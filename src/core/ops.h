// The operations the FlexGlue graph is written against. A backend implements this interface.
//
// Every tensor is a grid of h x w tokens with c channels, stored token-major (h, w, c): a row per token.
// A list of N vectors is a 1 x N x c tensor. Weights are passed as they are stored in the weights file.
#pragma once

#include <memory>
#include <vector>

#include "core/weights.h"

namespace jmt {

struct Tensor {
  int h = 0, w = 0, c = 0;
  virtual ~Tensor() = default;
  // Copies h * w * c floats to dst (token-major).
  virtual void read(float* dst) const = 0;
  int n() const { return h * w; }
  size_t size() const { return static_cast<size_t>(h) * w * c; }
};
using TensorPtr = std::shared_ptr<Tensor>;

enum class Act { None, Gelu, Relu };

class Ops {
 public:
  virtual ~Ops() = default;

  virtual TensorPtr upload(const float* data, int h, int w, int c) = 0;

  // rgb: height x width x 3 floats in [0, 1]. Bilinear resize to out_h x out_w (pixel centres aligned as
  // PyTorch's align_corners = false, no antialiasing), then (x - mean) / std per channel.
  virtual TensorPtr preprocess(const float* rgb, int height, int width, int out_h, int out_w, const float mean[3],
                               const float std[3]) = 0;

  // y = x W^T + b on the channel axis. weight: (out, in); bias may be null.
  virtual TensorPtr linear(const TensorPtr& x, const Param& weight, const Param* bias) = 0;
  // Convolution with a k x k kernel. weight: (out, in, k, k). Zero padding.
  virtual TensorPtr conv(const TensorPtr& x, const Param& weight, const Param& bias, int k, int stride, int pad) = 0;
  // Depthwise 7x7 convolution, padding 3. weight: (c, 1, 7, 7).
  virtual TensorPtr dwconv7(const TensorPtr& x, const Param& weight, const Param& bias) = 0;
  // Normalization over the channel axis: (x - mean) / sqrt(var + eps) * gain + bias. When gain has fewer
  // entries than x has channels, every consecutive group of that many channels is normalized on its own
  // with the same gain and bias (the per-head norms of the attention blocks).
  virtual TensorPtr layer_norm(const TensorPtr& x, const Param& gain, const Param& bias, float eps) = 0;
  virtual TensorPtr gelu(const TensorPtr& x) = 0;  // exact: x * Phi(x)
  virtual TensorPtr relu(const TensorPtr& x) = 0;
  virtual TensorPtr add(const TensorPtr& a, const TensorPtr& b) = 0;
  virtual TensorPtr concat(const TensorPtr& a, const TensorPtr& b) = 0;       // on the channel axis
  virtual TensorPtr slice(const TensorPtr& x, int first, int count) = 0;       // channels [first, first + count)
  virtual TensorPtr pixel_unshuffle2(const TensorPtr& x) = 0;                  // (h, w, c) -> (h/2, w/2, 4c)
  virtual TensorPtr resize(const TensorPtr& x, int out_h, int out_w) = 0;      // bilinear, align_corners = false
  virtual TensorPtr l2_normalize(const TensorPtr& x) = 0;                      // per token, eps 1e-12

  // Rotary position encoding. x: n tokens x (heads * 2 * pairs) channels; cos, sin: n x pairs.
  // Each consecutive channel pair (a, b) of every head becomes (a cos - b sin, b cos + a sin).
  // The caller keeps cos and sin unchanged and at the same address for as long as it uses the backend (the
  // model computes them once per grid size): a backend may keep a device copy keyed by that address.
  virtual TensorPtr rotary(const TensorPtr& x, const std::vector<float>& cos, const std::vector<float>& sin,
                           int heads) = 0;
  // Scaled dot-product attention per head. q: (P, heads * d); k, v: (S, heads * d) -> (P, heads * d).
  virtual TensorPtr attention(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v, int heads) = 0;

  // Bilinear samples of x at n points (x, y) in [-1, 1] (-1 and 1 are the outer edges of the map; outside
  // points take the nearest border value) -> 1 x n x c.
  virtual TensorPtr sample(const TensorPtr& x, const float* pts, int n) = 0;

  // Match read-out. query: 1 x N x d, search: h x w x d, both L2-normalized. For every query: logits =
  // temperature * cosine similarity to every cell, 1.5-entmax with the threshold computed from the 512
  // largest logits, then the expected cell-centre position in the 3x3 window around the best cell
  // (coarse: N x 2 in [-1, 1]) and the probability mass of that window (mass: N).
  virtual void readout(const TensorPtr& query, const TensorPtr& search, float temperature, std::vector<float>* coarse,
                       std::vector<float>* mass) = 0;

  // ---- Fused forms. A backend may override them to save passes over memory; the defaults compose the
  // operations above, so the result is the same.

  // act(x W^T + b) + residual. residual may be null.
  virtual TensorPtr linear_fused(const TensorPtr& x, const Param& weight, const Param* bias, Act act,
                                 const TensorPtr& residual) {
    TensorPtr y = activate(linear(x, weight, bias), act);
    return residual ? add(residual, y) : y;
  }
  // act(layer_norm(x)).
  virtual TensorPtr layer_norm_act(const TensorPtr& x, const Param& gain, const Param& bias, float eps, Act act) {
    return activate(layer_norm(x, gain, bias, eps), act);
  }
  // preprocess from 8-bit pixels: `channels` bytes per pixel (1: grey, 3 or 4: R, G, B first), rows of
  // `stride` bytes. Same result as preprocess on the pixels divided by 255.
  virtual TensorPtr preprocess_u8(const unsigned char* data, int height, int width, int stride, int channels,
                                  int out_h, int out_w, const float mean[3], const float std[3]) {
    std::vector<float> rgb(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const unsigned char* p = data + static_cast<size_t>(y) * stride + static_cast<size_t>(x) * channels;
        float* o = rgb.data() + (static_cast<size_t>(y) * width + x) * 3;
        for (int c = 0; c < 3; ++c) o[c] = static_cast<float>(p[channels == 1 ? 0 : c]) / 255.0f;
      }
    return preprocess(rgb.data(), height, width, out_h, out_w, mean, std);
  }

  // ---- Read-out on the device. When has_device_readout() is true, readout_device and sample_at replace
  // readout and sample in the head, so a pair needs no round trip to the host before its last operation.
  virtual bool has_device_readout() const { return false; }
  // As readout; coarse: 1 x N x 2 and mass: 1 x N x 1, both read back as float32 exactly.
  virtual void readout_device(const TensorPtr& /*query*/, const TensorPtr& /*search*/, float /*temperature*/,
                              TensorPtr* /*coarse*/, TensorPtr* /*mass*/) {}
  // As sample, with the points taken from a tensor (1 x n x 2) made by readout_device.
  virtual TensorPtr sample_at(const TensorPtr& /*x*/, const TensorPtr& /*pts*/) { return nullptr; }

  // Device buffers created so far (0 when the backend does not count them). Tests use it to check that a
  // second pair of the same size allocates nothing.
  virtual long device_allocations() const { return 0; }

  // GPU backends queue their work. flush starts what is queued without waiting for it; finish blocks until
  // every tensor created so far is computed. Reading a tensor waits for that tensor.
  virtual void flush() {}
  virtual void finish() {}

 private:
  TensorPtr activate(const TensorPtr& x, Act act) { return act == Act::Gelu ? gelu(x) : act == Act::Relu ? relu(x) : x; }
};

std::unique_ptr<Ops> make_cpu_ops(int num_threads);
// Only in builds with the Metal backend. device: index, 0 = first. fp16: store tensors and weights as half.
std::unique_ptr<Ops> make_metal_ops(int device, bool fp16);
// Only in builds with the CUDA backend. Same arguments.
std::unique_ptr<Ops> make_cuda_ops(int device, bool fp16);

}  // namespace jmt
