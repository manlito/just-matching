#include "models/lightglue/lightglue.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jmt {
namespace {

std::vector<float> read(const TensorPtr& t) {
  std::vector<float> v(t->size());
  if (!v.empty()) t->read(v.data());
  return v;
}

void emit(const Trace& trace, const std::string& name, const TensorPtr& t) {
  if (trace) trace(name, t);
}

std::string idx(const std::string& prefix, int i) { return prefix + std::to_string(i); }

float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }

}  // namespace

LightGlue::LightGlue(const Weights& weights) : w_(weights) {
  if (w_.meta("model") != "lightglue") throw std::runtime_error("not LightGlue weights");
  n_layers_ = w_.meta_int("n_layers");
  heads_ = w_.meta_int("heads");
  dim_ = w_.meta_int("dim");
  input_dim_ = w_.meta_int("input_dim");
}

TensorPtr LightGlue::lin(Ops& ops, const std::string& name, const TensorPtr& x, Act act, const TensorPtr& residual) const {
  return ops.linear_fused(x, w_.get(name + ".weight"), w_.find(name + ".bias"), act, residual);
}

// x + Linear(GELU(LayerNorm(Linear([x, message])))): entries 0, 1 and 3 of the upstream Sequential.
TensorPtr LightGlue::ffn(Ops& ops, const std::string& block, const TensorPtr& x, const TensorPtr& message) const {
  TensorPtr y = lin(ops, block + ".ffn.0", ops.concat(x, message));
  y = ops.layer_norm_act(y, w_.get(block + ".ffn.1.weight"), w_.get(block + ".ffn.1.bias"), 1e-5f, Act::Gelu);
  return lin(ops, block + ".ffn.3", y, Act::None, x);
}

TensorPtr LightGlue::self_block(Ops& ops, const std::string& p, const TensorPtr& x, const Image& im) const {
  TensorPtr qkv = lin(ops, p + ".Wqkv", x);  // [q | k | v], each (heads, dim / heads)
  TensorPtr q = ops.rotary_keypoints(ops.slice(qkv, 0, dim_), im.cos, im.sin, heads_);
  TensorPtr k = ops.rotary_keypoints(ops.slice(qkv, dim_, dim_), im.cos, im.sin, heads_);
  TensorPtr message = lin(ops, p + ".out_proj", ops.attention(q, k, ops.slice(qkv, 2 * dim_, dim_), heads_));
  return ffn(ops, p, x, message);
}

LightGlue::Image LightGlue::prepare(Ops& ops, const float* xy, const float* desc, int n, float width, float height,
                                    const Trace& trace, const std::string& tag) const {
  if (n < 0 || (n && (!xy || !desc))) throw std::runtime_error("bad keypoints");
  if (!(width > 0 && height > 0)) throw std::runtime_error("image size must be positive");
  Image im;
  im.n = n;
  if (n == 0) return im;
  // Normalized as upstream: centred on the image, divided by half the longer side.
  const Param& wr = w_.get("posenc.Wr.weight");  // (pairs, 2), no bias
  const int pairs = wr.shape[0];
  const float shift_x = width / 2, shift_y = height / 2, scale = std::max(width, height) / 2;
  im.cos_h.resize(static_cast<size_t>(n) * pairs);
  im.sin_h.resize(im.cos_h.size());
  for (int i = 0; i < n; ++i) {
    const float x = (xy[2 * i] - shift_x) / scale, y = (xy[2 * i + 1] - shift_y) / scale;
    for (int j = 0; j < pairs; ++j) {
      const float angle = x * wr.data[2 * j] + y * wr.data[2 * j + 1];
      im.cos_h[static_cast<size_t>(i) * pairs + j] = std::cos(angle);
      im.sin_h[static_cast<size_t>(i) * pairs + j] = std::sin(angle);
    }
  }
  im.cos = ops.upload(im.cos_h.data(), 1, n, pairs);
  im.sin = ops.upload(im.sin_h.data(), 1, n, pairs);
  im.tokens = lin(ops, "input_proj", ops.upload(desc, 1, n, input_dim_));
  emit(trace, "input_proj" + tag, im.tokens);
  return im;
}

void LightGlue::match(Ops& ops, const Image& im0, const Image& im1, const Options& o, Result* out,
                      const Trace& trace) const {
  if (o.n_layers < 1 || o.n_layers > n_layers_) throw std::runtime_error("n_layers out of range");
  out->matches0.assign(im0.n, -1), out->matches1.assign(im1.n, -1);
  out->scores0.assign(im0.n, 0.0f), out->scores1.assign(im1.n, 0.0f);
  out->matchability0.assign(im0.n, 0.0f), out->matchability1.assign(im1.n, 0.0f);
  out->stop_layer = 0;
  if (im0.n == 0 || im1.n == 0) return;  // nothing to attend to or match

  TensorPtr x0 = im0.tokens, x1 = im1.tokens;
  for (int k = 0; k < o.n_layers; ++k) {
    std::string p = idx("transformers.", k);
    x0 = self_block(ops, p + ".self_attn", x0, im0);
    x1 = self_block(ops, p + ".self_attn", x1, im1);
    p += ".cross_attn";
    // One similarity in both directions: the queries of one image are the keys of the other.
    TensorPtr qk0 = lin(ops, p + ".to_qk", x0), qk1 = lin(ops, p + ".to_qk", x1);
    TensorPtr v0 = lin(ops, p + ".to_v", x0), v1 = lin(ops, p + ".to_v", x1);
    TensorPtr m0 = lin(ops, p + ".to_out", ops.attention(qk0, qk1, v1, heads_));
    TensorPtr m1 = lin(ops, p + ".to_out", ops.attention(qk1, qk0, v0, heads_));
    x0 = ffn(ops, p, x0, m0);
    x1 = ffn(ops, p, x1, m1);
    emit(trace, idx("layer", k) + ".desc0", x0), emit(trace, idx("layer", k) + ".desc1", x1);
  }
  out->stop_layer = o.n_layers;

  const std::string head = idx("log_assignment.", o.n_layers - 1);
  TensorPtr z0 = lin(ops, head + ".matchability", x0), z1 = lin(ops, head + ".matchability", x1);
  std::vector<int> best0, best1;
  std::vector<float> max0, max1;
  ops.assignment(lin(ops, head + ".final_proj", x0), lin(ops, head + ".final_proj", x1), z0, z1, &best0, &max0,
                 &best1, &max1);
  if (trace) {
    trace("max0", ops.upload(max0.data(), 1, im0.n, 1));
    trace("max1", ops.upload(max1.data(), 1, im1.n, 1));
  }
  // Mutual best matches whose score clears the threshold (upstream filter_matches).
  for (int i = 0; i < im0.n; ++i) {
    const int j = best0[i];
    if (best1[j] != i) continue;
    const float score = std::exp(max0[i]);
    if (!(score > o.filter_threshold)) continue;
    out->matches0[i] = j, out->scores0[i] = score;
    out->matches1[j] = i, out->scores1[j] = score;
  }
  std::vector<float> h0 = read(z0), h1 = read(z1);
  for (int i = 0; i < im0.n; ++i) out->matchability0[i] = sigmoid(h0[i]);
  for (int j = 0; j < im1.n; ++j) out->matchability1[j] = sigmoid(h1[j]);
}

}  // namespace jmt
