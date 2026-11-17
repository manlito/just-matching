// Plain C++ reference backend. Written for clarity and agreement with tools/folded.py. Loops are arranged
// so the compiler can vectorize them and matrix products are cache-blocked, nothing more.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <thread>

#include "core/assignment.h"
#include "core/ops.h"
#include "core/preprocess.h"
#include "core/readout.h"

namespace jmt {
namespace {

struct CpuTensor : Tensor {
  std::vector<float> d;
  CpuTensor(int h_, int w_, int c_) : d(static_cast<size_t>(h_) * w_ * c_) { h = h_, w = w_, c = c_; }
  void read(float* dst) const override { std::memcpy(dst, d.data(), d.size() * sizeof(float)); }
  float* row(int i) { return d.data() + static_cast<size_t>(i) * c; }
  const float* row(int i) const { return d.data() + static_cast<size_t>(i) * c; }
};
using CpuPtr = std::shared_ptr<CpuTensor>;

const CpuTensor& cpu(const TensorPtr& t) { return static_cast<const CpuTensor&>(*t); }
CpuPtr make(int h, int w, int c) { return std::make_shared<CpuTensor>(h, w, c); }

inline void axpy(float a, const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) y[i] += a * x[i];
}

// y += a0 x0 + a1 x1 + a2 x2 + a3 x3: four rows of the transposed weight matrix per pass over y.
inline void axpy4(const float* a, const float* x0, const float* x1, const float* x2, const float* x3, float* y,
                  int n) {
  const float a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  for (int i = 0; i < n; ++i) y[i] += a0 * x0[i] + a1 * x1[i] + a2 * x2[i] + a3 * x3[i];
}

class CpuOps : public Ops {
 public:
  explicit CpuOps(int num_threads) {
    threads_ = num_threads > 0 ? num_threads : static_cast<int>(std::thread::hardware_concurrency());
    threads_ = std::max(1, threads_);
  }

  TensorPtr upload(const float* data, int h, int w, int c) override {
    auto out = make(h, w, c);
    std::memcpy(out->d.data(), data, out->d.size() * sizeof(float));
    return out;
  }

  TensorPtr preprocess(const float* rgb, int height, int width, int out_h, int out_w, const float mean[3],
                       const float std[3]) override {
    auto out = make(out_h, out_w, 3);
    parallel_for(out_h, [&](int y_begin, int y_end) {
      for (int y = y_begin; y < y_end; ++y) preprocess_row(rgb, height, width, out_h, out_w, mean, std, y, out->row(y * out_w));
    });
    return out;
  }

  TensorPtr linear(const TensorPtr& xp, const Param& weight, const Param* bias) override {
    const CpuTensor& x = cpu(xp);
    int out_c = weight.shape[0], in_c = weight.shape[1];
    if (weight.shape.size() != 2 || in_c != x.c) throw std::runtime_error("linear: shape mismatch");
    auto out = make(x.h, x.w, out_c);
    gemm(x.d.data(), x.n(), in_c, transposed(weight, out_c, in_c), bias ? bias->data : nullptr, out->d.data(), out_c);
    return out;
  }

  TensorPtr conv(const TensorPtr& xp, const Param& weight, const Param& bias, int k, int stride, int pad) override {
    const CpuTensor& x = cpu(xp);
    int out_c = weight.shape[0], in_c = weight.shape[1];
    if (weight.shape.size() != 4 || in_c != x.c || weight.shape[2] != k) throw std::runtime_error("conv: shape mismatch");
    int oh = (x.h + 2 * pad - k) / stride + 1, ow = (x.w + 2 * pad - k) / stride + 1, cols = k * k * in_c;
    // Patches (oh * ow, k * k * in), ordered (ky, kx, channel); then one matrix product.
    std::vector<float> patches(static_cast<size_t>(oh) * ow * cols, 0.0f);
    parallel_for(oh, [&](int y_begin, int y_end) {
      for (int y = y_begin; y < y_end; ++y)
        for (int xo = 0; xo < ow; ++xo) {
          float* p = patches.data() + (static_cast<size_t>(y) * ow + xo) * cols;
          for (int ky = 0; ky < k; ++ky) {
            int sy = y * stride - pad + ky;
            if (sy < 0 || sy >= x.h) continue;
            for (int kx = 0; kx < k; ++kx) {
              int sx = xo * stride - pad + kx;
              if (sx < 0 || sx >= x.w) continue;
              std::memcpy(p + (ky * k + kx) * in_c, x.row(sy * x.w + sx), in_c * sizeof(float));
            }
          }
        }
    });
    std::vector<float>& wt = cache_[weight.data];
    if (wt.empty()) {
      wt.resize(static_cast<size_t>(cols) * out_c);
      for (int o = 0; o < out_c; ++o)
        for (int ch = 0; ch < in_c; ++ch)
          for (int t = 0; t < k * k; ++t)
            wt[(static_cast<size_t>(t) * in_c + ch) * out_c + o] = weight.data[(static_cast<size_t>(o) * in_c + ch) * k * k + t];
    }
    auto out = make(oh, ow, out_c);
    gemm(patches.data(), oh * ow, cols, wt.data(), bias.data, out->d.data(), out_c);
    return out;
  }

  TensorPtr dwconv7(const TensorPtr& xp, const Param& weight, const Param& bias) override {
    const CpuTensor& x = cpu(xp);
    const int c = x.c;
    std::vector<float>& taps = cache_[weight.data];  // (49, c)
    if (taps.empty()) {
      taps.resize(static_cast<size_t>(49) * c);
      for (int ch = 0; ch < c; ++ch)
        for (int t = 0; t < 49; ++t) taps[static_cast<size_t>(t) * c + ch] = weight.data[static_cast<size_t>(ch) * 49 + t];
    }
    auto out = make(x.h, x.w, c);
    parallel_for(x.h, [&](int y_begin, int y_end) {
      for (int y = y_begin; y < y_end; ++y)
        for (int xo = 0; xo < x.w; ++xo) {
          float* o = out->row(y * x.w + xo);
          std::memcpy(o, bias.data, c * sizeof(float));
          for (int ky = 0; ky < 7; ++ky) {
            int sy = y + ky - 3;
            if (sy < 0 || sy >= x.h) continue;
            for (int kx = 0; kx < 7; ++kx) {
              int sx = xo + kx - 3;
              if (sx < 0 || sx >= x.w) continue;
              const float* in = x.row(sy * x.w + sx);
              const float* t = taps.data() + static_cast<size_t>(ky * 7 + kx) * c;
              for (int ch = 0; ch < c; ++ch) o[ch] += t[ch] * in[ch];
            }
          }
        }
    });
    return out;
  }

  TensorPtr layer_norm(const TensorPtr& xp, const Param& gain, const Param& bias, float eps) override {
    const CpuTensor& x = cpu(xp);
    const int g = static_cast<int>(gain.size());
    if (g <= 0 || x.c % g != 0) throw std::runtime_error("layer_norm: shape mismatch");
    auto out = make(x.h, x.w, x.c);
    parallel_for(x.n(), [&](int begin, int end) {
      for (int i = begin; i < end; ++i)
        for (int c0 = 0; c0 < x.c; c0 += g) {
          const float* in = x.row(i) + c0;
          float* o = out->row(i) + c0;
          float mean = 0;
          for (int j = 0; j < g; ++j) mean += in[j];
          mean /= static_cast<float>(g);
          float var = 0;
          for (int j = 0; j < g; ++j) var += (in[j] - mean) * (in[j] - mean);
          float inv = 1.0f / std::sqrt(var / static_cast<float>(g) + eps);
          for (int j = 0; j < g; ++j) o[j] = (in[j] - mean) * inv * gain.data[j] + bias.data[j];
        }
    });
    return out;
  }

  TensorPtr gelu(const TensorPtr& xp) override {
    return map(xp, [](float v) { return 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f)); });
  }
  TensorPtr relu(const TensorPtr& xp) override {
    return map(xp, [](float v) { return v > 0 ? v : 0.0f; });
  }

  TensorPtr add(const TensorPtr& ap, const TensorPtr& bp) override {
    const CpuTensor &a = cpu(ap), &b = cpu(bp);
    if (a.d.size() != b.d.size()) throw std::runtime_error("add: shape mismatch");
    auto out = make(a.h, a.w, a.c);
    for (size_t i = 0; i < a.d.size(); ++i) out->d[i] = a.d[i] + b.d[i];
    return out;
  }

  TensorPtr concat(const TensorPtr& ap, const TensorPtr& bp) override {
    const CpuTensor &a = cpu(ap), &b = cpu(bp);
    if (a.n() != b.n()) throw std::runtime_error("concat: shape mismatch");
    auto out = make(a.h, a.w, a.c + b.c);
    for (int i = 0; i < a.n(); ++i) {
      std::memcpy(out->row(i), a.row(i), a.c * sizeof(float));
      std::memcpy(out->row(i) + a.c, b.row(i), b.c * sizeof(float));
    }
    return out;
  }

  TensorPtr slice(const TensorPtr& xp, int first, int count) override {
    const CpuTensor& x = cpu(xp);
    auto out = make(x.h, x.w, count);
    for (int i = 0; i < x.n(); ++i) std::memcpy(out->row(i), x.row(i) + first, count * sizeof(float));
    return out;
  }

  TensorPtr pixel_unshuffle2(const TensorPtr& xp) override {
    // Output channel = c * 4 + dy * 2 + dx, as torch.nn.functional.pixel_unshuffle.
    const CpuTensor& x = cpu(xp);
    auto out = make(x.h / 2, x.w / 2, x.c * 4);
    for (int y = 0; y < out->h; ++y)
      for (int xo = 0; xo < out->w; ++xo) {
        float* o = out->row(y * out->w + xo);
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            const float* in = x.row((2 * y + dy) * x.w + 2 * xo + dx);
            for (int ch = 0; ch < x.c; ++ch) o[ch * 4 + dy * 2 + dx] = in[ch];
          }
      }
    return out;
  }

  TensorPtr resize(const TensorPtr& xp, int out_h, int out_w) override {
    const CpuTensor& x = cpu(xp);
    auto out = make(out_h, out_w, x.c);
    parallel_for(out_h, [&](int y_begin, int y_end) {
      for (int y = y_begin; y < y_end; ++y) {
        int y0, y1;
        float ty;
        resize_source(y, out_h, x.h, &y0, &y1, &ty);
        for (int xo = 0; xo < out_w; ++xo) {
          int x0, x1;
          float tx;
          resize_source(xo, out_w, x.w, &x0, &x1, &tx);
          const float *p00 = x.row(y0 * x.w + x0), *p01 = x.row(y0 * x.w + x1);
          const float *p10 = x.row(y1 * x.w + x0), *p11 = x.row(y1 * x.w + x1);
          float* o = out->row(y * out_w + xo);
          for (int ch = 0; ch < x.c; ++ch)
            o[ch] = (p00[ch] * (1 - tx) + p01[ch] * tx) * (1 - ty) + (p10[ch] * (1 - tx) + p11[ch] * tx) * ty;
        }
      }
    });
    return out;
  }

  TensorPtr l2_normalize(const TensorPtr& xp) override {
    const CpuTensor& x = cpu(xp);
    auto out = make(x.h, x.w, x.c);
    for (int i = 0; i < x.n(); ++i) {
      const float* in = x.row(i);
      double sq = 0;
      for (int j = 0; j < x.c; ++j) sq += static_cast<double>(in[j]) * in[j];
      float inv = 1.0f / std::max(static_cast<float>(std::sqrt(sq)), 1e-12f);
      for (int j = 0; j < x.c; ++j) out->row(i)[j] = in[j] * inv;
    }
    return out;
  }

  TensorPtr rotary(const TensorPtr& xp, const std::vector<float>& cos, const std::vector<float>& sin,
                   int heads) override {
    const CpuTensor& x = cpu(xp);
    const int pairs = x.c / heads / 2;
    if (static_cast<int>(cos.size()) != x.n() * pairs) throw std::runtime_error("rotary: shape mismatch");
    auto out = make(x.h, x.w, x.c);
    for (int i = 0; i < x.n(); ++i) {
      const float *c = cos.data() + static_cast<size_t>(i) * pairs, *s = sin.data() + static_cast<size_t>(i) * pairs;
      for (int hd = 0; hd < heads; ++hd) {
        const float* in = x.row(i) + hd * pairs * 2;
        float* o = out->row(i) + hd * pairs * 2;
        for (int j = 0; j < pairs; ++j) {
          o[2 * j] = in[2 * j] * c[j] - in[2 * j + 1] * s[j];
          o[2 * j + 1] = in[2 * j + 1] * c[j] + in[2 * j] * s[j];
        }
      }
    }
    return out;
  }

  TensorPtr attention(const TensorPtr& qp, const TensorPtr& kp, const TensorPtr& vp, int heads) override {
    const CpuTensor &q = cpu(qp), &k = cpu(kp), &v = cpu(vp);
    const int d = q.c / heads, P = q.n(), S = k.n();
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    // Keys per head as (d, S), so the logits of one query are d passes over S contiguous values.
    std::vector<float> kt(static_cast<size_t>(heads) * d * S);
    for (int s = 0; s < S; ++s)
      for (int j = 0; j < heads * d; ++j) kt[static_cast<size_t>(j) * S + s] = k.row(s)[j];
    auto out = make(q.h, q.w, q.c);
    parallel_for(P, [&](int begin, int end) {
      std::vector<float> logits(S);
      for (int p = begin; p < end; ++p)
        for (int hd = 0; hd < heads; ++hd) {
          const float* qr = q.row(p) + hd * d;
          std::fill(logits.begin(), logits.end(), 0.0f);
          for (int j = 0; j < d; ++j) axpy(qr[j] * scale, kt.data() + static_cast<size_t>(hd * d + j) * S, logits.data(), S);
          float top = logits[0];
          for (int s = 1; s < S; ++s) top = std::max(top, logits[s]);
          double sum = 0;
          for (int s = 0; s < S; ++s) {
            logits[s] = std::exp(logits[s] - top);
            sum += logits[s];
          }
          float* o = out->row(p) + hd * d;
          std::vector<double> acc(d, 0.0);
          const double inv = 1.0 / sum;
          for (int s = 0; s < S; ++s) {
            const double wgt = logits[s] * inv;
            const float* vr = v.row(s) + hd * d;
            for (int j = 0; j < d; ++j) acc[j] += wgt * vr[j];
          }
          for (int j = 0; j < d; ++j) o[j] = static_cast<float>(acc[j]);
        }
    });
    return out;
  }

  TensorPtr rotary_keypoints(const TensorPtr& xp, const TensorPtr& cosp, const TensorPtr& sinp, int heads) override {
    const CpuTensor &x = cpu(xp), &c = cpu(cosp), &sn = cpu(sinp);
    const int pairs = x.c / heads / 2;
    if (c.n() != x.n() || c.c != pairs || sn.n() != x.n() || sn.c != pairs) throw std::runtime_error("rotary: shape mismatch");
    auto out = make(x.h, x.w, x.c);
    out->d = x.d;
    parallel_for(x.n(), [&](int begin, int end) {
      rotary_rows(out->row(begin), c.d.data() + static_cast<size_t>(begin) * pairs,
                  sn.d.data() + static_cast<size_t>(begin) * pairs, end - begin, heads, pairs);
    });
    return out;
  }

  void assignment(const TensorPtr& ap, const TensorPtr& bp, const TensorPtr& zap, const TensorPtr& zbp,
                  std::vector<int>* best0, std::vector<float>* max0, std::vector<int>* best1,
                  std::vector<float>* max1) override {
    const CpuTensor &a = cpu(ap), &b = cpu(bp), &za = cpu(zap), &zb = cpu(zbp);
    if (a.c != b.c || za.n() != a.n() || zb.n() != b.n() || za.c != 1 || zb.c != 1)
      throw std::runtime_error("assignment: shape mismatch");
    assignment_host(a.d.data(), a.n(), b.d.data(), b.n(), a.c, za.d.data(), zb.d.data(),
                    [this](int n, const std::function<void(int, int)>& fn) { parallel_for(n, fn); }, best0, max0,
                    best1, max1);
  }

  TensorPtr sample(const TensorPtr& xp, const float* pts, int n) override {
    const CpuTensor& x = cpu(xp);
    auto out = make(1, n, x.c);
    for (int i = 0; i < n; ++i) {
      float fx = (pts[2 * i] + 1.0f) * 0.5f * static_cast<float>(x.w) - 0.5f;
      float fy = (pts[2 * i + 1] + 1.0f) * 0.5f * static_cast<float>(x.h) - 0.5f;
      fx = std::min(std::max(fx, 0.0f), static_cast<float>(x.w - 1));
      fy = std::min(std::max(fy, 0.0f), static_cast<float>(x.h - 1));
      int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
      int x1 = std::min(x0 + 1, x.w - 1), y1 = std::min(y0 + 1, x.h - 1);
      float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0);
      const float *p00 = x.row(y0 * x.w + x0), *p01 = x.row(y0 * x.w + x1);
      const float *p10 = x.row(y1 * x.w + x0), *p11 = x.row(y1 * x.w + x1);
      float* o = out->row(i);
      for (int ch = 0; ch < x.c; ++ch)
        o[ch] = p00[ch] * (1 - tx) * (1 - ty) + p01[ch] * tx * (1 - ty) + p10[ch] * (1 - tx) * ty + p11[ch] * tx * ty;
    }
    return out;
  }

  void readout(const TensorPtr& queryp, const TensorPtr& searchp, float temperature, std::vector<float>* coarse,
               std::vector<float>* mass) override {
    const CpuTensor &query = cpu(queryp), &search = cpu(searchp);
    const int N = query.n(), P = search.n(), d = query.c, h = search.h, w = search.w;
    std::vector<float> st(static_cast<size_t>(d) * P);
    for (int s = 0; s < P; ++s)
      for (int j = 0; j < d; ++j) st[static_cast<size_t>(j) * P + s] = search.row(s)[j];
    coarse->assign(static_cast<size_t>(N) * 2, 0.0f);
    mass->assign(N, 0.0f);
    parallel_for(N, [&](int begin, int end) {
      std::vector<float> logits(P);
      ReadoutScratch scratch;
      for (int i = begin; i < end; ++i) {
        std::fill(logits.begin(), logits.end(), 0.0f);
        for (int j = 0; j < d; ++j) axpy(query.row(i)[j] * temperature, st.data() + static_cast<size_t>(j) * P, logits.data(), P);
        readout_row(logits.data(), h, w, scratch, coarse->data() + 2 * static_cast<size_t>(i), mass->data() + i);
      }
    });
  }

 private:
  // (out, in) weight matrix as (in, out), built on first use.
  const float* transposed(const Param& weight, int out_c, int in_c) {
    std::vector<float>& wt = cache_[weight.data];
    if (wt.empty()) {
      wt.resize(static_cast<size_t>(in_c) * out_c);
      for (int o = 0; o < out_c; ++o)
        for (int i = 0; i < in_c; ++i) wt[static_cast<size_t>(i) * out_c + o] = weight.data[static_cast<size_t>(o) * in_c + i];
    }
    return wt.data();
  }

  // Y (n, out) = X (n, in) * Wt (in, out) + bias. Blocked so a slab of Wt and a block of Y stay in cache.
  void gemm(const float* X, int n, int in_c, const float* wt, const float* bias, float* Y, int out_c) const {
    const int row_block = 32, in_block = std::max(4, (1 << 16) / std::max(out_c, 1) / 4 * 4);
    parallel_for((n + row_block - 1) / row_block, [&](int b_begin, int b_end) {
      for (int b = b_begin; b < b_end; ++b) {
        const int r0 = b * row_block, r1 = std::min(n, r0 + row_block);
        for (int r = r0; r < r1; ++r) {
          float* y = Y + static_cast<size_t>(r) * out_c;
          if (bias) std::memcpy(y, bias, out_c * sizeof(float));
          else std::fill(y, y + out_c, 0.0f);
        }
        for (int i0 = 0; i0 < in_c; i0 += in_block) {
          const int i1 = std::min(in_c, i0 + in_block);
          for (int r = r0; r < r1; ++r) {
            const float* x = X + static_cast<size_t>(r) * in_c;
            float* y = Y + static_cast<size_t>(r) * out_c;
            int i = i0;
            for (; i + 4 <= i1; i += 4) {
              const float* w0 = wt + static_cast<size_t>(i) * out_c;
              axpy4(x + i, w0, w0 + out_c, w0 + 2 * static_cast<size_t>(out_c), w0 + 3 * static_cast<size_t>(out_c), y, out_c);
            }
            for (; i < i1; ++i) axpy(x[i], wt + static_cast<size_t>(i) * out_c, y, out_c);
          }
        }
      }
    });
  }

  template <typename F>
  TensorPtr map(const TensorPtr& xp, F f) const {
    const CpuTensor& x = cpu(xp);
    auto out = make(x.h, x.w, x.c);
    parallel_for(x.n(), [&](int begin, int end) {
      const size_t a = static_cast<size_t>(begin) * x.c, b = static_cast<size_t>(end) * x.c;
      for (size_t i = a; i < b; ++i) out->d[i] = f(x.d[i]);
    });
    return out;
  }

  // Runs fn(begin, end) over [0, n) split across the worker threads.
  void parallel_for(int n, const std::function<void(int, int)>& fn) const {
    int parts = std::min(threads_, n);
    if (parts <= 1) {
      if (n > 0) fn(0, n);
      return;
    }
    std::vector<std::thread> pool;
    for (int t = 1; t < parts; ++t)
      pool.emplace_back(fn, static_cast<int>(static_cast<long long>(n) * t / parts),
                        static_cast<int>(static_cast<long long>(n) * (t + 1) / parts));
    fn(0, n / parts);
    for (auto& th : pool) th.join();
  }

  int threads_;
  std::map<const float*, std::vector<float>> cache_;  // rearranged weights, keyed by the stored tensor
};

}  // namespace

std::unique_ptr<Ops> make_cpu_ops(int num_threads) { return std::make_unique<CpuOps>(num_threads); }

}  // namespace jmt
