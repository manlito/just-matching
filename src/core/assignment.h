// LightGlue's match assignment and keypoint rotary encoding on the host: the CPU backend's form, and the
// fallback of the base Ops class for backends that do not have their own (see core/ops.h).
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace jmt {

// Runs fn(begin, end) over [0, n), split as the caller likes.
using ParallelFor = std::function<void(int n, const std::function<void(int, int)>& fn)>;

// In place: each consecutive channel pair (a, b) of every head of every row becomes
// (a cos - b sin, b cos + a sin). x: n x (heads * 2 * pairs); cos, sin: n x pairs.
inline void rotary_rows(float* x, const float* cos, const float* sin, int n, int heads, int pairs) {
  for (int i = 0; i < n; ++i) {
    const float *c = cos + static_cast<size_t>(i) * pairs, *s = sin + static_cast<size_t>(i) * pairs;
    for (int hd = 0; hd < heads; ++hd) {
      float* r = x + (static_cast<size_t>(i) * heads + hd) * pairs * 2;
      for (int j = 0; j < pairs; ++j) {
        const float a = r[2 * j], b = r[2 * j + 1];
        r[2 * j] = a * c[j] - b * s[j];
        r[2 * j + 1] = b * c[j] + a * s[j];
      }
    }
  }
}

// log(sigmoid(z)), without overflow for large |z|.
inline float log_sigmoid(float z) { return std::min(z, 0.0f) - std::log1p(std::exp(-std::fabs(z))); }

// See Ops::assignment. a: M x d, b: N x d, row-major; za: M, zb: N.
inline void assignment_host(const float* a, int M, const float* b, int N, int d, const float* za, const float* zb,
                            const ParallelFor& parallel_for, std::vector<int>* best0, std::vector<float>* max0,
                            std::vector<int>* best1, std::vector<float>* max1) {
  best0->assign(M, -1), max0->assign(M, -INFINITY), best1->assign(N, -1), max1->assign(N, -INFINITY);
  if (M == 0 || N == 0) return;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  // b as (d, N), so a row of s is d passes over N contiguous values.
  std::vector<float> bt(static_cast<size_t>(d) * N);
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < d; ++k) bt[static_cast<size_t>(k) * N + j] = b[static_cast<size_t>(j) * d + k];
  std::vector<float> s(static_cast<size_t>(M) * N), row_lse(M), col_lse(N), ls0(M), ls1(N);
  parallel_for(M, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      float* r = s.data() + static_cast<size_t>(i) * N;
      std::fill(r, r + N, 0.0f);
      for (int k = 0; k < d; ++k) {
        const float x = a[static_cast<size_t>(i) * d + k] * scale;
        const float* bk = bt.data() + static_cast<size_t>(k) * N;
        for (int j = 0; j < N; ++j) r[j] += x * bk[j];
      }
      const float top = *std::max_element(r, r + N);
      float sum = 0;
      for (int j = 0; j < N; ++j) sum += std::exp(r[j] - top);
      row_lse[i] = top + std::log(sum);
      ls0[i] = log_sigmoid(za[i]);
    }
  });
  parallel_for(N, [&](int begin, int end) {
    std::vector<float> top(end - begin, -INFINITY), sum(end - begin, 0.0f);
    for (int i = 0; i < M; ++i) {
      const float* r = s.data() + static_cast<size_t>(i) * N;
      for (int j = begin; j < end; ++j) top[j - begin] = std::max(top[j - begin], r[j]);
    }
    for (int i = 0; i < M; ++i) {
      const float* r = s.data() + static_cast<size_t>(i) * N;
      for (int j = begin; j < end; ++j) sum[j - begin] += std::exp(r[j] - top[j - begin]);
    }
    for (int j = begin; j < end; ++j) {
      col_lse[j] = top[j - begin] + std::log(sum[j - begin]);
      ls1[j] = log_sigmoid(zb[j]);
    }
  });
  // The score in the original's order: the two log-softmaxes, then the certainties.
  auto score = [&](int i, int j, float sij) { return ((sij - row_lse[i]) + (sij - col_lse[j])) + (ls0[i] + ls1[j]); };
  parallel_for(M, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      const float* r = s.data() + static_cast<size_t>(i) * N;
      int arg = 0;
      float best = score(i, 0, r[0]);
      for (int j = 1; j < N; ++j) {
        const float v = score(i, j, r[j]);
        if (v > best) best = v, arg = j;
      }
      (*best0)[i] = arg, (*max0)[i] = best;
    }
  });
  parallel_for(N, [&](int begin, int end) {
    for (int i = 0; i < M; ++i) {
      const float* r = s.data() + static_cast<size_t>(i) * N;
      for (int j = begin; j < end; ++j) {
        const float v = score(i, j, r[j]);
        if (v > (*max1)[j]) (*max1)[j] = v, (*best1)[j] = i;
      }
    }
  });
}

}  // namespace jmt
