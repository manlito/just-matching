// Match read-out of one query from its logits over the h x w cells (see Ops::readout). Shared by the
// backends that do this step on the host.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace jmt {

struct ReadoutScratch {
  std::vector<float> sorted;
  std::vector<double> tau;
};

// logits: h * w values. scratch: reused between calls. coarse: 2 values (x, y in [-1, 1]); mass: 1 value.
inline void readout_row(const float* logits, int h, int w, ReadoutScratch& scratch, float* coarse, float* mass) {
  const int P = h * w, top_k = std::min(512, P);
  int best = 0;
  for (int s = 1; s < P; ++s)
    if (logits[s] > logits[best]) best = s;
  const float top = logits[best];
  scratch.sorted.assign(logits, logits + P);
  scratch.tau.resize(top_k);
  std::partial_sort(scratch.sorted.begin(), scratch.sorted.begin() + top_k, scratch.sorted.end(), std::greater<float>());
  // Threshold of 1.5-entmax, p = max(y - tau, 0)^2 with y = (logit - top) / 2, from the top_k largest:
  // tau_r is the threshold if only the r largest were in the support; the support is the number of r whose
  // value is still at or above tau_r.
  double sum = 0, sum_sq = 0;
  int support = 0;
  for (int r = 0; r < top_k; ++r) {
    const double y = (static_cast<double>(scratch.sorted[r]) - top) / 2, rho = r + 1;
    sum += y, sum_sq += y * y;
    const double mean = sum / rho, mean_sq = sum_sq / rho;
    scratch.tau[r] = mean - std::sqrt(std::max((1 - rho * (mean_sq - mean * mean)) / rho, 0.0));
    support += scratch.tau[r] <= y;
  }
  const double tau = scratch.tau[std::max(support, 1) - 1];
  double px = 0, py = 0, total = 0, inside = 0;
  const int by = best / w, bx = best % w;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int ny = by + dy, nx = bx + dx;
      const bool in = ny >= 0 && ny < h && nx >= 0 && nx < w;
      ny = std::min(std::max(ny, 0), h - 1), nx = std::min(std::max(nx, 0), w - 1);
      double p = std::max((static_cast<double>(logits[ny * w + nx]) - top) / 2 - tau, 0.0);
      p *= p;
      px += p * ((nx + 0.5) / w * 2 - 1), py += p * ((ny + 0.5) / h * 2 - 1);
      total += p;
      if (in) inside += p;
    }
  coarse[0] = static_cast<float>(px / total);
  coarse[1] = static_cast<float>(py / total);
  *mass = static_cast<float>(inside);
}

}  // namespace jmt
