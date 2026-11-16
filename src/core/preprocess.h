// Image preprocessing on the host (see Ops::preprocess), one output row at a time. Shared by the backends.
#pragma once

#include <algorithm>
#include <cstddef>

namespace jmt {

// Bilinear source position for output index i (PyTorch, align_corners = false).
inline void resize_source(int i, int out, int in, int* i0, int* i1, float* t) {
  float s = (static_cast<float>(i) + 0.5f) * (static_cast<float>(in) / static_cast<float>(out)) - 0.5f;
  if (s < 0) s = 0;
  *i0 = std::min(static_cast<int>(s), in - 1);
  *i1 = std::min(*i0 + 1, in - 1);
  *t = s - static_cast<float>(*i0);
}

// Row y of the out_h x out_w x 3 network input from rgb (height x width x 3 floats in [0, 1]).
inline void preprocess_row(const float* rgb, int height, int width, int out_h, int out_w, const float mean[3],
                           const float std[3], int y, float* out) {
  int y0, y1;
  float ty;
  resize_source(y, out_h, height, &y0, &y1, &ty);
  for (int x = 0; x < out_w; ++x) {
    int x0, x1;
    float tx;
    resize_source(x, out_w, width, &x0, &x1, &tx);
    const float* p00 = rgb + (static_cast<size_t>(y0) * width + x0) * 3;
    const float* p01 = rgb + (static_cast<size_t>(y0) * width + x1) * 3;
    const float* p10 = rgb + (static_cast<size_t>(y1) * width + x0) * 3;
    const float* p11 = rgb + (static_cast<size_t>(y1) * width + x1) * 3;
    for (int ch = 0; ch < 3; ++ch) {
      float top = p00[ch] * (1 - tx) + p01[ch] * tx, bottom = p10[ch] * (1 - tx) + p11[ch] * tx;
      out[x * 3 + ch] = ((top * (1 - ty) + bottom * ty) - mean[ch]) / std[ch];
    }
  }
}

}  // namespace jmt
