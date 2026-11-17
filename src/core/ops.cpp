// Host forms of the operations a backend may leave to the base class (see core/ops.h).
#include "core/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/assignment.h"

namespace jmt {
namespace {

std::vector<float> read(const TensorPtr& t) {
  std::vector<float> v(t->size());
  if (!v.empty()) t->read(v.data());
  return v;
}

}  // namespace

TensorPtr Ops::rotary_keypoints(const TensorPtr& x, const TensorPtr& cos, const TensorPtr& sin, int heads) {
  const int pairs = x->c / heads / 2;
  if (cos->n() != x->n() || cos->c != pairs || sin->n() != x->n() || sin->c != pairs)
    throw std::runtime_error("rotary: shape mismatch");
  std::vector<float> v = read(x);
  rotary_rows(v.data(), read(cos).data(), read(sin).data(), x->n(), heads, pairs);
  return upload(v.data(), x->h, x->w, x->c);
}

TensorPtr Ops::gather_rows(const TensorPtr& x, const std::vector<int>& rows) {
  const std::vector<float> v = read(x);
  std::vector<float> out(rows.size() * static_cast<size_t>(x->c));
  for (size_t r = 0; r < rows.size(); ++r) {
    if (rows[r] < 0 || rows[r] >= x->n()) throw std::runtime_error("gather_rows: index out of range");
    std::copy_n(v.data() + static_cast<size_t>(rows[r]) * x->c, x->c, out.data() + r * x->c);
  }
  return upload(out.data(), 1, static_cast<int>(rows.size()), x->c);
}

void Ops::assignment(const TensorPtr& a, const TensorPtr& b, const TensorPtr& za, const TensorPtr& zb,
                     std::vector<int>* best0, std::vector<float>* max0, std::vector<int>* best1,
                     std::vector<float>* max1) {
  if (a->c != b->c || za->n() != a->n() || zb->n() != b->n() || za->c != 1 || zb->c != 1)
    throw std::runtime_error("assignment: shape mismatch");
  assignment_host(read(a).data(), a->n(), read(b).data(), b->n(), a->c, read(za).data(), read(zb).data(),
                  [](int n, const std::function<void(int, int)>& fn) { if (n > 0) fn(0, n); }, best0, max0, best1, max1);
}

}  // namespace jmt
