// Compares every stage of the library against golden tensors dumped from the reference forward pass.
//   golden_test WEIGHTS.jmtw GOLDEN.jmtt
// Generate golden files with: python tools/dump_golden.py A B
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/flexglue/flexglue.h"

namespace {

int g_failed = 0;

// Golden maps are stored (c, h, w); the library works token-major (h, w, c).
std::vector<float> to_chw(const jmt::TensorPtr& t) {
  std::vector<float> v(t->size()), out(t->size());
  t->read(v.data());
  const size_t n = static_cast<size_t>(t->h) * t->w;
  for (size_t i = 0; i < n; ++i)
    for (int c = 0; c < t->c; ++c) out[static_cast<size_t>(c) * n + i] = v[i * t->c + c];
  return out;
}

// Passes when max|got - ref| <= tol * max(1, max|ref|).
void check(const std::string& name, const std::vector<float>& got, const jmt::Param* ref, double tol) {
  if (!ref || ref->size() != got.size()) {
    std::printf("%-22s FAIL (size: got %zu, golden %zu)\n", name.c_str(), got.size(), ref ? ref->size() : 0);
    ++g_failed;
    return;
  }
  double max_diff = 0, max_ref = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_diff = std::fmax(max_diff, std::fabs(static_cast<double>(got[i]) - ref->data[i]));
    max_ref = std::fmax(max_ref, std::fabs(static_cast<double>(ref->data[i])));
  }
  bool ok = max_diff <= tol * std::fmax(1.0, max_ref);
  std::printf("%-22s %s  max|diff| %.2e  (max|ref| %.2e, tol %.0e)\n", name.c_str(), ok ? "ok  " : "FAIL", max_diff,
              max_ref, tol);
  g_failed += !ok;
}

// Points: passes when at most `allowed` of them differ by more than `tol` pixels. A query whose two best
// cells are nearly equal may pick the other one, which moves its point by many pixels.
void check_points(const std::string& name, const std::vector<float>& got, const jmt::Param* ref, double tol,
                  double allowed) {
  if (!ref || ref->size() != got.size()) {
    std::printf("%-22s FAIL (size)\n", name.c_str());
    ++g_failed;
    return;
  }
  size_t n = got.size() / 2, off = 0;
  std::vector<double> d(n);
  for (size_t i = 0; i < n; ++i) {
    d[i] = std::hypot(got[2 * i] - ref->data[2 * i], got[2 * i + 1] - ref->data[2 * i + 1]);
    off += d[i] > tol;
  }
  std::sort(d.begin(), d.end());
  bool ok = static_cast<double>(off) <= allowed * static_cast<double>(n);
  std::printf("%-22s %s  median %.2e, %zu of %zu beyond %.0e\n", name.c_str(), ok ? "ok  " : "FAIL",
              n ? d[n / 2] : 0.0, off, n, tol);
  g_failed += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: golden_test WEIGHTS.jmtw GOLDEN.jmtt\n");
    return 2;
  }
  try {
    jmt::Weights weights = jmt::Weights::from_file(argv[1]);
    jmt::Weights golden = jmt::Weights::from_file(argv[2]);
    const int size = golden.meta_int("size");
    jmt::FlexGlue model(weights);
    auto ops = jmt::make_cpu_ops(0);

    jmt::TensorPtr feats[2];
    float w[2], h[2];
    for (int i = 0; i < 2; ++i) {
      const std::string im = "im" + std::to_string(i) + ".";
      const jmt::Param& image = golden.get("image" + std::to_string(i));  // (H, W, 3), values 0..255
      std::vector<float> rgb(image.size());
      for (size_t j = 0; j < rgb.size(); ++j) rgb[j] = image.data[j] / 255.0f;
      h[i] = static_cast<float>(image.shape[0]), w[i] = static_cast<float>(image.shape[1]);
      feats[i] = model.describe(*ops, rgb.data(), image.shape[0], image.shape[1], size,
                                [&](const std::string& name, const jmt::TensorPtr& t) {
                                  check(im + name, to_chw(t), golden.find(im + name), 1e-4);
                                });
    }
    jmt::MatchOptions options;
    options.n_layers = golden.meta_int("layers");
    options.filter_threshold = std::stof(golden.meta("filter_threshold"));
    options.max_distance_px = std::stof(golden.meta("max_distance_px"));
    const jmt::Param &k0 = golden.get("keypoints0"), &k1 = golden.get("keypoints1");
    jmt::MatchResult out;
    model.match(*ops, feats[0], k0.data, k0.shape[0], w[0], h[0], feats[1], k1.data, k1.shape[0], w[1], h[1],
                options, &out, [&](const std::string& name, const jmt::TensorPtr& t) {
                  std::vector<float> v(t->size());
                  t->read(v.data());
                  if (name.rfind("coarse", 0) == 0) check_points("pair." + name, v, golden.find("pair." + name), 1e-4, 0.005);
                  else check("pair." + name, v, golden.find("pair." + name), name.rfind("mass", 0) == 0 ? 1e-2 : 1e-3);
                });
    check_points("out.p2d0_i1", out.p2d0_i1, golden.find("out.p2d0_i1"), 0.05, 0.005);
    check_points("out.p2d1_i0", out.p2d1_i0, golden.find("out.p2d1_i0"), 0.05, 0.005);
    check("out.covisibility", {out.covisibility}, golden.find("out.covisibility"), 1e-3);

    const jmt::Param& ref = golden.get("out.matches0");
    size_t same = 0, n_ref = 0, n_got = 0;
    for (size_t i = 0; i < out.matches0.size(); ++i) {
      int r = static_cast<int>(ref.data[i]);
      n_ref += r >= 0, n_got += out.matches0[i] >= 0;
      same += r >= 0 && r == out.matches0[i];
    }
    bool ok = n_ref == n_got ? same >= n_ref - n_ref / 50 : false;
    ok = ok || (same + 2 >= std::max(n_ref, n_got));
    std::printf("%-22s %s  %zu matches, golden %zu, same %zu\n", "out.matches0", ok ? "ok  " : "FAIL", n_got, n_ref, same);
    g_failed += !ok;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all stages match\n", g_failed);
  return g_failed ? 1 : 0;
}
