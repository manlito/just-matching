// Compares every stage of the LightGlue graph against golden tensors dumped from the original.
//   lightglue_golden_test WEIGHTS.jmtw GOLDEN.jmtt [cpu|metal|cuda|webgpu, each GPU one also :fp16]
// The tokens grow less exact with depth in any float32 run: by layer 5 the original's own are 4e-4 away
// from a float64 run of it (values up to 14). So the stages after the transformer layers are measured
// against that float64 run, and pass when they are no further from it than 4 times the original's float32
// run (and within 1e-5 of the largest value in any case).
// Generate golden files with: tools/dump_golden_lightglue.sh (or tools/dump_golden_lightglue.py A B).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/ops.h"
#include "core/weights.h"
#include "models/lightglue/lightglue.h"

namespace {

int g_failed = 0;

// Passes when max|got - ref| <= tol * max(1, max|ref|).
void check(const std::string& name, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
  if (ref.size() != got.size()) {
    std::printf("%-22s FAIL (size: got %zu, golden %zu)\n", name.c_str(), got.size(), ref.size());
    ++g_failed;
    return;
  }
  double max_diff = 0, max_ref = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_diff = std::fmax(max_diff, std::fabs(static_cast<double>(got[i]) - ref[i]));
    max_ref = std::fmax(max_ref, std::fabs(static_cast<double>(ref[i])));
  }
  const bool ok = max_diff <= tol * std::fmax(1.0, max_ref);
  std::printf("%-22s %s  max|diff| %.2e  (max|ref| %.2e, tol %.0e)\n", name.c_str(), ok ? "ok  " : "FAIL", max_diff,
              max_ref, tol);
  g_failed += !ok;
}

std::vector<float> values(const jmt::Param& p) { return std::vector<float>(p.data, p.data + p.size()); }

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(static_cast<double>(a[i]) - b[i]));
  return m;
}

// got and the original's float32 result (orig), both against the float64 result (exact).
void check_exact(const std::string& name, const std::vector<float>& got, const std::vector<float>& orig,
                 const std::vector<float>& exact, double loose) {
  if (got.size() != exact.size() || orig.size() != exact.size()) {
    std::printf("%-22s FAIL (size: got %zu, golden %zu)\n", name.c_str(), got.size(), exact.size());
    ++g_failed;
    return;
  }
  double max_ref = 0;
  for (float v : exact) max_ref = std::fmax(max_ref, std::fabs(static_cast<double>(v)));
  const double ours = max_abs_diff(got, exact), theirs = max_abs_diff(orig, exact);
  const double tol = std::fmax(4 * theirs, 1e-5 * std::fmax(1.0, max_ref)) * loose;
  const bool ok = ours <= tol;
  std::printf("%-22s %s  from exact: %.2e, original %.2e  (max|ref| %.2e, tol %.1e)\n", name.c_str(), ok ? "ok  " : "FAIL",
              ours, theirs, max_ref, tol);
  g_failed += !ok;
}

std::vector<float> read(const jmt::TensorPtr& t) {
  std::vector<float> v(t->size());
  if (!v.empty()) t->read(v.data());
  return v;
}

// Matches: passes when the matched sets agree but for a few pairs whose score sits at the threshold or
// whose two best partners are nearly tied (allowed: 1 % of the golden matches, at least 2). Scores are
// probabilities; they pass within 2e-4 (the tokens differ from the original's by about 1e-4 of their size).
void check_matches(const std::string& name, const std::vector<int>& got, const std::vector<float>& got_scores,
                   const jmt::Param& ref, const jmt::Param& ref_scores, double score_tol) {
  size_t n_ref = 0, n_got = 0, same = 0;
  double max_diff = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const int r = static_cast<int>(ref.data[i]);
    n_ref += r >= 0, n_got += got[i] >= 0;
    if (r >= 0 && r == got[i]) {
      ++same;
      max_diff = std::fmax(max_diff, std::fabs(static_cast<double>(got_scores[i]) - ref_scores.data[i]));
    }
  }
  const size_t allowed = std::max<size_t>(2, n_ref / 100);
  const bool ok = same + allowed >= std::max(n_ref, n_got) && max_diff <= score_tol;
  std::printf("%-22s %s  %zu matches, golden %zu, same %zu; scores max|diff| %.2e\n", name.c_str(), ok ? "ok  " : "FAIL",
              n_got, n_ref, same, max_diff);
  g_failed += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string backend = argc == 4 ? argv[3] : "cpu";
  const bool fp16 = backend.find(":fp16") != std::string::npos;
  const double loose = fp16 ? 300 : 1;
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: lightglue_golden_test WEIGHTS.jmtw GOLDEN.jmtt [cpu|metal|cuda|webgpu, each GPU one also :fp16]\n");
    return 2;
  }
  try {
    jmt::Weights weights = jmt::Weights::from_file(argv[1]);
    jmt::Weights golden = jmt::Weights::from_file(argv[2]);
    jmt::LightGlue model(weights);
    std::unique_ptr<jmt::Ops> ops;
    if (backend == "cpu") ops = jmt::make_cpu_ops(0);
#ifdef JMT_HAVE_METAL
    else if (backend == "metal" || backend == "metal:fp16") ops = jmt::make_metal_ops(0, fp16);
#endif
#ifdef JMT_HAVE_CUDA
    else if (backend == "cuda" || backend == "cuda:fp16") ops = jmt::make_cuda_ops(0, fp16);
#endif
#ifdef JMT_HAVE_WEBGPU
    else if (backend == "webgpu" || backend == "webgpu:fp16") ops = jmt::make_webgpu_ops(fp16);
#endif
    else throw std::runtime_error("this build has no " + backend + " backend");

    jmt::LightGlue::Image im[2];
    for (int i = 0; i < 2; ++i) {
      const std::string s = std::to_string(i);
      const jmt::Param &kp = golden.get("keypoints" + s), &desc = golden.get("descriptors" + s);
      const jmt::Param& size = golden.get("image_size" + s);  // (width, height)
      im[i] = model.prepare(*ops, kp.data, desc.data, kp.shape[0], size.data[0], size.data[1],
                            [&](const std::string& name, const jmt::TensorPtr& t) {
                              check("pair." + name, read(t), values(golden.get("pair." + name)), 1e-5 * loose);
                            }, s);
      // Golden encoding: (2, n, 64), cos then sin, each angle repeated for both channels of its pair.
      const jmt::Param& enc = golden.get("pair.encoding" + s);
      const size_t n = enc.shape[enc.shape.size() - 2], pairs = enc.shape.back() / 2;
      std::vector<float> cos_ref(n * pairs), sin_ref(n * pairs);
      for (size_t r = 0; r < n; ++r)
        for (size_t j = 0; j < pairs; ++j) {
          cos_ref[r * pairs + j] = enc.data[r * 2 * pairs + 2 * j];
          sin_ref[r * pairs + j] = enc.data[(n + r) * 2 * pairs + 2 * j];
        }
      check("pair.encoding" + s + ".cos", im[i].cos_h, cos_ref, 1e-5);
      check("pair.encoding" + s + ".sin", im[i].sin_h, sin_ref, 1e-5);
    }

    // Best score of every row and column of the golden log assignment, without its dustbins.
    const jmt::Param& la = golden.get("pair.log_assignment");  // (n0 + 1, n1 + 1)
    const int n0 = im[0].n, n1 = im[1].n, cols = la.shape.back();
    std::vector<float> max0_ref(n0, -INFINITY), max1_ref(n1, -INFINITY);
    for (int r = 0; r < n0; ++r)
      for (int c = 0; c < n1; ++c) {
        const float v = la.data[static_cast<size_t>(r) * cols + c];
        max0_ref[r] = std::max(max0_ref[r], v), max1_ref[c] = std::max(max1_ref[c], v);
      }

    jmt::LightGlue::Options options;
    options.n_layers = golden.meta_int("layers");
    options.filter_threshold = std::stof(golden.meta("filter_threshold"));
    options.depth_confidence = options.width_confidence = -1;  // fixed depth, every keypoint
    jmt::LightGlue::Result out;
    model.match(*ops, im[0], im[1], options, &out, [&](const std::string& name, const jmt::TensorPtr& t) {
      const std::vector<float> orig = name == "max0" ? max0_ref : name == "max1" ? max1_ref : values(golden.get("pair." + name));
      check_exact("pair." + name, read(t), orig, values(golden.get("pair.fp64." + name)), loose);
    });
    check_exact("pair.matchability0", out.matchability0, values(golden.get("pair.matchability0")),
                values(golden.get("pair.fp64.matchability0")), loose);
    check_exact("pair.matchability1", out.matchability1, values(golden.get("pair.matchability1")),
                values(golden.get("pair.fp64.matchability1")), loose);
    check_matches("out.matches0", out.matches0, out.scores0, golden.get("out.matches0"),
                  golden.get("out.matching_scores0"), 2e-4 * loose);
    check_matches("out.matches1", out.matches1, out.scores1, golden.get("out.matches1"),
                  golden.get("out.matching_scores1"), 2e-4 * loose);

    // Adaptive depth and width with upstream's defaults: the same stop layer, the same points dropped after
    // the same layers (but for a few whose confidence or matchability sits at a threshold), the same matches.
    jmt::LightGlue::Options adaptive;
    adaptive.filter_threshold = options.filter_threshold;
    jmt::LightGlue::Result ad;
    model.match(*ops, im[0], im[1], adaptive, &ad);
    const int stop_ref = static_cast<int>(golden.get("adaptive.stop").data[0]);
    const bool stop_ok = ad.stop_layer == stop_ref;
    std::printf("%-22s %s  %d layers, golden %d\n", "adaptive.stop", stop_ok ? "ok  " : "FAIL", ad.stop_layer, stop_ref);
    g_failed += !stop_ok;
    for (int s = 0; s < 2; ++s) {
      const std::vector<int>& got = s ? ad.layers1 : ad.layers0;
      const jmt::Param& ref = golden.get(s ? "adaptive.prune1" : "adaptive.prune0");
      size_t differ = 0, dropped = 0;
      for (size_t k = 0; k < got.size(); ++k) {
        differ += got[k] != static_cast<int>(ref.data[k]);
        dropped += static_cast<int>(ref.data[k]) < stop_ref;
      }
      const bool ok = got.size() == ref.size() && differ <= std::max<size_t>(2, got.size() / 200);
      std::printf("%-22s %s  %zu of %zu keypoints differ (golden drops %zu)\n", s ? "adaptive.prune1" : "adaptive.prune0",
                  ok ? "ok  " : "FAIL", differ, got.size(), dropped);
      g_failed += !ok;
    }
    check_matches("adaptive.matches0", ad.matches0, ad.scores0, golden.get("adaptive.matches0"),
                  golden.get("adaptive.matching_scores0"), 2e-4 * loose);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf(g_failed ? "FAILED (%d)\n" : "all stages match\n", g_failed);
  return g_failed ? 1 : 0;
}
