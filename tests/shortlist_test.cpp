// Checks jmt_covisibility and jmt_shortlist on the image pair of a golden file and copies of it, so the
// right answer is known without a reference: images [a, b, a, b], where the copies have equal features.
// Agreement with the reference scores is tools/compare_covis.py.
//   shortlist_test WEIGHTS.jmtw GOLDEN.jmtt [cpu|metal|cuda|webgpu, each GPU one also :fp16]
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/api_internal.h"
#include "core/weights.h"
#include "jmt/jmt.h"

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what) {
  if (!ok) std::printf("FAIL: %s\n", what.c_str()), ++g_failed;
}

std::set<std::pair<int, int>> pairs_of(const jmt_pair_list& l) {
  std::set<std::pair<int, int>> s;
  for (int p = 0; p < l.n; ++p) s.emplace(l.idx[2 * p], l.idx[2 * p + 1]);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: shortlist_test WEIGHTS.jmtw GOLDEN.jmtt [backend]\n");
    return 2;
  }
  const std::string backend = argc == 4 ? argv[3] : "cpu";
  try {
    jmt::Weights golden = jmt::Weights::from_file(argv[2]);
    jmt_config config;
    jmt_config_default(&config);
    config.weights_path = argv[1];
    config.input_size = golden.meta_int("size");
    if (backend.rfind("metal", 0) == 0) config.backend = JMT_BACKEND_METAL;
    else if (backend.rfind("cuda", 0) == 0) config.backend = JMT_BACKEND_CUDA;
    else if (backend.rfind("webgpu", 0) == 0) config.backend = JMT_BACKEND_WEBGPU;
    if (backend.find(":fp16") != std::string::npos) config.precision = JMT_PRECISION_FP16;
    jmt_ctx* ctx = jmt_create(&config);
    if (!ctx) throw std::runtime_error(jmt_last_error());

    jmt_features* f[4];
    for (int i = 0; i < 4; ++i) {
      const jmt::Param& image = golden.get("image" + std::to_string(i % 2));  // (H, W, 3), values 0..255
      std::vector<uint8_t> pixels(image.size());
      for (size_t j = 0; j < pixels.size(); ++j) pixels[j] = static_cast<uint8_t>(std::lround(image.data[j]));
      jmt_image view{pixels.data(), image.shape[1], image.shape[0], 0, JMT_FORMAT_RGB8};
      f[i] = jmt_describe(ctx, &view);
      if (!f[i]) throw std::runtime_error(jmt_last_error());
    }
    auto score = [&](int i, int j, int level) {
      float s;
      if (jmt_covisibility(ctx, f[i], f[j], level, &s)) throw std::runtime_error(jmt_last_error());
      return s;
    };

    const float ab1 = score(0, 1, 1), ab2 = score(0, 1, 2), aa1 = score(0, 2, 1), aa2 = score(0, 2, 2);
    const float bb2 = score(1, 3, 2);
    std::printf("level 1: a-b %.4f, a-a %.4f; level 2: a-b %.4f, a-a %.4f, b-b %.4f\n", ab1, aa1, ab2, aa2, bb2);
    check(std::fabs(aa1 - 1) < 1e-5f, "level 1 of an image with itself is 1");
    check(score(1, 0, 1) == ab1, "level 1 is symmetric");
    check(score(1, 0, 2) == ab2, "level 2 is symmetric");
    check(ab2 >= 0 && ab2 <= 1 && aa2 <= 1, "level 2 is in [0, 1]");
    check(aa2 > ab2 && bb2 > ab2, "an image scores higher with itself than with the other image");
    float bad;
    check(jmt_covisibility(ctx, f[0], f[1], 3, &bad) != 0, "level 3 is refused");

    const jmt_features* images[4] = {f[0], f[1], f[2], f[3]};
    jmt_shortlist_options o;
    jmt_shortlist_options_default(&o);
    jmt_pair_list l;
    // k = 1: every image keeps its copy.
    o.k = 1;
    if (jmt_shortlist(ctx, images, 4, &o, &l)) throw std::runtime_error(jmt_last_error());
    check(pairs_of(l) == std::set<std::pair<int, int>>{{0, 2}, {1, 3}}, "k = 1 keeps the copies");
    // Neighbours in order add (0, 1), (1, 2), (2, 3).
    o.neighbours = 1;
    if (jmt_shortlist(ctx, images, 4, &o, &l)) throw std::runtime_error(jmt_last_error());
    check(pairs_of(l) == std::set<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}}, "neighbours are added");
    for (int p = 1; p < l.n; ++p) check(l.score[p - 1] >= l.score[p], "sorted by score");
    // Every pair, reranked by level 2; then a bar between the copies and the other pairs.
    o.k = 10, o.neighbours = 0, o.rerank = 1;
    if (jmt_shortlist(ctx, images, 4, &o, &l)) throw std::runtime_error(jmt_last_error());
    check(l.n == 6, "k >= n - 1 keeps every pair");
    for (int p = 0; p < l.n; ++p) {
      const int i = l.idx[2 * p], j = l.idx[2 * p + 1];
      check(i < j, "first index < second");
      check(std::fabs(l.score[p] - score(i, j, 2)) < 1e-6f, "rerank gives the level-2 score");
      if (p) check(l.score[p - 1] >= l.score[p], "sorted by level 2");
    }
    const long allocations = jmt::ops_of(ctx).device_allocations();
    o.min_score = 0.5f * (ab2 + std::min(aa2, bb2));
    if (jmt_shortlist(ctx, images, 4, &o, &l)) throw std::runtime_error(jmt_last_error());
    check(pairs_of(l) == std::set<std::pair<int, int>>{{0, 2}, {1, 3}}, "min_score drops the other pairs");
    const long more = jmt::ops_of(ctx).device_allocations() - allocations;
    check(more == 0, "a second shortlist creates no device buffers (" + std::to_string(more) + ")");

    for (jmt_features* x : f) jmt_features_free(x);
    jmt_destroy(ctx);
    std::printf(g_failed ? "FAILED (%d)\n" : "covisibility and shortlist pass\n", g_failed);
    return g_failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
