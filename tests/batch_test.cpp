// Checks the C API on the image pair of a golden file: jmt_match_batch gives what jmt_match gives pair by
// pair, and matching again allocates no device buffers.
//   batch_test WEIGHTS.jmtw GOLDEN.jmtt [cpu|metal|metal:fp16|cuda|cuda:fp16]
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/api_internal.h"
#include "core/weights.h"
#include "jmt/jmt.h"

namespace {

struct Copy {
  std::vector<int32_t> idx;
  std::vector<float> score, a_in_b, conf_a;
  float covisibility;
  explicit Copy(const jmt_matches& m)
      : idx(m.idx, m.idx + 2 * static_cast<size_t>(m.n)), score(m.score, m.score + m.n),
        a_in_b(m.a_in_b, m.a_in_b + 2 * static_cast<size_t>(m.n_a)), conf_a(m.conf_a, m.conf_a + m.n_a),
        covisibility(m.covisibility) {}
  bool operator==(const Copy& o) const {
    return idx == o.idx && score == o.score && a_in_b == o.a_in_b && conf_a == o.conf_a && covisibility == o.covisibility;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: batch_test WEIGHTS.jmtw GOLDEN.jmtt [backend]\n");
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
    if (backend.find(":fp16") != std::string::npos) config.precision = JMT_PRECISION_FP16;
    jmt_ctx* ctx = jmt_create(&config);
    if (!ctx) throw std::runtime_error(jmt_last_error());

    jmt_features* f[2];
    std::vector<float> xy[2];
    for (int i = 0; i < 2; ++i) {
      const jmt::Param& image = golden.get("image" + std::to_string(i));  // (H, W, 3), values 0..255
      std::vector<uint8_t> pixels(image.size());
      for (size_t j = 0; j < pixels.size(); ++j) pixels[j] = static_cast<uint8_t>(std::lround(image.data[j]));
      jmt_image view{pixels.data(), image.shape[1], image.shape[0], 0, JMT_FORMAT_RGB8};
      f[i] = jmt_describe(ctx, &view);
      if (!f[i]) throw std::runtime_error(jmt_last_error());
      const jmt::Param& k = golden.get("keypoints" + std::to_string(i));  // FlexGlue's convention: corner origin
      xy[i].resize(k.size());
      for (size_t j = 0; j < xy[i].size(); ++j) xy[i][j] = k.data[j] - 0.5f;
    }
    const int n0 = static_cast<int>(xy[0].size() / 2), n1 = static_cast<int>(xy[1].size() / 2);
    jmt_match_options options;
    jmt_match_options_default(&options);
    options.n_layers = golden.meta_int("layers");

    // One by one.
    jmt_matches m;
    if (jmt_match(ctx, f[0], xy[0].data(), n0, f[1], xy[1].data(), n1, &options, &m)) throw std::runtime_error(jmt_last_error());
    const Copy ab(m);
    if (jmt_match(ctx, f[1], xy[1].data(), n1, f[0], xy[0].data(), n0, &options, &m)) throw std::runtime_error(jmt_last_error());
    const Copy ba(m);

    // As a batch, longer than the pipeline, twice: the second time must not allocate on the device.
    const jmt_pair pab = {f[0], xy[0].data(), n0, f[1], xy[1].data(), n1}, pba = {f[1], xy[1].data(), n1, f[0], xy[0].data(), n0};
    const std::vector<jmt_pair> pairs = {pab, pba, pab, pab, pba, pab, pba};
    std::vector<jmt_matches> all(pairs.size());
    int failed = 0;
    long allocations = 0;
    for (int round = 0; round < 2; ++round) {
      if (jmt_match_batch(ctx, pairs.data(), static_cast<int>(pairs.size()), &options, all.data())) throw std::runtime_error(jmt_last_error());
      for (size_t i = 0; i < pairs.size(); ++i) {
        const bool same = Copy(all[i]) == (pairs[i].a == f[0] ? ab : ba);
        if (!same) std::printf("round %d, pair %zu: FAIL (differs from jmt_match)\n", round, i), ++failed;
      }
      const long now = jmt::ops_of(ctx).device_allocations();
      if (round == 1 && now != allocations) std::printf("FAIL: the second batch created %ld device buffers\n", now - allocations), ++failed;
      allocations = now;
    }
    std::printf("%zu pairs twice: %d matches a->b, %d b->a, %ld device buffers in all\n", pairs.size(),
                static_cast<int>(ab.score.size()), static_cast<int>(ba.score.size()), allocations);
    jmt_features_free(f[0]), jmt_features_free(f[1]);
    jmt_destroy(ctx);
    std::printf(failed ? "FAILED (%d)\n" : "batch matches single pairs\n", failed);
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
