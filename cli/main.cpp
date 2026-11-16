// jmt: command line front end of the just-matching library.
//
//   jmt match A B --kpts-a A.jkpt --kpts-b B.jkpt [--out OUT.jmtt] [options]
//
// A, B: images (PNG, JPEG, ...). The keypoint files are just-keypoints outputs (`jkp extract --out`), or any
// container with a float32 tensor "keypoints" (N, 2) in pixels with (0,0) at the centre of the top-left pixel.
// OUT holds matches (K, 2: index in A, index in B), scores (K), a_in_b (Na, 2), b_in_a (Nb, 2), conf_a (Na),
// conf_b (Nb), covisibility (1), stop_layer (1), all float32; with --dump-features also features_a and
// features_b (h, w, 256).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#include "core/api_internal.h"
#include "core/weights.h"
#include "jmt/jmt.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: jmt match A B --kpts-a A.jkpt --kpts-b B.jkpt [--out OUT.jmtt] [options]\n"
               "  --backend B            cpu (default), metal, metal:fp16, cuda, cuda:fp16\n"
               "  --weights FILE         weights file (default weights/flexglue.jmtw)\n"
               "  --size N               network input side, multiple of 32 (default 512)\n"
               "  --layers N             transformer layers, 0..9 (default 9)\n"
               "  --filter T             minimum match score (default 0.1)\n"
               "  --max-px D             maximum transfer error in network-input pixels (default 3; 0 = off)\n"
               "  --min-covisibility S   reject pairs scoring below S (default 0 = off)\n"
               "  --early-stop D         stop early on non-covisible pairs, decay D in [0,1) (default off)\n"
               "  --threads N            CPU threads (default all)\n"
               "  --dump-features        also write the dense features to OUT\n");
}

// Waits for the features to be computed (GPU backends return before the work is done).
void jmt_sync_features(jmt_features* f) {
  float first[1];
  std::vector<float> all(f->tensor->size());
  f->tensor->read(all.data());
  (void)first;
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Image {
  int w = 0, h = 0;
  unsigned char* pixels = nullptr;
  explicit Image(const std::string& path) {
    int n;
    pixels = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!pixels) throw std::runtime_error("cannot read image '" + path + "'");
  }
  ~Image() { stbi_image_free(pixels); }
  jmt_image view() const { return jmt_image{pixels, w, h, 0, JMT_FORMAT_RGB8}; }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "match") != 0) {
    usage();
    return 2;
  }
  try {
    std::string kpts_a, kpts_b, out_path, weights = "weights/flexglue.jmtw";
    std::vector<std::string> images;  // options may come before, between or after the two images
    jmt_config config;
    jmt_config_default(&config);
    jmt_match_options options;
    jmt_match_options_default(&options);
    bool dump_features = false;
    int runs = 1;
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      auto value = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--kpts-a") kpts_a = value();
      else if (a == "--kpts-b") kpts_b = value();
      else if (a == "--out") out_path = value();
      else if (a == "--weights") weights = value();
      else if (a == "--backend") {
        std::string b = value();
        if (b == "cpu") config.backend = JMT_BACKEND_CPU;
        else if (b == "metal") config.backend = JMT_BACKEND_METAL;
        else if (b == "metal:fp16") config.backend = JMT_BACKEND_METAL, config.precision = JMT_PRECISION_FP16;
        else if (b == "cuda") config.backend = JMT_BACKEND_CUDA;
        else if (b == "cuda:fp16") config.backend = JMT_BACKEND_CUDA, config.precision = JMT_PRECISION_FP16;
        else throw std::runtime_error("unknown backend " + b);
      }
      else if (a == "--runs") runs = std::stoi(value());
      else if (a == "--size") config.input_size = std::stoi(value());
      else if (a == "--layers") options.n_layers = std::stoi(value());
      else if (a == "--filter") options.filter_threshold = std::stof(value());
      else if (a == "--max-px") options.max_distance_px = std::stof(value());
      else if (a == "--min-covisibility") options.min_covisibility = std::stof(value());
      else if (a == "--early-stop") options.early_stop_decay = std::stof(value());
      else if (a == "--threads") config.num_threads = std::stoi(value());
      else if (a == "--dump-features") dump_features = true;
      else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a);
      else images.push_back(a);
    }
    if (images.size() != 2) throw std::runtime_error("two images are required");
    const std::string path_a = images[0], path_b = images[1];
    if (kpts_a.empty() || kpts_b.empty()) throw std::runtime_error("--kpts-a and --kpts-b are required");
    config.weights_path = weights.c_str();

    double t0 = now_ms();
    jmt_ctx* ctx = jmt_create(&config);
    if (!ctx) throw std::runtime_error(jmt_last_error());
    double t_load = now_ms() - t0;

    Image image_a(path_a), image_b(path_b);
    jmt::Weights file_a = jmt::Weights::from_file(kpts_a), file_b = jmt::Weights::from_file(kpts_b);
    const jmt::Param &ka = file_a.get("keypoints"), &kb = file_b.get("keypoints");
    const int na = static_cast<int>(ka.size() / 2), nb = static_cast<int>(kb.size() / 2);

    // With --runs N everything is run N times and the fastest run is reported (the first one compiles
    // shaders and fills caches on a GPU backend).
    jmt_image va = image_a.view(), vb = image_b.view();
    jmt_features *fa = nullptr, *fb = nullptr;
    jmt_matches m;
    double t_describe = 1e30, t_match = 1e30;
    for (int run = 0; run < std::max(1, runs); ++run) {
      jmt_features_free(fa), jmt_features_free(fb);
      t0 = now_ms();
      fa = jmt_describe(ctx, &va);
      fb = fa ? jmt_describe(ctx, &vb) : nullptr;
      if (!fa || !fb) throw std::runtime_error(jmt_last_error());
      jmt_sync_features(fb);
      t_describe = std::min(t_describe, (now_ms() - t0) / 2);
      t0 = now_ms();
      if (jmt_match(ctx, fa, ka.data, na, fb, kb.data, nb, &options, &m) != 0) throw std::runtime_error(jmt_last_error());
      t_match = std::min(t_match, now_ms() - t0);
    }

    std::printf("%d matches from %d and %d keypoints, covisibility %.4f, %d layers\n", m.n, na, nb, m.covisibility,
                m.stop_layer);
    std::printf("load %.0f ms, describe %.0f ms per image, match %.0f ms\n", t_load, t_describe, t_match);

    if (!out_path.empty()) {
      std::vector<float> idx(m.idx, m.idx + 2 * static_cast<size_t>(m.n));
      float covis = m.covisibility, layer = static_cast<float>(m.stop_layer);
      std::vector<jmt::NamedTensor> tensors = {
          {"matches", {m.n, 2}, idx.data()}, {"scores", {m.n}, m.score},
          {"a_in_b", {na, 2}, m.a_in_b},     {"b_in_a", {nb, 2}, m.b_in_a},
          {"conf_a", {na}, m.conf_a},        {"conf_b", {nb}, m.conf_b},
          {"covisibility", {1}, &covis},     {"stop_layer", {1}, &layer}};
      std::vector<float> da, db;
      if (dump_features) {
        da.resize(fa->tensor->size()), db.resize(fb->tensor->size());
        fa->tensor->read(da.data()), fb->tensor->read(db.data());
        tensors.push_back({"features_a", {fa->tensor->h, fa->tensor->w, fa->tensor->c}, da.data()});
        tensors.push_back({"features_b", {fb->tensor->h, fb->tensor->w, fb->tensor->c}, db.data()});
      }
      jmt::write_container(out_path, {{"kind", "matches"}, {"model", "flexglue"}}, tensors);
    }
    jmt_features_free(fa), jmt_features_free(fb);
    jmt_destroy(ctx);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
