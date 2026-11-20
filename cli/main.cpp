// jmt: command line front end of the just-matching library.
//
//   jmt match A B --kpts-a A.jkpt --kpts-b B.jkpt [--out OUT.jmtt] [options]
//   jmt match --model lightglue-aliked --kpts-a A.jkpt --kpts-b B.jkpt [--out OUT.jmtt] [options]
//   jmt shortlist IMAGE... [--out OUT.jmtt] [options]
//   jmt convert IN OUT.jmtw [--fp16]        a checkpoint as released (.pth) or a .jmtw file -> .jmtw
//
// A, B: images (PNG, JPEG, ...). The keypoint files are just-keypoints outputs (`jkp extract --out`), or any
// container with a float32 tensor "keypoints" (N, 2) in pixels with (0,0) at the centre of the top-left pixel.
// LightGlue reads no images: it needs "descriptors" (N, 128) and the image size (metadata width and height,
// as just-keypoints writes them) from the keypoint files. Its a_in_b and b_in_a hold the matched keypoints'
// positions (NaN when unmatched), conf_a and conf_b the matchability, covisibility -1.
// OUT holds matches (K, 2: index in A, index in B), scores (K), a_in_b (Na, 2), b_in_a (Nb, 2), conf_a (Na),
// conf_b (Nb), covisibility (1), stop_layer (1), all float32; with --dump-features also features_a and
// features_b (h, w, 256).
// The shortlist file holds pairs (K, 2: image indices), scores (K) and level1 (N, N), all float32.
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
               "       jmt match --model lightglue-aliked --kpts-a A.jkpt --kpts-b B.jkpt [--out OUT.jmtt] [options]\n"
               "  --model M              flexglue (default), lightglue-aliked, lightglue-raco-aliked or lightglue-xfeat\n"
               "  --backend B            cpu (default), metal, metal:fp16, cuda, cuda:fp16,\n"
               "                         webgpu, webgpu:fp16\n"
               "  --weights FILE         a checkpoint as released (.pth) or a .jmtw file (default weights/flexglue.jmtw;\n"
               "                         LightGlue: weights/<model>.jmtw if it exists, else the built-in weights;\n"
               "                         builtin: the built-in weights)\n"
               "  --size N               network input side, multiple of 32 (default 512)\n"
               "  --layers N             transformer layers, 0..9 (default 9)\n"
               "  --filter T             minimum match score (default 0.1)\n"
               "  --max-px D             maximum transfer error in network-input pixels (default 3; 0 = off)\n"
               "  --min-covisibility S   reject pairs scoring below S (default 0 = off)\n"
               "  --early-stop D         stop early on non-covisible pairs, decay D in [0,1) (default off)\n"
               "  --depth-confidence C   LightGlue: stop when more than C of the keypoints are confident\n"
               "                         (default 0.95; 0 = every layer)\n"
               "  --width-confidence C   LightGlue: drop confident keypoints with matchability <= 1 - C\n"
               "                         (default 0.99; 0 = keep all)\n"
               "  --threads N            CPU threads (default all)\n"
               "  --dump-features        also write the dense features to OUT\n"
               "  --runs N               repeat N times and report the fastest run (default 1)\n"
               "  --batch N              also time N copies of the pair through jmt_match_batch\n"
               "usage: jmt shortlist IMAGE... [--out OUT.jmtt] [options]\n"
               "  --k K                  partners kept per image by level 1 (default 10)\n"
               "  --neighbours N         also keep the N next images in the order given (default 0)\n"
               "  --rerank               score the kept pairs with level 2 (layer-0 grid round trip)\n"
               "  --min-score S          with --rerank, drop pairs scoring below S (default 0)\n"
               "  --backend, --weights, --size, --threads as for match\n"
               "usage: jmt convert IN OUT.jmtw [--fp16]\n"
               "  IN: a checkpoint as released (flexglue.pth, aliked_lightglue.pth, raco_aliked_lightglue.pth) or a\n"
               "  .jmtw file; OUT gets the tensors the library reads, as float32 or with --fp16 as float16\n");
}

bool file_exists(const std::string& path) {
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    std::fclose(f);
    return true;
  }
  return false;
}

double now_ms();

// jmt match --model lightglue-*: keypoints, descriptors and image sizes all come from the keypoint files.
int match_lightglue(jmt_config config, std::string weights, const std::string& kpts_a, const std::string& kpts_b,
                    const std::string& out_path, const jmt_match_options& options, int runs) {
  if (weights.empty() && file_exists("weights/" + std::string(config.model) + ".jmtw"))
    weights = "weights/" + std::string(config.model) + ".jmtw";
  if (weights == "builtin") weights.clear();
  config.weights_path = weights.empty() ? nullptr : weights.c_str();  // none: the built-in weights
  double t0 = now_ms();
  jmt_ctx* ctx = jmt_create(&config);
  if (!ctx) throw std::runtime_error(jmt_last_error());
  const double t_load = now_ms() - t0;
  jmt::Weights file_a = jmt::Weights::from_file(kpts_a), file_b = jmt::Weights::from_file(kpts_b);
  const jmt::Param &ka = file_a.get("keypoints"), &kb = file_b.get("keypoints");
  const jmt::Param &da = file_a.get("descriptors"), &db = file_b.get("descriptors");
  const int na = static_cast<int>(ka.size() / 2), nb = static_cast<int>(kb.size() / 2);
  const int dim = da.shape.size() == 2 ? da.shape[1] : 0;
  jmt_features *fa = nullptr, *fb = nullptr;
  jmt_matches m;
  double t_describe = 1e30, t_match = 1e30;
  for (int run = 0; run < std::max(1, runs); ++run) {
    jmt_features_free(fa), jmt_features_free(fb);
    t0 = now_ms();
    fa = jmt_describe_keypoints(ctx, file_a.meta_int("width"), file_a.meta_int("height"), ka.data, da.data, na, dim);
    fb = fa ? jmt_describe_keypoints(ctx, file_b.meta_int("width"), file_b.meta_int("height"), kb.data, db.data, nb, dim)
            : nullptr;
    if (!fa || !fb) throw std::runtime_error(jmt_last_error());
    t_describe = std::min(t_describe, (now_ms() - t0) / 2);
    t0 = now_ms();
    if (jmt_match(ctx, fa, ka.data, na, fb, kb.data, nb, &options, &m) != 0) throw std::runtime_error(jmt_last_error());
    t_match = std::min(t_match, now_ms() - t0);
  }
  std::printf("%d matches from %d and %d keypoints, %d layers\n", m.n, na, nb, m.stop_layer);
  std::printf("load %.0f ms, keypoints %.1f ms per image, match %.0f ms\n", t_load, t_describe, t_match);
  if (!out_path.empty()) {
    std::vector<float> idx(m.idx, m.idx + 2 * static_cast<size_t>(m.n));
    float covis = m.covisibility, layer = static_cast<float>(m.stop_layer);
    jmt::write_container(out_path, {{"kind", "matches"}, {"model", config.model}},
                         {{"matches", {m.n, 2}, idx.data()}, {"scores", {m.n}, m.score},
                          {"a_in_b", {na, 2}, m.a_in_b}, {"b_in_a", {nb, 2}, m.b_in_a},
                          {"conf_a", {na}, m.conf_a}, {"conf_b", {nb}, m.conf_b},
                          {"covisibility", {1}, &covis}, {"stop_layer", {1}, &layer}});
  }
  jmt_features_free(fa), jmt_features_free(fb);
  jmt_destroy(ctx);
  return 0;
}

void set_backend(const std::string& b, jmt_config* config) {
  if (b == "cpu") config->backend = JMT_BACKEND_CPU;
  else if (b == "metal") config->backend = JMT_BACKEND_METAL;
  else if (b == "metal:fp16") config->backend = JMT_BACKEND_METAL, config->precision = JMT_PRECISION_FP16;
  else if (b == "cuda") config->backend = JMT_BACKEND_CUDA;
  else if (b == "cuda:fp16") config->backend = JMT_BACKEND_CUDA, config->precision = JMT_PRECISION_FP16;
  else if (b == "webgpu") config->backend = JMT_BACKEND_WEBGPU;
  else if (b == "webgpu:fp16") config->backend = JMT_BACKEND_WEBGPU, config->precision = JMT_PRECISION_FP16;
  else throw std::runtime_error("unknown backend " + b);
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

// jmt shortlist: describe every image, then the pairs worth matching.
int shortlist(int argc, char** argv) {
  std::string out_path, weights = "weights/flexglue.jmtw";
  std::vector<std::string> paths;
  jmt_config config;
  jmt_config_default(&config);
  jmt_shortlist_options options;
  jmt_shortlist_options_default(&options);
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--out") out_path = value();
    else if (a == "--weights") weights = value();
    else if (a == "--backend") set_backend(value(), &config);
    else if (a == "--size") config.input_size = std::stoi(value());
    else if (a == "--threads") config.num_threads = std::stoi(value());
    else if (a == "--k") options.k = std::stoi(value());
    else if (a == "--neighbours") options.neighbours = std::stoi(value());
    else if (a == "--rerank") options.rerank = 1;
    else if (a == "--min-score") options.min_score = std::stof(value());
    else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a);
    else paths.push_back(a);
  }
  const int n = static_cast<int>(paths.size());
  if (n < 2) throw std::runtime_error("at least two images are required");
  config.weights_path = weights.c_str();
  jmt_ctx* ctx = jmt_create(&config);
  if (!ctx) throw std::runtime_error(jmt_last_error());
  std::vector<jmt_features*> features;
  double t0 = now_ms();
  for (const std::string& p : paths) {
    Image image(p);
    jmt_image view = image.view();
    features.push_back(jmt_describe(ctx, &view));
    if (!features.back()) throw std::runtime_error(jmt_last_error());
  }
  jmt::ops_of(ctx).finish();
  const double t_describe = now_ms() - t0;
  t0 = now_ms();
  std::vector<float> level1(static_cast<size_t>(n) * n, 1.0f);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      float s;
      if (jmt_covisibility(ctx, features[i], features[j], 1, &s)) throw std::runtime_error(jmt_last_error());
      level1[static_cast<size_t>(i) * n + j] = level1[static_cast<size_t>(j) * n + i] = s;
    }
  const double t_level1 = now_ms() - t0;
  t0 = now_ms();
  jmt_pair_list list;
  if (jmt_shortlist(ctx, features.data(), n, &options, &list)) throw std::runtime_error(jmt_last_error());
  const double t_shortlist = now_ms() - t0;
  for (int p = 0; p < list.n; ++p)
    std::printf("%s %s %.4f\n", paths[list.idx[2 * p]].c_str(), paths[list.idx[2 * p + 1]].c_str(), list.score[p]);
  std::printf("%d of %d pairs; describe %.0f ms per image, level 1 for all pairs %.1f ms, shortlist %.1f ms",
              list.n, n * (n - 1) / 2, t_describe / n, t_level1, t_shortlist);
  if (options.rerank && list.n) std::printf(" (%.2f ms per reranked pair, with the first use of each image)", t_shortlist / list.n);
  std::printf("\n");
  if (!out_path.empty()) {
    std::vector<float> idx(list.idx, list.idx + 2 * static_cast<size_t>(list.n));
    jmt::write_container(out_path, {{"kind", "shortlist"}, {"model", "flexglue"}},
                         {{"pairs", {list.n, 2}, idx.data()}, {"scores", {list.n}, list.score},
                          {"level1", {n, n}, level1.data()}});
  }
  for (jmt_features* f : features) jmt_features_free(f);
  jmt_destroy(ctx);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "convert") == 0) {
    try {
      std::vector<std::string> paths;
      bool fp16 = false;
      for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fp16") == 0) fp16 = true;
        else if (argv[i][0] == '-' && argv[i][1] == '-') throw std::runtime_error(std::string("unknown option ") + argv[i]);
        else paths.push_back(argv[i]);
      }
      if (paths.size() != 2) throw std::runtime_error("convert: IN and OUT are required");
      const jmt::Weights w = jmt::Weights::from_file(paths[0]);
      size_t values = 0;
      const auto names = w.names();
      for (const std::string& name : names) values += w.get(name).size();
      const std::string out = jmt::convert_weights(w, fp16);
      std::FILE* f = std::fopen(paths[1].c_str(), "wb");
      bool ok = f && std::fwrite(out.data(), 1, out.size(), f) == out.size();
      if (f) ok = std::fclose(f) == 0 && ok;
      if (!ok) throw std::runtime_error("cannot write '" + paths[1] + "'");
      std::printf("%s: %s, %zu tensors, %zu values, %s\n", paths[1].c_str(), w.meta("model").c_str(), names.size(), values,
                  fp16 ? "float16" : "float32");
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
    return 0;
  }
  if (argc >= 2 && std::strcmp(argv[1], "shortlist") == 0) {
    try {
      return shortlist(argc, argv);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc < 2 || std::strcmp(argv[1], "match") != 0) {
    usage();
    return 2;
  }
  try {
    std::string kpts_a, kpts_b, out_path, weights, model = "flexglue";
    std::vector<std::string> images;  // options may come before, between or after the two images
    jmt_config config;
    jmt_config_default(&config);
    jmt_match_options options;
    jmt_match_options_default(&options);
    bool dump_features = false;
    int runs = 1, batch = 0;
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
      else if (a == "--model") model = value();
      else if (a == "--backend") set_backend(value(), &config);
      else if (a == "--runs") runs = std::stoi(value());
      else if (a == "--batch") batch = std::stoi(value());
      else if (a == "--size") config.input_size = std::stoi(value());
      else if (a == "--layers") options.n_layers = std::stoi(value());
      else if (a == "--filter") options.filter_threshold = std::stof(value());
      else if (a == "--max-px") options.max_distance_px = std::stof(value());
      else if (a == "--min-covisibility") options.min_covisibility = std::stof(value());
      else if (a == "--early-stop") options.early_stop_decay = std::stof(value());
      else if (a == "--depth-confidence") options.depth_confidence = std::stof(value());
      else if (a == "--width-confidence") options.width_confidence = std::stof(value());
      else if (a == "--threads") config.num_threads = std::stoi(value());
      else if (a == "--dump-features") dump_features = true;
      else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a);
      else images.push_back(a);
    }
    if (kpts_a.empty() || kpts_b.empty()) throw std::runtime_error("--kpts-a and --kpts-b are required");
    config.model = model.c_str();
    if (model != "flexglue") return match_lightglue(config, weights, kpts_a, kpts_b, out_path, options, runs);
    if (images.size() != 2) throw std::runtime_error("two images are required");
    const std::string path_a = images[0], path_b = images[1];
    if (weights.empty()) weights = "weights/flexglue.jmtw";
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
      // GPU backends return before the work is done: wait, so the time is charged to describe. Reading
      // the features back also waits on a backend that does not implement finish.
      jmt::ops_of(ctx).finish();
      std::vector<float> wait(fb->tensor->size());
      fa->tensor->read(wait.data()), fb->tensor->read(wait.data());
      t_describe = std::min(t_describe, (now_ms() - t0) / 2);
      t0 = now_ms();
      if (jmt_match(ctx, fa, ka.data, na, fb, kb.data, nb, &options, &m) != 0) throw std::runtime_error(jmt_last_error());
      t_match = std::min(t_match, now_ms() - t0);
    }

    std::printf("%d matches from %d and %d keypoints, covisibility %.4f, %d layers\n", m.n, na, nb, m.covisibility,
                m.stop_layer);
    std::printf("load %.0f ms, describe %.0f ms per image, match %.0f ms\n", t_load, t_describe, t_match);
    if (batch > 0) {
      std::vector<jmt_pair> pairs(batch, jmt_pair{fa, ka.data, na, fb, kb.data, nb});
      std::vector<jmt_matches> all(batch);
      double t_batch = 1e30;
      for (int run = 0; run < std::max(1, runs); ++run) {
        t0 = now_ms();
        if (jmt_match_batch(ctx, pairs.data(), batch, &options, all.data()) != 0) throw std::runtime_error(jmt_last_error());
        t_batch = std::min(t_batch, (now_ms() - t0) / batch);
      }
      for (const jmt_matches& b : all)
        if (b.n != all[0].n) throw std::runtime_error("batch: results differ between copies of the pair");
      std::printf("batch of %d: %.1f ms per pair, %d matches each\n", batch, t_batch, all[0].n);
      m = all[0];
    }

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
