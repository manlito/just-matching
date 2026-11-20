#include <algorithm>
#include <cmath>
#include <deque>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/api_internal.h"
#include "core/ops.h"
#include "core/weights.h"
#include "jmt/jmt.h"
#include "models/flexglue/flexglue.h"
#include "models/lightglue/lightglue.h"

// What the arrays of one jmt_matches point into. Kept by the context and reused, so matching allocates
// nothing once the arrays have reached their size.
struct jmt_result {
  jmt::MatchResult result;
  jmt::LightGlue::Result lightglue;
  std::vector<int32_t> idx;
  std::vector<float> score, a_in_b, b_in_a, xy_a, xy_b;
};

struct jmt_ctx {
  jmt::Weights weights;
  std::unique_ptr<jmt::FlexGlue> model;         // one of the two
  std::unique_ptr<jmt::LightGlue> lightglue;
  std::unique_ptr<jmt::Ops> ops;
  int input_size = 512;
  std::deque<jmt_result> results;  // one per pair of the last call; a deque keeps the addresses stable
  std::vector<int32_t> shortlist_idx;  // what the last jmt_pair_list points into
  std::vector<float> shortlist_score;
};

namespace {
thread_local std::string g_error;

// To FlexGlue's convention ((0,0) = corner of the top-left pixel) and back.
void shifted(const float* xy, int n, float by, std::vector<float>* out) {
  out->resize(static_cast<size_t>(n) * 2);
  for (size_t i = 0; i < out->size(); ++i) (*out)[i] = xy[i] + by;
}

jmt::MatchOptions to_options(const jmt_match_options* options) {
  jmt::MatchOptions o;
  if (options) {
    o.n_layers = options->n_layers;
    o.filter_threshold = options->filter_threshold;
    o.max_distance_px = options->max_distance_px;
    o.min_covisibility = options->min_covisibility;
    o.early_stop_decay = options->early_stop_decay;
  }
  return o;
}

// Pairs in flight on the device ahead of the one the host is finishing.
const int kPipeline = 4;

void need_flexglue(jmt_ctx* ctx, const char* what) {
  if (!ctx->model) throw std::runtime_error(std::string(what) + " needs FlexGlue; this context runs LightGlue");
}

const char* const kLightGlueModels[] = {"lightglue-aliked", "lightglue-raco-aliked", "lightglue-xfeat"};

bool is_lightglue_model(const std::string& m) {
  return std::find(std::begin(kLightGlueModels), std::end(kLightGlueModels), m) != std::end(kLightGlueModels);
}

// LightGlue's results in the jmt_matches layout: positions are the matched keypoints', NaN when unmatched.
void lightglue_matches(const jmt_pair& p, jmt_result& r, jmt_matches* m) {
  const jmt::LightGlue::Result& lg = r.lightglue;
  r.idx.clear(), r.score.clear();
  r.a_in_b.assign(static_cast<size_t>(p.n_a) * 2, NAN), r.b_in_a.assign(static_cast<size_t>(p.n_b) * 2, NAN);
  for (int i = 0; i < p.n_a; ++i) {
    const int j = lg.matches0[i];
    if (j < 0) continue;
    r.idx.push_back(i), r.idx.push_back(j);
    r.score.push_back(lg.scores0[i]);
    r.a_in_b[2 * i] = p.xy_b[2 * j], r.a_in_b[2 * i + 1] = p.xy_b[2 * j + 1];
    r.b_in_a[2 * j] = p.xy_a[2 * i], r.b_in_a[2 * j + 1] = p.xy_a[2 * i + 1];
  }
  m->n = static_cast<int>(r.score.size());
  m->idx = r.idx.data(), m->score = r.score.data();
  m->n_a = p.n_a, m->n_b = p.n_b;
  m->a_in_b = r.a_in_b.data(), m->b_in_a = r.b_in_a.data();
  m->conf_a = lg.matchability0.data(), m->conf_b = lg.matchability1.data();
  m->covisibility = -1.0f, m->stop_layer = lg.stop_layer;
}

const std::vector<float>& global_of(jmt_ctx* ctx, const jmt_features* f) {
  need_flexglue(ctx, "covisibility");
  if (f->global.empty()) f->global = ctx->model->global_vector(f->tensor);
  return f->global;
}

jmt::FlexGlue::Image0 image0_of(jmt_ctx* ctx, const jmt_features* f) {
  need_flexglue(ctx, "covisibility");
  if (!f->search0) f->search0 = ctx->model->search0(*ctx->ops, f->tensor);
  return {f->tensor, f->search0};
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0;
  for (size_t k = 0; k < a.size(); ++k) s += static_cast<double>(a[k]) * b[k];
  return static_cast<float>(s);
}
}  // namespace

namespace jmt {

std::vector<float> to_rgb01(const jmt_image& image) {
  if (!image.data || image.width <= 0 || image.height <= 0) throw std::runtime_error("empty image");
  const int ch = image.format == JMT_FORMAT_GRAY8 ? 1 : image.format == JMT_FORMAT_RGB8 ? 3 : 4;
  const int stride = image.stride ? image.stride : image.width * ch;
  std::vector<float> rgb(static_cast<size_t>(image.width) * image.height * 3);
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < image.width; ++x) {
      const uint8_t* p = image.data + static_cast<size_t>(y) * stride + static_cast<size_t>(x) * ch;
      float* o = rgb.data() + (static_cast<size_t>(y) * image.width + x) * 3;
      for (int c = 0; c < 3; ++c) o[c] = static_cast<float>(p[ch == 1 ? 0 : c]) / 255.0f;
    }
  return rgb;
}

FlexGlue& model_of(jmt_ctx* ctx) {
  need_flexglue(ctx, "this");
  return *ctx->model;
}
Ops& ops_of(jmt_ctx* ctx) { return *ctx->ops; }
bool is_lightglue(const jmt_ctx* ctx) { return ctx->lightglue != nullptr; }

}  // namespace jmt

extern "C" {

void jmt_config_default(jmt_config* config) {
  *config = jmt_config{};
  config->backend = JMT_BACKEND_CPU;
  config->input_size = 512;
}

void jmt_match_options_default(jmt_match_options* o) {
  jmt::MatchOptions d;
  o->n_layers = d.n_layers;
  o->filter_threshold = d.filter_threshold;
  o->max_distance_px = d.max_distance_px;
  o->min_covisibility = d.min_covisibility;
  o->early_stop_decay = d.early_stop_decay;
  const jmt::LightGlue::Options l;
  o->depth_confidence = l.depth_confidence;
  o->width_confidence = l.width_confidence;
}

int jmt_backend_available(jmt_backend backend) {
#ifdef JMT_HAVE_METAL
  if (backend == JMT_BACKEND_METAL) return 1;
#endif
#ifdef JMT_HAVE_CUDA
  if (backend == JMT_BACKEND_CUDA) return 1;
#endif
#ifdef JMT_HAVE_WEBGPU
  if (backend == JMT_BACKEND_WEBGPU) return 1;
#endif
  return backend == JMT_BACKEND_CPU;
}

jmt_ctx* jmt_create(const jmt_config* config) {
  try {
    if (!config) throw std::runtime_error("config is null");
    auto ctx = std::make_unique<jmt_ctx>();
    const std::string model = config->model ? config->model : "";
    if (!model.empty() && model != "flexglue" && !is_lightglue_model(model))
      throw std::runtime_error("unknown model '" + model + "' (flexglue, lightglue-aliked, lightglue-raco-aliked, lightglue-xfeat)");
    const void* blob = nullptr;
    size_t blob_size = 0;
    if (config->weights_data && config->weights_size) {
      ctx->weights = jmt::Weights::from_memory(config->weights_data, config->weights_size);
    } else if (config->weights_path) {
      ctx->weights = jmt::Weights::from_file(config->weights_path);
    } else if (is_lightglue_model(model) && jmt::embedded_weights(model, &blob, &blob_size)) {
      ctx->weights = jmt::Weights::from_memory(blob, blob_size);
    } else if (is_lightglue_model(model)) {
      throw std::runtime_error("no weights given, and " + model + " is not built into this library "
                               "(JMT_EMBED_WEIGHTS): set weights_path to " + model + ".jmtw or the released .pth");
    } else {
      throw std::runtime_error("no weights given (weights_path or weights_data)");
    }
    const std::string family = ctx->weights.meta("model");
    if (!model.empty() && family != (model == "flexglue" ? "flexglue" : "lightglue"))
      throw std::runtime_error("the weights are " + family + "'s, not " + model + "'s");
    if (family == "lightglue") ctx->lightglue = std::make_unique<jmt::LightGlue>(ctx->weights);
    else ctx->model = std::make_unique<jmt::FlexGlue>(ctx->weights);
    if (!jmt_backend_available(config->backend)) throw std::runtime_error("backend not available in this build");
    const bool fp16 = config->precision == JMT_PRECISION_FP16;
    if (fp16 && config->backend == JMT_BACKEND_CPU) throw std::runtime_error("the cpu backend runs in fp32 only");
    if (config->backend == JMT_BACKEND_CPU) ctx->ops = jmt::make_cpu_ops(config->num_threads);
#ifdef JMT_HAVE_METAL
    if (config->backend == JMT_BACKEND_METAL) ctx->ops = jmt::make_metal_ops(config->device, fp16);
#endif
#ifdef JMT_HAVE_CUDA
    if (config->backend == JMT_BACKEND_CUDA) ctx->ops = jmt::make_cuda_ops(config->device, fp16);
#endif
#ifdef JMT_HAVE_WEBGPU
    if (config->backend == JMT_BACKEND_WEBGPU) ctx->ops = jmt::make_webgpu_ops(fp16);
#endif
    ctx->input_size = config->input_size > 0 ? config->input_size : 512;
    if (ctx->input_size % 32 != 0) throw std::runtime_error("input_size must be a multiple of 32");
    return ctx.release();
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

jmt_features* jmt_describe(jmt_ctx* ctx, const jmt_image* image) {
  try {
    if (!ctx || !image) throw std::runtime_error("null argument");
    if (!ctx->model) throw std::runtime_error("LightGlue matches keypoints: use jmt_describe_keypoints");
    if (!image->data || image->width <= 0 || image->height <= 0) throw std::runtime_error("empty image");
    const int ch = image->format == JMT_FORMAT_GRAY8 ? 1 : image->format == JMT_FORMAT_RGB8 ? 3 : 4;
    const int stride = image->stride ? image->stride : image->width * ch;
    auto f = std::make_unique<jmt_features>();
    f->tensor = ctx->model->describe_u8(*ctx->ops, image->data, image->height, image->width, stride, ch, ctx->input_size);
    f->width = image->width, f->height = image->height;
    return f.release();
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

jmt_features* jmt_describe_keypoints(jmt_ctx* ctx, int width, int height, const float* xy, const float* desc, int n,
                                     int dim) {
  try {
    if (!ctx || n < 0 || (n && (!xy || !desc))) throw std::runtime_error("bad argument");
    if (!ctx->lightglue) throw std::runtime_error("FlexGlue describes images: use jmt_describe");
    if (width <= 0 || height <= 0) throw std::runtime_error("image size must be positive");
    if (dim != ctx->lightglue->input_dim())
      throw std::runtime_error("descriptors must have " + std::to_string(ctx->lightglue->input_dim()) + " values");
    auto f = std::make_unique<jmt_features>();
    f->keypoints = std::make_shared<jmt::LightGlue::Image>(
        ctx->lightglue->prepare(*ctx->ops, xy, desc, n, static_cast<float>(width), static_cast<float>(height)));
    f->width = width, f->height = height;
    return f.release();
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

void jmt_features_free(jmt_features* features) { delete features; }

int jmt_match_batch(jmt_ctx* ctx, const jmt_pair* pairs, int n_pairs, const jmt_match_options* options, jmt_matches* matches) {
  try {
    if (!ctx || n_pairs < 0 || (n_pairs && (!pairs || !matches))) throw std::runtime_error("bad argument");
    for (int i = 0; i < n_pairs; ++i) {
      const jmt_pair& p = pairs[i];
      if (!p.a || !p.b || p.n_a < 0 || p.n_b < 0 || (p.n_a && !p.xy_a) || (p.n_b && !p.xy_b)) throw std::runtime_error("bad argument");
    }
    if (static_cast<int>(ctx->results.size()) < n_pairs) ctx->results.resize(n_pairs);
    if (ctx->lightglue) {
      jmt::LightGlue::Options lo;
      if (options) {
        lo.n_layers = options->n_layers, lo.filter_threshold = options->filter_threshold;
        lo.depth_confidence = options->depth_confidence, lo.width_confidence = options->width_confidence;
      }
      for (int i = 0; i < n_pairs; ++i) {
        const jmt_pair& p = pairs[i];
        if (!p.a->keypoints || !p.b->keypoints) throw std::runtime_error("features not made by jmt_describe_keypoints");
        if (p.n_a != p.a->keypoints->n || p.n_b != p.b->keypoints->n)
          throw std::runtime_error("keypoints differ from those given to jmt_describe_keypoints");
        ctx->lightglue->match(*ctx->ops, *p.a->keypoints, *p.b->keypoints, lo, &ctx->results[i].lightglue);
        lightglue_matches(p, ctx->results[i], matches + i);
      }
      return 0;
    }
    for (int i = 0; i < n_pairs; ++i)
      if (!pairs[i].a->tensor || !pairs[i].b->tensor) throw std::runtime_error("features not made by jmt_describe");
    const jmt::MatchOptions o = to_options(options);
    std::deque<std::shared_ptr<jmt::FlexGlue::Pair>> queued;
    for (int next = 0, done = 0; done < n_pairs; ++done) {
      // Keep the device busy: queue the pairs ahead before waiting for this one.
      for (; next < n_pairs && next < done + kPipeline; ++next) {
        const jmt_pair& p = pairs[next];
        jmt_result& r = ctx->results[next];
        shifted(p.xy_a, p.n_a, 0.5f, &r.xy_a), shifted(p.xy_b, p.n_b, 0.5f, &r.xy_b);
        queued.push_back(ctx->model->match_queue(*ctx->ops, p.a->tensor, r.xy_a.data(), p.n_a, static_cast<float>(p.a->width),
                                                 static_cast<float>(p.a->height), p.b->tensor, r.xy_b.data(), p.n_b,
                                                 static_cast<float>(p.b->width), static_cast<float>(p.b->height), o));
        ctx->ops->flush();
      }
      const jmt_pair& p = pairs[done];
      jmt_result& r = ctx->results[done];
      ctx->model->match_read(*queued.front(), &r.result);
      queued.pop_front();
      r.idx.clear(), r.score.clear();
      for (int i = 0; i < p.n_a; ++i)
        if (r.result.matches0[i] >= 0) {
          r.idx.push_back(i), r.idx.push_back(r.result.matches0[i]);
          r.score.push_back(r.result.scores0[i]);
        }
      shifted(r.result.p2d0_i1.data(), p.n_a, -0.5f, &r.a_in_b);
      shifted(r.result.p2d1_i0.data(), p.n_b, -0.5f, &r.b_in_a);
      jmt_matches* m = matches + done;
      m->n = static_cast<int>(r.score.size());
      m->idx = r.idx.data(), m->score = r.score.data();
      m->n_a = p.n_a, m->n_b = p.n_b;
      m->a_in_b = r.a_in_b.data(), m->b_in_a = r.b_in_a.data();
      m->conf_a = r.result.confidence0.data(), m->conf_b = r.result.confidence1.data();
      m->covisibility = r.result.covisibility, m->stop_layer = r.result.stop_layer;
    }
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

int jmt_match(jmt_ctx* ctx, const jmt_features* a, const float* xy_a, int n_a, const jmt_features* b,
              const float* xy_b, int n_b, const jmt_match_options* options, jmt_matches* m) {
  if (!m) {
    g_error = "bad argument";
    return -1;
  }
  const jmt_pair pair = {a, xy_a, n_a, b, xy_b, n_b};
  return jmt_match_batch(ctx, &pair, 1, options, m);
}

int jmt_covisibility(jmt_ctx* ctx, const jmt_features* a, const jmt_features* b, int level, float* score) {
  try {
    if (!ctx || !a || !b || !score) throw std::runtime_error("null argument");
    need_flexglue(ctx, "jmt_covisibility");
    if (level == 1) {
      *score = dot(global_of(ctx, a), global_of(ctx, b));
    } else if (level == 2) {
      *score = ctx->model->grid_scores(*ctx->ops, {{image0_of(ctx, a), image0_of(ctx, b)}})[0];
    } else {
      throw std::runtime_error("covisibility level must be 1 or 2 (level 3 is jmt_matches.covisibility)");
    }
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

void jmt_shortlist_options_default(jmt_shortlist_options* o) {
  *o = jmt_shortlist_options{};
  o->k = 10;
}

int jmt_shortlist(jmt_ctx* ctx, const jmt_features* const* images, int n, const jmt_shortlist_options* options,
                  jmt_pair_list* out) {
  try {
    if (!ctx || !out || n < 0 || (n && !images)) throw std::runtime_error("bad argument");
    need_flexglue(ctx, "jmt_shortlist");
    for (int i = 0; i < n; ++i)
      if (!images[i]) throw std::runtime_error("bad argument");
    jmt_shortlist_options o;
    jmt_shortlist_options_default(&o);
    if (options) o = *options;
    // Level 1 for every pair, then the k best partners of every image.
    std::vector<float> sim(static_cast<size_t>(n) * n, 0.0f);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        sim[static_cast<size_t>(i) * n + j] = sim[static_cast<size_t>(j) * n + i] =
            dot(global_of(ctx, images[i]), global_of(ctx, images[j]));
    std::vector<char> keep(static_cast<size_t>(n) * n, 0);
    std::vector<int> order;
    const int k = std::min(std::max(o.k, 0), n - 1);
    for (int i = 0; i < n && k > 0; ++i) {
      order.clear();
      for (int j = 0; j < n; ++j)
        if (j != i) order.push_back(j);
      const float* row = sim.data() + static_cast<size_t>(i) * n;
      std::partial_sort(order.begin(), order.begin() + k, order.end(),
                        [&](int x, int y) { return row[x] > row[y] || (row[x] == row[y] && x < y); });
      for (int r = 0; r < k; ++r) keep[static_cast<size_t>(std::min(i, order[r])) * n + std::max(i, order[r])] = 1;
    }
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j <= i + o.neighbours && j < n; ++j) keep[static_cast<size_t>(i) * n + j] = 1;
    std::vector<std::pair<int, int>> kept;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (keep[static_cast<size_t>(i) * n + j]) kept.emplace_back(i, j);
    std::vector<float> score(kept.size());
    if (o.rerank) {
      std::vector<std::pair<jmt::FlexGlue::Image0, jmt::FlexGlue::Image0>> pairs;
      pairs.reserve(kept.size());
      for (const auto& p : kept) pairs.emplace_back(image0_of(ctx, images[p.first]), image0_of(ctx, images[p.second]));
      score = ctx->model->grid_scores(*ctx->ops, pairs);
    } else {
      for (size_t p = 0; p < kept.size(); ++p) score[p] = sim[static_cast<size_t>(kept[p].first) * n + kept[p].second];
    }
    std::vector<size_t> rank;
    for (size_t p = 0; p < kept.size(); ++p)
      if (!o.rerank || score[p] >= o.min_score) rank.push_back(p);
    std::stable_sort(rank.begin(), rank.end(), [&](size_t x, size_t y) { return score[x] > score[y]; });
    ctx->shortlist_idx.clear(), ctx->shortlist_score.clear();
    for (size_t p : rank) {
      ctx->shortlist_idx.push_back(kept[p].first), ctx->shortlist_idx.push_back(kept[p].second);
      ctx->shortlist_score.push_back(score[p]);
    }
    out->n = static_cast<int>(ctx->shortlist_score.size());
    out->idx = ctx->shortlist_idx.data(), out->score = ctx->shortlist_score.data();
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

void jmt_destroy(jmt_ctx* ctx) { delete ctx; }

const char* jmt_last_error(void) { return g_error.c_str(); }

}  // extern "C"
