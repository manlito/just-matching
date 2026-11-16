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

// What the arrays of one jmt_matches point into. Kept by the context and reused, so matching allocates
// nothing once the arrays have reached their size.
struct jmt_result {
  jmt::MatchResult result;
  std::vector<int32_t> idx;
  std::vector<float> score, a_in_b, b_in_a, xy_a, xy_b;
};

struct jmt_ctx {
  jmt::Weights weights;
  std::unique_ptr<jmt::FlexGlue> model;
  std::unique_ptr<jmt::Ops> ops;
  int input_size = 512;
  std::deque<jmt_result> results;  // one per pair of the last call; a deque keeps the addresses stable
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

FlexGlue& model_of(jmt_ctx* ctx) { return *ctx->model; }
Ops& ops_of(jmt_ctx* ctx) { return *ctx->ops; }

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
}

int jmt_backend_available(jmt_backend backend) {
#ifdef JMT_HAVE_METAL
  if (backend == JMT_BACKEND_METAL) return 1;
#endif
#ifdef JMT_HAVE_CUDA
  if (backend == JMT_BACKEND_CUDA) return 1;
#endif
  return backend == JMT_BACKEND_CPU;
}

jmt_ctx* jmt_create(const jmt_config* config) {
  try {
    if (!config) throw std::runtime_error("config is null");
    auto ctx = std::make_unique<jmt_ctx>();
    if (config->weights_data && config->weights_size)
      ctx->weights = jmt::Weights::from_memory(config->weights_data, config->weights_size);
    else if (config->weights_path)
      ctx->weights = jmt::Weights::from_file(config->weights_path);
    else
      throw std::runtime_error("no weights given (weights_path or weights_data)");
    ctx->model = std::make_unique<jmt::FlexGlue>(ctx->weights);
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

void jmt_features_free(jmt_features* features) { delete features; }

int jmt_match_batch(jmt_ctx* ctx, const jmt_pair* pairs, int n_pairs, const jmt_match_options* options, jmt_matches* matches) {
  try {
    if (!ctx || n_pairs < 0 || (n_pairs && (!pairs || !matches))) throw std::runtime_error("bad argument");
    for (int i = 0; i < n_pairs; ++i) {
      const jmt_pair& p = pairs[i];
      if (!p.a || !p.b || p.n_a < 0 || p.n_b < 0 || (p.n_a && !p.xy_a) || (p.n_b && !p.xy_b)) throw std::runtime_error("bad argument");
    }
    const jmt::MatchOptions o = to_options(options);
    if (static_cast<int>(ctx->results.size()) < n_pairs) ctx->results.resize(n_pairs);
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

void jmt_destroy(jmt_ctx* ctx) { delete ctx; }

const char* jmt_last_error(void) { return g_error.c_str(); }

}  // extern "C"
