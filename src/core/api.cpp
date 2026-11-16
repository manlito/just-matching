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

struct jmt_ctx {
  jmt::Weights weights;
  std::unique_ptr<jmt::FlexGlue> model;
  std::unique_ptr<jmt::Ops> ops;
  int input_size = 512;
  jmt::MatchResult result;
  std::vector<int32_t> idx;
  std::vector<float> score, a_in_b, b_in_a;
};

namespace {
thread_local std::string g_error;

// To FlexGlue's convention ((0,0) = corner of the top-left pixel) and back.
std::vector<float> shifted(const float* xy, int n, float by) {
  std::vector<float> out(static_cast<size_t>(n) * 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = xy[i] + by;
  return out;
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
    std::vector<float> rgb = jmt::to_rgb01(*image);
    auto f = std::make_unique<jmt_features>();
    f->tensor = ctx->model->describe(*ctx->ops, rgb.data(), image->height, image->width, ctx->input_size);
    f->width = image->width, f->height = image->height;
    return f.release();
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

void jmt_features_free(jmt_features* features) { delete features; }

int jmt_match(jmt_ctx* ctx, const jmt_features* a, const float* xy_a, int n_a, const jmt_features* b,
              const float* xy_b, int n_b, const jmt_match_options* options, jmt_matches* m) {
  try {
    if (!ctx || !a || !b || !m || n_a < 0 || n_b < 0 || (n_a && !xy_a) || (n_b && !xy_b))
      throw std::runtime_error("bad argument");
    jmt::MatchOptions o;
    if (options) {
      o.n_layers = options->n_layers;
      o.filter_threshold = options->filter_threshold;
      o.max_distance_px = options->max_distance_px;
      o.min_covisibility = options->min_covisibility;
      o.early_stop_decay = options->early_stop_decay;
    }
    std::vector<float> ka = shifted(xy_a, n_a, 0.5f), kb = shifted(xy_b, n_b, 0.5f);
    jmt::MatchResult& r = ctx->result;
    ctx->model->match(*ctx->ops, a->tensor, ka.data(), n_a, static_cast<float>(a->width),
                      static_cast<float>(a->height), b->tensor, kb.data(), n_b, static_cast<float>(b->width),
                      static_cast<float>(b->height), o, &r);
    ctx->idx.clear(), ctx->score.clear();
    for (int i = 0; i < n_a; ++i)
      if (r.matches0[i] >= 0) {
        ctx->idx.push_back(i), ctx->idx.push_back(r.matches0[i]);
        ctx->score.push_back(r.scores0[i]);
      }
    ctx->a_in_b = shifted(r.p2d0_i1.data(), n_a, -0.5f);
    ctx->b_in_a = shifted(r.p2d1_i0.data(), n_b, -0.5f);
    m->n = static_cast<int>(ctx->score.size());
    m->idx = ctx->idx.data(), m->score = ctx->score.data();
    m->n_a = n_a, m->n_b = n_b;
    m->a_in_b = ctx->a_in_b.data(), m->b_in_a = ctx->b_in_a.data();
    m->conf_a = r.confidence0.data(), m->conf_b = r.confidence1.data();
    m->covisibility = r.covisibility, m->stop_layer = r.stop_layer;
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

void jmt_destroy(jmt_ctx* ctx) { delete ctx; }

const char* jmt_last_error(void) { return g_error.c_str(); }

}  // extern "C"
