// Flat entry points of the WebAssembly module, over the C API (include/jmt/jmt.h). web/jmt.js wraps
// them; an app that links just-matching into its own module calls the C API directly instead.
//
// jmtw_create, jmtw_describe, jmtw_match, jmtw_finish, jmtw_covisibility_level, jmtw_level1_matrix and
// jmtw_shortlist wait for the GPU, so they are JSPI exports: in JavaScript they return promises, and the
// module suspends at each wait instead of blocking the page.
#include <emscripten/emscripten.h>

#include <cmath>
#include <cstdint>
#include <new>

#include "core/api_internal.h"
#include "core/ops.h"
#include "jmt/jmt.h"

namespace {
struct Handle {
  jmt_ctx* ctx = nullptr;
  jmt_matches result{};
  jmt_pair_list pairs{};
};
}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int jmtw_backend_available(int backend) {
  return jmt_backend_available(static_cast<jmt_backend>(backend));
}

// weights: a .jmtw file (float32 or float16) in module memory, copied by jmt_create. Returns null on
// failure (jmtw_error).
EMSCRIPTEN_KEEPALIVE Handle* jmtw_create(int backend, int precision, const void* weights, int weights_size, int input_size) {
  jmt_config config;
  jmt_config_default(&config);
  config.backend = static_cast<jmt_backend>(backend);
  config.precision = static_cast<jmt_precision>(precision);
  config.weights_data = weights;
  config.weights_size = static_cast<size_t>(weights_size);
  config.input_size = input_size;
  config.num_threads = 1;
  jmt_ctx* ctx = jmt_create(&config);
  if (!ctx) return nullptr;
  Handle* h = new (std::nothrow) Handle;
  if (!h) {
    jmt_destroy(ctx);
    return nullptr;
  }
  h->ctx = ctx;
  return h;
}

// rgba: width * height * 4 bytes. Returns the features, or null (jmtw_error). The work may still be
// running on the GPU when it returns; jmtw_finish waits for it.
EMSCRIPTEN_KEEPALIVE jmt_features* jmtw_describe(Handle* h, const uint8_t* rgba, int width, int height) {
  jmt_image image{rgba, width, height, 0, JMT_FORMAT_RGBA8};
  return jmt_describe(h->ctx, &image);
}

EMSCRIPTEN_KEEPALIVE void jmtw_features_free(jmt_features* f) { jmt_features_free(f); }

// Waits until everything queued on the GPU is done (for timings). Returns 0, or -1 (jmtw_error).
EMSCRIPTEN_KEEPALIVE int jmtw_finish(Handle* h) {
  try {
    jmt::ops_of(h->ctx).finish();
    return 0;
  } catch (...) {
    return -1;
  }
}

// xy_a [n_a][2], xy_b [n_b][2] in pixels, (0, 0) = centre of the top-left pixel. Returns the number of
// matches, or -1 (jmtw_error). The results stay valid until the next jmtw_match.
EMSCRIPTEN_KEEPALIVE int jmtw_match(Handle* h, const jmt_features* a, const float* xy_a, int n_a, const jmt_features* b,
                                    const float* xy_b, int n_b, int n_layers, float filter_threshold, float max_distance_px,
                                    float min_covisibility, float early_stop_decay) {
  jmt_match_options o;
  jmt_match_options_default(&o);
  o.n_layers = n_layers;
  o.filter_threshold = filter_threshold;
  o.max_distance_px = max_distance_px;
  o.min_covisibility = min_covisibility;
  o.early_stop_decay = early_stop_decay;
  if (jmt_match(h->ctx, a, xy_a, n_a, b, xy_b, n_b, &o, &h->result) != 0) return -1;
  return h->result.n;
}

EMSCRIPTEN_KEEPALIVE const int32_t* jmtw_idx(Handle* h) { return h->result.idx; }
EMSCRIPTEN_KEEPALIVE const float* jmtw_score(Handle* h) { return h->result.score; }
EMSCRIPTEN_KEEPALIVE const float* jmtw_a_in_b(Handle* h) { return h->result.a_in_b; }
EMSCRIPTEN_KEEPALIVE const float* jmtw_b_in_a(Handle* h) { return h->result.b_in_a; }
EMSCRIPTEN_KEEPALIVE const float* jmtw_conf_a(Handle* h) { return h->result.conf_a; }
EMSCRIPTEN_KEEPALIVE const float* jmtw_conf_b(Handle* h) { return h->result.conf_b; }
EMSCRIPTEN_KEEPALIVE float jmtw_covisibility(Handle* h) { return h->result.covisibility; }
EMSCRIPTEN_KEEPALIVE int jmtw_stop_layer(Handle* h) { return h->result.stop_layer; }
// Covisibility of a pair at level 1 or 2 (jmt_covisibility). Returns NaN on failure (jmtw_error).
EMSCRIPTEN_KEEPALIVE float jmtw_covisibility_level(Handle* h, const jmt_features* a, const jmt_features* b, int level) {
  float score = 0.0f;
  return jmt_covisibility(h->ctx, a, b, level, &score) == 0 ? score : NAN;
}

// Level 1 of every pair of n images into out [n][n] (1 on the diagonal). Returns 0, or -1 (jmtw_error).
EMSCRIPTEN_KEEPALIVE int jmtw_level1_matrix(Handle* h, const jmt_features* const* images, int n, float* out) {
  for (int i = 0; i < n; ++i) {
    out[i * n + i] = 1.0f;
    for (int j = i + 1; j < n; ++j) {
      float score = 0.0f;
      if (jmt_covisibility(h->ctx, images[i], images[j], 1, &score) != 0) return -1;
      out[i * n + j] = out[j * n + i] = score;
    }
  }
  return 0;
}

// The pairs of n images worth matching (jmt_shortlist), best first. Returns their number, or -1
// (jmtw_error). jmtw_pairs_idx [n][2] and jmtw_pairs_score [n] stay valid until the next jmtw_shortlist.
EMSCRIPTEN_KEEPALIVE int jmtw_shortlist(Handle* h, const jmt_features* const* images, int n, int k, int neighbours,
                                        int rerank, float min_score) {
  jmt_shortlist_options o;
  jmt_shortlist_options_default(&o);
  o.k = k;
  o.neighbours = neighbours;
  o.rerank = rerank;
  o.min_score = min_score;
  if (jmt_shortlist(h->ctx, images, n, &o, &h->pairs) != 0) return -1;
  return h->pairs.n;
}

EMSCRIPTEN_KEEPALIVE const int32_t* jmtw_pairs_idx(Handle* h) { return h->pairs.idx; }
EMSCRIPTEN_KEEPALIVE const float* jmtw_pairs_score(Handle* h) { return h->pairs.score; }
EMSCRIPTEN_KEEPALIVE const char* jmtw_error() { return jmt_last_error(); }

EMSCRIPTEN_KEEPALIVE void jmtw_destroy(Handle* h) {
  if (!h) return;
  jmt_destroy(h->ctx);
  delete h;
}

}  // extern "C"
