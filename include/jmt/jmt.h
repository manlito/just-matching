/* just-matching: FlexGlue and LightGlue image matching with minimal dependencies. */
#ifndef JMT_H
#define JMT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jmt_ctx jmt_ctx;
/* Features of one image, computed once and used by every jmt_match with that image: FlexGlue's dense features
   (jmt_describe), or LightGlue's keypoints and descriptors made ready for matching (jmt_describe_keypoints). */
typedef struct jmt_features jmt_features;

typedef enum { JMT_BACKEND_CPU = 0, JMT_BACKEND_CUDA = 1, JMT_BACKEND_METAL = 2, JMT_BACKEND_WEBGPU = 3 } jmt_backend;

typedef enum { JMT_FORMAT_GRAY8 = 0, JMT_FORMAT_RGB8 = 1, JMT_FORMAT_RGBA8 = 2 } jmt_format;

/* FP32 is the reference mode: it matches the original network. FP16 (GPU backends only) stores feature
   maps and weights as 16-bit floats: faster, with slightly different results. */
typedef enum { JMT_PRECISION_FP32 = 0, JMT_PRECISION_FP16 = 1 } jmt_precision;

typedef struct {
  jmt_backend backend;
  /* Weights source, first that is set wins: memory blob, file path. A checkpoint as released (flexglue.pth,
     aliked_lightglue.pth, raco_aliked_lightglue.pth, read without PyTorch) or a .jmtw file. FlexGlue's weights
     are never built in or distributed: the backbone is under Meta's DINOv3 License. LightGlue's (Apache-2.0)
     are built in when the library is compiled with JMT_EMBED_WEIGHTS; then no source needs to be set. */
  const void* weights_data;
  size_t weights_size;
  const char* weights_path;
  /* Side of the square network input in pixels, a multiple of 32. Default 512. Images are squashed to it.
     384 is faster and less accurate; below that accuracy drops sharply. */
  int input_size;
  /* CPU backend worker threads; 0 = all hardware threads. */
  int num_threads;
  /* GPU backends: device index; 0 = first device. */
  int device;
  /* Default JMT_PRECISION_FP32. jmt_create fails when the backend does not support the precision. */
  jmt_precision precision;
  /* "flexglue", "lightglue-aliked" (ALIKED keypoints and descriptors) or "lightglue-raco-aliked" (RaCo
     keypoints described by ALIKED). NULL (default): the model of the weights given. A LightGlue model with no
     weights given uses the built-in ones. input_size applies to FlexGlue only. */
  const char* model;
} jmt_config;

typedef struct {
  const uint8_t* data; /* top-left origin, rows of `stride` bytes */
  int width;
  int height;
  int stride; /* bytes per row; 0 = width * bytes per pixel */
  jmt_format format;
} jmt_image;

typedef struct {
  int n_layers;           /* transformer layers to run, 0..9. Default 9; fewer is faster and less accurate */
  float filter_threshold; /* minimum match score. Default 0.1 */
  float max_distance_px;  /* maximum distance between a keypoint and the point predicted for its partner,
                             mean of both directions, in pixels of the network input. Default 3; <= 0: off */
  float min_covisibility; /* pairs whose covisibility is below it get no matches. Default 0 (off). Unrelated
                             scenes score 0.3 to 0.7, look-alikes near 0: the original suggests 0.8 */
  float early_stop_decay; /* in [0, 1): after layer k (2 <= k < n_layers) stop when the pair's covisibility
                             is below min_covisibility * (1 - early_stop_decay)^(9 - k), as the original
                             does; the pair then gets no matches and stop_layer = k. Lower stops earlier:
                             0.2 stopped 85 % of the pairs of different scenes and no pair that passes at
                             full depth (PLAN.md, Notes from C3). Needs min_covisibility > 0. Default -1 (off) */
} jmt_match_options;

/* LightGlue uses n_layers (1..9) and filter_threshold; the other fields are FlexGlue's.
   Keypoint coordinates, in and out, follow just-keypoints: pixels of the original image, x right / y down,
   (0,0) = centre of the top-left pixel. Arrays are owned by the context and stay valid until the next
   jmt_match, jmt_match_batch or jmt_destroy on it. */
typedef struct {
  int n;                  /* number of matches */
  const int32_t* idx;     /* [n][2]: keypoint index in a, keypoint index in b */
  const float* score;     /* [n] in (0, 1] */
  int n_a, n_b;           /* number of keypoints given */
  const float* a_in_b;    /* [n_a][2] point predicted in image b for every keypoint of a, matched or not */
  const float* b_in_a;    /* [n_b][2] */
  const float* conf_a;    /* [n_a] confidence of a_in_b */
  const float* conf_b;    /* [n_b] */
  float covisibility;     /* probability that the two images show the same place (1 when n_layers is 0) */
  int stop_layer;         /* layers actually run */
} jmt_matches;
/* With LightGlue, which predicts no positions and no covisibility: a_in_b and b_in_a hold the matched
   keypoint's position and NaN for unmatched keypoints, conf_a and conf_b the matchability of each keypoint
   (in [0, 1]), and covisibility is -1. */

void jmt_config_default(jmt_config* config);
void jmt_match_options_default(jmt_match_options* options);

/* Returns 1 when the backend is compiled into this build. */
int jmt_backend_available(jmt_backend backend);

/* Returns NULL on failure; see jmt_last_error(). */
jmt_ctx* jmt_create(const jmt_config* config);

/* FlexGlue. Returns NULL on failure. Free with jmt_features_free. Features must only be used with the context
   that made them, and freed before it is destroyed on GPU backends. */
jmt_features* jmt_describe(jmt_ctx* ctx, const jmt_image* image);

/* LightGlue: n keypoints xy [n][2] of an image of width x height pixels, and their descriptors desc [n][dim]
   (dim 128: just-keypoints' aliked-n16, alone or describing RaCo's keypoints). The arrays are copied.
   jmt_match then takes the same keypoints, in the same order. Returns NULL on failure, as jmt_describe. */
jmt_features* jmt_describe_keypoints(jmt_ctx* ctx, int width, int height, const float* xy, const float* desc, int n,
                                     int dim);
void jmt_features_free(jmt_features* features);

/* Matches keypoints xy_a [n_a][2] of image a with keypoints xy_b [n_b][2] of image b. options may be NULL.
   Returns 0 on success. A context must not be used from several threads at once. */
int jmt_match(jmt_ctx* ctx, const jmt_features* a, const float* xy_a, int n_a, const jmt_features* b,
              const float* xy_b, int n_b, const jmt_match_options* options, jmt_matches* matches);

/* One pair of a batch: the arguments of jmt_match. */
typedef struct {
  const jmt_features* a;
  const float* xy_a;
  int n_a;
  const jmt_features* b;
  const float* xy_b;
  int n_b;
} jmt_pair;

/* Matches n_pairs pairs; matches[i] receives the result of pairs[i], the same as jmt_match would give.
   On a GPU backend the pairs are pipelined: the device works on the next pairs while the host finishes
   the previous one (offsets, mutual nearest neighbours), and no pair waits for a round trip in between.
   Returns 0 on success; on failure no result is valid. */
int jmt_match_batch(jmt_ctx* ctx, const jmt_pair* pairs, int n_pairs, const jmt_match_options* options,
                    jmt_matches* matches);

/* ---- Covisibility: which pairs are worth matching, from the features alone. */

/* Score of a pair at a level of cost:
     1  cosine similarity of the two images' mean feature vectors: a dot product once both are known (each
        image's vector is computed on first use and kept with its features). In practice 0.8 to 1: compare
        pairs with each other rather than with a fixed bar.
     2  a 16 x 16 grid of points of each image through FlexGlue's layer-0 head to the other image and
        back; the share of points that return within one cell (8 pixels of the network input), mean of
        both directions, in [0, 1]. About 0.3 G multiply-adds per pair; each image keeps a 64-channel
        map (1 MB in FP32) with its features after first use.
   Level 3, FlexGlue's own score, is jmt_matches.covisibility: set min_covisibility and early_stop_decay
   in jmt_match_options and a pair that fails stops after a few layers, while one that passes runs on to
   full depth and is matched in the same call. FlexGlue only, as jmt_shortlist. Returns 0 on success. */
int jmt_covisibility(jmt_ctx* ctx, const jmt_features* a, const jmt_features* b, int level, float* score);

typedef struct {
  int k;            /* partners kept per image, by level 1; a pair is kept when either image keeps it.
                       Default 10; >= n - 1 keeps every pair */
  int neighbours;   /* also keep (i, i + 1) .. (i, i + neighbours), for images given in capture order.
                       Default 0 */
  int rerank;       /* 1: score the kept pairs with level 2 and sort by that score. Default 0 */
  float min_score;  /* with rerank: drop pairs whose level-2 score is below it. Default 0 */
} jmt_shortlist_options;

/* Pairs of images, owned by the context; valid until the next jmt_shortlist or jmt_destroy on it. */
typedef struct {
  int n;
  const int32_t* idx; /* [n][2]: indices into the images given, first < second */
  const float* score; /* [n] level 1, or level 2 with rerank */
} jmt_pair_list;

void jmt_shortlist_options_default(jmt_shortlist_options* options);

/* The pairs of n images worth matching, best first. options may be NULL. Returns 0 on success. */
int jmt_shortlist(jmt_ctx* ctx, const jmt_features* const* images, int n, const jmt_shortlist_options* options,
                  jmt_pair_list* pairs);

void jmt_destroy(jmt_ctx* ctx);

/* Message for the last failure on the calling thread. */
const char* jmt_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* JMT_H */
