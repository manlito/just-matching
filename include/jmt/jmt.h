/* just-matching: FlexGlue image matching with minimal dependencies. */
#ifndef JMT_H
#define JMT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jmt_ctx jmt_ctx;
/* Dense features of one image: computed once by jmt_describe, used by every jmt_match with that image. */
typedef struct jmt_features jmt_features;

typedef enum { JMT_BACKEND_CPU = 0, JMT_BACKEND_CUDA = 1, JMT_BACKEND_METAL = 2, JMT_BACKEND_WEBGPU = 3 } jmt_backend;

typedef enum { JMT_FORMAT_GRAY8 = 0, JMT_FORMAT_RGB8 = 1, JMT_FORMAT_RGBA8 = 2 } jmt_format;

/* FP32 is the reference mode: it matches the original network. FP16 (GPU backends only) stores feature
   maps and weights as 16-bit floats: faster, with slightly different results. */
typedef enum { JMT_PRECISION_FP32 = 0, JMT_PRECISION_FP16 = 1 } jmt_precision;

typedef struct {
  jmt_backend backend;
  /* Weights source, first that is set wins: memory blob, file path. Weights are never built in. */
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
  float min_covisibility; /* pairs whose covisibility is below it get no matches. Default 0 (off) */
  float early_stop_decay; /* in [0, 1): stop after an early layer when the pair is confidently not
                             covisible (needs min_covisibility > 0). Default -1 (off) */
} jmt_match_options;

/* Keypoint coordinates, in and out, follow just-keypoints: pixels of the original image, x right / y down,
   (0,0) = centre of the top-left pixel. Arrays are owned by the context and stay valid until the next
   jmt_match or jmt_destroy on it. */
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

void jmt_config_default(jmt_config* config);
void jmt_match_options_default(jmt_match_options* options);

/* Returns 1 when the backend is compiled into this build. */
int jmt_backend_available(jmt_backend backend);

/* Returns NULL on failure; see jmt_last_error(). */
jmt_ctx* jmt_create(const jmt_config* config);

/* Returns NULL on failure. Free with jmt_features_free. Features must only be used with the context
   that made them, and freed before it is destroyed on GPU backends. */
jmt_features* jmt_describe(jmt_ctx* ctx, const jmt_image* image);
void jmt_features_free(jmt_features* features);

/* Matches keypoints xy_a [n_a][2] of image a with keypoints xy_b [n_b][2] of image b. options may be NULL.
   Returns 0 on success. A context must not be used from several threads at once. */
int jmt_match(jmt_ctx* ctx, const jmt_features* a, const float* xy_a, int n_a, const jmt_features* b,
              const float* xy_b, int n_b, const jmt_match_options* options, jmt_matches* matches);

void jmt_destroy(jmt_ctx* ctx);

/* Message for the last failure on the calling thread. */
const char* jmt_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* JMT_H */
