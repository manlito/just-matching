# C API

The whole public interface is one header, `jmt/jmt.h`, in C. Errors are reported by return values; the message
for the last failure on the calling thread comes from `jmt_last_error()`.

## Functions

| Function | Meaning |
|---|---|
| `void jmt_config_default(jmt_config*)` | fills a configuration with the defaults: CPU, 512 px, FP32, all threads |
| `int jmt_backend_available(jmt_backend)` | 1 when the backend is compiled into this build (whether the device works is checked by `jmt_create`) |
| `jmt_ctx* jmt_create(const jmt_config*)` | loads the weights and prepares the backend; `NULL` on failure |
| `jmt_features* jmt_describe(jmt_ctx*, const jmt_image*)` | FlexGlue: runs the backbone on one image; `NULL` on failure |
| `jmt_features* jmt_describe_keypoints(ctx, width, height, xy, desc, n, dim)` | LightGlue: an image's `n` keypoints `xy` `[n][2]` and descriptors `desc` `[n][dim]` (`dim` 128: just-keypoints' `aliked-n16`), for an image of `width` x `height` pixels; copied. `NULL` on failure |
| `void jmt_features_free(jmt_features*)` | frees an image's features |
| `void jmt_match_options_default(jmt_match_options*)` | fills match options with the defaults |
| `int jmt_match(ctx, a, xy_a, n_a, b, xy_b, n_b, options, jmt_matches*)` | matches the keypoints of two described images; 0 on success |
| `int jmt_match_batch(ctx, const jmt_pair*, n_pairs, options, jmt_matches*)` | matches many pairs, the same results as one `jmt_match` each; 0 on success |
| `int jmt_covisibility(ctx, a, b, level, float* score)` | FlexGlue: the score of one pair at level 1 or 2; 0 on success |
| `void jmt_shortlist_options_default(jmt_shortlist_options*)` | fills shortlist options with the defaults |
| `int jmt_shortlist(ctx, const jmt_features* const* images, n, options, jmt_pair_list*)` | FlexGlue: the pairs of n images worth matching, best first; 0 on success |
| `void jmt_destroy(jmt_ctx*)` | frees the context and its results |
| `const char* jmt_last_error(void)` | message for the last failure on the calling thread |

Options may be `NULL` for the defaults. Features belong to the context that made them; on the GPU backends,
free them before destroying it. A context runs one model: `jmt_describe` fails on a LightGlue context and
`jmt_describe_keypoints` on a FlexGlue one, and the covisibility functions are FlexGlue's. With LightGlue,
`jmt_match` takes the keypoints given to `jmt_describe_keypoints`, in the same order. A context must not be used from several threads at once: use one context per
thread, or per GPU.

## `jmt_config`

| Field | Meaning |
|---|---|
| `backend` | `JMT_BACKEND_CPU`, `JMT_BACKEND_CUDA`, `JMT_BACKEND_METAL` or `JMT_BACKEND_WEBGPU` |
| `weights_data`, `weights_size` | a weights file in memory (copied by `jmt_create`): a checkpoint as released (`flexglue.pth`, `aliked_lightglue.pth`, `raco_aliked_lightglue.pth`) or a `.jmtw` file |
| `weights_path` | the same, on disk; used when no memory blob is given |
| `model` | `"flexglue"`, `"lightglue-aliked"` or `"lightglue-raco-aliked"`; `NULL` (default): the model of the weights given. A LightGlue model with no weights given uses the weights built into the library ([Build](install.md)) |
| `input_size` | FlexGlue: side of the square network input, a multiple of 32; default 512. Images are resized to it |
| `num_threads` | CPU backend worker threads; 0 = all hardware threads |
| `device` | GPU backends: device index, 0 = the first |
| `precision` | `JMT_PRECISION_FP32` (default, matches the original) or `JMT_PRECISION_FP16` (GPU backends); `jmt_create` fails when the backend does not support it |

## `jmt_image`

| Field | Meaning |
|---|---|
| `data` | pixels, top-left origin, in host memory |
| `width`, `height` | in pixels; any size |
| `stride` | bytes per row; 0 = `width` times the bytes per pixel |
| `format` | `JMT_FORMAT_GRAY8`, `JMT_FORMAT_RGB8` or `JMT_FORMAT_RGBA8` |

## `jmt_match_options`

| Field | Default | Meaning |
|---|---|---|
| `n_layers` | 9 | transformer layers to run, 0 to 9 (LightGlue: 1 to 9) |
| `filter_threshold` | 0.1 | minimum match score |
| `max_distance_px` | 3 | maximum distance between a keypoint and the point predicted for its partner, mean of both directions, in pixels of the network input; 0 or less: off |
| `min_covisibility` | 0 (off) | pairs scoring below it get no matches; 0.8 is the original's suggestion |
| `early_stop_decay` | -1 (off) | in [0, 1): after layer k, from 2 to `n_layers` - 1, stop when the score is below `min_covisibility` (1 - decay)^(9 - k). Needs `min_covisibility` > 0 |
| `depth_confidence` | 0.95 | LightGlue: after each layer but the last, stop when more than this share of the keypoints are confident, as the original does; 0 or less: every layer |
| `width_confidence` | 0.99 | LightGlue: drop confident keypoints whose matchability is at most 1 - this before the next layer (they stay unmatched); 0 or less: every keypoint |

`max_distance_px`, `min_covisibility` and `early_stop_decay` are FlexGlue's; LightGlue ignores them, and FlexGlue
ignores `depth_confidence` and `width_confidence`.

## `jmt_matches`

Keypoints, in and out, are pixels of the original image, x right and y down, with (0, 0) at the centre of the
top-left pixel. The arrays belong to the context and stay valid until the next `jmt_match`, `jmt_match_batch` or
`jmt_destroy` on it.

| Field | Meaning |
|---|---|
| `n` | number of matches |
| `idx` | `[n][2]`: keypoint index in a, keypoint index in b |
| `score` | `[n]` match score, in (0, 1] |
| `n_a`, `n_b` | number of keypoints given |
| `a_in_b`, `b_in_a` | `[n_a][2]`, `[n_b][2]`: the point predicted in the other image for every keypoint, matched or not |
| `conf_a`, `conf_b` | `[n_a]`, `[n_b]`: the confidence of each predicted point |
| `covisibility` | the probability that the two images show the same place (1 when `n_layers` is 0) |
| `stop_layer` | layers actually run; fewer than `n_layers` when the pair stopped early |

LightGlue predicts no points: `a_in_b` and `b_in_a` hold the matched keypoint's position, and NaN for an
unmatched keypoint; `conf_a` and `conf_b` hold each keypoint's matchability, in [0, 1]; `covisibility` is -1.

## `jmt_pair`

One pair of a batch: the arguments of `jmt_match`, `a`, `xy_a`, `n_a`, `b`, `xy_b`, `n_b`. `jmt_match_batch`
writes the result of `pairs[i]` to `matches[i]`; on failure no result is valid.

## `jmt_shortlist_options` and `jmt_pair_list`

| Field | Default | Meaning |
|---|---|---|
| `k` | 10 | partners kept per image by level 1; a pair is kept when either image keeps it. n - 1 or more keeps every pair |
| `neighbours` | 0 | also keep (i, i + 1) to (i, i + `neighbours`), for images given in capture order |
| `rerank` | 0 | 1: score the kept pairs with level 2 and sort by that score |
| `min_score` | 0 | with `rerank`: drop pairs whose level-2 score is below it |

`jmt_pair_list`: `n` pairs, `idx` `[n][2]` (indices into the images given, first < second) and `score` `[n]`
(level 1, or level 2 with `rerank`), best first. It belongs to the context and stays valid until the next
`jmt_shortlist` or `jmt_destroy`.

The levels: [Which pairs to match](covisibility.md).

## JavaScript

The browser wrapper `jmt.js` has the same model: `create(options)` returns a matcher (a context) with
`describe` or `describeKeypoints`, `match`, `covisibility`, `level1` and `shortlist`. See
[In the browser](browser.md).
