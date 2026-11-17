# Quick start

## LightGlue, from C or C++

LightGlue matches keypoints by their descriptors: just-keypoints' `aliked-n16` gives both. With the weights
built into the library ([Build](install.md)), nothing needs to be loaded:

```c
#include "jmt/jmt.h"

jmt_config config;
jmt_config_default(&config);
config.model = "lightglue-aliked";           /* built-in weights; or weights_path = "aliked_lightglue.pth" */
if (jmt_backend_available(JMT_BACKEND_METAL)) config.backend = JMT_BACKEND_METAL, config.precision = JMT_PRECISION_FP16;
jmt_ctx* ctx = jmt_create(&config);
if (!ctx) { fprintf(stderr, "%s\n", jmt_last_error()); return 1; }

/* xy: [n][2] keypoints, desc: [n][128] descriptors, from just-keypoints (aliked-n16) */
jmt_features* fa = jmt_describe_keypoints(ctx, width_a, height_a, xy_a, desc_a, n_a, 128);
jmt_features* fb = jmt_describe_keypoints(ctx, width_b, height_b, xy_b, desc_b, n_b, 128);
jmt_matches m;
if (jmt_match(ctx, fa, xy_a, n_a, fb, xy_b, n_b, NULL, &m) == 0)
  for (int i = 0; i < m.n; ++i) { int ka = m.idx[2 * i], kb = m.idx[2 * i + 1]; float s = m.score[i]; }
jmt_features_free(fa);
jmt_features_free(fb);
jmt_destroy(ctx);
```

By default LightGlue stops early on easy pairs and drops keypoints that cannot match, as the original does
(`depth_confidence`, `width_confidence` in [C API](api.md)). `lightglue-raco-aliked` takes RaCo's keypoints
described by `aliked-n16` instead; on pairs with a large change of scale it finds fewer matches than ALIKED
with its own keypoints.

## FlexGlue, from C or C++

```c
#include "jmt/jmt.h"

jmt_config config;
jmt_config_default(&config);                 /* CPU, 512 px, FP32 */
if (jmt_backend_available(JMT_BACKEND_METAL)) config.backend = JMT_BACKEND_METAL;
config.precision = JMT_PRECISION_FP16;       /* GPU backends: the fast mode */
config.weights_path = "flexglue.pth";        /* from FlexGlue's release, see Build */
jmt_ctx* ctx = jmt_create(&config);
if (!ctx) { fprintf(stderr, "%s\n", jmt_last_error()); return 1; }

jmt_image ia = {pixels_a, width_a, height_a, 0, JMT_FORMAT_RGB8};
jmt_image ib = {pixels_b, width_b, height_b, 0, JMT_FORMAT_RGB8};
jmt_features* fa = jmt_describe(ctx, &ia);   /* once per image */
jmt_features* fb = jmt_describe(ctx, &ib);

jmt_matches m;
if (jmt_match(ctx, fa, xy_a, n_a, fb, xy_b, n_b, NULL, &m) == 0) {
  for (int i = 0; i < m.n; ++i) {
    int ka = m.idx[2 * i], kb = m.idx[2 * i + 1];   /* keypoint ka of a matches keypoint kb of b */
    float s = m.score[i];
  }
  /* m.covisibility: the probability that the two images show the same place */
}
jmt_features_free(fa);
jmt_features_free(fb);
jmt_destroy(ctx);
```

Link `libjmt.a` (and, for the GPU backends, the CUDA runtime, Metal.framework or Dawn). Create a context once and
reuse it: it holds the weights and the GPU buffers. `jmt_describe` runs the backbone on one image and keeps its
dense features on the device; `jmt_match` runs the matcher on two described images, so a set of images is
described once however many pairs it is matched in.

Keypoints `xy_a` and `xy_b` are `[n][2]` floats in pixels of the original image, x right and y down, with (0, 0)
at the centre of the top-left pixel: what just-keypoints returns. FlexGlue reads the images, not descriptors, so
any keypoints work; RaCo's, 2048 per image, are what its authors use and what the results on this site were
measured with.

Besides the matches, every keypoint gets the point FlexGlue predicts for it in the other image (`a_in_b`,
`b_in_a`) with a confidence (`conf_a`, `conf_b`), matched or not. The arrays belong to the context and stay valid
until the next `jmt_match` on it.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `config.input_size` | 512 | images are resized (squashed) to this square; 384 is 2.2 times as fast, with a median error twice as large; below 384 the accuracy drops sharply |
| `options.n_layers` | 9 | transformer layers; 5 is 1.7 times as fast and loses little on easy pairs; 3 or fewer lose sub-pixel accuracy |
| `options.filter_threshold` | 0.1 | minimum match score |
| `options.max_distance_px` | 3 | a match's keypoints must lie within this distance of each other's predicted points (mean of both directions, in pixels of the network input) |
| `options.min_covisibility`, `options.early_stop_decay` | off | reject pairs FlexGlue scores as not covisible, and stop early on them: see [Which pairs to match](covisibility.md) |

`jmt_match_options_default` fills the defaults; `NULL` options mean the same.

## FP32 and FP16

FP32 is the default on every backend and matches the original network. On the GPU backends
`config.precision = JMT_PRECISION_FP16` stores feature maps and weights as 16-bit floats and uses the GPU's
matrix hardware: 3 times as fast on Metal and CUDA, 1.5 times on WebGPU. It gives nearly the same matches: natively, 97 to
100 % of them on the test pairs, with confident points moving by a few hundredths of a pixel. The original runs
its transformer in FP16 too. See [Accuracy and speed](accuracy-and-speed.md).

## Many pairs

`jmt_match_batch` takes an array of `jmt_pair` and fills an array of `jmt_matches`, the same results as one
`jmt_match` per pair. On a GPU backend it keeps the device busy while the host finishes each pair. Which pairs
to give it: [Which pairs to match](covisibility.md).

## From the command line

```bash
jkp extract a.jpg --top-k 2048 --out a.jkpt                                 # ALIKED keypoints and descriptors
jkp extract b.jpg --top-k 2048 --out b.jkpt
jmt match --model lightglue-aliked --kpts-a a.jkpt --kpts-b b.jkpt --backend metal:fp16 --out matches.jmtt
```

LightGlue reads no images: the keypoint files carry the keypoints, the descriptors and the image size. Its
weights: `--weights FILE` (a `.pth` as released or a `.jmtw` file), else `weights/<model>.jmtw` when it
exists, else the built-in ones; `--weights builtin` forces those. `--depth-confidence C` and
`--width-confidence C` set the adaptive options (0 turns them off). With FlexGlue:

```bash
jkp extract a.jpg --model raco --no-covariance --no-ranker --out a.jkpt     # keypoints, from just-keypoints
jkp extract b.jpg --model raco --no-covariance --no-ranker --out b.jkpt
jmt match a.jpg b.jpg --kpts-a a.jkpt --kpts-b b.jkpt --weights flexglue.pth --backend metal:fp16 --out matches.jmtt
jmt shortlist images/*.jpg --weights flexglue.pth --k 10 --neighbours 2 --rerank --out pairs.jmtt
```

| Option of `match` | Meaning |
|---|---|
| `--model M` | `flexglue` (default), `lightglue-aliked` or `lightglue-raco-aliked` |
| `--backend B` | `cpu` (default), `metal`, `metal:fp16`, `cuda`, `cuda:fp16`, `webgpu`, `webgpu:fp16` |
| `--weights FILE` | a checkpoint as released or a `.jmtw` file (FlexGlue's default: `weights/flexglue.jmtw`) |
| `--size N` | network input side, a multiple of 32 (default 512) |
| `--layers N` | transformer layers, 0 to 9 (default 9) |
| `--filter T` | minimum match score (default 0.1) |
| `--max-px D` | maximum distance to the predicted point, in network-input pixels (default 3; 0 = off) |
| `--min-covisibility S`, `--early-stop D` | reject pairs scoring below S; stop early on them with decay D |
| `--runs N`, `--batch N` | timing: report the fastest of N runs; also time N copies of the pair as one batch |
| `--threads N` | CPU threads (default all) |
| `--dump-features` | also write the dense features |

`jmt convert IN OUT.jmtw [--fp16]` turns a checkpoint as released (or a `.jmtw` file) into the library's own
container, in float32 or float16 ([Weights format](weights-format.md)).

The options of `shortlist` are on [Which pairs to match](covisibility.md). Output files use the same container as
the weights ([Weights format](weights-format.md)): `match` writes the tensors `matches`, `scores`, `a_in_b`,
`b_in_a`, `conf_a` and `conf_b`; `shortlist` writes `pairs` and `scores`.
