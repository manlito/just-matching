# just-matching

Image matching (FlexGlue) and covisibility with minimal dependencies, the companion of
[just-keypoints](../just-keypoints). No PyTorch, no ONNX Runtime at run time. Plan and status: [PLAN.md](PLAN.md).

Status: the CPU reference backend and the Metal, CUDA and WebGPU backends work and reproduce the original
FlexGlue; WebGPU also runs in the browser (WebAssembly). On the 6 test pairs at two settings, in FP32, they return the same matches as the original;
predicted points agree within 0.0004 px. How to check this yourself, with the commands and their output:
[docs/reproducing.md](docs/reproducing.md).

Per pair at full settings (512 px, 9 layers, 2048 keypoints per image), against the original under
PyTorch on the same GPU:

| | jmt FP32 | PyTorch FP32 | jmt FP16 | PyTorch FP16 | describe per image, jmt FP32 / FP16 |
|---|---|---|---|---|---|
| M5 Pro, Metal | 0.31 s | 0.38 s | 0.105 s | 0.16 s | 31 / 13 ms |
| RTX 2070 Max-Q laptop, CUDA | 0.42 s | 0.63 s | 0.14 s | 0.21 s | 54 / 18 ms |

WebGPU (Dawn natively, Chrome in the browser) on the M5 Pro: see PLAN.md, M6. Covisibility levels 1 and 2
(which pairs to match, from the features alone) and the shortlist are in; level 3 is FlexGlue's own score,
with early stop on pairs it rejects, as in the original.

## Build

Needs CMake >= 3.16 and a C++17 compiler. Nothing else for the CPU backend. The Metal backend is built on
macOS (`-DJMT_METAL=OFF` to leave it out); it uses only Metal.framework and compiles its shaders at run time.
Its fast kernels need macOS 26 (Metal 4); on older systems it runs slower ones, with the same results in FP32.
The CUDA backend is built when a CUDA compiler is found (`-DJMT_CUDA=OFF` to leave it out); it needs only
the CUDA toolkit (nvcc and its runtime library), no cuBLAS or cuDNN. The WebGPU backend is built when
Dawn's prebuilt release is found (`-DJMT_WEBGPU=OFF` to leave it out); `tools/fetch_webgpu.sh` unpacks it
under `third_party/webgpu`, or point `JMT_DAWN_DIR` at it. Its kernels are WGSL, compiled at run time.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

The weights are not in the repository (567 MB; the backbone is under the DINOv3 License). Produce them once,
see [tools/README.md](tools/README.md):

```sh
tools/fetch_upstream.sh && python tools/export_weights.py     # -> weights/flexglue.jmtw
```

### In the browser

With Emscripten and emdawnwebgpu from the same Dawn release (`tools/fetch_webgpu.sh browser`):

```sh
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release -DJMT_BUILD_CLI=OFF -DJMT_BUILD_TESTS=OFF
cmake --build build/wasm      # -> build/wasm/web: jmt_wasm.js / .wasm, jmt.js, test.html, demo.html
```

`web/jmt.js` wraps the module (`create`, `describe`, `match`, and for covisibility `covisibility`, `level1`,
`shortlist`); it needs JSPI (Chrome, Edge). The weights are served next to it and may be the FP16 file (284 MB
instead of 567; widened to float32 on load). The test page compares the module with the native CPU backend on
the 6 test pairs, without and with early stop (`?settings=2,3` runs only the latter), then the shortlist of
their 10 images (`?shortlist=0` skips it):

```sh
python tools/web_testdata.py --out build/wasm/web/testdata
ln -s $PWD/weights/flexglue.jmtw build/wasm/web/
python -m http.server 8765 --directory build/wasm/web      # then open /test.html
```

The demo (`demo.html`) takes a set of images (12 samples in `web/samples`, or your own), shows the level-1
matrix and the shortlist, matches the kept pairs with early stop, and draws the matches of any pair with the
timings of each step. Its keypoints come from just-keypoints' WebAssembly module (RaCo, best 2048): build
just-keypoints with Emscripten too and point `JMT_JKP_WEB_DIR` at its `build/wasm/web` (the default is
`../just-keypoints/build/wasm/web`); the build copies `jkp.js`, its module and `raco.jkpw` into `web/jkp`. It
loads `flexglue-fp16.jmtw` (and `flexglue.jmtw` for FP32) from next to the page; `?run=1` runs on load.

## Command line

Keypoints come from [just-keypoints](../just-keypoints); only their positions are used.

```sh
jkp extract a.jpg --model raco --no-covariance --no-ranker --out a.jkpt
jkp extract b.jpg --model raco --no-covariance --no-ranker --out b.jkpt
build/release/jmt match a.jpg b.jpg --kpts-a a.jkpt --kpts-b b.jkpt --out matches.jmtt
```

Options: `--backend cpu|metal|metal:fp16|cuda|cuda:fp16|webgpu|webgpu:fp16`, `--runs N` (report the fastest), `--batch N` (also
time N copies of the pair as one batch), `--weights FILE`, `--size N` (network input, default 512), `--layers N` (0..9, default 9),
`--filter T`, `--max-px D`, `--min-covisibility S`, `--early-stop D`, `--threads N`, `--dump-features`.

The pairs of a set of images worth matching, best first:

```sh
build/release/jmt shortlist images/*.jpg --k 10 --neighbours 2 --rerank --out pairs.jmtt
```

Options: `--k K` (partners kept per image by level 1, the mean feature vector; default 10), `--neighbours N`
(also the N next images in the order given), `--rerank` (score the kept pairs with level 2, a grid of points
through the layer-0 head and back), `--min-score S` (with `--rerank`, drop pairs below S), and `--backend`,
`--weights`, `--size`, `--threads`.

## C API

```c
#include "jmt/jmt.h"

jmt_config config;
jmt_config_default(&config);                 /* CPU, 512 px */
if (jmt_backend_available(JMT_BACKEND_METAL)) config.backend = JMT_BACKEND_METAL;
config.weights_path = "weights/flexglue.jmtw";
jmt_ctx* ctx = jmt_create(&config);

jmt_image ia = {pixels_a, width_a, height_a, 0, JMT_FORMAT_RGB8};
jmt_features* fa = jmt_describe(ctx, &ia);   /* once per image */
jmt_features* fb = jmt_describe(ctx, &ib);

jmt_matches m;
if (jmt_match(ctx, fa, xy_a, n_a, fb, xy_b, n_b, NULL, &m) == 0) {
  /* m.n matches: keypoint m.idx[2*i] of a with keypoint m.idx[2*i+1] of b, m.score[i];
     m.covisibility: probability that the two images show the same place */
}
jmt_features_free(fa); jmt_features_free(fb);
jmt_destroy(ctx);
```

Keypoints are `[n][2]` pixel coordinates with (0,0) at the centre of the top-left pixel, as just-keypoints
returns them.

For many pairs, `jmt_match_batch` takes an array of `jmt_pair` and fills an array of `jmt_matches`: the same
results, with the GPU kept busy while the host finishes each pair.

Which pairs to match, from the features `jmt_describe` already computed:

```c
jmt_shortlist_options so;
jmt_shortlist_options_default(&so);          /* k = 10 partners per image by level 1 */
so.rerank = 1;                               /* sort the kept pairs by level 2 */
jmt_pair_list pairs;
jmt_shortlist(ctx, features, n_images, &so, &pairs);   /* pairs.idx[2*i], pairs.idx[2*i+1], pairs.score[i] */

float s;
jmt_covisibility(ctx, fa, fb, 2, &s);        /* one pair, level 1 or 2 */

jmt_match_options mo;                        /* level 3: FlexGlue decides while matching */
jmt_match_options_default(&mo);
mo.min_covisibility = 0.8f;                  /* pairs below it get no matches */
mo.early_stop_decay = 0.2f;                  /* and stop after a few layers: m.stop_layer < 9 */
```

## Validation against the original

Step by step, with what each command prints and how to read it: [docs/reproducing.md](docs/reproducing.md).
In short:

```sh
python tools/compare.py a.jpg b.jpg --b "cmd:build/release/jmt match --backend metal" --save build/repro
diff build/repro/a-b.512x9.a.txt build/repro/a-b.512x9.b.txt   # the original's matches vs jmt's
tools/compare_all.sh metal                   # the 6 test pairs at two settings against the original
python tools/dump_golden.py a.jpg b.jpg      # then: cmake -S . -B build/release && ctest --test-dir build/release
```

## Licenses

Code in this repository: to be decided. FlexGlue's code and matcher weights are MIT (`weights/LICENSE-FLEXGLUE`);
its backbone weights are under the DINOv3 License (`weights/LICENSE-DINOv3.md`). No weights are committed.
`cli/stb_image.h` is public domain / MIT and is used only by the CLI.
