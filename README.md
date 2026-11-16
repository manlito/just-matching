# just-matching

Image matching (FlexGlue) and covisibility with minimal dependencies, the companion of
[just-keypoints](../just-keypoints). No PyTorch, no ONNX Runtime at run time. Plan and status: [PLAN.md](PLAN.md).

Status: the CPU reference backend and the Metal and CUDA backends work and match the original FlexGlue. Metal
is not as fast as PyTorch yet: about 0.8 s per pair at full settings on an M5 Pro (0.56 s in FP16), against
7.5 s on the CPU and 0.4 s (0.16 s in FP16) for PyTorch on the same GPU. CUDA takes 0.73 s per pair on a
laptop RTX 2070 Max-Q, and 0.35 s in FP16, where its kernels use the tensor cores.
The WebGPU backend and the covisibility functions are planned.

## Build

Needs CMake >= 3.16 and a C++17 compiler. Nothing else for the CPU backend. The Metal backend is built on
macOS (`-DJMT_METAL=OFF` to leave it out); it uses only Metal.framework and compiles its shaders at run time.
The CUDA backend is built when a CUDA compiler is found (`-DJMT_CUDA=OFF` to leave it out); it needs only
the CUDA toolkit (nvcc and its runtime library), no cuBLAS or cuDNN.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

The weights are not in the repository (567 MB; the backbone is under the DINOv3 License). Produce them once,
see [tools/README.md](tools/README.md):

```sh
tools/fetch_upstream.sh && python tools/export_weights.py     # -> weights/flexglue.jmtw
```

## Command line

Keypoints come from [just-keypoints](../just-keypoints); only their positions are used.

```sh
jkp extract a.jpg --model raco --no-covariance --no-ranker --out a.jkpt
jkp extract b.jpg --model raco --no-covariance --no-ranker --out b.jkpt
build/release/jmt match a.jpg b.jpg --kpts-a a.jkpt --kpts-b b.jkpt --out matches.jmtt
```

Options: `--backend cpu|metal|metal:fp16|cuda|cuda:fp16`, `--runs N` (report the fastest), `--weights FILE`, `--size N` (network input, default 512), `--layers N` (0..9, default 9),
`--filter T`, `--max-px D`, `--min-covisibility S`, `--early-stop D`, `--threads N`, `--dump-features`.

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

## Validation against the original

```sh
python tools/compare.py a.jpg b.jpg --a upstream --b "cmd:build/release/jmt match"
python tools/compare.py a.jpg b.jpg --a upstream --b "cmd:build/release/jmt match --backend metal"
python tools/dump_golden.py a.jpg b.jpg      # then: cmake -S . -B build/release && ctest --test-dir build/release
```

## Licenses

Code in this repository: to be decided. FlexGlue's code and matcher weights are MIT (`weights/LICENSE-FLEXGLUE`);
its backbone weights are under the DINOv3 License (`weights/LICENSE-DINOv3.md`). No weights are committed.
`cli/stb_image.h` is public domain / MIT and is used only by the CLI.
