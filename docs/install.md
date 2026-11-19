# Install and build

## Python

```bash
pip install just-matching        # CPU, and Metal on macOS 13 or later (arm64)
pip install just-matching-cu12   # CPU and CUDA 12: Linux and Windows x86_64, NVIDIA driver R525 or later
pip install just-matching-cu13   # CPU and CUDA 13: Linux and Windows x86_64, NVIDIA driver R580 or later
```

Install one: all three provide the module `just_matching`, for Python 3.12 or later. The CUDA packages need only
the NVIDIA driver (the CUDA runtime is built in) and a GPU of compute capability 7.5 (Turing) or later. No weights
are in any package: give `just_matching.Matcher` the checkpoint (see [Weights](#weights) below).

```python
import just_matching as jmt

m = jmt.Matcher("aliked_lightglue.pth", backend="cuda")
a = m.describe_keypoints(width_a, height_a, xy_a, desc_a)  # xy (n, 2), desc (n, 128)
b = m.describe_keypoints(width_b, height_b, xy_b, desc_b)
r = m.match(a, b)                                          # r.idx (n, 2), r.score (n,)

f = jmt.Matcher("flexglue.pth", backend="cuda")
fa, fb = f.describe(image_a), f.describe(image_b)          # uint8 (h, w, 3)
r = f.match(fa, fb, xy_a, xy_b)
pairs, scores = f.shortlist([fa, fb], k=10)
```

`Matcher` takes the options of `jmt_config` as keyword arguments (`backend`, `model`, `input_size`, `threads`,
`device`, `precision`); `match` and `match_batch` those of `jmt_match_options`; arrays come back as NumPy arrays.

## C and C++

just-matching is built from source with CMake 3.16 or later and a C++17 compiler. The CPU backend needs nothing
else. Each GPU backend is built when its toolkit is found, and uses no library beyond it.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

This gives the static library `libjmt.a`, the single public header `jmt/jmt.h` and the `jmt` command-line tool.

## Backends

| Backend | Built when | Notes |
|---|---|---|
| CPU | always | Plain C++, multi-threaded. The reference the others are checked against. |
| CUDA | the CUDA toolkit (`nvcc`) is found | Needs only the CUDA runtime library: no cuBLAS, no cuDNN. FP16 uses the tensor cores. |
| Metal | on macOS | Needs only Metal.framework; the shaders are compiled at run time. The fast kernels need macOS 26 (Metal 4); older systems run slower ones, with the same results in FP32. |
| WebGPU | Dawn's prebuilt release is found | The same source runs natively (Dawn) and in the browser. FP16 needs the adapter's `shader-f16` feature. |

## Options

| Option | Default | Meaning |
|---|---|---|
| `JMT_CUDA` | `AUTO` | `AUTO` = build it when a CUDA compiler is found, `ON` = require it, `OFF` |
| `JMT_METAL` | `AUTO` | `AUTO` = build it on Apple platforms, `ON` = require it, `OFF` |
| `JMT_WEBGPU` | `AUTO` | `AUTO` = build it when Dawn (or, under Emscripten, emdawnwebgpu) is found, `ON` = require it, `OFF` |
| `JMT_DAWN_DIR`, `JMT_EMDAWNWEBGPU_DIR` | `third_party/webgpu/…` | where Dawn's release and emdawnwebgpu are unpacked |
| `JMT_BUILD_WASM` | `ON` | under Emscripten: build the WebAssembly module and its JavaScript wrapper |
| `JMT_BUILD_CLI` | `ON` | build the `jmt` tool |
| `JMT_BUILD_TESTS` | `ON` | build the tests |
| `JMT_EMBED_WEIGHTS` | `OFF` | build LightGlue's weights into the library (`#embed`: Clang 19 or GCC 15); 47.6 MB per model |
| `JMT_EMBED_MODELS` | both | which: `lightglue-aliked`, `lightglue-raco-aliked`, read from `weights/<model>.jmtw` |

## Weights

LightGlue's weights (Apache-2.0) can be built into the library. Fetch the checkpoints, convert them, then
configure with `JMT_EMBED_WEIGHTS=ON`; a context created with `config.model = "lightglue-aliked"` and no
weights then uses them:

```bash
curl -fL -o aliked_lightglue.pth https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth
shasum -a 256 aliked_lightglue.pth    # d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503
build/release/jmt convert aliked_lightglue.pth weights/lightglue-aliked.jmtw
cmake -S . -B build/release -DJMT_EMBED_WEIGHTS=ON -DJMT_EMBED_MODELS=lightglue-aliked && cmake --build build/release
```

Without them, give the checkpoint or the `.jmtw` file at run time, as for FlexGlue.

FlexGlue's weights are not part of the library. Download `flexglue.pth` (567 MB) from
<a href="https://github.com/cvg/FlexGlue/releases/tag/v0.1">FlexGlue's v0.1 release</a> and give its path to the
library or to `jmt --weights`; it reads the checkpoint as it is, without PyTorch. Its backbone is under Meta's
DINOv3 License, which you accept by using it ([Licences](licences.md)).

```bash
curl -fLO https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth
shasum -a 256 flexglue.pth    # d7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068
```

The library also reads its own container, `.jmtw` ([Weights format](weights-format.md)), which the repository's
`tools/export_weights.py` makes from the checkpoint, in float32 or in float16 (284 MB). Reading the checkpoint
gives the float32 file's tensors bit for bit. The float16 file is widened to float32 when the context is created,
so the original's FP32 results need the checkpoint or the float32 file.

## WebGPU

Natively the WebGPU backend links Dawn; in the browser it uses Emscripten's emdawnwebgpu. Both come from the
same Dawn release, v20261113.214538, which a script downloads into the source tree:

```bash
tools/fetch_webgpu.sh            # emdawnwebgpu, and Dawn for this machine
tools/fetch_webgpu.sh browser    # emdawnwebgpu only
```

## The browser build

The browser build needs Emscripten 7.0.0, and Python 3.10 or later for Emscripten (set `EMSDK_PYTHON` when the
default `python3` is older):

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release -DJMT_BUILD_CLI=OFF -DJMT_BUILD_TESTS=OFF
cmake --build build/wasm
```

`build/wasm/web` then holds the module (`jmt_wasm.js`, 100 KB, and `jmt_wasm.wasm`, 399 KB) and the wrapper
`jmt.js`. Serve them and a weights file as static files; see [In the browser](browser.md).
