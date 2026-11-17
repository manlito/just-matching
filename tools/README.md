# Dev tools

Python + PyTorch are used here only to export weights and validate against the original FlexGlue.
They are never a runtime dependency of the library.

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r tools/requirements.txt
tools/fetch_upstream.sh                 # original at its pinned commit -> third_party/FlexGlue, flexglue.pth (567 MB)
tools/fetch_lightglue.sh                # LightGlue at its pinned commit -> third_party/LightGlue, two ALIKED checkpoints (2 x 48 MB)
cmake -S ../just-keypoints -B ../just-keypoints/build/release -DCMAKE_BUILD_TYPE=Release \
      -DJKP_EMBED_MODELS="aliked-n16;raco" && cmake --build ../just-keypoints/build/release   # keypoints
A=third_party/FlexGlue/assets

python tools/export_weights.py          # -> weights/flexglue.jmtw (add --fp16 for the half-size file)
python tools/export_weights.py --model lightglue-aliked      # -> weights/lightglue-aliked.jmtw (also lightglue-raco-aliked)
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg                       # original vs reference pass, CPU
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg --layers 3 --size 256
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg --a upstream:cpu --b upstream:mps,fp16
python tools/dump_golden.py $A/berlin1.jpg $A/berlin2.jpg                   # -> build/golden/*.jmtt (not committed)
tools/dump_golden_lightglue.sh                                              # -> build/golden/lightglue-*/ (6 pairs, both models)
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg --b "cmd:build/release/jmt match"   # original vs the C++ library
python tools/presets.py $A/berlin*.jpg $A/lamar*.jpg --neg $A/lamar_cab_1446.jpg $A/lamar_cab_1450.jpg
python tools/covis.py SCENE_DIR... --single IMAGE...                        # cheap covisibility scores vs FlexGlue's
python tools/compare_covis.py IMAGE... --backend metal                      # jmt shortlist's levels 1 and 2 vs the reference
```

| File | Purpose |
|---|---|
| `jmtfile.py` | weights / test-tensor container reader and writer (the just-keypoints `.jkpw` format, unchanged) |
| `export_weights.py` | checkpoint -> `weights/flexglue.jmtw`, or with `--model lightglue-aliked` / `lightglue-raco-aliked` -> `weights/<model>.jmtw`; the few changes it makes are listed at its top |
| `folded.py` | forward pass from the weights file only; the executable specification for the C++ backends |
| `upstream.py` | runs the original FlexGlue behind the same interface and exposes its tokens per layer |
| `common.py` | image loading, keypoints from the `jkp` CLI (`$JKP` overrides its path), implementation selection |
| `compare.py` | compares two implementations (`upstream`, `folded`, or `cmd:<program>` for the `jmt` CLI) on image pairs; `--save DIR` writes both match lists as text for `diff` (see [docs/reproducing.md](../docs/reproducing.md)) |
| `dump_golden.py` | golden test vectors from the reference pass |
| `fetch_lightglue.sh` | the original LightGlue at its pinned commit and its two checkpoints for ALIKED descriptors (Apache-2.0), checked by sha256 |
| `dump_golden_lightglue.py` | golden test vectors from the original LightGlue on just-keypoints' ALIKED or RaCo-ALIKED features: tokens per layer, assignment, matches, and an adaptive run; `dump_golden_lightglue.sh` runs it on the 6 test pairs |
| `presets.py` | accuracy and time against input size, layer count and precision, on synthetic pairs with exact ground truth |
| `covis.py` | evaluates the covisibility levels of the plan on folders of images |
| `compare_covis.py` | levels 1 and 2 of `jmt shortlist` against `covis.py`'s reference, every pair of the images given |
| `compare_all.sh` | `compare.py` on the 6 test pairs at 512 px / 9 layers and 384 px / 5 layers for one backend: `tools/compare_all.sh metal:fp16 [--save DIR]`; `JMT="tools/remote.sh user@host DIR build/release/jmt"` runs the CLI on another machine |
| `bench_torch.py` | time of the original under PyTorch on the pair `bench.sh` uses, FP32 and FP16, for the benchmark table |
| `bench.sh` | wall time and GPU time per kernel of the `jmt` CLI on the Berlin pair: `tools/bench.sh metal:fp16` (WebGPU: wall time per kernel, a round trip each) |
| `remote.sh` | runs the `jmt` CLI on another machine over SSH, so `compare.py` can check a backend this machine lacks: `--b "cmd:tools/remote.sh user@host DIR build/release/jmt match --backend cuda"` |
| `fetch_webgpu.sh` | unpacks the pinned Dawn release (native) and emdawnwebgpu (browser) under `third_party/webgpu` |
| `web_testdata.py` | the browser test page's data: the 6 test pairs as raw RGBA, their keypoints and the native CPU backend's results, and the native shortlist of their 10 images |
| `bench_onnx.py` | times upstream's ONNX models under ONNX Runtime (`pip install onnxruntime`, or `onnxruntime-gpu` for `--providers cuda`; models from release v0.1 in `third_party/onnx/`) |

Keypoints: just-keypoints returns `(0,0)` at the centre of the top-left pixel, FlexGlue expects it at that
pixel's corner. `common.keypoints` adds 0.5; everything in these tools is in FlexGlue's convention.
