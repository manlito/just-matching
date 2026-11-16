# Dev tools

Python + PyTorch are used here only to export weights and validate against the original FlexGlue.
They are never a runtime dependency of the library.

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r tools/requirements.txt
tools/fetch_upstream.sh                 # original at its pinned commit -> third_party/FlexGlue, flexglue.pth (567 MB)
cmake -S ../just-keypoints -B ../just-keypoints/build/release -DCMAKE_BUILD_TYPE=Release \
      -DJKP_EMBED_MODELS="aliked-n16;raco" && cmake --build ../just-keypoints/build/release   # keypoints
A=third_party/FlexGlue/assets

python tools/export_weights.py          # -> weights/flexglue.jmtw (add --fp16 for the half-size file)
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg                       # original vs reference pass, CPU
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg --layers 3 --size 256
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg --a upstream:cpu --b upstream:mps,fp16
python tools/dump_golden.py $A/berlin1.jpg $A/berlin2.jpg                   # -> build/golden/*.jmtt (not committed)
python tools/presets.py $A/berlin*.jpg $A/lamar*.jpg --neg $A/lamar_cab_1446.jpg $A/lamar_cab_1450.jpg
python tools/covis.py SCENE_DIR... --single IMAGE...                        # cheap covisibility scores vs FlexGlue's
```

| File | Purpose |
|---|---|
| `jmtfile.py` | weights / test-tensor container reader and writer (the just-keypoints `.jkpw` format, unchanged) |
| `export_weights.py` | checkpoint -> `weights/flexglue.jmtw`; the few changes it makes are listed at its top |
| `folded.py` | forward pass from the weights file only; the executable specification for the C++ backends |
| `upstream.py` | runs the original FlexGlue behind the same interface and exposes its tokens per layer |
| `common.py` | image loading, keypoints from the `jkp` CLI (`$JKP` overrides its path), implementation selection |
| `compare.py` | compares two implementations (`upstream`, `folded`; CPU, MPS or CUDA; FP32 or FP16) on image pairs |
| `dump_golden.py` | golden test vectors from the reference pass |
| `presets.py` | accuracy and time against input size, layer count and precision, on synthetic pairs with exact ground truth |
| `covis.py` | evaluates the covisibility levels of the plan on folders of images |

Keypoints: just-keypoints returns `(0,0)` at the centre of the top-left pixel, FlexGlue expects it at that
pixel's corner. `common.keypoints` adds 0.5; everything in these tools is in FlexGlue's convention.
