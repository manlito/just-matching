# Reproducing FlexGlue's results

just-matching is a port of [FlexGlue](https://github.com/cvg/FlexGlue) (release v0.1). "Reproduces" means,
here: given the same two images and the same keypoints, the original under PyTorch and jmt return the same
list of matches, and the numbers behind them (predicted points, confidences, covisibility) agree to within
the tolerances below. This page shows how to check that yourself: the commands, and what they print on
our machines.

The reference is always the original code, unchanged, at a pinned commit, with the released checkpoint:

| | |
|---|---|
| Code | `cvg/FlexGlue` at commit `99920296d082c9405ebfd612d674efe288e74881`, cloned into `third_party/FlexGlue` |
| Weights | `flexglue.pth` from release v0.1, SHA-256 `d7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068` |
| How it is run | `tools/upstream.py`: `FlexGlue(weights=...)` and its own `descriptor.describe` and `match(..., sparse=True)`, with the defaults of `match` (filter 0.1, 3 px) |
| Precision | float32 on the CPU (`mp=None`). FlexGlue's own default is float16 mixed precision; `upstream:mps,fp16` or `upstream:cuda,fp16` runs that |

jmt reads the same checkpoint, converted once to `weights/flexglue.jmtw` by `tools/export_weights.py`.
None of the changes the conversion makes alters the function; they are listed at the top of that script.
They are: layer scale folded into the next product, `Wqkv` rows reordered so that q, k and v are
contiguous, 1x1 convolutions stored as matrices, and renamed tensors.

## 1. Set up (once)

```sh
# jmt
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release

# Python, for the original and the comparison tools only (never needed by jmt itself)
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r tools/requirements.txt
source .venv/bin/activate

# The original at its pinned commit, and the checkpoint (567 MB, checked against its SHA-256)
tools/fetch_upstream.sh
python tools/export_weights.py                  # -> weights/flexglue.jmtw

# Keypoints come from just-keypoints (RaCo); build its CLI next to this repository
cmake -S ../just-keypoints -B ../just-keypoints/build/release -DCMAKE_BUILD_TYPE=Release \
      -DJKP_EMBED_MODELS="aliked-n16;raco" && cmake --build ../just-keypoints/build/release
export JKP=../just-keypoints/build/release/jkp  # only if it is elsewhere
```

Keypoints are computed once per image and cached in `build/keypoints/`. Both implementations get exactly
the same keypoint file, so a difference in the result can only come from the matcher.

## 2. One pair, end to end

The Berlin pair from FlexGlue's own `assets/` folder, 2048 keypoints per image, 512 px, 9 layers (the
defaults).

### Run jmt

```sh
A=third_party/FlexGlue/assets; K=build/keypoints
build/release/jmt match $A/berlin1.jpg $A/berlin2.jpg \
    --kpts-a $K/berlin1.jpg.raco.2048.jkpt --kpts-b $K/berlin2.jpg.raco.2048.jkpt --backend metal
```
```
70 matches from 2048 and 2048 keypoints, covisibility 0.9932, 9 layers
load 353 ms, describe 59 ms per image, match 346 ms
```

(`--backend cpu`, `metal`, `metal:fp16`, `cuda` or `cuda:fp16`; `--out FILE` writes the matches. The
first run includes compiling the shaders; `--runs 8` reports the fastest of 8.)

### Run the original on the same input, and compare

`tools/compare.py` runs both on the same images and keypoints. `--a` is the original by default; `--b`
here is the jmt CLI. `--save` also writes both match lists as text.

```sh
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg \
    --b "cmd:build/release/jmt match --backend metal" --save build/repro/metal
```
```
berlin1.jpg - berlin2.jpg: feat 3.08e-05  tokens nan  pos max 1.73e-04 median 3.05e-05 px  conf 2.56e-06  covis 0.9932 / 0.9932  matches 70 / 70, same 70
worst: feat 3.08e-05  tokens 0.00e+00  pos 1.73e-04 px  conf 2.56e-06  same matches 100.00 %
```

How to read it (original first, jmt second):

| Field | Meaning | Here |
|---|---|---|
| `feat` | largest difference of the dense features of either image (values around 1) | 3e-5 |
| `tokens` | largest difference of the tokens after the last layer; `nan` for the CLI, which does not export them | |
| `pos max / median` | difference of the predicted position of every confident query (confidence > 0.1) in the other image, in pixels of the original image | at most 0.00017 px |
| `conf` | largest difference of any query's confidence (0 to 1) | 3e-6 |
| `covis` | covisibility score of each | 0.9932 both |
| `matches a / b, same n` | matches of each, and how many are the same pair of keypoints | 70, 70, all 70 the same |

### Look at the matches themselves

Each file has one match per line: index of the keypoint in image A, in image B, the two keypoint positions
(pixels, FlexGlue's convention: (0, 0) is the corner of the top-left pixel) and the score.

```sh
head -4 build/repro/metal/berlin1-berlin2.512x9.a.txt     # .a = the original, .b = jmt
```
```
# idx_a idx_b x_a y_a x_b y_b score
9 768 733.68 444.24 10.70 623.93 0.8049
45 2024 839.27 375.54 473.59 341.65 0.5828
97 637 767.06 325.73 485.42 180.53 0.1069
```

In FP32 the two lists are the same, here even the scores to the 4 decimals written (on other pairs a
few scores differ in the 4th decimal: 6 of 1534 lines on the largest pair):

```sh
diff build/repro/metal/berlin1-berlin2.512x9.a.txt build/repro/metal/berlin1-berlin2.512x9.b.txt && echo same
```
```
same
```

The same holds for the CUDA backend (`--backend cuda`). In FP16 a match can change. To see which, compare
the first 6 columns, the matched pairs and their positions (in FP16 the scores differ in the 3rd decimal):

```sh
diff <(cut -d' ' -f1-6 build/repro/cuda-fp16/berlin1-berlin2.512x9.a.txt) \
     <(cut -d' ' -f1-6 build/repro/cuda-fp16/berlin1-berlin2.512x9.b.txt)
```
```
5c5
< 126 198 759.92 345.54 124.66 249.18
---
> 126 49 759.92 345.54 112.35 247.90
```

Keypoint 126 of image A is matched to its neighbour 12 px away (keypoint 49 instead of 198). The other 69
matches are the same pairs. This is the size of difference to expect from half precision.

## 3. All test pairs, two settings

`tools/compare_all.sh` runs the comparison on 6 pairs at the default setting (512 px, 9 layers) and a fast
one (384 px, 5 layers):

| Pair | What it tests |
|---|---|
| berlin1 - berlin2 | FlexGlue's example pair |
| lamar_cab_1446 - lamar_cab_1450 | a hard pair: low covisibility (0.06), few matches |
| berlin1 - lamar_cab_1446 | two different places: covisibility must stay low, almost no matches |
| berlin1_h1 - berlin1_h2, lamar_cab_1446_h1 - _h2 | synthetic homographies of one image: over 1000 matches |
| berlin2_h1 - lamar_cab_1450_h2 | synthetic, different places |

The synthetic images are made by `tools/presets.py` with a fixed random seed (0), into `build/synth/`.
Running it once at its cheapest preset writes them:

```sh
python tools/presets.py $A/berlin*.jpg $A/lamar*.jpg --presets 256x0 --impl upstream:cpu
tools/compare_all.sh metal --save build/repro/all-metal        # or metal:fp16, cpu, cuda, cuda:fp16
```

With `--save`, the matched pairs of every run can be diffed at once (files are named
`<A>-<B>.<size>x<layers>.a.txt` / `.b.txt`):

```sh
for a in build/repro/all-metal/*.a.txt; do
  diff -q <(cut -d' ' -f1-6 $a) <(cut -d' ' -f1-6 ${a%.a.txt}.b.txt) > /dev/null || echo "differs: $a"
done
```

This prints nothing for `metal` and `cuda` (FP32): the 12 match lists are identical.

The CUDA output below comes from a Mac driving the NVIDIA laptop over SSH (see section 4):

```
=== cuda, 512 px, 9 layers
berlin1.jpg - berlin2.jpg: feat 3.14e-05  tokens nan  pos max 1.44e-04 median 6.10e-05 px  conf 9.72e-06  covis 0.9932 / 0.9932  matches 70 / 70, same 70
lamar_cab_1446.jpg - lamar_cab_1450.jpg: feat 1.15e-05  tokens nan  pos max 4.11e-04 median 6.10e-05 px  conf 1.51e-05  covis 0.0647 / 0.0647  matches 39 / 39, same 39
berlin1.jpg - lamar_cab_1446.jpg: feat 3.14e-05  tokens nan  pos max 9.16e-05 median 4.58e-05 px  conf 5.96e-06  covis 0.6495 / 0.6495  matches 3 / 3, same 3
berlin1_h1.png - berlin1_h2.png: feat 2.52e-05  tokens nan  pos max 1.83e-04 median 1.53e-05 px  conf 4.32e-06  covis 0.9995 / 0.9995  matches 1256 / 1256, same 1256
lamar_cab_1446_h1.png - lamar_cab_1446_h2.png: feat 1.05e-05  tokens nan  pos max 1.36e-04 median 0.00e+00 px  conf 4.32e-06  covis 0.9998 / 0.9998  matches 1534 / 1534, same 1534
berlin2_h1.png - lamar_cab_1450_h2.png: feat 1.73e-05  tokens nan  pos max 1.36e-04 median 6.29e-05 px  conf 8.46e-06  covis 0.7038 / 0.7038  matches 1 / 1, same 1
worst: feat 3.14e-05  tokens 0.00e+00  pos 4.11e-04 px  conf 1.51e-05  same matches 100.00 %
=== cuda, 384 px, 5 layers
berlin1.jpg - berlin2.jpg: feat 2.29e-05  tokens nan  pos max 1.86e-04 median 6.10e-05 px  conf 6.65e-06  covis 0.4460 / 0.4460  matches 71 / 71, same 71
lamar_cab_1446.jpg - lamar_cab_1450.jpg: feat 1.10e-05  tokens nan  pos max 1.53e-04 median 5.50e-05 px  conf 6.99e-06  covis 0.0259 / 0.0259  matches 66 / 66, same 66
berlin1.jpg - lamar_cab_1446.jpg: feat 2.29e-05  tokens nan  pos max 1.22e-04 median 6.10e-05 px  conf 5.92e-06  covis 0.6452 / 0.6452  matches 3 / 3, same 3
berlin1_h1.png - berlin1_h2.png: feat 2.80e-05  tokens nan  pos max 2.52e-04 median 3.05e-05 px  conf 3.46e-06  covis 0.9994 / 0.9994  matches 1317 / 1317, same 1317
lamar_cab_1446_h1.png - lamar_cab_1446_h2.png: feat 9.54e-06  tokens nan  pos max 1.36e-04 median 3.05e-05 px  conf 5.16e-06  covis 0.9997 / 0.9997  matches 1518 / 1518, same 1518
berlin2_h1.png - lamar_cab_1450_h2.png: feat 1.13e-05  tokens nan  pos max 2.46e-04 median 6.10e-05 px  conf 3.49e-06  covis 0.6515 / 0.6515  matches 5 / 5, same 5
worst: feat 2.80e-05  tokens 0.00e+00  pos 2.52e-04 px  conf 6.99e-06  same matches 100.00 %
```

What we measured on every backend (12 runs each: 6 pairs x 2 settings):

| Backend | Same matches as the original | Confident points, largest difference | Confidences |
|---|---|---|---|
| `metal` (M5 Pro) | all 12 runs | 1.9e-4 px | 5.6e-6 |
| `cuda` (RTX 2070 Max-Q) | all 12 runs | 4.1e-4 px | 1.5e-5 |
| `metal:fp16` | 11 of 12 runs identical; 38 of 39 on the other | median 0.005 to 0.034 px | |
| `cuda:fp16` | 9 of 12 runs identical; 1 match differs on each of the other 3 (98.5 to 98.6 %) | median 0.003 to 0.027 px | within 0.09 |

In FP16 a few unconfident queries land on another cell (the `pos max` of several pixels in those runs).
That is the case FlexGlue's own float16 mode is in too: `python tools/compare.py ... --a upstream:cpu
--b upstream:mps,fp16` (or `upstream:cuda,fp16`) shows the original against itself in half precision.

## 4. A backend this machine does not have

`tools/remote.sh` runs the CLI on another machine over SSH: input files are copied there, the result is
copied back. It needs a checkout with a build and `weights/` on that machine. The comparison itself
(and the original under PyTorch) still runs locally:

```sh
python tools/compare.py $A/berlin1.jpg $A/berlin2.jpg \
    --b "cmd:tools/remote.sh user@gpu-box /path/to/just-matching build/release/jmt match --backend cuda"
JMT="tools/remote.sh user@gpu-box /path/to/just-matching build/release/jmt" tools/compare_all.sh cuda
```

## 5. Every stage, not only the output: golden tests

`tools/dump_golden.py` runs the reference forward pass (`tools/folded.py`, checked against the original
by `compare.py --b folded`) on a pair and stores every intermediate result: input, each backbone stage,
the neck, the tokens after every layer, the read-out's coarse positions and masses, the final points and
the matches. `ctest` checks each backend against every stage:

```sh
python tools/dump_golden.py $A/berlin1.jpg $A/berlin2.jpg --size 384 --layers 9
python tools/dump_golden.py $A/lamar_cab_1446.jpg $A/lamar_cab_1450.jpg --size 256 --layers 3
cmake -S . -B build/release && cmake --build build/release && ctest --test-dir build/release
```

One stage per line. A tensor passes when its largest difference is at most `tol` times max(1, `max|ref|`);
points pass when few enough of them move more than the tolerance (an unconfident query whose two best
cells are close may land on the other).
The end of `golden_test weights/flexglue.jmtw build/golden/berlin1-berlin2.jmtt cuda` (384 px, 9 layers,
512 keypoints per image):

```
pair.tokens1_l9        ok    max|diff| 4.09e-04  (max|ref| 4.04e+01, tol 1e-03)
pair.coarse0           ok    median 2.11e-07, 0 of 512 beyond 1e-04
pair.mass0             ok    max|diff| 4.44e-06  (max|ref| 7.35e-01, tol 1e-02)
pair.coarse1           ok    median 7.03e-08, 0 of 512 beyond 1e-04
pair.mass1             ok    max|diff| 1.53e-05  (max|ref| 1.00e+00, tol 1e-02)
out.p2d0_i1            ok    median 6.82e-05, 0 of 512 beyond 5e-02
out.p2d1_i0            ok    median 3.05e-05, 0 of 512 beyond 5e-02
out.covisibility       ok    max|diff| 5.22e-06  (max|ref| 3.22e-01, tol 1e-03)
out.matches0           ok    7 matches, golden 7, same 7
all stages match
```

`batch_test` checks that `jmt_match_batch` gives exactly what `jmt_match` gives pair by pair, and that a
second batch of pairs allocates nothing on the device.

## 6. Speed, measured the same way

```sh
tools/bench.sh cuda:fp16                          # jmt: best of 6 runs, then time per kernel
build/release/jmt match ... --backend cuda --size 384 --layers 9 --runs 8 --batch 8
python tools/bench_torch.py --impl upstream:cuda  # the original under PyTorch, FP32 and FP16, 3 settings
python tools/bench_onnx.py $A/berlin1.jpg $A/berlin2.jpg --providers cuda   # upstream's ONNX models
```

All on the Berlin pair with 2048 keypoints per image, best of 8 runs, match time per pair:

| Machine | Size, layers | jmt FP32 | PyTorch FP32 | jmt FP16 | PyTorch FP16 |
|---|---|---|---|---|---|
| M5 Pro, Metal | 512, 9 | 314 ms | 376 ms | 105 ms | 157 ms |
| | 512, 5 | 180 ms | 215 ms | 62 ms | 92 ms |
| | 384, 9 | 140 ms | 167 ms | 48 ms | 82 ms |
| RTX 2070 Max-Q, CUDA | 512, 9 | 417 ms | 629 ms | 141 ms | 207 ms |
| | 512, 5 | 245 ms | 360 ms | 83 ms | 121 ms |
| | 384, 9 | 190 ms | 249 ms | 54 ms | 97 ms |

ONNX Runtime with upstream's ONNX models (512 px, 9 layers only): 2452 ms on the M5 Pro's CPU, 660 ms
with the CUDA provider. Its FP16 matcher returns NaN on the RTX 2070. Its FP32 matcher returns 90
matches on the Berlin pair where the PyTorch original (and jmt) return 70, with the same covisibility. We
have not traced this; it may scale `max_distance_px` differently. PLAN.md has the details and how each
number was measured.

The laptop's GPU clock moves with its power cap, so its times vary by up to 30 % from run to run. The
tables use the best run.
