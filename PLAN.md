# just-matching — plan

Minimal-dependency image matching for bundle-adjustment pipelines: FlexGlue, written the same way as
`../just-keypoints` (C++ core, hand-written CPU / CUDA / Metal / WebGPU kernels, no PyTorch or ONNX Runtime
at run time), plus a covisibility function that decides which image pairs are worth matching.

Status: M1 and M2 done: the C++ library runs on the CPU and matches the original. No GPU backend yet. Numbers marked "estimate" are computed
from the layer sizes, not measured.

## What FlexGlue is (reviewed at release v0.1, 2026-10-26)

Source: https://github.com/cvg/FlexGlue, paper arXiv:2610.32976.

It is not a descriptor matcher like LightGlue. Three facts shape the whole plan:

1. **It reads the images, not descriptors.** A frozen DINOv3 ConvNeXt-B backbone turns each image
   (resized to 512x512 by default) into a 64x64 grid of 256-dim features. Keypoints are only query
   positions. ALIKED descriptors from just-keypoints are not used; only `xy` is.
2. **It is large.** 567 MB in FP32 (141.7 M parameters: 89.6 M backbone and neck, 52.2 M matcher; ALIKED is 0.68 M
   and 2.7 MB). FP16 halves that to 284 MB.
3. **It already predicts covisibility.** Every transformer layer has a head that scores "do these two images
   show the same place", trained with Doppelgangers negatives, and inference can stop early on pairs that
   score low.

### Graph

- **Per image, once** (`describe`): ImageNet normalization, ConvNeXt-B (depths 3/3/27/3, widths
  128/256/512/1024), a small head that merges the four stages at stride 8 -> `features` 64x64x256.
  About 80 G multiply-adds at 512x512 (estimate). The result can be cached: 2.1 MB per image in FP16.
- **Per pair** (`match`): linear 256 -> 512, then 9 layers of self-attention (rotary position encoding) and
  cross-attention over 2 x 4096 tokens, width 512, 8 heads. About 110 G multiply-adds per layer, 1000 G for
  all 9 (estimate); a third of it is the attention products, the rest linear layers.
- **Per query point** (head): bilinear sample of the token grid, project to 64-dim, cosine similarity
  against the 4096 tokens of the other image, 1.5-entmax, expected position in the 3x3 window around the
  best cell, then an MLP that adds a sub-cell offset (16 bins per axis). Confidence = matchability x window
  mass. Cheap next to the transformer.
- **Sparse mode** (keypoints in both images): predictions in both directions, mutual nearest neighbours
  within 3 px, confidence >= 0.1.
- A head also exists for layer 0 (raw backbone features, no transformer). It gives matches but no
  matchability and no covisibility score.

The authors report 28.7 ms per pair on an RTX 4090 in mixed precision. An independent ONNX Runtime test
reported 122 ms per pair (GPU not stated).

### Operations to write

| Part | Operations |
|---|---|
| Backbone | 4x4 stride-4 conv, 7x7 depthwise conv, 2x2 stride-2 conv, 3x3 conv, LayerNorm over channels, linear / 1x1 conv, GELU, pixel-unshuffle, bilinear resize |
| Transformer | linear, LayerNorm, GELU, rotary encoding, scaled dot-product attention |
| Head | bilinear sampling at points, L2 normalize, similarity product, 1.5-entmax with window read-out, small MLPs (ReLU), softmax, sigmoid |
| Sparse matching | cycle error, mutual nearest neighbour |

Almost all of the time is matrix multiplication (linear layers and attention). In just-keypoints the work was
convolution kernels; here it is one very good GEMM and one attention kernel per backend, in FP16.

## Decisions (proposed)

| Topic | Choice |
|---|---|
| Model | FlexGlue v0.1, the only matcher. No refiner (RoMa v2) in v1. |
| Keypoints | Any `xy` array. The CLI reads just-keypoints `.jkpt` files. RaCo without the ALIKED descriptor pass is enough. |
| Coordinates | Public API uses the just-keypoints convention ((0,0) = centre of the top-left pixel). FlexGlue's corner convention (+0.5) is applied inside. |
| Two-step API | `describe` per image, `match` per pair, so features are computed once and reused over all pairs. |
| Precision | FP32 reference mode matching upstream; FP16 is the fast mode and, unlike in just-keypoints, the expected default on GPU (upstream itself runs the transformer in FP16). |
| Weights | Same flat container as `.jkpw`. Not embedded and not committed (275-550 MB): fetched by script, loaded from path, memory or URL. |
| Speed presets | Number of layers and input size are run-time options. From M1: 512 px with 5-9 layers; 384 px is the smallest useful size; 256 px is not offered. |
| Backends | `cpu` (reference), `metal`, `cuda`, `webgpu`, same pluggable structure as just-keypoints. |
| Code sharing | Start by copying the just-keypoints core (weights loader, buffer pool, backend glue, CMake embedding, test harness). Extract a shared library only if both repos keep changing the same code. |
| License | Code: to be decided. Matcher weights MIT. Backbone weights are under the DINOv3 License: redistribution allowed with a copy of the license, with trade-control and military-use restrictions. |

## Public API (C, plain buffers)

```c
jmt_ctx* jmt_create(const jmt_config*);                       /* backend, weights, precision, input size */
jmt_features* jmt_describe(jmt_ctx*, const jmt_image*);       /* once per image; features stay on the device */
int  jmt_match(jmt_ctx*, const jmt_features* a, const float* xy_a, int na,
               const jmt_features* b, const float* xy_b, int nb,
               const jmt_match_options*, jmt_matches*);
int  jmt_covisibility(jmt_ctx*, const jmt_features* a, const jmt_features* b, int level, float* score);
int  jmt_shortlist(jmt_ctx*, const jmt_features* const* images, int n, int k, jmt_pairs*);
void jmt_features_free(jmt_features*);
void jmt_destroy(jmt_ctx*);
```

- `jmt_matches`: `idx` int32 `[K,2]`, `score` float32 `[K]`, the predicted sub-pixel point for every query in
  both directions (`xy_a_in_b`, `xy_b_in_a`), `covisibility`, `stop_layer`.
- `jmt_match_options`: `n_layers`, `filter_threshold` (0.1), `max_distance_px` (3), `min_covisibility`,
  `early_stop_decay`.
- `jmt_features` can be saved and loaded, so a capture is described once (not written yet).
- Implemented so far: `jmt_create`, `jmt_describe`, `jmt_match`; see `include/jmt/jmt.h`. `jmt_covisibility` and
  `jmt_shortlist` belong to the covisibility track.

## Covisibility

MegaLoc is DINOv2-B plus a trained aggregation layer. FlexGlue already runs a DINOv3 backbone on every image
and already has a covisibility head, so no second network is needed. Three levels, each using only what the
previous steps computed:

| Level | What it does | Cost per pair | Quality |
|---|---|---|---|
| 1. Global vector | Pool the cached 64x64x256 features into one vector per image; cosine similarity; keep the top k per image. Two poolings to try: GeM, and VLAD with a vocabulary clustered from the capture itself (as AnyLoc does). | A dot product | Untested with these features; measure against MegaLoc |
| 2. Layer-0 matching | A coarse grid of queries (e.g. 16x16) through FlexGlue's layer-0 head in both directions; score = share of queries that are confident and cycle-consistent. | ~0.1 G multiply-adds (estimate) | Untested; geometric evidence but no protection against look-alikes |
| 3. FlexGlue head | Run the transformer with early stop; the pair score is the model's own covisibility output. Pairs that pass continue to full depth, so nothing is computed twice. | 220 G (stop at layer 2) to 1000 G | The trained answer, including look-alike scenes |

Proposed default: level 1 (plus neighbours in capture order, when the capture is ordered) builds the shortlist,
level 2 optionally re-ranks it, level 3 gives the verdict during matching. For a small capture (up to ~50
images) level 2 can run on all pairs and level 1 can be skipped.

Alternative considered: ALIKED keypoints on small images with a nearest-neighbour matcher. It works, but it
needs a second network and descriptors FlexGlue does not otherwise use; level 2 is the same idea with
features that are already there.

Levels 1 and 2 are the unknowns. They are tested in Python in M1 on real captures, with MegaLoc as the
reference for recall, before any C++ is written for them.

## Milestones

- [x] **M1 — Reference locked and presets measured**
  Done: upstream pinned at `9992029`, checkpoint `v0.1/flexglue.pth` (sha256 in `tools/fetch_upstream.sh`);
  `weights/flexglue.jmtw` is 870 tensors, 141,741,444 values, 567 MB (284 MB as FP16); not committed.
  The reference forward pass from that file (`tools/folded.py`) matches the original on 3 pairs with RaCo
  keypoints at 512 px and 9 layers: identical matches, predicted points within 2.1e-4 px, confidence within
  6.1e-6, features within 2.0e-5, tokens after layer 9 within 1.9e-4. Also checked at 256 px / 3 layers,
  384 px / 0 layers, and on MPS. Golden dump and `compare.py` exist. Presets and covisibility: see
  "Notes from M1". Not done: the MegaLoc comparison on our own captures (no capture was at hand).
- [x] **M2 — First matches from our own code**
  Done: `jmt match A B --kpts-a A.jkpt --kpts-b B.jkpt` (CPU, C++, no dependencies) matches the original on
  6 pairs (FlexGlue and ALIKED sample images, 2048 RaCo keypoints each), at 512 px / 9 layers and at
  384 px / 5 layers: identical matches (up to 1209 per pair), predicted points within 4.8e-4 px, confidence
  within 1.6e-5, features within 2.5e-5, covisibility equal to 4 digits. Early stop gives the same layer
  and score as the original. Every intermediate tensor of two golden dumps (384 px / 9 layers,
  256 px / 3 layers) matches within 1e-5 of its largest value (tokens after 9 layers: 5.1e-4 on values up to 67). Speed is not a goal here: on the M5 Pro, 0.59 s per
  image and 7.5 s per pair at 512 px / 9 layers (4.4 s with 5 layers, 2.5 s at 384 px); 67 s on one thread.
- [ ] **M3 — Runs on the Mac GPU** (Metal, FP32 then FP16)
- [ ] **M4 — Runs on NVIDIA** (CUDA; order of M3/M4 depends on the machine in use)
- [ ] **M5 — Fast**
  GEMM and attention kernels tuned per backend, pairs batched, zero allocations per pair. Benchmark table
  against PyTorch and the upstream ONNX models.
- [ ] **M6 — Runs in the browser** (WebGPU via Dawn, then Emscripten; FP16 when `shader-f16` is available)
- [ ] **M7 — Demo and docs**: pick images, see the shortlist, matches and timings; just-keypoints supplies the keypoints.
- [ ] **M8 — v1.0**: CMake package, npm package, weights fetch script.

### Covisibility track (parallel, like the RaCo track in just-keypoints)

- [ ] **C1** — levels 1 and 2 evaluated in Python. Partly done in M1 on sample scenes (`tools/covis.py`): all levels
  separate different scenes. Open: recall within one real capture, against MegaLoc; then choose the pooling.
- [ ] **C2** — `jmt_shortlist` and `jmt_covisibility` levels 1-2 on CPU. Needs only `describe` and the layer-0 head, so it can land before the transformer.
- [ ] **C3** — level 3 with early stop, on each backend once that backend passes M2-M4.
- [ ] **C4** — shortlist view in the demo.

## Risks

- **Speed.** A pair at 512 px and 9 layers takes 158 ms in PyTorch on the M5 Pro GPU in FP16 (395 ms in
  FP32), plus 41 ms per image for the features; the authors report 29 ms on an RTX 4090. Input size is not
  the lever hoped for: 256 px loses the accuracy (see Notes from M1). What is left: 5 layers instead of 9
  (223 ms instead of 395 in FP32), 384 px (172 ms), FP16, and the shortlist.
- **Matching upstream's speed.** PyTorch uses vendor GEMM and fused attention. Hand-written kernels beat it
  on ALIKED's small convolutions; matching it on large matrix products is harder.
- **Browser.** A 275 MB download, about 1 GB of GPU memory, and WebGPU buffer limits for the 4096x4096
  attention products (tile them).
- **1.5-entmax** needs a sort or a bisection per query over 4096 values; pick the form that is exact on GPU.
- **Bit-level agreement.** FP16 attention will not reproduce upstream exactly; tests compare matches and
  positions with tolerances, as the RaCo tests do.
- **Upstream is three weeks old** (v0.1); weights and code may still change.

## Notes from M1

- `tools/folded.py` is the executable specification: it uses only the weights file and plain tensor
  operations. `tools/upstream.py` runs the original behind the same interface.
- Export changes (none alters the function): ConvNeXt layer scale folded into `pwconv2`, `Wqkv` rows
  reordered so q, k and v are contiguous, 1x1 convolutions stored as matrices. See `tools/export_weights.py`.
- The layer-0 descriptor (`desc0`, 256 -> 64) is `input_proj` and `heads.0.to_search` fused by upstream.
- 1.5-entmax: upstream computes the threshold from the 512 largest of the 4096 logits (exact when at most
  512 cells have non-zero probability). The reference does the same; a backend needs a top-512 selection per
  query, not a full sort.
- Sparse matching: the nearest neighbour is taken on the sum of the two squared transfer errors, the reported
  distance is the mean of the two errors. Cycle errors are not computed in sparse mode.
- The best cell is an argmax over cosine similarities: a query with two near-equal cells can flip between
  implementations (seen once in 6000 queries at 256 px / 3 layers: one point 23 px away, one match
  different). Tests compare match sets with a tolerance, not exactly.
- FP16 transformer (upstream's default) against FP32, same pairs: same matches, points move by 0.007 px
  (median), covisibility by 2e-4.
- Sparse mode needs keypoints that repeat in both images. Berlin pair, 2048 keypoints per image: 70 matches
  with RaCo, 194 with ALIKED; 219 and 418 with 4096. The second image is a close-up, so only 102 of the
  2048 RaCo keypoints of the first have a confident counterpart at all.

### Presets (`tools/presets.py`)

Original model, PyTorch on the M5 Pro GPU. 8 synthetic pairs (4 photos, each warped by a mild and a strong
homography with a brightness change), 2048 RaCo keypoints per image, so the truth is exact. This is easier
than real pairs; read it as trends.

| Size | Layers | Precision | Median error | Within 1 px | Within 3 px | Sparse matches | Correct | Describe | Match |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 9 | FP32 | 0.20 px | 98.9 % | 99.8 % | 1475 | 99.5 % | 44 ms | 395 ms |
| 512 | 9 | FP16 | 0.20 px | 98.9 % | 99.8 % | 1476 | 99.5 % | 40 ms | 158 ms |
| 640 | 9 | FP32 | 0.18 px | 99.2 % | 99.9 % | 1488 | 99.7 % | 61 ms | 770 ms |
| 384 | 9 | FP32 | 0.40 px | 93.3 % | 99.5 % | 1490 | 99.2 % | 24 ms | 172 ms |
| 256 | 9 | FP32 | 2.06 px | 15.5 % | 76.1 % | 1446 | 95.1 % | 13 ms | 69 ms |
| 512 | 7 | FP32 | 0.22 px | 98.4 % | 99.8 % | 1501 | 99.5 % | 40 ms | 303 ms |
| 512 | 5 | FP32 | 0.33 px | 95.3 % | 99.8 % | 1568 | 99.5 % | 41 ms | 223 ms |
| 512 | 3 | FP32 | 0.72 px | 68.8 % | 98.6 % | 1558 | 99.2 % | 41 ms | 137 ms |
| 512 | 2 | FP32 | 1.33 px | 34.4 % | 90.2 % | 1464 | 98.0 % | 41 ms | 99 ms |
| 512 | 1 | FP32 | 1.94 px | 19.3 % | 73.4 % | 1282 | 96.5 % | 41 ms | 57 ms |
| 512 | 0 | FP32 | 4.53 px | 3.8 % | 28.0 % | 877 | 87.3 % | 41 ms | 15 ms |
| 256 | 5 | FP32 | 3.07 px | 6.7 % | 48.6 % | 1443 | 89.2 % | 13 ms | 45 ms |

Errors are in pixels of the original images (about 1000 px wide). Error and "within" are over all keypoints
whose true position is inside the other image; "correct" is the share of sparse matches within 3 px.

- 256 px is out: 2 px median error, and the real Berlin pair drops from 70 matches to 10 and from a
  covisibility score of 0.99 to 0.36.
- 5 layers cost little accuracy here; 3 or fewer lose sub-pixel accuracy. Whether 5 holds on hard real pairs
  is not known from this test.
- Time grows faster than the token count from 384 to 640 px, as expected from attention.

### Covisibility (`tools/covis.py`)

53 images: 20 TUM frames, 20 KITTI frames, 5 + 5 photos of two landmarks, 3 single images; 1378 pairs, 400
in the same scene. "Covisible" here means "same scene", so this tests scene separation only.

| Score | AUC | Same scene: lowest / median | Other scene: median / 99th percentile / highest |
|---|---|---|---|
| Level 1, mean feature vector | 0.9997 | 0.921 / 0.995 | 0.882 / 0.921 / 0.987 |
| Level 1, VLAD (16 centres from these images) | 0.9982 | 0.242 / 0.707 | 0.022 / 0.271 / 0.762 |
| Level 2, layer-0 grid round trip | 0.9998 | 0.096 / 0.626 | 0.059 / 0.104 / 0.291 |
| Level 3, FlexGlue after 2 layers | 0.9996 | 0.632 / 1.000 | 0.441 / 0.770 / 0.935 |
| Level 3, FlexGlue after 3 layers | 1.0000 | 0.907 / 0.999 | 0.342 / 0.905 / 0.987 |
| Level 3, FlexGlue after 9 layers | 1.0000 | 0.999 / 1.000 | 0.480 / 0.914 / 0.992 |

- FlexGlue's score is not near zero for unrelated scenes: typically 0.3-0.7. It is near zero for look-alikes
  (the LaMAR pair: 0.06). The threshold has to be high: at >= 0.8 after 3 layers every same-scene pair is
  kept and 1.6 % of the others; upstream's example also uses 0.8.
- After 1 layer the score is not usable (AUC 0.947); early stop from layer 2 as upstream does.
- Levels 1 and 2 separate scenes about as well as FlexGlue's score at a fraction of the cost (level 2:
  6.7 ms per pair in PyTorch; level 3: 57 ms for one layer, 395 ms for nine). The mean vector has a narrow
  range (0.88 against 0.99), VLAD and the grid score a wide one.
- Not tested: ranking partial overlap inside one capture, which is the real job of the shortlist. Needs a
  capture with known poses or MegaLoc as the reference.

## Notes from M2

- The graph (`src/models/flexglue/`) is written once against `src/core/ops.h`: 17 operations on token-major
  tensors (h, w, c). A GPU backend implements that interface. The head's last steps (offset expectation,
  sigmoid, mutual nearest neighbours) run on the host from values read back: a few thousand numbers per pair.
  Expect to revise the interface when the first GPU backend shows what it needs (fused blocks, no read-backs
  inside the layer loop for early stop).
- Convolutions with a stride (stem 4x4, downsampling 2x2) and the neck's 3x3 are patch extraction plus one
  matrix product; everything else heavy is `linear`, `attention` and `dwconv7`. One good matrix product per
  backend covers most of the time.
- The CPU matrix product is blocked (32 rows by a 256 KB slab of weights) and written as `y += a * x` loops
  so the compiler vectorizes it without `-ffast-math`. Unlike just-keypoints the library is built with the
  compiler's default floating-point contraction: FP16 on the GPU backends rules out bit-level agreement
  anyway, and fused multiply-add is worth the speed here.
- Weights are held twice on the CPU backend (as stored, and transposed on first use): about 1.1 GB.
- The public API takes and returns keypoints in the just-keypoints convention and adds 0.5 inside.
- `tools/compare.py` writes lossless PNGs for the program under test: JPEG decoders differ, and both sides
  must see the same pixels for the comparison to mean anything.
- `tests/golden_test.cpp` checks every stage against a golden dump (`ctest`, 2 dumps, 5 s) and is the main
  debugging tool for new backends. Golden maps are stored (c, h, w); the test converts.
