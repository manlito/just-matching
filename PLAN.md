# just-matching — plan

Minimal-dependency image matching for bundle-adjustment pipelines: FlexGlue, written the same way as
`../just-keypoints` (C++ core, hand-written CPU / CUDA / Metal / WebGPU kernels, no PyTorch or ONNX Runtime
at run time), plus a covisibility function that decides which image pairs are worth matching.

Status: M1-M4 done: the C++ library runs on the CPU, on the Mac GPU (Metal) and on NVIDIA (CUDA) and matches
the original. M5 is done on Metal: 314 ms per pair in FP32 and 105 ms in FP16 at full settings on an M5 Pro,
against 376 and 157 ms for PyTorch on the same GPU, and on CUDA. M6 is done: the WebGPU backend matches the
original natively (Dawn) and in Chrome (WebAssembly): per pair at full settings on the M5 Pro, 478 ms in
FP32 and 313 ms in FP16 natively, 500 and 335 ms in Chrome. C2 is done: `jmt_covisibility` (levels 1 and 2)
and `jmt_shortlist` on every backend. C3 is done: level 3 with early stop stops where the original does on
every backend. Numbers marked "estimate" are computed from the layer sizes, not measured.

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
int  jmt_match_batch(jmt_ctx*, const jmt_pair* pairs, int n_pairs, const jmt_match_options*, jmt_matches*);
int  jmt_covisibility(jmt_ctx*, const jmt_features* a, const jmt_features* b, int level, float* score);
int  jmt_shortlist(jmt_ctx*, const jmt_features* const* images, int n, const jmt_shortlist_options*,
                   jmt_pair_list*);
void jmt_features_free(jmt_features*);
void jmt_destroy(jmt_ctx*);
```

- `jmt_matches`: `idx` int32 `[K,2]`, `score` float32 `[K]`, the predicted sub-pixel point for every query in
  both directions (`xy_a_in_b`, `xy_b_in_a`), `covisibility`, `stop_layer`.
- `jmt_match_options`: `n_layers`, `filter_threshold` (0.1), `max_distance_px` (3), `min_covisibility`,
  `early_stop_decay`.
- `jmt_features` can be saved and loaded, so a capture is described once (not written yet).
- `jmt_shortlist_options`: `k` (partners per image by level 1), `neighbours` (capture order), `rerank`
  (level 2), `min_score`. `jmt_pair_list`: `idx` int32 `[n,2]`, `score`, best first.
- Implemented so far: everything above except saving features; see `include/jmt/jmt.h`.

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
- [x] **M3 — Runs on the Mac GPU** (Metal, FP32 and FP16)
  Done (M5 Pro): `jmt match --backend metal` matches the original on the same 6 pairs and 2 settings as M2:
  identical matches, predicted points within 4.5e-4 px, confidence within 1.4e-5, features within 2.5e-5;
  every intermediate tensor matches the golden dumps as on the CPU. `--backend metal:fp16` (tensors and
  weights stored as half, arithmetic in float): 5 of 6 pairs with identical matches, 686 of 687 on the
  sixth; confident points move by 0.003-0.02 px (median), covisibility by at most 3e-4.
  Not tuned yet, per pair with 2048 keypoints per image, best of 4 runs:

  | Size, layers | Metal FP32 | Metal FP16 | PyTorch MPS FP32 / FP16 | CPU |
  |---|---|---|---|---|
  | 512, 9 | 1090 ms | 918 ms | 395 / 158 ms | 7.5 s |
  | 512, 5 | 625 ms | 519 ms | 223 ms / - | 4.4 s |
  | 384, 9 | 452 ms | 377 ms | 172 ms / - | 2.5 s |

  Describe: 87 ms per image in FP32, 67 ms in FP16 (PyTorch MPS: 44 ms; CPU: 590 ms).
- [x] **M4 — Runs on NVIDIA** (CUDA; order of M3/M4 depends on the machine in use)
  Done: `src/backend/cuda/cuda_ops.cu`, FP32 and FP16 (`--backend cuda`, `cuda:fp16`). FP32 gives the
  original's matches on the 6 pairs x 2 settings (points within 1.9e-4 px). Per pair at 512 px / 9 layers
  on an RTX 2070 Max-Q: 726 ms in FP32, 349 ms in FP16 (tensor cores). See Notes from M4.
- [x] **M5 — Fast**
  GEMM and attention kernels tuned per backend, pairs batched, zero allocations per pair. Benchmark table
  against PyTorch and the upstream ONNX models.
  Done on Metal (M5 Pro, macOS 26). Per pair, 2048 keypoints per image, Berlin pair, best of 8 runs;
  PyTorch measured the same way on the same day (`tools/bench_torch.py`):

  | Size, layers | Metal FP32 | PyTorch MPS FP32 | Metal FP16 | PyTorch MPS FP16 | Metal FP32 / FP16 after M3 |
  |---|---|---|---|---|---|
  | 512, 9 | 314 ms | 376 ms | 105 ms | 157 ms | 1090 / 918 ms |
  | 512, 5 | 180 ms | 215 ms | 62 ms | 92 ms | 625 / 519 ms |
  | 384, 9 | 140 ms | 167 ms | 48 ms | 82 ms | 452 / 377 ms |

  Describe, per image: 31 ms in FP32 and 13 ms in FP16 at 512 px, 19 and 8 ms at 384 px (PyTorch: 39 and
  23 ms in both precisions; after M3: 87 and 67 ms at 512 px).
  In a batch of 8 (`jmt_match_batch`): 311 / 176 / 137 ms per pair in FP32 and 103 / 59 / 45 ms in FP16.
  Agreement with the original on the 6 pairs x 2 settings (`tools/compare_all.sh`): FP32 identical matches
  on all 12 runs, points within 1.9e-4 px, confidences within 5.6e-6, features within 3.1e-5. FP16: identical
  matches on 11 of the 12 runs and 38 of 39 on the other; confident points move by 0.005-0.034 px (median),
  covisibility by at most 3.2e-3.
  The second batch of the same pairs creates no device buffers (`tests/batch_test.cpp`).
  See "Notes from M5, second pass". The fast kernels need macOS 26; older systems use the kernels of the
  first pass (786 ms in FP32, 549 ms in FP16).
  Done on CUDA (RTX 2070 Max-Q laptop): the same hooks, attention without the logits buffer in both modes
  (tensor cores, `mma.sync` from inline PTX; FP32 on operands split in two halves), the read-out and
  preprocessing on the device. Per pair,
  2048 keypoints per image, Berlin pair, best of 8 runs over three sweeps; PyTorch (2.15, CUDA 13.2,
  `tools/bench_torch.py --impl upstream:cuda`) measured the same way on the same day:

  | Size, layers | CUDA FP32 | PyTorch CUDA FP32 | CUDA FP16 | PyTorch CUDA FP16 | FP32 / FP16 after M4 |
  |---|---|---|---|---|---|
  | 512, 9 | 417 ms | 629 ms | 141 ms | 207 ms | 726 / 349 ms |
  | 512, 5 | 245 ms | 360 ms | 83 ms | 121 ms | 434 / 219 ms |
  | 384, 9 | 190 ms | 249 ms | 54 ms | 97 ms | 331 / 173 ms |

  Describe, per image: 54 ms in FP32 and 18 ms in FP16 at 512 px, 40 and 14 ms at 384 px (PyTorch: 54
  and 31 ms in both precisions; after M4: 66 / 30 and 47 / 19 ms).
  ONNX Runtime 1.32, CUDA provider, upstream's release v0.1 models (512 px, 9 layers only): describe 66
  ms, match 660 ms with the FP32 matcher (90 matches, see the Mac notes); the FP16 matcher runs in 184 ms
  but returns NaN covisibility and no matches on this GPU. FP16 is 1.5 to 1.8 times PyTorch's speed,
  FP32 1.3 to 1.5 times. With its attention through the float logits buffer (`JMT_CUDA_SPLIT=0`), FP32
  takes 634 / 367 / 282 ms. In a batch of 8: 419 / 246 / 191 ms per pair in FP32 and 144 / 85 / 66 ms in
  FP16: no gain, one pair already fills the GPU. GPU memory in use: 0.93 GB in FP32, 0.54 GB in FP16 (M4:
  1.5 / 0.8 GB).
  Agreement (`tools/compare_all.sh` through `tools/remote.sh`, 6 pairs x 2 settings): FP32 identical
  matches on all 12 runs, points within 4.1e-4 px, confidences within 1.5e-5 (through the logits buffer:
  2.5e-4 px, 1.1e-5). FP16: 98.5 to 100 % of the
  matches per pair (identical on 9 of the 12 runs; 1 match differs on each of the others), as after M4.
  The second batch creates no device buffers on CUDA either. See "Notes from M5 (CUDA)".
  Open: the products of describe at small grids, slices and concatenations as views.
- [x] **M6 — Runs in the browser** (WebGPU via Dawn, then Emscripten; FP16 when `shader-f16` is available)
  Done (M5 Pro, macOS 26, Dawn v20261113.214538 natively, Chrome in the desktop app's browser pane):
  `src/backend/webgpu/` (WGSL kernels, host over `webgpu.h`, the same source natively and in the browser),
  `--backend webgpu` / `webgpu:fp16`, and the WebAssembly module with `web/jmt.js` and `web/test.html`.
  - Natively, against the original (`tools/compare_all.sh`, 6 pairs x 2 settings): FP32 identical matches
    on all 12 runs, points within 2.4e-4 px, confidences within 1.5e-5, features within 2.7e-5 (Metal:
    1.9e-4, 5.6e-6, 3.1e-5). FP16: identical matches on 11 of the 12 runs, 65 of 66 on the other;
    confident points move by 0.005-0.034 px (median), covisibility by at most 2.4e-3 (Metal FP16: 11 of 12,
    38 of 39; 0.005-0.034 px). Every intermediate tensor of both golden dumps matches in both precisions;
    every operation matches the CPU backend (`tests/ops_test.cpp`, new).
  - In Chrome (`web/test.html`, the same 12 runs on raw RGBA from `tools/web_testdata.py`, against the
    native CPU backend): FP32 gives the native matches on all 12 runs, points within 3.1e-4 px,
    covisibility within 1.8e-6. FP16: identical on 7 of 12; on the others 1 or 2 matches differ, and 5 of
    70 on the Berlin pair at 512 px (native WebGPU FP16: none). The results repeat exactly; the difference
    is the half-precision attention (`k_attnh`), which Chrome compiles less closely than native Dawn: with
    f32 attention the Berlin pair has all 70, at 407 instead of 335 ms per pair. Kept for its speed; the
    page's FP16 bar was loosened from 3 % to 10 % of the matches. One confident query of the LaMAR pair at
    384 px moves 180 px in Chrome FP16, with either attention. The FP16 weights file (284 MB to download
    instead of 567, widened to float32 on load) passed the FP16 runs with the first kernels; not re-run.
  - Speed, per pair, 2048 keypoints per image, Berlin pair, best of 8 (`jmt match --runs 8`):

    | Size, layers | WebGPU FP32 | WebGPU FP16 | Metal FP32 | Metal FP16 | CPU | Chrome FP32 / FP16 |
    |---|---|---|---|---|---|---|
    | 512, 9 | 478 ms | 313 ms | 315 ms | 105 ms | 7.2 s | 499-502 / 333-337 ms |
    | 512, 5 | 271 ms | 178 ms | 181 ms | 62 ms | 4.3 s | |
    | 384, 9 | 209 ms | 139 ms | 141 ms | 49 ms | 2.5 s | |
    | 384, 5 | 121 ms | 81 ms | 82 ms | 29 ms | 1.4 s | 139-147 / 91-103 ms |

    CPU: the native CPU backend on all cores of the M5 Pro, best of 3. Chrome: the warm runs of the test
    page (pairs 2 to 6). Describe, per image: 46 / 30 ms at 512 px and 28 / 18 ms at 384 px in FP32 / FP16
    (Metal: 32 / 13 and 20 / 8 ms; CPU: 0.41 and 0.27 s); 50 / 33 and 30 / 20 ms in Chrome. The first
    version of these kernels took 496 / 499 ms per pair and 50 / 48 ms per image; the second pass (Notes
    from M6) is half arithmetic in FP16 and operands read straight from the buffers.
  - Module: 302 KB of wasm and 100 KB of JS; weights fetched separately.
  - Not checked: Safari and Firefox (JSPI), Dawn on Vulkan or Direct3D 12, a GPU other than the M5 Pro.
- [ ] **M7 — Demo and docs**: pick images, see the shortlist, matches and timings; just-keypoints supplies the keypoints.
- [ ] **M8 — v1.0**: CMake package, npm package, weights fetch script.

### Covisibility track (parallel, like the RaCo track in just-keypoints)

- [ ] **C1** — levels 1 and 2 evaluated in Python. Partly done in M1 on sample scenes (`tools/covis.py`): all levels
  separate different scenes. Open: recall within one real capture, against MegaLoc; then choose the pooling.
- [x] **C2** — `jmt_shortlist` and `jmt_covisibility` levels 1-2 on CPU.
  Done, on every backend (it is written against the backend operations, like the rest of the graph):
  level 1 is the mean feature vector (the pooling is C1's open choice), level 2 the layer-0 grid round
  trip of `tools/covis.py`. `jmt shortlist IMAGE... [--k K] [--neighbours N] [--rerank] [--min-score S]`.
  Against the reference (`tools/compare_covis.py`, 8 images of 5 scenes, all 28 pairs): level 1 within
  6.0e-5 (1.2e-4 in FP16), level 2 within 3 of 512 grid points (4 in WebGPU FP16) on CPU, Metal and WebGPU
  in both precisions. On the 53 images of `covis.py` with k = 5 it keeps the same 156 pairs. See "Notes from C2".
- [x] **C3** — level 3 with early stop, on each backend once that backend passes M2-M4.
  Done: `min_covisibility` and `early_stop_decay` in `jmt_match_options` (`--min-covisibility`,
  `--early-stop`), the original's rule after every layer from 2 on. A pair that passes runs on to full
  depth in the same pass and is matched; nothing is computed twice. Against the original
  (`tools/compare_all.sh B --min-covisibility 0.8 --early-stop D`, D = 0, 0.2, 0.5, which stop the 6 pairs
  after layer 2, 4 or 8 or not at all): the same stop layer on every pair in all 42 runs (CPU, Metal, CUDA,
  WebGPU, each GPU one in FP32 and FP16); the original's matches in FP32, and in FP16 the agreement of M5
  (1 of 70 differs on the Berlin pair in CUDA FP16). `batch_test` checks a batch against single pairs with
  early stop too. A pair stopped after layer 2 costs about a quarter of a full one (24-29 %); the check
  after each layer costs 1 to 3.4 % on pairs that pass. See "Notes from C3".
- [ ] **C4** — shortlist view in the demo.

## Risks

- **Speed.** A pair at 512 px and 9 layers takes 105 ms on the M5 Pro GPU in FP16 (314 ms in FP32), plus
  13 ms per image for the features; the authors report 29 ms on an RTX 4090. Input size is not the lever
  hoped for: 256 px loses the accuracy (see Notes from M1). What is left: 5 layers instead of 9 (62 ms in
  FP16), 384 px (48 ms), and the shortlist.
- **Matching upstream's speed.** Settled on Metal (1.2x faster than PyTorch in FP32, 1.5x in FP16) once the
  kernels used the same matrix hardware as the vendor library. Open on CUDA: 70 % of cuBLAS in FP32.
- **The fast Metal path depends on Metal 4** (macOS 26) and, for its FP16 speed, on the matrix hardware of
  the M5 GPU. On older chips it has not been measured.
- **Browser.** A 284 MB download (FP16 weights file) and about 1 GB of GPU memory. Attention needs no
  4096 x 4096 buffer (single pass). Speed is set by WGSL's lack of matrix instructions: 1.5x Metal FP32
  and 3.0x Metal FP16 on the M5 Pro. Dawn's experimental subgroup matrices would reach the matrix hardware
  but are not in stable browsers.
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

## Notes from M3

- `src/backend/metal/kernels.metal` and `metal_ops.mm`; dependencies: Metal.framework and Foundation only.
  The `Ops` interface did not change. Context, buffer pool, run-time shader compilation and the profiler
  (`JMT_METAL_PROFILE=1`) follow just-keypoints.
- The matrix product (`k_linear`) is register-tiled: one thread computes 4 tokens x 8 output channels, with
  weights stored input-major and padded so the 8 weights of an input are adjacent. Of the variants tried
  (`JMT_METAL_LINEAR`), 4 x 8 is the fastest: 1090 ms per pair against 1300 (2 x 8), 1417 (8 x 8),
  1651 (4 x 16) and 1685 (1 x 8). All 354 products of a pair take 420 ms of GPU time.
- Attention, first attempt: one thread per (query, head) walking the 4096 source tokens with a running
  softmax, 100+ vector registers of state. 18 s per pair, 97 % of the GPU time: per-thread state that large
  is what makes a GPU kernel slow, not the arithmetic. Now four passes over a float buffer (heads, P, S):
  transpose the keys, logits as a tiled product, exp and row sums, weighted sum of the values as a tiled
  product. 730 ms per pair, and each product runs at the speed of `k_linear`.
- That buffer is 512 MB at 512 px (8 heads x 4096 x 4096 floats) and is float in FP16 mode too. Peak
  memory of a run is 3.8 GB, of which 1.1 GB is the weights held twice (host copy and device copy).
- GPU time per pair at 512 px / 9 layers, FP32: products 420 ms, softmax 280 ms, attention values 235 ms,
  attention logits 220 ms, everything else 70 ms. Next for M5: the softmax (one thread per row, three
  scalar passes over 4096 values), fusing the small operations around each product (bias, norm, GELU, add),
  and FP16 arithmetic where it is safe. FP16 storage alone gains only 16 %.
- The read-out computes its logits on the GPU and selects the 512 largest per query on host threads
  (shared memory, no copy). A bisection for the entmax threshold would keep it on the GPU.
- Image preprocessing runs on the host (512 x 512 x 3 values).
- GELU uses a polynomial erf (Abramowitz-Stegun 7.1.26, error below 1.5e-7): Metal has no erf.
- FP16 golden tests are a sanity check only: on a hard pair up to 10 % of the unconfident queries land on
  another cell. `tools/compare.py`, which looks at confident queries and matches, is the FP16 check.
- The CLI accepts options before, between or after the two images, so `compare.py` can pass
  `"cmd:build/release/jmt match --backend metal"`.

### Against the upstream ONNX models (`tools/bench_onnx.py`)

ONNX Runtime 1.32 on the same M5 Pro, upstream's release v0.1 models (512 px, 9 layers), Berlin pair with
2048 RaCo keypoints per image, best of 3 runs after a warm-up:

| Runtime | Describe, per image | Match, per pair |
|---|---|---|
| ONNX Runtime, CPU provider, FP32 matcher | 163 ms | 2452 ms |
| ONNX Runtime, CPU provider, FP16 matcher | 163 ms | 3025 ms |
| ONNX Runtime, CoreML provider, FP32 matcher | 201 ms | 5130 ms |
| ONNX Runtime, CoreML provider, FP16 matcher | 201 ms | 3090 ms |
| jmt CPU | 590 ms | 7500 ms |
| jmt Metal FP32 | 31 ms | 314 ms |
| jmt Metal FP16 | 13 ms | 105 ms |
| PyTorch MPS FP32 / FP16 | 39 ms | 376 / 157 ms |

- ONNX Runtime has no useful GPU path on the Mac: CoreML takes 194 of the matcher's 2107 nodes (23
  partitions) and the rest runs on the CPU, which is slower than the CPU provider alone. Its CUDA provider
  can only be measured on the NVIDIA machine (M4).
- jmt on Metal is 7.8x faster than the best ONNX Runtime result (23x in FP16); jmt on the CPU is 3x
  slower than ONNX Runtime on the CPU, as expected from a reference backend. (The ONNX Runtime rows were
  measured before the second pass of M5 and not repeated; the jmt and PyTorch rows are current.)
- The ONNX matcher returns 90 matches on this pair where the PyTorch model (and jmt) returns 70, with the
  same covisibility score. Not investigated; its documentation describes `max_distance_px` as scaled by
  the long image side, where the PyTorch code scales each axis by the squashed size, which would accept
  more matches on a non-square image.

## Notes from M4 (CUDA)

Machine: laptop, RTX 2070 Max-Q (8 GB, power-capped near 78 W), CUDA 13.4, 12 CPU threads.

- `src/backend/cuda/cuda_ops.cu`, one file; dependencies: the CUDA toolkit only (no cuBLAS, no cuDNN).
  The `Ops` interface did not change. Kernels are templates on the storage type (float or half); pool,
  profiler (`JMT_CUDA_PROFILE=1`) and build option (`JMT_CUDA`) follow just-keypoints.
- **Agreement** (`tools/compare.py` driving the laptop through `tools/remote.sh`, against upstream on the
  CPU; 6 pairs, at 512 px / 9 layers and 384 px / 5 layers). FP32: identical matches on all 12 runs, points
  within 1.9e-4 px, confidences within 1.3e-5. FP16: 97.0 to 100 % of the matches per pair (the same on 9
  of the 12 runs; 1, 1 and 2 matches differ on the others), median point difference 0.03 px or less,
  confidences within 0.12, as on Metal. Early stop gives the same result as Metal (LaMAR
  pair, `--min-covisibility 0.5 --early-stop 0.3`: stops after 3 layers, score 0.0541). Golden tests pass;
  the FP16 one needed its allowance for unconfident queries at 256 px raised from 15 % to 25 % (CUDA
  moves 18 % of them, Metal 11 %).
- **One matrix-product kernel** (`k_gemm`) serves the linear layers, the convolutions, both products of
  attention and the read-out logits. A block of 16 x 16 threads computes a 128 x 128 tile of the output
  (8 x 8 values per thread, in registers) from tiles of both operands in shared memory; operands are
  addressed with strides, so keys, values and search tokens are used in place.
- **Attention in two passes instead of three.** The logits pass also records each row's maximum (folded
  within a warp, then one atomic update per row and block); the value pass takes the exponentials while it
  loads its tiles and collects the row sums on the way. This removed the separate softmax pass over the
  512 MB buffer (257 ms per pair).
- What moved the time, per pair at 512 px / 9 layers, FP32:

  | Step | Match |
  |---|---|
  | First working version (three attention passes) | 1470 ms |
  | Softmax folded into the value pass | 1267 ms |
  | Tile shape scan (thread 4x4 ... 8x8, block 64 ... 512 threads) | no gain, 8x8 / 256 stays |
  | Next tile fetched into registers while the current one is computed | 1171 ms |
  | Row maxima folded within a warp before the atomic update | 956 ms |

- **Timings** (2048 keypoints per image, best of 5), after the second pass below:

  | Size, layers | CUDA FP32 | CUDA FP16 | Describe, per image (FP32 / FP16) |
  |---|---|---|---|
  | 512, 9 | 726 ms | 349 ms | 66 / 30 ms |
  | 512, 5 | 434 ms | 219 ms | 67 / 31 ms |
  | 384, 9 | 331 ms | 173 ms | 47 / 19 ms |

  CPU backend on this laptop: 38.7 s per pair, 1.8 s describe. GPU memory in use: 1.5 GB in FP32, 0.8 GB
  in FP16. FP16 kernel time per pair: attention 160 to 200 ms, other products 64, layer norm 40,
  depthwise convolution 11, rest 20.
- Not measured: PyTorch and ONNX Runtime on this GPU (neither is installed on the laptop).
- Not on the device yet, as on Metal: preprocessing and the read-out's top-512 selection.

### Second pass: why cuBLAS was faster, and what closed the gap

Method: `k_gemm` and cuBLAS run side by side in one program on the same shapes (the GPU's clock moves with
its power cap, so only interleaved numbers compare), the inner dimension varied to separate the fixed cost
of a call from its slope, and the compiled kernel disassembled (`cuobjdump -sass`) and its resources read
(`-Xptxas -v`). Rates in T (10^12) multiply-adds per second, n 4096, k 1024, m 1024 unless noted.

FP32 (cuBLAS: 3.5 T, fixed cost 80 us per call):

| Step | k_gemm | Fixed cost |
|---|---|---|
| Start | 1.75 T | 320 us |
| Index arithmetic and bounds checks out of the loop | 1.77 T | 320 us |
| Warp of 8 x 4 threads and bank-ordered shared tiles; full tiles stored 4 values at a time | 2.37 T | 95 us |
| Two copies of the tiles, one barrier per step | 2.33 T (no gain, kept) | |
| 16 columns per step instead of 8 | 1.96 T (worse: 168 registers, one block per SM) | |

- **Shared-memory bank conflicts** were the main loss in the loop. With 16 x 16 threads per block a warp
  held 16 threads along the outputs, whose 128-bit reads of the w tile fell 4 deep on the same banks.
  A warp of 8 (outputs) x 4 (rows) threads, with each tile stored so that those reads are consecutive,
  removes them.
- **The fixed cost** was 64 guarded scalar stores per thread; it decides the short products (k = 64:
  the attention logits).
- What did not matter: threads per block and blocks per SM (the first scan), the second barrier.
- Still open against cuBLAS in FP32 (about 70 % of it): not investigated further.

FP16:

- **Tensor cores are usable from hand-written kernels**: `nvcuda::wmma` in the toolkit's `mma.h` compiles
  to the HMMA instructions (16 x 16 x 16 products, half operands, float or half accumulation). No library.
  The earlier note that they are reachable only through NVIDIA's libraries was wrong.
- cuBLAS on the tensor cores: 22 T with half accumulation (`CUBLAS_COMPUTE_16F`, what the first
  measurement used), 13.6 T with float accumulation. `k_gemm_tc` accumulates in float, so 13.6 T is the
  number to compare with. Half accumulation was not adopted: sums of up to 4096 products in half.

| Step | k_gemm_tc | Fixed cost (m 4096) |
|---|---|---|
| First version: tiles with rows of KC values, w stored by column | 5.7 T | 500 us |
| One or two blocks per SM, float or half accumulators | no change | |
| Rows of KC + 8 values in both tiles, w stored by output | 8.5 T (KC 32), 9.7 T (KC 16) | |
| Accumulators read out conflict-free (a thread takes one column of 8 rows) | 10.2 T | 360 us |

- Bank conflicts again, worse: with rows of 32 halves the fragment loads of a warp hit two banks.
- **Attention logits as half without losing the softmax**: a logit is stored relative to the largest of
  its group (its row, 32 consecutive sources), and the group maxima (float) go to a small side table.
  Logits near the top of their group, the ones with weight, are small numbers; the coarse ones have
  weight e^-8 or less. The buffer drops from 512 to 256 MB.
- Linear layers in FP16 mode keep their weights as stored, (out, in): the tensor-core kernel wants the
  columns contiguous.
- Agreement after this pass (same 6 pairs x 2 settings): FP32 identical matches on all 12 runs (points
  within 1.9e-4 px); FP16 98.5 to 100 % of the matches per pair (identical on 10 of the 12 runs),
  confidences within 0.12. Golden tests pass in both modes; early stop unchanged.
- Result per pair, 512 px / 9 layers: FP32 956 -> 726 ms, FP16 942 -> 349 ms; describe 84 -> 66 ms and
  80 -> 30 ms.
- Open: attention is still 160 to 200 ms of the FP16 pair (logits 3 ms per call, bound by the read-out of
  16 M values per head through shared memory); layer norm is now 12 % of the kernel time. The same
  bank-conflict review has not been done for the Metal kernels, where threadgroup memory is not used for
  the products.

## Notes from M5, first pass (Metal, plain kernels)

These are the kernels a system older than macOS 26 still runs.

What the measurements showed, in the order tried. All times are GPU time per pair at 512 px / 9 layers.

- **Attention is bound by memory traffic through its weights buffer** (heads x P x S values, 512 MB in
  float). Vectorizing the softmax pass changed nothing (281 -> 271 ms). Fusing logits and softmax into one
  kernel did: a threadgroup holds 4 queries of one head (one thread per 8 source tokens), the row maximum
  and row sum are reduced in threadgroup memory, and the buffer is written once and read once instead of
  written twice and read three times. Logits + softmax: 500 -> 280 ms. In FP16 the buffer is half: 162 ms.
- **The matrix product is bound by how often a thread reloads weights, and by registers.** Time is
  inversely proportional to the tokens per thread (1 x 8: 1514 ms, 2 x 8: 761, 4 x 8: 431), and collapses
  once a thread holds more than 32 float accumulators (8 x 8: 836, 4 x 16: 1187, 16 x 4: 7183).
  8 tokens x 4 channels is the best shape: 318 ms.
- **Did nothing:** storing weights contiguously per output block (431 -> 431), loading the inputs four at
  a time (431 -> 431), the shape of the attention value product (4 x 8, 2 x 16, 1 x 32, 8 x 4: 212-274 ms).
  32-bit instead of 64-bit offsets in the inner loops gave 7 %.
- **Made it worse:** grouping the product's threads by output block instead of by token block (431 ->
  1232 ms: the threads of a group must share the same input rows). Running attention in chunks of queries
  to keep the buffer in cache (4 chunks: +8 %, 64 chunks: 4x slower): a small dispatch does not fill the GPU.
  The chunk size stays as an option (`JMT_METAL_ATTENTION_CHUNK`) because it bounds the buffer's memory.
- **Half-precision accumulators** in the products (FP16 mode) gained 10-25 % on the products and changed
  matches; not kept.
- **Ceiling of this kernel style:** with its memory reads confined to a few kilobytes the attention value
  product still takes 149 of 214 ms, so about 2.1 T multiply-adds per second is what a thread with 32
  accumulators reaches on this GPU; the kernels run at 1.3-1.5 T. PyTorch's FP32 time corresponds to about
  2.9 T over the whole pair, so closing the gap needs a different structure, not more tuning of this one.
- Where the time is now, FP32: products 318 ms, attention weights 290, attention values 215, layer norm
  31, the rest 30. FP16: 245, 162, 176, 25, 20.
- Not done: fusing the small operations around each product (about 60 ms in total), the read-out's
  selection on the GPU, preprocessing on the GPU, batching pairs.
- Peak memory: 3.1 GB in FP16 (M3: 3.8 GB in FP32).
- `tools/bench.sh [backend] [options]` prints the wall time and the GPU time per kernel for the Berlin pair.

## Notes from M5, second pass (Metal)

- **Where the limit was.** A kernel that only multiplies and adds in registers reaches 2.9 T (10^12)
  multiply-adds per second in float and 6.2 T in half on this GPU. `k_linear` ran at 2.3 T (2.9 T with
  half accumulators), and the shapes that should have helped (more accumulators, 8 values per load, weights and
  inputs kept in cache) did not: a thread slows down sharply past about 64 float registers (128 half), and
  a load costs as much as 4 to 6 scalar multiply-adds. That put the best plain kernel near 3.6 T.
- **PyTorch was above that limit**: 13.5 T for a half matrix product, 3.6 T in float. So it was not using
  plain shader arithmetic.
- **Metal Performance Primitives** (Metal 4, macOS 26; headers in the system, no library to link) is what
  reaches that hardware from a kernel of our own: `matmul2d` computes one tile of a product cooperatively
  on the threads of a SIMD group. It plays the part the tensor-core instructions play on CUDA. The shader
  source is still compiled at run time; when it does not compile with `JMT_MPP` (older system) the backend
  falls back to the kernels of the first pass (`JMT_METAL_MPP=0` forces that).
  Measured, 4096 x 1024 x 1024: 8 to 16 T in half (dispatches this short time unevenly), 3.8 T in float.
  `simdgroup_matrix` (8 x 8 products, available since macOS 13) gives 3.5 T in both float and half: a gain
  in float only, not used.
- **Products** (`k_mm`): a tile of 64 x 64 outputs per threadgroup of 4 SIMD groups, weights as stored
  (no rearranged copy). The tile is a float "cooperative tensor" held in registers, so bias, GELU or ReLU
  and the residual are applied before it is written: the separate passes for them are gone. Tensors are
  clipped at their edges by the primitive, so any size works.
- **Attention** (`k_attn`) no longer has a weights buffer (512 MB in float at 512 px). One SIMD group
  takes 32 queries of one head and walks the source tokens 32 at a time: logits, row maxima, weights,
  row sums and the product with the values all stay in registers; when a row's maximum grows, what it has
  accumulated is rescaled. 9.4 ms per attention before (FP16), 1.4 ms now; 14 ms to 5 ms in FP32.
  Computing the logits twice (first for the maxima) instead of rescaling cost 20 ms more per pair in FP16
  and 87 ms in FP32.
- **Half overflows in attention.** Before the division by the row sum, an output is a sum over 4096
  tokens. With flat weights and values near 16 it exceeds 65504: NaN at 512 px, nothing wrong at 384. Row
  sums and outputs are accumulated in float in both modes; that also cut the error of an attention output
  from 2.5e-3 to 1.7e-4 on a test with values up to 0.4.
- **FP16 accuracy is bound by half storage**, not by where the products round: float tiles in `k_mm`
  instead of half ones changed neither the agreement nor the time.
- **Layer norm** (`k_layer_norm_sg`): one SIMD group per token, each thread holding 1/32 of the channels
  in registers, mean and variance reduced across the group, activation applied in the same pass.
- **Read-out on the device** (`k_readout`): one SIMD group per query. The 1.5-entmax threshold is the
  root of a convex decreasing function; Newton's method from the left end climbs to it without passing
  it, which needs no sort. The reference's cut at the 512 largest logits is reproduced: when more than
  512 cells can have weight, the 512th largest is found by bisection. Against the golden dumps it is as
  close as the host version: coarse positions within 2e-7 (median), masses within 2e-5.
- **One round trip per pair.** Nothing is read back before the last operation (unless early stop is on):
  `FlexGlue::match_queue` issues a pair, `match_read` reads it. Per pair: 2 ms to encode and 4 ms on the
  host (offsets and mutual nearest neighbours); the rest is GPU work.
- **Batches are a pipeline, not a stack.** One pair already fills the GPU, so `jmt_match_batch` keeps 4
  pairs queued and finishes pair i on the host while the device works on the next ones. Gain: 1 to 7 %.
  Stacking pairs into larger tensors would gain nothing here.
- **Preprocessing on the device** from the 8-bit image (`k_preprocess`). The source rows and columns of
  the resize are computed on the host: computed in the shader, some differed in the last bit and the
  features moved by 4e-4 at 384 px.
- **Allocations.** Device buffers are pooled and a pair of a size seen before creates none (tested). On
  the host the results reuse their arrays and the rotary tables are computed once per grid size; tensor
  handles and parameter names are still allocated per operation.
- **The profile is no longer a measure.** With `JMT_METAL_PROFILE=1` every dispatch is its own command
  buffer; now that most take under a millisecond the GPU's clock drops between them and the totals vary
  2x from run to run. Wall time over several runs is what the tables use.
- Where the time is now (FP16, profile, one of the faster runs, both images and the pair): products 57 ms,
  attention 49, layer norm 11, depthwise convolution 7, channel copies 4, read-out 3.
- Peak memory: 2.8 GB in FP16, 3.2 GB in FP32 (first pass: 3.1 GB in FP16).
- Not done: slices and concatenations as views (4 ms), the depthwise convolution, float logits in FP16
  attention (a float tile as the left operand of the second product gave wrong results; not investigated).

## Notes from M5 (CUDA)

Same laptop as M4 (RTX 2070 Max-Q, CUDA 13.4). Starting point: the Metal M5 branch, whose new `Ops` hooks
have default implementations; `cuda_ops.cu` compiled unchanged and its 8 tests passed.

- **The hooks.** `linear_fused`: bias, activation and residual applied to the tile before it is stored, in
  all three product kernels (`k_gemm`, `k_gemm_tc`, `k_linear_small`). `layer_norm_act`: the activation in
  the same pass. `readout_device` / `sample_at`: `k_readout`, a port of the Metal kernel (one warp per
  query). `preprocess_u8`: `k_preprocess`, the resize taps from the host and every step rounded as written
  (`__fmul_rn` and friends, no fused multiply-adds), so the input is the host's to the last bit.
  `finish` waits for the device; `flush` has nothing to do (a kernel starts when it is launched).
  `device_allocations` counts `cudaMalloc` calls: the batch test creates 728 buffers, the second batch none.
- **Queuing a pair never waits.** `cudaMemcpy` from pageable memory waits for the device to go idle first,
  and every rotary call uploaded its cos and sin tables that way (36 per pair), so the host stopped in the
  middle of every pair. Uploads are now `cudaMemcpyAsync` on the stream (the driver stages pageable memory
  before it returns), and the tables are uploaded once per address, as `Ops::rotary` allows.
- **Attention without the logits buffer** (`k_flash`, FP16 mode). A warp takes 16 queries of one head and
  walks the source tokens 64 at a time; logits, row maxima, weights and the product with the values stay
  in registers, and a row is rescaled when its maximum grows, as in Metal's `k_attn`. The products use
  `mma.sync.m16n8k8` (sm_75) through inline PTX rather than `nvcuda::wmma`: its register layout is
  documented, and the accumulator of the logits is laid out like the left operand of the second product,
  so the weights go from one to the other without leaving the registers (with `wmma` the layout is
  opaque and the tile would go through shared memory). Keys and values are read with `ldmatrix` (its
  `.trans` form gives the value operand), from tiles whose rows are padded to 72 values so that the 8 rows
  of one read fall on distinct banks.
  Warps per block (16 queries each), per pair at 512 px / 9 layers: 2: 241 ms, 4: 171, 8: 152, 16: 148.
  Asking for 2 to 4 blocks per SM (`__launch_bounds__`, fewer registers) made it slower: 4 warps x 3
  blocks 170 ms, x 4 185 ms. 16 warps needs 128 registers with a few spilled and is still the fastest:
  a larger block shares the key and value tiles among more queries. Attention is now 1.5 to 2.1 ms per
  call (8 to 12 T multiply-adds per second; the clock moves with the power cap), against 5.5 ms for the
  two passes over the logits (3.1 + 2.5 ms, same profile run).
- **FP32 attention on split halves.** The same kernel with every operand split into two halves,
  x = hi + lo (hi = half(x), lo = half(x - hi)), and three products per tile (hi hi + hi lo + lo hi,
  float accumulation): about 22 bits per product instead of 24. Pair at 512 px / 9 layers: 634 -> 417 ms.
  - First version: the same matches as the original on all 12 runs, but points moved by up to 2e-3 px
    instead of 2.5e-4, and on the LaMAR pair one confident query moved 4.1 px (confidence 0.018).
  - Not the logits: splitting queries and keys in three halves (6 products, about 33 bits) changed
    nothing (last-layer tokens 3.9e-3 from the golden dump either way; 3.2e-4 through the float buffer).
  - The output accumulator: each output summed its 4096 source tokens inside the tensor-core instruction,
    whose float accumulation on this GPU does not round as float additions do. With a fresh accumulator
    per step of 32 tokens, added to the outputs with float additions, tokens are within 4.1e-4 of the
    golden dump and the 12 runs within 4.1e-4 px (the LaMAR query: 4.1e-4 px). Cost: 4 % (409 -> 417 ms).
    It is the default; `JMT_CUDA_SPLIT=0` goes back to the float logits buffer.
  - The same fresh accumulator in FP16 mode changed nothing measurable (its error is set by half
    storage), so FP16 keeps one accumulator.
  - Tried before this and dropped: plain float kernels (no tensor cores) without the logits buffer. One
    query per thread (64 query values and 64 outputs in registers, keys and values read as broadcasts):
    661 ms per pair, 1.4 T multiply-adds per second, slower than the two passes. Four lanes sharing four
    queries (a quarter of the shared-memory reads, two shuffles per logit): 697 ms. Both run at the
    1.4 to 1.6 T the two passes over the logits reach at these shapes.
- **Layer norm** with 16-byte loads (`k_layer_norm_vec`): a group of g channels is held by g / 8 lanes (FP16)
  or g / 4 (FP32), several groups per warp when g is small. The per-head norms (g = 64) had run at 61 GB/s
  with a warp per group and 2-byte loads: 7.4 -> 1.5 ms per pair for their 54 calls; all layer norms of a
  pair and its two images 15 -> 6 ms.
- **Depthwise 7x7 convolution** (`k_dwconv7_vec`): a thread takes 16 bytes of channels and 4 neighbouring
  outputs of a row, so every input it loads serves the taps of all 4; the sum runs in the same order as
  before (bitwise the same in FP32). 15 -> 2.8 ms for the two images.
- **Read-out.** The Metal kernel scans the row 30 times for the bisection and up to 40 times for Newton's
  method. Here many rows have thousands of cells above -1 (the cells that can have weight), so the
  bisection ran on most rows: 5 ms per pair. Gathering those cells into shared memory did not help (a
  list of 1024 overflowed; 4096 allowed only 4 warps per SM: 3.3 ms). What did: three passes of a 256-bin
  histogram, each within the bin the previous one chose, place the 512th largest y in a cell of width
  2^-24; the cells that share it stand in as copies of their largest value, and the y that count (at most
  512) go to shared memory for Newton's method. 1.6 ms per pair. Against the golden dumps: coarse
  positions within 3.3e-7 (median), masses within 1e-5, as close as the host version; in FP16, on the same
  device logits, the host and the device read-outs move as many unconfident queries (41 and 42 of 512).
- **The profile** (`JMT_CUDA_PROFILE=1`) now brackets every launch with two events instead of waiting
  after it. Waiting let the GPU's clock drop between kernels and inflated the totals several times (3.8 s
  of attention reported for a pair that took 0.63 s). Even with events, totals vary by up to 30 % from run to run
  on this laptop (power cap), so the tables use the best wall time of several runs.
- **Results** (Berlin pair, 2048 keypoints per image, best of 8 runs over two sweeps; table in M5 above).
  FP16 at 512 px / 9 layers: 349 ms after M4, 274 with the hooks (fused products, device read-out,
  uploads that do not wait, warp layer norm), 148 with `k_flash`, 143 with the vectorized layer norm,
  depthwise convolution and the histogram read-out. FP32: 726 -> 634 ms with the same steps, 417 with
  attention on split halves (the two passes over 512 MB of float logits had been 53 % of its kernel
  time).
- **Batches.** `jmt_match_batch` gains nothing here (FP16 at 512 px: 143 ms alone, 145 ms per pair in a
  batch of 8): one pair fills the GPU, and the host's part (4 ms) already overlaps.
- **Where the time is now** (FP16, profile of both images and the pair, 195 ms of kernel time in that run;
  another run gave 139): products 89 ms, attention 78, copies for slices and concatenations 8, layer norm 6,
  depthwise convolution 3, read-out 2. About a quarter of the products' time is the two images: the
  products of stage 3 (32 x 32 tokens) fill only 32 blocks for 36 SMs and run at 6 T.
- Not done: products with smaller tiles for the small grids of describe, slices and concatenations as
  views (the value slice could go to `k_flash` with its stride). The products of FP32 mode (about 300 ms of
  kernel time per pair and its two images) could use the same split halves on the tensor cores.

## Notes from M6 (WebGPU)

The first pass first; the second pass, at the end, replaced its products and attention.

- **Structure.** `kernels.wgsl` holds one section per kernel; the host compiles each instance as its own
  module (prelude + section + entry point) on first use. Sections are either per thread (`fn run`, 1-D grid
  of 256-thread groups) or per workgroup (`fn group`, `//@group N`). WGSL has no templates: an instance name
  carries its parameters (`k_mm_r4c4_v4_f32`: 4 x 4 per thread, vector loads, float output), replaced in
  the source (`$R`, `$C`, `$O`) or selecting lines marked `//?v4` / `//?s`. The context (one compute pass
  per batch, a 256-byte uniform slot per dispatch for its scalars, a bind group per dispatch, error scopes
  checked at the next read, writes that submit the open batch first) is just-keypoints'.
- **Workgroup memory: never write different components of one vector from different threads.** The first
  product stored transposed tiles as `xs[i][lane % 4] = v`; on Dawn's Metal path that compiles to a
  read-modify-write of the whole vector and loses writes (outputs off by up to the whole value, at
  random). Every vector in workgroup memory is now written whole by one thread. `ops_test` found it at once.
- **Matrix product** (`k_mm`): 64 x 64 tile, 4 x 4 outputs per thread interleaved (thread (ty, tx) has
  rows ty + 16 r and columns tx + 16 c), operands through workgroup memory as rows of 16 inputs stored as
  4 vectors plus one of padding, dot products along the input. Measured on 4096 tokens (ops_bench, new):
  scalar tiles 1.7-2.0 T multiply-adds per second, vector tiles 2.2-2.3 T; 8 x 4, 4 x 8 and 8 x 8 per
  thread are 1.8 to 7 times slower (registers); fetching the next tile into registers while multiplying changed
  nothing. Metal's plain kernels reached 2.3-2.9 T on this GPU, its matrix primitives 3.7 T in FP32 and
  10-15 T in FP16. Convolutions are patches + this product; the read-out's logits are this product with
  the search tokens as the weight matrix and float output.
- **Attention** (`k_attn`), single pass, no weights buffer: a workgroup takes 64 queries of one head and
  walks the keys 64 at a time; per tile the logits as a 64 x 64 product (4 x 4 per thread), the row
  maximum from the 16 threads that share a row, rescaling when it grows, and the value product 16 tokens at
  a time. Each thread keeps its part of the row sums; they are added once at the end. One call (P = S =
  4096, 8 heads): one query per thread with its 64 values and 64 outputs in registers, 68 ms (0.25 T: the
  registers spill); blocked, scalar workgroup reads, 11.4 ms; vector reads with dynamic component indices,
  14 ms; the same with static indices (`l[r] += vec4(dot(a, b0), ..)`), 8.3 ms (2.1 T). Metal: 5.0 ms in
  FP32, 1.3 ms in FP16.
- **Layer norm**: 16, 32 or 64 threads per group by group size, several groups per workgroup; the per-head
  norms (64 channels) 0.31 -> 0.11 ms per call.
- **Read-out on the device** as on Metal (bisection, then Newton), one 128-thread workgroup per query with
  the row in registers (up to 8192 cells, input up to 720 px; larger falls back to the host). Reductions
  in workgroup memory; the loop decisions use `workgroupUniformLoad`, which WGSL's uniformity rules require
  for a branch around barriers.
- **Arithmetic**: WGSL gives no control over contraction or division rounding, so FP32 differs from the CPU
  backend in the last bits (single operations within 7e-6 of values of order 1). `preprocess_u8` divides with a
  correctly rounded division (`div_rn`, from just-keypoints) and matches the host within 7e-7.
- **Where the time is** (FP16 profile, a round trip per dispatch, so shares only): products 43 %,
  attention 39 %, layer norm 9 %, copies for slices and concatenations 3 %, depthwise convolution 2.5 %.
- **The browser module**: `-sJSPI` (`jmtw_create`, `jmtw_describe`, `jmtw_finish`, `jmtw_match` suspend at
  the GPU waits), `-fwasm-exceptions`, `MODULARIZE`, `EXPORT_ES6`, memory up to 4 GB. The weights are copied
  into the module (567 MB, or 284 MB for the FP16 file, widened to 567 MB of float32 by the loader). Times
  in Chrome equal the native ones: the work is on the GPU, and the host part of a pair is small.
- **Test page data** (`tools/web_testdata.py`): images decoded once (OpenCV, as `compare.py`), written as
  PNG for the native CLI and as raw RGBA for the page, so both see the same pixels; the reference is the
  native CPU backend.
- **Not done**: subgroup matrices (`chromium_experimental_subgroup_matrix`, the only way to the matrix
  hardware from WGSL; behind a flag in Chrome), f16 arithmetic in FP16 mode, slices as views, timestamp
  queries for GPU time per kernel. The other GPU workload on the machine during part of this work (another
  project's benchmark) made single timings vary up to 5x; the tables are from a quiet GPU.

### Second pass (a review by another model, measured and adopted)

- **Half arithmetic in FP16 mode, 8 halves per load.** `k_mmh`: f16 dot products of 8 inputs, accumulated
  in f32, x and W bound as `array<vec4<u32>>` and split with `bitcast<vec4<f16>>`; 2.2 -> 3.7 T
  multiply-adds per second (f16 products alone: 3.1 T; 8 instead of 4 halves per load: 3.0 -> 3.7 T).
  The read-out's logits (float output) keep f32 arithmetic: in f16 the read-out mass error in `ops_test`
  went from 2e-6 to 3e-3. All-half accumulators were only 5 % faster and changed matches on Metal (M5).
- **Workgroup tiles bought nothing on this GPU.** Products and attention reading their operands straight
  from the buffers, without staging or barriers, run as fast or faster (`k_mmd`, FP32: +7 %). The tiled
  `k_mm` stays only for cin not a multiple of 4. May not hold on a discrete GPU (not measured).
- **Attention.** `k_attnh` (FP16): q and k read straight from the buffers 8 channels per load, the values
  transposed once per call (`k_tokens_t`, 8 tokens of one channel per load), only the weights through
  workgroup memory; barriers per tile 16 -> 3; sums of 8 terms in f16, everything across them f32.
  8.1 -> 5.1 ms per call; ops_test error 3.9-5.3e-3 (Metal FP16: 3.2-5.3e-3; the f32 kernel: 4.9e-4).
  `k_attnd` (FP32): the same structure in f32, 8.3 -> 7.9 ms, 16 KB of workgroup memory (maxima and sums
  share the weights' array). In Chrome `k_attnh` gives results less close to the reference than natively
  (M6 entry).
- **Depthwise convolution** with vector bindings (it did 4 scalar loads per vector): describe 31 -> 30 ms.
- **Larger tiles per thread, revisited.** Part of the earlier "1.8 to 7 times slower" was Tint's code for
  loops over local arrays: unrolled by hand, 8 x 4 per thread reaches 2.1 T in f32 and 3.3 T in f16, still
  below 4 x 4 (2.3 and 3.7 T). Loops over local arrays of more than about 16 elements are a trap.
- **Rejected:** f16 sums over 16 terms in attention (slower and less accurate), half `exp`, fewer barriers
  (within 1-2 %), other workgroup shapes, FP32 on split halves (no faster at a 6.2 T half ceiling,
  estimate), subgroups (shared loads cost the same as buffer loads; not prototyped).
- **Profile caveat.** `JMT_WEBGPU_PROFILE` adds a round trip of about 0.2 ms per dispatch, so small kernels
  look larger: timed directly, attention was about 60 % of a pair's match time before this pass, not 39 %.
- **Left:** layer norm, copies, rotary, add and unary still load and store one scalar at a time (layer norm
  about 13 ms per pair, estimate 5-8 ms to gain); slices as views and norms writing into the concat buffer
  (estimate 5-9 ms); convolutions reading x directly instead of through the 75 MB of patches. By the
  measured register-only ceilings (6.2 T in half, 2.9 T in float) FP16 attention could reach about 2.8 ms
  per call; it is at 5.1. The remote RTX 2070 could not be used for this pass (permission), so there are
  no Dawn/Vulkan numbers.

## Notes from C2 (covisibility levels 1 and 2)

- **Level 2 without the 512-channel tokens.** `desc0` (64 x 256) equals `heads.0.to_search` after
  `input_proj` to 4.8e-7, and bilinear sampling commutes with a linear layer (the weights sum to 1), so
  the search map is `l2_normalize(desc0(features))`, computed once per image and kept with its features
  (1 MB in FP32), and `input_proj` runs only on the sampled points (the query and its best cell). A pair
  is four layer-0 heads of 256 queries: about 0.3 G multiply-adds (estimate), instead of 1.3 G with the
  token maps of both images. `FlexGlue::head_tail` is the part both head forms share.
- **Batching.** The way back starts from the way there, whose offsets are added on the host, so a pair
  needs two round trips. `grid_scores` queues 8 pairs (16 heads) before reading any of them.
- **Speed**, the 53 images of `covis.py`, k = 5 and rerank (156 pairs), per reranked pair including each
  image's first search map: Metal 2.5 ms in FP32, 2.0 ms in FP16; WebGPU FP16 3.9 ms; CPU 25 ms (PyTorch
  MPS in `covis.py`: 6.7 ms). Level 1 for all 1378 pairs: 11 ms on the CPU, 18-43 ms on a GPU (reading
  the 53 feature maps back once).
- **Level 2 and scale.** The Berlin pair (a close-up of the same facade) scores 0.11 at level 2 and 0.80 at
  level 1; the LaMAR look-alikes (not the same place) score 0.39 and 0.985. A grid of 16 x 16 points mostly
  lands outside a close-up, so level 2 under-rates scale changes and does not reject look-alikes. That is
  why `min_score` is off by default; level 3 decides. Whether level 2 re-ranks partial overlap inside one
  capture well is C1's question.
- **Not done:** a faster CPU level 2 (10 times Metal; where its time goes is not measured), the shortlist in the WebAssembly module, saving the kept per-image vectors with
  the features.

## Notes from C3 (level 3, early stop)

- **Nothing to restructure.** The check is the covisibility head after layer k, read back on the host; a
  pair that passes continues from the same tokens, so the shortlist -> verdict -> matches path is one call
  per pair. The read-back is one round trip per layer from 2 to n_layers - 1. Doing these in step across
  the pairs of a batch was considered and not done: on pairs that pass, the checks cost
  2-5 ms per pair (1 % in FP32, 2 % in FP16) on Metal and WebGPU, 14 / 2 ms on CUDA (3 / 1 %), in a batch
  of 8 as for single pairs.
- **Speed**, per pair, 512 px, 9 layers, 2048 keypoints per image, best of 8 (M5 Pro; CUDA: RTX 2070
  Max-Q). Passing: the Berlin pair; stopped after 8: berlin1 - lamar_cab_1446 (different places, 0.63 after
  layer 8) at decay 0.2; after 4: the LaMAR look-alikes at decay 0.5; after 2: the same at decay 0.2.

  | Backend | Off | Passing, checked | Stopped after 8 | after 4 | after 2 |
  |---|---|---|---|---|---|
  | Metal FP32 | 311 ms | 313 ms | 282 ms | 147 ms | 77 ms |
  | Metal FP16 | 102 ms | 104 ms | 96 ms | 53 ms | 30 ms |
  | WebGPU FP32 | 471 ms | 475 ms | 424 ms | 217 ms | 113 ms |
  | WebGPU FP16 | 308 ms | 313 ms | 280 ms | 145 ms | 77 ms |
  | CUDA FP32 | 413 ms | 427 ms | 389 ms | 201 ms | 106 ms |
  | CUDA FP16 | 144 ms | 146 ms | 131 ms | 70 ms | 40 ms |

  A stopped pair still runs the head at its stop layer for every keypoint (as the original does: its
  predicted points and confidences are returned, only the matches are dropped); that is most of what is
  left after layer 2.
- **Which decay.** From the per-layer scores of the 53 images of `covis.py` (all 1378 pairs; 400 in the
  same scene), with min_covisibility 0.8, the rule replayed on the reference scores:

  | Decay | Pairs of other scenes stopped | Mean stop layer | Transformer work left | Pairs lost |
  |---|---|---|---|---|
  | 0 | 99.8 % | 2.0 | 45 % | 1 same-scene, 18 others that pass at 9 |
  | 0.1 | 95.4 % | 3.7 | 60 % | 1 other |
  | 0.2 | 84.9 % | 5.6 | 77 % | none |
  | 0.5 | 41.9 % | 7.9 | 96 % | none |

  "Lost": pairs that would score >= 0.8 after 9 layers but stop early. Unrelated scenes score 0.3-0.7 at
  full depth, and the bar after layer k is 0.8 (1 - decay)^(9 - k): at 0.5 it is below 0.03 up to layer 4,
  so only look-alikes (near 0) stop early. The original's README uses 0.2 in its example, its ONNX notes 0.5
  (ETH3D, 42 % of pairs skip most of the work). 0.2 is what `jmt.h` points to; the default stays off.
- **After a shortlist there is little left to stop.** On the k = 5 shortlist of the same images (156 pairs,
  20 of other scenes) decay 0.2 stops 16 pairs and saves 6 % of the transformer; k = 10 (329 pairs, 59 of
  other scenes): 50 pairs, 8 %. Early stop pays for exhaustive matching (small captures, where level 2 or
  nothing builds the list) and for look-alikes the shortlist lets through.
- **CUDA FP32, LaMAR pair stopped after layer 4:** one confidence differs from the original by 1.4e-3
  (elsewhere within 2e-5; points within 2.2e-4 px). Likely a query at the edge of the read-out's support;
  the pair has no matches either way. Not followed up.
- **Not done:** the browser (the rule is host code shared with native WebGPU; Chrome FP16's attention
  differs slightly, so a pair near a bar could stop one layer apart), a capture where the shortlist and
  early stop are measured together on real overlap (C1's capture).
