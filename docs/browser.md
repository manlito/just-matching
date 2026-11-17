# In the browser

The browser build is the same C++ library compiled to WebAssembly, with the WebGPU backend running the network
on the visitor's GPU. A small JavaScript wrapper, `jmt.js`, loads it. Keypoints come from just-keypoints' own
browser module.

```js
import * as jmt from './jmt.js';
import * as jkp from './jkp/jkp.js';

// weightsFile: flexglue.pth from FlexGlue's release, picked or dropped by the user
const matcher = await jmt.create({ weights: await weightsFile.arrayBuffer(), precision: 'fp16' });
const extractor = await jkp.create({ weightsUrl: 'jkp/raco.jkpw', topK: 2048, skipRanker: true, skipCovariance: true });

const [imA, imB] = await Promise.all([fileA, fileB].map(f => createImageBitmap(f)));
const kA = await extractor.extract(imA), kB = await extractor.extract(imB);
const a = await matcher.describe(imA), b = await matcher.describe(imB);   // once per image
const m = await matcher.match(a, kA.xy, b, kB.xy);
// m.n matches: m.matches Int32Array [n][2] (keypoint of a, keypoint of b), m.scores [n], m.covisibility
a.destroy(); b.destroy(); matcher.destroy();
```

Serve `jmt.js`, `jmt_wasm.js` and `jmt_wasm.wasm` as static files; nothing runs on a server. The weights come from
the user: FlexGlue's backbone is under Meta's DINOv3 License ([Licences](licences.md)), so a page should not serve
them. <a href="demo/index.html">The live demo</a> asks for `flexglue.pth` with a button or a drop, reads it in the
browser, and keeps a copy in the origin's private file system for the next visit only when the visitor ticks
"Keep in this browser".

The module converts the checkpoint while it creates the matcher. Meanwhile memory holds the file and the converted
weights: about 1.2 GB for the 567 MB checkpoint.

## Options of `create`

| Option | Default | Meaning |
|---|---|---|
| `weights` or `weightsUrl` | | the bytes of `flexglue.pth` or of a `.jmtw` file (`ArrayBuffer`, `Uint8Array`), or a URL of either or of a parts manifest (below) |
| `backend` | `'webgpu'` | `'webgpu'`, or `'cpu'` (the CPU backend in WebAssembly: one thread, very slow) |
| `precision` | `'fp32'` | `'fp16'` needs the adapter's `shader-f16` feature |
| `inputSize` | `512` | side of the network input, a multiple of 32 |

## Methods of the matcher

| Method | Returns |
|---|---|
| `describe(image)` | the image's features; free them with `destroy()`. `image`: an `ImageBitmap`, `ImageData`, image element, video frame or canvas |
| `match(a, xyA, b, xyB, options)` | `{ n, matches, scores, aInB, bInA, confA, confB, covisibility, stopLayer }`, as `jmt_matches`. `xyA`, `xyB`: `Float32Array` `[n][2]` |
| `covisibility(a, b, level)` | the score of one pair at level 1 or 2 |
| `level1(features)` | level 1 of every pair of n images: `Float32Array` `[n][n]` |
| `shortlist(features, options)` | `{ n, pairs: Int32Array [n][2], scores }`, best first |
| `finish()` | waits for the GPU |
| `destroy()` | frees the matcher |

Options of `match`: `layers` (9), `filterThreshold` (0.1), `maxDistancePx` (3), `minCovisibility` (0),
`earlyStopDecay` (-1, off). Options of `shortlist`: `k` (10), `neighbours` (0), `rerank` (false), `minScore` (0).
Their meaning is the C API's: [C API](api.md), [Which pairs to match](covisibility.md).

Results are copies. Calls on one matcher must not overlap.

## Weights in parts

For a copy of the weights you host yourself, in your own deployment and under the licences above: some static
hosts cap the size of a file (Cloudflare Pages: 25 MiB). `jmt.fetchWeights(url, { onProgress })`,
which `create` uses for `weightsUrl`, loads a plain `.jmtw` file or, when the URL ends in `.json`, a manifest of
parts, fetched four at a time and joined in memory:

```json
{ "format": "jmtw-parts", "size": 283537672,
  "parts": [ { "url": "flexglue-fp16.part00", "size": 25165824 },
             { "url": "flexglue-fp16.part01", "size": 25165824 } ] }
```

Part URLs are relative to the manifest (two of the twelve parts are shown). `onProgress(loaded, total)` reports bytes as they arrive, for a progress
bar; fetch the bytes this way and pass them to `create` as `weights`.

## Browser support

The module waits for the GPU through JavaScript promise integration (JSPI), so that the C++ code can wait for
results without blocking the page. Chrome and Edge 137 and later have both WebGPU and JSPI. Safari and Firefox have
not been tried.

Memory: the weights are copied into the module once, and the GPU holds about 1 GB.

In Chrome, FP32 gives the native CPU backend's matches on all the test pairs (points within 3.1e-4 px). FP16
differs on a few matches per pair, up to 5 of 70 on the hardest one: Chrome compiles the half-precision attention
a little differently from native Dawn.
