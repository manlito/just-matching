// just-matching in the browser: a thin wrapper over the WebAssembly module (jmt_wasm.js + .wasm).
//
//   import * as jmt from './jmt.js';
//   const matcher = await jmt.create({ weightsUrl: 'flexglue-fp16.jmtw', precision: 'fp16' });
//   const a = await matcher.describe(imageA), b = await matcher.describe(imageB);
//   const m = await matcher.match(a, xyA, b, xyB);   // xy: Float32Array [n][2] from just-keypoints
//   const s = await matcher.shortlist([a, b, c, ...], { k: 5 });  // the pairs worth matching, best first
//
// Keypoints are pixels of the original image, (0, 0) = centre of the top-left pixel. Results are copies:
// matches Int32Array [n][2] (index in a, index in b), scores [n], aInB [nA][2] and bInA [nB][2] (the point
// predicted in the other image for every keypoint), confA [nA], confB [nB], covisibility, stopLayer.
// Covisibility: covisibility(a, b, level) scores one pair at level 1 (cosine of the mean feature vectors) or 2
// (a 16 x 16 grid through the layer-0 head and back); level1([features]) is the level-1 matrix of n images
// (Float32Array [n][n]); shortlist([features], options) returns { n, pairs: Int32Array [n][2] (first <
// second), scores } best first. Level 3 is match() with minCovisibility and earlyStopDecay.
// The module waits for the GPU through JSPI (Chrome, Edge).
import createJmt from './jmt_wasm.js';

const BACKENDS = { cpu: 0, cuda: 1, metal: 2, webgpu: 3 };
const PRECISIONS = { fp32: 0, fp16: 1 };

let modulePromise = null;

// The module, loaded once per page.
export function module() {
  if (!modulePromise) modulePromise = createJmt();
  return modulePromise;
}

export function webgpuSupported() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

// RGBA pixels of an ImageBitmap, ImageData, image element, video frame or canvas.
export function rgba(image) {
  if (typeof ImageData !== 'undefined' && image instanceof ImageData) return image;
  if (image && image.data instanceof Uint8Array && image.width && image.height) return image;  // { width, height, data }
  const w = image.width ?? image.videoWidth, h = image.height ?? image.videoHeight;
  const canvas = new OffscreenCanvas(w, h);
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.drawImage(image, 0, 0);
  return g.getImageData(0, 0, w, h);
}

// The bytes of a weights file, from its URL or from the URL of a parts manifest (a .json file): hosts such as
// Cloudflare Pages cap files at 25 MiB, so the site serves each .jmtw as parts listed by
//   { "format": "jmtw-parts", "size": total bytes, "parts": [{ "url": relative to the manifest, "size": bytes }] }
// onProgress(loaded, total) is called as bytes arrive (total is 0 when the server does not say).
export async function fetchWeights(url, { onProgress } = {}) {
  const report = onProgress ?? (() => {});
  if (!/\.json(\?|#|$)/.test(url)) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`cannot load ${url}: ${r.status}`);
    const total = Number(r.headers.get('content-length')) || 0;
    if (!r.body || !total) {
      const out = new Uint8Array(await r.arrayBuffer());
      report(out.length, out.length);
      return out;
    }
    const out = new Uint8Array(total);
    let got = 0;
    for (const reader = r.body.getReader(); ;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > total) throw new Error(`${url}: more bytes than its content-length`);
      out.set(value, got);
      got += value.length;
      report(got, total);
    }
    if (got !== total) throw new Error(`${url}: ${got} of ${total} bytes`);
    return out;
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`cannot load ${url}: ${r.status}`);
  const manifest = await r.json();
  if (manifest.format !== 'jmtw-parts' || !Array.isArray(manifest.parts)) throw new Error(`${url}: not a parts manifest`);
  const total = manifest.size;
  if (manifest.parts.reduce((s, p) => s + p.size, 0) !== total) throw new Error(`${url}: part sizes do not add up`);
  const base = new URL(url, location.href);
  const out = new Uint8Array(total);
  const offsets = [];
  manifest.parts.reduce((o, p) => (offsets.push(o), o + p.size), 0);
  let got = 0, next = 0;
  // A few parts at a time, each written to its place as it arrives.
  const worker = async () => {
    while (next < manifest.parts.length) {
      const i = next++;
      const part = manifest.parts[i], partUrl = new URL(part.url, base).href;
      const pr = await fetch(partUrl);
      if (!pr.ok) throw new Error(`cannot load ${partUrl}: ${pr.status}`);
      let at = offsets[i];
      const end = at + part.size;
      for (const reader = pr.body.getReader(); ;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (at + value.length > end) throw new Error(`${partUrl}: larger than ${part.size} bytes`);
        out.set(value, at);
        at += value.length;
        got += value.length;
        report(got, total);
      }
      if (at !== end) throw new Error(`${partUrl}: ${at - offsets[i]} of ${part.size} bytes`);
    }
  };
  await Promise.all(Array.from({ length: Math.min(4, manifest.parts.length) }, worker));
  return out;
}

// options: weightsUrl (a .jmtw file or a parts manifest, see fetchWeights) or weights (ArrayBuffer /
// Uint8Array of a .jmtw file, float32 or float16); backend 'webgpu' (default) or 'cpu'; precision 'fp32'
// (default) or 'fp16' (needs shader-f16); inputSize (512).
export async function create(options = {}) {
  const M = await module();
  const backend = options.backend ?? 'webgpu';
  const precision = options.precision ?? 'fp32';
  if (!(backend in BACKENDS)) throw new Error(`unknown backend '${backend}'`);
  if (!(precision in PRECISIONS)) throw new Error(`unknown precision '${precision}'`);
  if (!M._jmtw_backend_available(BACKENDS[backend])) throw new Error(`backend '${backend}' is not in this build`);
  let weights = options.weights;
  if (!weights) {
    if (!options.weightsUrl) throw new Error('give weightsUrl or weights');
    weights = await fetchWeights(options.weightsUrl);
  }
  const bytes = weights instanceof Uint8Array ? weights : new Uint8Array(weights);
  const wp = M._malloc(bytes.length);
  if (!wp) throw new Error('out of memory for the weights');
  M.HEAPU8.set(bytes, wp);
  let handle;
  try {
    handle = await M._jmtw_create(BACKENDS[backend], PRECISIONS[precision], wp, bytes.length, options.inputSize ?? 512);
  } finally {
    M._free(wp);  // jmt_create copies the weights
  }
  if (!handle) throw new Error(M.UTF8ToString(M._jmtw_error()));
  return new Matcher(M, handle, backend, precision);
}

// Dense features of one image, made by Matcher.describe; free them with destroy() when done.
class Features {
  constructor(matcher, ptr, width, height) {
    this.matcher = matcher;
    this.ptr = ptr;
    this.width = width;
    this.height = height;
  }
  destroy() {
    if (this.ptr) this.matcher.M._jmtw_features_free(this.ptr);
    this.ptr = 0;
  }
}

class Matcher {
  constructor(M, handle, backend, precision) {
    this.M = M;
    this.handle = handle;
    this.backend = backend;
    this.precision = precision;
    this.buffers = {};  // reused module allocations, by role
  }

  // Module memory of at least `bytes` for `role`, reused between calls.
  scratch(role, bytes) {
    const b = this.buffers[role];
    if (b && b.bytes >= bytes) return b.ptr;
    if (b) this.M._free(b.ptr);
    const ptr = this.M._malloc(Math.max(bytes, 16));
    if (!ptr) throw new Error('out of memory');
    this.buffers[role] = { ptr, bytes };
    return ptr;
  }

  // Calls on one matcher must not overlap.
  async describe(image) {
    const M = this.M;
    if (!this.handle) throw new Error('matcher destroyed');
    const data = rgba(image);
    const bytes = data.width * data.height * 4;
    const px = this.scratch('pixels', bytes);
    M.HEAPU8.set(data.data, px);
    const f = await M._jmtw_describe(this.handle, px, data.width, data.height);
    if (!f) throw new Error(M.UTF8ToString(M._jmtw_error()));
    return new Features(this, f, data.width, data.height);
  }

  // Waits for the GPU (describe may return before its work is done).
  async finish() {
    if ((await this.M._jmtw_finish(this.handle)) !== 0) throw new Error('webgpu: ' + this.M.UTF8ToString(this.M._jmtw_error()));
  }

  // options: layers (9), filterThreshold (0.1), maxDistancePx (3), minCovisibility (0), earlyStopDecay (-1).
  async match(a, xyA, b, xyB, options = {}) {
    const M = this.M;
    if (!this.handle) throw new Error('matcher destroyed');
    const nA = xyA.length / 2, nB = xyB.length / 2;
    const pa = this.scratch('xyA', xyA.length * 4), pb = this.scratch('xyB', xyB.length * 4);
    M.HEAPF32.set(xyA, pa / 4);
    M.HEAPF32.set(xyB, pb / 4);
    const n = await M._jmtw_match(this.handle, a.ptr, pa, nA, b.ptr, pb, nB, options.layers ?? 9,
                                  options.filterThreshold ?? 0.1, options.maxDistancePx ?? 3,
                                  options.minCovisibility ?? 0, options.earlyStopDecay ?? -1);
    if (n < 0) throw new Error(M.UTF8ToString(M._jmtw_error()));
    const h = this.handle;
    const f32 = (ptr, count) => new Float32Array(M.HEAPF32.buffer, ptr, count).slice();
    return {
      n,
      matches: new Int32Array(M.HEAP32.buffer, M._jmtw_idx(h), 2 * n).slice(),
      scores: f32(M._jmtw_score(h), n),
      aInB: f32(M._jmtw_a_in_b(h), 2 * nA),
      bInA: f32(M._jmtw_b_in_a(h), 2 * nB),
      confA: f32(M._jmtw_conf_a(h), nA),
      confB: f32(M._jmtw_conf_b(h), nB),
      covisibility: M._jmtw_covisibility(h),
      stopLayer: M._jmtw_stop_layer(h),
    };
  }

  // Score of a pair at level 1 or 2 (see the top of this file).
  async covisibility(a, b, level = 1) {
    if (!this.handle) throw new Error('matcher destroyed');
    const s = await this.M._jmtw_covisibility_level(this.handle, a.ptr, b.ptr, level);
    if (Number.isNaN(s)) throw new Error(this.M.UTF8ToString(this.M._jmtw_error()));
    return s;
  }

  // Module memory holding the features pointers of `images`.
  pointers(images) {
    const p = this.scratch('images', images.length * 4);
    for (const [i, f] of images.entries()) {
      if (!f.ptr) throw new Error('features destroyed');
      this.M.HEAPU32[p / 4 + i] = f.ptr;
    }
    return p;
  }

  // Level 1 of every pair: Float32Array [n][n], 1 on the diagonal.
  async level1(images) {
    const M = this.M, n = images.length;
    if (!this.handle) throw new Error('matcher destroyed');
    const p = this.pointers(images), out = this.scratch('level1', n * n * 4);
    if ((await M._jmtw_level1_matrix(this.handle, p, n, out)) !== 0) throw new Error(M.UTF8ToString(M._jmtw_error()));
    return new Float32Array(M.HEAPF32.buffer, out, n * n).slice();
  }

  // options: k (10) partners per image by level 1, neighbours (0) in the order given, rerank (false: score
  // the kept pairs with level 2 and sort by it), minScore (0, with rerank).
  async shortlist(images, options = {}) {
    const M = this.M;
    if (!this.handle) throw new Error('matcher destroyed');
    const p = this.pointers(images);
    const n = await M._jmtw_shortlist(this.handle, p, images.length, options.k ?? 10, options.neighbours ?? 0,
                                      options.rerank ? 1 : 0, options.minScore ?? 0);
    if (n < 0) throw new Error(M.UTF8ToString(M._jmtw_error()));
    return {
      n,
      pairs: new Int32Array(M.HEAP32.buffer, M._jmtw_pairs_idx(this.handle), 2 * n).slice(),
      scores: new Float32Array(M.HEAPF32.buffer, M._jmtw_pairs_score(this.handle), n).slice(),
    };
  }

  destroy() {
    if (!this.handle) return;
    this.M._jmtw_destroy(this.handle);
    for (const b of Object.values(this.buffers)) this.M._free(b.ptr);
    this.buffers = {};
    this.handle = 0;
  }
}
