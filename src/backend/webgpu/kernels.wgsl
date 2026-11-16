// WebGPU backend kernels (WGSL) for the operations of core/ops.h. Tensors are token-major: element
// (token, channel) is at token * c + channel. Mostly a port of src/backend/metal/kernels.metal (same names,
// same arithmetic per output value); the matrix product and attention are written for WGSL's workgroup
// memory instead of Metal's matrix primitives.
//
// Layout of this file (read by webgpu_ops.cpp):
// - Everything before the first `//@kernel` line is the prelude, included in every kernel's module.
// - `//@kernel NAME` starts a kernel section. Each kernel is compiled as its own module (prelude +
//   section), so sections declare their own bindings and may reuse names and binding numbers.
// - A section defines either `fn run(i: u32)`, called once per thread of a 1-D grid (the host appends the
//   entry point; surplus threads have i >= the kernel's count), or, with a `//@group N` line,
//   `fn group(g: u32, li: u32)`, called by every thread of N-thread workgroup number g (li: thread in the
//   group; surplus groups have g >= the kernel's count).
// - `$O` is the output type of a kernel instance: `real`, or `f32` for an instance named with `_f32`.
//   An instance name may also carry `_r{R}c{C}` (replaces `$R` and `$C`) and flags such as `_v4`: a line
//   that starts with `//?v4 ` is kept (without that prefix) only in instances with the flag, a line that
//   starts with `//?s ` only in instances without any flag.
// - Binding 0 holds the kernel's scalar arguments (`P`, 64 words, read with pi / pu / pf by position, in
//   the order the host passes them). Buffers bind to 1, 2, ... in the order the host passes them.
//
// Precision: `real` is the storage type of tensors and weights: f32, or f16 in FP16 mode (the host
// prepends `enable f16;` and the alias). Arithmetic and accumulation are f32 in both modes.
//
// Differences from the Metal source forced by WGSL: no 8-bit storage (the image is read from 32-bit
// words), no infinity literal, and no way to turn off contraction into fused multiply-adds.

struct Params {
  v: array<vec4<u32>, 16>,
}
@group(0) @binding(0) var<uniform> P: Params;

fn pu(i: u32) -> u32 { return P.v[i >> 2u][i & 3u]; }
fn pi(i: u32) -> i32 { return bitcast<i32>(pu(i)); }
fn pf(i: u32) -> f32 { return bitcast<f32>(pu(i)); }

const kGelu = 0;
const kRelu = 1;
const kNone = 2;
const kLowest = -3.4028234663852886e38;

// erf by Abramowitz and Stegun 7.1.26: absolute error below 1.5e-7 (as the Metal kernels).
fn erf_approx(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  let y = 1.0 - poly * exp(-a * a);
  return select(y, -y, x < 0.0);
}

fn activate(v: f32, act: i32) -> f32 {
  if (act == kGelu) { return 0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752440)); }
  if (act == kRelu) { return max(v, 0.0); }
  return v;
}

// a / b rounded to nearest. WGSL's division may be approximate (it is on Dawn's Metal path); two
// corrections with fused multiply-adds round it correctly.
fn div_rn(a: f32, b: f32) -> f32 {
  let y = 1.0 / b;
  var q = a * y;
  q = fma(fma(-q, b, a), y, q);
  return fma(fma(-q, b, a), y, q);
}

//@kernel k_mm
// Matrix product: out (n, cout) = act(scale * x (n, cin) W^T + bias) + residual, W stored (cout, cin) as in
// the weights file. Scalars: n, cin, cout, flags (1 = bias, 2 = residual (n, cout)), act, scale.
// One workgroup: a (16 R) x (16 C) tile of the output; one thread: R tokens x C channels, interleaved
// (thread (ty, tx) has tokens ty + 16 r and channels tx + 16 c), so neighbouring threads store
// neighbouring outputs. The operands go through workgroup memory 16 inputs at a time, a row as 4 vectors
// (plus one of padding against bank conflicts); a thread reads one vector per row and input quad and
// accumulates dot products. Every vector in workgroup memory is written whole by one thread (threads
// writing different components of one vector race on some compilers).
// Instances: k_mm_r{R}c{C}, with _v4 when cin is a multiple of 4 (vector loads from memory).
// Groups: ceil(cout / (16 C)) * ceil(n / (16 R)), channel tile fastest.
//@group 256
//?v4 @group(0) @binding(1) var<storage, read> x: array<vec4<real>>;
//?v4 @group(0) @binding(2) var<storage, read> wt: array<vec4<real>>;
//?s @group(0) @binding(1) var<storage, read> x: array<real>;
//?s @group(0) @binding(2) var<storage, read> wt: array<real>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read> residual: array<real>;
@group(0) @binding(5) var<storage, read_write> out: array<$O>;
const TR = $Ru;
const TC = $Cu;
const BR = 16u * TR;
const BC = 16u * TC;
const SV = 5u;  // vectors per tile row: 16 inputs and one of padding
var<workgroup> xs: array<vec4<f32>, BR * SV>;
var<workgroup> ws: array<vec4<f32>, BC * SV>;

// Inputs k .. k + 3 of row `row` (zeros past cin); row * cin + k is a multiple of 4 in the _v4 instance.
fn load4x(row: u32, k: u32, cin: u32) -> vec4<f32> {
//?v4 if (k >= cin) { return vec4<f32>(0.0); }
//?v4 return vec4<f32>(x[(row * cin + k) / 4u]);
//?s var r = vec4<f32>(0.0);
//?s for (var j = 0u; j < 4u; j++) { if (k + j < cin) { r[j] = f32(x[row * cin + k + j]); } }
//?s return r;
}
fn load4w(row: u32, k: u32, cin: u32) -> vec4<f32> {
//?v4 if (k >= cin) { return vec4<f32>(0.0); }
//?v4 return vec4<f32>(wt[(row * cin + k) / 4u]);
//?s var r = vec4<f32>(0.0);
//?s for (var j = 0u; j < 4u; j++) { if (k + j < cin) { r[j] = f32(wt[row * cin + k + j]); } }
//?s return r;
}

fn group(g: u32, li: u32) {
  let n = pu(0u);
  let cin = pu(1u);
  let cout = pu(2u);
  let flags = pu(3u);
  let ctiles = (cout + BC - 1u) / BC;
  if (g >= ctiles * ((n + BR - 1u) / BR)) { return; }
  let c0 = (g % ctiles) * BC;
  let r0 = (g / ctiles) * BR;
  let tx = li % 16u;
  let ty = li / 16u;
  var acc: array<f32, TR * TC>;
  for (var i = 0u; i < TR * TC; i++) { acc[i] = 0.0; }
  // Loads: vector e of a tile is inputs 4 (e % 4) .. + 3 of its row e / 4. The next tile's are fetched
  // into registers while the current one is multiplied.
  var xv: array<vec4<f32>, TR / 4u>;
  var wv: array<vec4<f32>, TC / 4u>;
  for (var i = 0u; i < TR / 4u; i++) {
    let e = li + 256u * i;
    xv[i] = load4x(min(r0 + e / 4u, n - 1u), (e % 4u) * 4u, cin);
  }
  for (var i = 0u; i < TC / 4u; i++) {
    let e = li + 256u * i;
    wv[i] = load4w(min(c0 + e / 4u, cout - 1u), (e % 4u) * 4u, cin);
  }
  for (var k0 = 0u; k0 < cin; k0 += 16u) {
    workgroupBarrier();
    for (var i = 0u; i < TR / 4u; i++) {
      let e = li + 256u * i;
      xs[(e / 4u) * SV + e % 4u] = xv[i];
    }
    for (var i = 0u; i < TC / 4u; i++) {
      let e = li + 256u * i;
      ws[(e / 4u) * SV + e % 4u] = wv[i];
    }
    workgroupBarrier();
    if (k0 + 16u < cin) {
      for (var i = 0u; i < TR / 4u; i++) {
        let e = li + 256u * i;
        xv[i] = load4x(min(r0 + e / 4u, n - 1u), k0 + 16u + (e % 4u) * 4u, cin);
      }
      for (var i = 0u; i < TC / 4u; i++) {
        let e = li + 256u * i;
        wv[i] = load4w(min(c0 + e / 4u, cout - 1u), k0 + 16u + (e % 4u) * 4u, cin);
      }
    }
    for (var q = 0u; q < 4u; q++) {
      var b: array<vec4<f32>, TC>;
      for (var c = 0u; c < TC; c++) { b[c] = ws[(tx + 16u * c) * SV + q]; }
      for (var r = 0u; r < TR; r++) {
        let a = xs[(ty + 16u * r) * SV + q];
        for (var c = 0u; c < TC; c++) { acc[r * TC + c] += dot(a, b[c]); }
      }
    }
  }
  let scale = pf(5u);
  let act = pi(4u);
  for (var r = 0u; r < TR; r++) {
    let row = r0 + ty + 16u * r;
    if (row >= n) { break; }
    for (var c = 0u; c < TC; c++) {
      let col = c0 + tx + 16u * c;
      if (col >= cout) { break; }
      var v = acc[r * TC + c] * scale;
      if ((flags & 1u) != 0u) { v += f32(bias[col]); }
      v = activate(v, act);
      let o = row * cout + col;
      if ((flags & 2u) != 0u) { v += f32(residual[o]); }
      out[o] = $O(v);
    }
  }
}

//@kernel k_patches
// Patches for a k x k convolution: out (oh * ow, k * k * c), ordered (ky, kx, channel); zeros outside.
// Scalars: h, w, c, k, stride, pad, oh, ow.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let h = pi(0u);
  let w = pi(1u);
  let c = pi(2u);
  let k = pi(3u);
  let ow = pi(7u);
  let cols = u32(k * k * c);
  if (gid >= u32(pi(6u) * ow) * cols) { return; }
  let tok = i32(gid / cols);
  let col = i32(gid % cols);
  let tap = col / c;
  let ch = col % c;
  let sy = (tok / ow) * pi(4u) - pi(5u) + tap / k;
  let sx = (tok % ow) * pi(4u) - pi(5u) + tap % k;
  if (sy >= 0 && sy < h && sx >= 0 && sx < w) {
    out[gid] = x[u32((sy * w + sx) * c + ch)];
  } else {
    out[gid] = real(0.0);
  }
}

//@kernel k_dwconv7
// Depthwise 7x7, padding 3. taps: (49, c). One thread: 4 channels of 4 neighbouring outputs of a row, so
// every input it loads serves the taps of all 4. Scalars: h, w, c. Grid: h * ceil(w / 4) * c / 4.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> taps: array<real>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;

fn ld4(a: u32) -> vec4<f32> { return vec4<f32>(f32(x[a]), f32(x[a + 1u]), f32(x[a + 2u]), f32(x[a + 3u])); }
fn tap4(a: u32) -> vec4<f32> {
  return vec4<f32>(f32(taps[a]), f32(taps[a + 1u]), f32(taps[a + 2u]), f32(taps[a + 3u]));
}

fn run(gid: u32) {
  let h = pi(0u);
  let w = pi(1u);
  let c = pi(2u);
  let c4 = c / 4;
  let wq = (w + 3) / 4;
  if (gid >= u32(h * wq * c4)) { return; }
  let q = i32(gid) % c4;
  let rest = i32(gid) / c4;
  let x0 = (rest % wq) * 4;
  let y = rest / wq;
  let b = vec4<f32>(f32(bias[q * 4]), f32(bias[q * 4 + 1]), f32(bias[q * 4 + 2]), f32(bias[q * 4 + 3]));
  var acc = array<vec4<f32>, 4>(b, b, b, b);
  for (var ky = 0; ky < 7; ky++) {
    let sy = y + ky - 3;
    if (sy < 0 || sy >= h) { continue; }
    // Inputs x0 - 3 .. x0 + 6 of the row; output o uses inputs o .. o + 6 of them.
    for (var s = 0; s < 10; s++) {
      let sx = x0 - 3 + s;
      if (sx < 0 || sx >= w) { continue; }
      let v = ld4(u32((sy * w + sx) * c + q * 4));
      for (var o = 0; o < 4; o++) {
        let kx = s - o;
        if (kx >= 0 && kx < 7) { acc[o] = fma(tap4(u32((ky * 7 + kx) * c + q * 4)), v, acc[o]); }
      }
    }
  }
  for (var o = 0; o < 4; o++) {
    if (x0 + o >= w) { break; }
    let a = u32((y * w + x0 + o) * c + q * 4);
    out[a] = real(acc[o].x);
    out[a + 1u] = real(acc[o].y);
    out[a + 2u] = real(acc[o].z);
    out[a + 3u] = real(acc[o].w);
  }
}

//@kernel k_layer_norm
// LayerNorm over groups of g channels (g <= 1024), then an activation. A workgroup of 256 threads takes
// 256 / L groups, L threads each (L = 16 for g <= 64, 32 for g <= 128, else 64); a thread holds up to 16
// values. Scalars: groups (n * c / g), g, eps, act, L. Groups: ceil(groups / (256 / L)).
//@group 256
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> gain: array<real>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;
var<workgroup> part: array<f32, 256>;

// Sum over the L threads of each group (threads li - li % L ..).
fn reduce(v: f32, li: u32, L: u32) -> f32 {
  part[li] = v;
  workgroupBarrier();
  for (var s = L / 2u; s > 0u; s >>= 1u) {
    if (li % L < s) { part[li] += part[li + s]; }
    workgroupBarrier();
  }
  let total = part[li - li % L];
  workgroupBarrier();
  return total;
}

fn group(g: u32, li: u32) {
  let groups = pu(0u);
  let gs = pu(1u);
  let L = pu(4u);
  if (g * (256u / L) >= groups) { return; }
  let mine = g * (256u / L) + li / L;
  let lane = li % L;
  let valid = mine < groups;
  let base = min(mine, groups - 1u) * gs;
  var vals: array<f32, 16>;
  var s = 0.0;
  for (var j = 0u; j < 16u; j++) {
    let at = lane + j * L;
    if (at < gs) {
      vals[j] = f32(x[base + at]);
      s += vals[j];
    }
  }
  let mean = reduce(s, li, L) / f32(gs);
  var sq = 0.0;
  for (var j = 0u; j < 16u; j++) {
    if (lane + j * L < gs) {
      vals[j] -= mean;
      sq = fma(vals[j], vals[j], sq);
    }
  }
  let inv = 1.0 / sqrt(reduce(sq, li, L) / f32(gs) + pf(2u));
  let act = pi(3u);
  if (!valid) { return; }
  for (var j = 0u; j < 16u; j++) {
    let at = lane + j * L;
    if (at < gs) { out[base + at] = real(activate(vals[j] * inv * f32(gain[at]) + f32(bias[at]), act)); }
  }
}

//@kernel k_unary
// Scalars: count, op (kGelu, kRelu).
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  if (gid >= pu(0u)) { return; }
  out[gid] = real(activate(f32(x[gid]), pi(1u)));
}

//@kernel k_add
// Scalars: count.
@group(0) @binding(1) var<storage, read> a: array<real>;
@group(0) @binding(2) var<storage, read> b: array<real>;
@group(0) @binding(3) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  if (gid >= pu(0u)) { return; }
  out[gid] = real(f32(a[gid]) + f32(b[gid]));
}

//@kernel k_copy_channels
// Copies channels [first, first + count) of x (c channels) to channels [at, at + count) of out (oc
// channels). Serves slice (at = 0, oc = count) and both halves of concat. Scalars: n, c, first, count, at, oc.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let count = pu(3u);
  if (gid >= pu(0u) * count) { return; }
  let tok = gid / count;
  let j = gid % count;
  out[tok * pu(5u) + pu(4u) + j] = x[tok * pu(1u) + pu(2u) + j];
}

//@kernel k_pixel_unshuffle2
// (h, w, c) -> (h / 2, w / 2, 4c); output channel = c * 4 + dy * 2 + dx. Scalars: h, w, c.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let h = pu(0u);
  let w = pu(1u);
  let c = pu(2u);
  let oc = 4u * c;
  let ow = w / 2u;
  if (gid >= (h / 2u) * ow * oc) { return; }
  let tok = gid / oc;
  let j = gid % oc;
  let sy = 2u * (tok / ow) + (j % 4u) / 2u;
  let sx = 2u * (tok % ow) + j % 2u;
  out[gid] = x[(sy * w + sx) * c + j / 4u];
}

//@kernel k_resize
// Bilinear resize (align_corners = false). Scalars: h, w, c, oh, ow.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;

struct Taps {
  i0: i32,
  i1: i32,
  t: f32,
}

fn source(i: i32, o: i32, n: i32) -> Taps {
  let s = max((f32(i) + 0.5) * (f32(n) / f32(o)) - 0.5, 0.0);
  let i0 = min(i32(s), n - 1);
  return Taps(i0, min(i0 + 1, n - 1), s - f32(i0));
}

fn run(gid: u32) {
  let h = pi(0u);
  let w = pi(1u);
  let c = pi(2u);
  let oh = pi(3u);
  let ow = pi(4u);
  if (gid >= u32(oh * ow * c)) { return; }
  let tok = i32(gid) / c;
  let ch = i32(gid) % c;
  let ty = source(tok / ow, oh, h);
  let tx = source(tok % ow, ow, w);
  let p00 = f32(x[u32((ty.i0 * w + tx.i0) * c + ch)]);
  let p01 = f32(x[u32((ty.i0 * w + tx.i1) * c + ch)]);
  let p10 = f32(x[u32((ty.i1 * w + tx.i0) * c + ch)]);
  let p11 = f32(x[u32((ty.i1 * w + tx.i1) * c + ch)]);
  out[gid] = real((p00 * (1.0 - tx.t) + p01 * tx.t) * (1.0 - ty.t) + (p10 * (1.0 - tx.t) + p11 * tx.t) * ty.t);
}

//@kernel k_l2_normalize
// Per token: x / max(|x|, 1e-12). Scalars: n, c.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let c = pu(1u);
  if (gid >= pu(0u)) { return; }
  let base = gid * c;
  var sq = 0.0;
  for (var j = 0u; j < c; j++) {
    let v = f32(x[base + j]);
    sq = fma(v, v, sq);
  }
  let inv = 1.0 / max(sqrt(sq), 1e-12);
  for (var j = 0u; j < c; j++) { out[base + j] = real(f32(x[base + j]) * inv); }
}

//@kernel k_rotary
// Rotary encoding. cosv, sinv: (n, pairs). Scalars: n, heads, pairs.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> cosv: array<f32>;
@group(0) @binding(3) var<storage, read> sinv: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let pairs = pu(2u);
  let per_token = pu(1u) * pairs;
  if (gid >= pu(0u) * per_token) { return; }
  let tok = gid / per_token;
  let j = (gid % per_token) % pairs;
  let c = cosv[tok * pairs + j];
  let s = sinv[tok * pairs + j];
  let a = f32(x[2u * gid]);
  let b = f32(x[2u * gid + 1u]);
  out[2u * gid] = real(a * c - b * s);
  out[2u * gid + 1u] = real(b * c + a * s);
}

//@kernel k_sample
// Bilinear samples at points in [-1, 1], border clamped. Scalars: h, w, c, n.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> pts: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let h = pi(0u);
  let w = pi(1u);
  let c = pi(2u);
  if (gid >= u32(pi(3u) * c)) { return; }
  let i = i32(gid) / c;
  let ch = i32(gid) % c;
  let fx = clamp((pts[2 * i] + 1.0) * 0.5 * f32(w) - 0.5, 0.0, f32(w - 1));
  let fy = clamp((pts[2 * i + 1] + 1.0) * 0.5 * f32(h) - 0.5, 0.0, f32(h - 1));
  let x0 = i32(fx);
  let y0 = i32(fy);
  let x1 = min(x0 + 1, w - 1);
  let y1 = min(y0 + 1, h - 1);
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let p00 = f32(x[u32((y0 * w + x0) * c + ch)]);
  let p01 = f32(x[u32((y0 * w + x1) * c + ch)]);
  let p10 = f32(x[u32((y1 * w + x0) * c + ch)]);
  let p11 = f32(x[u32((y1 * w + x1) * c + ch)]);
  out[gid] = real(p00 * (1.0 - tx) * (1.0 - ty) + p01 * tx * (1.0 - ty) + p10 * (1.0 - tx) * ty + p11 * tx * ty);
}

//@kernel k_attn
// Attention with 64 channels per head, without a weights buffer: out = softmax(q k^T / 8) v per head.
// One workgroup: 64 queries of one head; it walks the source tokens 64 at a time. Thread (ty, tx) holds
// queries ty + 16 r (r < 4) and, per tile, the logits of tokens 4 tx .. 4 tx + 3. Per tile:
//   logits    q k^T, the operands through workgroup memory 16 channels at a time as vectors of 4 channels;
//             a thread reads its keys' vectors in an order rotated by tx, which spreads the reads of the
//             16 threads of a row over the banks
//   softmax   the largest logit of each query so far, from the 16 threads that share its row; when it
//             grows, the outputs and the partial sums so far are scaled to the new reference
//   values    out += weights v, 16 tokens at a time: the weights as vectors of 4 tokens (each written
//             whole by the thread that holds them), the values as vectors of 4 channels; a thread holds
//             outputs 4 tx .. 4 tx + 3 of its queries
// Each thread keeps the sum of its own weights; the 16 partial sums of a row are added at the end.
// Scalars: P, S, heads. Groups: heads * ceil(P / 64).
//@group 256
@group(0) @binding(1) var<storage, read> q: array<vec4<real>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<real>>;
@group(0) @binding(3) var<storage, read> v: array<vec4<real>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<real>>;
const SV = 5u;  // vectors per row of a logits operand tile: 4 and one of padding
// Logits: queries at [0, 320) and keys at [320, 640), row-major with SV vectors per row.
// Values phase: weights at [0, 256) (query * 4 + token quad), values at [256, 512) (token * 16 + quad).
var<workgroup> sm: array<vec4<f32>, 640>;
var<workgroup> red: array<f32, 1024>;  // [64 queries][16 threads]: partial maxima, then partial sums

fn group(g: u32, li: u32) {
  let Pq = pu(0u);
  let S = pu(1u);
  let heads = pu(2u);
  let C4 = heads * 16u;  // vectors per token
  let pblocks = (Pq + 63u) / 64u;
  if (g >= heads * pblocks) { return; }
  let hd = g / pblocks;
  let p0 = (g % pblocks) * 64u;
  let tx = li % 16u;
  let ty = li / 16u;
  // Logits loads: vector li % 4 of the 16-channel chunk of row li / 4, for queries and keys.
  let lr = li / 4u;
  let lq = li % 4u;
  let qbase = min(p0 + lr, Pq - 1u) * C4 + hd * 16u + lq;
  // Values loads: vector li % 16 of token li / 16 of a 16-token chunk.
  let vt = li / 16u;
  var o: array<vec4<f32>, 4>;
  var top = vec4<f32>(kLowest);
  var part = vec4<f32>(0.0);  // this thread's share of each row's sum of weights
  for (var r = 0u; r < 4u; r++) { o[r] = vec4<f32>(0.0); }
  for (var s0 = 0u; s0 < S; s0 += 64u) {
    let kbase = min(s0 + lr, S - 1u) * C4 + hd * 16u + lq;
    var l: array<vec4<f32>, 4>;  // l[r][c]: query ty + 16 r, token s0 + 4 tx + c
    for (var r = 0u; r < 4u; r++) { l[r] = vec4<f32>(0.0); }
    for (var c0 = 0u; c0 < 16u; c0 += 4u) {
      workgroupBarrier();
      sm[lr * SV + lq] = vec4<f32>(q[qbase + c0]);
      sm[320u + lr * SV + lq] = vec4<f32>(k[kbase + c0]);
      workgroupBarrier();
      for (var j = 0u; j < 4u; j++) {
        let qv = (j + tx) % 4u;
        let kb = 320u + tx * 4u * SV + qv;
        let b0 = sm[kb];
        let b1 = sm[kb + SV];
        let b2 = sm[kb + 2u * SV];
        let b3 = sm[kb + 3u * SV];
        for (var r = 0u; r < 4u; r++) {
          let a = sm[(ty + 16u * r) * SV + qv];
          l[r] += vec4<f32>(dot(a, b0), dot(a, b1), dot(a, b2), dot(a, b3));
        }
      }
    }
    // Scaled logits; source tokens past S get the lowest value. Then the row maxima.
    let valid = vec4<u32>(s0 + tx * 4u) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(S);
    for (var r = 0u; r < 4u; r++) {
      l[r] = select(vec4<f32>(kLowest), l[r] * 0.125, valid);  // 1 / sqrt(64)
      red[(ty + 16u * r) * 16u + tx] = max(max(l[r].x, l[r].y), max(l[r].z, l[r].w));
    }
    workgroupBarrier();
    var grown = top;
    for (var t = 0u; t < 16u; t++) {
      grown = max(grown, vec4<f32>(red[ty * 16u + t], red[(ty + 16u) * 16u + t], red[(ty + 32u) * 16u + t],
                                   red[(ty + 48u) * 16u + t]));
    }
    let shrink = exp(top - grown);  // 0 on the first tile, 1 for rows whose maximum did not grow
    top = grown;
    part *= shrink;
    for (var r = 0u; r < 4u; r++) {
      o[r] *= shrink[r];
      l[r] = select(vec4<f32>(0.0), exp(l[r] - vec4<f32>(top[r])), valid);  // the weights
      part[r] += (l[r].x + l[r].y) + (l[r].z + l[r].w);
    }
    for (var t0 = 0u; t0 < 64u; t0 += 16u) {
      if (tx / 4u == t0 / 16u) {
        for (var r = 0u; r < 4u; r++) { sm[(ty + 16u * r) * 4u + tx % 4u] = l[r]; }
      }
      sm[256u + li] = vec4<f32>(v[min(s0 + t0 + vt, S - 1u) * C4 + hd * 16u + tx]);
      workgroupBarrier();
      for (var j = 0u; j < 4u; j++) {
        let vb = 256u + j * 64u + tx;
        let b0 = sm[vb];
        let b1 = sm[vb + 16u];
        let b2 = sm[vb + 32u];
        let b3 = sm[vb + 48u];
        for (var r = 0u; r < 4u; r++) {
          let w = sm[(ty + 16u * r) * 4u + j];
          o[r] = fma(vec4<f32>(w.x), b0, fma(vec4<f32>(w.y), b1, fma(vec4<f32>(w.z), b2, fma(vec4<f32>(w.w), b3, o[r]))));
        }
      }
      workgroupBarrier();
    }
  }
  // Row sums from the 16 partial sums, then the outputs.
  for (var r = 0u; r < 4u; r++) { red[(ty + 16u * r) * 16u + tx] = part[r]; }
  workgroupBarrier();
  var sum = vec4<f32>(0.0);
  for (var t = 0u; t < 16u; t++) {
    sum += vec4<f32>(red[ty * 16u + t], red[(ty + 16u) * 16u + t], red[(ty + 32u) * 16u + t], red[(ty + 48u) * 16u + t]);
  }
  for (var r = 0u; r < 4u; r++) {
    let p = p0 + ty + 16u * r;
    if (p < Pq) { out[p * C4 + hd * 16u + tx] = vec4<real>(o[r] / sum[r]); }
  }
}

//@kernel k_readout
// Match read-out of one query from its float logits over the h x w cells (see Ops::readout and
// core/readout.h, which this follows; the same method as the Metal kernel). One workgroup of 128 threads
// per query; every thread holds up to 64 cells (P <= 8192) and the group reduces in workgroup memory.
//
// The 1.5-entmax threshold tau solves sum over T of max(y - tau, 0)^2 = 1 with y = (logit - top) / 2 and
// T the 512 largest logits. The sum is convex and decreasing in tau and is at least 1 at tau = -1, so
// Newton's method from -1 climbs to the root without passing it. Cells with y <= -1 never count; when
// more than 512 cells lie above -1, the 512th largest y is found by bisection and the cells below it are
// left out. Scalars: N, h, w. Groups: N.
//@group 128
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;
@group(0) @binding(3) var<storage, read_write> mass: array<f32>;
var<workgroup> fpart: array<f32, 128>;
var<workgroup> ipart: array<i32, 128>;
var<workgroup> fpart2: array<f32, 128>;
var<workgroup> rf: f32;
var<workgroup> rf2: vec2<f32>;
var<workgroup> ri: i32;

fn sum_f(v: f32, li: u32) -> f32 {
  fpart[li] = v;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (li < s) { fpart[li] += fpart[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { rf = fpart[0]; }
  return workgroupUniformLoad(&rf);
}

fn sum2_f(a: f32, b: f32, li: u32) -> vec2<f32> {
  fpart[li] = a;
  fpart2[li] = b;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (li < s) {
      fpart[li] += fpart[li + s];
      fpart2[li] += fpart2[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) { rf2 = vec2<f32>(fpart[0], fpart2[0]); }
  return workgroupUniformLoad(&rf2);
}

fn sum_i(v: i32, li: u32) -> i32 {
  ipart[li] = v;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (li < s) { ipart[li] += ipart[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { ri = ipart[0]; }
  return workgroupUniformLoad(&ri);
}

fn group(g: u32, li: u32) {
  let N = pu(0u);
  let h = pi(1u);
  let w = pi(2u);
  if (g >= N) { return; }
  let Pc = h * w;
  let row = g * u32(Pc);
  // Thread li holds cells li, li + 128, ...
  var ys: array<f32, 64>;
  var best = kLowest;
  var at = Pc;
  for (var j = 0; j < 64; j++) {
    let s = i32(li) + j * 128;
    if (s < Pc) {
      let val = logits[row + u32(s)];
      ys[j] = val;
      if (val > best) { best = val; at = s; }
    }
  }
  // Largest logit and the first cell that has it.
  fpart[li] = best;
  ipart[li] = at;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (li < s) {
      let o = li + s;
      if (fpart[o] > fpart[li] || (fpart[o] == fpart[li] && ipart[o] < ipart[li])) {
        fpart[li] = fpart[o];
        ipart[li] = ipart[o];
      }
    }
    workgroupBarrier();
  }
  if (li == 0u) { rf = fpart[0]; ri = ipart[0]; }
  let top = workgroupUniformLoad(&rf);
  let arg = workgroupUniformLoad(&ri);
  for (var j = 0; j < 64; j++) { ys[j] = (ys[j] - top) * 0.5; }
  // Cells that can have weight: y > cut. `extra` copies of the value `cut` stand for the cells that tie
  // at the 512th place.
  var cut = -1.0;
  var c = 0;
  for (var j = 0; j < 64; j++) {
    if (i32(li) + j * 128 < Pc && ys[j] > cut) { c++; }
  }
  let count = sum_i(c, li);
  var extra = 0;
  if (count > 512) {
    var lo = -1.0;
    var hi = 0.0;  // count(y > lo) >= 512 > count(y > hi)
    for (var it = 0; it < 30; it++) {
      let mid = 0.5 * (lo + hi);
      c = 0;
      for (var j = 0; j < 64; j++) {
        if (i32(li) + j * 128 < Pc && ys[j] > mid) { c++; }
      }
      if (sum_i(c, li) >= 512) { lo = mid; } else { hi = mid; }
    }
    c = 0;
    for (var j = 0; j < 64; j++) {
      if (i32(li) + j * 128 < Pc && ys[j] > hi) { c++; }
    }
    cut = hi;
    extra = 512 - sum_i(c, li);
  }
  var tau = -1.0;
  for (var it = 0; it < 40; it++) {
    var a = 0.0;
    var b = 0.0;
    for (var j = 0; j < 64; j++) {
      if (i32(li) + j * 128 < Pc && ys[j] > cut) {
        let d = max(ys[j] - tau, 0.0);
        a += d;
        b = fma(d, d, b);
      }
    }
    let de = max(cut - tau, 0.0) * f32(extra);
    let ab = sum2_f(a, b, li);
    let sa = ab.x + de;
    let sb = ab.y + de * max(cut - tau, 0.0);
    let next = tau + (sb - 1.0) / (2.0 * sa);
    if (!(next > tau)) { break; }
    tau = next;
  }
  if (li != 0u) { return; }
  // Expected cell-centre position in the 3 x 3 window around the best cell, and the window's mass.
  let by = arg / w;
  let bx = arg % w;
  var px = 0.0;
  var py = 0.0;
  var total = 0.0;
  var inside = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let ny = by + dy;
      let nx = bx + dx;
      let isin = ny >= 0 && ny < h && nx >= 0 && nx < w;
      let cy = clamp(ny, 0, h - 1);
      let cx = clamp(nx, 0, w - 1);
      var pr = max((logits[row + u32(cy * w + cx)] - top) * 0.5 - tau, 0.0);
      pr *= pr;
      px += pr * ((f32(cx) + 0.5) / f32(w) * 2.0 - 1.0);
      py += pr * ((f32(cy) + 0.5) / f32(h) * 2.0 - 1.0);
      total += pr;
      if (isin) { inside += pr; }
    }
  }
  coarse[2u * g] = px / total;
  coarse[2u * g + 1u] = py / total;
  mass[g] = inside;
}

//@kernel k_preprocess
// Network input from 8-bit pixels: bilinear resize to oh x ow of pixel / 255, then (v - mean) / std per
// channel. The image is read from 32-bit words (WGSL has no 8-bit storage). The source rows and columns
// come from the host (taps: 3 floats per output row, then 3 per output column: first index, second index,
// weight of the second), so they are the same as on the other backends to the last bit.
// Scalars: stride, ch, oh, ow, mean (3), std (3).
@group(0) @binding(1) var<storage, read> img: array<u32>;
@group(0) @binding(2) var<storage, read> taps: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<real>;

fn byte(at: u32) -> f32 { return f32((img[at / 4u] >> (8u * (at % 4u))) & 255u); }

fn run(gid: u32) {
  let stride = pu(0u);
  let ch = pu(1u);
  let oh = pu(2u);
  let ow = pu(3u);
  if (gid >= oh * ow) { return; }
  let ty3 = 3u * (gid / ow);
  let tx3 = 3u * (oh + gid % ow);
  let y0 = u32(taps[ty3]);
  let y1 = u32(taps[ty3 + 1u]);
  let ty = taps[ty3 + 2u];
  let x0 = u32(taps[tx3]);
  let x1 = u32(taps[tx3 + 1u]);
  let tx = taps[tx3 + 2u];
  for (var c = 0u; c < 3u; c++) {
    let o = select(c, 0u, ch == 1u);
    let p00 = div_rn(byte(y0 * stride + x0 * ch + o), 255.0);
    let p01 = div_rn(byte(y0 * stride + x1 * ch + o), 255.0);
    let p10 = div_rn(byte(y1 * stride + x0 * ch + o), 255.0);
    let p11 = div_rn(byte(y1 * stride + x1 * ch + o), 255.0);
    let top = p00 * (1.0 - tx) + p01 * tx;
    let bottom = p10 * (1.0 - tx) + p11 * tx;
    out[gid * 3u + c] = real(div_rn((top * (1.0 - ty) + bottom * ty) - pf(4u + c), pf(7u + c)));
  }
}
