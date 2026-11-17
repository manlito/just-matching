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
//   An instance name may also carry `_r{R}c{C}` (replaces `$R` and `$C`).
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
// The general form, for any cin; k_mmd (cin a multiple of 4) and k_mmh (FP16, cin a multiple of 8) are the
// fast ones. One workgroup: a (16 R) x (16 C) tile of the output; one thread: R tokens x C channels,
// interleaved (thread (ty, tx) has tokens ty + 16 r and channels tx + 16 c), so neighbouring threads store
// neighbouring outputs. The operands go through workgroup memory 16 inputs at a time, a row as 4 vectors
// plus one of padding. Every vector in workgroup memory is written whole by one thread (threads writing
// different components of one vector race on some compilers). Instances: k_mm_r{R}c{C}.
// Groups: ceil(cout / (16 C)) * ceil(n / (16 R)), channel tile fastest.
//@group 256
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> wt: array<real>;
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

// Inputs k .. k + 3 of the row starting at `base` (zeros past cin).
fn load4x(base: u32, k: u32, cin: u32) -> vec4<f32> {
  var r = vec4<f32>(0.0);
  for (var j = 0u; j < 4u; j++) { if (k + j < cin) { r[j] = f32(x[base + k + j]); } }
  return r;
}
fn load4w(base: u32, k: u32, cin: u32) -> vec4<f32> {
  var r = vec4<f32>(0.0);
  for (var j = 0u; j < 4u; j++) { if (k + j < cin) { r[j] = f32(wt[base + k + j]); } }
  return r;
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
  for (var k0 = 0u; k0 < cin; k0 += 16u) {
    // Vector e of a tile is inputs 4 (e % 4) .. + 3 of its row e / 4.
    workgroupBarrier();
    for (var i = 0u; i < TR / 4u; i++) {
      let e = li + 256u * i;
      xs[(e / 4u) * SV + e % 4u] = load4x(min(r0 + e / 4u, n - 1u) * cin, k0 + (e % 4u) * 4u, cin);
    }
    for (var i = 0u; i < TC / 4u; i++) {
      let e = li + 256u * i;
      ws[(e / 4u) * SV + e % 4u] = load4w(min(c0 + e / 4u, cout - 1u) * cin, k0 + (e % 4u) * 4u, cin);
    }
    workgroupBarrier();
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

//@kernel k_mmh
// k_mm with half arithmetic (FP16 mode, cin a multiple of 8), without workgroup memory: a thread reads its
// R rows of x and C rows of W straight from the buffers, 8 values per 16-byte load (two vec4<f16> in a
// vec4<u32>), and accumulates f16 dot products of 8 inputs in f32. On the M5 Pro, staging the operands in
// workgroup memory gained nothing, and 8 halves per load instead of 4 gave 3.0 -> 3.7 T multiply-adds/s.
//@group 256
@group(0) @binding(1) var<storage, read> x: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wt: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read> residual: array<real>;
@group(0) @binding(5) var<storage, read_write> out: array<$O>;
const TR = $Ru;
const TC = $Cu;
const BR = 16u * TR;
const BC = 16u * TC;

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
  var xb: array<u32, TR>;
  var wb: array<u32, TC>;
  for (var r = 0u; r < TR; r++) { xb[r] = min(r0 + ty + 16u * r, n - 1u) * (cin / 8u); }
  for (var c = 0u; c < TC; c++) { wb[c] = min(c0 + tx + 16u * c, cout - 1u) * (cin / 8u); }
  for (var k0 = 0u; k0 < cin / 8u; k0 += 1u) {
    var b: array<vec4<u32>, TC>;
    for (var c = 0u; c < TC; c++) { b[c] = wt[wb[c] + k0]; }
    for (var r = 0u; r < TR; r++) {
      let a = x[xb[r] + k0];
      let a0 = bitcast<vec4<f16>>(a.xy);
      let a1 = bitcast<vec4<f16>>(a.zw);
      for (var c = 0u; c < TC; c++) { acc[r * TC + c] += f32(dot(a0, bitcast<vec4<f16>>(b[c].xy)) + dot(a1, bitcast<vec4<f16>>(b[c].zw))); }
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

//@kernel k_mmd
// k_mm without workgroup memory (cin a multiple of 4): a thread reads its R rows of x and C rows of W
// straight from the buffers, 4 values per load. Arithmetic in f32. The products of FP32 mode, and those of
// FP16 mode with float output (the read-out's logits) or cin not a multiple of 8.
//@group 256
@group(0) @binding(1) var<storage, read> x: array<vec4<real>>;
@group(0) @binding(2) var<storage, read> wt: array<vec4<real>>;
@group(0) @binding(3) var<storage, read> bias: array<real>;
@group(0) @binding(4) var<storage, read> residual: array<real>;
@group(0) @binding(5) var<storage, read_write> out: array<$O>;
const TR = $Ru;
const TC = $Cu;
const BR = 16u * TR;
const BC = 16u * TC;

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
  var xb: array<u32, TR>;
  var wb: array<u32, TC>;
  for (var r = 0u; r < TR; r++) { xb[r] = min(r0 + ty + 16u * r, n - 1u) * (cin / 4u); }
  for (var c = 0u; c < TC; c++) { wb[c] = min(c0 + tx + 16u * c, cout - 1u) * (cin / 4u); }
  for (var k0 = 0u; k0 < cin / 4u; k0 += 1u) {
    var b: array<vec4<f32>, TC>;
    for (var c = 0u; c < TC; c++) { b[c] = vec4<f32>(wt[wb[c] + k0]); }
    for (var r = 0u; r < TR; r++) {
      let a = vec4<f32>(x[xb[r] + k0]);
      for (var c = 0u; c < TC; c++) { acc[r * TC + c] += dot(a, b[c]); }
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
@group(0) @binding(1) var<storage, read> x: array<vec4<real>>;
@group(0) @binding(2) var<storage, read> taps: array<vec4<real>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<real>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<real>>;

// c is a multiple of 4, so 4 channels are one vector of the buffers.
fn ld4(a: u32) -> vec4<f32> { return vec4<f32>(x[a / 4u]); }
fn tap4(a: u32) -> vec4<f32> { return vec4<f32>(taps[a / 4u]); }

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
  let b = vec4<f32>(bias[q]);
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
    out[a / 4u] = vec4<real>(acc[o]);
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

//@kernel k_rotary_t
// Rotary encoding with the tables as tensors in the storage type (LightGlue: made per image). cosv, sinv:
// (n, pairs). Scalars: n, heads, pairs.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> cosv: array<real>;
@group(0) @binding(3) var<storage, read> sinv: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let pairs = pu(2u);
  let per_token = pu(1u) * pairs;
  if (gid >= pu(0u) * per_token) { return; }
  let tok = gid / per_token;
  let j = (gid % per_token) % pairs;
  let c = f32(cosv[tok * pairs + j]);
  let s = f32(sinv[tok * pairs + j]);
  let a = f32(x[2u * gid]);
  let b = f32(x[2u * gid + 1u]);
  out[2u * gid] = real(a * c - b * s);
  out[2u * gid + 1u] = real(b * c + a * s);
}

//@kernel k_gather_rows
// out[r] = x[rows[r]], whole rows of c values. Scalars: c, count.
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read> rows: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<real>;

fn run(gid: u32) {
  let c = pu(0u);
  if (gid >= c * pu(1u)) { return; }
  out[gid] = x[u32(rows[gid / c]) * c + gid % c];
}

//@kernel k_lse_rows
// LightGlue's match assignment (see core/ops.h and the Metal kernels of the same names): the log-sum-exp of
// every row of s (M, N) float. Scalars: M, N. Groups: M.
//@group 128
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> lse: array<f32>;
var<workgroup> part: array<f32, 128>;
var<workgroup> result: f32;

fn reduce(v: f32, li: u32, take_max: bool) -> f32 {
  part[li] = v;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) {
    if (li < k) { part[li] = select(part[li] + part[li + k], max(part[li], part[li + k]), take_max); }
    workgroupBarrier();
  }
  if (li == 0u) { result = part[0]; }
  return workgroupUniformLoad(&result);
}

fn group(g: u32, li: u32) {
  let M = pu(0u);
  let N = pu(1u);
  if (g >= M) { return; }
  let row = g * N;
  var top = kLowest;
  for (var j = li; j < N; j += 128u) { top = max(top, s[row + j]); }
  top = reduce(top, li, true);
  var sum = 0.0;
  for (var j = li; j < N; j += 128u) { sum += exp(s[row + j] - top); }
  sum = reduce(sum, li, false);
  if (li == 0u) { lse[g] = top + log(sum); }
}

//@kernel k_lse_cols
// The log-sum-exp of every column of s (M, N); each thread walks one column. Scalars: M, N.
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> lse: array<f32>;

fn run(gid: u32) {
  let M = pu(0u);
  let N = pu(1u);
  if (gid >= N) { return; }
  var top = kLowest;
  for (var i = 0u; i < M; i++) { top = max(top, s[i * N + gid]); }
  var sum = 0.0;
  for (var i = 0u; i < M; i++) { sum += exp(s[i * N + gid] - top); }
  lse[gid] = top + log(sum);
}

//@kernel k_best_rows
// The best column of every row: score(i, j) = ((s - row_lse_i) + (s - col_lse_j)) + (logsigmoid(za_i) +
// logsigmoid(zb_j)), ties to the lower index. best: f32 (indices are exact below 2^24). Scalars: M, N.
// Groups: M.
//@group 128
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> row_lse: array<f32>;
@group(0) @binding(3) var<storage, read> col_lse: array<f32>;
@group(0) @binding(4) var<storage, read> za: array<real>;
@group(0) @binding(5) var<storage, read> zb: array<real>;
@group(0) @binding(6) var<storage, read_write> best: array<f32>;
@group(0) @binding(7) var<storage, read_write> value: array<f32>;
var<workgroup> fpart: array<f32, 128>;
var<workgroup> ipart: array<u32, 128>;

fn log_sigmoid(z: f32) -> f32 { return min(z, 0.0) - log(1.0 + exp(-abs(z))); }

fn group(g: u32, li: u32) {
  let M = pu(0u);
  let N = pu(1u);
  if (g >= M) { return; }
  let row = g * N;
  let lr = row_lse[g];
  let l0 = log_sigmoid(f32(za[g]));
  var top = kLowest;
  var at = N;
  for (var j = li; j < N; j += 128u) {
    let v = ((s[row + j] - lr) + (s[row + j] - col_lse[j])) + (l0 + log_sigmoid(f32(zb[j])));
    if (v > top) { top = v; at = j; }
  }
  fpart[li] = top;
  ipart[li] = at;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) {
    if (li < k) {
      let o = li + k;
      if (fpart[o] > fpart[li] || (fpart[o] == fpart[li] && ipart[o] < ipart[li])) {
        fpart[li] = fpart[o];
        ipart[li] = ipart[o];
      }
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    best[g] = f32(ipart[0]);
    value[g] = fpart[0];
  }
}

//@kernel k_best_cols
// The best row of every column, as k_best_rows; each thread walks one column. Scalars: M, N.
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> row_lse: array<f32>;
@group(0) @binding(3) var<storage, read> col_lse: array<f32>;
@group(0) @binding(4) var<storage, read> za: array<real>;
@group(0) @binding(5) var<storage, read> zb: array<real>;
@group(0) @binding(6) var<storage, read_write> best: array<f32>;
@group(0) @binding(7) var<storage, read_write> value: array<f32>;

fn log_sigmoid(z: f32) -> f32 { return min(z, 0.0) - log(1.0 + exp(-abs(z))); }

fn run(gid: u32) {
  let M = pu(0u);
  let N = pu(1u);
  if (gid >= N) { return; }
  let lc = col_lse[gid];
  let l1 = log_sigmoid(f32(zb[gid]));
  var top = kLowest;
  var at = 0u;
  for (var i = 0u; i < M; i++) {
    let sij = s[i * N + gid];
    let v = ((sij - row_lse[i]) + (sij - lc)) + (log_sigmoid(f32(za[i])) + l1);
    if (v > top) { top = v; at = i; }
  }
  best[gid] = f32(at);
  value[gid] = top;
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

//@kernel k_tokens_t
// Values of attention, transposed and packed for k_attnh: out[(channel * S8 + s / 8)] holds channel
// `channel` of tokens s .. s + 7 as 8 f16 (zeros past S). Scalars: S, C (channels), S8 = ceil(S / 8).
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<u32>>;

fn run(gid: u32) {
  let S = pu(0u);
  let C = pu(1u);
  let S8 = pu(2u);
  if (gid >= C * S8) { return; }
  let ch = gid / S8;
  let s = (gid % S8) * 8u;
  var h: array<real, 8>;
  for (var j = 0u; j < 8u; j++) {
    h[j] = real(0.0);
    if (s + j < S) { h[j] = x[(s + j) * C + ch]; }
  }
  out[gid] = vec4<u32>(bitcast<vec2<u32>>(vec4<real>(h[0], h[1], h[2], h[3])), bitcast<vec2<u32>>(vec4<real>(h[4], h[5], h[6], h[7])));
}

//@kernel k_attnh
// k_attnd (attention, see there) with half arithmetic (FP16 mode). The logits read q and k straight from the buffers, 8 channels
// per 16-byte load; only the weights go through workgroup memory (vec4<f16> of 4 tokens, each written
// whole by the thread that holds them); the values come transposed (vt, from k_tokens_t), 8 tokens of
// one channel per load. Products and sums of 8 terms are f16, everything across them f32.
// Scalars: P, S, heads. Groups: heads * ceil(P / 64).
//@group 256
@group(0) @binding(1) var<storage, read> q: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> vt: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f16>>;
var<workgroup> sw: array<vec4<f16>, 1024>;  // [64 queries][16 token quads]: the weights of a tile
var<workgroup> red: array<vec4<f32>, 256>;  // [thread]: the maxima, then the sums, of its part of its 4 rows

fn h0(v: vec4<u32>) -> vec4<f16> { return bitcast<vec4<f16>>(v.xy); }
fn h1(v: vec4<u32>) -> vec4<f16> { return bitcast<vec4<f16>>(v.zw); }
fn dot8(a: vec4<u32>, b: vec4<u32>) -> f16 { return dot(h0(a), h0(b)) + dot(h1(a), h1(b)); }

fn group(g: u32, li: u32) {
  let Pq = pu(0u);
  let S = pu(1u);
  let heads = pu(2u);
  let C8 = heads * 8u;  // loads per token
  let S8 = (S + 7u) / 8u;
  let pblocks = (Pq + 63u) / 64u;
  if (g >= heads * pblocks) { return; }
  let hd = g / pblocks;
  let p0 = (g % pblocks) * 64u;
  let tx = li % 16u;
  let ty = li / 16u;
  // Rows past P and tokens past S are clamped to the last one; their results are not used.
  var qb: array<u32, 4>;
  for (var r = 0u; r < 4u; r++) { qb[r] = min(p0 + ty + 16u * r, Pq - 1u) * C8 + hd * 8u; }
  let vbase = (hd * 64u + 4u * tx) * S8;
  var o: array<vec4<f32>, 4>;
  var top = vec4<f32>(kLowest);
  var part = vec4<f32>(0.0);
  for (var r = 0u; r < 4u; r++) { o[r] = vec4<f32>(0.0); }
  for (var s0 = 0u; s0 < S; s0 += 64u) {
    var kb: array<u32, 4>;
    for (var c = 0u; c < 4u; c++) { kb[c] = min(s0 + 4u * tx + c, S - 1u) * C8 + hd * 8u; }
    var l: array<vec4<f32>, 4>;  // l[r][c]: query ty + 16 r, token s0 + 4 tx + c
    for (var r = 0u; r < 4u; r++) { l[r] = vec4<f32>(0.0); }
    for (var j = 0u; j < 8u; j++) {
      let b0 = k[kb[0] + j];
      let b1 = k[kb[1] + j];
      let b2 = k[kb[2] + j];
      let b3 = k[kb[3] + j];
      for (var r = 0u; r < 4u; r++) {
        let a = q[qb[r] + j];
        l[r] += vec4<f32>(vec4<f16>(dot8(a, b0), dot8(a, b1), dot8(a, b2), dot8(a, b3)));
      }
    }
    let valid = vec4<u32>(s0 + tx * 4u) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(S);
    for (var r = 0u; r < 4u; r++) { l[r] = select(vec4<f32>(kLowest), l[r] * 0.125, valid); }  // 1 / sqrt(64)
    // The 16 threads with the same ty hold the same 4 rows.
    red[li] = vec4<f32>(max(max(l[0].x, l[0].y), max(l[0].z, l[0].w)), max(max(l[1].x, l[1].y), max(l[1].z, l[1].w)),
                        max(max(l[2].x, l[2].y), max(l[2].z, l[2].w)), max(max(l[3].x, l[3].y), max(l[3].z, l[3].w)));
    workgroupBarrier();
    var grown = top;
    for (var t = 0u; t < 16u; t++) { grown = max(grown, red[ty * 16u + t]); }
    let shrink = exp(top - grown);  // 0 on the first tile, 1 for rows whose maximum did not grow
    top = grown;
    part *= shrink;
    for (var r = 0u; r < 4u; r++) {
      o[r] *= shrink[r];
      l[r] = select(vec4<f32>(0.0), exp(l[r] - vec4<f32>(top[r])), valid);  // the weights
      part[r] += (l[r].x + l[r].y) + (l[r].z + l[r].w);
      sw[(ty + 16u * r) * 16u + tx] = vec4<f16>(l[r]);
    }
    workgroupBarrier();
    for (var t = 0u; t < 8u; t++) {
      let at = s0 / 8u + t;
      if (at >= S8) { break; }
      let v0 = vt[vbase + at];
      let v1 = vt[vbase + S8 + at];
      let v2 = vt[vbase + 2u * S8 + at];
      let v3 = vt[vbase + 3u * S8 + at];
      for (var r = 0u; r < 4u; r++) {
        let wa = sw[(ty + 16u * r) * 16u + 2u * t];
        let wb = sw[(ty + 16u * r) * 16u + 2u * t + 1u];
        o[r] += vec4<f32>(vec4<f16>(dot(wa, h0(v0)) + dot(wb, h1(v0)), dot(wa, h0(v1)) + dot(wb, h1(v1)),
                                    dot(wa, h0(v2)) + dot(wb, h1(v2)), dot(wa, h0(v3)) + dot(wb, h1(v3))));
      }
    }
    workgroupBarrier();
  }
  // Row sums from the 16 partial sums, then the outputs.
  red[li] = part;
  workgroupBarrier();
  var sum = vec4<f32>(0.0);
  for (var t = 0u; t < 16u; t++) { sum += red[ty * 16u + t]; }
  for (var r = 0u; r < 4u; r++) {
    let p = p0 + ty + 16u * r;
    if (p < Pq) { out[p * heads * 16u + hd * 16u + tx] = vec4<f16>(o[r] / sum[r]); }
  }
}

//@kernel k_tokens_t4
// Values of attention, transposed for k_attnd: out[(channel * S4 + s / 4)] holds channel `channel` of
// tokens s .. s + 3 (zeros past S). Scalars: S, C (channels), S4 = ceil(S / 4).
@group(0) @binding(1) var<storage, read> x: array<real>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<real>>;

fn run(gid: u32) {
  let S = pu(0u);
  let C = pu(1u);
  let S4 = pu(2u);
  if (gid >= C * S4) { return; }
  let ch = gid / S4;
  let s = (gid % S4) * 4u;
  var h: array<real, 4>;
  for (var j = 0u; j < 4u; j++) {
    h[j] = real(0.0);
    if (s + j < S) { h[j] = x[(s + j) * C + ch]; }
  }
  out[gid] = vec4<real>(h[0], h[1], h[2], h[3]);
}

//@kernel k_attnd
// Attention with 64 channels per head, without a weights buffer: out = softmax(q k^T / 8) v per head
// (FP32; k_attnh is the FP16 form). One workgroup: 64 queries of one head; it walks the source tokens 64
// at a time. Thread (ty, tx) holds queries ty + 16 r (r < 4) and, per tile, the logits of tokens
// 4 tx .. + 3. Per tile: the logits; the largest logit of each query so far, from the 16 threads that
// share its row (when it grows, the outputs and partial sums so far are scaled to the new reference);
// then out += weights v. Each thread keeps the sum of its own weights; a row's 16 partial sums are added
// at the end. Operand tiles in workgroup memory gained nothing here: q and k are read straight from the
// buffers. The logits read q and k straight
// from the buffers; only the weights go through workgroup memory (a vec4 of 4 tokens, written whole by
// the thread that holds them); the values come transposed (vt, from k_tokens_t4), 4 tokens of one
// channel per load. The row maxima and sums share the weights' memory.
// Scalars: P, S, heads. Groups: heads * ceil(P / 64).
//@group 256
@group(0) @binding(1) var<storage, read> q: array<vec4<real>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<real>>;
@group(0) @binding(3) var<storage, read> vt: array<vec4<real>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<real>>;
var<workgroup> sw: array<vec4<f32>, 1024>;  // [64 queries][16 token quads]: the weights of a tile

fn group(g: u32, li: u32) {
  let Pq = pu(0u);
  let S = pu(1u);
  let heads = pu(2u);
  let C4 = heads * 16u;  // loads per token
  let S4 = (S + 3u) / 4u;
  let pblocks = (Pq + 63u) / 64u;
  if (g >= heads * pblocks) { return; }
  let hd = g / pblocks;
  let p0 = (g % pblocks) * 64u;
  let tx = li % 16u;
  let ty = li / 16u;
  var qb: array<u32, 4>;
  for (var r = 0u; r < 4u; r++) { qb[r] = min(p0 + ty + 16u * r, Pq - 1u) * C4 + hd * 16u; }
  let vbase = (hd * 64u + 4u * tx) * S4;
  var o: array<vec4<f32>, 4>;
  var top = vec4<f32>(kLowest);
  var part = vec4<f32>(0.0);
  for (var r = 0u; r < 4u; r++) { o[r] = vec4<f32>(0.0); }
  for (var s0 = 0u; s0 < S; s0 += 64u) {
    var kb: array<u32, 4>;
    for (var c = 0u; c < 4u; c++) { kb[c] = min(s0 + 4u * tx + c, S - 1u) * C4 + hd * 16u; }
    var l: array<vec4<f32>, 4>;  // l[r][c]: query ty + 16 r, token s0 + 4 tx + c
    for (var r = 0u; r < 4u; r++) { l[r] = vec4<f32>(0.0); }
    for (var j = 0u; j < 16u; j++) {
      let b0 = vec4<f32>(k[kb[0] + j]);
      let b1 = vec4<f32>(k[kb[1] + j]);
      let b2 = vec4<f32>(k[kb[2] + j]);
      let b3 = vec4<f32>(k[kb[3] + j]);
      for (var r = 0u; r < 4u; r++) {
        let a = vec4<f32>(q[qb[r] + j]);
        l[r] += vec4<f32>(dot(a, b0), dot(a, b1), dot(a, b2), dot(a, b3));
      }
    }
    let valid = vec4<u32>(s0 + tx * 4u) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(S);
    for (var r = 0u; r < 4u; r++) { l[r] = select(vec4<f32>(kLowest), l[r] * 0.125, valid); }  // 1 / sqrt(64)
    // The 16 threads with the same ty hold the same 4 rows.
    sw[li] = vec4<f32>(max(max(l[0].x, l[0].y), max(l[0].z, l[0].w)), max(max(l[1].x, l[1].y), max(l[1].z, l[1].w)),
                        max(max(l[2].x, l[2].y), max(l[2].z, l[2].w)), max(max(l[3].x, l[3].y), max(l[3].z, l[3].w)));
    workgroupBarrier();
    var grown = top;
    for (var t = 0u; t < 16u; t++) { grown = max(grown, sw[ty * 16u + t]); }
    workgroupBarrier();
    let shrink = exp(top - grown);  // 0 on the first tile, 1 for rows whose maximum did not grow
    top = grown;
    part *= shrink;
    for (var r = 0u; r < 4u; r++) {
      o[r] *= shrink[r];
      l[r] = select(vec4<f32>(0.0), exp(l[r] - vec4<f32>(top[r])), valid);  // the weights
      part[r] += (l[r].x + l[r].y) + (l[r].z + l[r].w);
      sw[(ty + 16u * r) * 16u + tx] = l[r];
    }
    workgroupBarrier();
    for (var t = 0u; t < 16u; t++) {
      let at = s0 / 4u + t;
      if (at >= S4) { break; }
      let v0 = vec4<f32>(vt[vbase + at]);
      let v1 = vec4<f32>(vt[vbase + S4 + at]);
      let v2 = vec4<f32>(vt[vbase + 2u * S4 + at]);
      let v3 = vec4<f32>(vt[vbase + 3u * S4 + at]);
      for (var r = 0u; r < 4u; r++) {
        let w = sw[(ty + 16u * r) * 16u + t];
        o[r] += vec4<f32>(dot(w, v0), dot(w, v1), dot(w, v2), dot(w, v3));
      }
    }
    workgroupBarrier();
  }
  // Row sums from the 16 partial sums, then the outputs.
  sw[li] = part;
  workgroupBarrier();
  var sum = vec4<f32>(0.0);
  for (var t = 0u; t < 16u; t++) { sum += sw[ty * 16u + t]; }
  for (var r = 0u; r < 4u; r++) {
    let p = p0 + ty + 16u * r;
    if (p < Pq) { out[p * heads * 16u + hd * 16u + tx] = vec4<real>(o[r] / sum[r]); }
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
