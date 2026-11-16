// Metal backend kernels for the operations of core/ops.h. Tensors are token-major: element (token, channel)
// is at token * c + channel.
//
// The source is compiled at run time (newLibraryWithSource), so building needs no Metal toolchain.
// Scalar arguments are passed one per buffer index with setBytes.
//
// Precision: `real` is the storage type of tensors and weights: float, or half when compiled with JMT_FP16.
// Values are converted to float when read, and all arithmetic and accumulation is float32 in both modes.
#include <metal_stdlib>
using namespace metal;

#ifdef JMT_FP16
typedef half real;
typedef packed_half4 real4;
#else
typedef float real;
typedef packed_float4 real4;
#endif

constant int kGelu = 0;
constant int kRelu = 1;

// erf by Abramowitz and Stegun 7.1.26: absolute error below 1.5e-7.
inline float erf_approx(float x) {
  const float a = fabs(x), t = 1.0f / (1.0f + 0.3275911f * a);
  const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float y = 1.0f - poly * precise::exp(-a * a);
  return x < 0 ? -y : y;
}

// ---------------------------------------------------------------------------------------------
// Matrix product: out (n, cout) = scale * x (n, cin) * W + bias.
// Weights are stored by output block: w[((o / T) * cin + i) * T + o % T], with cout padded to a multiple of
// T by zeros, so the weights a thread needs are one contiguous run of cin * T values. cin must be a multiple
// of 4. One thread computes an R x T block of the output: R tokens, T channels.
// Grid: ceil(n / R) * (cpad / T), channel block fastest.
// ---------------------------------------------------------------------------------------------
template <int R, int T, typename O>
kernel void k_linear(device const real* x [[buffer(0)]], device const real4* w [[buffer(1)]],
                     device const real* bias [[buffer(2)]], constant int& has_bias [[buffer(3)]],
                     constant int& n [[buffer(4)]], constant int& cin [[buffer(5)]], constant int& cout [[buffer(6)]],
                     constant float& scale [[buffer(7)]], device O* out [[buffer(8)]],
                     uint gid [[thread_position_in_grid]]) {
  const int blocks = (cout + T - 1) / T, rblocks = (n + R - 1) / R;
  if (int(gid) >= blocks * rblocks) return;
  const int rb = int(gid) / blocks, ob = int(gid) % blocks;
  // 32-bit offsets in the loop (64-bit integer arithmetic is slow on the GPU): in units of 4 values, within x.
  float4 acc[R][T / 4];
  uint row[R];
  for (int r = 0; r < R; ++r) {
    row[r] = uint(min(rb * R + r, n - 1)) * uint(cin / 4);
    for (int q = 0; q < T / 4; ++q) acc[r][q] = float4(0.0f);
  }
  // Four inputs per step: one vector load per token, then the T weights of each of the four inputs.
  device const real4* wp = w + size_t(ob) * size_t(cin) * (T / 4);
  device const real4* x4 = (device const real4*)x;
  for (uint i = 0, wi = 0; i < uint(cin / 4); ++i, wi += uint(T)) {
    float4 xv[R];
    for (int r = 0; r < R; ++r) xv[r] = float4(x4[row[r] + i]);
    for (int j = 0; j < 4; ++j) {
      float4 wv[T / 4];
      for (int q = 0; q < T / 4; ++q) wv[q] = float4(wp[wi + uint(j * (T / 4) + q)]);
      for (int r = 0; r < R; ++r) {
        const float4 v = float4(xv[r][j]);
        for (int q = 0; q < T / 4; ++q) acc[r][q] = fma(wv[q], v, acc[r][q]);
      }
    }
  }
  for (int r = 0; r < R; ++r) {
    const int tok = rb * R + r;
    if (tok >= n) break;
    for (int t = 0; t < T; ++t) {
      const int o = ob * T + t;
      if (o >= cout) break;
      float v = float(acc[r][t / 4][t % 4]) * scale;
      if (has_bias) v += float(bias[o]);
      out[size_t(tok) * size_t(cout) + size_t(o)] = O(v);
    }
  }
}

#define JMT_LINEAR(NAME, R, T, O)                                                                             \
  template [[host_name(NAME)]] kernel void k_linear<R, T, O>(                                                  \
      device const real*, device const real4*, device const real*, constant int&, constant int&, constant int&, \
      constant int&, constant float&, device O*, uint);
JMT_LINEAR("k_linear_r1_t8", 1, 8, real)
JMT_LINEAR("k_linear_r2_t8", 2, 8, real)
JMT_LINEAR("k_linear_r4_t8", 4, 8, real)
JMT_LINEAR("k_linear_r4_t16", 4, 16, real)
JMT_LINEAR("k_linear_r8_t8", 8, 8, real)
JMT_LINEAR("k_linear_r8_t4", 8, 4, real)
#ifdef JMT_FP16
JMT_LINEAR("k_linear_f32", 4, 8, float)  // float output: the logits of the read-out (FP32: k_linear_r4_t8)
#endif

// (rows, cols) -> weights of k_linear with output r and input c (see there for the layout); zeros for
// r >= rows. Turns a tensor into the weight matrix of a product. Grid: cols * rpad.
kernel void k_as_weights(device const real* x [[buffer(0)]], constant int& rows [[buffer(1)]],
                         constant int& cols [[buffer(2)]], constant int& rpad [[buffer(3)]],
                         constant int& tile [[buffer(4)]], device real* out [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= cols * rpad) return;
  const int t = int(gid) % tile, c = (int(gid) / tile) % cols, r = int(gid) / tile / cols * tile + t;
  out[gid] = r < rows ? x[size_t(r) * size_t(cols) + size_t(c)] : real(0);
}

// ---------------------------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------------------------

// Patches for a k x k convolution: out (oh * ow, k * k * c), ordered (ky, kx, channel); zeros outside.
// Grid: oh * ow * k * k * c.
kernel void k_patches(device const real* x [[buffer(0)]], constant int& h [[buffer(1)]], constant int& w [[buffer(2)]],
                      constant int& c [[buffer(3)]], constant int& k [[buffer(4)]], constant int& stride [[buffer(5)]],
                      constant int& pad [[buffer(6)]], constant int& oh [[buffer(7)]], constant int& ow [[buffer(8)]],
                      device real* out [[buffer(9)]], uint gid [[thread_position_in_grid]]) {
  const int cols = k * k * c;
  if (gid >= uint(oh * ow) * uint(cols)) return;
  const int tok = int(gid / uint(cols)), col = int(gid % uint(cols)), tap = col / c, ch = col % c;
  const int sy = (tok / ow) * stride - pad + tap / k, sx = (tok % ow) * stride - pad + tap % k;
  const bool inside = sy >= 0 && sy < h && sx >= 0 && sx < w;
  out[gid] = inside ? x[(size_t(sy) * size_t(w) + size_t(sx)) * size_t(c) + size_t(ch)] : real(0);
}

// Depthwise 7x7, padding 3. taps: (49, c). One thread per (token, 4 channels). Grid: h * w * c / 4.
kernel void k_dwconv7(device const real4* x [[buffer(0)]], device const real4* taps [[buffer(1)]],
                      device const real4* bias [[buffer(2)]], constant int& h [[buffer(3)]], constant int& w [[buffer(4)]],
                      constant int& c [[buffer(5)]], device real4* out [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
  const int c4 = c / 4;
  if (int(gid) >= h * w * c4) return;
  const int tok = int(gid) / c4, q = int(gid) % c4, y = tok / w, xo = tok % w;
  float4 acc = float4(bias[q]);
  for (int ky = 0; ky < 7; ++ky) {
    const int sy = y + ky - 3;
    if (sy < 0 || sy >= h) continue;
    for (int kx = 0; kx < 7; ++kx) {
      const int sx = xo + kx - 3;
      if (sx < 0 || sx >= w) continue;
      acc = fma(float4(taps[(ky * 7 + kx) * c4 + q]), float4(x[size_t(sy * w + sx) * size_t(c4) + size_t(q)]), acc);
    }
  }
  out[gid] = real4(acc);
}

// ---------------------------------------------------------------------------------------------
// Normalization and elementwise operations
// ---------------------------------------------------------------------------------------------

// LayerNorm over groups of g channels. Grid: n * (c / g).
kernel void k_layer_norm(device const real* x [[buffer(0)]], device const real* gain [[buffer(1)]],
                         device const real* bias [[buffer(2)]], constant int& n [[buffer(3)]], constant int& c [[buffer(4)]],
                         constant int& g [[buffer(5)]], constant float& eps [[buffer(6)]], device real* out [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
  const int groups = c / g;
  if (int(gid) >= n * groups) return;
  const size_t base = size_t(gid) * size_t(g);
  float mean = 0;
  for (int j = 0; j < g; ++j) mean += float(x[base + j]);
  mean /= float(g);
  float var = 0;
  for (int j = 0; j < g; ++j) {
    const float d = float(x[base + j]) - mean;
    var = fma(d, d, var);
  }
  const float inv = 1.0f / precise::sqrt(var / float(g) + eps);
  for (int j = 0; j < g; ++j) out[base + j] = real((float(x[base + j]) - mean) * inv * float(gain[j]) + float(bias[j]));
}

kernel void k_unary(device const real* x [[buffer(0)]], constant int& count [[buffer(1)]], constant int& op [[buffer(2)]],
                    device real* out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= count) return;
  const float v = float(x[gid]);
  out[gid] = real(op == kGelu ? 0.5f * v * (1.0f + erf_approx(v * 0.70710678118654752440f)) : max(v, 0.0f));
}

kernel void k_add(device const real* a [[buffer(0)]], device const real* b [[buffer(1)]], constant int& count [[buffer(2)]],
                  device real* out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= count) return;
  out[gid] = real(float(a[gid]) + float(b[gid]));
}

// Copies channels [first, first + count) of x (c channels) to channels [at, at + count) of out (oc channels).
// Grid: n * count. Serves slice (at = 0, oc = count) and both halves of concat.
kernel void k_copy_channels(device const real* x [[buffer(0)]], constant int& n [[buffer(1)]], constant int& c [[buffer(2)]],
                            constant int& first [[buffer(3)]], constant int& count [[buffer(4)]],
                            constant int& at [[buffer(5)]], constant int& oc [[buffer(6)]], device real* out [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(n) * uint(count)) return;
  const size_t tok = gid / uint(count), j = gid % uint(count);
  out[tok * size_t(oc) + size_t(at) + j] = x[tok * size_t(c) + size_t(first) + j];
}

// (h, w, c) -> (h / 2, w / 2, 4c); output channel = c * 4 + dy * 2 + dx. Grid: output elements.
kernel void k_pixel_unshuffle2(device const real* x [[buffer(0)]], constant int& h [[buffer(1)]],
                               constant int& w [[buffer(2)]], constant int& c [[buffer(3)]], device real* out [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
  const int oc = 4 * c, ow = w / 2;
  if (gid >= uint(h / 2) * uint(ow) * uint(oc)) return;
  const int tok = int(gid / uint(oc)), j = int(gid % uint(oc)), ch = j / 4, dy = (j % 4) / 2, dx = j % 2;
  const int sy = 2 * (tok / ow) + dy, sx = 2 * (tok % ow) + dx;
  out[gid] = x[(size_t(sy) * size_t(w) + size_t(sx)) * size_t(c) + size_t(ch)];
}

inline void resize_source(int i, int out, int in, thread int& i0, thread int& i1, thread float& t) {
  float s = (float(i) + 0.5f) * (float(in) / float(out)) - 0.5f;
  s = max(s, 0.0f);
  i0 = min(int(s), in - 1);
  i1 = min(i0 + 1, in - 1);
  t = s - float(i0);
}

// Bilinear resize (align_corners = false). Grid: oh * ow * c.
kernel void k_resize(device const real* x [[buffer(0)]], constant int& h [[buffer(1)]], constant int& w [[buffer(2)]],
                     constant int& c [[buffer(3)]], constant int& oh [[buffer(4)]], constant int& ow [[buffer(5)]],
                     device real* out [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(oh * ow) * uint(c)) return;
  const int tok = int(gid / uint(c)), ch = int(gid % uint(c));
  int y0, y1, x0, x1;
  float ty, tx;
  resize_source(tok / ow, oh, h, y0, y1, ty);
  resize_source(tok % ow, ow, w, x0, x1, tx);
  const float p00 = float(x[size_t(y0 * w + x0) * size_t(c) + ch]), p01 = float(x[size_t(y0 * w + x1) * size_t(c) + ch]);
  const float p10 = float(x[size_t(y1 * w + x0) * size_t(c) + ch]), p11 = float(x[size_t(y1 * w + x1) * size_t(c) + ch]);
  out[gid] = real((p00 * (1 - tx) + p01 * tx) * (1 - ty) + (p10 * (1 - tx) + p11 * tx) * ty);
}

// Per token: x / max(|x|, 1e-12). Grid: n.
kernel void k_l2_normalize(device const real* x [[buffer(0)]], constant int& n [[buffer(1)]], constant int& c [[buffer(2)]],
                           device real* out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= n) return;
  const size_t base = size_t(gid) * size_t(c);
  float sq = 0;
  for (int j = 0; j < c; ++j) {
    const float v = float(x[base + j]);
    sq = fma(v, v, sq);
  }
  const float inv = 1.0f / max(precise::sqrt(sq), 1e-12f);
  for (int j = 0; j < c; ++j) out[base + j] = real(float(x[base + j]) * inv);
}

// Rotary encoding. cs: (n, pairs) cos then (n, pairs) sin. Grid: n * heads * pairs.
kernel void k_rotary(device const real* x [[buffer(0)]], device const float* cosv [[buffer(1)]],
                     device const float* sinv [[buffer(2)]], constant int& n [[buffer(3)]], constant int& heads [[buffer(4)]],
                     constant int& pairs [[buffer(5)]], device real* out [[buffer(6)]],
                     uint gid [[thread_position_in_grid]]) {
  const int per_token = heads * pairs;
  if (int(gid) >= n * per_token) return;
  const int tok = int(gid) / per_token, j = (int(gid) % per_token) % pairs;
  const float c = cosv[tok * pairs + j], s = sinv[tok * pairs + j];
  const float a = float(x[2 * size_t(gid)]), b = float(x[2 * size_t(gid) + 1]);
  out[2 * size_t(gid)] = real(a * c - b * s);
  out[2 * size_t(gid) + 1] = real(b * c + a * s);
}

// Bilinear samples at points in [-1, 1], border clamped. Grid: n * c.
kernel void k_sample(device const real* x [[buffer(0)]], device const float* pts [[buffer(1)]], constant int& h [[buffer(2)]],
                     constant int& w [[buffer(3)]], constant int& c [[buffer(4)]], constant int& n [[buffer(5)]],
                     device real* out [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(n) * uint(c)) return;
  const int i = int(gid / uint(c)), ch = int(gid % uint(c));
  float fx = (pts[2 * i] + 1.0f) * 0.5f * float(w) - 0.5f, fy = (pts[2 * i + 1] + 1.0f) * 0.5f * float(h) - 0.5f;
  fx = clamp(fx, 0.0f, float(w - 1)), fy = clamp(fy, 0.0f, float(h - 1));
  const int x0 = int(fx), y0 = int(fy), x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  const float tx = fx - float(x0), ty = fy - float(y0);
  const float p00 = float(x[size_t(y0 * w + x0) * size_t(c) + ch]), p01 = float(x[size_t(y0 * w + x1) * size_t(c) + ch]);
  const float p10 = float(x[size_t(y1 * w + x0) * size_t(c) + ch]), p11 = float(x[size_t(y1 * w + x1) * size_t(c) + ch]);
  out[gid] = real(p00 * (1 - tx) * (1 - ty) + p01 * tx * (1 - ty) + p10 * (1 - tx) * ty + p11 * tx * ty);
}

// ---------------------------------------------------------------------------------------------
// Attention with 64 channels per head, in three passes:
//   keys      k (S, heads * 64) -> kt (heads * 64, Spad): the keys of a head as the weights of a product
//   weights   exp(q k^T / 8 - row maximum) -> buffer (heads, P, S), and 1 / row sum to a side buffer
//   apply     out = weights * v / row sum, tiled like k_linear
// The last two run on a chunk of queries at a time (P queries starting at p0), so the buffer can be small.
// The weights buffer is the large one (512 MB at 64 x 64 tokens in float) and memory traffic through it is
// what bounds attention, so it is written once and read once: the row maximum and the row sum are reduced
// inside a threadgroup instead of in passes of their own. Its element type `wreal` is float, or half in
// FP16 mode (weights are in [0, 1]).
// ---------------------------------------------------------------------------------------------
#ifdef JMT_FP16
typedef half wreal;
#else
typedef float wreal;
#endif

// kt[(head * 64 + c) * spad + s] = k[s][head * 64 + c]; zeros for s >= S. Grid: heads * 64 * spad.
kernel void k_attn_keys(device const real* k [[buffer(0)]], constant int& S [[buffer(1)]], constant int& heads [[buffer(2)]],
                        constant int& spad [[buffer(3)]], device real* kt [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(heads) * 64u * uint(spad)) return;
  const int s = int(gid % uint(spad)), ch = int(gid / uint(spad));
  kt[gid] = s < S ? k[size_t(s) * size_t(heads) * 64 + size_t(ch)] : real(0);
}

constant int kReduce = 32;  // threads that fold the per-thread values of a group

// One threadgroup: 4 queries of one head; one thread: 8 source tokens. Threadgroups: heads * ceil(P / 4),
// threads per group: spad / 8 (at most 1024).
kernel void k_attn_weights(device const real* q [[buffer(0)]], device const real4* kt [[buffer(1)]],
                           constant int& P [[buffer(2)]], constant int& S [[buffer(3)]], constant int& heads [[buffer(4)]],
                           constant int& spad [[buffer(5)]], device wreal* weights [[buffer(6)]],
                           device float* inv [[buffer(7)]], constant int& p0 [[buffer(8)]],
                           uint group [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
  threadgroup float4 part[1024];
  threadgroup float4 folded[kReduce];
  const int pblocks = (P + 3) / 4, sblocks = spad / 8, sb = int(lid);
  const int pb = int(group % uint(pblocks)), hd = int(group / uint(pblocks));
  const int folds = min(kReduce, sblocks);
  uint qrow[4];  // in units of 4 values
  float4 acc[4][2];
  for (int r = 0; r < 4; ++r) {
    qrow[r] = (uint(p0 + min(pb * 4 + r, P - 1)) * uint(heads) + uint(hd)) * 16u;
    acc[r][0] = float4(0.0f), acc[r][1] = float4(0.0f);
  }
  device const real4* q4 = (device const real4*)q;
  device const real4* wp = kt + (size_t(hd) * 64 * size_t(spad)) / 4 + size_t(sb) * 2;
  const uint wstep = uint(spad) / 4u;
  for (uint c = 0, wi = 0; c < 16u; ++c) {
    float4 qv[4];
    for (int r = 0; r < 4; ++r) qv[r] = float4(q4[qrow[r] + c]);
    for (int j = 0; j < 4; ++j, wi += wstep) {
      const float4 w0 = float4(wp[wi]), w1 = float4(wp[wi + 1u]);
      for (int r = 0; r < 4; ++r) {
        const float4 v = float4(qv[r][j]);
        acc[r][0] = fma(w0, v, acc[r][0]);
        acc[r][1] = fma(w1, v, acc[r][1]);
      }
    }
  }
  // Logits; source tokens past S (padding) get -inf. Then the maximum of each of the 4 rows over the group.
  const int4 s0 = int4(sb * 8) + int4(0, 1, 2, 3);
  const bool4 in0 = s0 < int4(S), in1 = (s0 + 4) < int4(S);
  float4 mine = float4(-INFINITY);
  for (int r = 0; r < 4; ++r) {
    acc[r][0] = select(float4(-INFINITY), acc[r][0] * 0.125f, in0);  // 1 / sqrt(64)
    acc[r][1] = select(float4(-INFINITY), acc[r][1] * 0.125f, in1);
    const float4 m = max(acc[r][0], acc[r][1]);
    mine[r] = max(max(m.x, m.y), max(m.z, m.w));
  }
  part[lid] = mine;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(lid) < folds) {
    float4 m = part[lid];
    for (int j = int(lid) + folds; j < sblocks; j += folds) m = max(m, part[j]);
    folded[lid] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 top = folded[0];
  for (int j = 1; j < folds; ++j) top = max(top, folded[j]);
  // exp(logit - maximum), stored; then the sum of each row over the group.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 sums = float4(0.0f);
  for (int r = 0; r < 4; ++r) {
    const float4 e0 = exp(acc[r][0] - top[r]), e1 = exp(acc[r][1] - top[r]);
    sums[r] = (e0.x + e0.y + e0.z + e0.w) + (e1.x + e1.y + e1.z + e1.w);
    const int p = pb * 4 + r;
    if (p >= P) continue;
    const size_t base = (size_t(hd) * size_t(P) + size_t(p)) * size_t(S) + size_t(sb) * 8;
    for (int t = 0; t < 4; ++t) {
      if (in0[t]) weights[base + size_t(t)] = wreal(e0[t]);
      if (in1[t]) weights[base + size_t(4 + t)] = wreal(e1[t]);
    }
  }
  part[lid] = sums;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(lid) < folds) {
    float4 m = part[lid];
    for (int j = int(lid) + folds; j < sblocks; j += folds) m += part[j];
    folded[lid] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float4 total = folded[0];
    for (int j = 1; j < folds; ++j) total += folded[j];
    for (int r = 0; r < 4; ++r)
      if (pb * 4 + r < P) inv[hd * P + pb * 4 + r] = 1.0f / total[r];
  }
}

// out (P, heads * 64) = weights (heads, P, S) * v (S, heads * 64) * inv, per head. One thread: R queries x T
// channels. Grid: heads * ceil(P / R) * (64 / T), channel block fastest.
template <int R, int T>
kernel void k_attn_apply(device const wreal* weights [[buffer(0)]], device const real4* v [[buffer(1)]],
                         constant int& P [[buffer(2)]], constant int& S [[buffer(3)]], constant int& heads [[buffer(4)]],
                         device const float* inv [[buffer(5)]], device real4* out [[buffer(6)]],
                         constant int& p0 [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  const int pblocks = (P + R - 1) / R, oblocks = 64 / T;
  if (gid >= uint(heads) * uint(pblocks) * uint(oblocks)) return;
  const int ob = int(gid % uint(oblocks)), pb = int((gid / uint(oblocks)) % uint(pblocks));
  const int hd = int(gid / uint(oblocks) / uint(pblocks));
  // 32-bit offsets in the loop: 64-bit integer arithmetic is slow on the GPU. The weights of one head are
  // P * S values, below 2^32 for any size the backend accepts.
  device const wreal* wh = weights + size_t(hd) * size_t(P) * size_t(S);
  uint row[R];
  float4 acc[R][T / 4];
  for (int r = 0; r < R; ++r) {
    row[r] = uint(min(pb * R + r, P - 1)) * uint(S);
    for (int j = 0; j < T / 4; ++j) acc[r][j] = float4(0.0f);
  }
  device const real4* vp = v + size_t(hd) * 16 + size_t(ob) * (T / 4);
  const uint vstep = uint(heads) * 16u;
  for (uint s = 0, vi = 0; s < uint(S); ++s, vi += vstep) {
    float4 vv[T / 4];
    for (int j = 0; j < T / 4; ++j) vv[j] = float4(vp[vi + uint(j)]);
    for (int r = 0; r < R; ++r) {
      const float4 a = float4(float(wh[row[r] + s]));
      for (int j = 0; j < T / 4; ++j) acc[r][j] = fma(vv[j], a, acc[r][j]);
    }
  }
  for (int r = 0; r < R; ++r) {
    const int p = pb * R + r;
    if (p >= P) break;
    const float scale = inv[hd * P + p];
    const size_t base = (size_t(p0 + p) * size_t(heads) + size_t(hd)) * 16 + size_t(ob) * (T / 4);
    for (int j = 0; j < T / 4; ++j) out[base + j] = real4(acc[r][j] * scale);
  }
}
#define JMT_APPLY(NAME, R, T)                                                                              \
  template [[host_name(NAME)]] kernel void k_attn_apply<R, T>(                                              \
      device const wreal*, device const real4*, constant int&, constant int&, constant int&, device const float*, \
      device real4*, constant int&, uint);
JMT_APPLY("k_attn_apply_r4_t8", 4, 8)
JMT_APPLY("k_attn_apply_r2_t16", 2, 16)
