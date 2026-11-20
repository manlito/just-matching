// Metal backend kernels for the operations of core/ops.h. Tensors are token-major: element (token, channel)
// is at token * c + channel.
//
// The source is compiled at run time (newLibraryWithSource), so building needs no Metal toolchain.
// Scalar arguments are passed one per buffer index with setBytes.
//
// Precision: `real` is the storage type of tensors and weights: float, or half when compiled with JMT_FP16.
// In the kernels written with plain arithmetic, values are converted to float when read and all arithmetic
// and accumulation is float32 in both modes.
//
// With JMT_MPP (Metal 4, macOS 26) the matrix products and attention are built on Metal Performance
// Primitives instead: matmul2d runs a product cooperatively on the threads of a SIMD group, on whatever
// matrix hardware the GPU has. It is to Metal what the tensor-core instructions are to CUDA: a primitive
// used from our own kernels, not a library call. See the section at the end of this file.
#include <metal_stdlib>
#ifdef JMT_MPP
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
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

// Rotary encoding with the tables as tensors in the storage type (LightGlue: made per image). cos, sin:
// (n, pairs). Grid: n * heads * pairs.
kernel void k_rotary_t(device const real* x [[buffer(0)]], device const real* cosv [[buffer(1)]],
                       device const real* sinv [[buffer(2)]], constant int& n [[buffer(3)]], constant int& heads [[buffer(4)]],
                       constant int& pairs [[buffer(5)]], device real* out [[buffer(6)]],
                       uint gid [[thread_position_in_grid]]) {
  const int per_token = heads * pairs;
  if (int(gid) >= n * per_token) return;
  const int tok = int(gid) / per_token, j = (int(gid) % per_token) % pairs;
  const float c = float(cosv[tok * pairs + j]), s = float(sinv[tok * pairs + j]);
  const float a = float(x[2 * size_t(gid)]), b = float(x[2 * size_t(gid) + 1]);
  out[2 * size_t(gid)] = real(a * c - b * s);
  out[2 * size_t(gid) + 1] = real(b * c + a * s);
}

// out[r] = x[rows[r]], whole rows of c values. Grid: count * c.
kernel void k_gather_rows(device const real* x [[buffer(0)]], device const int* rows [[buffer(1)]], constant int& c [[buffer(2)]],
                          constant int& count [[buffer(3)]], device real* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(count) * uint(c)) return;
  const uint r = gid / uint(c), ch = gid % uint(c);
  out[gid] = x[size_t(rows[r]) * size_t(c) + ch];
}

// ---------------------------------------------------------------------------------------------
// LightGlue's match assignment (core/ops.h) on s (M, N) float, made by the read-out product:
//   score(i, j) = ((s_ij - row_lse_i) + (s_ij - col_lse_j)) + (logsigmoid(za_i) + logsigmoid(zb_j))
// in four passes: the log-sum-exp of every row and of every column, then the best of every row and of
// every column. The order of operations is the host's (core/assignment.h); ties go to the lower index.
// ---------------------------------------------------------------------------------------------
inline float log_sigmoid(float z) { return min(z, 0.0f) - precise::log(1.0f + precise::exp(-fabs(z))); }

// Threadgroups: M, 32 threads each.
kernel void k_lse_rows(device const float* s [[buffer(0)]], constant int& N [[buffer(1)]], device float* lse [[buffer(2)]],
                       uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  device const float* row = s + size_t(tg) * size_t(N);
  float top = -INFINITY;
  for (int j = lane; j < N; j += 32) top = max(top, row[j]);
  top = simd_max(top);
  float sum = 0;
  for (int j = lane; j < N; j += 32) sum += precise::exp(row[j] - top);
  sum = simd_sum(sum);
  if (lane == 0) lse[tg] = top + precise::log(sum);
}

// Grid: N; each thread walks one column down the M rows (neighbouring threads read neighbouring values).
kernel void k_lse_cols(device const float* s [[buffer(0)]], constant int& M [[buffer(1)]], constant int& N [[buffer(2)]],
                       device float* lse [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(N)) return;
  float top = -INFINITY;
  for (int i = 0; i < M; ++i) top = max(top, s[size_t(i) * size_t(N) + gid]);
  float sum = 0;
  for (int i = 0; i < M; ++i) sum += precise::exp(s[size_t(i) * size_t(N) + gid] - top);
  lse[gid] = top + precise::log(sum);
}

// Best column of every row. best: float (indices are exact below 2^24). Threadgroups: M, 32 threads each.
kernel void k_best_rows(device const float* s [[buffer(0)]], constant int& N [[buffer(1)]], device const float* row_lse [[buffer(2)]],
                        device const float* col_lse [[buffer(3)]], device const real* za [[buffer(4)]],
                        device const real* zb [[buffer(5)]], device float* best [[buffer(6)]], device float* value [[buffer(7)]],
                        uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  device const float* row = s + size_t(tg) * size_t(N);
  const float lr = row_lse[tg], l0 = log_sigmoid(float(za[tg]));
  float top = -INFINITY;
  int at = N;
  for (int j = lane; j < N; j += 32) {
    const float v = ((row[j] - lr) + (row[j] - col_lse[j])) + (l0 + log_sigmoid(float(zb[j])));
    if (v > top) top = v, at = j;
  }
  const float m = simd_max(top);
  const int arg = simd_min(top == m ? at : N);
  if (lane == 0) best[tg] = float(arg), value[tg] = m;
}

// Best row of every column. Grid: N.
kernel void k_best_cols(device const float* s [[buffer(0)]], constant int& M [[buffer(1)]], constant int& N [[buffer(2)]],
                        device const float* row_lse [[buffer(3)]], device const float* col_lse [[buffer(4)]],
                        device const real* za [[buffer(5)]], device const real* zb [[buffer(6)]], device float* best [[buffer(7)]],
                        device float* value [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(N)) return;
  const float lc = col_lse[gid], l1 = log_sigmoid(float(zb[gid]));
  float top = -INFINITY;
  int at = 0;
  for (int i = 0; i < M; ++i) {
    const float sij = s[size_t(i) * size_t(N) + gid];
    const float v = ((sij - row_lse[i]) + (sij - lc)) + (log_sigmoid(float(za[i])) + l1);
    if (v > top) top = v, at = i;
  }
  best[gid] = float(at), value[gid] = top;
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

// kt[(head * D + c) * spad + s] = k[s][head * D + c]; zeros for s >= S. Grid: heads * D * spad. D, the
// channels per head, is 64 for LightGlue and FlexGlue, 96 for LighterGlue (one head).
template <int D>
kernel void k_attn_keys(device const real* k [[buffer(0)]], constant int& S [[buffer(1)]], constant int& heads [[buffer(2)]],
                        constant int& spad [[buffer(3)]], device real* kt [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(heads) * uint(D) * uint(spad)) return;
  const int s = int(gid % uint(spad)), ch = int(gid / uint(spad));
  kt[gid] = s < S ? k[size_t(s) * size_t(heads) * D + size_t(ch)] : real(0);
}
template [[host_name("k_attn_keys")]] kernel void k_attn_keys<64>(device const real*, constant int&, constant int&,
                                                                   constant int&, device real*, uint);
template [[host_name("k_attn_keys_d96")]] kernel void k_attn_keys<96>(device const real*, constant int&, constant int&,
                                                                       constant int&, device real*, uint);

constant int kReduce = 32;  // threads that fold the per-thread values of a group

// One threadgroup: 4 queries of one head; one thread: 8 source tokens. Threadgroups: heads * ceil(P / 4),
// threads per group: spad / 8 (at most 1024). D channels per head; the logits are scaled by 1 / sqrt(D).
template <int D>
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
    qrow[r] = (uint(p0 + min(pb * 4 + r, P - 1)) * uint(heads) + uint(hd)) * uint(D / 4);
    acc[r][0] = float4(0.0f), acc[r][1] = float4(0.0f);
  }
  device const real4* q4 = (device const real4*)q;
  device const real4* wp = kt + (size_t(hd) * D * size_t(spad)) / 4 + size_t(sb) * 2;
  const uint wstep = uint(spad) / 4u;
  for (uint c = 0, wi = 0; c < uint(D / 4); ++c) {
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
  const float scale = D == 64 ? 0.125f : 1.0f / precise::sqrt(float(D));  // as the CPU backend
  for (int r = 0; r < 4; ++r) {
    acc[r][0] = select(float4(-INFINITY), acc[r][0] * scale, in0);
    acc[r][1] = select(float4(-INFINITY), acc[r][1] * scale, in1);
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

#define JMT_ATTN_WEIGHTS(NAME, D)                                                                           \
  template [[host_name(NAME)]] kernel void k_attn_weights<D>(device const real*, device const real4*, constant int&,  \
      constant int&, constant int&, constant int&, device wreal*, device float*, constant int&, uint, uint);
JMT_ATTN_WEIGHTS("k_attn_weights", 64)
JMT_ATTN_WEIGHTS("k_attn_weights_d96", 96)

// out (P, heads * D) = weights (heads, P, S) * v (S, heads * D) * inv, per head. One thread: R queries x T
// channels. Grid: heads * ceil(P / R) * (D / T), channel block fastest.
template <int R, int T, int D>
kernel void k_attn_apply(device const wreal* weights [[buffer(0)]], device const real4* v [[buffer(1)]],
                         constant int& P [[buffer(2)]], constant int& S [[buffer(3)]], constant int& heads [[buffer(4)]],
                         device const float* inv [[buffer(5)]], device real4* out [[buffer(6)]],
                         constant int& p0 [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  const int pblocks = (P + R - 1) / R, oblocks = D / T;
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
  device const real4* vp = v + size_t(hd) * (D / 4) + size_t(ob) * (T / 4);
  const uint vstep = uint(heads) * uint(D / 4);
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
    const size_t base = (size_t(p0 + p) * size_t(heads) + size_t(hd)) * (D / 4) + size_t(ob) * (T / 4);
    for (int j = 0; j < T / 4; ++j) out[base + j] = real4(acc[r][j] * scale);
  }
}
#define JMT_APPLY(NAME, R, T, D)                                                                           \
  template [[host_name(NAME)]] kernel void k_attn_apply<R, T, D>(                                           \
      device const wreal*, device const real4*, constant int&, constant int&, constant int&, device const float*, \
      device real4*, constant int&, uint);
JMT_APPLY("k_attn_apply_r4_t8", 4, 8, 64)
JMT_APPLY("k_attn_apply_r2_t16", 2, 16, 64)
JMT_APPLY("k_attn_apply_r4_t8_d96", 4, 8, 96)
JMT_APPLY("k_attn_apply_r2_t16_d96", 2, 16, 96)

#ifdef JMT_MPP
// ---------------------------------------------------------------------------------------------
// Kernels on Metal Performance Primitives.
//
// A tensor here is a view of a buffer with extents (columns, rows). matmul2d computes one tile of the
// output per threadgroup from slices of its operands; slices are clipped to the tensors, so tiles on the
// edge need no code of their own. A cooperative tensor is a tile held in the registers of the group's
// threads: the result can be changed there (bias, activation, residual, softmax) before it is stored.
//
// Measured on an M5 Pro, 4096 x 1024 x 1024: 10 to 15 T multiply-adds per second in half and 3.8 T in
// float, against 2.9 T and 2.3 T for k_linear.
// ---------------------------------------------------------------------------------------------
using namespace mpp::tensor_ops;
typedef dextents<int32_t, 2> ext2;

constant int kNone = 2;  // activation: kGelu, kRelu or none

inline float activate(float v, int act) {
  if (act == kGelu) return 0.5f * v * (1.0f + erf_approx(v * 0.70710678118654752440f));
  if (act == kRelu) return max(v, 0.0f);
  return v;
}

// Product: out (n, cout) = act(scale * x (n, cin) W^T + bias) + residual. w is stored (cout, cin), as in
// the weights file. flags: 1 = bias, 2 = residual (n, cout).
// Threadgroups: (ceil(cout / 64), ceil(n / 64)), each of 4 SIMD groups.
constant int kMmRows = 64;
constant int kMmCols = 64;
constant int kMmGroups = 4;

kernel void k_mm(device real* x [[buffer(0)]], device real* w [[buffer(1)]], device const real* bias [[buffer(2)]],
                 device const real* residual [[buffer(3)]], constant int& n [[buffer(4)]], constant int& cin [[buffer(5)]],
                 constant int& cout [[buffer(6)]], constant int& flags [[buffer(7)]], constant int& act [[buffer(8)]],
                 constant float& scale [[buffer(9)]], device real* out [[buffer(10)]],
                 uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(kMmRows, kMmCols, static_cast<int>(dynamic_extent), false, true);
  matmul2d<desc, execution_simdgroups<kMmGroups>> op;
  auto X = tensor<device real, ext2, tensor_inline>(x, ext2(cin, n));
  auto W = tensor<device real, ext2, tensor_inline>(w, ext2(cin, cout));
  auto C = tensor<device real, ext2, tensor_inline>(out, ext2(cout, n));
  const int r0 = int(tg.y) * kMmRows, c0 = int(tg.x) * kMmCols;
  auto tX = X.slice(0, r0);
  auto tW = W.slice(0, c0);
  auto tC = C.slice(c0, r0);
#ifndef JMT_FP16
  if (flags == 0 && act == kNone && scale == 1.0f) {
    op.run(tX, tW, tC);
    return;
  }
#endif
  // The tile is float in both modes, so the bias, the activation and the residual are applied to the
  // unrounded product and the result is rounded to the storage type once.
  auto tile = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  op.run(tX, tW, tile);
  for (auto it = tile.begin(); it != tile.end(); it++) {
    if (!tile.is_valid_element(it)) continue;
    const auto at = tile.get_multidimensional_index(it);
    const int col = c0 + at[0], row = r0 + at[1];
    if (col >= cout || row >= n) continue;
    float v = *it * scale;
    if (flags & 1) v += float(bias[col]);
    v = activate(v, act);
    const size_t o = size_t(row) * size_t(cout) + size_t(col);
    if (flags & 2) v += float(residual[o]);
    out[o] = real(v);
  }
}

// Attention with D channels per head (64; LighterGlue: 96), without the weights buffer: out = softmax(q k^T /
// sqrt(D)) v per head.
// One SIMD group takes 32 queries of one head and walks the source tokens 32 at a time, once. Per tile:
// the logits (a 32 x 32 x 64 product), the largest of each row so far, the weights exp(logit - largest)
// in registers, their row sums, and the product of the tile with the values, accumulated into 32 x 64
// outputs. When a row's largest logit grows, what the row has accumulated is scaled down to the new
// reference first, so the result is the same as with the true row maximum known in advance. The outputs
// are divided by the row sums at the end. Source tokens past S in the last tile are masked.
// Threadgroups: heads * ceil(P / 32), each of one SIMD group (32 threads).
constant int kAtRows = 32;
constant int kAtCols = 32;

template <int D>
kernel void k_attn(device real* q [[buffer(0)]], device real* k [[buffer(1)]], device real* v [[buffer(2)]],
                   constant int& P [[buffer(3)]], constant int& S [[buffer(4)]], constant int& heads [[buffer(5)]],
                   device real* out [[buffer(6)]], uint tg [[threadgroup_position_in_grid]]) {
  const int C = heads * D, pblocks = (P + kAtRows - 1) / kAtRows;
  const int pb = int(tg) % pblocks, hd = int(tg) / pblocks;
  constexpr auto logits_desc = matmul2d_descriptor(kAtRows, kAtCols, D, false, true);
  constexpr auto apply_desc =
      matmul2d_descriptor(kAtRows, D, kAtCols, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<logits_desc, execution_simdgroup> logits_op;
  matmul2d<apply_desc, execution_simdgroup> apply_op;
  auto Q = tensor<device real, ext2, tensor_inline>(q, ext2(C, P));
  auto K = tensor<device real, ext2, tensor_inline>(k, ext2(C, S));
  auto V = tensor<device real, ext2, tensor_inline>(v, ext2(C, S));
  auto tQ = Q.slice(hd * D, pb * kAtRows);
  typedef decltype(tQ) slice_t;

  // Logits and weights are tiles of `real`. Row sums and outputs are float in both modes: before the
  // division an output is a sum over all source tokens, which overflows half (4096 weights near 1 times a
  // value of 16), and half sums of many small terms lose most of their digits.
  auto tile = logits_op.template get_destination_cooperative_tensor<slice_t, slice_t, real>();
  auto top = logits_op.template get_row_reduction_destination_cooperative_tensor<slice_t, slice_t, real>();
  auto part = logits_op.template get_row_reduction_destination_cooperative_tensor<slice_t, slice_t, real>();
  auto sum = logits_op.template get_row_reduction_destination_cooperative_tensor<slice_t, slice_t, float>();
  auto shrink = logits_op.template get_row_reduction_destination_cooperative_tensor<slice_t, slice_t, float>();
  auto acc = apply_op.template get_destination_cooperative_tensor<decltype(tile), slice_t, float>();
  const real lowest = real(-1e30f);  // -inf in half
  const float scale = D == 64 ? 0.125f : 1.0f / precise::sqrt(float(D));  // as the CPU backend
  #pragma unroll
  for (ushort i = 0; i < top.get_capacity(); ++i)
    if (top.is_valid_element(i)) top[i] = lowest, sum[i] = 0.0f;
  #pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  for (int s = 0; s < S; s += kAtCols) {
    auto tK = K.slice(hd * D, s);
    auto tV = V.slice(hd * D, s);
    logits_op.run(tQ, tK, tile);
    const bool last = s + kAtCols > S;
    if (last)
      for (auto it = tile.begin(); it != tile.end(); it++)
        if (tile.is_valid_element(it) && s + tile.get_multidimensional_index(it)[0] >= S) *it = lowest;
    // New row maxima; rows whose maximum grew are scaled to it (shrink is 1 for the others, 0 at the start).
    reduce_rows(tile, part, reduction_operation::max, lowest);
    bool grew = false;
    #pragma unroll
    for (ushort i = 0; i < top.get_capacity(); ++i)
      if (top.is_valid_element(i)) {
        const real t = max(top[i], part[i]);
        shrink[i] = t > top[i] ? exp((float(top[i]) - float(t)) * scale) : 1.0f;
        grew = grew || t > top[i];
        sum[i] *= shrink[i];
        top[i] = t;
      }
    if (simd_any(grew))
      for (auto it = acc.begin(); it != acc.end(); it++) *it *= *shrink.map_iterator(it);
    for (auto it = tile.begin(); it != tile.end(); it++) *it = exp((*it - *top.map_iterator(it)) * real(scale));
    if (last)
      for (auto it = tile.begin(); it != tile.end(); it++)
        if (tile.is_valid_element(it) && s + tile.get_multidimensional_index(it)[0] >= S) *it = real(0);
    reduce_rows(tile, part, reduction_operation::sum, real(0));
    #pragma unroll
    for (ushort i = 0; i < sum.get_capacity(); ++i)
      if (sum.is_valid_element(i)) sum[i] += float(part[i]);
    auto weights = apply_op.template get_left_input_cooperative_tensor<real, real, float>(tile);
    apply_op.run(weights, tV, acc);
  }
  for (auto it = acc.begin(); it != acc.end(); it++) {
    if (!acc.is_valid_element(it)) continue;
    const auto at = acc.get_multidimensional_index(it);
    const int row = pb * kAtRows + at[1];
    if (row < P) out[size_t(row) * size_t(C) + size_t(hd * D + at[0])] = real(*it / *sum.map_iterator(it));
  }
}
template [[host_name("k_attn")]] kernel void k_attn<64>(device real*, device real*, device real*, constant int&,
                                                      constant int&, constant int&, device real*, uint);
template [[host_name("k_attn_d96")]] kernel void k_attn<96>(device real*, device real*, device real*, constant int&,
                                                          constant int&, constant int&, device real*, uint);

// LayerNorm over groups of g channels (g a multiple of 32, at most 1024), then an activation. One SIMD
// group (32 threads) per token and group: each thread holds g / 32 values in registers, and the mean and
// the variance are reduced across the group. Threadgroups: n * (c / g), 32 threads each.
kernel void k_layer_norm_sg(device const real* x [[buffer(0)]], device const real* gain [[buffer(1)]],
                            device const real* bias [[buffer(2)]], constant int& n [[buffer(3)]],
                            constant int& c [[buffer(4)]], constant int& g [[buffer(5)]], constant float& eps [[buffer(6)]],
                            constant int& act [[buffer(7)]], device real* out [[buffer(8)]],
                            uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int per = g / 32;
  const uint j0 = uint(lane) * uint(per);
  const size_t base = size_t(tg) * size_t(g) + size_t(j0);
  float vals[32];
  float s = 0;
  for (int j = 0; j < per; ++j) {
    vals[j] = float(x[base + j]);
    s += vals[j];
  }
  const float mean = simd_sum(s) / float(g);
  float sq = 0;
  for (int j = 0; j < per; ++j) {
    vals[j] -= mean;
    sq = fma(vals[j], vals[j], sq);
  }
  const float inv = 1.0f / precise::sqrt(simd_sum(sq) / float(g) + eps);
  for (int j = 0; j < per; ++j)
    out[base + j] = real(activate(vals[j] * inv * float(gain[j0 + j]) + float(bias[j0 + j]), act));
}

// Match read-out of one query from its logits over the h x w cells (see Ops::readout and core/readout.h,
// which this follows). One SIMD group (32 threads) per query; every thread scans a 32nd of the cells and
// the group reduces. Threadgroups: N, 32 threads each.
//
// The 1.5-entmax threshold tau solves sum over T of max(y - tau, 0)^2 = 1 with y = (logit - top) / 2 and
// T the 512 largest logits (the reference sorts them and uses a closed form per prefix; the root is the
// same). The sum is convex and decreasing in tau and is at least 1 at tau = -1, so Newton's method from
// -1 climbs to the root without passing it. Cells with y <= -1 never count; when more than 512 cells lie
// above -1, the 512th largest y is found by bisection and the cells below it are left out.
kernel void k_readout(device const float* logits [[buffer(0)]], constant int& h [[buffer(1)]], constant int& w [[buffer(2)]],
                      device float* coarse [[buffer(3)]], device float* mass [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int P = h * w, per = (P + 31) / 32;
  const int s0 = min(P, int(lane) * per), s1 = min(P, s0 + per);
  device const float* row = logits + size_t(tg) * size_t(P);
  // Largest logit and the first cell that has it.
  float best = -INFINITY;
  int at = P;
  for (int s = s0; s < s1; ++s) {
    const float v = row[s];
    if (v > best) best = v, at = s;
  }
  const float top = simd_max(best);
  const int arg = simd_min(best == top ? at : P);
  // Cells that can have weight: y > cut. `extra` copies of the value `cut` stand for the cells that tie
  // at the 512th place.
  float cut = -1.0f;
  int count = 0;
  for (int s = s0; s < s1; ++s) count += int((row[s] - top) * 0.5f > cut);
  count = simd_sum(count);
  int extra = 0;
  if (count > 512) {
    float lo = -1.0f, hi = 0.0f;  // count(y > lo) >= 512 > count(y > hi)
    for (int it = 0; it < 30; ++it) {
      const float mid = 0.5f * (lo + hi);
      int c = 0;
      for (int s = s0; s < s1; ++s) c += int((row[s] - top) * 0.5f > mid);
      if (simd_sum(c) >= 512) lo = mid; else hi = mid;
    }
    int c = 0;
    for (int s = s0; s < s1; ++s) c += int((row[s] - top) * 0.5f > hi);
    cut = hi, extra = 512 - simd_sum(c);
  }
  float tau = -1.0f;
  for (int it = 0; it < 40; ++it) {
    float a = 0, b = 0;
    for (int s = s0; s < s1; ++s) {
      const float y = (row[s] - top) * 0.5f;
      const float d = y > cut ? max(y - tau, 0.0f) : 0.0f;
      a += d, b = fma(d, d, b);
    }
    const float de = max(cut - tau, 0.0f) * float(extra);
    a = simd_sum(a) + de, b = simd_sum(b) + de * max(cut - tau, 0.0f);
    const float next = tau + (b - 1.0f) / (2.0f * a);
    if (!(next > tau)) break;
    tau = next;
  }
  if (lane != 0) return;
  // Expected cell-centre position in the 3 x 3 window around the best cell, and the window's mass.
  const int by = arg / w, bx = arg % w;
  float px = 0, py = 0, total = 0, inside = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int ny = by + dy, nx = bx + dx;
      const bool in = ny >= 0 && ny < h && nx >= 0 && nx < w;
      ny = clamp(ny, 0, h - 1), nx = clamp(nx, 0, w - 1);
      float p = max((row[ny * w + nx] - top) * 0.5f - tau, 0.0f);
      p *= p;
      px += p * ((float(nx) + 0.5f) / float(w) * 2.0f - 1.0f), py += p * ((float(ny) + 0.5f) / float(h) * 2.0f - 1.0f);
      total += p;
      if (in) inside += p;
    }
  coarse[2 * tg] = px / total, coarse[2 * tg + 1] = py / total;
  mass[tg] = inside;
}

// Network input from 8-bit pixels: bilinear resize to oh x ow of pixel / 255, then (v - mean) / std per
// channel. ch bytes per pixel (1: grey; 3 or 4: R, G, B first). The source rows and columns come from the
// host (taps: 3 floats per output row, then 3 per output column: first index, second index, weight of the
// second), so they are the same as on the other backends to the last bit. Grid: oh * ow.
kernel void k_preprocess(device const uchar* img [[buffer(0)]], device const float* taps [[buffer(1)]],
                         constant int& stride [[buffer(2)]], constant int& ch [[buffer(3)]], constant int& oh [[buffer(4)]],
                         constant int& ow [[buffer(5)]], constant float3& mean [[buffer(6)]], constant float3& stdev [[buffer(7)]],
                         device real* out [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  if (int(gid) >= oh * ow) return;
  device const float* ty3 = taps + 3 * (int(gid) / ow);
  device const float* tx3 = taps + 3 * (oh + int(gid) % ow);
  const int y0 = int(ty3[0]), y1 = int(ty3[1]), x0 = int(tx3[0]), x1 = int(tx3[1]);
  const float ty = ty3[2], tx = tx3[2];
  for (int c = 0; c < 3; ++c) {
    const int o = ch == 1 ? 0 : c;
    const float p00 = float(img[size_t(y0) * stride + size_t(x0 * ch + o)]) / 255.0f;
    const float p01 = float(img[size_t(y0) * stride + size_t(x1 * ch + o)]) / 255.0f;
    const float p10 = float(img[size_t(y1) * stride + size_t(x0 * ch + o)]) / 255.0f;
    const float p11 = float(img[size_t(y1) * stride + size_t(x1 * ch + o)]) / 255.0f;
    const float top = p00 * (1 - tx) + p01 * tx, bottom = p10 * (1 - tx) + p11 * tx;
    out[size_t(gid) * 3 + c] = real(((top * (1 - ty) + bottom * ty) - mean[c]) / stdev[c]);
  }
}
#endif  // JMT_MPP
