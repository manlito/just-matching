// CUDA backend: hand-written kernels for the operations of core/ops.h. Tensors are token-major: element
// (token, channel) is at token * c + channel.
//
// Precision: every kernel is a template on the storage type R of tensors and weights, float or __half.
// Values are converted to float when read; arithmetic and accumulation are float32 in both modes. The
// logits of the read-out and of attention are float in both modes.
//
// Matrix products: one kernel, k_gemm, serves the linear layers, the convolutions (on patches) and the
// logits of the read-out. A block computes a tile of the output from tiles of both operands held in
// shared memory, so every operand value is read from device memory once per tile instead of once per
// thread. Operands are addressed with strides, so the search tokens of the read-out are used where they
// are, without a copy. Bias, activation and residual are applied to the tile before it is stored
// (Ops::linear_fused). FP16 mode runs the products on the tensor cores (k_gemm_tc, through the toolkit's
// warp matrix interface; no library), with float accumulation.
//
// Attention (k_flash) keeps logits, weights and the product with the values of a tile of queries in
// registers on the tensor cores and rescales when a row's maximum grows, so there is no logits buffer. In
// FP32 mode every operand is split into two halves (see there). JMT_CUDA_SPLIT=0 sends FP32 mode, and
// JMT_CUDA_FLASH=0 both modes, through two passes over a buffer of heads x P x S float logits (k_gemm /
// k_gemm_tc with kTops and kSoftmax).
//
// The read-out (k_readout: the 1.5-entmax threshold over the 512 largest logits) and the preprocessing of
// 8-bit images (k_preprocess) run on the device, so a pair needs no round trip before its results are read.
//
// Kernels run in the order they are launched (one stream), so a buffer returned to the pool can be
// handed out again at once. Uploads are queued on the stream too (cudaMemcpyAsync from pageable memory,
// staged by the driver), and tables the model keeps (rotary cos and sin) are uploaded once: queuing a
// pair never waits for the device. The host waits only when it reads a value.
//
// Environment: JMT_CUDA_PROFILE=1 times every launch with events and prints time per operation when the
// backend is destroyed. JMT_CUDA_GEMM picks how many columns the matrix product loads per step (k8, k16),
// for tuning. JMT_CUDA_TENSOR=0 keeps FP16 mode off the tensor cores. JMT_CUDA_FLASH: 0 as above, or the
// warps per attention block (4, 8, 16). JMT_CUDA_READOUT=0: the read-out's selection on host threads.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/half.h"
#include "core/ops.h"
#include "core/preprocess.h"
#include "core/readout.h"

namespace jmt {
namespace {

// ---------------------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------------------

enum : int { kNoAct = 0, kGelu = 1, kRelu = 2 };  // activations; a zeroed Gemm has none
constexpr int kBlock = 256;    // threads per block of the one-dimensional kernels

__device__ __forceinline__ float ld(float v) { return v; }
__device__ __forceinline__ float ld(__half v) { return __half2float(v); }
__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }

// T consecutive values (T a multiple of 4), 4 at a time where the type allows; p is aligned to 4 values.
template <int T>
__device__ __forceinline__ void st_row(float* p, const float* v) {
  #pragma unroll
  for (int q = 0; q < T / 4; ++q) reinterpret_cast<float4*>(p)[q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
}
template <int T>
__device__ __forceinline__ void st_row(__half* p, const float* v) {
  #pragma unroll
  for (int q = 0; q < T / 2; ++q) reinterpret_cast<__half2*>(p)[q] = __floats2half2_rn(v[2 * q], v[2 * q + 1]);
}

__device__ __forceinline__ float activate(float v, int act) {
  if (act == kGelu) return 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
  if (act == kRelu) return fmaxf(v, 0.0f);
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
  return v;
}
__device__ __forceinline__ int warp_sum(int v) {
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, s));
  return v;
}
__device__ __forceinline__ int warp_min(int v) {
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, s));
  return v;
}

__device__ __forceinline__ long long thread_index() {
  return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// 16 bytes of the storage type as floats (8 halves or 4 floats).
__device__ __forceinline__ void ld16(const __half* p, float* v) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __half2* h = reinterpret_cast<const __half2*>(&u);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __half22float2(h[i]);
    v[2 * i] = f.x, v[2 * i + 1] = f.y;
  }
}
__device__ __forceinline__ void ld16(const float* p, float* v) {
  const float4 f = *reinterpret_cast<const float4*>(p);
  v[0] = f.x, v[1] = f.y, v[2] = f.z, v[3] = f.w;
}
__device__ __forceinline__ void st16(__half* p, const float* v) {
  uint4 u;
  __half2* h = reinterpret_cast<__half2*>(&u);
  #pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void st16(float* p, const float* v) { *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]); }

// Sum of one value per thread of a 256-thread block. `red`: 256 shared floats.
__device__ __forceinline__ float block_sum(float* red, float v) {
  const int t = threadIdx.x;
  red[t] = v;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();
  return r;
}

// ---------------------------------------------------------------------------------------------
// Matrix product: out (n, cout) = scale * x (n, cin) * w (cin, cout) + bias, for `batches` independent
// products (grid z). Operands are addressed with strides:
//   x   (b, row, k)  at b * x_batch + row * x_row + k
//   w   (b, k, o)    at b * w_batch + k * w_in + o * w_out
//   out (b, row, o)  at b * o_batch + row * o_row + o
// A block of bx x by threads computes by * M rows x bx * M outputs; one thread an M x M part of it, in
// registers. Per step the block loads KC columns of x and KC rows of w into shared memory as floats.
// Grid: ceil(cout / (bx * M)) x ceil(n / (by * M)) x batches.
// ---------------------------------------------------------------------------------------------
struct Gemm {
  int n, cin, cout;
  long long x_row, x_batch, w_in, w_out, w_batch, o_row, o_batch;
  float scale;
  int vec;  // rows of the output start at multiples of 4 values: full tiles are stored 4 values at a time
  float* group_top;  // k_gemm_tc, kTops and kSoftmax: see there
  // kPlain only: out = act(product) + residual, residual laid out as out (same strides, storage type).
  const void* residual;
  int act;
};

// Row maxima are shared between blocks as integers that order like the floats they encode, so that
// atomicMax applies.
__device__ __forceinline__ int order_bits(float f) {
  const int i = __float_as_int(f);
  return i ^ ((i >> 31) & 0x7fffffff);
}
__device__ __forceinline__ float order_value(int i) { return __int_as_float(i ^ ((i >> 31) & 0x7fffffff)); }

// MODE selects the two halves of attention:
//   kPlain    the product as described
//   kTops     also folds the largest output of every row into tops[b * n + row] (encoded by order_bits)
//   kSoftmax  x holds logits: the product uses softmax(x) per row, computed while the tiles are loaded
//             (exp(x - tops) on load, the row sums collected on the way and divided out at the end)
enum : int { kPlain = 0, kTops = 1, kSoftmax = 2 };

template <typename X, typename W, typename O, int MR, int MT, int KC, int BX, int BY, int MODE>
__global__ void k_gemm(const X* x, const W* w, const float* bias, int* tops, Gemm a, O* out) {
  // A block is BX x BY threads; one thread computes MR rows x MT outputs. Everything below is constant.
  constexpr int ROWS = BY * MR, OUTS = BX * MT, THREADS = BX * BY;
  constexpr int JX = ROWS * KC / THREADS, JW = OUTS * KC / THREADS;  // values of each tile a thread loads per step
  constexpr int LK = KC == 8 ? 3 : KC == 16 ? 4 : 5;                 // log2(KC)
  static_assert(JX * THREADS == ROWS * KC && JW * THREADS == OUTS * KC, "tiles must divide among the threads");
  static_assert(BX % 8 == 0 && BY % 4 == 0 && MR % 4 == 0 && MT % 4 == 0, "a warp is 8 x 4 threads");
  // Shared tiles. A warp is 8 threads along the outputs by 4 along the rows, and the tiles are laid out
  // so that the 128-bit reads of a warp fall on distinct banks: value (step kk, row r) of x is at
  // [kk][(r % MR) / 4][r / MR][r % 4], and likewise for w with outputs and MT.
  // There are two copies of each tile, used by alternate steps: a thread can store the tiles of the next
  // step while others still compute on the current ones, so one barrier per step is enough.
  __shared__ float4 xs4[2 * KC * ROWS / 4];
  __shared__ float4 ws4[2 * KC * OUTS / 4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tx = (lane & 7) + 8 * (warp % (BX / 8)), ty = (lane >> 3) + 4 * (warp / (BX / 8));
  const int row0 = blockIdx.y * ROWS, o0 = blockIdx.x * OUTS;
  x += blockIdx.z * a.x_batch;
  w += blockIdx.z * a.w_batch;
  out += blockIdx.z * a.o_batch;
  tops += blockIdx.z * static_cast<long long>(a.n);
  const bool along_out = a.w_out == 1;  // which axis of w is contiguous in memory

  // The values this thread brings in per step: element e = tid + j * THREADS of each tile.
  //   x tile: row e / KC, column e % KC. w tile: (column, output) = (e / OUTS, e % OUTS) when the outputs
  //   are contiguous in memory, (e % KC, e / KC) otherwise.
  // They are read into registers one step ahead (xr, wr), so that the loads are in flight while the block
  // computes, and stored to shared memory at the start of the step that uses them. Where each value comes
  // from and goes to is worked out once, here: the loop only advances pointers. cin is a multiple of KC.
  const X* xp[JX];
  const W* wp[JW];
  int xi[JX], wi[JW];  // positions in the shared tiles
  bool wok[JW];        // false: an output past cout, read as zero
  #pragma unroll
  for (int j = 0; j < JX; ++j) {
    const int e = tid + j * THREADS, r = e >> LK, kk = e & (KC - 1);
    xp[j] = x + static_cast<long long>(min(row0 + r, a.n - 1)) * a.x_row + kk;
    xi[j] = kk * ROWS + ((r % MR) / 4) * (4 * BY) + (r / MR) * 4 + (r & 3);
  }
  #pragma unroll
  for (int j = 0; j < JW; ++j) {
    const int e = tid + j * THREADS;
    const int kk = along_out ? e / OUTS : e & (KC - 1), o = along_out ? e % OUTS : e >> LK;
    wok[j] = o0 + o < a.cout;
    wp[j] = w + kk * a.w_in + (wok[j] ? o0 + o : 0) * a.w_out;
    wi[j] = kk * OUTS + ((o % MT) / 4) * (4 * BX) + (o / MT) * 4 + (o & 3);
  }
  const long long wstep = KC * a.w_in;
  float xr[JX], wr[JW];
  auto fetch = [&]() {
    #pragma unroll
    for (int j = 0; j < JX; ++j) {
      xr[j] = ld(*xp[j]);
      xp[j] += KC;
    }
    #pragma unroll
    for (int j = 0; j < JW; ++j) {
      wr[j] = wok[j] ? ld(*wp[j]) : 0.0f;
      wp[j] += wstep;
    }
  };

  float acc[MR][MT];
  #pragma unroll
  for (int r = 0; r < MR; ++r)
    #pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.0f;
  // kSoftmax: the maxima and running sums of the rows this thread loads.
  float top[JX], part[JX];
  if (MODE == kSoftmax) {
    #pragma unroll
    for (int j = 0; j < JX; ++j) {
      top[j] = order_value(tops[min(row0 + ((tid + j * THREADS) >> LK), a.n - 1)]);
      part[j] = 0.0f;
    }
  }

  fetch();
  for (int k0 = 0; k0 < a.cin; k0 += KC) {
    const int odd = (k0 >> LK) & 1;
    float* xs = reinterpret_cast<float*>(xs4) + odd * (KC * ROWS);
    float* ws = reinterpret_cast<float*>(ws4) + odd * (KC * OUTS);
    #pragma unroll
    for (int j = 0; j < JX; ++j) {
      float v = xr[j];
      if (MODE == kSoftmax) {
        v = __expf(v - top[j]);
        part[j] += v;
      }
      xs[xi[j]] = v;
    }
    #pragma unroll
    for (int j = 0; j < JW; ++j) ws[wi[j]] = wr[j];
    __syncthreads();
    if (k0 + KC < a.cin) fetch();
    const float4* xt = xs4 + odd * (KC * ROWS / 4);
    const float4* wt = ws4 + odd * (KC * OUTS / 4);
    #pragma unroll
    for (int kk = 0; kk < KC; ++kk) {
      float xv[MR], wv[MT];
      #pragma unroll
      for (int q = 0; q < MR / 4; ++q) {
        const float4 xq = xt[kk * (ROWS / 4) + q * BY + ty];
        xv[4 * q] = xq.x, xv[4 * q + 1] = xq.y, xv[4 * q + 2] = xq.z, xv[4 * q + 3] = xq.w;
      }
      #pragma unroll
      for (int q = 0; q < MT / 4; ++q) {
        const float4 wq = wt[kk * (OUTS / 4) + q * BX + tx];
        wv[4 * q] = wq.x, wv[4 * q + 1] = wq.y, wv[4 * q + 2] = wq.z, wv[4 * q + 3] = wq.w;
      }
      #pragma unroll
      for (int r = 0; r < MR; ++r)
        #pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = fmaf(xv[r], wv[t], acc[r][t]);
    }
  }
  float* xs = reinterpret_cast<float*>(xs4);
  // kSoftmax: the KC partial sums of row r end up at xs[r * KC ...].
  if (MODE == kSoftmax) {
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < JX; ++j) xs[tid + j * THREADS] = part[j];
    __syncthreads();
  }
  float bv[MT];
  #pragma unroll
  for (int t = 0; t < MT; ++t) bv[t] = bias && o0 + tx * MT + t < a.cout ? bias[o0 + tx * MT + t] : 0.0f;
  const bool full = a.vec && row0 + ROWS <= a.n && o0 + OUTS <= a.cout;  // the same for the whole block
  const O* res = MODE == kPlain && a.residual ? static_cast<const O*>(a.residual) + blockIdx.z * a.o_batch : nullptr;
  #pragma unroll
  for (int r = 0; r < MR; ++r) {
    const long long row = row0 + ty * MR + r;
    float scale = a.scale;
    if (MODE == kSoftmax) {
      float sum = 0.0f;
      #pragma unroll
      for (int kk = 0; kk < KC; ++kk) sum += xs[((ty * MR + r) << LK) + kk];
      scale /= sum;
    }
    float v[MT], best = -INFINITY;
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
      v[t] = acc[r][t] * scale + bv[t];
      if (o0 + tx * MT + t < a.cout) best = fmaxf(best, v[t]);
    }
    if (MODE == kPlain) {
      #pragma unroll
      for (int t = 0; t < MT; ++t) v[t] = activate(v[t], a.act);
      if (res && row < a.n) {
        #pragma unroll
        for (int t = 0; t < MT; ++t)
          if (o0 + tx * MT + t < a.cout) v[t] += ld(res[row * a.o_row + o0 + tx * MT + t]);
      }
    }
    O* dst = out + row * a.o_row + o0 + tx * MT;
    if (full) {
      st_row<MT>(dst, v);
    } else if (row < a.n) {
      #pragma unroll
      for (int t = 0; t < MT; ++t)
        if (o0 + tx * MT + t < a.cout) st(dst + t, v[t]);
    }
    if (MODE == kTops) {
      // The 8 threads of a warp that share a row are adjacent lanes: fold their maxima, then one atomic
      // update per row and warp.
      #pragma unroll
      for (int s = 1; s < 8; s <<= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, s));
      if ((lane & 7) == 0 && row < a.n) atomicMax(tops + row, order_bits(best));
    }
  }
}

// ---------------------------------------------------------------------------------------------
// The same product on the tensor cores (FP16 mode): half operands, float accumulation, through the
// toolkit's warp matrix interface (16 x 16 x 16 products). Same arguments and modes as k_gemm, with
// these limits: cin a multiple of KC; the rows of x start at multiples of 4 values; w contiguous along
// its columns (w_in == 1, outputs at multiples of 4 values) or along its outputs (w_out == 1, columns at
// multiples of 4 values, cout a multiple of 4).
// The logits of attention are stored as half without losing what matters: kTops writes every logit
// relative to the largest of its group (its row, 32 consecutive outputs), which goes to
// group_top[(b * n + row) * ceil(cout / 32) + output / 32] as float. Logits near the top of their group,
// the ones that carry weight, are then small numbers, where half is fine. kSoftmax reads x that way
// (group_top indexed by its columns) and sends the exponentials to the tensor cores as half.
// A block is 8 warps, WR along the rows by WC along the outputs; a warp computes 64 rows x 32 outputs
// (8 accumulators). Per step the block loads KC columns of x and of w into shared memory, 4 values per
// load, one step ahead as in k_gemm. Both tiles are stored with rows of KC + 8 values: with rows of KC the
// reads of a warp would fall on the same few banks of shared memory and run one after another. Grid: ceil(cout / (WC * 32)) x ceil(n / (WR * 64)) x batches.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long ld_quad(const __half* p) { return *reinterpret_cast<const unsigned long long*>(p); }

template <typename X, typename O, int KC, int WR, int WC, int MODE>
__global__ void k_gemm_tc(const X* x, const __half* w, const float* bias, int* tops, Gemm a, O* out) {
  using namespace nvcuda;
  constexpr int ROWS = WR * 64, OUTS = WC * 32, THREADS = 256, KQ = KC / 4;
  constexpr int LD = KC + 8, LQ = LD / 4;  // row length of the shared tiles, in values and in loads
  constexpr int JX = ROWS * KQ / THREADS, JW = OUTS * KQ / THREADS;  // loads of 4 values per thread and step
  constexpr bool kLogits = MODE == kSoftmax;
  static_assert(WR * WC == 8 && KC % 16 == 0 && KC <= 32, "8 warps per block; a load stays within a group");
  static_assert(sizeof(X) == sizeof(__half) && (MODE != kTops || sizeof(O) == sizeof(__half)), "half operands");
  __shared__ unsigned long long xs8[ROWS * LQ];  // x tile as half, (row, column) at row * LD + column
  __shared__ unsigned long long ws8[OUTS * LQ];  // w tile as half, (column, output) at output * LD + column
  __shared__ float scratch[8][256];              // one accumulator of each warp, on its way out
  __shared__ int rowtop[ROWS];
  __half* xh = reinterpret_cast<__half*>(xs8);
  __half* wh = reinterpret_cast<__half*>(ws8);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wr = warp / WC, wc = warp % WC;
  const int row0 = blockIdx.y * ROWS, o0 = blockIdx.x * OUTS;
  x += blockIdx.z * a.x_batch;
  w += blockIdx.z * a.w_batch;
  out += blockIdx.z * a.o_batch;
  tops += blockIdx.z * static_cast<long long>(a.n);
  const bool along_out = a.w_out == 1;  // w is contiguous along its outputs instead of its columns
  const int groups = ((MODE == kSoftmax ? a.cin : a.cout) + 31) / 32;
  float* group_top = MODE == kPlain ? nullptr : a.group_top + blockIdx.z * static_cast<long long>(a.n) * groups;

  // Load e = tid + j * THREADS of each tile: row (x) or output (w) e / KQ, columns 4 * (e % KQ) ... + 3;
  // for w that is contiguous along its outputs, see below.
  const X* xp[JX];
  const __half* wp[JW];
  bool wok[JW];
  #pragma unroll
  for (int j = 0; j < JX; ++j) {
    const int e = tid + j * THREADS;
    xp[j] = x + static_cast<long long>(min(row0 + e / KQ, a.n - 1)) * a.x_row + (e % KQ) * 4;
  }
  #pragma unroll
  for (int j = 0; j < JW; ++j) {
    const int e = tid + j * THREADS;
    if (along_out) {  // column e / (OUTS / 4), outputs 4 * (e % (OUTS / 4)) ... + 3
      const int o = o0 + (e % (OUTS / 4)) * 4;
      wok[j] = o < a.cout;
      wp[j] = w + (e / (OUTS / 4)) * a.w_in + (wok[j] ? o : 0);
    } else {
      const int o = o0 + e / KQ;
      wok[j] = o < a.cout;
      wp[j] = w + (wok[j] ? o : 0) * a.w_out + (e % KQ) * 4;
    }
  }
  const long long wstep = along_out ? KC * a.w_in : KC;
  unsigned long long xr[JX], wr4[JW];
  float top[JX], part[JX], shift[JX];  // kSoftmax: row maximum, running row sum, group maximum - row maximum
  const float* gp[JX];
  if (MODE == kSoftmax) {
    #pragma unroll
    for (int j = 0; j < JX; ++j) {
      const long long row = min(row0 + (tid + j * THREADS) / KQ, a.n - 1);
      top[j] = order_value(tops[row]);
      part[j] = 0.0f;
      gp[j] = group_top + row * groups;
    }
  }
  auto fetch = [&](int k0) {
    #pragma unroll
    for (int j = 0; j < JX; ++j) {
      xr[j] = ld_quad(xp[j]);
      xp[j] += KC;
      if (MODE == kSoftmax) shift[j] = gp[j][(k0 + ((tid + j * THREADS) % KQ) * 4) >> 5] - top[j];
    }
    #pragma unroll
    for (int j = 0; j < JW; ++j) {
      wr4[j] = wok[j] ? ld_quad(wp[j]) : 0ull;
      wp[j] += wstep;
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
  #pragma unroll
  for (int fr = 0; fr < 4; ++fr)
    #pragma unroll
    for (int fc = 0; fc < 2; ++fc) wmma::fill_fragment(acc[fr][fc], 0.0f);

  fetch(0);
  for (int k0 = 0; k0 < a.cin; k0 += KC) {
    #pragma unroll
    for (int j = 0; j < JX; ++j) {
      const int e = tid + j * THREADS;
      if constexpr (kLogits) {
        const __half2* h = reinterpret_cast<const __half2*>(&xr[j]);
        const float2 la = __half22float2(h[0]), lb = __half22float2(h[1]);
        const float e0 = __expf(la.x + shift[j]), e1 = __expf(la.y + shift[j]), e2 = __expf(lb.x + shift[j]), e3 = __expf(lb.y + shift[j]);
        part[j] += (e0 + e1) + (e2 + e3);
        __half2* dst = reinterpret_cast<__half2*>(xs8 + (e / KQ) * LQ + e % KQ);
        dst[0] = __floats2half2_rn(e0, e1);
        dst[1] = __floats2half2_rn(e2, e3);
      } else {
        xs8[(e / KQ) * LQ + e % KQ] = xr[j];
      }
    }
    #pragma unroll
    for (int j = 0; j < JW; ++j) {
      const int e = tid + j * THREADS;
      if (along_out) {
        const __half* q = reinterpret_cast<const __half*>(&wr4[j]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) wh[((e % (OUTS / 4)) * 4 + i) * LD + e / (OUTS / 4)] = q[i];
      } else {
        ws8[(e / KQ) * LQ + e % KQ] = wr4[j];
      }
    }
    __syncthreads();
    if (k0 + KC < a.cin) fetch(k0 + KC);
    #pragma unroll
    for (int kk = 0; kk < KC; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
      #pragma unroll
      for (int fr = 0; fr < 4; ++fr) wmma::load_matrix_sync(af[fr], xh + (wr * 64 + fr * 16) * LD + kk, LD);
      #pragma unroll
      for (int fc = 0; fc < 2; ++fc) wmma::load_matrix_sync(bf[fc], wh + (wc * 32 + fc * 16) * LD + kk, LD);
      #pragma unroll
      for (int fr = 0; fr < 4; ++fr)
        #pragma unroll
        for (int fc = 0; fc < 2; ++fc) wmma::mma_sync(acc[fr][fc], af[fr], bf[fc], acc[fr][fc]);
    }
    __syncthreads();
  }

  // Out: the accumulators go through shared memory one at a time. Of each (16 x 16 values) a thread takes
  // column lane % 16 of 8 rows, values lane + 32 t: the reads of a warp then fall on distinct banks, and the
  // 16 lanes that share a row write 16 consecutive outputs.
  // kSoftmax: the KQ partial sums of row r are first gathered at psum[r * KQ ...], then 1 / sum at rinv[r].
  float* psum = reinterpret_cast<float*>(xs8);
  float* rinv = reinterpret_cast<float*>(ws8);
  static_assert(sizeof(xs8) >= ROWS * KQ * sizeof(float) && sizeof(ws8) >= ROWS * sizeof(float), "scratch space");
  if (MODE == kSoftmax) {
    #pragma unroll
    for (int j = 0; j < JX; ++j) psum[tid + j * THREADS] = part[j];
    __syncthreads();
    for (int r = tid; r < ROWS; r += THREADS) {
      float sum = 0.0f;
      #pragma unroll
      for (int kq = 0; kq < KQ; ++kq) sum += psum[r * KQ + kq];
      rinv[r] = 1.0f / sum;
    }
  }
  if (MODE == kTops)
    for (int i = tid; i < ROWS; i += THREADS) rowtop[i] = INT_MIN;
  __syncthreads();
  const int lc = lane & 15, lr = lane >> 4;
  const O* res = MODE == kPlain && a.residual ? static_cast<const O*>(a.residual) + blockIdx.z * a.o_batch : nullptr;
  #pragma unroll
  for (int fr = 0; fr < 4; ++fr) {
    const int r0 = wr * 64 + fr * 16 + lr;  // this thread's rows: r0 + 2 t
    float v[2][8];
    #pragma unroll
    for (int fc = 0; fc < 2; ++fc) {
      wmma::store_matrix_sync(scratch[warp], acc[fr][fc], 16, wmma::mem_row_major);
      __syncwarp();
      const int o = o0 + wc * 32 + fc * 16 + lc;
      const float bo = bias && o < a.cout ? bias[o] : 0.0f;
      #pragma unroll
      for (int t = 0; t < 8; ++t) {
        const float scale = MODE == kSoftmax ? a.scale * rinv[r0 + 2 * t] : a.scale;
        v[fc][t] = scratch[warp][lane + 32 * t] * scale + bo;
      }
      __syncwarp();
    }
    const int oa = o0 + wc * 32 + lc, ob = oa + 16;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int r = r0 + 2 * t;
      const long long row = row0 + r;
      if (MODE == kTops) {
        // The 16 lanes that share a row hold its group of 32 outputs between them.
        float best = fmaxf(oa < a.cout ? v[0][t] : -INFINITY, ob < a.cout ? v[1][t] : -INFINITY);
        #pragma unroll
        for (int sh = 1; sh < 16; sh <<= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, sh));
        if (lc == 0 && row < a.n && oa < a.cout) {
          atomicMax(rowtop + r, order_bits(best));
          group_top[row * groups + (oa >> 5)] = best;
        }
        v[0][t] -= best, v[1][t] -= best;
      }
      if (row < a.n) {
        if (MODE == kPlain) {
          v[0][t] = activate(v[0][t], a.act), v[1][t] = activate(v[1][t], a.act);
          if (res && oa < a.cout) v[0][t] += ld(res[row * a.o_row + oa]);
          if (res && ob < a.cout) v[1][t] += ld(res[row * a.o_row + ob]);
        }
        if (oa < a.cout) st(out + row * a.o_row + oa, v[0][t]);
        if (ob < a.cout) st(out + row * a.o_row + ob, v[1][t]);
      }
    }
  }
  if (MODE == kTops) {
    __syncthreads();
    for (int i = tid; i < ROWS; i += THREADS)
      if (row0 + i < a.n) atomicMax(tops + row0 + i, rowtop[i]);
  }
}

// ---------------------------------------------------------------------------------------------
// Attention without the logits buffer (FP16 mode, 64 channels per head): out = softmax(q k^T / 8) v.
// A warp takes 16 queries of one head and walks the source tokens 64 at a time. Per step: the logits
// (16 x 64, k = 64), the largest logit of each row so far, the weights exp(logit - largest) and the
// product of the weights with the values (16 x 64, k = 64) all stay in registers; when a row's largest
// logit grows, what the row has accumulated is scaled down to the new reference first, so the result is
// that of the softmax with the row maximum known in advance. Outputs are divided by the row sums at the
// end. Logits, row sums and outputs are float; the weights go to the second product as half.
//
// The products use the tensor-core instruction mma.sync m16n8k8 (sm_75) directly, whose register layout
// is documented: the accumulator of the logits (row g and g + 8 of the warp's 16, two columns per thread)
// is laid out like the left operand of the second product, so the weights never leave the registers.
// Keys and values go through shared memory (rows padded to 72 values, so that the 8 rows an ldmatrix
// reads fall on distinct banks), two copies used by alternate steps; the next step's tiles are read into
// registers while the current one is computed. Source tokens past S are zero and masked.
// Block: W warps (16 W queries of one head). Grid: ceil(P / (16 W)) x heads.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma16816(float* c, unsigned a0, unsigned a1, unsigned b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ void ldsm_x4(unsigned* r, const __half* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const __half* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ unsigned pack_half2(float a, float b) {
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const unsigned*>(&h);
}

// For float tensors (FP32 mode) the same kernel runs with every operand split in two halves, x = hi + lo
// (hi = half(x), lo = half(x - hi)): a product is hi hi + hi lo + lo hi, accumulated in float, which keeps
// about 22 bits of each term instead of 24. Keys and values then take 4 tiles of shared memory, so a step
// covers 32 source tokens instead of 64. The tensor cores of this GPU do not round their float
// accumulation as float additions do: an output summed over 4096 tokens inside the instruction was off by
// 10 times the error of float (a confident point moved 4 px on one test pair). So each step's products go
// to a fresh accumulator, which is added to the outputs with float additions; that brings the error back
// to that of the float path (splitting queries and keys in three halves instead of two did not).
__device__ __forceinline__ void split_half2(float a, float b, unsigned* hi, unsigned* lo) {
  const __half2 h = __floats2half2_rn(a, b);
  const float2 f = __half22float2(h);
  const __half2 l = __floats2half2_rn(a - f.x, b - f.y);
  *hi = *reinterpret_cast<const unsigned*>(&h), *lo = *reinterpret_cast<const unsigned*>(&l);
}
template <typename T, int W, int MINB>
__global__ void __launch_bounds__(W * 32, MINB) k_flash(const T* q, const T* k, const T* v, int P, int S, int q_row,
                                                        int k_row, int v_row, int o_row, T* out) {
  constexpr bool kSplit = std::is_same<T, float>::value;
  constexpr int TERMS = kSplit ? 2 : 1;  // half tiles per operand: hi (and lo)
  constexpr int NT = kSplit ? 4 : 8;     // token tiles of 8 per step
  constexpr int KT = NT * 8, LD = 72, THREADS = W * 32, VL = 16 / sizeof(T);  // VL: values per 16-byte load
  constexpr int LOADS = KT * (64 / VL) / THREADS;                             // 16-byte loads per tile and thread
  static_assert(LOADS * THREADS == KT * (64 / VL), "a tile divides among the threads");
  __shared__ __align__(16) __half ks[2][TERMS][KT * LD];
  __shared__ __align__(16) __half vs[2][TERMS][KT * LD];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
  const int hd = blockIdx.y, row0 = blockIdx.x * (16 * W) + warp * 16;
  q += hd * 64, k += hd * 64, v += hd * 64, out += hd * 64;

  // The warp's queries as left operands: qa[t][kk] holds dims kk * 8 + 2 tig (+1) of rows g and g + 8.
  unsigned qa[TERMS][8][2];
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    const T* src = q + static_cast<long long>(min(row0 + g + 8 * r, P - 1)) * q_row + 2 * tig;
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      if constexpr (kSplit) {
        const float2 f = *reinterpret_cast<const float2*>(src + kk * 8);
        split_half2(f.x, f.y, &qa[0][kk][r], &qa[TERMS - 1][kk][r]);
      } else {
        qa[0][kk][r] = *reinterpret_cast<const unsigned*>(src + kk * 8);
      }
    }
  }
  // Tile loads: element e = tid + j * THREADS is token e / (64 / VL), dims VL (e % (64 / VL)) ... + VL - 1.
  uint4 kr[LOADS], vr[LOADS];
  auto fetch = [&](int s0) {
    #pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      const int e = tid + j * THREADS, t = s0 + e / (64 / VL), c = (e % (64 / VL)) * VL;
      const bool ok = t < S;
      kr[j] = ok ? *reinterpret_cast<const uint4*>(k + static_cast<long long>(t) * k_row + c) : make_uint4(0, 0, 0, 0);
      vr[j] = ok ? *reinterpret_cast<const uint4*>(v + static_cast<long long>(t) * v_row + c) : make_uint4(0, 0, 0, 0);
    }
  };
  auto stash = [&](int buf) {
    #pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      const int e = tid + j * THREADS, at = (e / (64 / VL)) * LD + (e % (64 / VL)) * VL;
      if constexpr (kSplit) {
        const float* kf = reinterpret_cast<const float*>(&kr[j]);
        const float* vf = reinterpret_cast<const float*>(&vr[j]);
        uint2 kh, kl, vh, vl;
        split_half2(kf[0], kf[1], &kh.x, &kl.x), split_half2(kf[2], kf[3], &kh.y, &kl.y);
        split_half2(vf[0], vf[1], &vh.x, &vl.x), split_half2(vf[2], vf[3], &vh.y, &vl.y);
        *reinterpret_cast<uint2*>(&ks[buf][0][at]) = kh, *reinterpret_cast<uint2*>(&ks[buf][TERMS - 1][at]) = kl;
        *reinterpret_cast<uint2*>(&vs[buf][0][at]) = vh, *reinterpret_cast<uint2*>(&vs[buf][TERMS - 1][at]) = vl;
      } else {
        *reinterpret_cast<uint4*>(&ks[buf][0][at]) = kr[j];
        *reinterpret_cast<uint4*>(&vs[buf][0][at]) = vr[j];
      }
    }
  };

  float o[8][4];
  #pragma unroll
  for (int dt = 0; dt < 8; ++dt) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
  float top[2] = {-INFINITY, -INFINITY}, sum[2] = {0.0f, 0.0f};  // rows g and g + 8; sum: this thread's columns
  constexpr float kScale = 0.125f * 1.4426950408889634f;              // 1 / sqrt(64), and exp via exp2
  // ldmatrix: lane l gives the address of row l % 8 of matrix l / 8.
  const int lrow = lane & 7, lmat = lane >> 3;

  fetch(0);
  stash(0);
  __syncthreads();
  const int steps = (S + KT - 1) / KT;
  for (int step = 0; step < steps; ++step) {
    const int buf = step & 1, s0 = step * KT;
    if (step + 1 < steps) fetch(s0 + KT);
    // Logits: c[nt] covers tokens nt * 8 ... + 7.
    float c[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.0f;
      unsigned kb[TERMS][8];
      #pragma unroll
      for (int t = 0; t < TERMS; ++t) {
        ldsm_x4(kb[t], ks[buf][t] + (nt * 8 + lrow) * LD + lmat * 8);
        ldsm_x4(kb[t] + 4, ks[buf][t] + (nt * 8 + lrow) * LD + 32 + lmat * 8);
      }
      #pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        if constexpr (kSplit) {  // the small terms first
          mma16816(c[nt], qa[1][kk][0], qa[1][kk][1], kb[0][kk]);
          mma16816(c[nt], qa[0][kk][0], qa[0][kk][1], kb[1][kk]);
        }
        mma16816(c[nt], qa[0][kk][0], qa[0][kk][1], kb[0][kk]);
      }
    }
    if (s0 + KT > S) {
      #pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          if (s0 + nt * 8 + 2 * tig + (i & 1) >= S) c[nt][i] = -INFINITY;
    }
    // Row maxima over the step (the 4 threads of a row hold its logits), and the rescaling.
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      float m = -INFINITY;
      #pragma unroll
      for (int nt = 0; nt < NT; ++nt) m = fmaxf(m, fmaxf(c[nt][2 * r], c[nt][2 * r + 1]));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
      const float t = fmaxf(top[r], m);
      if (t > top[r]) {
        const float shrink = exp2f((top[r] - t) * kScale);  // 0 on the first step
        sum[r] *= shrink;
        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) o[dt][2 * r] *= shrink, o[dt][2 * r + 1] *= shrink;
        top[r] = t;
      }
    }
    // Weights, their sums, and the product with the values: the weights of tokens nt * 8 ... + 7 are the
    // left operand of k-slice nt. Split mode (float) adds the step's products in a fresh accumulator and
    // that to the outputs with float additions (see the comment above the kernel).
    float ps[kSplit ? 8 : 1][4];
    if constexpr (kSplit) {
      #pragma unroll
      for (int dt = 0; dt < 8; ++dt) ps[dt][0] = ps[dt][1] = ps[dt][2] = ps[dt][3] = 0.0f;
    }
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float p0 = exp2f((c[nt][0] - top[0]) * kScale), p1 = exp2f((c[nt][1] - top[0]) * kScale);
      const float p2 = exp2f((c[nt][2] - top[1]) * kScale), p3 = exp2f((c[nt][3] - top[1]) * kScale);
      sum[0] += p0 + p1, sum[1] += p2 + p3;
      unsigned a[TERMS][2];
      if constexpr (kSplit) {
        split_half2(p0, p1, &a[0][0], &a[1][0]), split_half2(p2, p3, &a[0][1], &a[1][1]);
      } else {
        a[0][0] = pack_half2(p0, p1), a[0][1] = pack_half2(p2, p3);
      }
      unsigned vb[TERMS][8];
      #pragma unroll
      for (int t = 0; t < TERMS; ++t) {
        ldsm_x4_t(vb[t], vs[buf][t] + (nt * 8 + lrow) * LD + lmat * 8);
        ldsm_x4_t(vb[t] + 4, vs[buf][t] + (nt * 8 + lrow) * LD + 32 + lmat * 8);
      }
      #pragma unroll
      for (int dt = 0; dt < 8; ++dt) {
        if constexpr (kSplit) {
          mma16816(ps[dt], a[1][0], a[1][1], vb[0][dt]);
          mma16816(ps[dt], a[0][0], a[0][1], vb[1][dt]);
          mma16816(ps[dt], a[0][0], a[0][1], vb[0][dt]);
        } else {
          mma16816(o[dt], a[0][0], a[0][1], vb[0][dt]);
        }
      }
    }
    if constexpr (kSplit) {
      #pragma unroll
      for (int dt = 0; dt < 8; ++dt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o[dt][i] += ps[dt][i];
    }
    if (step + 1 < steps) stash(buf ^ 1);
    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 1);
    sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 2);
    const int row = row0 + g + 8 * r;
    if (row >= P) continue;
    const float inv = 1.0f / sum[r];
    T* dst = out + static_cast<long long>(row) * o_row + 2 * tig;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      if constexpr (kSplit) *reinterpret_cast<float2*>(dst + dt * 8) = make_float2(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
      else *reinterpret_cast<unsigned*>(dst + dt * 8) = pack_half2(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
    }
  }
}

// The same product for outputs too few to fill a tile: one thread per output value. w: (cin, cout).
// Grid: n * cout.
template <typename R, typename O>
__global__ void k_linear_small(const R* x, const R* w, const float* bias, int n, int cin, int cout, float scale,
                               int act, const O* residual, O* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(n) * cout) return;
  const long long tok = i / cout;
  const int o = static_cast<int>(i % cout);
  const R* xr = x + tok * cin;
  float acc = 0.0f;
  for (int k = 0; k < cin; ++k) acc = fmaf(ld(xr[k]), ld(w[static_cast<long long>(k) * cout + o]), acc);
  float v = activate(acc * scale + (bias ? bias[o] : 0.0f), act);
  if (residual) v += ld(residual[i]);
  st(out + i, v);
}

// ---------------------------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------------------------

// Patches for a k x k convolution: out (oh * ow, k * k * c), ordered (ky, kx, channel); zeros outside.
template <typename R>
__global__ void k_patches(const R* x, int h, int w, int c, int k, int stride, int pad, int oh, int ow, R* out) {
  const long long i = thread_index();
  const int cols = k * k * c;
  if (i >= static_cast<long long>(oh) * ow * cols) return;
  const int tok = static_cast<int>(i / cols), col = static_cast<int>(i % cols), tap = col / c, ch = col % c;
  const int sy = (tok / ow) * stride - pad + tap / k, sx = (tok % ow) * stride - pad + tap % k;
  const bool inside = sy >= 0 && sy < h && sx >= 0 && sx < w;
  if (inside) out[i] = x[(static_cast<long long>(sy) * w + sx) * c + ch];
  else st(out + i, 0.0f);
}

// Depthwise 7x7, padding 3. taps: (49, c). One thread per value. Grid: h * w * c.
template <typename R>
__global__ void k_dwconv7(const R* x, const R* taps, const R* bias, int h, int w, int c, R* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(h) * w * c) return;
  const int tok = static_cast<int>(i / c), ch = static_cast<int>(i % c), y = tok / w, xo = tok % w;
  float acc = ld(bias[ch]);
  for (int ky = 0; ky < 7; ++ky) {
    const int sy = y + ky - 3;
    if (sy < 0 || sy >= h) continue;
    for (int kx = 0; kx < 7; ++kx) {
      const int sx = xo + kx - 3;
      if (sx < 0 || sx >= w) continue;
      acc = fmaf(ld(taps[(ky * 7 + kx) * c + ch]), ld(x[(static_cast<long long>(sy) * w + sx) * c + ch]), acc);
    }
  }
  st(out + i, acc);
}

// The same with 16 bytes of channels (V values) and TX neighbouring outputs of a row per thread: every
// input loaded serves the taps of all TX outputs. The sum runs over the taps in the same order as in
// k_dwconv7. c a multiple of V. Grid: h * ceil(w / TX) * (c / V), channels fastest.
template <typename R, int TX>
__global__ void k_dwconv7_vec(const R* x, const R* taps, const R* bias, int h, int w, int c, R* out) {
  constexpr int V = 16 / sizeof(R);
  const int cv = c / V, wx = (w + TX - 1) / TX;
  const long long i = thread_index();
  if (i >= static_cast<long long>(h) * wx * cv) return;
  const int ch = static_cast<int>(i % cv) * V;
  const int t = static_cast<int>(i / cv), y = t / wx, x0 = (t % wx) * TX;
  float acc[TX][V], b[V];
  ld16(bias + ch, b);
  #pragma unroll
  for (int o = 0; o < TX; ++o)
    #pragma unroll
    for (int j = 0; j < V; ++j) acc[o][j] = b[j];
  for (int ky = 0; ky < 7; ++ky) {
    const int sy = y + ky - 3;
    if (sy < 0 || sy >= h) continue;
    float tp[7][V];
    #pragma unroll
    for (int kx = 0; kx < 7; ++kx) ld16(taps + (ky * 7 + kx) * c + ch, tp[kx]);
    #pragma unroll
    for (int j = 0; j < TX + 6; ++j) {
      const int sx = x0 + j - 3;
      if (sx < 0 || sx >= w) continue;
      float in[V];
      ld16(x + (static_cast<long long>(sy) * w + sx) * c + ch, in);
      #pragma unroll
      for (int kx = 0; kx < 7; ++kx) {
        const int o = j - kx;
        if (o < 0 || o >= TX) continue;
        #pragma unroll
        for (int q = 0; q < V; ++q) acc[o][q] = fmaf(tp[kx][q], in[q], acc[o][q]);
      }
    }
  }
  #pragma unroll
  for (int o = 0; o < TX; ++o)
    if (x0 + o < w) st16(out + (static_cast<long long>(y) * w + x0 + o) * c + ch, acc[o]);
}

// ---------------------------------------------------------------------------------------------
// Normalization and elementwise operations
// ---------------------------------------------------------------------------------------------

// LayerNorm over groups of g channels, then an activation. One thread per (token, group). Grid: n * (c / g).
template <typename R>
__global__ void k_layer_norm(const R* x, const R* gain, const R* bias, int n, int c, int g, float eps, int act, R* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(n) * (c / g)) return;
  const R* xr = x + i * g;
  float mean = 0;
  for (int j = 0; j < g; ++j) mean += ld(xr[j]);
  mean /= static_cast<float>(g);
  float var = 0;
  for (int j = 0; j < g; ++j) {
    const float d = ld(xr[j]) - mean;
    var = fmaf(d, d, var);
  }
  const float inv = 1.0f / sqrtf(var / static_cast<float>(g) + eps);
  for (int j = 0; j < g; ++j) st(out + i * g + j, activate((ld(xr[j]) - mean) * inv * ld(gain[j]) + ld(bias[j]), act));
}

// The same with 16-byte loads, for g = NC * gl * V (V values in 16 bytes, gl lanes per group, a power of two
// up to 32): lane `sub` of a group holds chunks sub, sub + gl, ... of V channels in registers; mean and
// variance are reduced across the group's lanes. A warp holds 32 / gl groups.
// Blocks of 256 threads. Grid: ceil(groups / (8 * 32 / gl)).
template <typename R, int NC>
__global__ void k_layer_norm_vec(const R* x, const R* gain, const R* bias, long long groups, int g, int gl, float eps, int act,
                                 R* out) {
  constexpr int V = 16 / sizeof(R);
  const int lane = threadIdx.x & 31, sub = lane & (gl - 1);
  const long long grp = (static_cast<long long>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * (32 / gl) + lane / gl;
  const bool ok = grp < groups;
  const R* xr = x + (ok ? grp : 0) * g;
  float v[NC][V];
  float s = 0;
  #pragma unroll
  for (int c = 0; c < NC; ++c) {
    ld16(xr + (c * gl + sub) * V, v[c]);
    #pragma unroll
    for (int i = 0; i < V; ++i) s += v[c][i];
  }
  for (int o = gl / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / static_cast<float>(g);
  float sq = 0;
  #pragma unroll
  for (int c = 0; c < NC; ++c)
    #pragma unroll
    for (int i = 0; i < V; ++i) v[c][i] -= mean, sq = fmaf(v[c][i], v[c][i], sq);
  for (int o = gl / 2; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
  const float inv = 1.0f / sqrtf(sq / static_cast<float>(g) + eps);
  if (!ok) return;
  #pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int j = (c * gl + sub) * V;
    float gv[V], bv[V];
    ld16(gain + j, gv), ld16(bias + j, bv);
    #pragma unroll
    for (int i = 0; i < V; ++i) v[c][i] = activate(v[c][i] * inv * gv[i] + bv[i], act);
    st16(out + grp * g + j, v[c]);
  }
}

// LayerNorm for wide groups: one block per (token, group), the sums reduced over the block.
// Grid: n * (c / g) blocks.
template <typename R>
__global__ void k_layer_norm_wide(const R* x, const R* gain, const R* bias, int g, float eps, int act, R* out) {
  __shared__ float red[kBlock];
  const long long base = static_cast<long long>(blockIdx.x) * g;
  float sum = 0;
  for (int j = threadIdx.x; j < g; j += kBlock) sum += ld(x[base + j]);
  const float mean = block_sum(red, sum) / static_cast<float>(g);
  float var = 0;
  for (int j = threadIdx.x; j < g; j += kBlock) {
    const float d = ld(x[base + j]) - mean;
    var = fmaf(d, d, var);
  }
  const float inv = 1.0f / sqrtf(block_sum(red, var) / static_cast<float>(g) + eps);
  for (int j = threadIdx.x; j < g; j += kBlock)
    st(out + base + j, activate((ld(x[base + j]) - mean) * inv * ld(gain[j]) + ld(bias[j]), act));
}

template <typename R>
__global__ void k_unary(const R* x, long long count, int op, R* out) {
  const long long i = thread_index();
  if (i >= count) return;
  const float v = ld(x[i]);
  st(out + i, activate(v, op));
}

template <typename R>
__global__ void k_add(const R* a, const R* b, long long count, R* out) {
  const long long i = thread_index();
  if (i >= count) return;
  st(out + i, ld(a[i]) + ld(b[i]));
}

// Copies channels [first, first + count) of x (c channels) to channels [at, at + count) of out (oc channels).
// Grid: n * count. Serves slice (at = 0, oc = count) and both halves of concat.
template <typename R>
__global__ void k_copy_channels(const R* x, int n, int c, int first, int count, int at, int oc, R* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(n) * count) return;
  const long long tok = i / count, j = i % count;
  out[tok * oc + at + j] = x[tok * c + first + j];
}

// (h, w, c) -> (h / 2, w / 2, 4c); output channel = c * 4 + dy * 2 + dx. Grid: output elements.
template <typename R>
__global__ void k_pixel_unshuffle2(const R* x, int h, int w, int c, R* out) {
  const long long i = thread_index();
  const int oc = 4 * c, ow = w / 2;
  if (i >= static_cast<long long>(h / 2) * ow * oc) return;
  const int tok = static_cast<int>(i / oc), j = static_cast<int>(i % oc), ch = j / 4, dy = (j % 4) / 2, dx = j % 2;
  const int sy = 2 * (tok / ow) + dy, sx = 2 * (tok % ow) + dx;
  out[i] = x[(static_cast<long long>(sy) * w + sx) * c + ch];
}

__device__ __forceinline__ void resize_src(int i, int out, int in, int* i0, int* i1, float* t) {
  float s = (static_cast<float>(i) + 0.5f) * (static_cast<float>(in) / static_cast<float>(out)) - 0.5f;
  s = fmaxf(s, 0.0f);
  *i0 = min(static_cast<int>(s), in - 1);
  *i1 = min(*i0 + 1, in - 1);
  *t = s - static_cast<float>(*i0);
}

// Bilinear resize (align_corners = false). Grid: oh * ow * c.
template <typename R>
__global__ void k_resize(const R* x, int h, int w, int c, int oh, int ow, R* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(oh) * ow * c) return;
  const int tok = static_cast<int>(i / c), ch = static_cast<int>(i % c);
  int y0, y1, x0, x1;
  float ty, tx;
  resize_src(tok / ow, oh, h, &y0, &y1, &ty);
  resize_src(tok % ow, ow, w, &x0, &x1, &tx);
  const float p00 = ld(x[(static_cast<long long>(y0) * w + x0) * c + ch]), p01 = ld(x[(static_cast<long long>(y0) * w + x1) * c + ch]);
  const float p10 = ld(x[(static_cast<long long>(y1) * w + x0) * c + ch]), p11 = ld(x[(static_cast<long long>(y1) * w + x1) * c + ch]);
  st(out + i, (p00 * (1 - tx) + p01 * tx) * (1 - ty) + (p10 * (1 - tx) + p11 * tx) * ty);
}

// Per token: x / max(|x|, 1e-12). Grid: n.
template <typename R>
__global__ void k_l2_normalize(const R* x, int n, int c, R* out) {
  const long long i = thread_index();
  if (i >= n) return;
  const R* xr = x + i * c;
  float sq = 0;
  for (int j = 0; j < c; ++j) {
    const float v = ld(xr[j]);
    sq = fmaf(v, v, sq);
  }
  const float inv = 1.0f / fmaxf(sqrtf(sq), 1e-12f);
  for (int j = 0; j < c; ++j) st(out + i * c + j, ld(xr[j]) * inv);
}

// Rotary encoding. cosv, sinv: (n, pairs). Grid: n * heads * pairs.
template <typename R>
__global__ void k_rotary(const R* x, const float* cosv, const float* sinv, int n, int heads, int pairs, R* out) {
  const long long i = thread_index();
  const int per_token = heads * pairs;
  if (i >= static_cast<long long>(n) * per_token) return;
  const long long tok = i / per_token;
  const int j = static_cast<int>(i % per_token) % pairs;
  const float c = cosv[tok * pairs + j], s = sinv[tok * pairs + j];
  const float a = ld(x[2 * i]), b = ld(x[2 * i + 1]);
  st(out + 2 * i, a * c - b * s);
  st(out + 2 * i + 1, b * c + a * s);
}

// Rotary encoding with the tables as tensors in the storage type (LightGlue: made per image). cosv, sinv:
// (n, pairs). Grid: n * heads * pairs.
template <typename R>
__global__ void k_rotary_t(const R* x, const R* cosv, const R* sinv, int n, int heads, int pairs, R* out) {
  const long long i = thread_index();
  const int per_token = heads * pairs;
  if (i >= static_cast<long long>(n) * per_token) return;
  const long long tok = i / per_token;
  const int j = static_cast<int>(i % per_token) % pairs;
  const float c = ld(cosv[tok * pairs + j]), s = ld(sinv[tok * pairs + j]);
  const float a = ld(x[2 * i]), b = ld(x[2 * i + 1]);
  st(out + 2 * i, a * c - b * s);
  st(out + 2 * i + 1, b * c + a * s);
}

// out[r] = x[rows[r]], whole rows of c values. Grid: count * c.
template <typename R>
__global__ void k_gather_rows(const R* x, const int* rows, int c, int count, R* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(count) * c) return;
  out[i] = x[static_cast<long long>(rows[i / c]) * c + i % c];
}

// ---------------------------------------------------------------------------------------------
// LightGlue's match assignment (core/ops.h) on s (M, N) float, made by the read-out's product:
//   score(i, j) = ((s_ij - row_lse_i) + (s_ij - col_lse_j)) + (logsigmoid(za_i) + logsigmoid(zb_j))
// in four passes: the log-sum-exp of every row and of every column, then the best of every row and of
// every column. The order of operations is the host's (core/assignment.h); ties go to the lower index.
// Row kernels: one warp per row, 8 per block (grid ceil(M / 8)). Column kernels: one thread per column.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float log_sigmoid(float z) { return fminf(z, 0.0f) - log1pf(expf(-fabsf(z))); }

__global__ void k_lse_rows(const float* s, int M, int N, float* lse) {
  const int lane = threadIdx.x & 31, r = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (r >= M) return;
  const float* row = s + static_cast<long long>(r) * N;
  float top = -INFINITY;
  for (int j = lane; j < N; j += 32) top = fmaxf(top, row[j]);
  for (int o = 16; o > 0; o >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o));
  float sum = 0.0f;
  for (int j = lane; j < N; j += 32) sum += expf(row[j] - top);
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) lse[r] = top + logf(sum);
}

__global__ void k_lse_cols(const float* s, int M, int N, float* lse) {
  const long long j = thread_index();
  if (j >= N) return;
  float top = -INFINITY;
  for (int i = 0; i < M; ++i) top = fmaxf(top, s[static_cast<long long>(i) * N + j]);
  float sum = 0.0f;
  for (int i = 0; i < M; ++i) sum += expf(s[static_cast<long long>(i) * N + j] - top);
  lse[j] = top + logf(sum);
}

template <typename R>
__global__ void k_best_rows(const float* s, int M, int N, const float* row_lse, const float* col_lse, const R* za, const R* zb,
                            int* best, float* value) {
  const int lane = threadIdx.x & 31, r = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (r >= M) return;
  const float* row = s + static_cast<long long>(r) * N;
  const float lr = row_lse[r], l0 = log_sigmoid(ld(za[r]));
  float top = -INFINITY;
  int at = N;
  for (int j = lane; j < N; j += 32) {
    const float v = ((row[j] - lr) + (row[j] - col_lse[j])) + (l0 + log_sigmoid(ld(zb[j])));
    if (v > top) top = v, at = j;
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float tv = __shfl_xor_sync(0xffffffffu, top, o);
    const int ta = __shfl_xor_sync(0xffffffffu, at, o);
    if (tv > top || (tv == top && ta < at)) top = tv, at = ta;
  }
  if (lane == 0) best[r] = at, value[r] = top;
}

template <typename R>
__global__ void k_best_cols(const float* s, int M, int N, const float* row_lse, const float* col_lse, const R* za, const R* zb,
                            int* best, float* value) {
  const long long j = thread_index();
  if (j >= N) return;
  const float lc = col_lse[j], l1 = log_sigmoid(ld(zb[j]));
  float top = -INFINITY;
  int at = 0;
  for (int i = 0; i < M; ++i) {
    const float sij = s[static_cast<long long>(i) * N + j];
    const float v = ((sij - row_lse[i]) + (sij - lc)) + (log_sigmoid(ld(za[i])) + l1);
    if (v > top) top = v, at = i;
  }
  best[j] = at, value[j] = top;
}

// Bilinear samples at points in [-1, 1], border clamped. Grid: n * c.
template <typename R>
__global__ void k_sample(const R* x, const float* pts, int h, int w, int c, int n, R* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(n) * c) return;
  const int p = static_cast<int>(i / c), ch = static_cast<int>(i % c);
  float fx = (pts[2 * p] + 1.0f) * 0.5f * static_cast<float>(w) - 0.5f;
  float fy = (pts[2 * p + 1] + 1.0f) * 0.5f * static_cast<float>(h) - 0.5f;
  fx = fminf(fmaxf(fx, 0.0f), static_cast<float>(w - 1)), fy = fminf(fmaxf(fy, 0.0f), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy), x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0);
  const float p00 = ld(x[(static_cast<long long>(y0) * w + x0) * c + ch]), p01 = ld(x[(static_cast<long long>(y0) * w + x1) * c + ch]);
  const float p10 = ld(x[(static_cast<long long>(y1) * w + x0) * c + ch]), p11 = ld(x[(static_cast<long long>(y1) * w + x1) * c + ch]);
  st(out + i, p00 * (1 - tx) * (1 - ty) + p01 * tx * (1 - ty) + p10 * (1 - tx) * ty + p11 * tx * ty);
}

// Match read-out of one query from its logits over the h x w cells (see Ops::readout and core/readout.h,
// which this follows). One warp per query; lane l scans cells l, l + 32, ... and the warp reduces.
// Blocks of 256 threads (8 queries). Grid: ceil(N / 8).
//
// The 1.5-entmax threshold tau solves sum over T of max(y - tau, 0)^2 = 1 with y = (logit - top) / 2 and
// T the 512 largest logits (the reference sorts them and uses a closed form per prefix; the root is the
// same). The sum is convex and decreasing in tau and is at least 1 at tau = -1, so Newton's method from
// -1 climbs to the root without passing it. Cells with y <= -1 never count.
// When more than 512 cells lie above -1 (often thousands), the 512th largest y is located with three
// passes of a 256-bin histogram, each within the bin the previous one chose: that pins it to a cell of
// width 2^-24, and the cells that share it with the 512th stand in as `extra` copies of their largest y.
// The y that count (at most 512) are then gathered into shared memory, where Newton's method reads them.
// Every lane ends a butterfly reduction with the same value, so the loops branch alike on all lanes.
constexpr int kReadoutList = 512;

// Histogram bin of y in (-1, 0] at level 0, 1, 2: successive base-256 digits of (y + 1) * 256.
__device__ __forceinline__ int readout_key(float y) {
  float f = (y + 1.0f) * 256.0f;
  const int b0 = min(255, static_cast<int>(f));
  f = (f - static_cast<float>(b0)) * 256.0f;
  const int b1 = min(255, static_cast<int>(f));
  f = (f - static_cast<float>(b1)) * 256.0f;
  const int b2 = min(255, static_cast<int>(f));
  return (b0 << 16) | (b1 << 8) | b2;
}

__global__ void k_readout(const float* logits, int N, int h, int w, float* coarse, float* mass) {
  __shared__ float lists[8][kReadoutList];
  __shared__ int hists[8][256];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, q = blockIdx.x * 8 + warp;
  if (q >= N) return;
  const int P = h * w;
  const float* row = logits + static_cast<long long>(q) * P;
  float* list = lists[warp];
  int* hist = hists[warp];
  // Largest logit and the first cell that has it.
  float best = -INFINITY;
  int at = P;
  for (int s = lane; s < P; s += 32) {
    const float v = row[s];
    if (v > best) best = v, at = s;
  }
  const float top = warp_max(best);
  const int arg = warp_min(best == top ? at : P);

  // Locate the 512th largest y when needed: `cell` is its key (the y with a larger key count in full),
  // `above` how many have a larger key.
  int count = 0;
  for (int s = lane; s < P; s += 32) count += (row[s] - top) * 0.5f > -1.0f;
  count = warp_sum(count);
  int cell = -1, above = 0;
  if (count > kReadoutList) {
    int prefix = 0;
    for (int level = 0; level < 3; ++level) {
      const int shift = 16 - 8 * level;
      for (int b = lane; b < 256; b += 32) hist[b] = 0;
      __syncwarp();
      for (int s = lane; s < P; s += 32) {
        const float y = (row[s] - top) * 0.5f;
        if (!(y > -1.0f)) continue;
        const int k = readout_key(y);
        if ((k >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255], 1);
      }
      __syncwarp();
      // Bins from the top: lane l holds bins 8 l ... 8 l + 7. The bin where the count from the top reaches
      // 512 - above holds the 512th.
      int part[8], total = 0;
      #pragma unroll
      for (int j = 0; j < 8; ++j) part[j] = hist[8 * lane + j], total += part[j];
      int incl = total;  // bins of lanes l ... 31
      #pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_down_sync(0xffffffffu, incl, o);
        if (lane + o < 32) incl += t;
      }
      const int need = kReadoutList - above, excl = incl - total;
      int bin = -1, more = 0;
      if (excl < need && incl >= need) {
        int cum = excl;
        for (int j = 7; j >= 0; --j) {
          if (cum + part[j] >= need) {
            bin = 8 * lane + j, more = cum;
            break;
          }
          cum += part[j];
        }
      }
      const int src = __ffs(__ballot_sync(0xffffffffu, bin >= 0)) - 1;
      bin = __shfl_sync(0xffffffffu, bin, src), more = __shfl_sync(0xffffffffu, more, src);
      prefix |= bin << shift, above += more;
      __syncwarp();
    }
    cell = prefix;
  }
  // Gather the y that count, and the largest y of the cell of the 512th.
  int n = 0;
  float tie = -INFINITY;
  for (int s0 = 0; s0 < P; s0 += 32) {
    const int s = s0 + lane;
    const float y = s < P ? (row[s] - top) * 0.5f : -1.0f;
    bool take = y > -1.0f;
    if (cell >= 0 && take) {
      const int k = readout_key(y);
      if (k == cell) tie = fmaxf(tie, y);
      take = k > cell;
    }
    const unsigned mask = __ballot_sync(0xffffffffu, take);
    if (take) list[n + __popc(mask & ((1u << lane) - 1))] = y;
    n += __popc(mask);
  }
  __syncwarp();
  const float cut = cell >= 0 ? warp_max(tie) : -1.0f;
  const int extra = cell >= 0 ? kReadoutList - n : 0;
  float tau = -1.0f;
  for (int it = 0; it < 40; ++it) {
    float a = 0, b = 0;
    for (int i = lane; i < n; i += 32) {
      const float d = fmaxf(list[i] - tau, 0.0f);
      a += d, b = fmaf(d, d, b);
    }
    const float gap = fmaxf(cut - tau, 0.0f), de = gap * static_cast<float>(extra);
    a = warp_sum(a) + de, b = warp_sum(b) + de * gap;
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
      ny = min(max(ny, 0), h - 1), nx = min(max(nx, 0), w - 1);
      float p = fmaxf((row[ny * w + nx] - top) * 0.5f - tau, 0.0f);
      p *= p;
      px += p * ((static_cast<float>(nx) + 0.5f) / static_cast<float>(w) * 2.0f - 1.0f);
      py += p * ((static_cast<float>(ny) + 0.5f) / static_cast<float>(h) * 2.0f - 1.0f);
      total += p;
      if (in) inside += p;
    }
  coarse[2 * q] = px / total, coarse[2 * q + 1] = py / total;
  mass[q] = inside;
}

// Network input from 8-bit pixels: bilinear resize to oh x ow of pixel / 255, then (v - mean) / std per
// channel. ch bytes per pixel (1: grey; 3 or 4: R, G, B first). The source rows and columns come from the
// host (taps: 3 floats per output row, then 3 per output column: first index, second index, weight of the
// second), and the arithmetic is rounded step by step as written (no fused multiply-adds), so the result
// is the host's (core/preprocess.h) to the last bit. Grid: oh * ow.
template <typename R>
__global__ void k_preprocess(const unsigned char* img, const float* taps, int stride, int ch, int oh, int ow, float3 mean,
                             float3 sd, R* out) {
  const long long i = thread_index();
  if (i >= static_cast<long long>(oh) * ow) return;
  const float* ty3 = taps + 3 * (i / ow);
  const float* tx3 = taps + 3 * (oh + i % ow);
  const int y0 = static_cast<int>(ty3[0]), y1 = static_cast<int>(ty3[1]), x0 = static_cast<int>(tx3[0]), x1 = static_cast<int>(tx3[1]);
  const float ty = ty3[2], tx = tx3[2];
  const float m[3] = {mean.x, mean.y, mean.z}, d[3] = {sd.x, sd.y, sd.z};
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int o = ch == 1 ? 0 : c;
    const float p00 = __fdiv_rn(static_cast<float>(img[static_cast<long long>(y0) * stride + x0 * ch + o]), 255.0f);
    const float p01 = __fdiv_rn(static_cast<float>(img[static_cast<long long>(y0) * stride + x1 * ch + o]), 255.0f);
    const float p10 = __fdiv_rn(static_cast<float>(img[static_cast<long long>(y1) * stride + x0 * ch + o]), 255.0f);
    const float p11 = __fdiv_rn(static_cast<float>(img[static_cast<long long>(y1) * stride + x1 * ch + o]), 255.0f);
    const float top = __fadd_rn(__fmul_rn(p00, __fsub_rn(1.0f, tx)), __fmul_rn(p01, tx));
    const float bottom = __fadd_rn(__fmul_rn(p10, __fsub_rn(1.0f, tx)), __fmul_rn(p11, tx));
    const float v = __fadd_rn(__fmul_rn(top, __fsub_rn(1.0f, ty)), __fmul_rn(bottom, ty));
    st(out + i * 3 + c, __fdiv_rn(__fsub_rn(v, m[c]), d[c]));
  }
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("cuda: " + what); }

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(status));
}
#define JMT_CUDA_CHECK(call) check((call), #call)

// JMT_CUDA_PROFILE: time per operation label (see the top of the file). Every launch is bracketed by two
// events on the stream and nothing waits for them until there are many: the GPU runs as it does without
// the profile (waiting after every launch let its clock drop between kernels and doubled the totals).
struct Profiler {
  bool on = false;
  std::string label;
  std::map<std::string, std::pair<double, long>> stats;  // label -> (ms, launches)
  struct Mark {
    std::string label;
    cudaEvent_t start, stop;
  };
  std::vector<Mark> marks;    // recorded, not yet read
  std::vector<cudaEvent_t> spare;

  Profiler() {
    const char* env = std::getenv("JMT_CUDA_PROFILE");
    on = env && *env && std::string(env) != "0";
  }
  cudaEvent_t event() {
    cudaEvent_t e;
    if (!spare.empty()) {
      e = spare.back();
      spare.pop_back();
    } else {
      cudaEventCreate(&e);
    }
    return e;
  }
  // Reads the recorded marks into stats (waits for them).
  void collect() {
    for (Mark& m : marks) {
      cudaEventSynchronize(m.stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, m.start, m.stop);
      auto& st = stats[m.label];
      st.first += ms, ++st.second;
      spare.push_back(m.start), spare.push_back(m.stop);
    }
    marks.clear();
  }
  void tag(const char* fmt, ...) {
    if (!on) return;
    char buf[160];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    label = buf;
  }
  void report() {
    if (!on) return;
    collect();
    if (stats.empty()) return;
    using Row = std::pair<std::string, std::pair<double, long>>;
    double total = 0;
    std::map<std::string, std::pair<double, long>> by_kind;  // first word of the label
    for (const auto& r : stats) {
      total += r.second.first;
      auto& k = by_kind[r.first.substr(0, r.first.find(' '))];
      k.first += r.second.first, k.second += r.second.second;
    }
    auto print = [&](const char* title, std::vector<Row> rows) {
      std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.second.first > b.second.first; });
      std::fprintf(stderr, "\n%-58s %10s %7s %6s\n", title, "ms", "share", "calls");
      for (size_t i = 0; i < rows.size() && i < 40; ++i)
        std::fprintf(stderr, "%-58s %10.2f %6.1f%% %6ld\n", rows[i].first.c_str(), rows[i].second.first,
                     100.0 * rows[i].second.first / total, rows[i].second.second);
    };
    std::fprintf(stderr, "\njmt cuda profile: kernel time %.1f ms\n", total);
    print("operation", std::vector<Row>(stats.begin(), stats.end()));
    print("kind", std::vector<Row>(by_kind.begin(), by_kind.end()));
    stats.clear();
  }
};
Profiler g_profile;

template <typename T>
struct Identity {
  using type = T;
};

// Launches `kernel` on the given grid. The kernel's parameter types fix the argument types.
template <typename... Args>
void launch_grid(void (*kernel)(Args...), dim3 grid, dim3 block, typename Identity<Args>::type... args) {
  if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
  if (!g_profile.on) {
    kernel<<<grid, block>>>(args...);
    check(cudaGetLastError(), "kernel launch");
    return;
  }
  Profiler::Mark m{g_profile.label, g_profile.event(), g_profile.event()};
  cudaEventRecord(m.start, 0);
  kernel<<<grid, block>>>(args...);
  check(cudaGetLastError(), "kernel launch");
  cudaEventRecord(m.stop, 0);
  g_profile.marks.push_back(m);
  if (g_profile.marks.size() >= 8192) g_profile.collect();
}

// One-dimensional kernels: at least n threads.
template <typename... Args>
void launch(void (*kernel)(Args...), long long n, typename Identity<Args>::type... args) {
  if (n <= 0) return;
  const long long blocks = (n + kBlock - 1) / kBlock;
  if (blocks > INT_MAX) fail("grid too large");
  launch_grid(kernel, dim3(static_cast<unsigned>(blocks)), dim3(kBlock), args...);
}

// Device memory with reuse: released blocks are kept and handed out again for the same size.
class Pool {
 public:
  explicit Pool(int device) : device_(device) {}
  ~Pool() {
    if (cudaSetDevice(device_) == cudaSuccess) {
      cudaDeviceSynchronize();
      trim();
    }
  }
  int device() const { return device_; }

  void* alloc(size_t bytes) {
    bytes = std::max<size_t>(bytes, 16);
    auto it = free_.find(bytes);
    if (it != free_.end()) {
      void* p = it->second;
      free_.erase(it);
      return p;
    }
    void* p = nullptr;
    ++created_;
    cudaError_t status = cudaMalloc(&p, bytes);
    if (status == cudaErrorMemoryAllocation) {  // give back what is cached and try once more
      cudaGetLastError();
      cudaDeviceSynchronize();
      trim();
      status = cudaMalloc(&p, bytes);
    }
    check(status, "cudaMalloc");
    return p;
  }
  void release(void* p, size_t bytes) { free_.emplace(std::max<size_t>(bytes, 16), p); }
  long created() const { return created_; }
  void trim() {
    for (auto& kv : free_) cudaFree(kv.second);
    free_.clear();
  }

 private:
  int device_;
  std::multimap<size_t, void*> free_;
  long created_ = 0;  // cudaMalloc calls
};
using PoolPtr = std::shared_ptr<Pool>;

// A block of bytes from the pool, returned to it on destruction.
class Buffer {
 public:
  Buffer(const PoolPtr& pool, size_t bytes) : pool_(pool), bytes_(bytes), p_(pool->alloc(bytes)) {}
  ~Buffer() { pool_->release(p_, bytes_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template <typename T>
  T* as() const { return static_cast<T*>(p_); }
  // Queued after the work launched so far. From pageable memory the driver stages the bytes before it
  // returns, so src may change at once; unlike cudaMemcpy it does not wait for the device to go idle, which
  // would stall the host in the middle of every pair.
  void upload(const void* src, size_t bytes) {
    if (bytes) JMT_CUDA_CHECK(cudaMemcpyAsync(p_, src, bytes, cudaMemcpyHostToDevice, 0));
  }
  void download(void* dst, size_t bytes) const {
    if (bytes) JMT_CUDA_CHECK(cudaMemcpy(dst, p_, bytes, cudaMemcpyDeviceToHost));
  }

 private:
  PoolPtr pool_;
  size_t bytes_;
  void* p_;
};

// Elements are float, or __half when `half` is set (the read-out's positions and masses are float in both
// modes).
struct CudaTensor : Tensor {
  bool half;
  Buffer buf;
  CudaTensor(const PoolPtr& pool, int h_, int w_, int c_, bool half_)
      : half(half_), buf(pool, static_cast<size_t>(h_) * w_ * c_ * (half_ ? sizeof(__half) : sizeof(float))) {
    h = h_, w = w_, c = c_;
  }
  void read(float* dst) const override {
    const size_t count = size();
    if (!half) {
      buf.download(dst, count * sizeof(float));
      return;
    }
    std::vector<uint16_t> bits(count);
    buf.download(bits.data(), count * sizeof(uint16_t));
    for (size_t i = 0; i < count; ++i) dst[i] = half_to_float(bits[i]);
  }
};

template <typename R>
class CudaOps : public Ops {
  static constexpr bool kHalf = sizeof(R) == sizeof(__half);
  using Ptr = std::shared_ptr<CudaTensor>;

 public:
  explicit CudaOps(int device) {
    int devices = 0;
    JMT_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (device < 0 || device >= devices)
      fail("device " + std::to_string(device) + " not found (" + std::to_string(devices) + " available)");
    JMT_CUDA_CHECK(cudaSetDevice(device));
    pool_ = std::make_shared<Pool>(device);
    const char* e = std::getenv("JMT_CUDA_GEMM");
    if (e && std::string(e) == "k16") chunk_ = 16;
    else if (e && *e && std::string(e) != "k8") fail("unknown JMT_CUDA_GEMM");
    if ((e = std::getenv("JMT_CUDA_TENSOR"))) tensor_ = std::atoi(e);
    if ((e = std::getenv("JMT_CUDA_READOUT"))) readout_device_ = std::atoi(e) != 0;
    if ((e = std::getenv("JMT_CUDA_FLASH"))) flash_ = std::atoi(e);
    if ((e = std::getenv("JMT_CUDA_SPLIT"))) split_ = std::atoi(e) != 0;
    threads_ = std::max(1u, std::thread::hardware_concurrency());
  }
  ~CudaOps() override {
    if (cudaSetDevice(pool_->device()) == cudaSuccess) {
      cudaDeviceSynchronize();
      for (auto& kv : params_) cudaFree(kv.second);
    }
    g_profile.report();
  }

  TensorPtr upload(const float* data, int h, int w, int c) override {
    Ptr out = make(h, w, c);
    store(data, out->size(), out->buf);
    return out;
  }

  TensorPtr preprocess(const float* rgb, int height, int width, int out_h, int out_w, const float mean[3],
                       const float std[3]) override {
    std::vector<float> v(static_cast<size_t>(out_h) * out_w * 3);
    const int parts = std::min(threads_, out_h);
    std::vector<std::thread> pool;
    for (int t = 0; t < parts; ++t)
      pool.emplace_back([&, t] {
        for (int y = out_h * t / parts; y < out_h * (t + 1) / parts; ++y)
          preprocess_row(rgb, height, width, out_h, out_w, mean, std, y, v.data() + static_cast<size_t>(y) * out_w * 3);
      });
    for (auto& th : pool) th.join();
    return upload(v.data(), out_h, out_w, 3);
  }

  TensorPtr preprocess_u8(const unsigned char* data, int height, int width, int stride, int channels, int out_h,
                          int out_w, const float mean[3], const float std[3]) override {
    const size_t bytes = static_cast<size_t>(height - 1) * stride + static_cast<size_t>(width) * channels;
    Buffer image(pool_, (bytes + 4095) / 4096 * 4096);  // rounded up so images of similar size reuse it
    image.upload(data, bytes);
    // Source rows and columns of the resize, computed here so they match the host implementation exactly.
    std::vector<float> taps(static_cast<size_t>(out_h + out_w) * 3);
    for (int i = 0; i < out_h + out_w; ++i) {
      float* t = taps.data() + 3 * i;
      int i0, i1;
      if (i < out_h) resize_source(i, out_h, height, &i0, &i1, t + 2);
      else resize_source(i - out_h, out_w, width, &i0, &i1, t + 2);
      t[0] = static_cast<float>(i0), t[1] = static_cast<float>(i1);
    }
    Buffer dtaps(pool_, taps.size() * sizeof(float));
    dtaps.upload(taps.data(), taps.size() * sizeof(float));
    Ptr out = make(out_h, out_w, 3);
    g_profile.tag("preprocess");
    launch(k_preprocess<R>, static_cast<long long>(out_h) * out_w, image.template as<unsigned char>(), dtaps.template as<float>(),
           stride, channels, out_h, out_w, make_float3(mean[0], mean[1], mean[2]), make_float3(std[0], std[1], std[2]),
           out->buf.template as<R>());
    return out;
  }

  TensorPtr linear(const TensorPtr& xp, const Param& weight, const Param* bias) override {
    return linear_fused(xp, weight, bias, Act::None, nullptr);
  }

  TensorPtr linear_fused(const TensorPtr& xp, const Param& weight, const Param* bias, Act act,
                         const TensorPtr& residual) override {
    const CudaTensor& x = dev(xp);
    if (weight.shape.size() != 2 || weight.shape[1] != x.c) fail("linear: shape mismatch");
    const int out_c = weight.shape[0], in_c = weight.shape[1];
    if (residual && (residual->n() != x.n() || residual->c != out_c)) fail("linear: residual shape mismatch");
    const bool by_output = tensor_product(x.n(), in_c, out_c);
    const R* w = matrix(weight, in_c, out_c, by_output, [&](int o, int i) { return weight.data[static_cast<size_t>(o) * in_c + i]; });
    g_profile.tag("linear %d -> %d, n %d", in_c, out_c, x.n());
    return product(x.buf.template as<R>(), x.h, x.w, in_c, w, by_output, bias, out_c, act_code(act),
                   residual ? dev(residual).buf.template as<R>() : nullptr);
  }

  TensorPtr conv(const TensorPtr& xp, const Param& weight, const Param& bias, int k, int stride, int pad) override {
    const CudaTensor& x = dev(xp);
    if (weight.shape.size() != 4 || weight.shape[1] != x.c || weight.shape[2] != k) fail("conv: shape mismatch");
    const int out_c = weight.shape[0], in_c = weight.shape[1];
    const int oh = (x.h + 2 * pad - k) / stride + 1, ow = (x.w + 2 * pad - k) / stride + 1, cols = k * k * in_c;
    Ptr patches = make(oh, ow, cols);
    g_profile.tag("conv%d c%d -> %d, %dx%d", k, in_c, out_c, ow, oh);
    launch(k_patches<R>, static_cast<long long>(patches->size()), x.buf.template as<R>(), x.h, x.w, x.c, k, stride, pad, oh,
           ow, patches->buf.template as<R>());
    const bool by_output = tensor_product(oh * ow, cols, out_c);
    const R* w = matrix(weight, cols, out_c, by_output, [&](int o, int i) {  // input i = (ky, kx, channel)
      return weight.data[(static_cast<size_t>(o) * in_c + i % in_c) * k * k + i / in_c];
    });
    return product(patches->buf.template as<R>(), oh, ow, cols, w, by_output, &bias, out_c, kNoAct, nullptr);
  }

  TensorPtr dwconv7(const TensorPtr& xp, const Param& weight, const Param& bias) override {
    const CudaTensor& x = dev(xp);
    const int c = x.c;
    const R* taps = param<R>(weight, 1, static_cast<size_t>(49) * c, [&](float* dst) {
      for (int ch = 0; ch < c; ++ch)
        for (int t = 0; t < 49; ++t) dst[static_cast<size_t>(t) * c + ch] = weight.data[static_cast<size_t>(ch) * 49 + t];
    });
    Ptr out = make(x.h, x.w, c);
    g_profile.tag("dwconv7 c%d %dx%d", c, x.w, x.h);
    constexpr int V = 16 / sizeof(R), TX = 4;
    if (c % V == 0)
      launch(k_dwconv7_vec<R, TX>, static_cast<long long>(x.h) * ((x.w + TX - 1) / TX) * (c / V), x.buf.template as<R>(), taps,
             vec<R>(bias), x.h, x.w, c, out->buf.template as<R>());
    else
      launch(k_dwconv7<R>, static_cast<long long>(x.size()), x.buf.template as<R>(), taps, vec<R>(bias), x.h, x.w, c,
             out->buf.template as<R>());
    return out;
  }

  TensorPtr layer_norm(const TensorPtr& xp, const Param& gain, const Param& bias, float eps) override {
    return layer_norm_act(xp, gain, bias, eps, Act::None);
  }

  TensorPtr layer_norm_act(const TensorPtr& xp, const Param& gain, const Param& bias, float eps, Act act) override {
    const CudaTensor& x = dev(xp);
    const int g = static_cast<int>(gain.size());
    if (g <= 0 || x.c % g != 0) fail("layer_norm: shape mismatch");
    Ptr out = make(x.h, x.w, x.c);
    const long long groups = static_cast<long long>(x.n()) * (x.c / g);
    const int a = act_code(act);
    g_profile.tag("layer_norm c%d g%d n %d", x.c, g, x.n());
    // 16-byte loads: g / V chunks of V values, a power of two up to 32 (one group per gl lanes) or 32 lanes
    // with nc = 2, 4 or 8 chunks each.
    constexpr int V = 16 / sizeof(R);
    const int chunks = g % V == 0 ? g / V : 0;
    const int gl = chunks >= 32 ? 32 : chunks, nc = chunks / std::max(gl, 1);
    const bool pow2 = gl > 0 && (gl & (gl - 1)) == 0 && gl * nc == chunks && (nc == 1 || nc == 2 || nc == 4 || nc == 8);
    const long long per_block = 8LL * (32 / std::max(gl, 1)), blocks = (groups + per_block - 1) / per_block;
    if (pow2 && blocks <= INT_MAX) {
      auto kern = nc == 1 ? k_layer_norm_vec<R, 1> : nc == 2 ? k_layer_norm_vec<R, 2> : nc == 4 ? k_layer_norm_vec<R, 4> : k_layer_norm_vec<R, 8>;
      launch_grid(kern, dim3(static_cast<unsigned>(blocks)), dim3(kBlock), x.buf.template as<R>(), vec<R>(gain), vec<R>(bias),
                  groups, g, gl, eps, a, out->buf.template as<R>());
    } else if (g >= kBlock && groups <= INT_MAX)
      launch_grid(k_layer_norm_wide<R>, dim3(static_cast<unsigned>(groups)), dim3(kBlock), x.buf.template as<R>(),
                  vec<R>(gain), vec<R>(bias), g, eps, a, out->buf.template as<R>());
    else
      launch(k_layer_norm<R>, groups, x.buf.template as<R>(), vec<R>(gain), vec<R>(bias), x.n(), x.c, g, eps, a,
             out->buf.template as<R>());
    return out;
  }

  TensorPtr gelu(const TensorPtr& xp) override { return unary(xp, kGelu, "gelu"); }
  TensorPtr relu(const TensorPtr& xp) override { return unary(xp, kRelu, "relu"); }

  TensorPtr add(const TensorPtr& ap, const TensorPtr& bp) override {
    const CudaTensor &a = dev(ap), &b = dev(bp);
    if (a.size() != b.size()) fail("add: shape mismatch");
    Ptr out = make(a.h, a.w, a.c);
    g_profile.tag("add");
    const long long count = static_cast<long long>(a.size());
    launch(k_add<R>, count, a.buf.template as<R>(), b.buf.template as<R>(), count, out->buf.template as<R>());
    return out;
  }

  TensorPtr concat(const TensorPtr& ap, const TensorPtr& bp) override {
    const CudaTensor &a = dev(ap), &b = dev(bp);
    if (a.n() != b.n()) fail("concat: shape mismatch");
    Ptr out = make(a.h, a.w, a.c + b.c);
    g_profile.tag("concat");
    launch(k_copy_channels<R>, static_cast<long long>(a.n()) * a.c, a.buf.template as<R>(), a.n(), a.c, 0, a.c, 0, out->c,
           out->buf.template as<R>());
    launch(k_copy_channels<R>, static_cast<long long>(b.n()) * b.c, b.buf.template as<R>(), b.n(), b.c, 0, b.c, a.c, out->c,
           out->buf.template as<R>());
    return out;
  }

  TensorPtr slice(const TensorPtr& xp, int first, int count) override {
    const CudaTensor& x = dev(xp);
    Ptr out = make(x.h, x.w, count);
    g_profile.tag("slice");
    launch(k_copy_channels<R>, static_cast<long long>(x.n()) * count, x.buf.template as<R>(), x.n(), x.c, first, count, 0,
           count, out->buf.template as<R>());
    return out;
  }

  TensorPtr pixel_unshuffle2(const TensorPtr& xp) override {
    const CudaTensor& x = dev(xp);
    Ptr out = make(x.h / 2, x.w / 2, x.c * 4);
    g_profile.tag("pixel_unshuffle2");
    launch(k_pixel_unshuffle2<R>, static_cast<long long>(out->size()), x.buf.template as<R>(), x.h, x.w, x.c,
           out->buf.template as<R>());
    return out;
  }

  TensorPtr resize(const TensorPtr& xp, int out_h, int out_w) override {
    const CudaTensor& x = dev(xp);
    Ptr out = make(out_h, out_w, x.c);
    g_profile.tag("resize");
    launch(k_resize<R>, static_cast<long long>(out->size()), x.buf.template as<R>(), x.h, x.w, x.c, out_h, out_w,
           out->buf.template as<R>());
    return out;
  }

  TensorPtr l2_normalize(const TensorPtr& xp) override {
    const CudaTensor& x = dev(xp);
    Ptr out = make(x.h, x.w, x.c);
    g_profile.tag("l2_normalize");
    launch(k_l2_normalize<R>, x.n(), x.buf.template as<R>(), x.n(), x.c, out->buf.template as<R>());
    return out;
  }

  TensorPtr rotary(const TensorPtr& xp, const std::vector<float>& cos, const std::vector<float>& sin, int heads) override {
    const CudaTensor& x = dev(xp);
    const int pairs = x.c / heads / 2;
    if (static_cast<int>(cos.size()) != x.n() * pairs) fail("rotary: shape mismatch");
    Ptr out = make(x.h, x.w, x.c);
    g_profile.tag("rotary");
    launch(k_rotary<R>, static_cast<long long>(x.n()) * heads * pairs, x.buf.template as<R>(), table(cos), table(sin), x.n(),
           heads, pairs, out->buf.template as<R>());
    return out;
  }

  TensorPtr attention(const TensorPtr& qp, const TensorPtr& kp, const TensorPtr& vp, int heads) override {
    const CudaTensor &q = dev(qp), &k = dev(kp), &v = dev(vp);
    if (q.c % heads != 0 || k.c != q.c || v.c != q.c || k.n() != v.n()) fail("attention: shape mismatch");
    const int P = q.n(), S = k.n(), d = q.c / heads;
    Ptr out = make(q.h, q.w, q.c);
    // Without the logits buffer, on the tensor cores (see k_flash).
    if (flash_ && d == 64 && (kHalf ? tensor_ != 0 : split_) && q.c % 8 == 0 && k.c % 8 == 0 && v.c % 8 == 0) {
      g_profile.tag("attention P %d S %d", P, S);
      const int wq = flash_ == 4 || flash_ == 8 ? flash_ : 16;  // 16 measured fastest (8: 3 % slower, 4: 15 %)
      auto kern = wq == 4 ? k_flash<R, 4, 1> : wq == 8 ? k_flash<R, 8, 1> : k_flash<R, 16, 1>;
      launch_grid(kern, dim3((P + 16 * wq - 1) / (16 * wq), heads), dim3(32 * wq), q.buf.template as<R>(), k.buf.template as<R>(),
                  v.buf.template as<R>(), P, S, q.c, k.c, v.c, q.c, out->buf.template as<R>());
      return out;
    }
    // k_flash masks the source tokens past S; the path through the logits buffer needs whole tiles.
    if (d % 16 != 0 || S % 16 != 0) fail("attention: channels per head and tokens per image must be multiples of 16");
    // Logits (heads, P, S) and the largest of every row.
    Buffer tops(pool_, static_cast<size_t>(heads) * P * sizeof(int));
    JMT_CUDA_CHECK(cudaMemset(tops.template as<int>(), 0x80, static_cast<size_t>(heads) * P * sizeof(int)));  // below any float
    Gemm a{}, b{};
    a.n = P, a.cin = d, a.cout = S, a.scale = 1.0f / std::sqrt(static_cast<float>(d));
    a.x_row = q.c, a.x_batch = d;
    a.w_in = 1, a.w_out = k.c, a.w_batch = d;
    a.o_row = S, a.o_batch = static_cast<long long>(P) * S;
    b.n = P, b.cin = S, b.cout = d, b.scale = 1.0f;
    b.x_row = S, b.x_batch = static_cast<long long>(P) * S;
    b.w_in = v.c, b.w_out = 1, b.w_batch = d;
    b.o_row = q.c, b.o_batch = d;
    if constexpr (kHalf) {
      if (tensor_fits(a) && tensor_fits(b)) {
        // Tensor cores: logits as half, relative to the largest of every 32 (see k_gemm_tc).
        Buffer logits(pool_, static_cast<size_t>(heads) * P * S * sizeof(__half));
        Buffer group_top(pool_, static_cast<size_t>(heads) * P * ((S + 31) / 32) * sizeof(float));
        a.group_top = b.group_top = group_top.template as<float>();
        g_profile.tag("attention logits P %d S %d", P, S);
        gemm_tc<kTops>(q.buf.template as<__half>(), k.buf.template as<__half>(), nullptr, tops.template as<int>(), a,
                       logits.template as<__half>(), heads);
        g_profile.tag("attention values P %d S %d", P, S);
        gemm_tc<kSoftmax>(logits.template as<__half>(), v.buf.template as<__half>(), nullptr, tops.template as<int>(), b,
                          out->buf.template as<__half>(), heads);
        return out;
      }
    }
    Buffer logits(pool_, static_cast<size_t>(heads) * P * S * sizeof(float));  // float in both modes
    g_profile.tag("attention logits P %d S %d", P, S);
    gemm<kTops>(q.buf.template as<R>(), k.buf.template as<R>(), nullptr, tops.template as<int>(), a,
                logits.template as<float>(), heads);
    g_profile.tag("attention values P %d S %d", P, S);
    gemm<kSoftmax>(logits.template as<float>(), v.buf.template as<R>(), nullptr, tops.template as<int>(), b,
                   out->buf.template as<R>(), heads);
    return out;
  }

  TensorPtr rotary_keypoints(const TensorPtr& xp, const TensorPtr& cosp, const TensorPtr& sinp, int heads) override {
    const CudaTensor &x = dev(xp), &c = dev(cosp), &sn = dev(sinp);
    const int pairs = x.c / heads / 2;
    if (c.n() != x.n() || c.c != pairs || sn.n() != x.n() || sn.c != pairs || c.half != x.half || sn.half != x.half)
      fail("rotary_keypoints: shape mismatch");
    Ptr out = make(x.h, x.w, x.c);
    g_profile.tag("rotary");
    launch(k_rotary_t<R>, static_cast<long long>(x.n()) * heads * pairs, x.buf.template as<R>(), c.buf.template as<R>(),
           sn.buf.template as<R>(), x.n(), heads, pairs, out->buf.template as<R>());
    return out;
  }

  TensorPtr gather_rows(const TensorPtr& xp, const std::vector<int>& rows) override {
    const CudaTensor& x = dev(xp);
    const int count = static_cast<int>(rows.size());
    for (int r : rows)
      if (r < 0 || r >= x.n()) fail("gather_rows: index out of range");
    Ptr out = make(1, count, x.c);
    if (count == 0) return out;
    Buffer idx(pool_, rows.size() * sizeof(int));
    idx.upload(rows.data(), rows.size() * sizeof(int));
    g_profile.tag("gather");
    launch(k_gather_rows<R>, static_cast<long long>(count) * x.c, x.buf.template as<R>(), idx.template as<int>(), x.c, count,
           out->buf.template as<R>());
    return out;
  }

  // The similarity is the read-out's product (float in both precisions); the rest is four small passes.
  void assignment(const TensorPtr& ap, const TensorPtr& bp, const TensorPtr& zap, const TensorPtr& zbp,
                  std::vector<int>* best0, std::vector<float>* max0, std::vector<int>* best1,
                  std::vector<float>* max1) override {
    const CudaTensor &a = dev(ap), &b = dev(bp), &za = dev(zap), &zb = dev(zbp);
    const int M = a.n(), N = b.n(), d = a.c;
    if (b.c != d || za.n() != M || zb.n() != N || za.c != 1 || zb.c != 1) fail("assignment: shape mismatch");
    best0->assign(M, -1), max0->assign(M, -INFINITY), best1->assign(N, -1), max1->assign(N, -INFINITY);
    if (M == 0 || N == 0) return;
    if (d % 16 != 0) fail("assignment: the number of channels must be a multiple of 16");
    Buffer s(pool_, static_cast<size_t>(M) * N * sizeof(float));
    Gemm g{};
    g.n = M, g.cin = d, g.cout = N, g.scale = 1.0f / std::sqrt(static_cast<float>(d));
    g.x_row = d, g.w_in = 1, g.w_out = d, g.o_row = N;
    g_profile.tag("assignment M %d N %d", M, N);
    gemm<kPlain>(a.buf.template as<R>(), b.buf.template as<R>(), nullptr, nullptr, g, s.template as<float>(), 1);
    Buffer row_lse(pool_, M * sizeof(float)), col_lse(pool_, N * sizeof(float));
    Buffer arg0(pool_, M * sizeof(int)), val0(pool_, M * sizeof(float)), arg1(pool_, N * sizeof(int)), val1(pool_, N * sizeof(float));
    const float* sp = s.template as<float>();
    launch_grid(k_lse_rows, dim3((M + 7) / 8), dim3(kBlock), sp, M, N, row_lse.template as<float>());
    launch(k_lse_cols, N, sp, M, N, col_lse.template as<float>());
    launch_grid(k_best_rows<R>, dim3((M + 7) / 8), dim3(kBlock), sp, M, N, row_lse.template as<float>(),
                col_lse.template as<float>(), za.buf.template as<R>(), zb.buf.template as<R>(), arg0.template as<int>(),
                val0.template as<float>());
    launch(k_best_cols<R>, N, sp, M, N, row_lse.template as<float>(), col_lse.template as<float>(), za.buf.template as<R>(),
           zb.buf.template as<R>(), arg1.template as<int>(), val1.template as<float>());
    arg0.download(best0->data(), M * sizeof(int)), val0.download(max0->data(), M * sizeof(float));
    arg1.download(best1->data(), N * sizeof(int)), val1.download(max1->data(), N * sizeof(float));
  }

  TensorPtr sample(const TensorPtr& xp, const float* pts, int n) override {
    const CudaTensor& x = dev(xp);
    Buffer p(pool_, static_cast<size_t>(n) * 2 * sizeof(float));
    p.upload(pts, static_cast<size_t>(n) * 2 * sizeof(float));
    Ptr out = make(1, n, x.c);
    g_profile.tag("sample");
    launch(k_sample<R>, static_cast<long long>(n) * x.c, x.buf.template as<R>(), p.template as<float>(), x.h, x.w, x.c, n,
           out->buf.template as<R>());
    return out;
  }

  void readout(const TensorPtr& queryp, const TensorPtr& searchp, float temperature, std::vector<float>* coarse,
               std::vector<float>* mass) override {
    const CudaTensor &query = dev(queryp), &search = dev(searchp);
    const int N = query.n(), P = search.n(), d = query.c, h = search.h, w = search.w;
    coarse->assign(static_cast<size_t>(N) * 2, 0.0f);
    mass->assign(N, 0.0f);
    if (N == 0) return;
    if (d % 16 != 0) fail("readout: the number of channels must be a multiple of 16");
    // Logits (N, P) = temperature * query * search^T on the device, the search tokens as the second operand.
    Buffer dev_logits(pool_, static_cast<size_t>(N) * P * sizeof(float));
    Gemm a{};
    a.n = N, a.cin = d, a.cout = P, a.scale = temperature;
    a.x_row = d, a.w_in = 1, a.w_out = d, a.o_row = P;
    g_profile.tag("readout logits N %d P %d", N, P);
    gemm<kPlain>(query.buf.template as<R>(), search.buf.template as<R>(), nullptr, nullptr, a, dev_logits.template as<float>(), 1);
    std::vector<float> logits(static_cast<size_t>(N) * P);
    dev_logits.download(logits.data(), logits.size() * sizeof(float));
    const int parts = std::min(threads_, N);
    std::vector<std::thread> pool;
    for (int t = 0; t < parts; ++t)
      pool.emplace_back([&, t] {
        ReadoutScratch scratch;
        const int begin = static_cast<int>(static_cast<long long>(N) * t / parts);
        const int end = static_cast<int>(static_cast<long long>(N) * (t + 1) / parts);
        for (int i = begin; i < end; ++i)
          readout_row(logits.data() + static_cast<size_t>(i) * P, h, w, scratch, coarse->data() + 2 * static_cast<size_t>(i),
                      mass->data() + i);
      });
    for (auto& th : pool) th.join();
  }

  bool has_device_readout() const override { return readout_device_; }

  void readout_device(const TensorPtr& queryp, const TensorPtr& searchp, float temperature, TensorPtr* coarse,
                      TensorPtr* mass) override {
    const CudaTensor &query = dev(queryp), &search = dev(searchp);
    const int N = query.n(), P = search.n(), d = query.c;
    if (d % 16 != 0) fail("readout: the number of channels must be a multiple of 16");
    auto c = std::make_shared<CudaTensor>(pool_, 1, N, 2, false), m = std::make_shared<CudaTensor>(pool_, 1, N, 1, false);
    *coarse = c, *mass = m;
    if (N == 0) return;
    Buffer logits(pool_, static_cast<size_t>(N) * P * sizeof(float));
    Gemm a{};
    a.n = N, a.cin = d, a.cout = P, a.scale = temperature;
    a.x_row = d, a.w_in = 1, a.w_out = d, a.o_row = P;
    g_profile.tag("readout logits N %d P %d", N, P);
    gemm<kPlain>(query.buf.template as<R>(), search.buf.template as<R>(), nullptr, nullptr, a, logits.template as<float>(), 1);
    g_profile.tag("readout N %d P %d", N, P);
    launch_grid(k_readout, dim3((N + 7) / 8), dim3(kBlock), logits.template as<float>(), N, search.h, search.w,
                c->buf.template as<float>(), m->buf.template as<float>());
  }

  TensorPtr sample_at(const TensorPtr& xp, const TensorPtr& ptsp) override {
    const CudaTensor& x = dev(xp);
    const CudaTensor& pts = static_cast<const CudaTensor&>(*ptsp);
    if (pts.half || pts.c != 2) fail("sample_at: points must be a float tensor of pairs");
    const int n = pts.n();
    Ptr out = make(1, n, x.c);
    g_profile.tag("sample");
    launch(k_sample<R>, static_cast<long long>(n) * x.c, x.buf.template as<R>(), pts.buf.template as<float>(), x.h, x.w, x.c, n,
           out->buf.template as<R>());
    return out;
  }

  long device_allocations() const override { return pool_->created() + static_cast<long>(params_.size()); }
  // Kernels start when they are launched: there is nothing to flush.
  void flush() override {}
  void finish() override { JMT_CUDA_CHECK(cudaDeviceSynchronize()); }

 private:
  static int act_code(Act a) { return a == Act::Gelu ? kGelu : a == Act::Relu ? kRelu : kNoAct; }

  // Device copy of a table the caller keeps unchanged at one address (the rotary cos and sin, see
  // Ops::rotary), made on first use.
  const float* table(const std::vector<float>& v) {
    auto& slot = tables_[v.data()];
    if (!slot || slot->n != v.size()) {
      slot = std::make_unique<Table>(pool_, v.size());
      slot->buf.upload(v.data(), v.size() * sizeof(float));
    }
    return slot->buf.template as<float>();
  }

  static const CudaTensor& dev(const TensorPtr& t) {
    const CudaTensor& x = static_cast<const CudaTensor&>(*t);
    if (x.half != kHalf) fail("tensor has another element type");
    return x;
  }
  Ptr make(int h, int w, int c) { return std::make_shared<CudaTensor>(pool_, h, w, c, kHalf); }

  // Host floats -> the storage type, on the device.
  void store(const float* src, size_t n, Buffer& dst) {
    if (!kHalf) {
      dst.upload(src, n * sizeof(float));
      return;
    }
    std::vector<uint16_t> bits(n);
    for (size_t i = 0; i < n; ++i) bits[i] = float_to_half(src[i]);
    dst.upload(bits.data(), n * sizeof(uint16_t));
  }

  // Device copy of a weight tensor rearranged by `fill(dst)` into n values of type T (float or __half),
  // made on first use and kept. Weights outlive the backend (see jmt_ctx). `kind` separates different
  // arrangements of one tensor.
  template <typename T, typename Fill>
  const T* param(const Param& p, int kind, size_t n, Fill fill) {
    const auto key = std::make_pair(p.data, kind * 2 + static_cast<int>(sizeof(T) == sizeof(__half)));
    auto it = params_.find(key);
    if (it != params_.end()) return static_cast<const T*>(it->second);
    std::vector<float> values(n, 0.0f);
    fill(values.data());
    void* d = nullptr;
    JMT_CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(n, 4) * sizeof(T)));
    params_[key] = d;
    if (sizeof(T) == sizeof(float)) {
      JMT_CUDA_CHECK(cudaMemcpy(d, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    } else {
      std::vector<uint16_t> bits(n);
      for (size_t i = 0; i < n; ++i) bits[i] = float_to_half(values[i]);
      JMT_CUDA_CHECK(cudaMemcpy(d, bits.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    return static_cast<const T*>(d);
  }

  // A vector parameter (bias, gain) as T.
  template <typename T>
  const T* vec(const Param& p) {
    return param<T>(p, 0, p.size(), [&](float* dst) { std::memcpy(dst, p.data, p.size() * sizeof(float)); });
  }

  // The tiled matrix product: 8 x 8 values per thread; a block of 16 x 16 threads, or 8 x 32 when the
  // output has at most 64 channels (the values of attention).
  template <int MODE, typename X, typename W, typename O>
  void gemm(const X* x, const W* w, const float* bias, int* tops, Gemm a, O* out, int batches) {
    a.vec = a.o_row % 4 == 0 && a.o_batch % 4 == 0;
    // FP16: the tensor-core kernel, where its limits allow (attention calls it itself).
    if constexpr (MODE == kPlain && std::is_same<W, __half>::value && std::is_same<X, __half>::value) {
      if (tensor_fits(a)) return gemm_tc<kPlain>(x, w, bias, tops, a, out, batches);
    }
    const bool narrow = a.cout <= 64;
    const int bx = narrow ? 8 : 16, by = narrow ? 32 : 16;
    const dim3 grid((a.cout + bx * 8 - 1) / (bx * 8), (a.n + by * 8 - 1) / (by * 8), batches), block(bx * by);
    if (narrow && chunk_ == 8) launch_grid(k_gemm<X, W, O, 8, 8, 8, 8, 32, MODE>, grid, block, x, w, bias, tops, a, out);
    else if (narrow) launch_grid(k_gemm<X, W, O, 8, 8, 16, 8, 32, MODE>, grid, block, x, w, bias, tops, a, out);
    else if (chunk_ == 8) launch_grid(k_gemm<X, W, O, 8, 8, 8, 16, 16, MODE>, grid, block, x, w, bias, tops, a, out);
    else launch_grid(k_gemm<X, W, O, 8, 8, 16, 16, 16, MODE>, grid, block, x, w, bias, tops, a, out);
  }

  // Whether k_gemm_tc can run this product (see its limits).
  bool tensor_fits(const Gemm& a) const {
    const bool quads = a.x_row % 4 == 0 && a.x_batch % 4 == 0 && a.w_batch % 4 == 0 &&
                       (a.w_in == 1 ? a.w_out % 4 == 0 : a.w_out == 1 && a.w_in % 4 == 0 && a.cout % 4 == 0);
    return tensor_ && quads && a.cin % 32 == 0;
  }
  template <int MODE, typename O>
  void gemm_tc(const __half* x, const __half* w, const float* bias, int* tops, Gemm a, O* out, int batches) {
    a.vec = a.o_row % 4 == 0 && a.o_batch % 4 == 0;
    const bool narrow = a.cout <= 64;
    const int wr = narrow ? 4 : 2, wc = narrow ? 2 : 4;
    const dim3 grid((a.cout + wc * 32 - 1) / (wc * 32), (a.n + wr * 64 - 1) / (wr * 64), batches), block(256);
    // Columns per step: 16, or 32 for narrow outputs, where the loads of x dominate (measured).
    if (narrow) launch_grid(k_gemm_tc<__half, O, 32, 4, 2, MODE>, grid, block, x, w, bias, tops, a, out);
    else launch_grid(k_gemm_tc<__half, O, 16, 2, 4, MODE>, grid, block, x, w, bias, tops, a, out);
  }

  // Weights of a product as (in, out), or as (out, in) for the tensor-core kernel, which wants the
  // columns contiguous. at(o, i): the weight of input i for output o.
  bool tensor_product(int n, int in_c, int out_c) const {
    return kHalf && tensor_ && n >= 32 && out_c >= 32 && in_c % 32 == 0;
  }
  template <typename At>
  const R* matrix(const Param& p, int in_c, int out_c, bool by_output, At at) {
    return param<R>(p, by_output ? 2 : 1, static_cast<size_t>(in_c) * out_c, [&](float* dst) {
      for (int o = 0; o < out_c; ++o)
        for (int i = 0; i < in_c; ++i)
          dst[by_output ? static_cast<size_t>(o) * in_c + i : static_cast<size_t>(i) * out_c + o] = at(o, i);
    });
  }

  // out (h * w, out_c) = act(x (h * w, in_c) * W + bias) + residual; W as `matrix` made it.
  TensorPtr product(const R* x, int h, int w, int in_c, const R* weights, bool by_output, const Param* bias, int out_c,
                    int act, const R* residual) {
    Ptr out = make(h, w, out_c);
    const int n = h * w;
    const float* b = bias ? vec<float>(*bias) : nullptr;
    if (!by_output && (out_c < 32 || n < 32 || in_c % 16 != 0)) {
      launch(k_linear_small<R, R>, static_cast<long long>(n) * out_c, x, weights, b, n, in_c, out_c, 1.0f, act, residual,
             out->buf.template as<R>());
    } else {
      Gemm a{};
      a.n = n, a.cin = in_c, a.cout = out_c, a.scale = 1.0f;
      a.x_row = in_c, a.o_row = out_c;
      a.w_in = by_output ? 1 : out_c, a.w_out = by_output ? in_c : 1;
      a.act = act, a.residual = residual;
      gemm<kPlain>(x, weights, b, nullptr, a, out->buf.template as<R>(), 1);
    }
    return out;
  }

  TensorPtr unary(const TensorPtr& xp, int op, const char* name) {
    const CudaTensor& x = dev(xp);
    Ptr out = make(x.h, x.w, x.c);
    g_profile.tag("%s", name);
    const long long count = static_cast<long long>(x.size());
    launch(k_unary<R>, count, x.buf.template as<R>(), count, op, out->buf.template as<R>());
    return out;
  }

  PoolPtr pool_;
  std::map<std::pair<const float*, int>, void*> params_;
  struct Table {
    size_t n;
    Buffer buf;
    Table(const PoolPtr& pool, size_t n_) : n(n_), buf(pool, n_ * sizeof(float)) {}
  };
  std::map<const float*, std::unique_ptr<Table>> tables_;
  bool readout_device_ = true;  // JMT_CUDA_READOUT=0: the read-out's selection on host threads
  int flash_ = 16;              // JMT_CUDA_FLASH: warps per attention block; 0 = attention through the logits buffer
  bool split_ = true;           // JMT_CUDA_SPLIT=0: FP32 attention through the float logits buffer
  int chunk_ = 8, threads_ = 1;
  int tensor_ = 1;  // FP16: use the tensor cores (JMT_CUDA_TENSOR=0 turns them off)
};

}  // namespace

std::unique_ptr<Ops> make_cuda_ops(int device, bool fp16) {
  if (fp16) return std::make_unique<CudaOps<__half>>(device);
  return std::make_unique<CudaOps<float>>(device);
}

}  // namespace jmt
