// Sizes for the GPU backends' buffer pools.
//
// A pool hands a released buffer out again only for a request of the same size. FlexGlue's tensors have
// the same sizes for every pair, but LightGlue's change with the keypoints and with every point dropped
// (adaptive width), so exact sizes would keep a new set of buffers for almost every pair: memory grew
// without bound over a project's pairs (64 GB of footprint for 5,400 LighterGlue pairs on Metal). Sizes
// are therefore rounded up to one of eight steps per power of two (at most 12.5 % larger), and the idle
// buffers a pool keeps are capped.
#pragma once

#include <cstddef>

namespace jmt {

inline size_t pool_bucket(size_t bytes) {
  if (bytes <= 4096) return (bytes + 255) / 256 * 256;
  size_t top = 1;
  while (top < bytes) top <<= 1;  // bytes in (top / 2, top]
  const size_t step = top / 16;   // eight steps between top / 2 and top
  return (bytes + step - 1) / step * step;
}

// Idle bytes a pool keeps; beyond it the largest idle buffers are let go.
constexpr size_t kPoolIdleBytes = size_t(1) << 30;

}  // namespace jmt
