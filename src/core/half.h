// IEEE 754 binary16 <-> binary32 on the host, for FP16 weights and for reading FP16 tensors back.
#pragma once

#include <cstdint>
#include <cstring>

namespace jmt {

// Round to nearest, ties to even; overflow gives infinity.
inline uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  x &= 0x7fffffffu;
  if (x > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u);   // NaN
  if (x >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // >= 65520 rounds to infinity
  if (x <= 0x33000000u) return static_cast<uint16_t>(sign);            // <= 2^-25 rounds to zero
  const int e = static_cast<int>(x >> 23) - 127;
  const uint32_t m = (x & 0x7fffffu) | 0x800000u;
  const int shift = e < -14 ? 13 + (-14 - e) : 13;  // mantissa bits dropped (more for subnormals)
  uint32_t h = m >> shift;
  const uint32_t rem = m & ((1u << shift) - 1), halfway = 1u << (shift - 1);
  if (rem > halfway || (rem == halfway && (h & 1u))) ++h;
  // h carries the implicit bit (0x400) for normal numbers, so a mantissa overflow moves to the exponent.
  if (e >= -14) h += static_cast<uint32_t>(e + 14) << 10;
  return static_cast<uint16_t>(sign | h);
}

inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 31u, m = h & 0x3ffu, x;
  if (e == 0) {
    if (m == 0) {
      x = sign;
    } else {  // subnormal: m * 2^-24
      uint32_t s = 0;
      for (; !(m & 0x400u); ++s) m <<= 1;
      x = sign | ((113u - s) << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (e == 31) {
    x = sign | 0x7f800000u | (m << 13);
  } else {
    x = sign | ((e + 112u) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

}  // namespace jmt
