#include "tensor/dequant.hpp"

#include <cmath>
#include <limits>

namespace dbinfer::tensor {

float f16_to_f32(std::uint16_t h) {
  // IEEE half: [sign:1][exp:5][mant:10], bias 15. Reconstruct by exponent case:
  // exp==0 -> ±0 (mant 0) or subnormal mant*2^-24; exp==31 -> ±inf (mant 0) or
  // NaN; otherwise normal (1+mant/1024)*2^(exp-15). Powers of two and the
  // mantissa fractions are exact in f32, so each branch is exact.
  const std::uint32_t sign = (h >> 15) & 0x1u;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;
  const float s = sign ? -1.0f : 1.0f;

  if (exp == 0) {
    if (mant == 0)
      return s * 0.0f;
    return s * (static_cast<float>(mant) * (1.0f / 16777216.0f)); // 2^-24
  }
  if (exp == 31) {
    if (mant == 0)
      return s * std::numeric_limits<float>::infinity();
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float frac = 1.0f + static_cast<float>(mant) * (1.0f / 1024.0f);
  return s * std::ldexp(frac, static_cast<int>(exp) - 15);
}

} // namespace dbinfer::tensor
