#include "tensor/dequant.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
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

std::uint16_t f32_to_f16(float f) {
  const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
  const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t exp_bits = (x >> 23) & 0xFFu;
  std::uint32_t mant = x & 0x7FFFFFu;

  if (exp_bits == 0xFFu)
    return sign | (mant != 0 ? 0x7E00u : 0x7C00u);

  const std::int32_t exp = static_cast<std::int32_t>(exp_bits) - 127 + 15;

  if (exp >= 0x1F)
    return sign | 0x7C00u;

  if (exp <= 0) {
    if (exp < -10)
      return sign;
    // restore the implicit leading one, then shift into subnormal position.
    mant |= 0x800000u;
    const std::int32_t shift = 14 - exp;
    std::uint32_t q = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (q & 1u)))
      ++q;
    return sign | static_cast<std::uint16_t>(q);
  }

  std::uint16_t result =
      sign | static_cast<std::uint16_t>(exp << 10) | static_cast<std::uint16_t>(mant >> 13);
  const std::uint32_t rem = mant & 0x1FFFu;
  // carry from a round-up propagates into the exponent field, up to inf.
  if (rem > 0x1000u || (rem == 0x1000u && (result & 1u)))
    ++result;
  return result;
}

void dequant_row_q8_0(const std::byte *block_base, std::size_t in, float *out) {
  const std::size_t nblocks = in / kBlockSize;
  for (std::size_t b = 0; b < nblocks; ++b) {
    const std::byte *blk = block_base + b * sizeof(BlockQ8_0);
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, blk, sizeof(d_bits));
    const float d = f16_to_f32(d_bits);
    for (std::size_t i = 0; i < kBlockSize; ++i)
      out[b * kBlockSize + i] = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + i]));
  }
}

void quantize_row_q8_0(const float *x, std::size_t n, std::byte *out) {
  const std::size_t nblocks = n / kBlockSize;
  for (std::size_t b = 0; b < nblocks; ++b) {
    const float *xb = x + b * kBlockSize;
    float amax = 0.0f;
    for (std::size_t i = 0; i < kBlockSize; ++i)
      amax = std::max(amax, std::fabs(xb[i]));
    // ggml derives id from the unrounded scale, then stores d as fp16.
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const std::uint16_t d_bits = f32_to_f16(d);
    std::byte *blk = out + b * sizeof(BlockQ8_0);
    std::memcpy(blk, &d_bits, sizeof(d_bits));
    for (std::size_t i = 0; i < kBlockSize; ++i) {
      const long q = std::clamp(std::lround(xb[i] * id), -127L, 127L);
      blk[2 + i] = static_cast<std::byte>(static_cast<std::int8_t>(q));
    }
  }
}

void dequant_row_q4_0(const std::byte *block_base, std::size_t in, float *out) {
  const std::size_t nblocks = in / kBlockSize;
  for (std::size_t b = 0; b < nblocks; ++b) {
    const std::byte *blk = block_base + b * sizeof(BlockQ4_0);
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, blk, sizeof(d_bits));
    const float d = f16_to_f32(d_bits);
    for (std::size_t j = 0; j < 16; ++j) {
      const std::uint8_t q = static_cast<std::uint8_t>(blk[2 + j]);
      out[b * kBlockSize + j] = d * (static_cast<float>(q & 0x0Fu) - 8.0f);
      out[b * kBlockSize + j + 16] = d * (static_cast<float>(q >> 4) - 8.0f);
    }
  }
}

void dequant_row(QuantMatrix w, std::size_t row, std::size_t in, float *out) {
  switch (w.type) {
  case gguf::GgmlType::F32: {
    // tensor data is aligned to the gguf tensor alignment, so a float view is
    // well-defined here.
    const float *r = reinterpret_cast<const float *>(w.data) + row * in;
    std::copy(r, r + in, out);
    return;
  }
  case gguf::GgmlType::F16: {
    const std::uint16_t *r = reinterpret_cast<const std::uint16_t *>(w.data) + row * in;
    for (std::size_t i = 0; i < in; ++i)
      out[i] = f16_to_f32(r[i]);
    return;
  }
  case gguf::GgmlType::Q8_0:
    dequant_row_q8_0(w.data + row * (in / kBlockSize) * sizeof(BlockQ8_0), in, out);
    return;
  case gguf::GgmlType::Q4_0:
    dequant_row_q4_0(w.data + row * (in / kBlockSize) * sizeof(BlockQ4_0), in, out);
    return;
  default:
    __builtin_unreachable(); // load validated w.type is one of the above
  }
}

} // namespace dbinfer::tensor
