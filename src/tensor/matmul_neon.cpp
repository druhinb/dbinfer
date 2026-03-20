#include "tensor/matmul_neon.hpp"

#include "tensor/dequant.hpp"

#include <arm_neon.h>
#include <cstdint>
#include <cstring>

namespace dbinfer::tensor {

namespace {

std::uint16_t load_d(const std::byte *blk) {
  std::uint16_t d = 0;
  std::memcpy(&d, blk, sizeof(d));
  return d;
}

std::int32_t dot_q8_block(const std::int8_t *wq, const std::int8_t *xq) {
  int32x4_t p = vdotq_s32(vdupq_n_s32(0), vld1q_s8(wq), vld1q_s8(xq));
  p = vdotq_s32(p, vld1q_s8(wq + 16), vld1q_s8(xq + 16));
  return vaddvq_s32(p);
}

std::int32_t dot_q4_block(const std::uint8_t *packed, const std::int8_t *xq) {
  const uint8x16_t v = vld1q_u8(packed);
  const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
  const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), vdupq_n_s8(8));
  int32x4_t p = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(xq));
  p = vdotq_s32(p, hi, vld1q_s8(xq + 16));
  return vaddvq_s32(p);
}

} // namespace

void matvec_q8_0_sdot(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                      std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  const std::size_t row_bytes = nb * sizeof(BlockQ8_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte *blk = row + b * sizeof(BlockQ8_0);
      // weight quants sit unaligned in the mmap, neon vld1 permits that.
      const std::int32_t sumi =
          dot_q8_block(reinterpret_cast<const std::int8_t *>(blk + 2), xq[b].qs);
      acc += f16_to_f32(load_d(blk)) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

void matvec_q4_0_sdot(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                      std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  const std::size_t row_bytes = nb * sizeof(BlockQ4_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte *blk = row + b * sizeof(BlockQ4_0);
      const std::int32_t sumi =
          dot_q4_block(reinterpret_cast<const std::uint8_t *>(blk + 2), xq[b].qs);
      acc += f16_to_f32(load_d(blk)) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

} // namespace dbinfer::tensor
