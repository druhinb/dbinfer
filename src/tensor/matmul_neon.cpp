#include "tensor/matmul_neon.hpp"

#include <arm_neon.h>

#include <cstdint>
#include <cstring>

#include "tensor/dequant.hpp"

namespace dbinfer::tensor {

namespace {

std::uint16_t load_d(const std::byte* blk) {
  std::uint16_t d = 0;
  std::memcpy(&d, blk, sizeof(d));
  return d;
}

std::int32_t dot_q8_block(const std::int8_t* wq, const std::int8_t* xq) {
  int32x4_t p = vdotq_s32(vdupq_n_s32(0), vld1q_s8(wq), vld1q_s8(xq));
  p = vdotq_s32(p, vld1q_s8(wq + 16), vld1q_s8(xq + 16));
  return vaddvq_s32(p);
}

std::int32_t dot_q4_block(const std::uint8_t* packed, const std::int8_t* xq) {
  const uint8x16_t v = vld1q_u8(packed);
  const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
  const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), vdupq_n_s8(8));
  int32x4_t p = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(xq));
  p = vdotq_s32(p, hi, vld1q_s8(xq + 16));
  return vaddvq_s32(p);
}

void unpack_q4_block(const std::uint8_t* packed, std::int8_t* out) {
  for (std::size_t j = 0; j < 16; ++j) {
    out[j] = static_cast<std::int8_t>(static_cast<std::int32_t>(packed[j] & 0x0Fu) - 8);
    out[j + 16] = static_cast<std::int8_t>(static_cast<std::int32_t>(packed[j] >> 4) - 8);
  }
}

// smmla forms a 2x2 int32 tile from a 2x8 by 2x8 product; lane 0 is row0 dotted
// with the activation, lane 3 is row1. the activation duplicates into both b rows.
void dot2_block(const std::int8_t* w0, const std::int8_t* w1, const std::int8_t* x,
                std::int32_t& s0, std::int32_t& s1) {
  int32x4_t acc = vdupq_n_s32(0);
  for (std::size_t c = 0; c < kBlockSize; c += 8) {
    const int8x16_t a = vcombine_s8(vld1_s8(w0 + c), vld1_s8(w1 + c));
    const int8x8_t xc = vld1_s8(x + c);
    acc = vmmlaq_s32(acc, a, vcombine_s8(xc, xc));
  }
  s0 = vgetq_lane_s32(acc, 0);
  s1 = vgetq_lane_s32(acc, 3);
}

}  // namespace

void matvec_q8_0_sdot(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  const std::size_t row_bytes = nb * sizeof(BlockQ8_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte* row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte* blk = row + b * sizeof(BlockQ8_0);
      // weight quants sit unaligned in the mmap, neon vld1 permits that.
      const std::int32_t sumi =
          dot_q8_block(reinterpret_cast<const std::int8_t*>(blk + 2), xq[b].qs);
      acc += f16_to_f32(load_d(blk)) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

void matvec_q4_0_sdot(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  const std::size_t row_bytes = nb * sizeof(BlockQ4_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte* row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte* blk = row + b * sizeof(BlockQ4_0);
      const std::int32_t sumi =
          dot_q4_block(reinterpret_cast<const std::uint8_t*>(blk + 2), xq[b].qs);
      acc += f16_to_f32(load_d(blk)) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

void matvec_q8_0_i8mm(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  const std::size_t row_bytes = nb * sizeof(BlockQ8_0);
  std::size_t o = 0;
  for (; o + 1 < out; o += 2) {
    const std::byte* r0 = W + o * row_bytes;
    const std::byte* r1 = W + (o + 1) * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte* b0 = r0 + b * sizeof(BlockQ8_0);
      const std::byte* b1 = r1 + b * sizeof(BlockQ8_0);
      std::int32_t s0 = 0;
      std::int32_t s1 = 0;
      dot2_block(reinterpret_cast<const std::int8_t*>(b0 + 2),
                 reinterpret_cast<const std::int8_t*>(b1 + 2), xq[b].qs, s0, s1);
      const float dx = f16_to_f32(xq[b].d);
      acc0 += f16_to_f32(load_d(b0)) * dx * static_cast<float>(s0);
      acc1 += f16_to_f32(load_d(b1)) * dx * static_cast<float>(s1);
    }
    y[o] = acc0;
    y[o + 1] = acc1;
  }
  if (o < out) {
    const std::byte* row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte* blk = row + b * sizeof(BlockQ8_0);
      const std::int32_t sumi =
          dot_q8_block(reinterpret_cast<const std::int8_t*>(blk + 2), xq[b].qs);
      acc += f16_to_f32(load_d(blk)) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

void matvec_q4_0_i8mm(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  const std::size_t row_bytes = nb * sizeof(BlockQ4_0);
  std::size_t o = 0;
  for (; o + 1 < out; o += 2) {
    const std::byte* r0 = W + o * row_bytes;
    const std::byte* r1 = W + (o + 1) * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte* b0 = r0 + b * sizeof(BlockQ4_0);
      const std::byte* b1 = r1 + b * sizeof(BlockQ4_0);
      std::int8_t w0[kBlockSize];
      std::int8_t w1[kBlockSize];
      unpack_q4_block(reinterpret_cast<const std::uint8_t*>(b0 + 2), w0);
      unpack_q4_block(reinterpret_cast<const std::uint8_t*>(b1 + 2), w1);
      std::int32_t s0 = 0;
      std::int32_t s1 = 0;
      dot2_block(w0, w1, xq[b].qs, s0, s1);
      const float dx = f16_to_f32(xq[b].d);
      acc0 += f16_to_f32(load_d(b0)) * dx * static_cast<float>(s0);
      acc1 += f16_to_f32(load_d(b1)) * dx * static_cast<float>(s1);
    }
    y[o] = acc0;
    y[o + 1] = acc1;
  }
  if (o < out) {
    const std::byte* row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::byte* blk = row + b * sizeof(BlockQ4_0);
      const std::int32_t sumi =
          dot_q4_block(reinterpret_cast<const std::uint8_t*>(blk + 2), xq[b].qs);
      acc += f16_to_f32(load_d(blk)) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

}  // namespace dbinfer::tensor
