#ifndef DBINFER_TENSOR_DEQUANT_HPP
#define DBINFER_TENSOR_DEQUANT_HPP

#include "tensor/matmul.hpp"

#include <cstddef>
#include <cstdint>

namespace dbinfer::tensor {

// converts an IEEE-754 binary16 bit pattern to float32, exactly (including
// zero, subnormals, infinities, and NaN).
float f16_to_f32(std::uint16_t h);

// inverse of f16_to_f32: rounds a float32 to binary16 bits, round to nearest
// even, subnormals preserved, out-of-range magnitude to infinity.
std::uint16_t f32_to_f16(float f);

constexpr std::size_t kBlockSize = 32;

// ggml Q8_0 block: fp16 scale d then 32 int8 quants. the 34-byte stride
// leaves d unaligned in the mmap, so never load through a BlockQ8_0 view.
struct BlockQ8_0 {
  std::uint16_t d;
  std::int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34);

// dequantizes in values of one Q8_0 row (in/32 blocks from block_base) into
// out: out[i] = f16_to_f32(d) * qs[i]. in must be a multiple of 32.
void dequant_row_q8_0(const std::byte *block_base, std::size_t in, float *out);

// quantizes n activations into n/32 Q8_0 blocks at out. n must be a multiple
// of 32. block d is stored fp16 so the vec dot scale matches ggml at parity.
void quantize_row_q8_0(const float *x, std::size_t n, std::byte *out);

// ggml Q4_0 block: fp16 scale d then 32 4-bit quants packed two per byte. the
// 18-byte stride leaves d unaligned in the mmap, so never load through a view.
struct BlockQ4_0 {
  std::uint16_t d;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18);

// dequantizes in values of one Q4_0 row (in/32 blocks from block_base) into
// out: within each byte the low nibble is element j and the high nibble is
// element j+16, both mapped as d*(nibble-8). in must be a multiple of 32.
void dequant_row_q4_0(const std::byte *block_base, std::size_t in, float *out);

// ggml Q5_0 block: fp16 scale d, a 32-bit fifth-bit field qh, then 32 low
// 4-bit quants packed two per byte. d and qh sit unaligned in the mmap.
struct BlockQ5_0 {
  std::uint16_t d;
  std::uint8_t qh[4];
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ5_0) == 22);

// dequantizes in values of one Q5_0 row (in/32 blocks from block_base) into
// out: each 5-bit value is the low nibble plus its fifth bit from qh, mapped
// as d*(value-16). in must be a multiple of 32.
void dequant_row_q5_0(const std::byte *block_base, std::size_t in, float *out);

constexpr std::size_t kSuperBlockSize = 256;

// ggml Q4_K super-block: fp16 d, fp16 dmin, 12 packed 6-bit scale/min bytes,
// then 256 4-bit quants two per byte. d and dmin sit unaligned in the mmap.
struct BlockQ4_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144);

// ggml Q6_K super-block: 256 lower 4-bit quants, 256 upper 2-bit quants, 16
// signed int8 sub-block scales, then fp16 d unaligned in the mmap.
struct BlockQ6_K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210);

// dequantizes in values of one Q4_K row (in/256 super-blocks from block_base)
// into out. in must be a multiple of 256.
void dequant_row_q4_k(const std::byte *block_base, std::size_t in, float *out);

// dequantizes in values of one Q6_K row (in/256 super-blocks from block_base)
// into out. in must be a multiple of 256.
void dequant_row_q6_k(const std::byte *block_base, std::size_t in, float *out);

// dequantizes one weight row into out ([in]) per w.type.
void dequant_row(QuantMatrix w, std::size_t row, std::size_t in, float *out);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_DEQUANT_HPP
