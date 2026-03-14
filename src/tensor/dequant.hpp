#ifndef DBINFER_TENSOR_DEQUANT_HPP
#define DBINFER_TENSOR_DEQUANT_HPP

#include "tensor/matmul.hpp"

#include <cstddef>
#include <cstdint>

namespace dbinfer::tensor {

// converts an IEEE-754 binary16 bit pattern to float32, exactly (including
// zero, subnormals, infinities, and NaN).
float f16_to_f32(std::uint16_t h);

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

// dequantizes one weight row into out ([in]) per w.type (F32, F16, Q8_0, Q4_0).
void dequant_row(QuantMatrix w, std::size_t row, std::size_t in, float *out);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_DEQUANT_HPP
