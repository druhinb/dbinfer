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

// dequantizes one weight row into out ([in]) per w.type (F32, F16, or Q8_0).
void dequant_row(QuantMatrix w, std::size_t row, std::size_t in, float *out);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_DEQUANT_HPP
