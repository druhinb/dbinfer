#ifndef DBINFER_TENSOR_MATMUL_NEON_HPP
#define DBINFER_TENSOR_MATMUL_NEON_HPP

#include <cstddef>

#include "tensor/dequant.hpp"

namespace dbinfer::tensor {

// matvec over pre-quantized Q8_0 activations xq (in/32 blocks), one weight
// dtype per function. matches the matvec_q8_0_scalar / matvec_q4_0_scalar
// reference (matmul.hpp) to 1e-6; reached only after cpu_features().dotprod
// is set.
void matvec_q8_0_sdot(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in);
void matvec_q4_0_sdot(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in);

// i8mm variants dot two output rows per smmla tile; reached only after
// cpu_features().i8mm is set. an odd final row falls back to the sdot path.
void matvec_q8_0_i8mm(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in);
void matvec_q4_0_i8mm(const std::byte* W, const BlockQ8_0* xq, float* y, std::size_t out,
                      std::size_t in);

}  // namespace dbinfer::tensor

#endif  // DBINFER_TENSOR_MATMUL_NEON_HPP
