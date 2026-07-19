#ifndef DBINFER_TENSOR_MATMUL_NEON_HPP
#define DBINFER_TENSOR_MATMUL_NEON_HPP

#include "tensor/dequant.hpp"

#include <cstddef>

namespace dbinfer::tensor {

// matvec over pre-quantized Q8_0 activations xq (in/32 blocks), one weight
// dtype per function. the scalar pair is the 1e-6 reference the simd kernels
// match; the sdot pair is reached only after cpu_features().dotprod is set.
void matvec_q8_0_scalar(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                        std::size_t in);
void matvec_q4_0_scalar(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                        std::size_t in);

void matvec_q8_0_sdot(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                      std::size_t in);
void matvec_q4_0_sdot(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                      std::size_t in);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_MATMUL_NEON_HPP
