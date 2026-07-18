#ifndef DBINFER_TENSOR_MATMUL_HPP
#define DBINFER_TENSOR_MATMUL_HPP

#include <cstddef>
#include <cstdint>

namespace dbinfer::tensor {

// y = W @ x for row-major W of shape [out, in]: each output element is the
// dot product of x with one row of W.
void matvec(const float *W, const float *x, float *y, std::size_t out, std::size_t in);

// same as matvec, but W is stored as F16 and dequantized one element at a
// time in the inner loop rather than converted up front.
void matvec_f16(const std::uint16_t *W, const float *x, float *y, std::size_t out, std::size_t in);

// batched matvec: C = A @ W^T for m rows of A ([m, in]) against the same
// row-major weight W ([out, in]), writing C as [m, out].
void matmul(const float *A, const float *W, float *C, std::size_t m, std::size_t out,
            std::size_t in);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_MATMUL_HPP
