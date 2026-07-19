#ifndef DBINFER_TENSOR_MATMUL_HPP
#define DBINFER_TENSOR_MATMUL_HPP

#include "gguf/gguf.hpp"

#include <cstddef>
#include <cstdint>

namespace dbinfer::tensor {

// y = W @ x for row-major W of shape [out, in]: each output element is the
// dot product of x with one row of W.
void matvec(const float *W, const float *x, float *y, std::size_t out, std::size_t in);

// same as matvec, but W is stored as F16 and dequantized one element at a
// time in the inner loop rather than converted up front.
void matvec_f16(const std::uint16_t *W, const float *x, float *y, std::size_t out, std::size_t in);

// non-owning view of a weight matrix in the mmap, tagged with its ggml dtype
// so matvec_quant dispatches without materializing a float copy.
struct QuantMatrix {
  const std::byte *data;
  gguf::GgmlType type;
};

// same as matvec, but W is Q8_0-packed row-major [out, in], dequantized one
// block at a time in the inner loop. in must be a multiple of 32.
void matvec_q8_0(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in);

// same as matvec, but W is Q4_0-packed row-major [out, in], dequantized one
// block at a time in the inner loop. in must be a multiple of 32.
void matvec_q4_0(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in);

// y = W @ x dispatching on w.type (F32, F16, Q8_0, or Q4_0).
void matvec_quant(QuantMatrix w, const float *x, float *y, std::size_t out, std::size_t in);

// batched matvec: C = A @ W^T for m rows of A ([m, in]) against the same
// row-major weight W ([out, in]), writing C as [m, out].
void matmul(const float *A, const float *W, float *C, std::size_t m, std::size_t out,
            std::size_t in);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_MATMUL_HPP
