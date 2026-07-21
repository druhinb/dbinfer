#ifndef DBINFER_TENSOR_MATMUL_HPP
#define DBINFER_TENSOR_MATMUL_HPP

#include <cstddef>
#include <cstdint>

#include "gguf/gguf.hpp"

namespace dbinfer::tensor {

// y = W @ x for row-major W of shape [out, in]: each output element is the
// dot product of x with one row of W.
void matvec(const float* W, const float* x, float* y, std::size_t out, std::size_t in);

// same as matvec, but W is stored as F16 and dequantized one element at a
// time in the inner loop rather than converted up front.
void matvec_f16(const std::uint16_t* W, const float* x, float* y, std::size_t out, std::size_t in);

// non-owning view of a weight matrix in the mmap, tagged with its ggml dtype
// so matvec_quant dispatches without materializing a float copy.
struct QuantMatrix {
  const std::byte* data;
  gguf::GgmlType type;
};

// same as matvec, but W is Q8_0-packed row-major [out, in], dequantized one
// block at a time in the inner loop. in must be a multiple of 32.
void matvec_q8_0(const std::byte* W, const float* x, float* y, std::size_t out, std::size_t in);

// same as matvec, but W is Q4_0-packed row-major [out, in], dequantized one
// block at a time in the inner loop. in must be a multiple of 32.
void matvec_q4_0(const std::byte* W, const float* x, float* y, std::size_t out, std::size_t in);

// same as matvec, but W is Q5_0-packed row-major [out, in], dequantized one
// block at a time into fp32 scratch. in must be a multiple of 32.
void matvec_q5_0(const std::byte* W, const float* x, float* y, std::size_t out, std::size_t in);

// same as matvec, but W is Q4_K-packed row-major [out, in], dequantized one
// super-block at a time into fp32 scratch. in must be a multiple of 256.
void matvec_q4_k(const std::byte* W, const float* x, float* y, std::size_t out, std::size_t in);

// same as matvec, but W is Q6_K-packed row-major [out, in], dequantized one
// super-block at a time into fp32 scratch. in must be a multiple of 256.
void matvec_q6_k(const std::byte* W, const float* x, float* y, std::size_t out, std::size_t in);

// y = W @ x dispatching on w.type (F32, F16, Q8_0, Q4_0, Q4_K, or Q6_K).
void matvec_quant(QuantMatrix w, const float* x, float* y, std::size_t out, std::size_t in);

// one matvec in a fused batch: weight w ([out, in]) times the shared x, into y.
struct MatvecJob {
  QuantMatrix w;
  float* y;
  std::size_t out;
};

// several independent matvecs that share x and in (e.g. the Q/K/V or gate/up
// projections), run under a single thread-pool barrier instead of one each.
// x is quantized at most once. Bitwise-identical to calling matvec_quant per
// job; only the parallelization boundary changes.
void matvec_quant_fused(const MatvecJob* jobs, std::size_t njobs, const float* x, std::size_t in);

// batched matvec: C = A @ W^T for m rows of A ([m, in]) against the same
// row-major weight W ([out, in]), writing C as [m, out].
void matmul(const float* A, const float* W, float* C, std::size_t m, std::size_t out,
            std::size_t in);

// batched projection for a chunk of tokens: C = A @ w^T for the m rows of A,
// reducing each (row, output) over in through the per-row matvec_quant kernel.
// every row of C is bitwise identical to matvec_quant(w, A + r * in, ...), so a
// chunk matches per-token prefill to the bit.
void matmul_quant(QuantMatrix w, const float* A, float* C, std::size_t m, std::size_t out,
                  std::size_t in);

// batched prefill: C = A @ W^T with W dequantized from w.type once, then one
// cblas_sgemm call. reduces reordering vs the scalar matmul, so results match
// only within GEMM tolerance. off Apple this falls back to the scalar path.
void matmul_accel(const float* A, QuantMatrix w, float* C, std::size_t m, std::size_t out,
                  std::size_t in);

}  // namespace dbinfer::tensor

#endif  // DBINFER_TENSOR_MATMUL_HPP
