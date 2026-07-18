#ifndef DBINFER_TENSOR_OPS_HPP
#define DBINFER_TENSOR_OPS_HPP

#include <cstddef>
#include <cstdint>

namespace dbinfer::tensor {

// RMSNorm over each of rows rows of dim: out = x / rms(x) * weight, with eps
// inside the sqrt for numerical stability at rms ~ 0.
void rmsnorm(const float *x, const float *weight, float eps, float *out, std::size_t rows,
             std::size_t dim);

// applies rotary position embedding in place, GPT-NeoX style: within each
// head_dim-sized head, pairs element j with element j+head_dim/2 (not
// adjacent pairs) and rotates by an angle that depends on position and a
// per-dimension frequency derived from theta. positions has one entry per
// sequence element (seq), broadcast across all n_heads.
void rope(float *x, const std::int32_t *positions, float theta, std::size_t n_heads,
          std::size_t seq, std::size_t head_dim);

// numerically stable softmax: subtracts the max logit before exponentiating
// so the sum doesn't overflow for large inputs.
void softmax(const float *in, float *out, std::size_t n);

void silu(const float *in, float *out, std::size_t n);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_OPS_HPP
