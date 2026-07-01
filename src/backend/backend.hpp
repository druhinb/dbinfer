#ifndef DBINFER_BACKEND_BACKEND_HPP
#define DBINFER_BACKEND_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

namespace dbinfer::backend {

struct Error {
  std::string message;
};

// kernel seam for Phase 11 slice (a). the only offloaded op is the batched f16
// matmul; the interface grows in slice (b) with rmsnorm, rope, and attention.
class Backend {
public:
  Backend() = default;
  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;
  Backend(Backend &&) = delete;
  Backend &operator=(Backend &&) = delete;
  virtual ~Backend() = default;

  // C[m, out] = A[m, in] x W[out, in]^T, W stored as F16, A and C fp32. matches
  // tensor::matmul_quant's F16 path within GPU reduction tolerance. an error
  // means the caller falls back to the CPU kernel.
  [[nodiscard]] virtual std::expected<void, Error> mul_mat_f16(const std::uint16_t *W,
                                                               const float *A, float *C,
                                                               std::size_t m, std::size_t out,
                                                               std::size_t in) = 0;
};

// resolves once from DBINFER_BACKEND. nullptr selects the CPU reference path,
// which is the default and stays bitwise unchanged.
Backend *active_backend();

} // namespace dbinfer::backend

#endif // DBINFER_BACKEND_BACKEND_HPP
