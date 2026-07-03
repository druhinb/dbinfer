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

// one F16 transformer block's weights and hyperparameters, all pointers into
// the mmap or model scratch. offload runs the whole block in one command
// buffer, so every field the block reads is passed at once.
struct LayerF16 {
  const float *attn_norm = nullptr; // [dim] F32
  const float *ffn_norm = nullptr;  // [dim] F32
  const std::uint16_t *wq = nullptr;
  const std::uint16_t *wk = nullptr;
  const std::uint16_t *wv = nullptr;
  const std::uint16_t *wo = nullptr;
  const std::uint16_t *wgate = nullptr;
  const std::uint16_t *wup = nullptr;
  const std::uint16_t *wdown = nullptr;
  const float *bq = nullptr; // optional F32 bias, nullptr when absent
  const float *bk = nullptr;
  const float *bv = nullptr;
  std::size_t dim = 0;
  std::size_t ff = 0;
  std::size_t n_heads = 0;
  std::size_t n_kv_heads = 0;
  std::size_t head_dim = 0;
  float rms_eps = 0.0f;
  float rope_theta = 0.0f;
};

// kernel seam for the Metal backend. slice (a) offloads the batched f16 matmul;
// slice (b) adds rmsnorm, rope, attention, and a fused per-layer decode.
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

  // out = x / rms(x) * weight over each of rows rows of dim, eps inside the
  // sqrt. mirrors tensor::rmsnorm; sequential mean-of-squares matches it.
  [[nodiscard]] virtual std::expected<void, Error> rmsnorm(const float *x, const float *weight,
                                                           float eps, float *out, std::size_t rows,
                                                           std::size_t dim) = 0;

  // GPT-NeoX rotary embedding in place, pairing element j with j+head_dim/2 per
  // head. mirrors tensor::rope; cos/sin run in fp32 so results match to
  // tolerance, not bitwise.
  [[nodiscard]] virtual std::expected<void, Error> rope(float *x, const std::int32_t *positions,
                                                        float theta, std::size_t n_heads,
                                                        std::size_t seq, std::size_t head_dim) = 0;

  // causal attention for one query position over the dense fp32 KV cache slice
  // [n_positions][n_kv_heads][head_dim]. GQA broadcasts by index, scale is
  // 1/sqrt(head_dim). mirrors the dense path in Model::decode_layer.
  [[nodiscard]] virtual std::expected<void, Error>
  attention(const float *q, const float *k_cache, const float *v_cache, float *out,
            std::size_t n_positions, std::size_t n_heads, std::size_t n_kv_heads,
            std::size_t head_dim) = 0;

  // runs one F16 block on x in place under a single command buffer: rmsnorm,
  // QKV, bias, rope, attention over (k_hist ++ this position), output proj,
  // residual, then the FFN half. k_hist/v_hist hold the dense cache positions
  // [0, pos); k_out/v_out receive this position's post-rope k and raw v
  // ([n_kv_heads*head_dim] each) for the caller to append.
  [[nodiscard]] virtual std::expected<void, Error>
  decode_layer(const LayerF16 &layer, float *x, std::int32_t pos, const float *k_hist,
               const float *v_hist, float *k_out, float *v_out) = 0;
};

// resolves once from DBINFER_BACKEND. nullptr selects the CPU reference path,
// which is the default and stays bitwise unchanged.
Backend *active_backend();

} // namespace dbinfer::backend

#endif // DBINFER_BACKEND_BACKEND_HPP
