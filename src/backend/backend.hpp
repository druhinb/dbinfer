#ifndef DBINFER_BACKEND_BACKEND_HPP
#define DBINFER_BACKEND_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>

namespace dbinfer::backend {

struct Error {
  std::string message;
};

// weight dtypes the Metal decode path can read. Q5_0/Q4_K/Q6_K stay CPU-only.
enum class WeightType : std::uint8_t { F16, Q8_0, Q4_0 };

// non-owning view of one packed weight matrix plus its dtype, for the offload
// decode. data points into the model mmap.
struct QWeight {
  const std::byte *data = nullptr;
  WeightType type = WeightType::F16;
};

// one transformer block's weights and hyperparameters for the offload decode.
// the matmul weights carry their dtype so the device can run F16, Q8_0, or Q4_0
// blocks. pointers view the mmap or model scratch.
struct LayerQ {
  const float *attn_norm = nullptr;
  const float *ffn_norm = nullptr;
  QWeight wq{};
  QWeight wk{};
  QWeight wv{};
  QWeight wo{};
  QWeight wgate{};
  QWeight wup{};
  QWeight wdown{};
  const float *bq = nullptr;
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

  // quantizes rows rows of in fp32 activations into Q8_0 blocks (BlockQ8_0
  // layout, 34 bytes/block), bit-exact to tensor::quantize_row_q8_0. out holds
  // rows*(in/32) blocks. in must be a multiple of 32.
  [[nodiscard]] virtual std::expected<void, Error>
  quantize_q8_0(const float *x, std::size_t rows, std::size_t in, std::byte *out) = 0;

  // C[m, out] = A[m, in] x W[out, in]^T for Q8_0 weights, quantizing each A row
  // to Q8_0 on the device, then the same integer block dot and sequential fp32
  // block accumulation as tensor::matvec_q8_0_scalar. bitwise identical to it.
  [[nodiscard]] virtual std::expected<void, Error> mul_mat_q8_0(const std::byte *W, const float *A,
                                                                float *C, std::size_t m,
                                                                std::size_t out,
                                                                std::size_t in) = 0;

  // same as mul_mat_q8_0 for Q4_0 weights, bitwise identical to
  // tensor::matvec_q4_0_scalar (activations still quantized to Q8_0).
  [[nodiscard]] virtual std::expected<void, Error> mul_mat_q4_0(const std::byte *W, const float *A,
                                                                float *C, std::size_t m,
                                                                std::size_t out,
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

  // copies each offloaded block's weights to the device once and wraps the dense
  // fp32 KV cache no-copy. k_base/v_base point at the page-aligned
  // [n_layers][capacity][pos_stride] arrays; layers cover the first layers.size()
  // blocks. output_norm and lm_head feed the on-device final norm and logits
  // projection used by decode_token_full; lm_head has vocab_size rows of dim.
  // must be called before decode_token and re-called if the model or cache
  // layout changes.
  [[nodiscard]] virtual std::expected<void, Error>
  setup_offload(std::span<const LayerQ> layers, float *k_base, float *v_base, std::size_t capacity,
                std::size_t pos_stride, const float *output_norm, QWeight lm_head,
                std::size_t vocab_size) = 0;

  // runs every prepared block of one token in a single command buffer: rmsnorm,
  // QKV, bias, rope, cache append into the bound KV, attention, output proj,
  // residual, FFN. x is the residual stream [dim], updated in place. reads and
  // writes the bound KV at slot pos of each block. one waitUntilCompleted.
  [[nodiscard]] virtual std::expected<void, Error> decode_token(float *x, std::int32_t pos) = 0;

  // runs every prepared block, the final rmsnorm, and the lm_head projection on
  // one command buffer. x is the input embedding row [dim]; the returned pointer
  // views the shared logits buffer [vocab_size], valid until the next call. use
  // only when every block is offloaded (no CPU blocks run after). writes the
  // bound KV at slot pos. one waitUntilCompleted.
  [[nodiscard]] virtual std::expected<const float *, Error> decode_token_full(const float *x,
                                                                              std::int32_t pos) = 0;
};

// resolves once from DBINFER_BACKEND. nullptr selects the CPU reference path,
// which is the default and stays bitwise unchanged.
Backend *active_backend();

} // namespace dbinfer::backend

#endif // DBINFER_BACKEND_BACKEND_HPP
