#ifndef DBINFER_MODEL_MODEL_HPP
#define DBINFER_MODEL_MODEL_HPP

#include "gguf/gguf.hpp"
#include "tensor/matmul.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

namespace dbinfer::model {

struct Config {
  std::size_t n_layers = 0;
  std::size_t embedding_length = 0;
  std::size_t n_heads = 0;
  std::size_t n_kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t gqa_factor = 0;
  std::size_t ffn_length = 0;
  std::size_t vocab_size = 0;
  std::size_t context_length = 0;
  float rope_theta = 0.0f;
  float rms_eps = 0.0f;
  bool tied_embeddings = false;
};

struct LayerWeights {
  const float *attn_norm = nullptr;
  tensor::QuantMatrix attn_q{};
  tensor::QuantMatrix attn_k{};
  tensor::QuantMatrix attn_v{};
  tensor::QuantMatrix attn_output{};
  const float *attn_q_bias = nullptr;
  const float *attn_k_bias = nullptr;
  const float *attn_v_bias = nullptr;
  const float *ffn_norm = nullptr;
  tensor::QuantMatrix ffn_gate{};
  tensor::QuantMatrix ffn_up{};
  tensor::QuantMatrix ffn_down{};
};

enum class KvDtype { F32, Int8 };

// int8 V quantizes each head_dim slice in blocks of this many elements, one
// symmetric scale per block. matches the Q8_0 layout, finer than a per-head
// scale that would lose more to outlier channels.
inline constexpr std::size_t kKvBlock = 32;

// int8 K in the dense cache uses a per-channel scale shared across a token
// group of this size. K outliers are channel-consistent, so a per-channel
// scale keeps a spiking channel from crushing the rest of a token's slice
// (KIVI). the group bounds the fp32 residual retained to requantize it.
inline constexpr std::size_t kKvKGroup = 32;

// ring-buffer selection for the KV cache. window == 0 keeps the dense
// full-context path. window > 0 pins n_sink initial positions and slides a
// window of the most recent tokens. dtype Int8 stores keys per-channel and
// values per-block as int8; in the dense path n_sink initial tokens (the
// attention-sink outliers) stay fp32.
struct KvPolicy {
  std::size_t n_sink = 0;
  std::size_t window = 0;
  KvDtype dtype = KvDtype::F32;
};

// [layer][slot][kv_head][head_dim] storage for past keys/values, sized
// up front for the full context length (or n_sink+window under a ring policy).
// append() writes one position's worth of k/v for every kv_head of a layer at
// once (as produced by decode_layer); key()/value() then hand back a pointer
// to one kv_head's slice indexed by slot for the attention dot products.
// Callers own bounds checking on layer/slot/kv_head.
class KVCache {
public:
  // buffer slot of a resident key with its cache-relative rope position.
  struct Resident {
    std::size_t slot;
    std::int32_t rope_pos;
  };

  KVCache(std::size_t n_layers, std::size_t max_seq, std::size_t n_kv_heads, std::size_t head_dim,
          KvPolicy policy = {});

  void append(std::size_t layer, std::size_t pos, const float *k, const float *v);

  const float *key(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const float *value(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  // head_dim int8 slice for a resident slot. valid only when int8() and, for a
  // key, only when slot >= n_sink_fp32().
  const std::int8_t *key_i8(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const std::int8_t *value_i8(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  // per-channel key scales (head_dim of them) for slot's token group in the
  // dense cache, or n_blocks() per-block scales in the ring cache. value scales
  // are n_blocks() per-block. dequantize the matching slice.
  const float *key_scales(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const float *value_scales(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  // fp32 key/value for a retained sink slot. valid only when slot < n_sink_fp32().
  const float *key_sink(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const float *value_sink(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  std::size_t n_blocks() const { return n_blocks_; }
  std::size_t n_sink_fp32() const { return n_sink_; }
  bool k_per_channel() const { return k_group_ > 0; }
  bool int8() const { return policy_.dtype == KvDtype::Int8; }
  bool ring() const { return policy_.window > 0; }
  std::size_t capacity() const { return capacity_; }
  std::size_t n_seen() const { return n_seen_; }
  const KvPolicy &policy() const { return policy_; }
  void reset() { n_seen_ = 0; }

  // maps absolute token position to its ring slot. valid only under ring().
  std::size_t slot_for(std::size_t pos) const;

  // fills out (>= capacity_ entries) with the resident (slot, rope_pos) list,
  // sinks first then window tokens oldest to newest, returns the count.
  std::size_t residents(Resident *out) const;

  std::size_t n_kv_heads() const { return n_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }

private:
  // requantizes token group g of a layer over its resident non-sink slots,
  // computing one symmetric scale per channel from the fp32 residual.
  void requantize_kgroup(std::size_t layer, std::size_t g, std::size_t within);

  KvPolicy policy_;
  std::size_t capacity_;
  std::size_t layer_stride_;
  std::size_t pos_stride_;
  std::size_t n_kv_heads_;
  std::size_t head_dim_;
  std::size_t n_blocks_;
  std::size_t scale_stride_;
  std::size_t k_group_ = 0;        // per-channel K token-group size, 0 = ring per-block
  std::size_t n_kgroups_ = 0;      // K token groups spanning capacity_
  std::size_t k_scale_stride_ = 0; // per-layer stride into k_scale_
  std::size_t n_sink_ = 0;         // sink tokens kept fp32 in the dense int8 cache
  std::size_t n_seen_ = 0;
  std::vector<float> k_;
  std::vector<float> v_;
  std::vector<std::int8_t> k8_;
  std::vector<std::int8_t> v8_;
  std::vector<float> k_scale_;
  std::vector<float> v_scale_;
  std::vector<float> k_raw_;  // current group's raw K, requantized in place
  std::vector<float> k_sink_; // fp32 retained sink keys
  std::vector<float> v_sink_; // fp32 retained sink values
};

// optional hooks for decode_layer to write out intermediate activations,
// used by golden-tensor tests to compare against reference values mid-layer.
struct DebugCapture {
  float *attn_head0 = nullptr; // [head_dim] pre-attn_output, query head 0
  float *layer_out = nullptr;  // [embedding_length] residual stream after layer
};

// owns a loaded model's weights (views into the GgufFile's mmap, not copies)
// plus the scratch buffers and KV cache needed to run one token at a time
// through the transformer. Not movable/copyable in practice since the
// scratch vectors are sized once in load(); construct via load() only.
class Model {
public:
  // validates architecture metadata and every weight tensor's shape/dtype
  // against Config, binds weight pointers into the GgufFile's mmap, and
  // allocates scratch buffers and the KV cache sized for this model.
  static std::expected<Model, gguf::Error> load(const gguf::GgufFile &file);

  const Config &config() const { return cfg_; }

  // dequantizes token's row of the embedding table into out ([embedding_length]).
  void embed(std::int32_t token, float *out) const;

  // runs one transformer block on x in place: attn_norm -> QKV -> RoPE ->
  // causal self-attention over kv (which is appended to first) -> output
  // projection -> residual add, then the same shape for the FFN half. pos is
  // this token's absolute position, used for RoPE and as the attention
  // window's upper bound.
  void decode_layer(std::size_t layer, float *x, std::int32_t pos, KVCache &kv,
                    DebugCapture *dbg = nullptr);

  // runs one token through embed -> every decode_layer -> output norm -> lm
  // head, appending to this model's own KV cache. Returns a pointer to
  // logits_, valid until the next forward() call. Callers must invoke this
  // with strictly increasing pos within a sequence, once per token, prefill
  // included. restarting pos at 0 starts a fresh sequence and overwrites the
  // cache slots it reuses.
  const float *forward(std::int32_t token, std::int32_t pos);

  // rebuilds the KV cache under policy and resizes the score/resident scratch.
  // window > 0 selects the ring path; window == 0 restores the dense default.
  void configure_kv(KvPolicy policy);

  // clears the resident count so the next token starts a fresh stream.
  void reset_kv() { kv_.reset(); }

private:
  Config cfg_;
  std::vector<LayerWeights> layers_;
  tensor::QuantMatrix token_embd_{};
  const float *output_norm_ = nullptr;
  tensor::QuantMatrix lm_head_{};

  std::vector<float> x_;
  std::vector<float> normed_;
  std::vector<float> q_;
  std::vector<float> k_;
  std::vector<float> v_;
  std::vector<float> attn_;
  std::vector<float> proj_;
  std::vector<float> scores_;
  std::vector<float> ffn_gate_;
  std::vector<float> ffn_up_;
  std::vector<float> ffn_down_;
  std::vector<float> logits_;
  std::vector<KVCache::Resident> resident_;
  KVCache kv_{0, 0, 0, 0};
};

} // namespace dbinfer::model

#endif // DBINFER_MODEL_MODEL_HPP
