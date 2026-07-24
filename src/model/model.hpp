#ifndef DBINFER_MODEL_MODEL_HPP
#define DBINFER_MODEL_MODEL_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "gguf/gguf.hpp"
#include "tensor/matmul.hpp"

namespace dbinfer::backend {
class Backend;
}

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
  const float* attn_norm = nullptr;
  tensor::QuantMatrix attn_q{};
  tensor::QuantMatrix attn_k{};
  tensor::QuantMatrix attn_v{};
  tensor::QuantMatrix attn_output{};
  const float* attn_q_bias = nullptr;
  const float* attn_k_bias = nullptr;
  const float* attn_v_bias = nullptr;
  const float* ffn_norm = nullptr;
  tensor::QuantMatrix ffn_gate{};
  tensor::QuantMatrix ffn_up{};
  tensor::QuantMatrix ffn_down{};
};

// page-aligned fp32 storage for the dense KV cache. the 16384-byte base lets
// the Metal backend wrap K and V with newBufferWithBytesNoCopy, so CPU and GPU
// share the same bytes with no per-token copy. for the CPU path it behaves like
// the vector it replaces: assign sizes and zeroes, data hands back the base.
class AlignedF32 {
 public:
  AlignedF32() = default;
  ~AlignedF32();
  AlignedF32(const AlignedF32&) = delete;
  AlignedF32& operator=(const AlignedF32&) = delete;
  AlignedF32(AlignedF32&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  AlignedF32& operator=(AlignedF32&& o) noexcept;

  void assign(std::size_t n, float value);
  float* data() { return ptr_; }
  const float* data() const { return ptr_; }
  float& operator[](std::size_t i) { return ptr_[i]; }
  float operator[](std::size_t i) const { return ptr_[i]; }
  std::size_t size() const { return size_; }

 private:
  float* ptr_ = nullptr;
  std::size_t size_ = 0;
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

  void append(std::size_t layer, std::size_t pos, const float* k, const float* v);

  const float* key(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const float* value(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  // head_dim int8 slice for a resident slot. valid only when int8() and, for a
  // key, only when slot >= n_sink_fp32().
  const std::int8_t* key_i8(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const std::int8_t* value_i8(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  // per-channel key scales (head_dim of them) for slot's token group in the
  // dense cache, or n_blocks() per-block scales in the ring cache. value scales
  // are n_blocks() per-block. dequantize the matching slice.
  const float* key_scales(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const float* value_scales(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  // fp32 key/value for a retained sink slot. valid only when slot < n_sink_fp32().
  const float* key_sink(std::size_t layer, std::size_t slot, std::size_t kv_head) const;
  const float* value_sink(std::size_t layer, std::size_t slot, std::size_t kv_head) const;

  std::size_t n_blocks() const { return n_blocks_; }
  std::size_t n_sink_fp32() const { return n_sink_; }
  bool k_per_channel() const { return k_group_ > 0; }
  bool int8() const { return policy_.dtype == KvDtype::Int8; }
  bool ring() const { return policy_.window > 0; }

  // full-context fp32 path, the only one the prefix cache serializes.
  bool dense_f32() const { return !int8() && !ring(); }
  std::size_t n_layers() const { return n_layers_; }
  // fp32 elements per K (and per V) covering positions [0, prefix_len) of
  // every layer. dense fp32 only.
  std::size_t prefix_elems(std::size_t prefix_len) const {
    return n_layers_ * prefix_len * pos_stride_;
  }

  // packs positions [0, prefix_len) of every layer into k_out and v_out,
  // prefix_elems() floats each. dense fp32 only, caller bounds-checks.
  void copy_prefix_out(std::size_t prefix_len, float* k_out, float* v_out) const;
  // restores positions [0, prefix_len) of every layer from k_in and v_in and
  // sets n_seen to prefix_len. dense fp32 only, caller bounds-checks.
  void copy_prefix_in(std::size_t prefix_len, const float* k_in, const float* v_in);

  std::size_t capacity() const { return capacity_; }
  std::size_t n_seen() const { return n_seen_; }
  const KvPolicy& policy() const { return policy_; }
  void reset() { n_seen_ = 0; }

  // maps absolute token position to its ring slot. valid only under ring().
  std::size_t slot_for(std::size_t pos) const;

  // fills out (>= capacity_ entries) with the resident (slot, rope_pos) list,
  // sinks first then window tokens oldest to newest, returns the count.
  std::size_t residents(Resident* out) const;

  std::size_t n_kv_heads() const { return n_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t pos_stride() const { return pos_stride_; }

  // page-aligned dense fp32 K/V storage base, for the Metal backend to wrap
  // no-copy. valid only under dense_f32().
  float* dense_k() { return k_.data(); }
  float* dense_v() { return v_.data(); }

  // records that position pos is resident after the GPU offload wrote it into
  // the shared cache in place. dense fp32 only.
  void mark_position(std::size_t pos) { n_seen_ = pos + 1; }

 private:
  // requantizes token group g of a layer over its resident non-sink slots,
  // computing one symmetric scale per channel from the fp32 residual.
  void requantize_kgroup(std::size_t layer, std::size_t g, std::size_t within);

  KvPolicy policy_;
  std::size_t n_layers_;
  std::size_t capacity_;
  std::size_t layer_stride_;
  std::size_t pos_stride_;
  std::size_t n_kv_heads_;
  std::size_t head_dim_;
  std::size_t n_blocks_;
  std::size_t scale_stride_;
  std::size_t k_group_ = 0;         // per-channel K token-group size, 0 = ring per-block
  std::size_t n_kgroups_ = 0;       // K token groups spanning capacity_
  std::size_t k_scale_stride_ = 0;  // per-layer stride into k_scale_
  std::size_t n_sink_ = 0;          // sink tokens kept fp32 in the dense int8 cache
  std::size_t n_seen_ = 0;
  AlignedF32 k_;
  AlignedF32 v_;
  std::vector<std::int8_t> k8_;
  std::vector<std::int8_t> v8_;
  std::vector<float> k_scale_;
  std::vector<float> v_scale_;
  std::vector<float> k_raw_;   // current group's raw K, requantized in place
  std::vector<float> k_sink_;  // fp32 retained sink keys
  std::vector<float> v_sink_;  // fp32 retained sink values
};

// optional hooks for decode_layer to write out intermediate activations,
// used by golden-tensor tests to compare against reference values mid-layer.
struct DebugCapture {
  float* attn_head0 = nullptr;  // [head_dim] pre-attn_output, query head 0
  float* layer_out = nullptr;   // [embedding_length] residual stream after layer
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
  static std::expected<Model, gguf::Error> load(const gguf::GgufFile& file);

  const Config& config() const { return cfg_; }

  // dequantizes token's row of the embedding table into out ([embedding_length]).
  void embed(std::int32_t token, float* out) const;

  // runs one transformer block on x in place: attn_norm -> QKV -> RoPE ->
  // causal self-attention over kv (which is appended to first) -> output
  // projection -> residual add, then the same shape for the FFN half. pos is
  // this token's absolute position, used for RoPE and as the attention
  // window's upper bound.
  void decode_layer(std::size_t layer, float* x, std::int32_t pos, KVCache& kv,
                    DebugCapture* dbg = nullptr);

  // runs one token through embed -> every decode_layer -> output norm -> lm
  // head, appending to this model's own KV cache. Returns a pointer to
  // logits_, valid until the next forward() call. Callers must invoke this
  // with strictly increasing pos within a sequence, once per token, prefill
  // included. restarting pos at 0 starts a fresh sequence and overwrites the
  // cache slots it reuses.
  const float* forward(std::int32_t token, std::int32_t pos);

  // runs n tokens at positions [pos0, pos0 + n) through the whole stack together,
  // appending each to this model's KV cache and writing n * vocab_size logits to
  // out (row per token). attention masks causally within the chunk and against
  // the existing cache. requires the dense fp32 cache and strictly increasing
  // pos0 across calls, matching per-token forward(). every logit is bitwise
  // identical to forward() run one token at a time over the same positions.
  void forward_chunk(const std::int32_t* tokens, std::int32_t pos0, std::size_t n, float* out);

  // true when the KV cache is the default full-context fp32 layout, the only one
  // forward_chunk supports.
  bool kv_dense_f32() const { return kv_.dense_f32(); }

  // offloads the first n_layers transformer blocks of forward() to backend,
  // copying their weights to the device and wrapping the KV cache no-copy.
  // n_layers == 0 leaves the pure-CPU path untouched. valid only on the dense
  // fp32 cache with layers whose weights layers_offloadable accepts. returns
  // false and stays on CPU if the backend setup fails.
  bool set_gpu_offload(backend::Backend* backend, std::size_t n_layers);

  // true when the first n transformer blocks store their matmul weights as F16,
  // the only dtype the Metal decode path reads.
  bool layers_f16(std::size_t n) const;

  // true when the first n blocks' matmul weights are all F16, Q8_0, or Q4_0,
  // the dtypes the Metal decode path reads.
  bool layers_offloadable(std::size_t n) const;

  // rebuilds the KV cache under policy and resizes the score/resident scratch.
  // window > 0 selects the ring path; window == 0 restores the dense default.
  void configure_kv(KvPolicy policy);

  // clears the resident count so the next token starts a fresh stream.
  void reset_kv() { kv_.reset(); }

  // serializes the dense fp32 KV cache prefix [0, prefix_tokens.size()) plus
  // the prefix token ids to path. requires the default dense fp32 policy and
  // that prefill has filled at least that many positions.
  [[nodiscard]] std::expected<void, gguf::Error> save_kv_prefix(
      const std::string& path, std::span<const std::int32_t> prefix_tokens) const;

  // validates a prefix cache header against this model and prefix_tokens
  // against the saved ids, restores K/V, sets the cache position to the saved
  // prefix length, and returns it. leaves the cache untouched on any mismatch.
  [[nodiscard]] std::expected<std::size_t, gguf::Error> load_kv_prefix(
      const std::string& path, std::span<const std::int32_t> prefix_tokens);

 private:
  // per-chunk activation buffers, sized for n tokens once per forward_chunk.
  struct ChunkScratch {
    std::vector<float> x;
    std::vector<float> normed;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn;
    std::vector<float> proj;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
  };

  // runs one transformer block over the whole chunk, mirroring decode_layer per
  // token. x holds n residual rows of embedding_length, updated in place.
  void decode_layer_chunk(std::size_t layer, float* x, std::int32_t pos0, std::size_t n,
                          ChunkScratch& s, KVCache& kv);

  // runs the first gpu_layers_ blocks of one token on the backend in a single
  // command buffer, matching decode_layer's dense fp32 result to tolerance.
  // falls back to per-layer decode_layer on a backend error.
  void decode_token_gpu(float* x, std::int32_t pos);

  // binds every weight tensor's mmap view into this model after parse_config
  // has filled cfg_, validating each tensor's shape and dtype against it.
  [[nodiscard]] std::expected<void, gguf::Error> bind_weights(const gguf::GgufFile& file);

  // self-attention over the keys and values already appended to kv, split by
  // cache layout. the score, softmax, and weighted-value math repeats across
  // the three because the fp32, int8, and ring key paths diverge too much to
  // share one loop. each writes one query head's result into attn_ and passes
  // scale as 1/sqrt(head_dim).
  void attend_dense_f32(std::size_t layer, std::int32_t pos, KVCache& kv, float scale,
                        DebugCapture* dbg);
  void attend_dense_int8(std::size_t layer, std::int32_t pos, KVCache& kv, float scale,
                         DebugCapture* dbg);
  // ring path re-ropes each resident key by its cache-relative position, so it
  // appends and ropes the query itself.
  void attend_ring(std::size_t layer, std::int32_t pos, KVCache& kv, float scale,
                   DebugCapture* dbg);

  Config cfg_;
  std::vector<LayerWeights> layers_;
  tensor::QuantMatrix token_embd_{};
  const float* output_norm_ = nullptr;
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
  backend::Backend* backend_ = nullptr;
  std::size_t gpu_layers_ = 0;
  bool gpu_head_ = false;
};

}  // namespace dbinfer::model

#endif  // DBINFER_MODEL_MODEL_HPP
