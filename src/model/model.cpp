#include "model/model.hpp"

#include "backend/backend.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul.hpp"
#include "tensor/ops.hpp"
#include "tensor/thread_pool.hpp"
#include "try.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string>

namespace dbinfer::model {

AlignedF32::~AlignedF32() { std::free(ptr_); }

AlignedF32 &AlignedF32::operator=(AlignedF32 &&o) noexcept {
  if (this != &o) {
    std::free(ptr_);
    ptr_ = o.ptr_;
    size_ = o.size_;
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

void AlignedF32::assign(std::size_t n, float value) {
  std::free(ptr_);
  ptr_ = nullptr;
  size_ = 0;
  if (n == 0)
    return;
  void *p = nullptr;
  // 16384 is the Apple Silicon page newBufferWithBytesNoCopy wraps zero-copy.
  // rounding the length to a page lets the backend wrap the whole span safely.
  constexpr std::size_t kPage = 16384;
  const std::size_t bytes = ((n * sizeof(float) + kPage - 1) / kPage) * kPage;
  if (posix_memalign(&p, kPage, bytes) != 0 || p == nullptr) {
    std::fprintf(stderr, "fatal: KV cache allocation of %zu floats failed\n", n);
    std::abort();
  }
  ptr_ = static_cast<float *>(p);
  std::fill(ptr_, ptr_ + n, value);
  size_ = n;
}

using gguf::Error;
using gguf::GgmlType;
using gguf::GgufFile;
using gguf::MetaValue;
using gguf::TensorInfo;

namespace {

std::optional<std::uint64_t> as_u64(const MetaValue &mv) {
  if (auto p = std::get_if<std::uint8_t>(&mv.value))
    return *p;
  if (auto p = std::get_if<std::uint16_t>(&mv.value))
    return *p;
  if (auto p = std::get_if<std::uint32_t>(&mv.value))
    return *p;
  if (auto p = std::get_if<std::uint64_t>(&mv.value))
    return *p;
  if (auto p = std::get_if<std::int32_t>(&mv.value))
    return *p >= 0 ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(*p)) : std::nullopt;
  if (auto p = std::get_if<std::int64_t>(&mv.value))
    return *p >= 0 ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(*p)) : std::nullopt;
  return std::nullopt;
}

std::optional<float> as_f32(const MetaValue &mv) {
  if (auto p = std::get_if<float>(&mv.value))
    return *p;
  if (auto p = std::get_if<double>(&mv.value))
    return static_cast<float>(*p);
  return std::nullopt;
}

const TensorInfo *find(const GgufFile &f, const std::string &name) {
  for (const auto &t : f.tensors)
    if (t.name == name)
      return &t;
  return nullptr;
}

struct BlockLayout {
  std::size_t elems;
  std::size_t bytes;
};

// elements and bytes per storage block for a weight dtype. k-quants pack
// 256-element super-blocks, the legacy formats 32.
BlockLayout block_layout(GgmlType type) {
  switch (type) {
  case GgmlType::F16:
    return {32, 64};
  case GgmlType::Q8_0:
    return {32, 34};
  case GgmlType::Q4_0:
    return {32, 18};
  case GgmlType::Q5_0:
    return {32, 22};
  case GgmlType::Q4_K:
    return {256, 144};
  case GgmlType::Q6_K:
    return {256, 210};
  default:
    return {0, 0};
  }
}

bool is_weight_type(GgmlType type) {
  switch (type) {
  case GgmlType::F16:
  case GgmlType::Q8_0:
  case GgmlType::Q4_0:
  case GgmlType::Q5_0:
  case GgmlType::Q4_K:
  case GgmlType::Q6_K:
    return true;
  default:
    return false;
  }
}

} // namespace

KVCache::KVCache(std::size_t n_layers, std::size_t max_seq, std::size_t n_kv_heads,
                 std::size_t head_dim, KvPolicy policy)
    : policy_(policy), n_layers_(n_layers),
      capacity_(policy.window > 0 ? policy.n_sink + policy.window : max_seq),
      layer_stride_(capacity_ * n_kv_heads * head_dim), pos_stride_(n_kv_heads * head_dim),
      n_kv_heads_(n_kv_heads), head_dim_(head_dim), n_blocks_((head_dim + kKvBlock - 1) / kKvBlock),
      scale_stride_(capacity_ * n_kv_heads * n_blocks_) {
  if (policy_.dtype != KvDtype::Int8) {
    k_.assign(n_layers * layer_stride_, 0.0f);
    v_.assign(n_layers * layer_stride_, 0.0f);
    return;
  }
  k8_.assign(n_layers * layer_stride_, 0);
  v8_.assign(n_layers * layer_stride_, 0);
  v_scale_.assign(n_layers * scale_stride_, 0.0f);
  // the ring path evicts tokens, so per-channel groups cannot stay coherent
  // across a slot's lifetime; keep its keys per-block. the dense path groups
  // keys per-channel over kKvKGroup tokens and pins n_sink initial tokens fp32.
  if (ring()) {
    k_scale_.assign(n_layers * scale_stride_, 0.0f);
    return;
  }
  k_group_ = kKvKGroup;
  n_kgroups_ = (capacity_ + k_group_ - 1) / k_group_;
  k_scale_stride_ = n_kgroups_ * n_kv_heads * head_dim;
  n_sink_ = policy_.n_sink;
  k_scale_.assign(n_layers * k_scale_stride_, 0.0f);
  k_raw_.assign(n_layers * k_group_ * pos_stride_, 0.0f);
  if (n_sink_ > 0) {
    k_sink_.assign(n_layers * n_sink_ * pos_stride_, 0.0f);
    v_sink_.assign(n_layers * n_sink_ * pos_stride_, 0.0f);
  }
}

namespace {

// symmetric per-block quant. scale = max_abs/127 per kKvBlock block. zero block
// yields zeros and a zero scale, no divide by zero.
void quant_head(const float *src, std::size_t n, std::int8_t *dst, float *scales) {
  std::size_t b = 0;
  for (std::size_t start = 0; start < n; start += kKvBlock, ++b) {
    const std::size_t end = std::min(start + kKvBlock, n);
    float max_abs = 0.0f;
    for (std::size_t i = start; i < end; ++i)
      max_abs = std::max(max_abs, std::fabs(src[i]));
    const float scale = max_abs / 127.0f;
    scales[b] = scale;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (std::size_t i = start; i < end; ++i) {
      const long q = std::lround(src[i] * inv);
      dst[i] = static_cast<std::int8_t>(std::clamp<long>(q, -127, 127));
    }
  }
}

} // namespace

std::size_t KVCache::slot_for(std::size_t pos) const {
  if (pos < policy_.n_sink)
    return pos;
  return policy_.n_sink + (pos - policy_.n_sink) % policy_.window;
}

// per-channel symmetric requantization of token group g over the slots filled
// so far ([g*group, g*group+within]), skipping sink slots held fp32. the scale
// row doubles as the max-abs accumulator before it is divided by 127.
void KVCache::requantize_kgroup(std::size_t layer, std::size_t g, std::size_t within) {
  const std::size_t first = g * k_group_;
  float *sc_base = k_scale_.data() + layer * k_scale_stride_ + g * n_kv_heads_ * head_dim_;
  const float *raw_base = k_raw_.data() + layer * k_group_ * pos_stride_;
  for (std::size_t kh = 0; kh < n_kv_heads_; ++kh) {
    float *sc = sc_base + kh * head_dim_;
    for (std::size_t c = 0; c < head_dim_; ++c)
      sc[c] = 0.0f;
    for (std::size_t w = 0; w <= within; ++w) {
      if (first + w < n_sink_)
        continue;
      const float *raw = raw_base + w * pos_stride_ + kh * head_dim_;
      for (std::size_t c = 0; c < head_dim_; ++c)
        sc[c] = std::max(sc[c], std::fabs(raw[c]));
    }
    for (std::size_t c = 0; c < head_dim_; ++c)
      sc[c] /= 127.0f;
    for (std::size_t w = 0; w <= within; ++w) {
      if (first + w < n_sink_)
        continue;
      const float *raw = raw_base + w * pos_stride_ + kh * head_dim_;
      std::int8_t *dst =
          k8_.data() + layer * layer_stride_ + (first + w) * pos_stride_ + kh * head_dim_;
      for (std::size_t c = 0; c < head_dim_; ++c) {
        const float inv = sc[c] > 0.0f ? 1.0f / sc[c] : 0.0f;
        const long q = std::lround(raw[c] * inv);
        dst[c] = static_cast<std::int8_t>(std::clamp<long>(q, -127, 127));
      }
    }
  }
}

void KVCache::append(std::size_t layer, std::size_t pos, const float *k, const float *v) {
  if (k_group_ > 0) {
    const std::size_t within = pos % k_group_;
    const std::size_t g = pos / k_group_;
    float *raw = k_raw_.data() + layer * k_group_ * pos_stride_ + within * pos_stride_;
    for (std::size_t i = 0; i < pos_stride_; ++i)
      raw[i] = k[i];
    if (pos < n_sink_) {
      float *ks = k_sink_.data() + layer * n_sink_ * pos_stride_ + pos * pos_stride_;
      float *vs = v_sink_.data() + layer * n_sink_ * pos_stride_ + pos * pos_stride_;
      for (std::size_t i = 0; i < pos_stride_; ++i) {
        ks[i] = k[i];
        vs[i] = v[i];
      }
    } else {
      const std::size_t base = layer * layer_stride_ + pos * pos_stride_;
      const std::size_t sbase = layer * scale_stride_ + pos * n_kv_heads_ * n_blocks_;
      for (std::size_t kh = 0; kh < n_kv_heads_; ++kh) {
        const std::size_t off = kh * head_dim_;
        quant_head(v + off, head_dim_, v8_.data() + base + off,
                   v_scale_.data() + sbase + kh * n_blocks_);
      }
    }
    requantize_kgroup(layer, g, within);
    n_seen_ = pos + 1;
    return;
  }

  const std::size_t slot = ring() ? slot_for(pos) : pos;
  const std::size_t base = layer * layer_stride_ + slot * pos_stride_;
  if (policy_.dtype == KvDtype::Int8) {
    const std::size_t sbase = layer * scale_stride_ + slot * n_kv_heads_ * n_blocks_;
    for (std::size_t kh = 0; kh < n_kv_heads_; ++kh) {
      const std::size_t off = kh * head_dim_;
      const std::size_t soff = sbase + kh * n_blocks_;
      quant_head(k + off, head_dim_, k8_.data() + base + off, k_scale_.data() + soff);
      quant_head(v + off, head_dim_, v8_.data() + base + off, v_scale_.data() + soff);
    }
  } else {
    for (std::size_t i = 0; i < pos_stride_; ++i) {
      k_[base + i] = k[i];
      v_[base + i] = v[i];
    }
  }
  n_seen_ = pos + 1;
}

std::size_t KVCache::residents(Resident *out) const {
  const std::size_t n_sink = policy_.n_sink;
  const std::size_t window = policy_.window;
  const std::size_t sinks = std::min(n_seen_, n_sink);
  std::size_t count = 0;
  for (std::size_t t = 0; t < sinks; ++t)
    out[count++] = {t, static_cast<std::int32_t>(t)};

  std::size_t window_start = n_sink;
  if (n_seen_ > n_sink && n_seen_ - n_sink > window)
    window_start = n_seen_ - window;
  for (std::size_t t = window_start; t < n_seen_; ++t)
    out[count++] = {slot_for(t), static_cast<std::int32_t>(n_sink + (t - window_start))};
  return count;
}

const float *KVCache::key(std::size_t layer, std::size_t slot, std::size_t kv_head) const {
  return k_.data() + layer * layer_stride_ + slot * pos_stride_ + kv_head * head_dim_;
}

const float *KVCache::value(std::size_t layer, std::size_t slot, std::size_t kv_head) const {
  return v_.data() + layer * layer_stride_ + slot * pos_stride_ + kv_head * head_dim_;
}

const std::int8_t *KVCache::key_i8(std::size_t layer, std::size_t slot, std::size_t kv_head) const {
  return k8_.data() + layer * layer_stride_ + slot * pos_stride_ + kv_head * head_dim_;
}

const std::int8_t *KVCache::value_i8(std::size_t layer, std::size_t slot,
                                     std::size_t kv_head) const {
  return v8_.data() + layer * layer_stride_ + slot * pos_stride_ + kv_head * head_dim_;
}

const float *KVCache::key_scales(std::size_t layer, std::size_t slot, std::size_t kv_head) const {
  if (k_group_ > 0) {
    const std::size_t g = slot / k_group_;
    return k_scale_.data() + layer * k_scale_stride_ + (g * n_kv_heads_ + kv_head) * head_dim_;
  }
  return k_scale_.data() + layer * scale_stride_ + (slot * n_kv_heads_ + kv_head) * n_blocks_;
}

const float *KVCache::value_scales(std::size_t layer, std::size_t slot, std::size_t kv_head) const {
  return v_scale_.data() + layer * scale_stride_ + (slot * n_kv_heads_ + kv_head) * n_blocks_;
}

const float *KVCache::key_sink(std::size_t layer, std::size_t slot, std::size_t kv_head) const {
  return k_sink_.data() + layer * n_sink_ * pos_stride_ + slot * pos_stride_ + kv_head * head_dim_;
}

const float *KVCache::value_sink(std::size_t layer, std::size_t slot, std::size_t kv_head) const {
  return v_sink_.data() + layer * n_sink_ * pos_stride_ + slot * pos_stride_ + kv_head * head_dim_;
}

void KVCache::copy_prefix_out(std::size_t prefix_len, float *k_out, float *v_out) const {
  const std::size_t span = prefix_len * pos_stride_;
  for (std::size_t l = 0; l < n_layers_; ++l) {
    const float *ks = k_.data() + l * layer_stride_;
    const float *vs = v_.data() + l * layer_stride_;
    std::copy(ks, ks + span, k_out + l * span);
    std::copy(vs, vs + span, v_out + l * span);
  }
}

void KVCache::copy_prefix_in(std::size_t prefix_len, const float *k_in, const float *v_in) {
  const std::size_t span = prefix_len * pos_stride_;
  for (std::size_t l = 0; l < n_layers_; ++l) {
    std::copy(k_in + l * span, k_in + (l + 1) * span, k_.data() + l * layer_stride_);
    std::copy(v_in + l * span, v_in + (l + 1) * span, v_.data() + l * layer_stride_);
  }
  n_seen_ = prefix_len;
}

std::expected<Model, Error> Model::load(const GgufFile &file) {
  auto fail = [](std::string msg) -> std::unexpected<Error> {
    return std::unexpected(Error{std::move(msg), "", 0});
  };

  const MetaValue *arch_mv = file.find_meta("general.architecture");
  if (arch_mv == nullptr)
    return fail("general.architecture missing");
  const auto *arch_p = std::get_if<std::string>(&arch_mv->value);
  if (arch_p == nullptr)
    return fail("general.architecture is not a string");
  const std::string arch = *arch_p;

  auto req_u64 = [&](const std::string &key,
                     std::uint64_t &dst) -> std::optional<std::unexpected<Error>> {
    const MetaValue *mv = file.find_meta(key);
    if (mv == nullptr)
      return std::unexpected(Error{key + " missing", "", 0});
    auto v = as_u64(*mv);
    if (!v)
      return std::unexpected(Error{key + " is not an integer", "", 0});
    dst = *v;
    return std::nullopt;
  };
  auto req_f32 = [&](const std::string &key, float &dst) -> std::optional<std::unexpected<Error>> {
    const MetaValue *mv = file.find_meta(key);
    if (mv == nullptr)
      return std::unexpected(Error{key + " missing", "", 0});
    auto v = as_f32(*mv);
    if (!v)
      return std::unexpected(Error{key + " is not a float", "", 0});
    dst = *v;
    return std::nullopt;
  };

  Config cfg;
  std::uint64_t u = 0;
  if (auto e = req_u64(arch + ".block_count", u))
    return std::move(*e);
  cfg.n_layers = u;
  if (auto e = req_u64(arch + ".embedding_length", u))
    return std::move(*e);
  cfg.embedding_length = u;
  if (auto e = req_u64(arch + ".attention.head_count", u))
    return std::move(*e);
  cfg.n_heads = u;
  if (auto e = req_u64(arch + ".attention.head_count_kv", u))
    return std::move(*e);
  cfg.n_kv_heads = u;
  if (auto e = req_u64(arch + ".feed_forward_length", u))
    return std::move(*e);
  cfg.ffn_length = u;
  if (auto e = req_u64(arch + ".context_length", u))
    return std::move(*e);
  cfg.context_length = u;
  if (auto e = req_f32(arch + ".rope.freq_base", cfg.rope_theta))
    return std::move(*e);
  if (auto e = req_f32(arch + ".attention.layer_norm_rms_epsilon", cfg.rms_eps))
    return std::move(*e);

  if (cfg.n_heads == 0 || cfg.n_kv_heads == 0 || cfg.embedding_length == 0)
    return fail("degenerate head/embedding config");
  if (cfg.embedding_length % cfg.n_heads != 0)
    return fail("embedding_length not divisible by head_count");
  if (cfg.n_heads % cfg.n_kv_heads != 0)
    return fail("head_count not divisible by head_count_kv");
  cfg.head_dim = cfg.embedding_length / cfg.n_heads;
  cfg.gqa_factor = cfg.n_heads / cfg.n_kv_heads;

  std::optional<std::uint64_t> vocab;
  if (const MetaValue *vs = file.find_meta(arch + ".vocab_size"))
    vocab = as_u64(*vs);
  if (!vocab) {
    if (const MetaValue *toks = file.find_meta("tokenizer.ggml.tokens"))
      if (const auto *a = std::get_if<gguf::MetaArray>(&toks->value))
        vocab = a->values.size();
  }
  if (!vocab)
    return fail("cannot determine vocab size");
  cfg.vocab_size = *vocab;

  Model m;
  m.cfg_ = cfg;

  const std::size_t dim = cfg.embedding_length;
  const std::size_t ff = cfg.ffn_length;
  const std::size_t hd = cfg.head_dim;

  auto weight = [&](const std::string &name, std::size_t out, std::size_t in,
                    tensor::QuantMatrix &dst) -> std::optional<std::unexpected<Error>> {
    const TensorInfo *t = find(file, name);
    if (t == nullptr)
      return std::unexpected(Error{name + " missing", "", 0});
    if (!is_weight_type(t->type))
      return std::unexpected(Error{name + " expected F16, Q8_0, Q4_0, Q5_0, Q4_K, or Q6_K", "", 0});
    if (t->shape[0] != out || t->shape[1] != in)
      return std::unexpected(Error{
          name + " shape mismatch: expected [" + std::to_string(out) + ", " + std::to_string(in) +
              "], found [" + std::to_string(t->shape[0]) + ", " + std::to_string(t->shape[1]) + "]",
          "", t->offset});
    const BlockLayout bl = block_layout(t->type);
    if (in % bl.elems != 0)
      return std::unexpected(
          Error{name + " in not a multiple of " + std::to_string(bl.elems), "", t->offset});
    std::size_t elems = 0;
    if (__builtin_mul_overflow(out, in, &elems))
      return std::unexpected(Error{name + " out*in overflow", "", t->offset});
    if (elems % bl.elems != 0)
      return std::unexpected(
          Error{name + " out*in not a multiple of " + std::to_string(bl.elems), "", t->offset});
    const std::size_t expect = (elems / bl.elems) * bl.bytes;
    if (t->nbytes != expect)
      return std::unexpected(Error{name + " nbytes mismatch: expected " + std::to_string(expect) +
                                       ", found " + std::to_string(t->nbytes),
                                   "", t->offset});
    dst = tensor::QuantMatrix{t->data, t->type};
    return std::nullopt;
  };
  auto normw = [&](const std::string &name, std::size_t n,
                   const float *&dst) -> std::optional<std::unexpected<Error>> {
    const TensorInfo *t = find(file, name);
    if (t == nullptr)
      return std::unexpected(Error{name + " missing", "", 0});
    if (t->type != GgmlType::F32)
      return std::unexpected(Error{name + " expected F32", "", 0});
    if (t->shape[0] != n)
      return std::unexpected(Error{name + " shape mismatch: expected [" + std::to_string(n) +
                                       "], found [" + std::to_string(t->shape[0]) + "]",
                                   "", t->offset});
    dst = reinterpret_cast<const float *>(t->data);
    return std::nullopt;
  };
  auto optbias = [&](const std::string &name, std::size_t n,
                     const float *&dst) -> std::optional<std::unexpected<Error>> {
    const TensorInfo *t = find(file, name);
    if (t == nullptr) {
      dst = nullptr;
      return std::nullopt;
    }
    if (t->type != GgmlType::F32)
      return std::unexpected(Error{name + " expected F32", "", 0});
    if (t->shape[0] != n)
      return std::unexpected(Error{name + " bias shape mismatch: expected [" + std::to_string(n) +
                                       "], found [" + std::to_string(t->shape[0]) + "]",
                                   "", t->offset});
    dst = reinterpret_cast<const float *>(t->data);
    return std::nullopt;
  };

  if (auto e = weight("token_embd.weight", cfg.vocab_size, dim, m.token_embd_))
    return std::move(*e);

  if (auto e = normw("output_norm.weight", dim, m.output_norm_))
    return std::move(*e);

  const TensorInfo *out_w = find(file, "output.weight");
  if (out_w == nullptr) {
    cfg.tied_embeddings = true;
    m.lm_head_ = m.token_embd_;
  } else {
    cfg.tied_embeddings = false;
    if (auto e = weight("output.weight", cfg.vocab_size, dim, m.lm_head_))
      return std::move(*e);
  }
  m.cfg_.tied_embeddings = cfg.tied_embeddings;

  m.layers_.resize(cfg.n_layers);
  for (std::size_t l = 0; l < cfg.n_layers; ++l) {
    LayerWeights &L = m.layers_[l];
    const std::string p = "blk." + std::to_string(l) + ".";
    if (auto e = normw(p + "attn_norm.weight", dim, L.attn_norm))
      return std::move(*e);
    if (auto e = weight(p + "attn_q.weight", cfg.n_heads * hd, dim, L.attn_q))
      return std::move(*e);
    if (auto e = weight(p + "attn_k.weight", cfg.n_kv_heads * hd, dim, L.attn_k))
      return std::move(*e);
    if (auto e = weight(p + "attn_v.weight", cfg.n_kv_heads * hd, dim, L.attn_v))
      return std::move(*e);
    if (auto e = weight(p + "attn_output.weight", dim, cfg.n_heads * hd, L.attn_output))
      return std::move(*e);
    if (auto e = optbias(p + "attn_q.bias", cfg.n_heads * hd, L.attn_q_bias))
      return std::move(*e);
    if (auto e = optbias(p + "attn_k.bias", cfg.n_kv_heads * hd, L.attn_k_bias))
      return std::move(*e);
    if (auto e = optbias(p + "attn_v.bias", cfg.n_kv_heads * hd, L.attn_v_bias))
      return std::move(*e);
    if (auto e = normw(p + "ffn_norm.weight", dim, L.ffn_norm))
      return std::move(*e);
    if (auto e = weight(p + "ffn_gate.weight", ff, dim, L.ffn_gate))
      return std::move(*e);
    if (auto e = weight(p + "ffn_up.weight", ff, dim, L.ffn_up))
      return std::move(*e);
    if (auto e = weight(p + "ffn_down.weight", dim, ff, L.ffn_down))
      return std::move(*e);
  }

  m.x_.assign(dim, 0.0f);
  m.normed_.assign(dim, 0.0f);
  m.q_.assign(cfg.n_heads * hd, 0.0f);
  m.k_.assign(cfg.n_kv_heads * hd, 0.0f);
  m.v_.assign(cfg.n_kv_heads * hd, 0.0f);
  m.attn_.assign(cfg.n_heads * hd, 0.0f);
  m.proj_.assign(dim, 0.0f);
  m.scores_.assign(cfg.n_heads * cfg.context_length, 0.0f);
  m.ffn_gate_.assign(ff, 0.0f);
  m.ffn_up_.assign(ff, 0.0f);
  m.ffn_down_.assign(dim, 0.0f);
  m.logits_.assign(cfg.vocab_size, 0.0f);
  m.resident_.assign(cfg.context_length, {});
  m.kv_ = KVCache(cfg.n_layers, cfg.context_length, cfg.n_kv_heads, hd);

  return m;
}

void Model::configure_kv(KvPolicy policy) {
  kv_ = KVCache(cfg_.n_layers, cfg_.context_length, cfg_.n_kv_heads, cfg_.head_dim, policy);
  scores_.assign(cfg_.n_heads * kv_.capacity(), 0.0f);
  resident_.assign(kv_.capacity(), {});
}

void Model::embed(std::int32_t token, float *out) const {
  const std::size_t dim = cfg_.embedding_length;
  tensor::dequant_row(token_embd_, static_cast<std::size_t>(token), dim, out);
}

void Model::decode_layer(std::size_t layer, float *x, std::int32_t pos, KVCache &kv,
                         DebugCapture *dbg) {
  const LayerWeights &L = layers_[layer];
  const std::size_t dim = cfg_.embedding_length;
  const std::size_t hd = cfg_.head_dim;
  const std::size_t nh = cfg_.n_heads;
  const std::size_t nkv = cfg_.n_kv_heads;
  const std::size_t gqa = cfg_.gqa_factor;

  tensor::rmsnorm(x, L.attn_norm, cfg_.rms_eps, normed_.data(), 1, dim);
  // Q/K/V share the normed input; fuse them under one barrier.
  const tensor::MatvecJob qkv[] = {
      {L.attn_q, q_.data(), nh * hd},
      {L.attn_k, k_.data(), nkv * hd},
      {L.attn_v, v_.data(), nkv * hd},
  };
  tensor::matvec_quant_fused(qkv, 3, normed_.data(), dim);
  if (L.attn_q_bias)
    for (std::size_t i = 0; i < nh * hd; ++i)
      q_[i] += L.attn_q_bias[i];
  if (L.attn_k_bias)
    for (std::size_t i = 0; i < nkv * hd; ++i)
      k_[i] += L.attn_k_bias[i];
  if (L.attn_v_bias)
    for (std::size_t i = 0; i < nkv * hd; ++i)
      v_[i] += L.attn_v_bias[i];

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  if (!kv.ring()) {
    const std::int32_t p = pos;
    tensor::rope(q_.data(), &p, cfg_.rope_theta, nh, 1, hd);
    tensor::rope(k_.data(), &p, cfg_.rope_theta, nkv, 1, hd);

    kv.append(layer, static_cast<std::size_t>(pos), k_.data(), v_.data());

    const std::size_t last = static_cast<std::size_t>(pos);
    const std::size_t cl = cfg_.context_length;
    if (!kv.int8()) {
      tensor::parallel_for(tensor::thread_pool(), nh, 1, [&](std::size_t hbegin, std::size_t hend) {
        for (std::size_t h = hbegin; h < hend; ++h) {
          const std::size_t kh = h / gqa;
          const float *qh = q_.data() + h * hd;
          float *sc = scores_.data() + h * cl;
          for (std::size_t pp = 0; pp <= last; ++pp) {
            const float *kp = kv.key(layer, pp, kh);
            float dot = 0.0f;
            for (std::size_t i = 0; i < hd; ++i)
              dot += qh[i] * kp[i];
            sc[pp] = dot * scale;
          }
          tensor::softmax(sc, sc, last + 1);
          float *outh = attn_.data() + h * hd;
          for (std::size_t i = 0; i < hd; ++i)
            outh[i] = 0.0f;
          for (std::size_t pp = 0; pp <= last; ++pp) {
            const float *vp = kv.value(layer, pp, kh);
            const float w = sc[pp];
            for (std::size_t i = 0; i < hd; ++i)
              outh[i] += w * vp[i];
          }
          if (dbg != nullptr && dbg->attn_head0 != nullptr && h == 0)
            for (std::size_t i = 0; i < hd; ++i)
              dbg->attn_head0[i] = outh[i];
        }
      });
    } else {
      const std::size_t n_sink = kv.n_sink_fp32();
      tensor::parallel_for(tensor::thread_pool(), nh, 1, [&](std::size_t hbegin, std::size_t hend) {
        for (std::size_t h = hbegin; h < hend; ++h) {
          const std::size_t kh = h / gqa;
          const float *qh = q_.data() + h * hd;
          float *sc = scores_.data() + h * cl;
          const std::size_t nb = kv.n_blocks();
          for (std::size_t pp = 0; pp <= last; ++pp) {
            float dot = 0.0f;
            if (pp < n_sink) {
              const float *kp = kv.key_sink(layer, pp, kh);
              for (std::size_t i = 0; i < hd; ++i)
                dot += qh[i] * kp[i];
            } else {
              const std::int8_t *kp = kv.key_i8(layer, pp, kh);
              const float *ks = kv.key_scales(layer, pp, kh);
              for (std::size_t i = 0; i < hd; ++i)
                dot += qh[i] * ks[i] * static_cast<float>(kp[i]);
            }
            sc[pp] = dot * scale;
          }
          tensor::softmax(sc, sc, last + 1);
          float *outh = attn_.data() + h * hd;
          for (std::size_t i = 0; i < hd; ++i)
            outh[i] = 0.0f;
          for (std::size_t pp = 0; pp <= last; ++pp) {
            const float w = sc[pp];
            if (pp < n_sink) {
              const float *vp = kv.value_sink(layer, pp, kh);
              for (std::size_t i = 0; i < hd; ++i)
                outh[i] += w * vp[i];
              continue;
            }
            const std::int8_t *vp = kv.value_i8(layer, pp, kh);
            const float *vs = kv.value_scales(layer, pp, kh);
            for (std::size_t b = 0; b < nb; ++b) {
              const std::size_t start = b * kKvBlock;
              const std::size_t end = std::min(start + kKvBlock, hd);
              const float wv = w * vs[b];
              for (std::size_t i = start; i < end; ++i)
                outh[i] += wv * static_cast<float>(vp[i]);
            }
          }
          if (dbg != nullptr && dbg->attn_head0 != nullptr && h == 0)
            for (std::size_t i = 0; i < hd; ++i)
              dbg->attn_head0[i] = outh[i];
        }
      });
    }
  } else {
    kv.append(layer, static_cast<std::size_t>(pos), k_.data(), v_.data());
    const std::size_t n_res = kv.residents(resident_.data());
    const std::int32_t q_pos = resident_[n_res - 1].rope_pos;
    tensor::rope(q_.data(), &q_pos, cfg_.rope_theta, nh, 1, hd);

    const std::size_t stride = kv.capacity();
    const std::size_t nb = kv.n_blocks();
    const KVCache::Resident *res = resident_.data();
    const bool i8 = kv.int8();
    tensor::parallel_for(tensor::thread_pool(), nh, 1, [&](std::size_t hbegin, std::size_t hend) {
      std::vector<float> kbuf(hd, 0.0f);
      for (std::size_t h = hbegin; h < hend; ++h) {
        const std::size_t kh = h / gqa;
        const float *qh = q_.data() + h * hd;
        float *sc = scores_.data() + h * stride;
        for (std::size_t r = 0; r < n_res; ++r) {
          if (i8) {
            const std::int8_t *kp = kv.key_i8(layer, res[r].slot, kh);
            const float *ks = kv.key_scales(layer, res[r].slot, kh);
            for (std::size_t i = 0; i < hd; ++i)
              kbuf[i] = ks[i / kKvBlock] * static_cast<float>(kp[i]);
          } else {
            const float *kp = kv.key(layer, res[r].slot, kh);
            for (std::size_t i = 0; i < hd; ++i)
              kbuf[i] = kp[i];
          }
          const std::int32_t rp = res[r].rope_pos;
          tensor::rope(kbuf.data(), &rp, cfg_.rope_theta, 1, 1, hd);
          float dot = 0.0f;
          for (std::size_t i = 0; i < hd; ++i)
            dot += qh[i] * kbuf[i];
          sc[r] = dot * scale;
        }
        tensor::softmax(sc, sc, n_res);
        float *outh = attn_.data() + h * hd;
        for (std::size_t i = 0; i < hd; ++i)
          outh[i] = 0.0f;
        for (std::size_t r = 0; r < n_res; ++r) {
          const float w = sc[r];
          if (i8) {
            const std::int8_t *vp = kv.value_i8(layer, res[r].slot, kh);
            const float *vs = kv.value_scales(layer, res[r].slot, kh);
            for (std::size_t b = 0; b < nb; ++b) {
              const std::size_t start = b * kKvBlock;
              const std::size_t end = std::min(start + kKvBlock, hd);
              const float wv = w * vs[b];
              for (std::size_t i = start; i < end; ++i)
                outh[i] += wv * static_cast<float>(vp[i]);
            }
          } else {
            const float *vp = kv.value(layer, res[r].slot, kh);
            for (std::size_t i = 0; i < hd; ++i)
              outh[i] += w * vp[i];
          }
        }
        if (dbg != nullptr && dbg->attn_head0 != nullptr && h == 0)
          for (std::size_t i = 0; i < hd; ++i)
            dbg->attn_head0[i] = outh[i];
      }
    });
  }

  tensor::matvec_quant(L.attn_output, attn_.data(), proj_.data(), dim, nh * hd);
  for (std::size_t i = 0; i < dim; ++i)
    x[i] += proj_[i];

  const std::size_t ff = cfg_.ffn_length;
  tensor::rmsnorm(x, L.ffn_norm, cfg_.rms_eps, normed_.data(), 1, dim);
  // gate and up share the normed input; fuse them under one barrier.
  const tensor::MatvecJob gate_up[] = {
      {L.ffn_gate, ffn_gate_.data(), ff},
      {L.ffn_up, ffn_up_.data(), ff},
  };
  tensor::matvec_quant_fused(gate_up, 2, normed_.data(), dim);
  tensor::silu(ffn_gate_.data(), ffn_gate_.data(), ff);
  for (std::size_t i = 0; i < ff; ++i)
    ffn_gate_[i] *= ffn_up_[i];
  tensor::matvec_quant(L.ffn_down, ffn_gate_.data(), ffn_down_.data(), dim, ff);
  for (std::size_t i = 0; i < dim; ++i)
    x[i] += ffn_down_[i];

  if (dbg != nullptr && dbg->layer_out != nullptr)
    for (std::size_t i = 0; i < dim; ++i)
      dbg->layer_out[i] = x[i];
}

bool Model::layers_f16(std::size_t n) const {
  const std::size_t limit = std::min(n, layers_.size());
  for (std::size_t l = 0; l < limit; ++l) {
    const LayerWeights &L = layers_[l];
    const gguf::GgmlType t[] = {L.attn_q.type,   L.attn_k.type, L.attn_v.type,  L.attn_output.type,
                                L.ffn_gate.type, L.ffn_up.type, L.ffn_down.type};
    for (gguf::GgmlType ty : t)
      if (ty != GgmlType::F16)
        return false;
  }
  return true;
}

namespace {

std::optional<backend::WeightType> to_weight_type(gguf::GgmlType t) {
  switch (t) {
  case gguf::GgmlType::F16:
    return backend::WeightType::F16;
  case gguf::GgmlType::Q8_0:
    return backend::WeightType::Q8_0;
  case gguf::GgmlType::Q4_0:
    return backend::WeightType::Q4_0;
  default:
    return std::nullopt;
  }
}

backend::QWeight to_qweight(const tensor::QuantMatrix &w) {
  return {w.data, to_weight_type(w.type).value_or(backend::WeightType::F16)};
}

} // namespace

bool Model::layers_offloadable(std::size_t n) const {
  const std::size_t limit = std::min(n, layers_.size());
  for (std::size_t l = 0; l < limit; ++l) {
    const LayerWeights &L = layers_[l];
    const gguf::GgmlType t[] = {L.attn_q.type,   L.attn_k.type, L.attn_v.type,  L.attn_output.type,
                                L.ffn_gate.type, L.ffn_up.type, L.ffn_down.type};
    for (gguf::GgmlType ty : t)
      if (!to_weight_type(ty))
        return false;
  }
  return true;
}

bool Model::set_gpu_offload(backend::Backend *backend, std::size_t n_layers) {
  backend_ = backend;
  gpu_layers_ = n_layers;
  if (backend == nullptr || n_layers == 0)
    return true;

  std::vector<backend::LayerQ> gl(n_layers);
  for (std::size_t l = 0; l < n_layers; ++l) {
    const LayerWeights &L = layers_[l];
    backend::LayerQ &g = gl[l];
    g.attn_norm = L.attn_norm;
    g.ffn_norm = L.ffn_norm;
    g.wq = to_qweight(L.attn_q);
    g.wk = to_qweight(L.attn_k);
    g.wv = to_qweight(L.attn_v);
    g.wo = to_qweight(L.attn_output);
    g.wgate = to_qweight(L.ffn_gate);
    g.wup = to_qweight(L.ffn_up);
    g.wdown = to_qweight(L.ffn_down);
    g.bq = L.attn_q_bias;
    g.bk = L.attn_k_bias;
    g.bv = L.attn_v_bias;
    g.dim = cfg_.embedding_length;
    g.ff = cfg_.ffn_length;
    g.n_heads = cfg_.n_heads;
    g.n_kv_heads = cfg_.n_kv_heads;
    g.head_dim = cfg_.head_dim;
    g.rms_eps = cfg_.rms_eps;
    g.rope_theta = cfg_.rope_theta;
  }
  // the on-device final norm and lm_head only run when every block is offloaded
  // and the head dtype is one the Metal matvec reads. otherwise the CPU keeps
  // the tail and a null head disables decode_token_full.
  gpu_head_ = n_layers == cfg_.n_layers && to_weight_type(lm_head_.type).has_value();
  const backend::QWeight head =
      gpu_head_ ? to_qweight(lm_head_) : backend::QWeight{nullptr, backend::WeightType::F16};
  auto r = backend_->setup_offload(gl, kv_.dense_k(), kv_.dense_v(), kv_.capacity(),
                                   kv_.pos_stride(), output_norm_, head, cfg_.vocab_size);
  if (!r) {
    std::fprintf(stderr, "metal: offload setup failed (%s), staying on CPU\n",
                 r.error().message.c_str());
    gpu_layers_ = 0;
    gpu_head_ = false;
    return false;
  }
  return true;
}

void Model::decode_token_gpu(float *x, std::int32_t pos) {
  auto r = backend_->decode_token(x, pos);
  if (!r) {
    for (std::size_t l = 0; l < gpu_layers_; ++l)
      decode_layer(l, x, pos, kv_, nullptr);
    return;
  }
  kv_.mark_position(static_cast<std::size_t>(pos));
}

const float *Model::forward(std::int32_t token, std::int32_t pos) {
  const std::size_t dim = cfg_.embedding_length;
  embed(token, x_.data());
  const bool offload = backend_ != nullptr && gpu_layers_ > 0 && kv_.dense_f32();
  if (offload && gpu_head_) {
    if (auto r = backend_->decode_token_full(x_.data(), pos)) {
      kv_.mark_position(static_cast<std::size_t>(pos));
      return *r;
    }
    // GPU tail failed, redo the whole token on CPU from the embedding.
    for (std::size_t l = 0; l < cfg_.n_layers; ++l)
      decode_layer(l, x_.data(), pos, kv_, nullptr);
  } else {
    if (offload)
      decode_token_gpu(x_.data(), pos);
    for (std::size_t l = offload ? gpu_layers_ : 0; l < cfg_.n_layers; ++l)
      decode_layer(l, x_.data(), pos, kv_, nullptr);
  }
  tensor::rmsnorm(x_.data(), output_norm_, cfg_.rms_eps, normed_.data(), 1, dim);
  tensor::matvec_quant(lm_head_, normed_.data(), logits_.data(), cfg_.vocab_size, dim);
  return logits_.data();
}

void Model::decode_layer_chunk(std::size_t layer, float *x, std::int32_t pos0, std::size_t n,
                               ChunkScratch &s, KVCache &kv) {
  const LayerWeights &L = layers_[layer];
  const std::size_t dim = cfg_.embedding_length;
  const std::size_t hd = cfg_.head_dim;
  const std::size_t nh = cfg_.n_heads;
  const std::size_t nkv = cfg_.n_kv_heads;
  const std::size_t gqa = cfg_.gqa_factor;

  tensor::rmsnorm(x, L.attn_norm, cfg_.rms_eps, s.normed.data(), n, dim);
  tensor::matmul_quant(L.attn_q, s.normed.data(), s.q.data(), n, nh * hd, dim);
  tensor::matmul_quant(L.attn_k, s.normed.data(), s.k.data(), n, nkv * hd, dim);
  tensor::matmul_quant(L.attn_v, s.normed.data(), s.v.data(), n, nkv * hd, dim);

  // bias, rope, and append every position before any query attends, so each
  // token still reads only its own causal prefix within the chunk.
  for (std::size_t c = 0; c < n; ++c) {
    float *qc = s.q.data() + c * nh * hd;
    float *kc = s.k.data() + c * nkv * hd;
    float *vc = s.v.data() + c * nkv * hd;
    if (L.attn_q_bias)
      for (std::size_t i = 0; i < nh * hd; ++i)
        qc[i] += L.attn_q_bias[i];
    if (L.attn_k_bias)
      for (std::size_t i = 0; i < nkv * hd; ++i)
        kc[i] += L.attn_k_bias[i];
    if (L.attn_v_bias)
      for (std::size_t i = 0; i < nkv * hd; ++i)
        vc[i] += L.attn_v_bias[i];
    const std::int32_t p = pos0 + static_cast<std::int32_t>(c);
    tensor::rope(qc, &p, cfg_.rope_theta, nh, 1, hd);
    tensor::rope(kc, &p, cfg_.rope_theta, nkv, 1, hd);
    kv.append(layer, static_cast<std::size_t>(p), kc, vc);
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::size_t last = static_cast<std::size_t>(pos0) + n - 1;
  tensor::parallel_for(tensor::thread_pool(), nh, 1, [&](std::size_t hbegin, std::size_t hend) {
    std::vector<float> sc(last + 1, 0.0f);
    for (std::size_t h = hbegin; h < hend; ++h) {
      const std::size_t kh = h / gqa;
      for (std::size_t c = 0; c < n; ++c) {
        const std::size_t pos_c = static_cast<std::size_t>(pos0) + c;
        const float *qh = s.q.data() + c * nh * hd + h * hd;
        for (std::size_t pp = 0; pp <= pos_c; ++pp) {
          const float *kp = kv.key(layer, pp, kh);
          float dot = 0.0f;
          for (std::size_t i = 0; i < hd; ++i)
            dot += qh[i] * kp[i];
          sc[pp] = dot * scale;
        }
        tensor::softmax(sc.data(), sc.data(), pos_c + 1);
        float *outh = s.attn.data() + c * nh * hd + h * hd;
        for (std::size_t i = 0; i < hd; ++i)
          outh[i] = 0.0f;
        for (std::size_t pp = 0; pp <= pos_c; ++pp) {
          const float *vp = kv.value(layer, pp, kh);
          const float w = sc[pp];
          for (std::size_t i = 0; i < hd; ++i)
            outh[i] += w * vp[i];
        }
      }
    }
  });

  tensor::matmul_quant(L.attn_output, s.attn.data(), s.proj.data(), n, dim, nh * hd);
  for (std::size_t i = 0; i < n * dim; ++i)
    x[i] += s.proj[i];

  const std::size_t ff = cfg_.ffn_length;
  tensor::rmsnorm(x, L.ffn_norm, cfg_.rms_eps, s.normed.data(), n, dim);
  tensor::matmul_quant(L.ffn_gate, s.normed.data(), s.gate.data(), n, ff, dim);
  tensor::matmul_quant(L.ffn_up, s.normed.data(), s.up.data(), n, ff, dim);
  tensor::silu(s.gate.data(), s.gate.data(), n * ff);
  for (std::size_t i = 0; i < n * ff; ++i)
    s.gate[i] *= s.up[i];
  tensor::matmul_quant(L.ffn_down, s.gate.data(), s.down.data(), n, dim, ff);
  for (std::size_t i = 0; i < n * dim; ++i)
    x[i] += s.down[i];
}

void Model::forward_chunk(const std::int32_t *tokens, std::int32_t pos0, std::size_t n,
                          float *out) {
  const std::size_t dim = cfg_.embedding_length;
  const std::size_t hd = cfg_.head_dim;
  const std::size_t nh = cfg_.n_heads;
  const std::size_t nkv = cfg_.n_kv_heads;
  const std::size_t ff = cfg_.ffn_length;

  ChunkScratch s;
  s.x.assign(n * dim, 0.0f);
  s.normed.assign(n * dim, 0.0f);
  s.q.assign(n * nh * hd, 0.0f);
  s.k.assign(n * nkv * hd, 0.0f);
  s.v.assign(n * nkv * hd, 0.0f);
  s.attn.assign(n * nh * hd, 0.0f);
  s.proj.assign(n * dim, 0.0f);
  s.gate.assign(n * ff, 0.0f);
  s.up.assign(n * ff, 0.0f);
  s.down.assign(n * dim, 0.0f);

  for (std::size_t c = 0; c < n; ++c)
    embed(tokens[c], s.x.data() + c * dim);
  for (std::size_t l = 0; l < cfg_.n_layers; ++l)
    decode_layer_chunk(l, s.x.data(), pos0, n, s, kv_);
  tensor::rmsnorm(s.x.data(), output_norm_, cfg_.rms_eps, s.normed.data(), n, dim);
  tensor::matmul_quant(lm_head_, s.normed.data(), out, n, cfg_.vocab_size, dim);
}

namespace {

// on-disk prefix cache is little-endian fixed-width. header is magic, version,
// (n_layers, n_kv_heads, head_dim, prefix_len), then prefix_len int32 token
// ids, then prefix_elems fp32 keys followed by the same count of values.
constexpr std::array<std::byte, 8> kKvMagic{std::byte{'D'}, std::byte{'B'}, std::byte{'K'},
                                            std::byte{'V'}, std::byte{'C'}, std::byte{'A'},
                                            std::byte{'C'}, std::byte{'H'}};
constexpr std::uint32_t kKvVersion = 1;

template <typename T> void put_scalar(std::vector<std::byte> &buf, const T &v) {
  const std::size_t off = buf.size();
  buf.resize(off + sizeof(T));
  std::memcpy(buf.data() + off, &v, sizeof(T));
}

struct ByteReader {
  const std::byte *base;
  std::size_t size;
  std::size_t pos = 0;

  template <typename T> std::expected<T, Error> scalar() {
    if (pos + sizeof(T) > size)
      return std::unexpected(Error{"kv prefix cache truncated", "", pos});
    T v;
    std::memcpy(&v, base + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
};

std::expected<std::vector<std::byte>, Error> read_whole_file(const std::string &path) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (f == nullptr)
    return std::unexpected(Error{"cannot open kv prefix cache " + path, path, 0});
  std::vector<std::byte> data;
  std::byte buf[65536];
  std::size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof buf, f)) > 0)
    data.insert(data.end(), buf, buf + got);
  const bool ok = std::ferror(f) == 0;
  std::fclose(f);
  if (!ok)
    return std::unexpected(Error{"read error on kv prefix cache " + path, path, 0});
  return data;
}

} // namespace

std::expected<void, Error>
Model::save_kv_prefix(const std::string &path, std::span<const std::int32_t> prefix_tokens) const {
  if (!kv_.dense_f32())
    return std::unexpected(Error{"kv prefix cache supports only the dense fp32 cache", path, 0});
  const std::size_t prefix_len = prefix_tokens.size();
  if (prefix_len == 0)
    return std::unexpected(Error{"kv prefix is empty", path, 0});
  if (kv_.n_seen() < prefix_len)
    return std::unexpected(Error{"cache holds fewer positions than the prefix", path, 0});

  std::vector<std::byte> buf;
  for (std::byte b : kKvMagic)
    put_scalar(buf, b);
  put_scalar(buf, kKvVersion);
  put_scalar(buf, static_cast<std::uint32_t>(cfg_.n_layers));
  put_scalar(buf, static_cast<std::uint32_t>(cfg_.n_kv_heads));
  put_scalar(buf, static_cast<std::uint32_t>(cfg_.head_dim));
  put_scalar(buf, static_cast<std::uint32_t>(prefix_len));
  for (std::int32_t t : prefix_tokens)
    put_scalar(buf, t);

  const std::size_t elems = kv_.prefix_elems(prefix_len);
  std::vector<float> kbuf(elems);
  std::vector<float> vbuf(elems);
  kv_.copy_prefix_out(prefix_len, kbuf.data(), vbuf.data());
  const std::size_t off = buf.size();
  buf.resize(off + 2 * elems * sizeof(float));
  std::memcpy(buf.data() + off, kbuf.data(), elems * sizeof(float));
  std::memcpy(buf.data() + off + elems * sizeof(float), vbuf.data(), elems * sizeof(float));

  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr)
    return std::unexpected(Error{"cannot open kv prefix cache for writing " + path, path, 0});
  const std::size_t wrote = std::fwrite(buf.data(), 1, buf.size(), f);
  const bool ok = wrote == buf.size() && std::ferror(f) == 0;
  std::fclose(f);
  if (!ok)
    return std::unexpected(Error{"write error on kv prefix cache " + path, path, 0});
  return {};
}

std::expected<std::size_t, Error>
Model::load_kv_prefix(const std::string &path, std::span<const std::int32_t> prefix_tokens) {
  if (!kv_.dense_f32())
    return std::unexpected(Error{"kv prefix cache supports only the dense fp32 cache", path, 0});

  std::vector<std::byte> data = TRY(read_whole_file(path));
  ByteReader r{data.data(), data.size()};

  for (std::byte want : kKvMagic) {
    const std::byte got = TRY(r.scalar<std::byte>());
    if (got != want)
      return std::unexpected(Error{"kv prefix cache bad magic", path, 0});
  }
  const std::uint32_t version = TRY(r.scalar<std::uint32_t>());
  if (version != kKvVersion)
    return std::unexpected(
        Error{"kv prefix cache version " + std::to_string(version) + " unsupported", path, 0});

  const std::uint32_t n_layers = TRY(r.scalar<std::uint32_t>());
  const std::uint32_t n_kv_heads = TRY(r.scalar<std::uint32_t>());
  const std::uint32_t head_dim = TRY(r.scalar<std::uint32_t>());
  const std::uint32_t prefix_len = TRY(r.scalar<std::uint32_t>());
  if (n_layers != cfg_.n_layers || n_kv_heads != cfg_.n_kv_heads || head_dim != cfg_.head_dim)
    return std::unexpected(Error{"kv prefix cache hyperparams do not match this model", path, 0});
  if (prefix_len == 0)
    return std::unexpected(Error{"kv prefix cache has a zero-length prefix", path, 0});
  if (prefix_len > prefix_tokens.size())
    return std::unexpected(Error{"kv prefix is longer than the prompt", path, 0});
  if (prefix_len > kv_.capacity())
    return std::unexpected(Error{"kv prefix exceeds cache capacity", path, 0});

  for (std::uint32_t i = 0; i < prefix_len; ++i) {
    const std::int32_t saved = TRY(r.scalar<std::int32_t>());
    if (saved != prefix_tokens[i])
      return std::unexpected(
          Error{"prompt prefix differs from saved cache at token " + std::to_string(i), path, 0});
  }

  const std::size_t elems = kv_.prefix_elems(prefix_len);
  if (r.pos + 2 * elems * sizeof(float) != data.size())
    return std::unexpected(Error{"kv prefix cache payload size mismatch", path, 0});

  std::vector<float> kbuf(elems);
  std::vector<float> vbuf(elems);
  std::memcpy(kbuf.data(), data.data() + r.pos, elems * sizeof(float));
  std::memcpy(vbuf.data(), data.data() + r.pos + elems * sizeof(float), elems * sizeof(float));
  kv_.copy_prefix_in(prefix_len, kbuf.data(), vbuf.data());
  return static_cast<std::size_t>(prefix_len);
}

} // namespace dbinfer::model
