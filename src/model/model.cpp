#include "model/model.hpp"

#include "tensor/dequant.hpp"
#include "tensor/matmul.hpp"
#include "tensor/ops.hpp"
#include "tensor/thread_pool.hpp"

#include <cmath>
#include <optional>
#include <string>

namespace dbinfer::model {

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

// bytes for one 32-element block of a weight dtype
std::size_t block_bytes_q32(GgmlType type) {
  switch (type) {
  case GgmlType::F16:
    return 64;
  case GgmlType::Q8_0:
    return 34;
  case GgmlType::Q4_0:
    return 18;
  default:
    return 0;
  }
}

} // namespace

KVCache::KVCache(std::size_t n_layers, std::size_t max_seq, std::size_t n_kv_heads,
                 std::size_t head_dim)
    : layer_stride_(max_seq * n_kv_heads * head_dim), pos_stride_(n_kv_heads * head_dim),
      n_kv_heads_(n_kv_heads), head_dim_(head_dim) {
  k_.assign(n_layers * layer_stride_, 0.0f);
  v_.assign(n_layers * layer_stride_, 0.0f);
}

void KVCache::append(std::size_t layer, std::size_t pos, const float *k, const float *v) {
  const std::size_t base = layer * layer_stride_ + pos * pos_stride_;
  for (std::size_t i = 0; i < pos_stride_; ++i) {
    k_[base + i] = k[i];
    v_[base + i] = v[i];
  }
}

const float *KVCache::key(std::size_t layer, std::size_t pos, std::size_t kv_head) const {
  return k_.data() + layer * layer_stride_ + pos * pos_stride_ + kv_head * head_dim_;
}

const float *KVCache::value(std::size_t layer, std::size_t pos, std::size_t kv_head) const {
  return v_.data() + layer * layer_stride_ + pos * pos_stride_ + kv_head * head_dim_;
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
    if (t->type != GgmlType::F16 && t->type != GgmlType::Q8_0 && t->type != GgmlType::Q4_0)
      return std::unexpected(Error{name + " expected F16, Q8_0, or Q4_0", "", 0});
    if (t->shape[0] != out || t->shape[1] != in)
      return std::unexpected(Error{
          name + " shape mismatch: expected [" + std::to_string(out) + ", " + std::to_string(in) +
              "], found [" + std::to_string(t->shape[0]) + ", " + std::to_string(t->shape[1]) + "]",
          "", t->offset});
    if (in % tensor::kBlockSize != 0)
      return std::unexpected(Error{name + " in not a multiple of 32", "", t->offset});
    std::size_t elems = 0;
    if (__builtin_mul_overflow(out, in, &elems))
      return std::unexpected(Error{name + " out*in overflow", "", t->offset});
    if (elems % tensor::kBlockSize != 0)
      return std::unexpected(Error{name + " out*in not a multiple of 32", "", t->offset});
    const std::size_t expect = (elems / tensor::kBlockSize) * block_bytes_q32(t->type);
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
  m.kv_ = KVCache(cfg.n_layers, cfg.context_length, cfg.n_kv_heads, hd);

  return m;
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
  tensor::matvec_quant(L.attn_q, normed_.data(), q_.data(), nh * hd, dim);
  tensor::matvec_quant(L.attn_k, normed_.data(), k_.data(), nkv * hd, dim);
  tensor::matvec_quant(L.attn_v, normed_.data(), v_.data(), nkv * hd, dim);
  if (L.attn_q_bias)
    for (std::size_t i = 0; i < nh * hd; ++i)
      q_[i] += L.attn_q_bias[i];
  if (L.attn_k_bias)
    for (std::size_t i = 0; i < nkv * hd; ++i)
      k_[i] += L.attn_k_bias[i];
  if (L.attn_v_bias)
    for (std::size_t i = 0; i < nkv * hd; ++i)
      v_[i] += L.attn_v_bias[i];

  const std::int32_t p = pos;
  tensor::rope(q_.data(), &p, cfg_.rope_theta, nh, 1, hd);
  tensor::rope(k_.data(), &p, cfg_.rope_theta, nkv, 1, hd);

  kv.append(layer, static_cast<std::size_t>(pos), k_.data(), v_.data());

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::size_t last = static_cast<std::size_t>(pos);
  const std::size_t cl = cfg_.context_length;
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

  tensor::matvec_quant(L.attn_output, attn_.data(), proj_.data(), dim, nh * hd);
  for (std::size_t i = 0; i < dim; ++i)
    x[i] += proj_[i];

  const std::size_t ff = cfg_.ffn_length;
  tensor::rmsnorm(x, L.ffn_norm, cfg_.rms_eps, normed_.data(), 1, dim);
  tensor::matvec_quant(L.ffn_gate, normed_.data(), ffn_gate_.data(), ff, dim);
  tensor::matvec_quant(L.ffn_up, normed_.data(), ffn_up_.data(), ff, dim);
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

const float *Model::forward(std::int32_t token, std::int32_t pos) {
  const std::size_t dim = cfg_.embedding_length;
  embed(token, x_.data());
  for (std::size_t l = 0; l < cfg_.n_layers; ++l)
    decode_layer(l, x_.data(), pos, kv_, nullptr);
  tensor::rmsnorm(x_.data(), output_norm_, cfg_.rms_eps, normed_.data(), 1, dim);
  tensor::matvec_quant(lm_head_, normed_.data(), logits_.data(), cfg_.vocab_size, dim);
  return logits_.data();
}

} // namespace dbinfer::model
