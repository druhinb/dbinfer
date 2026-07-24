#include "tokenizer/tokenizer.hpp"

#include <cstdint>
#include <optional>
#include <variant>

#include "tokenizer/unicode.hpp"

namespace dbinfer::tokenizer {

namespace {

using gguf::Error;
using gguf::MetaArray;
using gguf::MetaValue;

const MetaArray* find_array(const gguf::GgufFile& f, std::string_view key) {
  const MetaValue* mv = f.find_meta(key);
  if (mv == nullptr) return nullptr;
  return std::get_if<MetaArray>(&mv->value);
}

std::optional<std::int64_t> as_int(const MetaValue& mv) {
  if (auto p = std::get_if<std::uint32_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::int32_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::uint64_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::int64_t>(&mv.value)) return *p;
  if (auto p = std::get_if<std::uint16_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  return std::nullopt;
}

}  // namespace

std::expected<Tokenizer, Error> Tokenizer::from_gguf(const gguf::GgufFile& file) {
  Tokenizer tok;

  const MetaValue* model_mv = file.find_meta("tokenizer.ggml.model");
  const std::string* model = model_mv ? std::get_if<std::string>(&model_mv->value) : nullptr;
  if (model == nullptr)
    return std::unexpected(Error{"tokenizer.ggml.model missing or not a string", "", 0});
  if (*model != "gpt2")
    return std::unexpected(
        Error{"unsupported tokenizer model '" + *model + "', expected 'gpt2'", "", 0});

  const MetaArray* tokens = find_array(file, "tokenizer.ggml.tokens");
  if (tokens == nullptr)
    return std::unexpected(Error{"tokenizer.ggml.tokens missing or not an array", "", 0});

  tok.id_to_token_.reserve(tokens->values.size());
  for (std::size_t i = 0; i < tokens->values.size(); ++i) {
    const std::string* s = std::get_if<std::string>(&tokens->values[i].value);
    if (s == nullptr)
      return std::unexpected(Error{"tokenizer.ggml.tokens element is not a string", "", i});
    tok.token_to_id_.emplace(*s, static_cast<std::int32_t>(i));
    tok.id_to_token_.push_back(*s);
  }

  const MetaArray* types = find_array(file, "tokenizer.ggml.token_type");
  if (types == nullptr)
    return std::unexpected(Error{"tokenizer.ggml.token_type missing or not an array", "", 0});
  if (types->values.size() != tokens->values.size())
    return std::unexpected(Error{"tokenizer.ggml.token_type length != tokens length", "", 0});
  for (const MetaValue& v : types->values) {
    if (std::get_if<std::int32_t>(&v.value) == nullptr)
      return std::unexpected(Error{"tokenizer.ggml.token_type element is not int32", "", 0});
  }

  const MetaArray* merges = find_array(file, "tokenizer.ggml.merges");
  if (merges == nullptr)
    return std::unexpected(Error{"tokenizer.ggml.merges missing or not an array", "", 0});
  tok.ranks_.reserve(merges->values.size());
  for (std::size_t i = 0; i < merges->values.size(); ++i) {
    const std::string* m = std::get_if<std::string>(&merges->values[i].value);
    if (m == nullptr)
      return std::unexpected(Error{"tokenizer.ggml.merges element is not a string", "", i});
    std::size_t pos = m->find(' ', 1);
    if (pos == std::string::npos)
      return std::unexpected(Error{"malformed merge rule '" + *m + "'", "", i});
    tok.ranks_.emplace(std::make_pair(m->substr(0, pos), m->substr(pos + 1)),
                       static_cast<std::int32_t>(i));
  }

  if (const MetaValue* bos = file.find_meta("tokenizer.ggml.bos_token_id"))
    if (auto v = as_int(*bos)) tok.bos_id_ = static_cast<std::int32_t>(*v);
  if (const MetaValue* eos = file.find_meta("tokenizer.ggml.eos_token_id"))
    if (auto v = as_int(*eos)) tok.eos_id_ = static_cast<std::int32_t>(*v);
  if (const MetaValue* add = file.find_meta("tokenizer.ggml.add_bos_token"))
    if (auto b = std::get_if<bool>(&add->value)) tok.add_bos_ = *b;

  return tok;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool add_special) const {
  std::vector<std::int32_t> out;
  if (add_special && add_bos_ && bos_id_ >= 0) out.push_back(bos_id_);
  for (const std::string& word : pretokenize(text)) bpe_word(word, ranks_, token_to_id_, out);
  return out;
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids) const {
  std::string joined;
  for (std::int32_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) continue;
    joined += id_to_token_[static_cast<std::size_t>(id)];
  }

  std::string out;
  for (std::uint32_t cpt : utf8_to_cpts(joined)) {
    std::uint8_t b;
    if (cpt_to_byte(cpt, b))
      out.push_back(static_cast<char>(b));
    else
      out += cpt_to_utf8(cpt);
  }
  return out;
}

}  // namespace dbinfer::tokenizer
