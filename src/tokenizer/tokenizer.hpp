#ifndef DBINFER_TOKENIZER_TOKENIZER_HPP
#define DBINFER_TOKENIZER_TOKENIZER_HPP

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gguf/gguf.hpp"

namespace dbinfer::tokenizer {

// boost-style hash_combine for a pair of strings, so BpeRanks can be an
// unordered_map without a custom bucket structure.
struct PairHash {
  std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
    std::hash<std::string> h;
    std::size_t a = h(p.first);
    std::size_t b = h(p.second);
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
  }
};

using BpeRanks = std::unordered_map<std::pair<std::string, std::string>, std::int32_t, PairHash>;
using Vocab = std::unordered_map<std::string, std::int32_t>;

// splits text into pretoken chunks per the Qwen2/GPT2-style pretokenizer
// regex (contractions, letter runs, digit runs, punctuation runs, whitespace
// runs), then byte-encodes each chunk so raw bytes survive BPE as printable
// codepoints. Each returned string is fed to bpe_word independently.
std::vector<std::string> pretokenize(std::string_view text);

// runs BPE merges on one byte-encoded word using ranks (lower rank = merge
// earlier), appending the resulting vocab ids to out. Falls back to
// single-byte tokens for any final piece missing from vocab.
void bpe_word(const std::string& word, const BpeRanks& ranks, const Vocab& vocab,
              std::vector<std::int32_t>& out);

// GPT2-family BPE tokenizer, built entirely from a GGUF file's
// tokenizer.ggml.* metadata (vocab, merge ranks, special token ids). Only
// the "gpt2" tokenizer model is supported; from_gguf rejects anything else.
class Tokenizer {
 public:
  static std::expected<Tokenizer, gguf::Error> from_gguf(const gguf::GgufFile& file);

  // pretokenizes and BPE-encodes text, optionally prefixing bos_id() when
  // the model's metadata says to add one.
  std::vector<std::int32_t> encode(std::string_view text, bool add_special) const;
  // joins ids' token strings and reverses the byte-level codec, so the
  // result is the original UTF-8 text (unknown ids are silently skipped).
  std::string decode(std::span<const std::int32_t> ids) const;

  std::int32_t bos_id() const { return bos_id_; }
  std::int32_t eos_id() const { return eos_id_; }

 private:
  std::vector<std::string> id_to_token_;
  Vocab token_to_id_;
  BpeRanks ranks_;
  std::int32_t bos_id_ = -1;
  std::int32_t eos_id_ = -1;
  bool add_bos_ = false;
};

}  // namespace dbinfer::tokenizer

#endif  // DBINFER_TOKENIZER_TOKENIZER_HPP
