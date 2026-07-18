#ifndef DBINFER_TOKENIZER_UNICODE_HPP
#define DBINFER_TOKENIZER_UNICODE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dbinfer::tokenizer {

// UTF-8 -> codepoints. Invalid bytes are consumed one at a time and emitted as
// U+FFFD, matching llama.cpp unicode_cpts_from_utf8 so invalid input tokenizes
// identically to the oracle.
std::vector<std::uint32_t> utf8_to_cpts(std::string_view s);

std::string cpt_to_utf8(std::uint32_t cpt);

// Expected UTF-8 sequence length from a lead byte (llama.cpp unicode_len_utf8).
std::size_t utf8_len(unsigned char lead);

bool is_letter(std::uint32_t cpt);
bool is_number(std::uint32_t cpt);
bool is_whitespace(std::uint32_t cpt);

// GPT2 byte-level codec. byte_to_unicode maps a raw byte to the UTF-8 of its
// mapped codepoint; cpt_to_byte reverses a single mapped codepoint.
const std::string &byte_to_unicode(std::uint8_t b);
bool cpt_to_byte(std::uint32_t cpt, std::uint8_t &out);

} // namespace dbinfer::tokenizer

#endif // DBINFER_TOKENIZER_UNICODE_HPP
