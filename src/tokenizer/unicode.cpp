#include "tokenizer/unicode.hpp"

#include <algorithm>
#include <array>
#include <unordered_map>

#include "tokenizer/unicode_data.inc"

namespace dbinfer::tokenizer {

namespace {

constexpr std::uint16_t kFlagNumber = 0x0002;
constexpr std::uint16_t kFlagLetter = 0x0004;

// kRangeStart holds the sorted start of each contiguous flag run; upper_bound
// finds the first range starting after cpt, so the run containing cpt is the
// one before it.
std::uint16_t flags_for(std::uint32_t cpt) {
  if (cpt >= 0x110000) return 0x0001;  // UNDEFINED
  const std::uint32_t* begin = detail::kRangeStart;
  const std::uint32_t* end = detail::kRangeStart + detail::kRangesCount;
  const std::uint32_t* it = std::upper_bound(begin, end, cpt);
  std::size_t idx = static_cast<std::size_t>(it - begin) - 1;
  return detail::kRangeFlags[idx];
}

// GPT2's bytes_to_unicode: bytes that are already printable single-byte
// Latin-1 characters map to themselves; every other byte (controls, etc.)
// gets an arbitrary unused codepoint starting at 256, so all 256 byte values
// end up printable and round-trippable through cpt_to_byte.
struct ByteCodec {
  std::array<std::uint32_t, 256> byte_to_cpt{};
  std::array<std::string, 256> byte_to_str{};
  std::unordered_map<std::uint32_t, std::uint8_t> cpt_to_byte;

  ByteCodec() {
    std::array<bool, 256> direct{};
    auto mark = [&](int lo, int hi) {
      for (int c = lo; c <= hi; ++c) direct[static_cast<std::size_t>(c)] = true;
    };
    mark(0x21, 0x7E);
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
      std::uint32_t cpt;
      if (direct[static_cast<std::size_t>(b)]) {
        cpt = static_cast<std::uint32_t>(b);
      } else {
        cpt = static_cast<std::uint32_t>(256 + n);
        ++n;
      }
      byte_to_cpt[static_cast<std::size_t>(b)] = cpt;
      byte_to_str[static_cast<std::size_t>(b)] = cpt_to_utf8(cpt);
      cpt_to_byte[cpt] = static_cast<std::uint8_t>(b);
    }
  }
};

const ByteCodec& codec() {
  static const ByteCodec c;
  return c;
}

}  // namespace

std::size_t utf8_len(unsigned char lead) {
  static const std::size_t lookup[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  return lookup[lead >> 4];
}

std::vector<std::uint32_t> utf8_to_cpts(std::string_view s) {
  std::vector<std::uint32_t> out;
  out.reserve(s.size());
  std::size_t i = 0;
  const std::size_t n = s.size();
  auto cont = [&](std::size_t k) {
    return k < n && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80;
  };
  while (i < n) {
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (!(c0 & 0x80)) {
      out.push_back(c0);
      i += 1;
    } else if (!(c0 & 0x40)) {
      out.push_back(0xFFFD);
      i += 1;
    } else if (!(c0 & 0x20)) {
      if (cont(i + 1)) {
        out.push_back(((c0 & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu));
        i += 2;
      } else {
        out.push_back(0xFFFD);
        i += 1;
      }
    } else if (!(c0 & 0x10)) {
      if (cont(i + 1) && cont(i + 2)) {
        out.push_back(((c0 & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                      (static_cast<unsigned char>(s[i + 2]) & 0x3Fu));
        i += 3;
      } else {
        out.push_back(0xFFFD);
        i += 1;
      }
    } else if (!(c0 & 0x08)) {
      if (cont(i + 1) && cont(i + 2) && cont(i + 3)) {
        out.push_back(((c0 & 0x07u) << 18) |
                      ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                      ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                      (static_cast<unsigned char>(s[i + 3]) & 0x3Fu));
        i += 4;
      } else {
        out.push_back(0xFFFD);
        i += 1;
      }
    } else {
      out.push_back(0xFFFD);
      i += 1;
    }
  }
  return out;
}

std::string cpt_to_utf8(std::uint32_t cpt) {
  std::string r;
  if (cpt <= 0x7F) {
    r.push_back(static_cast<char>(cpt));
  } else if (cpt <= 0x7FF) {
    r.push_back(static_cast<char>(0xC0 | ((cpt >> 6) & 0x1F)));
    r.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else if (cpt <= 0xFFFF) {
    r.push_back(static_cast<char>(0xE0 | ((cpt >> 12) & 0x0F)));
    r.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    r.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else {
    r.push_back(static_cast<char>(0xF0 | ((cpt >> 18) & 0x07)));
    r.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
    r.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    r.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  }
  return r;
}

bool is_letter(std::uint32_t cpt) { return (flags_for(cpt) & kFlagLetter) != 0; }
bool is_number(std::uint32_t cpt) { return (flags_for(cpt) & kFlagNumber) != 0; }

bool is_whitespace(std::uint32_t cpt) {
  for (std::uint32_t i = 0; i < detail::kWhitespaceCount; ++i)
    if (detail::kWhitespace[i] == cpt) return true;
  return false;
}

const std::string& byte_to_unicode(std::uint8_t b) { return codec().byte_to_str[b]; }

bool cpt_to_byte(std::uint32_t cpt, std::uint8_t& out) {
  const auto& m = codec().cpt_to_byte;
  auto it = m.find(cpt);
  if (it == m.end()) return false;
  out = it->second;
  return true;
}

}  // namespace dbinfer::tokenizer
