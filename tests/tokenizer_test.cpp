// ctest for the byte-level BPE tokenizer

#include "tokenizer/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gguf/gguf.hpp"
#include "tokenizer/fixture_generated.hpp"

#ifndef DBINFER_TEST_GGUF
#error "DBINFER_TEST_GGUF must be defined by the build (path to the qwen2.5-0.5b gguf)"
#endif

namespace {

int g_failures = 0;

std::string as_bytes(std::initializer_list<std::uint8_t> in) {
  std::string s;
  for (std::uint8_t b : in) s.push_back(static_cast<char>(b));
  return s;
}

bool is_valid_utf8(const std::string& s) {
  std::size_t i = 0, n = s.size();
  auto cont = [&](std::size_t k) {
    return k < n && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80;
  };
  while (i < n) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (!(c & 0x80)) {
      i += 1;
    } else if ((c & 0xE0) == 0xC0) {
      if (!cont(i + 1)) return false;
      i += 2;
    } else if ((c & 0xF0) == 0xE0) {
      if (!cont(i + 1) || !cont(i + 2)) return false;
      i += 3;
    } else if ((c & 0xF8) == 0xF0) {
      if (!cont(i + 1) || !cont(i + 2) || !cont(i + 3)) return false;
      i += 4;
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  auto loaded = dbinfer::gguf::load(DBINFER_TEST_GGUF);
  if (!loaded) {
    std::printf("FAIL cannot load gguf: %s\n", dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }
  auto tok = dbinfer::tokenizer::Tokenizer::from_gguf(*loaded);
  if (!tok) {
    std::printf("FAIL from_gguf: %s\n", dbinfer::gguf::to_string(tok.error()).c_str());
    return 1;
  }

  constexpr std::size_t n =
      sizeof(dbinfer::tokenizer_fixture::kCases) / sizeof(dbinfer::tokenizer_fixture::kCases[0]);

  int parity_fail = 0, roundtrip_fail = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& c = dbinfer::tokenizer_fixture::kCases[i];
    std::string input = as_bytes(c.input);

    std::vector<std::int32_t> got = tok->encode(input, /*add_special=*/false);
    std::vector<std::int32_t> want(c.ids.begin(), c.ids.end());
    if (got != want) {
      std::printf("FAIL parity case %zu (%zu bytes): got %zu ids, want %zu\n", i, input.size(),
                  got.size(), want.size());
      std::printf("     got: ");
      for (auto x : got) std::printf("%d ", x);
      std::printf("\n     want:");
      for (auto x : want) std::printf("%d ", x);
      std::printf("\n");
      ++parity_fail;
    }

    std::string dec = tok->decode(want);
    // Valid UTF-8 must round-trip exactly; invalid input round-trips at the byte
    // level too (llama.cpp maps invalid bytes through U+FFFD, so we compare the
    // decode of the recorded ids against that same lossy-but-deterministic form
    // only when the input was valid UTF-8).
    if (is_valid_utf8(input)) {
      if (dec != input) {
        std::printf("FAIL roundtrip case %zu: decode mismatch\n", i);
        ++roundtrip_fail;
      }
    }
  }

  // Byte-level round-trip on the encoder path: decode(encode(x)) reproduces the
  // encoder's own view of x for every case, valid or not.
  int enc_roundtrip_fail = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& c = dbinfer::tokenizer_fixture::kCases[i];
    std::string input = as_bytes(c.input);
    std::string dec = tok->decode(tok->encode(input, false));
    if (is_valid_utf8(input) && dec != input) {
      std::printf("FAIL enc-roundtrip case %zu\n", i);
      ++enc_roundtrip_fail;
    }
  }

  g_failures = parity_fail + roundtrip_fail + enc_roundtrip_fail;
  std::printf("---\n%zu cases: %d parity fail, %d roundtrip fail, %d enc-roundtrip fail\n", n,
              parity_fail, roundtrip_fail, enc_roundtrip_fail);
  return g_failures == 0 ? 0 : 1;
}
