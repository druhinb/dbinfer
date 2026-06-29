#ifndef DBINFER_DBMF_HUFFMAN_HPP
#define DBINFER_DBMF_HUFFMAN_HPP

#include "gguf/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace dbinfer::dbmf {

// f16 entropy coder. each f16 element splits into a low byte (mantissa, high
// entropy) and a high byte (sign, exponent, top mantissa bits, low entropy).
// the high-byte plane is canonical-huffman coded, the low-byte plane stored
// raw. quant weights are already high-entropy and are never passed here.

struct CompressResult {
  std::vector<std::byte> bytes;
  bool compressed = false; // false when coding did not shrink the input
};

// raw is the verbatim f16 bytes (even length). returns compressed=false when
// the coded blob would not be smaller, so the caller stores raw instead.
CompressResult compress_f16(std::span<const std::byte> raw);

// decodes a blob produced by compress_f16 into out (out.size() must equal the
// original raw length). fully bounds-checked against a truncated or malformed
// stream; a bad blob returns an Error and never reads out of bounds.
std::expected<void, gguf::Error> decompress_f16(std::span<const std::byte> blob,
                                                std::span<std::byte> out);

} // namespace dbinfer::dbmf

#endif // DBINFER_DBMF_HUFFMAN_HPP
