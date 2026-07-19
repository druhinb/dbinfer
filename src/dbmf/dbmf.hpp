#ifndef DBINFER_DBMF_DBMF_HPP
#define DBINFER_DBMF_DBMF_HPP

#include "gguf/gguf.hpp"

#include <cstdint>
#include <expected>
#include <string_view>

// dbmf: dbinfer model format. a mmap-friendly container for the same weights
// and hyperparameters a gguf carries, restructured for this engine.
//
// on-disk layout (all integers little-endian, matching the host):
//   [header 128B]  magic, version, page alignment, counts, section offsets,
//                  file size, xxhash64 of the header itself
//   [metadata]     typed key/value section, same value encoding as gguf, so it
//                  carries every hyperparam and the whole tokenizer payload
//   [string pool]  tensor name bytes, referenced by table records
//   [tensor table] fixed 128-byte records for O(1) indexed access: name ref,
//                  dtype, dims, data offset, sizes, flags, per-tensor xxhash64
//   [tensor data]  each tensor's bytes at a 16384-byte (Apple Silicon page)
//                  aligned offset so Metal newBufferWithBytesNoCopy can wrap
//                  them zero-copy in a later phase
//
// dims are stored already normalized to row-major, the same convention
// gguf::load exposes on TensorInfo::shape, so the loader path is identical for
// both formats. quant tensor bytes are copied verbatim, keeping the source
// quantization exactly. tensors are laid out in forward-pass first-use
// order so prefill faults pages sequentially. identical tensors (same dtype,
// size, and xxhash64 with byte-equal contents) are stored once and shared.
//
// with --compress, f16 tensors are entropy-coded (see huffman.hpp); the loader
// decodes them once into an anonymous mapping and verifies each against its
// stored hash. quant tensors stay raw and mmap-direct.

namespace dbinfer::dbmf {

// 'DBMF' little-endian.
inline constexpr std::uint32_t kMagic = 0x464D4244u;
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::uint64_t kAlignment = 16384; // Apple Silicon page size
inline constexpr std::uint64_t kHeaderSize = 128;
inline constexpr std::uint64_t kRecordSize = 128;

// header/tensor flag bits.
inline constexpr std::uint32_t kFlagCompressed = 1u; // tensor: entropy-coded f16

struct ConvertOptions {
  bool compress = false; // entropy-code f16 tensors
};

struct ReadOptions {
  bool verify = false; // recompute and check every tensor's xxhash64 at load
};

// writes src to path in the dbmf container. tensor bytes are copied verbatim
// from src's mapping (once, necessarily). fails with an actionable error on any
// io problem.
[[nodiscard]] std::expected<void, gguf::Error>
convert(const gguf::GgufFile &src, std::string_view path, const ConvertOptions &opts = {});

// mmaps and parses a dbmf file into the same in-memory shape gguf::load yields.
// every field is bounds-checked against the file size with overflow guards.
[[nodiscard]] std::expected<gguf::GgufFile, gguf::Error> read(std::string_view path,
                                                              const ReadOptions &opts = {});

// sniffs the leading magic and dispatches to gguf::load or dbmf::read, so the
// engine loads either format transparently.
[[nodiscard]] std::expected<gguf::GgufFile, gguf::Error> load_model(std::string_view path,
                                                                    const ReadOptions &opts = {});

} // namespace dbinfer::dbmf

#endif // DBINFER_DBMF_DBMF_HPP
