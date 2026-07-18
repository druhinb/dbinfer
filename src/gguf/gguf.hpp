#ifndef DBINFER_GGUF_GGUF_HPP
#define DBINFER_GGUF_GGUF_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace dbinfer::gguf {

enum class GgmlType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
};

const char *to_string(GgmlType type);

enum class MetaType : std::uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

const char *to_string(MetaType type);

struct Error {
  std::string message;
  std::string file;
  std::uint64_t offset = 0;
};

// "file:offset: message"
std::string to_string(const Error &err);

struct MetaValue;

struct MetaArray {
  MetaType elem_type{};
  std::vector<MetaValue> values;
};

struct MetaValue {
  std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t, std::uint32_t, std::int32_t,
               float, bool, std::string, MetaArray, std::uint64_t, std::int64_t, double>
      value;
};

struct TensorInfo {
  std::string name;
  GgmlType type{};
  std::array<std::uint64_t, 4> shape{1, 1, 1, 1};
  std::uint32_t n_dims = 0;
  std::uint64_t offset = 0; // relative to the tensor data section
  const std::byte *data = nullptr;
  std::uint64_t nbytes = 0;
};

} // namespace dbinfer::gguf

#include "gguf/mmap_file.hpp"

namespace dbinfer::gguf {

struct GgufFile {
  std::uint32_t version = 0;
  std::uint64_t alignment = 0;
  std::vector<std::pair<std::string, MetaValue>> metadata; // insertion order
  std::vector<TensorInfo> tensors;
  MappedFile mapping;

  // linear scan; metadata is small (dozens of keys) so a map is not worth it
  const MetaValue *find_meta(std::string_view key) const;
};

// mmaps path and parses the GGUF header, metadata, and tensor table. Tensor
// shapes are normalized from GGUF's column-major storage to row-major (see
// TensorInfo::shape), and each TensorInfo::data points directly into the
// mapping, so file and GgufFile must outlive any use of tensor data.
std::expected<GgufFile, Error> load(std::string_view path);

} // namespace dbinfer::gguf

#endif
