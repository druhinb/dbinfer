#ifndef DBINFER_GGUF_READER_HPP
#define DBINFER_GGUF_READER_HPP

#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <optional>
#include <string>

#include "gguf/gguf.hpp"
#include "try.hpp"

// shared parsing internals for the GGUF byte format, used by both the loader
// (gguf.cpp) and the dbmf converter, which embeds a GGUF metadata block.
namespace dbinfer::gguf::detail {

inline std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

inline bool meta_type_in_range(std::uint32_t v) {
  return v <= static_cast<std::uint32_t>(MetaType::Float64);
}

// bounds-checked forward-only reader over mmap'd file bytes; every read
// either advances pos and returns a value or leaves pos untouched and
// returns an Error, so callers can TRY() through the format without
// separately tracking how much of the buffer is left.
struct Cursor {
  const std::byte* base;
  std::uint64_t size;
  const std::string& path;
  std::uint64_t pos = 0;

  Error eof(std::uint64_t need) const {
    return Error{"unexpected EOF: need " + std::to_string(need) + " bytes at offset " +
                     std::to_string(pos) + ", file is " + std::to_string(size) + " bytes",
                 path, pos};
  }

  // mmap bytes are unaligned for T. a typed load would be UB.
  template <typename T>
  std::expected<T, Error> read_scalar() {
    if (pos + sizeof(T) > size) {
      return std::unexpected(eof(sizeof(T)));
    }
    T value;
    std::memcpy(&value, base + pos, sizeof(T));
    pos += sizeof(T);
    return value;
  }

  std::expected<std::string, Error> read_string() {
    auto len = TRY(read_scalar<std::uint64_t>());
    if (len > size - pos) {
      return std::unexpected(eof(len));
    }
    std::string s(reinterpret_cast<const char*>(base + pos), static_cast<std::size_t>(len));
    pos += len;
    return s;
  }
};

template <typename T>
std::expected<MetaValue, Error> read_meta_scalar(Cursor& c) {
  return MetaValue{TRY(c.read_scalar<T>())};
}

// reads one metadata value of the given type, recursing into read_meta_value
// for array elements. GGUF arrays are flat (scalars only), so the recursive
// call can never see MetaType::Array again; that case is rejected below.
inline std::expected<MetaValue, Error> read_meta_value(Cursor& c, MetaType type) {
  switch (type) {
    case MetaType::UInt8:
      return read_meta_scalar<std::uint8_t>(c);
    case MetaType::Int8:
      return read_meta_scalar<std::int8_t>(c);
    case MetaType::UInt16:
      return read_meta_scalar<std::uint16_t>(c);
    case MetaType::Int16:
      return read_meta_scalar<std::int16_t>(c);
    case MetaType::UInt32:
      return read_meta_scalar<std::uint32_t>(c);
    case MetaType::Int32:
      return read_meta_scalar<std::int32_t>(c);
    case MetaType::Float32:
      return read_meta_scalar<float>(c);
    case MetaType::UInt64:
      return read_meta_scalar<std::uint64_t>(c);
    case MetaType::Int64:
      return read_meta_scalar<std::int64_t>(c);
    case MetaType::Float64:
      return read_meta_scalar<double>(c);
    case MetaType::Bool:
      return MetaValue{TRY(c.read_scalar<std::uint8_t>()) != 0};
    case MetaType::String:
      return MetaValue{TRY(c.read_string())};
    case MetaType::Array: {
      std::uint64_t elem_start = c.pos;
      auto elem_raw = TRY(c.read_scalar<std::uint32_t>());
      if (!meta_type_in_range(elem_raw)) {
        return std::unexpected(Error{"array element type out of range: found " +
                                         std::to_string(elem_raw) + ", expected 0..12",
                                     c.path, elem_start});
      }
      auto elem_type = static_cast<MetaType>(elem_raw);

      // GGUF arrays hold scalars only; an array-of-arrays is malformed
      if (elem_type == MetaType::Array) {
        return std::unexpected(
            Error{"nested metadata arrays are not permitted by GGUF", c.path, elem_start});
      }
      auto count = TRY(c.read_scalar<std::uint64_t>());
      if (count > c.size - c.pos) {
        return std::unexpected(Error{
            "array count " + std::to_string(count) + " exceeds remaining bytes", c.path, c.pos});
      }
      MetaArray arr;
      arr.elem_type = elem_type;
      arr.values.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        arr.values.push_back(TRY(read_meta_value(c, elem_type)));
      }
      return MetaValue{std::move(arr)};
    }
  }
  return std::unexpected(Error{"unknown metadata value type", c.path, c.pos});
}

// coerces any integer metadata value to uint64_t; nullopt if not an integer
// type or the stored value is negative.
inline std::optional<std::uint64_t> as_u64(const MetaValue& mv) {
  if (auto p = std::get_if<std::uint8_t>(&mv.value)) return *p;
  if (auto p = std::get_if<std::uint16_t>(&mv.value)) return *p;
  if (auto p = std::get_if<std::uint32_t>(&mv.value)) return *p;
  if (auto p = std::get_if<std::uint64_t>(&mv.value)) return *p;
  if (auto p = std::get_if<std::int32_t>(&mv.value))
    return *p >= 0 ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(*p)) : std::nullopt;
  if (auto p = std::get_if<std::int64_t>(&mv.value))
    return *p >= 0 ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(*p)) : std::nullopt;
  return std::nullopt;
}

// coerces a Float32 or Float64 metadata value to float; nullopt otherwise.
inline std::optional<float> as_f32(const MetaValue& mv) {
  if (auto p = std::get_if<float>(&mv.value)) return *p;
  if (auto p = std::get_if<double>(&mv.value)) return static_cast<float>(*p);
  return std::nullopt;
}

// coerces any integer metadata value to int64_t; nullopt if not an integer
// type or a uint64 value too large to represent as signed.
inline std::optional<std::int64_t> as_i64(const MetaValue& mv) {
  if (auto p = std::get_if<std::uint8_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::int8_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::uint16_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::int16_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::uint32_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::int32_t>(&mv.value)) return static_cast<std::int64_t>(*p);
  if (auto p = std::get_if<std::int64_t>(&mv.value)) return *p;
  if (auto p = std::get_if<std::uint64_t>(&mv.value)) {
    return *p <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
               ? std::optional<std::int64_t>(static_cast<std::int64_t>(*p))
               : std::nullopt;
  }
  return std::nullopt;
}

}  // namespace dbinfer::gguf::detail

#endif
