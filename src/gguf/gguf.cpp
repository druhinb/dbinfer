#include "gguf/gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>

#include "gguf/reader.hpp"
#include "try.hpp"

namespace dbinfer::gguf {

using detail::Cursor;
using std::expected;
using std::get_if;
using std::string;
using std::string_view;
using std::uint32_t;
using std::uint64_t;
using std::unexpected;
using std::vector;

const char* to_string(GgmlType type) {
  switch (type) {
    case GgmlType::F32:
      return "F32";
    case GgmlType::F16:
      return "F16";
    case GgmlType::Q4_0:
      return "Q4_0";
    case GgmlType::Q4_1:
      return "Q4_1";
    case GgmlType::Q5_0:
      return "Q5_0";
    case GgmlType::Q5_1:
      return "Q5_1";
    case GgmlType::Q8_0:
      return "Q8_0";
    case GgmlType::Q8_1:
      return "Q8_1";
    case GgmlType::Q2_K:
      return "Q2_K";
    case GgmlType::Q3_K:
      return "Q3_K";
    case GgmlType::Q4_K:
      return "Q4_K";
    case GgmlType::Q5_K:
      return "Q5_K";
    case GgmlType::Q6_K:
      return "Q6_K";
    case GgmlType::Q8_K:
      return "Q8_K";
    case GgmlType::I8:
      return "I8";
    case GgmlType::I16:
      return "I16";
    case GgmlType::I32:
      return "I32";
    case GgmlType::I64:
      return "I64";
    case GgmlType::F64:
      return "F64";
    case GgmlType::BF16:
      return "BF16";
  }
  return "UNKNOWN";
}

const char* to_string(MetaType type) {
  switch (type) {
    case MetaType::UInt8:
      return "UINT8";
    case MetaType::Int8:
      return "INT8";
    case MetaType::UInt16:
      return "UINT16";
    case MetaType::Int16:
      return "INT16";
    case MetaType::UInt32:
      return "UINT32";
    case MetaType::Int32:
      return "INT32";
    case MetaType::Float32:
      return "FLOAT32";
    case MetaType::Bool:
      return "BOOL";
    case MetaType::String:
      return "STRING";
    case MetaType::Array:
      return "ARRAY";
    case MetaType::UInt64:
      return "UINT64";
    case MetaType::Int64:
      return "INT64";
    case MetaType::Float64:
      return "FLOAT64";
  }
  return "UNKNOWN";
}

string to_string(const Error& err) {
  return err.file + ":" + std::to_string(err.offset) + ": " + err.message;
}

// block_size = elements per stored block, type_size = bytes per block.
// K-quant sizes (QK_K=256) taken from ggml's block struct layouts.
TypeInfo type_info(GgmlType t) {
  switch (t) {
    case GgmlType::F32:
      return {1, 4, true};
    case GgmlType::F16:
      return {1, 2, true};
    case GgmlType::Q4_0:
      return {32, 18, true};
    case GgmlType::Q4_1:
      return {32, 20, true};
    case GgmlType::Q5_0:
      return {32, 22, true};
    case GgmlType::Q5_1:
      return {32, 24, true};
    case GgmlType::Q8_0:
      return {32, 34, true};
    case GgmlType::Q8_1:
      return {32, 36, true};
    case GgmlType::Q2_K:
      return {256, 84, true};
    case GgmlType::Q3_K:
      return {256, 110, true};
    case GgmlType::Q4_K:
      return {256, 144, true};
    case GgmlType::Q5_K:
      return {256, 176, true};
    case GgmlType::Q6_K:
      return {256, 210, true};
    case GgmlType::Q8_K:
      return {256, 292, true};
    case GgmlType::I8:
      return {1, 1, true};
    case GgmlType::I16:
      return {1, 2, true};
    case GgmlType::I32:
      return {1, 4, true};
    case GgmlType::I64:
      return {1, 8, true};
    case GgmlType::F64:
      return {1, 8, true};
    case GgmlType::BF16:
      return {1, 2, true};
  }
  return {0, 0, false};
}

namespace {

bool ggml_type_known(uint32_t v) { return type_info(static_cast<GgmlType>(v)).known; }

const TensorInfo* find_tensor(const GgufFile& f, string_view name) {
  for (const auto& t : f.tensors) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

const TensorInfo* find_tensor_suffix(const GgufFile& f, string_view suffix) {
  for (const auto& t : f.tensors) {
    if (t.name.ends_with(suffix)) return &t;
  }
  return nullptr;
}

// cross-checks embedding shape against arch metadata so a truncated file
// fails at load.
expected<void, Error> validate_against_metadata(const GgufFile& f, const string& path) {
  const TensorInfo* embd = find_tensor(f, "token_embd.weight");
  if (embd == nullptr) {
    return {};
  }

  const MetaValue* arch_mv = f.find_meta("general.architecture");
  if (arch_mv == nullptr) {
    return unexpected(Error{"token_embd.weight present but general.architecture missing", path, 0});
  }
  const auto* arch = get_if<string>(&arch_mv->value);
  if (arch == nullptr) {
    return unexpected(Error{"general.architecture is not a string", path, 0});
  }

  const MetaValue* emb_mv = f.find_meta(*arch + ".embedding_length");
  if (emb_mv == nullptr) {
    return unexpected(Error{*arch + ".embedding_length missing", path, 0});
  }
  auto emb = detail::as_u64(*emb_mv);
  if (!emb) {
    return unexpected(Error{*arch + ".embedding_length is not an integer", path, 0});
  }

  std::optional<uint64_t> vocab;
  if (const MetaValue* vs = f.find_meta(*arch + ".vocab_size")) {
    vocab = detail::as_u64(*vs);
  }
  if (!vocab) {
    if (const MetaValue* toks = f.find_meta("tokenizer.ggml.tokens")) {
      if (const auto* arr = get_if<MetaArray>(&toks->value)) {
        vocab = arr->values.size();
      }
    }
  }
  if (!vocab) {
    return unexpected(
        Error{"cannot determine vocab size (no " + *arch + ".vocab_size, no tokenizer.ggml.tokens)",
              path, 0});
  }

  if (embd->shape[0] != *vocab || embd->shape[1] != *emb) {
    return unexpected(Error{"token_embd.weight shape mismatch: expected [" +
                                std::to_string(*vocab) + ", " + std::to_string(*emb) +
                                "], found [" + std::to_string(embd->shape[0]) + ", " +
                                std::to_string(embd->shape[1]) + "]",
                            path, embd->offset});
  }

  if (const TensorInfo* attn = find_tensor_suffix(f, "attn_q.weight")) {
    if (attn->shape[1] != *emb) {
      return unexpected(Error{"attention weight " + attn->name + " cols mismatch: expected " +
                                  std::to_string(*emb) + ", found " +
                                  std::to_string(attn->shape[1]),
                              path, attn->offset});
    }
  }

  return {};
}

struct Header {
  uint32_t version;
  uint64_t tensor_count;
  uint64_t kv_count;
};

// reads and validates the fixed 24-byte GGUF header (magic, version, counts).
// only version 3 is supported. older layouts are rejected.
expected<Header, Error> read_header(Cursor& c) {
  auto magic = TRY(c.read_scalar<uint32_t>());
  if (magic != 0x46554747u) {
    return unexpected(Error{"bad magic: expected 'GGUF' (0x46554747), found 0x" +
                                [&] {
                                  char buf[16];
                                  std::snprintf(buf, sizeof(buf), "%08X", magic);
                                  return string(buf);
                                }(),
                            c.path, 0});
  }
  auto version = TRY(c.read_scalar<uint32_t>());
  if (version != 3) {
    return unexpected(Error{
        "unsupported GGUF version: found " + std::to_string(version) + ", expected 3", c.path, 4});
  }
  auto tensor_count = TRY(c.read_scalar<uint64_t>());
  auto kv_count = TRY(c.read_scalar<uint64_t>());
  return Header{version, tensor_count, kv_count};
}

// reads the kv_count metadata entries following the header, preserving file
// order for GgufFile::find_meta's linear scan.
expected<vector<std::pair<string, MetaValue>>, Error> read_metadata(Cursor& c, uint64_t kv_count) {
  vector<std::pair<string, MetaValue>> metadata;
  metadata.reserve(static_cast<std::size_t>(std::min<uint64_t>(kv_count, 1024)));
  for (uint64_t i = 0; i < kv_count; ++i) {
    auto key = TRY(c.read_string());
    uint64_t type_at = c.pos;
    auto type_raw = TRY(c.read_scalar<uint32_t>());
    if (!detail::meta_type_in_range(type_raw)) {
      return unexpected(Error{"metadata value type out of range for key '" + key + "': found " +
                                  std::to_string(type_raw) + ", expected 0..12",
                              c.path, type_at});
    }
    auto value = TRY(detail::read_meta_value(c, static_cast<MetaType>(type_raw)));
    metadata.emplace_back(std::move(key), std::move(value));
  }
  return metadata;
}

// gguf defaults tensor alignment to 32. general.alignment overrides it.
// llama.cpp requires a power of two.
expected<uint64_t, Error> resolve_alignment(const GgufFile& f, const string& path) {
  const MetaValue* am = f.find_meta("general.alignment");
  if (am == nullptr) {
    return 32;
  }
  const auto* a = get_if<uint32_t>(&am->value);
  if (a == nullptr) {
    return unexpected(Error{"general.alignment must be UINT32", path, 0});
  }
  if (*a == 0 || (*a & (*a - 1)) != 0) {
    return unexpected(
        Error{"general.alignment must be a power of two, found " + std::to_string(*a), path, 0});
  }
  return *a;
}

// reads the tensor_info table: for each tensor, its name, dims (stored
// column-major per GGML convention and flipped into row-major here), ggml
// type, and byte offset relative to the tensor data section. Also computes
// each tensor's byte size from its element count and per-type block layout,
// which bind_tensor_data later uses to bounds-check against the file.
expected<vector<TensorInfo>, Error> read_tensor_infos(Cursor& c, uint64_t tensor_count) {
  vector<TensorInfo> tensors;
  tensors.reserve(static_cast<std::size_t>(std::min<uint64_t>(tensor_count, 65536)));
  for (uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo info;
    info.name = TRY(c.read_string());

    uint64_t ndims_at = c.pos;
    auto n_dims = TRY(c.read_scalar<uint32_t>());
    if (n_dims == 0 || n_dims > 4) {
      return unexpected(Error{"tensor '" + info.name + "' n_dims out of range: found " +
                                  std::to_string(n_dims) + ", expected 1..4",
                              c.path, ndims_at});
    }
    info.n_dims = n_dims;

    std::array<uint64_t, 4> file_dims{1, 1, 1, 1};
    for (uint32_t d = 0; d < n_dims; ++d) {
      file_dims[d] = TRY(c.read_scalar<uint64_t>());
    }

    uint64_t type_at = c.pos;
    auto type_raw = TRY(c.read_scalar<uint32_t>());
    if (!ggml_type_known(type_raw)) {
      return unexpected(
          Error{"tensor '" + info.name + "' has unknown ggml type " + std::to_string(type_raw),
                c.path, type_at});
    }
    info.type = static_cast<GgmlType>(type_raw);

    info.offset = TRY(c.read_scalar<uint64_t>());

    // normalize to row major from column major
    info.shape = {1, 1, 1, 1};
    for (uint32_t d = 0; d < n_dims; ++d) {
      info.shape[d] = file_dims[n_dims - 1 - d];
    }

    uint64_t nelem = 1;
    for (uint64_t dim : info.shape) {
      if (__builtin_mul_overflow(nelem, dim, &nelem)) {
        return unexpected(
            Error{"tensor '" + info.name + "' element count overflows uint64", c.path, type_at});
      }
    }
    TypeInfo tr = type_info(info.type);
    if (nelem % tr.block_size != 0) {
      return unexpected(Error{"tensor '" + info.name + "' element count " + std::to_string(nelem) +
                                  " not a multiple of block size " + std::to_string(tr.block_size) +
                                  " for type " + to_string(info.type),
                              c.path, type_at});
    }
    if (__builtin_mul_overflow(nelem / tr.block_size, tr.type_size, &info.nbytes)) {
      return unexpected(
          Error{"tensor '" + info.name + "' byte size overflows uint64", c.path, type_at});
    }

    tensors.push_back(std::move(info));
  }
  return tensors;
}

// points each TensorInfo::data into the mmap, after checking the tensor's
// offset respects the file's alignment and that [offset, offset+nbytes) is
// fully inside the file. This is the last chance to catch a truncated or
// malformed file before callers start dereferencing tensor data directly.
expected<void, Error> bind_tensor_data(vector<TensorInfo>& tensors, const std::byte* base,
                                       uint64_t file_size, uint64_t data_section,
                                       uint64_t alignment, const string& path) {
  for (auto& info : tensors) {
    if (info.offset % alignment != 0) {
      return unexpected(Error{"tensor '" + info.name + "' offset " + std::to_string(info.offset) +
                                  " not aligned to " + std::to_string(alignment),
                              path, info.offset});
    }
    uint64_t data_off;
    uint64_t end;
    if (__builtin_add_overflow(data_section, info.offset, &data_off) ||
        __builtin_add_overflow(data_off, info.nbytes, &end) || end > file_size) {
      return unexpected(Error{"tensor '" + info.name + "' data at offset " +
                                  std::to_string(data_off) + " + " + std::to_string(info.nbytes) +
                                  " bytes exceeds file size " + std::to_string(file_size),
                              path, data_off});
    }
    info.data = base + data_off;
  }
  return {};
}

}  // namespace

const MetaValue* GgufFile::find_meta(string_view key) const {
  for (const auto& kv : metadata) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

expected<GgufFile, Error> load(string_view path) {
  string path_str(path);
  GgufFile file;
  file.mapping = TRY(MappedFile::open(path));

  Cursor c{file.mapping.data(), file.mapping.size(), path_str, 0};

  Header header = TRY(read_header(c));
  file.version = header.version;
  file.metadata = TRY(read_metadata(c, header.kv_count));
  file.alignment = TRY(resolve_alignment(file, path_str));

  auto tensors = TRY(read_tensor_infos(c, header.tensor_count));

  uint64_t data_section = detail::align_up(c.pos, file.alignment);
  if (data_section > file.mapping.size()) {
    return unexpected(Error{"tensor data section start " + std::to_string(data_section) +
                                " exceeds file size " + std::to_string(file.mapping.size()),
                            path_str, c.pos});
  }

  TRY(bind_tensor_data(tensors, file.mapping.data(), file.mapping.size(), data_section,
                       file.alignment, path_str));
  file.tensors = std::move(tensors);

  TRY(validate_against_metadata(file, path_str));
  return file;
}

}  // namespace dbinfer::gguf
