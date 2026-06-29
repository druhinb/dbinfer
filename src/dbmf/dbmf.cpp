#include "dbmf/dbmf.hpp"

#include "dbmf/huffman.hpp"
#include "dbmf/xxhash.hpp"
#include "try.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace dbinfer::dbmf {

using gguf::Error;
using gguf::GgmlType;
using gguf::GgufFile;
using gguf::MappedFile;
using gguf::MetaArray;
using gguf::MetaType;
using gguf::MetaValue;
using gguf::TensorInfo;

namespace {

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

// ---- writing -------------------------------------------------------------

struct ByteBuf {
  std::vector<std::byte> data;

  template <typename T> void put(T v) {
    std::byte tmp[sizeof(T)];
    std::memcpy(tmp, &v, sizeof(T));
    data.insert(data.end(), tmp, tmp + sizeof(T));
  }
  void put_bytes(const std::byte *p, std::size_t n) { data.insert(data.end(), p, p + n); }
  void put_string(std::string_view s) {
    put<std::uint64_t>(s.size());
    put_bytes(reinterpret_cast<const std::byte *>(s.data()), s.size());
  }
  void pad_to(std::size_t n) { data.resize(std::max(data.size(), n), std::byte{0}); }
};

MetaType meta_type_of(const MetaValue &mv) {
  // variant alternatives are declared in MetaType numeric order.
  return static_cast<MetaType>(mv.value.index());
}

void put_meta_value(ByteBuf &b, const MetaValue &mv) {
  std::visit(
      [&](const auto &v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
          b.put_string(v);
        } else if constexpr (std::is_same_v<T, bool>) {
          b.put<std::uint8_t>(v ? 1 : 0);
        } else if constexpr (std::is_same_v<T, MetaArray>) {
          b.put<std::uint32_t>(static_cast<std::uint32_t>(v.elem_type));
          b.put<std::uint64_t>(v.values.size());
          for (const auto &e : v.values)
            put_meta_value(b, e);
        } else {
          b.put<T>(v);
        }
      },
      mv.value);
}

// forward-pass first-use order: token_embd, then blk.N in ascending layer, then
// everything else (norms, biases), then the lm head. a stable sort keeps the
// original order within a group.
std::uint64_t order_key(const std::string &name) {
  if (name == "token_embd.weight")
    return 0;
  if (name.starts_with("blk.")) {
    std::uint64_t layer = 0;
    std::size_t i = 4;
    for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i)
      layer = layer * 10 + static_cast<std::uint64_t>(name[i] - '0');
    return (1ULL << 40) + (layer << 8);
  }
  if (name == "output.weight")
    return 3ULL << 40;
  return 2ULL << 40;
}

std::vector<std::size_t> usage_order(const std::vector<TensorInfo> &tensors) {
  std::vector<std::size_t> order(tensors.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return order_key(tensors[a].name) < order_key(tensors[b].name);
  });
  return order;
}

struct Prepared {
  std::uint64_t logical = 0;
  std::uint64_t stored = 0;
  std::uint64_t xxh = 0;
  bool compressed = false;
  std::vector<std::byte> blob; // owns coded bytes when compressed
  std::uint64_t data_offset = 0;
  bool owner = false;
};

struct FileCloser {
  void operator()(std::FILE *f) const {
    if (f != nullptr)
      std::fclose(f);
  }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

struct Writer {
  std::FILE *f;
  std::string path;
  std::uint64_t pos = 0;

  std::expected<void, Error> write(const std::byte *p, std::size_t n) {
    if (n > 0 && std::fwrite(p, 1, n, f) != n)
      return std::unexpected(Error{"short write", path, pos});
    pos += n;
    return {};
  }
  std::expected<void, Error> pad_to(std::uint64_t target) {
    static const std::byte zeros[4096] = {};
    while (pos < target) {
      const std::uint64_t n = std::min<std::uint64_t>(target - pos, sizeof(zeros));
      TRY(write(zeros, static_cast<std::size_t>(n)));
    }
    return {};
  }
};

std::expected<void, Error> convert_impl(const GgufFile &src, const std::string &path,
                                        const ConvertOptions &opts) {
  const std::size_t nt = src.tensors.size();
  const std::vector<std::size_t> order = usage_order(src.tensors);

  std::vector<Prepared> prep(nt);
  for (std::size_t i = 0; i < nt; ++i) {
    const TensorInfo &t = src.tensors[i];
    if (t.data == nullptr)
      return std::unexpected(Error{"tensor '" + t.name + "' has no bound data", path, 0});
    prep[i].logical = t.nbytes;
    prep[i].stored = t.nbytes;
    prep[i].xxh = xxhash64(t.data, static_cast<std::size_t>(t.nbytes));
    if (opts.compress && t.type == GgmlType::F16 && t.nbytes > 0) {
      CompressResult r = compress_f16({t.data, static_cast<std::size_t>(t.nbytes)});
      if (r.compressed) {
        prep[i].blob = std::move(r.bytes);
        prep[i].stored = prep[i].blob.size();
        prep[i].compressed = true;
      }
    }
  }

  ByteBuf meta;
  for (const auto &kv : src.metadata) {
    meta.put_string(kv.first);
    meta.put<std::uint32_t>(static_cast<std::uint32_t>(meta_type_of(kv.second)));
    put_meta_value(meta, kv.second);
  }

  ByteBuf pool;
  std::vector<std::uint64_t> name_off(nt);
  for (std::size_t j = 0; j < order.size(); ++j) {
    const std::size_t i = order[j];
    name_off[i] = pool.data.size();
    pool.put_bytes(reinterpret_cast<const std::byte *>(src.tensors[i].name.data()),
                   src.tensors[i].name.size());
  }

  const std::uint64_t metadata_offset = kHeaderSize;
  const std::uint64_t metadata_size = meta.data.size();
  const std::uint64_t string_pool_offset = metadata_offset + metadata_size;
  const std::uint64_t string_pool_size = pool.data.size();
  const std::uint64_t tensor_table_offset = string_pool_offset + string_pool_size;
  const std::uint64_t tensor_table_size = kRecordSize * nt;
  const std::uint64_t data_offset = align_up(tensor_table_offset + tensor_table_size, kAlignment);

  // assign data offsets in write order, deduplicating byte-equal tensors.
  using DedupKey = std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>;
  std::map<DedupKey, std::size_t> owners;
  std::uint64_t running = data_offset;
  std::uint64_t file_size = data_offset;
  bool any_compressed = false;
  for (const std::size_t i : order) {
    const TensorInfo &t = src.tensors[i];
    const DedupKey key{prep[i].xxh, prep[i].logical, static_cast<std::uint32_t>(t.type)};
    bool shared = false;
    if (auto it = owners.find(key); it != owners.end()) {
      const std::size_t o = it->second;
      if (prep[i].compressed == prep[o].compressed &&
          std::memcmp(t.data, src.tensors[o].data, static_cast<std::size_t>(prep[i].logical)) ==
              0) {
        prep[i].data_offset = prep[o].data_offset;
        shared = true;
      }
    }
    if (!shared) {
      const std::uint64_t off = align_up(running, kAlignment);
      prep[i].data_offset = off;
      prep[i].owner = true;
      running = off + prep[i].stored;
      file_size = running;
      owners.emplace(key, i);
    }
    any_compressed = any_compressed || prep[i].compressed;
  }

  ByteBuf table;
  for (std::size_t j = 0; j < order.size(); ++j) {
    const std::size_t i = order[j];
    const TensorInfo &t = src.tensors[i];
    const std::size_t start = table.data.size();
    table.put<std::uint64_t>(name_off[i]);
    table.put<std::uint32_t>(static_cast<std::uint32_t>(t.name.size()));
    table.put<std::uint32_t>(static_cast<std::uint32_t>(t.type));
    table.put<std::uint32_t>(t.n_dims);
    table.put<std::uint32_t>(prep[i].compressed ? kFlagCompressed : 0u);
    for (int d = 0; d < 4; ++d)
      table.put<std::uint64_t>(t.shape[static_cast<std::size_t>(d)]);
    table.put<std::uint64_t>(prep[i].data_offset);
    table.put<std::uint64_t>(prep[i].logical);
    table.put<std::uint64_t>(prep[i].stored);
    table.put<std::uint64_t>(prep[i].xxh);
    table.pad_to(start + kRecordSize);
  }

  ByteBuf header;
  header.put<std::uint32_t>(kMagic);
  header.put<std::uint32_t>(kVersion);
  header.put<std::uint32_t>(static_cast<std::uint32_t>(kAlignment));
  header.put<std::uint32_t>(any_compressed ? kFlagCompressed : 0u);
  header.put<std::uint64_t>(nt);
  header.put<std::uint64_t>(src.metadata.size());
  header.put<std::uint64_t>(metadata_offset);
  header.put<std::uint64_t>(metadata_size);
  header.put<std::uint64_t>(string_pool_offset);
  header.put<std::uint64_t>(string_pool_size);
  header.put<std::uint64_t>(tensor_table_offset);
  header.put<std::uint64_t>(tensor_table_size);
  header.put<std::uint64_t>(data_offset);
  header.put<std::uint64_t>(file_size);
  const std::uint64_t header_checksum = xxhash64(header.data.data(), header.data.size());
  header.put<std::uint64_t>(header_checksum);
  header.pad_to(kHeaderSize);

  FilePtr fp(std::fopen(path.c_str(), "wb"));
  if (!fp)
    return std::unexpected(
        Error{std::string("cannot open for write: ") + std::strerror(errno), path, 0});
  Writer w{fp.get(), path};
  TRY(w.write(header.data.data(), header.data.size()));
  TRY(w.write(meta.data.data(), meta.data.size()));
  TRY(w.write(pool.data.data(), pool.data.size()));
  TRY(w.write(table.data.data(), table.data.size()));
  TRY(w.pad_to(data_offset));
  for (const std::size_t i : order) {
    if (!prep[i].owner)
      continue;
    TRY(w.pad_to(prep[i].data_offset));
    if (prep[i].compressed)
      TRY(w.write(prep[i].blob.data(), prep[i].blob.size()));
    else
      TRY(w.write(src.tensors[i].data, static_cast<std::size_t>(prep[i].logical)));
  }
  if (std::fflush(fp.get()) != 0)
    return std::unexpected(Error{"flush failed", path, w.pos});
  return {};
}

// ---- reading -------------------------------------------------------------

struct Cursor {
  const std::byte *base;
  std::uint64_t size;
  const std::string &path;
  std::uint64_t pos = 0;

  Error eof(std::uint64_t need) const {
    return Error{"unexpected EOF: need " + std::to_string(need) + " bytes at offset " +
                     std::to_string(pos) + ", file is " + std::to_string(size) + " bytes",
                 path, pos};
  }
  template <typename T> std::expected<T, Error> read_scalar() {
    if (pos + sizeof(T) > size)
      return std::unexpected(eof(sizeof(T)));
    T value;
    std::memcpy(&value, base + pos, sizeof(T));
    pos += sizeof(T);
    return value;
  }
  std::expected<std::string, Error> read_string() {
    auto len = TRY(read_scalar<std::uint64_t>());
    if (len > size - pos)
      return std::unexpected(eof(len));
    std::string s(reinterpret_cast<const char *>(base + pos), static_cast<std::size_t>(len));
    pos += len;
    return s;
  }
};

bool meta_type_in_range(std::uint32_t v) {
  return v <= static_cast<std::uint32_t>(MetaType::Float64);
}

template <typename T> std::expected<MetaValue, Error> read_scalar_meta(Cursor &c) {
  return MetaValue{TRY(c.read_scalar<T>())};
}

std::expected<MetaValue, Error> read_meta_value(Cursor &c, MetaType type) {
  switch (type) {
  case MetaType::UInt8:
    return read_scalar_meta<std::uint8_t>(c);
  case MetaType::Int8:
    return read_scalar_meta<std::int8_t>(c);
  case MetaType::UInt16:
    return read_scalar_meta<std::uint16_t>(c);
  case MetaType::Int16:
    return read_scalar_meta<std::int16_t>(c);
  case MetaType::UInt32:
    return read_scalar_meta<std::uint32_t>(c);
  case MetaType::Int32:
    return read_scalar_meta<std::int32_t>(c);
  case MetaType::Float32:
    return read_scalar_meta<float>(c);
  case MetaType::UInt64:
    return read_scalar_meta<std::uint64_t>(c);
  case MetaType::Int64:
    return read_scalar_meta<std::int64_t>(c);
  case MetaType::Float64:
    return read_scalar_meta<double>(c);
  case MetaType::Bool:
    return MetaValue{TRY(c.read_scalar<std::uint8_t>()) != 0};
  case MetaType::String:
    return MetaValue{TRY(c.read_string())};
  case MetaType::Array: {
    const std::uint64_t at = c.pos;
    auto elem_raw = TRY(c.read_scalar<std::uint32_t>());
    if (!meta_type_in_range(elem_raw) || elem_raw == static_cast<std::uint32_t>(MetaType::Array))
      return std::unexpected(
          Error{"bad array element type " + std::to_string(elem_raw), c.path, at});
    const auto elem_type = static_cast<MetaType>(elem_raw);
    auto count = TRY(c.read_scalar<std::uint64_t>());
    if (count > c.size - c.pos)
      return std::unexpected(Error{"array count exceeds remaining bytes", c.path, c.pos});
    MetaArray arr;
    arr.elem_type = elem_type;
    arr.values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i)
      arr.values.push_back(TRY(read_meta_value(c, elem_type)));
    return MetaValue{std::move(arr)};
  }
  }
  return std::unexpected(Error{"unknown metadata type", c.path, c.pos});
}

struct Header {
  std::uint32_t alignment;
  std::uint32_t flags;
  std::uint64_t tensor_count;
  std::uint64_t metadata_count;
  std::uint64_t metadata_offset;
  std::uint64_t metadata_size;
  std::uint64_t string_pool_offset;
  std::uint64_t string_pool_size;
  std::uint64_t tensor_table_offset;
  std::uint64_t tensor_table_size;
  std::uint64_t data_offset;
  std::uint64_t file_size;
};

std::expected<Header, Error> read_header(const std::byte *base, std::uint64_t size,
                                         const std::string &path) {
  if (size < kHeaderSize)
    return std::unexpected(Error{"file smaller than dbmf header", path, 0});
  Cursor c{base, size, path, 0};
  const auto magic = TRY(c.read_scalar<std::uint32_t>());
  if (magic != kMagic) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%08X", magic);
    return std::unexpected(
        Error{std::string("bad magic: expected 'DBMF', found 0x") + buf, path, 0});
  }
  const auto version = TRY(c.read_scalar<std::uint32_t>());
  if (version != kVersion)
    return std::unexpected(Error{"unsupported dbmf version " + std::to_string(version), path, 4});
  Header h{};
  h.alignment = TRY(c.read_scalar<std::uint32_t>());
  h.flags = TRY(c.read_scalar<std::uint32_t>());
  h.tensor_count = TRY(c.read_scalar<std::uint64_t>());
  h.metadata_count = TRY(c.read_scalar<std::uint64_t>());
  h.metadata_offset = TRY(c.read_scalar<std::uint64_t>());
  h.metadata_size = TRY(c.read_scalar<std::uint64_t>());
  h.string_pool_offset = TRY(c.read_scalar<std::uint64_t>());
  h.string_pool_size = TRY(c.read_scalar<std::uint64_t>());
  h.tensor_table_offset = TRY(c.read_scalar<std::uint64_t>());
  h.tensor_table_size = TRY(c.read_scalar<std::uint64_t>());
  h.data_offset = TRY(c.read_scalar<std::uint64_t>());
  h.file_size = TRY(c.read_scalar<std::uint64_t>());
  const auto stored_checksum = TRY(c.read_scalar<std::uint64_t>());

  if (h.alignment == 0 || (h.alignment & (h.alignment - 1)) != 0)
    return std::unexpected(Error{"alignment not a power of two", path, 8});
  const std::uint64_t computed = xxhash64(base, 96);
  if (computed != stored_checksum)
    return std::unexpected(Error{"header checksum mismatch", path, 96});

  auto region_ok = [&](std::uint64_t off, std::uint64_t len) {
    std::uint64_t end = 0;
    return !__builtin_add_overflow(off, len, &end) && end <= size;
  };
  if (h.file_size > size)
    return std::unexpected(Error{"header file_size exceeds mapping", path, 88});
  if (!region_ok(h.metadata_offset, h.metadata_size) ||
      !region_ok(h.string_pool_offset, h.string_pool_size) ||
      !region_ok(h.tensor_table_offset, h.tensor_table_size) || h.data_offset > size)
    return std::unexpected(Error{"a header section runs past the file", path, 0});
  std::uint64_t table_bytes = 0;
  if (__builtin_mul_overflow(h.tensor_count, kRecordSize, &table_bytes) ||
      table_bytes > h.tensor_table_size)
    return std::unexpected(Error{"tensor table smaller than declared count", path, 64});
  return h;
}

struct Record {
  std::string name;
  GgmlType type;
  std::uint32_t n_dims;
  std::uint32_t flags;
  std::array<std::uint64_t, 4> dims;
  std::uint64_t data_offset;
  std::uint64_t logical;
  std::uint64_t stored;
  std::uint64_t xxh;
};

std::expected<Record, Error> read_record(const std::byte *base, std::uint64_t size, const Header &h,
                                         std::uint64_t idx, const std::string &path) {
  Cursor c{base, size, path, h.tensor_table_offset + idx * kRecordSize};
  const auto name_off = TRY(c.read_scalar<std::uint64_t>());
  const auto name_len = TRY(c.read_scalar<std::uint32_t>());
  Record r{};
  r.type = static_cast<GgmlType>(TRY(c.read_scalar<std::uint32_t>()));
  r.n_dims = TRY(c.read_scalar<std::uint32_t>());
  r.flags = TRY(c.read_scalar<std::uint32_t>());
  for (int d = 0; d < 4; ++d)
    r.dims[static_cast<std::size_t>(d)] = TRY(c.read_scalar<std::uint64_t>());
  r.data_offset = TRY(c.read_scalar<std::uint64_t>());
  r.logical = TRY(c.read_scalar<std::uint64_t>());
  r.stored = TRY(c.read_scalar<std::uint64_t>());
  r.xxh = TRY(c.read_scalar<std::uint64_t>());

  std::uint64_t name_end = 0;
  if (__builtin_add_overflow(name_off, name_len, &name_end) || name_end > h.string_pool_size)
    return std::unexpected(Error{"tensor name out of string pool", path, h.tensor_table_offset});
  r.name.assign(reinterpret_cast<const char *>(base + h.string_pool_offset + name_off), name_len);

  const gguf::TypeInfo ti = gguf::type_info(r.type);
  if (!ti.known)
    return std::unexpected(Error{"tensor '" + r.name + "' unknown dtype", path, r.data_offset});
  if (r.n_dims == 0 || r.n_dims > 4)
    return std::unexpected(Error{"tensor '" + r.name + "' bad n_dims", path, r.data_offset});
  std::uint64_t nelem = 1;
  for (std::uint32_t d = 0; d < r.n_dims; ++d)
    if (__builtin_mul_overflow(nelem, r.dims[d], &nelem))
      return std::unexpected(Error{"tensor '" + r.name + "' element count overflow", path, 0});
  if (nelem % ti.block_size != 0)
    return std::unexpected(
        Error{"tensor '" + r.name + "' element count not block aligned", path, 0});
  std::uint64_t expect = 0;
  if (__builtin_mul_overflow(nelem / ti.block_size, ti.type_size, &expect))
    return std::unexpected(Error{"tensor '" + r.name + "' byte size overflow", path, 0});
  if (r.logical != expect)
    return std::unexpected(Error{"tensor '" + r.name + "' logical size " +
                                     std::to_string(r.logical) + " != computed " +
                                     std::to_string(expect),
                                 path, r.data_offset});

  if (r.data_offset % h.alignment != 0)
    return std::unexpected(
        Error{"tensor '" + r.name + "' data offset not page aligned", path, r.data_offset});
  std::uint64_t end = 0;
  if (__builtin_add_overflow(r.data_offset, r.stored, &end) || end > size)
    return std::unexpected(
        Error{"tensor '" + r.name + "' data runs past the file", path, r.data_offset});
  if ((r.flags & kFlagCompressed) != 0) {
    if (r.type != GgmlType::F16)
      return std::unexpected(Error{"tensor '" + r.name + "' compressed but not f16", path, 0});
  } else if (r.stored != r.logical) {
    return std::unexpected(Error{"tensor '" + r.name + "' raw stored size mismatch", path, 0});
  }
  return r;
}

std::expected<GgufFile, Error> read_impl(std::string_view path_view, const ReadOptions &opts) {
  const std::string path(path_view);
  GgufFile file;
  file.mapping = TRY(MappedFile::open(path_view));
  const std::byte *base = file.mapping.data();
  const std::uint64_t size = file.mapping.size();

  const Header h = TRY(read_header(base, size, path));
  file.version = kVersion;
  file.alignment = h.alignment;

  Cursor mc{base, h.metadata_offset + h.metadata_size, path, h.metadata_offset};
  file.metadata.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(h.metadata_count, 1024)));
  for (std::uint64_t i = 0; i < h.metadata_count; ++i) {
    auto key = TRY(mc.read_string());
    const auto type_raw = TRY(mc.read_scalar<std::uint32_t>());
    if (!meta_type_in_range(type_raw))
      return std::unexpected(Error{"metadata type out of range for '" + key + "'", path, mc.pos});
    auto value = TRY(read_meta_value(mc, static_cast<MetaType>(type_raw)));
    file.metadata.emplace_back(std::move(key), std::move(value));
  }

  std::vector<Record> records;
  records.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(h.tensor_count, 65536)));
  for (std::uint64_t i = 0; i < h.tensor_count; ++i)
    records.push_back(TRY(read_record(base, size, h, i, path)));

  // decode compressed tensors once into a single anonymous mapping, unique by
  // data offset so deduplicated tensors decode only once.
  struct CompInfo {
    std::uint64_t logical;
    std::uint64_t stored;
    std::uint64_t xxh;
  };
  std::map<std::uint64_t, CompInfo> comp;
  for (const Record &r : records)
    if ((r.flags & kFlagCompressed) != 0)
      comp.emplace(r.data_offset, CompInfo{r.logical, r.stored, r.xxh});
  std::uint64_t aux_size = 0;
  for (const auto &[off, ci] : comp)
    aux_size += ci.logical;

  std::map<std::uint64_t, const std::byte *> comp_ptr;
  if (aux_size > 0) {
    file.aux = TRY(MappedFile::anonymous(static_cast<std::size_t>(aux_size)));
    std::byte *cursor = file.aux.data_mut();
    for (const auto &[off, ci] : comp) {
      std::span<const std::byte> blob{base + off, static_cast<std::size_t>(ci.stored)};
      std::span<std::byte> out{cursor, static_cast<std::size_t>(ci.logical)};
      TRY(decompress_f16(blob, out));
      if (xxhash64(cursor, static_cast<std::size_t>(ci.logical)) != ci.xxh)
        return std::unexpected(Error{
            "decoded tensor at offset " + std::to_string(off) + " fails its checksum", path, off});
      comp_ptr.emplace(off, cursor);
      cursor += ci.logical;
    }
  }

  file.tensors.reserve(records.size());
  for (const Record &r : records) {
    TensorInfo info;
    info.name = r.name;
    info.type = r.type;
    info.n_dims = r.n_dims;
    info.shape = {1, 1, 1, 1};
    for (std::uint32_t d = 0; d < r.n_dims; ++d)
      info.shape[d] = r.dims[d];
    info.offset = r.data_offset;
    info.nbytes = r.logical;
    if ((r.flags & kFlagCompressed) != 0) {
      info.data = comp_ptr.at(r.data_offset);
    } else {
      info.data = base + r.data_offset;
      if (opts.verify && xxhash64(info.data, static_cast<std::size_t>(r.logical)) != r.xxh)
        return std::unexpected(
            Error{"tensor '" + r.name + "' fails its checksum", path, r.data_offset});
    }
    file.tensors.push_back(std::move(info));
  }
  return file;
}

} // namespace

std::expected<void, Error> convert(const GgufFile &src, std::string_view path,
                                   const ConvertOptions &opts) {
  return convert_impl(src, std::string(path), opts);
}

std::expected<GgufFile, Error> read(std::string_view path, const ReadOptions &opts) {
  return read_impl(path, opts);
}

std::expected<GgufFile, Error> load_model(std::string_view path, const ReadOptions &opts) {
  const std::string path_str(path);
  FilePtr fp(std::fopen(path_str.c_str(), "rb"));
  if (!fp)
    return std::unexpected(Error{std::string("cannot open: ") + std::strerror(errno), path_str, 0});
  std::uint32_t magic = 0;
  const std::size_t got = std::fread(&magic, 1, sizeof magic, fp.get());
  fp.reset();
  if (got < sizeof magic)
    return std::unexpected(Error{"file too small to identify format", path_str, 0});
  if (magic == kMagic)
    return read_impl(path, opts);
  return gguf::load(path);
}

} // namespace dbinfer::dbmf
