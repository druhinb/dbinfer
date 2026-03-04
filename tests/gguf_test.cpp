#include "gguf/gguf.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include <unistd.h>

namespace {

int g_failures = 0;

void check(bool ok, const char *what) {
  if (ok) {
    std::printf("PASS %s\n", what);
  } else {
    std::printf("FAIL %s\n", what);
    ++g_failures;
  }
}

struct Buf {
  std::vector<std::uint8_t> bytes;

  void raw(const void *p, std::size_t n) {
    const auto *b = static_cast<const std::uint8_t *>(p);
    bytes.insert(bytes.end(), b, b + n);
  }
  template <typename T> void scalar(T v) { raw(&v, sizeof(T)); }
  void u32(std::uint32_t v) { scalar(v); }
  void u64(std::uint64_t v) { scalar(v); }
  void str(const std::string &s) {
    u64(s.size());
    raw(s.data(), s.size());
  }
  void meta_key_type(const std::string &key, dbinfer::gguf::MetaType t) {
    str(key);
    u32(static_cast<std::uint32_t>(t));
  }
  std::size_t size() const { return bytes.size(); }
};

// returns the full buffer; sets *infos_start to the byte offset where tensor
// infos begin (for the truncation test).
std::vector<std::uint8_t> build_good(std::size_t *infos_start) {
  using MT = dbinfer::gguf::MetaType;
  Buf b;
  b.u32(0x46554747u); // 'GGUF'
  b.u32(3);           // version
  b.u64(2);           // tensor_count
  b.u64(14);          // kv_count

  b.meta_key_type("general.alignment", MT::UInt32);
  b.u32(32);
  b.meta_key_type("k.u8", MT::UInt8);
  b.scalar<std::uint8_t>(0xAB);
  b.meta_key_type("k.i8", MT::Int8);
  b.scalar<std::int8_t>(-5);
  b.meta_key_type("k.u16", MT::UInt16);
  b.scalar<std::uint16_t>(0x1234);
  b.meta_key_type("k.i16", MT::Int16);
  b.scalar<std::int16_t>(-1000);
  b.meta_key_type("k.u32", MT::UInt32);
  b.u32(0xDEADBEEFu);
  b.meta_key_type("k.i32", MT::Int32);
  b.scalar<std::int32_t>(-123456);
  b.meta_key_type("k.f32", MT::Float32);
  b.scalar<float>(3.5f);
  b.meta_key_type("k.bool", MT::Bool);
  b.scalar<std::uint8_t>(1);
  b.meta_key_type("k.str", MT::String);
  b.str("hello");
  b.meta_key_type("k.u64", MT::UInt64);
  b.u64(0x1122334455667788ull);
  b.meta_key_type("k.i64", MT::Int64);
  b.scalar<std::int64_t>(-9000000000ll);
  b.meta_key_type("k.f64", MT::Float64);
  b.scalar<double>(2.718281828);
  b.meta_key_type("k.arr", MT::Array);
  b.u32(static_cast<std::uint32_t>(MT::Int32));
  b.u64(3);
  b.scalar<std::int32_t>(10);
  b.scalar<std::int32_t>(20);
  b.scalar<std::int32_t>(30);

  *infos_start = b.size();

  // Tensor A: file dims (in=4, out=8) => row-major shape [8,4]; F32, 32 elems.
  b.str("tA");
  b.u32(2);
  b.u64(4);
  b.u64(8);
  b.u32(static_cast<std::uint32_t>(dbinfer::gguf::GgmlType::F32));
  b.u64(0);
  // Tensor B: 1-D of 16; F32; offset 128 (aligned to 32).
  b.str("tB");
  b.u32(1);
  b.u64(16);
  b.u32(static_cast<std::uint32_t>(dbinfer::gguf::GgmlType::F32));
  b.u64(128);

  const std::size_t align = 32;
  while (b.size() % align != 0)
    b.scalar<std::uint8_t>(0);

  // Data blob: 128 bytes (tA) + 64 bytes (tB) = 192, with a marker pattern.
  for (std::size_t i = 0; i < 192; ++i)
    b.scalar<std::uint8_t>(static_cast<std::uint8_t>(i & 0xFF));

  return b.bytes;
}

std::string write_temp(const std::vector<std::uint8_t> &bytes) {
  char tmpl[] = "/tmp/gguf_test_XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    std::printf("FAIL cannot create temp file\n");
    ++g_failures;
    return {};
  }
  std::size_t off = 0;
  while (off < bytes.size()) {
    ssize_t w = ::write(fd, bytes.data() + off, bytes.size() - off);
    if (w <= 0)
      break;
    off += static_cast<std::size_t>(w);
  }
  ::close(fd);
  return std::string(tmpl);
}

template <typename T> bool meta_is(const dbinfer::gguf::GgufFile &f, const char *key, T expected) {
  const dbinfer::gguf::MetaValue *mv = f.find_meta(key);
  if (mv == nullptr)
    return false;
  const auto *p = std::get_if<T>(&mv->value);
  return p != nullptr && *p == expected;
}

void test_happy() {
  std::size_t infos_start = 0;
  auto bytes = build_good(&infos_start);
  std::string path = write_temp(bytes);
  auto loaded = dbinfer::gguf::load(path);
  ::unlink(path.c_str());

  if (!loaded) {
    std::printf("FAIL happy load: %s\n", dbinfer::gguf::to_string(loaded.error()).c_str());
    ++g_failures;
    return;
  }
  const auto &f = *loaded;

  check(f.version == 3, "version == 3");
  check(f.alignment == 32, "alignment == 32");
  check(f.tensors.size() == 2, "tensor_count == 2");
  check(f.metadata.size() == 14, "kv_count == 14");

  check(meta_is<std::uint8_t>(f, "k.u8", 0xAB), "k.u8 decodes");
  check(meta_is<std::int8_t>(f, "k.i8", -5), "k.i8 decodes");
  check(meta_is<std::uint16_t>(f, "k.u16", 0x1234), "k.u16 decodes");
  check(meta_is<std::int16_t>(f, "k.i16", -1000), "k.i16 decodes");
  check(meta_is<std::uint32_t>(f, "k.u32", 0xDEADBEEFu), "k.u32 decodes");
  check(meta_is<std::int32_t>(f, "k.i32", -123456), "k.i32 decodes");
  check(meta_is<float>(f, "k.f32", 3.5f), "k.f32 decodes");
  check(meta_is<bool>(f, "k.bool", true), "k.bool decodes");
  check(meta_is<std::string>(f, "k.str", std::string("hello")), "k.str decodes");
  check(meta_is<std::uint64_t>(f, "k.u64", 0x1122334455667788ull), "k.u64 decodes");
  check(meta_is<std::int64_t>(f, "k.i64", -9000000000ll), "k.i64 decodes");
  check(meta_is<double>(f, "k.f64", 2.718281828), "k.f64 decodes");

  const auto *arr_mv = f.find_meta("k.arr");
  const auto *arr = arr_mv ? std::get_if<dbinfer::gguf::MetaArray>(&arr_mv->value) : nullptr;
  bool arr_ok =
      arr != nullptr && arr->elem_type == dbinfer::gguf::MetaType::Int32 && arr->values.size() == 3;
  if (arr_ok) {
    for (int i = 0; i < 3; ++i) {
      const auto *v = std::get_if<std::int32_t>(&arr->values[static_cast<std::size_t>(i)].value);
      arr_ok = arr_ok && v != nullptr && *v == (i + 1) * 10;
    }
  }
  check(arr_ok, "k.arr decodes (Int32 x3 = [10,20,30])");

  check(f.find_meta("does.not.exist") == nullptr, "find_meta miss returns null");

  const auto &tA = f.tensors[0];
  const auto &tB = f.tensors[1];
  check(tA.name == "tA" && tA.n_dims == 2, "tA name/n_dims");
  check(tA.shape[0] == 8 && tA.shape[1] == 4, "tA shape reversal file(4,8)->[8,4]");
  check(tA.nbytes == 128, "tA nbytes == 128");
  check(tB.shape[0] == 16 && tB.n_dims == 1, "tB shape [16]");
  check(tB.nbytes == 64, "tB nbytes == 64");

  const std::byte *base = f.mapping.data();
  const std::byte *end = base + f.mapping.size();
  bool a_in = tA.data >= base && tA.data + tA.nbytes <= end;
  bool b_in = tB.data >= base && tB.data + tB.nbytes <= end;
  check(a_in, "tA data pointer within mapping");
  check(b_in, "tB data pointer within mapping");

  // tB data begins at data_section + 128; first byte of the marker pattern
  // there is (128 & 0xFF) == 128.
  if (b_in) {
    check(std::to_integer<std::uint8_t>(tB.data[0]) == 128, "tB data content offset correct");
  }
}

void test_bad_magic() {
  std::size_t infos_start = 0;
  auto bytes = build_good(&infos_start);
  bytes[0] = 0x00; // corrupt magic
  std::string path = write_temp(bytes);
  auto loaded = dbinfer::gguf::load(path);
  ::unlink(path.c_str());
  check(!loaded, "bad magic -> Error");
}

void test_truncated() {
  std::size_t infos_start = 0;
  auto bytes = build_good(&infos_start);
  bytes.resize(infos_start + 3); // cut mid tensor-info
  std::string path = write_temp(bytes);
  auto loaded = dbinfer::gguf::load(path);
  ::unlink(path.c_str());
  check(!loaded, "truncated mid tensor-info -> Error");
}

void test_offset_past_eof() {
  std::size_t infos_start = 0;
  auto bytes = build_good(&infos_start);
  // Overwrite tA's offset (first tensor info: name "tA" = 8(len)+2, n_dims=4,
  // dims 8+8, type 4, then offset u64) with a huge aligned value.
  std::size_t p = infos_start;
  p += 8 + 2; // name
  p += 4;     // n_dims
  p += 16;    // two dims
  p += 4;     // type
  std::uint64_t huge = 1ull << 30;
  std::memcpy(bytes.data() + p, &huge, sizeof(huge));
  std::string path = write_temp(bytes);
  auto loaded = dbinfer::gguf::load(path);
  ::unlink(path.c_str());
  check(!loaded, "tensor offset past EOF -> Error");
}

// An array whose element type is itself Array is malformed per the GGUF spec and
// must be rejected, not recursed into.
void test_nested_array() {
  using MT = dbinfer::gguf::MetaType;
  Buf b;
  b.u32(0x46554747u);
  b.u32(3);
  b.u64(0); // no tensors
  b.u64(1); // one kv
  b.meta_key_type("k.nested", MT::Array);
  b.u32(static_cast<std::uint32_t>(MT::Array)); // element type = Array
  b.u64(1);
  std::string path = write_temp(b.bytes);
  auto loaded = dbinfer::gguf::load(path);
  ::unlink(path.c_str());
  check(!loaded, "nested array -> Error");
}

// Dims whose product overflows uint64 must be trapped, not wrapped into a small
// element count that slips past the data-region bounds check.
void test_dim_overflow() {
  std::size_t infos_start = 0;
  auto bytes = build_good(&infos_start);
  // tA's two dims live at infos_start + 8(len)+2(name) + 4(n_dims).
  std::size_t p = infos_start + 8 + 2 + 4;
  std::uint64_t big = 0xFFFFFFFFFFFFFFFFull;
  std::memcpy(bytes.data() + p, &big, sizeof(big));
  std::memcpy(bytes.data() + p + 8, &big, sizeof(big));
  std::string path = write_temp(bytes);
  auto loaded = dbinfer::gguf::load(path);
  ::unlink(path.c_str());
  check(!loaded, "dim product overflow -> Error");
}

} // namespace

int main() {
  test_happy();
  test_bad_magic();
  test_truncated();
  test_offset_past_eof();
  test_nested_array();
  test_dim_overflow();

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
