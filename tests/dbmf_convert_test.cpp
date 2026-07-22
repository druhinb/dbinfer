#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "dbmf/dbmf.hpp"
#include "dbmf/huffman.hpp"
#include "dbmf/xxhash.hpp"
#include "gguf/gguf.hpp"
#include "model/model.hpp"

// dbmf gate: a converted model reproduces the gguf original's logits to the
// bit, for q8_0 raw, f16 raw, and f16 compressed. plus round-trip, dedup,
// corruption detection, and a header-parser fuzz, all under the same binary.

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
  std::printf("FAIL %s\n", msg.c_str());
  ++g_failures;
}

std::vector<std::int32_t> make_prompt() {
  std::vector<std::int32_t> ids;
  for (std::size_t i = 0; i < 24; ++i)
    ids.push_back(static_cast<std::int32_t>((i * 137 + 5) % 811));
  return ids;
}

std::vector<float> run_logits(const dbinfer::gguf::GgufFile& file,
                              const std::vector<std::int32_t>& ids) {
  auto mret = dbinfer::model::Model::load(file);
  if (!mret) {
    fail(std::string("model load: ") + dbinfer::gguf::to_string(mret.error()));
    return {};
  }
  dbinfer::model::Model& model = *mret;
  const std::size_t vocab = model.config().vocab_size;
  std::vector<float> out(ids.size() * vocab);
  for (std::size_t s = 0; s < ids.size(); ++s) {
    const float* l = model.forward(ids[s], static_cast<std::int32_t>(s));
    std::memcpy(out.data() + s * vocab, l, vocab * sizeof(float));
  }
  return out;
}

std::vector<std::byte> read_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  std::vector<std::byte> data;
  if (f == nullptr) return data;
  std::byte buf[65536];
  std::size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) data.insert(data.end(), buf, buf + got);
  std::fclose(f);
  return data;
}

void write_file(const std::string& path, std::span<const std::byte> data) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return;
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
}

std::string dir() { return std::string(DBINFER_DBMF_DIR) + "/"; }

// bitwise logits gate for one source model in one storage mode.
void logits_gate(const char* label, const char* gguf_path, bool compress,
                 const std::vector<std::int32_t>& ids) {
  auto gloaded = dbinfer::gguf::load(gguf_path);
  if (!gloaded) {
    fail(std::string(label) + " load gguf: " + dbinfer::gguf::to_string(gloaded.error()));
    return;
  }
  const std::string out = dir() + "gate_" + label + (compress ? "_c.dbmf" : ".dbmf");
  dbinfer::dbmf::ConvertOptions copts;
  copts.compress = compress;
  if (auto ok = dbinfer::dbmf::convert(*gloaded, out, copts); !ok) {
    fail(std::string(label) + " convert: " + dbinfer::gguf::to_string(ok.error()));
    return;
  }
  auto dloaded = dbinfer::dbmf::read(out);
  if (!dloaded) {
    fail(std::string(label) + " read dbmf: " + dbinfer::gguf::to_string(dloaded.error()));
    return;
  }

  std::vector<float> a = run_logits(*gloaded, ids);
  std::vector<float> b = run_logits(*dloaded, ids);
  if (a.empty() || a.size() != b.size()) {
    fail(std::string(label) + " logit size mismatch");
    return;
  }
  if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
    std::size_t diffs = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
      if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++diffs;
    fail(std::string(label) + " logits differ in " + std::to_string(diffs) + "/" +
         std::to_string(a.size()) + " positions");
    return;
  }
  std::printf("PASS %-14s bitwise identical over %zu logits (%s)\n", label, a.size(),
              compress ? "compressed" : "raw");
}

// ---- synthetic model for round-trip, dedup, corruption, fuzz -------------

struct Owned {
  std::vector<std::byte> bytes;
};

dbinfer::gguf::GgufFile make_synthetic(std::vector<Owned>& store) {
  using dbinfer::gguf::GgmlType;
  using dbinfer::gguf::GgufFile;
  using dbinfer::gguf::MetaArray;
  using dbinfer::gguf::MetaType;
  using dbinfer::gguf::MetaValue;
  using dbinfer::gguf::TensorInfo;
  using dbinfer::gguf::type_info;
  using dbinfer::gguf::TypeInfo;
  GgufFile f;
  f.version = 3;
  f.alignment = 32;
  f.metadata.emplace_back("general.architecture", MetaValue{std::string("qwen2")});
  f.metadata.emplace_back("qwen2.block_count", MetaValue{std::uint32_t{1}});
  MetaArray toks;
  toks.elem_type = MetaType::String;
  toks.values.push_back(MetaValue{std::string("hello")});
  toks.values.push_back(MetaValue{std::string("world")});
  f.metadata.emplace_back("tokenizer.ggml.tokens", MetaValue{std::move(toks)});

  auto add = [&](const std::string& name, GgmlType type, std::uint64_t rows, std::uint64_t cols,
                 std::byte fill) {
    const TypeInfo ti = type_info(type);
    const std::uint64_t nelem = rows * cols;
    const std::uint64_t nbytes = nelem / ti.block_size * ti.type_size;
    Owned o;
    o.bytes.assign(static_cast<std::size_t>(nbytes), fill);
    for (std::size_t i = 0; i < o.bytes.size(); ++i)
      o.bytes[i] = static_cast<std::byte>((i * 31 + std::to_integer<int>(fill)) & 0xFF);
    store.push_back(std::move(o));
    TensorInfo t;
    t.name = name;
    t.type = type;
    t.n_dims = 2;
    t.shape = {rows, cols, 1, 1};
    t.nbytes = nbytes;
    t.data = store.back().bytes.data();
    f.tensors.push_back(std::move(t));
  };

  add("token_embd.weight", GgmlType::F16, 64, 32, std::byte{0x11});
  add("blk.0.attn_q.weight", GgmlType::Q8_0, 32, 32, std::byte{0x22});
  add("blk.0.ffn_down.weight", GgmlType::F32, 4, 8, std::byte{0x33});
  // byte-identical twin of token_embd to exercise dedup.
  add("output.weight", GgmlType::F16, 64, 32, std::byte{0x11});
  return f;
}

void roundtrip_and_dedup(std::vector<Owned>& store) {
  dbinfer::gguf::GgufFile src = make_synthetic(store);
  const std::string out = dir() + "synthetic.dbmf";
  if (auto ok = dbinfer::dbmf::convert(src, out); !ok) {
    fail("synthetic convert: " + dbinfer::gguf::to_string(ok.error()));
    return;
  }
  auto dloaded = dbinfer::dbmf::read(out, {/*verify=*/true});
  if (!dloaded) {
    fail("synthetic read: " + dbinfer::gguf::to_string(dloaded.error()));
    return;
  }
  const dbinfer::gguf::GgufFile& df = *dloaded;
  if (df.tensors.size() != src.tensors.size()) {
    fail("synthetic tensor count mismatch");
    return;
  }
  for (const auto& st : src.tensors) {
    const dbinfer::gguf::TensorInfo* dt = nullptr;
    for (const auto& t : df.tensors)
      if (t.name == st.name) dt = &t;
    if (dt == nullptr || dt->nbytes != st.nbytes ||
        std::memcmp(dt->data, st.data, static_cast<std::size_t>(st.nbytes)) != 0) {
      fail("synthetic tensor bytes mismatch: " + st.name);
      return;
    }
  }
  // token_embd and its twin must share one data offset after dedup.
  const dbinfer::gguf::TensorInfo* e = nullptr;
  const dbinfer::gguf::TensorInfo* o = nullptr;
  for (const auto& t : df.tensors) {
    if (t.name == "token_embd.weight") e = &t;
    if (t.name == "output.weight") o = &t;
  }
  if (e == nullptr || o == nullptr || e->offset != o->offset || e->data != o->data) {
    fail("dedup did not share identical tensors");
    return;
  }
  std::printf("PASS synthetic     round-trip + dedup (tied tensor shares offset %llu)\n",
              static_cast<std::uint64_t>(e->offset));
}

void corruption_gate(std::vector<Owned>& store) {
  dbinfer::gguf::GgufFile src = make_synthetic(store);
  const std::string good = dir() + "corrupt_src.dbmf";
  if (auto ok = dbinfer::dbmf::convert(src, good); !ok) {
    fail("corruption convert: " + dbinfer::gguf::to_string(ok.error()));
    return;
  }
  auto base = dbinfer::dbmf::read(good, {/*verify=*/true});
  if (!base) {
    fail("corruption base read: " + dbinfer::gguf::to_string(base.error()));
    return;
  }
  // locate the q8 tensor's data offset to flip a byte inside it.
  std::uint64_t off = 0;
  std::string target;
  for (const auto& t : base->tensors)
    if (t.name == "blk.0.attn_q.weight") {
      off = t.offset;
      target = t.name;
    }
  std::vector<std::byte> bytes = read_file(good);

  // 1. flipped tensor byte, named in the error.
  auto flipped = bytes;
  flipped[static_cast<std::size_t>(off)] ^= std::byte{0xFF};
  const std::string fpath = dir() + "corrupt_flip.dbmf";
  write_file(fpath, flipped);
  auto r1 = dbinfer::dbmf::read(fpath, {/*verify=*/true});
  if (r1 || r1.error().message.find(target) == std::string::npos) {
    fail("flipped byte not caught with tensor name");
  } else {
    std::printf("PASS corruption    flipped byte -> %s\n",
                dbinfer::gguf::to_string(r1.error()).c_str());
  }

  // 2. wrong magic.
  auto badmagic = bytes;
  badmagic[0] = std::byte{0x00};
  const std::string mpath = dir() + "corrupt_magic.dbmf";
  write_file(mpath, badmagic);
  auto r2 = dbinfer::dbmf::read(mpath);
  if (r2)
    fail("wrong magic accepted");
  else
    std::printf("PASS corruption    wrong magic -> %s\n",
                dbinfer::gguf::to_string(r2.error()).c_str());

  // 3. truncated file.
  const std::string tpath = dir() + "corrupt_trunc.dbmf";
  write_file(tpath, std::span<const std::byte>(bytes.data(), bytes.size() / 2));
  auto r3 = dbinfer::dbmf::read(tpath);
  if (r3)
    fail("truncated file accepted");
  else
    std::printf("PASS corruption    truncated -> %s\n",
                dbinfer::gguf::to_string(r3.error()).c_str());

  // 4. corrupted header field caught by the header checksum.
  auto badhdr = bytes;
  badhdr[16] ^= std::byte{0x01};  // tensor_count low byte
  const std::string hpath = dir() + "corrupt_hdr.dbmf";
  write_file(hpath, badhdr);
  auto r4 = dbinfer::dbmf::read(hpath);
  if (r4)
    fail("corrupted header accepted");
  else
    std::printf("PASS corruption    header field -> %s\n",
                dbinfer::gguf::to_string(r4.error()).c_str());
}

// random single-byte mutations of a valid file must never crash the parser.
void fuzz_parser(std::vector<Owned>& store) {
  dbinfer::gguf::GgufFile src = make_synthetic(store);
  const std::string good = dir() + "fuzz_src.dbmf";
  if (auto ok = dbinfer::dbmf::convert(src, good, {/*compress=*/true}); !ok) {
    fail("fuzz convert: " + dbinfer::gguf::to_string(ok.error()));
    return;
  }
  std::vector<std::byte> bytes = read_file(good);
  std::mt19937 rng(12345);
  std::uniform_int_distribution<std::size_t> pos(0, bytes.empty() ? 0 : bytes.size() - 1);
  std::uniform_int_distribution<int> val(0, 255);
  const std::string fpath = dir() + "fuzz_mut.dbmf";
  int crashes_avoided = 0;
  for (int iter = 0; iter < 64; ++iter) {
    auto mutated = bytes;
    const int flips = 1 + (iter % 6);
    for (int k = 0; k < flips; ++k) mutated[pos(rng)] = static_cast<std::byte>(val(rng));
    write_file(fpath, mutated);
    auto r = dbinfer::dbmf::read(fpath, {/*verify=*/true});
    (void)r;  // ok or actionable error, never a crash (ASan-checked)
    ++crashes_avoided;
  }
  std::printf("PASS fuzz          %d mutated files parsed without crash\n", crashes_avoided);
}

void huffman_unit() {
  // an f16 plane the coder actually shrinks, then a bad blob the decoder rejects.
  std::vector<std::byte> raw(4096);
  for (std::size_t i = 0; i < raw.size(); i += 2) {
    raw[i] = static_cast<std::byte>(i & 0xFF);          // low byte varies
    raw[i + 1] = static_cast<std::byte>((i / 64) & 3);  // high byte low entropy
  }
  dbinfer::dbmf::CompressResult r = dbinfer::dbmf::compress_f16(raw);
  if (!r.compressed) {
    fail("huffman did not compress a low-entropy plane");
    return;
  }
  std::vector<std::byte> out(raw.size());
  if (auto ok = dbinfer::dbmf::decompress_f16(r.bytes, out); !ok) {
    fail("huffman decode: " + dbinfer::gguf::to_string(ok.error()));
    return;
  }
  if (std::memcmp(out.data(), raw.data(), raw.size()) != 0) {
    fail("huffman round-trip not bit-exact");
    return;
  }
  std::vector<std::byte> junk(8, std::byte{0x55});
  if (dbinfer::dbmf::decompress_f16(junk, out))
    fail("huffman decoder accepted a truncated blob");
  else
    std::printf("PASS huffman       round-trip exact, short blob rejected\n");
}

}  // namespace

int main() {
  const std::vector<std::int32_t> ids = make_prompt();

#ifdef DBINFER_QUANT_Q8
  logits_gate("q8_0", DBINFER_QUANT_Q8, false, ids);
#endif
#ifdef DBINFER_TEST_GGUF
  logits_gate("f16", DBINFER_TEST_GGUF, false, ids);
  logits_gate("f16_compressed", DBINFER_TEST_GGUF, true, ids);
#endif

  huffman_unit();
  {
    std::vector<Owned> store;
    roundtrip_and_dedup(store);
  }
  {
    std::vector<Owned> store;
    corruption_gate(store);
  }
  {
    std::vector<Owned> store;
    fuzz_parser(store);
  }

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
