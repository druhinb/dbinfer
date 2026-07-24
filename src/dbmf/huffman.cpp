#include "dbmf/huffman.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <queue>
#include <string>
#include <utility>

#include "try.hpp"

namespace dbinfer::dbmf {

namespace {

// canonical huffman codes over a 256-symbol alphabet are capped at this length.
// a distribution whose optimal code exceeds it falls back to raw storage, so
// the decoder never sees a longer code and its per-length tables stay bounded.
constexpr std::uint32_t kMaxLen = 24;
constexpr std::size_t kBlobHeader = 16 + 256;  // n_elems, hi_stream_bytes, code_len[256]

void put_u64(std::vector<std::byte>& b, std::uint64_t v) {
  std::byte tmp[8];
  std::memcpy(tmp, &v, sizeof v);
  b.insert(b.end(), tmp, tmp + sizeof v);
}

std::uint64_t get_u64(const std::byte* p) {
  std::uint64_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

std::unexpected<gguf::Error> huffman_error(std::string msg) {
  return std::unexpected(gguf::Error{"dbmf huffman: " + std::move(msg), "", 0});
}

// huffman code length per symbol from a frequency table, using an optimal
// bottom-up merge. lengths are read back off the tree by depth.
std::array<std::uint8_t, 256> code_lengths(const std::array<std::uint64_t, 256>& freq,
                                           std::uint32_t& max_len) {
  std::array<std::uint8_t, 256> len{};
  std::vector<std::uint64_t> nfreq;
  std::vector<int> left, right;
  std::array<int, 256> leaf_id{};
  leaf_id.fill(-1);

  using Item = std::pair<std::uint64_t, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
  for (int s = 0; s < 256; ++s) {
    if (freq[s] == 0) continue;
    const int id = static_cast<int>(nfreq.size());
    nfreq.push_back(freq[s]);
    left.push_back(-1);
    right.push_back(-1);
    leaf_id[s] = id;
    heap.push({freq[s], id});
  }

  max_len = 0;
  if (heap.empty()) return len;
  if (heap.size() == 1) {
    for (int s = 0; s < 256; ++s)
      if (freq[s] > 0) len[s] = 1;
    max_len = 1;
    return len;
  }

  while (heap.size() > 1) {
    const Item a = heap.top();
    heap.pop();
    const Item b = heap.top();
    heap.pop();
    const int id = static_cast<int>(nfreq.size());
    nfreq.push_back(a.first + b.first);
    left.push_back(a.second);
    right.push_back(b.second);
    heap.push({a.first + b.first, id});
  }

  const int root = heap.top().second;
  std::vector<std::pair<int, std::uint32_t>> stack{{root, 0}};
  while (!stack.empty()) {
    const auto [id, depth] = stack.back();
    stack.pop_back();
    if (left[id] < 0) {
      const std::uint32_t d = depth == 0 ? 1 : depth;
      max_len = std::max(max_len, d);
      for (int s = 0; s < 256; ++s)
        if (leaf_id[s] == id) len[s] = static_cast<std::uint8_t>(d);
      continue;
    }
    stack.push_back({left[id], depth + 1});
    stack.push_back({right[id], depth + 1});
  }
  return len;
}

// ---- compression -----------------------------------------------------------

struct Planes {
  std::vector<std::byte> lo;
  std::vector<std::uint8_t> hi;
  std::array<std::uint64_t, 256> freq{};
};

Planes split_planes(std::span<const std::byte> raw, std::size_t n) {
  Planes p;
  p.lo.resize(n);
  p.hi.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    p.lo[i] = raw[2 * i];
    const auto h = std::to_integer<std::uint8_t>(raw[2 * i + 1]);
    p.hi[i] = h;
    ++p.freq[h];
  }
  return p;
}

std::array<std::uint32_t, 256> assign_canonical_codes(const std::array<std::uint8_t, 256>& len,
                                                      std::uint32_t max_len) {
  std::array<std::uint32_t, kMaxLen + 1> bl_count{};
  for (int s = 0; s < 256; ++s)
    if (len[s] > 0) ++bl_count[len[s]];

  std::array<std::uint32_t, kMaxLen + 1> next_code{};
  std::uint32_t code = 0;
  for (std::uint32_t l = 1; l <= max_len; ++l) {
    code = (code + bl_count[l - 1]) << 1;
    next_code[l] = code;
  }

  std::array<std::uint32_t, 256> sym_code{};
  for (int s = 0; s < 256; ++s)
    if (len[s] > 0) sym_code[s] = next_code[len[s]]++;
  return sym_code;
}

std::vector<std::byte> pack_hi_stream(const std::vector<std::uint8_t>& hi,
                                      const std::array<std::uint8_t, 256>& len,
                                      const std::array<std::uint32_t, 256>& sym_code) {
  std::vector<std::byte> bits;
  bits.reserve(hi.size());
  std::uint32_t acc = 0;
  int nbits = 0;
  for (const std::uint8_t s : hi) {
    const std::uint32_t l = len[s];
    const std::uint32_t c = sym_code[s];
    for (std::uint32_t b = 0; b < l; ++b) {
      acc = (acc << 1) | ((c >> (l - 1 - b)) & 1u);
      if (++nbits == 8) {
        bits.push_back(static_cast<std::byte>(acc & 0xFF));
        acc = 0;
        nbits = 0;
      }
    }
  }
  if (nbits > 0) bits.push_back(static_cast<std::byte>((acc << (8 - nbits)) & 0xFF));
  return bits;
}

// ---- decompression ----------------------------------------------------------

struct DecodeHeader {
  std::uint64_t n = 0;
  std::uint64_t hi_bytes = 0;
  const std::byte* lens = nullptr;
};

std::expected<DecodeHeader, gguf::Error> parse_decode_header(std::span<const std::byte> blob,
                                                             std::span<std::byte> out) {
  if (blob.size() < kBlobHeader) return huffman_error("blob shorter than header");

  DecodeHeader h;
  h.n = get_u64(blob.data());
  h.hi_bytes = get_u64(blob.data() + 8);
  h.lens = blob.data() + 16;

  if (out.size() != h.n * 2)
    return huffman_error("output size " + std::to_string(out.size()) +
                         " does not match element count " + std::to_string(h.n));
  std::uint64_t body = 0;
  if (__builtin_add_overflow(h.hi_bytes, h.n, &body) ||
      __builtin_add_overflow(body, kBlobHeader, &body) || body != blob.size())
    return huffman_error("blob size mismatch");
  return h;
}

struct LengthHistogram {
  std::array<std::uint32_t, kMaxLen + 1> bl_count{};
  std::uint32_t max_len = 0;
};

std::expected<LengthHistogram, gguf::Error> histogram_code_lengths(const std::byte* lens) {
  LengthHistogram hist;
  for (int s = 0; s < 256; ++s) {
    const auto l = std::to_integer<std::uint8_t>(lens[s]);
    if (l > kMaxLen) return huffman_error("code length " + std::to_string(l) + " exceeds cap");
    if (l > 0) {
      ++hist.bl_count[l];
      hist.max_len = std::max(hist.max_len, static_cast<std::uint32_t>(l));
    }
  }
  return hist;
}

// canonical first-code and first-symbol-index per length, plus symbols
// ordered by (length, symbol) for the reverse lookup.
struct DecodeTables {
  std::array<std::uint32_t, kMaxLen + 1> first_code{};
  std::array<std::uint32_t, kMaxLen + 1> first_index{};
  std::array<std::uint8_t, 256> ordered{};
};

DecodeTables build_decode_tables(const std::byte* lens, const LengthHistogram& hist) {
  DecodeTables t;
  std::uint32_t code = 0;
  std::uint32_t index = 0;
  for (std::uint32_t l = 1; l <= hist.max_len; ++l) {
    code = (code + hist.bl_count[l - 1]) << 1;
    t.first_code[l] = code;
    t.first_index[l] = index;
    std::uint32_t placed = 0;
    for (int s = 0; s < 256; ++s)
      if (std::to_integer<std::uint8_t>(lens[s]) == l)
        t.ordered[index + placed++] = static_cast<std::uint8_t>(s);
    index += hist.bl_count[l];
  }
  return t;
}

std::expected<void, gguf::Error> decode_symbols(std::span<const std::byte> blob,
                                                const DecodeHeader& h, const LengthHistogram& hist,
                                                const DecodeTables& t, std::span<std::byte> out) {
  const std::byte* stream = blob.data() + kBlobHeader;
  const std::byte* lo = stream + h.hi_bytes;
  const std::uint64_t total_bits = h.hi_bytes * 8;

  std::uint64_t bitpos = 0;
  for (std::uint64_t i = 0; i < h.n; ++i) {
    std::uint32_t cur = 0;
    std::uint32_t l = 0;
    for (;;) {
      if (bitpos >= total_bits) return huffman_error("bitstream underrun");
      const std::byte byte = stream[bitpos >> 3];
      const std::uint32_t bit = (std::to_integer<std::uint8_t>(byte) >> (7 - (bitpos & 7))) & 1u;
      ++bitpos;
      cur = (cur << 1) | bit;
      ++l;
      if (l > hist.max_len) return huffman_error("no code matched within max length");
      if (hist.bl_count[l] != 0 && cur - t.first_code[l] < hist.bl_count[l]) {
        const std::uint8_t sym = t.ordered[t.first_index[l] + (cur - t.first_code[l])];
        out[2 * i] = lo[i];
        out[2 * i + 1] = static_cast<std::byte>(sym);
        break;
      }
    }
  }
  return {};
}

}  // namespace

CompressResult compress_f16(std::span<const std::byte> raw) {
  CompressResult result;
  if (raw.size() % 2 != 0 || raw.empty()) return result;

  const std::size_t n = raw.size() / 2;
  const Planes planes = split_planes(raw, n);

  std::uint32_t max_len = 0;
  const std::array<std::uint8_t, 256> len = code_lengths(planes.freq, max_len);
  if (max_len == 0 || max_len > kMaxLen) return result;

  const std::array<std::uint32_t, 256> sym_code = assign_canonical_codes(len, max_len);
  const std::vector<std::byte> bits = pack_hi_stream(planes.hi, len, sym_code);

  const std::size_t total = kBlobHeader + bits.size() + n;
  if (total >= raw.size()) return result;

  std::vector<std::byte> blob;
  blob.reserve(total);
  put_u64(blob, n);
  put_u64(blob, bits.size());
  for (int s = 0; s < 256; ++s) blob.push_back(static_cast<std::byte>(len[s]));
  blob.insert(blob.end(), bits.begin(), bits.end());
  blob.insert(blob.end(), planes.lo.begin(), planes.lo.end());

  result.bytes = std::move(blob);
  result.compressed = true;
  return result;
}

std::expected<void, gguf::Error> decompress_f16(std::span<const std::byte> blob,
                                                std::span<std::byte> out) {
  const DecodeHeader h = TRY(parse_decode_header(blob, out));
  const LengthHistogram hist = TRY(histogram_code_lengths(h.lens));
  const DecodeTables tables = build_decode_tables(h.lens, hist);
  return decode_symbols(blob, h, hist, tables, out);
}

}  // namespace dbinfer::dbmf
