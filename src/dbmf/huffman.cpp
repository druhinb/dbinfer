#include "dbmf/huffman.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <queue>
#include <string>
#include <utility>

namespace dbinfer::dbmf {

namespace {

// canonical huffman codes over a 256-symbol alphabet are capped at this length.
// a distribution whose optimal code exceeds it falls back to raw storage, so
// the decoder never sees a longer code and its per-length tables stay bounded.
constexpr std::uint32_t kMaxLen = 24;
constexpr std::size_t kBlobHeader = 16 + 256; // n_elems, hi_stream_bytes, code_len[256]

void put_u64(std::vector<std::byte> &b, std::uint64_t v) {
  std::byte tmp[8];
  std::memcpy(tmp, &v, sizeof v);
  b.insert(b.end(), tmp, tmp + sizeof v);
}

std::uint64_t get_u64(const std::byte *p) {
  std::uint64_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

// huffman code length per symbol from a frequency table, using an optimal
// bottom-up merge. lengths are read back off the tree by depth.
std::array<std::uint8_t, 256> code_lengths(const std::array<std::uint64_t, 256> &freq,
                                           std::uint32_t &max_len) {
  std::array<std::uint8_t, 256> len{};
  std::vector<std::uint64_t> nfreq;
  std::vector<int> left, right;
  std::array<int, 256> leaf_id{};
  leaf_id.fill(-1);

  using Item = std::pair<std::uint64_t, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
  for (int s = 0; s < 256; ++s) {
    if (freq[s] == 0)
      continue;
    const int id = static_cast<int>(nfreq.size());
    nfreq.push_back(freq[s]);
    left.push_back(-1);
    right.push_back(-1);
    leaf_id[s] = id;
    heap.push({freq[s], id});
  }

  max_len = 0;
  if (heap.empty())
    return len;
  if (heap.size() == 1) {
    for (int s = 0; s < 256; ++s)
      if (freq[s] > 0)
        len[s] = 1;
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
        if (leaf_id[s] == id)
          len[s] = static_cast<std::uint8_t>(d);
      continue;
    }
    stack.push_back({left[id], depth + 1});
    stack.push_back({right[id], depth + 1});
  }
  return len;
}

} // namespace

CompressResult compress_f16(std::span<const std::byte> raw) {
  CompressResult result;
  if (raw.size() % 2 != 0 || raw.empty())
    return result;

  const std::size_t n = raw.size() / 2;
  std::vector<std::byte> lo(n);
  std::vector<std::uint8_t> hi(n);
  std::array<std::uint64_t, 256> freq{};
  for (std::size_t i = 0; i < n; ++i) {
    lo[i] = raw[2 * i];
    const auto h = std::to_integer<std::uint8_t>(raw[2 * i + 1]);
    hi[i] = h;
    ++freq[h];
  }

  std::uint32_t max_len = 0;
  const std::array<std::uint8_t, 256> len = code_lengths(freq, max_len);
  if (max_len == 0 || max_len > kMaxLen)
    return result;

  std::array<std::uint32_t, kMaxLen + 1> bl_count{};
  for (int s = 0; s < 256; ++s)
    if (len[s] > 0)
      ++bl_count[len[s]];
  std::array<std::uint32_t, kMaxLen + 1> next_code{};
  std::uint32_t code = 0;
  for (std::uint32_t l = 1; l <= max_len; ++l) {
    code = (code + bl_count[l - 1]) << 1;
    next_code[l] = code;
  }
  std::array<std::uint32_t, 256> sym_code{};
  for (int s = 0; s < 256; ++s)
    if (len[s] > 0)
      sym_code[s] = next_code[len[s]]++;

  std::vector<std::byte> bits;
  bits.reserve(n);
  std::uint32_t acc = 0;
  int nbits = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t s = hi[i];
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
  if (nbits > 0)
    bits.push_back(static_cast<std::byte>((acc << (8 - nbits)) & 0xFF));

  const std::size_t total = kBlobHeader + bits.size() + n;
  if (total >= raw.size())
    return result;

  std::vector<std::byte> blob;
  blob.reserve(total);
  put_u64(blob, n);
  put_u64(blob, bits.size());
  for (int s = 0; s < 256; ++s)
    blob.push_back(static_cast<std::byte>(len[s]));
  blob.insert(blob.end(), bits.begin(), bits.end());
  blob.insert(blob.end(), lo.begin(), lo.end());

  result.bytes = std::move(blob);
  result.compressed = true;
  return result;
}

std::expected<void, gguf::Error> decompress_f16(std::span<const std::byte> blob,
                                                std::span<std::byte> out) {
  auto bad = [](std::string msg) -> std::unexpected<gguf::Error> {
    return std::unexpected(gguf::Error{"dbmf huffman: " + std::move(msg), "", 0});
  };

  if (blob.size() < kBlobHeader)
    return bad("blob shorter than header");
  const std::uint64_t n = get_u64(blob.data());
  const std::uint64_t hi_bytes = get_u64(blob.data() + 8);
  const std::byte *lens = blob.data() + 16;

  if (out.size() != n * 2)
    return bad("output size " + std::to_string(out.size()) + " does not match element count " +
               std::to_string(n));
  std::uint64_t body = 0;
  if (__builtin_add_overflow(hi_bytes, n, &body) ||
      __builtin_add_overflow(body, kBlobHeader, &body) || body != blob.size())
    return bad("blob size mismatch");

  std::array<std::uint32_t, kMaxLen + 1> bl_count{};
  std::uint32_t max_len = 0;
  for (int s = 0; s < 256; ++s) {
    const auto l = std::to_integer<std::uint8_t>(lens[s]);
    if (l > kMaxLen)
      return bad("code length " + std::to_string(l) + " exceeds cap");
    if (l > 0) {
      ++bl_count[l];
      max_len = std::max(max_len, static_cast<std::uint32_t>(l));
    }
  }

  // canonical first-code and first-symbol-index per length, plus symbols
  // ordered by (length, symbol) for the reverse lookup.
  std::array<std::uint32_t, kMaxLen + 1> first_code{};
  std::array<std::uint32_t, kMaxLen + 1> first_index{};
  std::array<std::uint8_t, 256> ordered{};
  std::uint32_t code = 0;
  std::uint32_t index = 0;
  for (std::uint32_t l = 1; l <= max_len; ++l) {
    code = (code + bl_count[l - 1]) << 1;
    first_code[l] = code;
    first_index[l] = index;
    std::uint32_t placed = 0;
    for (int s = 0; s < 256; ++s)
      if (std::to_integer<std::uint8_t>(lens[s]) == l)
        ordered[index + placed++] = static_cast<std::uint8_t>(s);
    index += bl_count[l];
  }

  const std::byte *stream = blob.data() + kBlobHeader;
  const std::byte *lo = stream + hi_bytes;

  std::uint64_t bitpos = 0;
  const std::uint64_t total_bits = hi_bytes * 8;
  for (std::uint64_t i = 0; i < n; ++i) {
    std::uint32_t cur = 0;
    std::uint32_t l = 0;
    for (;;) {
      if (bitpos >= total_bits)
        return bad("bitstream underrun");
      const std::byte byte = stream[bitpos >> 3];
      const std::uint32_t bit = (std::to_integer<std::uint8_t>(byte) >> (7 - (bitpos & 7))) & 1u;
      ++bitpos;
      cur = (cur << 1) | bit;
      ++l;
      if (l > max_len)
        return bad("no code matched within max length");
      if (bl_count[l] != 0 && cur - first_code[l] < bl_count[l]) {
        const std::uint8_t sym = ordered[first_index[l] + (cur - first_code[l])];
        out[2 * i] = lo[i];
        out[2 * i + 1] = static_cast<std::byte>(sym);
        break;
      }
    }
  }
  return {};
}

} // namespace dbinfer::dbmf
