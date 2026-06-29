#include "dbmf/xxhash.hpp"

#include <cstring>

namespace dbinfer::dbmf {

namespace {

constexpr std::uint64_t kP1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kP2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kP3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kP4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kP5 = 0x27D4EB2F165667C5ULL;

std::uint64_t rotl(std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

std::uint64_t read64(const std::byte *p) {
  std::uint64_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

std::uint32_t read32(const std::byte *p) {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

std::uint64_t round64(std::uint64_t acc, std::uint64_t input) {
  acc += input * kP2;
  acc = rotl(acc, 31);
  acc *= kP1;
  return acc;
}

std::uint64_t merge_round(std::uint64_t acc, std::uint64_t val) {
  val = round64(0, val);
  acc ^= val;
  acc = acc * kP1 + kP4;
  return acc;
}

} // namespace

std::uint64_t xxhash64(const std::byte *p, std::size_t len, std::uint64_t seed) {
  const std::byte *const end = p + len;
  std::uint64_t h;

  if (len >= 32) {
    const std::byte *const limit = end - 32;
    std::uint64_t v1 = seed + kP1 + kP2;
    std::uint64_t v2 = seed + kP2;
    std::uint64_t v3 = seed;
    std::uint64_t v4 = seed - kP1;
    do {
      v1 = round64(v1, read64(p));
      p += 8;
      v2 = round64(v2, read64(p));
      p += 8;
      v3 = round64(v3, read64(p));
      p += 8;
      v4 = round64(v4, read64(p));
      p += 8;
    } while (p <= limit);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge_round(h, v1);
    h = merge_round(h, v2);
    h = merge_round(h, v3);
    h = merge_round(h, v4);
  } else {
    h = seed + kP5;
  }

  h += static_cast<std::uint64_t>(len);

  while (p + 8 <= end) {
    h ^= round64(0, read64(p));
    h = rotl(h, 27) * kP1 + kP4;
    p += 8;
  }
  if (p + 4 <= end) {
    h ^= static_cast<std::uint64_t>(read32(p)) * kP1;
    h = rotl(h, 23) * kP2 + kP3;
    p += 4;
  }
  while (p < end) {
    h ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(*p)) * kP5;
    h = rotl(h, 11) * kP1;
    ++p;
  }

  h ^= h >> 33;
  h *= kP2;
  h ^= h >> 29;
  h *= kP3;
  h ^= h >> 32;
  return h;
}

} // namespace dbinfer::dbmf
