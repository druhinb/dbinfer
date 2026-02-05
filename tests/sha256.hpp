// Self-contained SHA-256. Public-domain algorithm (FIPS 180-4), our own code.
// Keeps the stdlib-only rule intact: no CommonCrypto / OpenSSL dependency.
// Header-only, no exceptions, no allocations beyond the returned std::string.
#ifndef DBINFER_TESTS_SHA256_HPP
#define DBINFER_TESTS_SHA256_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace dbinfer::sha256 {

class Sha256 {
public:
  Sha256() { reset(); }

  void reset() {
    len_ = 0;
    buf_len_ = 0;
    h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  }

  void update(const std::uint8_t *data, std::size_t n) {
    len_ += n;
    for (std::size_t i = 0; i < n; ++i) {
      buf_[buf_len_++] = data[i];
      if (buf_len_ == 64) {
        process(buf_.data());
        buf_len_ = 0;
      }
    }
  }

  // Finalizes and returns the lowercase hex digest. Consumes the state.
  std::string hex() {
    std::uint64_t bit_len = len_ * 8;
    std::uint8_t pad = 0x80;
    update(&pad, 1);
    std::uint8_t zero = 0x00;
    while (buf_len_ != 56)
      update(&zero, 1);
    std::array<std::uint8_t, 8> lenbytes{};
    for (int i = 0; i < 8; ++i)
      lenbytes[i] = static_cast<std::uint8_t>(bit_len >> (56 - 8 * i));
    update(lenbytes.data(), 8);

    static const char *hexd = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint32_t word : h_) {
      for (int b = 3; b >= 0; --b) {
        std::uint8_t byte = static_cast<std::uint8_t>(word >> (8 * b));
        out.push_back(hexd[byte >> 4]);
        out.push_back(hexd[byte & 0x0f]);
      }
    }
    return out;
  }

private:
  static std::uint32_t rotr(std::uint32_t x, std::uint32_t n) { return (x >> n) | (x << (32 - n)); }

  void process(const std::uint8_t *p) {
    static const std::uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};

    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(p[i * 4]) << 24) |
             (static_cast<std::uint32_t>(p[i * 4 + 1]) << 16) |
             (static_cast<std::uint32_t>(p[i * 4 + 2]) << 8) |
             (static_cast<std::uint32_t>(p[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
      std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    std::uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
      std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      std::uint32_t ch = (e & f) ^ (~e & g);
      std::uint32_t t1 = hh + S1 + ch + k[i] + w[i];
      std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t t2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += hh;
  }

  std::array<std::uint32_t, 8> h_{};
  std::array<std::uint8_t, 64> buf_{};
  std::size_t buf_len_ = 0;
  std::uint64_t len_ = 0;
};

// Convenience: digest a byte span in one call.
inline std::string hex_digest(const std::uint8_t *data, std::size_t n) {
  Sha256 s;
  s.update(data, n);
  return s.hex();
}

} // namespace dbinfer::sha256

#endif // DBINFER_TESTS_SHA256_HPP
