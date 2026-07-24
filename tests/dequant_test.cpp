#include "tensor/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "test_util.hpp"

namespace {

using dbinfer::test::g_failures;

void check_eq(const char* what, float got, float want) {
  if (got == want) {
    std::printf("PASS %-24s %.9g\n", what, static_cast<double>(got));
  } else {
    std::printf("FAIL %-24s got %.9g want %.9g\n", what, static_cast<double>(got),
                static_cast<double>(want));
    ++g_failures;
  }
}

void check_le(const char* what, float got, float bound) {
  if (got <= bound) {
    std::printf("PASS %-24s %.9g <= %.9g\n", what, static_cast<double>(got),
                static_cast<double>(bound));
  } else {
    std::printf("FAIL %-24s %.9g > %.9g\n", what, static_cast<double>(got),
                static_cast<double>(bound));
    ++g_failures;
  }
}

void check_u16(const char* what, std::uint16_t got, std::uint16_t want) {
  if (got == want) {
    std::printf("PASS %-24s 0x%04X\n", what, got);
  } else {
    std::printf("FAIL %-24s got 0x%04X want 0x%04X\n", what, got, want);
    ++g_failures;
  }
}

bool signbit_of(float v) { return std::signbit(v); }

void test_f16_to_f32_special_values() {
  using dbinfer::tensor::f16_to_f32;

  check_eq("0x0000 -> +0", f16_to_f32(0x0000), 0.0f);
  if (signbit_of(f16_to_f32(0x0000))) {
    std::printf("FAIL 0x0000 sign is negative\n");
    ++g_failures;
  }

  {
    float v = f16_to_f32(0x8000);
    if (v == 0.0f && signbit_of(v)) {
      std::printf("PASS %-24s -0\n", "0x8000 -> -0");
    } else {
      std::printf("FAIL 0x8000 -> not -0 (%.9g)\n", static_cast<double>(v));
      ++g_failures;
    }
  }

  check_eq("0x3C00 -> 1", f16_to_f32(0x3C00), 1.0f);
  check_eq("0xC000 -> -2", f16_to_f32(0xC000), -2.0f);

  {
    float v = f16_to_f32(0x7C00);
    if (std::isinf(v) && v > 0.0f) {
      std::printf("PASS %-24s +inf\n", "0x7C00 -> +inf");
    } else {
      std::printf("FAIL 0x7C00 -> not +inf (%.9g)\n", static_cast<double>(v));
      ++g_failures;
    }
  }

  {
    float v = f16_to_f32(0xFC00);
    if (std::isinf(v) && v < 0.0f) {
      std::printf("PASS %-24s -inf\n", "0xFC00 -> -inf");
    } else {
      std::printf("FAIL 0xFC00 -> not -inf (%.9g)\n", static_cast<double>(v));
      ++g_failures;
    }
  }

  {
    float v = f16_to_f32(0x7E00);  // a NaN pattern (exp=31, mant!=0)
    if (std::isnan(v)) {
      std::printf("PASS %-24s NaN\n", "0x7E00 -> NaN");
    } else {
      std::printf("FAIL 0x7E00 -> not NaN (%.9g)\n", static_cast<double>(v));
      ++g_failures;
    }
  }

  check_eq("0x0001 -> 2^-24", f16_to_f32(0x0001), std::ldexp(1.0f, -24));
  check_eq("0x7BFF -> 65504", f16_to_f32(0x7BFF), 65504.0f);
}

void test_q8_0_block_layout() {
  using dbinfer::tensor::dequant_row_q8_0;
  using dbinfer::tensor::f16_to_f32;

  std::byte block[34];
  const std::uint16_t d_bits = 0x3800;  // 0.5 in fp16
  std::memcpy(block, &d_bits, sizeof(d_bits));
  std::int8_t qs[32];
  for (int i = 0; i < 32; ++i) {
    qs[i] = static_cast<std::int8_t>(i - 16);
    block[2 + i] = static_cast<std::byte>(qs[i]);
  }

  float out[32];
  dequant_row_q8_0(block, 32, out);
  const float d = f16_to_f32(d_bits);

  for (int i = 0; i < 32; ++i) {
    char name[32];
    std::snprintf(name, sizeof name, "q8_0 layout [%d]", i);
    check_eq(name, out[i], d * static_cast<float>(qs[i]));
  }
}

void test_q8_0_roundtrip() {
  using dbinfer::tensor::dequant_row_q8_0;

  // amax = 127/16 makes d = 1/16 = 0.0625, exactly representable in fp16.
  const float xs[32] = {7.9375f, -7.9375f, 0.0f,   0.0625f, -0.0625f, 1.25f, -3.3f, 5.5f,
                        -6.1f,   2.2f,     -0.4f,  4.75f,   -5.9f,    0.9f,  -1.1f, 3.14f,
                        -2.72f,  6.6f,     -7.0f,  0.03f,   -0.5f,    7.1f,  -4.4f, 1.9f,
                        -6.8f,   3.7f,     -0.03f, 5.05f,   -2.5f,    0.75f, -7.5f, 4.2f};
  float amax = 0.0f;
  for (float v : xs) amax = std::max(amax, std::fabs(v));
  const float d = amax / 127.0f;

  const std::uint16_t d_bits = 0x2C00;  // 0.0625 in fp16, equals d exactly
  std::byte block[34];
  std::memcpy(block, &d_bits, sizeof(d_bits));
  for (int i = 0; i < 32; ++i) {
    float qf = std::round(xs[i] / d);
    qf = std::clamp(qf, -127.0f, 127.0f);
    block[2 + i] = static_cast<std::byte>(static_cast<std::int8_t>(qf));
  }

  float out[32];
  dequant_row_q8_0(block, 32, out);
  float max_err = 0.0f;
  for (int i = 0; i < 32; ++i) max_err = std::max(max_err, std::fabs(out[i] - xs[i]));
  check_le("q8_0 roundtrip", max_err, d * 0.5f + 1e-6f);
}

void test_q4_0_nibble_order() {
  using dbinfer::tensor::dequant_row_q4_0;
  using dbinfer::tensor::f16_to_f32;

  // low nibble of byte j is element j, high nibble is element j+16, so a
  // distinct value per nibble catches a swapped nibble order.
  std::byte block[18];
  const std::uint16_t d_bits = 0x3C00;  // 1.0 in fp16
  std::memcpy(block, &d_bits, sizeof(d_bits));
  for (int j = 0; j < 16; ++j) {
    const std::uint8_t lo = static_cast<std::uint8_t>(j);       // 0..15
    const std::uint8_t hi = static_cast<std::uint8_t>(15 - j);  // 15..0
    block[2 + j] = static_cast<std::byte>(lo | (hi << 4));
  }

  float out[32];
  dequant_row_q4_0(block, 32, out);
  const float d = f16_to_f32(d_bits);

  for (int j = 0; j < 16; ++j) {
    char name[32];
    std::snprintf(name, sizeof name, "q4_0 lo nibble [%d]", j);
    check_eq(name, out[j], d * static_cast<float>(j - 8));
    std::snprintf(name, sizeof name, "q4_0 hi nibble [%d]", j);
    check_eq(name, out[j + 16], d * static_cast<float>((15 - j) - 8));
  }
}

void test_q4_0_roundtrip() {
  using dbinfer::tensor::dequant_row_q4_0;

  // max magnitude 0.875 maps to level 7, so d = 0.875/7 = 0.125 exactly.
  const float xs[32] = {0.875f,  -0.875f, 0.0f,    0.125f, -0.125f, 0.25f,  -0.5f,   0.75f,
                        -0.625f, 0.375f,  -0.75f,  0.5f,   -0.375f, 0.625f, -0.25f,  0.125f,
                        -0.125f, 0.75f,   -0.875f, 0.0f,   -0.5f,   0.875f, -0.75f,  0.25f,
                        -0.625f, 0.375f,  -0.25f,  0.5f,   -0.5f,   0.625f, -0.875f, 0.75f};
  float amax = 0.0f;
  for (float v : xs) amax = std::max(amax, std::fabs(v));
  const float d = amax / 7.0f;

  const std::uint16_t d_bits = 0x3000;  // 0.125 in fp16, equals d exactly
  std::byte block[18];
  std::memcpy(block, &d_bits, sizeof(d_bits));
  for (int j = 0; j < 16; ++j) {
    auto pack = [&](float x) -> std::uint8_t {
      float qf = std::round(x / d) + 8.0f;
      qf = std::clamp(qf, 0.0f, 15.0f);
      return static_cast<std::uint8_t>(qf);
    };
    const std::uint8_t lo = pack(xs[j]);
    const std::uint8_t hi = pack(xs[j + 16]);
    block[2 + j] = static_cast<std::byte>(lo | (hi << 4));
  }

  float out[32];
  dequant_row_q4_0(block, 32, out);
  float max_err = 0.0f;
  for (int i = 0; i < 32; ++i) max_err = std::max(max_err, std::fabs(out[i] - xs[i]));
  check_le("q4_0 roundtrip", max_err, d + 1e-6f);
}

void test_q5_0_qh_bit_layout() {
  using dbinfer::tensor::dequant_row_q5_0;
  using dbinfer::tensor::f16_to_f32;

  // qh carries the fifth bit per element; the pattern sets it for some
  // elements in each half and clears it for others.
  const std::uint16_t d_bits = 0x3C00;  // 1.0 in fp16
  const std::uint32_t qh = 0xAAAA5555u;
  std::byte block[22];
  std::memcpy(block, &d_bits, sizeof(d_bits));
  std::memcpy(block + 2, &qh, sizeof(qh));
  for (int j = 0; j < 16; ++j) {
    const std::uint8_t lo = static_cast<std::uint8_t>(j);       // 0..15
    const std::uint8_t hi = static_cast<std::uint8_t>(15 - j);  // 15..0
    block[6 + j] = static_cast<std::byte>(lo | (hi << 4));
  }

  float out[32];
  dequant_row_q5_0(block, 32, out);
  const float d = f16_to_f32(d_bits);

  for (int j = 0; j < 16; ++j) {
    const int fifth_lo = static_cast<int>((qh >> j) & 1u);
    const int fifth_hi = static_cast<int>((qh >> (j + 16)) & 1u);
    const int v_lo = (j) + fifth_lo * 16 - 16;
    const int v_hi = (15 - j) + fifth_hi * 16 - 16;
    char name[32];
    std::snprintf(name, sizeof name, "q5_0 lo [%d]", j);
    check_eq(name, out[j], d * static_cast<float>(v_lo));
    std::snprintf(name, sizeof name, "q5_0 hi [%d]", j);
    check_eq(name, out[j + 16], d * static_cast<float>(v_hi));
  }
}

void test_q5_0_roundtrip() {
  using dbinfer::tensor::dequant_row_q5_0;

  // max magnitude 3.75 maps to level 31 (value 15), so d = 3.75/15 = 0.25.
  const float xs[32] = {3.75f, -4.0f, 0.0f,   0.25f, -0.25f, 0.5f,  -1.0f,  1.5f,
                        -3.5f, 0.75f, -2.0f,  2.25f, -0.75f, 1.25f, -1.5f,  3.0f,
                        -2.5f, 1.75f, -3.75f, 0.5f,  -1.25f, 3.25f, -2.75f, 0.25f,
                        -0.5f, 2.75f, -1.75f, 2.0f,  -3.0f,  1.0f,  -2.25f, 3.5f};
  const float d = 0.25f;
  const std::uint16_t d_bits = 0x3400;  // 0.25 in fp16, equals d exactly

  std::byte block[22];
  std::memcpy(block, &d_bits, sizeof(d_bits));
  std::uint32_t qh = 0;
  auto pack = [&](float x) -> std::uint8_t {
    float qf = std::round(x / d) + 16.0f;
    qf = std::clamp(qf, 0.0f, 31.0f);
    return static_cast<std::uint8_t>(qf);
  };
  for (int j = 0; j < 16; ++j) {
    const std::uint8_t lo = pack(xs[j]);
    const std::uint8_t hi = pack(xs[j + 16]);
    qh |= static_cast<std::uint32_t>((lo >> 4) & 1u) << j;
    qh |= static_cast<std::uint32_t>((hi >> 4) & 1u) << (j + 16);
    block[6 + j] = static_cast<std::byte>((lo & 0x0Fu) | ((hi & 0x0Fu) << 4));
  }
  std::memcpy(block + 2, &qh, sizeof(qh));

  float out[32];
  dequant_row_q5_0(block, 32, out);
  float max_err = 0.0f;
  for (int i = 0; i < 32; ++i) max_err = std::max(max_err, std::fabs(out[i] - xs[i]));
  check_le("q5_0 roundtrip", max_err, 0.5f * d + 1e-6f);
}

void test_q4_k_block_layout() {
  using dbinfer::tensor::dequant_row_q4_k;

  // scales chosen so both scale_min_k4 branches fire; the top two bits of
  // scales[0..3] feed the high nibble of sub-blocks 4..7. sc/m are the
  // independently hand-computed unpack results the dequant must reproduce.
  const int sc[8] = {1, 2, 3, 4, 17, 18, 19, 20};
  const int m[8] = {5, 6, 7, 8, 20, 21, 22, 23};
  const std::uint8_t packed_scales[12] = {0x41, 0x42, 0x43, 0x44, 0x45, 0x46,
                                          0x47, 0x48, 0x41, 0x52, 0x63, 0x74};

  const float d = 1.0f, dmin = 0.5f;
  std::uint8_t qn[256];
  for (int e = 0; e < 256; ++e) qn[e] = static_cast<std::uint8_t>(e % 16);

  std::byte blk[144];
  std::memset(blk, 0, sizeof blk);
  const std::uint16_t d_bits = 0x3C00, dmin_bits = 0x3800;
  std::memcpy(blk, &d_bits, 2);
  std::memcpy(blk + 2, &dmin_bits, 2);
  std::memcpy(blk + 4, packed_scales, 12);
  for (int e = 0; e < 256; ++e) {
    const int g = e / 64, w = e % 64, l = w % 32;
    auto cur = static_cast<std::uint8_t>(blk[16 + g * 32 + l]);
    if (w < 32)
      cur = static_cast<std::uint8_t>((cur & 0xF0) | (qn[e] & 0xF));
    else
      cur = static_cast<std::uint8_t>((cur & 0x0F) | ((qn[e] & 0xF) << 4));
    blk[16 + g * 32 + l] = static_cast<std::byte>(cur);
  }

  float out[256];
  dequant_row_q4_k(blk, 256, out);
  for (int e = 0; e < 256; ++e) {
    const int g = e / 64, w = e % 64;
    const int sb = 2 * g + (w >= 32 ? 1 : 0);
    const float want = d * sc[sb] * static_cast<float>(qn[e]) - dmin * m[sb];
    char name[32];
    std::snprintf(name, sizeof name, "q4_k [%d]", e);
    check_eq(name, out[e], want);
  }
}

void test_q4_k_roundtrip() {
  using dbinfer::tensor::dequant_row_q4_k;

  // same hand-computed sc/m table as test_q4_k_block_layout.
  const int sc[8] = {1, 2, 3, 4, 17, 18, 19, 20};
  const int m[8] = {5, 6, 7, 8, 20, 21, 22, 23};
  const std::uint8_t packed_scales[12] = {0x41, 0x42, 0x43, 0x44, 0x45, 0x46,
                                          0x47, 0x48, 0x41, 0x52, 0x63, 0x74};

  // quantize arbitrary values through the packed format; recovery is bounded
  // by half the largest sub-block step d*sc.
  const float d = 1.0f, dmin = 0.5f;
  float xs[256];
  float max_step = 0.0f;
  for (int e = 0; e < 256; ++e) {
    const int g = e / 64, w = e % 64;
    const int sb = 2 * g + (w >= 32 ? 1 : 0);
    const float step = d * static_cast<float>(sc[sb]);
    max_step = std::max(max_step, step);
    const float t = static_cast<float>(3 + (e % 10));
    xs[e] = step * (t + 0.3f) - dmin * static_cast<float>(m[sb]);
  }

  std::byte blk[144];
  std::memset(blk, 0, sizeof blk);
  const std::uint16_t d_bits = 0x3C00, dmin_bits = 0x3800;
  std::memcpy(blk, &d_bits, 2);
  std::memcpy(blk + 2, &dmin_bits, 2);
  std::memcpy(blk + 4, packed_scales, 12);
  for (int e = 0; e < 256; ++e) {
    const int g = e / 64, w = e % 64, l = w % 32;
    const int sb = 2 * g + (w >= 32 ? 1 : 0);
    const float step = d * static_cast<float>(sc[sb]);
    float qf = std::round((xs[e] + dmin * static_cast<float>(m[sb])) / step);
    qf = std::clamp(qf, 0.0f, 15.0f);
    const auto q = static_cast<std::uint8_t>(qf);
    auto cur = static_cast<std::uint8_t>(blk[16 + g * 32 + l]);
    if (w < 32)
      cur = static_cast<std::uint8_t>((cur & 0xF0) | (q & 0xF));
    else
      cur = static_cast<std::uint8_t>((cur & 0x0F) | ((q & 0xF) << 4));
    blk[16 + g * 32 + l] = static_cast<std::byte>(cur);
  }

  float out[256];
  dequant_row_q4_k(blk, 256, out);
  float max_err = 0.0f;
  for (int e = 0; e < 256; ++e) max_err = std::max(max_err, std::fabs(out[e] - xs[e]));
  check_le("q4_k roundtrip", max_err, 0.5f * max_step + 1e-4f);
}

void test_q6_k_block_layout() {
  using dbinfer::tensor::dequant_row_q6_k;

  const float d = 0.5f;
  std::int8_t scales[16];
  for (int i = 0; i < 16; ++i) scales[i] = static_cast<std::int8_t>(i - 8);
  int qn[256];
  for (int e = 0; e < 256; ++e) qn[e] = (e % 64) - 32;

  std::byte blk[210];
  std::memset(blk, 0, sizeof blk);

  for (int n = 0; n < 256; n += 128) {
    const int g = n / 128;
    const int ql_base = g * 64, qh_base = g * 32;
    for (int l = 0; l < 32; ++l) {
      auto put = [&](int e, int ql_off, int ql_shift, int qh_shift) {
        const int u = qn[e] + 32;
        auto qb = static_cast<std::uint8_t>(blk[ql_base + ql_off]);
        qb = static_cast<std::uint8_t>(qb | ((u & 0xF) << ql_shift));
        blk[ql_base + ql_off] = static_cast<std::byte>(qb);
        auto hb = static_cast<std::uint8_t>(blk[128 + qh_base + l]);
        hb = static_cast<std::uint8_t>(hb | (((u >> 4) & 3) << qh_shift));
        blk[128 + qh_base + l] = static_cast<std::byte>(hb);
      };
      put(n + l, l, 0, 0);
      put(n + l + 32, l + 32, 0, 2);
      put(n + l + 64, l, 4, 4);
      put(n + l + 96, l + 32, 4, 6);
    }
  }

  std::memcpy(blk + 192, scales, 16);
  const std::uint16_t d_bits = 0x3800;
  std::memcpy(blk + 208, &d_bits, 2);

  float out[256];
  dequant_row_q6_k(blk, 256, out);
  for (int e = 0; e < 256; ++e) {
    const int g = e / 128, w = e % 128, l = w % 32;
    const int scidx = g * 8 + l / 16 + (w / 32) * 2;
    const float want = d * static_cast<float>(scales[scidx]) * static_cast<float>(qn[e]);
    char name[32];
    std::snprintf(name, sizeof name, "q6_k [%d]", e);
    check_eq(name, out[e], want);
  }
}

void test_q6_k_roundtrip() {
  using dbinfer::tensor::dequant_row_q6_k;

  // avoid scale 0 so the inverse step is finite; err bounded by half d*sc.
  const float d = 0.5f;
  std::int8_t scales[16];
  for (int i = 0; i < 16; ++i) scales[i] = static_cast<std::int8_t>(i < 8 ? i - 8 : i - 7);

  float xs[256];
  float max_step = 0.0f;
  for (int e = 0; e < 256; ++e) {
    const int g = e / 128, w = e % 128;
    const int scidx = g * 8 + (w % 32) / 16 + (w / 32) * 2;
    const float step = d * std::fabs(static_cast<float>(scales[scidx]));
    max_step = std::max(max_step, step);
    const float t = static_cast<float>((e % 40) - 20);
    xs[e] = d * static_cast<float>(scales[scidx]) * (t + 0.3f);
  }

  std::byte blk[210];
  std::memset(blk, 0, sizeof blk);
  for (int n = 0; n < 256; n += 128) {
    const int g = n / 128;
    const int ql_base = g * 64, qh_base = g * 32;
    for (int l = 0; l < 32; ++l) {
      auto put = [&](int e, int quad, int ql_off, int ql_shift, int qh_shift) {
        const int scidx = g * 8 + l / 16 + quad * 2;
        const float step = d * static_cast<float>(scales[scidx]);
        float qf = std::round(xs[e] / step) + 32.0f;
        qf = std::clamp(qf, 0.0f, 63.0f);
        const int u = static_cast<int>(qf);
        auto qb = static_cast<std::uint8_t>(blk[ql_base + ql_off]);
        qb = static_cast<std::uint8_t>(qb | ((u & 0xF) << ql_shift));
        blk[ql_base + ql_off] = static_cast<std::byte>(qb);
        auto hb = static_cast<std::uint8_t>(blk[128 + qh_base + l]);
        hb = static_cast<std::uint8_t>(hb | (((u >> 4) & 3) << qh_shift));
        blk[128 + qh_base + l] = static_cast<std::byte>(hb);
      };
      put(n + l, 0, l, 0, 0);
      put(n + l + 32, 1, l + 32, 0, 2);
      put(n + l + 64, 2, l, 4, 4);
      put(n + l + 96, 3, l + 32, 4, 6);
    }
  }

  std::memcpy(blk + 192, scales, 16);
  const std::uint16_t d_bits = 0x3800;
  std::memcpy(blk + 208, &d_bits, 2);

  float out[256];
  dequant_row_q6_k(blk, 256, out);
  float max_err = 0.0f;
  for (int e = 0; e < 256; ++e) max_err = std::max(max_err, std::fabs(out[e] - xs[e]));
  check_le("q6_k roundtrip", max_err, 0.5f * max_step + 1e-3f);
}

void test_f32_to_f16_special_values() {
  using dbinfer::tensor::f32_to_f16;

  check_u16("f32->f16 +0", f32_to_f16(0.0f), 0x0000);
  check_u16("f32->f16 -0", f32_to_f16(-0.0f), 0x8000);
  check_u16("f32->f16 1.0", f32_to_f16(1.0f), 0x3C00);
  check_u16("f32->f16 -2.0", f32_to_f16(-2.0f), 0xC000);
  check_u16("f32->f16 65504", f32_to_f16(65504.0f), 0x7BFF);
  check_u16("f32->f16 overflow", f32_to_f16(70000.0f), 0x7C00);
  check_u16("f32->f16 2^-24", f32_to_f16(std::ldexp(1.0f, -24)), 0x0001);
  // 1 + 3/2048 sits halfway between mant 1 and mant 2, even wins.
  check_u16("f32->f16 tie even", f32_to_f16(1.00146484375f), 0x3C02);
  // 1 + 1/2048 sits halfway between mant 0 and mant 1, even wins.
  check_u16("f32->f16 tie down", f32_to_f16(1.00048828125f), 0x3C00);
}

void test_f16_roundtrip() {
  using dbinfer::tensor::f16_to_f32;
  using dbinfer::tensor::f32_to_f16;

  for (std::uint16_t bits :
       {static_cast<std::uint16_t>(0x0000), static_cast<std::uint16_t>(0x8000),
        static_cast<std::uint16_t>(0x3C00), static_cast<std::uint16_t>(0xC000),
        static_cast<std::uint16_t>(0x0001), static_cast<std::uint16_t>(0x7BFF),
        static_cast<std::uint16_t>(0x2C00), static_cast<std::uint16_t>(0x3800)}) {
    char name[32];
    std::snprintf(name, sizeof name, "f16 roundtrip 0x%04X", bits);
    check_u16(name, f32_to_f16(f16_to_f32(bits)), bits);
  }
}

void test_q8_0_quantize() {
  using dbinfer::tensor::f32_to_f16;
  using dbinfer::tensor::quantize_row_q8_0;

  {
    float xs[32];
    for (int i = 0; i < 32; ++i) xs[i] = static_cast<float>(i - 16) * 0.5f;
    xs[0] = 300.0f;  // amax makes d well above one, forces clamp on the rest.
    float amax = 0.0f;
    for (float v : xs) amax = std::max(amax, std::fabs(v));

    std::byte block[34];
    quantize_row_q8_0(xs, 32, block);
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, block, sizeof(d_bits));
    check_u16("q8 quant d fp16", d_bits, f32_to_f16(amax / 127.0f));
    check_eq("q8 quant amax->127", static_cast<float>(static_cast<std::int8_t>(block[2])), 127.0f);
  }

  {
    float xs[32];
    for (int i = 0; i < 32; ++i) xs[i] = -1.0f;
    xs[5] = 1.0f;
    std::byte block[34];
    quantize_row_q8_0(xs, 32, block);
    check_eq("q8 quant -clamp", static_cast<float>(static_cast<std::int8_t>(block[2])), -127.0f);
  }
}

}  // namespace

int main() {
  test_f16_to_f32_special_values();
  test_q8_0_block_layout();
  test_q8_0_roundtrip();
  test_q4_0_nibble_order();
  test_q4_0_roundtrip();
  test_q5_0_qh_bit_layout();
  test_q5_0_roundtrip();
  test_q4_k_block_layout();
  test_q4_k_roundtrip();
  test_q6_k_block_layout();
  test_q6_k_roundtrip();
  test_f32_to_f16_special_values();
  test_f16_roundtrip();
  test_q8_0_quantize();

  return dbinfer::test::summary();
}
