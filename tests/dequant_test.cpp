#include "tensor/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;

void check_eq(const char *what, float got, float want) {
  if (got == want) {
    std::printf("PASS %-24s %.9g\n", what, static_cast<double>(got));
  } else {
    std::printf("FAIL %-24s got %.9g want %.9g\n", what, static_cast<double>(got),
                static_cast<double>(want));
    ++g_failures;
  }
}

void check_le(const char *what, float got, float bound) {
  if (got <= bound) {
    std::printf("PASS %-24s %.9g <= %.9g\n", what, static_cast<double>(got),
                static_cast<double>(bound));
  } else {
    std::printf("FAIL %-24s %.9g > %.9g\n", what, static_cast<double>(got),
                static_cast<double>(bound));
    ++g_failures;
  }
}

void check_u16(const char *what, std::uint16_t got, std::uint16_t want) {
  if (got == want) {
    std::printf("PASS %-24s 0x%04X\n", what, got);
  } else {
    std::printf("FAIL %-24s got 0x%04X want 0x%04X\n", what, got, want);
    ++g_failures;
  }
}

bool signbit_of(float v) { return std::signbit(v); }

} // namespace

int main() {
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
    if (std::isinf(v) && v > 0.0f)
      std::printf("PASS %-24s +inf\n", "0x7C00 -> +inf");
    else {
      std::printf("FAIL 0x7C00 -> not +inf (%.9g)\n", static_cast<double>(v));
      ++g_failures;
    }
  }
  {
    float v = f16_to_f32(0xFC00);
    if (std::isinf(v) && v < 0.0f)
      std::printf("PASS %-24s -inf\n", "0xFC00 -> -inf");
    else {
      std::printf("FAIL 0xFC00 -> not -inf (%.9g)\n", static_cast<double>(v));
      ++g_failures;
    }
  }
  {
    float v = f16_to_f32(0x7E00); // a NaN pattern (exp=31, mant!=0)
    if (std::isnan(v))
      std::printf("PASS %-24s NaN\n", "0x7E00 -> NaN");
    else {
      std::printf("FAIL 0x7E00 -> not NaN (%.9g)\n", static_cast<double>(v));
      ++g_failures;
    }
  }

  check_eq("0x0001 -> 2^-24", f16_to_f32(0x0001), std::ldexp(1.0f, -24));
  check_eq("0x7BFF -> 65504", f16_to_f32(0x7BFF), 65504.0f);

  using dbinfer::tensor::dequant_row_q8_0;

  {
    std::byte block[34];
    const std::uint16_t d_bits = 0x3800; // 0.5 in fp16
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

  {
    // amax = 127/16 makes d = 1/16 = 0.0625, exactly representable in fp16.
    const float xs[32] = {7.9375f, -7.9375f, 0.0f,   0.0625f, -0.0625f, 1.25f, -3.3f, 5.5f,
                          -6.1f,   2.2f,     -0.4f,  4.75f,   -5.9f,    0.9f,  -1.1f, 3.14f,
                          -2.72f,  6.6f,     -7.0f,  0.03f,   -0.5f,    7.1f,  -4.4f, 1.9f,
                          -6.8f,   3.7f,     -0.03f, 5.05f,   -2.5f,    0.75f, -7.5f, 4.2f};
    float amax = 0.0f;
    for (float v : xs)
      amax = std::max(amax, std::fabs(v));
    const float d = amax / 127.0f;
    const std::uint16_t d_bits = 0x2C00; // 0.0625 in fp16, equals d exactly
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
    for (int i = 0; i < 32; ++i)
      max_err = std::max(max_err, std::fabs(out[i] - xs[i]));
    check_le("q8_0 roundtrip", max_err, d * 0.5f + 1e-6f);
  }

  using dbinfer::tensor::dequant_row_q4_0;

  {
    // low nibble of byte j is element j, high nibble is element j+16, so a
    // distinct value per nibble catches a swapped nibble order.
    std::byte block[18];
    const std::uint16_t d_bits = 0x3C00; // 1.0 in fp16
    std::memcpy(block, &d_bits, sizeof(d_bits));
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t lo = static_cast<std::uint8_t>(j);      // 0..15
      const std::uint8_t hi = static_cast<std::uint8_t>(15 - j); // 15..0
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

  {
    // max magnitude 0.875 maps to level 7, so d = 0.875/7 = 0.125 exactly.
    const float xs[32] = {0.875f,  -0.875f, 0.0f,    0.125f, -0.125f, 0.25f,  -0.5f,   0.75f,
                          -0.625f, 0.375f,  -0.75f,  0.5f,   -0.375f, 0.625f, -0.25f,  0.125f,
                          -0.125f, 0.75f,   -0.875f, 0.0f,   -0.5f,   0.875f, -0.75f,  0.25f,
                          -0.625f, 0.375f,  -0.25f,  0.5f,   -0.5f,   0.625f, -0.875f, 0.75f};
    float amax = 0.0f;
    for (float v : xs)
      amax = std::max(amax, std::fabs(v));
    const float d = amax / 7.0f;
    const std::uint16_t d_bits = 0x3000; // 0.125 in fp16, equals d exactly
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
    for (int i = 0; i < 32; ++i)
      max_err = std::max(max_err, std::fabs(out[i] - xs[i]));
    check_le("q4_0 roundtrip", max_err, d + 1e-6f);
  }

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

  for (std::uint16_t bits : {std::uint16_t(0x0000), std::uint16_t(0x8000), std::uint16_t(0x3C00),
                             std::uint16_t(0xC000), std::uint16_t(0x0001), std::uint16_t(0x7BFF),
                             std::uint16_t(0x2C00), std::uint16_t(0x3800)}) {
    char name[32];
    std::snprintf(name, sizeof name, "f16 roundtrip 0x%04X", bits);
    check_u16(name, f32_to_f16(f16_to_f32(bits)), bits);
  }

  using dbinfer::tensor::quantize_row_q8_0;

  {
    float xs[32];
    for (int i = 0; i < 32; ++i)
      xs[i] = static_cast<float>(i - 16) * 0.5f;
    xs[0] = 300.0f; // amax makes d well above one, forces clamp on the rest.
    float amax = 0.0f;
    for (float v : xs)
      amax = std::max(amax, std::fabs(v));
    std::byte block[34];
    quantize_row_q8_0(xs, 32, block);
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, block, sizeof(d_bits));
    check_u16("q8 quant d fp16", d_bits, f32_to_f16(amax / 127.0f));
    check_eq("q8 quant amax->127", static_cast<float>(static_cast<std::int8_t>(block[2])), 127.0f);
  }

  {
    float xs[32];
    for (int i = 0; i < 32; ++i)
      xs[i] = -1.0f;
    xs[5] = 1.0f;
    std::byte block[34];
    quantize_row_q8_0(xs, 32, block);
    check_eq("q8 quant -clamp", static_cast<float>(static_cast<std::int8_t>(block[2])), -127.0f);
  }

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
