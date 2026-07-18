#include "tensor/dequant.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

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

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
