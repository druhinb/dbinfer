#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "tensor/dequant.hpp"
#include "tensor/matmul.hpp"

// the int8-dot matvec is the reference activation-quant kernel. two properties
// pin it: it reproduces the block-scaled integer dot bit for bit, and it stays
// within the exact per-element quant error of the fp32 dot on dequant weights.

namespace {

using dbinfer::tensor::BlockQ8_0;
using dbinfer::tensor::f16_to_f32;
using dbinfer::tensor::f32_to_f16;
using dbinfer::tensor::kBlockSize;

int g_failures = 0;

void check_eq(const char* what, float got, float want) {
  if (got == want) {
    std::printf("PASS %-28s %.9g\n", what, static_cast<double>(got));
  } else {
    std::printf("FAIL %-28s got %.9g want %.9g\n", what, static_cast<double>(got),
                static_cast<double>(want));
    ++g_failures;
  }
}

void check_le(const char* what, float got, float bound) {
  if (got <= bound) {
    std::printf("PASS %-28s %.6g <= %.6g\n", what, static_cast<double>(got),
                static_cast<double>(bound));
  } else {
    std::printf("FAIL %-28s %.6g > %.6g\n", what, static_cast<double>(got),
                static_cast<double>(bound));
    ++g_failures;
  }
}

// block-scaled integer dot recomputed independently, the exact value matvec
// must produce (same order, same per-block fp scale).
float reference_dot(const std::int8_t* wq, const std::uint16_t* dw, const BlockQ8_0* xq,
                    std::size_t nblocks) {
  float acc = 0.0f;
  for (std::size_t b = 0; b < nblocks; ++b) {
    std::int32_t sumi = 0;
    for (std::size_t i = 0; i < kBlockSize; ++i)
      sumi += static_cast<std::int32_t>(wq[b * kBlockSize + i]) *
              static_cast<std::int32_t>(xq[b].qs[i]);
    acc += f16_to_f32(dw[b]) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
  }
  return acc;
}

}  // namespace

int main() {
  using dbinfer::tensor::matvec_q4_0;
  using dbinfer::tensor::matvec_q8_0;
  using dbinfer::tensor::quantize_row_q8_0;

  std::mt19937 rng(0xC0FFEEu);
  std::uniform_real_distribution<float> act(-4.0f, 4.0f);
  std::uniform_real_distribution<float> scale(0.001f, 0.5f);
  std::uniform_int_distribution<int> q8(-127, 127);
  std::uniform_int_distribution<int> nib(0, 15);

  const std::size_t nblocks = 8;
  const std::size_t in = nblocks * kBlockSize;
  const std::size_t out = 4;

  float max_q8_exact = 0.0f;
  float max_q4_exact = 0.0f;
  float max_q8_noise = 0.0f;
  float max_q4_noise = 0.0f;

  for (int trial = 0; trial < 200; ++trial) {
    std::vector<float> x(in);
    for (float& v : x) v = act(rng);

    std::vector<BlockQ8_0> xq(nblocks);
    quantize_row_q8_0(x.data(), in, reinterpret_cast<std::byte*>(xq.data()));

    // per-element activation quant error, the exact bound on the fp32 gap.
    std::vector<float> xdeq(in);
    for (std::size_t b = 0; b < nblocks; ++b)
      for (std::size_t i = 0; i < kBlockSize; ++i)
        xdeq[b * kBlockSize + i] = f16_to_f32(xq[b].d) * static_cast<float>(xq[b].qs[i]);

    {
      std::vector<BlockQ8_0> w(out * nblocks);
      std::vector<std::int8_t> wq(out * in);
      std::vector<std::uint16_t> dw(out * nblocks);
      for (std::size_t o = 0; o < out; ++o)
        for (std::size_t b = 0; b < nblocks; ++b) {
          const std::uint16_t d = f32_to_f16(scale(rng));
          w[o * nblocks + b].d = d;
          dw[o * nblocks + b] = d;
          for (std::size_t i = 0; i < kBlockSize; ++i) {
            const std::int8_t q = static_cast<std::int8_t>(q8(rng));
            w[o * nblocks + b].qs[i] = q;
            wq[o * in + b * kBlockSize + i] = q;
          }
        }

      std::vector<float> y(out);
      matvec_q8_0(reinterpret_cast<const std::byte*>(w.data()), x.data(), y.data(), out, in);

      for (std::size_t o = 0; o < out; ++o) {
        const float ref =
            reference_dot(wq.data() + o * in, dw.data() + o * nblocks, xq.data(), nblocks);
        max_q8_exact = std::max(max_q8_exact, std::fabs(y[o] - ref));

        float fp32 = 0.0f;
        float bound = 0.0f;
        for (std::size_t b = 0; b < nblocks; ++b)
          for (std::size_t i = 0; i < kBlockSize; ++i) {
            const std::size_t idx = b * kBlockSize + i;
            const float wf = f16_to_f32(dw[o * nblocks + b]) * static_cast<float>(wq[o * in + idx]);
            fp32 += wf * x[idx];
            bound += std::fabs(wf * (x[idx] - xdeq[idx]));
          }
        max_q8_noise = std::max(max_q8_noise,
                                std::fabs(y[o] - fp32) - (bound + 1e-3f + 1e-4f * std::fabs(fp32)));
      }
    }

    {
      std::vector<dbinfer::tensor::BlockQ4_0> w(out * nblocks);
      std::vector<std::int8_t> wq(out * in);
      std::vector<std::uint16_t> dw(out * nblocks);
      for (std::size_t o = 0; o < out; ++o)
        for (std::size_t b = 0; b < nblocks; ++b) {
          const std::uint16_t d = f32_to_f16(scale(rng));
          w[o * nblocks + b].d = d;
          dw[o * nblocks + b] = d;
          for (std::size_t j = 0; j < 16; ++j) {
            const std::uint8_t lo = static_cast<std::uint8_t>(nib(rng));
            const std::uint8_t hi = static_cast<std::uint8_t>(nib(rng));
            w[o * nblocks + b].qs[j] = static_cast<std::uint8_t>(lo | (hi << 4));
            wq[o * in + b * kBlockSize + j] = static_cast<std::int8_t>(lo) - 8;
            wq[o * in + b * kBlockSize + j + 16] = static_cast<std::int8_t>(hi) - 8;
          }
        }

      std::vector<float> y(out);
      matvec_q4_0(reinterpret_cast<const std::byte*>(w.data()), x.data(), y.data(), out, in);

      for (std::size_t o = 0; o < out; ++o) {
        const float ref =
            reference_dot(wq.data() + o * in, dw.data() + o * nblocks, xq.data(), nblocks);
        max_q4_exact = std::max(max_q4_exact, std::fabs(y[o] - ref));

        float fp32 = 0.0f;
        float bound = 0.0f;
        for (std::size_t b = 0; b < nblocks; ++b)
          for (std::size_t i = 0; i < kBlockSize; ++i) {
            const std::size_t idx = b * kBlockSize + i;
            const float wf = f16_to_f32(dw[o * nblocks + b]) * static_cast<float>(wq[o * in + idx]);
            fp32 += wf * x[idx];
            bound += std::fabs(wf * (x[idx] - xdeq[idx]));
          }
        max_q4_noise = std::max(max_q4_noise,
                                std::fabs(y[o] - fp32) - (bound + 1e-3f + 1e-4f * std::fabs(fp32)));
      }
    }
  }

  check_eq("q8 int8-dot exact", max_q8_exact, 0.0f);
  check_eq("q4 int8-dot exact", max_q4_exact, 0.0f);
  check_le("q8 vs fp32 within noise", max_q8_noise, 0.0f);
  check_le("q4 vs fp32 within noise", max_q4_noise, 0.0f);

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
