#include "tensor/cpu.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul_neon.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// property test: the sdot kernels reproduce the scalar reference within atol
// 1e-6 over random blocks plus zero, saturated, and denormal edge cases. runs
// before the sdot path is trusted in the forward pass.

namespace {

using dbinfer::tensor::BlockQ8_0;
using dbinfer::tensor::f32_to_f16;
using dbinfer::tensor::kBlockSize;

int g_failures = 0;
constexpr float kAtol = 1e-6f;

float max_diff(const std::vector<float> &a, const std::vector<float> &b) {
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

void check_q8(const char *what, const std::vector<std::byte> &w, const std::vector<BlockQ8_0> &xq,
              std::size_t out, std::size_t in) {
  std::vector<float> ref(out);
  std::vector<float> got(out);
  dbinfer::tensor::matvec_q8_0_scalar(w.data(), xq.data(), ref.data(), out, in);
  dbinfer::tensor::matvec_q8_0_sdot(w.data(), xq.data(), got.data(), out, in);
  const float d = max_diff(ref, got);
  if (d > kAtol) {
    std::printf("FAIL %-24s max_diff %.3e\n", what, static_cast<double>(d));
    ++g_failures;
  }
}

void check_q4(const char *what, const std::vector<std::byte> &w, const std::vector<BlockQ8_0> &xq,
              std::size_t out, std::size_t in) {
  std::vector<float> ref(out);
  std::vector<float> got(out);
  dbinfer::tensor::matvec_q4_0_scalar(w.data(), xq.data(), ref.data(), out, in);
  dbinfer::tensor::matvec_q4_0_sdot(w.data(), xq.data(), got.data(), out, in);
  const float d = max_diff(ref, got);
  if (d > kAtol) {
    std::printf("FAIL %-24s max_diff %.3e\n", what, static_cast<double>(d));
    ++g_failures;
  }
}

} // namespace

int main() {
  if (!dbinfer::tensor::cpu_features().dotprod) {
    std::printf("SKIP dotprod unavailable on this host\n");
    std::printf("---\n0 checks failed\n");
    return 0;
  }

  std::mt19937 rng(0x5EEDu);
  std::uniform_real_distribution<float> scale(0.001f, 0.5f);
  std::uniform_int_distribution<int> q8(-127, 127);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> nb_dist(1, 8);
  std::uniform_int_distribution<int> out_dist(1, 4);

  std::size_t q8_blocks = 0;
  std::size_t q4_blocks = 0;

  for (int trial = 0; trial < 3000; ++trial) {
    const std::size_t nblocks = static_cast<std::size_t>(nb_dist(rng));
    const std::size_t out = static_cast<std::size_t>(out_dist(rng));
    const std::size_t in = nblocks * kBlockSize;

    std::vector<BlockQ8_0> xq(nblocks);
    for (auto &blk : xq) {
      blk.d = f32_to_f16(scale(rng));
      for (std::size_t i = 0; i < kBlockSize; ++i)
        blk.qs[i] = static_cast<std::int8_t>(q8(rng));
    }

    {
      std::vector<BlockQ8_0> w(out * nblocks);
      for (auto &blk : w) {
        blk.d = f32_to_f16(scale(rng));
        for (std::size_t i = 0; i < kBlockSize; ++i)
          blk.qs[i] = static_cast<std::int8_t>(q8(rng));
      }
      std::vector<std::byte> wb(out * nblocks * sizeof(BlockQ8_0));
      std::memcpy(wb.data(), w.data(), wb.size());
      check_q8("q8 random", wb, xq, out, in);
      q8_blocks += out * nblocks;
    }

    {
      std::vector<std::byte> wb(out * nblocks * sizeof(dbinfer::tensor::BlockQ4_0));
      for (std::size_t k = 0; k < out * nblocks; ++k) {
        std::byte *blk = wb.data() + k * sizeof(dbinfer::tensor::BlockQ4_0);
        const std::uint16_t d = f32_to_f16(scale(rng));
        std::memcpy(blk, &d, sizeof(d));
        for (std::size_t j = 0; j < 16; ++j)
          blk[2 + j] = static_cast<std::byte>(byte(rng));
      }
      check_q4("q4 random", wb, xq, out, in);
      q4_blocks += out * nblocks;
    }
  }

  // edge: all-zero block, saturated +-127, denormal fp16 scales.
  {
    std::vector<BlockQ8_0> xq(1);
    xq[0].d = 0;
    std::memset(xq[0].qs, 0, sizeof(xq[0].qs));
    std::vector<BlockQ8_0> w(1);
    w[0].d = f32_to_f16(0.25f);
    for (std::size_t i = 0; i < kBlockSize; ++i)
      w[0].qs[i] = static_cast<std::int8_t>(q8(rng));
    std::vector<std::byte> wb(sizeof(BlockQ8_0));
    std::memcpy(wb.data(), w.data(), wb.size());
    check_q8("q8 zero activation", wb, xq, 1, kBlockSize);

    std::vector<std::byte> w4(sizeof(dbinfer::tensor::BlockQ4_0));
    const std::uint16_t d = f32_to_f16(0.25f);
    std::memcpy(w4.data(), &d, sizeof(d));
    for (std::size_t j = 0; j < 16; ++j)
      w4[2 + j] = static_cast<std::byte>(byte(rng));
    check_q4("q4 zero activation", w4, xq, 1, kBlockSize);
  }

  {
    std::vector<BlockQ8_0> xq(1);
    xq[0].d = f32_to_f16(0.5f);
    for (std::size_t i = 0; i < kBlockSize; ++i)
      xq[0].qs[i] = (i & 1u) ? std::int8_t(127) : std::int8_t(-127);
    std::vector<BlockQ8_0> w(1);
    w[0].d = f32_to_f16(0.5f);
    for (std::size_t i = 0; i < kBlockSize; ++i)
      w[0].qs[i] = (i & 1u) ? std::int8_t(-127) : std::int8_t(127);
    std::vector<std::byte> wb(sizeof(BlockQ8_0));
    std::memcpy(wb.data(), w.data(), wb.size());
    check_q8("q8 saturated", wb, xq, 1, kBlockSize);

    std::vector<std::byte> w4(sizeof(dbinfer::tensor::BlockQ4_0));
    const std::uint16_t d = f32_to_f16(0.5f);
    std::memcpy(w4.data(), &d, sizeof(d));
    for (std::size_t j = 0; j < 16; ++j)
      w4[2 + j] = static_cast<std::byte>(0x0F); // both nibbles level 15
    check_q4("q4 saturated", w4, xq, 1, kBlockSize);
  }

  {
    std::vector<BlockQ8_0> xq(1);
    xq[0].d = 0x0001; // fp16 denormal 2^-24
    for (std::size_t i = 0; i < kBlockSize; ++i)
      xq[0].qs[i] = static_cast<std::int8_t>(q8(rng));
    std::vector<BlockQ8_0> w(1);
    w[0].d = 0x0002; // fp16 denormal 2^-23
    for (std::size_t i = 0; i < kBlockSize; ++i)
      w[0].qs[i] = static_cast<std::int8_t>(q8(rng));
    std::vector<std::byte> wb(sizeof(BlockQ8_0));
    std::memcpy(wb.data(), w.data(), wb.size());
    check_q8("q8 denormal scale", wb, xq, 1, kBlockSize);

    std::vector<std::byte> w4(sizeof(dbinfer::tensor::BlockQ4_0));
    const std::uint16_t d = 0x0002;
    std::memcpy(w4.data(), &d, sizeof(d));
    for (std::size_t j = 0; j < 16; ++j)
      w4[2 + j] = static_cast<std::byte>(byte(rng));
    check_q4("q4 denormal scale", w4, xq, 1, kBlockSize);
  }

  std::printf("q8 blocks tested %zu, q4 blocks tested %zu\n", q8_blocks, q4_blocks);
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
