#include "gguf/gguf.hpp"
#include "model/model.hpp"
#include "tensor/cpu.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul_neon.hpp"
#include "tensor/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double seconds(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// calls/s of one quant matvec at m=1 on a synthetic weight of shape [out, in].
template <typename Fn>
double kernel_rate(Fn fn, const std::byte *W, const dbinfer::tensor::BlockQ8_0 *xq, float *y,
                   std::size_t out, std::size_t in) {
  constexpr int kIters = 200;
  std::vector<double> rates;
  for (int r = 0; r < 5; ++r) {
    const auto t0 = clock_type::now();
    for (int i = 0; i < kIters; ++i)
      fn(W, xq, y, out, in);
    rates.push_back(static_cast<double>(kIters) / seconds(t0, clock_type::now()));
  }
  return median(rates);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf> [prefill=512] [decode=256]\n", argv[0]);
    return 2;
  }
  const char *path = argv[1];
  const int n_pre = argc > 2 ? std::atoi(argv[2]) : 512;
  const int n_dec = argc > 3 ? std::atoi(argv[3]) : 256;

  auto loaded = dbinfer::gguf::load(path);
  if (!loaded) {
    std::fprintf(stderr, "error: load gguf: %s\n",
                 dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }
  auto mret = dbinfer::model::Model::load(*loaded);
  if (!mret) {
    std::fprintf(stderr, "error: load model: %s\n", dbinfer::gguf::to_string(mret.error()).c_str());
    return 1;
  }
  dbinfer::model::Model &model = *mret;
  const auto &cfg = model.config();

  const dbinfer::tensor::CpuFeatures &feat = dbinfer::tensor::cpu_features();
  std::printf("model %s  dotprod=%d i8mm=%d\n", path, feat.dotprod, feat.i8mm);

  const std::int32_t tok = 40;
  for (std::size_t nt = 1; nt <= dbinfer::tensor::p_core_count(); ++nt) {
    dbinfer::tensor::configure_thread_count(nt);
    std::vector<double> pre_rates;
    std::vector<double> dec_rates;
    for (int run = 0; run < 5; ++run) {
      std::int32_t pos = 0;
      const auto t0 = clock_type::now();
      for (int i = 0; i < n_pre; ++i)
        model.forward(tok, pos++);
      const auto t1 = clock_type::now();
      for (int i = 0; i < n_dec; ++i)
        model.forward(tok, pos++);
      const auto t2 = clock_type::now();
      pre_rates.push_back(static_cast<double>(n_pre) / seconds(t0, t1));
      dec_rates.push_back(static_cast<double>(n_dec) / seconds(t1, t2));
    }
    std::printf("threads %zu: prefill %.2f tok/s  decode %.2f tok/s\n", nt, median(pre_rates),
                median(dec_rates));
  }
  dbinfer::tensor::configure_thread_count(1);

  // isolate the quant kernel at m=1 on the ffn shape, sdot against i8mm.
  const std::size_t in = cfg.embedding_length;
  const std::size_t out = cfg.ffn_length;
  const std::size_t nb = in / dbinfer::tensor::kBlockSize;

  std::mt19937 rng(0xBEEFu);
  std::uniform_int_distribution<int> q8(-127, 127);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(0.001f, 0.5f);
  std::uniform_real_distribution<float> act(-4.0f, 4.0f);

  std::vector<float> x(in);
  for (float &v : x)
    v = act(rng);
  std::vector<dbinfer::tensor::BlockQ8_0> xq(nb);
  dbinfer::tensor::quantize_row_q8_0(x.data(), in, reinterpret_cast<std::byte *>(xq.data()));
  std::vector<float> y(out);

  std::vector<dbinfer::tensor::BlockQ8_0> w8(out * nb);
  for (auto &blk : w8) {
    blk.d = dbinfer::tensor::f32_to_f16(scale(rng));
    for (std::size_t i = 0; i < dbinfer::tensor::kBlockSize; ++i)
      blk.qs[i] = static_cast<std::int8_t>(q8(rng));
  }
  const std::byte *w8b = reinterpret_cast<const std::byte *>(w8.data());

  std::vector<std::byte> w4(out * nb * sizeof(dbinfer::tensor::BlockQ4_0));
  for (std::size_t k = 0; k < out * nb; ++k) {
    std::byte *blk = w4.data() + k * sizeof(dbinfer::tensor::BlockQ4_0);
    const std::uint16_t d = dbinfer::tensor::f32_to_f16(scale(rng));
    std::memcpy(blk, &d, sizeof(d));
    for (std::size_t j = 0; j < 16; ++j)
      blk[2 + j] = static_cast<std::byte>(byte(rng));
  }

  std::printf("kernel microbench [out=%zu in=%zu] m=1:\n", out, in);
  std::printf("  q8 scalar %.0f calls/s\n",
              kernel_rate(dbinfer::tensor::matvec_q8_0_scalar, w8b, xq.data(), y.data(), out, in));
  std::printf("  q4 scalar %.0f calls/s\n", kernel_rate(dbinfer::tensor::matvec_q4_0_scalar,
                                                        w4.data(), xq.data(), y.data(), out, in));
  std::printf("  q8 sdot %.0f calls/s\n",
              kernel_rate(dbinfer::tensor::matvec_q8_0_sdot, w8b, xq.data(), y.data(), out, in));
  std::printf("  q4 sdot %.0f calls/s\n", kernel_rate(dbinfer::tensor::matvec_q4_0_sdot, w4.data(),
                                                      xq.data(), y.data(), out, in));
  if (feat.i8mm) {
    std::printf("  q8 i8mm %.0f calls/s\n",
                kernel_rate(dbinfer::tensor::matvec_q8_0_i8mm, w8b, xq.data(), y.data(), out, in));
    std::printf("  q4 i8mm %.0f calls/s\n", kernel_rate(dbinfer::tensor::matvec_q4_0_i8mm,
                                                        w4.data(), xq.data(), y.data(), out, in));
  }
  return 0;
}
