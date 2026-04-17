#include "tensor/matmul.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

// matmul_accel reorders the reduction through cblas_sgemm, so it matches the
// scalar matmul only within GEMM tolerance. this pins the correctness bound
// before the wrapper feeds any forward pass, and records a prefill timing.

namespace {

int g_failures = 0;

} // namespace

int main() {
  constexpr std::size_t m = 512, out = 512, in = 896;

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> A(m * in), W(out * in);
  for (float &v : A)
    v = dist(rng);
  for (float &v : W)
    v = dist(rng);

  std::vector<float> c_scalar(m * out), c_accel(m * out);
  dbinfer::tensor::matmul(A.data(), W.data(), c_scalar.data(), m, out, in);

  const dbinfer::tensor::QuantMatrix w{reinterpret_cast<const std::byte *>(W.data()),
                                       dbinfer::gguf::GgmlType::F32};
  dbinfer::tensor::matmul_accel(A.data(), w, c_accel.data(), m, out, in);

  float max_err = 0.0f;
  for (std::size_t i = 0; i < m * out; ++i)
    max_err = std::max(max_err, std::fabs(c_accel[i] - c_scalar[i]));
  if (max_err <= 1e-3f) {
    std::printf("PASS accel vs scalar max_err %.6g <= 1e-3\n", static_cast<double>(max_err));
  } else {
    std::printf("FAIL accel vs scalar max_err %.6g > 1e-3\n", static_cast<double>(max_err));
    ++g_failures;
  }

  using clock = std::chrono::steady_clock;
  constexpr int iters = 20;
  auto t0 = clock::now();
  for (int t = 0; t < iters; ++t)
    dbinfer::tensor::matmul(A.data(), W.data(), c_scalar.data(), m, out, in);
  auto t1 = clock::now();
  for (int t = 0; t < iters; ++t)
    dbinfer::tensor::matmul_accel(A.data(), w, c_accel.data(), m, out, in);
  auto t2 = clock::now();
  const double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  const double accel_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
  std::printf("BENCH matmul [%zu x %zu x %zu] scalar %.3f ms  accel %.3f ms  speedup %.2fx\n", m,
              out, in, scalar_ms, accel_ms, scalar_ms / accel_ms);

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
