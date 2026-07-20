// tolerance gate for the simdgroup_matrix GEMM used on the m>1 batched paths.
// the 8x8 tile reduction reorders vs the CPU sequential matmul, so it matches
// tensor::matmul_quant's F16 path within GEMM tolerance (atol 1e-3), not
// bitwise. skips cleanly with no Metal device or a driver lacking the op.

#include "backend/backend.hpp"
#include "backend/metal_backend.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

struct Lcg {
  std::uint64_t s;
  float next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::uint32_t hi = static_cast<std::uint32_t>(s >> 32);
    return static_cast<float>(hi) / 2147483648.0f - 1.0f;
  }
};

double run_shape(dbinfer::backend::Backend &metal, std::size_t m, std::size_t out, std::size_t in) {
  Lcg rng{0x5EED ^ (m * 131 + out * 17 + in)};
  std::vector<std::uint16_t> W(out * in);
  std::vector<float> A(m * in);
  for (auto &w : W)
    w = dbinfer::tensor::f32_to_f16(rng.next());
  for (auto &a : A)
    a = rng.next();

  std::vector<float> c_cpu(m * out, 0.0f);
  std::vector<float> c_gpu(m * out, 0.0f);

  const dbinfer::tensor::QuantMatrix wm{reinterpret_cast<const std::byte *>(W.data()),
                                        dbinfer::gguf::GgmlType::F16};
  dbinfer::tensor::matmul_quant(wm, A.data(), c_cpu.data(), m, out, in);

  auto r = metal.mul_mat_f16_gemm(W.data(), A.data(), c_gpu.data(), m, out, in);
  if (!r) {
    std::printf("  gemm error: %s\n", r.error().message.c_str());
    ++g_failures;
    return 0.0;
  }

  double max_err = 0.0;
  for (std::size_t i = 0; i < m * out; ++i) {
    const double d = static_cast<double>(c_gpu[i]) - static_cast<double>(c_cpu[i]);
    max_err = std::max(max_err, d < 0 ? -d : d);
  }
  std::printf("  m=%zu out=%zu in=%zu max_err=%.3e\n", m, out, in, max_err);
  return max_err;
}

} // namespace

int main() {
  dbinfer::backend::Backend *metal = dbinfer::backend::metal_backend();
  if (metal == nullptr) {
    std::printf("SKIP no Metal device available\n");
    return 0;
  }
  // probe availability: a 8x8 call errors when the driver lacks simdgroup_matrix.
  {
    std::vector<std::uint16_t> w(64, 0);
    std::vector<float> a(8, 0.0f), c(8, 0.0f);
    if (!metal->mul_mat_f16_gemm(w.data(), a.data(), c.data(), 1, 8, 8)) {
      std::printf("SKIP simdgroup_matrix gemm unavailable on this device\n");
      return 0;
    }
  }

  // qwen2.5-0.5b prefill widths plus non-multiple-of-8 m to exercise padding.
  const std::size_t ms[] = {1, 5, 8, 64};
  const std::size_t shapes[][2] = {{896, 896}, {4864, 896}, {896, 4864}};
  double worst = 0.0;
  for (std::size_t m : ms)
    for (const auto &s : shapes)
      worst = std::max(worst, run_shape(*metal, m, s[0], s[1]));

  std::printf("worst max_err across shapes=%.3e (gate atol 1e-3)\n", worst);
  const bool ok = worst <= 1e-3;
  std::printf("%s simdgroup_matrix gemm within atol 1e-3 of CPU matmul\n", ok ? "PASS" : "FAIL");
  if (!ok)
    ++g_failures;

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
