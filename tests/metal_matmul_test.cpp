// per-kernel parity: Metal mul_mat_f16 vs the CPU matmul_quant F16 path.
//
// the GPU runs one simdgroup per row and reduces the in-dimension with simd_sum,
// so its summation order differs from the CPU sequential fma. the tolerance is a
// combined atol 1e-4 plus rtol 1e-4, the reordered fp32 sum reaching ~1e-5
// relative at in=4864. skips cleanly when no Metal device is present.

#include "backend/metal_backend.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

struct Lcg {
  std::uint64_t s;
  float next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint32_t hi = static_cast<std::uint32_t>(s >> 32);
    return static_cast<float>(hi) / 2147483648.0f - 1.0f;
  }
};

void check(bool ok, const char *what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok)
    ++g_failures;
}

double run_shape(dbinfer::backend::Backend &metal, std::size_t m, std::size_t out, std::size_t in) {
  Lcg rng{0xC0FFEE ^ (m * 131 + out * 17 + in)};
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

  auto r = metal.mul_mat_f16(W.data(), A.data(), c_gpu.data(), m, out, in);
  if (!r) {
    std::printf("  metal error: %s\n", r.error().message.c_str());
    ++g_failures;
    return 0.0;
  }

  double max_err = 0.0;
  double worst_ratio = 0.0;
  for (std::size_t i = 0; i < m * out; ++i) {
    const double d = static_cast<double>(c_gpu[i]) - static_cast<double>(c_cpu[i]);
    const double ad = d < 0 ? -d : d;
    max_err = std::max(max_err, ad);
    const double bound = 1e-4 + 1e-4 * std::abs(static_cast<double>(c_cpu[i]));
    worst_ratio = std::max(worst_ratio, ad / bound);
  }
  std::printf("  m=%zu out=%zu in=%zu max_err=%.3e ratio=%.2f\n", m, out, in, max_err, worst_ratio);
  return worst_ratio;
}

void test_zero_copy() {
  // DBMF aligns tensor data to 16384 so Metal wraps it no-copy. an unaligned
  // pointer must report false and take the copy path.
  constexpr std::size_t kAlign = 16384;
  constexpr std::size_t kBytes = 896 * 2;
  void *aligned = nullptr;
  if (posix_memalign(&aligned, kAlign, kAlign) != 0 || aligned == nullptr) {
    check(false, "posix_memalign for zero-copy probe");
    return;
  }
  const bool ok_aligned = dbinfer::backend::metal_can_wrap_nocopy(aligned, kBytes);
  auto *bytes = static_cast<std::byte *>(aligned);
  const bool ok_unaligned = dbinfer::backend::metal_can_wrap_nocopy(bytes + 32, kBytes);
  std::free(aligned);
  std::printf("  nocopy aligned=%d unaligned=%d\n", ok_aligned, ok_unaligned);
  check(ok_aligned && !ok_unaligned, "zero-copy wraps 16 KiB-aligned weights only");
}

} // namespace

int main() {
  dbinfer::backend::Backend *metal = dbinfer::backend::metal_backend();
  if (metal == nullptr) {
    std::printf("SKIP no Metal device available\n");
    return 0;
  }

  const std::size_t ms[] = {1, 4, 7};
  const std::size_t dims[] = {32, 896, 4864};
  double worst = 0.0;
  for (std::size_t m : ms)
    for (std::size_t out : dims)
      for (std::size_t in : dims)
        worst = std::max(worst, run_shape(*metal, m, out, in));

  std::printf("worst atol+rtol ratio across shapes=%.2f\n", worst);
  check(worst <= 1.0, "metal mul_mat_f16 within atol 1e-4 + rtol 1e-4 of CPU");

  test_zero_copy();

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
