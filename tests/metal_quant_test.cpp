// bitwise gate: Metal Q8_0/Q4_0 matvec and the Q8_0 activation quant kernel vs
// the CPU scalar references. the integer block dot is order-free and the fp32
// block accumulation is sequential, so GPU and CPU are memcmp-identical, not
// merely within tolerance. skips cleanly when no Metal device is present.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <span>
#include <vector>

#include "backend/backend.hpp"
#include "backend/metal_backend.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul.hpp"
#include "tensor/matmul_neon.hpp"

namespace {

int g_failures = 0;

struct Lcg {
  std::uint64_t s;
  float next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::uint32_t hi = static_cast<std::uint32_t>(s >> 32);
    return static_cast<float>(hi) / 2147483648.0f - 1.0f;
  }
  std::uint8_t byte() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint8_t>(s >> 40);
  }
};

void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

using dbinfer::tensor::BlockQ8_0;
using dbinfer::tensor::kBlockSize;

std::vector<std::byte> quant_weight_q8(Lcg& rng, std::size_t out, std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  std::vector<float> f(out * in);
  for (auto& v : f) v = rng.next() * (rng.byte() < 8 ? 0.0f : 3.0f);
  std::vector<std::byte> w(out * nb * sizeof(BlockQ8_0));
  for (std::size_t o = 0; o < out; ++o)
    dbinfer::tensor::quantize_row_q8_0(f.data() + o * in, in,
                                       w.data() + o * nb * sizeof(BlockQ8_0));
  return w;
}

// Q4_0 blocks are 18 bytes: fp16 scale then 16 nibble-pair bytes. synthesize
// random valid blocks directly; the scalar and GPU kernels read them the same.
std::vector<std::byte> rand_weight_q4(Lcg& rng, std::size_t out, std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  std::vector<std::byte> w(out * nb * 18);
  for (std::size_t b = 0; b < out * nb; ++b) {
    std::byte* blk = w.data() + b * 18;
    const std::uint16_t d = dbinfer::tensor::f32_to_f16(rng.next() * 2.0f);
    std::memcpy(blk, &d, sizeof(d));
    for (std::size_t j = 0; j < 16; ++j) blk[2 + j] = static_cast<std::byte>(rng.byte());
  }
  return w;
}

std::vector<float> rand_activation(Lcg& rng, std::size_t m, std::size_t in) {
  std::vector<float> a(m * in);
  for (auto& v : a) v = rng.next() * (rng.byte() < 8 ? 0.0f : 2.0f);
  return a;
}

void test_quant(dbinfer::backend::Backend& metal, std::size_t rows, std::size_t in) {
  Lcg rng{0xA1B2 ^ (rows * 131 + in)};
  const std::size_t nb = in / kBlockSize;
  std::vector<float> a = rand_activation(rng, rows, in);
  std::vector<std::byte> cpu(rows * nb * sizeof(BlockQ8_0));
  for (std::size_t r = 0; r < rows; ++r)
    dbinfer::tensor::quantize_row_q8_0(a.data() + r * in, in,
                                       cpu.data() + r * nb * sizeof(BlockQ8_0));
  std::vector<std::byte> gpu(rows * nb * sizeof(BlockQ8_0));
  auto res = metal.quantize_q8_0(a.data(), rows, in, gpu.data());
  if (!res) {
    std::printf("  quant error: %s\n", res.error().message.c_str());
    ++g_failures;
    return;
  }
  const bool ok = std::memcmp(cpu.data(), gpu.data(), cpu.size()) == 0;
  std::printf("  quant rows=%zu in=%zu %s\n", rows, in, ok ? "bit-exact" : "MISMATCH");
  if (!ok) ++g_failures;
}

void test_q8(dbinfer::backend::Backend& metal, std::size_t out, std::size_t in) {
  Lcg rng{0xC3D4 ^ (out * 17 + in)};
  std::vector<std::byte> w = quant_weight_q8(rng, out, in);
  std::vector<float> a = rand_activation(rng, 1, in);

  const std::size_t nb = in / kBlockSize;
  std::vector<std::byte> xq(nb * sizeof(BlockQ8_0));
  dbinfer::tensor::quantize_row_q8_0(a.data(), in, xq.data());
  std::vector<float> cpu(out, 0.0f);
  dbinfer::tensor::matvec_q8_0_scalar(w.data(), reinterpret_cast<const BlockQ8_0*>(xq.data()),
                                      cpu.data(), out, in);

  std::vector<float> gpu(out, 0.0f);
  auto res = metal.mul_mat_q8_0(w.data(), a.data(), gpu.data(), 1, out, in);
  if (!res) {
    std::printf("  q8 error: %s\n", res.error().message.c_str());
    ++g_failures;
    return;
  }
  const bool ok = std::memcmp(cpu.data(), gpu.data(), out * sizeof(float)) == 0;
  std::printf("  q8 out=%zu in=%zu %s\n", out, in, ok ? "bit-exact" : "MISMATCH");
  if (!ok) ++g_failures;
}

void test_q4(dbinfer::backend::Backend& metal, std::size_t out, std::size_t in) {
  Lcg rng{0xE5F6 ^ (out * 17 + in)};
  std::vector<std::byte> w = rand_weight_q4(rng, out, in);
  std::vector<float> a = rand_activation(rng, 1, in);

  const std::size_t nb = in / kBlockSize;
  std::vector<std::byte> xq(nb * sizeof(BlockQ8_0));
  dbinfer::tensor::quantize_row_q8_0(a.data(), in, xq.data());
  std::vector<float> cpu(out, 0.0f);
  dbinfer::tensor::matvec_q4_0_scalar(w.data(), reinterpret_cast<const BlockQ8_0*>(xq.data()),
                                      cpu.data(), out, in);

  std::vector<float> gpu(out, 0.0f);
  auto res = metal.mul_mat_q4_0(w.data(), a.data(), gpu.data(), 1, out, in);
  if (!res) {
    std::printf("  q4 error: %s\n", res.error().message.c_str());
    ++g_failures;
    return;
  }
  const bool ok = std::memcmp(cpu.data(), gpu.data(), out * sizeof(float)) == 0;
  std::printf("  q4 out=%zu in=%zu %s\n", out, in, ok ? "bit-exact" : "MISMATCH");
  if (!ok) ++g_failures;
}

std::vector<std::uint16_t> f16_weight(Lcg& rng, std::size_t out, std::size_t in) {
  std::vector<std::uint16_t> w(out * in);
  for (auto& v : w) v = dbinfer::tensor::f32_to_f16(rng.next());
  return w;
}

using dbinfer::backend::WeightType;

// grouped QKV/gate-up matvec must be bitwise identical to running each sibling
// weight through the single-weight kernel: the wide dispatch changes only the
// launch, not the per-row reduction.
void test_group(dbinfer::backend::Backend& metal, WeightType type, std::vector<std::size_t> outs,
                std::size_t in) {
  Lcg rng{0x9E37 ^ (outs[0] * 31 + outs.size() * 7 + in + static_cast<std::size_t>(type) * 101)};
  const std::size_t g = outs.size();
  std::vector<float> a = rand_activation(rng, 1, in);

  std::vector<std::vector<std::byte>> wq(g);
  std::vector<std::vector<std::uint16_t>> wf(g);
  std::vector<const std::byte*> wp(g);
  for (std::size_t i = 0; i < g; ++i) {
    if (type == WeightType::Q8_0) {
      wq[i] = quant_weight_q8(rng, outs[i], in);
      wp[i] = wq[i].data();
    } else if (type == WeightType::Q4_0) {
      wq[i] = rand_weight_q4(rng, outs[i], in);
      wp[i] = wq[i].data();
    } else {
      wf[i] = f16_weight(rng, outs[i], in);
      wp[i] = reinterpret_cast<const std::byte*>(wf[i].data());
    }
  }

  std::vector<std::vector<float>> sep(g);
  for (std::size_t i = 0; i < g; ++i) {
    sep[i].assign(outs[i], 0.0f);
    std::expected<void, dbinfer::backend::Error> r;
    if (type == WeightType::Q8_0)
      r = metal.mul_mat_q8_0(wp[i], a.data(), sep[i].data(), 1, outs[i], in);
    else if (type == WeightType::Q4_0)
      r = metal.mul_mat_q4_0(wp[i], a.data(), sep[i].data(), 1, outs[i], in);
    else
      r = metal.mul_mat_f16(reinterpret_cast<const std::uint16_t*>(wp[i]), a.data(), sep[i].data(),
                            1, outs[i], in);
    if (!r) {
      std::printf("  group separate error: %s\n", r.error().message.c_str());
      ++g_failures;
      return;
    }
  }

  std::vector<std::vector<float>> grp(g);
  std::vector<float*> cp(g);
  for (std::size_t i = 0; i < g; ++i) {
    grp[i].assign(outs[i], 0.0f);
    cp[i] = grp[i].data();
  }
  auto r = metal.mul_mat_group(type, std::span<const std::byte* const>(wp.data(), g),
                               std::span<const std::size_t>(outs.data(), g), a.data(),
                               std::span<float* const>(cp.data(), g), in);
  if (!r) {
    std::printf("  group error: %s\n", r.error().message.c_str());
    ++g_failures;
    return;
  }

  bool ok = true;
  for (std::size_t i = 0; i < g; ++i)
    ok = ok && std::memcmp(sep[i].data(), grp[i].data(), outs[i] * sizeof(float)) == 0;
  const char* tn = type == WeightType::Q8_0 ? "q8" : type == WeightType::Q4_0 ? "q4" : "f16";
  std::printf("  group %s g=%zu in=%zu %s\n", tn, g, in, ok ? "bit-exact vs separate" : "MISMATCH");
  if (!ok) ++g_failures;
}

}  // namespace

int main() {
  dbinfer::backend::Backend* metal = dbinfer::backend::metal_backend();
  if (metal == nullptr) {
    std::printf("SKIP no Metal device available\n");
    return 0;
  }

  for (std::size_t rows : {std::size_t{1}, std::size_t{4}})
    for (std::size_t in : {std::size_t{896}, std::size_t{4864}}) test_quant(*metal, rows, in);

  // out/in cover the qwen2.5-0.5b projection shapes plus the 151936-row lm_head.
  const std::size_t shapes[][2] = {{896, 896}, {4864, 896}, {896, 4864}, {151936, 896}};
  for (const auto& s : shapes) {
    test_q8(*metal, s[0], s[1]);
    test_q4(*metal, s[0], s[1]);
  }

  // QKV (14x64, 2x64, 2x64) and gate/up (4864, 4864) grouped vs separate, all
  // dtypes, at the qwen2.5-0.5b projection widths (in=896).
  for (WeightType type : {WeightType::Q8_0, WeightType::Q4_0, WeightType::F16}) {
    test_group(*metal, type, {896, 128, 128}, 896);
    test_group(*metal, type, {4864, 4864}, 896);
  }

  check(g_failures == 0, "GPU quant matvec and activation quant bitwise-identical to CPU scalar");
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
