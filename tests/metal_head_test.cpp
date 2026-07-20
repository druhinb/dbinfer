// lm_head-on-GPU gate: the final rmsnorm plus lm_head projection that
// decode_token_full appends to the per-token command buffer. two checks per
// dtype. fed the same normed hidden state, the GPU quant matvec is memcmp
// identical to the CPU scalar reference (F16 within tolerance). fed the raw
// hidden state, the GPU tail (its own reordered rmsnorm then the matvec) lands
// within 1e-4 of the CPU tail. shapes use the qwen2.5-0.5b lm_head [151936,896].
// skips cleanly when no Metal device is present.

#include "backend/metal_backend.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul.hpp"
#include "tensor/matmul_neon.hpp"
#include "tensor/ops.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
  std::uint8_t byte() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint8_t>(s >> 40);
  }
};

using dbinfer::tensor::BlockQ8_0;
using dbinfer::tensor::kBlockSize;

std::vector<std::byte> quant_weight_q8(Lcg &rng, std::size_t out, std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  std::vector<float> f(out * in);
  for (auto &v : f)
    v = rng.next() * (rng.byte() < 8 ? 0.0f : 3.0f);
  std::vector<std::byte> w(out * nb * sizeof(BlockQ8_0));
  for (std::size_t o = 0; o < out; ++o)
    dbinfer::tensor::quantize_row_q8_0(f.data() + o * in, in,
                                       w.data() + o * nb * sizeof(BlockQ8_0));
  return w;
}

std::vector<std::byte> rand_weight_q4(Lcg &rng, std::size_t out, std::size_t in) {
  const std::size_t nb = in / kBlockSize;
  std::vector<std::byte> w(out * nb * 18);
  for (std::size_t b = 0; b < out * nb; ++b) {
    std::byte *blk = w.data() + b * 18;
    const std::uint16_t d = dbinfer::tensor::f32_to_f16(rng.next() * 2.0f);
    std::memcpy(blk, &d, sizeof(d));
    for (std::size_t j = 0; j < 16; ++j)
      blk[2 + j] = static_cast<std::byte>(rng.byte());
  }
  return w;
}

std::vector<std::uint16_t> rand_weight_f16(Lcg &rng, std::size_t out, std::size_t in) {
  std::vector<std::uint16_t> w(out * in);
  for (auto &v : w)
    v = dbinfer::tensor::f32_to_f16(rng.next() * 0.1f);
  return w;
}

float max_abs_err(const std::vector<float> &a, const std::vector<float> &b) {
  float e = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i)
    e = std::max(e, std::fabs(a[i] - b[i]));
  return e;
}

void check(bool ok, const char *what, float err) {
  std::printf("%s %s (max_err %.3e)\n", ok ? "PASS" : "FAIL", what, err);
  if (!ok)
    ++g_failures;
}

// cpu tail for a quant lm_head: rmsnorm then quantize the normed row then the
// scalar block dot the GPU matvec mirrors bit for bit.
void cpu_tail_quant(void (*matvec)(const std::byte *, const BlockQ8_0 *, float *, std::size_t,
                                   std::size_t),
                    const std::byte *w, const std::vector<float> &normed, std::size_t out,
                    std::size_t in, std::vector<float> &logits) {
  const std::size_t nb = in / kBlockSize;
  std::vector<std::byte> xq(nb * sizeof(BlockQ8_0));
  dbinfer::tensor::quantize_row_q8_0(normed.data(), in, xq.data());
  logits.assign(out, 0.0f);
  matvec(w, reinterpret_cast<const BlockQ8_0 *>(xq.data()), logits.data(), out, in);
}

} // namespace

int main() {
  dbinfer::backend::Backend *metal = dbinfer::backend::metal_backend();
  if (metal == nullptr) {
    std::printf("SKIP no Metal device available\n");
    return 0;
  }

  const std::size_t dim = 896;
  const std::size_t vocab = 151936;
  const float eps = 1e-6f;

  Lcg rng{0x5EED};
  std::vector<float> h(dim);
  for (auto &v : h)
    v = rng.next() * 2.0f;
  std::vector<float> wn(dim);
  for (auto &v : wn)
    v = 1.0f + rng.next() * 0.1f;

  std::vector<float> normed_cpu(dim, 0.0f);
  dbinfer::tensor::rmsnorm(h.data(), wn.data(), eps, normed_cpu.data(), 1, dim);
  std::vector<float> normed_gpu(dim, 0.0f);
  if (auto r = metal->rmsnorm(h.data(), wn.data(), eps, normed_gpu.data(), 1, dim); !r) {
    std::printf("FAIL rmsnorm error: %s\n", r.error().message.c_str());
    ++g_failures;
  }
  check(max_abs_err(normed_cpu, normed_gpu) <= 1e-4f, "final rmsnorm GPU vs CPU within 1e-4",
        max_abs_err(normed_cpu, normed_gpu));

  // Q8_0 lm_head.
  {
    std::vector<std::byte> w = quant_weight_q8(rng, vocab, dim);
    std::vector<float> cpu_same, gpu_same;
    cpu_tail_quant(dbinfer::tensor::matvec_q8_0_scalar, w.data(), normed_cpu, vocab, dim, cpu_same);
    gpu_same.assign(vocab, 0.0f);
    if (auto r = metal->mul_mat_q8_0(w.data(), normed_cpu.data(), gpu_same.data(), 1, vocab, dim);
        !r)
      std::printf("  q8 error: %s\n", r.error().message.c_str());
    const bool bitwise = std::memcmp(cpu_same.data(), gpu_same.data(), vocab * sizeof(float)) == 0;
    std::printf("%s Q8_0 lm_head bitwise vs CPU on same normed\n", bitwise ? "PASS" : "FAIL");
    if (!bitwise)
      ++g_failures;

    std::vector<float> cpu_tail, gpu_tail(vocab, 0.0f);
    cpu_tail_quant(dbinfer::tensor::matvec_q8_0_scalar, w.data(), normed_cpu, vocab, dim, cpu_tail);
    if (auto r = metal->mul_mat_q8_0(w.data(), normed_gpu.data(), gpu_tail.data(), 1, vocab, dim);
        !r)
      std::printf("  q8 tail error: %s\n", r.error().message.c_str());
    const float e = max_abs_err(cpu_tail, gpu_tail);
    check(e <= 1e-4f, "Q8_0 tail (rmsnorm+lm_head) GPU vs CPU within 1e-4", e);
  }

  // Q4_0 lm_head.
  {
    std::vector<std::byte> w = rand_weight_q4(rng, vocab, dim);
    std::vector<float> cpu_same(vocab, 0.0f), gpu_same(vocab, 0.0f);
    cpu_tail_quant(dbinfer::tensor::matvec_q4_0_scalar, w.data(), normed_cpu, vocab, dim, cpu_same);
    if (auto r = metal->mul_mat_q4_0(w.data(), normed_cpu.data(), gpu_same.data(), 1, vocab, dim);
        !r)
      std::printf("  q4 error: %s\n", r.error().message.c_str());
    const bool bitwise = std::memcmp(cpu_same.data(), gpu_same.data(), vocab * sizeof(float)) == 0;
    std::printf("%s Q4_0 lm_head bitwise vs CPU on same normed\n", bitwise ? "PASS" : "FAIL");
    if (!bitwise)
      ++g_failures;
  }

  // F16 lm_head, tolerance only (the f16 matvec reorders its fold).
  {
    std::vector<std::uint16_t> w = rand_weight_f16(rng, vocab, dim);
    std::vector<float> cpu(vocab, 0.0f), gpu(vocab, 0.0f);
    dbinfer::tensor::matvec_f16(w.data(), normed_cpu.data(), cpu.data(), vocab, dim);
    if (auto r = metal->mul_mat_f16(w.data(), normed_gpu.data(), gpu.data(), 1, vocab, dim); !r)
      std::printf("  f16 error: %s\n", r.error().message.c_str());
    const float e = max_abs_err(cpu, gpu);
    check(e <= 1e-3f, "F16 tail (rmsnorm+lm_head) GPU vs CPU within 1e-3", e);
  }

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
