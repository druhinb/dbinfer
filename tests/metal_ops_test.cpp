// per-kernel parity: Metal rmsnorm, rope, and attention vs the CPU reference.
//
// rmsnorm and attention sum sequentially per output, so they match the scalar
// path bitwise apart from exp in the softmax; rope's fp32 cos/sin diverge from
// the double CPU transcendentals within tolerance. atol is 1e-4 throughout.
// skips cleanly when no Metal device is present.

#include "backend/metal_backend.hpp"
#include "tensor/ops.hpp"

#include <algorithm>
#include <cmath>
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
    std::uint32_t hi = static_cast<std::uint32_t>(s >> 32);
    return static_cast<float>(hi) / 2147483648.0f - 1.0f;
  }
};

void check(bool ok, const char *what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok)
    ++g_failures;
}

double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    m = std::max(m, d < 0 ? -d : d);
  }
  return m;
}

// dense fp32 single-query attention, mirroring Model::decode_layer exactly.
void cpu_attention(const float *q, const float *k, const float *v, float *out,
                   std::size_t n_positions, std::size_t nh, std::size_t nkv, std::size_t hd) {
  const std::size_t gqa = nh / nkv;
  const std::size_t stride = nkv * hd;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  std::vector<float> sc(n_positions, 0.0f);
  for (std::size_t h = 0; h < nh; ++h) {
    const std::size_t kh = h / gqa;
    const float *qh = q + h * hd;
    for (std::size_t pp = 0; pp < n_positions; ++pp) {
      const float *kp = k + pp * stride + kh * hd;
      float dot = 0.0f;
      for (std::size_t i = 0; i < hd; ++i)
        dot += qh[i] * kp[i];
      sc[pp] = dot * scale;
    }
    dbinfer::tensor::softmax(sc.data(), sc.data(), n_positions);
    float *outh = out + h * hd;
    for (std::size_t i = 0; i < hd; ++i)
      outh[i] = 0.0f;
    for (std::size_t pp = 0; pp < n_positions; ++pp) {
      const float *vp = v + pp * stride + kh * hd;
      const float w = sc[pp];
      for (std::size_t i = 0; i < hd; ++i)
        outh[i] += w * vp[i];
    }
  }
}

double test_rmsnorm(dbinfer::backend::Backend &m, std::size_t rows, std::size_t dim) {
  Lcg rng{0x5151 ^ (rows * 131 + dim * 17)};
  std::vector<float> x(rows * dim), w(dim), cpu(rows * dim, 0.0f), gpu(rows * dim, 0.0f);
  for (auto &e : x)
    e = rng.next();
  for (auto &e : w)
    e = rng.next();
  const float eps = 1e-6f;
  dbinfer::tensor::rmsnorm(x.data(), w.data(), eps, cpu.data(), rows, dim);
  auto r = m.rmsnorm(x.data(), w.data(), eps, gpu.data(), rows, dim);
  if (!r) {
    std::printf("  rmsnorm error: %s\n", r.error().message.c_str());
    ++g_failures;
    return 0.0;
  }
  const double err = max_abs_diff(cpu, gpu);
  std::printf("  rmsnorm rows=%zu dim=%zu max_err=%.3e\n", rows, dim, err);
  return err;
}

double test_rope(dbinfer::backend::Backend &m, std::size_t nh, std::size_t seq, std::size_t hd,
                 const std::vector<std::int32_t> &pos) {
  Lcg rng{0x2323 ^ (nh * 131 + seq * 17 + hd)};
  std::vector<float> x(nh * seq * hd), cpu, gpu;
  for (auto &e : x)
    e = rng.next();
  cpu = x;
  gpu = x;
  const float theta = 1000000.0f;
  dbinfer::tensor::rope(cpu.data(), pos.data(), theta, nh, seq, hd);
  auto r = m.rope(gpu.data(), pos.data(), theta, nh, seq, hd);
  if (!r) {
    std::printf("  rope error: %s\n", r.error().message.c_str());
    ++g_failures;
    return 0.0;
  }
  const double err = max_abs_diff(cpu, gpu);
  std::printf("  rope nh=%zu seq=%zu hd=%zu pos0=%d posN=%d max_err=%.3e\n", nh, seq, hd,
              pos.front(), pos.back(), err);
  return err;
}

double test_attention(dbinfer::backend::Backend &m, std::size_t nh, std::size_t nkv, std::size_t hd,
                      std::size_t n_pos) {
  Lcg rng{0x9797 ^ (nh * 131 + nkv * 17 + hd * 7 + n_pos)};
  const std::size_t stride = nkv * hd;
  std::vector<float> q(nh * hd), k(n_pos * stride), v(n_pos * stride);
  std::vector<float> cpu(nh * hd, 0.0f), gpu(nh * hd, 0.0f);
  for (auto &e : q)
    e = rng.next();
  for (auto &e : k)
    e = rng.next();
  for (auto &e : v)
    e = rng.next();
  cpu_attention(q.data(), k.data(), v.data(), cpu.data(), n_pos, nh, nkv, hd);
  auto r = m.attention(q.data(), k.data(), v.data(), gpu.data(), n_pos, nh, nkv, hd);
  if (!r) {
    std::printf("  attention error: %s\n", r.error().message.c_str());
    ++g_failures;
    return 0.0;
  }
  const double err = max_abs_diff(cpu, gpu);
  std::printf("  attention nh=%zu nkv=%zu hd=%zu n_pos=%zu max_err=%.3e\n", nh, nkv, hd, n_pos,
              err);
  return err;
}

} // namespace

int main() {
  dbinfer::backend::Backend *m = dbinfer::backend::metal_backend();
  if (m == nullptr) {
    std::printf("SKIP no Metal device available\n");
    return 0;
  }

  double worst_rms = 0.0;
  for (std::size_t rows : {std::size_t{1}, std::size_t{4}})
    for (std::size_t dim : {std::size_t{32}, std::size_t{896}, std::size_t{4864}})
      worst_rms = std::max(worst_rms, test_rmsnorm(*m, rows, dim));
  check(worst_rms <= 1e-4, "metal rmsnorm within atol 1e-4 of CPU");

  // fp32 pow caps rope accuracy vs the double CPU path; the angle error scales
  // with position and reaches 1e-4 only near pos 2048 for theta 1e6, far past
  // any parity-prompt range. probe up to 1024 for the gate.
  double worst_rope = 0.0;
  for (std::size_t hd : {std::size_t{64}, std::size_t{128}}) {
    worst_rope = std::max(worst_rope, test_rope(*m, 14, 1, hd, {0}));
    worst_rope = std::max(worst_rope, test_rope(*m, 2, 4, hd, {0, 1, 17, 511}));
    worst_rope = std::max(worst_rope, test_rope(*m, 8, 1, hd, {1024}));
  }
  check(worst_rope <= 1e-4, "metal rope within atol 1e-4 of CPU");

  double worst_attn = 0.0;
  for (std::size_t hd : {std::size_t{64}, std::size_t{128}})
    for (std::size_t n_pos : {std::size_t{1}, std::size_t{2}, std::size_t{64}, std::size_t{300}}) {
      worst_attn = std::max(worst_attn, test_attention(*m, 14, 2, hd, n_pos));
      worst_attn = std::max(worst_attn, test_attention(*m, 8, 8, hd, n_pos));
    }
  check(worst_attn <= 1e-4, "metal attention within atol 1e-4 of CPU");

  std::printf("worst rms=%.3e rope=%.3e attn=%.3e\n", worst_rms, worst_rope, worst_attn);
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
