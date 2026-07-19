// unit tests for int8 KV quantization: per-channel key round-trip and int8
// self-attention against the fp32 reference. keys carry a per-channel scale
// shared across a token group; values stay per-block per-token.

#include "model/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using dbinfer::model::kKvBlock;
using dbinfer::model::KVCache;
using dbinfer::model::KvDtype;
using dbinfer::model::KvPolicy;

int g_failures = 0;

void check(bool ok, const char *what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok)
    ++g_failures;
}

struct Lcg {
  std::uint64_t s;
  float next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint32_t hi = static_cast<std::uint32_t>(s >> 32);
    return static_cast<float>(hi) / 2147483648.0f - 1.0f;
  }
};

void test_roundtrip() {
  const std::size_t nkv = 3, hd = 40, slots = 5;
  Lcg rng{0xC0FFEE};
  KVCache kv(/*n_layers=*/1, /*max_seq=*/slots, nkv, hd, KvPolicy{0, 0, KvDtype::Int8});

  std::vector<float> k(nkv * hd), v(nkv * hd);
  bool ok = true;
  for (std::size_t s = 0; s < slots; ++s) {
    for (auto &x : k)
      x = rng.next() * 7.0f;
    for (auto &x : v)
      x = rng.next() * 0.03f;
    kv.append(0, s, k.data(), v.data());
    for (std::size_t h = 0; h < nkv; ++h) {
      const std::int8_t *k8 = kv.key_i8(0, s, h);
      const float *ks = kv.key_scales(0, s, h);
      for (std::size_t i = 0; i < hd; ++i) {
        const float sc = ks[i];
        const float err = std::fabs(k[h * hd + i] - sc * static_cast<float>(k8[i]));
        ok = ok && err <= sc + 1e-6f;
      }
    }
  }
  check(ok, "per-channel key quant/dequant error within one scale step");
}

// int8 self-attention vs fp32. per-channel key and per-block value quant error
// is bounded by scale/2 = max_abs/254 (~0.4%); softmax and the value mix keep
// the head output within 1% relative L2 on random inputs.
void test_attention_matches_fp32() {
  const std::size_t hd = 64, keys = 48;
  Lcg rng{0x1357};
  std::vector<float> q(hd), k(keys * hd), v(keys * hd);
  for (auto &x : q)
    x = rng.next();
  for (auto &x : k)
    x = rng.next();
  for (auto &x : v)
    x = rng.next();

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  auto softmax = [](std::vector<float> &s) {
    float m = s[0];
    for (float x : s)
      m = std::max(m, x);
    float sum = 0.0f;
    for (float &x : s) {
      x = std::exp(x - m);
      sum += x;
    }
    for (float &x : s)
      x /= sum;
  };

  std::vector<float> ref(hd, 0.0f);
  {
    std::vector<float> sc(keys);
    for (std::size_t p = 0; p < keys; ++p) {
      float dot = 0.0f;
      for (std::size_t i = 0; i < hd; ++i)
        dot += q[i] * k[p * hd + i];
      sc[p] = dot * scale;
    }
    softmax(sc);
    for (std::size_t p = 0; p < keys; ++p)
      for (std::size_t i = 0; i < hd; ++i)
        ref[i] += sc[p] * v[p * hd + i];
  }

  std::vector<float> got(hd, 0.0f);
  {
    KVCache kv(1, keys, 1, hd, KvPolicy{0, 0, KvDtype::Int8});
    for (std::size_t p = 0; p < keys; ++p)
      kv.append(0, p, k.data() + p * hd, v.data() + p * hd);
    std::vector<float> sc(keys);
    for (std::size_t p = 0; p < keys; ++p) {
      const std::int8_t *k8 = kv.key_i8(0, p, 0);
      const float *ks = kv.key_scales(0, p, 0);
      float dot = 0.0f;
      for (std::size_t i = 0; i < hd; ++i)
        dot += q[i] * ks[i] * static_cast<float>(k8[i]);
      sc[p] = dot * scale;
    }
    softmax(sc);
    for (std::size_t p = 0; p < keys; ++p) {
      const std::int8_t *v8 = kv.value_i8(0, p, 0);
      const float *vs = kv.value_scales(0, p, 0);
      for (std::size_t i = 0; i < hd; ++i)
        got[i] += sc[p] * vs[i / kKvBlock] * static_cast<float>(v8[i]);
    }
  }

  float num = 0.0f, den = 0.0f;
  for (std::size_t i = 0; i < hd; ++i) {
    const float d = got[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  const float rel = std::sqrt(num / den);
  std::printf("int8 attention relative L2 = %.5f\n", rel);
  check(rel <= 0.006f, "int8 attention output within 0.6% relative L2 of fp32");
}

} // namespace

int main() {
  test_roundtrip();
  test_attention_matches_fp32();
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
