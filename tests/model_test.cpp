#include "model/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gguf/gguf.hpp"
#include "test_util.hpp"

#ifndef DBINFER_TEST_GGUF
#error "DBINFER_TEST_GGUF must be defined by the build"
#endif

namespace {

using dbinfer::test::g_failures;
using dbinfer::test::load_bin;

// allclose per numpy
void check(const char* what, const std::vector<float>& got, const std::vector<float>& ref) {
  const double atol = 1e-3, rtol = 1e-3;
  double max_abs = 0.0, max_rel = 0.0;
  std::size_t nbad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double a = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    double denom = std::fabs(static_cast<double>(ref[i]));
    max_abs = std::max(max_abs, a);
    if (denom > 1e-6) max_rel = std::max(max_rel, a / denom);
    if (a > atol + rtol * denom) ++nbad;
  }
  bool ok = nbad == 0;
  std::printf("%s %-16s max_abs=%.3e max_rel=%.3e bad=%zu/%zu\n", ok ? "PASS" : "FAIL", what,
              max_abs, max_rel, nbad, got.size());
  if (!ok) ++g_failures;
}

}  // namespace

int main() {
  auto loaded = dbinfer::gguf::load(DBINFER_TEST_GGUF);
  if (!loaded) {
    std::printf("FAIL cannot load gguf: %s\n", dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }
  auto mret = dbinfer::model::Model::load(*loaded);
  if (!mret) {
    std::printf("FAIL model load: %s\n", dbinfer::gguf::to_string(mret.error()).c_str());
    return 1;
  }
  dbinfer::model::Model& model = *mret;
  const auto& cfg = model.config();

  const std::int32_t ids[6] = {40, 1079, 264, 4128, 1614, 13};
  const std::int32_t positions[6] = {0, 1, 2, 3, 4, 5};
  const std::size_t seq = 6;
  const std::size_t dim = cfg.embedding_length;
  const std::size_t hd = cfg.head_dim;

  // Step 4: embedding.
  auto emb_ref = load_bin("embedding.bin", seq * dim);
  if (emb_ref.empty()) return 1;
  std::vector<float> emb(seq * dim);
  for (std::size_t s = 0; s < seq; ++s) model.embed(ids[s], emb.data() + s * dim);
  check("embedding", emb, emb_ref);

  // Step 6: layer-0 head-0 attention output, fed from the embedding rows.
  {
    dbinfer::model::KVCache kv(cfg.n_layers, seq, cfg.n_kv_heads, hd);
    std::vector<float> head0(seq * hd);
    for (std::size_t s = 0; s < seq; ++s) {
      std::vector<float> x(emb.begin() + s * dim, emb.begin() + (s + 1) * dim);
      dbinfer::model::DebugCapture dbg;
      dbg.attn_head0 = head0.data() + s * hd;
      model.decode_layer(0, x.data(), positions[s], kv, &dbg);
    }
    auto ref = load_bin("attn_head0_out.bin", seq * hd);
    if (!ref.empty()) check("attn_head0", head0, ref);
  }

  // Step 7: full layer-0 residual output.
  {
    dbinfer::model::KVCache kv(cfg.n_layers, seq, cfg.n_kv_heads, hd);
    std::vector<float> hout(seq * dim);
    for (std::size_t s = 0; s < seq; ++s) {
      std::vector<float> x(emb.begin() + s * dim, emb.begin() + (s + 1) * dim);
      dbinfer::model::DebugCapture dbg;
      dbg.layer_out = hout.data() + s * dim;
      model.decode_layer(0, x.data(), positions[s], kv, &dbg);
    }
    auto ref = load_bin("layer0_out.bin", seq * dim);
    if (!ref.empty()) check("layer0_out", hout, ref);
  }

  // Step 8: full forward logits for all six positions.
  {
    std::vector<float> logits(seq * cfg.vocab_size);
    for (std::size_t s = 0; s < seq; ++s) {
      const float* l = model.forward(ids[s], positions[s]);
      std::copy(l, l + cfg.vocab_size, logits.data() + s * cfg.vocab_size);
    }
    auto ref = load_bin("logits.bin", seq * cfg.vocab_size);
    if (!ref.empty()) check("logits", logits, ref);
  }

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
