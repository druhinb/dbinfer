// speculative decoding correctness. two layers: a pure-logic check that the
// accept/reject merge over synthetic draft/target streams reproduces the target
// oracle's greedy stream exactly, and (when the two ggufs are present) an
// end-to-end check that speculative_generate is token-identical to pure target
// greedy for k in {4, 8}.

#include "model/speculative.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "gguf/gguf.hpp"
#include "model/model.hpp"

namespace {

int g_failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++g_failures;
  }
}

std::int32_t argmax(const float* logits, std::size_t n) {
  std::int32_t best = 0;
  float bestv = logits[0];
  for (std::size_t i = 1; i < n; ++i)
    if (logits[i] > bestv) {
      bestv = logits[i];
      best = static_cast<std::int32_t>(i);
    }
  return best;
}

// deterministic greedy oracle keyed only on the last token, so a pure integer
// simulation can stand in for a real target model.
std::int32_t oracle(std::int32_t x) { return (x * 7 + 3) % 101; }

// draft that agrees with the oracle on some tokens and diverges on others,
// controlled by wrong_mod: token x diverges when x % wrong_mod == 0.
std::int32_t draft_guess(std::int32_t x, std::int32_t wrong_mod) {
  const std::int32_t g = oracle(x);
  return (wrong_mod > 0 && x % wrong_mod == 0) ? (g + 1) % 101 : g;
}

// runs the speculative merge round by round against the oracle and returns the
// merged token stream, which must match pure oracle greedy.
std::vector<std::int32_t> simulate(std::int32_t seed, std::size_t k, std::size_t n,
                                   std::int32_t wrong_mod, dbinfer::model::SpecStats* stats) {
  std::vector<std::int32_t> gen;
  std::int32_t last = seed;
  dbinfer::model::SpecStats st;
  std::vector<std::int32_t> proposals(k);
  std::vector<std::int32_t> verified(k + 1);
  std::vector<std::int32_t> merged;
  bool stop = false;
  while (!stop && gen.size() < n) {
    std::int32_t cur = last;
    for (std::size_t i = 0; i < k; ++i) {
      cur = draft_guess(cur, wrong_mod);
      proposals[i] = cur;
    }
    verified[0] = oracle(last);
    for (std::size_t i = 0; i < k; ++i) verified[i + 1] = oracle(proposals[i]);

    merged.clear();
    const std::size_t accepted = dbinfer::model::speculative_merge(
        std::span<const std::int32_t>(proposals.data(), k),
        std::span<const std::int32_t>(verified.data(), k + 1), merged);
    st.proposed += k;
    st.accepted += accepted;
    ++st.rounds;
    last = merged.back();
    for (std::int32_t tok : merged) {
      if (gen.size() >= n) {
        stop = true;
        break;
      }
      gen.push_back(tok);
    }
  }
  if (stats != nullptr) *stats = st;
  return gen;
}

std::vector<std::int32_t> oracle_greedy(std::int32_t seed, std::size_t n) {
  std::vector<std::int32_t> out;
  std::int32_t cur = seed;
  while (out.size() < n) {
    cur = oracle(cur);
    out.push_back(cur);
  }
  return out;
}

void test_merge_logic() {
  // direct return-value checks.
  {
    const std::int32_t d[3] = {10, 20, 30};
    const std::int32_t t[4] = {10, 20, 30, 40};
    std::vector<std::int32_t> out;
    const std::size_t a = dbinfer::model::speculative_merge(
        std::span<const std::int32_t>(d, 3), std::span<const std::int32_t>(t, 4), out);
    expect(a == 3, "all-accept count");
    expect(out.size() == 4 && out[0] == 10 && out[1] == 20 && out[2] == 30 && out[3] == 40,
           "all-accept emits bonus");
  }
  {
    const std::int32_t d[3] = {10, 99, 30};
    const std::int32_t t[4] = {10, 20, 30, 40};
    std::vector<std::int32_t> out;
    const std::size_t a = dbinfer::model::speculative_merge(
        std::span<const std::int32_t>(d, 3), std::span<const std::int32_t>(t, 4), out);
    expect(a == 1, "reject-at-1 count");
    expect(out.size() == 2 && out[0] == 10 && out[1] == 20, "reject emits correction");
  }
  {
    const std::int32_t d[2] = {5, 6};
    const std::int32_t t[3] = {9, 8, 7};
    std::vector<std::int32_t> out;
    const std::size_t a = dbinfer::model::speculative_merge(
        std::span<const std::int32_t>(d, 2), std::span<const std::int32_t>(t, 3), out);
    expect(a == 0, "reject-at-0 count");
    expect(out.size() == 1 && out[0] == 9, "immediate reject emits target argmax");
  }

  // merged stream equals oracle greedy across seeds, k, and divergence rates.
  for (std::int32_t seed : {1, 42, 77}) {
    for (std::size_t k : {1u, 4u, 8u}) {
      for (std::int32_t wrong : {0, 2, 3, 1}) {  // 0 = perfect draft, 1 = always wrong
        dbinfer::model::SpecStats st;
        std::vector<std::int32_t> spec = simulate(seed, k, 50, wrong, &st);
        std::vector<std::int32_t> ref = oracle_greedy(seed, 50);
        bool same = spec.size() == ref.size();
        for (std::size_t i = 0; same && i < ref.size(); ++i) same = spec[i] == ref[i];
        expect(same, "synthetic merged stream equals oracle greedy");
        if (wrong == 0) expect(st.accepted == st.proposed, "perfect draft accepts every token");
        if (wrong == 1) expect(st.accepted == 0, "always-wrong draft accepts nothing");
      }
    }
  }
}

#if defined(DBINFER_SPEC_TARGET) && defined(DBINFER_SPEC_DRAFT)
std::vector<std::int32_t> ref_greedy(dbinfer::model::Model& m, std::span<const std::int32_t> prompt,
                                     std::size_t n) {
  m.reset_kv();
  const std::size_t vocab = m.config().vocab_size;
  const float* logits = nullptr;
  std::int32_t pos = 0;
  for (std::int32_t id : prompt) logits = m.forward(id, pos++);
  std::vector<std::int32_t> out;
  while (out.size() < n) {
    const std::int32_t next = argmax(logits, vocab);
    out.push_back(next);
    logits = m.forward(next, pos++);
  }
  return out;
}

void test_end_to_end() {
  auto tloaded = dbinfer::gguf::load(DBINFER_SPEC_TARGET);
  auto dloaded = dbinfer::gguf::load(DBINFER_SPEC_DRAFT);
  if (!tloaded || !dloaded) {
    std::printf("FAIL cannot load spec ggufs\n");
    ++g_failures;
    return;
  }
  auto target = dbinfer::model::Model::load(*tloaded);
  auto draft = dbinfer::model::Model::load(*dloaded);
  if (!target || !draft) {
    std::printf("FAIL spec model load\n");
    ++g_failures;
    return;
  }
  if (draft->config().vocab_size != target->config().vocab_size) {
    std::printf("FAIL spec vocab mismatch\n");
    ++g_failures;
    return;
  }

  const std::int32_t prompt[8] = {40, 1079, 264, 4128, 1614, 13, 358, 1079};
  const std::size_t n = 48;
  std::vector<std::int32_t> ref = ref_greedy(*target, std::span<const std::int32_t>(prompt, 8), n);

  for (std::size_t k : {4u, 8u}) {
    dbinfer::model::SpecStats st;
    std::vector<std::int32_t> spec = dbinfer::model::speculative_generate(
        *target, *draft, std::span<const std::int32_t>(prompt, 8), k, n, -1, &st);
    bool same = spec.size() == ref.size();
    for (std::size_t i = 0; same && i < ref.size(); ++i) same = spec[i] == ref[i];
    if (!same) {
      std::printf("FAIL k=%zu speculative stream differs from target greedy\n", k);
      ++g_failures;
    } else {
      std::printf("PASS k=%zu token-identical over %zu tokens, accepted %zu/%zu\n", k, n,
                  st.accepted, st.proposed);
    }
  }
}
#endif

}  // namespace

int main() {
  test_merge_logic();
#if defined(DBINFER_SPEC_TARGET) && defined(DBINFER_SPEC_DRAFT)
  test_end_to_end();
#endif
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
