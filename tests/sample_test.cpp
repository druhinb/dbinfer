// ctest for greedy argmax and the sampler stages.

#include "sample/sample.hpp"

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

using dbinfer::sample::Candidate;

std::vector<Candidate> make(const std::vector<float>& logits) {
  std::vector<Candidate> c;
  c.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i)
    c.push_back(Candidate{static_cast<std::int32_t>(i), logits[i], 0.0f});
  return c;
}

std::set<std::int32_t> ids(const std::vector<Candidate>& c) {
  std::set<std::int32_t> s;
  for (const Candidate& x : c) s.insert(x.id);
  return s;
}

bool approx(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) <= eps; }
}  // namespace

int main() {
  using namespace dbinfer::sample;

  {
    std::vector<float> v = {0.1f, 0.9f, 0.3f, -1.0f};
    check(argmax(v.data(), v.size()) == 1, "plain max at index 1");
  }
  {
    std::vector<float> v = {5.0f, 1.0f, 5.0f, 5.0f};
    check(argmax(v.data(), v.size()) == 0, "tie -> lowest index");
  }
  {
    std::vector<float> v = {-3.0f, -2.0f, -2.0f};
    check(argmax(v.data(), v.size()) == 1, "negatives, first max wins");
  }
  {
    std::vector<float> v = {42.0f};
    check(argmax(v.data(), v.size()) == 0, "single element");
  }

  {
    auto c = make({1.0f, 5.0f, 2.0f, 5.0f, 3.0f});
    top_k(c, 2);
    check((ids(c) == std::set<std::int32_t>{1, 3}), "top-k keeps logit-desc id-asc {1,3}");
  }
  {
    // three-way tie straddling k=2
    auto c = make({5.0f, 5.0f, 5.0f, 1.0f});
    top_k(c, 2);
    check((ids(c) == std::set<std::int32_t>{0, 1}), "top-k tie at cutoff keeps lower ids {0,1}");
  }

  {
    auto c = make({std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)});
    top_p(c, 0.7f);
    check((ids(c) == std::set<std::int32_t>{0, 1}), "top-p inclusive cutoff keeps {0,1}");
  }

  {
    auto c = make({1.0f, 3.0f, 2.0f});
    top_p(c, 0.0f);
    check((ids(c) == std::set<std::int32_t>{1}), "top-p 0 keeps exactly the argmax");
  }

  {
    auto c = make({0.0f, std::log(0.5f)});
    min_p(c, 0.5f);
    check((ids(c) == std::set<std::int32_t>{0, 1}), "min-p boundary keeps ratio == 0.5");
  }
  {
    auto c = make({0.0f, std::log(0.5f)});
    min_p(c, 0.5001f);
    check((ids(c) == std::set<std::int32_t>{0}), "min-p just above 0.5 drops the boundary token");
  }

  {
    SamplerParams p;
    p.temperature = 0.0f;
    Sampler s(p);
    std::vector<std::vector<float>> cases = {
        {0.1f, 0.9f, 0.3f}, {5.0f, 1.0f, 5.0f, 5.0f}, {-3.0f, -2.0f, -2.0f}};
    bool ok = true;
    for (auto& v : cases) ok = ok && s.sample(v.data(), v.size()) == argmax(v.data(), v.size());
    check(ok, "temperature 0 equals greedy argmax");
  }

  {
    SamplerParams p;
    p.temperature = 0.8f;
    p.seed = 12345ULL;
    Sampler a(p);
    Sampler b(p);
    std::vector<std::vector<float>> draws = {{0.2f, 1.5f, -0.3f, 0.7f, 2.1f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {-2.0f, 3.0f, 0.5f},
                                             {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}};
    std::vector<std::int32_t> seq_a, seq_b;
    for (auto& v : draws) {
      seq_a.push_back(a.sample(v.data(), v.size()));
      seq_b.push_back(b.sample(v.data(), v.size()));
    }
    check(seq_a == seq_b, "same seed produces identical id sequence");

    a.reseed(p.seed);
    std::int32_t first = a.sample(draws[0].data(), draws[0].size());
    check(first == seq_a[0], "reseed reproduces the first draw");
  }

  {
    auto c = make({2.0f});
    std::vector<std::int32_t> recent = {0};
    apply_penalties(c, recent, 2.0f, 0.0f, 0.0f);
    check(approx(c[0].logit, 1.0f), "repeat penalty halves positive logit");
  }
  {
    auto c = make({-2.0f});
    std::vector<std::int32_t> recent = {0};
    apply_penalties(c, recent, 2.0f, 0.0f, 0.0f);
    check(approx(c[0].logit, -4.0f), "repeat penalty doubles negative logit");
  }
  {
    auto c = make({0.0f});
    std::vector<std::int32_t> recent = {0};
    apply_penalties(c, recent, 2.0f, 0.0f, 0.0f);
    check(c[0].logit == 0.0f, "repeat penalty leaves zero logit at zero");
  }
  {
    auto c = make({5.0f});
    std::vector<std::int32_t> recent = {0, 0, 0};
    apply_penalties(c, recent, 1.0f, 0.5f, 0.0f);
    check(approx(c[0].logit, 3.5f), "frequency penalty scales with count");
  }
  {
    auto c = make({5.0f});
    std::vector<std::int32_t> recent = {0, 0, 0};
    apply_penalties(c, recent, 1.0f, 0.0f, 0.7f);
    check(approx(c[0].logit, 4.3f), "presence penalty applied once per distinct token");
  }
  {
    auto c = make({4.0f});
    std::vector<std::int32_t> recent = {0};
    apply_penalties(c, recent, 2.0f, 1.0f, 0.5f);
    check(approx(c[0].logit, 0.5f), "combined penalties: repeat before freq+presence");
  }
  {
    auto c = make({2.0f, -1.0f, 0.0f, 7.5f});
    auto before = c;
    std::vector<std::int32_t> recent = {0, 1, 2, 3};
    apply_penalties(c, recent, 1.0f, 0.0f, 0.0f);
    bool same = true;
    for (std::size_t i = 0; i < c.size(); ++i) same = same && c[i].logit == before[i].logit;
    check(same, "default penalties are a bit-identical no-op");
  }
  {
    // penalty_last_n=1 keeps only the last recent token (id 1) in-window.
    SamplerParams p;
    p.temperature = 0.0f;
    p.repeat_penalty = 100.0f;
    p.penalty_last_n = 1;
    Sampler s(p);
    std::vector<float> v = {1.0f, 2.0f};
    std::vector<std::int32_t> recent = {0, 1};
    check(s.sample(v.data(), v.size(), recent) == 0,
          "window truncation penalizes only in-window token");
  }
  {
    SamplerParams p;
    p.temperature = 0.5f;
    p.top_k = 1;
    p.freq_penalty = 3.5f;
    p.penalty_last_n = 2;
    Sampler s(p);
    std::vector<float> v = {1.0f, 4.0f};
    std::vector<std::int32_t> recent = {1};
    check(s.sample(v.data(), v.size(), recent) == 0,
          "penalties precede temperature in pipeline order");
  }
  {
    SamplerParams p;
    p.temperature = 0.0f;
    Sampler s(p);
    std::vector<std::vector<float>> cases = {
        {0.1f, 0.9f, 0.3f}, {5.0f, 1.0f, 5.0f, 5.0f}, {-3.0f, -2.0f, -2.0f}};
    std::vector<std::int32_t> recent = {0, 1};
    bool ok = true;
    for (auto& v : cases)
      ok = ok && s.sample(v.data(), v.size(), recent) == argmax(v.data(), v.size());
    check(ok, "greedy with default penalties equals argmax");
  }
  {
    SamplerParams p;
    p.temperature = 0.0f;
    p.repeat_penalty = 100.0f;
    Sampler s(p);
    std::vector<float> v = {5.0f, 4.9f};
    std::vector<std::int32_t> recent = {0};
    check(s.sample(v.data(), v.size(), recent) != argmax(v.data(), v.size()),
          "greedy with repetition penalty diverges from raw argmax");
  }
  {
    SamplerParams p;
    check(validate(p).has_value(), "validate accepts defaults");
    p.repeat_penalty = 0.0f;
    check(!validate(p).has_value(), "validate rejects repeat_penalty 0");
    p.repeat_penalty = -1.0f;
    check(!validate(p).has_value(), "validate rejects negative repeat_penalty");
  }

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
