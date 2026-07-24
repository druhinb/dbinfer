// ctest for greedy argmax and the sampler stages.

#include "sample/sample.hpp"

#include <cmath>
#include <set>
#include <vector>

#include "test_util.hpp"

namespace {

using dbinfer::sample::argmax;
using dbinfer::sample::Candidate;
using dbinfer::sample::min_p;
using dbinfer::sample::Sampler;
using dbinfer::sample::SamplerParams;
using dbinfer::sample::top_k;
using dbinfer::sample::top_p;
using dbinfer::test::check;

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

void test_argmax() {
  std::vector<float> plain = {0.1f, 0.9f, 0.3f, -1.0f};
  check(argmax(plain.data(), plain.size()) == 1, "plain max at index 1");

  std::vector<float> tie = {5.0f, 1.0f, 5.0f, 5.0f};
  check(argmax(tie.data(), tie.size()) == 0, "tie -> lowest index");

  std::vector<float> negatives = {-3.0f, -2.0f, -2.0f};
  check(argmax(negatives.data(), negatives.size()) == 1, "negatives, first max wins");

  std::vector<float> single = {42.0f};
  check(argmax(single.data(), single.size()) == 0, "single element");
}

void test_top_k() {
  auto c = make({1.0f, 5.0f, 2.0f, 5.0f, 3.0f});
  top_k(c, 2);
  check((ids(c) == std::set<std::int32_t>{1, 3}), "top-k keeps logit-desc id-asc {1,3}");

  // three-way tie straddling k=2
  auto tied = make({5.0f, 5.0f, 5.0f, 1.0f});
  top_k(tied, 2);
  check((ids(tied) == std::set<std::int32_t>{0, 1}), "top-k tie at cutoff keeps lower ids {0,1}");
}

void test_top_p() {
  auto c = make({std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)});
  top_p(c, 0.7f);
  check((ids(c) == std::set<std::int32_t>{0, 1}), "top-p inclusive cutoff keeps {0,1}");

  auto zero = make({1.0f, 3.0f, 2.0f});
  top_p(zero, 0.0f);
  check((ids(zero) == std::set<std::int32_t>{1}), "top-p 0 keeps exactly the argmax");
}

void test_min_p() {
  auto boundary = make({0.0f, std::log(0.5f)});
  min_p(boundary, 0.5f);
  check((ids(boundary) == std::set<std::int32_t>{0, 1}), "min-p boundary keeps ratio == 0.5");

  auto above = make({0.0f, std::log(0.5f)});
  min_p(above, 0.5001f);
  check((ids(above) == std::set<std::int32_t>{0}), "min-p just above 0.5 drops the boundary token");
}

void test_sampler_temperature_zero() {
  SamplerParams p;
  p.temperature = 0.0f;
  Sampler s(p);
  std::vector<std::vector<float>> cases = {
      {0.1f, 0.9f, 0.3f}, {5.0f, 1.0f, 5.0f, 5.0f}, {-3.0f, -2.0f, -2.0f}};
  bool ok = true;
  for (auto& v : cases) ok = ok && s.sample(v.data(), v.size()) == argmax(v.data(), v.size());
  check(ok, "temperature 0 equals greedy argmax");
}

void test_sampler_seed_determinism() {
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

void test_repeat_penalty() {
  auto positive = make({2.0f});
  std::vector<std::int32_t> r1 = {0};
  apply_penalties(positive, r1, 2.0f, 0.0f, 0.0f);
  check(approx(positive[0].logit, 1.0f), "repeat penalty halves positive logit");

  auto negative = make({-2.0f});
  std::vector<std::int32_t> r2 = {0};
  apply_penalties(negative, r2, 2.0f, 0.0f, 0.0f);
  check(approx(negative[0].logit, -4.0f), "repeat penalty doubles negative logit");

  auto zero = make({0.0f});
  std::vector<std::int32_t> r3 = {0};
  apply_penalties(zero, r3, 2.0f, 0.0f, 0.0f);
  check(zero[0].logit == 0.0f, "repeat penalty leaves zero logit at zero");
}

void test_frequency_and_presence_penalty() {
  auto freq = make({5.0f});
  std::vector<std::int32_t> r1 = {0, 0, 0};
  apply_penalties(freq, r1, 1.0f, 0.5f, 0.0f);
  check(approx(freq[0].logit, 3.5f), "frequency penalty scales with count");

  auto presence = make({5.0f});
  std::vector<std::int32_t> r2 = {0, 0, 0};
  apply_penalties(presence, r2, 1.0f, 0.0f, 0.7f);
  check(approx(presence[0].logit, 4.3f), "presence penalty applied once per distinct token");
}

void test_combined_penalty_order() {
  auto c = make({4.0f});
  std::vector<std::int32_t> recent = {0};
  apply_penalties(c, recent, 2.0f, 1.0f, 0.5f);
  check(approx(c[0].logit, 0.5f), "combined penalties: repeat before freq+presence");
}

void test_default_penalties_are_noop() {
  auto c = make({2.0f, -1.0f, 0.0f, 7.5f});
  auto before = c;
  std::vector<std::int32_t> recent = {0, 1, 2, 3};
  apply_penalties(c, recent, 1.0f, 0.0f, 0.0f);
  bool same = true;
  for (std::size_t i = 0; i < c.size(); ++i) same = same && c[i].logit == before[i].logit;
  check(same, "default penalties are a bit-identical no-op");
}

void test_repeat_penalty_window() {
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

void test_penalty_precedes_temperature() {
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

void test_sampler_default_penalties_greedy() {
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

void test_sampler_repetition_diverges_from_argmax() {
  SamplerParams p;
  p.temperature = 0.0f;
  p.repeat_penalty = 100.0f;
  Sampler s(p);
  std::vector<float> v = {5.0f, 4.9f};
  std::vector<std::int32_t> recent = {0};
  check(s.sample(v.data(), v.size(), recent) != argmax(v.data(), v.size()),
        "greedy with repetition penalty diverges from raw argmax");
}

void test_validate_repeat_penalty() {
  SamplerParams p;
  check(validate(p).has_value(), "validate accepts defaults");
  p.repeat_penalty = 0.0f;
  check(!validate(p).has_value(), "validate rejects repeat_penalty 0");
  p.repeat_penalty = -1.0f;
  check(!validate(p).has_value(), "validate rejects negative repeat_penalty");
}

}  // namespace

int main() {
  test_argmax();
  test_top_k();
  test_top_p();
  test_min_p();
  test_sampler_temperature_zero();
  test_sampler_seed_determinism();
  test_repeat_penalty();
  test_frequency_and_presence_penalty();
  test_combined_penalty_order();
  test_default_penalties_are_noop();
  test_repeat_penalty_window();
  test_penalty_precedes_temperature();
  test_sampler_default_penalties_greedy();
  test_sampler_repetition_diverges_from_argmax();
  test_validate_repeat_penalty();

  return dbinfer::test::summary();
}
