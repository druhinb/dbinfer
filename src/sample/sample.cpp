#include "sample/sample.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace dbinfer::sample {

namespace {
// first-max wins: strict > so an equal later candidate never displaces best.
template <class T, class Proj>
std::int32_t argmax_by(std::span<const T> c, Proj proj) {
  std::int32_t best = 0;
  float bestv = proj(c[0]);
  for (std::size_t i = 1; i < c.size(); ++i) {
    const float v = proj(c[i]);
    if (v > bestv) {
      bestv = v;
      best = static_cast<std::int32_t>(i);
    }
  }
  return best;
}
}  // namespace

std::int32_t argmax(const float* logits, std::size_t n) {
  return argmax_by(std::span<const float>(logits, n), [](float x) { return x; });
}

std::expected<void, Error> validate(const SamplerParams& p) {
  if (p.temperature < 0.0f) return std::unexpected(Error{"temperature must be >= 0"});
  if (p.top_p < 0.0f || p.top_p > 1.0f) return std::unexpected(Error{"top_p must be in [0, 1]"});
  if (p.min_p < 0.0f || p.min_p > 1.0f) return std::unexpected(Error{"min_p must be in [0, 1]"});
  if (p.repeat_penalty <= 0.0f) return std::unexpected(Error{"repeat_penalty must be > 0"});
  return {};
}

void apply_penalties(std::span<Candidate> c, std::span<const std::int32_t> recent,
                     float repeat_penalty, float freq_penalty, float presence_penalty) {
  if (repeat_penalty == 1.0f && freq_penalty == 0.0f && presence_penalty == 0.0f) return;

  std::unordered_map<std::int32_t, int> counts;
  for (std::int32_t id : recent) ++counts[id];
  for (const auto& [id, count] : counts) {
    if (id < 0 || static_cast<std::size_t>(id) >= c.size()) continue;
    float& logit = c[static_cast<std::size_t>(id)].logit;
    // positive logit divided, negative logit multiplied, zero stays zero.
    if (logit > 0.0f)
      logit /= repeat_penalty;
    else
      logit *= repeat_penalty;
    logit -= static_cast<float>(count) * freq_penalty + presence_penalty;
  }
}

void apply_temperature(std::span<Candidate> c, float temperature) {
  for (Candidate& x : c) x.logit /= temperature;
}

namespace {
bool logit_desc_id_asc(const Candidate& a, const Candidate& b) {
  if (a.logit != b.logit) return a.logit > b.logit;
  return a.id < b.id;
}
}  // namespace

void top_k(std::vector<Candidate>& c, int k) {
  if (k <= 0 || static_cast<std::size_t>(k) >= c.size()) return;
  std::partial_sort(c.begin(), c.begin() + k, c.end(), logit_desc_id_asc);
  c.resize(static_cast<std::size_t>(k));
}

void top_p(std::vector<Candidate>& c, float p) {
  if (p >= 1.0f) return;
  std::sort(c.begin(), c.end(), logit_desc_id_asc);
  softmax(c);
  float cum = 0.0f;
  std::size_t cut = 0;
  for (; cut < c.size(); ++cut) {
    cum += c[cut].p;
    if (cum >= p) break;
  }
  c.resize(std::min(cut + 1, c.size()));
}

void min_p(std::vector<Candidate>& c, float min_p) {
  if (min_p <= 0.0f) return;
  float max_logit = c[0].logit;
  for (const Candidate& x : c) max_logit = std::max(max_logit, x.logit);

  const float thresh = max_logit + std::log(min_p);
  std::erase_if(c, [thresh](const Candidate& x) { return x.logit < thresh; });
}

void softmax(std::span<Candidate> c) {
  float max_logit = c[0].logit;
  for (const Candidate& x : c) max_logit = std::max(max_logit, x.logit);
  float sum = 0.0f;
  for (Candidate& x : c) {
    x.p = std::exp(x.logit - max_logit);
    sum += x.p;
  }
  for (Candidate& x : c) x.p /= sum;
}

std::int32_t sample_from(std::span<const Candidate> c, std::mt19937_64& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float u = dist(rng);
  float cum = 0.0f;
  for (const Candidate& x : c) {
    cum += x.p;
    if (cum > u) return x.id;
  }
  return c.back().id;
}

Sampler::Sampler(const SamplerParams& p) : params_(p), rng_(p.seed) {}

void Sampler::reseed(std::uint64_t seed) { rng_.seed(seed); }

namespace {
bool penalties_active(const SamplerParams& p) {
  return p.repeat_penalty != 1.0f || p.freq_penalty != 0.0f || p.presence_penalty != 0.0f;
}
}  // namespace

std::int32_t Sampler::sample(const float* logits, std::size_t n) { return sample(logits, n, {}); }

std::int32_t Sampler::sample(const float* logits, std::size_t n,
                             std::span<const std::int32_t> recent_tokens) {
  const bool active = penalties_active(params_);

  if (params_.temperature == 0.0f && !active) return argmax(logits, n);

  std::span<const std::int32_t> window = recent_tokens;
  if (params_.penalty_last_n == 0)
    window = {};
  else if (params_.penalty_last_n > 0 &&
           recent_tokens.size() > static_cast<std::size_t>(params_.penalty_last_n))
    window = recent_tokens.last(static_cast<std::size_t>(params_.penalty_last_n));

  scratch_.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    scratch_[i] = Candidate{static_cast<std::int32_t>(i), logits[i], 0.0f};

  apply_penalties(scratch_, window, params_.repeat_penalty, params_.freq_penalty,
                  params_.presence_penalty);

  if (params_.temperature == 0.0f) {
    return argmax_by(std::span<const Candidate>(scratch_),
                     [](const Candidate& x) { return x.logit; });
  }

  apply_temperature(scratch_, params_.temperature);
  top_k(scratch_, params_.top_k);
  top_p(scratch_, params_.top_p);
  min_p(scratch_, params_.min_p);
  softmax(scratch_);
  return sample_from(scratch_, rng_);
}

}  // namespace dbinfer::sample
