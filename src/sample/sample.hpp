#ifndef DBINFER_SAMPLE_SAMPLE_HPP
#define DBINFER_SAMPLE_SAMPLE_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace dbinfer::sample {

std::int32_t argmax(const float *logits, std::size_t n);

inline constexpr std::uint64_t kDefaultSeed = 0x9E3779B97F4A7C15ULL;

struct Candidate {
  std::int32_t id;
  float logit;
  float p;
};

struct SamplerParams {
  float temperature = 1.0f;
  int top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repeat_penalty = 1.0f;
  float freq_penalty = 0.0f;
  float presence_penalty = 0.0f;
  int penalty_last_n = 64;
  std::uint64_t seed = kDefaultSeed;
};

struct Error {
  std::string message;
};

std::expected<void, Error> validate(const SamplerParams &p);

// penalizes candidates that appear in recent (the last penalty_last_n
// tokens): repeat_penalty scales the logit toward zero, freq_penalty and
// presence_penalty subtract per-occurrence-count and flat amounts. A no-op
// when all three are at their neutral values (1.0, 0, 0).
void apply_penalties(std::span<Candidate> c, std::span<const std::int32_t> recent,
                     float repeat_penalty, float freq_penalty, float presence_penalty);
void apply_temperature(std::span<Candidate> c, float temperature);
// keeps only the k candidates with the highest logit, sorting c by logit
// descending (ties broken by id ascending, for determinism) as a side effect.
void top_k(std::vector<Candidate> &c, int k);
// nucleus sampling: softmaxes c (sorting it by logit first), then keeps the
// smallest prefix whose cumulative probability reaches p.
void top_p(std::vector<Candidate> &c, float p);
// drops any candidate whose logit is more than log(min_p) below the max
// logit, i.e. whose probability relative to the top candidate is < min_p.
void min_p(std::vector<Candidate> &c, float min_p);
void softmax(std::span<Candidate> c);
// draws one candidate via inverse-CDF sampling over c's probabilities;
// c must already be softmax()'d.
std::int32_t sample_from(std::span<const Candidate> c, std::mt19937_64 &rng);

// stateful sampler: owns the RNG and a scratch candidate buffer across calls
// so per-token sampling doesn't reallocate. Applies the pipeline in the
// fixed order penalties -> temperature -> top-k -> top-p -> min-p -> softmax
// -> sample; temperature 0 (with no active penalties) short-circuits straight
// to greedy argmax.
class Sampler {
public:
  explicit Sampler(const SamplerParams &p);
  void reseed(std::uint64_t seed);
  std::int32_t sample(const float *logits, std::size_t n);
  // recent_tokens feeds apply_penalties, windowed to the last
  // penalty_last_n entries (0 disables penalties, negative means unbounded).
  std::int32_t sample(const float *logits, std::size_t n,
                      std::span<const std::int32_t> recent_tokens);

private:
  SamplerParams params_;
  std::mt19937_64 rng_;
  std::vector<Candidate> scratch_;
};

} // namespace dbinfer::sample

#endif // DBINFER_SAMPLE_SAMPLE_HPP
