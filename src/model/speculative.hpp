#ifndef DBINFER_MODEL_SPECULATIVE_HPP
#define DBINFER_MODEL_SPECULATIVE_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "model/model.hpp"

namespace dbinfer::model {

// greedy speculative merge. draft holds the k proposed tokens; target holds the
// target model's greedy argmax at each of the k+1 verify positions, where
// target[i] is the argmax after consuming draft[i-1] (target[0] follows the
// committed context). appends the accepted draft prefix followed by one target
// token (the correction at the first mismatch, or the bonus when all k accept)
// to out, and returns the count of accepted draft tokens.
std::size_t speculative_merge(std::span<const std::int32_t> draft,
                              std::span<const std::int32_t> target, std::vector<std::int32_t>& out);

struct SpecStats {
  std::size_t proposed = 0;
  std::size_t accepted = 0;
  std::size_t rounds = 0;
};

// greedy generation with draft-model speculation. the draft proposes k tokens
// per round; the target verifies them in one batched forward whose logits are
// bitwise identical to sequential per-token forward, so the returned token
// stream is identical to pure target greedy. both models must use the default
// dense fp32 KV cache and share a vocabulary. generates up to n tokens, halting
// at eos (which is excluded from the result). resets both KV caches first.
std::vector<std::int32_t> speculative_generate(Model& target, Model& draft,
                                               std::span<const std::int32_t> prompt, std::size_t k,
                                               std::size_t n, std::int32_t eos, SpecStats* stats);

}  // namespace dbinfer::model

#endif  // DBINFER_MODEL_SPECULATIVE_HPP
