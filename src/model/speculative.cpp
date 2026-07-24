#include "model/speculative.hpp"

#include <cstddef>
#include <cstdint>

#include "sample/sample.hpp"

namespace dbinfer::model {

std::size_t speculative_merge(std::span<const std::int32_t> draft,
                              std::span<const std::int32_t> target,
                              std::vector<std::int32_t>& out) {
  std::size_t j = 0;
  while (j < draft.size() && draft[j] == target[j]) {
    out.push_back(draft[j]);
    ++j;
  }

  // target[j] is the correction at the first mismatch, or target[k] as the free
  // bonus token when every draft token was accepted.
  out.push_back(target[j]);
  return j;
}

std::vector<std::int32_t> speculative_generate(Model& target, Model& draft,
                                               std::span<const std::int32_t> prompt, std::size_t k,
                                               std::size_t n, std::int32_t eos, SpecStats* stats) {
  std::vector<std::int32_t> gen;
  if (prompt.empty() || k == 0 || n == 0) return gen;

  target.reset_kv();
  draft.reset_kv();

  const std::size_t vocab = target.config().vocab_size;

  // prefill all but the last prompt token into both caches. the last prompt
  // token stays pending so the first round's draft and verify begin from it.
  const std::size_t prompt_len = prompt.size();
  for (std::size_t i = 0; i + 1 < prompt_len; ++i) {
    target.forward(prompt[i], static_cast<std::int32_t>(i));
    draft.forward(prompt[i], static_cast<std::int32_t>(i));
  }
  std::int32_t last = prompt[prompt_len - 1];
  std::int32_t pos = static_cast<std::int32_t>(prompt_len - 1);

  std::vector<std::int32_t> proposals(k);
  std::vector<std::int32_t> inputs(k + 1);
  std::vector<std::int32_t> verified(k + 1);
  std::vector<float> chunk((k + 1) * vocab);
  std::vector<std::int32_t> merged;
  SpecStats st;

  bool stop = false;
  while (!stop && gen.size() < n) {
    std::int32_t cur = last;
    for (std::size_t i = 0; i < k; ++i) {
      const float* ld = draft.forward(cur, pos + static_cast<std::int32_t>(i));
      cur = sample::argmax(ld, vocab);
      proposals[i] = cur;
    }

    inputs[0] = last;
    for (std::size_t i = 0; i < k; ++i) inputs[i + 1] = proposals[i];
    target.forward_chunk(inputs.data(), pos, k + 1, chunk.data());
    for (std::size_t i = 0; i <= k; ++i)
      verified[i] = sample::argmax(chunk.data() + i * vocab, vocab);

    merged.clear();
    const std::size_t accepted =
        speculative_merge(std::span<const std::int32_t>(proposals.data(), k),
                          std::span<const std::int32_t>(verified.data(), k + 1), merged);
    st.proposed += k;
    st.accepted += accepted;
    ++st.rounds;

    // on full acceptance the draft cache never wrote proposals[k-1] (it stayed
    // the pending argmax of the last draft forward). write it so the next
    // round's draft attention reads a committed slot at pos+k.
    if (accepted == k) draft.forward(proposals[k - 1], pos + static_cast<std::int32_t>(k));

    last = merged.back();
    pos += static_cast<std::int32_t>(accepted) + 1;

    for (std::int32_t tok : merged) {
      if (gen.size() >= n || tok == eos) {
        stop = true;
        break;
      }
      gen.push_back(tok);
    }
  }

  if (stats != nullptr) *stats = st;
  return gen;
}

}  // namespace dbinfer::model
