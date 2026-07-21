// unit tests for the StreamingLLM ring-buffer KV cache: slot mapping,
// resident-set membership, and cache-relative RoPE positions.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "model/model.hpp"

namespace {

using dbinfer::model::KVCache;
using dbinfer::model::KvPolicy;

int g_failures = 0;

void check(bool ok, const char* what) {
  if (ok) {
    std::printf("PASS %s\n", what);
  } else {
    std::printf("FAIL %s\n", what);
    ++g_failures;
  }
}

std::size_t expected_slot(std::size_t pos, std::size_t n_sink, std::size_t window) {
  if (pos < n_sink) return pos;
  return n_sink + (pos - n_sink) % window;
}

// feeds n_seen tokens, storing each token's absolute position as its k/v
// payload so key()/value() reveal which token currently owns a slot.
void run_case(std::size_t n_sink, std::size_t window, std::size_t n_seen) {
  char name[128];
  KVCache kv(/*n_layers=*/1, /*max_seq=*/0, /*n_kv_heads=*/1, /*head_dim=*/1,
             KvPolicy{n_sink, window, dbinfer::model::KvDtype::F32});

  for (std::size_t t = 0; t < n_seen; ++t) {
    const float payload = static_cast<float>(t);
    check(kv.slot_for(t) == expected_slot(t, n_sink, window), "slot_for matches reference");
    kv.append(0, t, &payload, &payload);
  }

  std::vector<KVCache::Resident> res(kv.capacity());
  const std::size_t n_res = kv.residents(res.data());

  const std::size_t sinks = std::min(n_seen, n_sink);
  std::size_t window_start = n_sink;
  if (n_seen > n_sink && n_seen - n_sink > window) window_start = n_seen - window;
  const std::size_t wlen = n_seen > n_sink ? n_seen - window_start : 0;

  std::snprintf(name, sizeof name, "n_res == sinks+window (sink=%zu win=%zu seen=%zu)", n_sink,
                window, n_seen);
  check(n_res == sinks + wlen, name);

  bool sinks_pinned = true;
  for (std::size_t r = 0; r < sinks; ++r) {
    sinks_pinned = sinks_pinned && res[r].slot == r &&
                   res[r].rope_pos == static_cast<std::int32_t>(r) &&
                   kv.key(0, res[r].slot, 0)[0] == static_cast<float>(r);
  }
  std::snprintf(name, sizeof name, "sinks never evicted (sink=%zu win=%zu seen=%zu)", n_sink,
                window, n_seen);
  check(sinks_pinned, name);

  bool window_recent = true;
  for (std::size_t r = sinks; r < n_res; ++r) {
    const std::size_t t = window_start + (r - sinks);
    window_recent = window_recent && res[r].slot == expected_slot(t, n_sink, window) &&
                    res[r].rope_pos == static_cast<std::int32_t>(n_sink + (t - window_start)) &&
                    kv.key(0, res[r].slot, 0)[0] == static_cast<float>(t);
  }
  std::snprintf(name, sizeof name, "window holds most recent tokens (sink=%zu win=%zu seen=%zu)",
                n_sink, window, n_seen);
  check(window_recent, name);

  bool contiguous = true;
  std::vector<std::size_t> slots;
  for (std::size_t r = 0; r < n_res; ++r) {
    contiguous = contiguous && res[r].rope_pos == static_cast<std::int32_t>(r);
    slots.push_back(res[r].slot);
  }
  std::snprintf(name, sizeof name, "rope positions pack to [0,n_res) (sink=%zu win=%zu seen=%zu)",
                n_sink, window, n_seen);
  check(contiguous, name);

  std::sort(slots.begin(), slots.end());
  const bool distinct = std::adjacent_find(slots.begin(), slots.end()) == slots.end();
  std::snprintf(name, sizeof name, "resident slots are distinct (sink=%zu win=%zu seen=%zu)",
                n_sink, window, n_seen);
  check(distinct, name);
}

}  // namespace

int main() {
  run_case(/*n_sink=*/4, /*window=*/8, /*n_seen=*/40);
  run_case(4, 8, 12);  // exactly n_sink+window
  run_case(4, 8, 11);  // window not yet full
  run_case(0, 8, 40);  // no sinks, naive window
  run_case(1, 4, 40);
  run_case(4, 8, 4);    // only sinks resident
  run_case(4, 8, 2);    // fewer than n_sink tokens
  run_case(4, 8, 100);  // many wraps
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
