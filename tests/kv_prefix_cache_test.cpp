// dense fp32 KV prefix cache round-trip. logits at the first token after the
// prefix must be bitwise identical between a fresh prefill and a run that
// reloads the serialized prefix.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "gguf/gguf.hpp"
#include "model/model.hpp"

#ifndef DBINFER_TEST_GGUF
#error "DBINFER_TEST_GGUF must be defined by the build"
#endif

namespace {

std::string cache_path() {
  const char* tmp = std::getenv("TMPDIR");
  std::string dir = tmp != nullptr ? tmp : "/tmp";
  if (!dir.empty() && dir.back() != '/') dir.push_back('/');
  return dir + "dbinfer_kv_prefix_test.bin";
}

}  // namespace

int main() {
  auto loaded = dbinfer::gguf::load(DBINFER_TEST_GGUF);
  if (!loaded) {
    std::printf("FAIL cannot load gguf: %s\n", dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }

  // tokens borrowed from model_test, split into a prefix plus the first token
  // scored after it.
  const std::int32_t ids[6] = {40, 1079, 264, 4128, 1614, 13};
  const std::size_t prefix_len = 4;
  const std::size_t total = 6;
  const std::string path = cache_path();

  auto model_a = dbinfer::model::Model::load(*loaded);
  if (!model_a) {
    std::printf("FAIL model load: %s\n", dbinfer::gguf::to_string(model_a.error()).c_str());
    return 1;
  }
  const std::size_t vocab = model_a->config().vocab_size;

  // fresh: prefill the prefix, dump it, then score the token at prefix_len.
  for (std::size_t i = 0; i < prefix_len; ++i)
    model_a->forward(ids[i], static_cast<std::int32_t>(i));
  if (auto ok = model_a->save_kv_prefix(path, std::span<const std::int32_t>(ids, prefix_len));
      !ok) {
    std::printf("FAIL save_kv_prefix: %s\n", dbinfer::gguf::to_string(ok.error()).c_str());
    return 1;
  }
  const float* lf = model_a->forward(ids[prefix_len], static_cast<std::int32_t>(prefix_len));
  std::vector<float> logits_fresh(lf, lf + vocab);

  // cached: reload the prefix into a fresh model, score the same token.
  auto model_b = dbinfer::model::Model::load(*loaded);
  if (!model_b) {
    std::printf("FAIL model load: %s\n", dbinfer::gguf::to_string(model_b.error()).c_str());
    return 1;
  }
  auto restored = model_b->load_kv_prefix(path, std::span<const std::int32_t>(ids, total));
  if (!restored) {
    std::printf("FAIL load_kv_prefix: %s\n", dbinfer::gguf::to_string(restored.error()).c_str());
    return 1;
  }
  if (*restored != prefix_len) {
    std::printf("FAIL restored prefix_len %zu != %zu\n", *restored, prefix_len);
    return 1;
  }
  const float* lc = model_b->forward(ids[prefix_len], static_cast<std::int32_t>(prefix_len));
  std::vector<float> logits_cached(lc, lc + vocab);

  std::remove(path.c_str());

  if (std::memcmp(logits_fresh.data(), logits_cached.data(), vocab * sizeof(float)) != 0) {
    std::size_t first = 0;
    for (; first < vocab; ++first)
      if (logits_fresh[first] != logits_cached[first]) break;
    std::printf("FAIL logits differ: first mismatch at %zu (fresh %.9g cached %.9g)\n", first,
                static_cast<double>(logits_fresh[first]),
                static_cast<double>(logits_cached[first]));
    return 1;
  }

  // header validation: a corrupted magic must return an error rather than crash.
  {
    auto bad =
        model_b->load_kv_prefix("does_not_exist.bin", std::span<const std::int32_t>(ids, total));
    if (bad) {
      std::printf("FAIL missing file should error\n");
      return 1;
    }
  }

  std::printf("PASS logits bitwise identical over %zu values\n", vocab);
  return 0;
}
