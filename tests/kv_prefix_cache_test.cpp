// dense fp32 KV prefix cache round-trip. logits at the first token after the
// prefix must be bitwise identical between a fresh prefill and a run that
// reloads the serialized prefix.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
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

// prefills the prefix into `model`, serializes it to `path`, then scores the
// token at prefix_len.
std::optional<std::vector<float>> run_fresh(dbinfer::model::Model& model, const std::int32_t* ids,
                                            std::size_t prefix_len, const std::string& path,
                                            std::size_t vocab) {
  for (std::size_t i = 0; i < prefix_len; ++i) model.forward(ids[i], static_cast<std::int32_t>(i));

  if (auto ok = model.save_kv_prefix(path, std::span<const std::int32_t>(ids, prefix_len)); !ok) {
    std::printf("FAIL save_kv_prefix: %s\n", dbinfer::gguf::to_string(ok.error()).c_str());
    return std::nullopt;
  }

  const float* lf = model.forward(ids[prefix_len], static_cast<std::int32_t>(prefix_len));
  return std::vector<float>(lf, lf + vocab);
}

// reloads the serialized prefix into `model`, then scores the token at
// prefix_len the same way run_fresh() did.
std::optional<std::vector<float>> run_cached(dbinfer::model::Model& model, const std::int32_t* ids,
                                             std::size_t prefix_len, std::size_t total,
                                             const std::string& path, std::size_t vocab) {
  auto restored = model.load_kv_prefix(path, std::span<const std::int32_t>(ids, total));
  if (!restored) {
    std::printf("FAIL load_kv_prefix: %s\n", dbinfer::gguf::to_string(restored.error()).c_str());
    return std::nullopt;
  }
  if (*restored != prefix_len) {
    std::printf("FAIL restored prefix_len %zu != %zu\n", *restored, prefix_len);
    return std::nullopt;
  }

  const float* lc = model.forward(ids[prefix_len], static_cast<std::int32_t>(prefix_len));
  return std::vector<float>(lc, lc + vocab);
}

// prints the first mismatching element on failure
bool logits_match(std::span<const float> fresh, std::span<const float> cached) {
  if (std::memcmp(fresh.data(), cached.data(), fresh.size() * sizeof(float)) == 0) return true;

  std::size_t first = 0;
  for (; first < fresh.size(); ++first)
    if (fresh[first] != cached[first]) break;
  std::printf("FAIL logits differ: first mismatch at %zu (fresh %.9g cached %.9g)\n", first,
              static_cast<double>(fresh[first]), static_cast<double>(cached[first]));
  return false;
}

// an absent cache file must surface an error
bool missing_file_errors(dbinfer::model::Model& model, const std::int32_t* ids, std::size_t total) {
  auto bad = model.load_kv_prefix("does_not_exist.bin", std::span<const std::int32_t>(ids, total));
  if (bad) {
    std::printf("FAIL missing file should error\n");
    return false;
  }
  return true;
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

  auto logits_fresh = run_fresh(*model_a, ids, prefix_len, path, vocab);
  if (!logits_fresh) return 1;

  auto model_b = dbinfer::model::Model::load(*loaded);
  if (!model_b) {
    std::printf("FAIL model load: %s\n", dbinfer::gguf::to_string(model_b.error()).c_str());
    return 1;
  }

  auto logits_cached = run_cached(*model_b, ids, prefix_len, total, path, vocab);
  if (!logits_cached) return 1;
  std::remove(path.c_str());

  if (!logits_match(*logits_fresh, *logits_cached)) return 1;

  if (!missing_file_errors(*model_b, ids, total)) return 1;

  std::printf("PASS logits bitwise identical over %zu values\n", vocab);
  return 0;
}
