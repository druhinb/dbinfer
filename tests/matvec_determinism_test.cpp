#include "model/model.hpp"

#include "gguf/gguf.hpp"
#include "tensor/cpu.hpp"
#include "tensor/thread_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// the fp32 matvec/attention parallelization must be bitwise-identical to the
// single-thread path. runs a fixed prompt through a fresh model at one thread
// and at P threads and asserts every logit matches to the bit.

namespace {

int g_failures = 0;

std::vector<std::int32_t> make_prompt() {
  std::vector<std::int32_t> ids;
  for (std::size_t i = 0; i < 40; ++i)
    ids.push_back(static_cast<std::int32_t>((i * 131 + 7) % 977));
  return ids;
}

std::vector<float> run_forward(const char *path, std::size_t threads,
                               const std::vector<std::int32_t> &ids) {
  dbinfer::tensor::configure_thread_count(threads);
  auto loaded = dbinfer::gguf::load(path);
  if (!loaded) {
    std::printf("FAIL load gguf %s: %s\n", path, dbinfer::gguf::to_string(loaded.error()).c_str());
    ++g_failures;
    return {};
  }
  auto mret = dbinfer::model::Model::load(*loaded);
  if (!mret) {
    std::printf("FAIL model load %s: %s\n", path, dbinfer::gguf::to_string(mret.error()).c_str());
    ++g_failures;
    return {};
  }
  dbinfer::model::Model &model = *mret;
  const std::size_t vocab = model.config().vocab_size;
  std::vector<float> out(ids.size() * vocab);
  for (std::size_t s = 0; s < ids.size(); ++s) {
    const float *l = model.forward(ids[s], static_cast<std::int32_t>(s));
    std::memcpy(out.data() + s * vocab, l, vocab * sizeof(float));
  }
  return out;
}

void check(const char *label, const char *path, const std::vector<std::int32_t> &ids,
           std::size_t P) {
  std::vector<float> single = run_forward(path, 1, ids);
  std::vector<float> multi = run_forward(path, P, ids);
  if (single.empty() || multi.empty())
    return;
  if (single.size() != multi.size() ||
      std::memcmp(single.data(), multi.data(), single.size() * sizeof(float)) != 0) {
    std::size_t diffs = 0;
    for (std::size_t i = 0; i < single.size(); ++i)
      if (std::memcmp(&single[i], &multi[i], sizeof(float)) != 0)
        ++diffs;
    std::printf("FAIL %-6s %zu/%zu logits differ between threads 1 and %zu\n", label, diffs,
                single.size(), P);
    ++g_failures;
    return;
  }
  std::printf("PASS %-6s bitwise identical over %zu logits, threads 1 vs %zu\n", label,
              single.size(), P);
}

} // namespace

int main() {
  const std::size_t P = dbinfer::tensor::p_core_count();
  const std::vector<std::int32_t> ids = make_prompt();

#ifdef DBINFER_DET_F16
  check("F16", DBINFER_DET_F16, ids, P);
#endif
#ifdef DBINFER_DET_Q8
  check("Q8_0", DBINFER_DET_Q8, ids, P);
#endif
#ifdef DBINFER_DET_Q4
  check("Q4_0", DBINFER_DET_Q4, ids, P);
#endif
#ifdef DBINFER_DET_Q6K
  check("Q6_K", DBINFER_DET_Q6K, ids, P);
#endif
#ifdef DBINFER_DET_Q4K
  check("Q4_K_M", DBINFER_DET_Q4K, ids, P);
#endif

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
