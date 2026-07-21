// chunked prefill must be bitwise identical to per-token prefill. prefills a
// fixed prompt both ways, with a chunk size that does not divide the prompt
// length so the last chunk is ragged, and asserts memcmp equality over the
// logits at every position.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gguf/gguf.hpp"
#include "model/model.hpp"

#ifndef DBINFER_TEST_GGUF
#error "DBINFER_TEST_GGUF must be defined by the build"
#endif

int main() {
  auto loaded = dbinfer::gguf::load(DBINFER_TEST_GGUF);
  if (!loaded) {
    std::printf("FAIL cannot load gguf: %s\n", dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }

  const std::int32_t ids[10] = {40, 1079, 264, 4128, 1614, 13, 358, 1079, 264, 4128};
  const std::size_t total = 10;
  const std::size_t chunk = 4;  // 4 does not divide 10: chunks 4, 4, 2.

  auto model_a = dbinfer::model::Model::load(*loaded);
  auto model_b = dbinfer::model::Model::load(*loaded);
  if (!model_a || !model_b) {
    std::printf("FAIL model load\n");
    return 1;
  }
  const std::size_t vocab = model_a->config().vocab_size;

  std::vector<float> per_token(total * vocab);
  for (std::size_t i = 0; i < total; ++i) {
    const float* l = model_a->forward(ids[i], static_cast<std::int32_t>(i));
    std::memcpy(per_token.data() + i * vocab, l, vocab * sizeof(float));
  }

  std::vector<float> chunked(total * vocab);
  std::vector<float> buf(chunk * vocab);
  for (std::size_t i = 0; i < total; i += chunk) {
    const std::size_t n = std::min(chunk, total - i);
    model_b->forward_chunk(ids + i, static_cast<std::int32_t>(i), n, buf.data());
    std::memcpy(chunked.data() + i * vocab, buf.data(), n * vocab * sizeof(float));
  }

  if (std::memcmp(per_token.data(), chunked.data(), total * vocab * sizeof(float)) != 0) {
    for (std::size_t p = 0; p < total; ++p) {
      for (std::size_t j = 0; j < vocab; ++j) {
        if (per_token[p * vocab + j] != chunked[p * vocab + j]) {
          std::printf("FAIL logits differ at pos %zu index %zu (per-token %.9g chunked %.9g)\n", p,
                      j, static_cast<double>(per_token[p * vocab + j]),
                      static_cast<double>(chunked[p * vocab + j]));
          return 1;
        }
      }
    }
    std::printf("FAIL memcmp mismatch without a located element\n");
    return 1;
  }

  std::printf("PASS chunked prefill bitwise identical over %zu positions x %zu logits\n", total,
              vocab);
  return 0;
}
