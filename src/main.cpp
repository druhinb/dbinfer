#include "args.hpp"
#include "gguf/gguf.hpp"
#include "model/model.hpp"
#include "sample/sample.hpp"
#include "tensor/thread_pool.hpp"
#include "tokenizer/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

std::optional<std::string> read_file(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (f == nullptr)
    return std::nullopt;
  std::string data;
  char buf[65536];
  std::size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof buf, f)) > 0)
    data.append(buf, got);
  const bool ok = std::ferror(f) == 0;
  std::fclose(f);
  if (!ok)
    return std::nullopt;
  return data;
}

// classic llama.cpp perplexity: non-overlapping 512-token windows, scoring
// only the second half so every scored token has 256+ tokens of left context.
int run_perplexity(dbinfer::model::Model &model, const dbinfer::tokenizer::Tokenizer &tok,
                   const dbinfer::cli::CliOptions &opts) {
  auto text = read_file(opts.perplexity_path.c_str());
  if (!text) {
    std::fprintf(stderr, "error: cannot read %s\n", opts.perplexity_path.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens = tok.encode(*text, /*add_special=*/false);
  const std::int32_t bos = tok.bos_id();
  if (bos >= 0)
    tokens.insert(tokens.begin(), bos);

  constexpr std::size_t n_ctx = 512;
  constexpr std::size_t score_lo = n_ctx / 2; // 256
  const std::size_t vocab = model.config().vocab_size;

  if (tokens.size() < n_ctx) {
    std::fprintf(stderr, "error: need at least %zu tokens, got %zu\n", n_ctx, tokens.size());
    return 1;
  }

  std::size_t n_chunk = tokens.size() / n_ctx;
  if (opts.ppl_chunks > 0 && static_cast<std::size_t>(opts.ppl_chunks) < n_chunk)
    n_chunk = static_cast<std::size_t>(opts.ppl_chunks);

  double nll = 0.0;
  std::size_t count = 0;
  for (std::size_t w = 0; w < n_chunk; ++w) {
    const std::size_t start = w * n_ctx;
    for (std::size_t j = 0; j < n_ctx; ++j) {
      const float *logits = model.forward(tokens[start + j], static_cast<std::int32_t>(j));
      if (j < score_lo || j + 1 >= n_ctx)
        continue;
      float max_logit = logits[0];
      for (std::size_t i = 1; i < vocab; ++i)
        max_logit = std::max(max_logit, logits[i]);
      double sum_exp = 0.0;
      for (std::size_t i = 0; i < vocab; ++i)
        sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
      const double logsumexp = static_cast<double>(max_logit) + std::log(sum_exp);
      const std::int32_t target = tokens[start + j + 1];
      nll += logsumexp - static_cast<double>(logits[target]);
      ++count;
    }
    std::fprintf(stderr, "chunk %zu/%zu ppl %.4f\n", w + 1, n_chunk,
                 std::exp(nll / static_cast<double>(count)));
  }

  std::printf("PPL = %.6f  tokens %zu  chunks %zu\n", std::exp(nll / static_cast<double>(count)),
              count, n_chunk);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  auto parsed = dbinfer::cli::parse_args(argc, argv);
  if (!parsed) {
    std::fprintf(stderr, "error: %s\n", parsed.error().message.c_str());
    std::fputs(dbinfer::cli::usage(argv[0]).c_str(), stderr);
    return 2;
  }
  const dbinfer::cli::CliOptions &opts = *parsed;

  auto loaded = dbinfer::gguf::load(opts.model_path);
  if (!loaded) {
    std::fprintf(stderr, "error: load gguf: %s\n",
                 dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }
  auto mret = dbinfer::model::Model::load(*loaded);
  if (!mret) {
    std::fprintf(stderr, "error: load model: %s\n", dbinfer::gguf::to_string(mret.error()).c_str());
    return 1;
  }
  auto tret = dbinfer::tokenizer::Tokenizer::from_gguf(*loaded);
  if (!tret) {
    std::fprintf(stderr, "error: tokenizer: %s\n", dbinfer::gguf::to_string(tret.error()).c_str());
    return 1;
  }
  dbinfer::model::Model &model = *mret;
  const dbinfer::tokenizer::Tokenizer &tok = *tret;

  if (opts.threads > 0)
    dbinfer::tensor::configure_thread_count(static_cast<std::size_t>(opts.threads));

  if (!opts.perplexity_path.empty())
    return run_perplexity(model, tok, opts);

  std::vector<std::int32_t> history = tok.encode(opts.prompt, /*add_special=*/false);
  if (history.empty()) {
    std::fprintf(stderr, "error: prompt tokenized to zero tokens\n");
    return 1;
  }

  dbinfer::sample::Sampler sampler(opts.params);

  const std::size_t vocab = model.config().vocab_size;
  std::int32_t pos = 0;
  const float *logits = nullptr;

  // prefill: run the whole prompt through the model one token at a time so
  // the KV cache is populated for every prompt position; only the final
  // token's logits are kept, to pick the first generated token from.
  for (std::size_t i = 0; i < history.size(); ++i)
    logits = model.forward(history[i], pos++);

  // decode: sample one token from the current logits, then feed it back in
  // to get the next position's logits, until eos or the -n budget runs out.
  const std::int32_t eos = tok.eos_id();
  for (int t = 0; t < opts.n; ++t) {
    std::int32_t next_tok = sampler.sample(logits, vocab, history);
    if (next_tok == eos)
      break;
    history.push_back(next_tok);
    if (opts.print_ids) {
      std::printf("%d\n", next_tok);
    } else {
      std::int32_t one[1] = {next_tok};
      std::string piece = tok.decode(std::span<const std::int32_t>(one, 1));
      std::fputs(piece.c_str(), stdout);
      std::fflush(stdout);
    }
    logits = model.forward(next_tok, pos++);
  }
  if (!opts.print_ids)
    std::fputc('\n', stdout);

  return 0;
}
