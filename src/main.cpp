#include "args.hpp"
#include "gguf/gguf.hpp"
#include "model/model.hpp"
#include "sample/sample.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

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
