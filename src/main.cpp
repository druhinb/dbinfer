#include "args.hpp"
#include "dbmf/dbmf.hpp"
#include "gguf/gguf.hpp"
#include "grammar/grammar.hpp"
#include "model/model.hpp"
#include "model/speculative.hpp"
#include "sample/sample.hpp"
#include "tensor/thread_pool.hpp"
#include "tokenizer/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

// DBMF_VERIFY=1 recomputes every tensor's xxhash64 at load. off by default so
// the mmap-direct path stays fast.
dbinfer::dbmf::ReadOptions read_opts() {
  const char *v = std::getenv("DBMF_VERIFY");
  return {v != nullptr && v[0] == '1'};
}

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

// streaming perplexity: one continuous context with monotonically increasing
// position, no window resets. per-token NLL bucketed by position exposes
// whether a ring cache stays flat or diverges as position passes the window.
int run_stream_perplexity(dbinfer::model::Model &model, const dbinfer::tokenizer::Tokenizer &tok,
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
  if (tokens.size() < 2) {
    std::fprintf(stderr, "error: need at least 2 tokens, got %zu\n", tokens.size());
    return 1;
  }

  constexpr std::size_t bucket_size = 512;
  const std::size_t vocab = model.config().vocab_size;
  std::size_t limit = tokens.size();
  if (opts.ppl_chunks > 0 && static_cast<std::size_t>(opts.ppl_chunks) * bucket_size < limit)
    limit = static_cast<std::size_t>(opts.ppl_chunks) * bucket_size;

  model.reset_kv();
  std::vector<double> bucket_nll;
  std::vector<std::size_t> bucket_count;
  double nll = 0.0;
  std::size_t count = 0;
  for (std::size_t j = 0; j + 1 < limit; ++j) {
    const float *logits = model.forward(tokens[j], static_cast<std::int32_t>(j));
    float max_logit = logits[0];
    for (std::size_t i = 1; i < vocab; ++i)
      max_logit = std::max(max_logit, logits[i]);
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < vocab; ++i)
      sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
    const double logsumexp = static_cast<double>(max_logit) + std::log(sum_exp);
    const std::int32_t target = tokens[j + 1];
    const double token_nll = logsumexp - static_cast<double>(logits[target]);

    const std::size_t b = j / bucket_size;
    if (b >= bucket_nll.size()) {
      bucket_nll.resize(b + 1, 0.0);
      bucket_count.resize(b + 1, 0);
    }
    bucket_nll[b] += token_nll;
    ++bucket_count[b];
    nll += token_nll;
    ++count;
  }

  for (std::size_t b = 0; b < bucket_nll.size(); ++b)
    std::fprintf(stderr, "pos %zu-%zu ppl %.4f\n", b * bucket_size, (b + 1) * bucket_size - 1,
                 std::exp(bucket_nll[b] / static_cast<double>(bucket_count[b])));

  std::printf("stream PPL = %.6f  tokens %zu\n", std::exp(nll / static_cast<double>(count)), count);
  return 0;
}

// speculative decode: a quantized draft proposes tokens, the F16 target verifies
// them in one batched forward. greedy, so the output is token-identical to
// target-only decode. loads the draft, checks vocab and cache compatibility,
// prints the generated tokens, and logs the acceptance rate to stderr.
int run_speculative(dbinfer::model::Model &target, const dbinfer::tokenizer::Tokenizer &tok,
                    const dbinfer::cli::CliOptions &opts) {
  auto dloaded = dbinfer::dbmf::load_model(opts.draft_model, read_opts());
  if (!dloaded) {
    std::fprintf(stderr, "error: load draft model: %s\n",
                 dbinfer::gguf::to_string(dloaded.error()).c_str());
    return 1;
  }
  auto dret = dbinfer::model::Model::load(*dloaded);
  if (!dret) {
    std::fprintf(stderr, "error: load draft model: %s\n",
                 dbinfer::gguf::to_string(dret.error()).c_str());
    return 1;
  }
  dbinfer::model::Model &draft = *dret;

  if (draft.config().vocab_size != target.config().vocab_size) {
    std::fprintf(stderr, "error: draft/target vocab mismatch (%zu vs %zu)\n",
                 draft.config().vocab_size, target.config().vocab_size);
    return 1;
  }
  if (!target.kv_dense_f32() || !draft.kv_dense_f32()) {
    std::fprintf(stderr, "error: speculative decode requires the dense fp32 cache\n");
    return 1;
  }

  std::vector<std::int32_t> history = tok.encode(opts.prompt, /*add_special=*/false);
  if (history.empty()) {
    std::fprintf(stderr, "error: prompt tokenized to zero tokens\n");
    return 1;
  }

  dbinfer::model::SpecStats stats;
  std::vector<std::int32_t> gen = dbinfer::model::speculative_generate(
      target, draft, history, static_cast<std::size_t>(opts.draft_k),
      static_cast<std::size_t>(opts.n), tok.eos_id(), &stats);

  for (std::int32_t next_tok : gen) {
    if (opts.print_ids) {
      std::printf("%d\n", next_tok);
    } else {
      std::int32_t one[1] = {next_tok};
      std::string piece = tok.decode(std::span<const std::int32_t>(one, 1));
      std::fputs(piece.c_str(), stdout);
      std::fflush(stdout);
    }
  }
  if (!opts.print_ids)
    std::fputc('\n', stdout);

  const double rate =
      stats.proposed > 0 ? static_cast<double>(stats.accepted) / static_cast<double>(stats.proposed)
                         : 0.0;
  std::fprintf(stderr, "speculative: k=%d proposed=%zu accepted=%zu rate=%.4f rounds=%zu\n",
               opts.draft_k, stats.proposed, stats.accepted, rate, stats.rounds);
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

  auto loaded = dbinfer::dbmf::load_model(opts.model_path, read_opts());
  if (!loaded) {
    std::fprintf(stderr, "error: load model: %s\n",
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

  if (opts.kv_window > 0 || opts.kv_int8)
    model.configure_kv(
        {static_cast<std::size_t>(opts.kv_sink), static_cast<std::size_t>(opts.kv_window),
         opts.kv_int8 ? dbinfer::model::KvDtype::Int8 : dbinfer::model::KvDtype::F32});

  if (!opts.perplexity_path.empty())
    return opts.ppl_stream ? run_stream_perplexity(model, tok, opts)
                           : run_perplexity(model, tok, opts);

  if (!opts.draft_model.empty())
    return run_speculative(model, tok, opts);

  std::vector<std::int32_t> history = tok.encode(opts.prompt, /*add_special=*/false);
  if (history.empty()) {
    std::fprintf(stderr, "error: prompt tokenized to zero tokens\n");
    return 1;
  }

  dbinfer::sample::Sampler sampler(opts.params);

  const std::size_t vocab = model.config().vocab_size;
  std::int32_t pos = 0;
  const float *logits = nullptr;

  std::size_t prefill_start = 0;
  if (!opts.kv_cache_load.empty()) {
    auto loaded_prefix = model.load_kv_prefix(opts.kv_cache_load, history);
    if (!loaded_prefix) {
      std::fprintf(stderr, "error: kv cache load: %s\n",
                   dbinfer::gguf::to_string(loaded_prefix.error()).c_str());
      return 1;
    }
    prefill_start = *loaded_prefix;
    pos = static_cast<std::int32_t>(prefill_start);
    if (prefill_start >= history.size()) {
      std::fprintf(stderr, "error: loaded prefix covers the whole prompt; extend the prompt\n");
      return 1;
    }
  }

  // prefill: run the prompt through the model so the KV cache is populated for
  // every prompt position; only the final token's logits are kept, to pick the
  // first generated token from. a loaded prefix already holds positions
  // [0, prefill_start), so decode resumes from there. --prefill-chunk > 1
  // processes fixed-size chunks together on the dense fp32 cache.
  const std::size_t chunk = static_cast<std::size_t>(opts.prefill_chunk);
  std::vector<float> chunk_logits;
  if (chunk > 1 && model.kv_dense_f32()) {
    chunk_logits.resize(chunk * vocab);
    for (std::size_t i = prefill_start; i < history.size(); i += chunk) {
      const std::size_t n = std::min(chunk, history.size() - i);
      model.forward_chunk(history.data() + i, pos, n, chunk_logits.data());
      pos += static_cast<std::int32_t>(n);
      logits = chunk_logits.data() + (n - 1) * vocab;
    }
  } else {
    for (std::size_t i = prefill_start; i < history.size(); ++i)
      logits = model.forward(history[i], pos++);
  }

  if (!opts.kv_cache_save.empty()) {
    auto saved = model.save_kv_prefix(
        opts.kv_cache_save, std::span<const std::int32_t>(history.data(), history.size()));
    if (!saved) {
      std::fprintf(stderr, "error: kv cache save: %s\n",
                   dbinfer::gguf::to_string(saved.error()).c_str());
      return 1;
    }
  }

  const std::int32_t eos = tok.eos_id();

  std::optional<dbinfer::grammar::Matcher> matcher;
  std::vector<float> masked;
  if (!opts.grammar_path.empty()) {
    auto gtext = read_file(opts.grammar_path.c_str());
    if (!gtext) {
      std::fprintf(stderr, "error: cannot read grammar %s\n", opts.grammar_path.c_str());
      return 1;
    }
    auto grammar = dbinfer::grammar::Grammar::parse(*gtext);
    if (!grammar) {
      std::fprintf(stderr, "error: grammar: %s\n", grammar.error().message.c_str());
      return 1;
    }
    std::vector<std::string> token_bytes(vocab);
    for (std::size_t id = 0; id < vocab; ++id) {
      const std::int32_t one[1] = {static_cast<std::int32_t>(id)};
      token_bytes[id] = tok.decode(std::span<const std::int32_t>(one, 1));
    }
    matcher.emplace(std::move(*grammar), std::move(token_bytes), eos);
    masked.resize(vocab);
  }

  // decode: sample one token from the current logits, then feed it back in
  // to get the next position's logits, until eos or the -n budget runs out.
  // a grammar, when present, masks disallowed tokens to -inf before sampling
  // and then advances on the chosen token.
  for (int t = 0; t < opts.n; ++t) {
    if (matcher) {
      std::copy(logits, logits + vocab, masked.begin());
      matcher->mask(masked);
      logits = masked.data();
    }
    std::int32_t next_tok = sampler.sample(logits, vocab, history);
    if (next_tok == eos)
      break;
    if (matcher)
      matcher->accept(next_tok);
    history.push_back(next_tok);
    if (opts.print_ids) {
      std::printf("%d\n", next_tok);
    } else {
      std::int32_t one[1] = {next_tok};
      std::string piece = tok.decode(std::span<const std::int32_t>(one, 1));
      std::fputs(piece.c_str(), stdout);
      std::fflush(stdout);
    }
    if (matcher && opts.grammar_stop && matcher->complete())
      break;
    logits = model.forward(next_tok, pos++);
  }
  if (!opts.print_ids)
    std::fputc('\n', stdout);

  return 0;
}
