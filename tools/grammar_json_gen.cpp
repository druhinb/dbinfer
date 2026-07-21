// generates many grammar-constrained completions for the JSON gate. loads the
// model once, then for each seed resets the KV cache, prefills the prompt, and
// decodes under the grammar until it reaches a complete state. each completion
// is written as one line to the output file for an external JSON validator.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "gguf/gguf.hpp"
#include "grammar/grammar.hpp"
#include "model/model.hpp"
#include "sample/sample.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

std::string read_file(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return {};
  std::string data;
  char buf[65536];
  std::size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) data.append(buf, got);
  std::fclose(f);
  return data;
}

const char* arg_value(int argc, char** argv, const char* flag, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = arg_value(argc, argv, "-m", nullptr);
  const char* grammar_path = arg_value(argc, argv, "-g", "grammars/json.gbnf");
  const char* out_path = arg_value(argc, argv, "-o", "build/grammar_json_out.jsonl");
  const char* prompt = arg_value(argc, argv, "-p", "Return one JSON object:\n");
  const int count = std::atoi(arg_value(argc, argv, "-c", "1000"));
  const int max_tokens = std::atoi(arg_value(argc, argv, "-n", "128"));
  const float temperature = static_cast<float>(std::atof(arg_value(argc, argv, "--temp", "1.0")));
  if (model_path == nullptr) {
    std::fprintf(stderr, "usage: %s -m <model.gguf> [-g grammar] [-o out] [-c count]\n", argv[0]);
    return 2;
  }

  auto loaded = dbinfer::gguf::load(model_path);
  if (!loaded) {
    std::fprintf(stderr, "load gguf: %s\n", dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }
  auto mret = dbinfer::model::Model::load(*loaded);
  if (!mret) {
    std::fprintf(stderr, "load model: %s\n", dbinfer::gguf::to_string(mret.error()).c_str());
    return 1;
  }
  auto tret = dbinfer::tokenizer::Tokenizer::from_gguf(*loaded);
  if (!tret) {
    std::fprintf(stderr, "tokenizer: %s\n", dbinfer::gguf::to_string(tret.error()).c_str());
    return 1;
  }
  dbinfer::model::Model& model = *mret;
  const dbinfer::tokenizer::Tokenizer& tok = *tret;

  const std::string gtext = read_file(grammar_path);
  auto grammar = dbinfer::grammar::Grammar::parse(gtext);
  if (!grammar) {
    std::fprintf(stderr, "grammar: %s\n", grammar.error().message.c_str());
    return 1;
  }

  const std::size_t vocab = model.config().vocab_size;
  const std::int32_t eos = tok.eos_id();
  std::vector<std::string> token_bytes(vocab);
  for (std::size_t id = 0; id < vocab; ++id) {
    const std::int32_t one[1] = {static_cast<std::int32_t>(id)};
    token_bytes[id] = tok.decode(std::span<const std::int32_t>(one, 1));
  }

  const std::vector<std::int32_t> prompt_ids = tok.encode(prompt, /*add_special=*/false);
  const float ninf = -std::numeric_limits<float>::infinity();

  std::FILE* out = std::fopen(out_path, "wb");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", out_path);
    return 1;
  }

  // the grammar mask tests every vocab token per step; feed() is const and
  // allocates locally, so the scan splits cleanly across threads.
  const std::size_t n_workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  auto mask_range = [&](const dbinfer::grammar::State& state, bool comp, std::span<float> logits,
                        std::size_t lo, std::size_t hi) {
    for (std::size_t id = lo; id < hi; ++id) {
      if (static_cast<std::int32_t>(id) == eos) {
        if (!comp) logits[id] = ninf;
        continue;
      }
      if (!grammar->feed(state, token_bytes[id])) logits[id] = ninf;
    }
  };

  std::vector<float> masked(vocab);
  int natural = 0;
  int forced = 0;
  for (int seed = 0; seed < count; ++seed) {
    model.reset_kv();
    dbinfer::sample::SamplerParams params;
    params.temperature = temperature;
    params.seed = static_cast<std::uint64_t>(seed);
    dbinfer::sample::Sampler sampler(params);
    dbinfer::grammar::State state = grammar->start();

    std::int32_t pos = 0;
    const float* logits = nullptr;
    for (std::int32_t id : prompt_ids) logits = model.forward(id, pos++);

    std::string text;
    std::vector<std::int32_t> history = prompt_ids;
    bool done = false;
    for (int t = 0; t < max_tokens; ++t) {
      std::copy(logits, logits + vocab, masked.begin());
      const bool comp = grammar->complete(state);
      std::vector<std::thread> workers;
      const std::size_t block = (vocab + n_workers - 1) / n_workers;
      for (std::size_t w = 0; w < n_workers; ++w) {
        const std::size_t lo = w * block;
        const std::size_t hi = std::min(lo + block, vocab);
        if (lo >= hi) break;
        workers.emplace_back(mask_range, std::cref(state), comp, std::span<float>(masked), lo, hi);
      }
      for (std::thread& wk : workers) wk.join();

      const std::int32_t next = sampler.sample(masked.data(), vocab, history);
      if (next == eos) break;
      if (auto adv = grammar->feed(state, token_bytes[static_cast<std::size_t>(next)]))
        state = std::move(*adv);
      history.push_back(next);
      const std::int32_t one[1] = {next};
      text += tok.decode(std::span<const std::int32_t>(one, 1));
      if (grammar->complete(state)) {
        done = true;
        break;
      }
      logits = model.forward(next, pos++);
    }

    if (done) {
      ++natural;
    } else if (auto suffix = grammar->complete_suffix(state)) {
      // the sampling budget ran out mid-value; close the structure along the
      // shortest grammar-valid path so the output is a complete document.
      text += *suffix;
      ++forced;
    }

    for (char& c : text)
      if (c == '\n' || c == '\r') c = ' ';
    std::fprintf(out, "%s\n", text.c_str());
    if ((seed + 1) % 50 == 0)
      std::fprintf(stderr, "generated %d/%d (natural %d, forced %d)\n", seed + 1, count, natural,
                   forced);
  }
  std::fclose(out);
  std::printf("wrote %d completions to %s (natural %d, forced-close %d)\n", count, out_path,
              natural, forced);
  return 0;
}
