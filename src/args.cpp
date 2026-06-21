#include "args.hpp"

#include "try.hpp"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>

namespace dbinfer::cli {

namespace {

sample::Error expects_number(std::string_view flag, const char *val) {
  return sample::Error{std::string(flag) + " expects a number, got '" + val + "'"};
}

sample::Error out_of_range(std::string_view flag, const char *val) {
  return sample::Error{std::string(flag) + " value '" + val + "' is out of range"};
}

sample::Error must_be(std::string_view flag, const char *val, const char *constraint) {
  return sample::Error{std::string(flag) + " value '" + val + "' " + constraint};
}

std::expected<float, sample::Error> parse_float(std::string_view flag, const char *val) {
  errno = 0;
  char *end = nullptr;
  float v = std::strtof(val, &end);
  if (end == val || *end != '\0')
    return std::unexpected(expects_number(flag, val));

  // reject nan, infinite values
  if (errno == ERANGE || !std::isfinite(v))
    return std::unexpected(out_of_range(flag, val));
  return v;
}

std::expected<int, sample::Error> parse_int(std::string_view flag, const char *val) {
  errno = 0;
  char *end = nullptr;
  long v = std::strtol(val, &end, 10);
  if (end == val || *end != '\0')
    return std::unexpected(expects_number(flag, val));
  if (errno == ERANGE || v < INT_MIN || v > INT_MAX)
    return std::unexpected(out_of_range(flag, val));
  return static_cast<int>(v);
}

std::expected<std::uint64_t, sample::Error> parse_ull(std::string_view flag, const char *val) {
  errno = 0;
  char *end = nullptr;
  unsigned long long v = std::strtoull(val, &end, 10);
  if (end == val || *end != '\0')
    return std::unexpected(expects_number(flag, val));
  if (errno == ERANGE)
    return std::unexpected(out_of_range(flag, val));
  return static_cast<std::uint64_t>(v);
}

using Apply = std::expected<void, sample::Error> (*)(CliOptions &opts, std::string_view flag,
                                                     const char *val);

struct FlagSpec {
  std::string_view name;
  std::string_view alias; // empty when the flag has no second spelling
  bool takes_value;
  Apply apply;
};

// Captureless lambdas decay to function pointers, so this table stays
// allocation-free. Each row owns its own parse, range check, and assignment;
// there is no second validation pass elsewhere.
constexpr FlagSpec kFlags[] = {
    {"-m", "", true,
     [](CliOptions &o, std::string_view, const char *v) -> std::expected<void, sample::Error> {
       o.model_path = v;
       return {};
     }},
    {"-p", "", true,
     [](CliOptions &o, std::string_view, const char *v) -> std::expected<void, sample::Error> {
       o.prompt = v;
       return {};
     }},
    {"-n", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       int x = TRY(parse_int(f, v));
       if (x <= 0)
         return std::unexpected(must_be(f, v, "must be > 0"));
       o.n = x;
       return {};
     }},
    {"--threads", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       int x = TRY(parse_int(f, v));
       if (x <= 0)
         return std::unexpected(must_be(f, v, "must be > 0"));
       o.threads = x;
       return {};
     }},
    {"--temp", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       float x = TRY(parse_float(f, v));
       if (x < 0.0f)
         return std::unexpected(must_be(f, v, "must be >= 0"));
       o.params.temperature = x;
       return {};
     }},
    {"--top-k", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       o.params.top_k = TRY(parse_int(f, v));
       return {};
     }},
    {"--top-p", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       float x = TRY(parse_float(f, v));
       if (x < 0.0f || x > 1.0f)
         return std::unexpected(must_be(f, v, "must be in [0, 1]"));
       o.params.top_p = x;
       return {};
     }},
    {"--min-p", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       float x = TRY(parse_float(f, v));
       if (x < 0.0f || x > 1.0f)
         return std::unexpected(must_be(f, v, "must be in [0, 1]"));
       o.params.min_p = x;
       return {};
     }},
    {"--repeat-penalty", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       float x = TRY(parse_float(f, v));
       if (x <= 0.0f)
         return std::unexpected(must_be(f, v, "must be > 0"));
       o.params.repeat_penalty = x;
       return {};
     }},
    {"--freq-penalty", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       o.params.freq_penalty = TRY(parse_float(f, v));
       return {};
     }},
    {"--presence-penalty", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       o.params.presence_penalty = TRY(parse_float(f, v));
       return {};
     }},
    {"--penalty-last-n", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       o.params.penalty_last_n = TRY(parse_int(f, v));
       return {};
     }},
    {"-s", "--seed", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       o.params.seed = TRY(parse_ull(f, v));
       return {};
     }},
    {"--print-ids", "", false,
     [](CliOptions &o, std::string_view, const char *) -> std::expected<void, sample::Error> {
       o.print_ids = true;
       return {};
     }},
    {"--perplexity", "", true,
     [](CliOptions &o, std::string_view, const char *v) -> std::expected<void, sample::Error> {
       o.perplexity_path = v;
       return {};
     }},
    {"--ppl-chunks", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       int x = TRY(parse_int(f, v));
       if (x <= 0)
         return std::unexpected(must_be(f, v, "must be > 0"));
       o.ppl_chunks = x;
       return {};
     }},
    {"--kv-window", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       int x = TRY(parse_int(f, v));
       if (x < 0)
         return std::unexpected(must_be(f, v, "must be >= 0"));
       o.kv_window = x;
       return {};
     }},
    {"--kv-sink", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       int x = TRY(parse_int(f, v));
       if (x < 0)
         return std::unexpected(must_be(f, v, "must be >= 0"));
       o.kv_sink = x;
       return {};
     }},
    {"--ppl-stream", "", false,
     [](CliOptions &o, std::string_view, const char *) -> std::expected<void, sample::Error> {
       o.ppl_stream = true;
       return {};
     }},
    {"--kv-int8", "", false,
     [](CliOptions &o, std::string_view, const char *) -> std::expected<void, sample::Error> {
       o.kv_int8 = true;
       return {};
     }},
    {"--kv-cache-save", "", true,
     [](CliOptions &o, std::string_view, const char *v) -> std::expected<void, sample::Error> {
       o.kv_cache_save = v;
       return {};
     }},
    {"--kv-cache-load", "", true,
     [](CliOptions &o, std::string_view, const char *v) -> std::expected<void, sample::Error> {
       o.kv_cache_load = v;
       return {};
     }},
    {"--prefill-chunk", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       int x = TRY(parse_int(f, v));
       if (x <= 0)
         return std::unexpected(must_be(f, v, "must be > 0"));
       o.prefill_chunk = x;
       return {};
     }},
    {"--grammar", "", true,
     [](CliOptions &o, std::string_view, const char *v) -> std::expected<void, sample::Error> {
       o.grammar_path = v;
       return {};
     }},
    {"--grammar-stop", "", false,
     [](CliOptions &o, std::string_view, const char *) -> std::expected<void, sample::Error> {
       o.grammar_stop = true;
       return {};
     }},
    {"--draft-model", "", true,
     [](CliOptions &o, std::string_view, const char *v) -> std::expected<void, sample::Error> {
       o.draft_model = v;
       return {};
     }},
    {"--draft-k", "", true,
     [](CliOptions &o, std::string_view f, const char *v) -> std::expected<void, sample::Error> {
       int x = TRY(parse_int(f, v));
       if (x <= 0)
         return std::unexpected(must_be(f, v, "must be > 0"));
       o.draft_k = x;
       return {};
     }},
};

const FlagSpec *find_flag(std::string_view a) {
  for (const FlagSpec &f : kFlags) {
    if (a == f.name || (!f.alias.empty() && a == f.alias))
      return &f;
  }
  return nullptr;
}

} // namespace

std::expected<CliOptions, sample::Error> parse_args(int argc, const char *const *argv) {
  CliOptions opts;
  // CLI default is greedy so flag-less behavior stays byte-identical to the
  // legacy binary; SamplerParams itself defaults temperature to 1.0 for the
  // library API.
  opts.params.temperature = 0.0f;

  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    const FlagSpec *spec = find_flag(a);
    if (spec == nullptr)
      return std::unexpected(sample::Error{"unknown argument " + std::string(a)});

    const char *val = nullptr;
    if (spec->takes_value) {
      if (i + 1 >= argc)
        return std::unexpected(sample::Error{std::string(a) + " requires an argument"});
      val = argv[++i];
    }
    if (auto ok = spec->apply(opts, a, val); !ok)
      return std::unexpected(ok.error());
  }

  if (opts.model_path.empty())
    return std::unexpected(sample::Error{"-m <model.gguf> is required"});
  if (opts.perplexity_path.empty() && opts.prompt.empty())
    return std::unexpected(sample::Error{"-p <prompt> is required"});

  // dense pos runs past context_length and overruns the fixed cache. the
  // stream gate needs the ring anyway.
  if (opts.ppl_stream && opts.kv_window <= 0)
    return std::unexpected(sample::Error{"--ppl-stream requires --kv-window > 0"});

  const bool kv_cache_io = !opts.kv_cache_save.empty() || !opts.kv_cache_load.empty();
  if (kv_cache_io && (opts.kv_window > 0 || opts.kv_int8))
    return std::unexpected(
        sample::Error{"--kv-cache-save/--kv-cache-load require the dense fp32 cache"});
  if (kv_cache_io && !opts.perplexity_path.empty())
    return std::unexpected(
        sample::Error{"--kv-cache-save/--kv-cache-load do not apply to perplexity"});

  if (!opts.draft_model.empty()) {
    if (!opts.perplexity_path.empty())
      return std::unexpected(sample::Error{"--draft-model does not apply to perplexity"});
    if (!opts.grammar_path.empty())
      return std::unexpected(sample::Error{"--draft-model does not support --grammar"});
    if (opts.kv_window > 0 || opts.kv_int8 || kv_cache_io)
      return std::unexpected(sample::Error{"--draft-model requires the dense fp32 cache"});
    if (opts.params.temperature != 0.0f)
      return std::unexpected(sample::Error{"--draft-model is greedy only (temp 0)"});
    if (opts.params.repeat_penalty != 1.0f || opts.params.freq_penalty != 0.0f ||
        opts.params.presence_penalty != 0.0f)
      return std::unexpected(sample::Error{"--draft-model does not support sampling penalties"});
  }

  if (auto ok = sample::validate(opts.params); !ok)
    return std::unexpected(ok.error());

  return opts;
}

std::string usage(const char *argv0) {
  std::string s = "usage: ";
  s += argv0;
  s += " -m <model.gguf> -p <prompt> [options]\n";
  s += "  -m <path>               GGUF model file (required)\n";
  s += "  -p <text>               prompt (required)\n";
  s += "  -n <int>                tokens to generate (default 128)\n";
  s += "  --threads <int>         worker threads, 0 = auto (default auto)\n";
  s += "  --temp <float>          temperature, 0 = greedy (default 0)\n";
  s += "  --top-k <int>           top-k cutoff, 0 = off (default 0)\n";
  s += "  --top-p <float>         top-p nucleus, [0,1] (default 1)\n";
  s += "  --min-p <float>         min-p cutoff, [0,1] (default 0)\n";
  s += "  --repeat-penalty <f>    repetition penalty, >0 (default 1)\n";
  s += "  --freq-penalty <f>      frequency penalty (default 0)\n";
  s += "  --presence-penalty <f>  presence penalty (default 0)\n";
  s += "  --penalty-last-n <int>  penalty window (default 64)\n";
  s += "  -s, --seed <uint>       RNG seed\n";
  s += "  --print-ids             print token ids instead of text\n";
  s += "  --perplexity <path>     score perplexity over a text file (no -p)\n";
  s += "  --ppl-chunks <int>      limit perplexity to N windows (default all)\n";
  s += "  --ppl-stream            stream one continuous context, bucket by position\n";
  s += "  --kv-window <int>       ring-buffer window, 0 = dense cache (default 0)\n";
  s += "  --kv-sink <int>         pinned attention sink positions (default 4)\n";
  s += "  --kv-int8               store keys per-channel and values per-block as int8\n";
  s += "  --kv-cache-save <path>  dump the dense fp32 prefix cache after prefill\n";
  s += "  --kv-cache-load <path>  load a prefix cache and continue from it\n";
  s += "  --prefill-chunk <int>   prefill tokens per chunk, 1 = per token (default 1)\n";
  s += "  --grammar <path>        constrain decoding to a GBNF grammar file\n";
  s += "  --grammar-stop          stop when the grammar first reaches a complete state\n";
  s += "  --draft-model <path>    draft GGUF for greedy speculative decoding\n";
  s += "  --draft-k <int>         draft tokens proposed per verify round (default 4)\n";
  return s;
}

} // namespace dbinfer::cli
