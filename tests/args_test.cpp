
#include "args.hpp"
#include "sample/sample.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int g_failures = 0;
void check(bool ok, const char *what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok)
    ++g_failures;
}

using dbinfer::cli::CliOptions;
using dbinfer::cli::parse_args;

std::expected<CliOptions, dbinfer::sample::Error> run(const std::vector<const char *> &args) {
  return parse_args(static_cast<int>(args.size()), args.data());
}

bool contains(const std::string &haystack, const char *needle) {
  return haystack.find(needle) != std::string::npos;
}
} // namespace

int main() {
  using dbinfer::sample::kDefaultSeed;
  using dbinfer::sample::validate;

  {
    auto r = run({"engine", "-m", "model.gguf", "-p", "hi"});
    bool ok = r.has_value();
    if (ok) {
      const auto &p = r->params;
      ok = p.temperature == 0.0f && p.top_k == 0 && p.top_p == 1.0f && p.min_p == 0.0f &&
           p.repeat_penalty == 1.0f && p.freq_penalty == 0.0f && p.presence_penalty == 0.0f &&
           p.penalty_last_n == 64 && p.seed == kDefaultSeed && r->n == 128 && !r->print_ids;
    }
    check(ok, "defaults with just -m/-p");
  }

  {
    auto r = run({"engine",     "-m",
                  "model.gguf", "-p",
                  "hi",         "--top-k",
                  "40",         "--top-p",
                  "0.9",        "--min-p",
                  "0.05",       "--repeat-penalty",
                  "1.1",        "--freq-penalty",
                  "0.2",        "--presence-penalty",
                  "0.3",        "--penalty-last-n",
                  "128",        "-s",
                  "123",        "--seed",
                  "123",        "--temp",
                  "0.8",        "--print-ids",
                  "-n",         "32"});
    bool ok = r.has_value();
    if (ok) {
      const auto &p = r->params;
      ok = p.top_k == 40 && p.top_p == 0.9f && p.min_p == 0.05f && p.repeat_penalty == 1.1f &&
           p.freq_penalty == 0.2f && p.presence_penalty == 0.3f && p.penalty_last_n == 128 &&
           p.seed == 123ULL && p.temperature == 0.8f && r->print_ids && r->n == 32;
    }
    check(ok, "each flag parses into the right field");
  }

  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--top-p", "1.5"});
    check(!r.has_value() && contains(r.error().message, "--top-p") &&
              contains(r.error().message, "1.5"),
          "--top-p 1.5 out of range names flag and value");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--min-p", "-0.1"});
    check(!r.has_value() && contains(r.error().message, "--min-p") &&
              contains(r.error().message, "-0.1"),
          "--min-p -0.1 out of range names flag and value");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--temp", "-1"});
    check(!r.has_value() && contains(r.error().message, "--temp") &&
              contains(r.error().message, "-1"),
          "--temp -1 out of range names flag and value");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--repeat-penalty", "0"});
    check(!r.has_value() && contains(r.error().message, "--repeat-penalty") &&
              contains(r.error().message, "0"),
          "--repeat-penalty 0 out of range names flag and value");
  }

  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--temp", "abc"});
    check(!r.has_value() && contains(r.error().message, "--temp"), "non-numeric --temp names flag");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--temp", "nan"});
    check(!r.has_value() && contains(r.error().message, "--temp"), "--temp nan rejected");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--top-p", "inf"});
    check(!r.has_value() && contains(r.error().message, "--top-p"), "--top-p inf rejected");
  }

  {
    auto r = run({"engine", "-p", "h"});
    check(!r.has_value() && contains(r.error().message, "-m"), "missing -m names it");
  }
  {
    auto r = run({"engine", "-m", "m"});
    check(!r.has_value() && contains(r.error().message, "-p"), "missing -p names it");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "-n", "0"});
    check(!r.has_value(), "-n 0 rejected");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--threads", "4"});
    check(r.has_value() && r->threads == 4, "--threads 4 parses into threads");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--threads", "0"});
    check(!r.has_value() && contains(r.error().message, "--threads"), "--threads 0 rejected");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--threads", "foo"});
    check(!r.has_value() && contains(r.error().message, "--threads"),
          "non-numeric --threads rejected");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--nope"});
    check(!r.has_value() && contains(r.error().message, "--nope"), "unknown --nope rejected");
  }
  {
    auto r = run({"engine", "-m", "m", "-p", "h", "--top-k"});
    check(!r.has_value() && contains(r.error().message, "--top-k"),
          "--top-k with no value rejected");
  }

  {
    auto r = run({"engine",     "-m",
                  "model.gguf", "-p",
                  "hi",         "--temp",
                  "0.7",        "--top-k",
                  "40",         "--top-p",
                  "0.95",       "--min-p",
                  "0.02",       "--repeat-penalty",
                  "1.1",        "--freq-penalty",
                  "0.1",        "--presence-penalty",
                  "0.1",        "--penalty-last-n",
                  "256",        "-s",
                  "7",          "-n",
                  "16"});
    bool ok = r.has_value() && validate(r->params).has_value();
    check(ok, "full valid combo parses and validates");
  }

  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
