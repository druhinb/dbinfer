
#include "args.hpp"

#include <string>
#include <vector>

#include "sample/sample.hpp"
#include "test_util.hpp"

namespace {

using dbinfer::cli::CliOptions;
using dbinfer::cli::parse_args;
using dbinfer::sample::kDefaultSeed;
using dbinfer::sample::validate;
using dbinfer::test::check;

std::expected<CliOptions, dbinfer::sample::Error> run(const std::vector<const char*>& args) {
  return parse_args(static_cast<int>(args.size()), args.data());
}

bool contains(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

void test_defaults_with_required_flags() {
  auto r = run({"engine", "-m", "model.gguf", "-p", "hi"});
  bool ok = r.has_value();
  if (ok) {
    const auto& p = r->params;
    ok = p.temperature == 0.0f && p.top_k == 0 && p.top_p == 1.0f && p.min_p == 0.0f &&
         p.repeat_penalty == 1.0f && p.freq_penalty == 0.0f && p.presence_penalty == 0.0f &&
         p.penalty_last_n == 64 && p.seed == kDefaultSeed && r->n == 128 && !r->print_ids;
  }
  check(ok, "defaults with just -m/-p");
}

void test_all_flags_parse_into_fields() {
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
    const auto& p = r->params;
    ok = p.top_k == 40 && p.top_p == 0.9f && p.min_p == 0.05f && p.repeat_penalty == 1.1f &&
         p.freq_penalty == 0.2f && p.presence_penalty == 0.3f && p.penalty_last_n == 128 &&
         p.seed == 123ULL && p.temperature == 0.8f && r->print_ids && r->n == 32;
  }
  check(ok, "each flag parses into the right field");
}

void test_out_of_range_numeric_flags() {
  auto top_p = run({"engine", "-m", "m", "-p", "h", "--top-p", "1.5"});
  check(!top_p.has_value() && contains(top_p.error().message, "--top-p") &&
            contains(top_p.error().message, "1.5"),
        "--top-p 1.5 out of range names flag and value");

  auto min_p = run({"engine", "-m", "m", "-p", "h", "--min-p", "-0.1"});
  check(!min_p.has_value() && contains(min_p.error().message, "--min-p") &&
            contains(min_p.error().message, "-0.1"),
        "--min-p -0.1 out of range names flag and value");

  auto temp = run({"engine", "-m", "m", "-p", "h", "--temp", "-1"});
  check(!temp.has_value() && contains(temp.error().message, "--temp") &&
            contains(temp.error().message, "-1"),
        "--temp -1 out of range names flag and value");

  auto repeat = run({"engine", "-m", "m", "-p", "h", "--repeat-penalty", "0"});
  check(!repeat.has_value() && contains(repeat.error().message, "--repeat-penalty") &&
            contains(repeat.error().message, "0"),
        "--repeat-penalty 0 out of range names flag and value");
}

void test_non_numeric_flag_values() {
  auto abc = run({"engine", "-m", "m", "-p", "h", "--temp", "abc"});
  check(!abc.has_value() && contains(abc.error().message, "--temp"),
        "non-numeric --temp names flag");

  auto nan = run({"engine", "-m", "m", "-p", "h", "--temp", "nan"});
  check(!nan.has_value() && contains(nan.error().message, "--temp"), "--temp nan rejected");

  auto inf = run({"engine", "-m", "m", "-p", "h", "--top-p", "inf"});
  check(!inf.has_value() && contains(inf.error().message, "--top-p"), "--top-p inf rejected");
}

void test_missing_required_flags() {
  auto no_m = run({"engine", "-p", "h"});
  check(!no_m.has_value() && contains(no_m.error().message, "-m"), "missing -m names it");

  auto no_p = run({"engine", "-m", "m"});
  check(!no_p.has_value() && contains(no_p.error().message, "-p"), "missing -p names it");
}

void test_n_zero_rejected() {
  auto r = run({"engine", "-m", "m", "-p", "h", "-n", "0"});
  check(!r.has_value(), "-n 0 rejected");
}

void test_threads_flag() {
  auto four = run({"engine", "-m", "m", "-p", "h", "--threads", "4"});
  check(four.has_value() && four->threads == 4, "--threads 4 parses into threads");

  auto zero = run({"engine", "-m", "m", "-p", "h", "--threads", "0"});
  check(!zero.has_value() && contains(zero.error().message, "--threads"), "--threads 0 rejected");

  auto foo = run({"engine", "-m", "m", "-p", "h", "--threads", "foo"});
  check(!foo.has_value() && contains(foo.error().message, "--threads"),
        "non-numeric --threads rejected");
}

void test_unknown_and_missing_value_flags() {
  auto nope = run({"engine", "-m", "m", "-p", "h", "--nope"});
  check(!nope.has_value() && contains(nope.error().message, "--nope"), "unknown --nope rejected");

  auto no_val = run({"engine", "-m", "m", "-p", "h", "--top-k"});
  check(!no_val.has_value() && contains(no_val.error().message, "--top-k"),
        "--top-k with no value rejected");
}

void test_ppl_stream_requires_kv_window() {
  auto r = run({"engine", "-m", "m", "--perplexity", "t", "--ppl-stream", "--kv-window", "256",
                "--kv-sink", "8"});
  check(r.has_value() && r->ppl_stream && r->kv_window == 256 && r->kv_sink == 8,
        "--ppl-stream/--kv-window/--kv-sink parse into fields");

  auto missing = run({"engine", "-m", "m", "--perplexity", "t", "--ppl-stream"});
  check(!missing.has_value() && contains(missing.error().message, "--ppl-stream") &&
            contains(missing.error().message, "--kv-window"),
        "--ppl-stream without --kv-window rejected");
}

void test_kv_window_and_sink_range() {
  auto window = run({"engine", "-m", "m", "-p", "h", "--kv-window", "-1"});
  check(!window.has_value() && contains(window.error().message, "--kv-window"),
        "--kv-window -1 rejected");

  auto sink = run({"engine", "-m", "m", "-p", "h", "--kv-sink", "-1"});
  check(!sink.has_value() && contains(sink.error().message, "--kv-sink"), "--kv-sink -1 rejected");
}

void test_full_combo_validates() {
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

}  // namespace

int main() {
  test_defaults_with_required_flags();
  test_all_flags_parse_into_fields();
  test_out_of_range_numeric_flags();
  test_non_numeric_flag_values();
  test_missing_required_flags();
  test_n_zero_rejected();
  test_threads_flag();
  test_unknown_and_missing_value_flags();
  test_ppl_stream_requires_kv_window();
  test_kv_window_and_sink_range();
  test_full_combo_validates();

  return dbinfer::test::summary();
}
