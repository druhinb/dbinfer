#ifndef DBINFER_ARGS_HPP
#define DBINFER_ARGS_HPP

#include "sample/sample.hpp"

#include <expected>
#include <string>

namespace dbinfer::cli {

struct CliOptions {
  std::string model_path;
  std::string prompt;
  std::string perplexity_path; // non-empty selects perplexity mode
  int ppl_chunks = 0;          // 0 means all windows
  int n = 128;
  bool print_ids = false;
  sample::SamplerParams params;
};

// parses argv into CliOptions, applying the CLI's greedy-by-default temperature
// and validating flag values (ranges, requiredness) before returning.
std::expected<CliOptions, sample::Error> parse_args(int argc, const char *const *argv);
std::string usage(const char *argv0);

} // namespace dbinfer::cli

#endif // DBINFER_ARGS_HPP
