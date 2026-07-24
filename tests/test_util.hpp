// shared test harness used by every ctest main() here. holds the failure
// counter, the PASS/FAIL printer, and the exit-code summary. golden-tensor
// helpers compile only when the build defines DBINFER_GOLDEN_DIR, so targets
// built without goldens can still include this header. every tolerance stays
// at the call site.
#ifndef DBINFER_TESTS_TEST_UTIL_HPP
#define DBINFER_TESTS_TEST_UTIL_HPP

#include <cstdio>

#ifdef DBINFER_GOLDEN_DIR
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>
#endif

namespace dbinfer::test {

inline int g_failures = 0;

inline void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

inline void check(bool ok, const char* what, double err) {
  std::printf("%s %-28s max_err=%.3e\n", ok ? "PASS" : "FAIL", what, err);
  if (!ok) ++g_failures;
}

// prints the tally and returns the process exit code
[[nodiscard]] inline int summary() {
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}

#ifdef DBINFER_GOLDEN_DIR

inline std::vector<float> load_bin(const char* name, std::size_t expect_n) {
  std::string path = std::string(DBINFER_GOLDEN_DIR) + "/" + name;

  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::printf("FAIL cannot open %s\n", path.c_str());
    ++g_failures;
    return {};
  }

  std::vector<float> v(expect_n);
  std::size_t got = expect_n == 0 ? 0 : std::fread(v.data(), sizeof(float), expect_n, f);
  std::fclose(f);

  if (got != expect_n) {
    std::printf("FAIL %s read %zu of %zu floats\n", name, got, expect_n);
    ++g_failures;
    return {};
  }

  return v;
}

inline double max_abs_err(std::span<const float> a, std::span<const float> b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return m;
}

#endif  // DBINFER_GOLDEN_DIR

}  // namespace dbinfer::test

#endif  // DBINFER_TESTS_TEST_UTIL_HPP
