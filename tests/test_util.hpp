// Shared golden-tensor test helpers: load a reference .bin from
// DBINFER_GOLDEN_DIR, compute max abs error, and print PASS/FAIL in the
// convention every ctest main() here follows. Tolerances are never baked in
// here; each call site passes its own threshold.
#ifndef DBINFER_TESTS_TEST_UTIL_HPP
#define DBINFER_TESTS_TEST_UTIL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#ifndef DBINFER_GOLDEN_DIR
#error "DBINFER_GOLDEN_DIR must be defined by the build (path to tests/golden)"
#endif

namespace dbinfer::test {

inline int g_failures = 0;

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

inline void check(bool ok, const char* what, double err) {
  if (ok) {
    std::printf("PASS %-28s max_err=%.3e\n", what, err);
  } else {
    std::printf("FAIL %-28s max_err=%.3e\n", what, err);
    ++g_failures;
  }
}

}  // namespace dbinfer::test

#endif  // DBINFER_TESTS_TEST_UTIL_HPP
