#include "tensor/thread_pool.hpp"

#include <cstddef>
#include <cstdio>
#include <vector>

#include "tensor/cpu.hpp"
#include "test_util.hpp"

// every index in [0,n) is covered by exactly one range, for a spread of thread
// counts, sizes, and alignments including the n<align grain-guard case.

namespace {

using dbinfer::test::g_failures;

void run_case(std::size_t count, std::size_t n, std::size_t align) {
  dbinfer::tensor::ThreadPool pool(count);
  std::vector<int> slots(n, 0);

  dbinfer::tensor::parallel_for(pool, n, align, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) slots[i] += 1;
  });

  std::size_t bad = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (slots[i] != 1) ++bad;
  if (bad != 0) {
    std::printf("FAIL count=%zu n=%zu align=%zu  %zu slots not written once\n", count, n, align,
                bad);
    ++g_failures;
  }
}

}  // namespace

int main() {
  const std::size_t P = dbinfer::tensor::p_core_count();
  const std::size_t counts[] = {1, 2, 7, P};
  const std::size_t ns[] = {0, 1, 2, 3, 16, 17, 63, 64, 65, 1000, 4864};
  const std::size_t aligns[] = {1, 2, 3, 64};

  for (std::size_t count : counts)
    for (std::size_t n : ns)
      for (std::size_t align : aligns) run_case(count, n, align);

  std::printf("P=%zu\n", P);
  return dbinfer::test::summary();
}
