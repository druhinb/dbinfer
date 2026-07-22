// ctest for rmsnorm/rope/softmax/silu

#include "tensor/ops.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "test_util.hpp"

namespace {

using dbinfer::test::check;
using dbinfer::test::g_failures;
using dbinfer::test::load_bin;
using dbinfer::test::max_abs_err;

void test_rmsnorm() {
  const std::size_t rows = 6, dim = 896;
  auto in = load_bin("rmsnorm_in.bin", rows * dim);
  auto weight = load_bin("rmsnorm_weight.bin", dim);
  auto ref = load_bin("rmsnorm_out.bin", rows * dim);
  if (in.empty() || weight.empty() || ref.empty()) return;
  std::vector<float> out(rows * dim);
  dbinfer::tensor::rmsnorm(in.data(), weight.data(), 1e-6f, out.data(), rows, dim);
  double err = max_abs_err(out, ref);
  check(err <= 1e-5, "rmsnorm vs rmsnorm_out", err);
}

void test_rope(const char* in_name, const char* out_name, std::size_t n_heads) {
  const std::size_t seq = 6, head_dim = 64;
  const std::size_t n = n_heads * seq * head_dim;
  auto x = load_bin(in_name, n);
  auto ref = load_bin(out_name, n);
  if (x.empty() || ref.empty()) return;
  const std::int32_t positions[6] = {0, 1, 2, 3, 4, 5};
  dbinfer::tensor::rope(x.data(), positions, 1e6f, n_heads, seq, head_dim);
  double err = max_abs_err(x, ref);
  check(err <= 1e-5, out_name, err);
}

void test_softmax() {
  const std::size_t rows = 8, cols = 512;
  auto in = load_bin("softmax_in.bin", rows * cols);
  auto ref = load_bin("softmax_out.bin", rows * cols);
  if (in.empty() || ref.empty()) return;
  std::vector<float> out(rows * cols);
  for (std::size_t r = 0; r < rows; ++r)
    dbinfer::tensor::softmax(in.data() + r * cols, out.data() + r * cols, cols);
  double err = max_abs_err(out, ref);
  check(err <= 1e-5, "softmax vs softmax_out", err);
}

void test_silu() {
  const std::size_t n = 4096;
  auto in = load_bin("silu_in.bin", n);
  auto ref = load_bin("silu_out.bin", n);
  if (in.empty() || ref.empty()) return;
  std::vector<float> out(n);
  dbinfer::tensor::silu(in.data(), out.data(), n);
  double err = max_abs_err(out, ref);
  check(err <= 1e-5, "silu vs silu_out", err);
}

}  // namespace

int main() {
  test_rmsnorm();
  test_rope("rope_q_in.bin", "rope_q_out.bin", 14);
  test_rope("rope_k_in.bin", "rope_k_out.bin", 2);
  test_softmax();
  test_silu();
  std::printf("---\n%d checks failed\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
