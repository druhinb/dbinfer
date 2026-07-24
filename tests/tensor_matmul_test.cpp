// ctest for the naive matvec/matmul kernels

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tensor/matmul.hpp"
#include "test_util.hpp"

namespace {

using dbinfer::test::check;

struct Lcg {
  std::uint64_t s;
  float next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint32_t hi = static_cast<std::uint32_t>(s >> 32);
    return static_cast<float>(hi) / 2147483648.0f - 1.0f;
  }
};

void test_matvec_rectangular() {
  const std::size_t out = 7, in = 13;
  Lcg rng{0x1234};
  std::vector<float> W(out * in), x(in), y(out);
  for (auto& w : W) w = rng.next();
  for (auto& v : x) v = rng.next();

  dbinfer::tensor::matvec(W.data(), x.data(), y.data(), out, in);

  double max_err = 0.0;
  for (std::size_t o = 0; o < out; ++o) {
    double ref = 0.0;
    for (std::size_t i = 0; i < in; ++i)
      ref += static_cast<double>(W[o * in + i]) * static_cast<double>(x[i]);
    max_err = std::max(max_err, std::fabs(static_cast<double>(y[o]) - ref));
  }
  std::printf("  matvec rectangular max_err=%.3e\n", max_err);
  check(max_err <= 1e-5, "matvec [7x13] vs double reference");
}

void test_matmul_multirow() {
  const std::size_t m = 5, out = 6, in = 9;
  Lcg rng{0xABCD};
  std::vector<float> A(m * in), W(out * in), C(m * out);
  for (auto& a : A) a = rng.next();
  for (auto& w : W) w = rng.next();

  dbinfer::tensor::matmul(A.data(), W.data(), C.data(), m, out, in);

  double max_err = 0.0;
  for (std::size_t r = 0; r < m; ++r)
    for (std::size_t o = 0; o < out; ++o) {
      double ref = 0.0;
      for (std::size_t i = 0; i < in; ++i)
        ref += static_cast<double>(A[r * in + i]) * static_cast<double>(W[o * in + i]);
      max_err = std::max(max_err, std::fabs(static_cast<double>(C[r * out + o]) - ref));
    }
  std::printf("  matmul multirow max_err=%.3e\n", max_err);
  check(max_err <= 1e-5, "matmul [5x9]*[6x9]^T vs double reference");
}

// NumPy reference:
//   import numpy as np
//   A = np.array([[1,2,3],[4,5,6]], dtype=np.float32)      # [2,3]
//   W = np.array([[1,0,-1],[2,2,2],[0,1,0],[1,1,1]], dtype=np.float32)  # [4,3]
//   C = A @ W.T
//   # C = [[-2, 12,  2,  6],
//   #      [-2, 30,  5, 15]]
//   x = np.array([1,2,3], dtype=np.float32)
//   y = W @ x  # = [-2, 12, 2, 6]
void test_numpy_fixture() {
  const float A[2 * 3] = {1, 2, 3, 4, 5, 6};
  const float W[4 * 3] = {1, 0, -1, 2, 2, 2, 0, 1, 0, 1, 1, 1};
  const float expect_C[2 * 4] = {-2, 12, 2, 6, -2, 30, 5, 15};
  const float x[3] = {1, 2, 3};
  const float expect_y[4] = {-2, 12, 2, 6};

  float C[2 * 4];
  dbinfer::tensor::matmul(A, W, C, 2, 4, 3);
  bool c_ok = true;
  for (std::size_t i = 0; i < 8; ++i) c_ok = c_ok && std::fabs(C[i] - expect_C[i]) <= 1e-5f;
  check(c_ok, "matmul matches NumPy fixture C = A @ W.T");

  float y[4];
  dbinfer::tensor::matvec(W, x, y, 4, 3);
  bool y_ok = true;
  for (std::size_t i = 0; i < 4; ++i) y_ok = y_ok && std::fabs(y[i] - expect_y[i]) <= 1e-5f;
  check(y_ok, "matvec matches NumPy fixture y = W @ x");
}

}  // namespace

int main() {
  test_matvec_rectangular();
  test_matmul_multirow();
  test_numpy_fixture();
  return dbinfer::test::summary();
}
