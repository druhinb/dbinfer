#include "tensor/ops.hpp"

#include <cmath>

namespace dbinfer::tensor {

void rmsnorm(const float* x, const float* weight, float eps, float* out, std::size_t rows,
             std::size_t dim) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * dim;
    float* outr = out + r * dim;
    float sumsq = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) sumsq += xr[i] * xr[i];
    float scale = 1.0f / std::sqrt(sumsq / static_cast<float>(dim) + eps);
    for (std::size_t i = 0; i < dim; ++i) outr[i] = xr[i] * scale * weight[i];
  }
}

void rope(float* x, const std::int32_t* positions, float theta, std::size_t n_heads,
          std::size_t seq, std::size_t head_dim) {
  const std::size_t half = head_dim / 2;
  for (std::size_t h = 0; h < n_heads; ++h) {
    for (std::size_t s = 0; s < seq; ++s) {
      float* v = x + (h * seq + s) * head_dim;
      const double p = static_cast<double>(positions[s]);
      for (std::size_t j = 0; j < half; ++j) {
        const double inv = std::pow(static_cast<double>(theta),
                                    -2.0 * static_cast<double>(j) / static_cast<double>(head_dim));
        const double c = std::cos(p * inv);
        const double sn = std::sin(p * inv);
        const float a = v[j];
        const float b = v[j + half];
        v[j] = static_cast<float>(a * c - b * sn);
        v[j + half] = static_cast<float>(b * c + a * sn);
      }
    }
  }
}

void softmax(const float* in, float* out, std::size_t n) {
  float m = in[0];
  for (std::size_t i = 1; i < n; ++i)
    if (in[i] > m) m = in[i];
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    float e = std::exp(in[i] - m);
    out[i] = e;
    sum += e;
  }
  float inv = 1.0f / sum;
  for (std::size_t i = 0; i < n; ++i) out[i] *= inv;
}

void silu(const float* in, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = in[i] / (1.0f + std::exp(-in[i]));
}

}  // namespace dbinfer::tensor
