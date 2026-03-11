#include "tensor/matmul.hpp"

#include "tensor/dequant.hpp"

#include <array>
#include <cstring>

namespace dbinfer::tensor {

void matvec(const float *W, const float *x, float *y, std::size_t out, std::size_t in) {
  for (std::size_t o = 0; o < out; ++o) {
    float acc = 0.0f;
    const float *row = W + o * in;
    for (std::size_t i = 0; i < in; ++i)
      acc += row[i] * x[i];
    y[o] = acc;
  }
}

void matvec_f16(const std::uint16_t *W, const float *x, float *y, std::size_t out, std::size_t in) {
  for (std::size_t o = 0; o < out; ++o) {
    float acc = 0.0f;
    const std::uint16_t *row = W + o * in;
    for (std::size_t i = 0; i < in; ++i)
      acc += f16_to_f32(row[i]) * x[i];
    y[o] = acc;
  }
}

void matvec_q8_0(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in) {
  const std::size_t nblocks = in / kBlockSize;
  const std::size_t row_bytes = nblocks * sizeof(BlockQ8_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const std::byte *blk = row + b * sizeof(BlockQ8_0);
      std::uint16_t d_bits = 0;
      std::memcpy(&d_bits, blk, sizeof(d_bits));
      const float d = f16_to_f32(d_bits);
      const float *xb = x + b * kBlockSize;
      for (std::size_t i = 0; i < kBlockSize; ++i)
        acc += d * static_cast<float>(static_cast<std::int8_t>(blk[2 + i])) * xb[i];
    }
    y[o] = acc;
  }
}

namespace {

void matvec_f32_view(const std::byte *W, const float *x, float *y, std::size_t out,
                     std::size_t in) {
  // tensor data is aligned to the gguf tensor alignment, so a float view holds.
  matvec(reinterpret_cast<const float *>(W), x, y, out, in);
}

void matvec_f16_view(const std::byte *W, const float *x, float *y, std::size_t out,
                     std::size_t in) {
  matvec_f16(reinterpret_cast<const std::uint16_t *>(W), x, y, out, in);
}

struct QuantKernel {
  gguf::GgmlType type;
  void (*fn)(const std::byte *, const float *, float *, std::size_t, std::size_t);
};

constexpr std::array<QuantKernel, 3> kQuantKernels{{
    {gguf::GgmlType::F32, matvec_f32_view},
    {gguf::GgmlType::F16, matvec_f16_view},
    {gguf::GgmlType::Q8_0, matvec_q8_0},
}};

} // namespace

void matvec_quant(QuantMatrix w, const float *x, float *y, std::size_t out, std::size_t in) {
  for (const auto &k : kQuantKernels)
    if (k.type == w.type) {
      k.fn(w.data, x, y, out, in);
      return;
    }
  __builtin_unreachable(); // load validated w.type is one of the table entries
}

void matmul(const float *A, const float *W, float *C, std::size_t m, std::size_t out,
            std::size_t in) {
  for (std::size_t r = 0; r < m; ++r) {
    const float *arow = A + r * in;
    float *crow = C + r * out;
    for (std::size_t o = 0; o < out; ++o) {
      float acc = 0.0f;
      const float *wrow = W + o * in;
      for (std::size_t i = 0; i < in; ++i)
        acc += arow[i] * wrow[i];
      crow[o] = acc;
    }
  }
}

} // namespace dbinfer::tensor
