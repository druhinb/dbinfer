#include "tensor/matmul.hpp"

#include "tensor/cpu.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul_neon.hpp"
#include "tensor/thread_pool.hpp"

#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

namespace dbinfer::tensor {

namespace {

// activations quantize once per matvec, reused across every output row.
const BlockQ8_0 *quantize_activation(const float *x, std::size_t in) {
  thread_local std::vector<BlockQ8_0> scratch;
  scratch.resize(in / kBlockSize);
  quantize_row_q8_0(x, in, reinterpret_cast<std::byte *>(scratch.data()));
  return scratch.data();
}

} // namespace

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

void matvec_q8_0_scalar(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                        std::size_t in) {
  const std::size_t nblocks = in / kBlockSize;
  const std::size_t row_bytes = nblocks * sizeof(BlockQ8_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const std::byte *blk = row + b * sizeof(BlockQ8_0);
      std::uint16_t dw_bits = 0;
      std::memcpy(&dw_bits, blk, sizeof(dw_bits));
      std::int32_t sumi = 0;
      for (std::size_t i = 0; i < kBlockSize; ++i)
        sumi += static_cast<std::int32_t>(static_cast<std::int8_t>(blk[2 + i])) *
                static_cast<std::int32_t>(xq[b].qs[i]);
      acc += f16_to_f32(dw_bits) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

void matvec_q4_0_scalar(const std::byte *W, const BlockQ8_0 *xq, float *y, std::size_t out,
                        std::size_t in) {
  const std::size_t nblocks = in / kBlockSize;
  const std::size_t row_bytes = nblocks * sizeof(BlockQ4_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const std::byte *blk = row + b * sizeof(BlockQ4_0);
      std::uint16_t dw_bits = 0;
      std::memcpy(&dw_bits, blk, sizeof(dw_bits));
      const std::int8_t *qx = xq[b].qs;
      std::int32_t sumi = 0;
      for (std::size_t j = 0; j < 16; ++j) {
        const std::uint8_t q = static_cast<std::uint8_t>(blk[2 + j]);
        sumi += (static_cast<std::int32_t>(q & 0x0Fu) - 8) * static_cast<std::int32_t>(qx[j]);
        sumi += (static_cast<std::int32_t>(q >> 4) - 8) * static_cast<std::int32_t>(qx[j + 16]);
      }
      acc += f16_to_f32(dw_bits) * f16_to_f32(xq[b].d) * static_cast<float>(sumi);
    }
    y[o] = acc;
  }
}

namespace {

using QuantDot = void (*)(const std::byte *, const BlockQ8_0 *, float *, std::size_t, std::size_t);

struct QuantDispatch {
  QuantDot q8;
  QuantDot q4;
};

// resolved once from runtime detection; scalar stays the reference fallback.
// m=1 decode is compute-bound at this model size, so i8mm outruns sdot by ~29%
// (BENCH.log). DBINFER_QUANT_KERNEL forces a path for benchmarking.
const QuantDispatch &quant_dispatch() {
  static const QuantDispatch d = [] {
    const CpuFeatures &f = cpu_features();
    const char *sel = std::getenv("DBINFER_QUANT_KERNEL");
    const std::string_view k = sel != nullptr ? sel : "";
    if (k == "scalar")
      return QuantDispatch{matvec_q8_0_scalar, matvec_q4_0_scalar};
    if (k == "sdot" && f.dotprod)
      return QuantDispatch{matvec_q8_0_sdot, matvec_q4_0_sdot};
    if (f.i8mm)
      return QuantDispatch{matvec_q8_0_i8mm, matvec_q4_0_i8mm};
    if (f.dotprod)
      return QuantDispatch{matvec_q8_0_sdot, matvec_q4_0_sdot};
    return QuantDispatch{matvec_q8_0_scalar, matvec_q4_0_scalar};
  }();
  return d;
}

} // namespace

void matvec_q8_0(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in) {
  quant_dispatch().q8(W, quantize_activation(x, in), y, out, in);
}

void matvec_q4_0(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in) {
  quant_dispatch().q4(W, quantize_activation(x, in), y, out, in);
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

// even row starts keep each i8mm two-row smmla tile inside one range.
constexpr std::size_t kRowTile = 2;

} // namespace

void matvec_quant(QuantMatrix w, const float *x, float *y, std::size_t out, std::size_t in) {
  ThreadPool &pool = thread_pool();

  if (w.type == gguf::GgmlType::Q8_0 || w.type == gguf::GgmlType::Q4_0) {
    const bool q8 = w.type == gguf::GgmlType::Q8_0;
    // quantize once on the caller thread, then share xq read-only across ranges.
    const BlockQ8_0 *xq = quantize_activation(x, in);
    const QuantDispatch &d = quant_dispatch();
    const QuantDot kern = q8 ? d.q8 : d.q4;
    const std::size_t row_bytes = (in / kBlockSize) * (q8 ? sizeof(BlockQ8_0) : sizeof(BlockQ4_0));
    parallel_for(pool, out, kRowTile, [&](std::size_t rb, std::size_t re) {
      kern(w.data + rb * row_bytes, xq, y + rb, re - rb, in);
    });
    return;
  }

  const bool f16 = w.type == gguf::GgmlType::F16;
  const std::size_t row_bytes = in * (f16 ? sizeof(std::uint16_t) : sizeof(float));
  const auto kern = f16 ? matvec_f16_view : matvec_f32_view;
  parallel_for(pool, out, kRowTile, [&](std::size_t rb, std::size_t re) {
    kern(w.data + rb * row_bytes, x, y + rb, re - rb, in);
  });
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
