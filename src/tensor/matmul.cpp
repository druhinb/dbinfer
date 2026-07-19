#include "tensor/matmul.hpp"

#include "tensor/cpu.hpp"
#include "tensor/dequant.hpp"
#include "tensor/matmul_neon.hpp"
#include "tensor/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#ifdef DBINFER_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

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
      acc = std::fma(f16_to_f32(row[i]), x[i], acc);
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

// Q5_0 and the k-quants dequant the weight to fp32 scratch then dot with fp32
// x. no activation quantization, so these share no kernel with Q8_0/Q4_0.
void matvec_q5_0_scalar(const std::byte *W, const float *x, float *y, std::size_t out,
                        std::size_t in) {
  thread_local std::vector<float> scratch;
  scratch.resize(kBlockSize);
  const std::size_t nblocks = in / kBlockSize;
  const std::size_t row_bytes = nblocks * sizeof(BlockQ5_0);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = 0; b < nblocks; ++b) {
      dequant_row_q5_0(row + b * sizeof(BlockQ5_0), kBlockSize, scratch.data());
      const float *xb = x + b * kBlockSize;
      for (std::size_t k = 0; k < kBlockSize; ++k)
        acc += scratch[k] * xb[k];
    }
    y[o] = acc;
  }
}

void matvec_q4_k_scalar(const std::byte *W, const float *x, float *y, std::size_t out,
                        std::size_t in) {
  thread_local std::vector<float> scratch;
  scratch.resize(kSuperBlockSize);
  const std::size_t nsb = in / kSuperBlockSize;
  const std::size_t row_bytes = nsb * sizeof(BlockQ4_K);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t s = 0; s < nsb; ++s) {
      dequant_row_q4_k(row + s * sizeof(BlockQ4_K), kSuperBlockSize, scratch.data());
      const float *xb = x + s * kSuperBlockSize;
      for (std::size_t k = 0; k < kSuperBlockSize; ++k)
        acc += scratch[k] * xb[k];
    }
    y[o] = acc;
  }
}

void matvec_q6_k_scalar(const std::byte *W, const float *x, float *y, std::size_t out,
                        std::size_t in) {
  thread_local std::vector<float> scratch;
  scratch.resize(kSuperBlockSize);
  const std::size_t nsb = in / kSuperBlockSize;
  const std::size_t row_bytes = nsb * sizeof(BlockQ6_K);
  for (std::size_t o = 0; o < out; ++o) {
    const std::byte *row = W + o * row_bytes;
    float acc = 0.0f;
    for (std::size_t s = 0; s < nsb; ++s) {
      dequant_row_q6_k(row + s * sizeof(BlockQ6_K), kSuperBlockSize, scratch.data());
      const float *xb = x + s * kSuperBlockSize;
      for (std::size_t k = 0; k < kSuperBlockSize; ++k)
        acc += scratch[k] * xb[k];
    }
    y[o] = acc;
  }
}

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

void matvec_q5_0(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in) {
  matvec_q5_0_scalar(W, x, y, out, in);
}

void matvec_q4_k(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in) {
  matvec_q4_k_scalar(W, x, y, out, in);
}

void matvec_q6_k(const std::byte *W, const float *x, float *y, std::size_t out, std::size_t in) {
  matvec_q6_k_scalar(W, x, y, out, in);
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

// only Q8_0/Q4_0 dot against a pre-quantized activation; every other dtype
// reads the fp32 x directly.
bool needs_activation_quant(gguf::GgmlType t) {
  return t == gguf::GgmlType::Q8_0 || t == gguf::GgmlType::Q4_0;
}

// compute rows [lo, hi) of weight w (shape [*, in]) into y[lo..hi), re-basing
// the weight pointer so each kernel call sees a contiguous [0, hi-lo) block.
// xq is the shared pre-quantized activation, used only by the Q8_0/Q4_0 path.
void matvec_rows(QuantMatrix w, const float *x, const BlockQ8_0 *xq, float *y, std::size_t lo,
                 std::size_t hi, std::size_t in) {
  if (lo >= hi)
    return;
  const std::size_t rows = hi - lo;
  switch (w.type) {
  case gguf::GgmlType::Q8_0:
  case gguf::GgmlType::Q4_0: {
    const bool q8 = w.type == gguf::GgmlType::Q8_0;
    const std::size_t row_bytes = (in / kBlockSize) * (q8 ? sizeof(BlockQ8_0) : sizeof(BlockQ4_0));
    const QuantDispatch &d = quant_dispatch();
    (q8 ? d.q8 : d.q4)(w.data + lo * row_bytes, xq, y + lo, rows, in);
    return;
  }
  case gguf::GgmlType::Q5_0: {
    const std::size_t row_bytes = (in / kBlockSize) * sizeof(BlockQ5_0);
    matvec_q5_0_scalar(w.data + lo * row_bytes, x, y + lo, rows, in);
    return;
  }
  case gguf::GgmlType::Q4_K:
  case gguf::GgmlType::Q6_K: {
    const bool q4k = w.type == gguf::GgmlType::Q4_K;
    const std::size_t row_bytes =
        (in / kSuperBlockSize) * (q4k ? sizeof(BlockQ4_K) : sizeof(BlockQ6_K));
    (q4k ? matvec_q4_k_scalar : matvec_q6_k_scalar)(w.data + lo * row_bytes, x, y + lo, rows, in);
    return;
  }
  case gguf::GgmlType::F16: {
    const std::size_t row_bytes = in * sizeof(std::uint16_t);
    matvec_f16_view(w.data + lo * row_bytes, x, y + lo, rows, in);
    return;
  }
  default: {
    const std::size_t row_bytes = in * sizeof(float);
    matvec_f32_view(w.data + lo * row_bytes, x, y + lo, rows, in);
    return;
  }
  }
}

} // namespace

void matvec_quant(QuantMatrix w, const float *x, float *y, std::size_t out, std::size_t in) {
  ThreadPool &pool = thread_pool();
  // quantize once on the caller thread, then share xq read-only across ranges.
  const BlockQ8_0 *xq = needs_activation_quant(w.type) ? quantize_activation(x, in) : nullptr;
  parallel_for(pool, out, kRowTile,
               [&](std::size_t rb, std::size_t re) { matvec_rows(w, x, xq, y, rb, re, in); });
}

void matvec_quant_fused(const MatvecJob *jobs, std::size_t njobs, const float *x, std::size_t in) {
  if (njobs == 0)
    return;
  ThreadPool &pool = thread_pool();

  // one quantization of x shared by every Q8_0/Q4_0 job. jobs share x and in,
  // so their pre-quantized activation is identical to the unfused path.
  bool any_quant = false;
  std::size_t total = 0;
  for (std::size_t j = 0; j < njobs; ++j) {
    any_quant = any_quant || needs_activation_quant(jobs[j].w.type);
    total += jobs[j].out;
  }
  const BlockQ8_0 *xq = any_quant ? quantize_activation(x, in) : nullptr;

  // one barrier over the concatenated [0, total) output-row space. a range may
  // span job boundaries, so clip it to each job and re-base into that job's y.
  parallel_for(pool, total, kRowTile, [&](std::size_t rb, std::size_t re) {
    std::size_t off = 0;
    for (std::size_t j = 0; j < njobs; ++j) {
      const std::size_t o0 = off;
      const std::size_t o1 = off + jobs[j].out;
      const std::size_t lo = std::max(rb, o0);
      const std::size_t hi = std::min(re, o1);
      if (lo < hi)
        matvec_rows(jobs[j].w, x, xq, jobs[j].y, lo - o0, hi - o0, in);
      off = o1;
    }
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

void matmul_quant(QuantMatrix w, const float *A, float *C, std::size_t m, std::size_t out,
                  std::size_t in) {
  // reuse the per-row kernel unchanged. sharing the weight dequant across rows
  // lets the compiler contract one reduction to fma and not the other, which
  // moved the f16 path by ~1e-5. per-row keeps a chunk row bit-identical.
  for (std::size_t r = 0; r < m; ++r)
    matvec_quant(w, A + r * in, C + r * out, out, in);
}

void matmul_accel(const float *A, QuantMatrix w, float *C, std::size_t m, std::size_t out,
                  std::size_t in) {
  thread_local std::vector<float> wdense;
  wdense.resize(out * in);
  for (std::size_t o = 0; o < out; ++o)
    dequant_row(w, o, in, wdense.data() + o * in);
#ifdef DBINFER_HAVE_ACCELERATE
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(out),
              static_cast<int>(in), 1.0f, A, static_cast<int>(in), wdense.data(),
              static_cast<int>(in), 0.0f, C, static_cast<int>(out));
#else
  matmul(A, wdense.data(), C, m, out, in);
#endif
}

} // namespace dbinfer::tensor
