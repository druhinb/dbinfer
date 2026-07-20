#include "backend/metal_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace dbinfer::backend {

namespace {

// Apple Silicon vm page. newBufferWithBytesNoCopy needs a page-aligned pointer,
// so DBMF's 16 KiB tensor alignment wraps zero-copy and a 32-byte gguf tensor
// falls back to a copy.
constexpr std::uintptr_t kPageSize = 16384;

bool page_aligned(const void *p) {
  return (reinterpret_cast<std::uintptr_t>(p) & (kPageSize - 1)) == 0;
}

std::size_t round_up_page(std::size_t n) {
  return (n + kPageSize - 1) & ~(kPageSize - 1);
}

// fast-math is disabled at compile time so the elementwise ops fuse and round
// like the scalar CPU path. mul_mat still calls fma explicitly, so its bitwise
// match with matvec_f16 is unaffected. rmsnorm and the attention dot sum
// sequentially per output, matching the CPU reduction; cos/sin/exp run in fp32
// and diverge from the double CPU transcendentals only within tolerance.
constexpr const char *kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// add_res folds the residual add into the matvec epilogue: when set, the output
// is acc + R[idx] rather than acc. fp32 add is commutative in IEEE, so this is
// bitwise identical to a separate residual add on the same acc. R is bound even
// when add_res is zero (aliasing C) so the buffer slot stays valid.
struct MatDims { uint m; uint n_out; uint n_in; uint add_res; };

// one simdgroup per output row, 32 lanes strided over the in-dimension. the
// per-lane partials combine through simd_sum, so the reduction order differs
// from matvec_f16's sequential fma. close within atol 1e-4, not bitwise.
kernel void mul_mat_f16(device const half   *W    [[buffer(0)]],
                        device const float  *A    [[buffer(1)]],
                        device       float  *C    [[buffer(2)]],
                        constant     MatDims &dims [[buffer(3)]],
                        device const float  *R    [[buffer(4)]],
                        uint2 tg   [[threadgroup_position_in_grid]],
                        uint  lane [[thread_index_in_simdgroup]]) {
  const uint o = tg.x;
  const uint r = tg.y;
  if (o >= dims.n_out || r >= dims.m)
    return;
  device const half  *wrow = W + (ulong)o * dims.n_in;
  device const float *arow = A + (ulong)r * dims.n_in;
  float acc = 0.0f;
  for (uint i = lane; i < dims.n_in; i += 32)
    acc = fma((float)wrow[i], arow[i], acc);
  acc = simd_sum(acc);
  if (lane == 0) {
    const ulong idx = (ulong)r * dims.n_out + o;
    C[idx] = dims.add_res ? acc + R[idx] : acc;
  }
}

struct RmsDims { uint rows; uint dim; float eps; };

// one threadgroup of 256 threads per row. lanes stride the dimension, sum their
// squares, reduce per simdgroup with simd_sum, then simdgroup 0 folds the 8
// partials. the tree reorders the sum vs the sequential CPU rmsnorm, within
// atol 1e-4.
constant uint kRmsThreads = 256;

kernel void rmsnorm(device const float *x   [[buffer(0)]],
                    device const float *w   [[buffer(1)]],
                    device       float *out [[buffer(2)]],
                    constant     RmsDims &d  [[buffer(3)]],
                    threadgroup  float *partial [[threadgroup(0)]],
                    uint r        [[threadgroup_position_in_grid]],
                    uint tid      [[thread_position_in_threadgroup]],
                    uint sg_lane  [[thread_index_in_simdgroup]],
                    uint sg_id    [[simdgroup_index_in_threadgroup]]) {
  if (r >= d.rows)
    return;
  device const float *xr = x + (ulong)r * d.dim;
  device       float *o  = out + (ulong)r * d.dim;
  float ss = 0.0f;
  for (uint i = tid; i < d.dim; i += kRmsThreads)
    ss = ss + xr[i] * xr[i];
  ss = simd_sum(ss);
  if (sg_lane == 0)
    partial[sg_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_id == 0) {
    const uint n_simd = kRmsThreads / 32;
    float v = sg_lane < n_simd ? partial[sg_lane] : 0.0f;
    v = simd_sum(v);
    if (sg_lane == 0)
      partial[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = 1.0f / sqrt(partial[0] / (float)d.dim + d.eps);
  for (uint i = tid; i < d.dim; i += kRmsThreads)
    o[i] = xr[i] * scale * w[i];
}

struct RopeDims { uint n_heads; uint seq; uint head_dim; float theta; };

kernel void rope(device       float *x   [[buffer(0)]],
                 device const int   *pos [[buffer(1)]],
                 constant     RopeDims &d  [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
  if (gid >= d.n_heads * d.seq)
    return;
  const uint h = gid / d.seq;
  const uint s = gid % d.seq;
  device float *v = x + ((ulong)h * d.seq + s) * d.head_dim;
  const uint half_d = d.head_dim / 2;
  const float p = (float)pos[s];
  const float two_pi = 6.28318530717958648f;
  for (uint j = 0; j < half_d; ++j) {
    // fp32 pow bounds accuracy vs the double CPU rope. angle error grows as
    // pos * ulp(inv), reaching ~1e-4 near pos 2048 for theta 1e6.
    const float inv = pow(d.theta, -2.0f * (float)j / (float)d.head_dim);
    float ang = p * inv;
    // reduce before the transcendental so large positions keep fp32 precision.
    ang = ang - two_pi * rint(ang / two_pi);
    const float c = precise::cos(ang);
    const float sn = precise::sin(ang);
    const float a = v[j];
    const float b = v[j + half_d];
    v[j] = a * c - b * sn;
    v[j + half_d] = b * c + a * sn;
  }
}

struct AttnDims { uint n_positions; uint n_heads; uint n_kv_heads; uint head_dim; uint pos_stride; };

// one simdgroup per query head, 32 lanes split the position range. QK^T and the
// softmax max/sum reduce across lanes with simd_max and simd_sum; the weighted-V
// accumulation splits the head dimension across lanes and sums positions in the
// CPU order. the softmax sum reorders vs the sequential CPU reduction, within
// the attention 1e-4 tolerance.
kernel void attention(device const float *q      [[buffer(0)]],
                      device const float *k      [[buffer(1)]],
                      device const float *v      [[buffer(2)]],
                      device       float *out    [[buffer(3)]],
                      device       float *scores [[buffer(4)]],
                      constant     AttnDims &d    [[buffer(5)]],
                      uint h    [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
  if (h >= d.n_heads)
    return;
  const uint gqa = d.n_heads / d.n_kv_heads;
  const uint kh = h / gqa;
  const float scale = 1.0f / sqrt((float)d.head_dim);
  device const float *qh = q + (ulong)h * d.head_dim;
  device       float *sc = scores + (ulong)h * d.n_positions;
  float lmax = -INFINITY;
  for (uint pp = lane; pp < d.n_positions; pp += 32) {
    device const float *kp = k + (ulong)pp * d.pos_stride + (ulong)kh * d.head_dim;
    float dot = 0.0f;
    for (uint i = 0; i < d.head_dim; ++i)
      dot = dot + qh[i] * kp[i];
    sc[pp] = dot * scale;
    lmax = max(lmax, sc[pp]);
  }
  const float m = simd_max(lmax);
  float lsum = 0.0f;
  for (uint pp = lane; pp < d.n_positions; pp += 32) {
    const float e = precise::exp(sc[pp] - m);
    sc[pp] = e;
    lsum = lsum + e;
  }
  const float inv = 1.0f / simd_sum(lsum);
  threadgroup_barrier(mem_flags::mem_device);
  device float *o = out + (ulong)h * d.head_dim;
  for (uint i = lane; i < d.head_dim; i += 32) {
    float acc = 0.0f;
    for (uint pp = 0; pp < d.n_positions; ++pp) {
      device const float *vp = v + (ulong)pp * d.pos_stride + (ulong)kh * d.head_dim;
      acc = acc + sc[pp] * inv * vp[i];
    }
    o[i] = acc;
  }
}

kernel void add_inplace(device       float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        constant     uint  &n [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < n)
    a[i] = a[i] + b[i];
}

kernel void swiglu(device       float *g [[buffer(0)]],
                   device const float *u [[buffer(1)]],
                   constant     uint  &n [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
  if (i >= n)
    return;
  const float xv = g[i];
  const float s = xv / (1.0f + precise::exp(-xv));
  g[i] = s * u[i];
}

struct QuantDims { uint rows; uint nblocks; };

// Q8_0 activation quant, one thread per 32-element block. mirrors
// quantize_row_q8_0: absmax, d = absmax/127, id = 1/d, q = round(x*id) clamped.
// d stored as fp16 through the hardware round-to-nearest-even, matching
// f32_to_f16 for these finite scales. writes the 34-byte BlockQ8_0 layout by
// byte so the unaligned fp16 scale needs no aligned typed store.
kernel void quantize_q8_0(device const float *A   [[buffer(0)]],
                          device       uchar *out [[buffer(1)]],
                          constant     QuantDims &d [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid >= d.rows * d.nblocks)
    return;
  device const float *xb = A + (ulong)gid * 32;
  float amax = 0.0f;
  for (uint i = 0; i < 32; ++i)
    amax = max(amax, fabs(xb[i]));
  const float dd = amax / 127.0f;
  const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
  const ushort dbits = as_type<ushort>((half)dd);
  device uchar *blk = out + (ulong)gid * 34;
  blk[0] = (uchar)(dbits & 0xff);
  blk[1] = (uchar)(dbits >> 8);
  for (uint i = 0; i < 32; ++i) {
    const int q = clamp((int)round(xb[i] * id), -127, 127);
    blk[2 + i] = as_type<uchar>((char)q);
  }
}

// one simdgroup per output row. 32 lanes split the row's blocks and each writes
// its exact integer block dot to threadgroup memory. lane 0 then accumulates the
// fp32 partials in block order, so acc + (dw*dx)*sumi still contracts to one fma
// per block like matvec_q8_0_scalar. the integer dot is order-free and the fp32
// order is unchanged, so this stays bitwise vs the scalar reference.
kernel void mul_mat_q8_0(device const uchar *W    [[buffer(0)]],
                         device const uchar *X    [[buffer(1)]],
                         device       float *C    [[buffer(2)]],
                         constant     MatDims &d   [[buffer(3)]],
                         device const float *R    [[buffer(4)]],
                         threadgroup  int   *sumi  [[threadgroup(0)]],
                         uint2 tg   [[threadgroup_position_in_grid]],
                         uint  lane [[thread_index_in_simdgroup]]) {
  const uint o = tg.x;
  const uint r = tg.y;
  if (o >= d.n_out || r >= d.m)
    return;
  const uint nb = d.n_in / 32;
  device const uchar *wrow = W + (ulong)o * nb * 34;
  device const uchar *xrow = X + (ulong)r * nb * 34;
  for (uint b = lane; b < nb; b += 32) {
    device const uchar *wb = wrow + (ulong)b * 34;
    device const uchar *xb = xrow + (ulong)b * 34;
    int s = 0;
    for (uint i = 0; i < 32; ++i)
      s += (int)as_type<char>(wb[2 + i]) * (int)as_type<char>(xb[2 + i]);
    sumi[b] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane != 0)
    return;
  float acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    device const uchar *wb = wrow + (ulong)b * 34;
    device const uchar *xb = xrow + (ulong)b * 34;
    const ushort dwb = (ushort)wb[0] | ((ushort)wb[1] << 8);
    const ushort dxb = (ushort)xb[0] | ((ushort)xb[1] << 8);
    const float dw = (float)as_type<half>(dwb);
    const float dx = (float)as_type<half>(dxb);
    acc = fma(dw * dx, (float)sumi[b], acc);
  }
  const ulong idx = (ulong)r * d.n_out + o;
  C[idx] = d.add_res ? acc + R[idx] : acc;
}

// same shape for Q4_0 weights: nibble j maps to element j, nibble j+16 to
// element j+16, both centered at 8. activations stay Q8_0. bitwise match with
// matvec_q4_0_scalar under the same split.
kernel void mul_mat_q4_0(device const uchar *W    [[buffer(0)]],
                         device const uchar *X    [[buffer(1)]],
                         device       float *C    [[buffer(2)]],
                         constant     MatDims &d   [[buffer(3)]],
                         device const float *R    [[buffer(4)]],
                         threadgroup  int   *sumi  [[threadgroup(0)]],
                         uint2 tg   [[threadgroup_position_in_grid]],
                         uint  lane [[thread_index_in_simdgroup]]) {
  const uint o = tg.x;
  const uint r = tg.y;
  if (o >= d.n_out || r >= d.m)
    return;
  const uint nb = d.n_in / 32;
  device const uchar *wrow = W + (ulong)o * nb * 18;
  device const uchar *xrow = X + (ulong)r * nb * 34;
  for (uint b = lane; b < nb; b += 32) {
    device const uchar *wb = wrow + (ulong)b * 18;
    device const uchar *xb = xrow + (ulong)b * 34;
    int s = 0;
    for (uint j = 0; j < 16; ++j) {
      const uint q = (uint)wb[2 + j];
      s += ((int)(q & 0x0Fu) - 8) * (int)as_type<char>(xb[2 + j]);
      s += ((int)(q >> 4) - 8) * (int)as_type<char>(xb[2 + j + 16]);
    }
    sumi[b] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane != 0)
    return;
  float acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    device const uchar *wb = wrow + (ulong)b * 18;
    device const uchar *xb = xrow + (ulong)b * 34;
    const ushort dwb = (ushort)wb[0] | ((ushort)wb[1] << 8);
    const ushort dxb = (ushort)xb[0] | ((ushort)xb[1] << 8);
    const float dw = (float)as_type<half>(dwb);
    const float dx = (float)as_type<half>(dxb);
    acc = fma(dw * dx, (float)sumi[b], acc);
  }
  const ulong idx = (ulong)r * d.n_out + o;
  C[idx] = d.add_res ? acc + R[idx] : acc;
}

// batched siblings: up to three weights sharing one input, three outputs. each
// output row runs the identical reduction as the single-weight kernel above, so
// the result is bitwise identical to three separate dispatches. one dispatch
// over out0+out1+out2 rows replaces three, cutting heavy launch latency. row o
// selects its weight and output by segment. out2 is zero for a two-weight group
// (gate/up), with W2/C2 aliasing W0/C0 so the third segment is never entered.
struct GroupDims { uint out0; uint out1; uint out2; uint n_in; };

kernel void mul_mat_f16_g3(device const half  *W0 [[buffer(0)]],
                           device const half  *W1 [[buffer(1)]],
                           device const half  *W2 [[buffer(2)]],
                           device const float *A  [[buffer(3)]],
                           device       float *C0 [[buffer(4)]],
                           device       float *C1 [[buffer(5)]],
                           device       float *C2 [[buffer(6)]],
                           constant GroupDims &d   [[buffer(7)]],
                           uint tg   [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint o = tg;
  device const half *wrow;
  device float *cout;
  uint lo;
  if (o < d.out0) { lo = o; wrow = W0 + (ulong)lo * d.n_in; cout = C0; }
  else if (o < d.out0 + d.out1) { lo = o - d.out0; wrow = W1 + (ulong)lo * d.n_in; cout = C1; }
  else if (o < d.out0 + d.out1 + d.out2) { lo = o - d.out0 - d.out1; wrow = W2 + (ulong)lo * d.n_in; cout = C2; }
  else return;
  float acc = 0.0f;
  for (uint i = lane; i < d.n_in; i += 32)
    acc = fma((float)wrow[i], A[i], acc);
  acc = simd_sum(acc);
  if (lane == 0)
    cout[lo] = acc;
}

kernel void mul_mat_q8_0_g3(device const uchar *W0 [[buffer(0)]],
                            device const uchar *W1 [[buffer(1)]],
                            device const uchar *W2 [[buffer(2)]],
                            device const uchar *X  [[buffer(3)]],
                            device       float *C0 [[buffer(4)]],
                            device       float *C1 [[buffer(5)]],
                            device       float *C2 [[buffer(6)]],
                            constant GroupDims &d   [[buffer(7)]],
                            threadgroup int *sumi   [[threadgroup(0)]],
                            uint tg   [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint o = tg;
  device const uchar *wrow;
  device float *cout;
  uint lo;
  const uint nb = d.n_in / 32;
  if (o < d.out0) { lo = o; wrow = W0 + (ulong)lo * nb * 34; cout = C0; }
  else if (o < d.out0 + d.out1) { lo = o - d.out0; wrow = W1 + (ulong)lo * nb * 34; cout = C1; }
  else if (o < d.out0 + d.out1 + d.out2) { lo = o - d.out0 - d.out1; wrow = W2 + (ulong)lo * nb * 34; cout = C2; }
  else return;
  device const uchar *xrow = X;
  for (uint b = lane; b < nb; b += 32) {
    device const uchar *wb = wrow + (ulong)b * 34;
    device const uchar *xb = xrow + (ulong)b * 34;
    int s = 0;
    for (uint i = 0; i < 32; ++i)
      s += (int)as_type<char>(wb[2 + i]) * (int)as_type<char>(xb[2 + i]);
    sumi[b] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane != 0)
    return;
  float acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    device const uchar *wb = wrow + (ulong)b * 34;
    device const uchar *xb = xrow + (ulong)b * 34;
    const ushort dwb = (ushort)wb[0] | ((ushort)wb[1] << 8);
    const ushort dxb = (ushort)xb[0] | ((ushort)xb[1] << 8);
    const float dw = (float)as_type<half>(dwb);
    const float dx = (float)as_type<half>(dxb);
    acc = fma(dw * dx, (float)sumi[b], acc);
  }
  cout[lo] = acc;
}

kernel void mul_mat_q4_0_g3(device const uchar *W0 [[buffer(0)]],
                            device const uchar *W1 [[buffer(1)]],
                            device const uchar *W2 [[buffer(2)]],
                            device const uchar *X  [[buffer(3)]],
                            device       float *C0 [[buffer(4)]],
                            device       float *C1 [[buffer(5)]],
                            device       float *C2 [[buffer(6)]],
                            constant GroupDims &d   [[buffer(7)]],
                            threadgroup int *sumi   [[threadgroup(0)]],
                            uint tg   [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint o = tg;
  device const uchar *wrow;
  device float *cout;
  uint lo;
  const uint nb = d.n_in / 32;
  if (o < d.out0) { lo = o; wrow = W0 + (ulong)lo * nb * 18; cout = C0; }
  else if (o < d.out0 + d.out1) { lo = o - d.out0; wrow = W1 + (ulong)lo * nb * 18; cout = C1; }
  else if (o < d.out0 + d.out1 + d.out2) { lo = o - d.out0 - d.out1; wrow = W2 + (ulong)lo * nb * 18; cout = C2; }
  else return;
  device const uchar *xrow = X;
  for (uint b = lane; b < nb; b += 32) {
    device const uchar *wb = wrow + (ulong)b * 18;
    device const uchar *xb = xrow + (ulong)b * 34;
    int s = 0;
    for (uint j = 0; j < 16; ++j) {
      const uint q = (uint)wb[2 + j];
      s += ((int)(q & 0x0Fu) - 8) * (int)as_type<char>(xb[2 + j]);
      s += ((int)(q >> 4) - 8) * (int)as_type<char>(xb[2 + j + 16]);
    }
    sumi[b] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane != 0)
    return;
  float acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    device const uchar *wb = wrow + (ulong)b * 18;
    device const uchar *xb = xrow + (ulong)b * 34;
    const ushort dwb = (ushort)wb[0] | ((ushort)wb[1] << 8);
    const ushort dxb = (ushort)xb[0] | ((ushort)xb[1] << 8);
    const float dw = (float)as_type<half>(dwb);
    const float dx = (float)as_type<half>(dxb);
    acc = fma(dw * dx, (float)sumi[b], acc);
  }
  cout[lo] = acc;
}
)MSL";

// simdgroup_matrix GEMM for the m>1 batched paths (chunked prefill, spec verify
// prefill). one 8x8 output tile per simdgroup, fp32 accumulate over 8-wide k
// tiles. inputs are pre-padded to multiples of 8 so every tile is in bounds.
// this reorders the reduction vs the CPU sequential matmul, so it is tolerance
// gated (atol 1e-3), never bitwise. compiled as a separate library so a driver
// without simdgroup_matrix leaves the main decode kernels intact.
constexpr const char *kGemmShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GemmDims { uint m_pad; uint n_pad; uint k_pad; };

kernel void gemm_f32(device const float *A [[buffer(0)]],
                     device const float *W [[buffer(1)]],
                     device       float *C [[buffer(2)]],
                     constant GemmDims &d   [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
  const uint ti = tg.x;
  const uint tj = tg.y;
  simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint k = 0; k < d.k_pad; k += 8) {
    simdgroup_matrix<float, 8, 8> am, bm;
    simdgroup_load(am, A + (ulong)tj * 8 * d.k_pad + k, d.k_pad);
    simdgroup_load(bm, W + (ulong)ti * 8 * d.k_pad + k, d.k_pad, ulong2(0, 0), true);
    simdgroup_multiply_accumulate(acc, am, bm, acc);
  }
  simdgroup_store(acc, C + (ulong)tj * 8 * d.n_pad + ti * 8, d.n_pad);
}
)MSL";

struct MatDims {
  std::uint32_t m;
  std::uint32_t n_out;
  std::uint32_t n_in;
  std::uint32_t add_res;
};

struct GemmDims {
  std::uint32_t m_pad;
  std::uint32_t n_pad;
  std::uint32_t k_pad;
};

struct RmsDims {
  std::uint32_t rows;
  std::uint32_t dim;
  float eps;
};

struct RopeDims {
  std::uint32_t n_heads;
  std::uint32_t seq;
  std::uint32_t head_dim;
  float theta;
};

struct AttnDims {
  std::uint32_t n_positions;
  std::uint32_t n_heads;
  std::uint32_t n_kv_heads;
  std::uint32_t head_dim;
  std::uint32_t pos_stride;
};

struct QuantDims {
  std::uint32_t rows;
  std::uint32_t nblocks;
};

struct GroupDims {
  std::uint32_t out0;
  std::uint32_t out1;
  std::uint32_t out2;
  std::uint32_t n_in;
};

// BlockQ8_0 stride matching tensor::BlockQ8_0 (2-byte fp16 scale + 32 int8).
constexpr std::size_t kQ8BlockBytes = 34;

// rmsnorm threadgroup width, mirroring kRmsThreads in the shader source.
constexpr std::uint32_t kRmsThreads = 256;

// one offloaded block's weights and biases copied to the device once at setup,
// plus the dims the encoder needs. buffers are strong under ARC.
struct GpuWeight {
  id<MTLBuffer> buf = nil;
  WeightType type = WeightType::F16;
  std::uint32_t out = 0;
  std::uint32_t in = 0;
};

struct LayerGpu {
  id<MTLBuffer> attn_norm = nil;
  id<MTLBuffer> ffn_norm = nil;
  GpuWeight wq, wk, wv, wo, wgate, wup, wdown;
  id<MTLBuffer> bq = nil;
  id<MTLBuffer> bk = nil;
  id<MTLBuffer> bv = nil;
  std::uint32_t dim = 0;
  std::uint32_t ff = 0;
  std::uint32_t nh = 0;
  std::uint32_t nkv = 0;
  std::uint32_t hd = 0;
  float eps = 0.0f;
  float theta = 0.0f;
};

std::size_t weight_bytes(WeightType t, std::size_t out, std::size_t in) {
  switch (t) {
  case WeightType::F16:
    return out * in * sizeof(std::uint16_t);
  case WeightType::Q8_0:
    return out * (in / 32) * 34;
  case WeightType::Q4_0:
    return out * (in / 32) * 18;
  }
  return 0;
}

class MetalBackend final : public Backend {
public:
  MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue,
               id<MTLComputePipelineState> mul_mat, id<MTLComputePipelineState> rmsnorm,
               id<MTLComputePipelineState> rope, id<MTLComputePipelineState> attention,
               id<MTLComputePipelineState> add_inplace, id<MTLComputePipelineState> swiglu,
               id<MTLComputePipelineState> quant_q8, id<MTLComputePipelineState> mul_mat_q8,
               id<MTLComputePipelineState> mul_mat_q4, id<MTLComputePipelineState> mul_mat_g3,
               id<MTLComputePipelineState> mul_mat_q8_g3, id<MTLComputePipelineState> mul_mat_q4_g3,
               id<MTLComputePipelineState> gemm)
      : device_(device), queue_(queue), mul_mat_(mul_mat), rmsnorm_(rmsnorm), rope_(rope),
        attention_(attention), add_inplace_(add_inplace), swiglu_(swiglu), quant_q8_(quant_q8),
        mul_mat_q8_(mul_mat_q8), mul_mat_q4_(mul_mat_q4), mul_mat_g3_(mul_mat_g3),
        mul_mat_q8_g3_(mul_mat_q8_g3), mul_mat_q4_g3_(mul_mat_q4_g3), gemm_(gemm) {}

  [[nodiscard]] std::expected<void, Error> mul_mat_f16(const std::uint16_t *W, const float *A,
                                                       float *C, std::size_t m, std::size_t out,
                                                       std::size_t in) override {
    @autoreleasepool {
      const std::size_t w_bytes = out * in * sizeof(std::uint16_t);
      const std::size_t a_bytes = m * in * sizeof(float);
      const std::size_t c_bytes = m * out * sizeof(float);

      id<MTLBuffer> buf_w = wrap_weights(W, w_bytes);
      if (buf_w == nil)
        return std::unexpected(Error{"metal: failed to allocate weight buffer"});
      // activations are freshly allocated host memory with no page-alignment
      // guarantee, so copy them in. weights are the tensors that must not copy.
      id<MTLBuffer> buf_a = new_from(A, a_bytes);
      id<MTLBuffer> buf_c = new_zeroed(c_bytes);
      if (buf_a == nil || buf_c == nil)
        return std::unexpected(Error{"metal: failed to allocate activation buffer"});

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      encode_mul_mat(enc, buf_w, 0, buf_a, buf_c, m, out, in);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: command buffer did not complete"});

      std::memcpy(C, buf_c.contents, c_bytes);
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error> quantize_q8_0(const float *x, std::size_t rows,
                                                         std::size_t in, std::byte *out) override {
    @autoreleasepool {
      const std::size_t nblocks = in / 32;
      const std::size_t a_bytes = rows * in * sizeof(float);
      const std::size_t q_bytes = rows * nblocks * kQ8BlockBytes;
      id<MTLBuffer> buf_a = new_from(x, a_bytes);
      id<MTLBuffer> buf_q = new_zeroed(q_bytes);
      if (buf_a == nil || buf_q == nil)
        return std::unexpected(Error{"metal: quantize buffer allocation failed"});

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      encode_quant_q8(enc, buf_a, buf_q, rows, nblocks);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: quantize command buffer did not complete"});

      std::memcpy(out, buf_q.contents, q_bytes);
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error> mul_mat_q8_0(const std::byte *W, const float *A,
                                                        float *C, std::size_t m, std::size_t out,
                                                        std::size_t in) override {
    return mul_mat_quant(mul_mat_q8_, W, (in / 32) * kQ8BlockBytes, A, C, m, out, in);
  }

  [[nodiscard]] std::expected<void, Error> mul_mat_q4_0(const std::byte *W, const float *A,
                                                        float *C, std::size_t m, std::size_t out,
                                                        std::size_t in) override {
    return mul_mat_quant(mul_mat_q4_, W, (in / 32) * 18, A, C, m, out, in);
  }

  [[nodiscard]] std::expected<void, Error>
  mul_mat_group(WeightType type, std::span<const std::byte *const> weights,
                std::span<const std::size_t> outs, const float *A, std::span<float *const> C,
                std::size_t in) override {
    const std::size_t g = weights.size();
    if (g < 2 || g > 3 || outs.size() != g || C.size() != g)
      return std::unexpected(Error{"metal: mul_mat_group needs 2 or 3 matched weights"});
    @autoreleasepool {
      GpuWeight w[3]{};
      id<MTLBuffer> cbuf[3]{nil, nil, nil};
      for (std::size_t i = 0; i < g; ++i) {
        w[i].type = type;
        w[i].out = static_cast<std::uint32_t>(outs[i]);
        w[i].in = static_cast<std::uint32_t>(in);
        w[i].buf = wrap_bytes(weights[i], weight_bytes(type, outs[i], in));
        cbuf[i] = new_zeroed(outs[i] * sizeof(float));
        if (w[i].buf == nil || cbuf[i] == nil)
          return std::unexpected(Error{"metal: mul_mat_group buffer allocation failed"});
      }
      id<MTLBuffer> buf_a = new_from(A, in * sizeof(float));
      id<MTLBuffer> buf_aq = nil;
      if (type != WeightType::F16) {
        buf_aq = new_zeroed((in / 32) * kQ8BlockBytes);
        if (buf_aq == nil)
          return std::unexpected(Error{"metal: mul_mat_group quant buffer allocation failed"});
      }
      if (buf_a == nil)
        return std::unexpected(Error{"metal: mul_mat_group activation buffer allocation failed"});

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      if (type != WeightType::F16)
        encode_quant_q8(enc, buf_a, buf_aq, 1, in / 32);
      const GpuWeight *w2 = g == 3 ? &w[2] : nullptr;
      id<MTLBuffer> c2 = g == 3 ? cbuf[2] : nil;
      encode_group(enc, w[0], w[1], w2, buf_a, buf_aq, cbuf[0], 0, cbuf[1], 0, c2, 0);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: mul_mat_group command buffer did not complete"});

      for (std::size_t i = 0; i < g; ++i)
        std::memcpy(C[i], cbuf[i].contents, outs[i] * sizeof(float));
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error> mul_mat_f16_gemm(const std::uint16_t *W, const float *A,
                                                            float *C, std::size_t m, std::size_t out,
                                                            std::size_t in) override {
    if (gemm_ == nil)
      return std::unexpected(Error{"metal: simdgroup_matrix gemm unavailable"});
    @autoreleasepool {
      const std::size_t mp = round8(m);
      const std::size_t np = round8(out);
      const std::size_t kp = round8(in);
      // zero-padded float staging keeps every 8x8 tile in bounds and lets the
      // GEMM read the mmap weights after one half-to-float widen.
      std::vector<float> ap(mp * kp, 0.0f);
      for (std::size_t r = 0; r < m; ++r)
        std::memcpy(ap.data() + r * kp, A + r * in, in * sizeof(float));
      std::vector<float> wp(np * kp, 0.0f);
      for (std::size_t o = 0; o < out; ++o)
        for (std::size_t i = 0; i < in; ++i)
          wp[o * kp + i] = half_to_f32(W[o * in + i]);

      id<MTLBuffer> buf_a = new_from(ap.data(), ap.size() * sizeof(float));
      id<MTLBuffer> buf_w = new_from(wp.data(), wp.size() * sizeof(float));
      id<MTLBuffer> buf_c = new_zeroed(mp * np * sizeof(float));
      if (buf_a == nil || buf_w == nil || buf_c == nil)
        return std::unexpected(Error{"metal: gemm buffer allocation failed"});

      const GemmDims d{static_cast<std::uint32_t>(mp), static_cast<std::uint32_t>(np),
                       static_cast<std::uint32_t>(kp)};
      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:gemm_];
      [enc setBuffer:buf_a offset:0 atIndex:0];
      [enc setBuffer:buf_w offset:0 atIndex:1];
      [enc setBuffer:buf_c offset:0 atIndex:2];
      [enc setBytes:&d length:sizeof(d) atIndex:3];
      const MTLSize groups = MTLSizeMake(np / 8, mp / 8, 1);
      const MTLSize group = MTLSizeMake(32, 1, 1);
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: gemm command buffer did not complete"});

      const float *cp = static_cast<const float *>(buf_c.contents);
      for (std::size_t r = 0; r < m; ++r)
        std::memcpy(C + r * out, cp + r * np, out * sizeof(float));
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error> rmsnorm(const float *x, const float *weight, float eps,
                                                   float *out, std::size_t rows,
                                                   std::size_t dim) override {
    @autoreleasepool {
      const std::size_t n = rows * dim;
      id<MTLBuffer> buf_x = new_from(x, n * sizeof(float));
      id<MTLBuffer> buf_w = new_from(weight, dim * sizeof(float));
      id<MTLBuffer> buf_o = new_zeroed(n * sizeof(float));
      if (buf_x == nil || buf_w == nil || buf_o == nil)
        return std::unexpected(Error{"metal: rmsnorm buffer allocation failed"});

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      encode_rmsnorm(enc, buf_x, 0, buf_w, buf_o, 0, rows, dim, eps);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: rmsnorm command buffer did not complete"});

      std::memcpy(out, buf_o.contents, n * sizeof(float));
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error> rope(float *x, const std::int32_t *positions,
                                                float theta, std::size_t n_heads, std::size_t seq,
                                                std::size_t head_dim) override {
    @autoreleasepool {
      const std::size_t n = n_heads * seq * head_dim;
      id<MTLBuffer> buf_x = new_from(x, n * sizeof(float));
      id<MTLBuffer> buf_p = new_from(positions, seq * sizeof(std::int32_t));
      if (buf_x == nil || buf_p == nil)
        return std::unexpected(Error{"metal: rope buffer allocation failed"});

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      encode_rope(enc, buf_x, 0, buf_p, theta, n_heads, seq, head_dim);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: rope command buffer did not complete"});

      std::memcpy(x, buf_x.contents, n * sizeof(float));
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error>
  attention(const float *q, const float *k_cache, const float *v_cache, float *out,
            std::size_t n_positions, std::size_t n_heads, std::size_t n_kv_heads,
            std::size_t head_dim) override {
    @autoreleasepool {
      const std::size_t pos_stride = n_kv_heads * head_dim;
      const std::size_t out_n = n_heads * head_dim;
      id<MTLBuffer> buf_q = new_from(q, out_n * sizeof(float));
      id<MTLBuffer> buf_k = new_from(k_cache, n_positions * pos_stride * sizeof(float));
      id<MTLBuffer> buf_v = new_from(v_cache, n_positions * pos_stride * sizeof(float));
      id<MTLBuffer> buf_o = new_zeroed(out_n * sizeof(float));
      id<MTLBuffer> buf_sc = new_zeroed(n_heads * n_positions * sizeof(float));
      if (buf_q == nil || buf_k == nil || buf_v == nil || buf_o == nil || buf_sc == nil)
        return std::unexpected(Error{"metal: attention buffer allocation failed"});

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      encode_attention(enc, buf_q, 0, buf_k, 0, buf_v, 0, buf_o, buf_sc, n_positions, n_heads,
                       n_kv_heads, head_dim);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: attention command buffer did not complete"});

      std::memcpy(out, buf_o.contents, out_n * sizeof(float));
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error>
  setup_offload(std::span<const LayerQ> layers, float *k_base, float *v_base, std::size_t capacity,
                std::size_t pos_stride, const float *output_norm, QWeight lm_head,
                std::size_t vocab_size) override {
    @autoreleasepool {
      offload_.clear();
      cache_capacity_ = capacity;
      cache_pos_stride_ = pos_stride;
      const std::size_t kv_bytes = layers.size() * capacity * pos_stride * sizeof(float);
      cache_k_ = wrap_bytes(reinterpret_cast<const std::byte *>(k_base), kv_bytes);
      cache_v_ = wrap_bytes(reinterpret_cast<const std::byte *>(v_base), kv_bytes);
      if (cache_k_ == nil || cache_v_ == nil)
        return std::unexpected(Error{"metal: failed to wrap KV cache"});

      offload_.reserve(layers.size());
      for (const LayerQ &L : layers) {
        LayerGpu g;
        g.dim = static_cast<std::uint32_t>(L.dim);
        g.ff = static_cast<std::uint32_t>(L.ff);
        g.nh = static_cast<std::uint32_t>(L.n_heads);
        g.nkv = static_cast<std::uint32_t>(L.n_kv_heads);
        g.hd = static_cast<std::uint32_t>(L.head_dim);
        g.eps = L.rms_eps;
        g.theta = L.rope_theta;
        const std::size_t q_n = L.n_heads * L.head_dim;
        const std::size_t kv_n = L.n_kv_heads * L.head_dim;
        g.attn_norm = new_from(L.attn_norm, L.dim * sizeof(float));
        g.ffn_norm = new_from(L.ffn_norm, L.dim * sizeof(float));
        g.wq = make_weight(L.wq, q_n, L.dim);
        g.wk = make_weight(L.wk, kv_n, L.dim);
        g.wv = make_weight(L.wv, kv_n, L.dim);
        g.wo = make_weight(L.wo, L.dim, q_n);
        g.wgate = make_weight(L.wgate, L.ff, L.dim);
        g.wup = make_weight(L.wup, L.ff, L.dim);
        g.wdown = make_weight(L.wdown, L.dim, L.ff);
        g.bq = L.bq ? new_from(L.bq, q_n * sizeof(float)) : nil;
        g.bk = L.bk ? new_from(L.bk, kv_n * sizeof(float)) : nil;
        g.bv = L.bv ? new_from(L.bv, kv_n * sizeof(float)) : nil;
        if (g.attn_norm == nil || g.ffn_norm == nil || g.wq.buf == nil || g.wk.buf == nil ||
            g.wv.buf == nil || g.wo.buf == nil || g.wgate.buf == nil || g.wup.buf == nil ||
            g.wdown.buf == nil)
          return std::unexpected(Error{"metal: failed to allocate offload weights"});
        offload_.push_back(g);
      }

      // a null lm_head leaves decode_token_full disabled, for the partial-offload
      // path where the CPU still runs the tail.
      if (lm_head.data != nullptr) {
        head_norm_ = new_from(output_norm, offload_.front().dim * sizeof(float));
        head_w_ = make_weight(lm_head, vocab_size, offload_.front().dim);
        logits_buf_ = new_zeroed(vocab_size * sizeof(float));
        if (head_norm_ == nil || head_w_.buf == nil || logits_buf_ == nil)
          return std::unexpected(Error{"metal: failed to allocate lm_head buffers"});
      }

      if (auto e = alloc_scratch(); !e)
        return e;
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error> decode_token(float *x, std::int32_t pos) override {
    if (offload_.empty())
      return std::unexpected(Error{"metal: decode_token before setup_offload"});
    @autoreleasepool {
      const std::uint32_t dim = offload_.front().dim;
      std::memcpy(buf_x_.contents, x, dim * sizeof(float));

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      for (std::size_t l = 0; l < offload_.size(); ++l)
        encode_layer(enc, offload_[l], l, buf_x_, scratch_, pos);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: decode_token command buffer did not complete"});

      std::memcpy(x, buf_x_.contents, dim * sizeof(float));
    }
    return {};
  }

  [[nodiscard]] std::expected<const float *, Error> decode_token_full(const float *x,
                                                                      std::int32_t pos) override {
    if (offload_.empty())
      return std::unexpected(Error{"metal: decode_token_full before setup_offload"});
    if (head_w_.buf == nil)
      return std::unexpected(Error{"metal: decode_token_full without lm_head"});
    @autoreleasepool {
      const std::uint32_t dim = offload_.front().dim;
      std::memcpy(buf_x_.contents, x, dim * sizeof(float));

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      for (std::size_t l = 0; l < offload_.size(); ++l)
        encode_layer(enc, offload_[l], l, buf_x_, scratch_, pos);
      // final rmsnorm reuses the free per-layer normed scratch, then the lm_head
      // projection writes logits directly into the shared buffer the CPU reads.
      encode_rmsnorm(enc, buf_x_, 0, head_norm_, scratch_.normed, 0, 1, dim, offload_.front().eps);
      if (head_w_.type != WeightType::F16)
        encode_quant_q8(enc, scratch_.normed, scratch_.normed_q, 1, dim / 32);
      encode_weight(enc, head_w_, scratch_.normed, scratch_.normed_q, logits_buf_, 0);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: decode_token_full command buffer did not complete"});
    }
    return static_cast<const float *>(logits_buf_.contents);
  }

private:
  struct Scratch {
    id<MTLBuffer> normed;
    id<MTLBuffer> normed_q;
    id<MTLBuffer> q;
    id<MTLBuffer> attn;
    id<MTLBuffer> attn_q;
    id<MTLBuffer> scores;
    id<MTLBuffer> gate;
    id<MTLBuffer> up;
    id<MTLBuffer> gate_q;
  };

  static std::size_t round8(std::size_t n) { return (n + 7) & ~std::size_t{7}; }

  // arm64 __fp16 is IEEE binary16, matching the gguf half weight layout.
  static float half_to_f32(std::uint16_t h) {
    __fp16 v;
    std::memcpy(&v, &h, sizeof(v));
    return static_cast<float>(v);
  }

  GpuWeight make_weight(const QWeight &w, std::size_t out, std::size_t in) {
    const std::size_t bytes = weight_bytes(w.type, out, in);
    GpuWeight g;
    g.type = w.type;
    g.out = static_cast<std::uint32_t>(out);
    g.in = static_cast<std::uint32_t>(in);
    g.buf = wrap_bytes(w.data, bytes);
    return g;
  }

  std::expected<void, Error> alloc_scratch() {
    const std::size_t dim = offload_.front().dim;
    const std::size_t ff = offload_.front().ff;
    const std::size_t q_n = static_cast<std::size_t>(offload_.front().nh) * offload_.front().hd;
    const std::size_t nh = offload_.front().nh;
    buf_x_ = new_zeroed(dim * sizeof(float));
    scratch_.normed = new_zeroed(dim * sizeof(float));
    scratch_.normed_q = new_zeroed((dim / 32) * kQ8BlockBytes);
    scratch_.q = new_zeroed(q_n * sizeof(float));
    scratch_.attn = new_zeroed(q_n * sizeof(float));
    scratch_.attn_q = new_zeroed((q_n / 32) * kQ8BlockBytes);
    scratch_.scores = new_zeroed(nh * cache_capacity_ * sizeof(float));
    scratch_.gate = new_zeroed(ff * sizeof(float));
    scratch_.up = new_zeroed(ff * sizeof(float));
    scratch_.gate_q = new_zeroed((ff / 32) * kQ8BlockBytes);
    if (buf_x_ == nil || scratch_.normed == nil || scratch_.normed_q == nil || scratch_.q == nil ||
        scratch_.attn == nil || scratch_.attn_q == nil || scratch_.scores == nil ||
        scratch_.gate == nil || scratch_.up == nil || scratch_.gate_q == nil)
      return std::unexpected(Error{"metal: scratch allocation failed"});
    return {};
  }

  // one weight matmul, dispatching on dtype. c is written at byte offset c_off,
  // used to land QKV projections directly at the KV cache slot. quant weights
  // read the pre-quantized activation aq, F16 reads the fp32 a.
  void encode_weight(id<MTLComputeCommandEncoder> enc, const GpuWeight &w, id<MTLBuffer> a,
                     id<MTLBuffer> aq, id<MTLBuffer> c, std::size_t c_off, bool add_res = false) {
    switch (w.type) {
    case WeightType::F16:
      encode_mul_mat(enc, w.buf, 0, a, c, 1, w.out, w.in, c_off, add_res);
      return;
    case WeightType::Q8_0:
      encode_mul_mat_quant(enc, mul_mat_q8_, w.buf, aq, c, 1, w.out, w.in, c_off, add_res);
      return;
    case WeightType::Q4_0:
      encode_mul_mat_quant(enc, mul_mat_q4_, w.buf, aq, c, 1, w.out, w.in, c_off, add_res);
      return;
    }
  }

  void encode_layer(id<MTLComputeCommandEncoder> enc, const LayerGpu &L, std::size_t layer_idx,
                    id<MTLBuffer> buf_x, const Scratch &s, std::int32_t pos) {
    const std::uint32_t dim = L.dim;
    const std::uint32_t ff = L.ff;
    const std::uint32_t q_n = L.nh * L.hd;
    const std::size_t pos_stride = cache_pos_stride_;
    const std::size_t layer_base = layer_idx * cache_capacity_ * pos_stride * sizeof(float);
    const std::size_t slot_off =
        layer_base + static_cast<std::size_t>(pos) * pos_stride * sizeof(float);
    const std::size_t n_pos = static_cast<std::size_t>(pos) + 1;

    encode_rmsnorm(enc, buf_x, 0, L.attn_norm, s.normed, 0, 1, dim, L.eps);
    if (quant_any(L.wq.type, L.wk.type, L.wv.type))
      encode_quant_q8(enc, s.normed, s.normed_q, 1, dim / 32);
    if (same_group(L.wq, L.wk) && same_group(L.wq, L.wv)) {
      encode_group(enc, L.wq, L.wk, &L.wv, s.normed, s.normed_q, s.q, 0, cache_k_, slot_off, cache_v_,
                   slot_off);
    } else {
      encode_weight(enc, L.wq, s.normed, s.normed_q, s.q, 0);
      encode_weight(enc, L.wk, s.normed, s.normed_q, cache_k_, slot_off);
      encode_weight(enc, L.wv, s.normed, s.normed_q, cache_v_, slot_off);
    }
    if (L.bq)
      encode_add(enc, s.q, 0, L.bq, q_n);
    if (L.bk)
      encode_add(enc, cache_k_, slot_off, L.bk, pos_stride);
    if (L.bv)
      encode_add(enc, cache_v_, slot_off, L.bv, pos_stride);
    encode_rope_pos(enc, s.q, 0, pos, L.theta, L.nh, L.hd);
    encode_rope_pos(enc, cache_k_, slot_off, pos, L.theta, L.nkv, L.hd);
    encode_attention(enc, s.q, 0, cache_k_, layer_base, cache_v_, layer_base, s.attn, s.scores,
                     n_pos, L.nh, L.nkv, L.hd);
    if (quant_any(L.wo.type))
      encode_quant_q8(enc, s.attn, s.attn_q, 1, q_n / 32);
    // fold the attention residual into the output projection epilogue.
    encode_weight(enc, L.wo, s.attn, s.attn_q, buf_x, 0, /*add_res=*/true);

    encode_rmsnorm(enc, buf_x, 0, L.ffn_norm, s.normed, 0, 1, dim, L.eps);
    if (quant_any(L.wgate.type, L.wup.type))
      encode_quant_q8(enc, s.normed, s.normed_q, 1, dim / 32);
    if (same_group(L.wgate, L.wup)) {
      encode_group(enc, L.wgate, L.wup, nullptr, s.normed, s.normed_q, s.gate, 0, s.up, 0, nil, 0);
    } else {
      encode_weight(enc, L.wgate, s.normed, s.normed_q, s.gate, 0);
      encode_weight(enc, L.wup, s.normed, s.normed_q, s.up, 0);
    }
    encode_swiglu(enc, s.gate, s.up, ff);
    if (quant_any(L.wdown.type))
      encode_quant_q8(enc, s.gate, s.gate_q, 1, ff / 32);
    // fold the ffn residual into the down projection epilogue.
    encode_weight(enc, L.wdown, s.gate, s.gate_q, buf_x, 0, /*add_res=*/true);
  }

  static bool quant_any(WeightType a) { return a != WeightType::F16; }
  static bool quant_any(WeightType a, WeightType b, WeightType c) {
    return quant_any(a) || quant_any(b) || quant_any(c);
  }
  static bool quant_any(WeightType a, WeightType b) { return quant_any(a) || quant_any(b); }

private:
  void encode_mul_mat(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> w, std::size_t w_off,
                      id<MTLBuffer> a, id<MTLBuffer> c, std::size_t m, std::size_t out,
                      std::size_t in, std::size_t c_off = 0, bool add_res = false) {
    const MatDims dims{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(out),
                       static_cast<std::uint32_t>(in), add_res ? 1u : 0u};
    [enc setComputePipelineState:mul_mat_];
    [enc setBuffer:w offset:w_off atIndex:0];
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBuffer:c offset:c_off atIndex:2];
    [enc setBytes:&dims length:sizeof(dims) atIndex:3];
    [enc setBuffer:c offset:c_off atIndex:4];
    const MTLSize groups = MTLSizeMake(out, m, 1);
    const MTLSize group = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
  }

  // quantizes A to Q8_0 then dots against the quant weight W, both in one serial
  // encoder so the matvec reads the freshly quantized activation. row_bytes is
  // the packed weight row stride (Q8_0 34/block, Q4_0 18/block).
  std::expected<void, Error> mul_mat_quant(id<MTLComputePipelineState> pipe, const std::byte *W,
                                           std::size_t row_bytes, const float *A, float *C,
                                           std::size_t m, std::size_t out, std::size_t in) {
    @autoreleasepool {
      const std::size_t nblocks = in / 32;
      const std::size_t w_bytes = out * row_bytes;
      const std::size_t a_bytes = m * in * sizeof(float);
      const std::size_t xq_bytes = m * nblocks * kQ8BlockBytes;
      const std::size_t c_bytes = m * out * sizeof(float);

      id<MTLBuffer> buf_w = wrap_bytes(W, w_bytes);
      id<MTLBuffer> buf_a = new_from(A, a_bytes);
      id<MTLBuffer> buf_xq = new_zeroed(xq_bytes);
      id<MTLBuffer> buf_c = new_zeroed(c_bytes);
      if (buf_w == nil || buf_a == nil || buf_xq == nil || buf_c == nil)
        return std::unexpected(Error{"metal: quant matvec buffer allocation failed"});

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      encode_quant_q8(enc, buf_a, buf_xq, m, nblocks);
      encode_mul_mat_quant(enc, pipe, buf_w, buf_xq, buf_c, m, out, in);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: quant matvec command buffer did not complete"});

      std::memcpy(C, buf_c.contents, c_bytes);
    }
    return {};
  }

  void encode_quant_q8(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, id<MTLBuffer> out,
                       std::size_t rows, std::size_t nblocks) {
    const QuantDims d{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(nblocks)};
    [enc setComputePipelineState:quant_q8_];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    dispatch_1d(enc, quant_q8_, rows * nblocks);
  }

  void encode_mul_mat_quant(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                            id<MTLBuffer> w, id<MTLBuffer> xq, id<MTLBuffer> c, std::size_t m,
                            std::size_t out, std::size_t in, std::size_t c_off = 0,
                            bool add_res = false) {
    const MatDims dims{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(out),
                       static_cast<std::uint32_t>(in), add_res ? 1u : 0u};
    [enc setComputePipelineState:pipe];
    [enc setBuffer:w offset:0 atIndex:0];
    [enc setBuffer:xq offset:0 atIndex:1];
    [enc setBuffer:c offset:c_off atIndex:2];
    [enc setBytes:&dims length:sizeof(dims) atIndex:3];
    [enc setBuffer:c offset:c_off atIndex:4];
    [enc setThreadgroupMemoryLength:(in / 32) * sizeof(std::int32_t) atIndex:0];
    const MTLSize groups = MTLSizeMake(out, m, 1);
    const MTLSize group = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
  }

  // batchable when the siblings share dtype and input width, the invariant the
  // grouped kernel relies on for a single shared input and one dispatch.
  static bool same_group(const GpuWeight &a, const GpuWeight &b) {
    return a.type == b.type && a.in == b.in;
  }

  // two or three sibling weights sharing input a (F16) or aq (quant), written to
  // c0/c1/c2 at their offsets, in one dispatch over out0+out1+out2 rows. w2 null
  // encodes a two-weight group, aliasing the third slots so validation passes.
  void encode_group(id<MTLComputeCommandEncoder> enc, const GpuWeight &w0, const GpuWeight &w1,
                    const GpuWeight *w2, id<MTLBuffer> a, id<MTLBuffer> aq, id<MTLBuffer> c0,
                    std::size_t c0_off, id<MTLBuffer> c1, std::size_t c1_off, id<MTLBuffer> c2,
                    std::size_t c2_off) {
    const std::uint32_t out2 = w2 ? w2->out : 0;
    const GroupDims dims{w0.out, w1.out, out2, w0.in};
    id<MTLBuffer> b2w = w2 ? w2->buf : w0.buf;
    id<MTLBuffer> b2c = w2 ? c2 : c0;
    const std::size_t b2c_off = w2 ? c2_off : c0_off;
    if (w0.type == WeightType::F16) {
      [enc setComputePipelineState:mul_mat_g3_];
      [enc setBuffer:a offset:0 atIndex:3];
    } else {
      [enc setComputePipelineState:w0.type == WeightType::Q8_0 ? mul_mat_q8_g3_ : mul_mat_q4_g3_];
      [enc setBuffer:aq offset:0 atIndex:3];
      [enc setThreadgroupMemoryLength:(w0.in / 32) * sizeof(std::int32_t) atIndex:0];
    }
    [enc setBuffer:w0.buf offset:0 atIndex:0];
    [enc setBuffer:w1.buf offset:0 atIndex:1];
    [enc setBuffer:b2w offset:0 atIndex:2];
    [enc setBuffer:c0 offset:c0_off atIndex:4];
    [enc setBuffer:c1 offset:c1_off atIndex:5];
    [enc setBuffer:b2c offset:b2c_off atIndex:6];
    [enc setBytes:&dims length:sizeof(dims) atIndex:7];
    const MTLSize groups = MTLSizeMake(w0.out + w1.out + out2, 1, 1);
    const MTLSize group = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
  }

  void encode_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, std::size_t x_off,
                      id<MTLBuffer> w, id<MTLBuffer> out, std::size_t out_off, std::size_t rows,
                      std::size_t dim, float eps) {
    const RmsDims d{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(dim), eps};
    [enc setComputePipelineState:rmsnorm_];
    [enc setBuffer:x offset:x_off atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:out offset:out_off atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    [enc setThreadgroupMemoryLength:(kRmsThreads / 32) * sizeof(float) atIndex:0];
    const MTLSize groups = MTLSizeMake(rows, 1, 1);
    const MTLSize group = MTLSizeMake(kRmsThreads, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
  }

  void encode_rope(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, std::size_t x_off,
                   id<MTLBuffer> pos, float theta, std::size_t n_heads, std::size_t seq,
                   std::size_t head_dim) {
    const RopeDims d{static_cast<std::uint32_t>(n_heads), static_cast<std::uint32_t>(seq),
                     static_cast<std::uint32_t>(head_dim), theta};
    [enc setComputePipelineState:rope_];
    [enc setBuffer:x offset:x_off atIndex:0];
    [enc setBuffer:pos offset:0 atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    dispatch_1d(enc, rope_, n_heads * seq);
  }

  // rope for a single position, feeding the position through a one-int buffer.
  void encode_rope_pos(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, std::size_t x_off,
                       std::int32_t pos, float theta, std::size_t n_heads, std::size_t head_dim) {
    id<MTLBuffer> buf_p = new_from(&pos, sizeof(std::int32_t));
    encode_rope(enc, x, x_off, buf_p, theta, n_heads, 1, head_dim);
  }

  void encode_attention(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> q, std::size_t q_off,
                        id<MTLBuffer> k, std::size_t k_off, id<MTLBuffer> v, std::size_t v_off,
                        id<MTLBuffer> out, id<MTLBuffer> scores, std::size_t n_positions,
                        std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim) {
    const AttnDims d{static_cast<std::uint32_t>(n_positions), static_cast<std::uint32_t>(n_heads),
                     static_cast<std::uint32_t>(n_kv_heads), static_cast<std::uint32_t>(head_dim),
                     static_cast<std::uint32_t>(n_kv_heads * head_dim)};
    [enc setComputePipelineState:attention_];
    [enc setBuffer:q offset:q_off atIndex:0];
    [enc setBuffer:k offset:k_off atIndex:1];
    [enc setBuffer:v offset:v_off atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBuffer:scores offset:0 atIndex:4];
    [enc setBytes:&d length:sizeof(d) atIndex:5];
    const MTLSize groups = MTLSizeMake(n_heads, 1, 1);
    const MTLSize group = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
  }

  void encode_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, std::size_t a_off,
                  id<MTLBuffer> b, std::size_t n) {
    const std::uint32_t count = static_cast<std::uint32_t>(n);
    [enc setComputePipelineState:add_inplace_];
    [enc setBuffer:a offset:a_off atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    dispatch_1d(enc, add_inplace_, n);
  }

  void encode_swiglu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> g, id<MTLBuffer> u,
                     std::size_t n) {
    const std::uint32_t count = static_cast<std::uint32_t>(n);
    [enc setComputePipelineState:swiglu_];
    [enc setBuffer:g offset:0 atIndex:0];
    [enc setBuffer:u offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    dispatch_1d(enc, swiglu_, n);
  }

  void dispatch_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                   std::size_t n) {
    const MTLSize grid = MTLSizeMake(n, 1, 1);
    const MTLSize group = MTLSizeMake(pipe.threadExecutionWidth, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
  }

  id<MTLBuffer> new_from(const void *p, std::size_t bytes) {
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }

  id<MTLBuffer> new_zeroed(std::size_t bytes) {
    return [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  }

  // zero-copy when the weight pointer is page-aligned, matching DBMF tensors.
  // rounding the length up to a page stays inside the file's page-granular
  // mapping. an unaligned gguf mmap pointer copies instead.
  id<MTLBuffer> wrap_weights(const std::uint16_t *W, std::size_t bytes) {
    return wrap_bytes(reinterpret_cast<const std::byte *>(W), bytes);
  }

  id<MTLBuffer> wrap_bytes(const std::byte *p, std::size_t bytes) {
    if (page_aligned(p))
      return [device_ newBufferWithBytesNoCopy:const_cast<std::byte *>(p)
                                        length:round_up_page(bytes)
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }

  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> mul_mat_;
  id<MTLComputePipelineState> rmsnorm_;
  id<MTLComputePipelineState> rope_;
  id<MTLComputePipelineState> attention_;
  id<MTLComputePipelineState> add_inplace_;
  id<MTLComputePipelineState> swiglu_;
  id<MTLComputePipelineState> quant_q8_;
  id<MTLComputePipelineState> mul_mat_q8_;
  id<MTLComputePipelineState> mul_mat_q4_;
  id<MTLComputePipelineState> mul_mat_g3_;
  id<MTLComputePipelineState> mul_mat_q8_g3_;
  id<MTLComputePipelineState> mul_mat_q4_g3_;
  id<MTLComputePipelineState> gemm_;

  std::vector<LayerGpu> offload_;
  id<MTLBuffer> cache_k_ = nil;
  id<MTLBuffer> cache_v_ = nil;
  std::size_t cache_capacity_ = 0;
  std::size_t cache_pos_stride_ = 0;

  // persistent per-token activations, allocated once in setup_offload and reused
  // every token. the CPU touches only buf_x_ (embedding in) and logits_buf_
  // (logits out); no per-token buffer allocation or host<->device copy remains.
  Scratch scratch_{};
  id<MTLBuffer> buf_x_ = nil;
  id<MTLBuffer> head_norm_ = nil;
  GpuWeight head_w_{};
  id<MTLBuffer> logits_buf_ = nil;
};

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib,
                                          const char *name) {
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (fn == nil) {
    std::fprintf(stderr, "metal: %s function not found\n", name);
    return nil;
  }
  NSError *err = nil;
  id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&err];
  if (pipe == nil)
    std::fprintf(stderr, "metal: %s pipeline creation failed: %s\n", name,
                 err.localizedDescription.UTF8String);
  return pipe;
}

std::unique_ptr<MetalBackend> build_backend() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "metal: no default device, falling back to CPU\n");
      return nullptr;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      std::fprintf(stderr, "metal: failed to create command queue\n");
      return nullptr;
    }

    // safe math so the elementwise reductions round like the scalar CPU path.
    // mul_mat's explicit fma keeps its bitwise match regardless.
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeSafe;
    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:kShaderSource];
    id<MTLLibrary> library = [device newLibraryWithSource:src options:opts error:&err];
    if (library == nil) {
      std::fprintf(stderr, "metal: shader compile failed: %s\n",
                   err.localizedDescription.UTF8String);
      return nullptr;
    }

    id<MTLComputePipelineState> mul_mat = make_pipeline(device, library, "mul_mat_f16");
    id<MTLComputePipelineState> rmsnorm = make_pipeline(device, library, "rmsnorm");
    id<MTLComputePipelineState> rope = make_pipeline(device, library, "rope");
    id<MTLComputePipelineState> attention = make_pipeline(device, library, "attention");
    id<MTLComputePipelineState> add_inplace = make_pipeline(device, library, "add_inplace");
    id<MTLComputePipelineState> swiglu = make_pipeline(device, library, "swiglu");
    id<MTLComputePipelineState> quant_q8 = make_pipeline(device, library, "quantize_q8_0");
    id<MTLComputePipelineState> mul_mat_q8 = make_pipeline(device, library, "mul_mat_q8_0");
    id<MTLComputePipelineState> mul_mat_q4 = make_pipeline(device, library, "mul_mat_q4_0");
    id<MTLComputePipelineState> mul_mat_g3 = make_pipeline(device, library, "mul_mat_f16_g3");
    id<MTLComputePipelineState> mul_mat_q8_g3 = make_pipeline(device, library, "mul_mat_q8_0_g3");
    id<MTLComputePipelineState> mul_mat_q4_g3 = make_pipeline(device, library, "mul_mat_q4_0_g3");
    if (mul_mat == nil || rmsnorm == nil || rope == nil || attention == nil ||
        add_inplace == nil || swiglu == nil || quant_q8 == nil || mul_mat_q8 == nil ||
        mul_mat_q4 == nil || mul_mat_g3 == nil || mul_mat_q8_g3 == nil || mul_mat_q4_g3 == nil)
      return nullptr;

    // the simdgroup_matrix GEMM compiles into its own library so a driver that
    // rejects the op leaves the decode kernels above intact; gemm_ stays nil and
    // the m>1 GEMM method reports unavailable.
    id<MTLComputePipelineState> gemm = nil;
    NSError *gerr = nil;
    NSString *gsrc = [NSString stringWithUTF8String:kGemmShaderSource];
    id<MTLLibrary> glib = [device newLibraryWithSource:gsrc options:opts error:&gerr];
    if (glib != nil)
      gemm = make_pipeline(device, glib, "gemm_f32");
    else
      std::fprintf(stderr, "metal: gemm shader compile failed: %s\n",
                   gerr.localizedDescription.UTF8String);

    return std::make_unique<MetalBackend>(device, queue, mul_mat, rmsnorm, rope, attention,
                                          add_inplace, swiglu, quant_q8, mul_mat_q8, mul_mat_q4,
                                          mul_mat_g3, mul_mat_q8_g3, mul_mat_q4_g3, gemm);
  }
}

} // namespace

Backend *metal_backend() {
  static const std::unique_ptr<MetalBackend> instance = build_backend();
  return instance.get();
}

bool metal_can_wrap_nocopy(const void *ptr, std::size_t bytes) {
  if (!page_aligned(ptr))
    return false;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil)
      return false;
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:const_cast<void *>(ptr)
                                                  length:round_up_page(bytes)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    return buf != nil;
  }
}

} // namespace dbinfer::backend
