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

struct MatDims { uint m; uint n_out; uint n_in; };

kernel void mul_mat_f16(device const half   *W    [[buffer(0)]],
                        device const float  *A    [[buffer(1)]],
                        device       float  *C    [[buffer(2)]],
                        constant     MatDims &dims [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const uint o = gid.x;
  const uint r = gid.y;
  if (o >= dims.n_out || r >= dims.m)
    return;
  device const half  *wrow = W + (ulong)o * dims.n_in;
  device const float *arow = A + (ulong)r * dims.n_in;
  float acc = 0.0f;
  for (uint i = 0; i < dims.n_in; ++i)
    acc = fma((float)wrow[i], arow[i], acc);
  C[(ulong)r * dims.n_out + o] = acc;
}

struct RmsDims { uint rows; uint dim; float eps; };

kernel void rmsnorm(device const float *x   [[buffer(0)]],
                    device const float *w   [[buffer(1)]],
                    device       float *out [[buffer(2)]],
                    constant     RmsDims &d  [[buffer(3)]],
                    uint r [[thread_position_in_grid]]) {
  if (r >= d.rows)
    return;
  device const float *xr = x + (ulong)r * d.dim;
  device       float *o  = out + (ulong)r * d.dim;
  float ss = 0.0f;
  for (uint i = 0; i < d.dim; ++i)
    ss = ss + xr[i] * xr[i];
  const float scale = 1.0f / sqrt(ss / (float)d.dim + d.eps);
  for (uint i = 0; i < d.dim; ++i)
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

kernel void attention(device const float *q      [[buffer(0)]],
                      device const float *k      [[buffer(1)]],
                      device const float *v      [[buffer(2)]],
                      device       float *out    [[buffer(3)]],
                      device       float *scores [[buffer(4)]],
                      constant     AttnDims &d    [[buffer(5)]],
                      uint h [[thread_position_in_grid]]) {
  if (h >= d.n_heads)
    return;
  const uint gqa = d.n_heads / d.n_kv_heads;
  const uint kh = h / gqa;
  const float scale = 1.0f / sqrt((float)d.head_dim);
  device const float *qh = q + (ulong)h * d.head_dim;
  device       float *sc = scores + (ulong)h * d.n_positions;
  for (uint pp = 0; pp < d.n_positions; ++pp) {
    device const float *kp = k + (ulong)pp * d.pos_stride + (ulong)kh * d.head_dim;
    float dot = 0.0f;
    for (uint i = 0; i < d.head_dim; ++i)
      dot = dot + qh[i] * kp[i];
    sc[pp] = dot * scale;
  }
  float m = sc[0];
  for (uint pp = 1; pp < d.n_positions; ++pp)
    m = max(m, sc[pp]);
  float sum = 0.0f;
  for (uint pp = 0; pp < d.n_positions; ++pp) {
    const float e = precise::exp(sc[pp] - m);
    sc[pp] = e;
    sum = sum + e;
  }
  const float inv = 1.0f / sum;
  for (uint pp = 0; pp < d.n_positions; ++pp)
    sc[pp] = sc[pp] * inv;
  device float *o = out + (ulong)h * d.head_dim;
  for (uint i = 0; i < d.head_dim; ++i)
    o[i] = 0.0f;
  for (uint pp = 0; pp < d.n_positions; ++pp) {
    device const float *vp = v + (ulong)pp * d.pos_stride + (ulong)kh * d.head_dim;
    const float w = sc[pp];
    for (uint i = 0; i < d.head_dim; ++i)
      o[i] = o[i] + w * vp[i];
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

// exact int8 block dot then sequential fp32 block accumulation, one thread per
// output row. bitwise match with matvec_q8_0_scalar: the integer dot is
// order-free, and acc + (dw*dx)*sumi contracts to one fma per block like the
// scalar reference under clang -ffp-contract=on.
kernel void mul_mat_q8_0(device const uchar *W    [[buffer(0)]],
                         device const uchar *X    [[buffer(1)]],
                         device       float *C    [[buffer(2)]],
                         constant     MatDims &d   [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const uint o = gid.x;
  const uint r = gid.y;
  if (o >= d.n_out || r >= d.m)
    return;
  const uint nb = d.n_in / 32;
  device const uchar *wrow = W + (ulong)o * nb * 34;
  device const uchar *xrow = X + (ulong)r * nb * 34;
  float acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    device const uchar *wb = wrow + (ulong)b * 34;
    device const uchar *xb = xrow + (ulong)b * 34;
    const ushort dwb = (ushort)wb[0] | ((ushort)wb[1] << 8);
    const ushort dxb = (ushort)xb[0] | ((ushort)xb[1] << 8);
    const float dw = (float)as_type<half>(dwb);
    const float dx = (float)as_type<half>(dxb);
    int sumi = 0;
    for (uint i = 0; i < 32; ++i)
      sumi += (int)as_type<char>(wb[2 + i]) * (int)as_type<char>(xb[2 + i]);
    acc = fma(dw * dx, (float)sumi, acc);
  }
  C[(ulong)r * d.n_out + o] = acc;
}

// same shape for Q4_0 weights: nibble j maps to element j, nibble j+16 to
// element j+16, both centered at 8. activations stay Q8_0. bitwise match with
// matvec_q4_0_scalar.
kernel void mul_mat_q4_0(device const uchar *W    [[buffer(0)]],
                         device const uchar *X    [[buffer(1)]],
                         device       float *C    [[buffer(2)]],
                         constant     MatDims &d   [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const uint o = gid.x;
  const uint r = gid.y;
  if (o >= d.n_out || r >= d.m)
    return;
  const uint nb = d.n_in / 32;
  device const uchar *wrow = W + (ulong)o * nb * 18;
  device const uchar *xrow = X + (ulong)r * nb * 34;
  float acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    device const uchar *wb = wrow + (ulong)b * 18;
    device const uchar *xb = xrow + (ulong)b * 34;
    const ushort dwb = (ushort)wb[0] | ((ushort)wb[1] << 8);
    const ushort dxb = (ushort)xb[0] | ((ushort)xb[1] << 8);
    const float dw = (float)as_type<half>(dwb);
    const float dx = (float)as_type<half>(dxb);
    int sumi = 0;
    for (uint j = 0; j < 16; ++j) {
      const uint q = (uint)wb[2 + j];
      sumi += ((int)(q & 0x0Fu) - 8) * (int)as_type<char>(xb[2 + j]);
      sumi += ((int)(q >> 4) - 8) * (int)as_type<char>(xb[2 + j + 16]);
    }
    acc = fma(dw * dx, (float)sumi, acc);
  }
  C[(ulong)r * d.n_out + o] = acc;
}
)MSL";

struct MatDims {
  std::uint32_t m;
  std::uint32_t n_out;
  std::uint32_t n_in;
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

// BlockQ8_0 stride matching tensor::BlockQ8_0 (2-byte fp16 scale + 32 int8).
constexpr std::size_t kQ8BlockBytes = 34;

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
               id<MTLComputePipelineState> mul_mat_q4)
      : device_(device), queue_(queue), mul_mat_(mul_mat), rmsnorm_(rmsnorm), rope_(rope),
        attention_(attention), add_inplace_(add_inplace), swiglu_(swiglu), quant_q8_(quant_q8),
        mul_mat_q8_(mul_mat_q8), mul_mat_q4_(mul_mat_q4) {}

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

  [[nodiscard]] std::expected<void, Error> setup_offload(std::span<const LayerQ> layers,
                                                         float *k_base, float *v_base,
                                                         std::size_t capacity,
                                                         std::size_t pos_stride) override {
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
    }
    return {};
  }

  [[nodiscard]] std::expected<void, Error> decode_token(float *x, std::int32_t pos) override {
    if (offload_.empty())
      return std::unexpected(Error{"metal: decode_token before setup_offload"});
    @autoreleasepool {
      const std::uint32_t dim = offload_.front().dim;
      const std::uint32_t ff = offload_.front().ff;
      const std::uint32_t q_n = offload_.front().nh * offload_.front().hd;
      const std::uint32_t nh = offload_.front().nh;

      id<MTLBuffer> buf_x = new_from(x, dim * sizeof(float));
      id<MTLBuffer> buf_normed = new_zeroed(dim * sizeof(float));
      id<MTLBuffer> buf_normed_q = new_zeroed((dim / 32) * kQ8BlockBytes);
      id<MTLBuffer> buf_q = new_zeroed(q_n * sizeof(float));
      id<MTLBuffer> buf_attn = new_zeroed(q_n * sizeof(float));
      id<MTLBuffer> buf_attn_q = new_zeroed((q_n / 32) * kQ8BlockBytes);
      id<MTLBuffer> buf_scores = new_zeroed(nh * cache_capacity_ * sizeof(float));
      id<MTLBuffer> buf_proj = new_zeroed(dim * sizeof(float));
      id<MTLBuffer> buf_gate = new_zeroed(ff * sizeof(float));
      id<MTLBuffer> buf_up = new_zeroed(ff * sizeof(float));
      id<MTLBuffer> buf_gate_q = new_zeroed((ff / 32) * kQ8BlockBytes);
      id<MTLBuffer> buf_down = new_zeroed(dim * sizeof(float));
      if (buf_x == nil || buf_normed == nil || buf_normed_q == nil || buf_q == nil ||
          buf_attn == nil || buf_attn_q == nil || buf_scores == nil || buf_proj == nil ||
          buf_gate == nil || buf_up == nil || buf_gate_q == nil || buf_down == nil)
        return std::unexpected(Error{"metal: decode_token scratch allocation failed"});

      const Scratch s{buf_normed, buf_normed_q, buf_q,   buf_attn, buf_attn_q, buf_scores,
                      buf_proj,   buf_gate,     buf_up,   buf_gate_q, buf_down};

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      for (std::size_t l = 0; l < offload_.size(); ++l)
        encode_layer(enc, offload_[l], l, buf_x, s, pos);
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: decode_token command buffer did not complete"});

      std::memcpy(x, buf_x.contents, dim * sizeof(float));
    }
    return {};
  }

private:
  struct Scratch {
    id<MTLBuffer> normed;
    id<MTLBuffer> normed_q;
    id<MTLBuffer> q;
    id<MTLBuffer> attn;
    id<MTLBuffer> attn_q;
    id<MTLBuffer> scores;
    id<MTLBuffer> proj;
    id<MTLBuffer> gate;
    id<MTLBuffer> up;
    id<MTLBuffer> gate_q;
    id<MTLBuffer> down;
  };

  GpuWeight make_weight(const QWeight &w, std::size_t out, std::size_t in) {
    const std::size_t bytes = weight_bytes(w.type, out, in);
    GpuWeight g;
    g.type = w.type;
    g.out = static_cast<std::uint32_t>(out);
    g.in = static_cast<std::uint32_t>(in);
    g.buf = wrap_bytes(w.data, bytes);
    return g;
  }

  // one weight matmul, dispatching on dtype. c is written at byte offset c_off,
  // used to land QKV projections directly at the KV cache slot. quant weights
  // read the pre-quantized activation aq, F16 reads the fp32 a.
  void encode_weight(id<MTLComputeCommandEncoder> enc, const GpuWeight &w, id<MTLBuffer> a,
                     id<MTLBuffer> aq, id<MTLBuffer> c, std::size_t c_off) {
    switch (w.type) {
    case WeightType::F16:
      encode_mul_mat(enc, w.buf, 0, a, c, 1, w.out, w.in, c_off);
      return;
    case WeightType::Q8_0:
      encode_mul_mat_quant(enc, mul_mat_q8_, w.buf, aq, c, 1, w.out, w.in, c_off);
      return;
    case WeightType::Q4_0:
      encode_mul_mat_quant(enc, mul_mat_q4_, w.buf, aq, c, 1, w.out, w.in, c_off);
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
    encode_weight(enc, L.wq, s.normed, s.normed_q, s.q, 0);
    encode_weight(enc, L.wk, s.normed, s.normed_q, cache_k_, slot_off);
    encode_weight(enc, L.wv, s.normed, s.normed_q, cache_v_, slot_off);
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
    encode_weight(enc, L.wo, s.attn, s.attn_q, s.proj, 0);
    encode_add(enc, buf_x, 0, s.proj, dim);

    encode_rmsnorm(enc, buf_x, 0, L.ffn_norm, s.normed, 0, 1, dim, L.eps);
    if (quant_any(L.wgate.type, L.wup.type))
      encode_quant_q8(enc, s.normed, s.normed_q, 1, dim / 32);
    encode_weight(enc, L.wgate, s.normed, s.normed_q, s.gate, 0);
    encode_weight(enc, L.wup, s.normed, s.normed_q, s.up, 0);
    encode_swiglu(enc, s.gate, s.up, ff);
    if (quant_any(L.wdown.type))
      encode_quant_q8(enc, s.gate, s.gate_q, 1, ff / 32);
    encode_weight(enc, L.wdown, s.gate, s.gate_q, s.down, 0);
    encode_add(enc, buf_x, 0, s.down, dim);
  }

  static bool quant_any(WeightType a) { return a != WeightType::F16; }
  static bool quant_any(WeightType a, WeightType b, WeightType c) {
    return quant_any(a) || quant_any(b) || quant_any(c);
  }
  static bool quant_any(WeightType a, WeightType b) { return quant_any(a) || quant_any(b); }

private:
  void encode_mul_mat(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> w, std::size_t w_off,
                      id<MTLBuffer> a, id<MTLBuffer> c, std::size_t m, std::size_t out,
                      std::size_t in, std::size_t c_off = 0) {
    const MatDims dims{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(out),
                       static_cast<std::uint32_t>(in)};
    [enc setComputePipelineState:mul_mat_];
    [enc setBuffer:w offset:w_off atIndex:0];
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBuffer:c offset:c_off atIndex:2];
    [enc setBytes:&dims length:sizeof(dims) atIndex:3];
    const MTLSize grid = MTLSizeMake(out, m, 1);
    const MTLSize group = MTLSizeMake(mul_mat_.threadExecutionWidth, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
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
                            std::size_t out, std::size_t in, std::size_t c_off = 0) {
    const MatDims dims{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(out),
                       static_cast<std::uint32_t>(in)};
    [enc setComputePipelineState:pipe];
    [enc setBuffer:w offset:0 atIndex:0];
    [enc setBuffer:xq offset:0 atIndex:1];
    [enc setBuffer:c offset:c_off atIndex:2];
    [enc setBytes:&dims length:sizeof(dims) atIndex:3];
    const MTLSize grid = MTLSizeMake(out, m, 1);
    const MTLSize group = MTLSizeMake(pipe.threadExecutionWidth, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
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
    dispatch_1d(enc, rmsnorm_, rows);
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
    dispatch_1d(enc, attention_, n_heads);
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

  std::vector<LayerGpu> offload_;
  id<MTLBuffer> cache_k_ = nil;
  id<MTLBuffer> cache_v_ = nil;
  std::size_t cache_capacity_ = 0;
  std::size_t cache_pos_stride_ = 0;
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
    if (mul_mat == nil || rmsnorm == nil || rope == nil || attention == nil ||
        add_inplace == nil || swiglu == nil || quant_q8 == nil || mul_mat_q8 == nil ||
        mul_mat_q4 == nil)
      return nullptr;

    return std::make_unique<MetalBackend>(device, queue, mul_mat, rmsnorm, rope, attention,
                                          add_inplace, swiglu, quant_q8, mul_mat_q8, mul_mat_q4);
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
