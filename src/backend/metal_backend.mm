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

class MetalBackend final : public Backend {
public:
  MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue,
               id<MTLComputePipelineState> mul_mat, id<MTLComputePipelineState> rmsnorm,
               id<MTLComputePipelineState> rope, id<MTLComputePipelineState> attention,
               id<MTLComputePipelineState> add_inplace, id<MTLComputePipelineState> swiglu)
      : device_(device), queue_(queue), mul_mat_(mul_mat), rmsnorm_(rmsnorm), rope_(rope),
        attention_(attention), add_inplace_(add_inplace), swiglu_(swiglu) {}

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

  [[nodiscard]] std::expected<void, Error> decode_layer(const LayerF16 &L, float *x,
                                                        std::int32_t pos, const float *k_hist,
                                                        const float *v_hist, float *k_out,
                                                        float *v_out) override {
    @autoreleasepool {
      const std::size_t dim = L.dim;
      const std::size_t ff = L.ff;
      const std::size_t nh = L.n_heads;
      const std::size_t nkv = L.n_kv_heads;
      const std::size_t hd = L.head_dim;
      const std::size_t pos_stride = nkv * hd;
      const std::size_t n_pos = static_cast<std::size_t>(pos) + 1;
      const std::size_t q_n = nh * hd;

      id<MTLBuffer> buf_x = new_from(x, dim * sizeof(float));
      id<MTLBuffer> buf_normed = new_zeroed(dim * sizeof(float));
      id<MTLBuffer> buf_q = new_zeroed(q_n * sizeof(float));
      // kv buffers hold history [0, pos) then this position's k/v at slot pos.
      id<MTLBuffer> buf_k = new_zeroed(n_pos * pos_stride * sizeof(float));
      id<MTLBuffer> buf_v = new_zeroed(n_pos * pos_stride * sizeof(float));
      id<MTLBuffer> buf_attn = new_zeroed(q_n * sizeof(float));
      id<MTLBuffer> buf_scores = new_zeroed(nh * n_pos * sizeof(float));
      id<MTLBuffer> buf_proj = new_zeroed(dim * sizeof(float));
      id<MTLBuffer> buf_gate = new_zeroed(ff * sizeof(float));
      id<MTLBuffer> buf_up = new_zeroed(ff * sizeof(float));
      id<MTLBuffer> buf_down = new_zeroed(dim * sizeof(float));

      id<MTLBuffer> wq = wrap_weights(L.wq, q_n * dim * sizeof(std::uint16_t));
      id<MTLBuffer> wk = wrap_weights(L.wk, pos_stride * dim * sizeof(std::uint16_t));
      id<MTLBuffer> wv = wrap_weights(L.wv, pos_stride * dim * sizeof(std::uint16_t));
      id<MTLBuffer> wo = wrap_weights(L.wo, dim * q_n * sizeof(std::uint16_t));
      id<MTLBuffer> wgate = wrap_weights(L.wgate, ff * dim * sizeof(std::uint16_t));
      id<MTLBuffer> wup = wrap_weights(L.wup, ff * dim * sizeof(std::uint16_t));
      id<MTLBuffer> wdown = wrap_weights(L.wdown, dim * ff * sizeof(std::uint16_t));
      id<MTLBuffer> attn_norm = new_from(L.attn_norm, dim * sizeof(float));
      id<MTLBuffer> ffn_norm = new_from(L.ffn_norm, dim * sizeof(float));
      if (buf_x == nil || buf_normed == nil || buf_q == nil || buf_k == nil || buf_v == nil ||
          buf_attn == nil || buf_scores == nil || buf_proj == nil || buf_gate == nil ||
          buf_up == nil || buf_down == nil || wq == nil || wk == nil || wv == nil || wo == nil ||
          wgate == nil || wup == nil || wdown == nil || attn_norm == nil || ffn_norm == nil)
        return std::unexpected(Error{"metal: decode_layer buffer allocation failed"});

      if (pos > 0) {
        std::memcpy(buf_k.contents, k_hist,
                    static_cast<std::size_t>(pos) * pos_stride * sizeof(float));
        std::memcpy(buf_v.contents, v_hist,
                    static_cast<std::size_t>(pos) * pos_stride * sizeof(float));
      }
      const std::size_t slot_off = static_cast<std::size_t>(pos) * pos_stride * sizeof(float);

      id<MTLBuffer> bq = L.bq ? new_from(L.bq, q_n * sizeof(float)) : nil;
      id<MTLBuffer> bk = L.bk ? new_from(L.bk, pos_stride * sizeof(float)) : nil;
      id<MTLBuffer> bv = L.bv ? new_from(L.bv, pos_stride * sizeof(float)) : nil;

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      encode_rmsnorm(enc, buf_x, 0, attn_norm, buf_normed, 0, 1, dim, L.rms_eps);
      encode_mul_mat(enc, wq, 0, buf_normed, buf_q, 1, q_n, dim);
      encode_mul_mat(enc, wk, 0, buf_normed, buf_k, 1, pos_stride, dim, slot_off);
      encode_mul_mat(enc, wv, 0, buf_normed, buf_v, 1, pos_stride, dim, slot_off);
      if (bq)
        encode_add(enc, buf_q, 0, bq, q_n);
      if (bk)
        encode_add(enc, buf_k, slot_off, bk, pos_stride);
      if (bv)
        encode_add(enc, buf_v, slot_off, bv, pos_stride);
      encode_rope_pos(enc, buf_q, 0, pos, L.rope_theta, nh, hd);
      encode_rope_pos(enc, buf_k, slot_off, pos, L.rope_theta, nkv, hd);
      encode_attention(enc, buf_q, 0, buf_k, 0, buf_v, 0, buf_attn, buf_scores, n_pos, nh, nkv, hd);
      encode_mul_mat(enc, wo, 0, buf_attn, buf_proj, 1, dim, q_n);
      encode_add(enc, buf_x, 0, buf_proj, dim);
      encode_rmsnorm(enc, buf_x, 0, ffn_norm, buf_normed, 0, 1, dim, L.rms_eps);
      encode_mul_mat(enc, wgate, 0, buf_normed, buf_gate, 1, ff, dim);
      encode_mul_mat(enc, wup, 0, buf_normed, buf_up, 1, ff, dim);
      encode_swiglu(enc, buf_gate, buf_up, ff);
      encode_mul_mat(enc, wdown, 0, buf_gate, buf_down, 1, dim, ff);
      encode_add(enc, buf_x, 0, buf_down, dim);

      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: decode_layer command buffer did not complete"});

      std::memcpy(x, buf_x.contents, dim * sizeof(float));
      const auto *k_slot = static_cast<const std::byte *>(buf_k.contents) + slot_off;
      const auto *v_slot = static_cast<const std::byte *>(buf_v.contents) + slot_off;
      std::memcpy(k_out, k_slot, pos_stride * sizeof(float));
      std::memcpy(v_out, v_slot, pos_stride * sizeof(float));
    }
    return {};
  }

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
    if (page_aligned(W))
      return [device_ newBufferWithBytesNoCopy:const_cast<std::uint16_t *>(W)
                                        length:round_up_page(bytes)
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
    return [device_ newBufferWithBytes:W length:bytes options:MTLResourceStorageModeShared];
  }

  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> mul_mat_;
  id<MTLComputePipelineState> rmsnorm_;
  id<MTLComputePipelineState> rope_;
  id<MTLComputePipelineState> attention_;
  id<MTLComputePipelineState> add_inplace_;
  id<MTLComputePipelineState> swiglu_;
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
    if (mul_mat == nil || rmsnorm == nil || rope == nil || attention == nil ||
        add_inplace == nil || swiglu == nil)
      return nullptr;

    return std::make_unique<MetalBackend>(device, queue, mul_mat, rmsnorm, rope, attention,
                                          add_inplace, swiglu);
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
