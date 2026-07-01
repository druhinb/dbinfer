#include "backend/metal_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

// one thread per (row, output). half weights read direct, fp32 accumulate. fma
// mirrors matvec_f16 so the reduction only differs by summation order.
constexpr const char *kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Dims { uint m; uint n_out; uint n_in; };

kernel void mul_mat_f16(device const half   *W    [[buffer(0)]],
                        device const float  *A    [[buffer(1)]],
                        device       float  *C    [[buffer(2)]],
                        constant     Dims   &dims [[buffer(3)]],
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
)MSL";

struct Dims {
  std::uint32_t m;
  std::uint32_t n_out;
  std::uint32_t n_in;
};

class MetalBackend final : public Backend {
public:
  MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue,
               id<MTLComputePipelineState> pipeline)
      : device_(device), queue_(queue), pipeline_(pipeline) {}

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
      id<MTLBuffer> buf_a = [device_ newBufferWithBytes:A
                                                 length:a_bytes
                                                options:MTLResourceStorageModeShared];
      id<MTLBuffer> buf_c = [device_ newBufferWithLength:c_bytes
                                                 options:MTLResourceStorageModeShared];
      if (buf_a == nil || buf_c == nil)
        return std::unexpected(Error{"metal: failed to allocate activation buffer"});

      const Dims dims{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(out),
                      static_cast<std::uint32_t>(in)};

      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pipeline_];
      [enc setBuffer:buf_w offset:0 atIndex:0];
      [enc setBuffer:buf_a offset:0 atIndex:1];
      [enc setBuffer:buf_c offset:0 atIndex:2];
      [enc setBytes:&dims length:sizeof(dims) atIndex:3];

      const NSUInteger width = pipeline_.threadExecutionWidth;
      const MTLSize grid = MTLSizeMake(out, m, 1);
      const MTLSize group = MTLSizeMake(width, 1, 1);
      [enc dispatchThreads:grid threadsPerThreadgroup:group];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];

      if (cmd.status != MTLCommandBufferStatusCompleted)
        return std::unexpected(Error{"metal: command buffer did not complete"});

      std::memcpy(C, buf_c.contents, c_bytes);
    }
    return {};
  }

private:
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
  id<MTLComputePipelineState> pipeline_;
};

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

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:kShaderSource];
    id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&err];
    if (library == nil) {
      std::fprintf(stderr, "metal: shader compile failed: %s\n",
                   err.localizedDescription.UTF8String);
      return nullptr;
    }
    id<MTLFunction> fn = [library newFunctionWithName:@"mul_mat_f16"];
    if (fn == nil) {
      std::fprintf(stderr, "metal: mul_mat_f16 function not found\n");
      return nullptr;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn
                                                                                 error:&err];
    if (pipeline == nil) {
      std::fprintf(stderr, "metal: pipeline creation failed: %s\n",
                   err.localizedDescription.UTF8String);
      return nullptr;
    }
    return std::make_unique<MetalBackend>(device, queue, pipeline);
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
