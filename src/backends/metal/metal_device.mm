#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_device.hpp"

namespace gpu_playground::backend
{

struct MetalDevice::Impl
{
  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  id<MTLLibrary> library{nil};
  id<MTLComputePipelineState> mat_add_ps{nil};
  id<MTLComputePipelineState> mat_mul_ps{nil};
  id<MTLComputePipelineState> mat_cmul_ps{nil};
  id<MTLComputePipelineState> mat_cdiv_ps{nil};

  Impl() : device(MTLCreateSystemDefaultDevice())
  {
    assert(this->device != nil);

    this->queue = [this->device newCommandQueue];
    assert(this->queue != nil);

    NSString *path = @METAL_LIB;
    NSURL *url     = [NSURL fileURLWithPath:path];
    assert([[NSFileManager defaultManager] fileExistsAtPath:path]);

    NSError *error = nil;
    this->library  = [device newLibraryWithURL:url error:&error];
    assert(this->library != nil);
    assert(error == nil);

    id<MTLFunction> fn = [this->library newFunctionWithName:@"mat_add"];
    assert(fn != nil);

    this->mat_add_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->mat_add_ps != nil);

    fn = [this->library newFunctionWithName:@"mat_mul"];
    assert(fn != nil);

    this->mat_mul_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->mat_mul_ps != nil);

    fn = [this->library newFunctionWithName:@"mat_cmul"];
    assert(fn != nil);

    this->mat_cmul_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->mat_cmul_ps != nil);

    fn = [this->library newFunctionWithName:@"mat_cdiv"];
    assert(fn != nil);

    this->mat_cdiv_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->mat_cdiv_ps != nil);

    [fn release];
  }

  Impl(Impl const &)            = delete;
  Impl(Impl &&)                 = delete;
  Impl &operator=(Impl const &) = delete;
  Impl &operator=(Impl &&)      = delete;

  ~Impl()
  {
    [this->mat_mul_ps release];
    [this->mat_add_ps release];
    [this->library release];
    [this->queue release];
    [this->device release];
  }
};

using MetalBuffer = id<MTLBuffer>;

MetalDevice::MetalDevice() : pimpl(std::make_unique<Impl>()) {}

MetalDevice::~MetalDevice() = default;

void MetalDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_same_shape(a, b, c);

    auto mtl_a = static_cast<MetalBuffer>(a.get());
    auto mtl_b = static_cast<MetalBuffer>(b.get());
    auto mtl_c = static_cast<MetalBuffer>(c.get());

    id<MTLCommandBuffer> cmd         = [this->pimpl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_add_ps];
    [enc setBuffer:mtl_a offset:0 atIndex:0];
    [enc setBuffer:mtl_b offset:0 atIndex:1];
    [enc setBuffer:mtl_c offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_add_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

void MetalDevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_compatible_mul(a, b, c);

    auto const [m, k] = a.shape();
    auto const n      = b.shape().cols;

    auto mtl_a = static_cast<MetalBuffer>(a.get());
    auto mtl_b = static_cast<MetalBuffer>(b.get());
    auto mtl_c = static_cast<MetalBuffer>(c.get());

    id<MTLCommandBuffer> cmd         = [this->pimpl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_mul_ps];
    [enc setBuffer:mtl_a offset:0 atIndex:0];
    [enc setBuffer:mtl_b offset:0 atIndex:1];
    [enc setBuffer:mtl_c offset:0 atIndex:2];
    [enc setBytes:&m length:sizeof(m) atIndex:3];
    [enc setBytes:&k length:sizeof(k) atIndex:4];
    [enc setBytes:&n length:sizeof(n) atIndex:5];

    MTLSize const gridSize = MTLSizeMake(n, m, 1);
    NSUInteger const tg    = 16;
    MTLSize const tgSize   = MTLSizeMake(tg, tg, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

void MetalDevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_same_shape(a, b, c);

    auto mtl_a = static_cast<MetalBuffer>(a.get());
    auto mtl_b = static_cast<MetalBuffer>(b.get());
    auto mtl_c = static_cast<MetalBuffer>(c.get());

    id<MTLCommandBuffer> cmd         = [this->pimpl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_cmul_ps];
    [enc setBuffer:mtl_a offset:0 atIndex:0];
    [enc setBuffer:mtl_b offset:0 atIndex:1];
    [enc setBuffer:mtl_c offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_cmul_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

void MetalDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_same_shape(a, b, c);

    auto mtl_a = static_cast<MetalBuffer>(a.get());
    auto mtl_b = static_cast<MetalBuffer>(b.get());
    auto mtl_c = static_cast<MetalBuffer>(c.get());

    id<MTLCommandBuffer> cmd         = [this->pimpl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_cdiv_ps];
    [enc setBuffer:mtl_a offset:0 atIndex:0];
    [enc setBuffer:mtl_b offset:0 atIndex:1];
    [enc setBuffer:mtl_c offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_cdiv_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

Buffer MetalDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  assert(this->pimpl->device != nil);

  MetalBuffer mtl_buffer = [this->pimpl->device newBufferWithBytes:data.data()
                                                            length:data.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
  return Buffer{
      HandlePtr{
          mtl_buffer,
          [](void *ptr) -> void
          {
            auto buf = static_cast<MetalBuffer>(ptr);
            [buf release];
          }
      },
      shape,
      MetalDevice::s_type
  };
}

void MetalDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  @autoreleasepool
  {
    assert_compatible_copy(from, to);

    auto metal_from = static_cast<MetalBuffer>(from.get());
    auto metal_to   = static_cast<MetalBuffer>(to.get());

    id<MTLCommandBuffer> cmd       = [this->pimpl->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    [blit copyFromBuffer:metal_from
             sourceOffset:0
                 toBuffer:metal_to
        destinationOffset:0
                     size:metal_from.length];

    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

std::vector<float> MetalDevice::cpu(Buffer const &buffer) const
{
  auto metal_buffer = static_cast<MetalBuffer>(buffer.get());

  std::vector<float> result(buffer.size());
  memcpy(result.data(), metal_buffer.contents, buffer.size() * sizeof(float));

  return result;
}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_metal_device()
{
  return std::make_shared<gpu_playground::backend::MetalDevice>();
}
