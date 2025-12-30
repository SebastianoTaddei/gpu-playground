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
  id<MTLComputePipelineState> vec_add_ps{nil};

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

    id<MTLFunction> fn = [this->library newFunctionWithName:@"vec_add"];
    assert(fn != nil);

    this->vec_add_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->vec_add_ps != nil);

    [fn release];
  }

  Impl(Impl const &)            = delete;
  Impl(Impl &&)                 = delete;
  Impl &operator=(Impl const &) = delete;
  Impl &operator=(Impl &&)      = delete;

  ~Impl()
  {
    [this->vec_add_ps release];
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
    assert_compatible(a, b, c);

    auto mtl_a = static_cast<MetalBuffer>(a.get());
    auto mtl_b = static_cast<MetalBuffer>(b.get());
    auto mtl_c = static_cast<MetalBuffer>(c.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->vec_add_ps];
    [enc setBuffer:mtl_a offset:0 atIndex:0];
    [enc setBuffer:mtl_b offset:0 atIndex:1];
    [enc setBuffer:mtl_c offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->vec_add_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

Buffer MetalDevice::new_buffer(std::vector<float> data) const
{
  assert(this->pimpl->device != nil);

  MetalBuffer mtl_buffer = [this->pimpl->device newBufferWithBytes:data.data()
                                                            length:data.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
  auto const size        = data.size();
  return Buffer{
      HandlePtr{
          mtl_buffer,
          [](void *ptr) -> void
          {
            auto buf = static_cast<MetalBuffer>(ptr);
            [buf release];
          }
      },
      size,
      MetalDevice::s_type
  };
}

void MetalDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  @autoreleasepool
  {
    assert_compatible(from, to);

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
