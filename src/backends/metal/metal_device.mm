#include "buffer.hpp"
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
    [this->device retain];
    assert(this->device != nil);

    this->queue = [this->device newCommandQueue];
    [this->queue retain];
    assert(this->queue != nil);

    NSString *path = @METAL_LIB;
    NSURL *url     = [NSURL fileURLWithPath:path];
    assert([[NSFileManager defaultManager] fileExistsAtPath:path]);

    NSError *error = nil;
    this->library  = [device newLibraryWithURL:url error:&error];
    [this->library retain];
    assert(this->library != nil);
    assert(error == nil);

    id<MTLFunction> fn = [this->library newFunctionWithName:@"vec_add"];
    [fn retain];
    assert(fn != nil);

    this->vec_add_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    [this->vec_add_ps retain];
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
    [this->queue release];
    [this->device release];
  }
};

using MetalBuffer = id<MTLBuffer>;

MetalDevice::MetalDevice() : pimpl(std::make_unique<Impl>()) {}

MetalDevice::~MetalDevice() = default;

void MetalDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible(a, b, c);

  auto const mtl_a = static_cast<MetalBuffer>(a.handle.get());
  auto const mtl_b = static_cast<MetalBuffer>(b.handle.get());
  auto mtl_c       = static_cast<MetalBuffer>(c.handle.get());

  id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];

  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

  [enc setComputePipelineState:this->pimpl->vec_add_ps];
  [enc setBuffer:mtl_a offset:0 atIndex:0];
  [enc setBuffer:mtl_b offset:0 atIndex:1];
  [enc setBuffer:mtl_c offset:0 atIndex:2];

  NSUInteger n = a.size;

  MTLSize gridSize  = MTLSizeMake(n, 1, 1);
  NSUInteger tgSize = this->pimpl->vec_add_ps.maxTotalThreadsPerThreadgroup;
  tgSize            = std::min<NSUInteger>(tgSize, n);

  MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);

  [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

Buffer MetalDevice::new_buffer(std::vector<float> data) const
{
  assert(this->pimpl->device != nil);

  MetalBuffer mtl_buffer = [this->pimpl->device newBufferWithBytes:data.data()
                                                            length:data.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];

  [mtl_buffer retain];

  return Buffer{
      .handle =
          HandlePtr{
              mtl_buffer,
              [](void *ptr) -> void
              {
                auto buf = static_cast<MetalBuffer>(ptr);
                [buf release];
              }
          },
      .size        = data.size(),
      .device_type = MetalDevice::s_type,
  };
}

void MetalDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible(from, to);

  auto const metal_from = static_cast<MetalBuffer>(from.handle.get());
  auto metal_to         = static_cast<MetalBuffer>(to.handle.get());

  id<MTLCommandBuffer> commandBuffer = [this->pimpl->device newCommandQueue].commandBuffer;
  id<MTLBlitCommandEncoder> blit     = [commandBuffer blitCommandEncoder];

  [blit copyFromBuffer:metal_from
           sourceOffset:0
               toBuffer:metal_to
      destinationOffset:0
                   size:metal_from.length];

  [blit endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

std::vector<float> MetalDevice::cpu(Buffer const &buffer) const
{
  auto metal_buffer = static_cast<MetalBuffer>(buffer.handle.get());

  std::vector<float> result(buffer.size);
  memcpy(result.data(), metal_buffer.contents, buffer.size * sizeof(float));

  return result;
}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_metal_device()
{
  return std::make_shared<gpu_playground::backend::MetalDevice>();
}
