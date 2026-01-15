#include "buffer.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_device.hpp"

namespace
{

void cmd_wait_release(id<MTLCommandBuffer> &cmd)
{
  if (cmd == nil)
  {
    return;
  }

  [cmd waitUntilCompleted];
  [cmd release];
  cmd = nil;
}

void cmd_swap(id<MTLCommandBuffer> &old_cmd, id<MTLCommandBuffer> &new_cmd)
{
  if (old_cmd != nil)
  {
    [old_cmd release];
  }
  old_cmd = new_cmd;
}

} // namespace

namespace gpu_playground::backend
{

struct MetalDevice::Impl
{
  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  id<MTLLibrary> library{nil};
  id<MTLComputePipelineState> mat_add_ps{nil};
  id<MTLComputePipelineState> mat_sub_ps{nil};
  id<MTLComputePipelineState> mat_mul_ps{nil};
  id<MTLComputePipelineState> mat_cmul_ps{nil};
  id<MTLComputePipelineState> mat_cdiv_ps{nil};
  id<MTLComputePipelineState> mat_smul_ps{nil};
  id<MTLComputePipelineState> mat_trans_ps{nil};

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

    fn = [this->library newFunctionWithName:@"mat_sub"];
    assert(fn != nil);

    this->mat_sub_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->mat_sub_ps != nil);

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

    fn = [this->library newFunctionWithName:@"mat_smul"];
    assert(fn != nil);

    this->mat_smul_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->mat_smul_ps != nil);

    fn = [this->library newFunctionWithName:@"mat_trans"];
    assert(fn != nil);

    this->mat_trans_ps = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->mat_trans_ps != nil);

    [fn release];
  }

  Impl(Impl const &)            = delete;
  Impl(Impl &&)                 = delete;
  Impl &operator=(Impl const &) = delete;
  Impl &operator=(Impl &&)      = delete;

  ~Impl()
  {
    [this->mat_trans_ps release];
    [this->mat_smul_ps release];
    [this->mat_cdiv_ps release];
    [this->mat_cmul_ps release];
    [this->mat_mul_ps release];
    [this->mat_add_ps release];
    [this->mat_sub_ps release];
    [this->library release];
    [this->queue release];
    [this->device release];
  }
};

// using MetalBuffer = id<MTLBuffer>;
struct MetalBuffer
{
  id<MTLBuffer> buffer{nil};
  mutable id<MTLCommandBuffer> last_cmd{nil};
};

MetalDevice::MetalDevice() : pimpl(std::make_unique<Impl>()) {}

MetalDevice::~MetalDevice() = default;

void MetalDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_same_shape(a, b, c);

    auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
    auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
    auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_add_ps];
    [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
    [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
    [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_add_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];

    cmd_swap(mtl_c->last_cmd, cmd);
  }
}

void MetalDevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_same_shape(a, b, c);

    auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
    auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
    auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_sub_ps];
    [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
    [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
    [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_sub_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];

    cmd_swap(mtl_c->last_cmd, cmd);
  }
}

void MetalDevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_compatible_mul(a, b, c);

    auto const [m, k] = a.shape();
    auto const n      = b.shape().cols;

    auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
    auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
    auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_mul_ps];
    [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
    [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
    [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];
    [enc setBytes:&m length:sizeof(m) atIndex:3];
    [enc setBytes:&k length:sizeof(k) atIndex:4];
    [enc setBytes:&n length:sizeof(n) atIndex:5];

    MTLSize const gridSize = MTLSizeMake(n, m, 1);
    NSUInteger const tg    = 16;
    MTLSize const tgSize   = MTLSizeMake(tg, tg, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    [enc endEncoding];
    [cmd commit];

    cmd_swap(mtl_c->last_cmd, cmd);
  }
}

void MetalDevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_same_shape(a, b, c);

    auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
    auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
    auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_cmul_ps];
    [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
    [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
    [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_cmul_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];

    cmd_swap(mtl_c->last_cmd, cmd);
  }
}

void MetalDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_same_shape(a, b, c);

    auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
    auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
    auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_cdiv_ps];
    [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
    [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
    [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_cdiv_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];

    cmd_swap(mtl_c->last_cmd, cmd);
  }
}

void MetalDevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  @autoreleasepool
  {
    assert_compatible_smul(a, b, c);

    auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
    auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
    auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_smul_ps];
    [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
    [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
    [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];

    NSUInteger const n = a.size();

    MTLSize const gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger const tgSize =
        std::min<NSUInteger>(this->pimpl->mat_smul_ps.maxTotalThreadsPerThreadgroup, n);

    MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [enc endEncoding];
    [cmd commit];

    cmd_swap(mtl_c->last_cmd, cmd);
  }
}

Buffer MetalDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  assert(this->pimpl->device != nil);

  MetalBuffer mtl_buffer{};
  mtl_buffer.buffer = [this->pimpl->device newBufferWithBytes:data.data()
                                                       length:data.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

  return Buffer{
      HandlePtr{
          new MetalBuffer(mtl_buffer),
          [](void *ptr) -> void
          {
            auto *buf = static_cast<MetalBuffer *>(ptr);
            [buf->last_cmd release];
            [buf->buffer release];
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

    auto const *mtl_from = static_cast<MetalBuffer const *>(from.get());
    auto *mtl_to         = static_cast<MetalBuffer *>(to.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    [blit copyFromBuffer:mtl_from->buffer
             sourceOffset:0
                 toBuffer:mtl_to->buffer
        destinationOffset:0
                     size:mtl_from->buffer.length];

    [blit endEncoding];
    [cmd commit];

    cmd_swap(mtl_to->last_cmd, cmd);
  }
}

void MetalDevice::transpose(Buffer const &from, Buffer &to) const
{
  @autoreleasepool
  {
    assert_compatible_transpose(from, to);

    auto const [m, n] = from.shape();

    auto const *mtl_from = static_cast<MetalBuffer const *>(from.get());
    auto *mtl_to         = static_cast<MetalBuffer *>(to.get());

    id<MTLCommandBuffer> cmd = [this->pimpl->queue commandBuffer];
    [cmd retain];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:this->pimpl->mat_trans_ps];
    [enc setBuffer:mtl_from->buffer offset:0 atIndex:0];
    [enc setBuffer:mtl_to->buffer offset:0 atIndex:1];
    [enc setBytes:&m length:sizeof(m) atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];

    MTLSize const gridSize = MTLSizeMake(n, m, 1);
    NSUInteger const tg    = 16;
    MTLSize const tgSize   = MTLSizeMake(tg, tg, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    [enc endEncoding];
    [cmd commit];

    cmd_swap(mtl_to->last_cmd, cmd);
  }
}

std::vector<float> MetalDevice::cpu(Buffer const &buffer) const
{
  auto const *mtl_buf = static_cast<MetalBuffer const *>(buffer.get());

  cmd_wait_release(mtl_buf->last_cmd);

  std::vector<float> result(buffer.size());
  memcpy(result.data(), mtl_buf->buffer.contents, buffer.size() * sizeof(float));

  return result;
}

void MetalDevice::sync(Buffer const &buffer) const
{
  auto const *mtl_buf = static_cast<MetalBuffer const *>(buffer.get());
  cmd_wait_release(mtl_buf->last_cmd);
}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_metal_device()
{
  return std::make_shared<gpu_playground::backend::MetalDevice>();
}
