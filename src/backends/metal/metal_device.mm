#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <unordered_map>

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

struct MetalBuffer
{
  id<MTLBuffer> buffer{nil};
  mutable id<MTLCommandBuffer> last_cmd{nil};
};

struct MetalDevice::Impl
{
  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  id<MTLLibrary> library{nil};
  std::unordered_map<std::string, id<MTLComputePipelineState>> ps;

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

    this->add_ps("mat_add");
    this->add_ps("mat_sub");
    this->add_ps("mat_mul");
    this->add_ps("mat_cmul");
    this->add_ps("mat_cdiv");
    this->add_ps("mat_sadd");
    this->add_ps("mat_ssub");
    this->add_ps("mat_smul");
    this->add_ps("mat_sdiv");
    this->add_ps("mat_trans");
  }

  Impl(Impl const &)            = delete;
  Impl(Impl &&)                 = delete;
  Impl &operator=(Impl const &) = delete;
  Impl &operator=(Impl &&)      = delete;

  ~Impl()
  {
    for (auto &[name, pipeline] : this->ps)
    {
      [pipeline release];
    }
    [this->library release];
    [this->queue release];
    [this->device release];
  }

  void add_ps(std::string const &kernel)
  {
    NSError *error = nil;
    id<MTLFunction> fn =
        [this->library newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
    assert(fn != nil);

    this->ps[kernel] = [this->device newComputePipelineStateWithFunction:fn error:&error];
    assert(this->ps[kernel] != nil);
    assert(error == nil);

    [fn release];
  }

  void cwise_op(Buffer const &a, Buffer const &b, Buffer &c, std::string const &kernel)
  {
    @autoreleasepool
    {
      assert_same_shape(a, b, c);

      auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
      auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
      auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

      id<MTLCommandBuffer> cmd = [this->queue commandBuffer];
      [cmd retain];

      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      [enc setComputePipelineState:this->ps[kernel]];
      [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
      [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
      [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];

      NSUInteger const n = a.size();

      MTLSize const gridSize = MTLSizeMake(n, 1, 1);
      NSUInteger const tgSize =
          std::min<NSUInteger>(this->ps[kernel].maxTotalThreadsPerThreadgroup, n);

      MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

      [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

      [enc endEncoding];
      [cmd commit];

      cmd_swap(mtl_c->last_cmd, cmd);
    }
  }

  void cwises_op(Buffer const &a, Buffer const &b, Buffer &c, std::string const &kernel)
  {
    @autoreleasepool
    {
      assert_compatible_sop(a, b, c);

      auto const *mtl_a = static_cast<MetalBuffer const *>(a.get());
      auto const *mtl_b = static_cast<MetalBuffer const *>(b.get());
      auto *mtl_c       = static_cast<MetalBuffer *>(c.get());

      id<MTLCommandBuffer> cmd = [this->queue commandBuffer];
      [cmd retain];

      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

      [enc setComputePipelineState:this->ps[kernel]];
      [enc setBuffer:mtl_a->buffer offset:0 atIndex:0];
      [enc setBuffer:mtl_b->buffer offset:0 atIndex:1];
      [enc setBuffer:mtl_c->buffer offset:0 atIndex:2];

      NSUInteger const n = a.size();

      MTLSize const gridSize = MTLSizeMake(n, 1, 1);
      NSUInteger const tgSize =
          std::min<NSUInteger>(this->ps[kernel].maxTotalThreadsPerThreadgroup, n);

      MTLSize const threadgroupSize = MTLSizeMake(tgSize, 1, 1);

      [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

      [enc endEncoding];
      [cmd commit];

      cmd_swap(mtl_c->last_cmd, cmd);
    }
  }
};

MetalDevice::MetalDevice() : pimpl(std::make_unique<Impl>()) {}

MetalDevice::~MetalDevice() = default;

void MetalDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  this->pimpl->cwise_op(a, b, c, "mat_add");
}

void MetalDevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  this->pimpl->cwise_op(a, b, c, "mat_sub");
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

    [enc setComputePipelineState:this->pimpl->ps["mat_mul"]];
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
  this->pimpl->cwise_op(a, b, c, "mat_cmul");
}

void MetalDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  this->pimpl->cwise_op(a, b, c, "mat_cdiv");
}

void MetalDevice::sadd(Buffer const &a, Buffer const &b, Buffer &c) const
{
  this->pimpl->cwises_op(a, b, c, "mat_sadd");
}

void MetalDevice::ssub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  this->pimpl->cwises_op(a, b, c, "mat_ssub");
}

void MetalDevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  this->pimpl->cwises_op(a, b, c, "mat_smul");
}

void MetalDevice::sdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  this->pimpl->cwises_op(a, b, c, "mat_sdiv");
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

    [enc setComputePipelineState:this->pimpl->ps["mat_trans"]];
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
