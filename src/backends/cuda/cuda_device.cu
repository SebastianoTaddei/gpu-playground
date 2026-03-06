#include <cuda_runtime.h>
#include <iostream>

#include "vec_add.cu"

#include "cuda_device.hpp"

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << cudaGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

namespace gpu_playground::backend
{

struct CUDABuffer
{
  float *buffer{nullptr};
};

struct CUDADevice::Impl
{
  int device{0};
  cudaDeviceProp prop{};
  cudaStream_t stream{};

  Impl()
  {
    CHECK(cudaSetDevice(this->device));
    CHECK(cudaGetDeviceProperties(&(this->prop), this->device));
    CHECK(cudaStreamCreate(&(this->stream)));

    std::cout << "Using GPU: " << this->prop.name << '\n';
  }

  Impl(Impl const &)            = delete;
  Impl(Impl &&)                 = delete;
  Impl &operator=(Impl const &) = delete;
  Impl &operator=(Impl &&)      = delete;

  ~Impl()
  {
    cudaStreamDestroy(this->stream);
  }
};

CUDADevice::CUDADevice() : pimpl(std::make_unique<Impl>()) {}

void CUDADevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer const *>(c.get());

  constexpr int N = 1024;
  constexpr int blockSize = 256;
  constexpr int gridSize = (N + blockSize - 1) / blockSize;

  vec_add<<<gridSize, blockSize, 0, this->pimpl->stream>>>(cu_a->buffer, cu_b->buffer, cu_c->buffer);
}

void CUDADevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const {}

void CUDADevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const {}

void CUDADevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const {}

void CUDADevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const {}

void CUDADevice::sadd(Buffer const &a, Buffer const &b, Buffer &c) const {}

void CUDADevice::ssub(Buffer const &a, Buffer const &b, Buffer &c) const {}

void CUDADevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const {}

void CUDADevice::sdiv(Buffer const &a, Buffer const &b, Buffer &c) const {}

Buffer CUDADevice::new_buffer(std::vector<float> data, Shape shape) const
{
  auto const bytes = data.size() * sizeof(float);
  CUDABuffer cu_buffer{};
  CHECK(cudaMallocAsync(&(cu_buffer.buffer), bytes, this->pimpl->stream));
  CHECK(cudaMemcpyAsync(cu_buffer.buffer, data.data(), bytes, cudaMemcpyHostToDevice, this->pimpl->stream));

  return Buffer{
    HandlePtr{
      new CUDADevice(cuda_device),
      [](void *ptr) -> void
      {
        auto cu_ptr = static_cast<CUDABuffer *>(ptr);
        cudaFreeAsync(cu_ptr->buffer, this->pimpl->stream);
        std::default_delete<CUDABuffer>{}(cu_ptr);
      }
    },
    shape,
    CUDADevice::s_type
  };
}

void copy_buffer(Buffer const &from, Buffer &to) const {}

void transpose(Buffer const &from, Buffer &to) const {}

std::vector<float> CUDADevice::cpu(Buffer const &buffer) const
{
  auto const *cu_ptr = static_cast<CUDABuffer const *>(buffer.get());

  auto const bytes = buffer.size() * sizeof(float);
  std::vector<float> result(buffer.size());
  cudaMemcpyAsync(result.data(), cu_ptr->buffer, bytes, cudaMemcpyDeviceToHost, this->pimpl->stream);

  cudaStreamSynchronize(this->pimpl->stream);

  return result;
}

void sync(Buffer const &buffer) const 
{
  cudaStreamSynchronize(this->pimpl->stream);
}

}
