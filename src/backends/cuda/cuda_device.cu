#include <cuda_runtime.h>
#include <iostream>

#include "mat_add.cu"
#include "mat_sub.cu"
#include "mat_cmul.cu"
#include "mat_cdiv.cu"
#include "mat_sadd.cu"
#include "mat_ssub.cu"
#include "mat_smul.cu"
#include "mat_sdiv.cu"
#include "mat_mul.cu"
#include "mat_trans.cu"

#include "cuda_device.hpp"

#define CHECK(call)                                                                                \
  {                                                                                                \
    const cudaError_t error = call;                                                                \
    if (error != cudaSuccess)                                                                      \
    {                                                                                              \
      std::cerr << "Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                                                      \
      exit(1);                                                                                     \
    }                                                                                              \
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

  ~Impl() { cudaStreamDestroy(this->stream); }
};

CUDADevice::CUDADevice() : pimpl(std::make_unique<Impl>()) {}

CUDADevice::~CUDADevice() = default;

void CUDADevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_add<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_same_shape(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_sub<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_compatible_mul(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  auto const [m, k]    = a.shape();
  int const n          = b.shape().cols;
  auto const blockSize = dim3(16, 16);
  auto const gridSize  = dim3((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

  mat_mul<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, m, k, n
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_same_shape(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_cmul<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_same_shape(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_cdiv<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::sadd(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_compatible_sop(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_sadd<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::ssub(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_compatible_sop(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_ssub<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_compatible_sop(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_smul<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

void CUDADevice::sdiv(Buffer const &a, Buffer const &b, Buffer &c) const 
{
  assert_compatible_sop(a, b, c);

  auto const *cu_a = static_cast<CUDABuffer const *>(a.get());
  auto const *cu_b = static_cast<CUDABuffer const *>(b.get());
  auto *cu_c       = static_cast<CUDABuffer *>(c.get());

  int const N         = a.size();
  int const blockSize = 256;
  int const gridSize  = (N + blockSize - 1) / blockSize;

  mat_sdiv<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_a->buffer, cu_b->buffer, cu_c->buffer, N
  );
  CHECK(cudaGetLastError());
}

Buffer CUDADevice::new_buffer(std::vector<float> data, Shape shape) const
{
  auto const bytes = data.size() * sizeof(float);
  CUDABuffer cu_buffer{};
  CHECK(cudaMallocAsync(&(cu_buffer.buffer), bytes, this->pimpl->stream));
  CHECK(cudaMemcpyAsync(
      cu_buffer.buffer, data.data(), bytes, cudaMemcpyHostToDevice, this->pimpl->stream
  ));

  return Buffer{
      HandlePtr{
          new CUDABuffer(cu_buffer),
          [this](void *ptr) -> void
          {
            auto cu_ptr = static_cast<CUDABuffer *>(ptr);
            CHECK(cudaFreeAsync(cu_ptr->buffer, this->pimpl->stream));
            std::default_delete<CUDABuffer>{}(cu_ptr);
          }
      },
      shape,
      CUDADevice::s_type
  };
}

void CUDADevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible_copy(from, to);

  auto const bytes    = from.size() * sizeof(float);
  auto const *cu_from = static_cast<CUDABuffer const *>(from.get());
  auto *cu_to         = static_cast<CUDABuffer *>(to.get());

  CHECK(cudaMemcpyAsync(
      cu_to->buffer, cu_from->buffer, bytes, cudaMemcpyDeviceToDevice, this->pimpl->stream
  ));
}

void CUDADevice::transpose(Buffer const &from, Buffer &to) const 
{
  assert_compatible_transpose(from, to);

  auto const *cu_from = static_cast<CUDABuffer const *>(from.get());
  auto *cu_to         = static_cast<CUDABuffer *>(to.get());

  auto const [m, n]    = from.shape();
  auto const blockSize = dim3(16, 16);
  auto const gridSize  = dim3((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

  mat_trans<<<gridSize, blockSize, 0, this->pimpl->stream>>>(
      cu_from->buffer, cu_to->buffer, m, n
  );
  CHECK(cudaGetLastError());
}

std::vector<float> CUDADevice::cpu(Buffer const &buffer) const
{
  auto const *cu_ptr = static_cast<CUDABuffer const *>(buffer.get());

  auto const bytes = buffer.size() * sizeof(float);
  std::vector<float> result(buffer.size());
  CHECK(cudaMemcpyAsync(
      result.data(), cu_ptr->buffer, bytes, cudaMemcpyDeviceToHost, this->pimpl->stream
  ));

  CHECK(cudaStreamSynchronize(this->pimpl->stream));

  return result;
}

void CUDADevice::sync(Buffer const &buffer) const
{
  CHECK(cudaStreamSynchronize(this->pimpl->stream));
}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_cuda_device()
{
  return std::make_shared<gpu_playground::backend::CUDADevice>();
}
