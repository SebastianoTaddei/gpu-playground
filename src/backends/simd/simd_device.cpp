#include "xsimd/xsimd.hpp"

#include "buffer.hpp"
#include "simd_device.hpp"
#include <iostream>

namespace gpu_playground::backend
{

using SIMDBuffer = std::vector<float, xsimd::aligned_allocator<float>>;

void SIMDDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible_add(a, b, c);

  auto const &simd_a = *static_cast<SIMDBuffer const *>(a.get());
  auto const &simd_b = *static_cast<SIMDBuffer const *>(b.get());
  auto &simd_c       = *static_cast<SIMDBuffer *>(c.get());

  size_t const size          = a.size();
  constexpr size_t simd_size = xsimd::simd_type<float>::size;
  size_t const vec_size      = size - (size % simd_size);

  for (size_t i{0}; i < vec_size; i += simd_size)
  {
    auto ba   = xsimd::load_aligned(&simd_a[i]);
    auto bb   = xsimd::load_aligned(&simd_b[i]);
    auto bres = ba + bb;
    bres.store_aligned(&simd_c[i]);
  }
  for (size_t i{vec_size}; i < size; i++)
  {
    simd_c[i] = simd_a[i] + simd_b[i];
  }
}

void SIMDDevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible_mul(a, b, c);

  auto const &simd_a = *static_cast<SIMDBuffer const *>(a.get());
  auto const &simd_b = *static_cast<SIMDBuffer const *>(b.get());
  auto &simd_c       = *static_cast<SIMDBuffer *>(c.get());

  std::cout << "SIMD matrix multiplication implemented as serial\n";

  auto const [a_rows, a_cols] = a.shape();
  auto const b_cols           = b.shape().cols;
  for (size_t i{0}; i < a_rows; i++)
  {
    for (size_t j{0}; j < b_cols; j++)
    {
      auto const idx = (i * b_cols) + j;
      float support{0.0};
      for (size_t k{0}; k < a_cols; k++)
      {
        support += simd_a[(i * a_cols) + k] * simd_b[(k * b_cols) + j];
      }
      simd_c[idx] = support;
    }
  }
}

Buffer SIMDDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  auto const size = data.size();
  return Buffer{
      HandlePtr{
          new SIMDBuffer(data.cbegin(), data.cend()),
          [](void *ptr) -> void { delete static_cast<SIMDBuffer *>(ptr); }
      },
      shape,
      SIMDDevice::s_type,
  };
}

void SIMDDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible_copy(from, to);

  auto const &simd_from = *static_cast<SIMDBuffer const *>(from.get());
  auto &simd_to         = *static_cast<SIMDBuffer *>(to.get());

  simd_to = simd_from;
}

std::vector<float> SIMDDevice::cpu(Buffer const &buffer) const
{
  auto simd_buffer = *static_cast<SIMDBuffer const *>(buffer.get());
  return {simd_buffer.cbegin(), simd_buffer.cend()};
}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_simd_device()
{
  return std::make_shared<gpu_playground::backend::SIMDDevice>();
}
