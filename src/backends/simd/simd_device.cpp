#include "xsimd/xsimd.hpp"

#include "buffer.hpp"
#include "simd_device.hpp"

namespace gpu_playground::backend
{

using SIMDBuffer = std::vector<float, xsimd::aligned_allocator<float>>;

void SIMDDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible(a, b, c);

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

Buffer SIMDDevice::new_buffer(std::vector<float> data) const
{
  auto const size = data.size();
  return Buffer{
      HandlePtr{
          new SIMDBuffer(data.cbegin(), data.cend()),
          [](void *ptr) -> void { delete static_cast<SIMDBuffer *>(ptr); }
      },
      size,
      SIMDDevice::s_type,
  };
}

void SIMDDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible(from, to);

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
