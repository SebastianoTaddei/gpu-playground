#include <cmath>

#include <memory>
#include <xsimd/xsimd.hpp>

#include "simd_device.hpp"

namespace gpu_playground::backend
{

using SIMDBuffer = std::vector<float, xsimd::aligned_allocator<float>>;

namespace
{

struct Add
{
  [[nodiscard]] xsimd::batch<float>
  operator()(xsimd::batch<float> const a, xsimd::batch<float> const b) const
  {
    return a + b;
  }

  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a + b; }
};

struct Sub
{
  [[nodiscard]] xsimd::batch<float>
  operator()(xsimd::batch<float> const a, xsimd::batch<float> const b) const
  {
    return a - b;
  }

  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a - b; }
};

struct Mul
{
  [[nodiscard]] xsimd::batch<float>
  operator()(xsimd::batch<float> const a, xsimd::batch<float> const b) const
  {
    return a * b;
  }

  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a * b; }
};

struct Div
{
  [[nodiscard]] xsimd::batch<float>
  operator()(xsimd::batch<float> const a, xsimd::batch<float> const b) const
  {
    return a / b;
  }

  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a / b; }
};

template <class Op>
void cwisem_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_same_shape(a, b, c);

  auto const &simd_a = *static_cast<SIMDBuffer const *>(a.get());
  auto const &simd_b = *static_cast<SIMDBuffer const *>(b.get());
  auto &simd_c       = *static_cast<SIMDBuffer *>(c.get());

  size_t const size          = a.size();
  constexpr size_t simd_size = xsimd::simd_type<float>::size;
  size_t const vec_size      = size - (size % simd_size);

  for (size_t i{0}; i < vec_size; i += simd_size)
  {
    auto const ba   = xsimd::load_aligned(&simd_a[i]);
    auto const bb   = xsimd::load_aligned(&simd_b[i]);
    auto const bres = op(ba, bb);
    bres.store_aligned(&simd_c[i]);
  }
  for (size_t i{vec_size}; i < size; i++)
  {
    simd_c[i] = op(simd_a[i], simd_b[i]);
  }
}

template <class Op>
void cwises_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_compatible_sop(a, b, c);

  auto const &simd_a = *static_cast<SIMDBuffer const *>(a.get());
  auto const &simd_b = *static_cast<SIMDBuffer const *>(b.get());
  auto &simd_c       = *static_cast<SIMDBuffer *>(c.get());

  size_t const size          = a.size();
  constexpr size_t simd_size = xsimd::simd_type<float>::size;
  size_t const vec_size      = size - (size % simd_size);

  auto const sb = simd_b.front();
  auto const bb = xsimd::broadcast(sb);
  for (size_t i{0}; i < vec_size; i += simd_size)
  {
    auto const ba   = xsimd::load_aligned(&simd_a[i]);
    auto const bres = op(ba, bb);
    bres.store_aligned(&simd_c[i]);
  }
  for (size_t i{vec_size}; i < size; i++)
  {
    simd_c[i] = op(simd_a[i], sb);
  }
}

} // namespace

void SIMDDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Add{});
}

void SIMDDevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Sub{});
}

void SIMDDevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible_mul(a, b, c);

  auto const &simd_a = *static_cast<SIMDBuffer const *>(a.get());
  auto const &simd_b = *static_cast<SIMDBuffer const *>(b.get());
  auto &simd_c       = *static_cast<SIMDBuffer *>(c.get());

  auto const [m, k]          = a.shape();
  auto const n               = b.shape().cols;
  constexpr size_t simd_size = xsimd::simd_type<float>::size;
  size_t const n_simd        = n - (n % simd_size);

  for (size_t i{0}; i < m; i++)
  {
    for (size_t p = 0; p < k; ++p)
    {
      auto const a_ip = xsimd::broadcast(simd_a[(i * k) + p]);

      for (size_t j{0}; j < n_simd; j += simd_size)
      {
        auto c       = xsimd::load_aligned(&simd_c[(i * n) + j]);
        auto const b = xsimd::load_aligned(&simd_b[(p * n) + j]);
        c            = xsimd::fma(a_ip, b, c);
        c.store_aligned(&simd_c[(i * n) + j]);
      }

      for (size_t j{n_simd}; j < n; j++)
      {
        simd_c[(i * n) + j] =
            std::fma(simd_a[(i * k) + p], simd_b[(p * n) + j], simd_c[(i * n) + j]);
      }
    }
  }
}

void SIMDDevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Mul{});
}

void SIMDDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Div{});
}

void SIMDDevice::sadd(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Add{});
}

void SIMDDevice::ssub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Sub{});
}

void SIMDDevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Mul{});
}

void SIMDDevice::sdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Div{});
}

Buffer SIMDDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  auto const size = data.size();
  return Buffer{
      HandlePtr{
          new SIMDBuffer(data.cbegin(), data.cend()),
          [](void *ptr) -> void
          { std::default_delete<SIMDBuffer>{}(static_cast<SIMDBuffer *>(ptr)); }
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

void SIMDDevice::transpose(Buffer const &from, Buffer &to) const
{
  assert_compatible_transpose(from, to);

  auto const &simd_from = *static_cast<SIMDBuffer const *>(from.get());
  auto &simd_to         = *static_cast<SIMDBuffer *>(to.get());

  auto const [rows, cols] = from.shape();
  for (size_t i{0}; i < rows; i++)
  {
    for (size_t j{0}; j < cols; j++)
    {
      simd_to[(j * rows) + i] = simd_from[(i * cols) + j];
    }
  }
}

std::vector<float> SIMDDevice::cpu(Buffer const &buffer) const
{
  auto simd_buffer = *static_cast<SIMDBuffer const *>(buffer.get());
  return {simd_buffer.cbegin(), simd_buffer.cend()};
}

void SIMDDevice::sync(Buffer const &buffer) const {}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_simd_device()
{
  return std::make_shared<gpu_playground::backend::SIMDDevice>();
}
