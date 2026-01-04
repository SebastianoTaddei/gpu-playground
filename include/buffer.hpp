#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>

#include "device_types.hpp"
#include "shape.hpp"

namespace gpu_playground::backend
{

using HandlePtr = std::unique_ptr<void, std::function<void(void *)>>;

class Buffer
{
private:
  HandlePtr m_handle;
  Shape m_shape;
  size_t m_size;
  DeviceType m_device_type;

public:
  Buffer()                          = delete;
  Buffer(Buffer const &)            = delete;
  Buffer &operator=(Buffer const &) = delete;
  Buffer(Buffer and)                = default;
  Buffer &operator=(Buffer and)     = default;
  ~Buffer()                         = default;

  Buffer(HandlePtr handle, Shape shape, DeviceType device_type)
      : m_handle(std::move(handle)), m_shape(shape), m_size(shape.rows * shape.cols),
        m_device_type(device_type)
  {
  }

  [[nodiscard]] void *get() { return this->m_handle.get(); }

  [[nodiscard]] void const *get() const { return this->m_handle.get(); }

  [[nodiscard]] Shape shape() const { return this->m_shape; }

  [[nodiscard]] size_t size() const { return this->m_size; }

  [[nodiscard]] DeviceType device_type() const { return this->m_device_type; }
};

template <typename... Rest>
inline void assert_is_buffer()
{
  static_assert((std::is_same_v<Buffer, Rest> and ...), "Only Buffers are supported");
}

template <typename... Rest>
inline void assert_same_device(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_is_buffer<Rest...>();
  DeviceType const ref = first.device_type();
  (assert(rest.device_type() == ref and "Buffers are on different devices"), ...);
#endif
}

template <typename... Rest>
inline void assert_size_nonzero(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_is_buffer<Rest...>();
  assert(first.size() > 0 and "Buffers have zero size");
  (assert(rest.size() > 0 and "Buffers have zero size"), ...);
#endif
}

template <typename... Rest>
inline void assert_valid_buffers(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_is_buffer<Rest...>();
  assert_same_device(first, rest...);
  assert_size_nonzero(first, rest...);
#endif
}

template <typename... Rest>
inline void assert_valid_copy(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  auto const rows = first.shape().rows;
  auto const cols = first.shape().cols;
  (assert(rest.shape().rows == rows and "Buffers must have the same number of rows"), ...);
  (assert(rest.shape().cols == cols and "Buffers must have the same number of columns"), ...);
#endif
}

inline void assert_valid_mul(Buffer const &a, Buffer const &b, Buffer const &c)
{
#ifndef NDEBUG
  auto const rows = a.shape().rows;
  auto const cols = b.shape().cols;
  assert(a.shape().cols == b.shape().rows and "Input buffers shape error");
  assert(
      (c.shape().rows == a.shape().rows and c.shape().cols == b.shape().cols) and
      "Output buffer shape error"
  );
#endif
}

template <typename... Rest>
inline void assert_compatible_copy(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_valid_buffers(first, rest...);
  assert_valid_copy(first, rest...);
#endif
}

template <typename... Rest>
inline void assert_same_shape(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_valid_buffers(first, rest...);
  auto const rows = first.shape().rows;
  auto const cols = first.shape().cols;
  (assert(rest.shape().rows == rows and "Buffers must have the same number of rows"), ...);
  (assert(rest.shape().cols == cols and "Buffers must have the same number of columns"), ...);
#endif
}

inline void assert_compatible_mul(Buffer const &a, Buffer const &b, Buffer const &c)
{
#ifndef NDEBUG
  assert_valid_buffers(a, b, c);
  assert_valid_mul(a, b, c);
#endif
}

} // namespace gpu_playground::backend
