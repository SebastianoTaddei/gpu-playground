#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>

#include "device_types.hpp"

namespace gpu_playground::backend
{

using HandlePtr = std::unique_ptr<void, std::function<void(void *)>>;

class Buffer
{
private:
  HandlePtr m_handle;
  size_t m_size;
  DeviceType m_device_type;

public:
  Buffer()                          = delete;
  Buffer(Buffer const &)            = delete;
  Buffer &operator=(Buffer const &) = delete;
  Buffer(Buffer &&)                 = default;
  Buffer &operator=(Buffer &&)      = default;
  ~Buffer()                         = default;

  Buffer(HandlePtr handle, size_t size, DeviceType device_type)
      : m_handle(std::move(handle)), m_size(size), m_device_type(device_type)
  {
  }

  [[nodiscard]] void *get() { return this->m_handle.get(); }

  [[nodiscard]] void const *get() const { return this->m_handle.get(); }

  [[nodiscard]] size_t size() const { return this->m_size; }

  [[nodiscard]] DeviceType device_type() const { return this->m_device_type; }
};

template <typename... Rest>
inline void assert_is_buffer()
{
  static_assert((std::is_same_v<Buffer, Rest> && ...), "Only Buffers are supported");
}

template <typename... Rest>
inline void assert_same_device(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_is_buffer<Rest...>();
  DeviceType const ref = first.device_type();
  ((assert(rest.device_type() == ref && "Buffers are on different devices")), ...);
#endif
}

template <typename... Rest>
inline void assert_same_size(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_is_buffer<Rest...>();
  std::size_t const ref = first.size();
  ((assert(rest.size() == ref && "Buffers have different sizes")), ...);
#endif
}

template <typename... Rest>
inline void assert_size_nonzero(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_is_buffer<Rest...>();
  assert(first.size() > 0 && "Buffers have zero size");
  ((assert(rest.size() > 0 && "Buffers have zero size")), ...);
#endif
}

template <typename... Rest>
inline void assert_compatible(Buffer const &first, Rest const &...rest)
{
#ifndef NDEBUG
  assert_is_buffer<Rest...>();
  assert_same_device(first, rest...);
  assert_same_size(first, rest...);
  assert_size_nonzero(first, rest...);
#endif
}

} // namespace gpu_playground::backend
