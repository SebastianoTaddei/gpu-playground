#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>

#include "device_types.hpp"

namespace gpu_playground
{

using HandlePtr = std::unique_ptr<void, std::function<void(void *)>>;

struct Buffer
{
  HandlePtr handle;
  size_t size;
  DeviceType device_type;

  Buffer() = delete;

  Buffer(Buffer const &)            = delete;
  Buffer &operator=(Buffer const &) = delete;

  Buffer(Buffer &&)            = default;
  Buffer &operator=(Buffer &&) = default;
};

namespace backend
{

template <typename... Rest>
inline void is_buffer()
{
  static_assert((std::is_same_v<Buffer, Rest> && ...), "Only Buffers are supported");
}

template <typename... Rest>
inline void assert_same_device(Buffer const &first, Rest const &...rest)
{
  is_buffer<Rest...>();
  const DeviceType ref = first.device_type;
  ((assert(rest.device_type == ref && "Buffers are on different devices")), ...);
}

template <typename... Rest>
inline void assert_same_size(Buffer const &first, Rest const &...rest)
{
  is_buffer<Rest...>();
  const std::size_t ref = first.size;
  ((assert(rest.size == ref && "Buffers have different sizes")), ...);
}

template <typename... Rest>
inline void assert_compatible(Buffer const &first, Rest const &...rest)
{
  is_buffer<Rest...>();
  assert_same_device(first, rest...);
  assert_same_size(first, rest...);
}

} // namespace backend

} // namespace gpu_playground
