#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace gpu_playground
{

#define DEVICE_TYPES                                                                               \
  X(SERIAL)                                                                                        \
  X(EIGEN)                                                                                         \
  X(SIMD)                                                                                          \
  X(METAL)

enum class DeviceType : uint8_t
{
#define X(type) type,
  DEVICE_TYPES
#undef X
      COUNT
};

enum DeviceIdx : uint8_t
{

#define X(type) type,
  DEVICE_TYPES
#undef X
      COUNT
};

inline constexpr std::array<std::string_view, DeviceIdx::COUNT> device_names{
#define X(name) #name,
    DEVICE_TYPES
#undef X
};

constexpr std::string_view get_device_name(DeviceType const type)
{
  return device_names.at(static_cast<size_t>(type));
}

} // namespace gpu_playground
