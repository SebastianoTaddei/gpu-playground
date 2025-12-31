#pragma once

#include <memory>
#include <vector>

#include "buffer.hpp"
#include "device_types.hpp"

namespace gpu_playground
{

class Device
{
public:
  virtual ~Device() = default;

  [[nodiscard]] virtual DeviceType type() const = 0;
  virtual void
  add(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;
  [[nodiscard]] virtual backend::Buffer new_buffer(std::vector<float> data) const   = 0;

  [[nodiscard]] backend::Buffer new_buffer_with_size(size_t size) const
  {
    return this->new_buffer(std::vector<float>(size, 0.0));
  }

  virtual void copy_buffer(backend::Buffer const &from, backend::Buffer &to) const  = 0;
  [[nodiscard]] virtual std::vector<float> cpu(backend::Buffer const &buffer) const = 0;
};

using DevicePtr = std::shared_ptr<Device>;

DevicePtr make_serial_device();

#ifdef GPU_PLAYGROUND_HAS_EIGEN
DevicePtr make_eigen_device();
#endif

#ifdef GPU_PLAYGROUND_HAS_SIMD
DevicePtr make_simd_device();
#endif

#ifdef GPU_PLAYGROUND_HAS_METAL
DevicePtr make_metal_device();
#endif

} // namespace gpu_playground
