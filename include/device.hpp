#pragma once

#include <memory>
#include <vector>

#include "buffer.hpp"

namespace gpu_playground
{

class Device
{
public:
  virtual ~Device() = default;

  [[nodiscard]] virtual DeviceType type() const = 0;

  virtual void
  add(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  sub(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  mul(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  cmul(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  cdiv(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  sadd(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  ssub(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  smul(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  virtual void
  sdiv(backend::Buffer const &a, backend::Buffer const &b, backend::Buffer &c) const = 0;

  [[nodiscard]] virtual backend::Buffer new_buffer(std::vector<float> data, Shape shape) const = 0;

  [[nodiscard]] backend::Buffer new_buffer_with_shape(Shape shape) const
  {
    return this->new_buffer(std::vector<float>(shape.rows * shape.cols, 0.0), shape);
  }

  virtual void copy_buffer(backend::Buffer const &from, backend::Buffer &to) const = 0;

  virtual void transpose(backend::Buffer const &from, backend::Buffer &to) const = 0;

  [[nodiscard]] virtual std::vector<float> cpu(backend::Buffer const &buffer) const = 0;

  virtual void sync(backend::Buffer const &buffer) const = 0;
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

inline std::array<DevicePtr, DeviceIdx::COUNT> make_devices()
{

  std::array<DevicePtr, DeviceIdx::COUNT> devices{};

  devices[DeviceIdx::SERIAL] = make_serial_device();
#ifdef GPU_PLAYGROUND_HAS_EIGEN
  devices[DeviceIdx::EIGEN] = make_eigen_device();
#else
  devices[DeviceIdx::EIGEN] = nullptr;
#endif
#ifdef GPU_PLAYGROUND_HAS_SIMD
  devices[DeviceIdx::SIMD] = make_simd_device();
#else
  devices[DeviceIdx::SIMD] = nullptr;
#endif
#ifdef GPU_PLAYGROUND_HAS_METAL
  devices[DeviceIdx::METAL] = make_metal_device();
#else
  devices[DeviceIdx::METAL] = nullptr;
#endif

  return devices;
}

} // namespace gpu_playground
