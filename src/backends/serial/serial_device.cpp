#include <cmath>

#include "serial_device.hpp"

namespace gpu_playground::backend
{

using SerialBuffer = std::vector<float>;

void SerialDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const &serial_a = *static_cast<SerialBuffer const *>(a.get());
  auto const &serial_b = *static_cast<SerialBuffer const *>(b.get());
  auto &serial_c       = *static_cast<SerialBuffer *>(c.get());

  for (size_t i{0}; i < a.size(); i++)
  {
    serial_c[i] = serial_a[i] + serial_b[i];
  }
}

void SerialDevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible_mul(a, b, c);

  auto const &serial_a = *static_cast<SerialBuffer const *>(a.get());
  auto const &serial_b = *static_cast<SerialBuffer const *>(b.get());
  auto &serial_c       = *static_cast<SerialBuffer *>(c.get());

  auto const [m, k] = a.shape();
  auto const n      = b.shape().cols;

  for (size_t i{0}; i < m; i++)
  {
    for (size_t p{0}; p < k; p++)
    {
      auto const a_ip = serial_a[(i * k) + p];

      for (size_t j{0}; j < n; j++)
      {
        serial_c[(i * n) + j] = std::fma(a_ip, serial_b[(p * n) + j], serial_c[(i * n) + j]);
      }
    }
  }
}

void SerialDevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const &serial_a = *static_cast<SerialBuffer const *>(a.get());
  auto const &serial_b = *static_cast<SerialBuffer const *>(b.get());
  auto &serial_c       = *static_cast<SerialBuffer *>(c.get());

  for (size_t i{0}; i < a.size(); i++)
  {
    serial_c[i] = serial_a[i] * serial_b[i];
  }
}

void SerialDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const &serial_a = *static_cast<SerialBuffer const *>(a.get());
  auto const &serial_b = *static_cast<SerialBuffer const *>(b.get());
  auto &serial_c       = *static_cast<SerialBuffer *>(c.get());

  for (size_t i{0}; i < a.size(); i++)
  {
    serial_c[i] = serial_a[i] / serial_b[i];
  }
}

Buffer SerialDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  return Buffer{
      HandlePtr{
          new SerialBuffer(std::move(data)),
          [](void *ptr) -> void { delete static_cast<SerialBuffer *>(ptr); }
      },
      shape,
      SerialDevice::s_type
  };
}

void SerialDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible_copy(from, to);

  auto const &serial_from = *static_cast<SerialBuffer const *>(from.get());
  auto &serial_to         = *static_cast<SerialBuffer *>(to.get());

  serial_to = serial_from;
}

std::vector<float> SerialDevice::cpu(Buffer const &buffer) const
{
  return *static_cast<SerialBuffer const *>(buffer.get());
}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_serial_device()
{
  return std::make_shared<gpu_playground::backend::SerialDevice>();
}
