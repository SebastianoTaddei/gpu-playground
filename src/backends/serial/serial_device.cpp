#include "serial_device.hpp"

namespace gpu_playground::backend
{

using SerialBuffer = std::vector<float>;

void SerialDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible_add(a, b, c);

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
        support += serial_a[(i * a_cols) + k] * serial_b[(k * b_cols) + j];
      }
      serial_c[idx] = support;
    }
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
