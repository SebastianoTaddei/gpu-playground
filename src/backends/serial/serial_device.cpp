#include <cmath>

#include "serial_device.hpp"

namespace gpu_playground::backend
{

using SerialBuffer = std::vector<float>;

namespace
{

struct Add
{
  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a + b; }
};

struct Sub
{
  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a - b; }
};

struct Mul
{
  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a * b; }
};

struct Div
{
  [[nodiscard]] constexpr float operator()(float const a, float const b) const { return a / b; }
};

template <class Op>
void cwisem_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_same_shape(a, b, c);

  auto const &serial_a = *static_cast<SerialBuffer const *>(a.get());
  auto const &serial_b = *static_cast<SerialBuffer const *>(b.get());
  auto &serial_c       = *static_cast<SerialBuffer *>(c.get());

  for (size_t i{0}; i < a.size(); i++)
  {
    serial_c[i] = op(serial_a[i], serial_b[i]);
  }
}

template <class Op>
void cwises_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_compatible_sop(a, b, c);

  auto const &serial_a = *static_cast<SerialBuffer const *>(a.get());
  auto const &serial_b = *static_cast<SerialBuffer const *>(b.get());
  auto &serial_c       = *static_cast<SerialBuffer *>(c.get());

  auto const scalar_b = serial_b.front();
  for (size_t i{0}; i < a.size(); i++)
  {
    serial_c[i] = op(serial_a[i], scalar_b);
  }
}

} // namespace

void SerialDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Add{});
}

void SerialDevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Sub{});
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
  cwisem_op(a, b, c, Mul{});
}

void SerialDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Div{});
}

void SerialDevice::sadd(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Add{});
}

void SerialDevice::ssub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Sub{});
}

void SerialDevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Mul{});
}

void SerialDevice::sdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Div{});
}

Buffer SerialDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  return Buffer{
      HandlePtr{
          new SerialBuffer(std::move(data)),
          [](void *ptr) -> void
          { std::default_delete<SerialBuffer>{}(static_cast<SerialBuffer *>(ptr)); }
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

void SerialDevice::transpose(Buffer const &from, Buffer &to) const
{
  assert_compatible_transpose(from, to);

  auto const &serial_from = *static_cast<SerialBuffer const *>(from.get());
  auto &serial_to         = *static_cast<SerialBuffer *>(to.get());

  auto const [rows, cols] = from.shape();
  for (size_t i{0}; i < rows; i++)
  {
    for (size_t j{0}; j < cols; j++)
    {
      serial_to[(j * rows) + i] = serial_from[(i * cols) + j];
    }
  }
}

std::vector<float> SerialDevice::cpu(Buffer const &buffer) const
{
  return *static_cast<SerialBuffer const *>(buffer.get());
}

void SerialDevice::sync([[maybe_unused]] Buffer const &buffer) const {}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_serial_device()
{
  return std::make_shared<gpu_playground::backend::SerialDevice>();
}
