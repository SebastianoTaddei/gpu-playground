#pragma once

#include <iostream>

#include "device.hpp"

namespace gpu_playground
{

class Tensor
{
private:
  DevicePtr device;
  backend::Buffer buffer;

public:
  Tensor()  = delete;
  ~Tensor() = default;

  Tensor(Tensor &&)            = default;
  Tensor &operator=(Tensor &&) = default;

  Tensor(std::vector<float> data, Shape shape, DevicePtr device)
      : device(std::move(device)), buffer(this->device->new_buffer(std::move(data), shape))
  {
  }

  Tensor(Shape shape, DevicePtr device)
      : device(std::move(device)), buffer(this->device->new_buffer_with_shape(shape))
  {
  }

  Tensor(Tensor const &other)
      : device(other.device), buffer(this->device->new_buffer_with_shape(other.buffer.shape()))
  {
    this->device->copy_buffer(other.buffer, this->buffer);
  }

  void to(DevicePtr device)
  {
    if (this->device == device)
    {
      return;
    }

    auto const data  = this->cpu();
    auto const shape = this->buffer.shape();
    *this            = Tensor(data, shape, std::move(device));
  }

  Tensor &operator=(Tensor const &other)
  {
    if (this == &other)
    {
      return *this;
    }

    this->device = other.device;
    this->device->copy_buffer(other.buffer, this->buffer);

    return *this;
  }

  Tensor &operator+=(Tensor const &rhs)
  {
    this->device->add(this->buffer, rhs.buffer, this->buffer);

    return *this;
  }

  Tensor &operator-=(Tensor const &rhs)
  {
    this->device->sub(this->buffer, rhs.buffer, this->buffer);

    return *this;
  }

  friend Tensor operator+(Tensor lhs, Tensor const &rhs);

  friend Tensor operator-(Tensor lhs, Tensor const &rhs);

  Tensor operator*(Tensor const &other) const
  {
    Tensor out{Shape{this->buffer.shape().rows, other.buffer.shape().cols}, this->device};
    this->device->mul(this->buffer, other.buffer, out.buffer);
    return out;
  }

  [[nodiscard]] Tensor cmul(Tensor const &other) const
  {
    Tensor out{this->buffer.shape(), this->device};
    this->device->cmul(this->buffer, other.buffer, out.buffer);
    return out;
  }

  [[nodiscard]] Tensor cdiv(Tensor const &other) const
  {
    Tensor out{this->buffer.shape(), this->device};
    this->device->cdiv(this->buffer, other.buffer, out.buffer);
    return out;
  }

  friend std::ostream &operator<<(std::ostream &os, Tensor const &t);

  [[nodiscard]] std::vector<float> cpu() const { return this->device->cpu(this->buffer); }

  [[nodiscard]] Shape shape() const { return this->buffer.shape(); }
};

inline Tensor operator+(Tensor lhs, Tensor const &rhs)
{
  lhs += rhs;
  return lhs;
}

inline Tensor operator-(Tensor lhs, Tensor const &rhs)
{
  lhs -= rhs;
  return lhs;
}

inline std::ostream &operator<<(std::ostream &os, Tensor const &t)
{
  auto const data         = t.cpu();
  auto const [rows, cols] = t.shape();

  os << "Tensor(" << rows << "x" << cols << ")\n";

  for (size_t i{0}; i < rows; i++)
  {
    os << "[ ";
    for (size_t j{0}; j < cols; j++)
    {
      os << data[(i * cols) + j] << ' ';
    }
    os << "]\n";
  }

  return os;
}

} // namespace gpu_playground
