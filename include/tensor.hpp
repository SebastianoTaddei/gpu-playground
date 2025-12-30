#pragma once

#include <utility>

#include "device.hpp"

namespace gpu_playground
{

class Tensor
{
private:
  std::shared_ptr<Device> device;
  backend::Buffer buffer;

public:
  Tensor()  = delete;
  ~Tensor() = default;

  Tensor(Tensor &&)            = default;
  Tensor &operator=(Tensor &&) = default;

  Tensor(std::vector<float> data, DevicePtr device)
      : device(std::move(device)), buffer(this->device->new_buffer(std::move(data)))
  {
  }

  Tensor(size_t size, DevicePtr device)
      : device(std::move(device)), buffer(this->device->new_buffer_with_size(size))
  {
  }

  Tensor(Tensor const &other)
      : device(other.device), buffer(this->device->new_buffer_with_size(other.buffer.size()))
  {
    this->device->copy_buffer(other.buffer, this->buffer);
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

  friend Tensor operator+(Tensor lhs, Tensor const &rhs);

  [[nodiscard]] std::vector<float> cpu() const { return this->device->cpu(this->buffer); }
};

inline Tensor operator+(Tensor lhs, Tensor const &rhs)
{
  lhs += rhs;
  return lhs;
}

} // namespace gpu_playground
