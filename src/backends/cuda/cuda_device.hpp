#pragma once

#include "device.hpp"

namespace gpu_playground::backend
{

class CUDADevice final : public Device
{
private:
  static constexpr DeviceType s_type{DeviceType::CUDA};
  struct Impl;
  std::unique_ptr<Impl> pimpl;

public:
  CUDADevice();

  CUDADevice(CUDADevice const &)            = delete;
  CUDADevice(CUDADevice &&)                 = delete;
  CUDADevice &operator=(CUDADevice const &) = delete;
  CUDADevice &operator=(CUDADevice &&)      = delete;
  ~CUDADevice() override;

  [[nodiscard]] DeviceType type() const override { return CUDADevice::s_type; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void sub(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void mul(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void cmul(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void cdiv(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void sadd(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void ssub(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void smul(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void sdiv(Buffer const &a, Buffer const &b, Buffer &c) const override;

  [[nodiscard]] Buffer new_buffer(std::vector<float> data, Shape shape) const override;

  void copy_buffer(Buffer const &from, Buffer &to) const override;

  void transpose(Buffer const &from, Buffer &to) const override;

  [[nodiscard]] std::vector<float> cpu(Buffer const &buffer) const override;

  void sync(Buffer const &buffer) const override;
};

} // namespace gpu_playground::backend
