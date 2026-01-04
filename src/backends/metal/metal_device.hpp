#pragma once

#include "device.hpp"

namespace gpu_playground::backend
{

class MetalDevice final : public Device
{
private:
  static constexpr DeviceType s_type{DeviceType::METAL};
  struct Impl;
  std::unique_ptr<Impl> pimpl;

public:
  MetalDevice();

  MetalDevice(MetalDevice const &)            = delete;
  MetalDevice(MetalDevice &&)                 = delete;
  MetalDevice &operator=(MetalDevice const &) = delete;
  MetalDevice &operator=(MetalDevice &&)      = delete;
  ~MetalDevice() override;

  [[nodiscard]] DeviceType type() const override { return MetalDevice::s_type; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void mul(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void cmul(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void cdiv(Buffer const &a, Buffer const &b, Buffer &c) const override;

  [[nodiscard]] Buffer new_buffer(std::vector<float> data, Shape shape) const override;

  void copy_buffer(Buffer const &from, Buffer &to) const override;

  [[nodiscard]] std::vector<float> cpu(Buffer const &buffer) const override;
};

} // namespace gpu_playground::backend
