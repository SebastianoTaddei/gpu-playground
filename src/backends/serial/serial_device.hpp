#pragma once

#include "device.hpp"

namespace gpu_playground::backend
{

class SerialDevice final : public Device
{
private:
  static constexpr DeviceType s_type{DeviceType::SERIAL};

public:
  SerialDevice() = default;

  SerialDevice(SerialDevice const &)            = default;
  SerialDevice(SerialDevice &&)                 = delete;
  SerialDevice &operator=(SerialDevice const &) = default;
  SerialDevice &operator=(SerialDevice &&)      = delete;
  ~SerialDevice() override                      = default;

  [[nodiscard]] DeviceType type() const override { return SerialDevice::s_type; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void mul(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void cmul(Buffer const &a, Buffer const &b, Buffer &c) const override;

  void cdiv(Buffer const &a, Buffer const &b, Buffer &c) const override;

  [[nodiscard]] Buffer new_buffer(std::vector<float> data, Shape shape) const override;

  void copy_buffer(Buffer const &from, Buffer &to) const override;

  [[nodiscard]] std::vector<float> cpu(Buffer const &buffer) const override;
};

} // namespace gpu_playground::backend
