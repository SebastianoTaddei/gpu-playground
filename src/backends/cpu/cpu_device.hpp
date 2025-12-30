#pragma once

#include "device.hpp"

namespace gpu_playground::backend
{

class CPUDevice final : public Device
{
private:
  static constexpr DeviceType s_type{DeviceType::CPU};

public:
  CPUDevice() = default;

  CPUDevice(CPUDevice const &)            = default;
  CPUDevice(CPUDevice &&)                 = delete;
  CPUDevice &operator=(CPUDevice const &) = default;
  CPUDevice &operator=(CPUDevice &&)      = delete;
  ~CPUDevice() override                   = default;

  [[nodiscard]] DeviceType type() const override { return CPUDevice::s_type; }

  void add(Buffer const &a, Buffer const &b, Buffer &c) const override;

  [[nodiscard]] Buffer new_buffer(std::vector<float> data) const override;

  void copy_buffer(Buffer const &from, Buffer &to) const override;

  [[nodiscard]] std::vector<float> cpu(Buffer const &buffer) const override;
};

} // namespace gpu_playground::backend
