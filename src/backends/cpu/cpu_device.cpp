#include "device.hpp"

namespace backend
{

class CPUDevice final : public Device
{
public:
  CPUDevice() = default;

  ~CPUDevice() = default;

  Type type() const override { return Type::CPU; }
};

std::shared_ptr<Device> make_cpu_device() { return std::make_shared<CPUDevice>(); }

} // namespace backend
