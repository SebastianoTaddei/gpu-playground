#include "cpu_device.hpp"

namespace gpu_playground::backend
{

using CPUBuffer = std::vector<float>;

void CPUDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible(a, b, c);

  auto const &cpu_a = *static_cast<CPUBuffer const *>(a.get());
  auto const &cpu_b = *static_cast<CPUBuffer const *>(b.get());
  auto &cpu_c       = *static_cast<CPUBuffer *>(c.get());

  for (size_t i{0}; i < a.size(); i++)
  {
    cpu_c[i] = cpu_a[i] + cpu_b[i];
  }
}

Buffer CPUDevice::new_buffer(std::vector<float> data) const
{
  auto const size = data.size();
  return Buffer{
      HandlePtr{
          new CPUBuffer(std::move(data)),
          [](void *ptr) -> void { delete static_cast<CPUBuffer *>(ptr); }
      },
      size,
      CPUDevice::s_type
  };
}

void CPUDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible(from, to);

  auto const &cpu_from = *static_cast<CPUBuffer const *>(from.get());
  auto &cpu_to         = *static_cast<CPUBuffer *>(to.get());

  cpu_to = cpu_from;
}

std::vector<float> CPUDevice::cpu(Buffer const &buffer) const
{
  return *static_cast<CPUBuffer const *>(buffer.get());
}

std::unique_ptr<Device> make_cpu_device();

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_cpu_device()
{
  return std::make_shared<gpu_playground::backend::CPUDevice>();
}
