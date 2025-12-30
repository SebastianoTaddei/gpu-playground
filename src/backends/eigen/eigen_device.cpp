#include "Eigen/Dense"

#include "eigen_device.hpp"

namespace gpu_playground::backend
{

using EigenBuffer = Eigen::VectorXf;

void EigenDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible(a, b, c);

  auto const &eigen_a = *static_cast<EigenBuffer const *>(a.get());
  auto const &eigen_b = *static_cast<EigenBuffer const *>(b.get());
  auto &eigen_c       = *static_cast<EigenBuffer *>(c.get());

  eigen_c = eigen_a + eigen_b;
}

Buffer EigenDevice::new_buffer(std::vector<float> data) const
{
  auto const size = data.size();
  return Buffer{
      HandlePtr{
          new EigenBuffer(
              Eigen::Map<EigenBuffer>(data.data(), static_cast<Eigen::Index>(data.size()))
          ),
          [](void *ptr) -> void { delete static_cast<EigenBuffer *>(ptr); }
      },
      size,
      EigenDevice::s_type
  };
}

void EigenDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible(from, to);

  auto const &eigen_from = *static_cast<EigenBuffer const *>(from.get());
  auto &eigen_to         = *static_cast<EigenBuffer *>(to.get());

  eigen_to = eigen_from;
}

std::vector<float> EigenDevice::cpu(Buffer const &buffer) const
{
  auto const &eigen_buffer = *static_cast<EigenBuffer const *>(buffer.get());
  return {eigen_buffer.data(), std::next(eigen_buffer.data(), eigen_buffer.size())};
}

std::unique_ptr<Device> make_eigen_device();

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_eigen_device()
{
  return std::make_shared<gpu_playground::backend::EigenDevice>();
}
