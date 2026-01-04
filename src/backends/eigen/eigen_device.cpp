#include "Eigen/Dense"

#include "eigen_device.hpp"

namespace gpu_playground::backend
{

using EigenBuffer = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void EigenDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const &eigen_a = *static_cast<EigenBuffer const *>(a.get());
  auto const &eigen_b = *static_cast<EigenBuffer const *>(b.get());
  auto &eigen_c       = *static_cast<EigenBuffer *>(c.get());

  eigen_c = eigen_a + eigen_b;
}

void EigenDevice::mul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_compatible_mul(a, b, c);

  auto const &eigen_a = *static_cast<EigenBuffer const *>(a.get());
  auto const &eigen_b = *static_cast<EigenBuffer const *>(b.get());
  auto &eigen_c       = *static_cast<EigenBuffer *>(c.get());

  eigen_c = eigen_a * eigen_b;
}

void EigenDevice::cmul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const &eigen_a = *static_cast<EigenBuffer const *>(a.get());
  auto const &eigen_b = *static_cast<EigenBuffer const *>(b.get());
  auto &eigen_c       = *static_cast<EigenBuffer *>(c.get());

  eigen_c = eigen_a.cwiseProduct(eigen_b);
}

void EigenDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  assert_same_shape(a, b, c);

  auto const &eigen_a = *static_cast<EigenBuffer const *>(a.get());
  auto const &eigen_b = *static_cast<EigenBuffer const *>(b.get());
  auto &eigen_c       = *static_cast<EigenBuffer *>(c.get());

  eigen_c = eigen_a.cwiseQuotient(eigen_b);
}

Buffer EigenDevice::new_buffer(std::vector<float> data, Shape shape) const
{
  return Buffer{
      HandlePtr{
          new EigenBuffer(
              Eigen::Map<EigenBuffer>(
                  data.data(),
                  static_cast<Eigen::Index>(shape.rows),
                  static_cast<Eigen::Index>(shape.cols)
              )
          ),
          [](void *ptr) -> void { delete static_cast<EigenBuffer *>(ptr); }
      },
      shape,
      EigenDevice::s_type
  };
}

void EigenDevice::copy_buffer(Buffer const &from, Buffer &to) const
{
  assert_compatible_copy(from, to);

  auto const &eigen_from = *static_cast<EigenBuffer const *>(from.get());
  auto &eigen_to         = *static_cast<EigenBuffer *>(to.get());

  eigen_to = eigen_from;
}

std::vector<float> EigenDevice::cpu(Buffer const &buffer) const
{
  auto const &eigen_buffer = *static_cast<EigenBuffer const *>(buffer.get());
  return {eigen_buffer.data(), std::next(eigen_buffer.data(), eigen_buffer.size())};
}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_eigen_device()
{
  return std::make_shared<gpu_playground::backend::EigenDevice>();
}
