#include "Eigen/Dense"
#include "buffer.hpp"

#include "eigen_device.hpp"

namespace gpu_playground::backend
{

using EigenBuffer = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace
{

struct Add
{
  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, EigenBuffer const &b) const
  {
    return a + b;
  }

  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, float const b) const
  {
    return a.array() + b;
  }
};

struct Sub
{
  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, EigenBuffer const &b) const
  {
    return a - b;
  }

  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, float const b) const
  {
    return a.array() - b;
  }
};

struct Mul
{
  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, EigenBuffer const &b) const
  {
    return a.cwiseProduct(b);
  }

  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, float const b) const { return a * b; }
};

struct Div
{
  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, EigenBuffer const &b) const
  {
    return a.cwiseQuotient(b);
  }

  [[nodiscard]] EigenBuffer operator()(EigenBuffer const &a, float const b) const { return a / b; }
};

template <class Op>
void cwisem_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_same_shape(a, b, c);

  auto const &eigen_a = *static_cast<EigenBuffer const *>(a.get());
  auto const &eigen_b = *static_cast<EigenBuffer const *>(b.get());
  auto &eigen_c       = *static_cast<EigenBuffer *>(c.get());

  eigen_c = op(eigen_a, eigen_b);
}

template <class Op>
void cwises_op(Buffer const &a, Buffer const &b, Buffer &c, Op const &op)
{
  assert_compatible_sop(a, b, c);

  auto const &eigen_a = *static_cast<EigenBuffer const *>(a.get());
  auto const &eigen_b = *static_cast<EigenBuffer const *>(b.get());
  auto &eigen_c       = *static_cast<EigenBuffer *>(c.get());

  auto const scalar_b = eigen_b(0);
  eigen_c             = op(eigen_a, scalar_b);
}

} // namespace

void EigenDevice::add(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Add{});
}

void EigenDevice::sub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Sub{});
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
  cwisem_op(a, b, c, Mul{});
}

void EigenDevice::cdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwisem_op(a, b, c, Div{});
}

void EigenDevice::sadd(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Add{});
}

void EigenDevice::ssub(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Sub{});
}

void EigenDevice::smul(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Mul{});
}

void EigenDevice::sdiv(Buffer const &a, Buffer const &b, Buffer &c) const
{
  cwises_op(a, b, c, Div{});
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

void EigenDevice::transpose(Buffer const &from, Buffer &to) const
{
  assert_compatible_transpose(from, to);

  auto const &eigen_from = *static_cast<EigenBuffer const *>(from.get());
  auto &eigen_to         = *static_cast<EigenBuffer *>(to.get());

  eigen_to = eigen_from.transpose();
}

std::vector<float> EigenDevice::cpu(Buffer const &buffer) const
{
  auto const &eigen_buffer = *static_cast<EigenBuffer const *>(buffer.get());
  return {eigen_buffer.data(), std::next(eigen_buffer.data(), eigen_buffer.size())};
}

void EigenDevice::sync([[maybe_unused]] Buffer const &buffer) const {}

} // namespace gpu_playground::backend

gpu_playground::DevicePtr gpu_playground::make_eigen_device()
{
  return std::make_shared<gpu_playground::backend::EigenDevice>();
}
