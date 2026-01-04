#include <iostream>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;
using vec = std::vector<float>;

namespace
{

void test_mat_mul(DevicePtr const &device, Tensor &a, Tensor &b)
{
  a.to(device);
  b.to(device);

  auto const c = a * b;

  std::cout << get_device_name(device->type()) << " mul:\n" << c << '\n';
}

} // namespace

int main()
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  std::vector<float> const a_data{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<float> const b_data{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  Shape const a_shape{.rows = 2, .cols = 3};
  Shape const b_shape{.rows = 3, .cols = 2};
  Tensor a(a_data, a_shape, serial_device);
  Tensor b(b_data, b_shape, serial_device);

  test_mat_mul(serial_device, a, b);
  test_mat_mul(eigen_device, a, b);
  test_mat_mul(simd_device, a, b);
  test_mat_mul(metal_device, a, b);

  return 0;
}
