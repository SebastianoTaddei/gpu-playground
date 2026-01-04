#include <iostream>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;

namespace
{

void test_vec_cdiv(DevicePtr const &device, Tensor &a, Tensor &b)
{
  a.to(device);
  b.to(device);

  auto const c = a.cdiv(b);

  std::cout << get_device_name(device->type()) << " cdiv:\n" << c << '\n';
}

} // namespace

int main()
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  std::vector<float> const a_data{0.0, 1.0, 2.0};
  std::vector<float> const b_data{3.0, 4.0, 5.0};
  Shape const shape{.rows = 3, .cols = 1};
  Tensor a(a_data, shape, serial_device);
  Tensor b(b_data, shape, serial_device);

  test_vec_cdiv(serial_device, a, b);
  test_vec_cdiv(eigen_device, a, b);
  test_vec_cdiv(simd_device, a, b);
  test_vec_cdiv(metal_device, a, b);

  return 0;
}
