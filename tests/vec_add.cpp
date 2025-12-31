#include <iostream>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;
using vec = std::vector<float>;

namespace
{

void test_vec_add(DevicePtr const &device, vec const &vec_a, vec const &vec_b)
{
  assert(vec_a.size() == vec_b.size());

  auto const tensor_a = gpu_playground::Tensor(vec_a, device);
  auto const tensor_b = gpu_playground::Tensor(vec_b, device);

  auto const tensor_c = tensor_a + tensor_b;
  auto const vec_c    = tensor_c.cpu();

  std::cout << get_device_name(device->type()) << " add: ";
  for (size_t i{0}; i < vec_a.size(); i++)
  {
    std::cout << vec_c.at(i) << " ";
  }
  std::cout << '\n';
}

} // namespace

int main()
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  std::vector<float> const vec_a{0.0, 1.0, 2.0};
  std::vector<float> const vec_b{3.0, 4.0, 5.0};

  test_vec_add(serial_device, vec_a, vec_b);
  test_vec_add(eigen_device, vec_a, vec_b);
  test_vec_add(simd_device, vec_a, vec_b);
  test_vec_add(metal_device, vec_a, vec_b);

  return 0;
}
