#include <iostream>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;
using vec = std::vector<float>;

namespace
{

void test_vec_mul(DevicePtr const &device, vec const &vec_a, vec const &vec_b)
{
  assert(vec_a.size() == vec_b.size());

  auto const rows = vec_a.size();
  auto const cols = vec_b.size();

  auto const tensor_a = gpu_playground::Tensor(vec_a, Shape{.rows = rows, .cols = 1}, device);
  auto const tensor_b = gpu_playground::Tensor(vec_b, Shape{.rows = 1, .cols = cols}, device);

  auto const tensor_c = tensor_a * tensor_b;
  auto const vec_c    = tensor_c.cpu();

  std::cout << get_device_name(device->type()) << " mul:\n";
  for (size_t i{0}; i < rows; i++)
  {
    for (size_t j{0}; j < cols; j++)
    {
      std::cout << vec_c.at((i * cols) + j) << " ";
    }
    std::cout << '\n';
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

  test_vec_mul(serial_device, vec_a, vec_b);
  test_vec_mul(eigen_device, vec_a, vec_b);
  test_vec_mul(simd_device, vec_a, vec_b);
  test_vec_mul(metal_device, vec_a, vec_b);

  return 0;
}
