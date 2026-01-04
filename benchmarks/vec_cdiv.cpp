#include <iostream>
#include <numeric>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"
#include "utils.hpp"

using namespace gpu_playground;

static constexpr size_t RUNS{1000};
static constexpr size_t ROWS{10000};

namespace
{

void benchmark_vec_cdiv(DevicePtr const &device, Tensor &a, Tensor &b)

{
  a.to(device);
  b.to(device);

  auto c = a.cdiv(b);

  std::cout << get_device_name(device->type()) << ": ";
  {
    benchmark::TimeIt timer{RUNS};

    for (size_t i{0}; i < RUNS; i++)
    {
      c = a.cdiv(b);
    }
  }
}

} // namespace

int main()
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  std::vector<float> a_data(ROWS);
  std::vector<float> b_data(ROWS);
  std::iota(a_data.begin(), a_data.end(), 0.0);
  std::iota(b_data.begin(), b_data.end(), 1.0);
  Shape const shape{.rows = ROWS, .cols = 1};
  Tensor a(a_data, shape, serial_device);
  Tensor b(b_data, shape, serial_device);

  benchmark_vec_cdiv(serial_device, a, b);
  benchmark_vec_cdiv(eigen_device, a, b);
  benchmark_vec_cdiv(simd_device, a, b);
  benchmark_vec_cdiv(metal_device, a, b);

  return 0;
}
