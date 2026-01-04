#include <iostream>
#include <numeric>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"
#include "utils.hpp"

using namespace gpu_playground;

static constexpr size_t RUNS{1000};
static constexpr size_t ROWS{1024};
static constexpr size_t COLS{1024};

namespace
{

void benchmark_mat_cmul(DevicePtr const &device, Tensor &a, Tensor &b)

{
  a.to(device);
  b.to(device);

  auto c = a.cmul(b);

  std::cout << get_device_name(device->type()) << ": ";
  {
    benchmark::TimeIt timer{RUNS};

    for (size_t i{0}; i < RUNS; i++)
    {
      c = a.cmul(b);
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

  std::vector<float> a_data(ROWS * COLS);
  std::vector<float> b_data(ROWS * COLS);
  std::iota(a_data.begin(), a_data.end(), 0.0);
  std::iota(b_data.begin(), b_data.end(), 1.0);
  Shape const a_shape{.rows = ROWS, .cols = COLS};
  Shape const b_shape{.rows = ROWS, .cols = COLS};
  Tensor a(a_data, a_shape, serial_device);
  Tensor b(b_data, b_shape, serial_device);

  benchmark_mat_cmul(serial_device, a, b);
  benchmark_mat_cmul(eigen_device, a, b);
  benchmark_mat_cmul(simd_device, a, b);
  benchmark_mat_cmul(metal_device, a, b);

  return 0;
}
