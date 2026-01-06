#include <numeric>
#include <string>
#include <vector>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"

#include "device.hpp"
#include "device_types.hpp"
#include "tensor.hpp"

using namespace gpu_playground;
using vec = std::vector<float>;

namespace
{

void benhc_mat_add(DevicePtr const &device, Tensor &a, Tensor &b)
{
  a.to(device);
  b.to(device);

  BENCHMARK(std::string(get_device_name(device->type()))) { return a + b; };
}

} // namespace

TEST_CASE("matrix: add", "[matrix]")
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  constexpr size_t rows{1'000};
  constexpr size_t cols{1'000};
  std::vector<float> a_data(rows * cols);
  std::vector<float> b_data(rows * cols);
  std::iota(a_data.begin(), a_data.end(), 0.0);
  std::iota(b_data.begin(), b_data.end(), 1.0);
  Shape const shape{rows, cols};
  Tensor a(a_data, shape, serial_device);
  Tensor b(b_data, shape, serial_device);

  benhc_mat_add(serial_device, a, b);
  benhc_mat_add(eigen_device, a, b);
  benhc_mat_add(simd_device, a, b);
  benhc_mat_add(metal_device, a, b);
}
