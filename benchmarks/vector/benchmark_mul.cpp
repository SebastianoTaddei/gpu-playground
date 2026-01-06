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

void bench_mat_mul(DevicePtr const &device, Tensor &a, Tensor &b)
{
  a.to(device);
  b.to(device);

  BENCHMARK(std::string(get_device_name(device->type()))) { return a * b; };
}

} // namespace

TEST_CASE("vector: mul", "[vector]")
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  constexpr size_t len{1'000};
  std::vector<float> a_data(len);
  std::vector<float> b_data(len);
  std::iota(a_data.begin(), a_data.end(), 0.0);
  std::iota(b_data.begin(), b_data.end(), 1.0);
  Shape const a_shape{len, 1};
  Shape const b_shape{1, len};
  Tensor a(a_data, a_shape, serial_device);
  Tensor b(b_data, b_shape, serial_device);

  bench_mat_mul(serial_device, a, b);
  bench_mat_mul(eigen_device, a, b);
  bench_mat_mul(simd_device, a, b);
  bench_mat_mul(metal_device, a, b);
}
