#include <numeric>
#include <string>
#include <vector>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;

TEST_CASE("matrix: cmul", "[matrix]")
{
  auto const devices = make_devices();

  constexpr size_t rows{1'000};
  constexpr size_t cols{1'000};
  std::vector<float> a_data(rows * cols);
  std::vector<float> b_data(rows * cols);
  std::iota(a_data.begin(), a_data.end(), 0.0);
  std::iota(b_data.begin(), b_data.end(), 1.0);
  Shape const shape{rows, cols};
  Tensor a(a_data, shape, devices[DeviceIdx::SERIAL]);
  Tensor b(b_data, shape, devices[DeviceIdx::SERIAL]);

  for (auto const &device : devices)
  {
    if (device != nullptr)
    {
      a.to(device);
      b.to(device);

      BENCHMARK(std::string(get_device_name(device->type()))) { return a.cmul(b); };
    }
  }
}
