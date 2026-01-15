#include <numeric>
#include <string>
#include <vector>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;

TEST_CASE("matrix: trans", "[matrix]")
{
  auto const devices = make_devices();

  constexpr size_t rows{1'000};
  constexpr size_t cols{1'000};
  std::vector<float> data(rows * cols);
  std::iota(data.begin(), data.end(), 0.0);
  Shape const shape{rows, cols};
  Tensor a(data, shape, devices[DeviceIdx::SERIAL]);

  for (auto const &device : devices)
  {
    if (device != nullptr)
    {
      a.to(device);

      BENCHMARK(std::string(get_device_name(device->type()))) { return a.transpose(); };
    }
  }
}
