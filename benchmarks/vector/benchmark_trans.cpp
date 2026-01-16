#include <numeric>
#include <string>
#include <vector>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;

TEST_CASE("vector: trans", "[vector]")
{
  auto const devices = make_devices();

  constexpr size_t len{1'000'000};
  std::vector<float> data(len);
  std::iota(data.begin(), data.end(), 0.0);
  Shape const shape{len, 1};
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
