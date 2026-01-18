#include <numeric>
#include <string>
#include <vector>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;

TEST_CASE("vector: ssub", "[vector]")
{
  auto const devices = make_devices();

  constexpr size_t len{1'000'000};
  std::vector<float> a_data(len);
  std::vector<float> const b_data{2.0};
  std::iota(a_data.begin(), a_data.end(), 0.0);
  Shape const a_shape{len, 1};
  Shape const b_shape{1, 1};
  Tensor a(a_data, a_shape, devices[DeviceIdx::SERIAL]);
  Tensor b(b_data, b_shape, devices[DeviceIdx::SERIAL]);

  for (auto const &device : devices)
  {
    if (device != nullptr)
    {
      a.to(device);
      b.to(device);

      BENCHMARK(std::string(get_device_name(device->type()))) { return a.ssub(b); };
    }
  }
}
