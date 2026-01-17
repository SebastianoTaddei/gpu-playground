#include <string>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"

#include "algorithms.hpp"
#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;

TEST_CASE("algorithms: gradient descent", "[algorithms]")
{
  auto const devices = make_devices();

  constexpr size_t rows{100};
  constexpr size_t cols{100};
  Shape const a_shape{rows, cols};
  Shape const b_shape{cols, 1};
  Tensor a  = Tensor::rand(a_shape, devices[DeviceIdx::SERIAL]);
  Tensor b  = Tensor::rand(b_shape, devices[DeviceIdx::SERIAL]);
  Tensor x0 = Tensor::zeros(b_shape, devices[DeviceIdx::SERIAL]);

  a = a.transpose() * a;

  for (auto const &device : devices)
  {
    if (device != nullptr)
    {
      a.to(device);
      b.to(device);
      x0.to(device);

      BENCHMARK(std::string(get_device_name(device->type())))
      {
        return gradient_descent(a, b, x0);
      };
    }
  }
}
