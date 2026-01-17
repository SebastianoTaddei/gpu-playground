#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"

#include "device.hpp"

#include "algorithms.hpp"
#include "matchers.hpp"
#include "tensor.hpp"

using namespace Catch::Matchers;
using namespace gpu_playground;

TEST_CASE("algorithms: gradient descent", "[algorithms]")
{
  auto const devices = make_devices();

  // clang-format off
  std::vector<float> const a_data{6.0, -1.0, 0.0, 0.0, 0.0, -1.0, 6.0, -1.0, 0.0, 0.0, 0.0, -1.0, 6.0, -1.0, 0.0, 0.0, 0.0, -1.0, 6.0, -1.0, 0.0, 0.0, 0.0, -1.0, 6.0};
  std::vector<float> const b_data{1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<float> const ref{0.24978355, 0.4987013, 0.74242425, 0.95584416, 0.9926407};
  // clang-format on
  Shape const a_shape{5, 5};
  Shape const b_shape{5, 1};
  Tensor a(a_data, a_shape, devices[DeviceIdx::SERIAL]);
  Tensor b(b_data, b_shape, devices[DeviceIdx::SERIAL]);
  Tensor x0 = Tensor::zeros(b_shape, devices[DeviceIdx::SERIAL]);

  for (auto const &device : devices)
  {
    if (device != nullptr)
    {
      SECTION(std::string(get_device_name(device->type())))
      {
        a.to(device);
        b.to(device);
        x0.to(device);

        auto const c = gradient_descent(a, b, x0);

        REQUIRE_THAT(c.cpu(), VectorsWithinAbsRel(ref));
      }
    }
  }
}
