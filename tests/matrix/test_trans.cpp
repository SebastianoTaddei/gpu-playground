#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"

#include "device.hpp"

#include "matchers.hpp"
#include "tensor.hpp"

using namespace Catch::Matchers;
using namespace gpu_playground;

TEST_CASE("matrix: trans", "[matrix]")
{
  auto const devices = make_devices();

  std::vector<float> const data{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<float> const ref{0.0, 3.0, 1.0, 4.0, 2.0, 5.0};
  Shape const shape{2, 3};
  Tensor a(data, shape, devices[DeviceIdx::SERIAL]);

  for (auto const &device : devices)
  {
    if (device != nullptr)
    {
      SECTION(std::string(get_device_name(device->type())))
      {
        a.to(device);

        auto const c            = a.transpose();
        auto const [rows, cols] = c.shape();

        REQUIRE(rows == shape.cols);
        REQUIRE(cols == shape.rows);
        REQUIRE_THAT(c.cpu(), VectorsWithinAbsRel(ref));
      }
    }
  }
}
