#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"

#include "device.hpp"

#include "matchers.hpp"
#include "tensor.hpp"

using namespace Catch::Matchers;
using namespace gpu_playground;

TEST_CASE("vector: sdiv", "[vector]")
{
  auto const devices = make_devices();

  std::vector<float> const a_data{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<float> const b_data{2.0};
  std::vector<float> const ref{0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
  Shape const a_shape{6, 1};
  Shape const b_shape{1, 1};
  Tensor a(a_data, a_shape, devices[DeviceIdx::SERIAL]);
  Tensor b(b_data, b_shape, devices[DeviceIdx::SERIAL]);

  for (auto const &device : devices)
  {
    if (device != nullptr)
    {
      SECTION(std::string(get_device_name(device->type())))
      {
        a.to(device);
        b.to(device);

        auto const c = a.sdiv(b);

        REQUIRE_THAT(c.cpu(), VectorsWithinAbsRel(ref));
      }
    }
  }
}
