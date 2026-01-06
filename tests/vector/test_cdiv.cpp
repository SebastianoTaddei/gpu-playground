#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"

#include "device.hpp"
#include "device_types.hpp"
#include "matchers.hpp"
#include "tensor.hpp"

using namespace Catch::Matchers;
using namespace gpu_playground;
using vec = std::vector<float>;

namespace
{

void test_mat_cdiv(DevicePtr const &device, Tensor &a, Tensor &b, vec const &ref)
{
  SECTION(std::string(get_device_name(device->type())))
  {
    a.to(device);
    b.to(device);

    auto const c = a.cdiv(b);

    REQUIRE_THAT(c.cpu(), VectorsWithinAbsRel(ref));
  }
}

} // namespace

TEST_CASE("vector: cdiv", "[vector]")
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  std::vector<float> const a_data{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<float> const b_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<float> const ref{0.0, 0.5, 2.0 / 3.0, 3.0 / 4.0, 4.0 / 5.0, 5.0 / 6.0};
  Shape const shape{1, 6};
  Tensor a(a_data, shape, serial_device);
  Tensor b(b_data, shape, serial_device);

  test_mat_cdiv(serial_device, a, b, ref);
  test_mat_cdiv(eigen_device, a, b, ref);
  test_mat_cdiv(simd_device, a, b, ref);
  test_mat_cdiv(metal_device, a, b, ref);
}
