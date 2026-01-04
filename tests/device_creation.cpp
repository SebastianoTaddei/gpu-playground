#include <iostream>

#include "device.hpp"

using namespace gpu_playground;

namespace
{

void print_device_type(DevicePtr const &device)
{
  std::cout << "The device type is: " << get_device_name(device->type()) << '\n';
}

} // namespace

int main()
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  print_device_type(serial_device);
  print_device_type(eigen_device);
  print_device_type(simd_device);
  print_device_type(metal_device);

  return 0;
}
