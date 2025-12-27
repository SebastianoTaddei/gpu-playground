#include <iostream>

#include "device.hpp"

int main()
{
  auto cpu_device   = backend::make_cpu_device();
  auto metal_device = backend::make_metal_device();

  std::cout << "The CPU device is:   " << backend::get_device_name(cpu_device->type()) << '\n';
  std::cout << "The Metal device is: " << backend::get_device_name(metal_device->type()) << '\n';
  return 0;
}
