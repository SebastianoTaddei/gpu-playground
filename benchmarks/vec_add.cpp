#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;
using vec = std::vector<float>;

static constexpr double NS_TO_MS{1e-6};
static constexpr size_t RUNS{1000};
static constexpr size_t LEN{10000};

namespace
{

double duration_as_ms(
    std::chrono::time_point<std::chrono::high_resolution_clock> const &start,
    std::chrono::time_point<std::chrono::high_resolution_clock> const &end
)
{
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
         ) *
         NS_TO_MS;
}

void benchmark_vec_add(DevicePtr const &device, vec const &vec_a, vec const &vec_b)
{
  assert(vec_a.size() == vec_b.size());

  auto const tensor_a = gpu_playground::Tensor(vec_a, device);
  auto const tensor_b = gpu_playground::Tensor(vec_b, device);
  auto tensor_c       = tensor_a + tensor_b;

  auto const start = std::chrono::high_resolution_clock::now();
  for (size_t i{0}; i < RUNS; i++)
  {
    tensor_c = tensor_a + tensor_b;
  }
  auto const end = std::chrono::high_resolution_clock::now();

  auto const elapsed  = duration_as_ms(start, end);
  auto const avg_time = elapsed / static_cast<double>(RUNS);
  std::cout << get_device_name(device->type()) << " avg time: " << avg_time << " ms\n";
}

} // namespace

int main()
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  std::vector<float> vec_a(LEN);
  std::vector<float> vec_b(LEN);
  std::iota(vec_a.begin(), vec_a.end(), 0.0);
  std::iota(vec_b.begin(), vec_b.end(), 1.0);

  benchmark_vec_add(serial_device, vec_a, vec_b);
  benchmark_vec_add(eigen_device, vec_a, vec_b);
  benchmark_vec_add(simd_device, vec_a, vec_b);
  benchmark_vec_add(metal_device, vec_a, vec_b);

  return 0;
}
