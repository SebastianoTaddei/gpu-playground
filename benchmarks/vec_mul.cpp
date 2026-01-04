#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "device.hpp"
#include "tensor.hpp"

using namespace gpu_playground;

static constexpr double NS_TO_MS{1e-6};
static constexpr size_t RUNS{1000};
static constexpr size_t ROWS{1024};

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

class TimeIt
{
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;

public:
  TimeIt() : start{std::chrono::high_resolution_clock::now()} {}

  TimeIt(TimeIt const &)            = default;
  TimeIt(TimeIt &&)                 = delete;
  TimeIt &operator=(TimeIt const &) = default;
  TimeIt &operator=(TimeIt &&)      = delete;

  ~TimeIt()
  {
    auto const end      = std::chrono::high_resolution_clock::now();
    auto const elapsed  = duration_as_ms(this->start, end);
    auto const avg_time = elapsed / static_cast<double>(RUNS);
    std::cout << avg_time << " ms\n";
  }
};

void benchmark_vec_mul(DevicePtr const &device, Tensor &a, Tensor &b)

{
  a.to(device);
  b.to(device);

  auto c = a * b;

  std::cout << get_device_name(device->type()) << ": ";
  {
    TimeIt timer{};

    for (size_t i{0}; i < RUNS; i++)
    {
      c = a * b;
    }
  }
}

} // namespace

int main()
{
  auto serial_device = make_serial_device();
  auto eigen_device  = make_eigen_device();
  auto simd_device   = make_simd_device();
  auto metal_device  = make_metal_device();

  std::vector<float> a_data(ROWS);
  std::vector<float> b_data(ROWS);
  std::iota(a_data.begin(), a_data.end(), 0.0);
  std::iota(b_data.begin(), b_data.end(), 1.0);
  Shape const a_shape{.rows = ROWS, .cols = 1};
  Shape const b_shape{.rows = 1, .cols = ROWS};
  Tensor a(a_data, a_shape, serial_device);
  Tensor b(b_data, b_shape, serial_device);

  benchmark_vec_mul(serial_device, a, b);
  benchmark_vec_mul(eigen_device, a, b);
  benchmark_vec_mul(simd_device, a, b);
  benchmark_vec_mul(metal_device, a, b);

  return 0;
}
