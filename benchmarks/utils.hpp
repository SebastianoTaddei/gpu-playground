#pragma once

#include <chrono>
#include <iostream>

namespace gpu_playground::benchmark
{

class TimeIt
{
private:
  static constexpr double NS_TO_MS{1e-6};

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  size_t runs;

public:
  explicit TimeIt(size_t runs) : start{std::chrono::high_resolution_clock::now()}, runs{runs} {}

  TimeIt(TimeIt const &)            = default;
  TimeIt(TimeIt &&)                 = delete;
  TimeIt &operator=(TimeIt const &) = default;
  TimeIt &operator=(TimeIt &&)      = delete;

  ~TimeIt()
  {
    auto const end      = std::chrono::high_resolution_clock::now();
    auto const elapsed  = TimeIt::duration_as_ms(this->start, end);
    auto const avg_time = elapsed / static_cast<double>(this->runs);
    std::cout << avg_time << " ms\n";
  }

  static double duration_as_ms(
      std::chrono::time_point<std::chrono::high_resolution_clock> const &start,
      std::chrono::time_point<std::chrono::high_resolution_clock> const &end
  )
  {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
           ) *
           TimeIt::NS_TO_MS;
  }
};

} // namespace gpu_playground::benchmark
