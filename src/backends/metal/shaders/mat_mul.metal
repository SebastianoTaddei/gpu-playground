#include <metal_stdlib>

using namespace metal;

kernel void mat_mul(
    const device float* a,
    const device float* b,
    device float* c,
    constant size_t& m,
    constant size_t& k,
    constant size_t& n,
    uint2 id [[thread_position_in_grid]]
)
{
    size_t row = id.y;
    size_t col = id.x;

    if (row >= m || col >= n)
    {
      return;
    }

    float support{0.0};
    for (size_t p{0}; p < k; p++)
    {
      support = fma(a[row * k + p], b[p * n + col], support);
    }

    c[row * n + col] = support;
}
