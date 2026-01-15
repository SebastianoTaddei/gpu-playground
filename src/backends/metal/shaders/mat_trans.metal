#include <metal_stdlib>

using namespace metal;

kernel void mat_trans(
    const device float* from,
    device float* to,
    constant size_t& m,
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

    to[col * m + row] = from[row * n + col];
}
