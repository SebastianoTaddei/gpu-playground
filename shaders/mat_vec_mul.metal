#include <metal_stdlib>
#include "shader_params.hpp"
using namespace metal;

kernel void mat_vec_mul(const device float* a,
                        const device float* x,
                        device float* y,
                        MatVecMulParams constant &params,
                        uint id [[thread_position_in_grid]])
{
    size_t const cols = params.cols;

    float sum{0.0};
    for (size_t i{0}; i < cols; i++)
    {
      size_t const idx = id * cols + i;
      sum += a[idx] * x[idx];
    }

    y[id] = sum;
}
