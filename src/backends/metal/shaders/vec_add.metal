#include <metal_stdlib>

using namespace metal;

kernel void vec_add(const device float* a,
                    const device float* b,
                    device float* c,
                    uint id [[thread_position_in_grid]])
{
    c[id] = a[id] + b[id];
}
