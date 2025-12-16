#include <metal_stdlib>
using namespace metal;

kernel void ax_minus_b(const device float* a      [[buffer(0)]],
                       const device float* x      [[buffer(1)]],
                       const device float* b      [[buffer(2)]],
                       device float* out          [[buffer(3)]],
                       uint id                    [[thread_position_in_grid]])
{
    out[id] = fma(a[id], x[id], -b[id]);
}
