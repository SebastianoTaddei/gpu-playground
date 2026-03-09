__global__ void mat_mul(const float* a, const float* b, float* c, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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
