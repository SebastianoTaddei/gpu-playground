__global__ void mat_smul(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
      c[i] = a[i] * b[0];
    }
}
