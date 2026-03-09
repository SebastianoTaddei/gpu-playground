__global__ void mat_trans(const float* from, float* to, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
    {
      return;
    }
    
    to[col * m + row] = from[row * n + col];
}
