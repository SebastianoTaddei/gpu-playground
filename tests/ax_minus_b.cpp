#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include "metal/metal.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

constexpr size_t len{1000000};
constexpr size_t rows{100};
constexpr size_t cols{100};

void generate_data(std::vector<float> &A, std::vector<float> &x, std::vector<float> &b)
{
  std::random_device dev;
  std::mt19937 rng(dev());

  std::normal_distribution<float> dist{};

  A.reserve(len);
  x.reserve(len);
  b.reserve(len);

  for (size_t i{0}; i < len; i++)
  {
    A.push_back(dist(rng));
    x.push_back(dist(rng));
    b.push_back(dist(rng));
  }
}

void ax_minus_b(
  std::vector<float> const &A, std::vector<float> const &x, std::vector<float> const &b, std::vector<float> &out
)
{
  for (size_t i{0}; i < A.size(); i++)
  {
    out[i] = std::fmaf(A[i], x[i], -b[i]);
  }
}

void mat_vec_mul(Eigen::MatrixXf const &A, Eigen::VectorXf const &x, Eigen::VectorXf &y)
{
  y = A * x;
}

int main()
{
  using namespace MTL;

  // Data
  std::vector<float> A;
  std::vector<float> x;
  std::vector<float> b;
  std::vector<float> out_cpu(len);
  std::vector<float> out_gpu(len);

  Eigen::MatrixXf A_eig;
  Eigen::VectorXf x_eig;
  Eigen::VectorXf y_eig;
  A_eig.setRandom(rows, cols);
  x_eig.setRandom(rows);
  x_eig.setZero(rows);

  constexpr size_t cache_runs{2};
  constexpr size_t runs{1000};

  std::cout
    << "Running with: cache_runs " << cache_runs << " runs " << runs << " rows " << rows << " cols "
    << cols << "\n";

  {
    for (size_t i{0}; i < cache_runs; i++)
    {
      mat_vec_mul(A_eig, x_eig, y_eig);
    }

    auto const start = std::chrono::steady_clock::now();

    for (size_t i{0}; i < runs; i++)
    {
      mat_vec_mul(A_eig, x_eig, y_eig);
    }

    auto const end = std::chrono::steady_clock::now();
    std::cout << "CPU took: " << (end - start).count() * 1e-9 / runs << "s\n";
  }

  // 1) Create device + queue
  auto device = CreateSystemDefaultDevice();
  auto queue  = device->newCommandQueue();

  // 2) Load metallib
  NS::Error *err = nullptr;

  auto url = NS::URL::fileURLWithPath(NS::String::string("ax_minus_b.metallib", NS::UTF8StringEncoding));

  auto lib = device->newLibrary(url, &err);

  if (!lib)
  {
    std::cerr
      << "Failed to load metallib: " << (err ? err->localizedDescription()->utf8String() : "unknown")
      << "\n";
    return 1;
  }

  // 3) Create compute pipeline
  auto fn = lib->newFunction(NS::String::string("ax_minus_b", NS::UTF8StringEncoding));
  if (!fn)
  {
    std::cerr << "Couldn't find the function `ax_minus_b`\n";
    return 1;
  }
  NS::Error *err2 = nullptr;
  auto pipeline   = device->newComputePipelineState(fn, &err2);

  std::cout << "Device name: " << device->name()->cString(NS::StringEncoding::ASCIIStringEncoding) << "\n";
  std::cout
    << "Device arch: "
    << device->architecture()->name()->cString(NS::StringEncoding::ASCIIStringEncoding) << "\n";
  std::cout << "Pipeline max threads: " << pipeline->maxTotalThreadsPerThreadgroup() << "\n";

  // 4) Buffers
  auto bufA   = device->newBuffer(A.data(), len * sizeof(float), ResourceStorageModeShared);
  auto bufx   = device->newBuffer(x.data(), len * sizeof(float), ResourceStorageModeShared);
  auto bufb   = device->newBuffer(b.data(), len * sizeof(float), ResourceStorageModeShared);
  auto bufOut = device->newBuffer(out_gpu.data(), len * sizeof(float), ResourceStorageModeShared);

  auto bufA_priv   = device->newBuffer(len * sizeof(float), ResourceStorageModePrivate);
  auto bufx_priv   = device->newBuffer(len * sizeof(float), ResourceStorageModePrivate);
  auto bufb_priv   = device->newBuffer(len * sizeof(float), ResourceStorageModePrivate);
  auto bufOut_priv = device->newBuffer(len * sizeof(float), ResourceStorageModePrivate);

  {
    auto cmd  = queue->commandBuffer();
    auto blit = cmd->blitCommandEncoder();
    blit->copyFromBuffer(bufA, 0, bufA_priv, 0, len * sizeof(float));
    blit->copyFromBuffer(bufx, 0, bufx_priv, 0, len * sizeof(float));
    blit->copyFromBuffer(bufb, 0, bufb_priv, 0, len * sizeof(float));
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
  }

  constexpr size_t threadsPerTG = 256;
  MTL::Size tgSize(threadsPerTG, 1, 1);
  MTL::Size tgCount((len + threadsPerTG - 1) / threadsPerTG, 1, 1);

  // 6) Run
  {
    auto cmdBuf = queue->commandBuffer();
    auto enc    = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    enc->setBuffer(bufA_priv, 0, 0);
    enc->setBuffer(bufx_priv, 0, 1);
    enc->setBuffer(bufb_priv, 0, 2);
    enc->setBuffer(bufOut_priv, 0, 3);

    for (size_t i{0}; i < cache_runs; i++)
    {
      enc->dispatchThreadgroups(tgCount, tgSize);
    }

    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
  }

  {
    auto const start = std::chrono::steady_clock::now();

    auto cmdBuf = queue->commandBuffer();
    auto enc    = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    enc->setBuffer(bufA_priv, 0, 0);
    enc->setBuffer(bufx_priv, 0, 1);
    enc->setBuffer(bufb_priv, 0, 2);
    enc->setBuffer(bufOut_priv, 0, 3);

    for (size_t i{0}; i < runs; i++)
    {
      enc->dispatchThreadgroups(tgCount, tgSize);
    }

    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    auto const end = std::chrono::steady_clock::now();
    std::cout << "GPU took: " << (end - start).count() * 1e-9 / runs << "s\n";
  }

  {
    auto cmd  = queue->commandBuffer();
    auto blit = cmd->blitCommandEncoder();
    blit->copyFromBuffer(bufOut_priv, 0, bufOut, 0, len * sizeof(float));
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
  }

  // Copy back (shared memory – already synced)
  memcpy(out_gpu.data(), bufOut->contents(), out_gpu.size() * sizeof(float));

  // Check result
  for (size_t i{0}; i < len; i++)
  {
    if (out_cpu[i] - out_gpu[i] != 0.0)
    {
      std::cerr << "Results differ!\n";
      return 1;
    }
  }

  return 0;
}
