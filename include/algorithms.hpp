#pragma once

#include "tensor.hpp"
#include <cmath>
#include <limits>

namespace gpu_playground
{

inline Tensor gradient_descent(
    Tensor const &a,
    Tensor const &b,
    Tensor const &x0,
    size_t const max_iter = 1000,
    float const tol       = std::numeric_limits<float>::epsilon()
)
{
  Tensor x_res{x0};

  auto r = b - a * x_res;

  for (size_t i{0}; i < max_iter; i++)
  {
    auto const r_t = r.transpose();
    auto const r_e = r_t * r;
    if (std::sqrt(r_e.cpu().front()) < tol)
    {
      return x_res;
    }

    auto const ar   = a * r;
    auto const eta  = r_e.cdiv(r_t * ar);
    x_res          += r.smul(eta);
    r              -= ar.smul(eta);
  }

  return x_res;
}

inline Tensor conjuaget_gradient(
    Tensor const &a,
    Tensor const &b,
    Tensor const &x0,
    size_t const max_iter = 1000,
    float const tol       = std::numeric_limits<float>::epsilon()
)
{
  Tensor x_res{x0};

  auto r = b - a * x_res;
  auto p = r;

  for (size_t i{0}; i < max_iter; i++)
  {
    auto const r_t = r.transpose();
    auto const r_e = r_t * r;
    if (std::sqrt(r_e.cpu().front()) < tol)
    {
      return x_res;
    }

    auto const p_t    = p.transpose();
    auto const ap     = a * p;
    auto const alpha  = r_e.cdiv(p_t * ap);
    x_res            += p.smul(alpha);
    r                -= ap.smul(alpha);
    auto const beta   = (r.transpose() * r).cdiv(r_e);
    p                 = r + p.smul(beta);
  }

  return x_res;
}

} // namespace gpu_playground
