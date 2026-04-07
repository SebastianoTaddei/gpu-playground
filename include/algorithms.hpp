#pragma once

#include <cmath>
#include <limits>

#include "tensor.hpp"

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

  constexpr size_t iter_check{10};
  for (size_t i{0}; i < max_iter; i++)
  {
    auto const r_t = r.transpose();
    auto const r_e = r_t * r;

    if (i % iter_check == 0 and std::sqrt(r_e.cpu().front()) < tol)
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

/*
    auto const r_t = r.transpose();
    auto const r_e = r_t * r;

    // if (i % iter_check == 0 and std::sqrt(r_e.cpu().front()) < tol)
    // {
    //   // std::cout << "Iter: " << i << '\n';
    //   return x_res;
    // }

    auto const p_t    = p.transpose();
    auto const ap     = a * p;
    auto const alpha  = r_e.cdiv(p_t * ap);
    x_res            += p.smul(alpha);
    r                -= ap.smul(alpha);
    auto const beta   = (r.transpose() * r).cdiv(r_e);
    p                 = r + p.smul(beta);
*/

inline Tensor conjugate_gradient(
    Tensor const &a,
    Tensor const &b,
    Tensor const &x0,
    size_t const max_iter            = 1000,
    [[maybe_unused]] float const tol = std::numeric_limits<float>::epsilon()
)
{
  Tensor x_res{x0};

  auto r = b - a * x_res;
  auto p = r;

  auto r_t     = r.transpose();
  auto r_e     = r_t * r;
  auto p_t     = p.transpose();
  auto ap      = a * p;
  auto alpha   = r_e.cdiv(p_t * ap);
  auto palpha  = p.smul(alpha);
  auto apalpha = ap.smul(alpha);
  auto beta    = (r.transpose() * r).cdiv(r_e);
  auto rtr     = r_t * r;
  auto pbeta   = p.smul(beta);
  auto ptap    = p_t * ap;

  constexpr size_t iter_check{10};
  size_t i{0};
  for (; i < max_iter; i++)
  {
    r.transpose(r_t);
    r_t.mul(r, r_e);

    if (i % iter_check == 0 and std::sqrt(r_e.cpu().front()) < tol)
    {
      return x_res;
    }

    p.transpose(p_t);
    a.mul(p, ap);
    p_t.mul(ap, ptap);
    r_e.cdiv(ptap, alpha);
    p.smul(alpha, palpha);
    x_res += palpha;
    ap.smul(alpha, apalpha);
    r -= apalpha;
    r.transpose(r_t);
    r_t.mul(r, rtr);
    rtr.cdiv(r_e, beta);
    p.smul(beta, pbeta);
    r.add(pbeta, p);
  }

  return x_res;
}

} // namespace gpu_playground
