#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>



using Scalar = double; // switch to float later, keep double for gradcheck

struct Shape {
  size_t r = 0, c = 0;
  size_t size() const { return r * c; }
  bool operator==(const Shape &o) const { return r == o.r && c == o.c; }
  bool operator!=(const Shape &o) const { return !(*this == o); }
};

struct Tensor {
  Shape s;
  std::vector<Scalar> v;

  Tensor() = default;
  explicit Tensor(Shape sh) : s(sh), v(sh.size(), 0) {}

  Scalar &operator()(size_t i, size_t j) { return v[i * s.c + j]; }
  Scalar operator()(size_t i, size_t j) const { return v[i * s.c + j]; }

  void zero() { std::fill(v.begin(), v.end(), Scalar(0)); }
  void fill(Scalar x) { std::fill(v.begin(), v.end(), x); }
  void resize(Shape sh) {
    s = sh;
    v.assign(sh.size(), Scalar(0));
  }
};

// C (+)= op(A) * op(B), where op is optional transpose.
// One kernel covers the forward product and both backward products.
// This is the only place that needs optimising (blocking / OpenMP / BLAS).
void gemm(const Tensor &A, bool tA, const Tensor &B, bool tB, Tensor &C, bool accumulate);