#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ============================================================================
// Tensor + kernels
// ============================================================================
#include "Tensor.h"


// ============================================================================
// Op interface
// ============================================================================

struct Op {
  virtual ~Op() = default;
  virtual const char *name() const = 0;
  virtual Shape outShape(const std::vector<Shape> &in) const = 0;
  virtual void forward(const std::vector<const Tensor *> &in, Tensor &out) = 0;
  // gradIn[i] == nullptr means input i does not need a gradient.
  // Implementations MUST accumulate (+=).
  virtual void backward(const std::vector<const Tensor *> &in,
                        const Tensor &out, const Tensor &gradOut,
                        const std::vector<Tensor *> &gradIn) = 0;
};

// ----------------------------------------------------------------------------
// C = A * B
// ----------------------------------------------------------------------------
struct MatMul : Op {
  const char *name() const override { return "MatMul"; }
  Shape outShape(const std::vector<Shape> &s) const override {
    if (s.size() != 2 || s[0].c != s[1].r)
      throw std::invalid_argument("MatMul: shape mismatch");
    return {s[0].r, s[1].c};
  }
  void forward(const std::vector<const Tensor *> &in, Tensor &out) override {
    gemm(*in[0], false, *in[1], false, out, false);
  }
  void backward(const std::vector<const Tensor *> &in, const Tensor &,
                const Tensor &G,
                const std::vector<Tensor *> &gin) override {
    if (gin[0]) gemm(G, false, *in[1], true, *gin[0], true); // dX += G * W^T
    if (gin[1]) gemm(*in[0], true, G, false, *gin[1], true); // dW += X^T * G
  }
};

// ----------------------------------------------------------------------------
// Y = X + b, with b of shape 1 x cols broadcast over rows
// ----------------------------------------------------------------------------
struct AddBias : Op {
  const char *name() const override { return "AddBias"; }
  Shape outShape(const std::vector<Shape> &s) const override {
    if (s.size() != 2 || s[1].r != 1 || s[1].c != s[0].c)
      throw std::invalid_argument("AddBias: bias must be 1 x cols");
    return s[0];
  }
  void forward(const std::vector<const Tensor *> &in, Tensor &out) override {
    const Tensor &X = *in[0], &b = *in[1];
    for (size_t i = 0; i < X.s.r; i++)
      for (size_t j = 0; j < X.s.c; j++) out(i, j) = X(i, j) + b(0, j);
  }
  void backward(const std::vector<const Tensor *> &, const Tensor &,
                const Tensor &G,
                const std::vector<Tensor *> &gin) override {
    if (gin[0])
      for (size_t k = 0; k < G.v.size(); k++) gin[0]->v[k] += G.v[k];
    if (gin[1])
      for (size_t i = 0; i < G.s.r; i++)
        for (size_t j = 0; j < G.s.c; j++) (*gin[1])(0, j) += G(i, j);
  }
};

// ----------------------------------------------------------------------------
// Elementwise activations
// ----------------------------------------------------------------------------
struct Relu : Op {
  const char *name() const override { return "Relu"; }
  Shape outShape(const std::vector<Shape> &s) const override { return s[0]; }
  void forward(const std::vector<const Tensor *> &in, Tensor &out) override {
    for (size_t k = 0; k < out.v.size(); k++)
      out.v[k] = in[0]->v[k] > 0 ? in[0]->v[k] : 0;
  }
  void backward(const std::vector<const Tensor *> &in, const Tensor &,
                const Tensor &G,
                const std::vector<Tensor *> &gin) override {
    if (!gin[0]) return;
    for (size_t k = 0; k < G.v.size(); k++)
      if (in[0]->v[k] > 0) gin[0]->v[k] += G.v[k];
  }
};

struct Sigmoid : Op {
  const char *name() const override { return "Sigmoid"; }
  Shape outShape(const std::vector<Shape> &s) const override { return s[0]; }
  void forward(const std::vector<const Tensor *> &in, Tensor &out) override {
    for (size_t k = 0; k < out.v.size(); k++)
      out.v[k] = 1.0 / (1.0 + std::exp(-in[0]->v[k]));
  }
  void backward(const std::vector<const Tensor *> &, const Tensor &out,
                const Tensor &G,
                const std::vector<Tensor *> &gin) override {
    if (!gin[0]) return;
    for (size_t k = 0; k < G.v.size(); k++)
      gin[0]->v[k] += G.v[k] * out.v[k] * (1.0 - out.v[k]);
  }
};

struct Tanh : Op {
  const char *name() const override { return "Tanh"; }
  Shape outShape(const std::vector<Shape> &s) const override { return s[0]; }
  void forward(const std::vector<const Tensor *> &in, Tensor &out) override {
    for (size_t k = 0; k < out.v.size(); k++) out.v[k] = std::tanh(in[0]->v[k]);
  }
  void backward(const std::vector<const Tensor *> &, const Tensor &out,
                const Tensor &G,
                const std::vector<Tensor *> &gin) override {
    if (!gin[0]) return;
    for (size_t k = 0; k < G.v.size(); k++)
      gin[0]->v[k] += G.v[k] * (1.0 - out.v[k] * out.v[k]);
  }
};

// ----------------------------------------------------------------------------
// L = mean((pred - target)^2) over all elements. Output is 1 x 1.
// ----------------------------------------------------------------------------
struct MSELoss : Op {
  const char *name() const override { return "MSELoss"; }
  Shape outShape(const std::vector<Shape> &s) const override {
    if (s.size() != 2 || s[0] != s[1])
      throw std::invalid_argument("MSELoss: pred/target shape mismatch");
    return {1, 1};
  }
  void forward(const std::vector<const Tensor *> &in, Tensor &out) override {
    const Tensor &P = *in[0], &T = *in[1];
    Scalar sum = 0;
    for (size_t k = 0; k < P.v.size(); k++) {
      Scalar d = P.v[k] - T.v[k];
      sum += d * d;
    }
    out.v[0] = sum / Scalar(P.v.size());
  }
  void backward(const std::vector<const Tensor *> &in, const Tensor &,
                const Tensor &G,
                const std::vector<Tensor *> &gin) override {
    const Tensor &P = *in[0], &T = *in[1];
    const Scalar scale = 2.0 * G.v[0] / Scalar(P.v.size());
    for (size_t k = 0; k < P.v.size(); k++) {
      Scalar d = scale * (P.v[k] - T.v[k]);
      if (gin[0]) gin[0]->v[k] += d;
      if (gin[1]) gin[1]->v[k] -= d;
    }
  }
};