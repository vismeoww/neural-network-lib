// autograd.cpp -- minimal reverse-mode autodiff graph (CPU), single file.
//
// Build:  g++ -O2 -std=c++17 -Wall -Wextra autograd.cpp -o autograd
//
// Conventions
//   * Row-major tensors, one row per sample:  Y = X * W + b
//   * Nodes can only reference earlier nodes, so creation order is already a
//     topological order: forward = loop first->last, backward = last->first.
//   * Every Op::backward must ACCUMULATE (+=) into its input gradients.

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
#include "Operator.h"

// ============================================================================
// Graph
// ============================================================================
#include "Graph.h"


// ============================================================================
// Gradient check: compares analytic gradients with central differences.
// Returns the max relative error over all (or up to maxPerParam) entries.
// ============================================================================

double gradCheck(Graph &g, NodeId loss, double eps = 1e-6,
                 size_t maxPerParam = 50) {
  g.forward();
  g.backward(loss);
  auto params = g.paramPairs();
  // Copy analytic grads because later forward() calls don't touch them, but
  // be safe in case someone changes that.
  std::vector<Tensor> analytic;
  for (auto &p : params) analytic.push_back(*p.second);

  double worst = 0;
  for (size_t pi = 0; pi < params.size(); pi++) {
    Tensor &w = *params[pi].first;
    const size_t n = w.v.size();
    const size_t stride = std::max<size_t>(1, n / maxPerParam);
    for (size_t k = 0; k < n; k += stride) {
      const Scalar orig = w.v[k];
      w.v[k] = orig + eps;
      g.forward();
      const double lp = g.value(loss).v[0];
      w.v[k] = orig - eps;
      g.forward();
      const double lm = g.value(loss).v[0];
      w.v[k] = orig;
      const double num = (lp - lm) / (2 * eps);
      const double ana = analytic[pi].v[k];
      const double rel =
          std::fabs(num - ana) / std::max(1e-8, std::fabs(num) + std::fabs(ana));
      worst = std::max(worst, rel);
    }
  }
  g.forward(); // leave graph in a consistent state
  return worst;
}

// ============================================================================
// Demo: fit a polynomial with a small MLP
// ============================================================================

static double truePoly(double x) {
  return 0.5 * x * x * x - 0.8 * x * x + 0.3 * x + 0.1;
}

static void runGradChecks() {
  std::printf("== Gradient checks (relative error, want < 1e-6) ==\n");
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);

  struct Case {
    const char *name;
    NodeId (*act)(Graph &, NodeId);
  } cases[] = {{"tanh", tanh_}, {"sigmoid", sigmoid}, {"relu", relu}};

  for (auto &c : cases) {
    Graph g(1);
    NodeId x = g.input({5, 3});
    NodeId t = g.input({5, 2});
    NodeId h = c.act(g, linear(g, x, 4));
    NodeId y = linear(g, h, 2);
    NodeId loss = mseLoss(g, y, t);
    for (auto &z : g.value(x).v) z = u(rng);
    for (auto &z : g.value(t).v) z = u(rng);
    // Non-zero biases so bias gradients are exercised meaningfully.
    for (auto &p : g.paramPairs())
      if (p.first->s.r == 1)
        for (auto &z : p.first->v) z = 0.1 * u(rng);
    double err = gradCheck(g, loss);
    std::printf("  MLP with %-8s : %.3e  %s\n", c.name, err,
                err < 1e-6 ? "OK" : "FAIL");
  }
}

int main() {
  runGradChecks();

  // ---- data ----------------------------------------------------------------
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> ux(-1.0, 1.0);
  std::normal_distribution<double> noise(0.0, 0.02);

  const size_t N = 512, B = 64;
  std::vector<double> X(N), Y(N);
  for (size_t i = 0; i < N; i++) {
    X[i] = ux(rng);
    Y[i] = truePoly(X[i]) + noise(rng);
  }

  // ---- model ---------------------------------------------------------------
  Graph g(42);
  NodeId x = g.input({B, 1}, "x");
  NodeId t = g.input({B, 1}, "target");
  NodeId h1 = tanh_(g, linear(g, x, 32, Init::Xavier));
  NodeId h2 = tanh_(g, linear(g, h1, 32, Init::Xavier));
  NodeId y = linear(g, h2, 1, Init::Xavier);
  NodeId loss = mseLoss(g, y, t);

  std::printf("\n== Graph ==\n");
  g.print();

  auto params = g.paramPairs();
  Adam opt(0.01);

  // ---- train ---------------------------------------------------------------
  std::printf("\n== Training (minibatch %zu, Adam) ==\n", B);
  std::vector<size_t> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  const int epochs = 800;
  for (int e = 1; e <= epochs; e++) {
    std::shuffle(idx.begin(), idx.end(), rng);
    double epochLoss = 0;
    size_t batches = 0;
    for (size_t s = 0; s + B <= N; s += B) {
      for (size_t i = 0; i < B; i++) {
        g.value(x)(i, 0) = X[idx[s + i]];
        g.value(t)(i, 0) = Y[idx[s + i]];
      }
      g.forward();
      g.backward(loss);
      opt.step(params);
      epochLoss += g.value(loss).v[0];
      batches++;
    }
    if (e == 1 || e % 50 == 0)
      std::printf("  epoch %4d  loss %.6f\n", e, epochLoss / batches);
  }

  // ---- evaluate on fresh points with a different batch size ----------------
  const size_t T = 9;
  g.setInputShape(x, {T, 1});
  g.setInputShape(t, {T, 1});
  g.reshape();
  for (size_t i = 0; i < T; i++) {
    g.value(x)(i, 0) = -1.0 + 2.0 * double(i) / double(T - 1);
    g.value(t)(i, 0) = truePoly(g.value(x)(i, 0));
  }
  g.forward();
  std::printf("\n== Evaluation (batch size changed to %zu) ==\n", T);
  std::printf("      x      true      pred     error\n");
  double maxErr = 0;
  for (size_t i = 0; i < T; i++) {
    double xi = g.value(x)(i, 0), ti = g.value(t)(i, 0),
           pi = g.value(y)(i, 0);
    maxErr = std::max(maxErr, std::fabs(pi - ti));
    std::printf("  %6.3f  %8.4f  %8.4f  %8.4f\n", xi, ti, pi, pi - ti);
  }
  std::printf("  test MSE %.6f, max abs error %.4f\n", g.value(loss).v[0],
              maxErr);
  return 0;
}