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

using NodeId = int;
enum class Init { Zero, He, Xavier };

struct Node {
  std::unique_ptr<Op> op; // null for leaves (inputs / params)
  std::vector<NodeId> inputs;
  Tensor value, grad;
  bool requiresGrad = false;
  bool isParam = false;
  std::string label;
};

class Graph {
  std::vector<Node> nodes_;
  std::mt19937 rng_;

  void initParam(Tensor &t, Init init) {
    const size_t fan_in = t.s.r, fan_out = t.s.c;
    switch (init) {
    case Init::Zero:
      t.zero();
      break;
    case Init::He: {
      std::normal_distribution<Scalar> d(0.0, std::sqrt(2.0 / fan_in));
      for (auto &x : t.v) x = d(rng_);
      break;
    }
    case Init::Xavier: {
      std::normal_distribution<Scalar> d(0.0,
                                         std::sqrt(2.0 / (fan_in + fan_out)));
      for (auto &x : t.v) x = d(rng_);
      break;
    }
    }
  }

public:
  explicit Graph(unsigned seed = 42) : rng_(seed) {}
  std::mt19937 &rng() { return rng_; }

  // Data / label leaf. No gradient.
  NodeId input(Shape s, std::string label = "") {
    Node n;
    n.value.resize(s);
    n.label = std::move(label);
    nodes_.push_back(std::move(n));
    return (NodeId)nodes_.size() - 1;
  }

  // Trainable leaf.
  NodeId param(Shape s, Init init, std::string label = "") {
    Node n;
    n.value.resize(s);
    n.grad.resize(s);
    n.requiresGrad = true;
    n.isParam = true;
    n.label = std::move(label);
    initParam(n.value, init);
    nodes_.push_back(std::move(n));
    return (NodeId)nodes_.size() - 1;
  }

  NodeId apply(std::unique_ptr<Op> op, std::vector<NodeId> in) {
    Node n;
    std::vector<Shape> shapes;
    for (NodeId i : in) {
      assert(i >= 0 && i < (NodeId)nodes_.size());
      shapes.push_back(nodes_[i].value.s);
      n.requiresGrad |= nodes_[i].requiresGrad;
    }
    Shape out = op->outShape(shapes);
    n.value.resize(out);
    if (n.requiresGrad) n.grad.resize(out);
    n.label = op->name();
    n.op = std::move(op);
    n.inputs = std::move(in);
    nodes_.push_back(std::move(n));
    return (NodeId)nodes_.size() - 1;
  }

  Tensor &value(NodeId id) { return nodes_[id].value; }
  Tensor &grad(NodeId id) { return nodes_[id].grad; }
  Shape shape(NodeId id) const { return nodes_[id].value.s; }
  size_t numNodes() const { return nodes_.size(); }

  void forward() {
    std::vector<const Tensor *> in;
    for (Node &n : nodes_) {
      if (!n.op) continue;
      in.clear();
      for (NodeId i : n.inputs) in.push_back(&nodes_[i].value);
      n.op->forward(in, n.value);
    }
  }

  void backward(NodeId loss) {
    if (!nodes_[loss].requiresGrad)
      throw std::logic_error("backward: loss does not depend on any param");
    for (Node &n : nodes_)
      if (n.requiresGrad) n.grad.zero();
    nodes_[loss].grad.fill(1);

    std::vector<const Tensor *> in;
    std::vector<Tensor *> gin;
    for (NodeId id = loss; id >= 0; --id) {
      Node &n = nodes_[id];
      if (!n.op || !n.requiresGrad) continue;
      in.clear();
      gin.clear();
      for (NodeId i : n.inputs) {
        in.push_back(&nodes_[i].value);
        gin.push_back(nodes_[i].requiresGrad ? &nodes_[i].grad : nullptr);
      }
      n.op->backward(in, n.value, n.grad, gin);
    }
  }

  // Change the shape of an input leaf (e.g. batch size) and re-infer every
  // downstream shape. Parameters keep their values.
  void setInputShape(NodeId id, Shape s) {
    if (nodes_[id].op || nodes_[id].isParam)
      throw std::invalid_argument("setInputShape: not an input leaf");
    nodes_[id].value.resize(s);
  }
  void reshape() {
    for (Node &n : nodes_) {
      if (!n.op) continue;
      std::vector<Shape> shapes;
      for (NodeId i : n.inputs) shapes.push_back(nodes_[i].value.s);
      Shape out = n.op->outShape(shapes);
      n.value.resize(out);
      if (n.requiresGrad) n.grad.resize(out);
    }
  }

  std::vector<std::pair<Tensor *, Tensor *>> paramPairs() {
    std::vector<std::pair<Tensor *, Tensor *>> p;
    for (Node &n : nodes_)
      if (n.isParam) p.emplace_back(&n.value, &n.grad);
    return p;
  }

  void print() const {
    for (size_t i = 0; i < nodes_.size(); i++) {
      const Node &n = nodes_[i];
      std::printf("%3zu  %-10s %zux%zu  %s", i, n.label.c_str(), n.value.s.r,
                  n.value.s.c, n.isParam ? "[param] " : "");
      if (!n.inputs.empty()) {
        std::printf("<- ");
        for (NodeId in : n.inputs) std::printf("%d ", in);
      }
      std::printf("\n");
    }
  }
};

// ============================================================================
// Layer helpers
// ============================================================================

NodeId linear(Graph &g, NodeId x, size_t out, Init init = Init::He) {
  const size_t in = g.shape(x).c;
  NodeId W = g.param({in, out}, init, "W");
  NodeId b = g.param({1, out}, Init::Zero, "b");
  NodeId xw = g.apply(std::make_unique<MatMul>(), {x, W});
  return g.apply(std::make_unique<AddBias>(), {xw, b});
}
NodeId relu(Graph &g, NodeId x) {
  return g.apply(std::make_unique<Relu>(), {x});
}
NodeId sigmoid(Graph &g, NodeId x) {
  return g.apply(std::make_unique<Sigmoid>(), {x});
}
NodeId tanh_(Graph &g, NodeId x) {
  return g.apply(std::make_unique<Tanh>(), {x});
}
NodeId mseLoss(Graph &g, NodeId pred, NodeId target) {
  return g.apply(std::make_unique<MSELoss>(), {pred, target});
}

// ============================================================================
// Optimizers
// ============================================================================

using ParamList = std::vector<std::pair<Tensor *, Tensor *>>; // (value, grad)

struct Optimizer {
  virtual ~Optimizer() = default;
  virtual void step(ParamList &params) = 0;
};

struct SGD : Optimizer {
  Scalar lr, momentum;
  std::vector<Tensor> vel;
  explicit SGD(Scalar lr, Scalar momentum = 0) : lr(lr), momentum(momentum) {}
  void step(ParamList &params) override {
    if (vel.empty())
      for (auto &p : params) vel.emplace_back(p.first->s);
    for (size_t i = 0; i < params.size(); i++) {
      Tensor &w = *params[i].first;
      const Tensor &g = *params[i].second;
      for (size_t k = 0; k < w.v.size(); k++) {
        vel[i].v[k] = momentum * vel[i].v[k] - lr * g.v[k];
        w.v[k] += vel[i].v[k];
      }
    }
  }
};

struct Adam : Optimizer {
  Scalar lr, b1, b2, eps;
  long t = 0;
  std::vector<Tensor> m, v;
  explicit Adam(Scalar lr = 1e-3, Scalar b1 = 0.9, Scalar b2 = 0.999,
                Scalar eps = 1e-8)
      : lr(lr), b1(b1), b2(b2), eps(eps) {}
  void step(ParamList &params) override {
    if (m.empty())
      for (auto &p : params) {
        m.emplace_back(p.first->s);
        v.emplace_back(p.first->s);
      }
    t++;
    const Scalar c1 = 1.0 - std::pow(b1, Scalar(t));
    const Scalar c2 = 1.0 - std::pow(b2, Scalar(t));
    for (size_t i = 0; i < params.size(); i++) {
      Tensor &w = *params[i].first;
      const Tensor &g = *params[i].second;
      for (size_t k = 0; k < w.v.size(); k++) {
        m[i].v[k] = b1 * m[i].v[k] + (1 - b1) * g.v[k];
        v[i].v[k] = b2 * v[i].v[k] + (1 - b2) * g.v[k] * g.v[k];
        w.v[k] -= lr * (m[i].v[k] / c1) / (std::sqrt(v[i].v[k] / c2) + eps);
      }
    }
  }
};

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