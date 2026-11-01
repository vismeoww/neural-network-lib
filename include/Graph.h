#pragma once

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

#include "Tensor.h"
#include "Operator.h"

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

  void initParam(Tensor &t, Init init);

public:
  explicit Graph(unsigned seed = 42) : rng_(seed) {}
  std::mt19937 &rng() { return rng_; }

  // Data / label leaf. No gradient.
  NodeId input(Shape s, std::string label = "");

  // Trainable leaf.
  NodeId param(Shape s, Init init, std::string label = "");

  NodeId apply(std::unique_ptr<Op> op, std::vector<NodeId> in); 

  Tensor &value(NodeId id) { return nodes_[id].value; }
  Tensor &grad(NodeId id) { return nodes_[id].grad; }
  Shape shape(NodeId id) const { return nodes_[id].value.s; }
  size_t numNodes() const { return nodes_.size(); }

  void forward(); 

  void backward(NodeId loss);

  // Change the shape of an input leaf (e.g. batch size) and re-infer every
  // downstream shape. Parameters keep their values.
  void setInputShape(NodeId id, Shape s);

  void reshape(); 

  std::vector<std::pair<Tensor *, Tensor *>> paramPairs(); 

  void print() const; 
};

// ============================================================================
// Layer helpers
// ============================================================================

NodeId linear(Graph &g, NodeId x, size_t out, Init init = Init::He); 
NodeId relu(Graph &g, NodeId x); 
NodeId sigmoid(Graph &g, NodeId x); 
NodeId tanh_(Graph &g, NodeId x); 
NodeId mseLoss(Graph &g, NodeId pred, NodeId target); 

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