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
#include "CPUOp.h"

#include "Node.hpp"

namespace cpu{

class Graph
{
  void initParam(Tensor &t, Init init);
  std::mt19937 rng_;
  std::vector<CPUNode> nodes_;

public:
  explicit Graph(unsigned seed = 42) : rng_(seed) {}

  // Data / label leaf. No gradient.
  // NodeId input(Shape s, std::string label = "");
  NodeId input(Shape s, std::string label = "")
  {
    CPUNode node;
    node.value.resize(s);
    node.label = std::move(label);
    nodes_.push_back(std::move(node));
    return (NodeId)nodes_.size() - 1;
  }
  Tensor &value(NodeId id) { return nodes_[id].value; }
  Tensor &grad(NodeId id) { return nodes_[id].grad; }
  Shape shape(NodeId id) const { return nodes_[id].value.s; }
  size_t numNodes() const { return nodes_.size(); }

  // Trainable leaf.
  NodeId param(Shape s, Init init, std::string label = "");

  NodeId apply(std::unique_ptr<Op> op, std::vector<NodeId> in);

  void forward();

  void backward(NodeId loss);

  // Change the shape of an input leaf (e.g. batch size) and re-infer every
  // downstream shape. Parameters keep their values.
  void setInputShape(NodeId id, Shape s);

  void reshape();

  std::vector<std::pair<Tensor *, Tensor *>> paramPairs();

  void print();
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

}