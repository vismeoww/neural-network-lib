#include "Graph.h"


void Graph::initParam(Tensor &t, Init init) {
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


NodeId Graph::input(Shape s, std::string label) {
    Node n;
    n.value.resize(s);
    n.label = std::move(label);
    nodes_.push_back(std::move(n));
    return (NodeId)nodes_.size() - 1;
}


NodeId Graph::param(Shape s, Init init, std::string label) {
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

NodeId Graph::apply(std::unique_ptr<Op> op, std::vector<NodeId> in) {
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


void Graph::forward() {
    std::vector<const Tensor *> in;
    for (Node &n : nodes_) {
      if (!n.op) continue;
      in.clear();
      for (NodeId i : n.inputs) in.push_back(&nodes_[i].value);
      n.op->forward(in, n.value);
    }
}

void Graph::backward(NodeId loss) {
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


void Graph::setInputShape(NodeId id, Shape s) {
    if (nodes_[id].op || nodes_[id].isParam)
      throw std::invalid_argument("setInputShape: not an input leaf");
    nodes_[id].value.resize(s);
}


void Graph::reshape() {
    for (Node &n : nodes_) {
        if (!n.op) continue;
        std::vector<Shape> shapes;
        for (NodeId i : n.inputs) shapes.push_back(nodes_[i].value.s);
        Shape out = n.op->outShape(shapes);
        n.value.resize(out);
        if (n.requiresGrad) n.grad.resize(out);
    }
}

std::vector<std::pair<Tensor *, Tensor *>> Graph::paramPairs() {
    std::vector<std::pair<Tensor *, Tensor *>> p;
    for (Node &n : nodes_)
      if (n.isParam) p.emplace_back(&n.value, &n.grad);
    return p;
}


void Graph::print() const {
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


NodeId linear(Graph &g, NodeId x, size_t out, Init init) {
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