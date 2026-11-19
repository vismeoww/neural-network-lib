#pragma once
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
protected: 
  std::vector<Node> nodes_;
  std::mt19937 rng_;

public:
  explicit Graph(unsigned int seed  = 42): rng_(seed) {}
  virtual ~Graph() = default;
  std::mt19937 &rng() { return rng_; }

  // Data / label leaf. No gradient.
  NodeId input(Shape s, std::string label = ""){
    Node node;
    node.value.resize(s);
    node.label = std::move(label);
    nodes_.push_back(std::move(node));
    return (NodeId)nodes_.size() - 1;
  }
  Tensor &value(NodeId id){return nodes_[id].value;}
  Tensor &grad(NodeId id){return nodes_[id].grad;}
  Shape shape(NodeId id) const {return nodes_[id].value.s;}
  size_t numNodes() const{return nodes_.size();}

  // Trainable leaf.
  virtual NodeId param(Shape s, Init init, std::string label = "") = 0;

  virtual NodeId apply(std::unique_ptr<Op> op, std::vector<NodeId> in) = 0; 


  virtual void forward() = 0; 

  virtual void backward(NodeId loss) = 0;

  // Change the shape of an input leaf (e.g. batch size) and re-infer every
  // downstream shape. Parameters keep their values.
  virtual void setInputShape(NodeId id, Shape s) = 0;

  virtual void reshape() = 0; 

  virtual std::vector<std::pair<Tensor *, Tensor *>> paramPairs() = 0; 

  virtual void print() = 0; 
};