#pragma once
#include "Tensor.h"
#include "CPU/CPUOp.h"

using NodeId = int;
enum class Init { Zero, He, Xavier };

struct CPUNode {
  std::unique_ptr<Op> op; // null for leaves (inputs / params)
  std::vector<NodeId> inputs;
  Tensor value, grad;
  bool requiresGrad = false;
  bool isParam = false;
  std::string label;
};
