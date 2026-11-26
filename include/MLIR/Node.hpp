#pragma once
#include "Tensor.h"
#include "MLIR/MLIROp.hpp"

namespace mlg{

using NodeId = int;
enum class MLIRInit { Zero, He, Xavier };

struct MLIRNode {
  std::unique_ptr<Op> op; // null for leaves (inputs / params)
  std::vector<NodeId> inputs;
  Tensor value, grad;
  bool requiresGrad = false;
  bool isParam = false;
  std::string label;
};

}