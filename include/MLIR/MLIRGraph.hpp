#pragma once
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>

#include "Node.hpp"
#include "Tensor.h"
#include "MLIROp.hpp"

namespace mlg{

struct MemRef2D {
    Scalar *allocated;
    Scalar *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

static MemRef2D toMemRef2D(Tensor &t) {
    int64_t rows = static_cast<int64_t>(t.s.r);
    int64_t cols = static_cast<int64_t>(t.s.c);
    return MemRef2D{
        t.v.data(), // allocated
        t.v.data(), // aligned
        0,          // offset
        {rows, cols},
        {cols, 1}   // row-major strides
    };
}

class Graph{
private:
    mlir::MLIRContext context_;
    mlir::OpBuilder builder_;
    mlir::ModuleOp module_;
    mlir::func::FuncOp fwdFunc_;
    std::map<NodeId,mlir::Value> mlirValues_;
    std::vector<NodeId> inputArgNodes_;
    std::vector<MLIRNode> nodes_;
    std::mt19937 rng_;
    std::vector<NodeId> outputNodes_;
    std::unique_ptr<mlir::ExecutionEngine> engine_;
    std::unique_ptr<mlir::ExecutionEngine> bwdEngine_;
    std::vector<NodeId> bwdParamNodes_;
    NodeId compiledLossNode_ = -1;

    void initParam(Tensor &t, MLIRInit init);
    mlir::Location makeNodeLoc(NodeId id);

public:
    explicit Graph(unsigned int seed = 42);
    ~Graph() = default;
    NodeId input(Shape s, std::string label = "")
    {
        MLIRNode node;
        node.value.resize(s);
        node.label = std::move(label);
        nodes_.push_back(std::move(node));
        return (NodeId)nodes_.size() - 1;
    }
    Tensor &value(NodeId id) { return nodes_[id].value; }
    Tensor &grad(NodeId id) { return nodes_[id].grad; }
    Shape shape(NodeId id) const { return nodes_[id].value.s; }
    size_t numNodes() const { return nodes_.size(); }

    NodeId param(Shape s, MLIRInit init, std::string label = "");
    mlir::Value getOrAddValue(NodeId id);
    NodeId apply(std::unique_ptr<Op> op, std::vector<NodeId> in);
   
    void finalizeForward(std::vector<NodeId> outputNodes);
    void forward();
    void compileBackwards(NodeId loss);
    void backward(NodeId loss);
    
    void setInputShape(NodeId id, Shape s);
    void reshape();
    
    std::vector<std::pair<Tensor *, Tensor *>> paramPairs();
    void print();
    void dump();
};

// ============================================================================
// Layer helpers
// ============================================================================

NodeId linear(Graph &g, NodeId x, size_t out, MLIRInit init = MLIRInit::He);
NodeId relu(Graph &g, NodeId x);
NodeId sigmoid(Graph &g, NodeId x);
NodeId tanh_(Graph &g, NodeId x);
NodeId mseLoss(Graph &g, NodeId pred, NodeId target);

}