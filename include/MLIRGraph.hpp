#pragma once
#include "Graph.hpp"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

class MLIRGraph : public Graph {
private:
    mlir::MLIRContext context_;
    mlir::OpBuilder builder_;
    mlir::ModuleOp module_;
    mlir::func::FuncOp fwdFunc_;
    std::map<NodeId,mlir::Value> mlirValues_;
    std::vector<NodeId> inputArgNodes_;

    void initParam(Tensor &t, Init init);

public:
    explicit MLIRGraph(unsigned int seed = 42);
    // ~MLIRGraph() override = default;

    NodeId param(Shape s, Init init, std::string label = "") override;
    mlir::Value getOrAddValue(NodeId id);
    NodeId apply(std::unique_ptr<Op> op, std::vector<NodeId> in) override;
   
    void finalizeForward(NodeId outputNode);
    void forward() override;
    void backward(NodeId loss) override;
    
    void setInputShape(NodeId id, Shape s) override;
    void reshape() override;
    
    std::vector<std::pair<Tensor *, Tensor *>> paramPairs() override;
    void print() override;
};