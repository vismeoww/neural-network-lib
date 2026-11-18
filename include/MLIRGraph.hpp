#pragma once
#include "BaseGraph.hpp"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

class MLIRGraph : public Graph {
private:
    mlir::MLIRContext context_;
    mlir::OpBuilder builder_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;

    void initParam(Tensor &t, Init init);

public:
    explicit MLIRGraph(unsigned int seed = 42);
    // ~MLIRGraph() override = default;

    NodeId param(Shape s, Init init, std::string label = "") override;
    NodeId apply(std::unique_ptr<Op> op, std::vector<NodeId> in) override;
    
    void forward() override;
    void backward(NodeId loss) override;
    
    void setInputShape(NodeId id, Shape s) override;
    void reshape() override;
    
    std::vector<std::pair<Tensor *, Tensor *>> paramPairs() override;
    void print() override;
};