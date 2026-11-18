#include "MLIRGraph.hpp"
#include <mlir/InitAllDialects.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <llvm/Support/raw_ostream.h>


void MLIRGraph::initParam(Tensor &t, Init init){
    assert(0&&"not implemented");
}

MLIRGraph::MLIRGraph(unsigned int seed) 
    : Graph(seed), builder_(&context_) {
    // Register all upstream dialects
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();

    // Create the top-level MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToStart(module_->getBody());
}

NodeId MLIRGraph::param(Shape s, Init init, std::string label) {
    Node node;
    node.value.resize(s);
    node.grad.resize(s);
    node.label = std::move(label);
    node.isParam = true;
    node.requiresGrad = true;
    
    initParam(node.value, init); // Your weight initialization logic
    nodes_.push_back(std::move(node));
    
    return nodes_.size() - 1;
}

NodeId MLIRGraph::apply(std::unique_ptr<Op> op, std::vector<NodeId> in) {
    Node node;
    node.op = std::move(op);
    node.inputs = std::vector<NodeId>{in.begin(), in.end()};
    
    // Infer output shape based on operator type and input tensors
    Shape outShape = node.op->outShape(
        std::vector<Shape>{nodes_[in[0]].value.s, nodes_[in[1]].value.s}
    );
    node.value.resize(outShape);
    node.grad.resize(outShape);
    
    // Emit linalg/tensor operations in MLIR
    auto loc = builder_.getUnknownLoc();
    auto tensorType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(outShape.r), static_cast<int64_t>(outShape.c)},
        builder_.getF32Type()
    );
    
    // TODO: Depending on node.op->name(), emit the specific linalg op:
    // e.g., if matrix multiplication, emit linalg::MatmulOp
    // if element-wise activation (ReLU, Sigmoid), emit linalg::GenericOp
    
    nodes_.push_back(std::move(node));
    return nodes_.size() - 1;
}

void MLIRGraph::forward(){
    assert(0&&"not implemented");
}

void MLIRGraph::backward(NodeId loss){
    assert(0&&"not implemented");
}

void MLIRGraph::setInputShape(NodeId i, Shape s){
    assert(0&&"not implemented");
}

void MLIRGraph::reshape() {
    assert(0&&"not implemented");
}


std::vector<std::pair<Tensor *, Tensor *>> MLIRGraph::paramPairs() {
    std::vector<std::pair<Tensor *, Tensor *>> p;
    for (Node &n : nodes_)
      if (n.isParam) p.emplace_back(&n.value, &n.grad);
    return p;
}

void MLIRGraph::print() {
    std::printf("=== MLIR Module Dump ===\n");
    if (module_) {
        // Dumps the entire MLIR module (functions, linalg ops, tensors, etc.) to stdout
        module_->dump();
        
        // Alternatively, if you prefer using LLVM's stream operator:
        // module_->print(llvm::outs());
        // llvm::outs() << "\n";
    } else {
        std::printf("(Module is empty or not initialized)\n");
    }
    std::printf("==========================\n");
}