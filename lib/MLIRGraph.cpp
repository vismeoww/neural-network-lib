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
    module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    // builder_.setInsertionPointToStart(module_.getBody());

    // In MLIRGraph constructor (after creating module_):
    auto loc = builder_.getUnknownLoc();
    auto funcType = builder_.getFunctionType({}, {}); // signature updated later
    fwdFunc_ = mlir::func::FuncOp::create(builder_, loc, "forward", funcType);
    mlir::Block *entryBlock = fwdFunc_.addEntryBlock();
    builder_.setInsertionPointToStart(entryBlock);
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

mlir::Value MLIRGraph::getOrAddValue(NodeId id) {
    if (auto it = mlirValues_.find(id); it != mlirValues_.end()) {
        return it->second;
    }

    // Leaf node (input or param) not yet in MLIR -> append as a func block argument
    Shape s = nodes_[id].value.s;
    auto tensorType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(s.r), static_cast<int64_t>(s.c)},
        builder_.getF32Type()
    );

    mlir::Block &entry = fwdFunc_.getBody().front();
    mlir::Value arg = entry.addArgument(tensorType, builder_.getUnknownLoc());
    mlirValues_[id] = arg;
    inputArgNodes_.push_back(id); // track arg order for ExecutionEngine later
    return arg;
}

NodeId MLIRGraph::apply(std::unique_ptr<Op> op, std::vector<NodeId> in) {
    Node node;
    node.op = std::move(op);
    node.inputs = std::vector<NodeId>{in.begin(), in.end()};
   
    std::vector<Shape> inShapes;
    inShapes.reserve(in.size());
    for(auto i : in) inShapes.push_back(nodes_[i].value.s);
    Shape outShape = node.op->outShape(inShapes);
    node.value.resize(outShape);
    node.grad.resize(outShape);

    // 2. Gather MLIR SSA values for all inputs
    llvm::SmallVector<mlir::Value, 4> mlirInputs;
    for (NodeId id : node.inputs) {
        mlirInputs.push_back(getOrAddValue(id));
    }

    // 3. Emit the op (using Option A dispatch table; for Option B use node.op->emitMLIR)
    auto loc = builder_.getUnknownLoc();
    // auto it = emitters_.find(node.op->name());
    // assert(it != emitters_.end() && "Unsupported MLIR op");
    // mlir::Value result = it->second(builder_, loc, mlirInputs);
    mlir::Value result = node.op->emitMLIR(builder_,loc,mlirInputs);

    // 4. Register the node and its MLIR value
    NodeId outId = static_cast<NodeId>(nodes_.size());
    mlirValues_[outId] = result;
    nodes_.push_back(std::move(node));
    return outId;
}

void MLIRGraph::finalizeForward(NodeId outputNode) {
    auto loc = builder_.getUnknownLoc();
    mlir::Value outVal = mlirValues_.at(outputNode);

    // 1. Emit func.return
    mlir::func::ReturnOp::create(builder_, loc, outVal);

    // 2. Sync the FunctionType with the entry block's actual argument types
    mlir::Block &entry = fwdFunc_.getBody().front();
    auto newType = builder_.getFunctionType(entry.getArgumentTypes(), {outVal.getType()});
    fwdFunc_.setFunctionType(newType);
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