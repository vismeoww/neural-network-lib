#include <mlir/InitAllDialects.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>

#include <mlir/Pass/PassManager.h>
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/ExecutionEngine/OptUtils.h"

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>

#include "MLIR/MLIRGraph.hpp"
#include "MLIR/MLIROp.hpp"
namespace mlg
{

void Graph::initParam(Tensor &t, MLIRInit init)
{
    // Zero-initialize for now (adjust to your Tensor's underlying storage, e.g., t.data)
    // std::fill(t.v.begin(), t.v.end(), 0.0f);
    // (void)init; // silence unused parameter warning until He/Xavier are added
    const size_t fan_in = t.s.r, fan_out = t.s.c;
    switch (init) {
    case MLIRInit::Zero:
      t.zero();
      break;
    case MLIRInit::He: {
      std::normal_distribution<Scalar> d(0.0, std::sqrt(2.0 / fan_in));
      for (auto &x : t.v) x = d(rng_);
      break;
    }
    case MLIRInit::Xavier: {
      std::normal_distribution<Scalar> d(0.0,
                                         std::sqrt(2.0 / (fan_in + fan_out)));
      for (auto &x : t.v) x = d(rng_);
      break;
    }
    }
}

mlir::Location Graph::makeNodeLoc(NodeId id)
{
    const std::string &label = nodes_[id].label;
    std::string name = label.empty() ? ("node" + std::to_string(id)) : label;
    return mlir::NameLoc::get(builder_.getStringAttr(name));
}

Graph::Graph(unsigned int seed)
    : rng_(seed), builder_(&context_)
{
    static std::once_flag llvmInitFlag;
    std::call_once(llvmInitFlag, []()
                    {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter(); });
    // Register all upstream dialects
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();

    // Create the top-level MLIR module
    module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToStart(module_.getBody());

    // In MLIRGraph constructor (after creating module_):
    auto loc = builder_.getUnknownLoc();
    auto funcType = builder_.getFunctionType({}, {}); // signature updated later
    fwdFunc_ = mlir::func::FuncOp::create(builder_, loc, "forward", funcType);
    mlir::Block *entryBlock = fwdFunc_.addEntryBlock();
    builder_.setInsertionPointToStart(entryBlock);
}

NodeId Graph::param(Shape s, MLIRInit init, std::string label)
{
    MLIRNode node;
    node.value.resize(s);
    node.grad.resize(s);
    node.label = std::move(label);
    node.isParam = true;
    node.requiresGrad = true;

    initParam(node.value, init);
    // Also zero out grad so it doesn't hold garbage
    std::fill(node.grad.v.begin(), node.grad.v.end(), 0.0f);

    NodeId id = static_cast<NodeId>(nodes_.size());
    nodes_.push_back(std::move(node));

    // Register the parameter as an MLIR function argument immediately
    auto tensorType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(s.r), static_cast<int64_t>(s.c)},
        builder_.getF32Type());

    mlir::Block &entry = fwdFunc_.getBody().front();
    mlir::Value arg = entry.addArgument(tensorType, makeNodeLoc(id));

    mlirValues_[id] = arg;
    inputArgNodes_.push_back(id);

    return id;
}

mlir::Value Graph::getOrAddValue(NodeId id)
{
    if (auto it = mlirValues_.find(id); it != mlirValues_.end())
    {
        return it->second;
    }

    // Leaf node (input or param) not yet in MLIR -> append as a func block argument
    Shape s = nodes_[id].value.s;
    auto tensorType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(s.r), static_cast<int64_t>(s.c)},
        builder_.getF32Type());

    mlir::Block &entry = fwdFunc_.getBody().front();
    mlir::Value arg = entry.addArgument(tensorType, makeNodeLoc(id));
    mlirValues_[id] = arg;
    inputArgNodes_.push_back(id); // track arg order for ExecutionEngine later
    return arg;
}

NodeId Graph::apply(std::unique_ptr<Op> op, std::vector<NodeId> in)
{
    MLIRNode node;
    node.op = std::move(op);
    node.inputs = std::vector<NodeId>{in.begin(), in.end()};

    std::vector<Shape> inShapes;
    inShapes.reserve(in.size());
    for (auto i : in)
        inShapes.push_back(nodes_[i].value.s);
    Shape outShape = node.op->outShape(inShapes);
    node.value.resize(outShape);
    node.grad.resize(outShape);

    // 2. Gather MLIR SSA values for all inputs
    llvm::SmallVector<mlir::Value, 4> mlirInputs;
    for (NodeId id : node.inputs)
    {
        // TODO: fix so that you see names instead of %arg0
        mlirInputs.push_back(getOrAddValue(id));
    }

    // 3. Emit the op (using Option A dispatch table; for Option B use node.op->emitMLIR)
    auto loc = builder_.getUnknownLoc();
    mlir::Value result = node.op->emitMLIR(builder_, loc, mlirInputs);

    // 4. Register the node and its MLIR value
    NodeId outId = static_cast<NodeId>(nodes_.size());
    mlirValues_[outId] = result;
    nodes_.push_back(std::move(node));
    return outId;
}

void Graph::finalizeForward(std::vector<NodeId> outputNodes)
{
    outputNodes_ = std::move(outputNodes);
    auto loc = builder_.getUnknownLoc();
    // mlir::Value outVal = mlirValues_.at(outputNode);

    // 1. Emit func.return
    // mlir::func::ReturnOp::create(builder_, loc, outVal);

    // 2. Sync the FunctionType with the entry block's actual argument types
    mlir::Block &entry = fwdFunc_.getBody().front();

    //append a destination memref argument for each requrested output 
    for(NodeId outId : outputNodes_){
        Shape s = nodes_[outId].value.s;
        auto mType = mlir::MemRefType::get(
            {static_cast<int64_t>(s.r), static_cast<int64_t>(s.c)},
            builder_.getF32Type()
        );
        mlir::Value destMemRef = entry.addArgument(mType,makeNodeLoc(outId));

        auto mat = mlir::bufferization::MaterializeInDestinationOp::create(
            builder_, loc, /*resultType=*/mlir::Type(), mlirValues_.at(outId),destMemRef
        );
        mat.setWritable(true);
    }

    mlir::func::ReturnOp::create(builder_,loc);
    auto newFnType = builder_.getFunctionType(entry.getArgumentTypes(), {});
    fwdFunc_.setFunctionType(newFnType);

    // Emit C-compatible wrapper: _mlir_ciface_forward(MemRef2D* out, MemRef2D* arg0, ...)
    // fwdFunc_->setAttr(mlir::LLVM::LLVMDialect::getEmitCInterfaceAttrName(),
    //                   mlir::UnitAttr::get(&context_));
    fwdFunc_->setAttr("llvm.emit_c_interface",
                        mlir::UnitAttr::get(fwdFunc_->getContext()));

    // Lowering pipeline: tensor/linalg -> memref/scf -> cf -> llvm
    mlir::PassManager pm(&context_);

    mlir::bufferization::OneShotBufferizePassOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    // pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(module_)))
    {
        throw std::runtime_error("MLIR lowering pipeline failed");
    }

    // Create the JIT ExecutionEngine
    mlir::ExecutionEngineOptions engineOpts;
    engineOpts.transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    auto maybeEngine = mlir::ExecutionEngine::create(module_, engineOpts);
    if (!maybeEngine)
    {
        llvm::handleAllErrors(maybeEngine.takeError(), [](const llvm::ErrorInfoBase &e)
                                { throw std::runtime_error("ExecutionEngine creation failed: " + e.message()); });
    }
    engine_ = std::move(*maybeEngine);
}

void Graph::forward()
{
    if (!engine_ || outputNodes_.empty())
    {
        throw std::runtime_error("Call finalizeForward() before forward()");
    }

    // 1. Prepare descriptors for all inputs/params in exact MLIR argument order
    std::vector<MemRef2D> descs;
    descs.reserve(inputArgNodes_.size() + outputNodes_.size());
    for (NodeId id : inputArgNodes_)descs.push_back(toMemRef2D(nodes_[id].value));
    for (NodeId id : outputNodes_)descs.push_back(toMemRef2D(nodes_[id].value));

    // 2. Output descriptor (populated by _mlir_ciface_forward)
    // MemRef2D outDesc{};

    // 3. Build pointer list: each C-interface arg is a MemRef2D*
    std::vector<MemRef2D *> argPtrs;
    argPtrs.reserve(descs.size());
    // argPtrs.push_back(&outDesc);
    for (auto &desc : descs) argPtrs.push_back(&desc);

    // invokePacked expects void** pointing to each argument slot
    llvm::SmallVector<void *, 16> packedArgs;
    packedArgs.reserve(argPtrs.size());
    for (auto &ptr : argPtrs) packedArgs.push_back(&ptr);

    // 4. Execute JIT-compiled function
    auto err = engine_->invokePacked("_mlir_ciface_forward", packedArgs);
    if (err)
    {
        llvm::handleAllErrors(std::move(err), [](const llvm::ErrorInfoBase &e)
                                { throw std::runtime_error("JIT execution failed: " + e.message()); });
    }

    // 5. Copy result back into outputNode_'s Tensor and free MLIR's allocation
    // Tensor &outTensor = nodes_[outputNode_].value;
    // std::memcpy(outTensor.v.data(), outDesc.aligned, outTensor.v.size() * sizeof(float));
    // std::free(outDesc.allocated);
}

void Graph::compileBackwards(NodeId loss){
    compiledLossNode_ = loss;
    bwdParamNodes_.clear();

    // Create a fresh module for the backward pass so it can be lowered independently
    mlir::OpBuilder b(&context_);
    auto loc = b.getUnknownLoc();
    auto bwdModule = mlir::ModuleOp::create(loc);
    b.setInsertionPointToStart(bwdModule.getBody());

    auto bwdFunc = mlir::func::FuncOp::create(b, loc, "backward", b.getFunctionType({}, {}));
    mlir::Block *entry = bwdFunc.addEntryBlock();
    b.setInsertionPointToStart(entry);

    // 1. Add input & param tensor arguments (same order as inputArgNodes_)
    std::map<NodeId, mlir::Value> fwdVals;
    for (NodeId id : inputArgNodes_) {
        Shape s = nodes_[id].value.s;
        auto tType = mlir::RankedTensorType::get(
            {static_cast<int64_t>(s.r), static_cast<int64_t>(s.c)}, b.getF32Type());
        fwdVals[id] = entry->addArgument(tType, makeNodeLoc(id));
    }

    // 2. Add destination memref arguments for each parameter's gradient
    std::map<NodeId, mlir::Value> gradDestMemrefs;
    for (NodeId id = 0; id < static_cast<NodeId>(nodes_.size()); ++id) {
        if (!nodes_[id].isParam) continue;
        Shape s = nodes_[id].grad.s;
        auto mType = mlir::MemRefType::get(
            {static_cast<int64_t>(s.r), static_cast<int64_t>(s.c)}, b.getF32Type());
        gradDestMemrefs[id] = entry->addArgument(mType, makeNodeLoc(id));
        bwdParamNodes_.push_back(id);
    }

    // 3. Re-emit forward ops up to `loss` so all intermediate SSA activations are in scope
    for (NodeId id = 0; id <= loss; ++id) {
        if (!nodes_[id].op) continue;
        llvm::SmallVector<mlir::Value, 4> inVals;
        for (NodeId inId : nodes_[id].inputs) inVals.push_back(fwdVals.at(inId));
        fwdVals[id] = nodes_[id].op->emitMLIR(b, loc, inVals);
    }

    // 4. Seed dL/dLoss = ones([1, 1])
    Shape lossShape = nodes_[loss].value.s;
    auto emptySeed = mlir::tensor::EmptyOp::create(
        b, loc, llvm::ArrayRef<int64_t>{static_cast<int64_t>(lossShape.r), static_cast<int64_t>(lossShape.c)},
        b.getF32Type());
    auto one = mlir::arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1.0f));
    mlir::Value seedGrad = mlir::linalg::FillOp::create(
        b, loc, mlir::ValueRange{one}, mlir::ValueRange{emptySeed}).getResult(0);

    std::map<NodeId, mlir::Value> grads;
    grads[loss] = seedGrad;

    // 5. Reverse-mode autodiff walk (nodes_ is already in topological order)
    for (NodeId id = loss; id >= 0; --id) {
        auto it = grads.find(id);
        if (it == grads.end() || !nodes_[id].op) continue;

        llvm::SmallVector<mlir::Value, 4> inVals;
        for (NodeId inId : nodes_[id].inputs) inVals.push_back(fwdVals.at(inId));

        auto inGrads = nodes_[id].op->emitBackward(b, loc, inVals, fwdVals.at(id), it->second);
        for (size_t i = 0; i < nodes_[id].inputs.size(); ++i) {
            NodeId inId = nodes_[id].inputs[i];
            if (grads.count(inId)) {
                grads[inId] = emitAdd(b, loc, grads[inId], inGrads[i]);
            } else {
                grads[inId] = inGrads[i];
            }
        }
    }

    // 6. Materialize each parameter gradient directly into its destination memref
    for (NodeId id : bwdParamNodes_) {
        if (!grads.count(id)) continue;
        auto mat = mlir::bufferization::MaterializeInDestinationOp::create(
            b, loc, /*resultType=*/mlir::Type(), grads[id], gradDestMemrefs[id]);
        mat.setWritable(true);
    }

    mlir::func::ReturnOp::create(b, loc);
    bwdFunc.setFunctionType(b.getFunctionType(entry->getArgumentTypes(), {}));
    // bwdFunc->setAttr(mlir::LLVM::LLVMDialect::getEmitCInterfaceAttrName(),
    //                  mlir::UnitAttr::get(&context_));
    bwdFunc->setAttr("llvm.emit_c_interface",
                        mlir::UnitAttr::get(&context_));

    // 7. Lower and JIT-compile (same pipeline as finalizeForward)
    mlir::PassManager pm(&context_);
    mlir::bufferization::OneShotBufferizePassOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    // pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(bwdModule))) {
        throw std::runtime_error("Backward MLIR lowering pipeline failed");
    }

    mlir::ExecutionEngineOptions engineOpts;
    engineOpts.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
    auto maybeEngine = mlir::ExecutionEngine::create(bwdModule, engineOpts);
    if (!maybeEngine) {
        llvm::handleAllErrors(maybeEngine.takeError(), [](const llvm::ErrorInfoBase &e) {
            throw std::runtime_error("Backward ExecutionEngine failed: " + e.message());
        });
    }
    bwdEngine_ = std::move(*maybeEngine);
}

void Graph::backward(NodeId loss)
{
    if (!bwdEngine_ || compiledLossNode_ != loss) {
        compileBackwards(loss);
    }

    // Build descriptors: [inputs/params..., param_grad_buffers...]
    std::vector<MemRef2D> descs;
    descs.reserve(inputArgNodes_.size() + bwdParamNodes_.size());
    for (NodeId id : inputArgNodes_) {
        descs.push_back(toMemRef2D(nodes_[id].value));
    }
    for (NodeId id : bwdParamNodes_) {
        descs.push_back(toMemRef2D(nodes_[id].grad));
    }

    std::vector<MemRef2D *> argPtrs;
    argPtrs.reserve(descs.size());
    for (auto &d : descs) argPtrs.push_back(&d);

    llvm::SmallVector<void *, 16> packedArgs;
    packedArgs.reserve(argPtrs.size());
    for (auto &ptr : argPtrs) packedArgs.push_back(&ptr);

    auto err = bwdEngine_->invokePacked("_mlir_ciface_backward", packedArgs);
    if (err) {
        llvm::handleAllErrors(std::move(err), [](const llvm::ErrorInfoBase &e) {
            throw std::runtime_error("Backward JIT execution failed: " + e.message());
        });
    }
}

void Graph::setInputShape(NodeId i, Shape s)
{
    assert(0 && "not implemented");
}

void Graph::reshape()
{
    assert(0 && "not implemented");
}

std::vector<std::pair<Tensor *, Tensor *>> Graph::paramPairs()
{
    std::vector<std::pair<Tensor *, Tensor *>> p;
    for (MLIRNode &n : nodes_)
        if (n.isParam)
            p.emplace_back(&n.value, &n.grad);
    return p;
}

void Graph::print()
{
    std::printf("=== MLIR Module Dump ===\n");
    if (module_)
    {
        mlir::OpPrintingFlags flags;
        module_->print(llvm::errs(), flags);
    }
    else
    {
        std::printf("(Module is empty or not initialized)\n");
    }
    std::printf("==========================\n");
}

void Graph::dump()
{
    std::printf("=== MLIR Module Dump ===\n");
    if (module_)
    {
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo();
        module_->print(llvm::errs(), flags);
    }
    else
    {
        std::printf("(Module is empty or not initialized)\n");
    }
    std::printf("==========================\n");
}

NodeId linear(Graph &g, NodeId x, size_t out, MLIRInit init)
{
    const size_t in = g.shape(x).c;
    NodeId W = g.param({in, out}, init, "W");
    NodeId b = g.param({1, out}, MLIRInit::Zero, "b");
    NodeId xw = g.apply(std::make_unique<MatMul>(), {x, W});
    return g.apply(std::make_unique<AddBias>(), {xw, b});
}
NodeId relu(Graph &g, NodeId x)
{
    return g.apply(std::make_unique<Relu>(), {x});
}
// NodeId sigmoid(Graph &g, NodeId x) {
//   return g.apply(std::make_unique<Sigmoid>(), {x});
// }
// NodeId tanh_(Graph &g, NodeId x) {
//   return g.apply(std::make_unique<Tanh>(), {x});
// }
NodeId mseLoss(Graph &g, NodeId pred, NodeId target)
{
    return g.apply(std::make_unique<MSELoss>(), {pred, target});
}

}