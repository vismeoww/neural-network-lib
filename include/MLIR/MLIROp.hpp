#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

#include "Tensor.h"

namespace mlg{

mlir::Value emitMatmul(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value lhs, mlir::Value rhs);

mlir::Value emitAddBias(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value input, mlir::Value bias);

mlir::Value emitRelu(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input);

mlir::Value emitMSELoss(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value pred, mlir::Value target);


struct Op {
  virtual ~Op() = default;
  virtual const char *name() const = 0;
  virtual Shape outShape(const std::vector<Shape> &in) const = 0;
  virtual mlir::Value emitMLIR(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange inputs) const = 0;
};

struct MatMul: Op{
  const char* name() const override {return "Matmul";}
  Shape outShape(const std::vector<Shape> &s) const override {
    if (s.size() != 2 || s[0].c != s[1].r)
      throw std::invalid_argument("MatMul: shape mismatch");
    return {s[0].r, s[1].c};
  }
  mlir::Value emitMLIR(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange inputs) const override{
    return emitMatmul(b,loc,inputs[0],inputs[1]);
  }
};


struct AddBias : Op {
  const char *name() const override { return "AddBias"; }
  Shape outShape(const std::vector<Shape> &s) const override {
    if (s.size() != 2 || s[1].r != 1 || s[1].c != s[0].c)
      throw std::invalid_argument("AddBias: bias must be 1 x cols");
    return s[0];
  }
  mlir::Value emitMLIR(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange inputs) const override{
    return emitAddBias(b,loc,inputs[0],inputs[1]);
  }
};

// ----------------------------------------------------------------------------
// Elementwise activations
// ----------------------------------------------------------------------------
struct Relu : Op {
  const char *name() const override { return "Relu"; }
  Shape outShape(const std::vector<Shape> &s) const override { return s[0]; }
  mlir::Value emitMLIR(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange inputs) const override{
    return emitRelu(b,loc,inputs[0]);
  }
};


// ----------------------------------------------------------------------------
// L = mean((pred - target)^2) over all elements. Output is 1 x 1.
// ----------------------------------------------------------------------------
struct MSELoss : Op {
  const char *name() const override { return "MSELoss"; }
  Shape outShape(const std::vector<Shape> &s) const override {
    if (s.size() != 2 || s[0] != s[1])
      throw std::invalid_argument("MSELoss: pred/target shape mismatch");
    return {1, 1};
  }
  mlir::Value emitMLIR(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange inputs) const override{
    return emitMSELoss(b,loc,inputs[0],inputs[1]);
  }
};


}