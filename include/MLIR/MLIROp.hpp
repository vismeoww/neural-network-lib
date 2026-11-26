#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

#include "Tensor.h"

mlir::Value emitMatmul(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value lhs, mlir::Value rhs);

mlir::Value emitAddBias(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value input, mlir::Value bias);

mlir::Value emitRelu(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input);

mlir::Value emitMSELoss(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value pred, mlir::Value target);


struct MLIROp {
  virtual ~MLIROp() = default;
  virtual const char *name() const = 0;
  virtual Shape outShape(const std::vector<Shape> &in) const = 0;
  virtual mlir::Value emitMLIR(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange inputs) const = 0;
};