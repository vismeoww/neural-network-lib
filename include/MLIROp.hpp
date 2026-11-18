#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

mlir::Value emitMatmul(mlir::OpBuilder &builder, mlir::Location loc, 
                       mlir::Value lhs, mlir::Value rhs); 

mlir::Value emitAddBias(mlir::OpBuilder &builder, mlir::Location loc, 
                        mlir::Value input, mlir::Value bias); 

mlir::Value emitRelu(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input); 