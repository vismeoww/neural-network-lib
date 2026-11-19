#include "MLIROp.hpp"

// 1. Matrix Multiplication
mlir::Value emitMatmul(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value lhs, mlir::Value rhs)
{
    auto lhsType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = mlir::cast<mlir::RankedTensorType>(rhs.getType());

    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];

    auto outputType = mlir::RankedTensorType::get({M, N}, builder.getF32Type());

    auto emptyTensor = mlir::tensor::EmptyOp::create(builder, loc, llvm::ArrayRef<int64_t>{M, N}, builder.getF32Type());

    // Emit linalg.matmul
    auto zero = mlir::arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(0.0f));
    auto zeroTensor = mlir::linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{emptyTensor}).getResult(0);
    auto matmulOp = mlir::linalg::MatmulOp::create(builder, loc, mlir::ValueRange{lhs, rhs}, mlir::ValueRange{zeroTensor});

    return matmulOp.getResult(0);
}

// 2. Bias Addition
mlir::Value emitAddBias(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value input, mlir::Value bias)
{
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t M = inputType.getShape()[0];
    int64_t N = inputType.getShape()[1];

    auto outputType = mlir::RankedTensorType::get({M, N}, builder.getF32Type());
    auto emptyTensor = mlir::tensor::EmptyOp::create(builder,
                                                     loc, llvm::ArrayRef<int64_t>{M, N}, builder.getF32Type());

    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);

    llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()),                               // input: [M, N]
        mlir::AffineMap::get(2, 0, {builder.getAffineConstantExpr(0), d1}, builder.getContext()), // bias:  [1, N]
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext())                                // output: [M, N]
    };

    llvm::SmallVector<mlir::utils::IteratorType, 2> iteratorTypes = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel};

    auto genericOp = mlir::linalg::GenericOp::create(
        builder, loc,
        outputType,
        mlir::ValueRange{input, bias},
        mlir::ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args)
        {
            auto add = mlir::arith::AddFOp::create(b, l, args[0], args[1]);
            mlir::linalg::YieldOp::create(b, l, add.getResult());
        });

    return genericOp.getResult(0);
}

// 3. ReLU Activation
mlir::Value emitRelu(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input)
{
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto outputType = inputType;

    auto emptyTensor = mlir::tensor::EmptyOp::create(builder,
                                                     loc, inputType.getShape(), builder.getF32Type());

    auto identityMap = builder.getMultiDimIdentityMap(inputType.getRank());
    llvm::SmallVector<mlir::utils::IteratorType, 2> iteratorTypes(
        inputType.getRank(), mlir::utils::IteratorType::parallel);

    auto genericOp = mlir::linalg::GenericOp::create(
        builder,
        loc,
        outputType,
        mlir::ValueRange{input},
        mlir::ValueRange{emptyTensor},
        llvm::ArrayRef<mlir::AffineMap>{identityMap, identityMap},
        iteratorTypes,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args)
        {
            auto zero = mlir::arith::ConstantOp::create(b, l, b.getF32FloatAttr(0.0f));
            auto max = mlir::arith::MaximumFOp::create(b, l, args[0], zero);
            mlir::linalg::YieldOp::create(b, l, max.getResult());
        });

    return genericOp.getResult(0);
}