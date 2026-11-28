#include "MLIR/MLIROp.hpp"

namespace mlg{

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

mlir::Value emitMSELoss(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value pred, mlir::Value target) {
    auto predType = mlir::cast<mlir::RankedTensorType>(pred.getType());
    int64_t M = predType.getShape()[0];
    int64_t N = predType.getShape()[1];
    float invCount = 1.0f / static_cast<float>(M * N);

    // Output is a [1, 1] tensor to match your 2D Shape{r, c} convention
    auto outputType = mlir::RankedTensorType::get({1, 1}, builder.getF32Type());

    // Zero-initialize the accumulator tensor before reducing into it
    auto emptyTensor = mlir::tensor::EmptyOp::create(
        builder, loc, llvm::ArrayRef<int64_t>{1, 1}, builder.getF32Type());
    auto zero = mlir::arith::ConstantOp::create(
        builder, loc, builder.getF32FloatAttr(0.0f));
    auto zeroTensor = mlir::linalg::FillOp::create(
        builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{emptyTensor}).getResult(0);

    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto c0 = builder.getAffineConstantExpr(0);

    llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()), // pred:   [M, N]
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()), // target: [M, N]
        mlir::AffineMap::get(2, 0, {c0, c0}, builder.getContext())  // output: [1, 1]
    };

    llvm::SmallVector<mlir::utils::IteratorType, 2> iteratorTypes = {
        mlir::utils::IteratorType::reduction,
        mlir::utils::IteratorType::reduction
    };

    auto genericOp = mlir::linalg::GenericOp::create(
        builder, loc,
        outputType,
        mlir::ValueRange{pred, target},
        mlir::ValueRange{zeroTensor},
        indexingMaps,
        iteratorTypes,
        [invCount](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
            // args[0] = pred, args[1] = target, args[2] = accumulator
            auto diff = mlir::arith::SubFOp::create(b, l, args[0], args[1]);
            auto sq = mlir::arith::MulFOp::create(b, l, diff, diff);
            auto scale = mlir::arith::ConstantOp::create(b, l, b.getF32FloatAttr(invCount));
            auto scaled = mlir::arith::MulFOp::create(b, l, sq, scale);
            auto acc = mlir::arith::AddFOp::create(b, l, args[2], scaled);
            mlir::linalg::YieldOp::create(b, l, acc.getResult());
        }
    );

    return genericOp.getResult(0);
}

// Elementwise addition for gradient accumulation
mlir::Value emitAdd(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) {
    auto type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto empty = mlir::tensor::EmptyOp::create(builder, loc, type.getShape(), builder.getF32Type());

    auto idMap = builder.getMultiDimIdentityMap(type.getRank());
    llvm::SmallVector<mlir::utils::IteratorType, 2> iters(
        type.getRank(), mlir::utils::IteratorType::parallel);

    return mlir::linalg::GenericOp::create(
        builder, loc, type,
        mlir::ValueRange{lhs, rhs}, mlir::ValueRange{empty},
        llvm::ArrayRef<mlir::AffineMap>{idMap, idMap, idMap}, iters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
            auto sum = mlir::arith::AddFOp::create(b, l, args[0], args[1]);
            mlir::linalg::YieldOp::create(b, l, sum.getResult());
        }).getResult(0);
}

// 2D Transpose: [M, N] -> [N, M]
mlir::Value emitTranspose(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input) {
    auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t M = inType.getShape()[0];
    int64_t N = inType.getShape()[1];

    auto outType = mlir::RankedTensorType::get({N, M}, builder.getF32Type());
    auto empty = mlir::tensor::EmptyOp::create(builder, loc, llvm::ArrayRef<int64_t>{N, M}, builder.getF32Type());

    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    llvm::SmallVector<mlir::AffineMap, 2> maps = {
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()),
        mlir::AffineMap::get(2, 0, {d1, d0}, builder.getContext())
    };
    llvm::SmallVector<mlir::utils::IteratorType, 2> iters = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel
    };

    return mlir::linalg::GenericOp::create(
        builder, loc, outType,
        mlir::ValueRange{input}, mlir::ValueRange{empty}, maps, iters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
            mlir::linalg::YieldOp::create(b, l, args[0]);
        }).getResult(0);
}

// AddBias backward wrt bias: reduce [M, N] -> [1, N] by summing rows
mlir::Value emitAddBiasBackwardBias(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value gradOut) {
    auto inType = mlir::cast<mlir::RankedTensorType>(gradOut.getType());
    int64_t N = inType.getShape()[1];

    auto outType = mlir::RankedTensorType::get({1, N}, builder.getF32Type());
    auto empty = mlir::tensor::EmptyOp::create(builder, loc, llvm::ArrayRef<int64_t>{1, N}, builder.getF32Type());
    auto zero = mlir::arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(0.0f));
    auto zeroTensor = mlir::linalg::FillOp::create(
        builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty}).getResult(0);

    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto c0 = builder.getAffineConstantExpr(0);
    llvm::SmallVector<mlir::AffineMap, 2> maps = {
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()), // gradOut: [M, N]
        mlir::AffineMap::get(2, 0, {c0, d1}, builder.getContext())  // db:      [1, N]
    };
    llvm::SmallVector<mlir::utils::IteratorType, 2> iters = {
        mlir::utils::IteratorType::reduction,
        mlir::utils::IteratorType::parallel
    };

    return mlir::linalg::GenericOp::create(
        builder, loc, outType,
        mlir::ValueRange{gradOut}, mlir::ValueRange{zeroTensor}, maps, iters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
            auto acc = mlir::arith::AddFOp::create(b, l, args[0], args[1]);
            mlir::linalg::YieldOp::create(b, l, acc.getResult());
        }).getResult(0);
}

// ReLU backward: dX = (X > 0) ? dY : 0
mlir::Value emitReluBackward(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value fwdInput, mlir::Value gradOut) {
    auto type = mlir::cast<mlir::RankedTensorType>(fwdInput.getType());
    auto empty = mlir::tensor::EmptyOp::create(builder, loc, type.getShape(), builder.getF32Type());

    auto idMap = builder.getMultiDimIdentityMap(type.getRank());
    llvm::SmallVector<mlir::utils::IteratorType, 2> iters(
        type.getRank(), mlir::utils::IteratorType::parallel);

    return mlir::linalg::GenericOp::create(
        builder, loc, type,
        mlir::ValueRange{fwdInput, gradOut}, mlir::ValueRange{empty},
        llvm::ArrayRef<mlir::AffineMap>{idMap, idMap, idMap}, iters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
            auto zero = mlir::arith::ConstantOp::create(b, l, b.getF32FloatAttr(0.0f));
            auto gt = mlir::arith::CmpFOp::create(b, l, mlir::arith::CmpFPredicate::OGT, args[0], zero);
            auto res = mlir::arith::SelectOp::create(b, l, gt, args[1], zero);
            mlir::linalg::YieldOp::create(b, l, res.getResult());
        }).getResult(0);
}

// MSELoss backward: dPred = dL * (2 / (M*N)) * (pred - target), dTarget = -dPred
std::pair<mlir::Value, mlir::Value> emitMSELossBackward(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value pred, mlir::Value target, mlir::Value gradOut) {
    auto type = mlir::cast<mlir::RankedTensorType>(pred.getType());
    int64_t M = type.getShape()[0];
    int64_t N = type.getShape()[1];
    float scaleVal = 2.0f / static_cast<float>(M * N);

    auto emptyPred = mlir::tensor::EmptyOp::create(builder, loc, type.getShape(), builder.getF32Type());
    auto emptyTarget = mlir::tensor::EmptyOp::create(builder, loc, type.getShape(), builder.getF32Type());

    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto c0 = builder.getAffineConstantExpr(0);
    llvm::SmallVector<mlir::AffineMap, 5> maps = {
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()), // pred
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()), // target
        mlir::AffineMap::get(2, 0, {c0, c0}, builder.getContext()), // gradOut: [1, 1]
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext()), // dPred
        mlir::AffineMap::get(2, 0, {d0, d1}, builder.getContext())  // dTarget
    };
    llvm::SmallVector<mlir::utils::IteratorType, 2> iters = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel
    };

    auto genericOp = mlir::linalg::GenericOp::create(
        builder, loc, mlir::TypeRange{type, type},
        mlir::ValueRange{pred, target, gradOut},
        mlir::ValueRange{emptyPred, emptyTarget},
        maps, iters,
        [scaleVal](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
            auto diff = mlir::arith::SubFOp::create(b, l, args[0], args[1]);
            auto scale = mlir::arith::ConstantOp::create(b, l, b.getF32FloatAttr(scaleVal));
            auto dPred = mlir::arith::MulFOp::create(b, l, diff, scale);
            dPred = mlir::arith::MulFOp::create(b, l, dPred, args[2]);
            auto dTarget = mlir::arith::NegFOp::create(b, l, dPred);
            mlir::linalg::YieldOp::create(b, l, mlir::ValueRange{dPred, dTarget});
        });

    return {genericOp.getResult(0), genericOp.getResult(1)};
}

}