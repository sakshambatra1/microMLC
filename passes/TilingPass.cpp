#include "stablehlo/transforms/TilingPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Hard-coded tile sizes for this example
constexpr int64_t tileSizeM = 4;
constexpr int64_t tileSizeN = 4;
constexpr int64_t tileSizeK = 4;

struct TileDotPattern : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op, PatternRewriter &rewriter) const override {
    // === FIX PART 1: Recursion Guard ===
    // If we have already tiled this op (marked by our tag), skip it.
    if (op->hasAttr("done_tiling")) {
      return failure();
    }

    // DEBUG: Start
    llvm::errs() << "DEBUG: Analyzing a DotGeneralOp...\n";

    auto lhsTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsTy = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!lhsTy || !rhsTy || !resTy) {
        return failure();
    }

    // DEBUG: Rank Check
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
       llvm::errs() << "DEBUG: Failed - Not rank 2.\n";
       return rewriter.notifyMatchFailure(op, "Only supports 2D tensors for this demo");
    }

    int64_t M = lhsTy.getShape()[0];
    int64_t K = lhsTy.getShape()[1];
    int64_t N = rhsTy.getShape()[1];

    // DEBUG: Dimension Check
    if (M % tileSizeM != 0 || N % tileSizeN != 0 || K % tileSizeK != 0) {
      llvm::errs() << "DEBUG: Failed - Dimensions not divisible by tile size (4).\n";
      return rewriter.notifyMatchFailure(op, "Dimensions not divisible by tile size");
    }

    llvm::errs() << "DEBUG: MATCH SUCCESS! Rewriting into loops...\n";

    Location loc = op.getLoc();

    // 1. Create Zero Init for the whole Result Tensor
    auto elementType = resTy.getElementType();
    auto zeroAttr = rewriter.getZeroAttr(elementType);
    auto fullZeroDense = DenseElementsAttr::get(resTy, zeroAttr);
    Value initC = rewriter.create<stablehlo::ConstantOp>(loc, fullZeroDense);

    // 2. Loop M (Outer)
    Value lbM = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value ubM = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, tileSizeM);

    auto loopM = rewriter.create<scf::ForOp>(loc, lbM, ubM, stepM, ValueRange{initC});
    
    // --- Inside M Loop ---
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value iIdx = loopM.getInductionVar();
    Value cInM = loopM.getRegionIterArg(0);

    // 3. Loop N (Middle)
    Value lbN = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value ubN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, tileSizeN);

    auto loopN = rewriter.create<scf::ForOp>(loc, lbN, ubN, stepN, ValueRange{cInM});

    // --- Inside N Loop ---
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value jIdx = loopN.getInductionVar();
    Value cInN = loopN.getRegionIterArg(0);

    // 4. Loop K (Inner - Reduction)
    Value lbK = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value ubK = rewriter.create<arith::ConstantIndexOp>(loc, K);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, tileSizeK);

    // Create a Zero Tile (4x4) for the accumulator
    auto tileType = RankedTensorType::get({tileSizeM, tileSizeN}, elementType);
    auto tileZeroDense = DenseElementsAttr::get(tileType, zeroAttr);
    Value zeroTile = rewriter.create<stablehlo::ConstantOp>(loc, tileZeroDense);

    auto loopK = rewriter.create<scf::ForOp>(loc, lbK, ubK, stepK, ValueRange{zeroTile});

    // --- Inside K Loop ---
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value kIdx = loopK.getInductionVar();
    Value accTile = loopK.getRegionIterArg(0);

    // Extract Slice A [i:i+tm, k:k+tk]
    SmallVector<OpFoldResult> offsetsA = { iIdx, kIdx }; 
    SmallVector<OpFoldResult> sizesA   = { rewriter.getIndexAttr(tileSizeM), rewriter.getIndexAttr(tileSizeK) };
    SmallVector<OpFoldResult> stridesA = { rewriter.getIndexAttr(1), rewriter.getIndexAttr(1) };
    Value tileA = rewriter.create<tensor::ExtractSliceOp>(loc, op.getLhs(), offsetsA, sizesA, stridesA);

    // Extract Slice B [k:k+tk, j:j+tn]
    SmallVector<OpFoldResult> offsetsB = { kIdx, jIdx };
    SmallVector<OpFoldResult> sizesB   = { rewriter.getIndexAttr(tileSizeK), rewriter.getIndexAttr(tileSizeN) };
    SmallVector<OpFoldResult> stridesB = { rewriter.getIndexAttr(1), rewriter.getIndexAttr(1) };
    Value tileB = rewriter.create<tensor::ExtractSliceOp>(loc, op.getRhs(), offsetsB, sizesB, stridesB);

    // Small Dot Calculation
    auto dotDims = stablehlo::DotDimensionNumbersAttr::get(op.getContext(), {}, {}, {1}, {0});
    auto precision = op.getPrecisionConfigAttr(); 
    
    // Create the inner OP
    auto partialOp = rewriter.create<stablehlo::DotGeneralOp>(
          loc, tileType, tileA, tileB, dotDims, precision);

    // === FIX PART 2: Tag the new Op ===
    // We add a "done_tiling" attribute so the pattern ignores this op next time.
    partialOp->setAttr("done_tiling", rewriter.getUnitAttr());

    Value partial = partialOp.getResult();

    // Accumulate
    Value nextAcc = rewriter.create<stablehlo::AddOp>(loc, accTile, partial);
    rewriter.create<scf::YieldOp>(loc, nextAcc);

    // --- After K Loop (Back in N Loop) ---
    rewriter.setInsertionPointAfter(loopK);
    Value resultingTile = loopK.getResult(0);

    // Insert tile back into C
    SmallVector<OpFoldResult> offsetsC = { iIdx, jIdx };
    SmallVector<OpFoldResult> sizesC   = { rewriter.getIndexAttr(tileSizeM), rewriter.getIndexAttr(tileSizeN) };
    SmallVector<OpFoldResult> stridesC = { rewriter.getIndexAttr(1), rewriter.getIndexAttr(1) };

    Value updatedC = rewriter.create<tensor::InsertSliceOp>(
          loc, resultingTile, cInN, offsetsC, sizesC, stridesC);

    rewriter.create<scf::YieldOp>(loc, updatedC);

    // --- After N Loop (Back in M Loop) ---
    rewriter.setInsertionPointAfter(loopN);
    rewriter.create<scf::YieldOp>(loc, loopN.getResult(0));

    // --- Final Replacement ---
    rewriter.replaceOp(op, loopM.getResult(0));

    return success();
  }
};

struct TilingPass : public PassWrapper<TilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass)

  StringRef getArgument() const final { return "stablehlo-tiling"; }
  StringRef getDescription() const final {
    return "Example tiling pass: tiles StableHLO DotGeneral into a 3D loop nest.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    tensor::TensorDialect,
                    scf::SCFDialect,
                    stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TileDotPattern>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

// Registration
namespace mlir {
namespace stablehlo {

void registerTilingPasses() {
  PassRegistration<TilingPass>();
}

} // namespace stablehlo
} // namespace mlir