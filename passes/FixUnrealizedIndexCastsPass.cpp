#include "stablehlo/transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct FixUnrealizedIndexCastPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle the simple 1:1 index <-> integer cases.
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value input = op.getInputs().front();
    Value output = op.getOutputs().front();

    auto srcType = input.getType();
    auto dstType = output.getType();

    auto srcIndex = srcType.isa<IndexType>();
    auto dstIndex = dstType.isa<IndexType>();
    auto srcInt = dyn_cast<IntegerType>(srcType);
    auto dstInt = dyn_cast<IntegerType>(dstType);

    // integer -> index
    if (!srcIndex && dstIndex && srcInt && !srcInt.isa<IndexType>()) {
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, dstType, input);
      return success();
    }
    // index -> integer
    if (srcIndex && !dstIndex && dstInt && !dstInt.isa<IndexType>()) {
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, dstType, input);
      return success();
    }
    return failure();
  }
};

struct FixUnrealizedIndexCastsPass
    : public PassWrapper<FixUnrealizedIndexCastsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixUnrealizedIndexCastsPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  StringRef getArgument() const final {
    return "fix-unrealized-index-casts";
  }

  StringRef getDescription() const final {
    return "Rewrite unrealized_conversion_cast iN -> index into arith.index_cast";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FixUnrealizedIndexCastPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir {
namespace stablehlo {

std::unique_ptr<Pass> createFixUnrealizedIndexCastsPass() {
  return std::make_unique<FixUnrealizedIndexCastsPass>();
}

}  // namespace stablehlo
}  // namespace mlir
