#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h" 
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "stablehlo-linalg-tiling"

using namespace mlir;

namespace {

// =============================================================================
// Helper: Profitability & Validation
// =============================================================================
static LogicalResult checkProfitability(linalg::LinalgOp op, 
                                        ArrayRef<int64_t> tileSizes, 
                                        int64_t minShapeSize, 
                                        PatternRewriter &rewriter) {
  
  unsigned numLoops = op.getNumLoops();
  
  // This check is now safe because we align sizes before calling this
  if (tileSizes.size() != numLoops) {
    return rewriter.notifyMatchFailure(op, "Tile sizes count mismatch (internal error)");
  }

  SmallVector<Range> ranges = op.createLoopRanges(rewriter, op.getLoc());
  
  llvm::errs() << "\n[DEBUG] Checking Profitability for Op: " << op->getName() << "\n";

  for (size_t i = 0; i < numLoops; ++i) {
    OpFoldResult sizeOfr = ranges[i].size;
    std::optional<int64_t> staticSize = getConstantIntValue(sizeOfr);
    int64_t tileSize = tileSizes[i];

    if (tileSize == 0) {
        // Tile size 0 means "skip this dimension"
        continue;
    }

    if (staticSize.has_value()) {
        llvm::errs() << "  Dim " << i << ": Static Size = " << *staticSize 
                     << ", Tile Size = " << tileSize << "\n";
        
        int64_t dim = *staticSize;
        if (dim < minShapeSize) {
           llvm::errs() << "    -> FAILED: Too small (< " << minShapeSize << ")\n";
           return failure();
        }
        
        // [ROBUST FIX] Removed strict divisibility check.
        // We now allow partial tiles (e.g., Dim 32, Tile 128).
        if (dim % tileSize != 0) {
           llvm::errs() << "    -> NOTE: Partial tile detected (" << dim << " % " << tileSize << " != 0). Allowing.\n";
        }
    } else {
        llvm::errs() << "  Dim " << i << ": Dynamic Size (Unknown) -> Allowing\n";
    }
  }

  llvm::errs() << "  -> SUCCESS: Profitability Check Passed\n";
  return success();
}

// =============================================================================
// The Pattern: Multi-Level Tiling (Generic for Matmul & BatchMatmul)
// =============================================================================
struct MultiLevelTilePattern : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  
  SmallVector<int64_t> l2TileSizes;
  SmallVector<int64_t> l1TileSizes;
  int64_t minShapeSize;

  MultiLevelTilePattern(MLIRContext *context, 
                        ArrayRef<int64_t> l2Sizes, 
                        ArrayRef<int64_t> l1Sizes, 
                        int64_t minSize)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context),
        l2TileSizes(l2Sizes.begin(), l2Sizes.end()),
        l1TileSizes(l1Sizes.begin(), l1Sizes.end()),
        minShapeSize(minSize) {}

  // --- Helper: Align user tile sizes to operation loop count ---
  // If user provides fewer sizes than loops, we pad with 0s on the LEFT (Batch dims).
  SmallVector<int64_t> alignSizes(linalg::LinalgOp op, ArrayRef<int64_t> userSizes) const {
      unsigned numLoops = op.getNumLoops();
      SmallVector<int64_t> aligned;
      
      if (userSizes.size() < numLoops) {
          // Case: Op is 4D (BatchMatmul), User provided 3 sizes.
          // Fill outer dims with 0.
          unsigned missing = numLoops - userSizes.size();
          for(unsigned i=0; i<missing; ++i) aligned.push_back(0);
          aligned.append(userSizes.begin(), userSizes.end());
      } else if (userSizes.size() > numLoops) {
          // Case: Op is 3D, User provided 4 sizes. Take the trailing N.
          auto start = userSizes.size() - numLoops;
          aligned.append(userSizes.begin() + start, userSizes.end());
      } else {
          // Perfect match
          aligned.append(userSizes.begin(), userSizes.end());
      }
      return aligned;
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp op, PatternRewriter &rewriter) const override {
    
    // Only target Matmul or BatchMatmul
    if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op.getOperation())) {
        return failure();
    }

    if (op->hasAttr("done_tiling")) return failure();
    
    // Prevent re-tiling loops that we just created
    if (op->getParentOfType<scf::ForOp>()) return failure(); 

    // Align the tile sizes to this specific op's rank
    SmallVector<int64_t> alignedL2 = alignSizes(op, l2TileSizes);
    SmallVector<int64_t> alignedL1 = alignSizes(op, l1TileSizes);

    // --- L2 Tiling ---
    llvm::errs() << ">>> Attempting L2 Tiling on " << op->getName() << " (Rank " << op.getNumLoops() << ")...\n";
    if (failed(checkProfitability(op, alignedL2, minShapeSize, rewriter))) {
      return failure();
    }

    auto optionsL2 = linalg::LinalgTilingOptions().setTileSizes(alignedL2);
    FailureOr<linalg::TiledLinalgOp> resultL2 = linalg::tileLinalgOp(rewriter, op, optionsL2);
    
    if (failed(resultL2)) return failure();

    // Mark L2 loops as done
    for (Operation *loop : resultL2->loops) {
        loop->setAttr("done_tiling", rewriter.getUnitAttr());
    }

    // --- L1 Tiling ---
    linalg::LinalgOp innerOp = resultL2->op;
    FailureOr<linalg::TiledLinalgOp> resultL1; 

    if (!l1TileSizes.empty() && innerOp) {
      
      llvm::errs() << ">>> Attempting L1 Tiling on Inner Op...\n";
      
      if (succeeded(checkProfitability(innerOp, alignedL1, 0, rewriter))) {
        
        auto optionsL1 = linalg::LinalgTilingOptions().setTileSizes(alignedL1);
        resultL1 = linalg::tileLinalgOp(rewriter, innerOp, optionsL1);

        if (succeeded(resultL1)) {
          llvm::errs() << ">>> L1 Tiling SUCCESS!\n";
          
          for (Operation *loop : resultL1->loops) {
              loop->setAttr("done_tiling", rewriter.getUnitAttr());
          }

          // Replace the intermediate op with the tiled result
          rewriter.replaceOp(innerOp, resultL1->tensorResults);
          innerOp = resultL1->op;
        }
      }
    }

    // --- Finalize ---
    if (innerOp) {
      innerOp->setAttr("done_tiling", rewriter.getUnitAttr());
    } else if (resultL2->op) {
      resultL2->op->setAttr("done_tiling", rewriter.getUnitAttr());
    }

    rewriter.replaceOp(op, resultL2->tensorResults);

    return success();
  }
};

// =============================================================================
// The Pass
// =============================================================================
struct LinalgTilingPass : public PassWrapper<LinalgTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgTilingPass)

  ListOption<int64_t> l2TileSizesOption{
      *this, "l2-tile-sizes",
      llvm::cl::desc("L2 Tile sizes (Outer loops)"),
      llvm::cl::ZeroOrMore};

  ListOption<int64_t> l1TileSizesOption{
      *this, "l1-tile-sizes",
      llvm::cl::desc("L1 Tile sizes (Inner loops)"),
      llvm::cl::ZeroOrMore};

  Option<int64_t> minShapeSizeOption{
      *this, "min-shape-size",
      llvm::cl::desc("Minimum shape size required to trigger tiling"),
      llvm::cl::init(0)};

  LinalgTilingPass() = default;
  LinalgTilingPass(const LinalgTilingPass &pass) : PassWrapper(pass) {
    l2TileSizesOption = pass.l2TileSizesOption;
    l1TileSizesOption = pass.l1TileSizesOption;
    minShapeSizeOption = pass.minShapeSizeOption;
  }

  StringRef getArgument() const final { return "stablehlo-linalg-tiling"; }
  StringRef getDescription() const final { return "Multi-level tiling for Linalg Matmul."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Default fallback sizes if none provided in CLI
    SmallVector<int64_t> l2Sizes(l2TileSizesOption.begin(), l2TileSizesOption.end());
    if (l2Sizes.empty()) l2Sizes = {1, 128, 128}; 

    SmallVector<int64_t> l1Sizes(l1TileSizesOption.begin(), l1TileSizesOption.end());
    if (l1Sizes.empty()) l1Sizes = {0, 32, 32};    

    patterns.add<MultiLevelTilePattern>(context, l2Sizes, l1Sizes, minShapeSizeOption);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespacea

namespace mlir {
namespace stablehlo {
void registerLinalgTilingPass() { PassRegistration<LinalgTilingPass>(); }
} 
}