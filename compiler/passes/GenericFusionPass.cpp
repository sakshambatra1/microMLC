#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h" 

using namespace mlir;
using namespace mlir::linalg;

namespace {

// ============================================================================
// Helper: Check for Reduction Iterators
// ============================================================================
static bool hasReductionIterator(LinalgOp op) {
    for (auto iter : op.getIteratorTypesArray()) {
        if (linalg::isReductionIterator(iter)) return true;
    }
    return false;
}

// ============================================================================
// Core Fusion Logic
// ============================================================================
static LogicalResult fuseProducerEpilogue(LinalgOp producer,
                                          GenericOp consumer,
                                          OpOperand &fusedOperand,
                                          PatternRewriter &rewriter) {
  
  // 1. Basic Validation
  if (producer->getNumRegions() == 0 || producer->getRegion(0).empty()) {
      return failure();
  }

  OpOperand *consOpOperand = &fusedOperand;
  unsigned consumerInputIdx = consOpOperand->getOperandNumber();

  // 2. Identify Producer Result
  OpResult producerResult = dyn_cast<OpResult>(fusedOperand.get());
  if (!producerResult || producerResult.getOwner() != producer.getOperation()) {
      return failure();
  }
  unsigned prodResultIdx = producerResult.getResultNumber();

  // 3. Map Validation (Strict Safety Check)
  SmallVector<AffineMap> prodMaps = producer.getIndexingMapsArray();
  SmallVector<AffineMap> consMaps = consumer.getIndexingMapsArray();

  // Validate bounds
  if (prodMaps.size() <= producer.getNumDpsInputs() + prodResultIdx) return failure();
  if (consMaps.size() <= consumerInputIdx) return failure();

  AffineMap producerResultMap = prodMaps[producer.getNumDpsInputs() + prodResultIdx];
  AffineMap consumerInputMap = consMaps[consumerInputIdx];

  // CRITICAL FIX: Only fuse if the producer writes to the tensor in the EXACT same pattern
  // that the consumer reads it. This avoids complex map composition math that causes crashes.
  if (producerResultMap != consumerInputMap) {
      return failure();
  }

  // 4. Construct Fused Maps
  SmallVector<AffineMap> fusedMaps;

  // A. Producer Input Maps (They take the place of the consumer input)
  for (unsigned i = 0; i < producer.getNumDpsInputs(); ++i) {
    fusedMaps.push_back(prodMaps[i]);
  }

  // B. Consumer Input Maps (Copy existing, skip the fused one)
  for (unsigned i = 0; i < consumer.getNumDpsInputs(); ++i) {
    if (i == consumerInputIdx) continue;
    fusedMaps.push_back(consMaps[i]);
  }

  // C. Consumer Output Maps (Copy existing)
  for (unsigned i = 0; i < consumer.getNumDpsInits(); ++i) {
    unsigned mapIdx = consumer.getNumDpsInputs() + i;
    fusedMaps.push_back(consMaps[mapIdx]);
  }

  // 5. Iterator Types (Must match)
  if (producer.getIteratorTypesArray() != consumer.getIteratorTypesArray()) {
      return failure();
  }
  SmallVector<utils::IteratorType> fusedIterators = consumer.getIteratorTypesArray();

  // 6. Collect Operands
  SmallVector<Value> fusedInputs;
  SmallVector<Value> fusedOutputs;

  for (OpOperand *opOperand : producer.getDpsInputOperands())
    fusedInputs.push_back(opOperand->get());

  for (OpOperand *opOperand : consumer.getDpsInputOperands()) {
    if (opOperand == &fusedOperand) continue;
    fusedInputs.push_back(opOperand->get());
  }

  for (OpOperand *opOperand : consumer.getDpsInitOperands())
    fusedOutputs.push_back(opOperand->get());

  // 7. Create Fused Op
  rewriter.setInsertionPoint(consumer);
  auto fusedOp = rewriter.create<GenericOp>(
      consumer.getLoc(), consumer->getResultTypes(),
      fusedInputs, fusedOutputs, fusedMaps,
      fusedIterators);

  // 8. Region Merging
  Block &fusedBlock = fusedOp.getRegion().emplaceBlock();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&fusedBlock);

  SmallVector<BlockArgument> newArgs;
  for (Value v : fusedInputs)
    newArgs.push_back(fusedBlock.addArgument(cast<TensorType>(v.getType()).getElementType(), consumer.getLoc()));
  for (Value v : fusedOutputs)
    newArgs.push_back(fusedBlock.addArgument(cast<TensorType>(v.getType()).getElementType(), consumer.getLoc()));

  IRMapping mapper;
  Block &prodBlock = producer->getRegion(0).front();
  
  // Map Producer Inputs
  for (unsigned i = 0; i < producer.getNumDpsInputs(); ++i) {
      if (i >= prodBlock.getNumArguments()) return failure();
      mapper.map(prodBlock.getArgument(i), newArgs[i]);
  }

  // CRITICAL FIX: Check for usage of Producer Output Args (Accumulators)
  // If the producer body uses its output argument (e.g. for reduction), we cannot safely fuse
  // without mapping it. Since we filtered reductions, this should be rare, but we must check.
  unsigned prodNumInputs = producer.getNumDpsInputs();
  for (unsigned i = 0; i < producer.getNumDpsInits(); ++i) {
      BlockArgument outArg = prodBlock.getArgument(prodNumInputs + i);
      if (!outArg.use_empty()) {
          // If the body uses the output accumulator, abort fusion to prevent crash.
          return failure();
      }
  }

  // Clone Producer Body
  SmallVector<Value> prodYieldValues;
  for (Operation &op : prodBlock.getOperations()) {
    if (auto yieldOp = dyn_cast<linalg::YieldOp>(op)) {
      for (Value v : yieldOp.getOperands()) prodYieldValues.push_back(mapper.lookup(v));
      continue;
    }
    rewriter.clone(op, mapper);
  }

  if (prodYieldValues.size() <= prodResultIdx) return failure();
  Value prodResultInBlock = prodYieldValues[prodResultIdx];

  // Map Consumer Block Arguments
  Block &consBlock = consumer.getRegion().front();
  unsigned nextInputIdx = producer.getNumDpsInputs();

  for (unsigned i = 0; i < consumer.getNumDpsInputs(); ++i) {
    if (i == consumerInputIdx) {
      mapper.map(consBlock.getArgument(i), prodResultInBlock); 
    } else {
      mapper.map(consBlock.getArgument(i), newArgs[nextInputIdx++]);
    }
  }

  unsigned outputStartIdx = producer.getNumDpsInputs() + (consumer.getNumDpsInputs() - 1);
  for (unsigned i = 0; i < consumer.getNumDpsInits(); ++i) {
    mapper.map(consBlock.getArgument(consumer.getNumDpsInputs() + i), newArgs[outputStartIdx + i]);
  }

  // Clone Consumer Body
  for (Operation &op : consBlock.getOperations()) {
    if (isa<linalg::YieldOp>(op)) continue;
    rewriter.clone(op, mapper);
  }

  auto consYield = cast<linalg::YieldOp>(consBlock.getTerminator());
  SmallVector<Value> yieldedVals;
  for (Value v : consYield.getOperands())
    yieldedVals.push_back(mapper.lookup(v));
  
  rewriter.create<linalg::YieldOp>(consYield.getLoc(), yieldedVals);

  // CRITICAL FIX: Print BEFORE invalidating the consumer op
  llvm::errs() << "[Fusion Debug] SUCCESS: Fused " << producer->getName().getStringRef() 
               << " into " << consumer->getName().getStringRef() << "\n";

  rewriter.replaceOp(consumer, fusedOp.getResults());
  
  return success();
}

struct GenericFusionPattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    
    if (!isElementwise(consumer)) return failure();

    for (OpOperand *input : consumer.getDpsInputOperands()) {
      Operation *defOp = input->get().getDefiningOp();
      if (!defOp) continue;

      auto producer = dyn_cast<LinalgOp>(defOp);
      if (!producer) continue;

      // Allow elementwise producers or contraction (if needed)
      // For safety, let's stick to elementwise fusion first
      if (!isElementwise(producer)) continue;

      if (hasReductionIterator(producer)) continue;
      if (producer->getNumRegions() == 0 || producer->getRegion(0).empty()) continue;

      if (succeeded(fuseProducerEpilogue(producer, consumer, *input, rewriter)))
        return success();
    }
    return failure();
  }
};

struct GenericFusionPass
    : public PassWrapper<GenericFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericFusionPass)

  StringRef getArgument() const final { return "generic-fusion"; }
  StringRef getDescription() const final { return "Fuses linalg producers into elementwise consumers"; }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<GenericFusionPattern>(ctx);
    
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 10; 
    
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config);
  }
};

} // namespace

namespace mlir {
namespace stablehlo {
    void registerGenericFusionPass() {
        PassRegistration<GenericFusionPass>();
    }
} // namespace stablehlo
} // namespace mlir