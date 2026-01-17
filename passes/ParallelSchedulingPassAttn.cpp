#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "parallel-schedule"

using namespace mlir;
using namespace mlir::linalg;

namespace {

// =========================================================
// HELPER: Conversion to scf.parallel
// =========================================================

static std::optional<int64_t> getConstTripCount(scf::ForOp forOp) {
  auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !ub || !step) return std::nullopt;

  int64_t stepVal = step.value();
  if (stepVal == 0) return std::nullopt;

  int64_t span = ub.value() - lb.value();
  if (span < 0) return std::nullopt;

  // ceildiv(span, step)
  return (span + stepVal - 1) / stepVal;
}

// Conservative disjointness: IV must appear in exactly one subview offset dim,
// and that slice size must equal the step so parallel iterations do not overlap.
static bool isDisjointSubviewForIV(Value memref, Value iv, int64_t stepVal) {
  // Trace through view-like ops; count exactly one IV in subview offsets and
  // require the tile size for that dim to match the loop step.
  Value cur = memref;
  int ivCount = 0;
  while (cur) {
    if (auto sub = cur.getDefiningOp<memref::SubViewOp>()) {
      auto offsets = sub.getMixedOffsets();
      auto sizes = sub.getMixedSizes();
      for (auto it : llvm::enumerate(offsets)) {
        if (auto v = it.value().dyn_cast<Value>()) {
          if (v == iv) {
            ivCount++;
            auto sz = sizes[it.index()];
            if (auto szAttr = sz.dyn_cast<Attribute>()) {
              if (auto intAttr = szAttr.dyn_cast<IntegerAttr>()) {
                if (intAttr.getInt() != stepVal) return false;
              }
            }
          }
        }
      }
      cur = sub.getSource();
      continue;
    }
    if (auto view = cur.getDefiningOp<memref::ViewOp>()) {
      cur = view.getSource();
      continue;
    }
    if (auto cast = cur.getDefiningOp<memref::ReinterpretCastOp>()) {
      cur = cast.getSource();
      continue;
    }
    if (auto c = cur.getDefiningOp<memref::CollapseShapeOp>()) {
      cur = c.getSrc();
      continue;
    }
    if (auto e = cur.getDefiningOp<memref::ExpandShapeOp>()) {
      cur = e.getSrc();
      continue;
    }
    break;
  }
  return ivCount == 1;
}

static bool ivAppearsInSubview(Value memref, Value iv) {
  Value cur = memref;
  while (cur) {
    if (auto sub = cur.getDefiningOp<memref::SubViewOp>()) {
      for (auto of : sub.getMixedOffsets())
        if (auto v = of.dyn_cast<Value>())
          if (v == iv) return true;
      cur = sub.getSource();
      continue;
    }
    if (auto view = cur.getDefiningOp<memref::ViewOp>()) {
      cur = view.getSource();
      continue;
    }
    if (auto cast = cur.getDefiningOp<memref::ReinterpretCastOp>()) {
      cur = cast.getSource();
      continue;
    }
    if (auto c = cur.getDefiningOp<memref::CollapseShapeOp>()) {
      cur = c.getSrc();
      continue;
    }
    if (auto e = cur.getDefiningOp<memref::ExpandShapeOp>()) {
      cur = e.getSrc();
      continue;
    }
    break;
  }
  return false;
}

static scf::ParallelOp convertForToParallel(IRRewriter &rewriter,
                                            scf::ForOp forOp) {
  rewriter.setInsertionPoint(forOp);

  // create scf.parallel
  auto par =
      rewriter.create<scf::ParallelOp>(forOp.getLoc(),
                                       /*lowerBounds=*/ValueRange{
                                           forOp.getLowerBound()},
                                       /*upperBounds=*/ValueRange{
                                           forOp.getUpperBound()},
                                       /*steps=*/ValueRange{forOp.getStep()});

  Block &oldBody = *forOp.getBody();
  Block &newBody = par.getRegion().front();

  // Remove the default terminator inserted by scf.parallel builder.
  if (!newBody.empty())
    rewriter.eraseOp(newBody.getTerminator());

  IRMapping mapping;

  // Use zip to map all arguments (IV is arg 0) safely
  for (auto [oldArg, newArg] :
       llvm::zip(oldBody.getArguments(), newBody.getArguments())) {
    mapping.map(oldArg, newArg);
  }

  // Clone body
  rewriter.setInsertionPointToStart(&newBody);
  for (Operation &op : oldBody.without_terminator()) {
    rewriter.clone(op, mapping);
  }

  rewriter.create<scf::YieldOp>(forOp.getLoc());
  rewriter.eraseOp(forOp);
  return par;
}

// ---------------------------------------------------------------------------
// SCF safety checks
// ---------------------------------------------------------------------------

static bool isParallelizableTileFor(scf::ForOp forOp) {
  if (forOp.getNumRegionIterArgs() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "Skip scf.for: has iter args\n");
    return false;
  }
  auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !ub || !step) {
    LLVM_DEBUG(llvm::dbgs() << "Skip scf.for: non-constant bounds/step\n");
    return false;
  }
  if (step.value() <= 0) {
    LLVM_DEBUG(llvm::dbgs() << "Skip scf.for: non-positive step\n");
    return false;
  }
  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || !yield.getOperands().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Skip scf.for: yield has operands (reduction)\n");
    return false;
  }
  return true;
}

static bool isParallelizableTileFor(scf::ParallelOp parOp) {
  // Helper for collapsing parallel->for: we only collapse simple parallel
  // loops without reductions.
  if (!parOp.getInitVals().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Skip collapse: scf.parallel has init vals\n");
    return false;
  }
  return true;
}

static bool isSafeToParallelizeSCF(scf::ForOp forOp) {
  if (!isParallelizableTileFor(forOp)) return false;

  bool ok = true;
  forOp.walk([&](Operation *op) {
    if (op == forOp.getOperation()) return WalkResult::advance();
    if (isa<scf::ForOp, scf::ParallelOp>(op))
      return WalkResult::advance(); // handled separately

    // Allow common non-side-effecting ops by dialect or type.
    if (isa<arith::ArithDialect>(op->getDialect()) ||
        isa<math::MathDialect>(op->getDialect()))
      return WalkResult::advance();
    if (isa<memref::LoadOp, memref::SubViewOp, memref::ViewOp,
            memref::ReinterpretCastOp, memref::CollapseShapeOp,
            memref::ExpandShapeOp, memref::AllocOp, memref::DeallocOp,
            vector::TransferReadOp, vector::TransferWriteOp,
            vector::ContractionOp>(op))
      return WalkResult::advance();

    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      // Ignore self-copies introduced by lowering; reject other copies.
      if (copy.getSource() == copy.getTarget())
        return WalkResult::advance();
      LLVM_DEBUG(llvm::dbgs() << "Skip scf.for: non-trivial memref.copy\n");
      ok = false;
      return WalkResult::interrupt();
    }

    if (op->getName().getStringRef() == "vector.multi_reduction")
      return WalkResult::advance();

    if (isa<func::CallOp, func::CallIndirectOp>(op) ||
        isa<memref::AtomicRMWOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Skip scf.for: found call/atomic " << *op
                              << "\n");
      ok = false;
      return WalkResult::interrupt();
    }

    if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffects(effects);
      bool hasWrite = llvm::any_of(effects, [](const auto &eff) {
        return isa<MemoryEffects::Write>(eff.getEffect());
      });
      if (!hasWrite)
        return WalkResult::advance();
      if (isa<memref::StoreOp>(op))
        return WalkResult::advance();
      LLVM_DEBUG(llvm::dbgs()
                 << "Skip scf.for: side-effecting op not allowed " << *op
                 << "\n");
      ok = false;
      return WalkResult::interrupt();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Skip scf.for: unknown op not allowed " << *op << "\n");
    ok = false;
    return WalkResult::interrupt();
  });

  return ok;
}

static bool hasDisjointWritesHeuristic(scf::ForOp forOp) {
  auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  int64_t stepVal = stepCst ? stepCst.value() : 0;
  if (stepVal <= 0) return false;

  Value iv = forOp.getInductionVar();
  bool sawWrite = false;
  bool sawIVWrite = false;

  WalkResult res = forOp.walk([&](Operation *op) {
    if (op == forOp.getOperation()) return WalkResult::advance();

    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      sawWrite = true;
      if (!isDisjointSubviewForIV(store.getMemref(), iv, stepVal)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip scf.for: store not disjoint for IV\n");
        return WalkResult::interrupt();
      }
      sawIVWrite = true;
      return WalkResult::advance();
    }
    if (auto tw = dyn_cast<vector::TransferWriteOp>(op)) {
      // Destination memref is operand #1 (vector is operand #0).
      Value target = tw->getNumOperands() > 1 ? tw->getOperand(1) : Value();
      if (!target) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip scf.for: transfer_write missing memref operand\n");
        return WalkResult::interrupt();
      }
      sawWrite = true;
      if (!isDisjointSubviewForIV(target, iv, stepVal)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip scf.for: transfer_write not disjoint for IV\n");
        return WalkResult::interrupt();
      }
      sawIVWrite = true;
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) return false;
  if (!sawWrite || !sawIVWrite) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skip scf.for: no IV-dependent writes detected\n");
    return false;
  }
  return true;
}

// Try privatizing writes and reducing back; only supports transfer_write
// targets with static shapes and no memref.store side effects.
static std::optional<scf::ParallelOp> privatizeAndParallelize(IRRewriter &rewriter,
                                                              scf::ForOp forOp) {
  // Gather transfer_writes and ensure a single destination shape.
  SmallVector<vector::TransferWriteOp> writes;
  forOp.walk([&](vector::TransferWriteOp tw) { writes.push_back(tw); });
  if (writes.empty()) return std::nullopt;

  // Require all writes target the same memref value.
  Value firstTarget = writes.front()->getOperand(1);
  for (auto tw : writes) {
    if (tw->getOperand(1) != firstTarget) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Privatize: multiple transfer_write destinations, bail\n");
      return std::nullopt;
    }
  }

  auto targetType = firstTarget.getType().dyn_cast<MemRefType>();
  if (!targetType || !targetType.hasStaticShape()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Privatize: non-static transfer_write destination, bail\n");
    return std::nullopt;
  }
  MemRefType privType =
      MemRefType::get(targetType.getShape(), targetType.getElementType());

  // Build the parallel skeleton.
  rewriter.setInsertionPoint(forOp);
  auto par = rewriter.create<scf::ParallelOp>(
      forOp.getLoc(), ValueRange{forOp.getLowerBound()},
      ValueRange{forOp.getUpperBound()}, ValueRange{forOp.getStep()});
  Block &newBody = par.getRegion().front();
  if (!newBody.empty()) rewriter.eraseOp(newBody.getTerminator());

  IRMapping mapping;
  for (auto [oldArg, newArg] :
       llvm::zip(forOp.getBody()->getArguments(), newBody.getArguments()))
    mapping.map(oldArg, newArg);

  rewriter.setInsertionPointToStart(&newBody);

  // Private tile alloc + zero fill.
  auto privAlloc =
      rewriter.create<memref::AllocOp>(forOp.getLoc(), privType);
  Value zeroVal;
  Type elemTy = targetType.getElementType();
  if (auto ft = elemTy.dyn_cast<FloatType>())
    zeroVal = rewriter.create<arith::ConstantOp>(
        forOp.getLoc(), elemTy, rewriter.getFloatAttr(ft, 0.0));
  else if (auto it = elemTy.dyn_cast<IntegerType>())
    zeroVal = rewriter.create<arith::ConstantOp>(
        forOp.getLoc(), elemTy, rewriter.getIntegerAttr(elemTy, 0));
  else
    return std::nullopt;
  rewriter.create<linalg::FillOp>(forOp.getLoc(), zeroVal,
                                  ValueRange{privAlloc});

  // Clone body, redirect transfer_writes to private alloc.
  Value destForReduction;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto tw = dyn_cast<vector::TransferWriteOp>(&op)) {
      SmallVector<Value> idx;
      for (Value iv : tw.getIndices())
        idx.push_back(mapping.lookupOrDefault(iv));
      auto vecVal = mapping.lookupOrDefault(tw.getVector());
      auto newTW = rewriter.create<vector::TransferWriteOp>(
          tw.getLoc(), vecVal, privAlloc, idx, tw.getPermutationMapAttr(),
          tw.getInBoundsAttr());
      if (!destForReduction)
        destForReduction = mapping.lookupOrDefault(tw->getOperand(1));
      (void)newTW;
      continue;
    }
    Operation *cloned = rewriter.clone(op, mapping);
    (void)cloned;
  }

  if (!destForReduction) {
    LLVM_DEBUG(llvm::dbgs()
               << "Privatize: no destination for reduction, bail\n");
    rewriter.eraseOp(par);
    return std::nullopt;
  }

  // Reduce private tile into the original destination.
  SmallVector<int64_t> shape(privType.getShape().begin(),
                             privType.getShape().end());
  auto vecType = VectorType::get(shape, elemTy);
  SmallVector<Value> zeros(privType.getRank());
  for (int i = 0, e = privType.getRank(); i < e; ++i)
    zeros[i] = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);

  auto idMap = AffineMap::getMultiDimIdentityMap(privType.getRank(),
                                                 rewriter.getContext());
  auto idMapAttr = AffineMapAttr::get(idMap);
  auto emptyArrayAttr = ArrayAttr();
  auto privVec = rewriter.create<vector::TransferReadOp>(
      forOp.getLoc(), vecType, privAlloc, zeros, idMapAttr, emptyArrayAttr);
  auto dstVec = rewriter.create<vector::TransferReadOp>(
      forOp.getLoc(), vecType, destForReduction, zeros, idMapAttr,
      emptyArrayAttr);

  Value sum;
  if (elemTy.isa<FloatType>())
    sum = rewriter.create<arith::AddFOp>(forOp.getLoc(), privVec, dstVec)
              .getResult();
  else
    sum = rewriter.create<arith::AddIOp>(forOp.getLoc(), privVec, dstVec)
              .getResult();

  rewriter.create<vector::TransferWriteOp>(forOp.getLoc(), sum,
                                           destForReduction, zeros, idMapAttr,
                                           emptyArrayAttr);

  rewriter.create<scf::YieldOp>(forOp.getLoc());
  rewriter.eraseOp(forOp);
  LLVM_DEBUG(llvm::dbgs() << "Privatized + parallelized loop with reduction\n");
  return par;
}

// Collapse a pattern scf.parallel -> scf.for into a single multi-IV parallel.
static bool canCollapseParallelFor(scf::ParallelOp par, scf::ForOp inner) {
  Block &body = par.getRegion().front();
  if (body.getOperations().size() != 2) return false;
  Operation &first = body.front();
  Operation &second = *std::next(body.begin());
  return &first == inner.getOperation() && isa<scf::YieldOp>(second);
}

static scf::ParallelOp collapseParallelFor(IRRewriter &rewriter,
                                           scf::ParallelOp par,
                                           scf::ForOp inner) {
  SmallVector<Value> lbs, ubs, steps;
  lbs.append(par.getLowerBound().begin(), par.getLowerBound().end());
  ubs.append(par.getUpperBound().begin(), par.getUpperBound().end());
  steps.append(par.getStep().begin(), par.getStep().end());
  lbs.push_back(inner.getLowerBound());
  ubs.push_back(inner.getUpperBound());
  steps.push_back(inner.getStep());

  rewriter.setInsertionPoint(par);
  auto newPar = rewriter.create<scf::ParallelOp>(par.getLoc(), lbs, ubs, steps);
  Block &newBody = newPar.getRegion().front();
  rewriter.eraseOp(newBody.getTerminator());

  IRMapping mapping;
  auto newArgs = newBody.getArguments();
  // Map existing parallel IVs.
  for (auto it : llvm::enumerate(par.getRegion().front().getArguments())) {
    mapping.map(it.value(), newArgs[it.index()]);
  }
  // Map the inner for IV to the last parallel IV.
  mapping.map(inner.getInductionVar(), newArgs[par.getNumLoops()]);

  rewriter.setInsertionPointToStart(&newBody);
  for (Operation &op : inner.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  rewriter.create<scf::YieldOp>(par.getLoc());
  rewriter.eraseOp(par);
  return newPar;
}

// ---------------------------------------------------------------------------
// Linalg helpers
// ---------------------------------------------------------------------------

static bool isDimOnlyInjective(AffineMap map) {
  DenseSet<unsigned> seen;
  for (AffineExpr e : map.getResults()) {
    auto d = e.dyn_cast<AffineDimExpr>();
    if (!d) return false;
    if (!seen.insert(d.getPosition()).second) return false;
  }
  return true;
}

static bool outputUsesDim(AffineMap map, unsigned dim) {
  for (AffineExpr e : map.getResults())
    if (auto d = e.dyn_cast<AffineDimExpr>())
      if (d.getPosition() == dim) return true;
  return false;
}

static std::optional<unsigned> chooseBestParallelLoop(GenericOp op) {
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  for (unsigned i = 0, e = ranges.size(); i < e; ++i) {
    int64_t trip = ranges[i];
    if (trip != ShapedType::kDynamic && trip >= 64)
      return i;
  }
  return std::nullopt;
}

static scf::ForOp findOutermostFor(ArrayRef<Operation *> loops,
                                   Block *parent) {
  scf::ForOp candidate;
  for (Operation *loopOp : loops) {
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
      if (forOp->getBlock() != parent) continue;
      // Pick the one that has no ancestor scf.for within this loop set.
      bool hasParentInSet = false;
      Operation *p = forOp->getParentOp();
      while (p) {
        if (llvm::is_contained(loops, p)) {
          hasParentInSet = true;
          break;
        }
        p = p->getParentOp();
      }
      if (!hasParentInSet) {
        candidate = forOp;
        break;
      }
    }
  }
  return candidate;
}

// =========================================================
// THE PASS
// =========================================================

struct FinalMVPParallelSchedulingPass
    : public PassWrapper<FinalMVPParallelSchedulingPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalMVPParallelSchedulingPass)

  StringRef getArgument() const final {
    return "final-mvp-parallel-scheduling";
  }
  StringRef getDescription() const final {
    return "Convert profitable linalg loops to scf.parallel";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // ====================================================
    // MODE A: scf-first scheduling (post-lowering/vectorization)
    // ====================================================
    SmallVector<scf::ForOp, 8> forCandidates;
    func.walk([&](scf::ForOp forOp) { forCandidates.push_back(forOp); });

    for (scf::ForOp forOp : forCandidates) {
      // Skip erased ops (defensive).
      if (!forOp || !forOp->getBlock()) continue;

      auto tripCount = getConstTripCount(forOp);
      if (!tripCount || *tripCount < 2) {
        LLVM_DEBUG(llvm::dbgs() << "Skip scf.for: small/dynamic tripcount\n");
        continue;
      }
      if (!isSafeToParallelizeSCF(forOp)) continue;
      if (!hasDisjointWritesHeuristic(forOp)) {
        // Attempt privatization-based parallelization for accumulation loops.
        if (auto par = privatizeAndParallelize(rewriter, forOp)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Privatized + parallelized accumulation loop\n");
          continue;
        }
        continue;
      }

      bool hasInnerFor = false;
      bool hasVectorActivity = false;
      forOp.walk([&](Operation *op) {
        if (op == forOp.getOperation()) return WalkResult::advance();
        if (isa<scf::ForOp>(op)) hasInnerFor = true;
        if (isa<vector::TransferReadOp, vector::TransferWriteOp,
                vector::ContractionOp>(op) ||
            op->getName().getStringRef() == "vector.multi_reduction")
          hasVectorActivity = true;
        if (hasInnerFor && hasVectorActivity)
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (!hasInnerFor || !hasVectorActivity) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip scf.for: no nested loop or no vector activity\n");
        continue;
      }

      if (auto parentPar = dyn_cast<scf::ParallelOp>(forOp->getParentOp())) {
        if (isParallelizableTileFor(parentPar) &&
            canCollapseParallelFor(parentPar, forOp)) {
          auto par = collapseParallelFor(rewriter, parentPar, forOp);
          LLVM_DEBUG(llvm::dbgs()
                     << "Collapsed scf.parallel + scf.for -> multi-d parallel: "
                     << par << "\n");
          continue;
        }
      }

      rewriter.setInsertionPoint(forOp);
      auto par = convertForToParallel(rewriter, forOp);
      LLVM_DEBUG(llvm::dbgs() << "Converted scf.for -> scf.parallel: " << par
                              << "\n");
    }

    // ====================================================
    // MODE B: linalg scheduling
    // ====================================================
    // Collect candidates to avoid invalidating iterator during modification
    SmallVector<GenericOp, 4> candidates;
    func.walk([&](GenericOp op) { candidates.push_back(op); });

    for (GenericOp op : candidates) {
      rewriter.setInsertionPoint(op);
      GenericOp processedOp = op;

      // ====================================================
      // PART 1: Interchange (Locality)
      // ====================================================

      bool allParallel = llvm::all_of(
          processedOp.getIteratorTypesArray(), [](utils::IteratorType t) {
            return t == utils::IteratorType::parallel;
          });

      if (allParallel) {
        auto indexingMaps = processedOp.getIndexingMapsArray();
        if (!indexingMaps.empty()) {
          // Heuristic: Last output expression = contiguous dimension
          AffineMap outputMap = indexingMaps.back();
          unsigned numLoops = processedOp.getNumLoops();

          if (outputMap.getNumResults() > 0) {
            AffineExpr fastDimExpr = outputMap.getResults().back();

            // Only interchange if it's a pure dimension (d0, d1...)
            if (auto dimExpr = fastDimExpr.dyn_cast<AffineDimExpr>()) {
              unsigned bestInnerLoop = dimExpr.getPosition();

              if (bestInnerLoop != numLoops - 1) {
                SmallVector<unsigned, 4> permutation;
                for (unsigned i = 0; i < numLoops; ++i) {
                  if (i != bestInnerLoop) permutation.push_back(i);
                }
                permutation.push_back(bestInnerLoop);

                FailureOr<GenericOp> res =
                    linalg::interchangeGenericOp(rewriter, processedOp,
                                                 permutation);
                if (succeeded(res)) {
                  processedOp = *res;
                }
              }
            }
          }
        }
      }

      // ====================================================
      // PART 2: Parallelization Legality & Profitability
      // ====================================================

      // A. Re-check Parallel Iterators (must be purely parallel)
      bool isParallelIter = llvm::all_of(
          processedOp.getIteratorTypesArray(), [](utils::IteratorType t) {
            return t == utils::IteratorType::parallel;
          });
      if (!isParallelIter) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip linalg.generic: non-parallel iterators\n");
        continue;
      }

      // Reductions: only interchange/lower, no parallelization.
      bool hasReduction = llvm::any_of(
          processedOp.getIteratorTypesArray(), [](utils::IteratorType it) {
            return it == utils::IteratorType::reduction;
          });

      // B. Pick candidate parallel loop (outermost static >=64).
      auto maybeLoop = chooseBestParallelLoop(processedOp);
      if (!maybeLoop) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip linalg.generic: no loop with tripcount >=64\n");
      }
      unsigned targetLoop = maybeLoop.value_or(0);

      // C. Injectivity per output map; allow multiple outputs.
      auto maps = processedOp.getIndexingMapsArray();
      bool outputsGood = true;
      for (unsigned i = 0, e = processedOp.getNumDpsInits(); i < e; ++i) {
        AffineMap outMap = maps[processedOp.getNumDpsInputs() + i];
        if (!isDimOnlyInjective(outMap) ||
            !outputUsesDim(outMap, targetLoop)) {
          outputsGood = false;
          break;
        }
      }
      if (!outputsGood) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip linalg.generic: output map not injective/uses dim\n");
        continue;
      }

      SmallVector<int64_t, 4> loopBounds = processedOp.getStaticLoopRanges();
      if (loopBounds.empty() || targetLoop >= loopBounds.size()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip linalg.generic: missing loop bounds\n");
        continue;
      }
      int64_t targetTripCount = loopBounds[targetLoop];
      if (targetTripCount == ShapedType::kDynamic ||
          targetTripCount < 64) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skip linalg.generic: target loop tripcount < 64\n");
        continue;
      }

      // ====================================================
      // PART 3: Lowering & Conversion
      // ====================================================

      Block *parentBlock = processedOp->getBlock();

      // Note: If linalgOpToLoops is missing/renamed in your version,
      // you may need to use `linalg::lowerToLoops` or check your specific
      // MLIR/XLA API.
      FailureOr<SmallVector<Operation *, 4>> loopsOrFail =
          linalg::linalgOpToLoops(rewriter, processedOp);

      if (failed(loopsOrFail)) {
        LLVM_DEBUG(llvm::dbgs() << "linalgOpToLoops failed\n");
        continue;
      }

      auto loops = std::move(*loopsOrFail);

      // Robustly find the outermost scf.for in the same block
      scf::ForOp outerLoop = findOutermostFor(loops, parentBlock);

      if (outerLoop && !hasReduction) {
        if (isParallelizableTileFor(outerLoop)) {
          auto par = convertForToParallel(rewriter, outerLoop);
          LLVM_DEBUG(llvm::dbgs()
                     << "Converted linalg outer scf.for -> scf.parallel: "
                     << par << "\n");
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Skip outer scf.for from linalg: not parallelizable\n");
        }
      }

      // Drop the original linalg op to avoid duplicate implementations.
      rewriter.eraseOp(processedOp);
    }

    // Remove trivially dead loads (e.g., stale loads overwritten immediately).
    SmallVector<Operation *> deadLoads;
    func.walk([&](memref::LoadOp load) {
      if (load.use_empty()) deadLoads.push_back(load.getOperation());
    });
    for (Operation *op : deadLoads) rewriter.eraseOp(op);

    // Cleanup: erase no-op copies (src==dst).
    SmallVector<memref::CopyOp> noopCopies;
    func.walk([&](memref::CopyOp copy) {
      if (copy.getSource() == copy.getTarget())
        noopCopies.push_back(copy);
    });
    for (auto copy : noopCopies) rewriter.eraseOp(copy);

    // Hoist simple allocs out of scf.parallel when safe.
    func.walk([&](scf::ParallelOp par) {
      SmallVector<memref::AllocOp> allocs;
      par.walk([&](memref::AllocOp alloc) {
        allocs.push_back(alloc);
      });
      for (auto alloc : allocs) {
        // Only hoist if alloc has static shape, no escapes outside parallel.
        auto type = alloc.getType();
        if (!type.hasStaticShape()) continue;
        bool escapes = false;
        for (Operation *user : alloc.getOperation()->getUsers()) {
          if (!par->isProperAncestor(user)) {
            escapes = true;
            break;
          }
        }
        if (escapes) continue;

        // Insert new alloc before the parallel, replace uses, and dealloc after.
        rewriter.setInsertionPoint(par);
        auto newAlloc = rewriter.create<memref::AllocOp>(alloc.getLoc(), type);
        alloc.replaceAllUsesWith(newAlloc.getResult());
        rewriter.setInsertionPointAfter(par);
        rewriter.create<memref::DeallocOp>(alloc.getLoc(), newAlloc);
        rewriter.eraseOp(alloc);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace stablehlo {

std::unique_ptr<Pass> createFinalMVPParallelSchedulingPass() {
  return std::make_unique<FinalMVPParallelSchedulingPass>();
}

void registerFinalMVPParallelSchedulingPass() {
  PassRegistration<FinalMVPParallelSchedulingPass>();
}

} // namespace stablehlo
} // namespace mlir
