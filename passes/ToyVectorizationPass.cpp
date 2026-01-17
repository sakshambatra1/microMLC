#include "stablehlo/transforms/ToyVectorizationPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

#define DEBUG_TYPE "toy-vectorize"

using namespace mlir;

namespace {

struct VectorizationOptions {
  int64_t vectorWidth = 4;
  bool conservative = true;
  bool enableReductions = false;
  bool verbose = false;
};

static void logDecision(bool verbose, Operation *op, StringRef msg) {
  if (!verbose) return;
  llvm::errs() << "[toy-vectorize] " << msg << "\n  op: " << *op << "\n";
}

/// Build a corrected permutation map for the matmul-style broadcast loads we
/// see in this pipeline. For the transfers we need to satisfy the verifier:
///   numDims == source rank, numResults == vector rank.
/// Vector rank is 3 (m, n, k) and source rank is 2.
/// - A-tile (1xK) -> vector<1xNxK>: map (d0, 0, d1)  (broadcast vector dim 1)
/// - B-tile (KxN) -> vector<1xNxK>: map (0, d1, d0)  (broadcast vector dim 0)
static AffineMap
buildMatmulBroadcastMap(ShapedType sourceType, VectorType vectorType,
                        MLIRContext *ctx) {
  if (!sourceType || !vectorType)
    return AffineMap();
  if (sourceType.getRank() != 2 || vectorType.getRank() != 3)
    return AffineMap();

  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr d1 = getAffineDimExpr(1, ctx);

  // If the first source dim is a broadcasted 1, treat it as the "m" dim.
  bool firstIsOne =
      sourceType.hasStaticShape() && sourceType.getDimSize(0) == 1;
  SmallVector<AffineExpr> results;
  if (firstIsOne) {
    results = {d0, getAffineConstantExpr(0, ctx), d1};
  } else {
    results = {getAffineConstantExpr(0, ctx), d1, d0};
  }
  return AffineMap::get(/*dimCount=*/sourceType.getRank(), /*symbolCount=*/0,
                        results, ctx);
}

/// ============================================================================
/// Helper: Shape/stride checks
/// ============================================================================
static bool hasStaticShape(Type t) {
  if (auto mr = dyn_cast<MemRefType>(t))
    return mr.hasStaticShape();
  if (auto rt = dyn_cast<RankedTensorType>(t))
    return rt.hasStaticShape();
  return false;
}

static bool isUnitStrideForLoopDim(linalg::LinalgOp op, int64_t loopDim,
                                   unsigned operandIdx) {
  auto maps = op.getIndexingMapsArray();
  AffineMap map = maps[operandIdx];
  auto mr = dyn_cast<MemRefType>(op->getOperand(operandIdx).getType());
  if (!mr)
    return true; // tensor operand, stride check not applicable

  if (!map.isPermutation())
    return false;

  int64_t memrefDim = -1;
  for (auto it : llvm::enumerate(map.getResults())) {
    if (auto d = it.value().dyn_cast<AffineDimExpr>()) {
      if (d.getPosition() == loopDim) {
        memrefDim = it.index();
        break;
      }
    }
  }
  if (memrefDim < 0)
    return true; // loop not used by this operand

  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(mr, strides, offset)))
    return false;
  if (memrefDim >= static_cast<int64_t>(strides.size()))
    return false;
  return strides[memrefDim] == 1;
}

static bool hasUnitStrideOnVectorDim(linalg::LinalgOp op, int64_t vecDim) {
  for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (!isUnitStrideForLoopDim(op, vecDim, i))
      return false;
  }
  return true;
}

/// ============================================================================
/// Legality: Ensure we are safe to run on Buffers/Tensors
/// ============================================================================
static bool isVectorizationLegal(linalg::LinalgOp op,
                                 const VectorizationOptions &opt) {
  // Skip trivially unfriendly ops.
  if (isa<linalg::CopyOp>(op.getOperation()))
    return false;

  bool isMatmulLike =
      isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op.getOperation());
  bool hasReductions = op.getNumReductionLoops() > 0;

  // Optional reduction support (allow matmul even when reductions are off).
  if (!opt.enableReductions && hasReductions && !isMatmulLike)
    return false;

  // Conservative mode: require static shapes for non-matmul ops and require
  // projected permutations for all ops. Allow matmul/bmm with dynamic tiles.
  if (opt.conservative) {
    if (!isMatmulLike) {
      for (Type t : op->getOperandTypes()) {
        if (!hasStaticShape(t))
          return false;
      }
    }
    for (AffineMap map : op.getIndexingMapsArray()) {
      if (!map.isProjectedPermutation(/*allowZeroInResults=*/false))
        return false;
    }
  }

  return true;
}

/// ============================================================================
/// Profitability: Adjusted for TILED loops
/// ============================================================================
static bool isVectorizationProfitable(linalg::LinalgOp op,
                                      const VectorizationOptions &opt) {
  SmallVector<int64_t> shape = op.getStaticLoopRanges();
  if (shape.empty()) return false;

  auto iterTypes = op.getIteratorTypesArray();
  bool hasVectorizableLoop = false;
  for (int64_t i = 0, e = iterTypes.size(); i < e; ++i) {
    if (iterTypes[i] != utils::IteratorType::parallel)
      continue;
    if (i >= static_cast<int64_t>(shape.size()))
      continue;
    int64_t s = shape[i];
    // Allow partial tiles (masking handles tails). Only skip pure scalars.
    if (s == ShapedType::kDynamic || s > 1) {
      hasVectorizableLoop = true;
      break;
    }
  }
  return hasVectorizableLoop;
}

/// Guard: refuse vectorization if any static loop range exceeds a register-ish
/// cap (prevents gigantic vector types like vector<128x32x32>).
static bool withinVectorSizeCap(linalg::LinalgOp op, int64_t cap) {
  SmallVector<int64_t> shape = op.getStaticLoopRanges();
  if (shape.empty()) return false;
  for (int64_t s : shape) {
    if (s == ShapedType::kDynamic) continue;
    if (s > cap) return false;
  }
  return true;
}

/// If loop sizes exceed the vector cap, tile to the cap so the inner op can
/// be vectorized (e.g. 8x8x8 -> 4x4x4). Only applies when all loop ranges
/// are static and divisible by the cap.
static FailureOr<linalg::LinalgOp>
tileForVectorization(PatternRewriter &rewriter, linalg::LinalgOp op,
                     int64_t cap) {
  if (op->hasAttr("toy.vector-tiled"))
    return op;
  SmallVector<int64_t> shape = op.getStaticLoopRanges();
  if (shape.empty())
    return failure();

  SmallVector<int64_t> tileSizes;
  tileSizes.reserve(shape.size());
  bool needsTiling = false;
  for (int64_t s : shape) {
    if (s == ShapedType::kDynamic)
      return failure();
    if (s > cap) {
      if (cap <= 0 || (s % cap) != 0)
        return failure();
      tileSizes.push_back(cap);
      needsTiling = true;
    } else {
      tileSizes.push_back(0);
    }
  }

  if (!needsTiling)
    return op;

  auto options = linalg::LinalgTilingOptions().setTileSizes(tileSizes);
  FailureOr<linalg::TiledLinalgOp> tiled =
      linalg::tileLinalgOp(rewriter, op, options);
  if (failed(tiled))
    return failure();

  if (!tiled->tensorResults.empty())
    rewriter.replaceOp(op, tiled->tensorResults);
  else
    rewriter.eraseOp(op);

  tiled->op->setAttr("toy.vector-tiled", rewriter.getUnitAttr());
  return tiled->op;
}

static SmallVector<int64_t> getLoopSizesWithCap(linalg::LinalgOp op,
                                                int64_t cap) {
  SmallVector<int64_t> loopSizes = op.getStaticLoopRanges();
  for (int64_t &size : loopSizes) {
    if (size == ShapedType::kDynamic)
      size = cap;
  }
  return loopSizes;
}

static Value maybeCastToStatic(RewriterBase &rewriter, Location loc, Value v,
                               ArrayRef<int64_t> shape,
                               SmallVectorImpl<Operation *> &createdCasts) {
  auto type = v.getType().dyn_cast<MemRefType>();
  if (!type || type.getRank() != static_cast<int64_t>(shape.size()))
    return v;

  bool needsCast = false;
  for (int64_t i = 0, e = type.getRank(); i < e; ++i) {
    int64_t dim = type.getDimSize(i);
    if (dim == ShapedType::kDynamic) {
      needsCast = true;
      continue;
    }
    if (dim != shape[i])
      return v;
  }

  if (!needsCast)
    return v;

  auto newType =
      MemRefType::get(shape, type.getElementType(), type.getLayout(),
                      type.getMemorySpace());
  auto cast = rewriter.create<memref::CastOp>(loc, newType, v);
  createdCasts.push_back(cast.getOperation());
  return cast;
}

static LogicalResult tryVectorizeMatmul(RewriterBase &rewriter,
                                        linalg::MatmulOp op, int64_t cap) {
  Location loc = op.getLoc();
  SmallVector<int64_t> loopSizes = getLoopSizesWithCap(op, cap);
  if (loopSizes.size() != 3)
    return linalg::vectorize(rewriter, op.getOperation());

  SmallVector<Operation *> createdCasts;
  Value a = maybeCastToStatic(rewriter, loc, op.getInputs()[0],
                              {loopSizes[0], loopSizes[2]}, createdCasts);
  Value b = maybeCastToStatic(rewriter, loc, op.getInputs()[1],
                              {loopSizes[2], loopSizes[1]}, createdCasts);
  Value c = maybeCastToStatic(rewriter, loc, op.getOutputs()[0],
                              {loopSizes[0], loopSizes[1]}, createdCasts);

  linalg::MatmulOp target = op;
  bool changed = a != op.getInputs()[0] || b != op.getInputs()[1] ||
                 c != op.getOutputs()[0];
  if (changed) {
    auto newOp = rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{a, b}, ValueRange{c});
    if (auto attr = op->getAttr("done_tiling"))
      newOp->setAttr("done_tiling", attr);
    target = newOp;
  }

  LogicalResult result = linalg::vectorize(rewriter, target.getOperation());
  if (failed(result)) {
    if (changed)
      rewriter.eraseOp(target);
    for (Operation *castOp : createdCasts)
      rewriter.eraseOp(castOp);
    return failure();
  }

  if (changed)
    rewriter.eraseOp(op);
  return success();
}

static LogicalResult tryVectorizeBatchMatmul(RewriterBase &rewriter,
                                             linalg::BatchMatmulOp op,
                                             int64_t cap) {
  Location loc = op.getLoc();
  SmallVector<int64_t> loopSizes = getLoopSizesWithCap(op, cap);
  if (loopSizes.size() != 4)
    return linalg::vectorize(rewriter, op.getOperation());

  SmallVector<Operation *> createdCasts;
  Value a = maybeCastToStatic(rewriter, loc, op.getInputs()[0],
                              {loopSizes[0], loopSizes[1], loopSizes[3]},
                              createdCasts);
  Value b = maybeCastToStatic(rewriter, loc, op.getInputs()[1],
                              {loopSizes[0], loopSizes[3], loopSizes[2]},
                              createdCasts);
  Value c = maybeCastToStatic(rewriter, loc, op.getOutputs()[0],
                              {loopSizes[0], loopSizes[1], loopSizes[2]},
                              createdCasts);

  linalg::BatchMatmulOp target = op;
  bool changed = a != op.getInputs()[0] || b != op.getInputs()[1] ||
                 c != op.getOutputs()[0];
  if (changed) {
    auto newOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, ValueRange{a, b}, ValueRange{c});
    if (auto attr = op->getAttr("done_tiling"))
      newOp->setAttr("done_tiling", attr);
    target = newOp;
  }

  LogicalResult result = linalg::vectorize(rewriter, target.getOperation());
  if (failed(result)) {
    if (changed)
      rewriter.eraseOp(target);
    for (Operation *castOp : createdCasts)
      rewriter.eraseOp(castOp);
    return failure();
  }

  if (changed)
    rewriter.eraseOp(op);
  return success();
}

/// Expand batch_matmul into a loop of matmul ops and vectorize the inner
/// matmul to avoid leaving batch_matmul unvectorized.
static LogicalResult expandBatchMatmulToMatmul(PatternRewriter &rewriter,
                                               linalg::BatchMatmulOp bmm,
                                               int64_t cap) {
  Location loc = bmm.getLoc();
  Value a = bmm.getInputs()[0];
  Value b = bmm.getInputs()[1];
  Value c = bmm.getOutputs()[0];

  auto aType = a.getType().dyn_cast<MemRefType>();
  auto bType = b.getType().dyn_cast<MemRefType>();
  auto cType = c.getType().dyn_cast<MemRefType>();
  if (!aType || !bType || !cType)
    return failure();
  if (aType.getRank() != 3 || bType.getRank() != 3 || cType.getRank() != 3)
    return failure();

  int64_t batch = aType.getDimSize(0);
  if (batch == ShapedType::kDynamic)
    return failure();
  if (batch > cap)
    return failure();

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value cBatch = rewriter.create<arith::ConstantIndexOp>(loc, batch);

  auto loop = rewriter.create<scf::ForOp>(loc, c0, cBatch, c1);
  rewriter.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();

  auto dimVal = [&](Value v, int64_t dim) -> OpFoldResult {
    auto t = v.getType().cast<MemRefType>();
    if (!t.isDynamicDim(dim))
      return rewriter.getIndexAttr(t.getDimSize(dim));
    return rewriter.create<memref::DimOp>(loc, v, dim).getResult();
  };
  auto dimOrDynamic = [&](MemRefType t, int64_t dim) -> int64_t {
    return t.isDynamicDim(dim) ? ShapedType::kDynamic : t.getDimSize(dim);
  };

  SmallVector<OpFoldResult> offsets = {iv, rewriter.getIndexAttr(0),
                                       rewriter.getIndexAttr(0)};
  SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(1)};

  // A: [batch, M, K] -> [M, K]
  SmallVector<OpFoldResult> aSizes = {rewriter.getIndexAttr(1), dimVal(a, 1),
                                      dimVal(a, 2)};
  SmallVector<int64_t> aShape = {dimOrDynamic(aType, 1),
                                 dimOrDynamic(aType, 2)};
  auto aSubviewType = memref::SubViewOp::inferRankReducedResultType(
                          aShape, aType, offsets, aSizes, strides)
                          .dyn_cast<MemRefType>();
  if (!aSubviewType)
    return failure();
  Value aSubview = rewriter.create<memref::SubViewOp>(
      loc, aSubviewType, a, offsets, aSizes, strides);

  // B: [batch, K, N] -> [K, N]
  SmallVector<OpFoldResult> bSizes = {rewriter.getIndexAttr(1), dimVal(b, 1),
                                      dimVal(b, 2)};
  SmallVector<int64_t> bShape = {dimOrDynamic(bType, 1),
                                 dimOrDynamic(bType, 2)};
  auto bSubviewType = memref::SubViewOp::inferRankReducedResultType(
                          bShape, bType, offsets, bSizes, strides)
                          .dyn_cast<MemRefType>();
  if (!bSubviewType)
    return failure();
  Value bSubview = rewriter.create<memref::SubViewOp>(
      loc, bSubviewType, b, offsets, bSizes, strides);

  // C: [batch, M, N] -> [M, N]
  SmallVector<OpFoldResult> cSizes = {rewriter.getIndexAttr(1), dimVal(c, 1),
                                      dimVal(c, 2)};
  SmallVector<int64_t> cShape = {dimOrDynamic(cType, 1),
                                 dimOrDynamic(cType, 2)};
  auto cSubviewType = memref::SubViewOp::inferRankReducedResultType(
                          cShape, cType, offsets, cSizes, strides)
                          .dyn_cast<MemRefType>();
  if (!cSubviewType)
    return failure();
  Value cSubview = rewriter.create<memref::SubViewOp>(
      loc, cSubviewType, c, offsets, cSizes, strides);

  auto matmul = rewriter.create<linalg::MatmulOp>(
      loc, ValueRange{aSubview, bSubview}, ValueRange{cSubview});
  if (failed(linalg::vectorize(rewriter, matmul.getOperation()))) {
    rewriter.eraseOp(loop);
    return failure();
  }

  rewriter.eraseOp(bmm);
  return success();
}

/// ============================================================================
/// Strategy: Pick the Stride-1 Dimension
/// ============================================================================
static FailureOr<int64_t> chooseVectorDim(linalg::LinalgOp op,
                                          const VectorizationOptions &opt) {
  auto iterTypes = op.getIteratorTypesArray();
  SmallVector<int64_t> shape = op.getStaticLoopRanges();
  int64_t numLoops = iterTypes.size();
  bool isMatmulLike =
      isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op.getOperation());
  bool hasReductions = op.getNumReductionLoops() > 0;

  // Iterate backwards (innermost loop first). 
  // This is usually the loop we tiled for vectorization.
  for (int64_t i = numLoops - 1; i >= 0; --i) {
    
    // 1. Must be Parallel
    if (iterTypes[i] != utils::IteratorType::parallel)
      continue;

    // 2. Width gate: allow partial tiles; only skip scalar loops.
    if (!shape.empty() && i < static_cast<int64_t>(shape.size())) {
      int64_t tripCount = shape[i];
      if (tripCount != ShapedType::kDynamic && tripCount == 1)
        continue;
    }

    // 3. Stride gate: require contiguous access when in conservative mode,
    // but allow matmul-like ops to bypass (they're common and often strided).
    if (opt.conservative && !isMatmulLike && !hasReductions &&
        !hasUnitStrideOnVectorDim(op, i))
      continue;

    return i;
  }
  
  return failure();
}

/// ============================================================================
/// Canonicalize matmul-shaped linalg.generic to linalg.matmul
/// ============================================================================
static bool isMatmulGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1)
    return false;
  auto iters = generic.getIteratorTypesArray();
  if (iters.size() != 3)
    return false;
  if (!(iters[0] == utils::IteratorType::parallel &&
        iters[1] == utils::IteratorType::parallel &&
        iters[2] == utils::IteratorType::reduction))
    return false;
  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  auto checkMap = [](AffineMap map, ArrayRef<int64_t> expected) {
    if (map.getNumResults() != expected.size())
      return false;
    for (auto it : llvm::enumerate(expected)) {
      auto d = map.getResult(it.index()).dyn_cast<AffineDimExpr>();
      if (!d || d.getPosition() != it.value())
        return false;
    }
    return true;
  };
  // lhs: (i,k), rhs: (k,j), out: (i,j)
  if (!checkMap(maps[0], {0, 2})) return false;
  if (!checkMap(maps[1], {2, 1})) return false;
  if (!checkMap(maps[2], {0, 1})) return false;
  return true;
}

struct GenericToMatmulPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    if (!isMatmulGeneric(generic))
      return failure();
    auto outs = generic.getOutputs();
    auto ins = generic.getInputs();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(generic, outs.getTypes(),
                                                  ins, outs);
    return success();
  }
};

/// Rewrite vector.multi_reduction(add) of a mul into vector.contract when the
/// shape matches a simple matmul-like contraction of rank-3 -> rank-2.
struct MultiReductionToContractPattern
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::MultiDimReductionOp red,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[toy-vectorize] Try red->contract: " << red
                            << "\n";);
    auto redDimsAttr = red.getReductionDimsAttr();
    if (!redDimsAttr || redDimsAttr.size() != 1) return failure();
    int64_t redDim = redDimsAttr[0].cast<IntegerAttr>().getInt();

    auto accType = red.getAcc().getType().dyn_cast<VectorType>();
    auto srcType = red.getSource().getType().dyn_cast<VectorType>();
    if (!accType || !srcType) return failure();
    int64_t rank = srcType.getRank();
    if (rank < 2) return failure();
    if (redDim != rank - 1) return failure(); // only handle tail reduction
    if (accType.getRank() != rank - 1) return failure();
    for (int64_t i = 0; i < rank - 1; ++i) {
      if (accType.getDimSize(i) != srcType.getDimSize(i))
        return failure();
    }

    // Strict operand check: source must be a mul of two same-shaped vectors.
    Value lhs, rhs;
    if (auto mulf = red.getSource().getDefiningOp<arith::MulFOp>()) {
      lhs = mulf.getLhs();
      rhs = mulf.getRhs();
    } else if (auto muli = red.getSource().getDefiningOp<arith::MulIOp>()) {
      lhs = muli.getLhs();
      rhs = muli.getRhs();
    } else {
      return failure();
    }
    auto lhsType = lhs.getType().dyn_cast<VectorType>();
    auto rhsType = rhs.getType().dyn_cast<VectorType>();
    if (!lhsType || !rhsType || lhsType != srcType || rhsType != srcType)
      return failure();

    // Sanity: verify iterator types size matches rank after we build them.
    auto ctx = rewriter.getContext();
    SmallVector<AffineExpr> dims;
    dims.reserve(rank);
    for (int64_t i = 0; i < rank; ++i)
      dims.push_back(mlir::getAffineDimExpr(i, ctx));

    // Identity maps for operands, drop the reduction dim for the result.
    SmallVector<AffineExpr> keepDims;
    keepDims.reserve(rank - 1);
    for (int64_t i = 0; i < rank; ++i)
      if (i != redDim) keepDims.push_back(dims[i]);

    auto mapA = AffineMap::get(rank, 0, dims, ctx);
    auto mapB = AffineMap::get(rank, 0, dims, ctx);
    auto mapC = AffineMap::get(rank, 0, keepDims, ctx);
    auto maps = rewriter.getAffineMapArrayAttr({mapA, mapB, mapC});

    SmallVector<Attribute> iterAttrs;
    iterAttrs.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      auto kind = (i == redDim) ? vector::IteratorType::reduction
                                : vector::IteratorType::parallel;
      iterAttrs.push_back(vector::IteratorTypeAttr::get(ctx, kind));
    }
    if (iterAttrs.size() != static_cast<size_t>(rank)) return failure();
    auto iterAttr = rewriter.getArrayAttr(iterAttrs);

    LLVM_DEBUG({
      llvm::dbgs() << "[toy-vectorize] Emit contract: rank=" << rank
                   << " redDim=" << redDim << " lhsType=" << lhsType
                   << " rhsType=" << rhsType << " accType=" << accType
                   << "\n";
    });
    auto contract = rewriter.create<vector::ContractionOp>(
        red.getLoc(), lhs, rhs, red.getAcc(), maps, iterAttr);
    rewriter.replaceOp(red, contract.getResult());

    return success();
  }
};

/// ============================================================================
/// Core Rewrite
/// ============================================================================
struct VectorizeLinalgOpPattern : public RewritePattern {
  VectorizeLinalgOpPattern(MLIRContext *ctx, VectorizationOptions opt,
                           bool disableSizeCap)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx), opt(opt),
        disableSizeCap(disableSizeCap) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(operation);
    if (!linalgOp) return failure();

    // Focus on matmul-like contractions; avoid vectorizing outer fused/huge
    // generics that degrade to multi_reduction.
    bool isMatmulLike =
        isa<linalg::MatmulOp, linalg::BatchMatmulOp>(linalgOp.getOperation());
    if (!isMatmulLike)
      return failure();

    // Keep batch/head dims small: only vectorize when loop ranges fit in a
    // small cap (register tile) and batch dims are <= cap.
    if (!disableSizeCap) {
      int64_t cap = opt.vectorWidth;
      if (!withinVectorSizeCap(linalgOp, cap)) {
        auto tiledOr = tileForVectorization(rewriter, linalgOp, cap);
        if (failed(tiledOr)) {
          logDecision(opt.verbose, linalgOp.getOperation(),
                      "reject: loop range exceeds vector cap");
          return failure();
        }
        linalgOp = *tiledOr;
        logDecision(opt.verbose, linalgOp.getOperation(),
                    "retiled for vector cap");
      }
      if (auto bmm = dyn_cast<linalg::BatchMatmulOp>(linalgOp.getOperation())) {
        SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
        if (!shape.empty() && shape[0] != ShapedType::kDynamic &&
            shape[0] > cap) {
          logDecision(opt.verbose, linalgOp.getOperation(),
                      "reject: batch dim exceeds vector cap");
          return failure();
        }
      }
    }

    if (!isVectorizationLegal(linalgOp, opt))
      return failure();

    if (!isVectorizationProfitable(linalgOp, opt))
      return failure();

    auto vecDimOr = chooseVectorDim(linalgOp, opt);
    if (failed(vecDimOr)) {
      logDecision(opt.verbose, linalgOp.getOperation(),
                  "reject: no contiguous vector dim");
      return failure();
    }
    int64_t vecDim = *vecDimOr;

    logDecision(opt.verbose, linalgOp.getOperation(),
                "accept: vectorizing dim " + std::to_string(vecDim));

    if (auto bmm = dyn_cast<linalg::BatchMatmulOp>(linalgOp.getOperation())) {
      if (succeeded(tryVectorizeBatchMatmul(rewriter, bmm, opt.vectorWidth)))
        return success();
      if (succeeded(expandBatchMatmulToMatmul(rewriter, bmm, opt.vectorWidth)))
        return success();
      logDecision(opt.verbose, linalgOp.getOperation(), "vectorize() failed");
      return failure();
    }

    if (auto mm = dyn_cast<linalg::MatmulOp>(linalgOp.getOperation())) {
      if (succeeded(tryVectorizeMatmul(rewriter, mm, opt.vectorWidth)))
        return success();
      logDecision(opt.verbose, linalgOp.getOperation(), "vectorize() failed");
      return failure();
    }
    return failure();
  }

  VectorizationOptions opt;
  bool disableSizeCap;
};

/// ============================================================================
/// The Pass
/// ============================================================================
struct ToyVectorizationPass
    : public PassWrapper<ToyVectorizationPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyVectorizationPass)

  ToyVectorizationPass() = default;
  ToyVectorizationPass(const ToyVectorizationPass &pass) : PassWrapper(pass) {
    vectorWidth = pass.vectorWidth;
    conservative = pass.conservative;
    enableReductions = pass.enableReductions;
    verbose = pass.verbose;
    dumpFirstBadTransfer = pass.dumpFirstBadTransfer;
  }

  Option<int64_t> vectorWidth{*this, "vector-width", llvm::cl::desc("Vector width"), llvm::cl::init(4)};
  Option<bool> conservative{*this, "conservative", llvm::cl::desc("Require static shapes + permutation maps"), llvm::cl::init(true)};
  Option<bool> enableReductions{*this, "enable-reductions", llvm::cl::desc("Allow vectorizing reduction loops"), llvm::cl::init(true)};
  Option<bool> verbose{*this, "verbose", llvm::cl::desc("Print decisions"), llvm::cl::init(false)};
  Option<bool> disableRedToContract{*this, "disable-red-to-contract", llvm::cl::desc("Disable multi_reduction -> vector.contract rewrite"), llvm::cl::init(false)};
  Option<bool> disableCleanup{*this, "disable-cleanup", llvm::cl::desc("Disable post-vectorization cleanup"), llvm::cl::init(false)};
  Option<bool> disableSizeCap{*this, "disable-size-cap", llvm::cl::desc("Disable vector size cap checks"), llvm::cl::init(false)};
  Option<bool> disableFallbackVec{*this, "disable-fallback-vec", llvm::cl::desc("Disable fallback matmul/bmm vectorization walker"), llvm::cl::init(false)};
  Option<bool> dumpFirstBadTransfer{*this, "dump-first-bad-transfer", llvm::cl::desc("Dump first invalid vector.transfer_{read,write} and fail"), llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    func::FuncDialect>();
  }

  StringRef getArgument() const final { return "toy-vectorize"; }
  StringRef getDescription() const final { return "Toy vectorization for Tiled+Buffered IR"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    opt.vectorWidth = vectorWidth;
    opt.conservative = conservative;
    opt.enableReductions = enableReductions;
    opt.verbose = verbose;

    // Stage 1: canonicalize matmul-shaped generics to linalg.matmul.
    RewritePatternSet prePatterns(&getContext());
    prePatterns.add<GenericToMatmulPattern>(&getContext());

    GreedyRewriteConfig config;
    config.enableRegionSimplification = true;
  if (failed(applyPatternsAndFoldGreedily(func, std::move(prePatterns),
                                          config))) {
    signalPassFailure();
    return;
  }

    // Stage 2: vectorize only matmul-like inner tiles.
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeLinalgOpPattern>(&getContext(), opt, disableSizeCap);
    if (!disableRedToContract)
      patterns.add<MultiReductionToContractPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns),
                                            config))) {
      signalPassFailure();
      return;
    }

    // Stage 2b: fallback walk to ensure matmul/batch_matmul get vectorized even
    // when nested in tiling/scf structure that patterns may miss.
    {
      IRRewriter rewriter(&getContext());
      func.walk([&](Operation *op) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
          return;
        // Skip if batch/head dims are large; keep them as loops.
        if (!disableSizeCap) {
          auto linalgOp = cast<linalg::LinalgOp>(op);
          if (!withinVectorSizeCap(linalgOp, opt.vectorWidth))
            return;
          if (auto bmm = dyn_cast<linalg::BatchMatmulOp>(op)) {
            SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
            if (!shape.empty() && shape[0] != ShapedType::kDynamic &&
                shape[0] > opt.vectorWidth)
              return;
          }
        }
        if (disableFallbackVec)
          return;
        rewriter.setInsertionPoint(op);
        if (auto mm = dyn_cast<linalg::MatmulOp>(op))
          (void)tryVectorizeMatmul(rewriter, mm, opt.vectorWidth);
        else if (auto bmm = dyn_cast<linalg::BatchMatmulOp>(op))
          (void)tryVectorizeBatchMatmul(rewriter, bmm, opt.vectorWidth);
        else
          (void)linalg::vectorize(rewriter, op);
      });
    }

    // Stage 3: light cleanup to enable downstream lowering.
    if (!disableCleanup) {
      RewritePatternSet cleanup(&getContext());
      linalg::populateLinalgTilingCanonicalizationPatterns(cleanup);
      vector::populateVectorToVectorCanonicalizationPatterns(cleanup);
      vector::TransferReadOp::getCanonicalizationPatterns(cleanup, &getContext());
      vector::TransferWriteOp::getCanonicalizationPatterns(cleanup, &getContext());
      (void)applyPatternsAndFoldGreedily(func, std::move(cleanup), config);
    }

    // --------------------------------------------------------------------
    // Validate and repair transfer permutation maps. The correct convention is:
    //   map.numDims == sourceRank
    //   map.numResults == vectorRank
    // indices.size == sourceRank
    // in_bounds.size == vectorRank (if present)
    //
    // We also repair the common broadcasted matmul loads:
    //   tensor<1xK>  -> vector<1xNxK> map (d0, d2)
    //   tensor<KxN>  -> vector<1xNxK> map (d2, d1)
    // --------------------------------------------------------------------
    IRRewriter rewriter(&getContext());
    bool sawError = false;
    auto validateAndFix = [&](Operation *op) -> WalkResult {
      auto dump = [&](auto transferOp, int64_t srcRank, int64_t vecRank) {
        if (!dumpFirstBadTransfer)
          return;
        llvm::errs() << "[toy-vectorize] Invalid transfer detected:\n";
        llvm::errs() << "  op: " << *transferOp << "\n";
        llvm::errs() << "  source type: " << transferOp.getSource().getType()
                     << " rank=" << srcRank << "\n";
        llvm::errs() << "  vector type: " << transferOp.getVectorType()
                     << " rank=" << vecRank << "\n";
        llvm::errs() << "  permutation_map: "
                     << transferOp.getPermutationMap() << "\n";
        llvm::errs() << "    numDims=" << transferOp.getPermutationMap().getNumDims()
                     << " numResults=" << transferOp.getPermutationMap().getNumResults()
                     << "\n";
        if (auto inBounds = transferOp.getInBoundsAttr()) {
          llvm::errs() << "  in_bounds size=" << inBounds.size() << "\n";
        } else {
          llvm::errs() << "  in_bounds = <none>\n";
        }
      };

      if (auto tr = dyn_cast<vector::TransferReadOp>(op)) {
        auto srcType = tr.getSource().getType().cast<ShapedType>();
        int64_t srcRank = srcType.getRank();
        int64_t vecRank = tr.getVectorType().getRank();
        AffineMap map = tr.getPermutationMap();
        bool needFix = map.getNumDims() != srcRank ||
                       map.getNumResults() != vecRank ||
                       static_cast<int64_t>(tr.getIndices().size()) != srcRank ||
                       (tr.getInBounds().has_value() &&
                        static_cast<int64_t>(tr.getInBounds()->size()) != vecRank);

        if (needFix) {
          dump(tr, srcRank, vecRank);

          rewriter.setInsertionPoint(tr);
          AffineMap newMap =
              buildMatmulBroadcastMap(srcType, tr.getVectorType(), &getContext());
          if (!newMap || newMap.getNumDims() != srcRank ||
              newMap.getNumResults() != vecRank) {
            sawError = true;
            return WalkResult::interrupt();
          }

          ArrayAttr newInBoundsAttr = tr.getInBoundsAttr();
          if (tr.getInBounds().has_value()) {
            SmallVector<bool> newInBounds(tr.getInBounds()->begin(),
                                          tr.getInBounds()->end());
            if (static_cast<int64_t>(newInBounds.size()) != vecRank)
              newInBounds.assign(vecRank, true);
            newInBoundsAttr = rewriter.getBoolArrayAttr(newInBounds);
          }

          auto newOp = rewriter.create<vector::TransferReadOp>(
              tr.getLoc(), tr.getVectorType(), tr.getSource(), tr.getIndices(),
              AffineMapAttr::get(newMap), tr.getPadding(), Value(),
              newInBoundsAttr);
          rewriter.replaceOp(tr, newOp.getResult());
          return WalkResult::advance();
        }
      } else if (auto tw = dyn_cast<vector::TransferWriteOp>(op)) {
        auto srcType = tw.getSource().getType().cast<ShapedType>();
        int64_t srcRank = srcType.getRank();
        int64_t vecRank = tw.getVectorType().getRank();
        AffineMap map = tw.getPermutationMap();
        bool needFix = map.getNumDims() != srcRank ||
                       map.getNumResults() != vecRank ||
                       static_cast<int64_t>(tw.getIndices().size()) != srcRank ||
                       (tw.getInBounds().has_value() &&
                        static_cast<int64_t>(tw.getInBounds()->size()) != vecRank);

        if (needFix) {
          dump(tw, srcRank, vecRank);

          rewriter.setInsertionPoint(tw);
          AffineMap newMap =
              buildMatmulBroadcastMap(srcType, tw.getVectorType(), &getContext());
          if (!newMap || newMap.getNumDims() != srcRank ||
              newMap.getNumResults() != vecRank) {
            sawError = true;
            return WalkResult::interrupt();
          }

          ArrayAttr newInBoundsAttr = tw.getInBoundsAttr();
          if (tw.getInBounds().has_value()) {
            SmallVector<bool> newInBounds(tw.getInBounds()->begin(),
                                          tw.getInBounds()->end());
            if (static_cast<int64_t>(newInBounds.size()) != vecRank)
              newInBounds.assign(vecRank, true);
            newInBoundsAttr = rewriter.getBoolArrayAttr(newInBounds);
          }

          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              tw, tw.getVector(), tw.getSource(), tw.getIndices(),
              AffineMapAttr::get(newMap), Value(), newInBoundsAttr);
          return WalkResult::advance();
        }
      }
      return WalkResult::advance();
    };

    if (func.walk(validateAndFix).wasInterrupted() || sawError) {
      signalPassFailure();
      return;
    }
  }

private:
  mutable VectorizationOptions opt;
};

} // namespace

namespace mlir {
namespace stablehlo {
std::unique_ptr<Pass> createToyVectorizationPass() {
  return std::make_unique<ToyVectorizationPass>();
}
void registerToyVectorizationPass() {
  PassRegistration<ToyVectorizationPass>();
}
} // namespace stablehlo
} // namespace mlir
