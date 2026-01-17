#include "stablehlo/transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

/// Rewrite CF branch operands and block arguments so that any block argument
/// of type `index` is replaced by an `i64` argument. Branches are updated to
/// pass i64 operands; index operands are cast to i64 at the branch site, and a
/// cast back to index is inserted at the top of the target block to satisfy
/// index-typed uses.
struct CfIntegerizeIndexBlockArgsPass
    : public PassWrapper<CfIntegerizeIndexBlockArgsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CfIntegerizeIndexBlockArgsPass)

  StringRef getArgument() const final {
    return "cf-integerize-index-block-args";
  }

  StringRef getDescription() const final {
    return "Rewrite CF block arguments from index to i64 and update branches.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect>();
  }

  // Adjust operands for a specific successor operand list.
  LogicalResult
  rewriteSuccessorOperands(MutableOperandRange succOperands,
                           ArrayRef<unsigned> indexPositions,
                           OpBuilder &builder) const {
    SmallVector<Value> newOperands;
    newOperands.reserve(succOperands.size());
    SmallVector<Value> savedIndexOperands;

    // Collect operands for non-index args and remember the ones corresponding
    // to index args for appending later.
    for (unsigned pos = 0; pos < succOperands.size(); ++pos) {
      Value operand = succOperands[pos];
      if (llvm::is_contained(indexPositions, pos)) {
        savedIndexOperands.push_back(operand);
      } else {
        newOperands.push_back(operand);
      }
    }

    // Append operands for the new trailing i64 args, casting index operands
    // to i64 as needed.
    for (Value operand : savedIndexOperands) {
      Type ty = operand.getType();
      if (ty.isIndex()) {
        operand = builder.create<arith::IndexCastOp>(
            operand.getLoc(), builder.getI64Type(), operand);
      }
      newOperands.push_back(operand);
    }

    if (newOperands.size() != succOperands.size())
      return failure();

    succOperands.assign(newOperands);
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      for (Block &block : func.getBody()) {
        SmallVector<unsigned> indexPositions;
        for (auto it : llvm::enumerate(block.getArguments())) {
          if (it.value().getType().isIndex())
            indexPositions.push_back(it.index());
        }
        if (indexPositions.empty())
          continue;

        // Append new i64 arguments corresponding to each index argument (in
        // order of appearance).
        SmallVector<BlockArgument> newArgs;
        newArgs.reserve(indexPositions.size());
        for (unsigned pos : indexPositions) {
          newArgs.push_back(
              block.addArgument(builder.getI64Type(),
                                block.getArgument(pos).getLoc()));
        }

        // Update predecessors' terminators.
        for (Block *pred : block.getPredecessors()) {
          Operation *term = pred->getTerminator();
          builder.setInsertionPoint(term);
          if (auto br = dyn_cast<cf::BranchOp>(term)) {
            if (failed(rewriteSuccessorOperands(br.getDestOperandsMutable(),
                                                indexPositions, builder))) {
              signalPassFailure();
              return;
            }
          } else if (auto cbr = dyn_cast<cf::CondBranchOp>(term)) {
            for (unsigned succIdx = 0; succIdx < cbr->getNumSuccessors();
                 ++succIdx) {
              if (cbr.getSuccessor(succIdx) != &block)
                continue;
              auto succOperands = cbr.getSuccessorOperands(succIdx);
              MutableOperandRange forwarded =
                  succOperands.slice(/*subStart=*/0, succOperands.size());
              if (failed(rewriteSuccessorOperands(forwarded, indexPositions,
                                                  builder))) {
                signalPassFailure();
                return;
              }
            }
          }
        }

        // Insert casts back to index inside the block and replace uses.
        builder.setInsertionPointToStart(&block);
        for (auto it : llvm::enumerate(indexPositions)) {
          unsigned oldPos = it.value();
          BlockArgument oldArg = block.getArgument(oldPos);
          BlockArgument newArg = newArgs[it.index()];
          Value idxVal = builder.create<arith::IndexCastOp>(
              newArg.getLoc(), builder.getIndexType(), newArg);
          oldArg.replaceAllUsesWith(idxVal);
        }

        // Erase old index arguments in descending order to keep indices valid.
        llvm::sort(indexPositions, std::greater<unsigned>());
        for (unsigned pos : indexPositions) {
          block.eraseArgument(pos);
        }
      }
    }
  }
};

}  // namespace

namespace mlir {
namespace stablehlo {

std::unique_ptr<Pass> createCfIntegerizeIndexBlockArgsPass() {
  return std::make_unique<CfIntegerizeIndexBlockArgsPass>();
}

}  // namespace stablehlo
}  // namespace mlir
