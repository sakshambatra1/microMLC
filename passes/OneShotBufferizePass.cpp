/* Copyright 2022 The StableHLO Authors.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
==============================================================================*/

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

// Wrap the common "cleanup -> one-shot bufferize -> finalize -> cleanup" flow
// so it can be reused from stablehlo-opt.
struct OneShotBufferizePass
    : public PassWrapper<OneShotBufferizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OneShotBufferizePass)

  StringRef getArgument() const final { return "my-one-shot-bufferize"; }
  StringRef getDescription() const final {
    return "Cleanup -> OneShot -> Finalizing -> Cleanup";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                func::FuncDialect, scf::SCFDialect, arith::ArithDialect,
                tensor::TensorDialect, linalg::LinalgDialect,
                math::MathDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpPassManager pm(ModuleOp::getOperationName());

    // Pre-cleanup to expose bufferization opportunities.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<func::FuncOp>(createSCFForLoopCanonicalizationPass());
    // Run empty-tensor lowering at function scope before bufferization.
    pm.addNestedPass<func::FuncOp>(
        bufferization::createEmptyTensorToAllocTensorPass());

    bufferization::OneShotBufferizationOptions opts;
    opts.bufferizeFunctionBoundaries = true;
    opts.allowUnknownOps = true;
    opts.allowReturnAllocs = true;

    pm.addPass(bufferization::createOneShotBufferizePass(opts));
    pm.addPass(bufferization::createFinalizingBufferizePass());

    // Post-cleanup to fold away leftover tensor ops.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(runPipeline(pm, module))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir {
namespace stablehlo {

void registerOneShotBufferizePass() {
  PassRegistration<OneShotBufferizePass>();
}

}  // namespace stablehlo
}  // namespace mlir
