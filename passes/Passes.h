/* Copyright 2022 The StableHLO Authors.
   ... (License text) ...
==============================================================================*/

#ifndef STABLEHLO_TRANSFORMS_PASSES_H
#define STABLEHLO_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace stablehlo {

// Forward declaration needed by generated registration.
std::unique_ptr<Pass> createCfIntegerizeIndexBlockArgsPass();

// --- TableGen Generated Passes ---
#define GEN_PASS_DECL_STABLEHLOCANONICALIZEDYNAMISMPASS
#define GEN_PASS_DECL_STABLEHLOLEGALIZETOVHLOPASS
#define GEN_PASS_DECL_STABLEHLOREFINESHAPESPASS
#define GEN_PASS_DECL_VHLOLEGALIZETOSTABLEHLOPASS
#define GEN_PASS_DECL_VHLOTOVERSIONPASS
#define GEN_PASS_DECL_CFINTEGERIZEINDEXBLOCKARGSPASS
#define GEN_PASS_DECL_FIXUNREALIZEDINDEXCASTSPASS
#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/Passes.h.inc"

// --- Manual Pass Declarations (Add these lines) ---

// Factory method to create the GenericFusionPass
std::unique_ptr<Pass> createGenericFusionPass();

// Registration method to expose the GenericFusionPass to the CLI
void registerGenericFusionPass();

// Final MVP parallel scheduling pass factory/registration
std::unique_ptr<Pass> createFinalMVPParallelSchedulingPass();
void registerFinalMVPParallelSchedulingPass();

// Toy vectorization pass factory/registration
std::unique_ptr<Pass> createToyVectorizationPass();
void registerToyVectorizationPass();

// Fix unrealized_conversion_cast iN -> index
std::unique_ptr<Pass> createFixUnrealizedIndexCastsPass();
void registerFixUnrealizedIndexCastsPass();

// CF index block args -> i64
std::unique_ptr<Pass> createCfIntegerizeIndexBlockArgsPass();
void registerCfIntegerizeIndexBlockArgsPass();

// Registration method for the One-Shot Bufferization pipeline
void registerOneShotBufferizePass();

// --------------------------------------------------

// Populates StableHLO ops to VHLO ops rewriting patterns.
void populateStablehloToVhloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates VHLO ops to StableHLO ops rewriting patterns.
void populateVhloToStablehloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates VHLO downgrade rewriting patterns.
void populateVhloToVersionPatterns(RewritePatternSet *patterns,
                                   TypeConverter *converter,
                                   MLIRContext *contexts);
}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_VHLO_OPS_H
