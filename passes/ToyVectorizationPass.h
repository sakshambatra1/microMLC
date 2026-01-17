#ifndef STABLEHLO_TRANSFORMS_TOYVECTORIZATIONPASS_H
#define STABLEHLO_TRANSFORMS_TOYVECTORIZATIONPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stablehlo {

std::unique_ptr<Pass> createToyVectorizationPass();
void registerToyVectorizationPass();

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TRANSFORMS_TOYVECTORIZATIONPASS_H
