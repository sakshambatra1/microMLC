#ifndef STABLEHLO_TRANSFORMS_TILINGPASS_H
#define STABLEHLO_TRANSFORMS_TILINGPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stablehlo {

// The old manual pass (you can keep or remove this)
void registerTilingPasses();

// ADD THIS LINE:
void registerLinalgTilingPass(); 

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TRANSFORMS_TILINGPASS_H