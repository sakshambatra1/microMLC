#!/usr/bin/env bash
# Build and run a runnable driver for the final stage (memref-based) IR.
# Produces:
#   /tmp/attn_stage5_driver.mlir  (adds @driver)
#   /tmp/attn_stage5_llvm.mlir    (LLVM dialect)
#   /tmp/attn_stage5_llvm.ll      (LLVM IR)
# Then runs mlir-cpu-runner on @driver.
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPT_BIN="${ROOT}/third_party/stablehlo/build/bin/stablehlo-opt"
TRANSLATE_BIN="/opt/homebrew/opt/llvm@17/bin/mlir-translate"
RUNNER_BIN="/opt/homebrew/opt/llvm@17/bin/mlir-cpu-runner"
SRC="${ROOT}/ir/attn_stage5_vectorized.mlir"

[[ -x "${OPT_BIN}" ]] || { echo "stablehlo-opt not found at ${OPT_BIN}"; exit 1; }
[[ -x "${TRANSLATE_BIN}" ]] || { echo "mlir-translate not found at ${TRANSLATE_BIN}"; exit 1; }
[[ -x "${RUNNER_BIN}" ]] || { echo "mlir-cpu-runner not found at ${RUNNER_BIN}"; exit 1; }
[[ -f "${SRC}" ]] || { echo "missing ${SRC}" >&2; exit 1; }

DRIVER=/tmp/attn_stage5_driver.mlir
STEP1=/tmp/attn_stage5_step1.mlir
LLVM_MLIR=/tmp/attn_stage5_llvm.mlir
LLVM_LL=/tmp/attn_stage5_llvm.ll

python3 - <<PY
from pathlib import Path
src = Path("${SRC}")
lines = src.read_text().splitlines()
# Drop the last line that looks like a closing module brace.
while lines and lines[-1].strip() == "":
    lines.pop()
if lines and lines[-1].strip() == "}":
    lines.pop()
Path("${DRIVER}").write_text("\n".join(lines) + "\n")
PY
cat >> "${DRIVER}" <<'EOF'

// Simple driver: alloc/fill zero, call main, return one element.
func.func @driver() -> f32 {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %A = memref.alloc() : memref<1x128x128xf32>
  %B = memref.alloc() : memref<1x128x128xf32>
  %C = memref.alloc() : memref<1x128x128xf32>
  %D = memref.alloc() : memref<1x128x128xf32>
  %Acast = memref.cast %A : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
  %Bcast = memref.cast %B : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
  %Ccast = memref.cast %C : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
  %Dcast = memref.cast %D : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
  linalg.fill ins(%zero : f32) outs(%A : memref<1x128x128xf32>)
  linalg.fill ins(%zero : f32) outs(%B : memref<1x128x128xf32>)
  linalg.fill ins(%zero : f32) outs(%C : memref<1x128x128xf32>)
  linalg.fill ins(%zero : f32) outs(%D : memref<1x128x128xf32>)
  %out = func.call @main(%Acast, %Bcast, %Ccast, %Dcast)
      : (memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>)
        -> memref<1x128x128xf32>
  %val = memref.load %out[%c0, %c0, %c0] : memref<1x128x128xf32>
  memref.dealloc %out : memref<1x128x128xf32>
  memref.dealloc %D : memref<1x128x128xf32>
  memref.dealloc %C : memref<1x128x128xf32>
  memref.dealloc %B : memref<1x128x128xf32>
  memref.dealloc %A : memref<1x128x128xf32>
  func.return %val : f32
}
}
EOF

# Stage 1: lower through index-to-llvm (this may introduce unrealized casts).
"${OPT_BIN}" "${DRIVER}" \
  -fix-unrealized-index-casts \
  -convert-linalg-to-loops \
  -lower-affine \
  -cf-integerize-index-block-args \
  -convert-scf-to-cf \
  -convert-index-to-llvm \
  -reconcile-unrealized-casts \
  -o "${STEP1}"

# Rewrite any remaining unrealized casts after index lowering.
python3 - <<PY
from pathlib import Path
path = Path("${STEP1}")
txt = path.read_text()
txt = txt.replace("builtin.unrealized_conversion_cast", "arith.index_cast")
# Crude fallback: convert remaining index types to i64 to placate lowering.
txt = txt.replace("index", "i64")
path.write_text(txt)
PY

# Stage 2: finish lowering to LLVM dialect.
"${OPT_BIN}" "${STEP1}" \
  -convert-index-to-llvm \
  -fix-unrealized-index-casts \
  -canonicalize \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -convert-vector-to-llvm \
  -convert-func-to-llvm \
  -finalize-memref-to-llvm \
  -reconcile-unrealized-casts \
  -o "${LLVM_MLIR}"

"${TRANSLATE_BIN}" --mlir-to-llvmir "${LLVM_MLIR}" -o "${LLVM_LL}"

"${RUNNER_BIN}" "${LLVM_LL}" -O3 -e driver -entry-point-result=f32 \
  -shared-libs=$(/opt/homebrew/opt/llvm@17/bin/llvm-config --libdir)/libmlir_runner_utils.dylib \
  -shared-libs=$(/opt/homebrew/opt/llvm@17/bin/llvm-config --libdir)/libmlir_c_runner_utils.dylib \
  -shared-libs=$(/opt/homebrew/opt/llvm@17/bin/llvm-config --libdir)/libmlir_async_runtime.dylib \
  -shared-libs=/opt/homebrew/opt/llvm/lib/libomp.dylib
