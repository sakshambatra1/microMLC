#!/usr/bin/env bash
# Lower each staged IR to LLVM dialect/IR and print simple memory op counts.
# Outputs:
#   /tmp/attn_stage{0..5}_llvm.mlir  (LLVM dialect)
#   /tmp/attn_stage{0..5}_llvm.ll    (LLVM IR)
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPT_BIN="${ROOT}/third_party/stablehlo/build/bin/stablehlo-opt"
TRANSLATE_BIN="/opt/homebrew/opt/llvm@17/bin/mlir-translate"
STAGES=(
  "attn_stage0_linalg"
  "attn_stage1_fused"
  "attn_stage2_tiled_tensor"
  "attn_stage3_bufferized"
  "attn_stage5_vectorized"
)
OUTDIR="${ROOT}/ir"

[[ -x "${OPT_BIN}" ]] || { echo "stablehlo-opt not found at ${OPT_BIN}"; exit 1; }
[[ -x "${TRANSLATE_BIN}" ]] || { echo "mlir-translate not found at ${TRANSLATE_BIN}"; exit 1; }

for stage in "${STAGES[@]}"; do
  SRC="${OUTDIR}/${stage}.mlir"
  LLVM_MLIR="/tmp/${stage}_llvm.mlir"
  LLVM_LL="/tmp/${stage}_llvm.ll"
  [[ -f "${SRC}" ]] || { echo "missing ${SRC}" >&2; exit 1; }

  "${OPT_BIN}" "${SRC}" \
    -fix-unrealized-index-casts \
    -convert-linalg-to-loops \
    -lower-affine \
    -convert-scf-to-cf \
    -convert-index-to-llvm \
    -convert-math-to-llvm \
    -convert-arith-to-llvm \
    -convert-vector-to-llvm \
    -convert-func-to-llvm \
    -finalize-memref-to-llvm \
    -reconcile-unrealized-casts \
    -o "${LLVM_MLIR}"

  "${TRANSLATE_BIN}" --mlir-to-llvmir "${LLVM_MLIR}" -o "${LLVM_LL}"

  echo "Stats for ${stage}:"
  rg -c "memref.copy" "${SRC}" || true
  rg -c "memref.load" "${SRC}" || true
  rg -c "memref.store" "${SRC}" || true
  rg -c "vector.transfer_read" "${SRC}" || true
  rg -c "vector.transfer_write" "${SRC}" || true
done

echo "Lowered LLVM files are in /tmp/attn_stage*_llvm.{mlir,ll}"
