#!/usr/bin/env bash
# Split pipeline: keep named matmuls, fuse epilogues on tensors, tile on tensors
# (L2=64, L1=4), bufferize, then vectorize. Promotion is disabled.
set -euo pipefail
set -x

OPT_BIN="third_party/stablehlo/build/bin/stablehlo-opt"
IR_DIR="ir"
INPUT="${IR_DIR}/attn_complete.mlir"

OUT0="${IR_DIR}/attn_stage0_linalg.mlir"
OUT1="${IR_DIR}/attn_stage1_fused.mlir"
OUT2="${IR_DIR}/attn_stage2_tiled_tensor.mlir"
OUT3="${IR_DIR}/attn_stage3_bufferized.mlir"
OUT5="${IR_DIR}/attn_stage5_vectorized.mlir"

[[ -x "${OPT_BIN}" ]] || { echo "stablehlo-opt not found at ${OPT_BIN}"; exit 1; }
mkdir -p "${IR_DIR}"
[[ -f "${INPUT}" ]] || { echo "Input MLIR missing at ${INPUT}"; exit 1; }

# Stage 0: initial stablehlo cleanup
"${OPT_BIN}" \
  -stablehlo-tiling \
  -canonicalize -cse \
  "${INPUT}" -o "${OUT0}"

# Stage 1: fuse epilogues while keeping named matmuls
"${OPT_BIN}" \
  -generic-fusion \
  -canonicalize -cse \
  "${OUT0}" -o "${OUT1}"

# Stage 2: tensor tiling (no promotion) with L1=4 for small vectors
"${OPT_BIN}" \
  "-stablehlo-linalg-tiling=l2-tile-sizes=64,64,64 l1-tile-sizes=4,4,4 enable-promotion=false" \
  -canonicalize -cse \
  "${OUT1}" -o "${OUT2}"

# Stage 3: bufferize to memrefs
"${OPT_BIN}" \
  -canonicalize -cse \
  -empty-tensor-to-alloc-tensor \
  "-one-shot-bufferize=bufferize-function-boundaries allow-return-allocs" \
  -convert-bufferization-to-memref \
  -canonicalize -cse \
  "${OUT2}" -o "${OUT3}"

# Stage 4 removed: vectorize directly after bufferization
"${OPT_BIN}" \
  "-toy-vectorize=vector-width=4 enable-reductions" \
  -canonicalize -cse \
  "${OUT3}" -o "${OUT5}"

echo "Pipeline completed. Outputs:"
echo "  ${OUT0}"
echo "  ${OUT1}"
echo "  ${OUT2}"
echo "  ${OUT3}"
echo "  ${OUT5}"
