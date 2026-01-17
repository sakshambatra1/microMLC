#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT}/passes"
DST_DIR="${ROOT}/third_party/stablehlo/stablehlo/transforms"

FILES=(
  CMakeLists.txt
  Passes.h
  Passes.td
  CfIntegerizeIndexBlockArgsPass.cpp
  FixUnrealizedIndexCastsPass.cpp
  GenericFusionPass.cpp
  LinalgTilingPass.cpp
  OneShotBufferizePass.cpp
  ParallelSchedulingPassAttn.cpp
  TilingPass.cpp
  TilingPass.h
  ToyVectorizationPass.cpp
  ToyVectorizationPass.h
)

[[ -d "${SRC_DIR}" ]] || { echo "Missing passes dir: ${SRC_DIR}" >&2; exit 1; }
[[ -d "${DST_DIR}" ]] || { echo "Missing stablehlo transforms dir: ${DST_DIR}" >&2; exit 1; }

for f in "${FILES[@]}"; do
  src="${SRC_DIR}/${f}"
  [[ -f "${src}" ]] || { echo "Missing ${src}" >&2; exit 1; }
  cp -f "${src}" "${DST_DIR}/"
done

echo "Synced passes into ${DST_DIR}"
