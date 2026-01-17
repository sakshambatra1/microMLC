#!/usr/bin/env bash
# Benchmark all final IR stages by injecting a bench+read_sink driver,
# lowering to LLVM dialect, and running mlir-cpu-runner.
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPT_BIN="${ROOT}/third_party/stablehlo/build/bin/stablehlo-opt"
LLVM_PREFIX=${LLVM_PREFIX:-$(brew --prefix llvm@17 2>/dev/null || brew --prefix llvm)}
LLVM_LIBDIR=$("$LLVM_PREFIX"/bin/llvm-config --libdir)
LIBOMP="${LLVM_PREFIX}/lib/libomp.dylib"
ITERS=${ITERS:-100}

[[ -x "${OPT_BIN}" ]] || { echo "stablehlo-opt not found at ${OPT_BIN}"; exit 1; }

OMP_ARG=""
if [[ -f "${LIBOMP}" ]]; then
  OMP_ARG="-shared-libs=${LIBOMP}"
fi

STAGES=(
  "attn_stage0_linalg"
  "attn_stage1_fused"
  "attn_stage2_tiled_tensor"
  "attn_stage3_bufferized"
  "attn_stage5_vectorized"
)

make_tensor_snippet() {
  cat > /tmp/bench_snippet.mlir <<EOF
  memref.global "public" @bench_sink : memref<f32> = dense<0.0>
  func.func @bench() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cI = arith.constant ${ITERS} : index
    %cst1 = arith.constant 1.0e-3 : f32
    %cst2 = arith.constant 2.0e-3 : f32
    %cst3 = arith.constant 3.0e-3 : f32
    %A = tensor.generate  {
    ^bb0(%i: index, %j: index, %k: index):
      %ii = arith.index_cast %i : index to i64
      %ji = arith.index_cast %j : index to i64
      %ki = arith.index_cast %k : index to i64
      %fi = arith.sitofp %ii : i64 to f32
      %fj = arith.sitofp %ji : i64 to f32
      %fk = arith.sitofp %ki : i64 to f32
      %vi = arith.mulf %fi, %cst1 : f32
      %vj = arith.mulf %fj, %cst2 : f32
      %vk = arith.mulf %fk, %cst3 : f32
      %s1 = arith.addf %vi, %vj : f32
      %s2 = arith.addf %s1, %vk : f32
      tensor.yield %s2 : f32
    } : tensor<1x128x128xf32>
    %B = tensor.generate  {
    ^bb0(%i: index, %j: index, %k: index):
      %ii = arith.index_cast %i : index to i64
      %ji = arith.index_cast %j : index to i64
      %ki = arith.index_cast %k : index to i64
      %fi = arith.sitofp %ii : i64 to f32
      %fj = arith.sitofp %ji : i64 to f32
      %fk = arith.sitofp %ki : i64 to f32
      %vi = arith.mulf %fi, %cst3 : f32
      %vj = arith.mulf %fj, %cst1 : f32
      %vk = arith.mulf %fk, %cst2 : f32
      %s1 = arith.addf %vi, %vj : f32
      %s2 = arith.addf %s1, %vk : f32
      tensor.yield %s2 : f32
    } : tensor<1x128x128xf32>
    %C = tensor.generate  {
    ^bb0(%i: index, %j: index, %k: index):
      %ii = arith.index_cast %i : index to i64
      %ji = arith.index_cast %j : index to i64
      %ki = arith.index_cast %k : index to i64
      %fi = arith.sitofp %ii : i64 to f32
      %fj = arith.sitofp %ji : i64 to f32
      %fk = arith.sitofp %ki : i64 to f32
      %vi = arith.mulf %fi, %cst2 : f32
      %vj = arith.mulf %fj, %cst3 : f32
      %vk = arith.mulf %fk, %cst1 : f32
      %s1 = arith.addf %vi, %vj : f32
      %s2 = arith.addf %s1, %vk : f32
      tensor.yield %s2 : f32
    } : tensor<1x128x128xf32>
    %D = tensor.generate  {
    ^bb0(%i: index, %j: index, %k: index):
      %ii = arith.index_cast %i : index to i64
      %ji = arith.index_cast %j : index to i64
      %ki = arith.index_cast %k : index to i64
      %fi = arith.sitofp %ii : i64 to f32
      %fj = arith.sitofp %ji : i64 to f32
      %fk = arith.sitofp %ki : i64 to f32
      %vi = arith.mulf %fi, %cst1 : f32
      %vj = arith.mulf %fj, %cst2 : f32
      %vk = arith.mulf %fk, %cst3 : f32
      %s1 = arith.addf %vi, %vj : f32
      %s2 = arith.addf %s1, %vk : f32
      tensor.yield %s2 : f32
    } : tensor<1x128x128xf32>
    %sink = memref.get_global @bench_sink : memref<f32>
    scf.for %t = %c0 to %cI step %c1 {
      %res = func.call @main(%A, %B, %C, %D)
          : (tensor<1x128x128xf32>, tensor<1x128x128xf32>,
             tensor<1x128x128xf32>, tensor<1x128x128xf32>)
            -> tensor<1x128x128xf32>
      %e = tensor.extract %res[%c0, %c0, %c0] : tensor<1x128x128xf32>
      memref.store %e, %sink[] : memref<f32>
    }
    func.return
  }
  func.func @read_sink() -> f32 {
    func.call @bench() : () -> ()
    %sink = memref.get_global @bench_sink : memref<f32>
    %v = memref.load %sink[] : memref<f32>
    func.return %v : f32
  }
EOF
}

make_memref_snippet() {
  cat > /tmp/bench_snippet.mlir <<EOF
  memref.global "public" @bench_sink : memref<f32> = dense<0.0>
  func.func @bench() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cI = arith.constant ${ITERS} : index
    %zero = arith.constant 0.0 : f32
    %A = memref.alloc() : memref<1x128x128xf32>
    %B = memref.alloc() : memref<1x128x128xf32>
    %C = memref.alloc() : memref<1x128x128xf32>
    %D = memref.alloc() : memref<1x128x128xf32>
    linalg.fill ins(%zero : f32) outs(%A : memref<1x128x128xf32>)
    linalg.fill ins(%zero : f32) outs(%B : memref<1x128x128xf32>)
    linalg.fill ins(%zero : f32) outs(%C : memref<1x128x128xf32>)
    linalg.fill ins(%zero : f32) outs(%D : memref<1x128x128xf32>)
    %Acast = memref.cast %A : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
    %Bcast = memref.cast %B : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
    %Ccast = memref.cast %C : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
    %Dcast = memref.cast %D : memref<1x128x128xf32> to memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>
    %sink = memref.get_global @bench_sink : memref<f32>
    scf.for %t = %c0 to %cI step %c1 {
      %res = func.call @main(%Acast, %Bcast, %Ccast, %Dcast)
          : (memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>,
             memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>,
             memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>,
             memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>)
            -> memref<1x128x128xf32>
      %e = memref.load %res[%c0, %c0, %c0] : memref<1x128x128xf32>
      memref.store %e, %sink[] : memref<f32>
      memref.dealloc %res : memref<1x128x128xf32>
    }
    memref.dealloc %D : memref<1x128x128xf32>
    memref.dealloc %C : memref<1x128x128xf32>
    memref.dealloc %B : memref<1x128x128xf32>
    memref.dealloc %A : memref<1x128x128xf32>
    func.return
  }
  func.func @read_sink() -> f32 {
    func.call @bench() : () -> ()
    %sink = memref.get_global @bench_sink : memref<f32>
    %v = memref.load %sink[] : memref<f32>
    func.return %v : f32
  }
EOF
}

# Lowering pipeline (from bench_attn.sh)
if "${OPT_BIN}" --help 2>&1 | rg -q "convert-index-to-llvm"; then
  LOWER_TENSOR=(
    -canonicalize -cse
    -empty-tensor-to-alloc-tensor
    "-one-shot-bufferize=bufferize-function-boundaries allow-return-allocs"
    -convert-bufferization-to-memref
    -convert-linalg-to-loops
    -lower-affine
    -convert-vector-to-scf
    -canonicalize -cse
    -fix-unrealized-index-casts
    -lower-affine
    -canonicalize -cse
    -convert-scf-to-cf
    -canonicalize -cse
    -cf-integerize-index-block-args
    -convert-index-to-llvm
    -convert-math-to-llvm
    -convert-arith-to-llvm
    -fix-unrealized-index-casts
    -vector-bufferize
    -convert-index-to-llvm
    -convert-vector-to-llvm
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -convert-cf-to-llvm
    -convert-func-to-llvm
    -canonicalize
    -lower-affine
    -canonicalize
    -fix-unrealized-index-casts
    -convert-index-to-llvm
    -convert-arith-to-llvm
    -reconcile-unrealized-casts
    -mlir-print-op-on-diagnostic
  )
  LOWER_MEMREF=(
    -convert-linalg-to-loops
    -lower-affine
    -convert-vector-to-scf
    -canonicalize -cse
    -fix-unrealized-index-casts
    -lower-affine
    -canonicalize -cse
    -convert-scf-to-cf
    -canonicalize -cse
    -cf-integerize-index-block-args
    -convert-index-to-llvm
    -convert-math-to-llvm
    -convert-arith-to-llvm
    -fix-unrealized-index-casts
    -vector-bufferize
    -convert-index-to-llvm
    -convert-vector-to-llvm
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -convert-cf-to-llvm
    -convert-func-to-llvm
    -canonicalize
    -lower-affine
    -canonicalize
    -fix-unrealized-index-casts
    -convert-index-to-llvm
    -convert-arith-to-llvm
    -reconcile-unrealized-casts
    -mlir-print-op-on-diagnostic
  )
else
  echo "convert-index-to-llvm not found in ${OPT_BIN}; falling back to mlir-opt is not supported in this script." >&2
  exit 1
fi

for stage in "${STAGES[@]}"; do
  SRC="${ROOT}/ir/${stage}.mlir"
  BENCH="/tmp/${stage}_bench.mlir"
  OUTLLVM="/tmp/${stage}_llvm.mlir"

  [[ -f "${SRC}" ]] || { echo "missing ${SRC}" >&2; exit 1; }

  if rg -n "func.func public @main\\([^)]*tensor<" "${SRC}" >/dev/null; then
    make_tensor_snippet
    PIPE=("${LOWER_TENSOR[@]}")
  else
    make_memref_snippet
    PIPE=("${LOWER_MEMREF[@]}")
  fi

  python3 - <<PY
from pathlib import Path
src = Path("${SRC}").read_text()
bench = Path("/tmp/bench_snippet.mlir").read_text()
out = src.replace("module {", "module {\\n" + bench + "\\n", 1)
Path("${BENCH}").write_text(out)
PY

  # Normalize any stray unrealized_conversion_casts (i64/index).
  python3 - <<PY
from pathlib import Path
import re

def fix_casts(text: str) -> str:
    text = re.sub(
        r'(%[\\w\\d]+)\\s*=\\s*builtin\\.unrealized_conversion_cast\\s+(%[\\w\\d]+)\\s*:\\s*i64\\s+to\\s+index',
        r'\\1 = arith.index_cast \\2 : i64 to index',
        text,
    )
    text = re.sub(
        r'(%[\\w\\d]+)\\s*=\\s*builtin\\.unrealized_conversion_cast\\s+(%[\\w\\d]+)\\s*:\\s*index\\s+to\\s+i64',
        r'\\1 = arith.index_cast \\2 : index to i64',
        text,
    )
    return text

p = Path("${BENCH}")
p.write_text(fix_casts(p.read_text()))
PY

  "${OPT_BIN}" "${BENCH}" "${PIPE[@]}" -o "${OUTLLVM}"

  # Run once to ensure it executes.
  mlir-cpu-runner "${OUTLLVM}" -O3 -e read_sink -entry-point-result=f32 \
    -shared-libs="${LLVM_LIBDIR}/libmlir_runner_utils.dylib" \
    -shared-libs="${LLVM_LIBDIR}/libmlir_c_runner_utils.dylib" \
    -shared-libs="${LLVM_LIBDIR}/libmlir_async_runtime.dylib" \
    ${OMP_ARG}

  if command -v hyperfine >/dev/null 2>&1; then
    hyperfine --warmup 1 \
      "mlir-cpu-runner ${OUTLLVM} -O3 -e bench -entry-point-result=void \
      -shared-libs=${LLVM_LIBDIR}/libmlir_runner_utils.dylib \
      -shared-libs=${LLVM_LIBDIR}/libmlir_c_runner_utils.dylib \
      -shared-libs=${LLVM_LIBDIR}/libmlir_async_runtime.dylib \
      ${OMP_ARG}"
  fi

  echo "Memory op counts for ${stage}:"
  rg -c "memref.copy" "${BENCH}" || true
  rg -c "memref.load" "${BENCH}" || true
  rg -c "memref.store" "${BENCH}" || true
  rg -c "vector.transfer_read" "${BENCH}" || true
  rg -c "vector.transfer_write" "${BENCH}" || true
done
