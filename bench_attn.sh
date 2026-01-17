#!/usr/bin/env bash
set -euo pipefail

SINK_MODE=${SINK_MODE:-global}  # global | alloc
CHECK=${CHECK:-0}               # 1 to run correctness check
ITERS=${ITERS:-100}             # iterations of @main in bench

BASE=ir/attn_complete.mlir
OPT=ir/attention_pipeline.mlir
LLVM_PREFIX=${LLVM_PREFIX:-$(brew --prefix llvm@17 2>/dev/null || brew --prefix llvm)}
LLVM_LIBDIR=$("$LLVM_PREFIX"/bin/llvm-config --libdir)

# Bump stack to avoid JITed vector scratch hitting guard pages on macOS.
ulimit -s 65532 || true

# ------------------------------------------------------------------
# Bench snippet (written to file). Two modes:
#  - global: uses a global sink and read_sink loads it
#  - alloc:  uses an alloc sink; read_sink calls bench then returns 0.0
# ------------------------------------------------------------------
if [[ "$SINK_MODE" == "global" ]]; then
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
else
cat > /tmp/bench_snippet.mlir <<EOF
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
    %sink = memref.alloc() : memref<f32>
    scf.for %t = %c0 to %cI step %c1 {
      %res = func.call @main(%A, %B, %C, %D)
          : (tensor<1x128x128xf32>, tensor<1x128x128xf32>,
             tensor<1x128x128xf32>, tensor<1x128x128xf32>)
            -> tensor<1x128x128xf32>
      %e = tensor.extract %res[%c0, %c0, %c0] : tensor<1x128x128xf32>
      memref.store %e, %sink[] : memref<f32>
    }
    memref.dealloc %sink : memref<f32>
    func.return
  }
  func.func @read_sink() -> f32 {
    func.call @bench() : () -> ()
    %z = arith.constant 0.0 : f32
    func.return %z : f32
  }
EOF
fi

# ------------------------------------------------------------------
# Inject bench into baseline and optimized
# ------------------------------------------------------------------
python3 - <<'PY'
from pathlib import Path
src = Path("ir/attn_complete.mlir").read_text()
bench = Path("/tmp/bench_snippet.mlir").read_text()
out = src.replace("module {", "module {\n" + bench + "\n", 1)
Path("/tmp/attn_baseline_bench.mlir").write_text(out)
PY

python3 - <<'PY'
from pathlib import Path
src = Path("ir/attention_pipeline.mlir").read_text()
bench = Path("/tmp/bench_snippet.mlir").read_text()
out = src.replace("module {", "module {\n" + bench + "\n", 1)
Path("/tmp/attn_opt_bench.mlir").write_text(out)
PY

# ------------------------------------------------------------------
# Normalize any stray unrealized_conversion_casts (i64/index) before lowering.
# This prevents later LLVM conversions from bailing out on illegal casts.
# ------------------------------------------------------------------
python3 - <<'PY'
from pathlib import Path
import re

def fix_casts(text: str) -> str:
    text = re.sub(
        r'(%[\w\d]+)\s*=\s*builtin\.unrealized_conversion_cast\s+(%[\w\d]+)\s*:\s*i64\s+to\s+index',
        r'\1 = arith.index_cast \2 : i64 to index',
        text,
    )
    text = re.sub(
        r'(%[\w\d]+)\s*=\s*builtin\.unrealized_conversion_cast\s+(%[\w\d]+)\s*:\s*index\s+to\s+i64',
        r'\1 = arith.index_cast \2 : index to i64',
        text,
    )
    return text

for path in ["/tmp/attn_baseline_bench.mlir", "/tmp/attn_opt_bench.mlir"]:
    p = Path(path)
    p.write_text(fix_casts(p.read_text()))
PY

# ------------------------------------------------------------------
# Pre-bufferize checks (vector.contract presence)
# ------------------------------------------------------------------
if command -v rg >/dev/null 2>&1; then
  rg "vector\\.contract" /tmp/attn_baseline_bench.mlir || true
  rg "vector\\.contract" /tmp/attn_opt_bench.mlir || true
else
  grep -n "vector.contract" /tmp/attn_baseline_bench.mlir || true
  grep -n "vector.contract" /tmp/attn_opt_bench.mlir || true
fi

# ------------------------------------------------------------------
# Lowering pipeline (same for both), retry with swapped vec/linalg order.
# Ensure SCF is eliminated right before LLVM lowering: canonicalize/cse -> convert-scf-to-cf -> canonicalize/cse,
# then convert-index-to-llvm before convert-arith-to-llvm.
# ------------------------------------------------------------------
if third_party/stablehlo/build/bin/stablehlo-opt --help | grep -q "convert-index-to-llvm"; then
  LOWER_BASE=(
    -canonicalize
    -cse
    -empty-tensor-to-alloc-tensor
    "-one-shot-bufferize=bufferize-function-boundaries allow-return-allocs"
    -convert-bufferization-to-memref
    -convert-linalg-to-loops
    -lower-affine
    -convert-vector-to-scf
    -canonicalize
    -cse
    -fix-unrealized-index-casts
    -lower-affine
    -canonicalize
    -cse
    -convert-scf-to-cf
    -canonicalize
    -cse
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
  LOWER_BASE=(
    -canonicalize
    -cse
    -empty-tensor-to-alloc-tensor
    "-one-shot-bufferize=bufferize-function-boundaries allow-return-allocs"
    -convert-bufferization-to-memref
    -convert-linalg-to-loops
    -lower-affine
    -convert-vector-to-scf
    -canonicalize
    -cse
    -fix-unrealized-index-casts
    -lower-affine
    -canonicalize
    -cse
    -convert-scf-to-cf
    -canonicalize
    -cse
    -cf-integerize-index-block-args
    -vector-bufferize
    -convert-math-to-llvm
    -convert-arith-to-llvm
    -fix-unrealized-index-casts
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
    -convert-arith-to-llvm
    -reconcile-unrealized-casts
    -mlir-print-op-on-diagnostic
  )
fi
LOWER_SWAP=("${LOWER_BASE[@]}")

try_lower() {
  local infile=$1 outfile=$2
  if third_party/stablehlo/build/bin/stablehlo-opt "$infile" "${LOWER_BASE[@]}" -o "$outfile" 2> /tmp/lower_err.txt; then
    echo "Lowered with order (vector before linalg): $outfile"
    return 0
  fi
  echo "First lowering failed; tail of stderr:"
  tail -n 60 /tmp/lower_err.txt || true
  echo "Retry lowering with swapped order (linalg before vector)..."
  if third_party/stablehlo/build/bin/stablehlo-opt "$infile" "${LOWER_SWAP[@]}" -o "$outfile"; then
    echo "Lowered with swapped order: $outfile"
    return 0
  fi
  return 1
}

try_lower /tmp/attn_baseline_bench.mlir /tmp/baseline.mlir
try_lower /tmp/attn_opt_bench.mlir /tmp/opt.mlir

# ------------------------------------------------------------------
# Stamp target triple/data layout for the JIT (arm64 macOS host).
# ------------------------------------------------------------------
python3 - <<'PY'
from pathlib import Path
targets = [Path("/tmp/baseline.mlir"), Path("/tmp/opt.mlir")]
layout = 'e-m:o-i64:64-i128:128-n32:64-S128'
triple = 'arm64-apple-macosx14.5.0'
for p in targets:
    if not p.exists():
        continue
    text = p.read_text()
    if 'llvm.target_triple' in text and layout in text:
        continue
    def repl(line: str) -> str:
        # Replace empty data layout if present.
        if 'llvm.data_layout = ""' in line:
            line = line.replace('llvm.data_layout = ""',
                                f'llvm.data_layout = "{layout}"')
        if 'llvm.target_triple' not in line:
            line = line.replace('module attributes {',
                                f'module attributes {{llvm.target_triple = "{triple}", ',
                                1)
        return line
    text = '\n'.join(repl(l) if l.strip().startswith('module attributes') else l
                     for l in text.splitlines())
    p.write_text(text)
PY

# ------------------------------------------------------------------
# Optional correctness check
# ------------------------------------------------------------------
LIBOMP_FALLBACK="$LLVM_PREFIX/lib/libomp.dylib"
LIBOMP_ALT="$(brew --prefix llvm@17 2>/dev/null)/lib/libomp.dylib"
if [[ -f "$LIBOMP_ALT" ]]; then
  LIBOMP="$LIBOMP_ALT"
else
  LIBOMP="$LIBOMP_FALLBACK"
fi

if [[ "$CHECK" == "1" ]]; then
  echo "Baseline read_sink:"
  mlir-cpu-runner /tmp/baseline.mlir -O3 -e read_sink -entry-point-result=f32 \
    -shared-libs=$LLVM_LIBDIR/libmlir_runner_utils.dylib \
    -shared-libs=$LLVM_LIBDIR/libmlir_c_runner_utils.dylib \
    -shared-libs=$LLVM_LIBDIR/libmlir_async_runtime.dylib \
    -shared-libs=$LIBOMP
  echo "Opt read_sink:"
  mlir-cpu-runner /tmp/opt.mlir -O3 -e read_sink -entry-point-result=f32 \
    -shared-libs=$LLVM_LIBDIR/libmlir_runner_utils.dylib \
    -shared-libs=$LLVM_LIBDIR/libmlir_c_runner_utils.dylib \
    -shared-libs=$LLVM_LIBDIR/libmlir_async_runtime.dylib \
    -shared-libs=$LIBOMP
fi

# ------------------------------------------------------------------
# Benchmark (identical runner invocations)
# ------------------------------------------------------------------
hyperfine --warmup 1 \
  "mlir-cpu-runner /tmp/baseline.mlir -O3 -e bench -entry-point-result=void \
     -shared-libs=$LLVM_LIBDIR/libmlir_runner_utils.dylib \
     -shared-libs=$LLVM_LIBDIR/libmlir_c_runner_utils.dylib \
     -shared-libs=$LLVM_LIBDIR/libmlir_async_runtime.dylib \
     -shared-libs=$LIBOMP" \
  "mlir-cpu-runner /tmp/opt.mlir -O3 -e bench -entry-point-result=void \
     -shared-libs=$LLVM_LIBDIR/libmlir_runner_utils.dylib \
     -shared-libs=$LLVM_LIBDIR/libmlir_c_runner_utils.dylib \
     -shared-libs=$LLVM_LIBDIR/libmlir_async_runtime.dylib \
     -shared-libs=$LIBOMP"

# ------------------------------------------------------------------
# Optional SIMD inspection
# ------------------------------------------------------------------
mlir-translate --mlir-to-llvmir /tmp/opt.mlir | tee /tmp/opt.ll
if command -v rg >/dev/null 2>&1; then rg "vfmadd|vmulps|llvm\\.intr" /tmp/opt.ll || true; else grep -nE "vfmadd|vmulps|llvm\\.intr" /tmp/opt.ll || true; fi
mlir-translate --mlir-to-llvmir /tmp/opt.mlir | llc -O3 -o /tmp/opt.s
if command -v rg >/dev/null 2>&1; then rg "vfmadd|vmulps|vfma" /tmp/opt.s || true; else grep -nE "vfmadd|vmulps|vfma" /tmp/opt.s || true; fi
