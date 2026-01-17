# toy-transformer Compiler Playground

A playground for exploring MLIR/StableHLO compiler passes on transformer workloads, with focus on tiling, fusion, bufferization, and vectorization optimizations.

## Repository Layout

```
toy-transformer/
├── passes/                 # Custom MLIR pass implementations
├── ir/                     # Input and staged output IRs
├── tools/                  # Build, sync, and benchmarking scripts
├── transformer/python/     # Transformer modeling code and IR generators
├── docs/                   # Design notes and performance analysis
├── experiments/            # Test cases and MLIR reproducers
├── results/                # Benchmark outputs and summaries
├── scripts/                # Build and execution helpers
└── third_party/            # StableHLO/LLVM checkouts (gitignored)
```

## Toolchain Versions

| Component | Version | Commit |
|-----------|---------|--------|
| StableHLO | v0.13.0 | `75c7095a97c6aaaee15dfab1fac529ce695e1d4a` |
| LLVM/MLIR (build) | llvmorg-18.1.8 | `3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff` |
| LLVM (runtime) | 17.0.6 | Homebrew `llvm@17` |

## Setup

### 1. Clone StableHLO and LLVM

```bash
mkdir -p third_party
git clone https://github.com/openxla/stablehlo.git third_party/stablehlo
git -C third_party/stablehlo checkout 75c7095a97c6aaaee15dfab1fac529ce695e1d4a

git clone https://github.com/llvm/llvm-project.git third_party/stablehlo/llvm-project
git -C third_party/stablehlo/llvm-project checkout 3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff
```

### 2. Sync Custom Passes

```bash
bash tools/sync_passes.sh
```

Copies pass implementations from [passes/](passes/) into `third_party/stablehlo/stablehlo/transforms/`.

### 3. Build LLVM/MLIR

```bash
cmake -S third_party/stablehlo/llvm-project/llvm \
  -B third_party/stablehlo/llvm-project/build \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/stablehlo/llvm-project/build
```

### 4. Build stablehlo-opt

```bash
cmake -S third_party/stablehlo \
  -B third_party/stablehlo/build \
  -G Ninja \
  -DMLIR_DIR=third_party/stablehlo/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=third_party/stablehlo/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/stablehlo/build -t stablehlo-opt
```

### 5. Install Runtime LLVM

```bash
brew install llvm@17
```

## Custom Passes

All pass implementations live in [passes/](passes/):

| Pass | Description |
|------|-------------|
| [TilingPass.cpp](passes/TilingPass.cpp) | Converts StableHLO to Linalg and applies initial tiling |
| [GenericFusionPass.cpp](passes/GenericFusionPass.cpp) | Fuses adjacent Linalg generic operations |
| [LinalgTilingPass.cpp](passes/LinalgTilingPass.cpp) | Multi-level cache-aware tiling (L2/L1) |
| [OneShotBufferizePass.cpp](passes/OneShotBufferizePass.cpp) | Tensor-to-memref bufferization |
| [ToyVectorizationPass.cpp](passes/ToyVectorizationPass.cpp) | SIMD vectorization with configurable vector width |
| [ParallelSchedulingPassAttn.cpp](passes/ParallelSchedulingPassAttn.cpp) | Parallel scheduling for attention operations |

**Registration files:** [Passes.td](passes/Passes.td), [Passes.h](passes/Passes.h), [CMakeLists.txt](passes/CMakeLists.txt)

After modifying a pass, resync and rebuild:

```bash
bash tools/sync_passes.sh
cmake --build third_party/stablehlo/build -t stablehlo-opt
```

## Pipeline Execution

Run the full optimization pipeline:

```bash
bash tools/run_attn_cache_pipeline.sh
```

**Pipeline stages** (input/output in [ir/](ir/)):

| Stage | Pass | Output |
|-------|------|--------|
| 0 | `-stablehlo-tiling` | [attn_stage0_linalg.mlir](ir/attn_stage0_linalg.mlir) |
| 1 | `-generic-fusion` | [attn_stage1_fused.mlir](ir/attn_stage1_fused.mlir) |
| 2 | `-stablehlo-linalg-tiling=l2-tile-sizes=64,64,64 l1-tile-sizes=4,4,4` | [attn_stage2_tiled_tensor.mlir](ir/attn_stage2_tiled_tensor.mlir) |
| 3 | `-empty-tensor-to-alloc-tensor`, `-one-shot-bufferize`, `-convert-bufferization-to-memref` | [attn_stage3_bufferized.mlir](ir/attn_stage3_bufferized.mlir) |
| 5 | `-toy-vectorize=vector-width=4 enable-reductions` | [attn_stage5_vectorized.mlir](ir/attn_stage5_vectorized.mlir) |

## Manual Pass Execution

Run `stablehlo-opt` directly with custom pass pipeline:

```bash
third_party/stablehlo/build/bin/stablehlo-opt \
  ir/attn_complete.mlir \
  -stablehlo-tiling \
  -generic-fusion \
  "-stablehlo-linalg-tiling=l2-tile-sizes=64,64,64 l1-tile-sizes=4,4,4 enable-promotion=false" \
  -canonicalize -cse \
  -o /tmp/attn_out.mlir
```

## Benchmarking

**All stages:**
```bash
ITERS=100 bash tools/benchmark_all_stages.sh
```

**Single stage (vectorized):**
```bash
bash tools/benchmark_stage5.sh
```

Both use `mlir-cpu-runner` to execute baseline ([attn_complete.mlir](ir/attn_complete.mlir)) vs optimized ([attention_pipeline.mlir](ir/attention_pipeline.mlir)) IR.
