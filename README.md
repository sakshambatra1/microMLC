# toy-transformer Compiler Playground

Transformer-style workloads really stress MLIR on CPUs. This repo is me figuring out multi-level tiling, epilogue fusion, vectorization, and scheduling on top of StableHLO/MLIR, while keeping upstream code separate from the experiments I care about.

## What you’ll find
- JAX → StableHLO → (manually/AI massaged) Linalg pipeline so I can reason about codegen before writing real passes.
- Benchmarks with matching baseline/tiled IR and a timing harness.
- Toolchain pinned to StableHLO v0.13.0 + LLVM/MLIR llvmorg-18.1.8, so numbers don’t drift.
- Clear split between my work and whatever third-party bits I pull locally.

## Repository Layout

```
toy-transformer/
├── compiler/            # Original dialects, passes, python prototypes, and tests
│   ├── python/          # Transformer model, IR generation, RoPE experiments
│   └── tests/           # (future) mlir-opt + python regression suites
├── docs/                # Design notes, architecture diagrams, performance write-ups
├── experiments/         # Benchmarks, datasets, standalone MLIR reproducers
├── results/             # Curated metrics, charts, notebooks (raw profiles ignored)
├── scripts/             # Build/run automation
├── third_party/         # (local only) StableHLO/LLVM/MLIR checkouts, ignored in git
├── patches/             # Patch queue for any third-party tweaks
└── venv/                # Local virtual environment (gitignored)
```

Only `compiler/`, `experiments/`, `docs/`, `results/`, `scripts/`, and `patches/` are checked in. The `third_party/` folder is local-only and stays out of GitHub; when I publish I’ll swap it for a submodule reference.

## Toolchain Versions & Known Limitations

| Component | Version | Commit |
|-----------|---------|--------|
| StableHLO | v0.13.0 | `75c7095a97c6aaaee15dfab1fac529ce695e1d4a` |
| LLVM/MLIR | llvmorg-18.1.8 | `3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff` |

The local StableHLO checkout I build against embeds a matching `llvm-project`. I stick to that toolchain (`cmake -GNinja -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=X86 ..`) so the dialects don’t drift under me.

> **StableHLO → Linalg reality check:** v0.13.0 doesn’t have a reliable `stablehlo-legalize-to-linalg` pipeline. Today I massage the StableHLO IR into Linalg (sometimes with AI assistance) before trying tiling/vectorization passes. Long term I’ll upgrade or script the missing pieces.

## Pass & IR Walkthrough

1. **JAX → StableHLO exporter (`compiler/python/compilerpass_selfatt.py`)**  
   Deterministic multi-head attention workload. Emits:
   - JAXPR for the high-level graph,
   - StableHLO text IR (`attn.mlir`) for downstream experiments,
   - Optional canonical HLO text for comparison.
2. **Attention building blocks (`compiler/python/selfattention.py`, `layers.py`, `embeddings.py`, `tokenizer.py`)**  
   JAX reference implementation of the transformer components that MLIR passes must preserve.
3. **Tiling prototype (`experiments/benchmarks/tiling/tiling_benchmark.py`)**  
   Python version of my planned MLIR tiling pass (tile matmul along L2/L1). Produces paired IR snapshots (`attn_before.mlir`, `attn_tiled.mlir`) and measures correctness + speedup.
4. **IR gallery (`experiments/ir/*.mlir`)**  
   Standalone MLIR modules to repro interesting lowering issues or highlight fusion opportunities.

As C++ MLIR passes land, they will live under `compiler/include/toy/` and `compiler/lib/passes/` with matching lit tests in `compiler/tests/mlir/`.

## Quickstart (Python + JAX pipeline)

```bash
git clone <repo> toy-transformer
cd toy-transformer
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install jax jaxlib numpy
```

Generate StableHLO IR:

```bash
python -m compiler.python.compilerpass_selfatt
# Outputs JAXPR + StableHLO text and writes attn.mlir at repo root.
mv attn.mlir experiments/ir/attn_latest.mlir  # optional snapshot
```

Run the tiling benchmark:

```bash
python experiments/benchmarks/tiling/tiling_benchmark.py
# Prints correctness diff + average CPU runtimes for baseline vs tiled kernels.
```

## Running StableHLO / MLIR Passes

After building StableHLO (inside `third_party/stablehlo`, run `cmake -B build -G Ninja && cmake --build build -t stablehlo-opt`), you can pipe any IR from `experiments/ir/` or `experiments/benchmarks/` through your pass pipeline.

Example tiling invocation:

```bash
export OPT_BIN=third_party/stablehlo/build/bin/stablehlo-opt
export INPUT=experiments/benchmarks/tiling/attn_before.mlir
export OUTPUT=experiments/benchmarks/tiling/attn_tiled_from_opt.mlir

"$OPT_BIN" "$INPUT" \
  --stablehlo-linalg-tiling="l2-tile-sizes=0,32,32 l1-tile-sizes=0,0,8" \
  -o "$OUTPUT"
```

Tips:
- Swap `INPUT` for any new StableHLO/Linalg module you produce.
- Adjust tile sizes or pass options (`vectorize`, `enable-epilogue-fusion`, etc.) while iterating.
- Use `--pass-pipeline=` when chaining multiple transformations.
- Commit resulting IRs/metrics into `experiments/benchmarks/<workload>/` and `results/summaries/`.

## Development Workflow

1. Implement/tweak passes under `compiler/` (Python prototypes today, MLIR C++ soon).
2. Save reproducible IR repros or inputs under `experiments/`.
3. Benchmark and archive results under `results/` with commentary in `docs/performance/`.
4. Track manual StableHLO changes as patches in `patches/` and keep `third_party/` pristine (convert to submodules before publishing).

It mirrors how upstream MLIR is laid out without pulling third-party code into GitHub, and keeps the focus on the passes/benchmarks/results I actually care about.
