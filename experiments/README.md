# Experiments

Curated benchmarks, IR snippets, and datasets used to validate tiling/fusion/vectorization ideas.

- `benchmarks/` hosts runnable harnesses (e.g., `tiling/` shows pre/post tiling IR plus drivers).
- `data/` stores large inputs and traces.  Keep raw blobs inside `data/raw/` (gitignored) and provide scripts to regenerate them when possible.
- `ir/` contains standalone MLIR reproducers that demonstrate interesting patterns or regressions.
