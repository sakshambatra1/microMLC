# Compiler Sources

This tree hosts all original compiler research code.

- `python/` contains the transformer modeling code, IR generators, and Python pass prototypes.
- `tests/` is where MLIR and Python regression suites live.  Add `mlir/` and `python/` sub-folders as the project grows.

Any C++ StableHLO/MLIR passes should live here too (mirror upstream `include/` + `lib/` structure) so reviewers can find them without sifting through vendored code.
