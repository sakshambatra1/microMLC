# Third-Party Dependencies

This directory is reserved for code pulled directly from upstream projects (StableHLO, LLVM/MLIR, etc.).

- `stablehlo/` currently points at a local checkout so development can proceed while offline.
- `archive/` can keep scratch copies or patches that should never be committed.

Before pushing to GitHub, replace any local copies with git submodules so the repository only tracks lightweight references:

```bash
git submodule add https://github.com/openxla/stablehlo.git third_party/stablehlo
( cd third_party/stablehlo && git checkout <commit-you-tested> )
```

If you carry local modifications, export them into `patches/stablehlo/*.patch` (create the `patches/` tree next to this README) and document how to apply them in that folder.
