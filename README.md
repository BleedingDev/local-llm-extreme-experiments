# supergemma-dflash-ddtree-mlx

Reproducible local benchmarking workspace for MLX-based model experiments on Apple Silicon.

This repository packages:
- pinned upstream sources (`dflash`, `ddtree`, `triattention`, `turboquant-mlx`)
- versioned local patches
- pinned Python runtime dependencies
- reproducible benchmark runners for Gemma 4 26B A4B variants

## Quick start

```bash
scripts/fetch_vendor_sources.sh
scripts/setup_env.sh
scripts/smoke_test.sh --venv-path .venv
```

## Run benchmarks

### 1) Variant preflight / dry-run

```bash
scripts/run_gemma4_a4b_variant.sh --variant baseline --preflight
scripts/run_gemma4_a4b_variant.sh --variant rotorquant --dry-run
```

### 2) Full matrix execution

```bash
scripts/benchmark_gemma4_a4b_variants.sh \
  --execute \
  --max-kv-size 512 \
  --max-kv-size 1024 \
  --max-tokens 256 \
  --repeats 1
```

Artifacts are written under `artifacts/benchmarks/gemma4-a4b/`.

## Reproducibility model

1. **Pinned sources:** `scripts/pinned_sources.sh`
2. **Deterministic fetch:** `scripts/fetch_vendor_sources.sh`
3. **Versioned patches:** `patches/*.patch` applied by `scripts/apply_vendor_patches.sh`
4. **Pinned runtime deps:** `requirements.txt`

Recreate the same state on another machine by rerunning:

```bash
scripts/fetch_vendor_sources.sh
scripts/setup_env.sh
```

## Project structure

- `scripts/` benchmark runners, setup, and orchestration
- `patches/` local vendor patch set
- `docs/` benchmark and bootstrap documentation
- `vendor/` pinned upstream checkouts (ignored in git)

## Notes

- This repo intentionally avoids global package installation.
- Heavy benchmark artifacts are local-only (`artifacts/`) and not committed.
- Some advanced combinations remain experimental and may hit Metal timeout/OOM depending on machine limits.
