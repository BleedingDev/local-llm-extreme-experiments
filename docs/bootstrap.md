# Reproducible bootstrap (MLX + SuperGemma + vendor patches)

This repository is set up for deterministic local runs on Apple Silicon:
- Python deps pinned in `requirements.txt`
- Upstream sources pinned by commit in `scripts/pinned_sources.sh`
- Local integration patches versioned in `patches/`

## 1) Fetch pinned vendor sources and apply local patches

```bash
scripts/fetch_vendor_sources.sh
```

This fetches pinned commits for:
- `dflash`
- `ddtree`
- `turboquant-mlx`
- `triattention`

Then it applies local patch set from `patches/` via:

```bash
scripts/apply_vendor_patches.sh
```

## 2) Bootstrap isolated Python environment

```bash
scripts/setup_env.sh
```

This creates/updates `.venv`, installs pinned dependencies, installs optional local packages when present, and runs `scripts/smoke_test.sh`.

## 3) Validate benchmark runners

```bash
scripts/run_gemma4_a4b_variant.sh --variant baseline --preflight
scripts/benchmark_gemma4_a4b_variants.sh --skip-preflight --variant baseline --max-kv-size 512 --max-tokens 64 --repeats 1
```

## 4) Full Gemma 4 26B A4B benchmark matrix

```bash
scripts/benchmark_gemma4_a4b_variants.sh --execute --repeats 1 --max-kv-size 512 --max-kv-size 1024 --max-tokens 256
```

## 5) Determinism notes

- Keep `scripts/pinned_sources.sh` commit hashes unchanged for reproducible vendor state.
- Keep `requirements.txt` pinned versions unchanged for reproducible runtime.
- If upstream vendor trees are refreshed, rerun `scripts/fetch_vendor_sources.sh` to reapply local patches.
