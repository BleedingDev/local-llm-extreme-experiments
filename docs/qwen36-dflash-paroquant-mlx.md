# Qwen 3.6 + MLX + DFlash + ParoQuant

This repo now has a dedicated MLX combo benchmark path for the three useful combinations:

- `dflash`: base MLX target + DFlash draft
- `paro`: ParoQuant MLX target only
- `both`: ParoQuant MLX target + DFlash draft

## Important constraint

These are **two MLX target-loading paths** that can be combined at generation time:

1. `dflash.model_mlx.load(...)` loads a normal MLX-LM target model.
2. `paroquant.inference.backends.mlx.load(...)` loads a PARO-formatted MLX target model.

The merged `both` mode works by:

1. loading the target through ParoQuant,
2. loading the draft through DFlash,
3. passing the ParoQuant-loaded target into `dflash.model_mlx.stream_generate(...)`.

That means the ParoQuant checkpoint must match the same target family the DFlash draft expects.

## Current upstream status

As checked on **2026-04-21**:

- `mlx-community/Qwen3.6-35B-A3B-4bit` is public on Hugging Face as the MLX 4-bit target.
- `z-lab/Qwen3.6-35B-A3B-DFlash` is public on Hugging Face.
- a public `Qwen3.6-35B-A3B` ParoQuant checkpoint was **not** found via the Hugging Face API.

So the repo supports a real merged benchmark, but for Qwen 3.6 you currently need to supply your own ParoQuant-compatible checkpoint path/id for `paro` or `both` mode.

## Setup

```bash
scripts/fetch_vendor_sources.sh --component paroquant
scripts/setup_paroquant_mlx.sh
```

Or install through the normal bootstrap:

```bash
PAROQUANT_INSTALL_SPEC=./vendor/paroquant scripts/setup_env.sh --skip-smoke-test
scripts/smoke_test.sh
```

## Qwen 3.6 wrapper

```bash
scripts/run_qwen36_mlx_combo_benchmark.sh \
  --modes dflash \
  --max-new-tokens 64
```

If you have a matching ParoQuant checkpoint for Qwen 3.6:

```bash
scripts/run_qwen36_mlx_combo_benchmark.sh \
  --paro-model <your_qwen36_paro_model_or_local_path> \
  --modes dflash,paro,both \
  --max-new-tokens 64
```

To inspect the prepared command without loading models:

```bash
scripts/run_qwen36_mlx_combo_benchmark.sh \
  --paro-model <your_qwen36_paro_model_or_local_path> \
  --modes dflash,paro,both \
  --dry-run
```

For DFlash-only preparation:

```bash
scripts/run_qwen36_dflash_mlx_benchmark.sh --dry-run
```

Artifacts are written under `artifacts/benchmarks/qwen-mlx-combo/`.
