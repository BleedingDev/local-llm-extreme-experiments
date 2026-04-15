# local-llm-extreme-experiments

Reproducible local benchmarking workspace for MLX-based model experiments on Apple Silicon.

This repository packages:
- pinned upstream sources (`dflash`, `ddtree`, `triattention`, `turboquant-mlx`)
- versioned local patches
- pinned Python runtime dependencies
- benchmark runners for both Gemma 4 and Qwen 3.5 experiment tracks

## Why these tests were run

The campaign targeted two practical questions:

1. **Gemma 4 26B A4B track:** Can a local 32 GB Apple Silicon machine run high-parameter Gemma with usable context and speed?
2. **Qwen 3.5 track:** Do DFlash/DDTree/TriAttention/TurboQuant/RotorQuant-style optimizations materially improve local agent workload behavior?

To answer that, we measured:
- low-context throughput (prompt/decode speed)
- high-context viability (how far context can be pushed before instability/OOM)
- long-generation stability (can it generate long outputs reliably)
- structured tool-call adherence (JSON/tool schema reliability)
- memory behavior and failure modes (timeout vs OOM)

## Quick start

```bash
scripts/fetch_vendor_sources.sh
scripts/setup_env.sh
scripts/smoke_test.sh --venv-path .venv
```

## Models and variants tested

### Gemma 4 experiment track

- `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` (Gemma 4 26B A4B)

### Additional allowed variant (same 26B A4B family)

- `majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-4bit`

### Draft model for speculative decoding only

- `mlx-community/gemma-4-e2b-it-4bit`

### Runtime variants tested

- `baseline`
- `speculative`
- `kv4`
- `triattention`
- `turboquant-v2-lean`
- `turboquant-v2-rot`
- `turboquant-v3-3.5` (compatibility-limited)
- `rotorquant`
- `speculative-rotorquant`

### Qwen 3.5 experiment track

Main family tested:
- `Qwen3.5-27B` variants in MLX-compatible forms (baseline, DFlash and quant/cache variants)

Optimization directions tested:
- DFlash speculative decode path
- DDTree integration/prototype path
- TriAttention merge path
- TurboQuant (v2/v3) cache substitutions
- RotorQuant model variant

Qwen-specific docs:
- `docs/qwen-tuning-report.md`
- `docs/qwen-tuning-benchmarks.md`
- `docs/qwen-dflash-sweep.md`
- `docs/qwen-turboquant-eval.md`
- `docs/qwen-ddtree-eval.md`
- `docs/qwen-optimization-report.md`

## Final benchmark outcomes (high-level)

### Best low-context speed

- **Decode speed winner:** `baseline`, KV 512 → ~**41.86 tok/s**
- **Prefill winner:** `triattention`, KV 512 → ~**62.44 tok/s**
- **Lowest memory (low-context set):** `turboquant-v2-lean`, KV 1536 → ~**14.28 GB**

### Practical high-context recommendation

- **Recommended profile:** `rotorquant` (non-speculative)
- Strong balance around KV 32768–36864 on this class of machine
- Verified run at ~111k prompt tokens + 32 decode tokens:
  - `rotorquant`, KV 36864 → prompt ~**495.75 tok/s**, decode ~**14.51 tok/s**, peak ~**25.67 GB**

### Context ceilings observed

- Stable meaningful decode verified at ~**111,360** prompt tokens (32 decode tokens)
- Stress probe reached ~**135,936** prompt tokens (1-token decode)
- Above that, failures become frequent due to **Metal OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`)

### Long-generation stability

- `baseline`: 8192 tokens succeeded (~31.67 tok/s)
- `rotorquant`: 8192 tokens succeeded (~37.28 tok/s)
- `speculative-rotorquant`: unstable for long decode (timeouts/crashes at higher lengths)

### Tool-calling reliability

- Structured JSON/tool-call adherence remained inconsistent across tested variants.
- Throughput/context can be made workable, but robust autonomous tool-calling reliability is still below production-grade expectations in this setup.

## Reproducing the same benchmark workflow

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
  --max-kv-size 1536 \
  --max-tokens 256 \
  --repeats 1
```

Artifacts are written under `artifacts/benchmarks/gemma4-a4b/`.

### Qwen track entry points

```bash
scripts/run_dflash_mlx_benchmark.sh --dry-run
scripts/run_qwen_mlx_kv_sweep.sh --help
scripts/run_qwen_turboquant_mlx.sh --help
scripts/run_qwen_ddtree_benchmark.sh --help
```

## What exactly gets stored per run

Each matrix/sweep run stores:
- resolved config
- per-run status/exit code
- raw logs per variant/KV/token point
- parsed metric fields (prompt TPS, generation TPS, peak memory, finish reason)

Core files:
- `results.csv`
- `summary.txt`
- `raw/*.log`

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

## Important operational notes

- This repo intentionally avoids global package installation.
- Heavy benchmark artifacts are local-only (`artifacts/`) and not committed.
- Some advanced combinations remain experimental and may hit Metal timeout/OOM depending on machine limits.
- `rotorquant` is the strongest high-context candidate in this campaign.
- `speculative-rotorquant` improved some short-context cases but was not stable in long-generation stress.
