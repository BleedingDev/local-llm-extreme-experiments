# SuperGemma MLX benchmarking

This repo includes a reproducible harness that runs `mlx_lm.generate` from `.venv` only and writes timestamped artifacts under `artifacts/benchmarks/`.

## 0) Reproducible setup (before benchmarks)

```bash
scripts/fetch_vendor_sources.sh
scripts/setup_env.sh
```

`fetch_vendor_sources.sh` uses pinned commits and automatically applies versioned local patches from `patches/`.

## 1) Matrix benchmark runs

Conservative defaults for a 32GB machine:
- `--max-tokens 64`
- `--max-kv-size 512`
- `--repeats 2`

Example (two KV sizes, 2 repeats each):

```bash
scripts/benchmark_supergemma_mlx.sh \
  --prompt "Write one short Czech sentence about local benchmarking." \
  --max-tokens 64 \
  --max-kv-size 512 \
  --max-kv-size 1024 \
  --repeats 2
```

You can pass extra `mlx_lm.generate` flags after `--`.

Dry-run (no model execution):

```bash
scripts/benchmark_supergemma_mlx.sh --dry-run --max-kv-size 512 --repeats 1
```

## 2) Context-limit probing

Probe increasing `--max-kv-size` until failure, then binary-refine the best known working value:

```bash
scripts/probe_supergemma_mlx_context_limit.sh \
  --prompt "Say hello." \
  --probe-start-kv 512 \
  --probe-max-kv 4096 \
  --probe-max-tokens 32
```

Integrated mode is also available:

```bash
scripts/benchmark_supergemma_mlx.sh --mode probe --probe-start-kv 512 --probe-max-kv 4096
```

## 3) Output files

Each run creates `artifacts/benchmarks/<timestamp>/` with:
- `results.csv` — one row per attempt (benchmark and/or probe)
- `summary.txt` — run status + probe conclusion
- `config.env` — resolved run settings
- `prompt.txt` — exact prompt used
- `raw/*.stdout.log` and `raw/*.stderr.log` — raw command streams
- `raw/*.combined.log` — merged log used for metric extraction
- `raw/*.cmd.txt` — exact command per attempt

`results.csv` includes extracted MLX metrics when present:
- prompt tokens + prompt tokens/sec
- generation tokens + generation tokens/sec
- peak memory (GB)

## 4) Gemma 4 26B A4B variant prep (safe mode)

For Gemma 4 26B A4B (including SuperGemma 26B A4B) there is a dedicated prep runner:

```bash
scripts/benchmark_gemma4_a4b_variants.sh
```

Default behavior is safe for live sessions:
- runs **preflight imports** only,
- runs **matrix dry-runs** only,
- does **not** execute heavy model benchmarks unless `--execute` is set.

Single-variant checks:

```bash
scripts/run_gemma4_a4b_variant.sh --variant triattention --preflight
scripts/run_gemma4_a4b_variant.sh --variant baseline --dry-run
```

Supported variants:
- `baseline`
- `speculative` (draft model is allowed to be smaller)
- `kv4` (MLX KV quantization)
- `triattention`
- `turboquant-v2-lean`, `turboquant-v2-rot`, `turboquant-v3-3.5`
- `rotorquant`
- `speculative-rotorquant`
