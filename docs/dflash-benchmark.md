# DFlash practical benchmark (local MLX)

Date: 2026-04-14

## Environment/setup

Executed (local `.venv`, repo scripts only):

```bash
scripts/setup_env.sh --skip-smoke-test
scripts/smoke_test.sh
```

Result: setup completed, DFlash import available in `.venv`, smoke test passed.

## Benchmark commands (real runs)

```bash
scripts/run_dflash_mlx_benchmark.sh --max-samples 1 -- --max-new-tokens 32
scripts/run_dflash_mlx_benchmark.sh --max-samples 2 -- --max-new-tokens 32
scripts/run_dflash_mlx_benchmark.sh --max-samples 3 -- --max-new-tokens 32
```

Backend/model pair followed wrapper defaults:
- backend: `mlx`
- target model: `Qwen/Qwen3.5-4B`
- draft model: `z-lab/Qwen3.5-4B-DFlash`
- dataset: `gsm8k`

## Artifacts/logs

- `artifacts/benchmarks/dflash-mlx-20260414-173127/benchmark.log`
- `artifacts/benchmarks/dflash-mlx-20260414-173612/benchmark.log`
- `artifacts/benchmarks/dflash-mlx-20260414-173921/benchmark.log`
- Dataset cache created/used: `vendor/dflash/cache/gsm8k.jsonl`

## Measured results (from logs)

| Run | max-samples | max-new-tokens | Baseline throughput (tok/s) | DFlash throughput (tok/s) | Decoding speedup | Avg acceptance length |
|---|---:|---:|---:|---:|---:|---:|
| dflash-mlx-20260414-173127 | 1 | 32 | 15.18 | 19.63 | 1.29x | 2.75 |
| dflash-mlx-20260414-173612 | 2 | 32 | 15.18 | 24.65 | 1.62x | 3.44 |
| dflash-mlx-20260414-173921 | 3 | 32 | 15.16 | 30.72 | 2.03x | 3.62 |

Observed acceptance histogram excerpts are included in each `benchmark.log`.

## Best current measurable run

Command:

```bash
scripts/run_dflash_mlx_benchmark.sh --max-samples 3 -- --max-new-tokens 32
```

Exact metric lines from `artifacts/benchmarks/dflash-mlx-20260414-173921/benchmark.log`:

```text
Baseline throughput: 15.16 tok/s
DFlash throughput:  30.72 tok/s
Decoding speedup: 2.03
Average Acceptance length: 3.62
```

## Notes / caveats

- First run incurred one-time model/dataset download and printed HF unauthenticated warning (`HF_TOKEN` not set), but benchmark execution completed successfully.
- Second run used cached artifacts and completed faster end-to-end.
