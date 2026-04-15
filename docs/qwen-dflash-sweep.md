# Qwen3.5 DFlash MLX sweep

Date: 2026-04-14

## Scope

- Environment: local `.venv` + repo scripts only.
- Base command path:

```bash
scripts/run_dflash_mlx_benchmark.sh --model Qwen/Qwen3.5-4B --draft-model z-lab/Qwen3.5-4B-DFlash
```

- Sweep artifacts:
  - `artifacts/benchmarks/qwen-dflash-sweep-20260414-192908/`
  - Per-run logs: `.../runs/*.log`
  - Parsed metrics: `.../metrics.tsv`
  - Aggregate summary: `.../summary.json`

## Option support check

- Verified supported flags via `-- --help`.
- Probed unsupported candidate:
  - `--speculative-num-draft-tokens 8` → unrecognized argument (captured in `unsupported-option-probe.log`).
- Adapted sweep to supported MLX-impacting options: `--max-samples`, `--max-new-tokens`, `--block-size`, `--temperature`.

## Sweep results

| Run | max-samples | max-new-tokens | block-size | temperature | Baseline tok/s | DFlash tok/s | Speedup | Avg acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run01-s1-t32-default | 1 | 32 | default | default | 15.16 | 22.49 | 1.48x | 2.75 |
| run02-s2-t32-default | 2 | 32 | default | default | 15.15 | 28.22 | 1.86x | 3.44 |
| run03-s3-t32-default | 3 | 32 | default | default | 13.55 | 27.93 | 2.06x | 3.62 |
| run04-s3-t64-default | 3 | 64 | default | default | 14.69 | 33.30 | 2.27x | 4.26 |
| run05-s3-t32-b8 | 3 | 32 | 8 | default | 14.58 | 31.34 | 2.15x | 3.77 |
| run06-s3-t32-b24 | 3 | 32 | 24 | default | 14.39 | 24.85 | 1.73x | 3.14 |
| run07-s3-t32-temp06 | 3 | 32 | default | 0.6 | 14.78 | 31.58 | 2.14x | 3.71 |
| run08-s3-t64-default-r2 | 3 | 64 | default | default | 12.94 | 30.98 | **2.39x** | 4.26 |
| run09-s3-t32-default-r2 | 3 | 32 | default | default | 13.55 | 27.27 | 2.01x | 3.62 |

## Best observed config

Best observed config (highest speedup; also highest acceptance length tier):

```bash
scripts/run_dflash_mlx_benchmark.sh --model Qwen/Qwen3.5-4B --draft-model z-lab/Qwen3.5-4B-DFlash --max-samples 3 -- --max-new-tokens 64
```

Best observed metrics:
- Baseline: 12.94 tok/s
- DFlash: 30.98 tok/s
- Speedup: 2.39x
- Avg acceptance length: 4.26

Also notable:
- Highest absolute DFlash throughput was on the same config family (`max-samples=3`, `max-new-tokens=64`): 33.30 tok/s (run04), 2.27x speedup.

## Notable variance

- Across all 9 runs:
  - Baseline: 12.94–15.16 tok/s
  - DFlash: 22.49–33.30 tok/s
  - Speedup: 1.48x–2.39x
  - Acceptance length: 2.75–4.26
- Repeated config variance:
  - `max-samples=3, max-new-tokens=32` (2 runs): speedup 2.01x–2.06x.
  - `max-samples=3, max-new-tokens=64` (2 runs): speedup 2.27x–2.39x.

## Practical takeaway

Within current local MLX toolchain, practical small-sweep tuning is exhausted across sample count, decode length, block size, and temperature. The most consistently strong setting is `max-samples=3` with `max-new-tokens=64`.
