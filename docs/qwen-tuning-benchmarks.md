# Qwen tuning benchmarks

Date: 2026-04-14

## Scope

Compared the currently available local paths in this repo:

- Qwen baseline MLX
- Qwen + DFlash (best config from prior sweep)
- Qwen + turboquant-mlx
- Qwen + DDTree-MLX prototype

## New apples-to-apples mini benchmark

- Artifacts root: `artifacts/benchmarks/qwen-tuning-comparison-20260414T194710Z/`
- Prompt: `Reply with one short Czech sentence about model tuning.`
- `max_new_tokens` / `max_tokens`: `24`
- Temperature: `0.0`
- Model: `Qwen/Qwen3.5-4B`
- Draft model (DFlash/DDTree): `z-lab/Qwen3.5-4B-DFlash`

| Path | Runnable | Generation tok/s | vs baseline (12.82 tok/s) | Peak memory (GB) | Key extra metric(s) | Evidence |
|---|---|---:|---:|---:|---|---|
| Qwen baseline MLX | Yes | 12.82 | 1.00x | 9.57 | 24 generated tokens | `artifacts/benchmarks/qwen-tuning-comparison-20260414T194710Z/dflash-prompt/result.json` |
| Qwen + DFlash (short-prompt run) | Yes | 7.18 | 0.56x | 9.65 | acceptance mean 1.14 | `artifacts/benchmarks/qwen-tuning-comparison-20260414T194710Z/dflash-prompt/result.json` |
| Qwen + turboquant-mlx (`tqv2_4bit_lean`) | Yes | 11.00 | 0.86x | n/a | `-13.33%` total cache bytes vs fp16; KV-layer compression `3.56x`; speed ratio vs its fp16 baseline `0.97x` | `artifacts/benchmarks/qwen-tuning-comparison-20260414T194710Z/turboquant/result.json` |
| Qwen + DDTree-MLX prototype (`tree_budget=4`, `depth=1`) | Yes | 4.69 | 0.37x | 9.64 | acceptance mean 0.47; fallback rate 52.94% | `artifacts/benchmarks/qwen-tuning-comparison-20260414T194710Z/ddtree-prototype/apples-to-apples/result.json` |

## DFlash best tuned config (from prior sweep)

Source:

- `docs/qwen-dflash-sweep.md`
- `artifacts/benchmarks/qwen-dflash-sweep-20260414-192908/summary.json`

Best command:

```bash
scripts/run_dflash_mlx_benchmark.sh --model Qwen/Qwen3.5-4B --draft-model z-lab/Qwen3.5-4B-DFlash --max-samples 3 -- --max-new-tokens 64
```

Best observed sweep metrics:

- Baseline: `12.94 tok/s`
- DFlash: `30.98 tok/s`
- Speedup: `2.39x`
- Average acceptance length: `4.26`

## Winners by metric (current best-known)

- **Highest observed throughput overall:** Qwen + DFlash best sweep config (`30.98 tok/s`, `2.39x` vs baseline in that sweep).
- **Highest throughput in the new short-prompt apples-to-apples run:** baseline MLX (`12.82 tok/s`).  
  Among tuned methods only: **turboquant-mlx** (`11.00 tok/s`).
- **Best cache-memory reduction signal:** **turboquant-mlx** (`-13.33%` total cache bytes vs fp16 in its run; KV-cache layers `3.56x` compression).

## Runnable status / blockers

All requested paths were runnable in this local environment for this benchmark pass; no blockers encountered.
