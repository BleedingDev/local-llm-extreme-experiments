# Qwen3.5 tuning report (local exhaustive pass)

Date: 2026-04-14

## Outcome

For this M5/32GB MLX setup, the best practical throughput path is still **DFlash**.  
`turboquant-mlx` is now integrated and useful mainly for **KV/cache size reduction**, not raw speed.  
The DDTree-style MLX prototype is functional but currently experimental and slower.

## What was exercised

| Track | Status | Entry point |
|---|---|---|
| DFlash sweep on Qwen3.5 | done | `scripts/run_dflash_mlx_benchmark.sh` |
| MLX KV-size sweep/probe | done | `scripts/run_qwen_kv_eval.sh` |
| TurboQuant-MLX integration + eval | done | `scripts/setup_turboquant_mlx.sh`, `scripts/run_qwen_turboquant_mlx.sh` |
| DDTree upstream applicability | done | `scripts/run_qwen_ddtree_benchmark.sh` |
| DDTree-style MLX prototype | done (experimental) | `scripts/run_qwen_ddtree_mlx_prototype.sh` |

## Measured highlights

### 1) DFlash (best observed throughput)

- Baseline: **12.94–15.16 tok/s**
- DFlash: **22.49–33.30 tok/s**
- Speedup: **1.48x–2.39x**
- Best config observed in sweep: `--max-samples 3 --max-new-tokens 64`

Reference: `docs/qwen-dflash-sweep.md` and `artifacts/benchmarks/qwen-dflash-sweep-20260414-192908/`.

### 2) TurboQuant-MLX (KV reduction focus)

Observed on Qwen3.5 with `tqv2_4bit_lean`:
- KV-related compression on selected KV layers: **~3.56x vs fp16 equivalent**
- Total cache reduction vs fp16: **~13.2–13.3%**
- Throughput ratio vs fp16 across measured runs: **~0.82x to 0.97x**

Observed with `tqv3_3.5bit_mixed`:
- Higher KV-layer compression (**~4.27x**), but large throughput drop (**~0.18x vs fp16** in measured run).

Reference: `docs/qwen-turboquant-eval.md` and `artifacts/benchmarks/qwen-turboquant-mlx-*`.

### 3) DDTree-style MLX prototype

- Runnable with round traces/fallback instrumentation.
- Apples-to-apples run (24 generated tokens): **~4.69 tok/s**, fallback rate **~52.9%**.
- Best short-run sample reached **~6.45 tok/s** (very short output), still below tuned DFlash path.

Reference: `docs/qwen-ddtree-mlx-prototype.md` and `artifacts/ddtree-mlx-prototype/runs/`.

### 4) Direct MLX KV-size tuning (without TurboQuant)

- Generation throughput trend improved from **5.62 tok/s (kv=512)** to **8.34 tok/s (kv=4096)** in sweep set.
- Probe succeeded to `max-kv-size=32768` on short prompt workload.

Reference: `artifacts/benchmarks/20260414T185523Z`, `artifacts/benchmarks/20260414T185732Z`.

## Final recommended profiles

1. **Max throughput (default):** DFlash Qwen3.5 (`run_dflash_mlx_benchmark.sh` tuned profile).
2. **Memory pressure experiments:** TurboQuant-MLX `tqv2_4bit_lean` (accepting small-to-moderate speed penalty for cache reduction).
3. **Research only:** DDTree-style MLX prototype for algorithm iteration, not production serving.

## Repro commands (quick start)

```bash
# DFlash tuned benchmark
scripts/run_dflash_mlx_benchmark.sh \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-samples 3 \
  -- --max-new-tokens 64

# TurboQuant-MLX comparison
scripts/run_qwen_turboquant_mlx.sh \
  --strategy tqv2_4bit_lean \
  --compare-fp16 \
  --prompt "Napiš jednu stručnou větu o kvantizaci KV cache."

# DDTree-style MLX prototype run
scripts/run_qwen_ddtree_mlx_prototype.sh \
  --prompt "Reply with one short Czech sentence about model tuning." \
  --tree-budget 4 \
  --depth 1 \
  --max-new-tokens 24
```
