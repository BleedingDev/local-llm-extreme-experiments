# Carnice 9B (Qwen 3.5) + MLX + DFlash evaluation

Date: 2026-04-15

Model target:
- `jason-schulz/Carnice-9b-MLX`

Draft model (DFlash):
- `z-lab/Qwen3.5-9B-DFlash`

Artifacts summary:
- `artifacts/benchmarks/carnice-9b-summary.json`

## What was tested

1. Baseline MLX throughput sweep (`max_kv_size`: 512/1024/2048; `max_tokens`: 64).
2. KV-cap probe (`max_kv_size` up to 131072).
3. Real long-prompt context scaling (up to ~121k prompt tokens, then short decode).
4. Long generation stability (baseline 1024-token run).
5. DFlash speedup on Carnice (short and longer decode runs).
6. Tool-calling format adherence using Carnice chat template with tool declarations.

## Key results

### Throughput (baseline MLX, short generation)

Source: `artifacts/benchmarks/20260415T140351Z/results.csv`

| max_kv_size | prompt tok/s | generation tok/s | peak mem (GB) |
|---:|---:|---:|---:|
| 512 | 33.346 | **20.096** | 5.169 |
| 1024 | **76.077** | 16.388 | 5.169 |
| 2048 | 67.706 | 20.064 | 5.169 |

- Best decode in this sweep: **20.096 tok/s** (`kv=512`).
- Best prefill in this sweep: **76.077 tok/s** (`kv=1024`).

### KV-cap probe

Source: `artifacts/benchmarks/20260415T142226Z/summary.txt`

- `probe_status=no-failure-observed`
- Largest tested `max_kv_size` without failure: **131072**

Important: this probe uses short prompts; it validates configured KV ceiling behavior, not full real-prompt occupancy.

### Real context on long prompts

Source: `artifacts/benchmarks/carnice-context-realprompt/results.json`

- Largest successful real prompt: **121,081 prompt tokens** (`target=131072`, decode 8 tokens, status `ok`).
- Estimated effective prefill throughput drops with context size:
  - ~595 tok/s at ~7.7k prompt tokens
  - ~429 tok/s at ~60.6k
  - ~312 tok/s at ~121.1k

### Long generation stability (baseline)

Source: `artifacts/benchmarks/20260415T151505Z/results.csv`

- Completed `max_tokens=1024` without failure.
- `generation_tps=15.954`, `prompt_tps=102.3`, `peak_memory_gb=5.176`.

### DFlash on Carnice

Short/tuned run source: `artifacts/benchmarks/carnice-dflash-latest.log`
- Baseline: `21.06 tok/s`
- DFlash: `34.67 tok/s`
- Speedup: **1.65x**
- Avg acceptance length: `5.03`

Longer decode source: `artifacts/benchmarks/carnice-dflash-longgen.log`
- Baseline: `24.63 tok/s`
- DFlash: `45.55 tok/s`
- Speedup: **1.85x**
- Avg acceptance length: `6.27`

### Tool-calling stability (Hermes template path)

Source: `artifacts/benchmarks/carnice-toolcall-eval/summary-hermes-template.json`

- 15/15 successful tool-call outputs (**100%**) under Carnice chat template + tool declarations.
- Exact XML tool-call format emitted and parsed (`<tool_call><function=...><parameter=...>`).

Note:
- Naive plain-JSON prompting without Hermes template failed consistently (0/15), while the model-specific tool template succeeded. This confirms the harness-format dependency.

## RotorQuant / TurboQuant / DDTree follow-up (higher-context tuning)

Summary artifact:
- `artifacts/benchmarks/carnice-quant-summary.json`

### TurboQuant matrix on Carnice (MLX path)

Source:
- `artifacts/benchmarks/carnice-turboquant-20260415T154912Z/*.json`

| Strategy | tok/s | vs fp16 | total cache bytes vs fp16 |
|---|---:|---:|---:|
| `tqv2_4bit_lean` | **14.86** | **1.002x** | **-12.71%** |
| `tqv2_3bit_rot_qjl` | 11.28 | 0.754x | -10.96% |
| `tqv2_4bit_rot` | 10.35 | 0.702x | -10.83% |
| `tqv3_3.5bit_mixed` | 11.22 | 0.758x | -11.20% |
| `tqv3_3bit` | 11.37 | 0.766x | -11.31% |

Takeaway:
- For this Carnice MLX setup, the only strategy that did not hurt speed was **`tqv2_4bit_lean`**, and it also reduced cache bytes.
- Aggressive variants (3-bit / rot / qjl / v3) reduced cache bytes but degraded decode throughput.

### KV quantization context probes

Sources:
- `artifacts/benchmarks/20260415T155052Z` (KV4)
- `artifacts/benchmarks/20260415T155130Z` (KV3)

Both KV4 and KV3 probe runs were stable to `max_kv_size=131072` on short prompts (no failure observed).

### Real long-prompt context checkpoints

Sources:
- `artifacts/benchmarks/carnice-quant-context/**/results.csv`

| Mode | ~122k prompt | 150k prompt | 160k prompt | 180k prompt | 220k prompt |
|---|---|---|---|---|---|
| fp16 | ok | ok | fail | fail | fail |
| kv4 | ok | n/a | ok | fail | fail |
| kv3 | ok | n/a | ok | n/a | fail |

Concrete successful checkpoints:
- `fp16` at ~150k prompt tokens: `prompt_tokens=150010`, `peak_memory_gb=21.532`.
- `kv4` at ~160k prompt tokens: `prompt_tokens=160010`, `peak_memory_gb=18.636`.
- `kv3` at ~160k prompt tokens: `prompt_tokens=160010`, `peak_memory_gb=18.302`.

Practical implication:
- In this run set, KV quantization extended practical single-window context from ~150k (`fp16`) to about ~160k (`kv4/kv3`) before instability.

### DDTree (MLX prototype) on Carnice

Source:
- `artifacts/ddtree-mlx-prototype/runs/qwen35-ddtree-mlx-20260415T165725Z/result.json`

Result:
- `generation_tps=7.20`
- `accepted_length_mean=0.923`
- `fallback_events=1`

Interpretation:
- DDTree-MLX prototype runs, but on this Carnice setup it is significantly slower than baseline/DFlash and is not the best path for higher practical context.

## Recommendation after this tuning pass

1. For higher context on Carnice MLX, prioritize **KV quantization (kv4/kv3)** over DDTree.
2. For TurboQuant variants, keep **`tqv2_4bit_lean`** as the only currently attractive default.
3. Keep DFlash for decode throughput; use quantized KV primarily for extending practical context headroom.

## DFlash + cache-fusion path (single runtime)

Implemented in `vendor/dflash/dflash`:
- `--cache-optimization none|kv-quant|turboquant`
- `--kv-bits`, `--kv-group-size`, `--quantized-kv-start`
- `--turboquant-strategy`, `--turboquant-seed`
- `--max-kv-size`

Reference commands:

```bash
# DFlash + KV quant
scripts/run_dflash_mlx_benchmark.sh \
  --model jason-schulz/Carnice-9b-MLX \
  --draft-model z-lab/Qwen3.5-9B-DFlash \
  --max-samples 1 \
  -- --max-new-tokens 16 --cache-optimization kv-quant --kv-bits 4 --kv-group-size 64 --quantized-kv-start 0

# DFlash + TurboQuant
scripts/run_dflash_mlx_benchmark.sh \
  --model jason-schulz/Carnice-9b-MLX \
  --draft-model z-lab/Qwen3.5-9B-DFlash \
  --max-samples 1 \
  -- --max-new-tokens 16 --cache-optimization turboquant --turboquant-strategy tqv2_4bit_lean --quantized-kv-start 0
```

Raw fused-runtime checks:
- `artifacts/benchmarks/carnice-fused-runtime-summary.json`
- `artifacts/benchmarks/carnice-fused-runtime-check/*.json`

Observed behavior in these checks:
- Fused runtime works on short prompts and on ~16k prompt window with valid Hermes tool call (`weather.get`).
- At ~32k prompt and above in fused DFlash path, runs hit Metal allocation failure on this machine.
