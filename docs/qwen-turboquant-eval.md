# Qwen3.5 TurboQuant evaluation (MLX path)

Date: 2026-04-14

## Scope

Goal: determine whether TurboQuant can be used in the current repo setup (Apple Silicon + MLX + Qwen3.5 path), and benchmark closest feasible KV-cache optimizations that work now.

Environment evidence:
- `scripts/smoke_test.sh` passed (MLX + mlx-lm + DFlash import available).
- `mlx_lm.generate --help` exposes `--max-kv-size`, `--kv-bits`, `--kv-group-size`, `--quantized-kv-start`.

## 1) TurboQuant feasibility verdict

### What “TurboQuant” refers to in thread/library context

Upstream evidence captured in:
- `artifacts/benchmarks/turboquant-source-snippets.txt`

Key snippets:
- `rotorquant` README: Turbo/iso/planar cache types are used through **llama.cpp** `llama-server --cache-type-k ... --cache-type-v ...` commands.
- `multi-turboquant` README: generates **llama.cpp/vLLM** launch commands; platform table says `macOS + Apple Silicon | iso/planar (4) | llama.cpp (Metal)`.

### Can it run on current MLX Qwen path?

Direct probe in this repo/toolchain:

```bash
.venv/bin/mlx_lm.generate ... --cache-type-k turbo3
.venv/bin/python -m dflash.benchmark ... --cache-type-k turbo3
```

Both fail with `error: unrecognized arguments: --cache-type-k turbo3` (logs in `artifacts/benchmarks/turboquant-flag-probe-*.log`).

**Verdict:** TurboQuant is **not feasible as a drop-in** for the current MLX/DFlash Qwen path.  
It would require a runtime migration to a llama.cpp GGUF path.

## 2) Feasible alternatives tested now

Test prompt:
- `artifacts/benchmarks/qwen-kv-eval-prompt.txt` (10,269 prompt tokens in run logs)

Measured command sets:

FP16 KV (no KV quantization):
```bash
scripts/benchmark_supergemma_mlx.sh \
  --model Qwen/Qwen3.5-4B \
  --prompt-file artifacts/benchmarks/qwen-kv-eval-prompt.txt \
  --max-tokens 256 \
  --max-kv-size 1024 --max-kv-size 2048 --max-kv-size 4096 \
  --repeats 2
```

4-bit KV quantization:
```bash
scripts/benchmark_supergemma_mlx.sh \
  --model Qwen/Qwen3.5-4B \
  --prompt-file artifacts/benchmarks/qwen-kv-eval-prompt.txt \
  --max-tokens 256 \
  --max-kv-size 1024 --max-kv-size 2048 --max-kv-size 4096 \
  --repeats 2 \
  -- --kv-bits 4 --kv-group-size 64 --quantized-kv-start 0
```

Artifacts:
- FP16 sweep: `artifacts/benchmarks/20260414T183925Z/results.csv`
- KV4 sweep: `artifacts/benchmarks/20260414T184250Z/results.csv`

## 3) Measured results

| Mode | max-kv-size | n | Avg gen tok/s | Avg peak memory (GB) |
|---|---:|---:|---:|---:|
| FP16 KV | 1024 | 2 | 12.907 | 10.679 |
| FP16 KV | 2048 | 2 | 12.927 | 10.679 |
| FP16 KV | 4096 | 2 | 12.911 | 10.679 |
| KV quantized (4-bit) | 1024 | 2 | 12.620 | 10.446 |
| KV quantized (4-bit) | 2048 | 2 | 12.516 | 10.446 |
| KV quantized (4-bit) | 4096 | 2 | 12.601 | 10.446 |

Derived from those runs:
- FP16 overall mean throughput: **12.915 tok/s**
- KV4 overall mean throughput: **12.579 tok/s** (about **-2.6%** vs FP16)
- Peak memory: **10.679 GB (FP16)** vs **10.446 GB (KV4)** (about **-0.233 GB**, **-2.2%**)

Observation on `--max-kv-size` in this workload:
- Within 1024/2048/4096, throughput and reported peak memory were effectively flat in both modes.

## 4) Recommendation for this repo

1. **Do not plan TurboQuant as an in-place MLX change** in this repo; it is not exposed by current MLX/DFlash CLIs.
2. If memory pressure is the priority on current MLX path, use tested KV quantization knobs:
   - `--kv-bits 4 --kv-group-size 64 --quantized-kv-start 0`
3. If throughput is the top priority and memory headroom is sufficient, stay with FP16 KV (faster in these measurements).
4. If TurboQuant/iso/planar is still desired, isolate it in a separate llama.cpp (Metal + GGUF) track.

## Repeatability helper

Added:
- `scripts/run_qwen_kv_eval.sh`

This script runs the same FP16 vs KV4 sweep and writes summarized outputs under:
- `artifacts/benchmarks/qwen-kv-eval-<timestamp>/`
