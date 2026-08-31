# Qwen3.6 35B A3B RotorQuant MLX optimization comparison

Date: 2026-04-29

Host:
- Apple M5
- 34,359,738,368 bytes unified memory
- macOS 26.3 build 25D5087f

Base model:
- `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit`
- Revision: `50113fa5f9a705d903b913e1998054c7fb1fdcb8`

DFlash draft:
- `z-lab/Qwen3.6-35B-A3B-DFlash`
- Revision observed via Hugging Face API: `42d3b34d588423cdae7ba8f53a8cf7789346a719`

Runtime:
- `mlx 0.31.1`
- `mlx-lm 0.31.2`

## Executive Summary

Vanilla MLX remains the best practical Hermes Agent profile for this model on this
34 GB M5 host.

DFlash is not useful for normal short/medium agent turns on this target. It decodes
around 19 tok/s in the 256-token throughput test versus 50-55 tok/s for vanilla/MLX
KV4. Long continuous generation is better: DFlash+KV4 and DFlash+TurboQuant-lean reach
about 65 tok/s for forced 8192-token generation. That only helps if the workload is a
single long uninterrupted generation.

Long context is the deciding failure mode. Vanilla and MLX KV4 reached 65K prompt tokens.
Every DFlash-family variant became impractical by 16K and dangerous by 32K: reported peak
memory was about 39 GB at 16K and about 62 GB at 32K, with decode collapsing to roughly
1.5-3 tok/s. No DFlash-family 65K run was attempted after that.

Do not use DFlash, DFlash+TurboQuant, or DFlash+TriAttention for Hermes long-context agent
sessions on this host.

## Variants Tested

| Variant | Meaning |
| --- | --- |
| `baseline` | Vanilla `mlx_lm.stream_generate` |
| `mlx-kv4` | Vanilla MLX with built-in 4-bit KV cache quantization |
| `dflash` | DFlash speculative path with Qwen3.6 DFlash draft |
| `dflash-kv4` | DFlash target cache with MLX 4-bit KV quantization |
| `dflash-turboquant-lean` | DFlash target cache with TurboQuant V2 4-bit lean cache |
| `dflash-turboquant-rot` | DFlash target cache with TurboQuant V2 4-bit rotation/normalization |
| `dflash-triattention` | DFlash with TriAttention MLX compression, norm-only scoring, KV budget 2048 |

TurboQuant and TriAttention were tested through the DFlash cache-fusion path already
present in this repository. I did not treat their current pure-cache prototypes as
production candidates because this repo's stable Qwen3.6 path is the DFlash integration.

## Short Throughput

Prompt: 34 tokens. Generation: 256 tokens. Values are means over 4 runs
(`kv2048` and `kv8192`, 2 repeats each).

| Variant | Decode tok/s | TTFT | Peak GB | Acceptance mean |
| --- | ---: | ---: | ---: | ---: |
| `mlx-kv4` | 55.07 | 0.503 s | 15.32 | n/a |
| `baseline` | 50.50 | 0.674 s | 15.32 | n/a |
| `dflash-turboquant-rot` | 22.84 | 0.479 s | 16.32 | 2.57 |
| `dflash-turboquant-lean` | 19.06 | 0.444 s | 16.32 | 2.10 |
| `dflash-kv4` | 18.92 | 0.523 s | 16.32 | 2.10 |
| `dflash` | 18.87 | 0.465 s | 16.33 | 2.05 |
| `dflash-triattention` | 18.66 | 0.430 s | 16.33 | 2.05 |

Interpretation: low acceptance makes DFlash slower than vanilla. MLX KV4 is marginally
faster than baseline on short decode but does not improve long-context behavior.

## Context Sweep

Each context row requested 32 generated tokens. Context targets above 65,536 are blocked
by the runner unless `--unsafe-allow-large-context` is passed, because a previous 131K
probe hard-crashed this Mac.

### Baseline

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s | Peak GB |
| ---: | ---: | ---: | ---: | ---: |
| 4,119 | 4.24 s | 1032.82 | 42.04 | 17.03 |
| 16,391 | 20.06 s | 829.06 | 44.82 | 18.09 |
| 32,771 | 48.84 s | 674.14 | 40.91 | 19.51 |
| 65,557 | 138.17 s | 475.40 | 28.12 | 22.41 |

### MLX KV4

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s | Peak GB |
| ---: | ---: | ---: | ---: | ---: |
| 4,119 | 4.71 s | 905.86 | 52.85 | 16.97 |
| 16,391 | 21.37 s | 773.34 | 38.88 | 17.85 |
| 32,771 | 54.89 s | 599.51 | 30.60 | 19.05 |
| 65,557 | 162.08 s | 405.04 | 18.92 | 21.48 |

### DFlash Family

| Variant | Prompt tokens | TTFT | Prefill tok/s | Decode tok/s | Peak GB | Tri cache len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dflash` | 4,119 | 6.18 s | 682.32 | 17.73 | 22.24 | n/a |
| `dflash` | 16,391 | 69.23 s | 237.28 | 1.55 | 39.57 | n/a |
| `dflash` | 32,771 | 197.89 s | 165.96 | 1.47 | 62.97 | n/a |
| `dflash-kv4` | 4,119 | 6.35 s | 666.24 | 15.86 | 22.24 | n/a |
| `dflash-kv4` | 16,391 | 59.81 s | 274.98 | 2.20 | 38.94 | n/a |
| `dflash-kv4` | 32,771 | 170.06 s | 193.17 | 2.24 | 61.71 | n/a |
| `dflash-turboquant-lean` | 4,119 | 6.63 s | 638.20 | 16.22 | 22.24 | n/a |
| `dflash-turboquant-lean` | 16,391 | 58.03 s | 283.20 | 3.15 | 38.94 | n/a |
| `dflash-turboquant-lean` | 32,771 | 133.26 s | 246.27 | 2.55 | 61.71 | n/a |
| `dflash-turboquant-rot` | 4,119 | 5.43 s | 785.57 | 11.39 | 22.11 | n/a |
| `dflash-turboquant-rot` | 16,391 | 68.68 s | 239.41 | 2.69 | 38.95 | n/a |
| `dflash-turboquant-rot` | 32,771 | 137.73 s | 238.35 | 2.08 | 61.85 | n/a |
| `dflash-triattention` | 4,119 | 5.54 s | 761.13 | 12.32 | 22.24 | 2,056 |
| `dflash-triattention` | 16,391 | 53.92 s | 304.82 | 2.85 | 39.57 | 2,055 |
| `dflash-triattention` | 32,771 | 134.99 s | 243.09 | 2.67 | 62.97 | 2,051 |

Interpretation: DFlash-family long context is not viable here. TriAttention keeps a
compressed cache around 2K entries, but total runtime memory and TTFT still blow up in
this integration.

## Max Generation

Forced 8192-token generation, short prompt, `--disable-eos-stop`, KV cap 12,288.

| Variant | Generated | Finish | TTFT | Decode tok/s | Peak GB | Acceptance mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `dflash-turboquant-lean` | 8,192 | length | 1.30 s | 65.32 | 16.55 | 7.88 |
| `dflash-kv4` | 8,192 | length | 0.87 s | 64.65 | 16.55 | 7.88 |
| `mlx-kv4` | 8,192 | length | 1.36 s | 54.02 | 15.32 | n/a |
| `baseline` | 8,192 | length | 0.90 s | 53.86 | 15.46 | n/a |
| `dflash` | 8,192 | length | 1.02 s | 50.10 | 16.73 | 6.16 |
| `dflash-triattention` | 8,192 | length | 0.99 s | 45.20 | 16.62 | 5.52 |
| `dflash-turboquant-rot` | 8,192 | length | 0.86 s | 13.79 | 16.56 | 1.68 |

Interpretation: DFlash can help only for long uninterrupted generation, and the best
observed long-generation profile is `dflash-turboquant-lean` or `dflash-kv4`. That is
not the typical Hermes Agent workload, where repeated tool turns and context growth
matter more than a single 8192-token uninterrupted stream.

## Max Context

| Variant | Max verified context | Status |
| --- | ---: | --- |
| `baseline` | 65,557 prompt tokens | Safe but TTFT is 138 s |
| `mlx-kv4` | 65,557 prompt tokens | Safe but slower than baseline at 65K |
| `dflash` | 32,771 prompt tokens | Unsafe to push further; 62.97 GB peak at 32K |
| `dflash-kv4` | 32,771 prompt tokens | Unsafe to push further; 61.71 GB peak at 32K |
| `dflash-turboquant-lean` | 32,771 prompt tokens | Unsafe to push further; 61.71 GB peak at 32K |
| `dflash-turboquant-rot` | 32,771 prompt tokens | Unsafe to push further; 61.85 GB peak at 32K |
| `dflash-triattention` | 32,771 prompt tokens | Unsafe to push further; 62.97 GB peak at 32K |

## Recommendation for Hermes Agent

Use vanilla `baseline` first.

Suggested production envelope:
- Default context budget: 8K to 32K.
- Hard local ceiling: 65K only for exceptional requests.
- Avoid 131K on this host.
- Avoid DFlash-family variants for agent loops or long-context sessions.

Potential specialized profile:
- `dflash-turboquant-lean` or `dflash-kv4` only for explicit long-form generation
  jobs where the prompt is short and the model is expected to produce thousands of
  tokens without tool calls.

Do not use:
- `dflash-turboquant-rot`; it is slow and output previews visibly degraded.
- DFlash-family context above 32K on this machine.

## Artifact Index

Main runner:
- `scripts/benchmark_qwen36_optimization_variant.py`

Short/context matrices:
- `artifacts/benchmarks/qwen36-optimization-matrix/matrix-baseline/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/matrix-mlx-kv4/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/matrix-dflash/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/safe-matrix-dflash-kv4/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/safe-matrix-dflash-turboquant-lean/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/safe-matrix-dflash-turboquant-rot/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/safe-matrix-dflash-triattention/result.json`

8192-token generation:
- `artifacts/benchmarks/qwen36-optimization-matrix/gen8192-baseline/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/gen8192-mlx-kv4/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/gen8192-dflash/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/gen8192-dflash-kv4/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/gen8192-dflash-turboquant-lean/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/gen8192-dflash-turboquant-rot/result.json`
- `artifacts/benchmarks/qwen36-optimization-matrix/gen8192-dflash-triattention/result.json`
