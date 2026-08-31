# Qwen3.6 35B A3B RotorQuant MLX 3-bit benchmark

Date: 2026-04-29

Host:
- Apple M5
- 34,359,738,368 bytes unified memory
- 10 CPU cores
- macOS 26.3 build 25D5087f

Model:
- `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit`
- Hugging Face revision: `50113fa5f9a705d903b913e1998054c7fb1fdcb8`
- Snapshot size reported by Hugging Face API: 15.2 GB
- Local runtime: `mlx 0.31.1`, `mlx-lm 0.31.2`

## Summary

The 3-bit MLX model is real and fast on this machine. Vanilla `mlx_lm.stream_generate`
loads it directly and short-context decode is consistently around 53 tok/s with normal
EOS handling. A forced 8192-token generation stress run reached 60.56 tok/s and stopped
by length.

The practical context ceiling on this 34 GB M5 is 65K prompt tokens. A 131K prompt-token
probe hard-crashed the Mac, so this runner now refuses context targets above 65,536 unless
`--unsafe-allow-large-context` is passed explicitly.

## Throughput

Artifact:
`artifacts/benchmarks/qwen36-rotorquant-mlx/vanilla-full-rotorquant-3bit-v2/result.json`

Short prompt, 34 prompt tokens, 256 generated tokens per run:

| KV cap | Mean decode tok/s | Mean TTFT |
| --- | ---: | ---: |
| 2048 | 53.36 | 0.69 s |
| 4096 | 52.64 | 0.29 s |
| 8192 | 54.41 | 0.26 s |

Warm-run mean excluding the first cold measured pass:
- Decode: 53.45 tok/s
- TTFT: 0.274 s
- Prefill: 190.46 tok/s
- Peak memory: 15.32 GB

## Context

Artifact:
`artifacts/benchmarks/qwen36-rotorquant-mlx/vanilla-context-rotorquant-3bit-v3/result.json`

Corrected string-prompt context sweep, 32 generated tokens requested:

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s | Peak memory |
| ---: | ---: | ---: | ---: | ---: |
| 4,119 | 6.05 s | 696.41 | 39.28 | 17.03 GB |
| 16,391 | 22.79 s | 723.57 | 48.28 | 18.09 GB |
| 32,771 | 47.61 s | 690.63 | 38.23 | 19.51 GB |
| 65,557 | 143.36 s | 457.87 | 28.43 | 22.41 GB |

131K prompt-token probe: failed at the system level; do not repeat on this host without a
separate watchdog and much stricter memory controls.

## Max Generation

Artifact:
`artifacts/benchmarks/qwen36-rotorquant-mlx/vanilla-generation-8192-noeos-rotorquant-3bit-v1/result.json`

Short prompt, EOS stop disabled for stress testing, KV cap 12,288:

| Requested | Generated | Finish | TTFT | Decode tok/s | Peak memory |
| ---: | ---: | --- | ---: | ---: | ---: |
| 8,192 | 8,192 | length | 1.37 s | 60.56 | 15.46 GB |

With normal EOS handling, the same model naturally stopped at 684 to 1,359 tokens on the
tested prompts, so forced stress generation should not be confused with normal assistant
behavior.

## Hermes Agent Readiness

Recommended vanilla profile for this host:
- Model: `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit`
- Context budget: keep production prompts below 32K by default
- Emergency/high-context budget: 65K max, with expected TTFT around 2.4 minutes
- Avoid: 131K context on this 34 GB M5; it hard-crashed the OS

Next work:
- Test Hermes prompt/tool-call behavior at 8K and 32K contexts.
- Only after that, try DFlash or other speculative paths. Do not combine that with >32K
  context until low-context behavior is stable.
