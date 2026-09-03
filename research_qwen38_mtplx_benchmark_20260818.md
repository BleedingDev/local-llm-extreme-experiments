# Qwen3.8 27B MTPLX context benchmark

Date: 2026-08-18

Machine: MacBook Pro Mac17,2, base Apple M5, 32 GB unified memory

Confidence: High for measured points; medium for the practical recommendation because unrelated Apple services repeatedly respawned during sustained runs

## Executive summary

The optimized Qwen3.8 27B checkpoint is a good interactive local model through roughly 8K prompt tokens on this Mac. Across quality-gated runs, decode averaged 12.36 tok/s at 525 prompt tokens, 12.14 tok/s at 2,061, and 10.01 tok/s at 8,205. At 32,781 prompt tokens, time to first token rose to 319.2 seconds and decode fell to 3.38 tok/s. A 65,536-target run failed during prefill with a Metal out-of-memory error.

The practical recommendation is therefore an 8K working context for interactive use. A 32K window is technically viable for occasional retrieval but is too slow for normal agent loops. 64K is not viable with unquantized prefill on this 32 GB configuration.

## Methodology

The benchmark follows this repository's existing philosophy: resolved configuration, deterministic inputs, raw response evidence, parsed metrics, failure classification, memory/thermal observation, and an explicit context-ceiling probe.

Each prompt contained unique indexed records with BLAKE2s checksums and one `BENCHMARK_NEEDLE` inserted after 73% of the filler. A run passed the basic quality gate only if Qwen recovered the exact needle. Requests used MTPLX 2.8.3, the `turbo` profile, native MTP depth 2, greedy sampling, reasoning off, serial scheduling, and no SSD session cache.

## Results

| Actual prompt tokens | Runs | Prefill tok/s | Decode tok/s | Mean TTFT | Max active memory | Peak memory | Needle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 525 | 2 | 121.38 | 12.36 | 4.34 s | 20.26 GB | 22.01 GB | 2/2 |
| 2,061 | 2 | 186.31 | 12.14 | 11.28 s | 21.45 GB | 23.81 GB | 2/2 |
| 8,205 | 2 | 266.13 | 10.01 | 33.51 s | 24.04 GB | 26.47 GB | 2/2 |
| 32,781 | 1 | 102.81 | 3.38 | 319.24 s | 23.07 GB | 27.82 GB | 1/1 |
| ~65,536 target | 1 | failed | failed | failed | — | OOM | no result |

The 32K request took 337.9 seconds end to end for 64 completion tokens. Its end-to-end generation rate was 0.189 tok/s once prompt processing was included. MTP draft acceptance remained 77.3% at depth 1 and 52.4% at depth 2, so the decode collapse is primarily a long-context/runtime cost rather than total draft rejection.

## Context ceiling

The 64K attempt repeatedly triggered MTPLX memory-pressure level 4 and failed in Metal with `kIOGPUCommandBufferCallbackErrorOutOfMemory` during suffix prefill. A fresh Q8 paged-KV server reported that Q8 only reduces decode KV memory and leaves peak prefill memory unchanged. A Q8 plus 512-token-chunk attempt was therefore treated as exploratory and cancelled when the hottest sensor reached 105.4 C and free memory fell to 9%; it is not a successful benchmark point.

## Thermal and background behavior

The 32K isolated run began at 55.2 C and ended at 83.4 C; battery temperature moved from 27.2 C to 29.1 C. During prefill, the hottest compute/memory sensor briefly exceeded 100 C even with the existing fan controller at maximum. macOS recorded no thermal or performance warning.

App Store, `analyticsd`, and `BiomeAgent` repeatedly respawned during sustained runs despite being terminated between points. `dasd` also consumed substantial CPU but was left untouched. Short-context repeats are relatively robust; the very long wall-clock points should be understood as realistic busy-Mac lower bounds rather than clean-room maxima.

## Strengths and weaknesses

Strengths:

- Quality gate passed at every completed context.
- MTP retains useful acceleration and acceptable draft rates through 8K.
- Battery temperature stayed low despite sustained compute load.
- The 21.3 GB model leaves enough memory for ordinary short-context serving.

Weaknesses:

- TTFT grows from seconds to more than five minutes by 32K.
- Decode drops from roughly 12 tok/s to 3.38 tok/s at 32K.
- 64K unquantized prefill exceeds the Metal memory budget.
- Repeated background-service activity makes extreme-context runs noisy.

## Artifacts

- Short-context matrix: `artifacts/benchmarks/qwen38-mtplx-context-20260818T174703Z/`
- Long-context/OOM probe: `artifacts/benchmarks/qwen38-mtplx-long-context-20260818T175337Z/`
- Aborted Q8/chunk-512 probe: `artifacts/benchmarks/qwen38-mtplx-q8-chunk512-20260818T181643Z/`
- Reusable runner: `scripts/benchmark_mtplx_qwen38_context.py`

## Sources

- Benchmark methodology: https://github.com/BleedingDev/local-llm-extreme-experiments
- MTPLX runtime: https://github.com/youssofal/MTPLX
- Model checkpoint: https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed

## Confidence assessment

- High: measured throughput, TTFT, memory telemetry, quality results, and the 64K Metal OOM.
- High: Q8 does not reduce peak prefill memory; this is declared by the running MTPLX health contract.
- Medium: the exact clean-system long-context throughput, because Apple background services repeatedly respawned during the campaign.
