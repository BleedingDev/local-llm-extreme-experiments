# Qwen3.8 27B MTPLX batching benchmark

Date: 2026-08-18

Machine: MacBook Pro Mac17,2, base Apple M5, 32 GB unified memory

Runtime: MTPLX 2.8.3, Qwen3.8 27B Optimized-Speed, Turbo profile, reasoning off

## Workload

Each request contained 555 actual prompt tokens and requested 256 completion
tokens. The output had to begin with a unique worker marker and continue into a
long checklist so early stopping would not make a high-concurrency result look
artificially fast. Aggregate throughput is total delivered completion tokens
divided by cohort wall time.

## Results

| Scheduler | Width | Delivered tokens | Cohort wall time | Aggregate tok/s | Mean job latency | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial MTP depth 2 | 1 | 256 | 24.49 s | 10.45 | 24.49 s | 1/1 |
| Serial MTP depth 2 | 2 | 512 | 50.24 s | 10.19 | 38.09 s | 2/2 |
| Serial MTP depth 2 | 4 | 1,024 | 104.22 s | 9.83 | 66.28 s | 4/4 |
| Batched AR | 1 | 256 | 46.13 s | 5.55 | 46.13 s | 1/1 |
| Batched AR | 2 | not delivered | >311 s | <=1.64 compute-only | >307 s | operational failure |

Serial MTP therefore sustains about 37,600 useful completion tokens/hour at B1.
Queueing four requests reduces aggregate throughput by about 6% and makes the
last job wait over 100 seconds, without increasing useful output.

The live B2 AR scheduler reported two active requests and a batch size of two, so
the bad result was not accidental serialization. Each generation took roughly
five minutes and final HTTP delivery remained blocked in post-commit work. B4 was
cancelled because no plausible crossover remained.

## Batched-MTP compatibility result

MTPLX's `mtp_batch` path was also probed. Version 2.8.3 required depth 1, a
131,072-token context window, `target_prefix` verification, and the stock verify
core. After satisfying those launch constraints, the runtime rejected this dense
27B checkpoint because the installed row-owned batch router requires the exact
Qwen A3B topology. Batched MTP is therefore unavailable for this model/runtime
combination; it is not a hidden speed setting for this checkpoint.

## Recommendation for unattended automation

Use serial MTP depth 2 with one active generation worker and an application-level
durable job queue. A second queued job is reasonable for fault tolerance and to
avoid idle gaps, but it is not a throughput multiplier. Keep prompts near or below
8K tokens, prefer longer completion jobs so prefill is amortized, and keep the
server resident overnight to avoid repeated model load and warmup.

Do not use `ar_batch`, `mtp_batch`, multiple local model processes, or Q8 KV as a
throughput strategy for this dense 27B checkpoint on 32 GB. Two model processes
cannot fit: one server reports approximately 19.8 GB of model weights before KV,
runtime, and macOS memory.

Actual horizontal scaling requires another Apple Silicon machine. Otherwise the
meaningful optimization lever is workload shaping: compact/RAG-selected context,
serial generation, bounded outputs, checkpointed jobs, and prefix/session reuse
for genuinely repeated histories.

## Evidence

- `artifacts/benchmarks/qwen38-mtplx-batch-20260818T192000Z/serial-mtp.json`
- `artifacts/benchmarks/qwen38-mtplx-batch-20260818T192000Z/ar-batch.json`
- `artifacts/benchmarks/qwen38-mtplx-batch-20260818T192000Z/AR_BATCH_ABORTED.md`
- `scripts/benchmark_mtplx_qwen38_batch.py`
