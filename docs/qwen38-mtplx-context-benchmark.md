# Qwen3.8 MTPLX context benchmark

`scripts/benchmark_mtplx_qwen38_context.py` adapts this repository's existing matrix and context-probe methodology to an OpenAI-compatible MTPLX server.

`scripts/benchmark_mtplx_qwen38_batch.py` measures aggregate delivered tokens per
second across concurrent requests. Run it against separately launched serial,
AR-batch, or compatible MTP-batch servers; it does not alter the server itself.

It measures actual prompt size rather than only configured KV capacity. Each prompt contains deterministic, unique indexed records and one needle after 73% of the filler. A run passes its basic quality gate only when the response recovers that exact needle.

Captured per run:

- prompt and completion tokens
- prefill, decode, and end-to-end throughput
- time to first token and phase timings
- active, peak, and cache memory
- MTP acceptance at draft depths 1 and 2
- fan speed, hottest sensor, and system load before and after
- battery temperature before and after
- finish reason, output text, raw response telemetry, and needle quality result

Dry-run the default matrix:

```bash
/opt/homebrew/var/mtplx/venv-2.8.3/bin/python \
  scripts/benchmark_mtplx_qwen38_context.py --dry-run
```

The default matrix benchmarks optimized MTP depth 2 at 512, 2K, 8K, 32K, and 64K prompt targets. It also records AR references at 512 and 8K. Short contexts run twice; long contexts run once. Larger targets remain available through `--contexts`, but are no longer defaults because the 32 GB M5 campaign found a Metal OOM at 64K before larger points could be useful.

By default, a run does not start until the hottest reported sensor is at or below 65 C. This prevents a context point from inheriting the previous point's accumulated heat. Override with `--max-start-temp` only when a different controlled thermal policy is intentional.

Artifacts follow the existing repository convention under `artifacts/benchmarks/qwen38-mtplx-context-<UTC>/`:

- `config.json`
- `prompt-spec.txt`
- `results.csv`
- `results.json`
- `summary.md`
- `raw/*.request-meta.json`
- `raw/*.response.json`
- `raw/*.output.txt`
