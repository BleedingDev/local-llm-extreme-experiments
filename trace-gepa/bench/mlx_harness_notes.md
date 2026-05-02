# MLX harness notes (`run_mlx.py`)

Phase-3 Bench Agent #7 — workhorse for the eventual fine-tuned local-model
evaluation (Option C).

## Environment

- `mlx_lm` is installed in `.venv` only (version `0.31.2`).
- `mlx_lm` is **NOT** installed in `.venv-gepa` (the venv requested by the
  spec). Smoke run was executed with `.venv/bin/python`. Per task constraints
  I did not auto-install. To unify:
  ```bash
  .venv-gepa/bin/pip install mlx-lm
  ```
- `mlx.core.get_peak_memory()` is the current API; the legacy
  `mx.metal.get_peak_memory()` is deprecated. The harness prefers the new one.
- HuggingFace cache: `~/.cache/huggingface/hub/`. No MLX-quantised weights
  were pre-cached; the harness pulls them on first load. The 1B-4bit Llama
  was 9 s download on a warm uplink.

## Models considered

| repo                                                  | size  | status                    |
|-------------------------------------------------------|-------|---------------------------|
| `mlx-community/Llama-3.2-1B-Instruct-4bit`            | ~700M | smoke target (downloaded) |
| `mlx-community/Qwen3.5-3B-Instruct-4bit`              | ~2GB  | not cached, would fit     |
| `mlx-community/gemma-4-e2b-it-4bit`                   | ~1.5GB| not cached                |
| `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2`     | ~14GB | skipped (large/slow)      |

The README's listed `Qwen3.5-3B`, `gemma-4-e2b-it`, and `supergemma4-26b` repos
were **not** present locally. Llama-3.2-1B is the smallest plausible chat
model with a Metal-resident MLX 4-bit conversion and was chosen as the smoke
target so we get real numbers instead of a "skipped" file.

## Smoke run (10 tool-routing tasks)

```
mlx-community/Llama-3.2-1B-Instruct-4bit
n_pass=4/10 (40 %)        peak_mem=0.99 GB
mean_tokens_per_sec=93.3  load=~1.0 s (cold) / 0.7 s (warm)
```

Results: `trace-gepa/bench/results/smoke_mlx.json`.

The 6 misses are predominantly `json_parse_fail` — Llama-3.2-1B emits the
JSON envelope inside Markdown fences or trailing prose for harder routing
prompts. The 4 passes are clean structural-JSON outputs. Stripping fences in
`_parse_predicted` recovered some but not all of these.

## Throughput / memory expectations

| model                              | tok/s   | peak GB | cold load |
|-------------------------------------|---------|---------|-----------|
| Llama-3.2-1B-Instruct-4bit (measured)| 90-100 | ~1.0    | ~1 s      |
| Qwen3.5-3B-Instruct-4bit (expected)  | 35-50  | ~2.5    | 5-10 s    |
| gemma-4-e2b-it-4bit (expected)       | 25-40  | ~2.0    | 5-10 s    |
| supergemma4-26b-mlx-4bit (expected)  | 6-12   | ~14     | 30-60 s   |

Expectations are typical figures for a 16-32 GB unified-memory M-series Mac.
First task in the smoke shows the pattern of one runaway generation
(`task_tool_routing_001`, 513 tokens, 5 s) versus terse routing decisions
(40-60 tokens, ~0.5 s). With 4-bit quant and `max_tokens=512` the harness
stays well under the 60 s per-task timeout for sub-3B models.

## Behavioural notes

- `_alarm` uses `SIGALRM`; only works on POSIX (macOS, Linux). Windows
  fallback is a no-op timeout.
- OOM detection is heuristic — Metal allocator failures surface as generic
  `RuntimeError` with `memory`/`alloc`/`metal` in the message. We bin those
  into `skipped_oom`.
- Output JSON shape mirrors the Anthropic/Codex aggregator (`pass_rate`,
  `by_category`, `by_verifier_kind`, `signal_counts`, `per_task`) plus
  MLX-specific `mean_tokens_per_sec`, `peak_memory_gb`, `load_seconds`.
- The verifier dispatch uses `bench/verifiers/tier1_regex.py` directly
  (no `bench.verifiers.verify` umbrella exists yet); the harness exposes its
  own `verify(task, predicted)` helper.

## Realistic deployment readiness

A 1B 4-bit model running 90 tok/s with 40 % pass on tool-routing is
**not** production-ready as a coding-agent backbone. It can hit easy routes
(Read, Bash) but blows the JSON envelope on context-heavy prompts. The
harness itself is production-shaped — single-threaded, alarm-bounded,
OOM-trapping, peak-memory aware — so the real next step is running the same
loop against Qwen3.5-3B or a fine-tune of the supergemma checkpoint, where
we expect 60-80 % pass at 30-50 tok/s, which is the band where on-device
inference starts to be useful as a fallback or first-pass reranker rather
than a primary agent.

## CLI

```bash
.venv/bin/python trace-gepa/bench/run_mlx.py \
    --tasks trace-gepa/data/benchmark_tasks.jsonl \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --limit 10 \
    --output trace-gepa/bench/results/smoke_mlx.json
```
