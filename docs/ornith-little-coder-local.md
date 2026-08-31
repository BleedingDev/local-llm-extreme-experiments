# Ornith-1.0-35B with Little Coder

The default local path is the MLX 4-bit checkpoint:

```text
mlx-community/Ornith-1.0-35B-4bit
```

Serve it with `mlx-vlm`, then use Little Coder as:

```text
omlx/mlx-community/Ornith-1.0-35B-4bit
```

The old GGUF/llama.cpp path is not the default. The first GGUF run was invalid
as a performance signal because the wrapper forced MoE expert layers to CPU.

## Runtime

Use the repo venv. It has the loader stack used for the speed gate:

```bash
.venv/bin/python - <<'PY'
import importlib.metadata as m
for dist in ["mlx-vlm", "mlx-lm", "mlx", "mlx-metal"]:
    print(dist, m.version(dist))
PY
```

Measured stack:

```text
mlx-vlm 0.6.3
mlx-lm 0.31.3
mlx 0.31.2
mlx-metal 0.31.2
```

The user Little Coder model registry lives at:

```text
~/.config/little-coder/models.json
```

It maps `omlx` to the local role-normalizing proxy at
`http://127.0.0.1:8001/v1`. The proxy forwards to the `mlx-vlm` server at
`http://127.0.0.1:8000/v1`.

## Speed Gate

Run the raw generation sweep:

```bash
ORNITH_MLX_VLM_SWEEP_REPEATS=2 \
ORNITH_MLX_VLM_SWEEP_MAX_TOKENS=256 \
scripts/sweep_ornith_mlx_vlm_speed.sh
```

Latest result on this M5 / 32 GB Mac:

| Case | Decode tok/s mean | Decode tok/s min | Prompt tok/s mean | Peak GB |
| --- | ---: | ---: | ---: | ---: |
| `p4096_no_thinking` | 55.39 | 55.34 | 254.46 | 21.00 |
| `p2048_kv4_no_thinking` | 54.80 | 54.58 | 258.67 | 21.00 |
| `p2048_thinking_budget256` | 54.46 | 54.27 | 254.35 | 21.00 |
| `p2048_no_thinking` | 51.87 | 48.45 | 256.13 | 21.00 |

Recommended raw profile:

```text
mlx-vlm server/generate
prefill-step-size=4096
no KV quantization for short agent turns
thinking disabled for speed-gate runs
```

Thinking is viable for comparison: with a 256-token thinking budget it stayed
above 54 tok/s in the raw generation sweep.

## Server

Start the OpenAI-compatible server:

```bash
scripts/run_ornith_mlx_vlm_server.sh
```

Single-client server speed through `/v1/chat/completions` cleared the 30 tok/s
gate:

```text
concurrency=1, 256 completion tokens, 36.46 tok/s
```

A later warm server pass showed:

```text
concurrency=1: 51.24 tok/s aggregate decode
concurrency=2: 78.06 tok/s aggregate decode
concurrency=4: 96.60 tok/s aggregate decode
```

The server should be stopped after benchmarking unless a Little Coder or
Terminal-Bench run is about to use it.

## Little Coder

Terminal 1:

```bash
scripts/run_ornith_mlx_vlm_server.sh
```

Terminal 2:

```bash
scripts/ornith_openai_proxy.py \
  --upstream http://127.0.0.1:8000 \
  --log bench/runs/ornith_openai_proxy.jsonl
```

Terminal 3:

```bash
scripts/run_little_coder_ornith.sh
```

The wrapper defaults to:

```bash
LITTLE_CODER_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
LITTLE_CODER_MAX_TOKENS=768
little-coder --model omlx/mlx-community/Ornith-1.0-35B-4bit
```

To compare thinking mode:

```bash
LITTLE_CODER_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}' \
scripts/run_little_coder_ornith.sh
```

## Terminal-Bench 2.1

Do not start full Terminal-Bench until the server is already running and a small
Little Coder smoke passes.

Prepare Little Coder's dev-only benchmark harness:

```bash
scripts/setup_little_coder_tb_harness.sh
```

Smoke one trial at low concurrency:

```bash
TB_K=1 TB_N_CONCURRENT=1 scripts/run_tb21_little_coder_ornith.sh \
  --include-task-name "terminal-bench/regex-log"
```

The TB wrapper defaults to:

```bash
TB_WARMUP=1
TB_MAX_TURNS=20
TB_MAX_OUTPUT_TOKENS=768
LITTLE_CODER_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
```

The warmup sends one small OpenAI-compatible tool-call request through the
proxy. This avoids charging the first real task for MLX's first-generation
compile path.

Thinking comparison:

```bash
TB_THINK=1 TB_K=1 TB_N_CONCURRENT=1 scripts/run_tb21_little_coder_ornith.sh \
  --include-task-name "terminal-bench/regex-log"
```

Full 2.1 evaluation uses 5 trials:

```bash
TB_K=5 TB_N_CONCURRENT=1 scripts/run_tb21_little_coder_ornith.sh
```

On this Mac, full TB 2.1 should be treated as local performance probing, not a
leaderboard-comparable score. The Ornith model card used a larger setup with
128K context, 4-hour task timeout, 32 CPU cores, 48 GB RAM, and five runs.

## Short Real-Task Probe

The current local profile was checked against `terminal-bench/regex-log`.

| Profile | Result | Wall | Model request time | Completion tokens | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| forced first tool, `1024` cap | pass | 100.23s | 73.27s | 1178 | 16.08 |
| forced first tool, `512` cap | fail | 66.89s | 42.42s | 1489 | 35.10 |
| full Little Coder profile, `768` cap, warm server, repeat 1 | pass | 54.63s | 29.38s | 1031 | 35.10 |
| full Little Coder profile, `768` cap, warm server, repeat 2 | pass | 55.36s | 29.41s | 1031 | 35.05 |

The 1024-token cap allowed a 915-token tool-call turn and the server later
crashed with a Metal out-of-memory error. The 512-token cap preserved speed but
hurt task quality. The restored full Little Coder profile with a 768-token cap
is the current default because it passed repeated real-task probes while keeping
aggregate real-agent decode above 30 tok/s and leaving the server alive.

Rejected experiments:

- TB lean prompt: reduced prompt tokens but caused prose-only failure.
- TB prose guard: recovered pass status but inflated wall time with extra
  768-token turns.
- Forced `tool_choice=ShellSession`: not reliable for real Little Coder
  conversation context on `mlx-vlm`.
- `enable_thinking=true`: did not fix the prose-only failure pattern in the
  `regex-log` probe.
