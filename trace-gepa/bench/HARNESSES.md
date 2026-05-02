# Harnesses

Three reference harnesses ship with v1. They share an output envelope (see
`SCHEMA.md`) but differ in models, transport, concurrency, and venv. All
three load tasks from `trace-gepa/data/benchmark_tasks.jsonl` and dispatch
scoring through `bench.verifiers.verify`.

## 1. Anthropic API — `bench/run_anthropic.py`

Calls the Anthropic Messages API directly via `agent_opt.llm.chat`.

### Models
- `claude-opus-4-7` (default; the 1M-context Opus 4.7).
- Any other Anthropic model id accepted by `agent_opt.llm` works via `--model`.

### Environment
- Venv: `trace-gepa/.venv-gepa` (the GEPA venv).
- Auth: reads `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`) from
  `trace-gepa/.env`. The SDK is loaded inside `agent_opt/llm.py`.

### CLI

```bash
.venv-gepa/bin/python trace-gepa/bench/run_anthropic.py \
    --tasks trace-gepa/data/benchmark_tasks.jsonl \
    --model claude-opus-4-7 \
    --max-workers 8 \
    --max-tokens 512 \
    --limit 0 \
    --output trace-gepa/bench/results/anthropic_full.json
```

### Wallclock + concurrency
- `ThreadPoolExecutor(max_workers=8)`.
- Smoke (20 tasks, all `tool_routing`): ~8 s. Full (105 tasks): ~45–90 s
  depending on contention. Token usage: ~7k in / ~500 out per 20-task batch.

### Gotchas
- **Opus 4.7 rejects `temperature`.** `agent_opt/llm.py` already handles this:
  the chat helper retries once without `temperature` if the API errors with
  `"temperature"` in the message. Do not re-introduce a hard-coded temperature
  override at the call site.
- The harness has an in-file fallback verifier (`_fallback_verify`) that
  handles `regex` and `structural_json` only. It is used iff
  `bench.verifiers.verify` cannot be imported. Production runs should use
  the umbrella verifier — confirm `verifier: "bench.verifiers.verify"` in the
  output JSON.

## 2. Codex CLI — `bench/run_codex.py`

Drives the local `codex` binary in non-interactive mode.

### Models
- Validated: `gpt-5.5` with reasoning effort `high` and `xhigh`.
- Config-listed but unvalidated: `gpt-5.5-mini`, `gpt-5.4`,
  `gpt-5.3-codex-spark`. Pass via `--model <name>`; failures show as non-zero
  exit codes plus `stderr_tail` in the output JSON.

### Environment
- Binary: `codex-cli 0.128.0` (`/opt/homebrew/bin/codex` on the dev box).
  Override with `--codex-bin` or `CODEX_BIN`.
- Auth: handled entirely by codex itself via `~/.codex/auth.json`. The
  harness does not touch that file. If the probe task hits common
  auth-failure phrases (`unauthorized`, `not authenticated`, `401`,
  `auth.json`, ...) every remaining task is short-circuited to
  `error="codex_auth_failure"`, score 0.
- Venv: `trace-gepa/.venv-gepa` for the Python wrapper; codex itself is a
  separate binary.

### CLI

```bash
.venv-gepa/bin/python trace-gepa/bench/run_codex.py \
    --tasks trace-gepa/data/benchmark_tasks.jsonl \
    --model gpt-5.5 \
    --reasoning xhigh \
    --max-workers 4 \
    --timeout 90 \
    --output trace-gepa/bench/results/codex_full.json
```

The script translates `--reasoning <eff>` to `-c model_reasoning_effort=<eff>`
on the codex command line. Other invariant flags it always passes:
`--json --skip-git-repo-check --ephemeral --ignore-rules
--dangerously-bypass-approvals-and-sandbox -c approval_policy=never`.

### Wallclock + concurrency
- `ThreadPoolExecutor(max_workers=4)`. Codex spawns its own process per call.
- Smoke (5 tasks, `high`): ~24 s. 20 tasks `xhigh` ~86 s. Stratified 20
  across all categories, `xhigh`: ~108 s.
- Per-task subprocess timeout 90 s. On expiry the row is scored 0 and
  flagged `timed_out: true`.

### Gotchas
- `--reasoning` is **not** a top-level flag on `codex exec` 0.128. It only
  works through the `-c key=value` override; the harness handles this.
- The agent is asked to *return JSON describing the next action*, not to
  actually execute the tool. Sandbox bypass is benign in that posture but
  remove it if you ever rewire the harness to accept tool side-effects.

## 3. MLX local — `bench/run_mlx.py`

Single-process Apple-Silicon harness for local 4-bit chat models.

### Models
- Smoke: `mlx-community/Llama-3.2-1B-Instruct-4bit` — 90 tok/s, ~1 GB peak,
  `pass_rate=0.40` on the 10-task tool-routing smoke. **The 1B model is too
  small for production agent use**; it blows the JSON envelope on
  context-heavy prompts. Use as a smoke target only.
- Plausible upgrades (uncached on dev box): `mlx-community/Qwen3.5-3B-Instruct-4bit`
  (~35–50 tok/s, ~2.5 GB), `mlx-community/gemma-4-e2b-it-4bit`,
  `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` (~6–12 tok/s, ~14 GB).

### Environment
- Venv: **`trace-gepa/.venv` only.** `mlx_lm` is **not** installed in
  `.venv-gepa`. To unify: `.venv-gepa/bin/pip install mlx-lm`.
- mlx-lm version: `0.31.2`. Apple Silicon required.
- HF cache: `~/.cache/huggingface/hub/`. First load downloads the repo.
- Memory: uses `mlx.core.get_peak_memory()` (the legacy `mx.metal.get_peak_memory()`
  is fallback only).

### CLI

```bash
.venv/bin/python trace-gepa/bench/run_mlx.py \
    --tasks trace-gepa/data/benchmark_tasks.jsonl \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --limit 10 \
    --max-tokens 512 \
    --task-timeout 60 \
    --output trace-gepa/bench/results/smoke_mlx.json
```

### Wallclock + concurrency
- Single-threaded (Metal allocator does not multiplex well across threads).
- Per-task `SIGALRM` timeout (default 60 s; POSIX-only — Windows is no-op).
- 10 tool-routing tasks on Llama-3.2-1B: ~10 s wallclock, ~93 mean tok/s.
- OOM detection is heuristic: `RuntimeError` with `memory|alloc|metal` in
  the message is binned as `skipped_oom`.

### Gotchas
- If `mlx_lm` is not importable, the harness writes a `status: mlx_not_installed`
  stub to `--output` and exits 0 — design choice so the leaderboard ingester
  can still record the run. Do not interpret an empty MLX result as a pass.
- The MLX harness uses `bench.verifiers.tier1_regex` directly rather than the
  composite umbrella; tier-2 (judge) and tier-3 (shell) verifiers are not
  reachable from this harness in v1.
