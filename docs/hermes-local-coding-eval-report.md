# Hermes Local Coding Evaluation: Qwen3.6 35B A3B RotorQuant MLX 3-bit

Date: 2026-04-29  
Host: macOS 26.3, Apple M5, 34 GB RAM  
Hermes: v0.7.0 (2026.4.3), Python 3.11.14, OpenAI SDK 2.30.0  
Runtime: `mlx_lm.server` 0.31.2, OpenAI-compatible `chat_completions`  
Model: `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit`

## Recommendation

This model is usable in Hermes Agent for small and medium local coding tasks, but I would not yet call it a fully reliable autonomous local coding model.

It passed 4/6 tasks on the first attempt and repaired both failures when given the failing evidence once. The tool loop is functional, the latency is acceptable for local work, and the model can run tests and edit files through Hermes. The weak spot is reliability on subtle requirements: money rounding was applied only at CLI presentation instead of in the required library function, and async scheduling needed explicit failure feedback.

For Hermes Agent use, I would run it with guardrails:

- Always require tests to run.
- Prefer short, focused tasks.
- Feed failing test output back once before giving up.
- Avoid high-context agent runs. The earlier 65K-context benchmark worked but had very high TTFT; 131K previously crashed the Mac.
- Use vanilla MLX baseline for Hermes. DFlash/TurboQuant-style variants were not better for agentic loops and were unsafe or impractical at long context in prior benchmarks.

## Local Setup

MLX server command:

```bash
.venv/bin/mlx_lm.server \
  --model majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit \
  --host 127.0.0.1 \
  --port 18080 \
  --max-tokens 4096 \
  --chat-template-args '{"enable_thinking":false}' \
  --prefill-step-size 2048
```

Hermes isolated profile:

```yaml
model:
  default: majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit
  provider: custom
  base_url: http://127.0.0.1:18080/v1
  api_mode: chat_completions
```

Profile used: `hermeslocalqwen`  
Eval artifacts: `artifacts/hermes-coding-eval/20260429-034258`  
Harness: `scripts/run_hermes_coding_eval.py`

## Tool Calling

Direct MLX `/v1/chat/completions` with a `tools` payload did not return native OpenAI `tool_calls`; it returned a JSON-looking tool request inside normal assistant content.

Hermes still worked. In the smoke test it successfully called `read_file` and `write_file`, changing `artifacts/hermes-coding-eval/smoke-task/answer.txt` from `wrong` to `ok`. So Hermes' prompting/parsing layer is sufficient for this model, even though the raw server response is not native OpenAI tool-call output.

## First-Pass Results

| Task | Coverage | Hermes time | Visible tests | Hidden check | First-pass result |
|---|---:|---:|---:|---:|---:|
| `py_slug_unicode` | Unicode slugify bugfix | 75.270s | pass | pass | pass |
| `py_intervals` | Interval sorting/merge/validation | 32.632s | pass | pass | pass |
| `py_parser` | Quote-aware config parser | 31.008s | pass | pass | pass |
| `py_multifile_cli` | Multi-file CSV/Decimal CLI | 67.030s | pass | fail | fail |
| `js_lru_cache` | Node LRU cache | 29.798s | pass | pass | pass |
| `py_async_limiter` | Async concurrency/rate limiting | 152.348s | fail | pass | fail |

First-pass score: 4/6 = 66.7%.

## Repair Results

I gave the two failures one additional prompt with the exact failing evidence.

| Task | Repair session | Repair duration from Hermes session timestamps | Result |
|---|---:|---:|---:|
| `py_multifile_cli` | `20260429_035042_7a4626` | 48.152s | repaired; visible and hidden checks pass |
| `py_async_limiter` | `20260429_035144_55bc83` | 26.972s | repaired; visible and hidden checks pass |

After one repair prompt: 6/6 = 100%.

## Failure Analysis

`py_multifile_cli`:

- The model implemented the package and CLI correctly for visible tests.
- It rounded output in `sales.cli`, but the requirement said `summarize(rows)` itself must return cents-rounded `Decimal` totals.
- Hidden check caught `Decimal('3.005')` instead of `Decimal('3.01')`.
- With the exact hidden failure, Hermes repaired `sales/summary.py` correctly using `ROUND_HALF_UP`.

`py_async_limiter`:

- The model understood the broad shape of concurrent scheduling but used `await asyncio.sleep(interval)` for every task after the first.
- That made task 1 and task 2 start at nearly the same time.
- It spent many turns trying and then hit the max-turn summary with visible tests still failing.
- With the exact failure explanation, it fixed the delay to `idx * interval` and passed.

## Observations

- Tool use worked across file reads, file writes, patch attempts, terminal commands, and test execution.
- Hermes sometimes tried `pytest` first even though the tasks used `unittest`; it recovered by running `python3 -m unittest`.
- The model can explain and fix errors after test feedback, but it does not consistently infer subtle edge requirements on the first attempt.
- Quiet-mode Hermes still produced enough logs for auditability, including diffs, commands, sessions, and final summaries.
- The isolated profile added about 10 MB; eval artifacts are about 268 KB.

## Bottom Line

Qwen3.6 35B A3B RotorQuant MLX 3-bit is a promising local Hermes model, especially considering speed and memory footprint. It is good enough for supervised local coding, test-driven fixes, and small multi-file edits. It is not good enough for unattended coding-agent work where subtle requirements or async/concurrency behavior matter.

The next comparison should be against at least one stronger local coding-specific model through the same Hermes harness. Keep this exact suite as the baseline so we compare model reliability rather than changing the test.
