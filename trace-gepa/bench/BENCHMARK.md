# trace-gepa Bench

A single-step agent action-selection benchmark derived from real Codex / Claude
Code trace prefixes. Given a frozen prefix (user request, available tools,
recent actions), a model must emit one JSON object naming the next tool call.
Scoring is deterministic per task via a multi-tier verifier suite (regex,
structural-JSON, optional LM judge / shell exec). v1 ships 105 tasks across
7 categories with three reference harnesses (Anthropic API, Codex CLI, MLX
local).

## Quickstart

```bash
# 1. Activate the GEPA venv (Anthropic + Codex harnesses).
source trace-gepa/.venv-gepa/bin/activate

# 2. Anthropic smoke (Claude Opus 4.7, 5 tasks).
python trace-gepa/bench/run_anthropic.py --limit 5 \
    --output trace-gepa/bench/results/anth_smoke.json

# 3. Codex smoke (gpt-5.5, xhigh reasoning, 5 tasks).
python trace-gepa/bench/run_codex.py \
    --tasks trace-gepa/data/benchmark_tasks.jsonl \
    --model gpt-5.5 --reasoning xhigh --limit 5 \
    --output trace-gepa/bench/results/codex_smoke.json

# 4. MLX smoke (uses .venv, NOT .venv-gepa).
trace-gepa/.venv/bin/python trace-gepa/bench/run_mlx.py \
    --tasks trace-gepa/data/benchmark_tasks.jsonl \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit --limit 10 \
    --output trace-gepa/bench/results/mlx_smoke.json

# 5. Refresh leaderboard from results/.
python trace-gepa/bench/leaderboard.py
```

## Headline numbers (v1)

From `trace-gepa/bench/LEADERBOARD.md` (generated 2026-05-02):

| harness   | model                                       | mean_score | n   | wallclock |
|-----------|---------------------------------------------|------------|-----|-----------|
| codex     | gpt-5.5 (xhigh)                              | 1.000      | 20  | 86 s      |
| anthropic | claude-opus-4-7                              | 1.000      | 20  | 8 s       |
| codex     | gpt-5.5 (stratified across all 7 categories)| 0.850      | 20  | 108 s     |
| mlx       | Llama-3.2-1B-Instruct-4bit                   | 0.400      | 10  | 10 s      |

Cost-effectiveness leader (mean_score / sec_per_task) is `claude-opus-4-7`
at 2.52, with `gpt-5.5` close behind on absolute pass rate.

## Caveats

- Pre-FIX1 leaderboard rows are partly bogus: the original `tier1_regex`
  verifier read keys (`pattern`, `schema`) that the dataset writes as
  `pattern_or_command`, so `regex` tasks scored `regex_no_pattern=0.0` and
  `structural_json` tasks passed on any parseable JSON. FIX1+ harnesses use
  the corrected dispatcher; older rows under `command_synthesis` should be
  treated as N/A. See `bench/codex_harness_notes.md` for the failure modes.
- The 1B MLX model is a smoke target only — not production-grade.

## Deeper docs

- `SCHEMA.md` — task + result JSON schema.
- `HARNESSES.md` — per-harness CLI, env, expected wallclock, gotchas.
- `CONTRIBUTING.md` — adding tasks / verifiers / harnesses / models.
- `CHANGELOG.md` — release history.
- `COMPARATIVE_POSITIONING.md` — how this bench relates to SWE-bench,
  Terminal-Bench, Aider Polyglot, HumanEval.
- `LEADERBOARD.md` — generated from `results/*.json`; do not hand-edit.
