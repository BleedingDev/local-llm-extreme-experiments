# Contributing to trace-gepa Bench

Conventions for extending v1: tasks, verifiers, harnesses, models. Keep all
contributions deterministic and replayable from JSONL + a single `--output`.

## Repo layout

```
trace-gepa/
  data/benchmark_tasks.jsonl           # the dataset
  bench/
    run_anthropic.py  run_codex.py  run_mlx.py
    verifiers/
      __init__.py        # KIND_TO_VERIFIER dispatcher
      tier1_regex.py     # regex / exact_match / structural_json / tool_*_match
      tier2_judge.py     # LM-judge verifier
      tier3_shell.py     # shell-exec verifier
      composite.py       # `composite` combiner
    leaderboard.py       # ingests results/*.json
    results/             # one JSON per run; never hand-edit
    BENCHMARK.md  SCHEMA.md  HARNESSES.md  CONTRIBUTING.md  CHANGELOG.md
    LEADERBOARD.md  COMPARATIVE_POSITIONING.md
```

## Adding a task

1. Pick a category from the v1 set (see `SCHEMA.md`). Open a new category
   only if the existing seven cannot describe the decision being graded —
   coordinate via the changelog.
2. Choose `difficulty` from `easy | medium | hard`. Calibrate against
   existing tasks in the same category.
3. Generate an `id` of the form `task_<category>_<NNN>` where `NNN` is the
   next free 3-digit suffix in that category.
4. Populate `prompt.user_request` with the literal user message that the
   upstream agent saw (Codex / Claude Code trace prefix). Truncate at
   ~6000 chars.
5. Populate `prompt.context`:
   - `available_tools`: the exact tool list the agent had at that step.
   - `available_skills`: optional skills list.
   - `recent_actions`, `recent_tool_results`: optional, last 4–6 entries.
6. Populate `expected.primary_action` with the gold tool name and any
   regex constraint on tool input. Use `must_avoid_*` to encode hard
   negatives.
7. Choose a verifier (see below) and write the matching `verifier_spec`.
8. Write `human_readable_summary` (one short sentence). Set
   `rubric_weight: 1.0` unless you have a calibrated reason to deviate.
9. Append the JSON object to `data/benchmark_tasks.jsonl` (one line, no
   trailing comma). Run the smoke harness with `--limit N` to confirm the
   verifier resolves a defined signal (not `unknown_kind` /
   `regex_no_pattern`).
10. Bump `CHANGELOG.md`.

### Trace-derived vs synthetic
Trace-derived tasks must list the upstream IDs in `source_record_ids`
(e.g. `cc_<8hex>_evt<5digit>`). Synthetic seeds use `[]`. Mark synthetic
tasks at category-level rate ~5% to avoid drift away from real
trajectories.

## Adding a verifier

1. Add a function `verify_<kind>(task: dict, predicted: Any) -> dict` to
   the right tier file (`tier1_regex.py`, `tier2_judge.py`, `tier3_shell.py`,
   or a new `tierN_<x>.py`). Return shape:
   `{"score": float in [0,1], "tier": int, "signal": str, "details": dict}`.
2. Register it in `bench/verifiers/__init__.py` under `KIND_TO_VERIFIER`.
3. **Read keys from `verifier_spec.pattern_or_command`** — that is the
   contract the dataset writes. Earlier verifiers read `pattern` / `schema`
   instead, which silently zeroed `regex` tasks; the composite `verifier`
   bug surfaced post-FIX1 and is the headline caveat in `BENCHMARK.md`.
4. Add a unit test under `tests/` (or wherever the project keeps them)
   covering: pass case, miss case, malformed-prediction case, malformed-spec
   case. Each branch must return a defined `signal`, never raise.
5. If the new verifier needs side-effects (network, shell), gate it behind
   an env var so the default `verify(...)` stays hermetic.

## Adding a harness

1. New harness: `bench/run_<name>.py`. Mirror the existing CLI surface —
   `--tasks`, `--model`, `--limit`, `--output`, `--max-workers`,
   `--task-timeout` — and the result envelope from `SCHEMA.md`
   (`{summary, config, results}`).
2. Always score through `bench.verifiers.verify(task, predicted)`. Per-task
   exceptions become `score=0` rows with an `error` string; never abort the
   whole run on a single task failure.
3. Capture per-task `latency_ms`/`elapsed_s`, parser status, raw output
   preview (≤1200 chars), and a token estimate (real if you have it,
   chars/4 if you don't).
4. Concurrency: thread pools are fine for HTTP / subprocess harnesses.
   Local-GPU harnesses (MLX, Metal) must be single-threaded with an
   alarm-bounded per-task timeout.
5. Wire the harness into `leaderboard.py` if it produces a new output shape;
   otherwise the existing ingestor will pick it up by the `harness` field.
6. Document venv, env vars, models, expected wallclock, and gotchas in
   `HARNESSES.md`. Add an entry to `CHANGELOG.md`.

## Adding a model

Within an existing harness, just pass `--model <id>`. To add a
permanently-listed model, also:

1. Confirm the model is reachable (auth, quota, available size).
2. Run a stratified smoke (one task per category, ≥10 tasks total).
3. Land the result JSON under `bench/results/` and rebuild the leaderboard.
4. Add the model to the relevant table in `HARNESSES.md`.

## Pull request checklist

- [ ] Smoke run committed under `bench/results/`.
- [ ] `LEADERBOARD.md` regenerated from `leaderboard.py` (do not hand-edit).
- [ ] `CHANGELOG.md` updated with date + summary.
- [ ] Verifier signals are all defined; no `unknown_kind` rows in the smoke.
- [ ] No secret material (`.env`, `auth.json`, etc.) added or echoed.
