# `BleedingDev/local-coding-benchmark` — `benchmark-quality-refactor` Evaluation

**Evaluator:** Bench-Track Agent #1
**Date:** 2026-05-01
**Local clone:** `/Users/satan/side/experiments/local-coding-benchmark`
**Working tree:** clean checkout of `benchmark-quality-refactor`

---

## 1. Branch Metadata

| Field | Value |
|---|---|
| HEAD commit | `a64bac32d580499b99cd662c5f8af2c71d79e489` |
| HEAD subject | `Build robust model-harness benchmark tiers` |
| Default branch | `main` (HEAD `e65b388` *Initial local coding benchmark extraction*) |
| Merge-base | `e65b388` (== `main` HEAD) |
| Divergence | **0 behind / 1 ahead** of `main` (single squash-style commit) |
| Files changed | **33** |
| Diff totals | **+3260 / −55** |
| Open PR for branch | **none** (`gh pr list --state all` returns `[]`) |
| `CHANGELOG.md` | not present |

The single commit is monolithic (≈3.2k+ LOC). No commit-by-commit narrative exists; the design rationale lives in `docs/benchmark-quality-plan.md` and `reports/benchmark-methodology.md`.

## 2. Refactor Goal (one paragraph)

The branch repositions the original 50-task single-file codegen suite from a presumed Terminal-Bench peer to an explicit **Tier-1 baseline**, and stands up a **Tier-2 terminal-agent** track running directory-based tasks against CLI harnesses with workspace mutation, hidden tests, trajectory artefacts, and a separate leaderboard. It introduces a four-layer data-driven config split (`adapters` / `models` / `agents` / `runs`), audit-grade run manifests (suite hash, host/git metadata, endpoint probe, sampling settings), a minimal-env CLI policy, calibration tasks (known-good + known-bad), a regression test suite, and stricter pass@k accounting (no fake pass@K when only 1 attempt was recorded; smoke runs flagged as subsets and never mixed with full-suite rankings).

## 3. New Schema / New Conventions

**New top-level layout (added on the branch):**

- `configs/{adapters,agents,models}.json` — versioned (`schema_version: 1`) config split, separate from runs.
- `configs/runs/*.json` — declarative run matrices (suite filter, attempts, sampling, agent_env_mode, output path).
- `tasks/terminal-agent/<task-id>/{task.json,prompt.md,scaffold/,tests/}` — directory-based tier-2 cases. Two calibration tasks ship: `calibration-config-parser`, `calibration-python-bugfix`.
- `docs/benchmark-quality-plan.md`, `docs/terminal-agent-tasks.md` — design + authoring rules.
- `tests/test_benchmark_regressions.py` — unittest suite (4 tests, all green via `uv run --with pytest`).
- `reports/artifacts/<run>/<task>/<attempt>/` — per-attempt trajectory dir: `agent.command.txt`, `agent.{stdout,stderr}.log`, `validation.{stdout,stderr}.log`, `workspace.diff`, `final-files.json`.

**JSONL row schema (current shape on this branch, e.g. `reports/direct-iq2xxs-128k-pass5.jsonl`):**

```jsonc
{
  "run":   { "run_id", "suite", "suite_version", "provider", "harness", "runtime",
             "model", "model_family", "quant", "context_tokens", "runtime_params",
             "notes", "attempts" },
  "task":  { "name", "language", "difficulty", "project_kind", "prompt", "tests",
             "scaffold", "max_tokens" },
  "record": { "name", "language", "difficulty", "project_kind", "ok", "stage",
              "detail", "elapsed_s", "prompt_tokens", "completion_tokens",
              "prompt_tps", "decode_tps", "finish_reason", "content_chars",
              "code_chars", "attempt_index", "check_passed", "check_total",
              "check_score" },
  "content": "...", "code": "..."
}
```

Sidecar `*.manifest.json` per run holds `suite_hash`, selected tasks, attempts, sampling (temperature/top_p/seed), host/python/git info, tool versions, `/v1/models` probe result, agent env mode. **Old rows on `main` lack `run` and the granular `check_*` / `attempt_index` fields and are explicitly archive-only.**

**Failure taxonomy (new):** `wrong_answer | runtime_error | syntax_error | timeout | policy_violation | format_leak | missing_symbol | harness_error | other_failure`.

**Composite score (new, `pass@1`-dominant):** `0.45*pass@1 + 0.15*check_score + 0.12*hard_pass + 0.08*scaffold_pass + 0.08*stable_output + 0.07*consistency + 0.05*speed`, with absent components renormalised.

## 4. Scripts Inventory

| File | Lines | One-line purpose |
|---|---:|---|
| `scripts/comprehensive_coding_eval.py` | 1449 | Tier-1 strict single-file codegen runner (50 tasks, granular checks, manifest emit). |
| `scripts/local_coding_eval.py` | 335 | Pre-existing simpler eval (kept for compat). |
| `scripts/summarize_coding_eval.py` | 133 | Summarise comprehensive JSONL files. |
| `scripts/benchmark_leaderboard.py` | 435 | Tier-1 leaderboard (Terminal-Bench-style MD); honours `attempts`, marks subsets. |
| `scripts/run_matrix.py` | 259 | Execute a `configs/runs/*.json` matrix against agent profiles (Tier 1). |
| `scripts/harness_inventory.py` | 126 | Render configured agents/models inventory as Markdown. |
| `scripts/terminal_agent_eval.py` | 703 | Tier-2 directory-task runner (workspace, prompt, validation, artefacts). |
| `scripts/run_terminal_matrix.py` | 234 | Tier-2 matrix driver (uses `terminal_command_template`). |
| `scripts/terminal_agent_leaderboard.py` | 255 | Tier-2 leaderboard (pass@k, stable, changed_files, elapsed). |
| `scripts/terminal_agent_oracle.sh` | 46 | Oracle harness selftest (deterministic correct-answer agent). |
| `scripts/validate_benchmark.py` | 158 | Validate configs + terminal task metadata. |
| `scripts/check_benchmark.sh` | 14 | CI bundle: `py_compile` + validate + `python -m unittest discover`. |

## 5. Build / Smoke Status

| Check | Result |
|---|---|
| `python3 scripts/comprehensive_coding_eval.py --list` | **PASS** — prints all 50 tasks, exit 0. |
| `python3 scripts/validate_benchmark.py` | **PASS** — `OK benchmark config and terminal tasks validated`. |
| `uv run --with pytest python -m pytest tests/ -q` | **PASS** — 4/4 in 0.75s. |
| Network/server-dependent runs | not exercised (no llama.cpp endpoint locally). |

## 6. Compatibility Verdict

**Clean fast-forward.** `git merge-base --is-ancestor main benchmark-quality-refactor` is true; `main` is `e65b388` and `HEAD` is one commit ahead. Zero conflicts. Merging is a single `git merge --ff-only`.

## 7. Concrete Merge Plan (Step A)

1. `git fetch origin` and confirm `main = e65b388`, `benchmark-quality-refactor = a64bac3`.
2. `git checkout main && git merge --ff-only origin/benchmark-quality-refactor` (FF-clean, no conflicts).
3. Sanity gate: `bash scripts/check_benchmark.sh` (compile + validate + unittests).
4. Tag the pre-merge state: `git tag pre-quality-refactor e65b388` for archive comparability of historical reports.
5. Push: `git push origin main` (and `git push origin pre-quality-refactor`).
6. Optionally delete the now-merged ref: `git push origin --delete benchmark-quality-refactor`.
7. Add a one-line `CHANGELOG.md` entry stamping the cutover (suite_hash, suite_version `2026-04-29.1`).

No rebase, no squash, no conflict resolution required.

## 8. Fit For Our `trace-gepa/` Artefacts

Our artefacts target a fundamentally different unit — **next-action selection on a single trace step** (175 tasks across `tool_routing | command_synthesis | edit_safety | path_grounding | debugging | recovery | planning`), scored by tier 1–4 verifiers (`regex | structural_json | tool_*_match | exact_match | lm_judge | shell_exec | composite`). The refactor's Tier-1 expects a **whole source file** evaluated by hidden Python/TS unit tests; Tier-2 expects a **workspace mutation** validated by a shell command. Neither of those matches our records 1:1. The right home for us is a **new Tier 0/3 — "trace-step action benchmark"** that lives alongside Tier-1 and Tier-2, sharing the manifest convention and `configs/agents.json` layer but NOT the codegen scoring.

| Our artefact | Refactor home | Translation work |
|---|---|---|
| `data/benchmark_tasks_full.jsonl` (175 tasks) | new `tasks/action-selection/v1/tasks.jsonl` (versioned, `suite: action-selection`, `suite_version: 2026-05-01.1`) | Add `suite_hash` of the file; add `suite`/`suite_version` to every emitted row. No record-shape change — our schema is already richer. |
| 4-tier verifier suite (`bench/verifiers/{tier1_regex,tier2_judge,tier3_shell,composite}.py`) | drop in as `scripts/action_agent_eval.py` + a sibling package `verifiers/` (or import directly). | Wire `verifier_kind` dispatch; emit failure taxonomy mapped to refactor's set: `wrong_answer` (verifier fail), `format_leak`/`syntax_error` (JSON parse fail), `runtime_error` (judge/shell crash), `harness_error` (CLI unavailable), `timeout`. |
| 3 harnesses (`run_anthropic.py`, `run_codex.py`, `run_mlx.py`) | register as agent profiles in `configs/agents.json` under a new adapter `action-step-json` (provider: `direct` for Anthropic SDK, `cli-command` for Codex CLI, `python-provider` for MLX). | Add `adapters.json` entry `action-step-json` with contract "produce one-line JSON `{tool_name, brief_reason}`". Move CLI invocation to `command_template`. Settings (`temperature`, `reasoning`, etc.) go to `agents.json`. |
| `bench/leaderboard.py` | rename to `scripts/action_agent_leaderboard.py`. | Replace its result-file scan with the JSONL+manifest pattern. Group by category/difficulty (already present in our summary). Mark smoke (`limit < 175`) as subset, same as Tier-1 does. |

Result-file format: our existing `{model, tasks_path, summary, per_task[]}` JSON should be **augmented**, not replaced — emit a parallel JSONL (one row per task per attempt) with `run`/`task`/`record` keys to match the refactor manifest contract, while keeping the rolled-up JSON as the leaderboard input. Our `per_task[].score` (0..1), `verifier_signal`, `parsed_output`, `latency_ms` map cleanly to `record.check_score`, `record.detail`, `record.code` (for the action JSON), `record.elapsed_s`.

## 9. On-Top Integration Plan (Step B + C)

**Step B — placement after FF merge:**

```
local-coding-benchmark/
├── configs/
│   ├── adapters.json              + action-step-json
│   ├── agents.json                + claude-opus-action / codex-gpt55-action / mlx-qwen3-action
│   └── runs/
│       └── action-selection-full.json   (new: 175 tasks, attempts=1, model matrix)
├── docs/
│   └── action-selection-tasks.md  (port trace-gepa/bench/SCHEMA.md; describe Tier-0)
├── scripts/
│   ├── action_agent_eval.py       (port of run_anthropic/codex/mlx, dispatch by adapter)
│   ├── action_agent_leaderboard.py(port of bench/leaderboard.py)
│   └── verifiers/                 (drop-in copy of trace-gepa/bench/verifiers/)
├── tasks/
│   └── action-selection/
│       └── v1/
│           ├── suite.json         (suite_hash, suite_version, task_count=175)
│           └── tasks.jsonl        (port of data/benchmark_tasks_full.jsonl)
├── reports/
│   └── action-*.jsonl + manifests (mirroring direct-*.jsonl convention)
└── tests/
    └── test_action_selection_regressions.py  (smoke against fixture rows)
```

**Step C — schema translation (per emitted JSONL row):**

```jsonc
{
  "run": {                                  // NEW wrapper, mirrors comprehensive-*
    "run_id": "<harness>-<model>-<suiteversion>",
    "suite": "action-selection",
    "suite_version": "2026-05-01.1",
    "suite_hash": "<sha256 of tasks.jsonl>",
    "provider": "anthropic" | "codex-cli" | "mlx",
    "harness":  "claude-opus-action" | "codex-gpt55-action" | "mlx-qwen3-action",
    "runtime":  "anthropic-sdk" | "codex-cli x.y" | "mlx 0.x",
    "model":    "claude-opus-4-7" | "gpt-5.5" | "qwen3-...-mlx",
    "attempts": 1
  },
  "task": {                                 // verbatim from our tasks.jsonl
    "id", "category", "difficulty", "prompt", "expected",
    "verifier_kind", "verifier_spec", "rubric_weight"
  },
  "record": {                               // mapped from our per_task[]
    "name":        task.id,                 // refactor uses .name; alias to id
    "category":    task.category,           // ADDED (not in refactor; carry through)
    "difficulty":  task.difficulty,
    "ok":          (score >= 1.0),
    "stage":       "verify",
    "detail":      verifier_signal,         // e.g. json_parsed / regex_fail
    "elapsed_s":   latency_ms / 1000.0,
    "prompt_tokens":     prompt_tokens_est,
    "completion_tokens": output_tokens_est,
    "attempt_index":     1,
    "check_passed":      int(score >= 1.0),
    "check_total":       1,
    "check_score":       score,             // float 0..1
    "failure_class":     <map verifier_signal to refactor taxonomy>
  },
  "content": raw_output,
  "code":    json.dumps(parsed_output)      // canonical action JSON
}
```

The composite-score formula needs a per-suite override: for action-selection there is no `hard_pass` / `scaffold_pass` / `stable_output` axis — those weights renormalise to zero, leaving `0.45*pass@1 + 0.15*check_score + 0.07*consistency + 0.05*speed`. The refactor's leaderboard already supports component omission, so this works without code changes.

## 10. Headline Numbers

- **+3260 / −55** across **33 files**, **1 commit**, **0 behind**.
- **Clean FF**, no conflicts.
- **All smoke checks green** (list, validate, pytest 4/4).
- **No open PR** — branch is dangling on the remote awaiting merge.

**Recommendation:** merge `benchmark-quality-refactor` to `main` via fast-forward immediately, then layer our action-selection tier on top as a fourth, parallel suite. The refactor's manifest + adapter/agent/run config split is exactly the substrate our 3 harnesses need; we get audit metadata, calibration discipline, and subset-marking for free, and we contribute a categorically different (trace-step) tier the refactor does not yet cover.
