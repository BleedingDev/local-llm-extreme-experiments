# BAG <-> LiveCodeBench v6 onboarding

This document covers how to run BleedingAgent (BAG) against a
date-pinned slice of LiveCodeBench v6, why we pin, and how it relates
to the rest of the BAG benchmark portfolio.

## TL;DR

- Adapter lives at `bench/livecodebench/`.
- Generic single-problem-stdin/stdout harness — works for any benchmark
  that ships per-problem (input, expected) test cases.
- Default pin: `release_v6_2025-01-04_to_2025-04-30` (the v5->v6 delta;
  every problem has `contest_date >= 2025-01-04`).
- No Docker. No GPU. No `pip install livecodebench` (see "On the
  upstream pip package" below).
- Smoke run command:

  ```bash
  python -m bench.livecodebench.run \
      --job-name livecodebench_smoke_5 \
      --n-tasks 5 \
      --concurrency 1
  ```

- Output: `bench/jobs/livecodebench_smoke_5/{config,result,summary}.{json,txt}`
  plus per-problem `task_<question_id>/{instruction.md, solution.py,
  bag.log, bag-acp-run.json, grade.log, result.json}`.

## Why LiveCodeBench (and why pinned)

LiveCodeBench (LCB) is a *contamination-free* code-generation
benchmark.  New problems are scraped from LeetCode, AtCoder, and
Codeforces *contests* on a rolling cadence; every record has a
`contest_date`. The dataset version (`release_v1` ... `release_v6`)
controls how many monthly increments are included.

Because LCB is rolling:

- **A `release_vN` label alone is not reproducible.** It might describe
  N=1055 problems today and N=1140 next month, depending on whether
  HuggingFace has pushed `test{n+1}.jsonl`.
- **Different model checkpoints have different training cutoffs.**
  Reporting an aggregate score over the whole benchmark mixes
  contaminated-and-uncontaminated problems, which is exactly the trap
  LCB exists to avoid.

The fix that the upstream paper recommends, and that we adopt here, is
to pin a closed `[start_date, end_date]` window inside a known
release. We pin to the **v5->v6 delta** by default:

| Property         | Value                              |
|------------------|------------------------------------|
| Source file(s)   | `livecodebench/code_generation_lite/test6.jsonl` |
| `start_date`     | `2025-01-04` (first contest in the file) |
| `end_date`       | `2025-04-30` (closes the window)   |
| Label            | `release_v6_2025-01-04_to_2025-04-30` |
| Approx size      | ~175 problems (full file)          |

This window is contamination-resistant for any model with a training
cutoff at or before late 2024 — which covers every Anthropic and
local-MLX model in the current BAG matrix.

To freeze even harder, set `HF_REPO_REVISION` in
`bench/livecodebench/dataset.py` to a commit SHA on
`huggingface.co/datasets/livecodebench/code_generation_lite`.

## Pin-rationale: avoiding rolling-window denominator drift

If the harness loads "release_v6" without filtering, three things go
wrong as time passes:

1. **The denominator changes.** `release_v6` today may be 1055
   problems. After HF pushes `test7.jsonl`, the implicit
   `release_latest` swells to ~1200, breaking comparability with
   yesterday's run.
2. **Contamination-vs-not gets mixed.** Problems from May 2023 sit
   alongside problems from March 2025 — fine for absolute capability
   evaluation, useless as a *control* for "did this BAG change improve
   over the raw model?" because we cannot tell whether a delta is
   memorisation drift or real reasoning lift.
3. **Score noise from ERRATA churn.** Upstream patches a small number
   of incorrect tests in each release; pinning a date window inside a
   single release sidesteps that noise.

Pinning by file-name **and** date range gives a stable, citable
benchmark cell.

## On the upstream pip package

The mission spec mentioned `pip install livecodebench`. As of
**2026-05-02**, that package does not exist on PyPI:

```bash
$ curl -s https://pypi.org/pypi/livecodebench/json
{"message": "Not Found"}
```

The official LCB harness is `lcb_runner`, distributed only via the
GitHub repo (`pip install -e .` from a checkout). It targets `vllm`
batch generation across the full release on a multi-GPU host — not
suitable for our per-problem ACP-style agent driver.

So this adapter **does not depend on the upstream Python package**.
Instead it:

1. Downloads the canonical jsonl files via `huggingface_hub.hf_hub_download`,
   filters by `contest_date`, and parses records itself
   (`bench/livecodebench/dataset.py`).
2. Vendors a stdin/stdout + functional-Solution-class grader
   (`bench/livecodebench/grader.py`), adapted from harbor's MIT-licensed
   LCB adapter (`bench/vendor/harbor/adapters/livecodebench/`), which
   itself derives from the upstream `apps`/`lcb_runner` checker. Same
   semantics: numeric-tolerance line comparison for stdin tests, direct
   Python-object equality for functional tests, base64+zlib+pickle
   private-test decoding.

If the upstream lands a real PyPI package later, the grader can be
swapped out — the rest of the harness only depends on the dataset
loader returning `LCBProblem` records.

## How a run unfolds

For each pinned problem the harness:

1. Creates `bench/jobs/<job-name>/task_<safe-question-id>/`.
2. Writes `instruction.md` (BAG-facing problem statement, derived from
   `question_title` + `question_content` + `starter_code`).
3. Writes `tests/config.json` (the full LCB record incl. private tests
   and metadata) plus `config.json` at workdir root.
4. Copies `check_solution.py` (an in-workdir public-tests-only helper)
   so the agent can iterate via `python check_solution.py`.
5. Spawns BAG via the same entry point used by other benches:
   ```
   node_modules/.bin/tsx --env-file=.env scripts/bag_acp_run.ts \
     "<instruction>" --workdir <task_dir> --out <task_dir>/bag-acp-run.json \
     --mode auto --timeout-ms <agent budget>
   ```
   ACP fs/terminal calls happen relative to the task workdir; the agent
   produces `solution.py` there (or fails to).
6. Runs the LCB-style grader on `solution.py` against the full set of
   public + private tests. Records pass/fail per test in `grade.log`
   and an aggregate `result.json`.

Aggregates land in:
- `bench/jobs/<job-name>/result.json` (full per-trial breakdown)
- `bench/jobs/<job-name>/summary.txt` (one-line pass-rate)
- `bench/jobs/<job-name>/job.log` (live stream of progress)

The headline metric is **pass-rate of problems where every test passes**
(strict pass@1) — same as the LCB leaderboard's `pass@1` column.

## Expected per-problem runtime

This is purely a function of BAG's own time budget, since the grader is
sub-second per problem (typically 1–3 seconds total for ~10–30 hidden
tests at <1s each).

| Configuration                | BAG cost / problem  | Notes |
|------------------------------|---------------------|-------|
| Anthropic Opus master, auto  | ~60–300s (median)   | matches swebenchpro_smoke_10 trial pacing |
| Anthropic Opus master, hard  | up to 900s (cap)    | the `--bag-timeout-sec` ceiling |
| Local MLX master (Qwen 3.5)  | ~5–20 minutes       | depends on KV/decode tps observed in `docs/qwen-tuning-benchmarks.md` |
| Local MLX speculative        | ~3–10 minutes       | rotorquant variants typically |

A 5-problem smoke against Anthropic-master BAG in `auto` mode with
`--bag-timeout-sec 900` finishes within roughly **5–25 minutes**
wallclock at concurrency=1, and roughly **5–10 minutes** at
concurrency=2 (subject to API rate limits).

For matrix runs over the full 175-problem v5->v6 delta, expect:

- ~3–6 hours on Anthropic Opus master, concurrency=4
- ~1–2 days on local MLX master, concurrency=1 (rotor/speculative)

## Comparison to other BAG benchmarks

| Benchmark           | Domain                       | Contam.-resistant? | Grader        | Why we run it |
|---------------------|------------------------------|--------------------|---------------|---------------|
| **LiveCodeBench v6 (this)** | Competitive-programming algo problems | yes (date-pinned) | stdin/stdout + functional | **Algorithmic-floor control.** Tests model's raw reasoning; a BAG that doesn't beat the bare model here is mostly orchestrating waste. |
| `cais/swebenchpro`  | Real-world repo bug fixes    | no (training data may include patches) | repo-level pytest | Ecosystem fit / multi-file editing. Captures the BAG-specific lift over single-shot generation. |
| `terminal-bench`    | Shell-task agents            | partly             | reward script | Trajectory quality; `bag_acp_run.ts` ACP loop. |
| `aider_polyglot`    | Multi-language one-shot edits | no                 | per-task    | Cross-language editing rigor. |

**Use LiveCodeBench as the control.**  When a BAG-side change improves
swebench but regresses LCB, that is a strong signal the change is
overfitting to repo-bug heuristics rather than improving the
underlying model's algorithmic reasoning. When a change moves both
together, that's a real win.

## Quick reproduction recipe

```bash
# One-time setup: ensure tsx + Anthropic token are in place
ls node_modules/.bin/tsx               # if missing: npm install
grep ANTHROPIC_AUTH_TOKEN .env         # if missing: export from ~/.zshrc

# 5-problem smoke
python -m bench.livecodebench.run \
    --job-name livecodebench_smoke_5 \
    --n-tasks 5 \
    --concurrency 1 \
    --bag-mode auto \
    --bag-timeout-sec 900

# Inspect
cat bench/jobs/livecodebench_smoke_5/summary.txt
cat bench/jobs/livecodebench_smoke_5/result.json | jq '.n_pass, .n_total, .trials | length'
```

To target a specific problem (e.g. for replay during development):

```bash
python -m bench.livecodebench.run \
    --job-name lcb_repro_abc387_b \
    --task-id abc387_b
```

To override the pin (e.g. evaluate against the full v6 release for a
calibration matrix on a clean leaderboard):

```bash
python -m bench.livecodebench.run \
    --job-name lcb_v6_full \
    --pin-label release_v6_full \
    --pin-files test.jsonl --pin-files test2.jsonl --pin-files test3.jsonl \
    --pin-files test4.jsonl --pin-files test5.jsonl --pin-files test6.jsonl \
    --pin-start 2023-05-01 --pin-end 2025-04-30
```

## License compatibility

LCB is MIT (see [LiveCodeBench/LiveCodeBench/LICENSE](
https://github.com/LiveCodeBench/LiveCodeBench)). The harbor adapter
that the grader is derived from is also MIT
(see `bench/vendor/harbor/LICENSE`). This repository's vendored
derivative (`bench/livecodebench/grader.py` and
`check_solution_template.py`) carries an MIT attribution comment and
is compatible with the project's existing licensing.
