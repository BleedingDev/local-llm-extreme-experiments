# Changelog

All notable changes to trace-gepa bench. Newest first. Dates ISO-8601.

## v1 (2026-05-02): initial release with 105 trace-derived tasks across 7 categories.

- Dataset: `data/benchmark_tasks.jsonl`, n=105.
  - Categories: `tool_routing` (28), `planning` (15), `debugging` (15),
    `recovery` (15), `command_synthesis` (12), `edit_safety` (10),
    `path_grounding` (10).
  - Difficulty mix: hard 46, medium 42, easy 17.
  - Verifier kinds in v1: `structural_json` (93), `regex` (12).
- Verifier suite (`bench/verifiers/`): tier-1 regex / exact / structural-JSON
  / tool-name / tool-family; tier-2 LM judge; tier-3 shell exec; composite
  combiner. Top-level `verify(task, predicted)` dispatcher.
- Reference harnesses:
  - `run_anthropic.py` — Claude Opus 4.7 via `agent_opt.llm`. Threaded
    (`max_workers=8`). Handles Opus-4.7 temperature deprecation in
    `agent_opt/llm.py`.
  - `run_codex.py` — `codex-cli 0.128` via `codex exec --json`. Threaded
    (`max_workers=4`), per-task subprocess timeout 90 s. Translates
    `--reasoning` to `-c model_reasoning_effort=<eff>`.
  - `run_mlx.py` — `mlx-lm 0.31.2`, single-threaded, SIGALRM-bounded.
    Smoke target Llama-3.2-1B-Instruct-4bit at ~90 tok/s, 0.4 pass-rate
    (1B is too small for production).
- Leaderboard ingestor (`leaderboard.py`) and rendered `LEADERBOARD.md`.
- Comparative positioning vs SWE-bench / Terminal-Bench / Aider Polyglot /
  HumanEval in `COMPARATIVE_POSITIONING.md`.
- Known caveat: pre-FIX1 verifier read keys (`pattern`, `schema`) the
  dataset never wrote; it writes `pattern_or_command`. Effect on early
  runs: `regex` tasks scored 0 (`regex_no_pattern`); `structural_json`
  tasks scored 1 on any parseable JSON. Fixed in the dispatcher post-FIX1;
  legacy rows under `command_synthesis` should be treated as N/A.
