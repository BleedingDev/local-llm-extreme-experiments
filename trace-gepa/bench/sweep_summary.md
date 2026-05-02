# Phase-3 Fix Agent #FIX5 Sweep Summary

_Run: 2026-05-02 (post-FIX1 verifiers)_
_Bench: `data/benchmarks/minibench.jsonl` (30 stratified tasks)_
_Sweep driver: `trace-gepa/bench/run_sweep.sh`_

## Pre-condition: FIX1 verifier landed

`trace-gepa/bench/results/smoke_anthropic_postfix.json` is present
(written 02:35 UTC) and reports `mean_score = 0.100` over 20 tasks
(non-trivial variance, no longer the prior 1.000 floor). The fixed
verifier path is therefore exercised by every run in this sweep.

## Overall ranked table (8 rows)

| rank | harness   | model                                          | prompt   | mean_score | n  | wallclock_s | lm_calls | in_tok  | out_tok |
|------|-----------|------------------------------------------------|----------|------------|----|-------------|----------|---------|---------|
| 1    | anthropic | claude-opus-4-7                                | seed     | 0.233      | 30 |   8.9       | 30       | 16225   | 677     |
| 2    | anthropic | claude-haiku-4-5                               | seed     | 0.200      | 30 |   4.5       | 30       | 16225   | 1077    |
| 3    | anthropic | claude-haiku-4-5                               | opt      | 0.200      | 30 |   6.3       | 30       | 28120   | 1215    |
| 4    | anthropic | claude-opus-4-7                                | opt      | 0.200      | 30 |  10.4       | 30       | 28120   | 862     |
| 5    | codex     | gpt-5.5                                        | high     | 0.200      | 30 | 116.7       | 30       | n/a     | n/a     |
| 6    | codex     | gpt-5.4                                        | high     | 0.200      | 30 | 183.8       | 30       | n/a     | n/a     |
| 7    | codex     | gpt-5.5                                        | xhigh    | 0.167      | 30 | 134.0       | 30       | n/a     | n/a     |
| 8    | mlx       | mlx-community/Llama-3.2-1B-Instruct-4bit       | default  | n/a        | -  | n/a         | -        | n/a     | n/a     |

(`mlx` row is incomplete: `mlx_lm` is not importable in `.venv-gepa`
on this host. Harness exited cleanly with `status=mlx_not_installed`
and zero LM calls. Apple Silicon + `pip install mlx-lm` would unblock
it. Nothing else in this sweep is affected.)

## Per-category breakdown (top 3 candidates)

Only categories with non-zero pass-rate variance shown.

| category          | rank-1 candidate                  | rank-2 candidate                  | rank-3 candidate                  |
|-------------------|-----------------------------------|-----------------------------------|-----------------------------------|
| tool_routing      | all 7 LM rows tied at 0.40 (n=5)  | -                                 | -                                 |
| edit_safety       | all 7 LM rows tied at 0.40 (n=5)  | -                                 | -                                 |
| recovery          | anthropic/opus/seed 0.75 (n=4)    | 5x other LM rows tied 0.50        | codex/gpt-5.5/xhigh 0.00          |
| planning          | codex/gpt-5.5/xhigh 0.25 (n=4)    | all other LM rows 0.00            | -                                 |
| command_synthesis | all 7 LM rows tied at 0.00 (n=4)  | -                                 | -                                 |
| debugging         | all 7 LM rows tied at 0.00 (n=4)  | -                                 | -                                 |
| path_grounding    | all 7 LM rows tied at 0.00 (n=4)  | -                                 | -                                 |

Three categories (`command_synthesis`, `debugging`, `path_grounding`)
score 0.00 across every candidate. Either the verifiers for those
categories are still mis-specified (a residual FIX1 issue) or the
prompt template is structurally wrong for these task families. Worth
flagging to FIX1 for a follow-up triage.

## Cost-effectiveness (pass-rate per second of wallclock)

| candidate                                      | mean_score | wallclock_s | score / sec |
|------------------------------------------------|-----------:|------------:|------------:|
| anthropic / claude-haiku-4-5 / seed            |      0.200 |         4.5 |   0.0444    |
| anthropic / claude-haiku-4-5 / opt             |      0.200 |         6.3 |   0.0317    |
| anthropic / claude-opus-4-7 / seed             |      0.233 |         8.9 |   0.0262    |
| anthropic / claude-opus-4-7 / opt              |      0.200 |        10.4 |   0.0192    |
| codex / gpt-5.5 / high                         |      0.200 |       116.7 |   0.0017    |
| codex / gpt-5.5 / xhigh                        |      0.167 |       134.0 |   0.0012    |
| codex / gpt-5.4 / high                         |      0.200 |       183.8 |   0.0011    |

The Anthropic harness is roughly 25-100x more wallclock-efficient than
Codex on this slice; that is dominated by Codex's reasoning tokens and
cold-start, not by per-task quality.

## Aggregate cost / time

- Total LM calls across the sweep: **210** (7 LM-running candidates x 30 tasks; MLX yielded zero).
- Total per-run wallclock summed: **466 s** (~7.8 min, well inside the 30-min budget).
- Rough cost estimate (back-of-envelope, USD):
  - Anthropic 4 runs * 30 tasks ~ 120 calls; ~28k in / ~3.8k out total. At Opus posted prices (~$15/$75 per 1M) and Haiku (~$1/$5 per 1M), ~$0.55 total Anthropic spend.
  - Codex 3 runs * 30 tasks via local `codex exec --json`; no direct $$ tracked, but 7-8 minutes of high-reasoning gpt-5.x usage is the dominant cost. Estimate ~$1-2.
  - MLX run: $0 (skipped).
  - **Estimated sweep total: ~$1.5 - $2.5.**

## Verdict

**Winner: `anthropic / claude-opus-4-7 / seed`** at `mean_score = 0.233`
on the 30-task minibench. Opus + the canonical seed prompt edged out
Opus + the optimised system prompt by exactly one task (one-task delta,
not statistically meaningful at n=30). Haiku matches Opus-opt at 0.200
in roughly half the wallclock, making it the cost-effectiveness winner.
Codex `gpt-5.5/high` is fully competitive on overall pass-rate (0.200)
and uniquely picks up `planning` credit at xhigh, but at 13-40x the
wallclock; raising reasoning from `high` to `xhigh` actually *hurt*
overall score on this slice (0.200 -> 0.167), suggesting xhigh
overthinks short single-step tool-selection tasks. The optimised system
prompt did not improve over seed for either Anthropic model on this
benchmark, indicating the optimiser's GEPA candidate has not generalised
to minibench's stratified mix - a clear next-iteration target. Overall
the headline number (~0.20) is low and the three "n/a 0.00" categories
suggest a residual verifier or schema bug worth handing back to FIX1
before declaring an absolute leaderboard.
