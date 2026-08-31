# Local Evidence Benchmark Results Scorecard

Generated for graph `local-evidence-flywheel-v1` on `2026-05-04T10:47:44Z`.

## Executive Summary

The strongest comparable local result family is terminal-bench-sample BAG job evidence: `52` BAG job summaries, `424` trials, `31` errors, average job mean `0.5731`, and trial-weighted mean `0.6304`. Across all parsed rewarded task results, the local mean reward is `0.6398` over `447` reward observations.

The best observed BAG local result is `bench/jobs/2026-05-02__07-13-33/result.json`: `bag__claude-opus-4-7__terminal-bench-sample`, `10/10` trials, `0` errors, mean metric `1.0`. Treat it as a high-water mark, not the expected aggregate.

The negative visible ACP run baseline is separate: `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/real-acp-run.headless-visible-20260504.manifest.json` has `0/9` passed, `8` failed, `1` cancelled, `0` changed files, `0` `fsWrite`, and `0` terminal creates. This is high-value no-write/end_turn failure evidence, not a terminal-bench reward result.

## Scorecards

| Family | Evidence | Sample | Result | Caveat |
| --- | --- | ---: | ---: | --- |
| Terminal-bench-sample BAG | `bench/jobs/**` | 52 jobs / 424 trials | avg job mean `0.5731`, weighted `0.6304` | Comparable only within same evaluator/task set. |
| Claude Code comparator | `bench/jobs/**` | 1 job / 10 trials | mean `0.9` | Comparator run, not BAG aggregate. |
| CAIS SWE-Bench Pro | `bench/jobs/**` | 8 trials / 2 errors | mean `0.4` | Different benchmark semantics. |
| Aider polyglot | `bench/aider_polyglot/results/**` | 5 problems | `4/5`, pass rate `0.8` | One task per language; Java failed. |
| Ablation grid | `bench/ablation/results/**` | 9 cells | `0` populated cells | No mode/model conclusion possible. |
| Real ACP replay | `.bag/replay-corpus/**` | 13 smoke+visible tasks | `0` passed, `11` failed, `1` error, `1` cancelled | Negative ACP behavior baseline. |
| LiveCodeBench smoke | `bench/jobs/livecodebench_smoke_5/result.json` | 5 questions | `5/5`, 216/216 tests | Small pinned smoke, not broad LCB. |
| SWE-Bench MM smoke | `bench/jobs/swe_bench_mm_smoke_3/result.json` | 3 instances | 3 nonempty patches, 1/3 used `view_image` | Local npm verification disabled. |
| METR TH smoke | `bench/jobs/metr_th_smoke_3/summary.json` | 3 tasks | `0/3` | Docker image pull failure, infra not model. |

## Repeated Task Signals

Strong repeated terminal-bench-sample tasks: `regex-log` (`41/45`, mean `0.9111`) and `log-summary-date-ranges` (`39/43`, mean `0.9070`).

Weak repeated tasks: `qemu-alpine-ssh` (`13/41`, mean `0.3171`), `qemu-startup` (`19/42`, mean `0.4524`), and `chess-best-move` (`26/46`, mean `0.5652`).

Top optimizer failure clusters are missing `/app/summary.csv` (`25`), missing `/app/report.jsonl` (`22`), SSH exit 255 (`19`), HTTP 000 webserver failure (`14`), and generic `assert false` (`10`).

## Optimizer Notes

`bench/.bag/optimizer` has `85` dataset records with mean reward `0.4353`, `12/12` candidates validated, and `26` failure clusters. Candidate generation is ready, but auto-promotion remains blocked by `post-promotion-monitor-window`.

Recommended order: first fix visible ACP no-write behavior, then use terminal-bench-sample as the comparable reward loop, then target the repeated missing-output and qemu/webserver failure clusters. Do not claim ablation lift until the ablation cells contain trials.

Machine-readable scorecard: `.bag/evidence/scorecards/benchmark-results.json`.
