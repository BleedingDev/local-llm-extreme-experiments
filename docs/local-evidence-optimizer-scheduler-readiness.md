# Local Evidence Optimizer Scheduler Readiness

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

Machine-readable contract: `.bag/evidence/optimizer/scheduler-readiness.json`.

## Current Decision

The scheduler should treat the current optimizer state as `promotion_blocked_candidate_generation_ready`.

Candidate generation may proceed under the current local GEPA thresholds: `bench/.bag/optimizer/gepa-readiness-report.json` reports 85 dataset records, 48 failed runs, 85 metric observations, and `candidateGenerationReady: true`. `bench/.bag/optimizer/candidates.json` has 12 candidates with 12/12 validation passed, scoped to rendered prompt/tool-contract artifacts.

Promotion and auto-promotion must remain blocked. The GEPA readiness report already blocks auto-promotion on `post-promotion-monitor-window` because the observed window is `0 ms` and the required window is `14,400,000 ms`. The scheduler contract adds another fail-closed blocker: visible ACP no-write/no-terminal failures must be represented in validation before any candidate can be promoted.

## Scorecards To Scheduler Gates

| Scorecard | Scheduler use | Promotion impact |
| --- | --- | --- |
| `tool-routing-scorecard` | Tool routing risk and ACP no-write validation | Blocks promotion until the visible ACP no-write/no-terminal slice is encoded as validation or a veto. |
| `scorecard.edit-strategy` | Edit policy attribution and observability gaps | Blocks edit-method promotion until real edit attempts carry strategy, rendered contract, hash, apply/write/verify, repair, and rollback telemetry. |
| `recovery-failure` | Failure taxonomy and repair-policy targets | Drives targeted validation for nonzero commands, hallucinated paths, timeouts, retry loops, verifier contracts, and ACP failed progress. |
| `scorecard.benchmark-results` | Comparable benchmark reporting and optimizer target selection | Confirms candidate generation readiness, but keeps benchmark gains separate from live ACP regressions. |

## Blocking Gates

- `post-promotion-monitor-window`: blocked. Current `0 ms`; required `14,400,000 ms`.
- `visible-acp-no-write-validation-represented`: blocked. The visible ACP run has 9 coding tasks, 8 failed, 1 cancelled, 0 changed files, 0 fsWrite, and 0 terminal creates. This must be a validation slice or promotion veto.
- `hidden-holdout-final-gate-ready`: blocked for promotion until the scheduler has explicit hidden-holdout evaluation and leakage results for the candidate.
- `operator-approval-for-promotion`: blocked. Existing scheduler flow is dry-run/advisory; actual promotion needs explicit operator approval and applies only to new sessions.

## Nonblocking Warnings

- Current local thresholds are permissive enough to allow candidate generation with zero real replay, zero visible replay, and zero repeated failure clusters. That is acceptable only for scoped candidate generation.
- The best observed 10/10 terminal-bench-sample run is a high-water mark, not the expected aggregate rate.
- Current edit strategy evidence does not identify one globally best edit method.
- Heterogeneous scorecard counts must not be compared across evaluator families without preserving source semantics.
- Raw local evidence remains local-only until privacy review passes.

## Recommended Scheduler Actions

1. Allow scoped GEPA candidate generation and visible evaluation as artifact-only, dry-run work.
2. Attach the scorecard suite to candidate validation, including tool routing, edit strategy, recovery, benchmark family, and no-write ACP slices.
3. Do not auto-promote or mark candidates promotion-ready until the monitor window, no-write ACP validation, hidden holdout, rollback checkpoint, and operator approval gates are represented.
4. Prioritize validation slices around missing required output files, qemu SSH/startup, HTTP 000 webserver failures, nonzero command repair, path grounding, timeout handling, and ACP failed progress.
5. Add edit-attempt telemetry before promoting edit-policy candidates.
