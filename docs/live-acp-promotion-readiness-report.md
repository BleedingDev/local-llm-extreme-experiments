# Live ACP Promotion Readiness Report

Date: 2026-05-05
Graph: `live-acp-evidence-readiness-v1`
Selection hash: `6fbc4883fa`
Lane: `Live ACP Evidence 03 Promotion Readiness Closure`

## Summary

Promotion is still not ready, but the evidence state is cleaner and more precise.

Current graph evidence now validates for `live-acp-evidence-readiness-v1`: scorecard suite metadata, optimizer gate suite metadata, and release proof all target the current graph. The old `scorecardsGraphMatchesCurrent` and `optimizerGraphMatchesCurrent` release-proof failures are closed.

Auto-promotion remains blocked. This is correct. The visible ACP no-write/no-terminal requirement is now represented by a concrete gate artifact instead of a vague missing-evidence blocker, and that gate now passes against the current visible run selection. Promotion remains blocked by holdout, approval/rollback, and monitor-window requirements.

Important quality caveat: the current headless ACP corpus still proves fail-closed behavior, not coding competence. The mutating tasks generate no file edits and then fail through terminal verification. That is acceptable for no-write/no-terminal gating, but it is not acceptable as a local coding model quality result.

## Current Artifacts

Current graph artifacts regenerated through BAG commands:

| Artifact | State |
| --- | --- |
| `.bag/evidence/scorecards/index.json` | retargeted to `scorecard-suite.live-acp-evidence-readiness-v1` |
| `.bag/evidence/edit-attempt-records.jsonl` | generated from optimizer-visible real ACP corpus manifests; 23 first-class edit attempt records |
| `.bag/evidence/scorecards/edit-attempt-projection.json` | generated from first-class edit attempt records; 23 projected records |
| `.bag/evidence/optimizer/index.json` | retargeted to `optimizer-gate-suite.live-acp-evidence-readiness-v1` |
| `.bag/evidence/optimizer/no-write-gate.json` | generated from `.bag/replay-corpus/**` visible ACP runs |
| `.bag/evidence/release-proof.json` | regenerated as `release-proof.live-acp-evidence-readiness-v1` |
| `docs/live-acp-current-release-proof-report.md` | regenerated from the current proof |

The underlying historical scorecard and optimizer contract documents are still inherited from `local-evidence-flywheel-v1`. They remain useful as local evidence lineage, but they are not fresh real-consumer proof.

## Validation State

`npm run bag -- evidence validate --graph-id live-acp-evidence-readiness-v1` now exits successfully for evidence validity:

| Field | Value |
| --- | --- |
| index records | `32` |
| scorecards | `4` |
| edit attempt records | `23` |
| optimizer contracts | `4` |
| release proof validation | `true` |
| promotionReady | `false` |

Release-proof validation statuses:

| Check | Status |
| --- | --- |
| `planGraphSnapshot` | passed |
| `evidenceIndexCommand` | passed |
| `scorecardsCommand` | passed |
| `optimizerGatesCommand` | passed |
| `scorecardsGraphMatchesCurrent` | passed |
| `optimizerGraphMatchesCurrent` | passed |
| `historicalProofPreserved` | passed |
| `historicalProofNotReportedAsCurrent` | passed |

## Visible ACP No-Write Gate

The new `.bag/evidence/optimizer/no-write-gate.json` is generated from replay corpus manifests through the existing no-write slice and no-write promotion gate helpers.

Gate result:

| Metric | Value |
| --- | ---: |
| included visible cases | `9` |
| passed | `9` |
| blocked | `0` |
| warned | `0` |
| status | `pass` |

The gate now checks only the current visible corpus run `real-acp-run.headless-current-visible-20260505`. The stale `real-acp-run.headless-visible-20260504` cases no longer win the current promotion gate selection.

This closes the missing-representation blocker without pretending the agent is ready. The separate coding-quality evidence still says the headless runner produced `0` edits on `8` mutating tasks and failed closed through verifier commands.

## Remaining Blockers

These blockers remain valid:

| Blocker | Current state | Required next work |
| --- | --- | --- |
| hidden holdout final gate | split contracts describe holdout exclusion, but no frozen-candidate hidden holdout result exists | freeze a candidate, run holdout evaluator, publish only aggregate proof |
| operator approval and rollback checkpoint | no current approval/checkpoint artifact exists for this graph | add dry-run approval/checkpoint artifact or monitored promotion workflow |
| post-promotion monitor window | no current monitor-window artifact exists; historical scheduler says observed window was `0 ms` against `14,400,000 ms` minimum | run monitored rollout or dry-run monitor proof after a candidate is frozen |
| real Glass/Zed consumer evidence | blocked by missing real-consumer executor wiring | implement real consumer executor adapter before claiming consumer parity |
| real coding quality | current headless corpus records `0/8` mutating tasks with file edits; each mutating task fails closed through terminal verification | wire a real model/executor path that can generate edits, then rerun visible and hidden gates |

Resolved after the first rollup:

- first-class edit attempt telemetry is now materialized from visible real ACP corpus manifests into `.bag/evidence/edit-attempt-records.jsonl`;
- edit-attempt scorecard projection is now materialized into `.bag/evidence/scorecards/edit-attempt-projection.json`;
- the optimizer gate suite no longer carries `edit-policy promotion needs first-class edit attempt telemetry` as a blocking reason.
- stale no-write blocker strings are filtered during optimizer-gate materialization;
- current no-write gate selection ranks runs by freshness, including date-bearing run ids, so stale lexicographically larger run ids cannot override current runs.

## Hidden Holdout Safety

Current evidence proves the policy shape, not a final holdout pass:

- `optimizer-input.hidden-holdout` is evaluation-only in `.bag/evidence/optimizer/input-slices.json`.
- The visible ACP no-write gate is built from optimizer-visible train/dev style evidence only.
- No hidden holdout content was copied into candidate-generation input by this lane.

The hidden holdout todo remains pending because there is no frozen candidate and no current holdout result artifact.

## Commands Run

```bash
npm run bag -- evidence validate --graph-id live-acp-evidence-readiness-v1
npm run bag -- evidence scorecards --write --graph-id live-acp-evidence-readiness-v1
npm run bag -- evidence optimizer-gates --write --graph-id live-acp-evidence-readiness-v1
npm run bag -- evidence release-proof --write --graph-id live-acp-evidence-readiness-v1
npm run bag -- evidence validate --graph-id live-acp-evidence-readiness-v1
bun test src/evidence/evidence-commands.test.ts src/replay/no-write-slice.test.ts src/replay/no-write-validation.test.ts
npx tsc -p tsconfig.json --noEmit --pretty false
npx tsc -p tsconfig.test.json --noEmit --pretty false
```

## Plan Status

Completed:

- `enumerate-current-blockers`
- `wire-visible-no-write-evidence`
- `wire-edit-telemetry-evidence`

Pending:

- `prove-hidden-holdout-final-gate`
- `prove-rollback-approval-monitor-window`

Reason: the remaining items require new runtime/source work or a frozen candidate. This lane intentionally did not fabricate promotion-ready artifacts.
