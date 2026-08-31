# Self-Evolving Runtime Gates Release Proof

Date: 2026-05-04
Graph: `self-evolving-runtime-gates-v1`
Selection hash: `30baf78610`
Snapshot: `.codex/plan-graphs/self-evolving-runtime-gates-v1/snapshot.json`

## Status

The runtime gates graph is implemented through lane 06 and release-proof validation is in progress for lane 07. The coding-agent harness now has enforceable runtime gate status, ACP no-write validation, sealed split helpers, edit-attempt telemetry, reproducible evidence command validation, scoped policy overlays, promotion rollback proof, and ACP session pin stability proof.

Auto-promotion remains intentionally blocked unless all evidence gates pass. Candidate generation remains allowed only as scoped dry-run work.

## Implemented Behavior

- Runtime optimizer readiness is loaded from `.bag/evidence/optimizer/index.json` and enforced before promotion.
- Missing, invalid, or blocking optimizer gate suites reject promotion fail-closed.
- ACP maintenance status surfaces runtime gate status and visible no-write replay slice details.
- ACP no-write/no-terminal failures are represented as structured replay validation cases and aggregate promotion gate evidence.
- Train/dev/hidden-holdout projection is deterministic and hidden holdout misuse fails closed.
- Edit attempt telemetry records strategy family, phases, hashes, verification, fallback, repair, rollback, and self-detected regression evidence.
- Edit-attempt records can be projected into scorecard aggregates when `.bag/evidence/edit-attempt-records.jsonl` exists.
- Policy overlays are scoped to exact model/profile/codebase/policy tuples and cover tool contracts, edit contracts, routing, verifier tactics, recovery hints, result style, and prompt fragments.
- `promoteCandidatePatch` now rejects when no artifact lineage decision is supplied. No active pointer update can happen merely because lineage was omitted.
- Successful promotion requires validation, eval scorecard pass, profile match, aggregate promotion gate pass, artifact lineage pass, runtime readiness, and rollback checkpoint creation.
- Promotion applies to new ACP sessions only; existing ACP sessions keep their resolved optimizer pin.
- Rollback restores the previous active pointer from the checkpoint.

## Proof Commands

Passed:

```bash
bun test tests/optimizer-promotion.test.ts tests/optimizer-session-pin-promotion.test.ts tests/edit-promotion-gates.test.ts tests/optimizer-promotion-runtime-lineage.test.ts tests/optimizer-artifact-lineage.test.ts
npx tsc -p tsconfig.json --noEmit --pretty false
npx tsc -p tsconfig.test.json --noEmit --pretty false
npm test
npm run bag -- evidence validate
npm run bag -- evidence release-proof
python /Users/satan/side/experiments/skills/plan-graph/scripts/plan_graph.py validate --plans-root .codex/plans/self-evolving-runtime-gates --glob '*.plan.md' --depends 'Self Evolving Runtime 01 Gate Integration:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 02 ACP No Write Validation Veto:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 03 Sealed Split Holdout Protection:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 04 Edit Attempt Telemetry:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 05 Reproducible Scorecard Gate Generation:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 06 Policy Overlay Promotion Rollback:Self Evolving Runtime 07 Release Proof And Dogfood'
bun test tests/acp-maintenance.test.ts tests/bag.test.ts tests/replay-split-redaction-holdout.test.ts tests/edit-strategy-telemetry.test.ts tests/replay-real-acp-scorecard.test.ts tests/optimizer-promotion-runtime-lineage.test.ts tests/optimizer-session-pin-promotion.test.ts
bun test src/replay/no-write-validation.test.ts src/replay/no-write-slice.test.ts src/optimizer/no-write-gate.test.ts src/optimizer/split-projection.test.ts src/acp/edit-attempt-record.test.ts src/evidence/generators/edit-attempt-scorecard-projection.test.ts
```

Observed full-suite result:

- `npm test`: 625 passed, 0 failed.
- `npm run bag -- evidence validate`: passed for 32 index records, 4 scorecards, and 4 optimizer contracts.
- `npm run bag -- evidence release-proof`: passed, but validates existing `local-evidence-flywheel-v1` proof artifacts rather than regenerating a fresh proof for `self-evolving-runtime-gates-v1`.

## Current Blockers

The evidence validator still reports `promotionReady: false` with these blockers:

- edit-policy promotion needs first-class edit attempt telemetry
- hidden holdout final gate is not ready for a frozen candidate
- operator approval and rollback checkpoint are required
- post-promotion-monitor-window is unsatisfied
- visible ACP no-write/no-terminal validation must be represented

Some of these are now implemented as runtime/testable mechanisms, but the local persisted `.bag/evidence/**` artifacts have not been regenerated to satisfy them. That distinction matters: code support exists, current local proof artifacts are still conservative.

## Residual Risks

- `bag evidence release-proof` still wraps an older proof artifact and should be upgraded to rebuild a current proof from graph metadata and command outputs.
- GEPA promotion orchestration can evaluate gates, but production auto-promotion should remain blocked until it supplies a complete artifact lineage decision for the candidate.
- Edit-attempt scorecard projection is optional unless `.bag/evidence/edit-attempt-records.jsonl` exists; real ACP dogfood runs need to populate that artifact.
- The dirty worktree contains large pre-existing untracked project work, so release hygiene should separate this graph's changes from older benchmark/model/harness work before committing.

## Handoff

Use this exact graph selection for any continuing `dag`, `helm`, or `subagent-graph` work:

```bash
python /Users/satan/side/experiments/skills/plan-graph/scripts/plan_graph.py summary --plans-root .codex/plans/self-evolving-runtime-gates --glob '*.plan.md' --graph-id self-evolving-runtime-gates-v1 --write-state --depends 'Self Evolving Runtime 01 Gate Integration:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 02 ACP No Write Validation Veto:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 03 Sealed Split Holdout Protection:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 04 Edit Attempt Telemetry:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 05 Reproducible Scorecard Gate Generation:Self Evolving Runtime 06 Policy Overlay Promotion Rollback' --depends 'Self Evolving Runtime 06 Policy Overlay Promotion Rollback:Self Evolving Runtime 07 Release Proof And Dogfood'
```

Next graph should focus on regenerating live evidence artifacts from actual ACP dogfood sessions, rebuilding release proof for this graph id, and keeping auto-promotion blocked until visible no-write, hidden holdout, lineage, rollback, and monitor-window evidence are all current.
