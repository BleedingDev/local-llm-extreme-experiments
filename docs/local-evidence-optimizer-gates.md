# Local Evidence Optimizer Gates

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

Machine-readable suite index: `.bag/evidence/optimizer/index.json`.

## Current Decision

Candidate generation is allowed only as scoped dry-run optimizer work. Promotion and auto-promotion are blocked.

Blocking reasons:

- Post-promotion monitor window is unsatisfied.
- Visible ACP no-write/no-terminal failures must be represented in validation.
- Hidden holdout evaluation is not ready for a frozen candidate.
- Operator approval and rollback checkpoint are required.
- Edit-policy promotion needs first-class edit attempt telemetry.

## Contracts

| Contract | Machine file | Operator summary | Purpose |
| --- | --- | --- | --- |
| Optimizer input slices | `.bag/evidence/optimizer/input-slices.json` | `docs/local-evidence-optimizer-input-slices.md` | Defines candidate-generation, dev-eval, hidden-holdout, monitoring, and live-rollout visibility. |
| Scheduler readiness | `.bag/evidence/optimizer/scheduler-readiness.json` | `docs/local-evidence-optimizer-scheduler-readiness.md` | Maps GEPA readiness artifacts plus scorecards into scheduler promotion gates. |
| Artifact lineage | `.bag/evidence/optimizer/artifact-lineage-contract.json` | `docs/local-evidence-optimizer-artifact-lineage.md` | Requires candidate identity, evidence bundles, scorecards, validation, holdout protection, rollback, and promotion decision metadata. |
| Policy gates | `.bag/evidence/optimizer/policy-gates.json` | `docs/local-evidence-optimizer-policy-gates.md` | Defines per-model/per-codebase overlay boundaries and fail-closed promotion gates. |

## Fail-Closed Rule

No candidate may become active if schema quality fails, splits leak, hidden holdout is exposed, lineage is incomplete, model/codebase profile mismatches, benchmark-only uplift hides ACP regression, rollback checkpoint is missing, or post-promotion monitoring is unavailable.

Tool descriptions, edit strategies, routing prompts, verifier tactics, recovery hints, and result style can all be optimized independently, but only for the exact evaluated `modelProfileId` plus `codebaseProfileId` policy tuple.
