# Local Evidence Optimizer Policy Gates

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

Machine-readable contract: `.bag/evidence/optimizer/policy-gates.json`.

## Purpose

This lane defines the promotion boundary for optimizer policy overlays. Tool descriptions, edit tools, routing prompts, verifier tactics, recovery hints, result style, and edit repair/fallback policy are independently optimizable, but only inside the resolved `modelProfileId` plus `codebaseProfileId` policy tuple.

The existing resolver already keeps model profiles, codebase profiles, and model-codebase policies separate. These gates make the promotion posture explicit: no candidate becomes active just because one scorecard improved, one edit method looked good in mixed traces, or a benchmark family produced a high-water result.

## Overlay Dimensions

| Dimension | What It Controls | Separation Rule |
| --- | --- | --- |
| Model profile | Runtime role, provider, endpoint kind, model server profile, context window, output budget, model-specific routing priors | Do not transfer uplift across models unless the evidence explicitly supports it. |
| Codebase profile | Repository or task-family fingerprint, protected-path risk, verifier availability, edit/task shape | Do not promote mixed-codebase priors as codebase-specific proof. |
| Model-codebase policy | Tool versions, rendered tool descriptions, result style, verifier policy, edit strategy, edit contract, repair/fallback policy | Only promoted policy records matching the active model and codebase tuple are selectable. |

## Fail-Closed Promotion Gates

Promotion must fail closed on all of these:

- Schema quality: selected evidence, candidate payloads, scorecards, lineage, and promotion decisions must parse and validate.
- Split safety: train/dev/holdout projection must be explicit, and leakage or unchecked split status blocks promotion.
- Scorecard uplift: gains must be measured within comparable evaluator families and must not hide live ACP regressions.
- Hidden-holdout protection: holdout evidence must be excluded from candidate generation and present in promotion evaluation.
- Rollback checkpoint: a checkpoint with the previous active pointer must exist before any active pointer update.
- ACP no-write validation: visible ACP no-write/no-terminal failures must be represented, and action-required replay cases must not silently end with no workspace progress.
- Post-promotion monitor window: promotion is incomplete until attributed post-promotion evidence is checked and rollback is available.

Additional blocking gates cover edit observability and artifact lineage. Edit strategy promotion needs attributable preview, apply/write, verify, repair, rollback, protected-path, stale-context, and applied-but-broken outcomes. Terminal command patterns and mixed action labels are useful priors, not standalone promotion evidence.

## Rollback Requirements

Promotion writes a checkpoint before updating the active optimizer pointer. If any blocking gate fails, the candidate can be recorded as rejected, but the active pointer stays unchanged. Rollback inspection must be able to read the checkpoint and report whether the previous pointer is available.

Promoted policies apply to new ACP sessions only. Existing sessions stay pinned to their resolved policy, and rollback must preserve that session-pin behavior.

## Operator Notes

Use the scorecards as scoped evidence, not as a global ranking table. Terminal-bench-style jobs, Aider polyglot smoke, real ACP replay, SWE-Bench-style results, and mixed action corpora have different semantics. A policy overlay can promote only when its evidence, model profile, codebase profile, evaluator family, holdout treatment, rollback checkpoint, and monitor window all line up.
