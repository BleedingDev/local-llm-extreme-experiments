# BleedingAgent GEPA Operations

Date: 2026-05-01

This note documents the GEPA Operations lane. The product claim is deliberately narrow: BleedingAgent can improve optimizer-controlled artifacts from measured traces, replay/eval scorecards, edit ablations, tool failures, and user correction signals. It must not freely rewrite runtime source code or silently change already-running sessions.

## Implemented Loop

1. Evidence readiness:
   - `src/optimizer/gepa-operations.ts` assesses whether enough evidence exists for candidate generation.
   - It separates candidate-generation readiness from auto-promotion readiness.
   - Gates cover real replay cases, visible replay safety, repeated failure clusters, edit failure volume, tool failure volume, user correction volume, metric observations, post-promotion monitoring window, and regression budget.

2. Feedback bundles:
   - `buildOperatorSafeGepaFeedbackBundle()` and `buildGepaFeedbackBundle()` assemble bounded feedback from replay/eval failures, trace evidence, edit ablations, test output, truncation mistakes, and LLM critiques.
   - Hidden holdout and raw/needs-review replay data are excluded from proposer and GEPA feedback input.

3. Candidate generation:
   - `runGepaOptimizer()` runs a bounded, resumable GEPA loop over feedback records.
   - Deterministic candidates cover model/codebase policy, rendered tool contracts, rendered edit contracts, edit routing/fallback/repair/verifier policy, verification gates, and result-style policy.
   - LLM-backed candidate proposal is optional and scope-checked against deterministic allowed artifact scopes.

4. Evaluation and promotion:
   - `runGepaCandidateEvaluation()` runs baseline-versus-candidate replay/eval checks on visible train/dev first, then hidden holdout when requested.
   - `promoteGepaCandidate()` promotes only passing candidates by writing active optimizer pointers for new sessions only.
   - Promotion writes registry records and checkpoint metadata.

5. Monitoring and rollback:
   - `monitorPostPromotionRollback()` detects regressions from scorecards, eval runs, and trace evidence bundles.
   - `rollbackOptimizerPromotion()` restores a previous active pointer checkpoint deterministically.

6. Operator UX:
   - ACP maintenance commands expose status, eval, optimize report, dry-run promote, and rollback readiness without showing maintenance controls in the normal coding command surface.

## Evidence

Focused GEPA checks:

```bash
bun test tests/optimizer-gepa-operations.test.ts tests/optimizer-candidates.test.ts tests/optimizer-gepa-feedback.test.ts tests/optimizer-gepa-runner.test.ts tests/optimizer-gepa-loop.test.ts tests/optimizer-promotion.test.ts tests/optimizer-gepa-checkpoints.test.ts tests/optimizer-gepa-pareto.test.ts
```

Current result:

- 42 tests passed.
- 0 failures.
- 162 assertions.

Full regression after this lane should also run:

```bash
npm run typecheck
npm test
```

## Boundary

This lane closes the operator-safe GEPA loop primitives. It does not claim a production daemon that autonomously harvests every new run, strengthens every weak oracle, promotes candidates without review, or silently rolls back in long-lived sessions.

Automatic scheduling can be layered on top later. It must reuse these gates and keep holdout leakage, active-pointer mutation, checkpointing, and post-promotion regression handling explicit.
