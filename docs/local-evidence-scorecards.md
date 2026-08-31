# Local Evidence Scorecard Suite

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

## Suite Contract

Machine-readable suite index: `.bag/evidence/scorecards/index.json`.

The suite links four local scorecards back to `.bag/evidence/index.jsonl`. It is intended to feed optimizer gates, not to declare a single global best model, edit method, or prompt. Every scorecard keeps benchmark family, trace source, model/source visibility, and observability caveats explicit.

## Scorecards

| Scorecard | Machine file | Operator summary | Main signal |
| --- | --- | --- | --- |
| Tool routing | `.bag/evidence/scorecards/tool-routing.json` | `docs/local-evidence-scorecard-tool-routing.md` | Terminal/process tools carry most observed tool risk; visible ACP failures are no-write/no-terminal progress failures, not failed read calls. |
| Edit strategy | `.bag/evidence/scorecards/edit-strategy.json` | `docs/local-evidence-scorecard-edit-strategy.md` | Local evidence does not support one globally best edit method; ACP no-write must be fixed before real ACP edit-method ranking is meaningful. |
| Recovery | `.bag/evidence/scorecards/recovery-failure.json` | `docs/local-evidence-scorecard-recovery.md` | Recovery pairs are dominated by nonzero terminal exits, hallucinated paths, timeouts, retry loops, and cancelled parallel batches. |
| Benchmark results | `.bag/evidence/scorecards/benchmark-results.json` | `docs/local-evidence-scorecard-benchmark-results.md` | Terminal-bench evidence has useful repeated-task signal; visible ACP replay remains a separate negative baseline. |

## Cross-Scorecard Gates

Auto-promotion must fail closed unless all of these are satisfied:

- Selected optimizer inputs pass schema/parse checks and split leakage checks.
- Hidden holdout evidence is excluded from candidate generation.
- Visible ACP no-write/no-terminal failures are represented in validation.
- Benchmark improvements are reported within comparable evaluator families.
- Edit policy candidates include enough telemetry to attribute preview, apply/write, verify, repair, rollback, and self-detected broken-file outcomes.
- A rollback checkpoint and post-promotion monitor window exist before promotion.

## Immediate Optimizer Direction

The next lane should connect these scorecards to the existing optimizer readiness and promotion modules. The priority is fail-closed policy selection per model and per codebase: tool descriptions, edit strategy, recovery hints, verifier tactics, and prompts must be independently optimizable and attributable to local evidence bundles.
