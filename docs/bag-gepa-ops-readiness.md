# BAG GEPA Operations Readiness

Date: 2026-05-01
Status: scaffolding-complete, dataset-bridged, candidate-generation-actionable

## Plan summary

Codex landed the GEPA operations primitives (`src/optimizer/gepa-*.ts`) covering
evidence readiness gates, feedback bundles, deterministic + LLM-backed candidate
proposers, validation, evaluation hooks, promotion + post-promotion monitoring,
and registry-backed checkpoint pointers. The ACP maintenance commands surface
status, eval, optimize report, dry-run promote, and rollback from these
primitives. Tests cover all scaffolding modules in isolation, but no end-to-end
wire connects the live `bench/.bag/optimizer/dataset.jsonl` (85 records) to the
readiness assessor or the proposer, and the promotion pointer does not yet
materialize an artifact at the path consumed by `loadOptimizedPlannerPrompt`.

The optimizer can therefore be exercised by unit tests but cannot yet:
1. Read the live dataset and decide whether to attempt candidate generation.
2. Hand seed prompt fragments to the proposer so generated candidates target
   the actual autonomous-coding-turn / task-shape-router prompt surfaces.
3. Materialize promoted candidate prompts at
   `artifacts/optimized-prompts/latest/best_candidate.json` so that
   `BAG_USE_OPTIMIZED_PROMPT=1` actually loads optimizer output.

## TODO inventory (carried over from gepa-* modules and planning notes)

Scheduler / readiness:
- [S] gepa-scheduler: wire `assessGepaEvidenceReadiness` against
  `bench/.bag/optimizer/dataset.jsonl` and emit a readiness report artifact.
- [S] threshold tuning CLI: expose `minMetricObservationCount` and
  `minRealReplayCases` overrides because we do not yet have replay capture.
- [M] periodic scheduler: cron / launchd / GitHub Actions hook to re-run
  readiness on each dataset refresh; nice-to-have, not blocking.

Candidate generation:
- [M] gepa-candidate-generation: register prompt artifacts as
  `OptimizerRegistryRecord`s and feed them through `runGepaOptimizer` with a
  feedback bundle derived from real failures.
- [M] LLM proposer adapter: small `LlmProposerClient` that proxies into
  `LlmRouter.chatText` so live runs can use the LLM-backed proposer instead of
  the deterministic fallback only.
- [L] holdout-aware candidate scoping: ensure the proposer only sees train/dev
  feedback. Already enforced upstream via gates; leaving here for visibility.

Promotion bridge:
- [S] gepa-promotion-pointer bridge: when `promoteCandidatePatch` writes the
  active optimizer pointer, also materialize
  `artifacts/optimized-prompts/<runId>/best_candidate.json` and rotate the
  `latest` symlink so `loadOptimizedPlannerPrompt()` returns the promoted text.
- [M] rollback bridge: `rollbackOptimizerPromotion` should re-point the
  `latest` symlink at the previous run dir or remove it when no prior run.
- [L] artifact retention policy: prune older `<runId>` dirs after N retained
  candidates.

Cross-cutting:
- [L] secret redaction audit: ensure the dataset adapter never leaks secret-
  bearing verifier output into feedback excerpts (sanitizer already exists in
  `gepa-feedback.ts`; just need a regression test using real records).
- [L] benchmark integration: surface the readiness verdict in
  `docs/benchmarking.md` operator runbook.

## Top 3 immediately actionable

1. gepa-scheduler (S, ~1h): adapter from dataset.jsonl rows to `EvalRunResult`,
   plus a `scripts/run_gepa_scheduler.ts` driver that prints + persists the
   readiness JSON. Unblocks gating decisions for any subsequent runner.

2. gepa-candidate-generation (M, ~2h): seed prompt registry records for
   `prompt.autonomous-coding-turn.system` and
   `prompt.task-shape-router.classifier`, build a `GepaFeedbackBundle` from
   reward=0 dataset rows, plug an `LlmRouter`-backed proposer into
   `runGepaOptimizer`, and emit `bench/.bag/optimizer/candidates.json`.

3. gepa-promotion-pointer bridge (S, ~1h): `materializePromotedPromptArtifact`
   helper that writes `artifacts/optimized-prompts/<runId>/best_candidate.json`
   and atomically rotates the `latest` symlink. Closes the loop with
   `loadOptimizedPlannerPrompt()`.

## Blockers

- No replay capture pipeline yet, so `minRealReplayCases` cannot be satisfied
  without relaxing thresholds. Mitigation: ship the scheduler with a
  `--threshold-min-metric-observations` override and document that
  `minRealReplayCases=0` is acceptable for the bring-up phase.
- `OptimizerRegistryRecordSchema` does not enumerate a `prompt-fragment`
  artifactKind. Workaround: encode prompt records under
  `rendered_tool_contract` kind with `promptFragments=[<promptText>]` and a
  stable `contentHash`. No schema change required.
- LLM proposer requires either Anthropic credentials (`ANTHROPIC_API_KEY`) or
  a routable local provider. Fallback to deterministic generator covers smoke.

## Cross-plan coordination

- Edit-strategy lane already produces ablation reports that flow into
  `buildGepaFeedbackBundle` directly; no extra glue needed.
- ACP maintenance commands consume the same primitives; they remain authoritative
  and this work only adds operator-runnable scripts and a symlink bridge.
- Runtime hot zones (`src/llm.ts`, `src/types.ts`, `src/config.ts`,
  `src/workspace.ts`, `src/acp-agent.ts`) stay untouched.

## Recommended sequence

1. Land the scheduler + dataset adapter; confirm readiness JSON shows
   `candidateGenerationReady: true` with relaxed minMetricObservationCount.
2. Register seed prompt records and run the proposer end-to-end (deterministic
   path is sufficient for first smoke; LLM path is best-effort).
3. Add the promotion-pointer bridge + bun test that asserts
   `loadOptimizedPlannerPrompt()` returns the freshly-promoted prompt.
4. Refresh `bench/bag-runtime/src` from `src/` so the runtime bundle ships
   the new helpers.
5. Layer monitoring (cron + dashboard) once steps 1-3 are stable.

## Out of scope for this readiness pass

- No autonomous loop closing without operator review.
- No silent in-session prompt mutation; promotion only affects
  `appliesToNewSessionsOnly` sessions per existing `PromotionDecision` schema.
- No runtime source rewriting; optimizer artifacts only.
