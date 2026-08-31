# BleedingAgent Edit Evidence Audit

Scope: read-only audit of the live edit portfolio, router, apply layer, fallback, repair, rollback, verification, tests, and evidence gaps. Repo root: `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx`. Branch observed: `master`.

## Current Status Refresh

The audit below preserves Wave 2/2C findings, including some gaps that were later narrowed. A
single-owner runtime follow-up now writes live ACP `replay-capture.json` artifacts that link prompt,
route decision, ACP consumer capability profile, provider/model/policy lineage, file-read hashes,
final edit attempts, tool calls, terminal verification records, trace refs, and run artifacts.
`src/replay/redaction.ts` and `src/replay/dataset.ts` then convert live captures into redacted
optimizer-safe replay cases, with holdout/raw-local leakage blocked by tests.

Remaining boundary: the system still needs more real model/codebase traces before any edit strategy
is considered learned for a model/codebase. Historical metrics are not yet broadly loaded back into
live routing, and several fine-grained attribution fields such as match counts, explicit read
snapshot refs, parent/child repair refs, and created/deleted-file rollback semantics remain future
hardening work.

## Executive Findings

The edit evidence model is real but uneven. `EditAttemptContract` has structured fields for strategy identity, rendered contract identity, phase results, content hashes, target hash rows, verification, post-apply consistency, self-detected regression evidence, repair count/refs, rollback status, fallback lineage/path, tokens, latency, changed counts, protected paths, redaction, and artifacts (`src/edit-strategy/types.ts:283-330`). Telemetry records those attempts as first-class `EDIT` spans with optimizer dimensions, target hash coverage, required phase coverage, fallback path, repair refs, capture completeness, and health classification (`src/telemetry.ts:256-340`, `src/telemetry.ts:559-590`).

The live ACP loop now routes through rendered edit contracts and previews with `applyEdit`, but live strategy selection currently passes no historical metrics or ablation reports into `routeEditStrategy` from `src/acp/edit-routing.ts`. That means the router can use evidence when supplied (`src/optimizer/edit-policy-router.ts:268-280`), but live runs are still mostly task-shape and conservative default routing, not learned from real edit attempts.

The strongest evidence today is synthetic: ablation probes, replay scenarios, and targeted tests cover parse failures, exact-match failures, stale context, protected paths, fallback success, applied-but-broken file consistency failures, applied-but-broken verification failures, self-detected regression, and promotion vetoes (`src/replay/edit-failure-scenarios.ts`, `tests/replay-edit-failure-scenarios.test.ts`). The previously missing implementation feed from live ACP replay captures now exists; the current gap is accumulating enough real model/codebase traces and loading measured routing evidence back into promotion-quality decisions.

## Wave 2C Status

This historical Wave 2C lane improved the synthetic edit evidence loop without touching `src/acp-agent.ts` and without claiming real live extraction. A later runtime follow-up added live ACP replay capture extraction; the boundaries below keep the distinction between synthetic coverage and learned real-world edit strategy quality.

- Ablation reports now declare selection discipline explicitly: ranking is scoped per model/codebase/strategy family, no global winner is selected, and hidden holdout runs are marked non-optimization input.
- Probe results now emit policy feedback targets and metrics for rendered contract fixes, routing, fallback order, repair instructions, verifier enforcement, rollback policy, protected-path policy, and stale-context policy. Family summaries aggregate those target counts for GEPA-facing feedback and router evidence.
- Replay fixtures now split applied-but-broken file consistency from command verification failure, and add a fixture-only promotion veto case that preserves failed holdout replay and gate-decision evidence.
- Promotion tests now cover candidate-family-scoped vetoes and token cost gates in addition to visible train/dev, hidden holdout, protected-path, post-apply, latency, and score gates.

Remaining live evidence gap: live ACP captures can now become replay cases, but promotion-quality
learning still depends on accumulating a larger real corpus and strengthening the attribution fields
listed in the current status refresh. Fixture coverage must not be represented as proof that a given
strategy is optimal for a real model/codebase.

## Current Portfolio

Canonical strategy families include whole-file, exact replace, multi exact replace, fenced diff, unified diff, structured apply_patch, hash/range, apply model, architect/editor, AST structured, range native, and custom (`src/edit-strategy/types.ts:4-17`). The canonical definitions classify maturity, application kind, deterministic apply, partial-read support, failure codes, and trace requirements (`src/edit-strategy/taxonomy.ts:64-81`, `src/edit-strategy/taxonomy.ts:116-357`).

Live deterministic apply supports only five families: `whole_file`, `exact_replace`, `unified_diff`, `apply_patch`, and `hash_range` (`src/edit-strategy/apply-layer.ts:67-73`, `src/edit-strategy/apply-layer.ts:519-522`). The live ACP context filters canonical definitions to those supported families and excludes future-gated definitions in `src/acp/edit-routing.ts`. This leaves `multi_exact_replace`, `fenced_diff`, `apply_model`, and `architect_editor` in the taxonomy and eval corpus, but not live deterministic apply.

Rendered contracts include strategy-specific input schemas and prompt fragments requiring repair, fallback, verifier, and stale-context evidence (`src/edit-strategy/contract-renderer.ts:61-216`, `src/edit-strategy/contract-renderer.ts:278-299`). The live model prompt in `src/acp/coding-generation.ts` asks for a JSON envelope containing edits and commands, then validates each `edit.payload` against the selected rendered input schema.

## Implemented Evidence

Strategy and routing:

- Router metrics can ingest trace, ablation, and manual evidence with parse pass rate, apply accepted rate, protected-path touch rate, stale rejection rate, and applied-but-broken rate (`src/optimizer/edit-policy-router.ts:40-59`).
- Router decisions include selected strategy, candidates, fallback rules, evidence used, and warnings (`src/optimizer/edit-policy-router.ts:134-153`).
- Candidate scoring penalizes protected-path touch rate, applied-but-broken rate, and stale rejection rate (`src/optimizer/edit-policy-router.ts:241-248`, `src/optimizer/edit-policy-router.ts:337-356`).
- Whole-file is a degraded baseline fallback if no eligible measured strategy exists (`src/optimizer/edit-policy-router.ts:193-205`, `src/optimizer/edit-policy-router.ts:522-523`).

Apply layer:

- Apply results expose strategy family, applied/skipped/failed status, changed file before/after content, error code/message, preview diff, and protected-path touch status (`src/edit-strategy/apply-layer.ts:76-91`).
- Exact replace distinguishes stale hash, missing text, and ambiguous matches (`src/edit-strategy/apply-layer.ts:216-226`).
- Hash/range validates content hash and range bounds before mutation (`src/edit-strategy/apply-layer.ts:251-258`).
- Unified and apply_patch hunk application rejects parse failures, missing context, and ambiguous context (`src/edit-strategy/apply-layer.ts:328-355`).
- Protected paths fail closed across supported strategies (`src/edit-strategy/apply-layer.ts:130-141`, `src/edit-strategy/apply-layer.ts:304-311`).

Live ACP loop:

- File reads capture per-file hashes in `CodingFileSnapshot` through `src/acp/coding-runner.ts` and `src/acp/workspace-io.ts`.
- Live routing writes an `edit-routing.json` artifact and traces selected strategy, selected family, rendered contract id, degradation, candidates, and warnings through `src/acp/edit-routing.ts` and `src/acp/coding-runner.ts`.
- Preview runs through `bag.edit.preview` with strategy id, family, rendered contract, target files, reason, and edit input in `src/acp/edit-lifecycle.ts`.
- Writes go through ACP `writeTextFile` with old/new hashes, edit strategy id/family, rendered contract version, permission handling, and diff content in `src/acp/workspace-io.ts`.
- Final lifecycle telemetry adds post-apply consistency, verification, self-check, rollback, and artifact refs to edit attempts through `src/acp/edit-telemetry.ts`.

Synthetic eval and replay:

- Ablation probes record parse status, apply status, expected outcome match, verification status, post-apply status, self-detected status, changed files, protected-path touch, policy feedback targets, and objective metrics (`src/eval-harness/edit-strategy-ablation.ts`).
- Family summaries aggregate parse pass rate, apply accepted rate, protected-path count, stale rejection count, applied-but-broken count, and policy feedback target counts (`src/eval-harness/edit-strategy-ablation.ts`).
- Replay scenarios include parse failure, apply failure, stale context, protected path, fallback success after primary failure, applied-but-broken file consistency failure, applied-but-broken verification failure, self-detected regression, and promotion veto fixtures (`src/replay/edit-failure-scenarios.ts`, `tests/replay-edit-failure-scenarios.test.ts`).

Schema and telemetry capture:

- `REAL_EDIT_ATTEMPT_REQUIRED_PHASES` now names the minimum real-attempt phases that must be visible for optimization: parse, validate, apply, write, post-apply consistency, and verify (`src/edit-strategy/types.ts:38-45`).
- Attempt contracts can carry explicit `renderedEditContractVersion`, per-target before/after hashes, structured fallback path steps, repair attempt refs, and structured self-detected regression evidence (`src/edit-strategy/types.ts:226-281`, `src/edit-strategy/types.ts:293-317`).
- Schema validation checks target hash consistency against input/output hash maps and read snapshots, fallback path consistency against top-level fallback lineage, repair count coverage, and confirmed self-regression evidence (`src/edit-strategy/types.ts:331-459`).
- `editAttemptCaptureIssues` reports partial capture without rejecting legacy attempts, so runtime spans can show exactly which evidence remains missing while older replay fixtures still parse (`src/edit-strategy/types.ts:469-558`).
- Edit telemetry now emits target hash counts/missing paths, required phase coverage, capture status/issues, fallback path length, repair attempt ref count, rendered contract version, and structured self-regression evidence count (`src/telemetry.ts:261-338`).

## Missing Evidence Fields

These gaps are implementation-ready because each has a source location and expected shape.

- Live router evidence input: `resolveLiveEditContext` in `src/acp/edit-routing.ts` does not pass `ablationReports` or `historicalMetrics` into `routeEditStrategy`, so live selection has no real measured edit evidence. Needed: load recent edit metrics by model/codebase/profile and pass them as `historicalMetrics`; record `evidenceUsed` in live artifacts.
- Read provenance: live edit attempts set `readSnapshotRefs: []` even though file snapshots have relative path, content, and hash in the `src/acp/coding-runner.ts` flow. Needed: snapshot ids, `wholeFileSeen`, viewed ranges, and artifact refs for every file read used by an edit.
- Task shape: live successful edit attempts still lose `editContext.taskShape` while parse-failure attempts preserve more routed shape evidence in the coding/edit telemetry path. Needed: preserve `editContext.taskShape` on all attempts.
- Structured target hash extraction: the schema can now hold `targetContentHashes`, and telemetry derives hash rows from existing maps, but `editAttemptFromAcpWrite` in `src/acp/edit-telemetry.ts` does not populate per-target `readSnapshotId`, `writeArtifactRef`, or explicit before/after rows yet. Needed: map `CodingFileSnapshot`, `EditApplyResult.changedFiles`, and ACP write artifacts into `targetContentHashes`.
- Rendered contract version in attempt payloads: the schema and telemetry can carry `renderedEditContractVersion`, but live attempts currently only include `renderedEditToolContractId`; the version exists in the session optimizer pin and write tool input across `src/acp/edit-telemetry.ts` and `src/acp/workspace-io.ts`. Needed: copy the pinned version into every edit attempt.
- Stale-context status: `editAttemptFromAcpWrite` in `src/acp/edit-telemetry.ts` still records `staleContextStatus: "not_checked"` even when apply failure is `hash_mismatch` or `anchor_stale`. Needed: map hash/anchor errors to `stale` and add a `stale_context_check` phase.
- Match and anchor diagnostics: exact and hunk match counts are computed internally but not returned (`src/edit-strategy/apply-layer.ts:220-226`, `src/edit-strategy/apply-layer.ts:345-350`). Needed: `matchCount`, `ambiguousMatchCount`, `operationCount`, `firstFailedOperationIndex`, `anchorKind`, `anchorHashAlgorithm`, and `anchorMismatchCount`.
- Apply operation semantics: canonical `apply_patch` describes add/update/delete/move operations (`src/edit-strategy/taxonomy.ts:227-247`), but the deterministic parser only supports `*** Update File` and rejects other sections (`src/edit-strategy/apply-layer.ts:432-478`). Needed: either implement add/delete/move or narrow the advertised contract until supported.
- Rendered output correlation: the canonical edit output schema includes `editAttemptId`, `status`, `errorCode`, and `artifactRefs` (`src/edit-strategy/contract-renderer.ts:44-59`), but live `generateCodingPatch` in `src/acp/coding-generation.ts` consumes only `summary`, `edits`, `commands`, and `risks`. Needed: decide whether the model emits attempt ids/status or the runtime owns them exclusively.
- Token and latency evidence: live edit attempts set token usage to zero and omit `latencyMs` in `src/acp/edit-telemetry.ts`. Needed: connect LLM/tool metrics and phase timings to attempts.
- Changed line counts: live attempts still set `changedLineCount: 0` in `src/acp/edit-telemetry.ts`. Needed: compute additions/deletions per changed file from before/after content.
- Fallback/repair/rollback structured refs: the schema can now hold `fallbackPath` and `repairAttemptRefs`, and telemetry exposes their counts, but live ACP still records one-hop fallback attributes and repair phases without parent/child attempt refs, prompt artifacts, or checkpoint/rollback refs in `src/acp/coding-runner.ts` and `src/acp/edit-telemetry.ts`. Needed: populate those new fields from fallback routing, repair generation, and rollback artifacts.
- Runtime capture completeness: telemetry now marks partial spans with `edit.capture_issues`, but the ACP constructors need to populate the new fields before live runs can reach `edit.capture_status: complete`.
- Diagnostic attribution: verification captures command output, but edit attempts only store failed command names, not diagnostic source/scope/caused-by-edit in `src/acp/coding-runner.ts` and `src/acp/edit-telemetry.ts`. This gap is already called out in research (`docs/edit-strategy-research/competitor-edit-harness.md:79-101`). Needed: structured diagnostic refs and causality fields.
- Anti-pattern evidence: research calls for shell edit, `git apply`, redirect write, and ad hoc script write tracking (`docs/edit-strategy-research/edit-format-research-synthesis.md:101-104`), but no live edit attempt fields or ACP trace attributes capture these.

## Applied-But-Broken Tracking

Implemented:

- Contract statuses distinguish post-apply consistency, verification, and self-detected regression (`src/edit-strategy/types.ts:97-123`, `src/edit-strategy/types.ts:225-232`).
- The attempt schema rejects `postApplyConsistencyStatus: "inconsistent"` unless verification or self-check evidence is present, and confirmed self-detected regressions now accept either legacy evidence refs or structured evidence entries (`src/edit-strategy/types.ts:380-399`).
- Ablation converts task assertion failures after apply into `postApplyConsistencyStatus`, `verificationStatus`, `selfDetectedRegressionStatus`, and `appliedButBroken` (`src/eval-harness/edit-strategy-ablation.ts:377-423`, `src/eval-harness/edit-strategy-ablation.ts:781-804`).
- Router and promotion gates use applied-but-broken signals as risk and veto inputs (`src/optimizer/edit-policy-router.ts:241-244`, `src/optimizer/edit-promotion-gates.ts:220-238`).
- Telemetry classifies inconsistent post-apply state, failed verification, confirmed self-detected regression, rollback failure, and protected path touch as unhealthy edit attempts (`src/telemetry.ts:559-590`).

Missing:

- Live post-apply consistency in `src/acp/edit-lifecycle.ts` only verifies that the client file hash matches the just-written hash. It does not detect syntax errors, type errors, test failures, expected-vs-actual changed files, or LSP diagnostics.
- Live `selfDetectedRegressionStatus` is derived mechanically from post-apply and verification statuses in `src/acp/edit-telemetry.ts`. It is not currently a model self-check with evidence about violated user intent.
- Verification failures become `verifier_error` in the coding/edit telemetry path regardless of whether the underlying failure is syntax, behavior, timeout, permission, or infrastructure.

## Fallback, Repair, And Rollback Evidence

Implemented:

- Router fallback rules cover parse failure, apply failure, stale context, protected path, post-apply inconsistency, verification failure, self-detected regression, and context budget exceeded (`src/optimizer/edit-policy-router.ts:454-490`). Protected path violations abort rather than fallback (`src/optimizer/edit-policy-router.ts:480-485`).
- Live fallback can select the next eligible strategy, persist `edit-fallback-routing.json`, trace primary failures, generate a fallback patch, and mark fallback attempts with `fallbackFromStrategyId`, `fallbackToStrategyId`, and trigger attributes through `src/acp/coding-runner.ts` and `src/acp/edit-telemetry.ts`.
- Live repair runs up to two rounds when terminal verification fails or post-apply consistency is inconsistent in `src/acp/coding-runner.ts`. Repair edits carry `repairRound` and record a repair phase through `src/acp/edit-telemetry.ts`.
- Live rollback writes baseline snapshots back through ACP after unrepaired verification or post-apply failure through `src/acp/coding-runner.ts` and `src/acp/edit-lifecycle.ts`.
- Final lifecycle computes rollback status per attempt as not needed, not attempted, succeeded, failed, or partial and records rollback phase evidence through `src/acp/edit-telemetry.ts`.
- Schema-level fallback and repair evidence now supports ordered fallback path steps and repair attempt refs, including trigger phase, repair round, status, and artifact refs (`src/edit-strategy/types.ts:244-265`, `src/edit-strategy/types.ts:312-317`).

Missing:

- Fallback is one-hop and based on the first next fallback-eligible candidate (`src/optimizer/edit-policy-router.ts:454-475`). Needed: ordered fallback chains with attempt indexes and final outcome attribution.
- Parse-failure attempts generated for fallback patches do not preserve fallback-from lineage because `recordPatchParseFailures` in `src/acp/edit-telemetry.ts` receives only the fallback edit context and parse failure text from `src/acp/coding-runner.ts`.
- Repair evidence is attached to repair edit attempts, not to the original failed attempt that triggered repair. Needed: parent attempt id, repair attempt id, repair input artifact refs, and repair result status on the failed primary attempt. The schema now has `repairAttemptRefs`; ACP still has to populate them.
- Rollback has status but no `rollbackRef`, checkpoint id, dirty-before hash, or restore artifact in `src/acp/edit-telemetry.ts`. Research explicitly called for `rollbackRef` (`docs/edit-strategy-research/edit-format-research-synthesis.md:101`).
- Created files without a baseline snapshot cannot be rolled back and are recorded as `rollback_failed` in the `src/acp/edit-lifecycle.ts` rollback path. Needed: add/delete rollback semantics and per-file baseline refs.

## Test Coverage

Covered by tests:

- Apply layer behavior for whole-file, exact replace, hash/range, unified diff, apply_patch, protected paths, stale hash, missing/ambiguous match, parse/context failure, and supported families (`tests/edit-strategy-apply-layer.test.ts:17-172`).
- Attempt contract invariants for applied-but-broken, defaults, complete real-attempt phase/hash/fallback/repair/rollback/self-check evidence, partial capture issue reporting, structured self-regression evidence, target hash consistency, fallback path consistency, repair count coverage, and failed phase error codes (`tests/edit-strategy-types.test.ts:73-280`).
- Telemetry dimensions for applied-but-broken attempts, target hash coverage, required phase coverage, capture status, fallback path length, repair refs, self-regression evidence counts, and trace queries (`tests/edit-strategy-telemetry.test.ts:138-186`).
- Ablation hidden-holdout discipline, parse/apply/stale/applied-but-broken outcomes, and no global strategy winner (`tests/edit-strategy-ablation.test.ts:9-107`).
- Router evidence use, fallback rules, holdout exclusion, task-shape constraints, protected-path risk, and applied-but-broken risk (`tests/edit-policy-router.test.ts:44-183`).
- Promotion gates for visible train/dev, hidden holdout, protected-path veto, post-apply consistency veto, candidate-family-scoped ablation vetoes, latency, token cost, and score thresholds (`tests/edit-promotion-gates.test.ts`).
- Live ACP strategy preview, fallback lineage span attributes, parse-failure attempts, final lifecycle telemetry, and rollback status (`tests/bag.test.ts:1305-1641`).
- Synthetic edit-failure replay scenarios and extraction, including fixture-only promotion veto evidence (`tests/replay-edit-failure-scenarios.test.ts`).

Gaps in tests:

- No live test proves `resolveLiveEditContext` consumes real historical metrics from prior `edit.attempt` spans.
- No live test asserts `readSnapshotRefs` are populated from ACP reads.
- No live test asserts ACP constructors populate `targetContentHashes`, `renderedEditContractVersion`, `fallbackPath`, `repairAttemptRefs`, or `edit.capture_status: complete`.
- No test covers `multi_exact_replace` or `fenced_diff` in live apply, despite canonical definitions.
- No test verifies match counts, operation indexes, or anchor diagnostics because the result schema does not expose them.
- No test covers real model malformed output preserving raw response artifacts.
- No test exercises rollback for created files, deleted files, partial rollback, or permission-denied rollback.

## Real Replay Cases Needed Next

Capture these from real ACP runs, not only synthetic fixtures. Each case should include raw model output artifact, rendered contract id, selected strategy, target file snapshots, input/output hashes, phase results, terminal outputs, and redacted source refs.

1. Malformed rendered payload: selected `apply_patch` or `hash_range`, schema validation failure, raw model output retained, no write.
2. Exact replace miss and ambiguity: include `matchCount`, `ambiguousMatchCount`, search text digest, and whether a fallback succeeded.
3. Stale context after concurrent or formatter change: prior read hash, current hash, stale phase, no write before refresh.
4. Protected path attempt in Safe mode: permission/protected-path evidence, requested path, no fallback masking.
5. Structured patch grammar mismatch: add/delete/move emitted by model while only update is supported, with parser error and repair prompt.
6. Applied-but-broken syntax failure: write succeeds, typecheck/lint fails, rollback attempted, diagnostic artifact linked to edit attempt.
7. Applied-but-broken behavior failure: tests fail with syntactically valid code, repair succeeds or fails, final attempt lineage preserved.
8. Self-detected regression: model flags its own inconsistency before verifier, with explicit evidence refs and no command failure required.
9. Fallback success after primary failure: final task passes while primary failed, primary failure remains optimizer-visible.
10. Repair churn: same verification error repeats across two repair rounds, with repair prompt/input/output artifacts.
11. Rollback edge cases: created file rollback, deleted file rollback, partial multi-file rollback, and rollback permission rejection.
12. No-op request: model emits no changes, runtime records skipped apply, verifier policy decision, and no write.

## Top Implementation Risks

1. Live routing looks optimizer-aware but is not yet evidence-fed. `routeEditStrategy` can score evidence, but `resolveLiveEditContext` in `src/acp/edit-routing.ts` supplies no real metrics.
2. The canonical portfolio is broader than the live apply layer. This can make rendered/eval promises drift from runtime behavior (`src/edit-strategy/taxonomy.ts:116-357`, `src/edit-strategy/apply-layer.ts:519-522`).
3. Applied-but-broken detection in live runs is mostly command-result based; post-apply consistency in `src/acp/edit-lifecycle.ts` is only hash readback.
4. Repair and rollback are operationally present but weakly attributed back to the original failed attempt across `src/acp/coding-runner.ts`, `src/acp/edit-lifecycle.ts`, and `src/acp/edit-telemetry.ts`.
5. Missing match/anchor diagnostics will limit optimizer learning for exact replace, unified diff, apply_patch, and hash/range failures.
6. Without real replay extraction, synthetic coverage can make promotion gates look stronger than the evidence available from actual model/codebase use.

Status: Wave 2C improved ablation, replay fixtures, policy feedback signals, and promotion gate coverage. The remaining blocker is real live extraction from ACP edit attempts into replay/eval evidence; fixture coverage must not be represented as live extraction.
