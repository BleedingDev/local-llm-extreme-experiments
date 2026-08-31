# Edit Format Research Synthesis

Date: 2026-04-30

Graph: `bleeding-agent-edit-strategy-optimizer-7d14174b76`

Plan todo: `edit-format-research`

Inputs:

- `docs/edit-strategy-research/aider-edit-formats.md`
- `docs/edit-strategy-research/apply-hash-range-systems.md`
- `docs/edit-strategy-research/academic-edit-benchmarks.md`
- `docs/edit-strategy-research/competitor-edit-harness.md`

## Executive Summary

The research does not identify a universal best edit tool for any current model, and it should not be used that way. Public leaderboards, product posts, and benchmark papers age quickly and usually confound model, prompt, provider, applier, verifier, task shape, and codebase.

The useful result is a taxonomy and measurement plan:

- Treat edit strategy as optimizer policy, not runtime folklore.
- Measure edit success as a pipeline: generation, parse, validation, preview, stale-context check, apply/write, post-apply consistency, verification, repair, rollback, and fallback.
- Track applied-but-broken edits as first-class failures even when parse/apply/write succeeded.
- Use external evidence to design local evals and trace fields, not to route models.
- Promote strategies only from BleedingAgent traces/evals for the target model, endpoint, quantization, codebase, and task mix.

## Source Quality

| Source area | Sources | Reliability for us | Use |
| --- | --- | --- | --- |
| Aider operational data | [Aider leaderboard](https://aider.chat/docs/leaderboards/), [edit formats](https://aider.chat/docs/more/edit-formats.html), [benchmark notes](https://aider.chat/docs/leaderboards/notes.html), [architect mode](https://aider.chat/docs/usage/modes.html) | High for taxonomy and measurement vocabulary, medium-low for current model routing. | Edit format families, well-formed vs correct distinction, stale leaderboard warning. |
| Classical patch/range mechanics | [GNU diff](https://web.mit.edu/gnu/doc/html/diff_3.html), [Git apply](https://git-scm.com/docs/git-apply/2.40.0.html), editor range concepts | High for mechanical invariants, not LLM behavior. | Context matching, hunk relocation, stale/range failure classes. |
| Hash/range and apply-model systems | [Can Boluk harness/hashline](https://blog.can.ac/2026/02/12/the-harness-problem/), [Cursor Instant Apply](https://cursor.com/blog/instant-apply), [Morph apply docs](https://docs.morphllm.com/models/apply), [Relace Apply 3](https://relace.ai/blog/relace-apply-3) | Medium. Product/community evidence; claims need local reproduction. | Candidate strategies, provenance, failure classes around anchors and apply models. |
| Academic/edit benchmarks | [EDIT-Bench](https://arxiv.org/abs/2511.04486), [Edit, But Verify](https://arxiv.org/html/2604.05100v1), [Diff-XYZ](https://openreview.net/pdf?id=1TgJd7uxOM), [LoCoDiff](https://abanteai.github.io/LoCoDiff-bench/) | High for labels and eval design, not production routing. | Oracle strength, parse/apply metrics, long-context state tracking, local fixture design. |
| Competitor harnesses | [ForgeCode GPT-5.4 writeup](https://forgecode.dev/blog/gpt-5-4-agent-improvements/), [ForgeCode evals](https://github.com/tailcallhq/forgecode/tree/main/benchmarks/evals), [OpenAI apply_patch docs](https://developers.openai.com/api/docs/guides/tools-apply-patch), [OpenCode tools](https://opencode.ai/docs/tools/), [Aider lint/test](https://aider.chat/docs/usage/lint-test.html), [oh-my-pi hashline](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/prompts/tools/hashline.md) | High for harness variables; portability is unproven. | Tool-contract rendering variables, verifier enforcement, fallback/rollback telemetry. |

## Edit Format Taxonomy

| Strategy family | Shape | What it tests | Primary risks |
| --- | --- | --- | --- |
| Whole-file write | Model returns complete new file content. | Removes patch syntax burden. Good baseline/control. | Truncation, accidental overwrite, high token cost, unrelated drift. |
| Exact search/replace | Model returns old text plus replacement text. | Compact local edits with deterministic match validation. | Old text not found, duplicate snippets, whitespace mismatch, stale reads. |
| Multi exact replace | One file gets ordered `oldText/newText` edits. | Batched exact replacements. | Overlap, partial failure attribution, ambiguous repair. |
| Fenced diff/search-replace | Aider-style `diff` / `diff-fenced`. | Same replacement semantics with format/fence/path variation. | Malformed fences, misplaced filename, exact-match failures. |
| Unified/context patch | Patch hunks with context. | Mature deterministic apply mechanics and replay artifacts. | Weak context, wrong-site application, malformed hunk grammar. |
| Structured apply_patch | Explicit add/update/delete/move patch operations. | Multi-file locality, parser feedback, path-safe mutation. | Grammar errors, missing operation fields, wrong fallback to shell. |
| Hash/range anchored edit | Edits target line/range plus content hashes or anchors. | Stale-context detection and less old-text copying. | Anchor transcription, short-hash collision risk, boundary mistakes, partial-read blindness. |
| Apply/editor model split | Planner emits lazy edit; specialized model merges. | Separates planning from mechanical merge. | Opaque false accepts, extra latency, attribution split, vendor dependence. |
| Architect/editor routing | Main model plans; editor model serializes edits. | Separates reasoning quality from edit-format compliance. | Handoff drift, cost, editor faithfully implementing flawed plan. |
| AST/structured edits | Edits target syntax nodes or structural operations. | Potentially precise localized mutation. | Parser coverage, language support, symbol ambiguity, future research only. |
| Range-native/LSP-backed edits | Editor protocol range edits and diagnostics. | Fresh buffer/version-aware edits and post-edit diagnostics. | This is future-gated. Do not implement or research deeper until explicitly approved. |

## External Model Observations Are Hypotheses Only

External sources agree that edit format interacts with model behavior:

- Aider exposes separate formats such as `whole`, `diff`, `diff-fenced`, `udiff`, and editor variants.
- ForgeCode reports model-sensitive tool-contract changes such as schema order, flattening, truncation wording, and enforced verification.
- Apply-model products show a split between planning and mechanical merge.

None of that is operational policy for BleedingAgent.

Use these observations only as hypotheses. The router must learn from local data with sample counts for each model/codebase/task-shape cell. Unknown model/codebase pairs should start from a named baseline and exploration budget, not inherit another project's leaderboard.

## Failure Map

Required failure classes:

- `parse_error`: output cannot be parsed as the selected edit format.
- `path_or_fence_error`: path markers, fences, or file headers are malformed or misplaced.
- `exact_match_not_found`: old text cannot be found.
- `exact_match_ambiguous`: old text matches multiple sites.
- `overlapping_edits`: multiple edits conflict within one file.
- `hunk_context_mismatch`: patch context does not match current file.
- `anchor_not_found`: hash/range anchor is absent.
- `anchor_stale`: anchor existed when read but no longer matches.
- `anchor_ambiguous`: anchor/range is not unique enough.
- `range_out_of_bounds`: range offsets/lines are invalid.
- `partial_apply`: some operations applied and others failed.
- `scope_violation`: edit touched files or regions outside policy.
- `truncation_induced_error`: model acted on truncated or incomplete file state.
- `post_apply_syntax_failure`: file is syntactically broken after successful write.
- `post_apply_behavior_failure`: tests/typecheck/build fail after successful write.
- `self_detected_regression`: model later identifies its own inconsistency with concrete evidence.
- `fallback_masked_failure`: final success came only after an earlier strategy failed.
- `rollback_failed`: rollback could not restore the pre-edit state.

## Trace Fields

The next contract step should include at least these field groups:

- Identity: `editAttemptId`, `modelProfileId`, `codebaseProfileId`, `policyId`, `taskShape`.
- Strategy: `editStrategyId`, `editStrategyFamily`, `canonicalEditToolSpecId`, `renderedEditToolContractId`.
- Rendering: `schemaShape`, `argNameVariant`, `requiredFieldOrder`, `exampleSetId`, `truncationWordingId`.
- Read provenance: `readSnapshotRefs`, `wholeFileSeen`, `viewedRanges`, `inputContentHashes`.
- Parse/apply: `parseStatus`, `parseErrorCode`, `applyStatus`, `applyErrorCode`, `operationCount`, `firstFailedOperationIndex`.
- Matching/anchors: `matchCount`, `ambiguousMatchCount`, `anchorKind`, `anchorHashAlgorithm`, `anchorHashLength`, `anchorMismatchCount`.
- Preview/write: `previewGenerated`, `changedFileCount`, `changedLineCount`, `outsideRequestedScopeChanges`, `permissionStatus`, `writeStatus`.
- Consistency: `postApplyConsistencyStatus`, `verificationStatus`, `diagnosticSource`, `diagnosticCausedByEdit`.
- Repair/recovery: `repairAttemptCount`, `fallbackFromStrategyId`, `fallbackToStrategyId`, `rollbackStatus`, `rollbackRef`.
- Cost: `latencyMs`, `generationLatencyMs`, `applyLatencyMs`, `promptTokens`, `completionTokens`, `estimatedCost`.
- Safety: `protectedPathTouched`, `redactionStatus`, `artifactRefs`.
- Anti-patterns: `shellEditAttempted`, `gitApplyAttempted`, `catRedirectWriteAttempted`, `adHocScriptWriteAttempted`.

## Eval Hypotheses

These are local eval hypotheses, not recommendations:

1. Whole-file writes will have low parse failure but high token/truncation/unrelated-drift risk on large files.
2. Exact replace will work well on unique snippets and fail predictably on duplicate, stale, or whitespace-sensitive snippets.
3. Fenced variants will isolate filename/fence placement failures from replacement semantics.
4. Unified/context patches will reduce exact-copy burden but need strict wrong-site and weak-context detection.
5. Structured `apply_patch` will give good multi-file replay artifacts, but local models may need simpler grammar/examples.
6. Hash/range anchors will reduce stale-context corruption but introduce anchor-copy, boundary, short-hash, and partial-read failures.
7. Apply/editor-model splits may reduce mechanical merge errors but require separate planner/editor attribution and applied-but-broken tracking.
8. Enforced verification will catch more applied-but-broken edits than prompt-only self-check.
9. Fallback chains can improve final pass rate while masking primary strategy failure, so fallback needs separate scoring.
10. Tool-contract rendering changes such as schema order, flattening, arg names, examples, and body-visible truncation wording should be ablated locally.

## Proposed First Experimental Strategy Set

This is the first portfolio to measure locally, not a claim that these are best:

1. `edit.whole-file.acp-write.v1`
   - Current baseline/control.
   - Needed to compare syntax-burden-free rewriting against compact edit formats.

2. `edit.exact-replace.v1`
   - Structured `oldText/newText` with uniqueness and overlap validation.
   - Gives clean error classes for exact-match, duplicate-match, stale-read, and no-op failures.

3. `edit.apply-patch.v1`
   - Structured patch or unified/context patch adapter.
   - Tests multi-file locality, parser feedback, path safety, and hunk context behavior.

4. `edit.hash-range.experimental.v1`
   - Hash/range anchored replacement behind an exploration flag.
   - Included because stale-context and repeated-snippet failures are central, but it must not be default until local evals prove value.

Do not include editor/apply-model split in the first implementation wave unless we already have a local candidate model and enough eval budget. It is important but adds attribution, cost, and infrastructure complexity.

## Backlog / Later Candidates

- Apply/editor-model split with planner/editor attribution.
- Architect/editor routing policy.
- AST/structured edits.
- Fuzzy exact-replace fallback as a separate measured strategy.
- Range-native/LSP-backed edits after explicit LSP gate approval.
- Body-visible truncation wording variants.
- Schema shape/name/order variants for GEPA.
- Anti-pattern eval gates for shell writes, `git apply`, ad hoc script writes, and broad file rewrites.

## Recommended Eval Fixtures

Minimum visible train/dev fixtures before policy routing:

- Small single-file exact edit.
- Large-file localized edit where whole-file rewriting is costly.
- Duplicate snippet exact-replace ambiguity.
- Stale read before write.
- Multi-file coordinated patch.
- Malformed patch grammar.
- Path/fence placement variants.
- Hash anchor mismatch and boundary ambiguity.
- Applied-but-broken syntax failure.
- Applied-but-broken typecheck/test failure.
- Self-detected regression with concrete evidence.
- Fallback success after primary strategy failure.
- Protected-path veto.
- Truncated file context.
- Oracle-weak task where tests pass but unrelated behavior is changed.

Each fixture should record visible train/dev/holdout split, oracle strength, changed-file expectations, protected paths, and whether the failure should be attributed to planner intent, edit serialization, apply mechanics, verification, or fallback policy.

## Open Questions

- What is the smallest error-code taxonomy that still distinguishes parser, anchor, apply, ACP write, verifier, and rollback failures?
- How many local samples are required before promoting a strategy for one model/codebase pair?
- Should fallback success count as partial credit, negative signal, or both?
- What default exploration budget is acceptable for unknown local models?
- How much source text can be stored in replay artifacts before redaction cost outweighs usefulness?
- Which verifier signals can reliably classify `caused_by_edit` versus pre-existing failures?
- Should hash/range anchors require whole-file read provenance, or can range reads be enough with explicit confidence?

## Next Plan Step

This research unblocks `edit-outcome-contract`.

The next work should define the normalized edit attempt contract and stable error codes. It should not implement edit tools yet.
