# Apply, Hash, and Range Anchored Edit Systems

## Summary

This lane covers edit-application mechanisms that separate "what should change" from "how bytes are modified": range-native editor edits, context patches, search/replace blocks, hash/range anchored edits, fast apply models, and architect/editor splits. The through-line is not that one format is best. It is that each format moves risk between planning, anchoring, mechanical application, and post-apply verification.

Risk map at a glance:

| Risk | Formats that reduce it | Formats that can reintroduce it |
| --- | --- | --- |
| Exact old text cannot be reproduced | Hashline anchors, apply models, range edits | Search/replace blocks, strict patches |
| Line numbers drift after earlier edits | Context patches, hash-verified anchors, full-file apply models | Plain range or line-number edits without version checks |
| Repeated snippets match the wrong site | Hash anchors, symbol/AST anchors, full-file semantic apply | Exact search/replace, fuzzy patching with weak context |
| Fast but wrong merge | Deterministic `--check`, syntax/test verification, rollback snapshots | Apply models that "smooth" underspecified edits |
| Partial-read edits miss surrounding constraints | Whole-file apply input, trace of viewed ranges, verifier/fallback | Range/hash tools when the model only saw grep fragments |
| Later verifier failures are hard to attribute | Rich apply traces and rollback links | Opaque editor/apply steps recorded only as final diffs |

The most useful BleedingAgent outcome is probably a trace schema that labels which failure happened, not a fixed model-to-strategy table. The same model may need different edit surfaces depending on file size, read coverage, repetition density, workspace concurrency, and verifier budget.

## Source Quality

| Source class | Examples | Quality note |
| --- | --- | --- |
| Long-lived specs/manuals | GNU diff/patch context formats, Git `apply`, LSP `TextEdit` | Strong for terminology and invariants. They describe mechanics, not LLM behavior. GNU diff notes context lets `patch` relocate hunks after small changes, while Git documents `--check`, context requirements, and `--3way` behavior when blob identities are available. Sources: [GNU diff output formats](https://web.mit.edu/gnu/doc/html/diff_3.html), [Git apply](https://git-scm.com/docs/git-apply/2.40.0.html), [LSP 3.17](https://ntaylormullen.github.io/language-server-protocol/specifications/specification-3-17/). |
| First-party product docs/blogs | Cursor, Fireworks, Morph, Aider, ForgeCode, Relace | Good for public architecture and workflow shape. Treat speed, accuracy, and benchmark claims as vendor-reported unless independently reproduced. Sources: [Cursor Instant Apply](https://cursor.com/blog/instant-apply), [Fireworks/Cursor case study](https://fireworks.ai/blog/cursor), [Morph Apply Model](https://docs.morphllm.com/models/apply), [Morph Fast Apply SDK](https://docs.morphllm.com/sdk/components/fast-apply), [Aider edit formats](https://aider.chat/docs/more/edit-formats.html), [Aider chat modes](https://aider.chat/docs/usage/modes.html), [ForgeCode agents/tools](https://forgecode.dev/docs/creating-agents/), [Relace Apply 3](https://relace.ai/blog/relace-apply-3). |
| Independent or community experiments | Can Boluk hashline, Geometric AST/hashline comparison, `hash-edit`, `ln-diff`, Opal edit-lines docs | Useful for provenance and failure taxonomies, but usually not enough to decide production routing. Many experiments are model-, benchmark-, and harness-specific. Sources: [Can Boluk harness/hashline post](https://blog.can.ac/2026/02/12/the-harness-problem/), [Geometric AST edits](https://geometricagi.github.io/2026/04/02/ast-edits.html), [hash-edit PyPI](https://pypi.org/project/hash-edit/), [ln-diff](https://github.com/dceluis/ln-diff), [Opal EditLines](https://hexdocs.pm/opal/Opal.Tool.EditLines.html). |

## Systems/Concepts

| Concept | Edit/application shape | Useful signals | Main risks |
| --- | --- | --- | --- |
| Range-native editor edits | A client/server sends `{ range, newText }`; LSP defines ranges as zero-based start/end positions and `TextEdit` as range plus inserted text. | Clean when edits are computed against an editor buffer version and applied immediately. Good trace fields include offset encoding, document version, and exact pre-edit digest. | Stale ranges after concurrent edits, UTF-16 vs byte/character offsets, line-ending drift, and under-context when the model only saw a slice. |
| Context/unified patches | Hunks include approximate line ranges and surrounding context. GNU `patch` can relocate hunks by searching for context; Git can check applicability, require context, and attempt 3-way merge if blob identities are recorded. | Mature deterministic apply surface; `git apply --check`, `-C<n>`, `--3way`, and `--reject` produce machine-classifiable outcomes. | Repeated context can still misapply; context-free patches are discouraged by Git; hand-edited patches can have stale hunk counts; whitespace settings change behavior. |
| Search/replace blocks | The model returns old text and replacement text. Aider's `diff` format uses search/replace blocks and `diff-fenced` variants for models with fencing problems. | Low token overhead and easy failure class: old text not found, ambiguous match, parse failure. | Requires exact reproduction of old content. Repeated snippets, indentation, whitespace, and model "lazy" elisions cause failures. |
| Hashline or hash/range anchors | Read output tags each line with a short content hash; edits refer to `line:hash` anchors or ranges. Can Boluk's proposal uses tags for replace line, replace range, and insert-after; `hash-edit` reports version conflicts and anchor mismatches. | Explicit stale-anchor rejection; lower need to copy old text; excellent traceability from read snapshot to apply attempt. | Short hashes can collide or be transcribed incorrectly. Line hashes only prove the referenced line matched, not that the model understood unseen surrounding code. Partial reads and concurrent formatters can cause retry churn. |
| Lazy snippets plus apply model | Planning model emits a partial "edit snippet" with markers like `// ... existing code ...`; a specialized model takes original file, snippet, and instruction and returns merged code. Cursor publicly described a fast-apply model and speculative edits; Morph and Relace expose similar apply-model products. | Moves brittle merge mechanics into a narrow model or service; can reduce full-file rewrite latency and search/replace retries. Trace must capture planner output and applied full diff separately. | "Applied" can mean semantically wrong. Apply models may smooth missing imports or infer intent beyond the snippet, which can help or hallucinate. Vendor accuracy/speed claims are not a substitute for our own evals. |
| Architect/editor split | Aider architect mode sends the request to a main model for solution design, then to an editor model that turns the proposal into file editing instructions. | Separates reasoning failures from edit-format failures. Useful when a reasoning model is poor at producing valid diffs. | Higher cost and latency; proposal-to-editor handoff can lose constraints; editor can faithfully implement a flawed plan. |
| Agent harness patch workflows | ForgeCode exposes tools such as `read`, `write`, `patch`, `search`, `undo`; its docs emphasize narrower tool grants and releases mention failure limits and `replace_all`. | Shows harness-level controls around which agents can edit and when repeated failures stop. Relevant to rollback/fallback design even without a public apply model. | Broad patch tools can still over-apply, especially `replace_all`; safety limits stop loops but do not classify why an edit was wrong. |

## Provenance Notes

- Context anchoring predates LLM agents. GNU diff/patch documentation explains that context lets a patch still apply after nearby line displacement, and Git inherits that family of patch mechanics with additional index/blob checks and 3-way fallback.
- Range edits come from editor and IDE protocols rather than LLM tooling. LSP makes ranges and text replacement a core abstraction; the missing piece for agent use is usually a reliable document version and read snapshot.
- Search/replace blocks are a natural LLM adaptation of ordinary editor commands. Aider documents `diff`, `diff-fenced`, `udiff`, `editor-diff`, and `editor-whole`, including that format choice was model-specific and affected "lazy coding" tendencies.
- Cursor appears to be the public origin point for the modern "fast apply" pattern: generate code in chat, then have a specialized apply model integrate it. Cursor's 2024 post describes synthetic data, fine-tuned DeepSeek/Llama model families, and speculative edits; Fireworks describes deploying Cursor's Llama-3-70B fast-apply fine-tune with speculative decoding.
- Morph and Relace package the same broad idea as a standalone apply-model API: the planning model emits a lazy diff or edit snippet, while a small specialized model performs the merge. Relace explicitly frames the apply model as "the merge algorithm" and discusses error categories such as functional merge error, hallucination, and truncation.
- Hashline provenance is recent and community-driven. Can Boluk's February 2026 post frames the problem as harness design and proposes short per-line content hashes so the model can edit by reference instead of reproducing exact old text. Geometric later used "hashline" as one of several edit formats in a comparative AST/hashline blog benchmark.
- ForgeCode is relevant as a harness example rather than a public apply-model system: public docs show role-specific agents, explicit tool grants, `patch`, `undo`, and failure limits. It is evidence that tool-boundary design and loop control are first-class product concerns.

## Failure Modes Solved

- Stale line or range detection: hashline tools can reject an edit when the referenced line hash no longer matches the file; Git can reject patches with insufficient matching context or use recorded blob identities for 3-way attempts.
- Exact text reproduction: hashline edits avoid copying the old text; apply models avoid requiring the planner to emit a mechanically valid patch; range edits avoid repeated old text when computed by a structured tool.
- Repeated snippets and ambiguous matches: content hashes, symbol/AST targets, and whole-file apply input give the applier more than "first occurrence of this string".
- Patch drift after earlier hunks: context patches can relocate hunks; independent hash/range edit blocks can be validated against the current file rather than assuming serial hunk application.
- Slow full-file rewriting: apply models and partial edit formats reduce the planner's output burden and can keep unchanged code out of the expensive reasoning model's completion.
- Format conformance failures: architect/editor splits and editor-specific prompts let one model reason and another model focus on producing a valid edit format.

## Failure Modes Introduced

- False accept: an edit applies cleanly but at the wrong semantic site, or an apply model returns plausible merged code that fails later tests.
- False reject: a safe edit is rejected because a formatter, import sorter, or concurrent agent changed a hash/range/context line.
- Anchor collision or transcription error: short hashes improve ergonomics but create collision and copy-error questions that need measured rates under our tokenization and model mix.
- Partial-read blindness: a model can correctly reference a visible line hash while missing invariants in unseen surrounding code.
- Smoothing ambiguity: apply models may fix small incomplete diffs, but the same behavior can become hallucinated imports, duplicated functions, or policy-violating extra changes.
- Opaque blame: if the trace only records "apply succeeded" and final diff, later verifier failures cannot be assigned to planner intent, anchor resolution, merge model, formatter, or fallback.
- Retry masking: fallback from hashline to search/replace to full rewrite may improve final pass rate while hiding expensive or damaging first-attempt failures.
- Rollback gaps: partial apply, `replace_all`, or multi-file apply can leave the workspace in a mixed state unless every attempt has a pre-edit snapshot and reversible diff.

## BleedingAgent Trace Fields

Minimum fields to make these systems evaluable:

| Field | Why it matters |
| --- | --- |
| `apply_attempt_id`, `parent_step_id`, `fallback_index` | Reconstructs retry chains instead of only final success. |
| `file_path`, `pre_edit_digest`, `post_edit_digest`, `git_blob_id`, `mtime_before` | Distinguishes stale workspace, concurrent edits, and actual content changes. |
| `read_snapshot_id`, `read_scope`, `viewed_ranges`, `whole_file_seen` | Separates partial-read failures from apply-format failures. |
| `requested_edit_format`, `observed_edit_format`, `anchor_kind` | Allows per-format evals: exact text, context diff, range, hashline, symbol, apply model, full rewrite. |
| `anchor_payload_digest`, `line_hash_algorithm`, `hash_length`, `range_encoding`, `line_ending` | Makes hash/range failures reproducible. |
| `anchor_match_count`, `anchor_mismatch_count`, `ambiguous_match_count`, `context_relocation_distance` | Captures "why did apply reject or choose a site?" |
| `planner_model`, `editor_model`, `apply_model`, `prompt_template_id`, `tool_version` | Separates model behavior from harness behavior. |
| `application_status`, `failure_class`, `partial_apply`, `conflict_markers_present` | Classifies rejected, partially applied, conflicted, or silently accepted edits. |
| `diff_stats`, `touched_ranges`, `outside_requested_scope_changes` | Flags broad rewrites and unexpected collateral changes. |
| `rollback_ref`, `undo_status`, `workspace_dirty_before` | Required for safe fallback and multi-agent recovery. |
| `syntax_status`, `typecheck_status`, `test_status`, `lint_status`, `verifier_failure_digest` | Connects applied-but-broken edits to downstream verification. |
| `latency_ms`, `input_tokens`, `output_tokens`, `retry_count`, `cost_estimate` | Evaluates fast workflows without ignoring hidden retry cost. |

Failure classes worth standardizing:

- `parse_error`
- `anchor_not_found`
- `anchor_ambiguous`
- `anchor_stale`
- `hash_mismatch`
- `range_out_of_bounds`
- `merge_model_error`
- `partial_apply`
- `scope_violation`
- `post_apply_syntax_failure`
- `post_apply_behavior_failure`
- `rollback_failed`

## Eval Hypotheses

- Hash/range anchors should reduce mechanical "old text not found" failures for models that struggle with exact search/replace, but may raise `anchor_stale` and `hash_mismatch` retries when files are formatted or edited concurrently.
- Apply models should lower end-to-end latency on large files and repeated-snippet edits when the planner emits lazy snippets, but they need a separate "applied-but-broken" metric because the merge can be semantically wrong even when it is syntactically valid.
- Range-native edits should perform best when generated from a fresh whole-file or editor-buffer snapshot and guarded by document version; the same format should degrade sharply under partial reads or stale line/character offsets.
- Context patches should be more robust than plain ranges to line displacement, but repeated context and low-context hunks should show higher wrong-site risk unless ambiguity detection or `--check`-style validation is strict.
- Architect/editor splits should improve tasks where the planner model is strong at reasoning but poor at edit syntax; the trace should show whether wins come from fewer format failures or from better task decomposition.
- Fallback chains should be evaluated by total damage, not only final pass rate: retries that eventually pass may still create broader diffs, higher cost, or temporary broken states that matter for multi-agent workspaces.
- Hash length is an eval variable, not a constant. Short tags may help model usability; longer tags may reduce collision risk but increase transcription errors and token cost.
- Verifier-aware apply should beat unverified fast apply on applied-but-broken rates if the harness can roll back and retry with the verifier failure included as structured context.

## Open Questions

- What hash length and hash algorithm minimize silent collision plus model transcription errors for our target models?
- Should a hashline edit require whole-file read provenance, or is a grep/range read enough when the tool can prove anchors match?
- How should an apply model handle "smoothing" such as missing imports: strict merge only, optional repair mode, or verifier-triggered second pass?
- Which verifier failures should trigger fallback to a different edit format versus asking the planner to revise intent?
- How much context relocation is acceptable before a context patch should be treated as risky even if it applies?
- Can rollback be made cheap enough to snapshot before every apply attempt in multi-file agent runs?
- Should trace summaries expose apply-model diffs to the planner on retry, or does that encourage the next planner turn to overfit to a bad merge?
- How should multi-agent ownership locks interact with stale-anchor rejection so one agent's formatter does not cause another agent's harmless edit to churn?
