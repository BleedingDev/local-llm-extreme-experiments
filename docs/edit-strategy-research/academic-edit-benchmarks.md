# Academic Edit Benchmark Notes

## Summary

Recent code-editing benchmarks are useful, but they measure different layers of an agentic edit system:

- **Intent and semantic edit quality**: Can the model infer the requested code change from an instruction and context? Examples: [EDIT-Bench](https://arxiv.org/abs/2511.04486), [CanItEdit](https://arxiv.org/abs/2312.12450), [CodeEditorBench](https://arxiv.org/abs/2404.03543), [RES-Q](https://arxiv.org/abs/2406.16801).
- **Edit representation and application reliability**: Can the model package an already-known edit in a format that parses, applies, and preserves exact text? Examples: [Diff-XYZ](https://openreview.net/pdf?id=1TgJd7uxOM), [Aider edit-format docs](https://aider.chat/docs/more/edit-formats.html), [AST Edits](https://geometricagi.github.io/2026/04/02/ast-edits.html).
- **Long-context state tracking**: Can the model reconstruct or update file state after long histories of diffs, branch merges, and repeated edits? Example: [LoCoDiff](https://abanteai.github.io/LoCoDiff-bench/).
- **Oracle strength and benchmark validity**: Can test suites distinguish "requested edit applied" from "only requested edit applied, without regressions"? Example: [Edit, But Verify](https://arxiv.org/html/2604.05100v1).

The main local takeaway is to evaluate edit strategies as a pipeline, not as static model rankings. A useful BleedingAgent eval should label failures by **transport validity**, **applicability**, **semantic correctness**, **regression preservation**, **scope control**, **retry repair**, and **cost/latency**, then let traces decide which strategy helps which local failure class.

Do not collapse these sources into a fixed model-to-edit-format table. The public results are stale quickly, have different task shapes, and often confound model quality with harness design, output format, retrieval, and oracle strength.

## Source Quality

| Source | Quality | What it measures well | Main stale-risk / limitation |
| --- | --- | --- | --- |
| [EDIT-Bench](https://arxiv.org/abs/2511.04486) | Strong benchmark paper with real IDE-origin data. | In-the-wild instructed single-file edits with instruction, full context, highlighted code, cursor position, multilingual instruction variants, and unit-test pass@1. | Public leaderboard ages quickly. Core data is mostly Python/JavaScript and tests are manually built after collection. The later [Edit, But Verify](https://arxiv.org/html/2604.05100v1) audit questions oracle strength and representativeness. |
| [Edit, But Verify](https://arxiv.org/html/2604.05100v1) | Strong empirical audit, April 2026 arXiv. | Benchmark-validity lens: language/domain coverage, test counts, coverage, fail-before/pass-after, test scope, duplicated contexts, artifact failures. | Audits only CanItEdit and EDIT-Bench after filtering. Some classification uses LLM-assisted labels, though prompts and artifacts are released. |
| [Diff-XYZ](https://openreview.net/pdf?id=1TgJd7uxOM) and [dataset card](https://huggingface.co/datasets/JetBrains-Research/diff-xyz) | Strong for edit-format mechanics; NeurIPS 2025 workshop paper and public dataset. | Isolates diff understanding across Apply, Anti-Apply, and Diff Generation with unified diff variants and search/replace. Reports parse rate, apply rate, exact match, IoU, and add/delete F1. | Single-file real commits, not natural-language intent following. Exact text reconstruction is stricter than many coding tasks and does not measure tests or multi-file consistency. |
| [LoCoDiff](https://abanteai.github.io/LoCoDiff-bench/) | Useful public benchmark, but not a peer-reviewed paper. | Long-context file-state tracking from real git histories; exact final-file match over 200 files from five repos. | Not instruction editing. Exact-match final-state reconstruction may overemphasize transcription and underemphasize semantic intent. |
| [CanItEdit](https://arxiv.org/abs/2312.12450) | Strong early instructed-edit benchmark with hand-authored tasks and tests. | Dual "lazy" and "descriptive" instructions, hidden test suites, fail-before/pass-after style validation, pass@1, and ExcessCode. | Python-only, relatively small, handcrafted, and whole-after-code generation rather than local patch application. |
| [CodeEditorBench](https://arxiv.org/abs/2404.03543) | Broad public benchmark; good task taxonomy and OJ harness. | Debug, translate, polish, and requirement-switch tasks across C++/Java/Python; many generated tests verified by online judge. | [Edit, But Verify](https://arxiv.org/html/2604.05100v1) excludes it from human-instruction audit because instructions are LLM-generated. Less representative of IDE edit traces. |
| [RES-Q](https://arxiv.org/abs/2406.16801) | Useful repo-scale agent benchmark. | Repository navigation plus edit execution over 100 handcrafted tasks from GitHub commits; measures end-to-end agent behavior and token efficiency. | Does not isolate edit application format. Confounds retrieval, planning, tool use, and patch quality. |
| [AST Edits](https://geometricagi.github.io/2026/04/02/ast-edits.html) | Low-to-medium source quality, high relevance to edit-format design. | Directly compares unified diff, search/replace, hashline variants, JSON ops, whole-file rewrites, and AST-targeted edits on 29 localized Python tasks. Separates format failures from logic failures. | Blog experiment, small Python-only task set, no peer review. Treat results as hypotheses for local fixtures, not production guidance. |
| [Aider edit formats](https://aider.chat/docs/more/edit-formats.html) and [leaderboard](https://aider.chat/docs/leaderboards/) | Practical tool evidence, not academic. | Tracks correct edit-format usage and task pass rates for whole-file, search/replace-style diff, and unified-diff-like formats across many models. | Exercises are coding-practice based; benchmark is coupled to Aider's prompts, applier, and supported formats. |

## Benchmarks Reviewed

### EDIT-Bench

[EDIT-Bench](https://arxiv.org/abs/2511.04486) collects real coding interactions through a VS Code extension. Users highlight code and write a short task description; collection can include the instruction, highlighted region, cursor position, prefix/suffix context, model responses, and whether the user accepted an edit. The paper reports 458 users and 2,672 accepted edit responses before curation, then a core benchmark and a translated complete set of 540 multilingual problems.

Task shape:

- Single-file instructed edits.
- Python and JavaScript focus after curation.
- Context includes full code, highlighted code, and optionally cursor position.
- Four functional edit categories: feature addition, feature modification, bug fixing, and optimization.
- Main metric is pass@1 against unit tests.

Local relevance:

- Good source for **realistic instruction ambiguity**, **highlighted-region dependence**, **short user prompts**, and **edit category labels**.
- Important evidence that context variants matter: highlighted code usually improves success, while cursor position has mixed effects.
- Useful label set for local traces: `feature_addition`, `feature_modification`, `bug_fix`, `optimization`, plus context flags for `highlight_available`, `cursor_available`, and `instruction_language`.

Risks:

- It evaluates full-file regeneration rather than patch-format generation.
- The [Edit, But Verify](https://arxiv.org/html/2604.05100v1) audit reports thin test suites for EDIT-Bench: median 4 tests per problem, lower whole-file coverage, and regression-detection gaps in low-coverage suites.
- A local optimizer should not copy the leaderboard. It should copy the idea of measuring context and task-shape sensitivity.

### Edit, But Verify

[Edit, But Verify](https://arxiv.org/html/2604.05100v1) audits instructed code-editing benchmarks rather than proposing a new edit strategy. It surveys code-related benchmarks and finds that only CanItEdit and EDIT-Bench survive criteria for human-authored instructions, existing-code edits, and test-based evaluation.

Key findings to translate locally:

- Both audited benchmarks are over 90% Python, while real AI-assisted coding activity spans many languages.
- Documentation, tests, build/CI, and maintenance edits are underrepresented or absent despite appearing in real PR activity.
- Test adequacy matters: the audit measures test count, whole-file coverage, diff-region coverage, and whether tests catch unwanted changes outside the edit region.
- EDIT-Bench has artifact and duplication issues: some universally unsolved problems trace to benchmark artifacts, and 29% of problems share a codebase with another benchmark problem.

Local relevance:

- Treat benchmark construction as a **validity problem**, not only a pass-rate problem.
- Add local labels for `oracle_strength`, `whole_file_coverage_bucket`, `diff_region_coverage_bucket`, `detects_unwanted_changes`, `fail_before_pass_after`, and `problem_independence`.
- Include out-of-scope and regression fixtures because "edit made" is weaker than "only requested edit made".

### Diff-XYZ

[Diff-XYZ](https://openreview.net/pdf?id=1TgJd7uxOM) isolates edit-format mechanics. It creates 1,000 real single-file edits from CommitPackFT, each represented as old code, new code, and multiple diff formats. The [dataset card](https://huggingface.co/datasets/JetBrains-Research/diff-xyz) reports 200 examples each for Python, JavaScript, Java, Kotlin, and Rust, balanced between single-hunk and multi-hunk edits, with 891 unique repositories.

Task shape:

- **Apply**: old code + diff -> new code.
- **Anti-Apply**: new code + diff -> old code.
- **Diff Generation**: old code + new code -> diff.
- Formats include standard unified diff, relaxed-header unified diff, verbose line-marker unified diff, and search/replace.

Metrics:

- Stripped exact match and line IoU for Apply and Anti-Apply.
- Parse rate, apply rate, post-apply exact match/IoU, and add/delete F1 for Diff Generation.

Local relevance:

- Best source for measuring **format packaging** independently from instruction understanding.
- Gives a direct local metric suite for edit application formats: `parse_rate`, `apply_rate`, `post_apply_exact_match`, `line_iou`, `added_line_f1`, `deleted_line_f1`.
- Supports fixture splits by `single_hunk`, `multi_hunk`, `small/medium/large_change`, `language`, and `change_kind`.

Risks:

- It does not test whether the edit is semantically requested or correct.
- Search/replace and unified-diff results vary by model scale and task phase, so use it to design ablations, not to choose a global format.

### LoCoDiff

[LoCoDiff](https://abanteai.github.io/LoCoDiff-bench/) asks a model to reconstruct the exact current state of a file from `git log -p --cc --reverse --topo-order` output. It contains 200 files, 40 each from Aider, Ghostty, tldraw, Qdrant, and React. It uses exact final-file match with no partial credit and prompts can reach roughly 100k tokens.

Local relevance:

- Captures the edit-strategy failure where a model loses track of file state over long contexts, repeated diffs, and merge conflict resolutions.
- Useful for stale-context and repeated-snippet fixtures: "patch was reasonable for an earlier file state, but not for the current one."
- Encourages a metric for `context_age`, `file_hash_at_prompt`, `file_hash_at_apply`, `prior_edit_count`, and `same_file_retry_count`.

Risks:

- Not an instructed edit benchmark.
- Exact reconstruction rewards transcription and state tracking more than patch design.

### CanItEdit

[CanItEdit](https://arxiv.org/abs/2312.12450) is a Python instructed-edit benchmark with hand-written problems, before/after code, lazy and descriptive instructions, and hidden tests. The paper emphasizes test completeness, correctness, and concealment; suites include unit tests, property-based tests, mocking, fuzzing, integration tests, and structural checks. It reports pass@1 and an ExcessCode metric that penalizes unnecessary successful-code additions.

Local relevance:

- Good model for **fail-before/pass-after**, **behavior-preserving constraints**, and **minimal edit / excess code** scoring.
- Lazy vs descriptive instructions map well to local traces where a planning step may generate a detailed edit spec before an editor applies it.

Risks:

- Python-only and small.
- Evaluates whole after-code generation, not tool-native patch formats.

### CodeEditorBench

[CodeEditorBench](https://arxiv.org/abs/2404.03543) covers four code-editing scenarios: debug, translate, polish, and requirement switch. It curates 7,961 tasks from programming challenge datasets, uses generated test cases whose outputs are verified through an online judge, and reports pass@1 / win-rate-style results.

Local relevance:

- Useful taxonomy source for `debug`, `translate`, `polish`, and `requirement_switch`.
- The OJ framing is useful for executable local fixtures where task success is objective.
- Its generated-test pipeline suggests separating `test_input_generation` from `expected_output_derivation`.

Risks:

- Less representative of natural IDE edits.
- LLM-generated instructions mean it is weaker evidence for real user intent.
- Programming-challenge source data misses build, docs, tests, config, and repo-specific conventions.

### RES-Q

[RES-Q](https://arxiv.org/abs/2406.16801) evaluates repository-editing systems on 100 handcrafted tasks derived from real GitHub commits. It requires the agent to interpret an instruction, navigate a repository, gather information, and construct an edit. It is a useful bridge from single-file benchmarks toward local coding-agent traces.

Local relevance:

- Add multi-file and repository-navigation fixture labels: `requires_search`, `requires_dependency_update`, `requires_test_update`, `requires_config_update`, `requires_cross_file_contract`.
- Measure not only final pass/fail but also token usage, file-read count, file-write count, and whether the edit touched the intended files.

Risks:

- It confounds planning/retrieval/tool-use with edit application.
- It is not a clean edit-format benchmark.

### AST / Structured Edit Discussions

[AST Edits](https://geometricagi.github.io/2026/04/02/ast-edits.html) compares output formats on localized Python tasks: unified diff, hashline unified diff, hashline search/replace, plain search/replace, hashline JSON ops, whole file, and AST-targeted edits. It reports a useful failure split: format failures where the edit cannot be applied, and logic failures where the edit applies but is semantically wrong. The experiment claims AST-targeted edits had zero format failures in its small setup, while unified diff failures were dominated by context/hunk mismatch.

Local relevance:

- Treat structured edits as a hypothesis: targeting by symbol may reduce transcription/context-match failures in large or repeated files.
- Add `target_resolution_failure` as a first-class failure type. AST operations can fail because the symbol is ambiguous, absent, generated, overloaded, language-unsupported, or modified concurrently.
- Compare structured edits against patch/search/whole-file on the same local fixtures instead of assuming a universal winner.

Risks:

- Blog-scale evidence only.
- Python `ast` support does not transfer directly to TypeScript, Rust, shell, markdown, generated files, or partial syntax.

### Aider Polyglot and Edit Formats

[Aider's edit-format docs](https://aider.chat/docs/more/edit-formats.html) are practical rather than academic, but they are directly relevant because Aider exposes whole-file, search/replace-style, and unified-diff-like formats and reports whether a model used the correct edit format in its benchmark. Its [leaderboard](https://aider.chat/docs/leaderboards/) says the polyglot benchmark uses 225 Exercism tasks across C++, Go, Java, JavaScript, Python, and Rust.

Local relevance:

- Use "correct edit format" as a separate metric from "task solved".
- Add a local `format_obedience` metric even when the applier is tolerant.

Risks:

- Results are coupled to Aider's prompt templates and applier.
- Exercism tasks are educational and do not represent messy repo traces.

## Metrics and Labels

### Metric Taxonomy

| Layer | Metric | Why it matters locally |
| --- | --- | --- |
| Output validity | `schema_valid`, `parse_rate`, `format_obedience`, `fence_valid`, `json_valid` | Separates "model knew the edit" from "model packaged it in a usable format." |
| Applicability | `apply_rate`, `hunk_context_match`, `search_unique_match`, `target_symbol_resolved`, `file_hash_match` | Captures stale context, repeated snippets, wrong line numbers, and ambiguous search blocks. |
| Text fidelity | `post_apply_exact_match`, `line_iou`, `added_line_f1`, `deleted_line_f1` | Useful when expected output is deterministic, especially for format ablations inspired by Diff-XYZ. |
| Semantic correctness | `unit_tests_pass`, `typecheck_pass`, `lint_pass`, `runtime_smoke_pass`, `golden_behavior_pass` | Captures applied-but-broken edits. |
| Regression preservation | `preexisting_tests_pass`, `non_edit_region_mutation_detected`, `public_api_compat_pass` | Checks the gap highlighted by Edit, But Verify: not only "was the edit made?" but "was only the requested edit made?" |
| Minimality and scope | `changed_file_count`, `changed_line_count`, `out_of_scope_lines`, `excess_code`, `untouched_constraint_pass` | Detects overbroad whole-file rewrites and unrelated cleanup. |
| Repair loop | `retry_count`, `same_error_repeat_count`, `verifier_repair_success`, `time_to_first_applicable`, `time_to_green` | Measures whether verification feedback helps or causes churn. |
| Cost | `input_tokens`, `output_tokens`, `wall_time_ms`, `tool_calls`, `applier_retries` | Needed because whole-file and multi-pass strategies can pass while being too expensive. |

### Label Taxonomy

Use labels that let the optimizer generalize from local traces:

- **Task intent**: `feat`, `fix`, `refactor`, `perf`, `style`, `docs`, `test`, `build`, `ci`, `chore`, `translate`, `polish`, `requirement_switch`.
- **Code domain**: `frontend`, `backend_api`, `cli`, `data_processing`, `ml_training`, `ai_native_app`, `build_system`, `test_infra`, `docs`.
- **Edit surface**: `function_body`, `function_signature`, `class_body`, `import_export`, `config`, `test_file`, `docs`, `generated_file`, `cross_file_contract`.
- **Context shape**: `short_file`, `long_file`, `long_prompt`, `highlighted_region`, `cursor_position`, `stale_context`, `prior_failed_attempt`, `repeated_snippet`, `multi_hunk`, `multi_file`.
- **Verifier shape**: `no_verifier`, `parser_only`, `unit_tests`, `typecheck`, `lint`, `coverage`, `mutation_proxy`, `snapshot`, `manual_oracle`.
- **Expected risk**: `format_sensitive`, `context_drift_sensitive`, `semantic_edge_case`, `scope_regression_sensitive`, `ambiguous_location`, `ambiguous_requirement`.

## Failure Modes

The table below is the risk map for local edit-strategy evaluation. It separates packaging/apply risks from semantic and verification risks so GEPA can learn from the failure layer, not just final pass/fail.

| Failure mode | Observable signal | Likely local fixture |
| --- | --- | --- |
| Malformed edit | Patch/parser/schema rejects output before touching files. | Invalid hunk header, invalid JSON op, missing code fence, mixed prose in patch. |
| Non-applicable edit | Output parses but applier cannot locate target. | Search text absent, search text matches multiple places, stale hunk context, wrong filename, target symbol missing. |
| Wrong-file or wrong-region edit | Edit applies outside intended target. | Repeated helper functions, same class name in two files, stale highlight, renamed file. |
| Applied-but-broken edit | Applier succeeds; parser/tests/typecheck fail. | Syntactic indentation bug, missing import, wrong branch condition, edge-case test failure. |
| Narrow-oracle pass with regression | Targeted tests pass; broader tests fail or unrelated code changes. | EDIT-Bench-style low-scope test fixture plus hidden regression suite. |
| Under-edit | Edit applies and compiles, but required behavior is missing. | Instruction asks for source + test update; model changes source only. |
| Over-edit / excess code | Correct behavior plus unrelated rewrites. | Whole-file rewrite changes formatting, public API, or unrelated helper. |
| Long-context state loss | Patch is valid for an earlier state, not current state. | LoCoDiff-inspired git-history prompt or repeated same-file retry trace. |
| Multi-file incoherence | Individual file edit is plausible but repo contract breaks. | Update exported function without imports, source without tests, config without docs. |
| Verifier repair churn | Same failing verifier reason repeats across retries. | A failing test log that requires one-line fix, but repeated retries edit unrelated code. |

## Gaps/Limitations

- **Static model ranks stale quickly**. EDIT-Bench and Aider leaderboards include contemporary model results, but they age faster than their task/metric designs. Use their methodology, not their rankings.
- **Most benchmarks under-measure edit transport**. CanItEdit and EDIT-Bench primarily score whole-code outputs against tests; they do not isolate patch parse/apply failures the way Diff-XYZ does.
- **Most edit-format benchmarks under-measure semantics**. Diff-XYZ can tell whether a model manipulates diffs faithfully, but not whether a natural-language change was the right change.
- **Regression preservation is under-tested**. Edit, But Verify shows that thin suites can miss unwanted modifications outside the requested edit region.
- **Language and domain skew are material**. Python-heavy benchmarks can hide TypeScript, Rust, Java, config, shell, build, docs, and test-infra failure modes.
- **Structured edits need parser coverage**. AST or tree-sitter operations can reduce text-match failures, but only when syntax is parseable and the target abstraction matches the edit.
- **Exact match can be too strict; tests can be too weak**. Local evals should combine exact/text metrics, executable checks, and scope/regression checks.
- **Multi-turn repair is underrepresented**. Local traces likely include verifier feedback, retries, partial patches, and stale intermediate states. Single-shot benchmarks miss this.

## BleedingAgent Eval Fixtures

Create fixture families that are small enough to run often but labeled enough to explain why a strategy won or lost.

1. **Malformed edit fixture**
   - Same semantic fix requested across output formats.
   - Inject expected failures: invalid hunk header, invalid JSON operation, missing file marker, unmatched fence.
   - Score: `schema_valid`, `parse_rate`, `first_error_type`, `retry_to_valid`.

2. **Non-applicable / stale-context fixture**
   - Prompt includes old file hash and a target snippet that has drifted by whitespace, rename, or adjacent edit.
   - Score: `file_hash_match`, `apply_rate`, `stale_context_detected`, `refresh_before_retry`.

3. **Repeated-snippet disambiguation fixture**
   - File has two or more identical helper blocks or repeated tests.
   - Instruction disambiguates by nearby symbol, call path, or test failure.
   - Score: `search_unique_match`, `target_region_correct`, `wrong_region_edit`.

4. **Applied-but-broken fixture**
   - Patch applies cleanly but misses a semantic edge case.
   - Score: `apply_rate`, `unit_tests_pass`, `edge_case_test_pass`, `repair_success_after_test_log`.

5. **Narrow-oracle over-edit fixture**
   - Public test checks only the requested behavior; hidden regression checks unrelated preserved behavior.
   - Score: `target_tests_pass`, `regression_tests_pass`, `out_of_scope_lines`, `excess_code`.

6. **Long-file localized multi-hunk fixture**
   - 2k to 5k line file, edits far apart, repeated class/function names.
   - Compare whole-file, unified diff, search/replace, and structured/symbol-targeted forms.
   - Score: `input_tokens`, `output_tokens`, `apply_rate`, `wrong_region_edit`, `latency`.

7. **Multi-file contract fixture**
   - Source change requires import/export, tests, schema, or docs update in another file.
   - Score: `changed_file_set_correct`, `typecheck_pass`, `tests_pass`, `missing_companion_edit`.

8. **Verifier repair fixture**
   - First attempt intentionally exposes a failing parser/test/typecheck log.
   - Score: `same_error_repeat_count`, `repair_delta_size`, `time_to_green`, `new_failure_introduced`.

9. **Format-choice ablation fixture**
   - Same task rendered as whole-file rewrite, unified diff, search/replace, and structured op.
   - Score all layers separately rather than only final pass/fail.
   - Hypothesis target: identify which local task labels predict format/applicability failures.

10. **Oracle-strength fixture**
    - Pair a weak suite and a strong suite for the same task.
    - Score: `weak_pass_strong_fail`, `diff_region_coverage_bucket`, `detects_unwanted_changes`.

### Eval Hypotheses

- **H1: Format failures and semantic failures respond to different fixes.** Prompting or constrained decoding may improve parse/apply rate without improving tests; verifier repair may improve tests without improving patch applicability.
- **H2: Whole-file rewriting should reduce applier failures but increase token cost and out-of-scope changes.** Measure both, do not infer from pass@1 alone.
- **H3: Search/replace and unified diff should be most fragile under repeated snippets, stale context, and long files.** Structured target-by-symbol edits may reduce those failures but introduce parser and symbol-resolution failures.
- **H4: Highlighted code should help when intent localization is the problem and hurt when the highlight is stale or misleading.** Track highlight freshness and whether the final edit overlaps the highlighted region.
- **H5: Smaller/local models may fail more at packaging than at intent.** Separate "could describe the right change" from `format_obedience` and `apply_rate`.
- **H6: Verifier feedback should be scored by error-class movement.** A retry that turns `malformed_edit` into `applied_but_broken` improved transport but not semantics; a retry that repeats the same error should be penalized.
- **H7: Exact-match metrics should be used only when the target file state is canonical.** For normal coding tasks, combine text fidelity with tests, typecheck, lint, and scope checks.

## Trace/GEPA Feedback Implications

Trace logging should preserve the failure layer, not just final success:

- Record `edit_format`, `applier`, `file_hash_before_prompt`, `file_hash_before_apply`, `file_hash_after_apply`, `changed_files`, `changed_lines`, and `target_region`.
- Normalize applier/verifier errors into a compact taxonomy: `format_invalid`, `parse_invalid`, `context_not_found`, `context_ambiguous`, `target_symbol_not_found`, `wrong_region`, `tests_failed`, `typecheck_failed`, `regression_failed`, `scope_violation`.
- Store verifier feedback as structured fields: failing command, failing test name, first stack frame, error message class, and whether the next retry addresses the same class.
- Reward movement through the pipeline: invalid -> applicable -> semantically closer -> green and scoped. Penalize same-error loops and unrelated diff growth.
- Feed GEPA with task labels and failure deltas, not global rules. A strategy that helps `context_ambiguous` fixtures may be neutral or harmful for `multi_file_contract` fixtures.
- Preserve local workload distribution. Public benchmarks underrepresent docs, tests, CI/build, and TypeScript-heavy work; local traces should determine their actual weights.
- Keep a small live canary suite refreshed from recent traces to detect prompt/format regressions when models, appliers, or repo conventions change.

## Open Questions

- What are the dominant local trace categories by language, file type, and edit intent?
- Which edit formats are already supported by the BleedingAgent applier, and can each emit machine-readable error classes?
- How much parser coverage is realistic for structured edits across the repo languages we care about?
- What minimum fixture count gives stable signal for malformed edit, stale context, repeated snippets, multi-file contracts, and verifier repair?
- Can we derive weak/strong oracle pairs automatically from existing tests plus mutation or coverage probes?
- How should we score partial progress in multi-file edits where one file is correct and another companion edit is missing?
- How often do local failures come from model intent errors versus applier constraints versus stale context supplied by orchestration?
- What privacy and provenance rules should govern adding real agent traces into a refreshed local eval corpus?
