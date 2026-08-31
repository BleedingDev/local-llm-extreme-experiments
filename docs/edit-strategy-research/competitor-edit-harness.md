# Competitor Edit Harness Patterns

## Summary

This note extracts edit-harness and tool-contract lessons from public product docs, source repos, benchmark harness files, and one issue-level signal. It does not recommend a fixed model-to-strategy mapping for BleedingAgent. The main conclusion is narrower: edit strategy, rendered tool contract, verification policy, repair policy, and fallback policy should be measured as separate optimizer-controlled artifacts.

Taxonomy and risk map:

| Edit family | Public pattern | Main risk | Trace signal needed |
| --- | --- | --- | --- |
| Whole-file write | Aider `whole`; OpenCode/Pi/oh-my-pi `write` tools | Expensive output, accidental overwrite, copied line prefixes, large unchanged text drift | file size, changed-line ratio, overwrite flag, prefix-stripped status, post-write diff |
| Exact search/replace | Aider `diff`; Anthropic text editor `str_replace`; Claude Code/Edit-style args; Pi exact `oldText`/`newText` | Exact-match failure, duplicate snippets, stale read, overlapping replacements | match count, exact vs fuzzy match, duplicate/overlap error, read snapshot ref |
| Multi-edit exact replace | OpenCode `edit`, Pi `edits[]`, Claude Code `MultiEdit` class | Ordering and overlap confusion, partial per-file repair ambiguity | per-edit index, sorted application order, atomicity mode, first failed edit |
| Structured patch | OpenAI/Codex `apply_patch`; OpenCode `apply_patch`; ForgeCode patch micro-evals | Malformed patch grammar, wrong tool chosen, missing context, shell fallback | parse error code, tool selected, path ops, hunk count, shell fallback attempted |
| Fenced/unified diff variants | Aider `diff-fenced` and `udiff` | Model-specific formatting noncompliance, lazy elisions, misplaced path/fence | format variant, malformed-response class, elision detector, apply failure |
| Hash/range anchored edit | oh-my-pi hashline | Anchor transcription errors, line hash collision risk, range boundary mistakes, stale context | anchor parse status, hash mismatch, auto-rebase, boundary warning, stale line display |
| Editor/apply model split | Aider architect/editor mode | Latency/cost and attribution split between planner and editor | planner model, editor model, editor format, inter-model instruction artifact |
| Post-edit verifier/hook loop | ForgeCode verification skill, Aider lint/test, Claude Code hooks, OpenCode LSP diagnostics | Applied-but-broken changes look successful at patch layer | verifier policy id, diagnostics, test command, repair count, rollback/fallback |

## Source Quality

| Source | Quality | Use in this note |
| --- | --- | --- |
| ForgeCode benchmark posts and services docs | Medium-high. Primary project posts with concrete failure categories, but some key runtime services and internal evals are proprietary or self-reported. | Tool naming, schema ordering, schema flattening, truncation wording, enforced verification, per-tool/per-model micro-evals. See [Part 1](https://forgecode.dev/blog/benchmarks-dont-matter/), [Part 2](https://forgecode.dev/blog/gpt-5-4-agent-improvements/), and [ForgeCode Services](https://forgecode.dev/docs/forge-services/). |
| ForgeCode public eval folders | High for what is present, limited for what is absent. The repo exposes focused eval fixtures and validations, not the full proprietary services layer. | Concrete harness pattern: inspect recorded tool calls and gate on tool selection or anti-patterns. See [benchmarks/evals](https://github.com/tailcallhq/forgecode/tree/main/benchmarks/evals), [patch_exact_match](https://github.com/tailcallhq/forgecode/blob/main/benchmarks/evals/patch_exact_match/task.yml), [multi_file_patch](https://github.com/tailcallhq/forgecode/blob/main/benchmarks/evals/multi_file_patch/task.yml), [refactoring_uses_patch](https://github.com/tailcallhq/forgecode/blob/main/benchmarks/evals/refactoring_uses_patch/task.yml), and [read_over_cat](https://github.com/tailcallhq/forgecode/blob/main/benchmarks/evals/read_over_cat/task.yml). |
| OpenAI Codex and apply_patch docs/source | High. Primary official docs and source. | Structured patch lifecycle, result feedback to model, path safety, parser/runtime verification, permission integration, portable apply-patch fixtures. See [Apply Patch docs](https://developers.openai.com/api/docs/guides/tools-apply-patch), [tool instructions](https://github.com/openai/codex/blob/main/codex-rs/apply-patch/apply_patch_tool_instructions.md), [handler](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/apply_patch.rs), and [scenario tests](https://github.com/openai/codex/tree/main/codex-rs/apply-patch/tests/fixtures/scenarios). |
| OpenCode docs/source | High. Public docs plus implementation. | Permission grouping, hook payload naming, apply_patch validation, diff metadata, LSP diagnostics after edit/write. See [OpenCode tools](https://opencode.ai/docs/tools/), [apply_patch.ts](https://github.com/sst/opencode/blob/dev/packages/opencode/src/tool/apply_patch.ts), [edit.ts](https://github.com/sst/opencode/blob/dev/packages/opencode/src/tool/edit.ts), and [write.ts](https://github.com/sst/opencode/blob/dev/packages/opencode/src/tool/write.ts). |
| Aider docs | High for declared public behavior, medium for model-performance extrapolation. | Edit-format families, model-specific format selection, architect/editor split, lint/test repair loop, troubleshooting context-size and format-compliance failures. See [edit formats](https://aider.chat/docs/more/edit-formats.html), [file editing problems](https://aider.chat/docs/troubleshooting/edit-errors.html), [linting/testing](https://aider.chat/docs/usage/lint-test.html), and [chat modes](https://aider.chat/docs/usage/modes.html). |
| Pi and oh-my-pi repos | High for implementation signals, medium for generalization. oh-my-pi is actively evolving and its README advertises many harness features. | Minimal exact-replace contract in Pi, hashline anchored edits in oh-my-pi, truncation and full-output hints, LSP diagnostics on write/edit. See Pi [README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md), Pi [edit tool](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/edit.ts), Pi [edit-diff](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/edit-diff.ts), oh-my-pi [README](https://github.com/can1357/oh-my-pi/blob/main/README.md), [hashline prompt](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/prompts/tools/hashline.md), [hashline implementation](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/edit/modes/hashline.ts), and [read tool](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/tools/read.ts). |
| Anthropic/Claude docs | High for API/tool schema behavior, medium for Claude Code product internals. | `str_replace` exactness, `max_characters` truncation, post-edit LSP diagnostics, hooks for post-edit validation. See [text editor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool), [Claude Code tools](https://code.claude.com/docs/en/tools-reference), and [hooks](https://code.claude.com/docs/en/hooks). |
| OpenAI Codex issue #13773 | Low-medium. Public user report, not verified benchmark evidence. | Only a cautionary signal that model/version/harness/OS combinations can regress edit reliability and must be locally re-tested. See [issue 13773](https://github.com/openai/codex/issues/13773). |

## Competitor Patterns

ForgeCode frames edit/tool reliability as harness engineering rather than model capability alone. Their Part 1 post separates wrong tool selection, wrong argument names, and wrong sequencing into micro-eval targets, and their public evals gate recorded traces for patch-tool use, avoidance of `git apply`, no missing patch operation fields, and no exact-match failures. Their Part 2 post adds four contract-level changes: put `required` before `properties`, flatten nested schemas, write truncation notices in the result body instead of relying only on metadata, and programmatically require verification before finishing.

OpenAI's `apply_patch` docs define a multi-turn edit loop: the model emits patch operations, the integration applies them, returns one output event per call with completed/failed status, and includes useful failure text for recovery. Codex CLI source then treats `apply_patch` as a mutating tool with parser verification, diff/progress events, hook payloads, sandbox/permission integration, and a fallback warning when patch is attempted through shell execution. Its scenario fixtures make the patch spec replayable across languages and platforms.

OpenCode groups `edit`, `write`, and `apply_patch` under a common `edit` permission, which is useful for policy simplicity but risky if trace data needs to distinguish overwrite, exact replace, and patch behavior. Its source pre-validates patch text, resolves embedded paths, renders diffs and per-file metadata before asking permission, applies updates, publishes file watcher events, touches LSP, and returns diagnostics when errors remain. Its hook docs explicitly distinguish `apply_patch` from a generic patch name and record `patchText` rather than a single `filePath`.

Aider exposes edit format as a first-class per-model configuration dimension. Its docs define whole-file, search/replace diff, diff-fenced, unified-diff-like `udiff`, and architect/editor edit variants. The troubleshooting docs point to context overload and prompt-format noncompliance as causes of failed edits, and suggest switching format or using architect mode. Its lint/test loop gives the model verifier output and asks it to repair failed edits.

Pi uses a minimal harness with default `read`, `write`, `edit`, and `bash` tools. Its edit tool accepts `edits[]` exact replacements, requires `oldText` uniqueness and non-overlap, supports legacy argument repair, queues per-file mutations, renders preview diffs, and preserves line endings/BOM. That is a compact example of exact-replace as a small structured contract rather than free-form shell mutation.

oh-my-pi adds a hashline mode: read output displays line-number plus content hash anchors; edit operations target anchors and ranges; the implementation validates hashes against current file content before mutation, may auto-rebase nearby shifted anchors, rejects no-op edits with diagnostics, and returns compact diff previews plus warnings. Its read tool emits explicit continuation text when output is truncated, which matches ForgeCode's lesson that truncation status must be obvious in the model-visible body.

Anthropic's standalone text editor tool centers on `str_replace`, where `old_str` must match including whitespace and indentation, and supports line-range view and `max_characters` truncation. Claude Code docs show a related product pattern: built-in edit/write tools can trigger LSP diagnostics after file edits, and hooks can run formatting, linting, security scans, or tests after `Edit`/`Write`.

## Tool Contract Lessons

Treat the rendered contract as an optimizer target, not documentation. ForgeCode's reported GPT-5.4 improvements came from schema field order, schema flattening, body-visible truncation wording, and enforced verification. These are not strategy choices; they are rendering choices around the same underlying tools.

Prefer familiar, unambiguous argument names. ForgeCode reports measurable improvement after renaming edit args to `old_string` and `new_string`. Pi, OpenCode, Claude Code, and Anthropic all converge on variants of `oldText`/`newText`, `oldString`/`newString`, or `old_str`/`new_str`. BleedingAgent should test naming families rather than assume internal names are harmless.

Flatten schemas where possible. Nested operation objects can express semantics, but ForgeCode observed nested `required` confusion. OpenCode's `apply_patch` minimizes its public schema to `patchText`, while Pi's exact edit tool uses a simple `{ path, edits: [{ oldText, newText }] }` shape. For a given strategy, compare flat and nested variants in micro-evals.

Put mandatory structure before explanatory detail. ForgeCode specifically reports lower schema error rates when `required` appears before `properties`. Whether that holds locally is an eval hypothesis, but rendered schema order should be recorded and mutable.

Make truncation impossible to miss. OpenAI and Anthropic docs expose truncation parameters or context; ForgeCode found metadata-only truncation too subtle; oh-my-pi adds plain continuation text such as remaining line counts and selector hints. Trace should store both machine metadata and exact body wording.

Separate edit families in trace even when permissions are grouped. OpenCode's docs map all file modifications to `edit` permission, but the harness still needs per-tool strategy ids. A whole-file write, exact replace, and structured patch have different failure classes even if the user approval surface is the same.

Render previewable diffs before mutation when possible. OpenCode and Pi compute diffs for UI/permission rendering. Codex streams patch-progress events from partial patch text. BleedingAgent should distinguish `preview_generated` from `write_committed`.

For exact replace, uniqueness and overlap rules are part of the contract. Pi rejects empty old text, duplicate matches, no-op replacements, and overlapping edits. Anthropic documents exact whitespace matching for `str_replace`. Those are deterministic validation gates, not model preferences.

For hash/range, anchors are both address and stale-context guard. oh-my-pi uses line-number plus hash references, validates them before mutation, and returns mismatch context. The risk is anchor transcription and range boundary mistakes, so the contract must emphasize full-anchor copying and inclusive range semantics.

## Verification/Repair Lessons

Prompting "verify your work" is weaker than enforced verification. ForgeCode explicitly distinguishes optional verification prompts from a required verification skill gate. A BleedingAgent policy should record whether verification was suggested, required, skipped by policy, or failed to execute.

Patch success is not task success. OpenAI `apply_patch` docs require status and output for each patch call, but that only covers edit application. Aider's lint/test loop and Claude/OpenCode LSP diagnostics catch applied-but-broken edits after the write. BleedingAgent should score final consistency separately from parse/apply/write success.

Return repairable errors. Codex returns parser/verification failures to the model; OpenAI docs recommend a failed status plus human-readable output; Pi distinguishes not-found, duplicate, empty, overlap, and no-change errors; oh-my-pi returns stale anchor mismatch context. Stable error codes should wrap these messages so GEPA sees categories, not brittle strings.

Record anti-patterns as first-class failures. ForgeCode public evals look for `git apply`, shell `cat`, and missing patch fields in the trace. This is useful beyond pass/fail: a task can pass while using a discouraged path that should penalize the rendered contract or tool availability policy.

Automated diagnostics should be attributed to the edit attempt. OpenCode attaches LSP diagnostics to edit/write/apply_patch output. Claude Code docs describe type warnings after edits. Aider expects failing lint/test output to drive repair. BleedingAgent needs `diagnostic_source`, `diagnostic_scope`, and `diagnostic_caused_by_edit` fields.

Do not hide fallback success. If a patch fails and whole-file rewrite succeeds, that is final task success but primary strategy failure. The trace should retain `fallback_from`, `fallback_to`, failure reason, and repair/fallback count.

## Trace Fields

Competitor signals imply these trace fields in addition to the plan's normalized edit-attempt contract:

- `editStrategyFamily`: `whole_file`, `exact_replace`, `multi_exact_replace`, `apply_patch`, `unified_diff`, `diff_fenced`, `hashline_range`, `editor_model`, or future families.
- `renderedContractVariant`: stable id for schema order, schema depth, arg names, examples, truncation wording, and repair wording.
- `toolNameRendered` and `toolNameInvoked`: captures wrong-tool selection and shell fallback.
- `schemaShape`: `flat`, `nested`, `freeform`, `provider_builtin`, plus required-field ordering.
- `generationFormat`: fenced path outside block, fenced path inside block, V4A/apply_patch, hashline anchors, exact JSON, whole file.
- `parsePhase`: parser id, parse status, parse error code, malformed class, hunk/edit count.
- `validationPhase`: path policy, protected path, unique match count, overlap status, stale-context status, no-op status.
- `previewPhase`: preview diff generated, changed files, additions/deletions, first changed line, preview truncation.
- `permissionPhase`: permission group, user/client approval status, preapproved paths, denied pattern, hook veto.
- `applyPhase`: adapter id, atomicity mode, first failed operation index, partial write status.
- `postApplySignals`: LSP diagnostics, format result, test/lint/build output refs, file watcher events, expected-vs-actual changed files.
- `truncationSignals`: total lines/bytes, selected range, body-visible notice, continuation hint, full-output artifact ref.
- `anchorSignals`: read snapshot id, line/content hash, mismatch lines, auto-rebase status, boundary warning.
- `repairSignals`: model-visible error summary, repair attempt count, repair strategy, fallback path, rollback status.
- `antiPatternSignals`: shell edit, `cat`/redirect write, `git apply`, Python ad hoc file write, manual patch-file editing.
- `attribution`: planner model, editor model, verifier model/tool, policy id, codebase profile id, task shape.

## Eval Hypotheses

H1: Schema field ordering affects malformed tool-call rate for some models. Compare `required` before vs after `properties` across identical tasks and record malformed JSON, wrong arg names, and recovery rate. Source motivation: ForgeCode Part 2.

H2: Flattened edit schemas reduce wrong-shape calls but may reduce semantic clarity for complex multi-file operations. Compare flat `file_path/old_string/new_string` against nested `change` objects, with repeated-snippet and multi-file tasks. Source motivation: ForgeCode Part 2 and OpenCode/Pi simple schemas.

H3: Body-visible truncation notices reduce incomplete-file assumptions more than metadata-only notices. Compare metadata-only, terse body notice, and explicit continuation instruction on long-file localized edits. Source motivation: ForgeCode Part 2, oh-my-pi read continuation, Anthropic/OpenAI truncation docs.

H4: Exact replace has high precision when snippets are unique but degrades sharply on duplicated or stale code. Eval repeated snippets, changed-between-read-and-write files, whitespace variants, and overlapping replacements. Source motivation: Anthropic `str_replace`, Pi `edit-diff`, Aider diff troubleshooting.

H5: Hash/range anchors reduce stale-context corruption but introduce anchor transcription and boundary-selection failures. Eval shifted-line files, duplicate line content, single-line replace, body-only range, block-with-closing-delimiter range, and truncated reads. Source motivation: oh-my-pi hashline prompt and implementation.

H6: Structured patch formats create better multi-file locality and replay artifacts, but malformed grammar and context mismatch need dedicated repair loops. Eval add/update/delete/move, missing hunk context, invalid markers, empty patches, and partial-success policy. Source motivation: OpenAI/Codex apply_patch docs/source/fixtures and ForgeCode patch evals.

H7: Enforced verification catches applied-but-broken edits better than optional self-checks. Compare no verifier, prompt-only verifier, required LSP/lint/test verifier, and required reviewer-model checklist. Score final task correctness, repair count, and false-positive verifier blocks. Source motivation: ForgeCode verification, Aider lint/test, OpenCode/Claude diagnostics.

H8: Editor-model splits improve format compliance for weak planner/editor combinations but complicate attribution and latency. Compare single-call edit generation with Aider-style architect/editor on same strategy and codebase slice. Source motivation: Aider architect mode.

H9: Grouped permissions simplify UX but can hide strategy-specific risk unless trace splits tool families. Compare traces where `write`, `edit`, and `apply_patch` share policy approval but retain separate strategy ids. Source motivation: OpenCode permission grouping.

H10: Anti-pattern trace gates predict future reliability regressions even when tasks pass. Add eval checks for shell editing, `git apply`, `cat`/redirect writes, and ad hoc Python file writes, then correlate with repair/fallback rates. Source motivation: ForgeCode public evals and Codex handler warnings for apply_patch through shell.

## Non-Portable Claims

Do not port ForgeCode's 81.8% TermBench result, GPT-5.4-vs-Opus runtime changes, or field-ordering outcome into BleedingAgent policy without local traces. The public post is useful because it identifies measurable variables, not because its winning settings are universal.

Do not assume `old_string/new_string` is the best argument naming for every model or provider. It is a strong candidate naming family because multiple public tools use similar terms, but naming must be evaluated per target model and endpoint.

Do not assume hashline/range anchors are safer globally. They solve stale-context and repeated-location classes, but add anchor-copy, hash mismatch, range-boundary, and truncation-display classes.

Do not assume a patch grammar that works in Codex will be equally reliable for local or quantized models. OpenAI's `apply_patch` tool is a specialized public contract with parser support and tests; other models may need different examples, simplified syntax, or fallback.

Do not treat Aider leaderboard or docs as routing policy. Aider's format selection is evidence that format is model-sensitive, but BleedingAgent must measure its own model, quantization, codebase, prompts, and verifier stack.

Do not count issue reports as benchmark proof. Codex issue #13773 is useful as a reminder that model/harness/OS/version interactions can regress, not as a measured claim about GPT-5.4.

## Open Questions

- Which initial strategy set gives the smallest representative portfolio for BleedingAgent evals without overfitting implementation effort?
- Should exact replace allow fuzzy fallback, as Pi does, or should fuzzy matching be a separate strategy with different risk scoring?
- What stable error-code taxonomy best covers exact-match, patch-grammar, hash-anchor, permission, ACP write, and verifier failures?
- How should fallback credit be assigned when primary strategy fails but fallback succeeds and verification passes?
- What is the right body-visible truncation wording for each model family, and should it include examples of continuation calls?
- How much source text can be stored in replay artifacts before redaction cost outweighs trace usefulness?
- Should LSP diagnostics be a default post-edit verifier or an optional codebase-profile capability?
- For editor-model splits, how should GEPA attribute failure to planner instruction, editor contract, edit strategy, or verifier policy?
