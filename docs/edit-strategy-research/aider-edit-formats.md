# Aider Edit Formats Research

## Summary

Aider is useful evidence for BleedingAgent because it separates code-task success from edit-application compliance. Its leaderboards report both task pass rate and format health fields such as `percent_cases_well_formed`, `num_malformed_responses`, `user_asks`, syntax errors, indentation errors, context exhaustion, timeouts, cost, model, date, version, and edit format. The main value is not a fixed model-to-format lookup. The value is a taxonomy of edit strategies, a risk map, and a measurement vocabulary for deciding when to retry, downgrade, or switch strategy.

The observed taxonomy is:

- `whole`: full updated file. Lowest parser burden, highest output-token and file-size pressure. Aider calls it the easiest format but warns that it is slower and costlier because the model returns entire files ([edit-format docs](https://aider.chat/docs/more/edit-formats.html), [benchmark notes](https://aider.chat/docs/leaderboards/notes.html)).
- `diff`: fenced search/replace blocks. Lower token volume and larger-file reach, but higher malformed-block risk because the model must preserve filenames, fences, and exact search context ([edit-format docs](https://aider.chat/docs/more/edit-formats.html)).
- `diff-fenced`: `diff` with the filename inside the fence. Aider says it exists mainly for Gemini-style fence-placement failures ([edit-format docs](https://aider.chat/docs/more/edit-formats.html)).
- `udiff`: simplified unified diff. Aider introduced it to reduce GPT-4 Turbo "lazy" omissions and avoid brittle line numbers ([unified diff writeup](https://aider.chat/2023/12/21/unified-diffs.html)).
- `editor-*` and architect/editor: two-step plan then edit application. Aider uses a main architect model to propose a solution and an editor model to produce edits; `editor-diff` and `editor-whole` use the same underlying formats with narrower prompts ([chat modes](https://aider.chat/docs/usage/modes.html), [architect writeup](https://aider.chat/2024/09/26/architect.html)).

## Source Quality

Aider sources are first-party and operationally relevant, but not independent. The strongest sources for this task are the official edit-format docs, leaderboard pages, benchmark notes, benchmark harness README, and the raw leaderboard YAML files in `aider/website/_data` ([polyglot YAML](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml), [old edit YAML](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/edit_leaderboard.yml), [architect YAML](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/architect.yml), [QwQ YAML](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/qwq.yml)).

The current polyglot leaderboard page says it tests 225 Exercism exercises across C++, Go, Java, JavaScript, Python, and Rust, and it was last updated on November 20, 2025 ([leaderboard](https://aider.chat/docs/leaderboards/)). On April 30, 2026, that means the data is useful but stale for "current best model" claims. The older code editing leaderboard says it has been replaced by the harder polyglot leaderboard and was last updated on April 12, 2025 ([old edit leaderboard](https://aider.chat/docs/leaderboards/edit.html)).

The benchmark harness README is especially useful for measurement semantics: it says benchmark reports include settings, commit hash, model, edit format, and pass-rate fields, and that `pass_rate_#` depends on the configured number of tries ([benchmark README](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md)). This means Aider pass rates are not pure one-shot parser-compliance measurements unless the run is configured that way.

## Edit Formats Observed

| Format | Shape | Taxonomy implication for BleedingAgent |
| --- | --- | --- |
| `whole` | Full source file per edited path | Baseline fallback and parser-control case. Measures whether correctness improves when edit syntax burden is minimized, while tracking output tokens, truncation, and accidental file overwrite risk. |
| `diff` | Search/replace blocks with path outside the fence | Primary compact edit family. Needs exact-match diagnostics, duplicate-match handling, missing-path handling, and repair-loop traces. |
| `diff-fenced` | Search/replace blocks with path inside the fence | Same patch semantics as `diff`, but isolates filename/fence placement as a measurable error axis. |
| `udiff` | Simplified unified diff without relying on line numbers | Tests whether familiar patch syntax lowers lazy-output and formatting burden. Needs hunk parser, context matching, and line-number distrust. |
| `editor-diff`, `editor-whole`, `editor-diff-fenced` | Editor-mode variants of existing formats | Separate "reasoning/planning" from "edit serialization". Aider's model code prefixes editor formats for `diff`, `whole`, and `diff-fenced` when selecting an editor format ([models.py](https://github.com/Aider-AI/aider/blob/main/aider/models.py)). |
| Architect/editor | Plan from architect, edits from editor | Not an edit syntax by itself. It is a routing pattern that can reduce malformed edits for models that reason better than they serialize edits. |

Search/replace is the core of Aider's `diff` family. Aider also demonstrates semantic search/replace workflows where the model emits multiple blocks for non-identical call-site changes, which is a useful fixture class for BleedingAgent because exact string replacement is only the final application step, not the user's semantic intent ([semantic search/replace transcript](https://aider.chat/examples/semantic-search-replace.html)).

## Metrics Observed

Aider exposes these metric families:

- Task success: `pass_rate_1`, `pass_rate_2`, `pass_num_1`, `pass_num_2`, `total_tests`.
- Format compliance: `percent_cases_well_formed`, `error_outputs`, `num_malformed_responses`, `num_with_malformed_responses`, `user_asks`.
- Code-quality/application failures: `lazy_comments`, `syntax_errors`, `indentation_errors`, `exhausted_context_windows`, `test_timeouts`.
- Repro and cost: `model`, `edit_format`, `editor_model`, `editor_edit_format`, `command`, `date`, `versions`, `commit_hash`, `seconds_per_case`, `total_cost`, token fields where present.

The raw polyglot YAML currently contains 69 rows: 47 `diff`, 15 `whole`, 4 `diff-fenced`, and 3 `architect` runs. In that file, averaged by recorded edit format, `diff` runs have about 92.2% well-formed cases, `whole` about 98.3%, `diff-fenced` about 97.3%, and `architect` 100.0%. These averages are descriptive only because model mix, dates, prompts, and providers differ across rows ([polyglot YAML](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml)).

Concrete examples show why BleedingAgent should log both correctness and edit-application health. In the polyglot data, `chatgpt-4o-latest (2025-03-29)` with `diff` records 45.3% pass rate and 64.4% well-formed cases, while `gemma-3-27b-it` with `whole` records 4.9% pass rate and 100.0% well-formed cases. Well-formed output is therefore necessary for unattended editing, but not sufficient for task success ([leaderboard](https://aider.chat/docs/leaderboards/)).

## Failure Modes

| Failure mode | Aider evidence | BleedingAgent risk |
| --- | --- | --- |
| Malformed edit syntax | Aider tracks malformed responses and cases with malformed responses in leaderboard YAML. | Parser failures can erase all task progress unless the retry preserves intent and diagnostic context. |
| Wrong fence/path placement | `diff-fenced` exists because some models fail Aider's normal fence/path arrangement. | Treat filename placement as a first-class failure axis, not just a generic parse error. |
| Exact-search miss | Search/replace blocks require matching source text. | Need trace fields for match count, normalized match attempts, fuzzy fallback, and stale-file detection. |
| Lazy or elided code | `udiff` was motivated by lazy-code reductions in Aider's GPT-4 Turbo tests. | Track placeholder comments and incomplete implementations separately from parser errors. |
| JSON/tool escaping burden | Aider reports worse code quality when source code is wrapped in JSON compared with plain markdown for whole-file returns ([JSON writeup](https://aider.chat/2024/08/14/code-in-json.html)). | Structured APIs do not remove the need to measure code payload quality and escaping-induced corruption. |
| Context distraction | Aider troubleshooting says excessive code/conversation context can reduce format conformance ([edit errors](https://aider.chat/docs/troubleshooting/edit-errors.html)). | Trace prompt token load and file count whenever malformed-rate regressions are analyzed. |
| Local/quantized model fragility | Aider warns weaker and quantized local models are more prone to edit problems ([edit errors](https://aider.chat/docs/troubleshooting/edit-errors.html)). | Local-model evals must log quantization, context limit, provider/runtime, and tokenizer settings. |
| Architect/editor cost and latency | Aider notes two requests can improve results but add time and cost ([chat modes](https://aider.chat/docs/usage/modes.html)). | Router must measure cost per successful applied patch, not just success rate. |

## What Is Useful For BleedingAgent

Aider's best contribution is an eval schema:

- Record both semantic success and edit serialization success.
- Split edit failure into parse, path, match, apply, syntax, lint/test, truncation, and lazy-output categories.
- Store the chosen format, fallback sequence, retry count, diagnostic prompt, model identity, provider, runtime, context size, output tokens, cost, and wall-clock time in every trace.
- Keep `whole` as a control condition for syntax-burden reduction and `diff`/`udiff`/`diff-fenced` as compact-edit candidates.
- Treat architect/editor as a routing pattern: planner output quality and editor serialization quality need separate spans.
- Add stale-source fixtures where the search block is correct for an earlier file version but fails against the current buffer.
- Add semantic batch-rewrite fixtures similar to Aider's multiple call-site replacement transcript.
- Add fence/path adversarial fixtures: path before fence, path inside fence, missing path, wrong path, nested fences, and extra commentary around blocks.

For trace fields, copy Aider's high-signal names where possible: `edit_format`, `percent_cases_well_formed`, `num_malformed_responses`, `num_with_malformed_responses`, `user_asks`, `lazy_comments`, `syntax_errors`, `indentation_errors`, `exhausted_context_windows`, `test_timeouts`, `seconds_per_case`, and `total_cost`. BleedingAgent should add fields Aider does not fully expose publicly: parser error code, block count, files touched, intended files, match cardinality, fuzzy-match distance, patch size, retry trigger, fallback format, and final user-visible diff size.

## What Must Not Be Assumed

Do not infer a fixed strategy for Qwen, GPT, Gemini, Claude, local models, or any current model from this report. Aider's own docs say different models do better or worse with different formats, and the leaderboard is a mixture of model versions, dates, providers, prompts, and Aider versions ([edit-format docs](https://aider.chat/docs/more/edit-formats.html), [leaderboard](https://aider.chat/docs/leaderboards/)).

Do not treat `percent_cases_well_formed` as correctness. A whole-file run can be perfectly well-formed and still fail most tasks; a diff run can have lower well-formedness and higher task success. The two metrics answer different questions.

Do not treat Aider's leaderboard as a format ablation table. Most rows vary model and format together. The old edit leaderboard includes `whole`, `diff`, `diff-fenced`, and `udiff`, but the polyglot leaderboard is not balanced by model across all formats ([old edit YAML](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/edit_leaderboard.yml), [polyglot YAML](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml)).

Do not assume architect/editor is always better. Aider examples show it can improve some cases, but it adds a second request and changes the failure surface from "model cannot edit" to "planner/editor handoff can drift" ([architect writeup](https://aider.chat/2024/09/26/architect.html)).

Do not use the November 20, 2025 leaderboard state as a current-model ranking on April 30, 2026. Use it as format and measurement evidence unless re-running or re-fetching fresh benchmark results.

## Eval Hypotheses

1. `whole` should have the lowest parse-error rate but the highest truncation, latency, and overwrite risk as file size grows.
2. `diff` should reduce output tokens and support larger files, but failures should cluster around missing paths, malformed delimiters, exact-search misses, and duplicate matches.
3. `diff-fenced` should improve models that misplace filenames relative to fences without changing search/replace semantics.
4. `udiff` should reduce lazy placeholder code on refactor-style tasks compared with custom search/replace blocks, but may need stronger hunk-context matching.
5. Architect/editor should reduce serialization failures for models that can plan but not reliably emit edit syntax, while increasing latency and handoff-drift failures.
6. Format conformance should degrade as irrelevant context, chat history, and file count increase; this should be measured independently from code-task difficulty.
7. Fallback routing should be evaluated by cost per successful applied patch, not pass rate alone: `diff -> diff-fenced -> whole` and `planner/editor -> editor-whole` are candidate sequences to test, not recommendations.
8. A "well-formed but wrong" fixture class is required: valid patches that compile but fail tests, patches to the wrong file, and whole-file rewrites that drop unrelated code.

## Open Questions

- What edit strategies are actually supported or planned in BleedingAgent today, and which parser errors can be classified without changing implementation?
- Do we want a one-shot benchmark, a repair-loop benchmark, or both? Aider's `pass_rate_#` fields depend on tries, so both views matter.
- Which fixture corpus should represent BleedingAgent's real target workloads: small single-file tasks, multi-file refactors, generated-code patching, or local-model constrained editing?
- Should the optimizer learn per-model policy online from traces, or only choose among predeclared strategy bundles during eval?
- How should user-visible safety be scored: minimal diff size, revertability, test pass, compile pass, or preservation of untouched code?
- Can we isolate "edit syntax burden" from "reasoning burden" by running the same model and task set across `whole`, `diff`, `diff-fenced`, and `udiff` with identical prompts except the edit format?
