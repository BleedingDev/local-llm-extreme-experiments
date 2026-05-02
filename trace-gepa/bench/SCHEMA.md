# Task + Result Schema

All tasks live in `trace-gepa/data/benchmark_tasks.jsonl` (one JSON object per
line). v1 has 105 tasks. This document is the contract between the dataset
and any harness.

## Task object

| field                | type                  | required | notes |
|----------------------|-----------------------|----------|-------|
| `id`                 | string                | yes      | `task_<category>_<3-digit>`; globally unique. |
| `category`           | enum string           | yes      | one of: `tool_routing`, `command_synthesis`, `edit_safety`, `path_grounding`, `debugging`, `recovery`, `planning`. |
| `difficulty`         | enum string           | yes      | one of: `easy`, `medium`, `hard`. |
| `source_record_ids`  | array of strings      | yes      | upstream trace event IDs the task was derived from (e.g. `cc_660da9c6_evt00017`). May be empty for synthetic seeds. |
| `prompt`             | object                | yes      | rendered into the user-facing prompt (see below). |
| `expected`           | object                | yes      | structured ground truth (see below). |
| `verifier_kind`      | enum string           | yes      | one of: `regex`, `exact_match`, `structural_json`, `tool_name_match`, `tool_family_match`, `lm_judge`, `shell_exec`, `composite`. v1 dataset only uses `regex` (12) and `structural_json` (93). |
| `verifier_spec`      | object                | yes      | tier-specific spec (see below). |
| `rubric_weight`      | float                 | yes      | aggregator weight; v1 always `1.0`. |
| `human_readable_summary` | string            | yes      | one-line description used by leaderboard tooling. |

### `prompt` sub-object

| field | type | notes |
|-------|------|-------|
| `prompt.user_request` | string | the natural-language request the agent saw. May be multi-paragraph. Truncated to 1500–6000 chars by harnesses. |
| `prompt.context.available_tools` | array of strings | tool names the agent could call at this step (e.g. `Read`, `Bash`, `ToolSearch`). Order is preserved from the upstream trace. |
| `prompt.context.available_skills` | array of strings | optional Skill names available. |
| `prompt.context.recent_actions` | array | last few assistant actions (objects with `name`/`tool_name` + `input`, or strings). |
| `prompt.context.recent_tool_results` | array | optional truncated results from the last few tool calls. |

### `expected` sub-object

| field | type | notes |
|-------|------|-------|
| `expected.primary_action.tool_name` | string | gold tool name. |
| `expected.primary_action.input_pattern_regex` | string | regex the predicted `input` should satisfy. `.*` if unconstrained. |
| `expected.must_avoid_actions` | array | tools the prediction MUST NOT pick. |
| `expected.must_include_keywords_in_reason` | array | substrings required in the reason field. |
| `expected.must_avoid_keywords_in_reason` | array | substrings disallowed in the reason field (e.g. `maybe`, `i think`). |

### `verifier_spec` sub-object

`verifier_spec.type` is informational (`json_schema`, `regex`, etc.). The
key payload is `verifier_spec.pattern_or_command` — interpreted by the
dispatcher in `bench/verifiers/__init__.py`.

| `verifier_kind`      | meaning of `pattern_or_command` |
|----------------------|---------------------------------|
| `regex`              | a regex string evaluated against the model's full output. |
| `structural_json`    | a JSONPath-style equality predicate, e.g. `$.tool_name == "Read"`. |
| `tool_name_match`    | exact tool name to match. |
| `tool_family_match`  | comma-separated list of acceptable tool names (any-of match). |
| `exact_match`        | literal string the prediction must equal after trim. |
| `lm_judge`           | rubric prompt for the tier-2 judge (`bench/verifiers/tier2_judge.py`). |
| `shell_exec`         | shell command; exit-0 means pass (`bench/verifiers/tier3_shell.py`). |
| `composite`          | spec dict with sub-verifiers and a combine rule (`all`, `any`, `weighted`). |

## Distribution (v1, n=105)

- By category: `tool_routing` 28, `planning` 15, `debugging` 15, `recovery` 15,
  `command_synthesis` 12, `edit_safety` 10, `path_grounding` 10.
- By difficulty: `hard` 46, `medium` 42, `easy` 17.
- By verifier kind: `structural_json` 93, `regex` 12.

## Result schema (per harness)

All harnesses emit the same envelope to `--output`:

```jsonc
{
  "summary": {
    "n": 105,
    "pass_rate": 0.85,
    "by_category":   { "<cat>":  { "n": 15, "pass_rate": 0.93 } },
    "by_difficulty": { "<diff>": { "n": 42, "pass_rate": 0.81 } },
    "parser_counts": { "json_direct": 90, "json_substring": 8, "raw_text": 7 },
    "error_counts":  { "codex_unavailable": 0 },
    "timeouts": 0,
    "mean_elapsed_s": 4.3
  },
  "config": { "model": "...", "tasks_path": "...", "elapsed_total_s": 510.2 },
  "results": [ /* per-task rows */ ]
}
```

### Per-task row fields (union across harnesses)

| field | source | notes |
|-------|--------|-------|
| `id`, `category`, `difficulty` | copied from task | always present. |
| `score` | float in [0,1] | from the verifier; 0 on any error. |
| `verifier` / `verifier_signal` / `verifier_details` | from dispatcher | the signal e.g. `tool_match`, `regex_no_pattern`, `json_parse_fail`. |
| `predicted_tool` | parsed | `predicted.tool_name` if JSON parsed. |
| `predicted_preview` / `parsed_output` | string / dict | up to 240 chars of the parsed prediction. |
| `raw_output` | string | up to 1200 chars of model output (Anthropic harness). |
| `parser_status` | string | one of `json_direct`, `json_fenced`, `json_substring`, `raw_text`, `empty`. |
| `latency_ms` / `elapsed_s` | numeric | per-task wallclock. |
| `prompt_tokens_est`, `output_tokens_est` | int | character-based estimates (Anthropic). |
| `gen_tokens`, `tokens_per_sec` | int / float | tokenizer-derived (MLX). |
| `exit_code`, `timed_out`, `stderr_tail` | subprocess | Codex only. |
| `error` | string \| null | populated when an llm/runner/verify exception occurred. |
