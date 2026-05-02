# Phase-3 Investigation: zero-score categories in FIX5 sweep

## Verdict: MIXED

| Category | Hypothesis | Evidence |
|---|---|---|
| `debugging` | **A** (verifier bug) | `no_repeat` clause read `recent_actions[i].input`, but minibench tasks store the value under `input_excerpt`. All 4 tasks landed on `signal=no_assertions_found`. Hand-crafted "good" outputs scored 0.00 — a verifier-only failure. |
| `command_synthesis` | **C** (pathological tasks) + B | Hand-crafted correct outputs score 1.00, so the verifier is fine. But none of the 4 sampled tasks list `Bash`/`exec_command` in `available_tools` — the regex pattern (`ls.*/Users/...`, `pnpm.*--noEmit`, etc.) requires a shell command, yet only narrative tools (TaskCreate, EnterPlanMode, MCP) are exposed. Models correctly route to those tools; verifier still says "miss." |
| `path_grounding` | **C** (verifier/context mismatch) | Hand-crafted correct outputs (with `tool_name="Bash"`) score 1.00, so the DSL works. But every sampled task lists `available_tools = ["exec_command","spawn_agent",...]` (codex naming) while the DSL requires `tool_name in {"Bash","Glob","Grep"}` (Anthropic naming). Models comply with the prompt and lose every point. |

## Three example tasks per zero-cat (predicted vs expected)

### debugging — VERIFIER BUG
- `task_debugging_001`: pattern `must not repeat failing command verbatim`. Haiku-seed predicted `{"tool_name":"exec_command","brief_reason":"Diagnose fallow installation and npm config issue before retrying health check"}` — does not repeat the failing `npx fallow health ...` command. Old: 0.00 (`no_assertions_found`). Re-scored with fix: **1.00 (`schema_ok`)**.
- `task_debugging_008`: Opus-seed predicted `{"tool_name":"","brief_reason":"No suitable tool available..."}`. Old: 0.00. Re-scored: **1.00**.
- `task_debugging_011`: same pattern, same `no_assertions_found`. Re-scored: **1.00**.

### command_synthesis — PATHOLOGICAL (Hyp C)
- `task_command_synthesis_001`: pattern `ls.*/Users/satan/side/experiments/`, but `available_tools` is 21 narrative-only tools (no `Bash`). Predicted `EnterPlanMode`. Score stays 0.00 even with verifier fix.
- `task_command_synthesis_003`: pattern `find.*/Users/satan/\.claude/projects`, `available_tools = []` (empty list!). Models dutifully say "Available tools list is empty/unknown." Task is impossible.
- `task_command_synthesis_011`: pattern `pnpm.*--noEmit`, tools list omits Bash. Models pick `TaskCreate` etc.

### path_grounding — PATHOLOGICAL (Hyp C)
- `task_path_grounding_002`/`003`/`006`/`007`: identical pattern `tool_name in {"Bash","Glob","Grep"}`, but every prompt's `available_tools` is `["exec_command","spawn_agent","update_plan","wait_agent","write_stdin"]`. Models faithfully output `tool_name = "exec_command"` and the `in` clause fails.

## Verifier fix (Hypothesis A)

`bench/verifiers/tier1_regex.py` — replaced the inline `no_repeat` extractor with `_extract_recent_commands(task)` that:

1. Reads `recent_actions[i].input`, then falls back to `.input_excerpt` (current dataset shape).
2. Handles bare-string actions like `'Bash: {"command":"..."}'` and pulls the inner `cmd`/`command` field via regex.
3. Also harvests the `Command: ...` line that debugging prompts include in `user_request`.
4. Substring guard: only chunks ≥ 12 chars count as "verbatim repeat" (avoids trivial overlap on words like "ls").

### Before/after on the 16-row debugging slice (4 models × 4 tasks)

| Model | Old debugging mean | New debugging mean |
|---|---|---|
| haiku-4-5 seed | 0.00 | 1.00 |
| haiku-4-5 opt  | 0.00 | 1.00 |
| opus-4-7 seed  | 0.00 | 1.00 |
| opus-4-7 opt   | 0.00 | 1.00 |
| codex gpt-5.4 high  | 0.00 | 0.25 (3/4 still json_parse_fail — truncated `predicted_preview`, separate harness issue) |
| codex gpt-5.5 high  | 0.00 | 0.25 |
| codex gpt-5.5 xhigh | 0.00 | 0.25 |

All Anthropic results were re-scored offline against the fixed verifier using the saved `parsed_output` from `bench/results/sweep/`. No new LM calls. (Codex previews are truncated during capture, so they don't parse — that's a separate non-verifier issue.)

Tests added in `tests/test_verifiers.py`:
- `test_no_repeat_handles_input_excerpt_shape`
- `test_no_repeat_handles_bare_string_recent_action`
- `test_no_repeat_uses_command_line_in_user_request`

All three pass. (Pre-existing failure in `test_tool_name_and_family_match` is unrelated.)

## Pathological tasks (Hypothesis C) — flagged for user

These should be repaired or relabeled in `data/benchmarks/minibench.jsonl`:

**command_synthesis (4/4 sampled):** verifier requires shell command, but `available_tools` excludes `Bash`/`exec_command`.
- `task_command_synthesis_001`, `..._003` (empty `available_tools`!), `..._010`, `..._011`.

**path_grounding (4/4 sampled):** verifier requires `Bash`/`Glob`/`Grep`, prompt offers only `exec_command`.
- `task_path_grounding_002`, `..._003`, `..._006`, `..._007`.

Suggested fixes (for user authorization):
1. **command_synthesis**: add `Bash` to `available_tools` in those 4 tasks, OR widen the regex to also accept `exec_command` JSON containing the path/command. Fix the empty-tools case in `task_command_synthesis_003` first.
2. **path_grounding**: change DSL to `tool_name in {"Bash","Glob","Grep","exec_command"}` (recognise codex naming) OR normalise `tool_name` through `_tool_family` before the `in` check.

## Most common model failure mode (Hypothesis B residual)

Across path_grounding/command_synthesis sweep traces, the dominant failure is **tool-vocabulary mismatch**: models pick the closest tool from `available_tools` (correctly) but that tool is not what the verifier expects. Mitigation: route the verifier through `agent_opt.adapter._tool_family` so codex's `exec_command` and Anthropic's `Bash` map to the same family before set-membership checks. This is a one-line change in `_eval_clause` for the `in` kind.

## Files touched
- `trace-gepa/bench/verifiers/tier1_regex.py` (replaced no_repeat extractor)
- `trace-gepa/tests/test_verifiers.py` (3 new tests)
- `trace-gepa/bench/zero_cat_investigation.md` (this report)

LM calls used: **0** (all re-scoring done offline against saved sweep traces).
