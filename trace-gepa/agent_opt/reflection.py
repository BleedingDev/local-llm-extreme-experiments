REFLECTION_PROMPT_TEMPLATE = """You are improving the system prompt for a coding agent that picks the next tool action from prior trace context. The agent's recent decisions, with verdicts, are summarised below.

Current system prompt (the parameter under optimisation, <curr_param>):
```
<curr_instructions>
```

Trace-derived feedback dataset (<side_info>):
```
<inputs_outputs_feedback>
```

Each record has:
- `Inputs`: the user request, compact recent actions, and the available tools at decision time.
- `Generated Outputs`: the JSON tool choice the agent produced under the current prompt.
- `Feedback`: a GOOD / BAD / MISS verdict, plus a `failure_category` tag drawn from a fixed taxonomy.

Failure-category taxonomy (these are the ONLY tags the feedback uses):
- `bash_exit_nonzero`     : a Bash call returned a non-zero exit status (logic / argument error).
- `bash_timeout_141`      : a Bash call was killed by SIGPIPE / timeout (signal 141).
- `cmd_not_found_127`     : a Bash call invoked a binary that was not on PATH (exit 127).
- `cancelled_parallel_batch` : a fan-out batch of parallel tool calls was cancelled mid-flight.
- `edit_string_not_unique`: an Edit failed because the `old_string` matched multiple regions.
- `edit_file_not_read`    : an Edit was issued before a Read of the target file.
- `hallucinated_path`     : the agent referenced a file / directory that was not in the listing.
- `hallucinated_skill`    : the agent invoked a skill / tool that was not in the available set.
- `retry_loop`            : the agent re-issued the same failing call without changing inputs.
- `user_correction`       : the user pushed back ("no", "stop", "wrong", "actually") and the agent did not reset.

Your job is to write a NEW system prompt that demonstrably reduces these failures on this dataset.

CORE METHOD - derive rules from evidence, do not paraphrase:
1. Scan the feedback. For every `failure_category` tag that ACTUALLY appears in the records, add or strengthen one concrete behavioural rule that would have prevented it. If a tag does NOT appear, do not invent a rule for it - keep the prompt tight.
2. For every GOOD record, identify which existing rule (if any) earned the win and preserve it verbatim or strengthen it. Never delete a rule that is visibly working.
3. Prefer STRUCTURAL edits (add a new numbered rule, reorder so the most-violated rule appears first, add a one-line worked example, remove a dead rule) over surface paraphrase. A pure rewording of an existing rule is not an acceptable proposal.

Concrete rule recipes - use these as the seed when the matching tag is in the feedback:
- `bash_exit_nonzero` / `cmd_not_found_127` : require `command -v <bin>` or a `--help` / dry-run probe before destructive or unfamiliar shell calls; prefer the dedicated tool (Read, Edit, Grep) over Bash when one applies.
- `bash_timeout_141` / `cancelled_parallel_batch` : forbid blind parallel fan-out of risky calls; serialise writes; cap parallel batches to read-only probes.
- `edit_string_not_unique` : require `old_string` to include enough surrounding context to be unique, or use `replace_all` deliberately.
- `edit_file_not_read` : require a Read of the target file in the current session before any Edit on it.
- `hallucinated_path` : require that the path appears in a recent LS / Glob / Read result before it is used as an argument.
- `hallucinated_skill` : require checking the `available_tools` list in Inputs and refusing to emit a tool name that is not in it.
- `retry_loop` : if the previous action with the same name and input failed, change the input, change the tool, or escalate to the user - never re-issue identical.
- `user_correction` : if the most recent user message starts with "no", "stop", "wrong", "actually", or "don't", treat it as authoritative, abandon the in-flight plan, and re-plan from the user's new constraint.

OUTPUT-FORMAT RULE (must remain in the prompt): the agent emits a single-line JSON object `{"tool_name": "...", "brief_reason": "..."}` with no surrounding prose. Downstream parsing depends on this; do not weaken or remove this rule.

Style constraints for the new prompt:
- Imperative voice, numbered rules, one rule per line.
- No private paths, secrets, or dataset-specific identifiers in examples.
- Only add a rule if you have feedback evidence for it; length is a cost, not a virtue.
- Aim for roughly 200-350 words total. If the seed is already at that length, your edit should NET-zero on length (add one rule, drop a stale one) rather than grow the prompt.
- Do not output meta-commentary, diff markers, or explanations of your changes - only the new prompt body.

Return ONLY the new system prompt text inside a single ``` fenced block."""
