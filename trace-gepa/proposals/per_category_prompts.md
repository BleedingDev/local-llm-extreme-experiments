# Per-Category Prompt Routing

## TLDR
- Single global prompt forces compromise across competing category rules; bash-bias v1 lifted tool_routing but regressed 15 edit_safety tasks because rules collided.
- Route prompts by `task.category` (already in bench): load `prompts/system_<category>.txt` per task, fall back to `system_default.txt`. Same model, same eval driver, different prompts.
- Add `--system-prompt-set <dirpath>` to `action_agent_eval.py` alongside existing `--system-prompt-file`; resolve `task.category` to a file at dispatch time.
- Decouples Pareto frontier — each category gets its own rule without polluting siblings; concrete target +5–10pp overall on 175-task bench, with per-category recall lift surfaced separately.

## Hypothesis
A global prompt is a forced compromise: rules that help category A often hurt category B. Routing per-category lets each rule live where it pays off and stay silent where it would interfere — recovering v1's tool_routing gains without paying the edit_safety tax.

## Design
1. **Bench already has `category`** ∈ {tool_routing, edit_safety, debugging, recovery, planning, command_synthesis, path_grounding}. No bench changes.
2. **Prompt set on disk:** a directory containing `system_<category>.txt` plus `system_default.txt`.
3. **CLI:** `--system-prompt-set <dir>` (mutually exclusive with `--system-prompt-file`). Loader: `path = dir/f"system_{task.category}.txt"`; if missing, fall back to `dir/system_default.txt`.
4. **Initial prompts:**
   - `system_tool_routing.txt`: "For shell-shaped tasks (listing, grep, process, git status), prefer Bash over Read/Grep tools." (the v1 rule, scoped).
   - `system_edit_safety.txt`: "Always Read before Edit. Verify `old_string` is unique; widen context if not."
   - `system_recovery.txt`: "After a tool failure, switch tool family rather than retrying the same call. Re-read state before mutating."
   - `system_default.txt`: today's built-in baseline.
5. **A/B methodology:** baseline (`--system-prompt-file default`) vs router (`--system-prompt-set prompts/router_v1/`) on full 175. Report overall pass + per-category recall delta. Sanity check: `system_default.txt` only run must equal current baseline (router determinism).

## ROI
Directly targets the v1 regression's Pareto failure mode. Cheap to build (4 prompt files + ~20-line dispatcher). If category prompts don't beat default per-category, deletes cleanly. Sets up future per-category GEPA optimization (each prompt evolves independently).

## Self-Critique
Risk: 7 prompts is more surface area to maintain and miscategorized tasks silently get the wrong rules — needs a category-mismatch audit and a fall-through metric.
