# GPT-5.5 xhigh-reasoning specialist consultation

**Model:** `gpt-5.5` via Codex CLI 0.128.0 with `model_reasoning_effort=xhigh`
**Tokens:** 18,506 input (7,552 cached) + 549 output + 289 reasoning
**Wallclock:** ~30s
**Date:** 2026-05-02

## Prompt
175-task action-selection benchmark from 26K Codex/Claude-Code traces. 7 categories. 4-tier verifier. Recent eval (n=30 fixed verifier): claude-opus-4-7 + seed = 0.233; claude-opus-4-7 + GEPA-optimised = 0.200; gpt-5.5 high = 0.200; gpt-5.5 xhigh = 0.167. Three categories scored 0.00 — debugging fixed, `command_synthesis` + `path_grounding` are pathological tasks.

Critique as senior LM-eval engineer.

## Answer (verbatim)

### 1. Strongest threat to validity
The benchmark is still measuring verifier/task-schema mismatches as much as model action selection. Any task where `available_tools` cannot satisfy the verifier is invalid and will distort category/model comparisons.

### 2. Split / merge
- Split `command_synthesis`: shell command construction vs tool/API argument synthesis.
- Split `path_grounding`: locating/selecting paths vs using paths safely in edits/commands.
- Consider merging parts of `planning` + `recovery` if both are mostly "choose next safe meta-action."
- Keep `edit_safety` separate; it is behaviorally distinct and high-value.

### 3. Highest-leverage next improvement
**Add a task-validity preflight**: for every item, mechanically prove or LM-assisted verify that at least one action using the declared `available_tools` can satisfy the verifier. **Block eval on invalid tasks.**

### 4. Why xhigh may hurt
Top hypothesis: for single-step tool selection, extra reasoning increases action entropy. The model over-deliberates, invents contingencies, chooses a "more generally sensible" action, or adds arguments that strict verifiers penalize, while `high` stays closer to the obvious trace-like prior.

## Action items derived
1. **Validity preflight** is the single highest-leverage move. Implement immediately.
2. Two category splits (command_synthesis, path_grounding) — defer until preflight clears the existing zero-cat distortions.
3. Reasoning-effort note: don't default to xhigh on single-step tasks. Keep it for multi-step / planning benchmarks only.
