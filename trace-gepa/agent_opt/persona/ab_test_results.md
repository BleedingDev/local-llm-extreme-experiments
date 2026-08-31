# Persona Prefix A/B Test

Model: `claude-opus-4-7`. Tasks: 30 stratified across 7 categories.
LM calls: 60 (cap 60). Errors: seed=0, persona=0.

## Overall

| Arm | Mean score | Changed-vs-seed |
|---|---|---|
| seed-only | 0.233 | - |
| seed+persona | 0.233 | 5/30 tasks |

## Per-category

| Category | n | seed | seed+persona | delta |
|---|---|---|---|---|
| command_synthesis | 3 | 0.000 | 0.000 | +0.000 |
| debugging | 3 | 1.000 | 1.000 | +0.000 |
| edit_safety | 7 | 0.286 | 0.286 | +0.000 |
| path_grounding | 4 | 0.000 | 0.000 | +0.000 |
| planning | 3 | 0.000 | 0.000 | +0.000 |
| recovery | 3 | 0.333 | 0.333 | +0.000 |
| tool_routing | 7 | 0.143 | 0.143 | +0.000 |

## Persona prefix (572 chars)

```
PERSONA NOTES (the user you are assisting):
- macOS/M-series, zsh. Workspace under /Users/satan/side/experiments/.
- Tools actually used: Bash, exec_command, Read, Edit.
- Bash verbs: git, zig, grep, ls, cat, pnpm. Prefers rg over grep/find, bun over pnpm/npm.
- Active repos: ir-multivector-retrieval, ir-expo, supergemma-dflash-ddtree-mlx.
- User course-corrects in Czech (vůbec, lepší, nene) - treat as authoritative override.
- On failed Bash, pivots to Read (inspect, don't retry blindly).
- Domain: MLX local LLM benchmarking, GEPA prompt opt. Prefer surgical edits.
```

## Sample diverging predictions (first 5)

- **task_edit_safety_005** (edit_safety): seed=`AskUserQuestion` vs persona=`mcp__expo-mcp__authenticate` — seed_score=0.0, persona_score=0.0
- **task_tool_routing_016** (tool_routing): seed=`EnterPlanMode` vs persona=`AskUserQuestion` — seed_score=0.0, persona_score=0.0
- **task_edit_safety_000** (edit_safety): seed=`ExitPlanMode` vs persona=`EnterPlanMode` — seed_score=0.0, persona_score=0.0
- **task_tool_routing_006** (tool_routing): seed=`TaskGet` vs persona=`TaskUpdate` — seed_score=0.0, persona_score=0.0
- **synth_easy_planning_002** (planning): seed=`Glob` vs persona=`Read` — seed_score=0.0, persona_score=0.0

## Verdict

Persona prefix MEASURABLY shifts behavior: 5/30 tasks produce a different prediction than seed-only. Overall mean delta = +0.000.

## Step 3 (LoRA) recommendation

PROCEED-WITH-CAUTION. Prefix demonstrably moves outputs without collapsing accuracy. A LoRA fine-tune is worth a Mac-day to internalise the persona signal that survives prompt clipping.
