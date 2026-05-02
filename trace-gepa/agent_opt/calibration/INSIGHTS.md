# Calibration Scorecard - Insights

Source: 175-task action-selection bench (`local-coding-benchmark`).
Models: Opus (`claude-opus-4-7`, 30.3% bench / 24.6% strict) vs GPT-5.5 high-reasoning (24.0% bench / 16.6% strict).
"Strict" accuracy requires `predicted_tool == gold.primary_action.tool_name AND check_score > 0`.

## Top 5 over-picked tools (model picks > gold expected)

**Opus**
- `Read`: 54 picked / 33 expected, +12.0pp (highest-magnitude over-pick of any frequent tool).
- `wait_agent`: 10 / 0, +5.7pp (not in canonical tool set; hallucinated).
- `EnterPlanMode`: 9 / 0, +5.1pp (over-uses planning meta-tool).
- `TaskGet`: 7 / 0, +4.0pp.
- `Skill`: 6 / 0, +3.4pp.

**GPT-5.5**
- `EnterWorktree`: 17 / 0, +9.7pp (large hallucinated preference; Opus over-picks the same tool only +2.9pp).
- `Read`: 44 / 33, +6.3pp.
- `wait_agent`: 9 / 0, +5.1pp.
- `EnterPlanMode`: 8 / 0, +4.6pp.
- `Skill`: 6 / 0, +3.4pp.

## Top 5 under-picked tools (model picks < gold expected)

**Opus**
- `Bash`: 31 / 74, -24.6pp - the single biggest miscalibration in the scorecard. Recall on Bash is 0.16 (only 12 of 74 Bash-correct gold rows recovered).
- `TaskCreate`: 4 / 19, -8.6pp (recall 0.11).
- `Edit`: 10 / 18, -4.6pp.
- `Agent`: 0 / 4, -2.3pp (never selected).
- `Glob`: 4 / 8, -2.3pp.

**GPT-5.5**
- `Bash`: 25 / 74, -28.0pp (worse than Opus; recall 0.09, F1 0.14).
- `TaskCreate`: 8 / 19, -6.3pp (recall 0.0 - never gets it right).
- `Edit`: 13 / 18, -2.9pp.
- `Agent`: 0 / 4, -2.3pp.
- `Glob`: 4 / 8, -2.3pp.

## Highest-precision tools (predicted_count >= 3)

**Opus**: `WebFetch` (P=0.67), `Edit` (0.50), `TaskCreate` (0.50).
**GPT-5.5**: `Glob` (P=0.75), `WebFetch` (0.67), `Edit` (0.31).

## Cross-model surprises (F1 gap >= 20pp on tools with reasonable evidence)

- `Glob`: GPT-5.5 F1 = 0.50 vs Opus F1 = 0.17 (-33pp for Opus). On the 8 Glob-gold tasks, GPT-5.5 is far better calibrated; Opus tends to substitute `Read` (Opus confused-with: `Read(7)`).
- `exec_command`: Opus F1 = 0.50 vs GPT-5.5 F1 = 0.22 (+28pp for Opus). Tiny N (expected_count 2) so noisy, but consistent direction.
- `EnterWorktree` over-pick gap: GPT-5.5 +9.7pp vs Opus +2.9pp - GPT-5.5 hallucinates EnterWorktree about 3x more often.

The dominant shared failure mode is the same: both models drastically under-pick `Bash` (Opus -24.6pp, GPT-5.5 -28pp). When they miss `Bash` they substitute `Read` (Opus 12, GPT-5.5 13), `wait_agent` (Opus 10, GPT-5.5 8), and increasingly `exec_command`. The 30.3 / 24.0 aggregate pass rates are largely a single tool's miscalibration: Bash misses alone account for roughly 35-40 lost tasks (74 expected x missed-recall) per model. `TaskCreate` is the secondary big miss; both models substitute `Bash` or `Read` for it.

## What to do about this (concrete suggestions)

The data argues for three targeted prompt edits and one routing rule rather than a broad re-tune. (1) **Bash bias correction**: add a positive few-shot block in the action-selection prompt that contains 4-5 short examples where the right answer is `Bash` for build/test/exec phrasing, and add a negative rule "do not pick `Read` to inspect a file when the user is asking to *run* something" - this is the single highest-leverage intervention because `Bash` accounts for ~42% of gold rows and both models miss it ~85% of the time. (2) **Strip hallucinated tool names from the candidate set**: `wait_agent`, `EnterPlanMode`, `EnterWorktree`, `TaskGet`, `Skill` together account for 30+ over-picks across both models and zero gold support; either remove them from the available-tools list when not actually available, or add an explicit "never pick X for task type Y" rule. GPT-5.5's `EnterWorktree` habit (+9.7pp) is so model-specific that a single negative rule would recover ~17 picks. (3) **TaskCreate confusion**: gold confusion shows TaskCreate misses go to `Bash` and `Read`; add one few-shot showing a multi-step delegation phrasing -> `TaskCreate`. (4) **Routing**: feed `cost_pareto_router` with the per-tool F1 matrix, not aggregate accuracy - on `Glob`-gold tasks, prefer GPT-5.5 (F1 0.50 vs 0.17); on most other gold-Bash and gold-Read tasks, prefer Opus. The scorecard now produces this matrix automatically, so the router gate becomes a lookup, not an estimate.
