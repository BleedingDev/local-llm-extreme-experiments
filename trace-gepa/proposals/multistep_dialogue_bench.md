# Multi-Step Dialogue Benchmark (Round-5 Member #R)

## TLDR
- Single-step action selection scores one decision; real coding sessions are 3-30 turns. We under-measure recovery, plan-coherence, and error-handling because every task resets context.
- Replay top-50 high-success sessions from `dataset_v2.jsonl`: freeze the first user message as the goal, then have the agent emit every subsequent `(tool, input)` while we mock observations from the gold trace.
- Score with a hybrid: trajectory BLEU over tool sequences + structural verifier (final files, key bash commands present) + LM-judge on plan coherence; tier lives at `tasks/multi-step/v1/`.
- Effort ~1 engineer-week. ROI: separates "right next move" agents from "reaches the goal" agents, which is what we actually ship.

## Hypothesis
Single-step benches reward locally-correct picks but cannot detect agents that loop, lose context after a tool error, or skip verification steps. A replay-based multi-step bench exposes these failure modes because the agent must keep producing coherent actions from a frozen goal across a long horizon.

## Concrete Design
**Source.** `trace-gepa/data/dataset_v2.jsonl` (mirrored from `supergemma-dflash-ddtree-mlx`). Filter:
- no `user_corrected` flag anywhere in the session,
- `ai_title` present (proxy for natural completion),
- `>= 10` agent turns,
- rank by composite success score, take top 50.

**Task shape.** Per session emit JSON:
```
{ "task_id": "ms-v1-<hash>",
  "initial_user_request": <first user message>,
  "gold_trajectory": [ {"action": {tool, input}, "observation": <mocked tool result>}, ... ],
  "terminal_signal": "ai_title" | "user_confirm",
  "key_artifacts": {"files_written": [...], "bash_commands_substr": [...]} }
```

**Eval driver** (`tasks/multi-step/v1/runner.py`):
1. Seed agent with `initial_user_request` only.
2. At each turn, agent proposes `(tool, input)`. Lookup the gold step at the same index; mock the observation from `gold_trajectory[i].observation` regardless of input (rationale: we grade trajectory shape, not real exec). If tool name diverges, still return the gold observation but mark turn as miss.
3. Stop when: (a) agent emits the gold final tool + matches `key_artifacts`, (b) `max_turns = 1.5 * len(gold)` exceeded, (c) `N=3` consecutive wrong-tool picks.

**Score.**
- Turn-level: tool-name accuracy, arg-key Jaccard.
- Trajectory: BLEU-2/3 over tool-name sequence vs gold.
- Final-state: structural verifier checks `files_written` substring match in any `Write/Edit` input and `bash_commands_substr` against any `Bash` input.
- Plan coherence: LM-judge (Opus) reads `(goal, agent_trajectory)` and rates 1-5 on coherence + recovery.
- Aggregate: `0.4 * structural + 0.3 * BLEU + 0.2 * turn_acc + 0.1 * judge`.

## Verifier
Structural pass = `files_written subset matched AND >= 80% bash substrings present`. LM-judge runs only on structural pass to save cost; failed runs auto-score 0 on judge.

## Integration
New tier directory `tasks/multi-step/v1/` mirroring `action-selection/v1` layout: `tasks.jsonl`, `runner.py`, `verifier.py`, `README.md`. Reuse harness config in `configs/` with new `suite: multi-step-v1`. Reports rendered into `reports/multi-step/`.

## Effort + ROI + Self-Critique
**Effort.** ~5 days: 1 day filtering + extraction, 2 days runner + mock-observation glue, 1 day verifier + judge, 1 day report wiring.

**ROI.** High. Single-step scores correlate weakly with end-to-end task completion in the wild; this tier directly grades the thing we ship and surfaces compounding-error agents that look fine on action-selection.

**Self-critique.** Mocking observations from the gold trace punishes legitimate alternative trajectories; a divergent-but-correct agent is graded as wrong, so headline scores will systematically under-rate exploratory models until we add a parallel free-rollout verifier track.