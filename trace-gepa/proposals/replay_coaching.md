# Replay-Coaching Mode

## TLDR
- **Per-decision grading on real sessions**: walk an actual trace event-by-event, pause at every `assistant.tool_use`, ask the candidate model "what next?", grade against the observed action — fine-grained signal that final-state scoring loses.
- **Hybrid eval + training data**: each decision point yields (prefix, candidate_action, gold_action, score) — usable for model comparison AND as SFT/DPO pairs without synthetic generation.
- **Decision divergence map**: aggregate per-event scores into a histogram showing WHERE in a session a model drifts from this user's expected behaviour (early planning? mid-debug tool choice? final commit?).
- **Reproducible via workflow snapshots**: pin to a snapshot from the round-5 sibling proposal so the prefix at each decision point is deterministic across runs.

## Hypothesis
Per-decision grades on real traces capture finer-grained signal than per-task final-state scoring, because (a) a session of 40 tool-uses produces 40 graded examples not 1, (b) grading happens on the user's actual prefix not a synthetic restart, and (c) divergences localize to specific decision contexts ("after a failed test, candidate runs `cat` but user runs `Read`") rather than diffuse end-state diff.

## Design
**Event walk.** For each `assistant.tool_use` event in trace `T`:
1. Build prefix = all events before this one (user msgs, prior tool_uses, tool_results) rendered as Anthropic message list.
2. Append system prompt + tool catalog (recovered from snapshot).
3. Sample candidate model with `tool_choice=any` until it emits a `tool_use` block — call this `candidate_action`.
4. Score against `observed_action` using TraceAdapter family:
   - Exact tool name match: 1.0
   - Same family (Read/Grep/Glob = "search"; Edit/Write = "mutate"; Bash variants by command verb): 0.5
   - Different family: 0.0
   - Argument similarity bonus: +0.2 if file path overlaps, capped at 1.0.
5. Emit `{event_id, prefix_hash, candidate, observed, score, family_match}`.

**Aggregation.** Per-session mean + per-family mean + decision-divergence map (heatmap: x=event_index, y=score, colored by tool family).

**Modes.**
- `eval`: run candidate fleet (Opus, Haiku, local-fine-tuned) on shared trace set, output leaderboard.
- `mine`: filter to score>=0.8 events, dump as SFT pairs `(prefix, observed_action)`.
- `explain`: aggregate divergences across N sessions, surface top-K decision contexts where this user systematically diverges from generic Opus.

## Effort + ROI
~3 days: trace walker (1d), candidate sampler with snapshot replay (1d), scoring + viz (1d). ROI high — same trace yields 40x more graded examples than final-state eval, and the SFT pairs are free.

## Self-critique
Tool-family partial credit is a coarse proxy — two different `Read` calls on different files are scored identically, hiding semantic divergence that matters for fine-tuning.
