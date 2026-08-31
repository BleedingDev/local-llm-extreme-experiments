# Wave-3 Status (compiled by Agent X, 2026-05-01)

## Cheap-reflection comparison (Agent S follow-up)

Agent S's `bag_cheap` run completed successfully (run_meta.json appeared during this session). Process PID 51744 finished with `elapsed_seconds=673.88`, `val_score_before=0.6`, `val_score_after=0.6` (delta 0.0). Best candidate is short (866 chars) and looks like a near-no-op edit — the haiku reflector failed to surface any improvement over the seed-bag prompt within budget=300.

3-way held-out test bench (n=60, seed=42, task_model=claude-haiku-4-5, file=`bench/results_cheap_reflection.json`):

| candidate  | source                                              | chars | pass_rate | bad   | good  | uconf |
|------------|-----------------------------------------------------|-------|-----------|-------|-------|-------|
| bag_opus   | bag_run_20260501T224339Z (reflection=opus-4-7)      | 2572  | 0.7667    | 0.800 | 0.745 | 1.000 |
| bag_haiku  | bag_cheap_run_20260501T230832Z (reflection=haiku)   | 866   | 0.5333    | 0.750 | 0.521 | 0.000 |
| seed       | agent_opt.seed:SEED_PROMPT                          | 958   | 0.4667    | 0.500 | 0.489 | 0.000 |

Cost ratio (estimated): Anthropic list price has Opus ~ 8x Haiku per output token. The reflection LM is the dominant cost during GEPA. So bag_cheap should have cost roughly **1/8** the reflection spend of bag_opus. Wallclocks are nearly identical (675s vs 674s), so the saving is purely on $/token, not latency. **Quality penalty: -0.2334 absolute pass_rate** on test (-30% relative). Insufficient instrumentation to give exact $ — runs do not log token usage.

## Verdict on haiku-reflection

**Not sufficient.** With train=80, val=40, budget=300, the haiku reflector produced a candidate that scored *equal to seed on val* (0.6 → 0.6) and *worse than the opus-reflected sibling on held-out test* (0.5333 vs 0.7667). The candidate prompt is also drastically shorter (866 vs 2572 chars), suggesting the haiku reflector mostly proposed minor deletions rather than the targeted rule additions opus produced. Recommendation: keep `claude-opus-4-7` as the default reflection LM. Haiku-reflection might be acceptable for `--budget >= 600` or in agreement-ensembled configurations, but Wave-3 evidence does not support it as a drop-in replacement.

## Wave-3 run inventory

| run_id                          | seed_module | budget | val_after | elapsed_s | reflection_model | status                |
|---------------------------------|-------------|--------|-----------|-----------|------------------|-----------------------|
| run_20260501T215441Z            | default     | 8      | 0.3750    | 22.8      | claude-opus-4-7  | completed (smoke)     |
| run_20260501T215538Z            | default     | 50     | 0.2500    | 71.9      | claude-opus-4-7  | completed             |
| run_20260501T215723Z            | default     | 50     | 0.2500    | 224.6     | claude-opus-4-7  | completed             |
| run_20260501T220148Z            | default     | 200    | 0.5667    | 1123.3    | claude-opus-4-7  | completed             |
| run_20260501T223837Z            | default     | 600    | 0.6200    | 1033.8    | claude-opus-4-7  | completed             |
| v2_run_20260501T224342Z         | v2          | 300    | 0.5625    | 589.4     | claude-opus-4-7  | completed             |
| bag_run_20260501T224339Z        | bag         | 300    | 0.6875    | 675.1     | claude-opus-4-7  | completed             |
| codex_run_20260501T224340Z      | codex       | 300    | 0.7500    | 645.3     | claude-opus-4-7  | completed             |
| hybrid_run_20260501T230908Z     | merge       | n/a    | n/a       | 12.2      | claude-opus-4-7  | completed (merge-only)|
| bag_cheap_run_20260501T230832Z  | bag         | 300    | 0.6000    | 673.9     | claude-haiku-4-5 | completed             |
| bag_xl_run_20260501T230613Z     | bag         | 600    | -         | -         | claude-opus-4-7  | in-flight (PID 46632) |
| v2_big_run_20260501T231552Z     | default     | 600    | -         | -         | claude-opus-4-7  | in-flight (PID 63160) |

(`-` = not yet written.)

## Per-Wave-3-agent attribution (best inference from artefacts; not all agents are tagged in metadata)

| agent | likely deliverable                                          | status              |
|-------|-------------------------------------------------------------|---------------------|
| N     | bag seed track (`agent_opt/seed_bag.py` + bag_run_*)        | delivered           |
| O     | codex seed track (`seed_codex.py` + codex_run_*)            | delivered           |
| P     | v2 dataset/optimize_v2 (`v2_run_*`, `dataset_v2.jsonl`)     | delivered           |
| Q     | hybrid merge (`agent_opt/merge_prompts.py` + hybrid_run_*)  | delivered           |
| R     | aggregator + REPORT.md                                      | delivered           |
| S     | cheap-reflection ablation (`bag_cheap_run_*`)               | delivered late (completed during Agent X session) |
| T     | bag_xl long run (budget=600 bag track)                      | in-flight           |
| U     | v2_big long run (budget=600 v2 track)                       | in-flight           |
| V     | BAG runtime wiring (TS) — owned, not measured here          | not in scope here   |
| W     | (unknown, no obvious artefact)                              | unknown             |
| X     | this status report + cheap-reflection comparison bench      | delivered           |

## Best-overall configuration

On the held-out test split (n=60):

- **`bag_run_20260501T224339Z`** — pass_rate **0.7667**, seed_module=`bag`, budget=300, reflection=`claude-opus-4-7`. Already the canonical winner of `bench/results_wave2_final.json` and the target of the `latest` symlink chain in the BAG runtime.

The codex track has the highest *val_score_after* (0.7500) but on test it lags bag (0.6500 in `results_wave2_final.json`), so val ranking and test ranking disagree — bag is the recommended deployment choice. Pending bag_xl/v2_big completions could displace it.

## Files produced this session

- `trace-gepa/bench/results_cheap_reflection.json` (3-way bench)
- `trace-gepa/WAVE3_STATUS.md` (this file)
