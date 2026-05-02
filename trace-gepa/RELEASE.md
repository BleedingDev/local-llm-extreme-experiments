# trace-gepa v1 — BAG executor prompt release

## Headline

`bag_exec_opus_v2_run_20260501T233901Z` is now the production prompt loaded by BAG's autonomous-coding-turn executor. Held-out test bench (n=60, opus-4-7 task LM): **0.767 vs seed 0.592 (+0.175 absolute, +30% relative).**

## What changed

- `latest` symlink now points at `bag_exec_opus_v2_run_20260501T233901Z` (was `bag_run_20260501T224339Z`, which dropped to rank 4 once we re-evaluated under opus task LM).
- BAG `src/optimized-prompt-loader.ts` is default-on; opt-out via `BAG_DISABLE_OPTIMIZED_PROMPT=1`.
- Loader is wired at `src/autonomous-coding-turn.ts:235` (executor system prompt). Was previously misrouted to the planner — that bug silently degraded real tasks to the hardcoded fallback. Fixed.

## Test bench (n=60 stratified, opus-4-7 task LM)

| rank | candidate | overall | bad | good | uconf |
|---|---|---|---|---|---|
| 1= | **bag_exec_opus** (now `latest`) | **0.767** | 0.800 | 0.745 | 1.000 |
| 1= | codex | 0.767 | 0.800 | 0.745 | 1.000 |
| 3 | v2_big_opus | 0.758 | 0.650 | 0.766 | 1.000 |
| 4 | bag (haiku-trained) | 0.683 | 0.800 | 0.638 | 1.000 |
| 7 | seed (baseline) | 0.592 | 0.650 | 0.553 | 1.000 |

Per-category dominance: optimised candidates (codex, bag_exec_opus, v2_big_opus) hit 0.750 on `hallucinated_path` (n=4) where seed and lower-ranked candidates score 0.250.

## Reproduction

```bash
cd /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx
.venv-gepa/bin/python trace-gepa/agent_opt/optimize.py \
    --budget 400 --train-size 100 --val-size 50 \
    --seed-module bag --run-name bag_exec_opus_v2 \
    --task-model claude-opus-4-7 --reflection-model claude-opus-4-7
```

Wallclock ~21 min. Cost: free (provided tokens).

## Roll back

```bash
cd trace-gepa/artifacts/optimized-prompts
ln -sfn bag_run_20260501T224339Z latest
```

Or set `BAG_DISABLE_OPTIMIZED_PROMPT=1` in the BAG environment to fall back to the in-tree `SYSTEM_PROMPT_DEFAULT` without changing the symlink.

## Pipeline state

- Dataset v1: 3,929 records (cc 948 + codex 2,981).
- Dataset v2: 26,384 records (top 200 sessions ranked by error+correction signals).
- Run dirs: 11 completed at `artifacts/optimized-prompts/`. Aggregator at `scripts/aggregate_runs.py` regenerates `REPORT.md`.
- Eval harnesses: `bench/eval_baseline.py` (2-way), `bench/eval_multi.py` (N-way), `bench/eval_ensemble.py` (judge-LM aggregator).

## Known limits

- Scoring is per-event tool-name match (with tool-family partial credit). Doesn't capture multi-step plan quality.
- `gpt55` track val=0.85 looked best but collapsed to 0.49 on broader test — overfits its narrow GPT-5.5-only subset.
- Hybrid prompt merging regressed; test-time ensemble didn't dominate the best single candidate.
