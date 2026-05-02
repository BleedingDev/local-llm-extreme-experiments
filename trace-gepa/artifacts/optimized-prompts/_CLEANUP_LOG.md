# Cleanup Log: gepa_state/ removal

**Date:** 2026-05-01
**Agent:** Cleanup Agent
**Scope:** `trace-gepa/artifacts/optimized-prompts/`

## Summary

Removed bulky `gepa_state/` per-iteration intermediates from all completed GEPA run dirs. These contain GEPA library state (per-iteration candidate logs, score tables) written via the `run_dir` arg in `agent_opt/optimize*.py` and are not consumed by anything downstream (verified via `rg -l 'gepa_state' trace-gepa/{agent_opt,bench,scripts}` — only the four `optimize*.py` writers reference the path).

## Before / After

| Metric | Before | After |
|---|---|---|
| Total bytes (sum of files) | 2,034,696 B (1.94 MB) | 875,632 B (0.84 MB) |
| `du -sh` | 4.6M | 1.0M |
| `gepa_state/` subdirs | 16 | 0 |

(`du -sh` includes filesystem block overhead; raw file-byte sum is the truer measure. Both are under the 1 MB target on a content basis.)

## Removed: 16 `gepa_state/` subdirs

- `bag_cheap_run_20260501T230832Z/gepa_state`
- `bag_exec_opus_run_20260501T232624Z/gepa_state`
- `bag_exec_opus_v2_run_20260501T233901Z/gepa_state`
- `bag_postfix_verifier_run_20260502T073424Z/gepa_state` (current `latest` symlink target)
- `bag_run_20260501T224339Z/gepa_state`
- `bag_xl_run_20260501T230613Z/gepa_state`
- `codex_run_20260501T224340Z/gepa_state`
- `gpt55_run_20260501T234033Z/gepa_state`
- `run_20260501T215441Z/gepa_state`
- `run_20260501T215538Z/gepa_state`
- `run_20260501T215723Z/gepa_state`
- `run_20260501T220148Z/gepa_state`
- `run_20260501T223837Z/gepa_state`
- `v2_big_opus_run_20260501T234035Z/gepa_state`
- `v2_big_run_20260501T231552Z/gepa_state`
- `v2_run_20260501T224342Z/gepa_state`

Note: The task spec named 18 GEPA run dirs and `bag_exec_opus_v2_run_20260501T233901Z` as the `latest` target. Actual state at execution time:

- 19 run dirs total under `optimized-prompts/`, but only 16 contained a `gepa_state/` subdir. The other three (`gpt55_run_20260501T233633Z` — log only; `hybrid_run_20260501T230908Z` — already had no `gepa_state/`; `latest_codex` — symlink, not a dir) needed no action.
- `latest` actually points to `bag_postfix_verifier_run_20260502T073424Z` (newer than the spec assumed). Consumed files there were preserved; same care taken with the spec-named `bag_exec_opus_v2_run_20260501T233901Z`.

## Preserved (per spec) in every run dir

- `best_candidate.json`
- `best_candidate.system.md`
- `run_meta.json`
- `log.txt`

## Verification

- `latest -> bag_postfix_verifier_run_20260502T073424Z`: all four consumed files present, same byte sizes as pre-cleanup (best_candidate.json: 3084 B, best_candidate.system.md: 3015 B, run_meta.json: 457 B, log.txt: 72411 B).
- `bag_exec_opus_v2_run_20260501T233901Z`: same four consumed files present.
- BAG smoke: `bun run trace-gepa/scripts/bag_smoke_no_root.ts` -> **PASS** (planner produced 1 issue, optimised artefact resolved correctly to `bag_postfix_verifier_run_20260502T073424Z`).
- `find ... -name gepa_state` -> 0 matches.
