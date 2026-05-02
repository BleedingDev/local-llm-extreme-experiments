# trace-gepa state — 2026-05-02 (post-correction)

## Honest headline

After fixing a verifier bug, **the GEPA-optimised prompt is statistically tied with the seed prompt** on a 175-task benchmark (0.291 vs 0.303 overall, opus task LM). Earlier-reported "0.767 vs 0.467 (+64% relative)" lift was an artifact of the broken verifier. See `FINAL_VERDICT.md`.

## What we built

1. **Trace dataset pipeline.** ~30K labelled records across 8 specialised datasets derived from 26 GB of Claude Code + Codex traces (`dataset.jsonl` v1, `dataset_v2.jsonl` v2, `dataset_toolcalling`, `dataset_recovery`, `dataset_corrections`, `dataset_synthetic`, `codex_gpt55_dataset`, `planner_dataset`, plus `finetune_dataset` chat-format).

2. **GEPA reflective optimisation.** 14+ runs in `artifacts/optimized-prompts/`. Adapter at `agent_opt/adapter.py`, reflection prompt at `agent_opt/reflection.py`. Drivers: `optimize.py`, `optimize_codex.py`, `optimize_gpt55.py`, `optimize_planner.py`, `optimize_v2.py`.

3. **Action-selection benchmark.** 175 tasks (105 trace-derived + 70 synthetic) across 7 categories. Sub-bench cuts at `data/benchmarks/`. Path: `data/benchmark_tasks_full.jsonl`.

4. **4-tier verifier suite** (`bench/verifiers/`): regex/JSON-DSL, LM-judge, sandboxed shell exec, weighted composite. 30+ tests.

5. **Three benchmark harnesses** (`bench/run_anthropic.py`, `run_codex.py`, `run_mlx.py`).

6. **Leaderboard infrastructure** (`bench/leaderboard.py`, `aggregate_runs.py`).

7. **BAG production wiring.** Loader is default-on; opt-out via `BAG_DISABLE_OPTIMIZED_PROMPT=1`. Insertion at `src/autonomous-coding-turn.ts:235` (executor step). Top-level `<repo>/artifacts/optimized-prompts -> trace-gepa/artifacts/optimized-prompts` symlink.

## Critical findings

1. **The verifier bug.** Pre-FIX1, `structural_json` tasks (89% of bench) trivially passed any parseable JSON. Post-FIX1 + INV1 the DSL parser handles `pattern_or_command`, `no_repeat` reads `input_excerpt`. 30 verifier tests pass.

2. **xhigh reasoning regresses GPT-5.5 on single-step tool selection.** 0.167 (xhigh) vs 0.200 (high) on the minibench. GPT-5.5 specialist explanation: "For single-step tool selection, extra reasoning increases action entropy."

3. **Three zero-category confounds:**
   - `command_synthesis`: pathological — `available_tools` excludes shell tools.
   - `path_grounding`: pathological — synthetic tasks offer codex tools but verifier requires Anthropic-style.
   - `debugging`: was a DSL gap, FIXED by INV1.

4. **GPT-5.5's highest-leverage suggestion: task-validity preflight.** For every task, mechanically prove that at least one action using `available_tools` can satisfy the verifier. **Not yet implemented.**

5. **Production status.** `latest` -> `bag_exec_opus_v2_run_20260501T233901Z`. With FINAL_VERDICT showing tied performance, rolling back is defensible.

## Headline numbers (n=175, opus task LM, FIXED verifier)

| | seed (236 chars) | optimised (2544 chars) |
|---|---:|---:|
| overall | **0.303** | 0.291 |
| edit_safety | 0.447 | **0.500** |
| recovery | **0.421** | 0.368 |
| debugging | 0.850 | 0.850 |
| tool_routing | 0.179 | 0.179 |
| planning | **0.105** | 0.000 |
| path_grounding | **0.083** | 0.042 |
| command_synthesis | 0.000 | 0.000 |

Excluding the 3 confounded categories: seed 0.422, optimised 0.431 — still tied.

## Open questions

1. Roll back `latest` to seed (or set `BAG_DISABLE_OPTIMIZED_PROMPT=1`)?
2. Build the validity preflight + re-run eval on cleaned bench?
3. Re-run GEPA optimisation with the FIXED verifier (corrected reflective signal)?
4. Move to MLX fine-tune (Option C)?

## Pointers

- Full verdict: `FINAL_VERDICT.md`
- GPT-5.5 critique: `bench/specialist_consultation.md`
- Zero-cat investigation: `bench/zero_cat_investigation.md`
- Comparative positioning: `bench/COMPARATIVE_POSITIONING.md`
- Bench docs: `bench/{BENCHMARK,SCHEMA,HARNESSES,CONTRIBUTING,CHANGELOG}.md`
- Datasets index: `data/DATASETS.md` / `data/datasets_index.json`
- Run aggregator: `scripts/aggregate_runs.py`
