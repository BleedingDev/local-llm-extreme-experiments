# Local Evidence Inventory

Generated for graph `local-evidence-flywheel-v1`.

## Executive Summary

This repository contains valuable local evidence for BleedingAgent self-improvement. It should not be treated as disposable dirty-worktree noise. The strongest evidence sources are trace datasets, sanitised mirrors, recovery/counterfactual corpora, benchmark/job outputs, ACP replay corpora, optimizer candidates, and prompt-optimization artifacts.

The immediate policy is: keep primary evidence, prefer sanitised data for optimizer/sharing, treat derived indexes as rebuildable but useful, and only clean dependency/build/runtime caches without explicit evidence approval.

## High-Value Evidence Sources

| Source | Size / Count | Value | Retention |
| --- | ---: | --- | --- |
| `trace-gepa/data/dataset_v2.jsonl` | 26,384 rows, 202 MB | largest general action/tool dataset | raw local primary evidence |
| `trace-gepa/data/sanitised/dataset_v2.jsonl` | 26,384 rows, 205 MB | canonical shareable/training mirror | canonical optimizer candidate input |
| `trace-gepa/data/cc_dataset_v2_new.jsonl` | 22,340 rows, 176 MB | Claude Code style action evidence | raw local primary evidence |
| `trace-gepa/data/codex_gpt55_dataset.jsonl` | 6,820 rows, 34 MB | GPT-5.5/Codex-specific action evidence | model-specific primary evidence |
| `trace-gepa/data/dataset_recovery.jsonl` | 4,055 rows | failure to recovery transitions | canonical recovery evidence after sanitised mirror |
| `trace-gepa/data/counterfactuals.jsonl` | 431 rows | tool/action alternatives, mean confidence 0.76 | keep |
| `trace-gepa/data/benchmark_tasks_full.jsonl` | 175 rows | complete task benchmark set | canonical benchmark definition |
| `trace-gepa/data/benchmarks/**` | 124 indexed sub-benchmark rows plus views | minibench/stress/category/difficulty views | keep as benchmark slices |
| `.bag/replay-corpus` | 13 real ACP attempts plus 50 adapter replay cases | ACP regression/failure evidence | keep canonical visible run and adapter export |
| `bench/jobs` | 77 job dirs, 541 result JSONs, 415 ACP summaries | observed benchmark behavior | high-risk keep |
| `bench/.bag/optimizer` | 85 dataset records, 12 candidates, 26 failure clusters | optimizer input/state | keep |
| `trace-gepa/artifacts/optimized-prompts` | 18 run dirs, 2 symlinks | prompt optimization history | keep metadata/candidates/logs |
| `bench/aider_polyglot/results` | 5-problem smoke, 4/5 pass | edit-strategy evidence | keep compact summaries/traces |

## Canonical Recommendations

- General supervised/action corpus: `trace-gepa/data/sanitised/dataset_v2.jsonl` plus `trace-gepa/data/splits_v2.json`.
- Recovery corpus: `trace-gepa/data/sanitised/dataset_recovery.jsonl` plus `trace-gepa/data/splits_recovery.json`.
- Benchmark corpus: `trace-gepa/data/sanitised/benchmark_tasks_full.jsonl`.
- Cheap regression benchmark: `trace-gepa/data/benchmarks/minibench.jsonl`.
- Hard regression benchmark: `trace-gepa/data/benchmarks/stress.jsonl`.
- Real ACP corpus baseline: `.bag/replay-corpus/index.jsonl` plus `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/**`.
- Source-adapter replay baseline: `.bag/replay-corpus/source-adapters/adapter-replay-export/**`.
- Terminal-bench observed behavior: `bench/jobs/**/{result.json,bag-acp-summary.json,audit.jsonl,exception.txt}`.
- Optimizer local state: `bench/.bag/optimizer/*`.

Do not use `latest` symlink names alone as truth. For example, `trace-gepa/artifacts/optimized-prompts/latest` currently points to a zero-delta run, while `latest_codex` points to a +0.175 validation delta run. Candidate selection must use lineage and scorecards, not names.

## Dataset Details

### Action And Tool Datasets

Common schema:

`id`, `src`, `src_path`, `src_event_idx`, `context`, `observed_action`, `label`, `failure_category`, `ideal_action_hint`, `next_user_message`.

`dataset_toolcalling.jsonl` additionally has `quality_score`.

Observed distributions:

- `dataset_v2.jsonl`: `good` 22,303; `bad` 3,421; `user_confirmed` 593; `user_corrected` 67.
- `dataset_v2.jsonl` failure categories: `bash_exit_nonzero` 2,725; `bash_timeout_141` 243; `hallucinated_path` 237; `retry_loop` 125; `cancelled_parallel_batch` 100; `user_correction` 67.
- `dataset_toolcalling.jsonl`: `good` 3,553; `bad` 439; `user_confirmed` 49; `user_corrected` 4.
- `codex_gpt55_dataset.jsonl`: `good` 5,389; `user_confirmed` 970; `bad` 357; `user_corrected` 104.

Canonicality:

- `dataset_v2.jsonl` contains all unique IDs from `dataset.jsonl`, all IDs from `cc_dataset_v2_new.jsonl`, and all IDs from `codex_dataset_v2_new.jsonl`.
- It does not subsume `codex_gpt55_dataset.jsonl` or `dataset_toolcalling.jsonl` by checked ID relationship.
- Older duplicate-risk sets: `cc_dataset.jsonl`, `dataset.jsonl`, and `planner_dataset.jsonl` have duplicate IDs.

### Recovery And Counterfactuals

`dataset_recovery.jsonl` has 4,055 pairs:

- `strong`: 3,520
- `weak`: 506
- `transient`: 29

Top recovery/failure signals:

- `bash_exit_nonzero`: 2,744 first failures.
- `hallucinated_path`: 455.
- `bash_timeout_141`: 251.
- `retry_loop`: 142.
- `cancelled_parallel_batch`: 105.

`counterfactuals.jsonl` has 431 valid annotations:

- `tool_swap`: 165
- `input_fix`: 133
- `verify_first`: 120
- `abort`: 9
- `decompose`: 4

## Benchmarks And Runs

`trace-gepa/data/benchmark_tasks_full.jsonl` is the full 175-task benchmark: 105 base plus 70 synthetic. `trace-gepa/data/benchmarks/**` are subsets or views.

`bench/jobs` contains:

- 77 top-level job dirs.
- 504 immediate task dirs.
- 541 `result.json` files.
- 415 `bag-acp-summary.json` files.
- 32 `exception.txt` files.
- 8 `audit.jsonl` files with 63 records.

Observed evaluator summary:

- `bag__claude-opus-4-7__terminal-bench-sample`: 52 runs, 424 trials, 31 errors, average mean metric 0.5731.
- `claude-code__claude-opus-4-7__terminal-bench-sample`: 10 trials, 0 errors, mean 0.9.
- `bag__claude-opus-4-7__cais/swebenchpro`: 8 trials, 2 errors, mean 0.4.

Task-level reward across parsed task results: 447 results, mean 0.6398. Strong local tasks include `regex-log` and `log-summary-date-ranges`; weak tasks include `qemu-alpine-ssh`, `qemu-startup`, and TB2 smoke tasks.

ACP summary signals from `bench/jobs`:

- `terminalCreate`: 12,983
- `terminalOutput`: 12,947
- `terminalExit`: 12,890
- `fsWrite`: 26
- `fsRead`: 40
- `permission`: 0

This is a rich observed-behavior corpus for tool routing, terminal behavior, edit behavior, and failure clustering.

## ACP Corpus

`.bag/replay-corpus` has 196 files total under `.bag`, with `.bag/replay-corpus` at about 1.1 MB.

Real ACP task results across smoke and visible runs:

- total: 13
- passed: 0
- failed: 11
- error: 1
- cancelled: 1

Canonical visible run:

- `.bag/replay-corpus/index.jsonl`: 9 records.
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/**`.
- Visible score: 0 passed, 8 failed, 1 cancelled, 0 changed files, 0 write tools.

Valuable signals:

- Early setup issue: `spawn tsx ENOENT`.
- Repeated no-write `end_turn` failures.
- Read-only behavior despite coding tasks.

This is negative quality evidence, but it is high-value because it identifies the current ACP coding bottleneck.

## Optimizer Artifacts

`bench/.bag/optimizer`:

- `dataset.jsonl`: 85 records, mean reward 0.4353.
- `candidates.json`: 12 candidates, 12/12 validation passed, 48 failure runs, 32 feedback records.
- `failure-clusters.json`: 143 failures, 26 clusters.
- `gepa-readiness-report.json`: candidate generation ready; auto-promotion blocked by `post-promotion-monitor-window`.

Top failure clusters include:

- missing `/app/summary.csv`: 25
- missing `/app/report.jsonl`: 22
- SSH exit 255: 19
- HTTP 000 webserver failure: 14
- assertion false: 10

`trace-gepa/artifacts/optimized-prompts`:

- 18 physical run dirs, 2 symlinks.
- 16 `best_candidate.json`.
- 16 `run_meta.json`.
- 17 `log.txt`.

Notable validation deltas:

- `v2_big_run_20260501T231552Z`: +0.2312.
- `v2_big_opus_run_20260501T234035Z`: +0.1833.
- `codex_run_20260501T224340Z`: +0.175.
- `bag_xl_run_20260501T230613Z`: +0.11.
- `bag_postfix_verifier_run_20260502T073424Z`: 0.

Per-iteration `gepa_state/` dirs were already removed according to `_CLEANUP_LOG.md`; final candidates/logs/metadata remain, but detailed local replay state is gone unless rerun.

## Derived Artifacts

RAG indexes:

- `rag_index`: 30,313 metadata rows, 71 MB.
- `rag_index_filtered`: 8,264 metadata rows, 11 MB.
- `rag_index_v2`: 8,264 metadata rows, 13 MB.

Anomaly models:

- `anomaly_iforest.pkl`: 43 MB.
- `anomaly_lof.pkl`: 22 MB.

These are derived but operationally useful. They should not be casually deleted; if removed, preserve rebuild commands and source data. Pickle files also carry security/privacy risk and should not be loaded from untrusted sources.

## Documentation Evidence

Current canonical docs:

- `docs/bleeding-agent.md`
- `docs/bleeding-agent-evidence-flywheel-release-proof.md`
- `docs/bleeding-agent-quality-execution-release-proof.md`
- `docs/bleeding-agent-release-readiness.md`
- `docs/bleeding-agent-packaging-inventory.md`
- `docs/bleeding-agent-ownership-manifest.md`
- `docs/bleeding-agent-green-gates.md`
- `docs/bleeding-agent-experiment-boundaries.md`
- `docs/bleeding-agent-acp-interop-report.md`
- `docs/bleeding-agent-edit-evidence-audit.md`
- `docs/bleeding-agent-gepa-operations.md`
- `docs/bleeding-agent-real-replay-dataset.md`
- `docs/bleeding-agent-provider-model-profile-audit.md`
- `docs/bleeding-agent-operator-runbook.md`
- `trace-gepa/STATE.md`
- `trace-gepa/FINAL_VERDICT.md`
- `trace-gepa/DELIVERABLES.md`
- `trace-gepa/COMMIT_MANIFEST.md`
- `trace-gepa/README.md`

Historical but useful docs include MLX/Qwen/DFlash reports, BAG benchmark forensics, trace mining strategy docs, edit-strategy research docs, and Trace-GEPA decision trails.

Likely obsolete or duplicative:

- `trace-gepa/RELEASE.md`: superseded by `STATE.md`, `FINAL_VERDICT.md`, and `DELIVERABLES.md`.
- `docs/bag-edit-strategy-study.md`: scaffold with `TO FILL` / `TBD`.
- `docs/bag-harness-ablation-study.md`: useful plan, but runs not executed.
- `docs/qwen-optimization-report.md`: explicitly historical and superseded by `docs/qwen-tuning-report.md`.
- `docs/bag-gepa-ops-readiness.md`: superseded by BleedingAgent GEPA docs and release proofs.

## Dirty Worktree Evidence

The source tree has meaningful but mixed changes:

- Staged `.mcp.json` to `.mcp.example.json` hygiene is packaging/config.
- Tracked MLX benchmark track changes affect benchmark scripts/docs and vendor patching.
- Tracked BleedingAgent core changes in `src/autonomous-coding-turn.ts` and `src/optimized-prompt-loader.ts` depend on untracked modules.
- Tracked Trace-GEPA integration changes depend on untracked `src/source-adapters/failures.ts`.
- Root untracked `src`, `tests`, `scripts`, `package.json`, `tsconfig.json`, and `rspack.config.ts` form the active BleedingAgent TypeScript workspace.
- `bench/bag-runtime` is a runtime staging copy/snapshot. It overlaps heavily with root `src` but differs; do not commit it as source-of-truth without an explicit fixture decision.

Known hygiene issue:

- `git diff --check` fails on trailing whitespace in `patches/dflash-triattention-mlx.patch`.

## Privacy And Safety

Raw `trace-gepa/data` contains many absolute `/Users/satan` paths and a few OpenAI-key-shaped strings in raw `codex_gpt55_dataset.jsonl`. Sanitised mirrors have zero `/Users/satan` matches and zero OpenAI-key-shaped matches in the same scan. Email-shaped strings remain numerous in both raw and sanitised data and need semantic review.

`.bag` manifests, telemetry, transcripts, and fixtures include absolute paths, env var names, model/provider metadata, registry IDs, token counts, cwd, and optimizer policy/profile IDs.

RAG metadata stores request excerpts and local paths. Pickle anomaly models are binary and should be treated as sensitive, local-only derived artifacts.

## Retention Policy

Keep / local primary evidence:

- `trace-gepa/data/*.jsonl`
- `trace-gepa/data/sanitised/*.jsonl`
- `.bag/replay-corpus/**`
- `.bag/telemetry/**` until an aggregation policy exists
- `bench/jobs/**/{result.json,config.json,trial.log,bag-acp-summary.json,audit.jsonl,exception.txt,reward.txt,test-stdout.txt}`
- `bench/.bag/optimizer/**`
- `bench/aider_polyglot/results/**` compact summaries and selected traces

Derived but useful:

- `trace-gepa/artifacts/rag_index*`
- `trace-gepa/artifacts/anomaly_*.pkl`
- `trace-gepa/artifacts/real_smoke/**`

Rebuildable and safe to remove only if space pressure requires it:

- `node_modules`
- `bench/bag-runtime/node_modules`
- `dist`
- `.venv`
- `__pycache__`

Rebuildable but approval first:

- `.venv-gepa`: capture `pip freeze` first; no exact lockfile found.
- `bench/.venv`: capture `pip freeze` first; likely benchmark-specific.
- `bench/vendor`: preserve current clone remotes/commits first.

Needs explicit decision:

- `bench/bag-runtime`: snapshot/fixture/staging copy, not plainly disposable.
- `.codex`: active local plan/orchestration evidence.
- `.claude`: local skills/worktree evidence.

## Unknowns And Next Checks

- Validate semantic correctness of labels and benchmark expected answers.
- Do full raw/sanitised semantic parity, not only row-count parity.
- Review sanitised proper noun audit manually.
- Determine whether `bench/bag-runtime` is a fixture, stale copy, or active runtime pack.
- Define a commit-safe metadata index that references high-value local evidence without storing sensitive raw rows.
- Replace `latest` pointer assumptions with explicit lineage and scorecard-based candidate selection.
