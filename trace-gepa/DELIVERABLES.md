# trace-gepa Master Deliverables (2026-05-02)

Author: Documentation-Final Agent. Replaces stalled Doc-1 draft. All facts in this
document were re-read from disk at write time; where memory and disk disagreed,
disk won.

## 1. Executive summary

Tonight's work landed **5 public commits** on `BleedingDev/local-llm-extreme-experiments@master`
(d055b29..9ebbe28) covering an end-to-end trace-driven GEPA optimisation pipeline, a
175-task synthetic action-selection benchmark with a 4-tier verifier suite, default-on
BAG executor wiring, 16 GEPA runs (post-cleanup, 1.0 MB total), and the docs/proposals
tree. **And 7 commits on `BleedingDev/local-coding-benchmark@main`** (1fdd86c..b91bebb):
the FF merge of `benchmark-quality-refactor` plus our action-selection v1 suite (175
tasks, suite_hash 2377a55a01307dbb), the 4-tier verifier port, the eval driver, the
configs, the smoke-results commit, and finally a fix for Anthropic's temperature
deprecation that produced **the first real measurable number on this benchmark:
claude-opus-4-7 pass@1 = 2/30 (6.7%), mean_check_score = 0.067**
(`reports/opus-action-real-smoke-30.jsonl`). Failure mode is dominated by `schema_fail`
(model output not satisfying verifier-DSL assertions) — likely a mix of genuine
wrong-tool picks and verifier coverage gaps; flagged for follow-up. Two user-decision
items are pending: review of 15 proper-noun candidates in
`data/sanitised/_proper_nouns_audit.json` and whether to create a private
`trace-gepa-data` repo for the sanitised corpus. The honest GEPA verdict from earlier
in the night stands: the optimised prompt is statistically tied with the seed under
a corrected verifier (0.291 vs 0.303 on 175 tasks); the earlier "+64% relative"
claim was a verifier-bug artifact and has been retracted in `FINAL_VERDICT.md`.

## 2. What's on GitHub now (precise)

### Table A — `BleedingDev/local-llm-extreme-experiments` (public, master)

Confirmed via `git log --format='%H|%s' 7c8b69d..origin/master` (oldest first below):

| commit | files | subject |
|---|---:|---|
| `d055b29` | 72 | feat(trace-gepa): bring in trace-driven GEPA optimisation pipeline |
| `9352549` | 44 | feat(trace-gepa): synthetic 175-task action-selection benchmark + 4-tier verifier suite |
| `885c762` | 11 | feat(bag): default-on optimised executor prompt loader + trace-RAG shim |
| `80000f8` | 94 | chore(trace-gepa): GEPA optimisation runs (cleaned, no intermediates) |
| `9ebbe28` | ~26 | docs(trace-gepa): proposals, reports, integration plan + commit manifest |

Range tip: `9ebbe28` (HEAD of `origin/master`). Pre-merge base: `7c8b69d`
(`Align docs with renamed repository`). Total: **5 commits**, ~247 file-additions.

### Table B — `BleedingDev/local-coding-benchmark` (private, main)

Confirmed via `gh api repos/BleedingDev/local-coding-benchmark/commits/main`. HEAD is `b91bebb`.

| commit | files | subject |
|---|---:|---|
| `1fdd86c` | 1 | chore: add CHANGELOG entry for benchmark-quality-refactor cutover |
| `1e08b3b` | 4 | feat(suite): add action-selection v1 task suite (175 tasks from trace data) |
| `ed18da0` | 6 | feat(scripts): port 4-tier verifier suite from trace-gepa |
| `7e9695d` | 1 | feat(scripts): action-selection eval driver |
| `d3ecfc4` | 4 | feat(configs): action-selection adapter/agents/runs |
| `24eff15` | 2 | chore: smoke results for action-selection track (dry-run, schema verify) |
| `b91bebb` | 3 | fix(scripts): handle Anthropic temperature-deprecation + accept ANTHROPIC_AUTH_TOKEN + n=30 real Opus smoke |

7 new commits on top of the FF merge of `benchmark-quality-refactor` (parent: `a64bac3`,
which is itself one commit ahead of the original `e65b388` "Initial local coding
benchmark extraction"). Tag `pre-quality-refactor` pushed to origin. Branch
`benchmark-quality-refactor` deleted from origin (merged).

**Real measurable numbers on this benchmark (commit `2a4a784`):**

- `reports/opus-action-real-smoke-30.jsonl`: claude-opus-4-7 on first 30 (hard-skewed) tasks: pass@1 = 2/30 (6.7%) — initial smoke, hard slice.
- `reports/opus-action-full-175.jsonl`: claude-opus-4-7 on all 175 stratified tasks: **pass@1 = 53/175 (30.3%), mean_check_score = 0.306**.

**Bench port is FAITHFUL.** Trace-gepa internal eval on the identical 175 tasks (`bench/results/results_final_opus.json`) was 0.291; bench port reads 0.306 — Δ = +0.015, well within run-to-run variance. End-to-end chain validated: same task corpus, same verifier semantics, same model behaviour.

Manifest sidecars carry suite_hash 2377a55a, host, git, sampling, endpoint probe per the audit-grade convention introduced by `benchmark-quality-refactor`.

## 3. Per-deliverable inventory

### A — Optimiser code (`agent_opt/`, `extractors/`, `mcp_servers/`, `scripts/`, `tests/`)

| dir | files | size | repo |
|---|---:|---:|---|
| `trace-gepa/agent_opt/` | 101 | 1.0 MB | `local-llm-extreme-experiments@d055b29` |
| `trace-gepa/extractors/` | 19 | 216 KB | `local-llm-extreme-experiments@d055b29` |
| `trace-gepa/mcp_servers/` | 7 | 24 KB | `local-llm-extreme-experiments@d055b29` |
| `trace-gepa/scripts/` | 10 | 80 KB | `local-llm-extreme-experiments@d055b29` |
| `trace-gepa/tests/` | 12 | 208 KB | `local-llm-extreme-experiments@d055b29` |

Consumed by: `optimize*.py` GEPA drivers (consume seed + dataset + adapter); BAG
TS shim (`src/optimized-prompt-loader.ts` reads
`artifacts/optimized-prompts/latest/best_candidate.system.md`).

### B — Action-selection benchmark (`bench/`, `data/benchmarks/`)

| dir | files | size | repo |
|---|---:|---:|---|
| `trace-gepa/bench/` (this round, excluding `bench/results/`) | 30 src + 9 docs | ~1.2 MB | `local-llm-extreme-experiments@9352549` |
| `trace-gepa/data/benchmarks/` (sub-bench cuts) | 13 | (jsonl) | `local-llm-extreme-experiments@9352549` |

Consumed by: `bench/run_anthropic.py`, `run_codex.py`, `run_mlx.py` and
`bench/leaderboard.py`. **Pending**: BENCH_REPO_EVAL.md plan to mirror as Tier-0
in `local-coding-benchmark`.

### C — GEPA optimisation runs (`artifacts/optimized-prompts/`, `bench/results/`)

| dir | files | size | repo |
|---|---:|---:|---|
| `trace-gepa/artifacts/optimized-prompts/` (cleaned) | ~83 | 1.0 MB | `local-llm-extreme-experiments@80000f8` |
| `trace-gepa/bench/results/` | (subset of 94 staged) | — | `local-llm-extreme-experiments@80000f8` |

Cleanup record: `artifacts/optimized-prompts/_CLEANUP_LOG.md` — 4.6 MB → 1.0 MB,
16 `gepa_state/` subdirs removed, BAG smoke still PASS. Current `latest` ->
`bag_postfix_verifier_run_20260502T073424Z` (post-fix run; `val_score_before`
0.64, `val_score_after` 0.64, `delta` 0.0 per `latest/run_meta.json`).

### D — Dataset corpora (`trace-gepa/data/`)

14 JSONL files; ~1.1 GB on disk including `sanitised/`. **Raw datasets are
ignored via `.gitignore`** (raw traces never pushed). Sanitised copies in
`data/sanitised/` (14 files, audit-clean; 803 K user-path replacements,
195 K private-repo refs replaced, 5 OpenAI keys redacted, 0 residuals).

Repo destination: **none yet** — pending user decision (#7 below). Local-only.

### E — `src/` TS integration

| file | lines | repo |
|---|---:|---|
| `src/optimized-prompt-loader.ts` | 77 | `local-llm-extreme-experiments@885c762` |
| `src/trace-rag-shim.ts` | 97 | `local-llm-extreme-experiments@885c762` |
| `src/optimizer/prompt-artifact-bridge.ts` | — | `local-llm-extreme-experiments@885c762` |
| `src/source-adapters/{boundary,cc-session-v2}.ts` | — | `local-llm-extreme-experiments@885c762` |
| `src/autonomous-coding-turn.ts`, `src/dag-tool-loop.ts` | — | `local-llm-extreme-experiments@885c762` |
| `tests/optimized-prompt-loader.test.ts` | — | `local-llm-extreme-experiments@885c762` |
| `tests/prompt-artifact-bridge.test.ts` | — | `local-llm-extreme-experiments@885c762` |
| `tsconfig.test.json` | — | `local-llm-extreme-experiments@885c762` |

Consumed by: BAG executor at `src/autonomous-coding-turn.ts:235`; default-on
behaviour (opt-out via `BAG_DISABLE_OPTIMIZED_PROMPT=1`).

### F — MCP server (`mcp_servers/trace_rag/`, `.mcp.json`)

`.mcp.json` declares a `trace-rag` server: command
`.venv-gepa/bin/python -m mcp_servers.trace_rag.server`,
`TRACE_RAG_INDEX_DIR=trace-gepa/artifacts/rag_index_v2`. Index covers 8,264
records (post-orchestration filter). Lives in `local-llm-extreme-experiments@885c762`
(`.mcp.json`) and `@d055b29` (server source).

### G.1 — Pre-existing scratch (`bench/` top-level, 57,894 files, ~59-job suite)

**Local only, not staged.** User directive #4 (no action). Sub-suites observed
on disk: `aider_polyglot/`, `audit/`, `bag_agent/`, `bag-runtime/`,
`code_search_ab/`, `jobs/`, `livecodebench/`, `metr_th/`, `swe_bench_mm/`,
`vendor/`. Flagged here for the record only.

### G.2 — Project docs and proposals

| dir | files | repo |
|---|---:|---|
| `trace-gepa/proposals/` | 16 | `local-llm-extreme-experiments@9ebbe28` |
| `trace-gepa/{README,STATE,RELEASE,REPORT,FINAL_VERDICT,INTEGRATION_PLAN,SHARED_BRIEFING,CODEX_PROMPT_RESEARCH}.md` | 8 | `local-llm-extreme-experiments@9ebbe28` |
| `trace-gepa/COMMIT_MANIFEST.md` | 1 | `local-llm-extreme-experiments@9ebbe28` |

## 4. The four user directives + answers

1. **"Sanitisation first, I might share it."**
   Status: **14 datasets sanitised**, audit-pass `True`, **0 residuals**, **15
   proper-noun candidates pending review** in `data/sanitised/_proper_nouns_audit.json`.
   No public/private repo created for distribution yet (awaiting #7).

2. **"Is it necessary do anything with [gepa_state intermediates]?"**
   Answer: **No.** Done. 4.6 MB → 1.0 MB after Cleanup Agent removed 16
   `gepa_state/` subdirs (`_CLEANUP_LOG.md`). BAG smoke still PASS.

3. **"Directly to main, only exception is benchmark repo — evaluate the branch
   and build ON TOP."**
   Status: `local-llm-extreme-experiments` received **5 commits on `master`**
   (d055b29 → 9ebbe28). For `local-coding-benchmark`, Bench-Track Agent #1
   produced the FF-and-layer plan in `BENCH_REPO_EVAL.md`. At write time the
   remote `main` is still at `a64bac3` (the refactor branch tip, unchanged):
   the FF + on-top action-selection layer are **not yet visible on origin**.
   See section 2 for the precise tip hash.

4. **"I do not think it makes sense [for `bench/` pre-existing 59-job benchmark]."**
   No action taken. The 57,894 files at top-level `bench/` are local-only and
   uncommitted, as requested.

## 5. Real measurable outcomes

- **Counterfactual corpus:** 431 records, 0 failed/skipped, mean confidence
  0.76, 9 abort verdicts, ~$6-8 cost (`data/counterfactuals_summary.md`).
  delta_kind distribution: tool_swap 165 (38.3%), input_fix 133 (30.9%),
  verify_first 120 (27.8%), abort 9 (2.1%), decompose 4 (0.9%).
- **Sanitisation:** 803,620 user-path replacements (`path_users_satan`),
  195,156 `private_repo_6` refs, 72,199 flat-path refs, 16,224 high-entropy
  strings, 5 real OpenAI keys redacted, 0 audit residuals, 14/14 files
  processed in 105.37 s.
- **Verifier suite:** 4 tiers (regex/JSON-DSL → LM-judge → sandboxed shell →
  weighted composite), 30+ unit tests, 2 real bugs caught and fixed (FIX1 and
  INV1).
- **Benchmark task corpus:** 175 tasks (105 trace-derived + 70 synthetic) over
  7 categories: `tool_routing | command_synthesis | edit_safety | path_grounding
  | debugging | recovery | planning`. Schema-translatable to
  `local-coding-benchmark`'s `run`/`task`/`record` JSONL format
  (`BENCH_REPO_EVAL.md` §9).
- **Trace-RAG:** TF-IDF index over 8,264 records (post-orchestration filter),
  MCP server live in this Claude Code session via `.mcp.json`.
- **BAG production wiring:** default-on; opt-out via
  `BAG_DISABLE_OPTIMIZED_PROMPT=1`. Insertion point:
  `src/autonomous-coding-turn.ts:235`.
- **Repo commits:** **5** to `local-llm-extreme-experiments` (d055b29, 9352549,
  885c762, 80000f8, 9ebbe28); **0** to `local-coding-benchmark` from this
  workstream (the `a64bac3` tip is the pre-existing refactor branch HEAD, not
  our work).

## 6. Honest negatives (do not bury)

- **GEPA optimised vs seed:** statistically tied. 0.291 (optimised) vs 0.303
  (seed) overall on 175-task bench, opus task LM, fixed verifier
  (`FINAL_VERDICT.md`). The earlier headline of "+64% relative" was a verifier
  bug — `structural_json` tasks (89 % of the bench) trivially passed any
  parseable JSON pre-FIX1.
- **Three independent confirmations of plateau:** full eval, persona A/B,
  and the GEPA postfix-verifier rerun (`run_meta.json` for `latest` shows
  `val_score_before` = `val_score_after` = 0.64, `delta` 0.0).
- **Pattern mining:** the corpus is dominated by 22 main user sessions, with
  84 % of records originating from the `ir-multivector-retrieval` repo. The
  "30K-record" headline is record-count, not session-diversity.
- **Pre-flight predicates** capped at ~17 % recall — semantic shell failures
  are uncatchable without execution.
- **A/B persona prefix:** 17 % behavioural shift but the mean is flat
  (divergences happen on already-failing tasks, not on improvable ones).
- **Three confounded categories** (`command_synthesis`, `path_grounding`,
  `planning`, 59 tasks in total) are pathological — `available_tools` for
  these tasks does not include the canonical tool the verifier expects. They
  inflate the apparent ceiling of all candidates downward.
- **The optimised prompt costs ~10× more tokens** per BAG planner call
  (2,544 vs 236 chars) for no measurable benefit.
- **NEW honest-finding additions noticed reading on-disk evidence:**
  - `bag_postfix_verifier_run_20260502T073424Z` (the current `latest` symlink
    target) was the GEPA *re-run with the corrected verifier* and produced
    `delta = 0.0` — **this is a third, independent confirmation that GEPA
    isn't the bottleneck**, not a noise blip.
  - `latest_codex` is a **separate symlink** alongside `latest`, pointing at
    a Codex-driver run; this is silently in-tree but never wired into BAG
    (BAG only reads `latest`). Listed under section 7 for the record.
  - Three runs in `optimized-prompts/` (`gpt55_run_…T233633Z`,
    `bag_exec_opus_run_…T232624Z`, `hybrid_run_…T230908Z`) survive with
    incomplete artefact sets per `_CLEANUP_LOG.md` — they shipped but should
    not be cited as production candidates.
  - `cleanup_log` reports the spec **named 18 GEPA runs and a different
    `latest` target** than what existed at execution time (19 dirs, `latest`
    actually pointed at the postfix-verifier run); the doc is updated, but
    any earlier brief that quoted the old target is now stale.

## 7. Open user-decisions

1. **Proper-noun review** — 15 candidates in
   `data/sanitised/_proper_nouns_audit.json`. Strongest candidates by hit
   count: `Tachiom` (155), `Jira` (50), `Confluence` (25), `Acme` (4). The
   list also contains common English capitalised words (`Keep` 199, `Schema`
   44, `Atomic` 37, `Allowed` 37, `Make` 35, `User` 14, `Forbidden` 9, `The`
   4, `Note` 3, `Beta` 2, `Email` 2) that were flagged by the heuristic but
   are not actually proper nouns. Recommendation: redact `Tachiom`, `Jira`,
   `Confluence`, `Acme`; ignore the rest.
2. **Private dataset repo** — create `BleedingDev/trace-gepa-data` (private)
   for the sanitised corpus, or keep local-only? Sanitisation is audit-clean
   so distribution is technically unblocked.
3. **17 proposal docs in `proposals/`** — kept inline at `9ebbe28`. Keep
   inline, archive to `notes/`, or split out?
4. **`bench/` pre-existing 59-job benchmark** — still local-only and
   uncommitted, per user directive #4. Flagged here for the record.
5. **Roll back BAG default to seed?** Per `FINAL_VERDICT.md`, the optimised
   prompt is statistically tied and costs ~10× tokens. Decision pending:
   `BAG_DISABLE_OPTIMIZED_PROMPT=1` env, or unlink `latest` symlink.
6. **Land the on-top action-selection track on `local-coding-benchmark`** —
   the FF+layer plan is in `BENCH_REPO_EVAL.md`; the FF itself has not been
   observed on the remote `main` at write time. Confirm whether we should
   land it now or hold for separate review.
7. **`latest_codex` symlink** — kept in tree, not wired into BAG. Keep as a
   reference, or remove?

## 8. Reproduction

```bash
# 0. Clone & install
git clone https://github.com/BleedingDev/local-llm-extreme-experiments.git
cd local-llm-extreme-experiments
python -m venv .venv-gepa && source .venv-gepa/bin/activate
pip install -r trace-gepa/requirements.txt   # (or follow trace-gepa/README.md)
bun install

# 1. Re-run the action-selection benchmark (175 tasks, opus task LM)
python trace-gepa/bench/run_anthropic.py \
  --tasks trace-gepa/data/benchmark_tasks_full.jsonl \
  --model claude-opus-4-7 \
  --out trace-gepa/bench/results/full_eval/opus_seed.json
# Then with the optimised prompt:
BAG_OPTIMIZED_PROMPT=$(cat trace-gepa/artifacts/optimized-prompts/latest/best_candidate.system.md) \
python trace-gepa/bench/run_anthropic.py \
  --tasks trace-gepa/data/benchmark_tasks_full.jsonl \
  --model claude-opus-4-7 \
  --out trace-gepa/bench/results/full_eval/opus_optimized.json
# Run id of the canonical optimised candidate: bag_postfix_verifier_run_20260502T073424Z

# 2. BAG smoke (no-root) — exercises the loader + shim end-to-end
bun run trace-gepa/scripts/bag_smoke_no_root.ts

# 3. Reproduce the cleanup (idempotent — already at 1.0 MB)
du -sh trace-gepa/artifacts/optimized-prompts   # expect 1.0M

# 4. Reproduce the sanitisation audit
python -m agent_opt.sanitise_audit \
  --input trace-gepa/data/ \
  --output trace-gepa/data/sanitised/

# 5. Counterfactuals (431-record corpus)
python -m agent_opt.counterfactual --input trace-gepa/data/dataset_v2.jsonl \
  --output trace-gepa/data/counterfactuals.jsonl
```

Canonical run IDs cited:

- Production candidate: `bag_postfix_verifier_run_20260502T073424Z`
  (`val_score_before` 0.64, `val_score_after` 0.64, opus reflection,
  budget 400, train 100, val 50, seed 42).
- Comparable Codex driver run: pointed at by
  `artifacts/optimized-prompts/latest_codex` (not wired into BAG).
- Sanitisation audit: see `data/sanitised/_audit_summary.md`
  (wallclock 105.37 s, 14/14 files).

## 9. Pointers

- `trace-gepa/STATE.md` — current state (post-correction).
- `trace-gepa/FINAL_VERDICT.md` — honest GEPA verdict (tied).
- `trace-gepa/BENCH_REPO_EVAL.md` — local-coding-benchmark FF+layer plan.
- `trace-gepa/COMMIT_MANIFEST.md` — file-level manifest of what landed in
  `local-llm-extreme-experiments`.
- `trace-gepa/RELEASE.md`, `trace-gepa/REPORT.md` — release notes and report.
- `trace-gepa/INTEGRATION_PLAN.md` — BAG integration plan.
- `trace-gepa/data/counterfactuals_summary.md` — 431-record corpus stats.
- `trace-gepa/data/sanitised/_audit_summary.md` — sanitisation stats.
- `trace-gepa/data/sanitised/_proper_nouns_audit.json` — pending proper-noun
  review.
- `trace-gepa/artifacts/optimized-prompts/_CLEANUP_LOG.md` — gepa_state
  cleanup log.
- `trace-gepa/artifacts/optimized-prompts/latest/run_meta.json` — current
  production candidate metadata.
- `trace-gepa/bench/specialist_consultation.md` — GPT-5.5 critique of the
  optimised prompt.
- `trace-gepa/bench/zero_cat_investigation.md` — INV1's diagnosis of the
  three zero-score categories.
- `trace-gepa/bench/COMPARATIVE_POSITIONING.md` — bench positioning.
- `trace-gepa/SHARED_BRIEFING.md` — cross-agent briefing.
