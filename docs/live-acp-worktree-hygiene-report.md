# Live ACP Worktree Hygiene Report

Generated: 2026-05-05

Graph: `live-acp-evidence-readiness-v1`  
Selection hash: `6fbc4883fa`  
Lane: `Live ACP Evidence 04 Dirty Worktree Evidence Hygiene`

## Scope

This is a non-destructive dirty-worktree inventory and retention plan. It does not delete, move, rewrite, or compress any project data. The goal is to separate release-critical BAG/ACP/self-optimization evidence from unrelated model benchmark work, trace-gepa experiments, generated artifacts, local config, and disposable residue.

## Commands Run

```bash
git status --short
git diff --name-only
git diff --cached --name-only
git diff --stat
git diff --cached --stat
git ls-files --others --exclude-standard | awk 'BEGIN{FS="/"} {top=$1; count[top]++} END{for (top in count) print count[top], top}' | sort -nr
du -sh .codex .bag bench docs scripts src tests trace-gepa types package.json package-lock.json rspack.config.ts tsconfig.json tsconfig.test.json 2>/dev/null | sort -h
find bench trace-gepa .bag -type f -print0 2>/dev/null | xargs -0 du -h | sort -hr | head -40
find .bag -maxdepth 4 -type f | sort | sed -n '1,160p'
find .codex/plans/live-acp-evidence-readiness .codex/plan-graphs/live-acp-evidence-readiness-v1 -maxdepth 3 -type f | sort
git status --short --ignored .bag .codex bench trace-gepa | sed -n '1,220p'
git ls-files --others --exclude-standard | awk 'BEGIN{FS="/"} {key=$1; if (NF>1) key=key"/"$2; count[key]++} END{for (key in count) print count[key], key}' | sort -nr | sed -n '1,120p'
du -sh bench/* trace-gepa/* .codex/* .claude/* 2>/dev/null | sort -hr | sed -n '1,80p'
git ls-files --others --exclude-standard src tests scripts docs package.json package-lock.json rspack.config.ts tsconfig.json types gepa-env.md | sed -n '1,260p'
git status --short package.json package-lock.json rspack.config.ts tsconfig.json tsconfig.test.json .gitignore .mcp.example.json .mcp.json gepa-env.md .claude types | sed -n '1,160p'
```

## Snapshot Summary

`git status --short` shows:

- Staged or partially staged local config hygiene:
  - `MM .gitignore`
  - `A  .mcp.example.json`
  - `D  .mcp.json`
- Tracked modifications:
  - 19 tracked files changed, with `983 insertions(+)` and `155 deletions(-)` from `git diff --stat`.
  - Main tracked clusters: DFlash/Qwen benchmark docs and scripts, `src/autonomous-coding-turn.ts`, prompt loading, Claude Code session adapter, trace RAG shim, and `trace-gepa` RAG files.
- Untracked top-level counts:
  - `bench`: 169
  - `src`: 154
  - `tests`: 101
  - `docs`: 78
  - `.codex`: 56
  - `trace-gepa`: 49
  - `scripts`: 31
  - `.claude`: 9
  - singletons: `package.json`, `package-lock.json`, `rspack.config.ts`, `tsconfig.json`, `types/bun-test.d.ts`, `gepa-env.md`
- Ignored but important evidence/generated areas:
  - `.bag/`
  - `.codex/plan-graphs/`
  - `bench/.bag/`, `bench/.venv/`, `bench/vendor/`, `bench/jobs/`, `bench/*/results/`, `bench/bag-runtime/node_modules/`
  - `trace-gepa/data/*.jsonl`, `trace-gepa/data/sanitised/`, `trace-gepa/artifacts/**`, Python caches

## Size Summary

Disk use from sampled roots:

| Path | Size | Classification |
|---|---:|---|
| `bench` | 1.4G | model/eval benchmark workspaces, vendored harnesses, venvs, generated outputs |
| `trace-gepa` | 1.3G | trace-gepa experiment datasets, sanitised copies, RAG indexes, anomaly models |
| `.bag` | 3.0M | BAG runtime evidence, replay corpus, telemetry, prior runs |
| `src` | 2.5M | core BAG/ACP/self-optimization source |
| `tests` | 1.2M | core BAG/ACP/self-optimization tests |
| `docs` | 1.1M | project reports and research notes |
| `scripts` | 884K | product, evidence, benchmark, and model scripts |
| `.codex` | 540K | plan files and graph state |

Largest individual files or clusters:

| Path | Size | Recommended action |
|---|---:|---|
| `bench/.venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude` | 206M | disposable/regenerable; archive only if exact env reproduction is needed |
| `trace-gepa/data/sanitised/dataset_v2.jsonl` | 205M | valuable sanitized dataset; archive or compress, do not delete |
| `trace-gepa/data/dataset_v2.jsonl` | 202M | raw dataset; manual privacy review before commit/archive |
| `trace-gepa/data/sanitised/cc_dataset_v2_new.jsonl` | 179M | valuable sanitized dataset; archive or compress |
| `trace-gepa/data/cc_dataset_v2_new.jsonl` | 176M | raw dataset; manual privacy review |
| `trace-gepa/artifacts/rag_index/tfidf_matrix.npz` | 52M | generated index; regenerate or archive with dataset manifest |
| `trace-gepa/artifacts/anomaly_iforest.pkl` | 43M | generated model artifact; archive with training metadata if useful |
| `bench/vendor/harbor/.../pyarrow/libarrow*.dylib` | 41M+ | vendored dependency; keep out of main release scope |

## Cluster Classification

### A. Core BAG Agent Code

Value: high. Ownership: product code.

Exact cluster examples:

- `src/acp-agent.ts`
- `src/acp/**`
- `src/replay/**`
- `src/evidence/**`
- `src/optimizer/**`
- `src/edit-strategy/**`, `src/edit-strategies/**`
- `src/eval-harness/**`
- `src/knowledge/**`
- `src/mcp/**`
- `src/sdk/**`
- `src/workspace*.ts`
- `tests/acp-*.test.ts`
- `tests/replay-*.test.ts`
- `tests/optimizer-*.test.ts`
- `tests/edit-strategy-*.test.ts`
- `tests/eval-harness-*.test.ts`
- `package.json`, `package-lock.json`, `rspack.config.ts`, `tsconfig.json`, `tsconfig.test.json`, `types/bun-test.d.ts`

Recommended action:

- Commit in coherent product-code slices, not one mega-commit.
- Keep source/test/package config in the release review path if needed by `live-acp-evidence-readiness-v1`.
- Require normal verification before commit: typecheck, relevant Bun tests, `bag evidence validate`.

### B. ACP / Self-Optimization Evidence

Value: high. Ownership: runtime evidence and release readiness.

Exact cluster examples:

- `.bag/evidence/index.jsonl`
- `.bag/evidence/release-proof.json`
- `.bag/evidence/optimizer/**`
- `.bag/evidence/scorecards/**`
- `.bag/replay-corpus/index.jsonl`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504*/**`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/**`
- `.bag/telemetry/events.jsonl`
- `.bag/telemetry/spans.jsonl`
- `.bag/telemetry/metrics.json`
- `.bag/acp-consumer-fixtures/local-consumer-validation-*.json`
- `docs/self-evolving-runtime-gates-release-proof.md`
- `docs/live-acp-worktree-hygiene-report.md`
- `.codex/plans/live-acp-evidence-readiness/*.plan.md`
- `.codex/plan-graphs/live-acp-evidence-readiness-v1/snapshot.json`

Recommended action:

- Preserve. This is the main input for the current live-evidence graph.
- Treat `.bag/**` as evidence artifacts, not ordinary source.
- Before promotion readiness, regenerate or validate evidence so it points to `live-acp-evidence-readiness-v1`, not only older graphs.
- Commit only redacted/generated-safe evidence. Keep raw telemetry under manual review if it may contain sensitive paths, prompts, or local content.

### C. Model Benchmark Research

Value: medium to high. Ownership: model research, not current BAG release.

Exact cluster examples:

- `bench/**`
- `docs/qwen36-dflash-paroquant-mlx.md`
- `docs/qwen36-optimization-comparison-report.md`
- `docs/qwen36-rotorquant-mlx-3bit-report.md`
- `docs/carnice-9b-eval.md`
- `docs/hermes-local-coding-eval-report.md`
- `docs/tq3-apple-migration.md`
- `scripts/benchmark_*.py`
- `scripts/run_qwen36_*`
- `scripts/run_carnice_dflash_max_tuning.sh`
- `scripts/run_osaurus_qwen36_smoke.sh`
- tracked benchmark support changes in `docs/benchmarking.md`, `patches/dflash-triattention-mlx.patch`, and `scripts/*dflash*`

Recommended action:

- Keep out of the live ACP evidence release commit.
- Archive/compress `bench` outputs separately if disk pressure matters.
- If useful long-term, split into a dedicated benchmark branch or artifact bundle with a manifest.
- Do not delete benchmark results without explicit approval because they may contain expensive model evaluation data.

### D. Trace-GEPA Experiment

Value: high research value, but separate from current BAG release unless explicitly imported.

Exact cluster examples:

- `trace-gepa/data/*.jsonl`
- `trace-gepa/data/sanitised/**`
- `trace-gepa/artifacts/rag_index/**`
- `trace-gepa/artifacts/rag_index_v2/**`
- `trace-gepa/artifacts/anomaly_*.pkl`
- `trace-gepa/agent_opt/**`
- `trace-gepa/proposals/*.md`
- `trace-gepa/bench/results/sweep/*.stdout`
- `trace-gepa/bench/results/sweep/*.stderr`
- tracked modifications in `trace-gepa/DELIVERABLES.md`, `trace-gepa/agent_opt/rag/embed.py`, and `trace-gepa/agent_opt/rag/test_rag.py`

Recommended action:

- Preserve and quarantine into a research/artifact boundary.
- Compress large JSONL and generated indexes if needed.
- Commit only curated summaries and lightweight code, not raw/sanitised dataset pairs, unless the repository policy explicitly allows it.
- Manual privacy review is required before publishing raw datasets.

### E. Generated Artifacts and Disposable Residue

Value: mixed. Ownership: local generated outputs.

Exact cluster examples:

- `bench/.venv/**`
- `bench/bag-runtime/node_modules/**`
- `bench/vendor/**`
- `bench/jobs/**/repo/.git/**`
- `bench/*/__pycache__/**`
- `trace-gepa/**/__pycache__/**`
- `trace-gepa/.pytest_cache/**`
- `trace-gepa/artifacts/rag_index*/**`
- `trace-gepa/artifacts/real_smoke/**`
- `trace-gepa/bench/results/sweep/*.stdout`
- `trace-gepa/bench/results/sweep/*.stderr`

Recommended action:

- Do not commit to main product history.
- Prefer `git clean -ndX` style previews only, not actual deletion, until the user explicitly approves.
- Compress or archive if reproducibility matters.
- Regenerate when possible from scripts and manifests.

### F. Local Secret / Config

Value: high operational value, high leakage risk. Ownership: local operator.

Exact cluster examples:

- staged deletion: `.mcp.json`
- staged addition: `.mcp.example.json`
- staged/unstaged edits: `.gitignore`
- `gepa-env.md`
- `.claude/**`
- `bench/bag-runtime/.env`
- `bench/bag-runtime/bag.config.json`

Recommended action:

- Keep `.mcp.json`, `.env`, and local provider config out of commits.
- Commit `.mcp.example.json` only after manual review confirms it has no secrets.
- Keep `.gitignore` changes if they protect local secret/config and generated outputs.
- Treat `gepa-env.md` as manual-review before commit because it may document local keys, env vars, or provider setup.

## Minimal Release Scope for `live-acp-evidence-readiness-v1`

The current graph should only need these clusters:

1. Plan and graph control:
   - `.codex/plans/live-acp-evidence-readiness/*.plan.md`
   - `.codex/plan-graphs/live-acp-evidence-readiness-v1/snapshot.json`
   - operator log should remain owned by the orchestrator, not this lane
2. Live ACP evidence:
   - `.bag/evidence/**`
   - `.bag/replay-corpus/index.jsonl`
   - `.bag/replay-corpus/real-acp-runs/**`
   - `.bag/telemetry/**`
   - `.bag/acp-consumer-fixtures/**`
3. Product code and tests required to generate/validate evidence:
   - `src/acp/**`
   - `src/replay/**`
   - `src/evidence/**`
   - `src/optimizer/**`
   - `src/index.ts`
   - relevant `tests/acp-*`, `tests/replay-*`, `tests/evidence-*`, `tests/optimizer-*`, and `tests/bag.test.ts`
   - `scripts/run_real_acp_corpus.ts`
   - `scripts/report_real_acp_scorecard.ts`
   - `scripts/report_real_acp_trace_scorecards.ts`
   - `scripts/report_optimizer_artifact_lineage.ts`
   - `scripts/verify_acp_consumer_setup.ts`
4. Release docs:
   - `docs/self-evolving-runtime-gates-release-proof.md`
   - `docs/live-acp-worktree-hygiene-report.md`
   - any later live-evidence release proof report created by the rollup lane
5. Toolchain config needed by the above:
   - `package.json`
   - `package-lock.json`
   - `rspack.config.ts`
   - `tsconfig.json`
   - `tsconfig.test.json`
   - `types/bun-test.d.ts`

Everything else should be outside this release graph unless another lane proves a direct dependency.

## Recommended Retention Plan

| Cluster | Action | Reason |
|---|---|---|
| Core BAG agent code | Commit in focused source/test slices | Needed for maintainable product history |
| ACP/self-optimization evidence | Preserve; commit only redacted/generated-safe artifacts | Required for live readiness gates and future optimizer learning |
| `.codex/plans/live-acp-evidence-readiness/**` | Commit or keep with graph state handoff | Required to resume graph precisely |
| `.codex/plan-graphs/live-acp-evidence-readiness-v1/snapshot.json` | Preserve with graph handoff | Required to target exact saved graph |
| `bench/**` | Quarantine/archive/compress outside release scope | Large, valuable benchmark research but unrelated to ACP readiness release |
| `trace-gepa/**` | Quarantine/archive/compress outside release scope | Valuable research, privacy-sensitive datasets, separate experiment |
| Python/Node caches, venvs, node_modules | Ignore or regenerate; preview before deleting | Large/regenerable local residue |
| `.mcp.json`, `.env`, local config | Never commit; manual review only | Secret/config leakage risk |
| `.mcp.example.json` | Commit only after manual review | Useful template if sanitized |
| model benchmark docs/scripts | Separate benchmark branch or artifact manifest | Useful but unrelated to current graph |

## Non-Destructive Commands for Next Operator

Preview ignored/generated cleanup without deleting:

```bash
git clean -ndX bench trace-gepa
```

Preview all untracked files without deleting:

```bash
git clean -nd bench trace-gepa
```

Create a compressed archive of benchmark research before any cleanup decision:

```bash
mkdir -p ../bag-artifact-archives
tar -czf ../bag-artifact-archives/bench-$(date +%Y%m%d).tar.gz bench
tar -czf ../bag-artifact-archives/trace-gepa-$(date +%Y%m%d).tar.gz trace-gepa
```

Create reviewable file manifests:

```bash
git status --short > .bag-worktree-status.txt
find .bag -type f | sort > .bag-file-manifest.txt
find bench trace-gepa -type f | sort > .benchmark-trace-file-manifest.txt
```

Stage only the live-evidence graph control and report files:

```bash
git add .codex/plans/live-acp-evidence-readiness docs/live-acp-worktree-hygiene-report.md
```

Stage core product slices separately, after review:

```bash
git add src/acp src/replay src/evidence src/optimizer tests scripts/run_real_acp_corpus.ts scripts/report_real_acp_scorecard.ts scripts/report_real_acp_trace_scorecards.ts scripts/report_optimizer_artifact_lineage.ts scripts/verify_acp_consumer_setup.ts
```

## Residual Risks

- `.mcp.json` is staged for deletion and `.mcp.example.json` is staged for addition. This is probably correct config hygiene, but the example file still needs manual secret review before commit.
- `.bag/**` is ignored but valuable. If release proof depends on it, the project needs a deliberate policy for whether generated evidence is committed, archived externally, or regenerated during release.
- `trace-gepa/data` contains both raw and sanitised dataset pairs. Do not publish raw files without privacy review.
- `bench/bag-runtime/.env` and related runtime config are ignored and should be treated as secret-bearing until proven otherwise.
- `bench/**` and `trace-gepa/**` together are roughly 2.7G. Disk cleanup pressure should target regenerable environments and caches first, but preservation should come before deletion.
- The worktree has many untracked product source and test files. Until they are committed or separated into branches, `git status` cannot distinguish completed BAG implementation from experimental additions.

## Decision

The live ACP evidence readiness work can proceed without destructive cleanup. The minimal release graph should isolate `.codex/plans/live-acp-evidence-readiness/**`, current `.bag` evidence/replay/telemetry artifacts, and the BAG ACP/evidence/optimizer/replay source/test/tooling needed to regenerate and validate them. Benchmark model work under `bench/**`, Qwen/DFlash docs/scripts, and trace-gepa datasets/artifacts should be preserved but kept outside the current release graph.
