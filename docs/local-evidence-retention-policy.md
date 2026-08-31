# Local Evidence Retention Archive And Cleanup Policy

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

This policy turns `docs/local-evidence-inventory.md` plus direct local checks into operator rules. The default posture is conservative: ignored local evidence is not disposable, cleanup begins with dry-run listings only, and deletion requires explicit approval at the boundary named below.

## Operator Summary

- Preserve primary evidence, sanitised mirrors, benchmark definitions, benchmark results, optimizer artifacts, and canonical evidence docs.
- Prefer sanitised evidence for optimizer input and sharing. Raw JSONL and ACP traces remain local-only unless a separate sanitisation review approves release.
- Treat derived indexes and anomaly models as rebuildable but operationally useful; remove them only after recording source data and rebuild commands.
- Treat runtime dependencies, build outputs, caches, and temporary workspaces as cleanup candidates, not evidence, but still require dry-run review before removal.
- Do not trust `latest` symlinks as canonical truth. Use lineage, scorecards, and timestamped run directories.
- This runbook authorizes no deletion.

## Retention Tiers

| Tier | Category | Paths | Retention | Cleanup Rule |
| --- | --- | --- | --- | --- |
| T0 | Primary evidence | `trace-gepa/data/dataset_v2.jsonl`, `cc_dataset_v2_new.jsonl`, `codex_gpt55_dataset.jsonl`, `dataset_recovery.jsonl`, `counterfactuals.jsonl`, `.bag/replay-corpus/**`, `.bag/runs/**`, `.bag/telemetry/**` | Indefinite | No deletion without explicit evidence-owner approval and archive |
| T1 | Sanitised evidence | `trace-gepa/data/sanitised/**`, split manifests | Indefinite | No deletion; preferred optimizer/shareable input after review |
| T2 | Benchmark definitions | `trace-gepa/data/benchmark_tasks_full.jsonl`, `trace-gepa/data/benchmarks/**`, benchmark summaries/audits | Indefinite | No deletion; archive before major rewrite |
| T3 | Benchmark results | `bench/jobs/**/{result.json,bag-acp-summary.json,audit.jsonl,exception.txt}`, `bench/aider_polyglot/results/**`, `trace-gepa/bench/results/**` | Keep canonical runs indefinitely | Archive before any approved pruning |
| T4 | Optimizer artifacts | `bench/.bag/optimizer/**`, `trace-gepa/artifacts/optimized-prompts/**/{best_candidate.json,run_meta.json,log.txt,_CLEANUP_LOG.md}` | Indefinite for candidates, lineage, scorecards, clusters | Do not collapse history to `latest` |
| T5 | Derived indexes and models | `trace-gepa/artifacts/rag_index*`, `trace-gepa/artifacts/anomaly_*.pkl` | Retain while active or until rebuild verified | Operator approval; archive recommended |
| T6 | Runtime dependencies | `node_modules/**`, `bench/bag-runtime/node_modules/**`, `.venv/**`, `bench/.venv/**` | Reinstallable | Dry-run list, then approval before deletion |
| T7 | Vendor checkouts | `vendor/**`, `bench/vendor/**` | Retain until pinned source fetch and patch state are verified | Record upstream URL, commit, and patch state first |
| T8 | Generated build outputs | `dist/**`, `build/**`, `target/**`, coverage dirs, `trace-gepa/artifacts/lora_adapters/**` | Ephemeral unless release artifact | Dry-run list, then approval before deletion |
| T9 | Temporary workspaces and caches | `tmp/**`, `.cache/**`, `**/__pycache__/**`, pytest/mypy/ruff caches, `.codex/plan-graphs/**`, `.claude/worktrees/**` | Short-lived | Dry-run list, then approval before deletion |
| T10 | Documentation and policy evidence | `docs/bleeding-agent*.md`, this policy, inventory, canonical `trace-gepa/*.md` state docs | Indefinite for current canonical docs | Supersede deliberately; avoid silent deletion |

Machine-readable tiers are in `.bag/evidence/retention-policy.json`.

## Archive Format

Use archive-before-prune for evidence tiers T0 through T5. Preferred format is `tar.zst`; fallback is `tar.gz` when zstd support is unavailable.

Archive path pattern:

```sh
.bag/evidence/archive/<tier>/<basename>-YYYYMMDDTHHMMSSZ.tar.zst
```

Each archive must have a sidecar `manifest.json` recording:

- policy id and tier
- creation timestamp
- source paths
- git HEAD and `git status --short`
- file count and byte size
- SHA-256 checksum
- sanitisation status
- approval reference
- restore command

Keep raw and sanitised evidence in separate archives. Preserve symlinks as symlinks and record resolved targets. Do not bundle `node_modules`, virtualenvs, caches, or build outputs into evidence archives unless an operator asks for a reproducibility bundle.

## Git Policy

The current `.gitignore` intentionally excludes large local evidence families such as `trace-gepa/data/*.jsonl`, `trace-gepa/data/sanitised/`, `bench/jobs/`, `bench/vendor/`, `node_modules/`, `dist/`, and `.bag/`. That does not make them disposable.

Track small policy files, manifests, curated benchmark definitions, and reviewed summaries. Keep raw evidence, large sanitised JSONL mirrors, benchmark run outputs, optimizer local state, derived indexes, runtime dependencies, vendor checkouts, and build outputs local by default. Do not commit vendor checkouts or runtime snapshots as source of truth.

## Dry-Run Cleanup Commands

These commands only list candidates. Do not pipe them to `rm` without explicit approval.

Runtime dependencies:

```sh
find . -path './node_modules' -prune -print -o -path './bench/bag-runtime/node_modules' -prune -print -o -path './.venv' -prune -print -o -path './bench/.venv' -prune -print
```

Python caches:

```sh
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -print
```

Generated build outputs:

```sh
find . -maxdepth 3 -type d \( -name 'dist' -o -name 'build' -o -name 'target' -o -name 'coverage' -o -name 'htmlcov' \) -print
```

Vendor checkout inventory:

```sh
find vendor bench/vendor -maxdepth 2 -type d -print 2>/dev/null
```

Benchmark result evidence inventory:

```sh
find bench/jobs -type f \( -name 'result.json' -o -name 'bag-acp-summary.json' -o -name 'audit.jsonl' -o -name 'exception.txt' \) -print
```

## Approval Boundaries

No approval is required for read-only inventory checks, archive creation that leaves source evidence unchanged, or dry-run reports that only print paths.

Operator approval is required before deleting runtime dependencies, vendor checkouts, generated build outputs, temporary workspaces, derived indexes, or anomaly models.

Explicit evidence-owner approval is required before deleting or rewriting raw trace JSONL, sanitised canonical mirrors, benchmark definitions, benchmark result directories, optimizer candidates, optimizer lineage, failure clusters, prompt optimization run metadata, or ACP replay corpora.

Publishing raw evidence outside this local machine is not allowed under this policy. Publish only sanitised evidence after review.

## Direct Local Checks Used

- `du -sh trace-gepa/data trace-gepa/artifacts bench .bag`: `1.1G`, `161M`, `1.4G`, `2.5M`.
- `find trace-gepa/data -maxdepth 3 -type f | wc -l`: 59 files.
- `find bench/jobs -type f ... | wc -l`: 996 benchmark result signal files.
- `find trace-gepa/artifacts/optimized-prompts -maxdepth 2 -type f ... | wc -l`: 49 optimizer prompt metadata files.
- Runtime dependency directories observed: `.venv`, `bench/.venv`, `node_modules`, `bench/bag-runtime/node_modules`.
- Vendor checkouts observed: `vendor/ddtree`, `vendor/dflash`, `vendor/paroquant`, `vendor/triattention`, `vendor/turboquant-mlx`, `bench/vendor/harbor`, `bench/vendor/polyglot-benchmark`.
