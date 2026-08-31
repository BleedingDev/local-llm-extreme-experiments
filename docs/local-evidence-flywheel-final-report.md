# Local Evidence Flywheel Final Report

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

## Executive Summary

The local evidence audit is closed as an evidence and planning graph. The valuable data is not random disk junk: it contains action traces, recovery pairs, ACP replay failures, benchmark results, optimizer candidates, prompt optimization lineage, and derived scorecards that can drive self-improvement.

The current optimizer decision is deliberately conservative:

- Candidate generation is allowed only as scoped dry-run optimizer work.
- Promotion and auto-promotion are blocked.
- Auto-promotion remains blocked by the existing post-promotion monitor window and by new evidence gates for visible ACP no-write/no-terminal validation, hidden holdout readiness, rollback checkpoint, operator approval, and edit-attempt telemetry.

## Canonical Outputs

| Area | Operator report | Machine-readable local artifact |
| --- | --- | --- |
| Inventory | `docs/local-evidence-inventory.md` | n/a |
| Quality audit | `docs/local-evidence-quality-audit.md` | `.bag/evidence/schema-audit.json` |
| Corpus index | `docs/local-evidence-corpus-index.md` | `.bag/evidence/index.jsonl`, `.bag/evidence/index.schema.json` |
| Retention policy | `docs/local-evidence-retention-policy.md` | `.bag/evidence/retention-policy.json` |
| Scorecards | `docs/local-evidence-scorecards.md` | `.bag/evidence/scorecards/index.json` |
| Optimizer gates | `docs/local-evidence-optimizer-gates.md` | `.bag/evidence/optimizer/index.json` |
| Release proof | this report | `.bag/evidence/release-proof.json` |

## High-Value Evidence

- `trace-gepa/data/dataset_v2.jsonl`: 26,384 action/tool rows.
- `trace-gepa/data/sanitised/dataset_v2.jsonl`: sanitised mirror for optimizer/shared use.
- `trace-gepa/data/dataset_recovery.jsonl`: 4,055 recovery pairs, including 3,520 strong pairs.
- `trace-gepa/data/counterfactuals.jsonl`: 431 counterfactual annotations.
- `.bag/replay-corpus/**`: real ACP replay evidence; current visible run is 0/9 pass with zero writes and zero terminal verification.
- `bench/jobs/**`: 541 result JSONs, 415 ACP summaries, terminal-bench and related benchmark runs.
- `bench/.bag/optimizer/**`: 85 optimizer records, 12 validated candidates, 26 failure clusters, readiness report.
- `trace-gepa/artifacts/optimized-prompts/**`: timestamped prompt optimization lineage.

## Key Findings

The visible ACP failures are progress failures: ACP read calls succeeded, but coding tasks ended with no file writes, no terminal verification, and no changed files. This must be a blocking validation slice before promotion.

The local evidence does not support choosing one globally best edit strategy. Shell heredoc, scripted writes, string replacement, structured ACP writes, Claude-style `Edit`/`Write`, and future `apply_patch`-style tools need per-model, per-codebase, per-task-shape measurement.

Recovery evidence is strong enough to guide policy work. The dominant classes are terminal nonzero exits, hallucinated paths, timeouts, retry loops, cancelled parallel batches, and verifier/output-file mismatches.

Benchmark evidence is useful but heterogeneous. Terminal-bench high-water results, Claude Code comparator runs, Aider polyglot, LiveCodeBench smoke, SWE-Bench MM smoke, METR TH smoke, and real ACP replay must stay separated by evaluator family.

## Quality And Retention

Audited central JSONL files parse cleanly. No audited split bucket leakage was found. Known caveats remain: duplicate IDs in legacy/RAG-derived files, one raw-vs-sanitised recovery ID parity mismatch, raw-local privacy patterns, and an action split manifest that needs a real train/dev/holdout projection before hidden holdout use.

No evidence deletion is authorized by this graph. Runtime dependencies, build outputs, and caches can be dry-run cleanup candidates, but primary evidence, optimizer lineage, replay corpora, and benchmark results need explicit evidence-owner approval before deletion or rewrite.

## Handoff Bundle

Use this exact graph selection for downstream `plan-graph`, `subagent-graph`, `dag`, or `helm` work:

- Graph ID: `local-evidence-flywheel-v1`
- Selection hash: `06eeb209cb`
- Plan root: `.codex/plans/local-evidence-flywheel`
- Glob: `*.plan.md`
- State dir: `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.codex/plan-graphs/local-evidence-flywheel-v1`
- Snapshot path: `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.codex/plan-graphs/local-evidence-flywheel-v1/snapshot.json`
- Selected plans: 7
- Dependency edges: 12

Dependency overlay:

```text
Local Evidence 01 Inventory And Classification:Local Evidence 02 Schema And Quality Audit
Local Evidence 01 Inventory And Classification:Local Evidence 03 Canonical Corpus Index
Local Evidence 01 Inventory And Classification:Local Evidence 06 Retention Archive And Cleanup Policy
Local Evidence 02 Schema And Quality Audit:Local Evidence 03 Canonical Corpus Index
Local Evidence 02 Schema And Quality Audit:Local Evidence 04 Mining And Scorecards
Local Evidence 02 Schema And Quality Audit:Local Evidence 05 Optimizer Integration And Gates
Local Evidence 03 Canonical Corpus Index:Local Evidence 04 Mining And Scorecards
Local Evidence 03 Canonical Corpus Index:Local Evidence 05 Optimizer Integration And Gates
Local Evidence 04 Mining And Scorecards:Local Evidence 05 Optimizer Integration And Gates
Local Evidence 04 Mining And Scorecards:Local Evidence 07 Release Proof And Next Graph
Local Evidence 05 Optimizer Integration And Gates:Local Evidence 07 Release Proof And Next Graph
Local Evidence 06 Retention Archive And Cleanup Policy:Local Evidence 07 Release Proof And Next Graph
```

Important operator note: do not use `--graph-id` alone if old completed plans are present. Reuse this exact plan root, glob, and dependency overlay.

## Validation Proof

Executed validation checks:

```sh
jq empty .bag/evidence/schema-audit.json .bag/evidence/retention-policy.json .bag/evidence/index.schema.json .bag/evidence/scorecards/*.json .bag/evidence/optimizer/*.json
awk 'NF' .bag/evidence/index.jsonl | while IFS= read -r line; do jq -e . >/dev/null <<<"$line"; done
jq -r 'select(.recordKind=="source") | .evidenceId' .bag/evidence/index.jsonl | sort > /tmp/bleeding-source-ids.txt
jq -r 'select(.recordKind=="slice") | .memberEvidenceIds[]' .bag/evidence/index.jsonl | sort -u > /tmp/bleeding-member-ids.txt
comm -13 /tmp/bleeding-source-ids.txt /tmp/bleeding-member-ids.txt
python /Users/satan/side/experiments/skills/plan-graph/scripts/plan_graph.py validate --plans-root .codex/plans/local-evidence-flywheel --glob '*.plan.md' --graph-id local-evidence-flywheel-v1 ...
git diff --check -- docs .codex/plans/local-evidence-flywheel .codex/plan-graphs/local-evidence-flywheel-v1/operator-log.md
```

All checks passed. The slice reference check returned no missing evidence IDs.

## Next Execution Frontier

1. Wire `.bag/evidence/optimizer/index.json` into runtime scheduler and `/maintenance status` so the agent reports the same fail-closed promotion state from the actual ACP surface.
2. Encode visible ACP no-write/no-terminal replay as a blocking validation suite and promotion veto.
3. Implement sealed train/dev/hidden-holdout projection for `dataset_v2` and prevent hidden evidence from candidate generation, RAG, clustering, or prompt drafting.
4. Add first-class edit attempt telemetry: strategy ID, rendered contract version, target hashes, preview/apply/write/verify/repair/rollback phases, protected-path events, stale-context detection, and applied-but-broken/self-detected regression.
5. Turn the scorecard and optimizer-gate generation into reproducible scripts or commands so future agents regenerate artifacts instead of hand-mining them.

Conflict boundaries for the next graph:

- `src/optimizer/**` and `src/acp/maintenance.ts` need a single integration owner.
- ACP replay validation should own replay/eval harness files only.
- Edit telemetry should own edit lifecycle/routing telemetry files only.
- Hidden holdout projection should own dataset/split tooling only.
- Generated evidence under `.bag/evidence/**` should stay local ignored unless a separate commit-safe export is explicitly produced.
