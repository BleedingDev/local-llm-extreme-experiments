# Live ACP Canonical Readiness Index

- epoch: `evidence-epoch.blocker-closure-v1.a49f7e68fb`
- graph: `blocker-closure-v1`
- selection hash: `a49f7e68fb`
- generated at: `2026-05-05T11:51:17.725Z`
- drift status: `blocked`
- promotion ready: `false`

## Current Evidence

- `.bag/evidence/canonical-epoch.json`
- `.codex/plan-graphs/blocker-closure-v1/snapshot.json`
- `docs/live-acp-canonical-readiness-index.md`

## Candidate Inputs

- `.bag/evidence/index.jsonl`

## Historical Context

- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-current-simple-20260505`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-current-visible-20260505`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504b`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504c`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504d`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504`
- `docs/local-evidence-flywheel-final-report.md`

## Stale Current Slots

- `.bag/evidence/optimizer/index.json`
- `.bag/evidence/release-proof.json`
- `.bag/evidence/scorecards/index.json`
- `docs/live-acp-current-release-proof-report.md`

## Drift Checks

- epoch.current-graph-selected: `passed` - Canonical graph is blocker-closure-v1 (a49f7e68fb).
- epoch.selected-plan..codex_plans_blocker-closure-v1_00-evidence-epoch-canonical-state.plan.md: `passed` - Selected plan exists: .codex/plans/blocker-closure-v1/00-evidence-epoch-canonical-state.plan.md
- epoch.selected-plan..codex_plans_blocker-closure-v1_01-optimizer-evidence-contracts.plan.md: `passed` - Selected plan exists: .codex/plans/blocker-closure-v1/01-optimizer-evidence-contracts.plan.md
- epoch.selected-plan..codex_plans_blocker-closure-v1_02-coding-progress-diagnostics.plan.md: `passed` - Selected plan exists: .codex/plans/blocker-closure-v1/02-coding-progress-diagnostics.plan.md
- epoch.selected-plan..codex_plans_blocker-closure-v1_03-real-mutating-headless-quality.plan.md: `passed` - Selected plan exists: .codex/plans/blocker-closure-v1/03-real-mutating-headless-quality.plan.md
- epoch.selected-plan..codex_plans_blocker-closure-v1_04-frozen-candidate-hidden-holdout.plan.md: `passed` - Selected plan exists: .codex/plans/blocker-closure-v1/04-frozen-candidate-hidden-holdout.plan.md
- epoch.selected-plan..codex_plans_blocker-closure-v1_05-real-acp-consumer-executor.plan.md: `passed` - Selected plan exists: .codex/plans/blocker-closure-v1/05-real-acp-consumer-executor.plan.md
- epoch.selected-plan..codex_plans_blocker-closure-v1_06-monitored-promotion-workflow.plan.md: `passed` - Selected plan exists: .codex/plans/blocker-closure-v1/06-monitored-promotion-workflow.plan.md
- epoch.scorecards.graph: `failed` - .bag/evidence/scorecards/index.json targets live-acp-evidence-readiness-v1, not canonical graph blocker-closure-v1.
- epoch.scorecards.generated-at: `failed` - .bag/evidence/scorecards/index.json generatedAt belongs to non-current graph live-acp-evidence-readiness-v1.
- epoch.optimizer.graph: `failed` - .bag/evidence/optimizer/index.json targets live-acp-evidence-readiness-v1, not canonical graph blocker-closure-v1.
- epoch.optimizer.generated-at: `failed` - .bag/evidence/optimizer/index.json generatedAt belongs to non-current graph live-acp-evidence-readiness-v1.
- epoch.release-proof.graph: `failed` - Release proof targets live-acp-evidence-readiness-v1, not canonical graph blocker-closure-v1.
- epoch.release-proof.selection: `failed` - Release proof selection 6fbc4883fa does not match a49f7e68fb.
- epoch.release-proof.mode: `passed` - Release proof is marked as current_graph.
- epoch.release-proof.generated-at: `failed` - .bag/evidence/release-proof.json generatedAt belongs to non-current graph live-acp-evidence-readiness-v1.
- epoch.current-release-report: `failed` - Current release-proof report is backed by stale proof for live-acp-evidence-readiness-v1.

## Graph Inventory

- `acp-agent-modularization-v1` (a85316a824): `historical`
- `bleeding-agent-codebase-quality-stabilization-74d9d4fb71` (74d9d4fb71): `historical`
- `bleeding-agent-evidence-flywheel-v1` (295f67f953): `historical`
- `bleeding-agent-quality-execution-v1` (e4bb63af71): `historical`
- `bleeding-agent-v1-acp-harness-closure` (5a733b5de4): `historical`
- `blocker-closure-v1` (a49f7e68fb): `current`
- `live-acp-evidence-readiness-v1` (6fbc4883fa): `historical`
- `local-evidence-flywheel-v1` (06eeb209cb): `historical`
- `self-evolving-runtime-gates-v1` (30baf78610): `historical`
