# Live ACP Readiness Rollup Handoff

Date: 2026-05-05
Graph: `live-acp-evidence-readiness-v1`
Selection hash: `6fbc4883fa`
Lane: `Live ACP Evidence 06 Readiness Rollup Handoff`

## Release Decision

The current release decision is evidence-valid but promotion blocked.

`npm run --silent bag -- evidence validate --graph-id live-acp-evidence-readiness-v1` validates the current graph evidence successfully. The release proof, scorecard suite, optimizer gate suite, visible ACP no-write gate, and plan snapshot now target `live-acp-evidence-readiness-v1` rather than the stale `local-evidence-flywheel-v1` proof slot.

Promotion must remain blocked. Candidate generation is allowed only as scoped dry-run work backed by existing evidence and must not auto-promote into runtime behavior.

Current optimizer decision:

| Field | Value |
| --- | --- |
| candidate generation | `allowed_as_scoped_dry_run` |
| auto promotion | `blocked` |
| promotion ready | `false` |
| proof mode | `current_graph` |

## Lane Results

| Lane | Status | Evidence |
| --- | --- | --- |
| 01 Dogfood Evidence Regeneration | partially completed, blocked | `docs/live-acp-evidence-regeneration-report.md` |
| 02 Current Graph Release Proof Rebuild | completed | `docs/live-acp-current-release-proof-report.md`, `.bag/evidence/release-proof.json` |
| 03 Promotion Readiness Closure | partially completed, blocked | `docs/live-acp-promotion-readiness-report.md` |
| 04 Dirty Worktree Evidence Hygiene | completed | `docs/live-acp-worktree-hygiene-report.md` |
| 05 Real Agent Quality Evaluation | partially completed, blocked | `docs/live-acp-real-agent-quality-report.md` |
| 06 Readiness Rollup Handoff | completed by this report | `docs/live-acp-readiness-rollup-handoff.md` |

Important correction from lane 03 and the follow-up integration pass: visible ACP no-write/no-terminal validation is now represented as concrete evidence in `.bag/evidence/optimizer/no-write-gate.json`. It passes for the current visible run selection: `9/9` checked cases pass, `0` block, `0` warn.

## Residual Blockers

These blockers are current and should not be bypassed:

1. No frozen-candidate hidden holdout final result exists. The split policy excludes hidden holdout from optimizer-visible input, but there is no final holdout pass for a frozen candidate.
2. No current operator approval plus rollback checkpoint artifact exists for this graph.
3. No post-promotion monitor-window artifact exists.
4. Real Glass/Zed consumer execution remains blocked by missing real-consumer executor wiring. The available evidence is headless/offline ACP evidence, not real ACP consumer parity proof.
5. The historical proof `release-proof.local-evidence-flywheel-v1` is preserved as lineage but stale for the current graph.
6. Lane 01 still has pending real capture/regeneration work because the system does not yet deterministically rebuild all `.bag/evidence/**` from live Glass/Zed ACP sessions.
7. Lane 05 still has pending real dogfood benchmark work because the runner refuses non-dry `real_consumer` mutation without a wired executor.
8. Current headless coding quality is still bad: mutating tasks fail closed after generating no file edits. This is no longer a no-write/no-terminal evidence blocker, but it is a real model/executor quality blocker.

Resolved after the original rollup:

- `.bag/evidence/edit-attempt-records.jsonl` now exists with `23` first-class edit attempt records generated from optimizer-visible real ACP corpus manifests.
- `.bag/evidence/scorecards/edit-attempt-projection.json` now exists with `23` projected records.
- The optimizer gate suite no longer reports `edit-policy promotion needs first-class edit attempt telemetry` as a blocking reason.
- The optimizer gate suite no longer reports visible ACP no-write/no-terminal blockers for the current visible corpus.

## Evidence Hashes

Current key artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| `.bag/evidence/release-proof.json` | `9408f518b5660a84df062f448047c74421bd95fa83a9f144bbadd28ce2aed2c7` |
| `.bag/evidence/optimizer/index.json` | `c78c53e5d0b5b20ed72dc50f218304d58c5cb1e5e633d1db4378fba2058a0f00` |
| `.bag/evidence/scorecards/index.json` | `f88df076642ea8f7700f584c4d89988f99e5e96eea807d45d94998108d28d40d` |
| `.bag/evidence/edit-attempt-records.jsonl` | `3fbacc3a90fc7a59871c983538f12b7842b753abd153cdc274e18ece3c7f1b7f` |
| `.bag/evidence/scorecards/edit-attempt-projection.json` | `30c2814194afbecc7c563efc698080c1084b00409c250deeca1409d66f70128f` |
| `.bag/evidence/optimizer/no-write-gate.json` | `b0fa70c6f4de459138cb046eee39204a5d85d86b39ae528a6a8bd2eb451f14c6` |
| `.codex/plan-graphs/live-acp-evidence-readiness-v1/snapshot.json` | `39899da5d8a439f0377b22ba550c808d003e17c90c3d4b1b4d646a2e5799f2f4` |

Release proof summary:

| Check | Status |
| --- | --- |
| `planGraphSnapshot` | passed |
| `evidenceIndexCommand` | passed |
| `scorecardsCommand` | passed |
| `optimizerGatesCommand` | passed |
| `scorecardsGraphMatchesCurrent` | passed |
| `optimizerGraphMatchesCurrent` | passed |
| `historicalProofPreserved` | passed |
| `historicalProofNotReportedAsCurrent` | passed |

Visible no-write gate:

| Metric | Value |
| --- | ---: |
| checked records | `9` |
| passed | `9` |
| blocked | `0` |
| warned | `0` |
| gate status | `pass` |

## Quality Finding

The headless visible quality run is useful negative evidence, but it is not proof that one edit strategy is better than another.

The visible run recorded:

| Metric | Value |
| --- | ---: |
| visible tasks | `9` |
| passed | `0` |
| failed | `8` |
| cancelled | `1` |
| changed files | `0` |
| write tool calls | `0` |
| terminal commands | `8` |
| read tool calls | `14` |

Interpretation: this is fail-closed quality evidence. Coding tasks reached no-edit failure, emitted terminal verification failures, and avoided silent success. It should feed model/executor quality and edit-generation optimization, not no-write/no-terminal promotion blocking.

## Worktree Hygiene And Boundaries

The worktree contains valuable data and should not be cleaned destructively as part of this graph.

Recommended commit boundaries:

1. Core BAG/ACP product code and tests: `src/acp/**`, `src/evidence/**`, `src/replay/**`, `src/optimizer/**`, ACP/runtime tests, package and TS/RsPack config required by the current implementation.
2. Current graph evidence and release proof: `.bag/evidence/**`, `.bag/replay-corpus/**`, `.bag/telemetry/**`, `docs/live-acp-*.md`, `.codex/plans/live-acp-evidence-readiness/*.plan.md`, `.codex/plan-graphs/live-acp-evidence-readiness-v1/snapshot.json`.
3. Worktree hygiene and operator control docs: `docs/live-acp-worktree-hygiene-report.md`, this handoff, and operator-log references.
4. Model benchmark research: `bench/**`, Qwen/DFlash/MLX docs and scripts. Keep separate from the live ACP release review.
5. Trace-GEPA research: `trace-gepa/**`. Preserve and quarantine; do not publish raw datasets without privacy review.
6. Local config and secrets: keep `.mcp.json`, `.env`-style files, provider config, and local operator notes out of commits; review `.mcp.example.json` manually before publishing.

Recommended artifact retention:

- Preserve `.bag/**` and `.codex/plan-graphs/live-acp-evidence-readiness-v1/**`; these are release evidence, not disposable clutter.
- Preserve `trace-gepa/data/**` and sanitized datasets unless explicitly archived or compressed.
- Treat `bench/.venv/**`, `bench/**/node_modules/**`, vendored benchmark dependencies, Python caches, and generated indexes as non-release artifacts, but only delete them after a separate approval because some benchmark runs were expensive.

## Next Graph Handoff

Canonical graph control bundle:

| Field | Value |
| --- | --- |
| graph id | `live-acp-evidence-readiness-v1` |
| selection hash | `6fbc4883fa` |
| plans root | `.codex/plans/live-acp-evidence-readiness` |
| snapshot | `.codex/plan-graphs/live-acp-evidence-readiness-v1/snapshot.json` |
| operator log | `.codex/plan-graphs/live-acp-evidence-readiness-v1/operator-log.md` |

Dependency overlay:

```text
Live ACP Evidence 01 Dogfood Evidence Regeneration -> Live ACP Evidence 02 Current Graph Release Proof Rebuild
Live ACP Evidence 01 Dogfood Evidence Regeneration -> Live ACP Evidence 03 Promotion Readiness Closure
Live ACP Evidence 02 Current Graph Release Proof Rebuild -> Live ACP Evidence 03 Promotion Readiness Closure
Live ACP Evidence 01 Dogfood Evidence Regeneration -> Live ACP Evidence 05 Real Agent Quality Evaluation
Live ACP Evidence 02 Current Graph Release Proof Rebuild -> Live ACP Evidence 06 Readiness Rollup Handoff
Live ACP Evidence 03 Promotion Readiness Closure -> Live ACP Evidence 06 Readiness Rollup Handoff
Live ACP Evidence 04 Dirty Worktree Evidence Hygiene -> Live ACP Evidence 06 Readiness Rollup Handoff
Live ACP Evidence 05 Real Agent Quality Evaluation -> Live ACP Evidence 06 Readiness Rollup Handoff
```

Completed lanes:

- Lane 02: current graph release proof rebuild.
- Lane 04: dirty worktree evidence hygiene.
- Lane 06: blocked-release rollup handoff.

Partially completed lanes with preserved blockers:

- Lane 01: evidence inventory, dogfood pack definition, and validation completed; real capture and full regeneration remain pending.
- Lane 03: blocker enumeration, visible no-write evidence, and first-class edit telemetry completed; hidden holdout final, rollback approval, and monitor window remain pending.
- Lane 05: quality matrix, representative tasks, baseline comparison, and report completed; real consumer dogfood benchmarks remain pending.

Remaining frontier:

- `run-or-script-acp-dogfood-capture`: emit replay-safe live ACP transcripts, edit-attempt records, no-write validation candidates, tool failure spans, and model/codebase/client/profile lineage.
- `regenerate-evidence-artifacts`: rebuild `.bag/evidence/**` deterministically from newly captured dogfood data.
- `prove-hidden-holdout-final-gate`: run hidden holdout only for a frozen candidate, without optimizer leakage.
- `prove-rollback-approval-monitor-window`: publish rollback checkpoint, operator approval, and monitor-window artifacts.
- `run-agent-dogfood-benchmarks`: run real Glass/Zed or equivalent ACP consumer tasks after executor wiring exists.
- `fix-real-edit-generation`: wire a real model/executor path that generates candidate edits, then rerun current visible and hidden gates.

Ownership boundaries for the next graph:

| Boundary | Owner shape |
| --- | --- |
| real consumer executor wiring | `src/replay/**`, scripts for isolated ACP consumer runs, focused real-consumer tests |
| edit attempt telemetry | ACP runtime/edit/write/verify/repair/rollback paths plus `.bag/evidence/edit-attempt-records.jsonl` projection |
| evidence generation | `src/evidence/**`, generated `.bag/evidence/**`, release proof docs |
| optimizer gates | `src/optimizer/**`, but only after evidence artifacts exist |
| worktree hygiene | docs-only or explicit cleanup task; no destructive action without approval |
| benchmark/model research | separate branch or artifact bundle |

Conflict hotspots:

- `.bag/evidence/**` must have one writer at a time because hashes and release proof lineage are sensitive.
- `src/evidence/**` is shared by proof generation, scorecard generation, optimizer gates, and validation.
- `src/replay/**` is shared by real ACP task packs, headless runner, real-consumer runner, and no-write slice generation.
- `src/acp-agent.ts` and ACP runtime modules should not be edited concurrently with telemetry wiring unless ownership is explicit.
- Hidden holdout artifacts must remain evaluation-only and must not be fed into optimizer candidate generation.

Important operator note: plain `frontier --graph-id live-acp-evidence-readiness-v1` still shows plan-level blockers because lanes 01, 03, and 05 intentionally retain pending blocker todos. Use this handoff and the operator log overlay when deciding the next graph, not the coarse frontier alone.

## Reproduction Commands

Validate current evidence:

```bash
npm run --silent bag -- evidence validate --graph-id live-acp-evidence-readiness-v1
```

Regenerate current graph proof without changing promotion readiness:

```bash
npm run --silent bag -- evidence scorecards --write --graph-id live-acp-evidence-readiness-v1
npm run --silent bag -- evidence optimizer-gates --write --graph-id live-acp-evidence-readiness-v1
npm run --silent bag -- evidence release-proof --write --graph-id live-acp-evidence-readiness-v1
npm run --silent bag -- evidence validate --graph-id live-acp-evidence-readiness-v1
```

Inspect exact graph frontier with overlay:

```bash
python /Users/satan/side/experiments/skills/plan-graph/scripts/plan_graph.py frontier \
  --plans-root .codex/plans/live-acp-evidence-readiness \
  --glob '*.plan.md' \
  --graph-id live-acp-evidence-readiness-v1 \
  --write-state \
  --lanes 10 \
  --max-depth 2 \
  --depends 'Live ACP Evidence 01 Dogfood Evidence Regeneration:Live ACP Evidence 02 Current Graph Release Proof Rebuild' \
  --depends 'Live ACP Evidence 01 Dogfood Evidence Regeneration:Live ACP Evidence 03 Promotion Readiness Closure' \
  --depends 'Live ACP Evidence 02 Current Graph Release Proof Rebuild:Live ACP Evidence 03 Promotion Readiness Closure' \
  --depends 'Live ACP Evidence 01 Dogfood Evidence Regeneration:Live ACP Evidence 05 Real Agent Quality Evaluation' \
  --depends 'Live ACP Evidence 02 Current Graph Release Proof Rebuild:Live ACP Evidence 06 Readiness Rollup Handoff' \
  --depends 'Live ACP Evidence 03 Promotion Readiness Closure:Live ACP Evidence 06 Readiness Rollup Handoff' \
  --depends 'Live ACP Evidence 04 Dirty Worktree Evidence Hygiene:Live ACP Evidence 06 Readiness Rollup Handoff' \
  --depends 'Live ACP Evidence 05 Real Agent Quality Evaluation:Live ACP Evidence 06 Readiness Rollup Handoff'
```

Hash current handoff artifacts:

```bash
shasum -a 256 \
  .bag/evidence/release-proof.json \
  .bag/evidence/optimizer/index.json \
  .bag/evidence/scorecards/index.json \
  .bag/evidence/edit-attempt-records.jsonl \
  .bag/evidence/scorecards/edit-attempt-projection.json \
  .bag/evidence/optimizer/no-write-gate.json \
  .codex/plan-graphs/live-acp-evidence-readiness-v1/snapshot.json
```

## Stop Condition

This lane is complete as a blocked-release rollup. It does not make promotion ready, does not edit runtime behavior, and does not weaken gates.
