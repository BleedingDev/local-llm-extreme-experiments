# Live ACP Evidence Regeneration Report

Date: 2026-05-05
Graph: `live-acp-evidence-readiness-v1`
Selection hash: `6fbc4883fa`
Lane: `Live ACP Evidence 01 Dogfood Evidence Regeneration`

## Summary

The current repo already contains useful headless ACP dogfood evidence under `.bag/replay-corpus/**`, including visible train/dev transcripts, replay exports, stability scorecards, trace scorecards, and replay source-adapter captures. That evidence is reusable as observed local headless ACP evidence, but it is not enough to claim fresh real-consumer ACP dogfood from Glass/Zed.

The current `.bag/evidence/**` artifacts validate cleanly, but they still point at the older `local-evidence-flywheel-v1` proof slot. They should be treated as stale proof data for this graph, not as a regenerated release proof for `live-acp-evidence-readiness-v1`.

No `.bag/evidence/**` JSON was hand-edited. No source files were edited.

## Current Evidence Inventory

Current `.bag/evidence/**` files:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `.bag/evidence/index.jsonl` | 33973 | `02cf3ee94a69ab3a7c666f142984d2e01a5c092d9334f1f4360562ce9efa1d4d` |
| `.bag/evidence/index.schema.json` | 7091 | `fd903668dcc713a79626b29af16b4a8d20b10b07b57ac54c550c48d3a80d3561` |
| `.bag/evidence/optimizer/artifact-lineage-contract.json` | 17469 | `a3ec5fec1293563a3ea79493f92a2eba08dc8dee000dea76f9b39c24b5e80892` |
| `.bag/evidence/optimizer/index.json` | 2855 | `9bef951064d03f9d5da986242595f739595f2c7d1b4d3b6384076240a6e58785` |
| `.bag/evidence/optimizer/input-slices.json` | 16836 | `50be9b595c5122a53806115dc4e5be3a29a59846ef24b4a7461aa63b35e4ba36` |
| `.bag/evidence/optimizer/policy-gates.json` | 18214 | `bb32c591b419d468073775877ea93c044f944bf44ca3b84952e0806299d43b3e` |
| `.bag/evidence/optimizer/scheduler-readiness.json` | 15669 | `4074ab42aa1d1f8e20f6ff5981c52e5fd957b77af7e2173b038b99040f1d51e0` |
| `.bag/evidence/release-proof.json` | 2303 | `7dc8fc228b833dc86daefa999c60561c21352ad299c02652eaaa193297ae5512` |
| `.bag/evidence/retention-policy.json` | 11975 | `97a211418f95f9720617f17032441b46a142f4cb8836225dafeb8f5fbcc9aef3` |
| `.bag/evidence/schema-audit.json` | 300248 | `fc4f8834b8e400463f4f1f31ddbc3f4fa7ad366710fe360aa0365fa2000337a1` |
| `.bag/evidence/scorecards/benchmark-results.json` | 19569 | `129cbf1832ff569276c29650dbcdf8b44c4ee16d87e5317350a2746a653dd2e0` |
| `.bag/evidence/scorecards/edit-strategy.json` | 21787 | `755ad92025c1af02e69417afba319e5a460cae5f7b7f6452612ef484ff5af87a` |
| `.bag/evidence/scorecards/index.json` | 2590 | `56f7a4a6195d6bb727ff7416aa9fdd82ea5a3754383f98fd004f4b8f848c3e41` |
| `.bag/evidence/scorecards/recovery-failure.json` | 23126 | `81fb53ca0a8d5daf80c02f845d43b2015305c8ae1545a4da177cb636d028e9ec` |
| `.bag/evidence/scorecards/tool-routing.json` | 17409 | `4a4091acd74ce6bc9f6fa57b33771841f66ae85df9573563696ac5c118980c43` |

`bag evidence index` reports 32 evidence records across 11 families:

- 26 `source` records
- 6 `slice` records
- key ACP records: `evidence.acp.replay-index`, `evidence.acp.visible-run`, `evidence.acp.adapter-replay-export`

The scorecard suite is `scorecard-suite.local-evidence-flywheel-v1`, not this graph. It exposes:

- `tool-routing-scorecard`
- `scorecard.edit-strategy`
- `recovery-failure`
- `scorecard.benchmark-results`

The optimizer gate suite is `optimizer-gate-suite.local-evidence-flywheel-v1`, not this graph. It reports:

- `candidateGeneration=allowed_as_scoped_dry_run`
- `autoPromotion=blocked`
- `promotionReady=false`

The release proof is `release-proof.local-evidence-flywheel-v1`, with selection hash `06eeb209cb`. That makes it stale for `live-acp-evidence-readiness-v1`.

## Reusable Observed Evidence

Reusable local observed evidence exists under `.bag/replay-corpus/**`.

Headless ACP corpus files include:

- `.bag/replay-corpus/index.jsonl`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504*/**`
- `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/**`
- `.bag/replay-corpus/source-adapters/adapter-replay-export/**`

The visible headless run contains 9 visible train/dev task results:

- 6 train
- 3 dev
- 8 failed
- 1 cancelled

Representative visible transcript hashes:

| Transcript | SHA-256 |
| --- | --- |
| `real-acp.task.simple-edit-greeting.json` | `44b2b5a7e13753bc68b9e94d426fe9a79be675243375956be0d9f291f977e356` |
| `real-acp.task.greenfield-slugify.json` | `065cd948246bbde8ba1eb76dc789515ef7423482b8a61bc12a33370df576dd3a` |
| `real-acp.task.cart-bugfix-fail-to-pass.json` | `2b666a35dbda38afa47aaa71cf9207f724dad0357a5de4d72d380d3765791e32` |
| `real-acp.task.protected-path-doc.json` | `8758768a3bd00e90facf1c1d04dd6ba9163cc1084215d586d206b5af0a6c44ae` |
| `real-acp.task.cancellation-mid-edit.json` | `c70998274b0f6ffb0364b4ab377c32eec72395726302c37d84a6c76b0fff8b3f` |
| `real-acp.task.applied-but-broken-import.json` | `80bd455a40601ec694d333fa4d2504cfc85afa4b05512e406ada65bdcbd7e4c9` |
| `real-acp.task.verifier-skip-docs.json` | `2d0524bac8a22b5b38aac7491a68f9cc0fe256b2097e42387ff3cc9c2409553e` |
| `real-acp.task.stale-context-anchor.json` | `f308cc3f6efb5f9b6918b0ccffce8a8d036658a78adc6f2cc91cea666ca31aa4` |
| `real-acp.task.user-correction-scope.json` | `247d2f7fabc024c49d2c83403872e6e0efee09c0f1ac7a16a2f869cfdd302ec9` |

The visible run artifacts:

- manifest: `8facb73c60cc12c0c89e92a8a935a6ef5f4a047646329f0aabb26df6b890c7a0`
- replay export: `62dd12ba803142dcbd593dff66e77d395f5419e6d5701e97e06ed69d45fa3e9d`
- run index: `836aa7d66444fd15af453162bee2e996c57872ee9ea9f46698a8d425f6640819`
- stability scorecard markdown: `4f0e5f13468e944009882c268db1a7da3ada36c02a7f083a88a002ec481d2477`
- trace scorecards markdown: `1f18d094a8955f36e5f28e365f2fe26480f6a8e33d41b0c8e2fddf5ede07ec7b`

## Stale Or Insufficient Evidence

These artifacts validate but must not be treated as fresh release proof for this graph:

- `.bag/evidence/release-proof.json`
- `.bag/evidence/scorecards/index.json`
- `.bag/evidence/optimizer/index.json`
- scorecard JSON and markdown documents that name `local-evidence-flywheel-v1`
- optimizer gate JSON and markdown documents that name `local-evidence-flywheel-v1`

Current missing first-class evidence:

- `.bag/evidence/edit-attempt-records.jsonl` does not exist, so `bag evidence scorecards` skips edit-attempt projection.
- No real Glass/Zed consumer session was driven in this lane.
- No deterministic command currently rebuilds `.bag/evidence/**` from `.bag/replay-corpus/**` for graph `live-acp-evidence-readiness-v1`.

## Dogfood Task Pack

The repo already defines a representative ACP dogfood task pack in `src/replay/real-acp-task-pack.ts`. Tests prove it covers 12 labels:

- `simple_edit`
- `greenfield_workspace`
- `bugfix_fail_to_pass`
- `refactor`
- `stale_context`
- `protected_path`
- `cancellation`
- `rollback`
- `applied_but_broken`
- `verifier_skip`
- `mcp_tool_failure`
- `user_correction`

The pack is split deterministically:

- train: 6
- dev: 3
- holdout: 3

Visible optimizer input is train/dev only. Holdout tasks are explicitly excluded from optimizer-visible input.

Coverage against the lane requirement:

| Requirement | Covered by pack | Current evidence status |
| --- | --- | --- |
| read-only chat | partially | routing/no-write fixtures cover read-only behavior; real-consumer chat transcript still missing |
| planning | partially | routing scenarios and ACP mode tests exist; real-consumer planning transcript still missing |
| file edits | yes | headless visible transcripts include edit tasks |
| MCP/tool calls | yes | task pack includes `mcp_tool_failure`; tool-call scenario tests pass |
| terminal verification | yes | task outcomes include verifier command assertions and terminal evidence tests |
| repair | yes | scorecard and replay tests cover repair signals |
| rollback | yes | task pack includes rollback holdout; holdout not optimizer-visible |
| cancellation | yes | visible run includes cancellation evidence |
| no-write failures | yes | no-write oracle and slice tests pass |

## Commands Run

Inventory and hashes:

```bash
find .bag/evidence -maxdepth 3 -type f
find .bag/evidence -type f -print0 | xargs -0 shasum -a 256
find .bag/evidence -type f -print0 | xargs -0 wc -c
find .bag/replay-corpus -maxdepth 4 -type f
find .bag/replay-corpus/real-acp-runs -type f -print0 | xargs -0 shasum -a 256
```

Evidence commands:

```bash
npm run bag -- evidence index
npm run bag -- evidence scorecards
npm run bag -- evidence optimizer-gates
npm run bag -- evidence release-proof
npm run bag -- evidence validate
```

Replay and validation tests:

```bash
bun test tests/replay-real-acp-task-pack.test.ts tests/replay-real-acp-headless-executor.test.ts tests/replay-real-acp-runner.test.ts tests/replay-real-acp-index.test.ts tests/replay-real-acp-scorecard.test.ts tests/replay-real-acp-trace-scorecards.test.ts tests/replay-split-redaction-holdout.test.ts tests/replay-capture-extraction.test.ts tests/replay-real-acp-redaction.test.ts

bun test src/replay/no-write-validation.test.ts src/replay/no-write-slice.test.ts tests/replay-tool-call-scenarios.test.ts tests/replay-edit-failure-scenarios.test.ts tests/replay-routing-scenarios.test.ts
```

## Verification Results

`npm run bag -- evidence index`:

- passed
- 32 index records
- 11 evidence families
- no missing slice source references

`npm run bag -- evidence scorecards`:

- passed
- 4 scorecards
- skipped edit-attempt projection because `.bag/evidence/edit-attempt-records.jsonl` is missing

`npm run bag -- evidence optimizer-gates`:

- passed
- 4 optimizer contracts
- `promotionReady=false`
- auto-promotion blocked

`npm run bag -- evidence release-proof`:

- passed
- validates old `release-proof.local-evidence-flywheel-v1`
- does not prove this graph

`npm run bag -- evidence validate`:

- passed
- `promotionReady=false`
- blocking reasons:
  - `edit-policy promotion needs first-class edit attempt telemetry`
  - `hidden holdout final gate is not ready for a frozen candidate`
  - `operator approval and rollback checkpoint are required`
  - `post-promotion-monitor-window is unsatisfied`
  - `visible ACP no-write/no-terminal validation must be represented`

Replay task-pack and headless-corpus tests:

- 34 pass
- 0 fail

No-write, routing, edit-failure, and tool-call tests:

- 23 pass
- 0 fail

## Regeneration Result

No `.bag/evidence/**` artifact was regenerated in this lane.

Reason: the current `bag evidence` commands validate or wrap existing artifacts. Their own `writes` sections say:

- index: `action=none`
- scorecards index: `action=none`
- optimizer index: `action=none`
- release proof: `action=none`
- edit-attempt projection: skipped because first-class edit-attempt records are absent

Running `--write` would not be an honest fix for this graph without a deterministic builder that projects `.bag/replay-corpus/**` into new `.bag/evidence/**` artifacts for `live-acp-evidence-readiness-v1`.

## Blockers

Lane 02, current graph release proof rebuild, is blocked on:

- a deterministic release proof builder that targets `live-acp-evidence-readiness-v1`
- validated command payloads whose `graphId` is this graph, not `local-evidence-flywheel-v1`
- a canonical mapping from `.bag/replay-corpus/**` to `.bag/evidence/**`

Lane 03, promotion readiness closure, is blocked on:

- first-class `.bag/evidence/edit-attempt-records.jsonl`
- a visible ACP no-write/no-terminal validation artifact in `.bag/evidence/**`
- hidden holdout final gate evidence
- rollback checkpoint and operator decision evidence
- post-promotion monitor window evidence

Lane 05, real agent quality evaluation, is partially unblocked by the existing headless task pack and replay tests, but still blocked for real-consumer claims on:

- Glass/Zed ACP consumer session capture
- real ACP client metadata and capabilities
- real edit/tool/terminal traces from the consumer substrate
- cancellation and rollback captured from the actual consumer, not only headless harness artifacts

## Recommended Next Work

1. Add a deterministic evidence materializer that reads `.bag/replay-corpus/real-acp-runs/**` plus source-adapter captures and writes graph-scoped `.bag/evidence/**` artifacts.
2. Emit first-class edit attempt records from ACP coding runs into `.bag/evidence/edit-attempt-records.jsonl`.
3. Promote the no-write slice output into a named evidence artifact instead of leaving it only as runtime/test behavior.
4. Add a real-consumer dogfood capture path for Glass/Zed that records ACP session id, client profile, capabilities, transcript path, edit attempts, tool failures, terminal commands, cancellation, rollback, and mode decisions.
5. Rebuild release proof only after artifacts name `live-acp-evidence-readiness-v1` and the hidden holdout remains sealed from optimizer input.
