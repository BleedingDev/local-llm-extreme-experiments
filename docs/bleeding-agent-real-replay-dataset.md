# BleedingAgent Real Replay Dataset

Date: 2026-05-01

This note documents the Real Replay Dataset lane for the ACP self-evolving harness. The goal is to turn live ACP coding-agent evidence into optimizer-safe replay cases without leaking private workspace content or hidden holdout examples into GEPA/proposer prompts.

## Implemented Contract

- Live ACP coding runs emit `replay-capture.json` artifacts that link user prompt, route decision, ACP consumer capabilities, model/provider/profile lineage, file-read hashes, edit attempts, MCP/tool calls, terminal verification commands, artifact refs, trace IDs, and outcome evidence.
- `src/replay/capture.ts` defines the normalized ACP replay capture schema and source/ref grouping.
- `src/replay/redaction.ts` converts raw local captures into optimizer-safe captures by default.
- `src/replay/dataset.ts` converts a redacted capture into a replay eval case skeleton with split metadata, oracle strength, source refs, routing summary, observed failures, and redaction summary.
- `src/replay/enforcement.ts` keeps hidden holdout and needs-review/raw-local cases out of proposer prompts, GEPA feedback, and optimizer selections.
- `src/replay/runner.ts` runs baseline versus candidate replay suites in temporary workspaces and emits scorecards for promotion gates.

## Redaction Rules

Default replay redaction is conservative:

- user/developer/assistant/system roles are preserved;
- prompt/tool/result text is secret-scrubbed and excerpt-capped;
- file-read content defaults to hash-only when a content hash exists;
- paths under a configured root become relative paths;
- absolute paths outside the configured root become stable path hashes;
- terminal output and artifacts remain hash/artifact references;
- explicit `includeRawLocalContent: true` is allowed only as a local opt-in and remains excluded from optimizer input.

## Coverage

Current replay packs and tests cover:

- greeting/no-side-effect routing regressions;
- read-only report routing;
- Auto mode temporary restoration;
- Safe/YOLO permission denial behavior;
- cancellation-like terminal failure evidence;
- edit parse/apply/stale-context failures;
- fallback success while preserving primary failure evidence;
- post-apply inconsistency and self-detected regression evidence;
- tool malformed-argument, timeout, truncation, permission, retry, and MCP lineage cases;
- terminal verification failures;
- hidden holdout exclusion.

## Evidence

Run the focused replay checks:

```bash
bun test tests/replay-live-dataset.test.ts tests/replay-capture-extraction.test.ts tests/replay-split-redaction-holdout.test.ts tests/replay-runner-integration.test.ts
```

Current result:

- 13 tests passed.
- 0 failures.
- 70 assertions.

Also run the broader replay packs:

```bash
bun test tests/replay-routing-scenarios.test.ts tests/replay-tool-call-scenarios.test.ts tests/replay-edit-failure-scenarios.test.ts
```

## Boundary

This lane closes the dataset adapter and safety boundary. It does not yet claim autonomous scheduling that continuously harvests every `.bag/runs/*/replay-capture.json`, ranks captures, strengthens weak oracles, and starts GEPA automatically. That operational loop belongs to the GEPA Operations and Release Evidence lanes.

Real-model quality by edit strategy, provider profile, ACP consumer, and codebase must be learned from accumulated traces and replay scorecards over time. It should not be hardcoded from external leaderboards.
