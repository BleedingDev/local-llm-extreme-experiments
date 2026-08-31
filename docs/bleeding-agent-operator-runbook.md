# BleedingAgent Operator Runbook

Date: 2026-05-01

This runbook is for operating BleedingAgent as an ACP self-evolving coding-agent harness. Normal
users should interact through any compatible ACP client; Zed and Glass are named setup examples, not
the product boundary. Operators use `bag` and hidden ACP maintenance commands to inspect traces,
evals, candidates, and rollback readiness.

## Preflight

Install dependencies first if the checkout is fresh:

```bash
npm install
```

Verify the TypeScript and test harness:

```bash
npm run typecheck
npm test
```

Check runtime configuration:

```bash
npm run bag -- doctor
npm run bag -- metrics
npm run bag -- metrics --json
```

Build the packaged `bag` binary when you need `dist/index.js`:

```bash
npm run build
```

## Start ACP

For local development from this repo:

```bash
npm run bag -- acp
```

For an installed/built package:

```bash
bag acp
```

Print the generic ACP launch snippet and named examples:

```bash
npm run bag -- acp-settings
```

Print only the Zed settings object when configuring Zed:

```bash
npm run bag -- acp-settings zed
```

Expected Zed shape:

```json
{
  "agent_servers": {
    "bleeding-agent": {
      "command": "bag",
      "args": ["acp"]
    }
  }
}
```

Keep the ACP process working directory at the project root so `.bag` telemetry and relative file paths resolve correctly.

Validate the local Glass/Zed-compatible launch target after changing the build or settings:

```bash
npm run build
npm run acp:verify-consumers -- --timeout-ms 45000 --out .bag/acp-consumer-fixtures/local-consumer-validation-latest.json
```

This verifies installed app bundles, the local `agent_servers.bleeding-agent` command, ACP
`initialize`/`session/new`/prompt handshake, and no-side-effect chat behavior. It does not automate
desktop rendering.

## ACP User Commands

Normal command surface:

- `/run <task>`: run a coding-agent task with reads, edits, verification, traces, and artifacts.
- `/plan <task>`: run read-only interview/PRD/DAG/report work.
- `/chat`: force no-side-effects chat.
- `/auto`: return to model-routed Auto mode.
- `/yolo`: allow file writes and terminal commands without prompts.
- `/safe`: require approval for file writes and terminal commands.
- `/skills`: list local skills visible to the agent.
- `/mcp`: show MCP servers attached to the current ACP session.
- `/metrics`: show telemetry, metrics, span, index, and artifact locations.
- `/traces`: show HALO-style trace dataset overview and recent failing traces.

Auto mode is the default. It should route greetings to Chat, read-only report work to Plan, and mutation/verification work to Run. Temporary Auto routes should restore Auto after the turn.

## Maintenance Commands

Maintenance commands are intentionally hidden from normal ACP command suggestions:

```text
/maintenance status
/maintenance eval
/maintenance optimize
/maintenance promote <candidate-id>
/maintenance rollback [checkpoint]
```

Semantics:

- `/maintenance status` reads optimizer registry state, session pinning, and background optimization evidence. No promotion or rollback is applied.
- `/maintenance eval` summarizes configured eval split metadata. It does not run evals.
- `/maintenance optimize` computes the existing safe optimization report from persisted metrics/manifests. It does not write candidate source changes.
- `/maintenance promote <candidate-id>` is a dry-run readiness inspection. It does not move the active pointer.
- `/maintenance rollback [checkpoint]` is a dry-run rollback target inspection. It does not restore the pointer.

Actual CLI-side self-optimization commands:

```bash
npm run bag -- optimize
npm run bag -- self-optimize
npm run bag -- self-optimize --apply
npm run bag -- apply-optimization <candidate-id>
```

`self-optimize --apply` and `apply-optimization` write only safe local optimizer artifacts such as `bag.config.json`, `.bag/tool-guidance.md`, and `.bag/self-improvement-plan.md`. They do not edit project source files.

## Inspect Artifacts And Traces

Primary locations:

- `.bag/runs/<run-id>/manifest.json`
- `.bag/runs/<run-id>/self-evaluation.json`
- `.bag/runs/<run-id>/optimization.json`
- `.bag/runs/<run-id>/coding-trace.json`
- `.bag/runs/<run-id>/planning-trace.json`
- `.bag/telemetry/events.jsonl`
- `.bag/telemetry/metrics.json`
- `.bag/telemetry/spans.jsonl`
- `.bag/telemetry/spans.jsonl.index.jsonl`
- `.bag/telemetry/spans.jsonl.index.meta.json`
- `.bag/optimizations/<candidate-id>.json`
- `.bag/optimizations/<candidate-id>.md`
- `.bag/optimizer/records/`
- `.bag/optimizer/checkpoints/`

Useful commands:

```bash
npm run bag -- metrics
npm run bag -- metrics --json
```

Inside ACP:

```text
/metrics
/traces
/maintenance status
```

Use `/traces` for a bounded overview. Do not dump large raw span files into prompts unless you have narrowed the trace or span IDs first.

## Replay And Eval Checks

There is no dedicated public `bag replay` CLI yet. Run replay and eval checks through the existing Bun test entry points.

Replay corpus checks:

```bash
bun test tests/replay-live-dataset.test.ts tests/replay-capture-extraction.test.ts tests/replay-split-redaction-holdout.test.ts tests/replay-runner-integration.test.ts
bun test tests/replay-runner-integration.test.ts
bun test tests/replay-routing-scenarios.test.ts tests/replay-tool-call-scenarios.test.ts tests/replay-edit-failure-scenarios.test.ts
bun test tests/replay-split-redaction-holdout.test.ts
```

Use `src/replay/dataset.ts` for explicit capture-to-replay extraction. By default it redacts local
captures before building replay cases; `includeRawLocalContent: true` remains local-only and is
rejected by optimizer input selection.

Eval harness checks:

```bash
bun test tests/eval-harness-runner.test.ts tests/eval-harness-scorer.test.ts tests/eval-harness-fixtures.test.ts tests/eval-harness-splits.test.ts
```

Edit strategy eval and promotion checks:

```bash
bun test tests/edit-strategy-apply-layer.test.ts tests/edit-strategy-ablation.test.ts tests/edit-policy-router.test.ts tests/edit-promotion-gates.test.ts
```

GEPA closed-loop checks:

```bash
bun test tests/optimizer-gepa-operations.test.ts
bun test tests/optimizer-gepa-loop.test.ts tests/optimizer-gepa-runner.test.ts tests/optimizer-gepa-feedback.test.ts tests/optimizer-gepa-checkpoints.test.ts tests/optimizer-promotion.test.ts
```

Parallel orchestration contract checks:

```bash
bun test tests/parallel-orchestration.test.ts
```

MCP runtime checks:

```bash
bun test tests/mcp-runtime-tools.test.ts
```

Full regression:

```bash
npm run typecheck
npm test
```

## Promotion Workflow

1. Collect evidence from ACP runs, metrics, traces, and replay/eval failures.
2. Inspect current state:

```text
/maintenance status
/maintenance eval
/maintenance optimize
```

3. Generate a candidate:

```bash
npm run bag -- self-optimize
```

4. Review `.bag/optimizations/<candidate-id>.json` and `.bag/optimizations/<candidate-id>.md`.
5. Run targeted tests for the affected area, then run:

```bash
npm run typecheck
npm test
```

6. Inspect promotion readiness in ACP:

```text
/maintenance promote <candidate-id>
```

7. Apply only when the candidate is bounded, reviewed, and backed by eval evidence:

```bash
npm run bag -- apply-optimization <candidate-id>
```

Promoted policies are intended to apply to new ACP sessions. Current sessions remain pinned to the policy resolved when they started.

## Rollback Workflow

Inspect available rollback state:

```text
/maintenance rollback
```

Inspect a specific checkpoint:

```text
/maintenance rollback <checkpoint-file>
```

The ACP rollback command is a readiness inspection only. Runtime rollback primitives exist in `src/optimizer/promotion.ts`, but this release lane should not claim an operator CLI that performs active-pointer rollback unless one is added and verified.

When a promoted candidate regresses:

1. Stop starting new ACP sessions against the bad candidate.
2. Preserve `.bag/optimizer/checkpoints/` and the relevant `.bag/telemetry/` files.
3. Record the failure as replay/eval evidence if possible.
4. Use the rollback primitives or a targeted follow-up implementation lane to restore the previous active pointer.
5. Add the regression to replay/eval coverage before re-promoting.

## Operational Boundaries

- Treat BleedingAgent as an ACP harness, not a finished TUI/CLI coding product.
- Do not promise LSP or browser automation in this release.
- Do not describe MCP as fully live in the model loop yet. Say the runtime bridge is tested and ACP server visibility exists, while live arbitrary tool proxying remains unfinished.
- Do not claim edit strategy quality is proven across real models/clients yet. Live ACP runs route through rendered edit contracts, but the final ACP transport write is still full-file `writeTextFile`, and the deterministic release dogfood covers the whole-file boundary path.
- Do not promote from visible train/dev evidence alone. Hidden holdout and post-promotion monitoring are required for strong promotion claims.
- Do not treat self-optimization as source-code mutation. It writes bounded local optimizer/config/guidance artifacts.
