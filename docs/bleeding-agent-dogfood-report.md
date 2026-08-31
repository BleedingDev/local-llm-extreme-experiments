# BleedingAgent Dogfood Report

Date: 2026-05-01

Scope: release evidence dogfood/regression lane for `BleedingAgent Next Release Evidence`.

Completed todo ids:

- `next-release-dogfood`
- `next-release-regression`

## Summary

The integrated ACP-style release-rollup harness passed deterministically without external model or network dependency. It covered session startup, greeting/chat, maintenance plan/report progress, ACP write-boundary edit with Safe-mode permission, terminal verification, trace inspection, and maintenance optimization dry run.

The local named ACP consumer launch target was also verified after a fresh `rspack` build. Glass and Zed are installed locally, the configured `agent_servers.bleeding-agent` command resolves to `node dist/index.js acp`, the ACP initialize/session/prompt handshake succeeded, and the smoke prompt `/chat Ahoj, co umis?` performed no file reads, file writes, terminal calls, or permission requests.

No ACP runtime changes were required.

## ACP-Style Dogfood Transcript

Harness: `tests/release-rollup.test.ts`

Workspace: temporary local workspace created by the test under the OS temp directory.

Consumer: in-process ACP-style fake connection, not desktop automation. The fake connection records ACP `sessionUpdate`, `requestPermission`, `readTextFile`, `writeTextFile`, and terminal calls while using a real local child process for terminal verification.

Scenario:

1. Initialize ACP agent.
   - Expected: agent advertises `bleeding-agent` identity and session capabilities.
   - Observed: initialization returned `agentInfo.name = "bleeding-agent"` and resume-capable session metadata.

2. Start a new session.
   - Expected: session is created and command surface is published.
   - Observed: session id returned; `available_commands_update` emitted.

3. Chat/greeting.
   - Prompt: `/chat`
   - Prompt: `hello, show the user-facing ACP surface only`
   - Expected: no file writes, no terminal calls, user-facing capability response only.
   - Observed: response contained `Ahoj. Jsem BleedingAgent ACP coding agent`; recorded writes = 0; recorded terminals = 0.

4. Plan/report via maintenance status.
   - Prompt: `/maintenance status`
   - Expected: ACP plan/tool progress for a read-only maintenance inspection.
   - Observed: plan updates included `Inspect optimizer registry`; final agent text contained `Maintenance optimizer status:`.

5. Code edit through ACP write boundary.
   - Prompt: `/safe`
   - Fixture before: `export const releaseRollup = 'before';`
   - Edit path: `src/rollup-fixture.ts`
   - Expected: preview through edit strategy layer, Safe-mode permission request, ACP `writeTextFile`, and consistent post-apply readback.
   - Observed: edit result `ok=true`, `editStrategyId=edit.whole-file.acp-write.v1`, `editStatus=applied`; one permission request recorded; one ACP write recorded; post-apply consistency was `consistent`.

6. Terminal verification.
   - Command: current Node executable with `-e "console.log('release-rollup-verification')"`
   - Expected: Safe-mode permission request, ACP terminal execution, exit code capture.
   - Observed: terminal result `exitCode=0`; output contained `release-rollup-verification`; total permission requests increased to 2.

7. Trace inspection.
   - Prompt: `/traces`
   - Expected: HALO-style trace summary is visible after telemetry-producing edit and terminal steps.
   - Observed: agent text contained `HALO-style trace dataset:` and a spans summary.

8. Maintenance optimization dry run.
   - Prompt: `/maintenance optimize`
   - Expected: no promotion or mutation; report computed from existing metrics/manifests.
   - Observed: agent text contained `Maintenance optimize report:` and completed tool output included `bag.maintenance.optimize`.

## Regression Evidence

Commands run:

```sh
npm run build
npm run acp:verify-consumers -- --timeout-ms 45000 --out .bag/acp-consumer-fixtures/local-consumer-validation-latest.json
bun test tests/release-rollup.test.ts
bun test tests/replay-live-dataset.test.ts tests/replay-capture-extraction.test.ts tests/replay-split-redaction-holdout.test.ts tests/replay-runner-integration.test.ts tests/replay-routing-scenarios.test.ts tests/replay-tool-call-scenarios.test.ts tests/replay-edit-failure-scenarios.test.ts tests/edit-strategy-apply-layer.test.ts tests/edit-policy-router.test.ts tests/optimizer-tool-renderer.test.ts tests/optimizer-gepa-operations.test.ts tests/optimizer-gepa-runner.test.ts tests/mcp-runtime-tools.test.ts tests/provider-role-model.test.ts tests/parallel-orchestration.test.ts tests/release-rollup.test.ts
npm run typecheck
npm test
```

Results:

- Build: passed.
- Local named ACP consumer launch-target validation: passed; Glass installed, Zed installed, handshake ok, no side effects.
- Focused release-rollup harness: 1 pass, 0 fail.
- Targeted replay/edit/tool/provider/GEPA/orchestration suite: 78 pass, 0 fail.
- Full local suite: 307 pass, 0 fail.
- Typecheck: passed.

## Follow-Ups And Risks

- The release-rollup harness intentionally avoided external model/network calls. A live `/plan <task>` or `/run <task>` model-generated end-to-end run remains an operator acceptance exercise, not a deterministic CI gate.
- The dogfood lane used an ACP-style harness plus launch-target validation, not Zed/Glass desktop rendering automation. That matches this lane's requirement, but consumer UI integration should remain covered by separate consumer smoke checks.
- Maintenance optimization was validated as a dry run. Promotion and rollback execution were not performed in this lane.

Changed paths:

- `tests/release-rollup.test.ts`
- `docs/bleeding-agent-dogfood-report.md`
- `.bag/acp-consumer-fixtures/local-consumer-validation-latest.json`
