# BleedingAgent Release Readiness

Date: 2026-05-01

Scope: final evidence rollup for `BleedingAgent Next Release Evidence`.

BleedingAgent should be described as an ACP self-evolving coding-agent harness. It is not a CLI-first coding-agent UI and should not be claimed as production-grade parity with Codex, OpenCode, ForgeCode, Pi, or Oh My Pi. ACP clients provide the visible editor/chat/diff/terminal surface; `bag` provides the runtime, traces, evals, optimizer artifacts, and operator maintenance hooks.

## Status Legend

- Real: implemented in source and covered by local tests or directly exposed by `bag`.
- Tested scaffold: implemented as reusable code and tested offline, but not fully wired into the live ACP coding loop.
- Scaffold: schema, helpers, or docs exist, but runtime closure is incomplete.
- Missing: not implemented yet.
- Future-gated: intentionally out of scope for this release lane.

## Integrated Checklist

| Area | Release state | Evidence | Operator command or check | Notes |
| --- | --- | --- | --- | --- |
| ACP runtime | Real | `src/acp-agent.ts`, `src/acp/surface.ts`, `src/acp/session.ts`, `src/acp/slash-router.ts`, `src/acp/tool-runner.ts`, `tests/bag.test.ts`, `scripts/verify_acp_consumer_setup.ts`, `.bag/acp-consumer-fixtures/local-consumer-validation-latest.json` | `npm run bag -- acp`; `npm run bag -- acp-settings`; `npm run build`; `npm run acp:verify-consumers -- --timeout-ms 45000 --out .bag/acp-consumer-fixtures/local-consumer-validation-latest.json`; `bun test tests/bag.test.ts` | ACP stdio agent supports initialize, sessions, Auto/Chat/Plan/Run, YOLO/Safe, slash commands, compact status updates, cancellation handling, traces, and maintenance commands. Validation includes protocol-level/offline transcript testing and local Glass/Zed-compatible launch-target validation, not desktop rendering automation. |
| Product positioning | Real in docs | `docs/bleeding-agent.md`, this file | N/A | Correct framing is ACP backend plus self-evolving harness. The CLI is for launch, diagnostics, planning pipeline, metrics, and maintenance, not a polished daily-driver TUI. |
| Live edit loop | Real but basic | `src/acp/coding-runner.ts`, `src/acp/coding-generation.ts`, `src/acp/edit-routing.ts`, `src/acp/edit-lifecycle.ts`, `src/acp/workspace-io.ts`, `src/acp/terminal.ts`, `tests/bag.test.ts` | `npm run bag -- acp`, then `/run <task>` from an ACP client | Live Run mode can read files, resolve a policy-backed edit strategy, render the selected edit contract, parse/apply model edit payloads, show diffs, write final content through ACP, run verification, repair, rollback, and persist `coding-trace.json`. The ACP client write boundary still uses `writeTextFile`, so the final transport write is whole-file even when the model-facing edit strategy is not. |
| Edit strategy portfolio | Partly live, still under-evidenced | `src/edit-strategy/*`, `src/eval-harness/edit-strategy-ablation.ts`, `src/optimizer/edit-policy-router.ts`, `tests/edit-strategy-*.test.ts`, `tests/edit-policy-router.test.ts`, `tests/edit-promotion-gates.test.ts`, `tests/bag.test.ts` | `bun test tests/edit-strategy-apply-layer.test.ts tests/edit-strategy-ablation.test.ts tests/edit-policy-router.test.ts tests/edit-promotion-gates.test.ts`; focused ACP edit-routing tests | Whole-file, exact replace, unified diff, apply patch, and hash/range families exist with parse/apply failure classes, protected-path and stale-context checks, ablation, routing, promotion gates, and live routing hooks. The missing evidence is broad real-model/real-client coverage proving which strategies work best per model and codebase. |
| MCP runtime | Tested scaffold | `src/mcp/runtime-tools.ts`, `tests/mcp-runtime-tools.test.ts` | `bun test tests/mcp-runtime-tools.test.ts`; ACP `/mcp` | MCP server metadata can be normalized into canonical optimizer tool specs, rendered into tool contracts, classified by side effect, executed through a runtime bridge, bounded/traced, and converted to optimizer feedback. ACP sessions can list attached MCP servers. Arbitrary ACP-attached MCP tools are not yet generally proxied into the live model loop. |
| Replay evals | Real adapter, tested offline runner | `src/replay/*`, `docs/bleeding-agent-real-replay-dataset.md`, `tests/replay-*.test.ts` | `bun test tests/replay-live-dataset.test.ts tests/replay-capture-extraction.test.ts tests/replay-split-redaction-holdout.test.ts tests/replay-runner-integration.test.ts`; broader replay scenario tests | Live ACP coding runs emit replay captures, captures can be redacted into optimizer-safe replay cases, holdout/raw-local leakage is blocked, and baseline/candidate replay scorecards run offline. Automatic harvesting, oracle strengthening, and GEPA scheduling over all live captures remain follow-on operations work. |
| Eval harness | Real for offline fixtures | `src/eval-harness/*`, `tests/eval-harness-*.test.ts` | `bun test tests/eval-harness-runner.test.ts tests/eval-harness-scorer.test.ts tests/eval-harness-fixtures.test.ts tests/eval-harness-splits.test.ts` | Temp-workspace eval runner, assertions, scorecards, train/dev/holdout split helpers, and edit ablations exist. Promotion-quality results still depend on building a larger real failure corpus. |
| GEPA closed loop | Real operator-safe loop primitives | `src/optimizer/gepa-*`, `src/optimizer/gepa-operations.ts`, `src/optimizer/candidates.ts`, `src/optimizer/promotion.ts`, `docs/bleeding-agent-gepa-operations.md`, `tests/optimizer-gepa-*.test.ts`, `tests/optimizer-promotion.test.ts` | `bun test tests/optimizer-gepa-operations.test.ts tests/optimizer-candidates.test.ts tests/optimizer-gepa-feedback.test.ts tests/optimizer-gepa-runner.test.ts tests/optimizer-gepa-loop.test.ts tests/optimizer-promotion.test.ts tests/optimizer-gepa-checkpoints.test.ts tests/optimizer-gepa-pareto.test.ts` | Readiness gates, feedback bundling, scoped candidate proposal, validation, train/dev/holdout eval gates, active-pointer promotion for new sessions, checkpoints, post-promotion regression detection, and rollback primitives are implemented and tested. This is operator-safe, not a silent auto-promoting daemon. |
| Parallel orchestration | Real contract/evidence substrate | `src/parallel-orchestration.ts`, `docs/bleeding-agent-parallel-orchestration.md`, `tests/parallel-orchestration.test.ts` | `bun test tests/parallel-orchestration.test.ts` | Bounded lane contracts, write-conflict detection, isolation choice, model/policy/risk-aware concurrency, merge verification planning, and optimizer evidence conversion exist. Runtime execution of real parallel model workers through ACP is a future layer over this contract. |
| ACP maintenance polish | Real, dry-run first | `src/acp/maintenance.ts`, `src/acp/slash-router.ts`, `tests/bag.test.ts` | In ACP: `/maintenance status`, `/maintenance eval`, `/maintenance optimize`, `/maintenance promote <candidate-id>`, `/maintenance rollback [checkpoint]` | Maintenance commands are hidden from normal command suggestions. Status/eval/optimize are inspections. ACP promote/rollback are dry-run readiness inspections; actual CLI application remains `bag self-optimize --apply` or `bag apply-optimization <id>`. |
| Profile automation | Tested scaffold | `src/optimizer/codebase-profile.ts`, `tests/optimizer-codebase-profile.test.ts`, `src/optimizer/policy-resolver.ts`, `tests/optimizer-policy-resolver.test.ts` | `bun test tests/optimizer-codebase-profile.test.ts tests/optimizer-policy-resolver.test.ts` | Codebase profiles can derive language/package-manager/verifier/protected-path facts and policy resolution can pin model/codebase/policy lineage. Automatic drift handling and live profile update scheduling are not closed. |
| Knowledge automation | Real substrate | `src/knowledge/*`, `tests/knowledge-*.test.ts` | `bun test tests/knowledge-store.test.ts tests/knowledge-codification.test.ts tests/knowledge-injection.test.ts tests/knowledge-retrieval.test.ts` | Knowledge entries, dedupe/consolidation, AI.md summary generation, retrieval, codification helpers, and untrusted-memory injection boundaries exist. Automatic codification from every successful/failed ACP run still needs runtime scheduling. |
| Trace and metrics | Real | `src/telemetry.ts`, `src/trace-store.ts`, `src/trace-analysis.ts`, `tests/bag.test.ts` | `npm run bag -- metrics`; `npm run bag -- metrics --json`; ACP `/metrics`; ACP `/traces` | Runs persist metrics, events, OpenInference/HALO-style spans, indexes, and optimizer lineage dimensions. More live ACP traces are needed before release claims can be based on operational evidence. |
| Tests | Real | `package.json`, `tests/`, `tests/release-rollup.test.ts` | `npm run typecheck`; `npm test`; `bun test tests/release-rollup.test.ts` | Current scripts are `tsc` typecheck and Bun tests. The release rollup adds a deterministic ACP-style dogfood harness. |
| Docs | Real after this lane | `docs/bleeding-agent.md`, `docs/bleeding-agent-competitive-comparison.md`, this file, `docs/bleeding-agent-operator-runbook.md` | Review markdown files | The docs now distinguish implemented behavior, tested scaffolding, missing closure, and future-gated work. |
| LSP | Future-gated | N/A | N/A | Do not research or implement in this release lane. Mention only as future work after edit/MCP/replay/GEPA closure. |
| Browser automation | Future-gated | N/A | N/A | Do not research or implement in this release lane. Mention only as future work after the core harness loop is closed. |

## Release Readiness Judgment

BleedingAgent is credible as a v0.1 ACP self-evolving harness if the release claim stays narrow:

- start `bag acp` as an ACP stdio backend;
- run chat, planning, and coding tasks through ACP;
- persist trace, metrics, self-evaluation, and artifact data;
- inspect telemetry and traces through CLI or ACP slash commands;
- run offline replay/eval/optimizer tests;
- generate and inspect safe self-optimization candidates;
- apply only bounded local optimizer artifacts from explicit CLI commands.

BleedingAgent is not ready to claim:

- production-grade daily-driver parity with Codex/OpenCode/ForgeCode/Pi/Oh My Pi;
- fully proven optimizer-aware edit strategy quality across real models, real ACP clients, and real project traces;
- arbitrary MCP tool use inside the live model loop;
- automatic failed-session-to-replay conversion;
- autonomous GEPA promotion from real live traces;
- post-promotion automatic rollback from live regressions;
- LSP or browser automation support.

## Known Limitations

1. Live editing has optimizer-aware strategy routing, but the ACP write transport still writes final file content through `writeTextFile`, and current release dogfood only exercises the whole-file write-boundary path deterministically.
2. MCP runtime bridge pieces are implemented and tested, but ACP-attached MCP tools are not yet universally available to the model as live runtime tools.
3. Replay evals include a live-capture adapter and synthetic/redacted regression packs, but they do not yet represent a large corpus of real ACP consumer failures gathered over time.
4. ACP compatibility is tested through fake-client transcripts plus local Glass/Zed-compatible launch-target validation. Broad desktop rendering automation for named consumers has not been run in this lane.
5. GEPA candidate generation, gating, promotion, and rollback primitives are implemented. Continuous autonomous scheduling and silent auto-promotion are intentionally not claimed.
6. Codebase profile and knowledge automation exist as tested substrates, but not as a complete always-on lifecycle.
7. Parallel orchestration has a contract/evidence substrate, but real ACP-launched parallel model workers are not yet productized.
8. Self-optimization applies safe local artifacts only; it must not be described as free-form self-rewriting.
