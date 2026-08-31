# BleedingAgent Quality Execution Release Proof

Date: 2026-05-04

Graph: `bleeding-agent-quality-execution-v1`
Selection hash: `e4bb63af71`

This report closes the quality execution graph for BleedingAgent as an ACP coding-agent backend plus self-evaluation and self-optimization harness. It does not claim that BleedingAgent is already a polished replacement for Codex CLI, Claude Code, ForgeCode, Pi, Oh My Pi, or OpenCode. The strong claim is narrower: the ACP backend, replay/eval substrate, edit strategy telemetry, MCP live-loop bridge, and GEPA-style optimizer primitives are now implemented and covered well enough for controlled operator use.

## Executive Judgment

BleedingAgent is credible as a v0.1 ACP self-evolving coding-agent harness.

It can be launched by ACP consumers, route normal chat without project side effects, run coding/planning flows, apply edits through a measured strategy layer, record traces and replay evidence, evaluate candidate policy changes, and promote/rollback bounded optimizer artifacts through gates.

It is not yet proven as a daily-driver product at the level of mature agents. The remaining high-risk gaps are real-client/live-model breadth, arbitrary live MCP transport wiring, large real replay corpora, and optimizer artifacts that demonstrate stable full-eval uplift rather than only infrastructure correctness.

## Main Verification

| Command | Result | What It Proves |
| --- | --- | --- |
| `npm run typecheck` | Passed | Core and test TypeScript projects typecheck. |
| `bun test tests` | `557 pass`, `0 fail`, `3488 expect()` calls across `87` files | Full Bun regression suite, including ACP, replay, eval harness, edit strategies, MCP runtime, GEPA optimizer, promotion, knowledge, provider, and release rollup coverage. |
| `python -m pytest trace-gepa/agent_opt/rag/test_rag.py -q` | `2 passed`, `1 skipped` | Scoped trace-GEPA RAG gate remains healthy in the lightweight environment. |
| `.venv-gepa/bin/python -m pytest trace-gepa/agent_opt/rag/test_rag.py trace-gepa/agent_opt/oracle/test_oracle.py trace-gepa/agent_opt/persona/test_persona.py trace-gepa/tests/test_adapter.py trace-gepa/tests/test_reflection.py trace-gepa/tests/test_verifiers.py -q` | `67 passed` | Full documented trace-GEPA Python gate passes in the project venv. |
| `npm run acp:verify-consumers -- --timeout-ms 45000 --out .bag/acp-consumer-fixtures/local-consumer-validation-latest.json` | Passed | Local Glass/Zed launch-target config spawns the built ACP server, handshakes, creates a session, runs `/chat Ahoj, co umis?`, and records no side effects. |
| `npm run build` | Rspack compiled successfully in `312 ms` | `dist/index.js` exists for the configured ACP launch target. |

## ACP Consumer Proof

The local consumer verifier read `/Users/satan/.config/zed/settings.json` and found `agent_servers.bleeding-agent` configured as:

```text
node /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/dist/index.js acp
```

Installed named consumers on this host:

| Consumer | App | Bundle | Version | Validation |
| --- | --- | --- | --- | --- |
| Glass | `/Applications/Glass.app` | `dev.glass.local` | `0.1.0` | Installed and covered by launch-target validation. |
| Zed | `/Applications/Zed.app` | `dev.zed.Zed` | `1.0.0` | Installed and covered by launch-target validation. |

Handshake evidence:

| Metric | Value |
| --- | ---: |
| `protocolCalls` | `6` |
| `sessionUpdates` | `3` |
| `fsRead` | `0` |
| `fsWrite` | `0` |
| `terminalCreate` | `0` |
| `permission` | `0` |
| `stopReason` | `end_turn` |

This proves the configured local ACP server target starts and basic chat does not accidentally inspect or mutate the project. It does not prove desktop UI rendering parity, permission-dialog visuals, diff widget behavior, terminal widget behavior, or full `/run` edit flows inside Glass/Zed.

## Independent Sidecar Verification

Two read-only release verifiers ran in parallel.

### ACP Dogfood Sidecar

Commands included focused ACP/RPC tests, headless ACP transcript tests, ACP settings snippets, local consumer verification, and typecheck.

Results:

| Slice | Result |
| --- | --- |
| Focused ACP/RPC tests | `37 pass`, `0 fail`, `202 expect()` calls |
| Headless ACP transcript tests | `5 pass`, `0 fail`, `147 expect()` calls |
| Combined targeted ACP tests | `42 pass`, `0 fail` |
| Typecheck | Passed |
| Consumer verifier | Passed |

Covered behavior includes consumer-neutral capability profiles, coding-focused command surface, hidden maintenance commands, slash routing, temporary mode restoration, YOLO/Safe policy, planning progress, cancellation artifacts, path policy, rich diff/text fallback, permission failures, terminal failures, and fake MCP execution through ACP updates.

### Eval And Optimizer Sidecar

Commands included replay/eval, MCP, edit, GEPA, typecheck, full trace-GEPA venv pytest, source-adapter regressions, autonomous-turn checks, prompt artifact bridge, and ACP edit routing.

Results:

| Slice | Result |
| --- | --- |
| Replay eval + eval harness | `52 pass`, `0 fail`, `506` assertions |
| MCP runtime/live-loop | `20 pass`, `0 fail`, `118` assertions |
| Edit ablation/telemetry/promotion | `52 pass`, `0 fail`, `638` assertions |
| GEPA optimizer/promotion/release rollup | `86 pass`, `0 fail`, `489` assertions |
| Typecheck | Passed |
| Full trace-GEPA venv tests | `67 passed` |
| Prompt/source-adapter regressions | `45 pass`, `0 fail`, `238` assertions |
| Autonomous turn + prompt artifact bridge | `8 pass`, `0 fail`, `52` assertions |
| ACP coding edit routing | `6 pass`, `0 fail`, `23` assertions |

The sidecar also inspected optimizer artifacts and found an important limitation: the latest BAG optimized prompt artifact does not prove full-eval uplift. Stored metadata showed latest BAG validation delta `0.0`, while an older stored full eval had optimized mean below seed. That is not a blocker for the harness infrastructure, but it is a blocker for claiming the optimizer has already improved coding quality in production.

## Completed Graph Lanes

| Lane | Status | Result |
| --- | --- | --- |
| Ownership map | Completed | Source/runtime/experiment/artifact boundaries documented in `docs/bleeding-agent-ownership-manifest.md`. |
| Green gates | Completed | TypeScript, Bun tests, and trace-GEPA scope repaired and documented. |
| Experiment quarantine | Completed | Local MCP config is ignored/exampled, optimized prompts and trace-RAG are opt-in, ESM path handling fixed. |
| ACP contract closure | Completed | Consumer-neutral ACP routing, YOLO/Safe behavior, slash commands, cancellation, artifacts, settings snippets, and transcript tests hardened. |
| Edit optimization loop | Completed | Multiple edit families retained, edit outcomes traced by phase, applied-but-broken and rollback evidence captured, task-shape routing added. |
| Replay source pipeline | Completed | CC session v2 and live ACP capture canonicalization, redaction, split discipline, source lineage, and failure extraction completed. |
| Live MCP loop | Completed | MCP tool contracts can be rendered into the model tool list and executed through ACP updates in the live tool-use runner path. |
| GEPA operations | Completed | Readiness gates, scoped candidates, train/dev/holdout gates, latency/token vetoes, promotion pointers, monitoring, and rollback primitives covered. |
| Release proof | Completed | This report plus command evidence above. |

## What We Can Claim Now

- `bag acp` is a working ACP stdio backend for local consumers.
- Glass and Zed are launch-target validated on this machine through the configured ACP server command.
- Normal greeting/chat does not trigger project scouting or side effects.
- Default YOLO and Safe behaviors are covered by tests; Safe prompts for risky writes/commands, YOLO avoids unnecessary approval in allowed paths.
- Auto/chat/plan/run routing and temporary mode restoration are covered.
- File edit behavior is not hardwired to one strategy family; whole-file, exact replace, unified diff, apply patch, and hash/range families exist behind measurable contracts.
- Edit telemetry tracks parse, apply, stale context, protected path, post-apply-broken, verification, repair, and rollback phases.
- Replay cases can be extracted from live ACP captures and source adapters with redaction and train/dev/hidden-holdout discipline.
- MCP runtime contracts and fake MCP live-loop execution are integrated into ACP tool-use routing.
- GEPA-style optimizer primitives exist for bounded prompt/tool/edit/verification policy artifacts, not arbitrary source rewriting.
- Promotion and rollback use active pointers, validation, eval gates, and regression monitoring primitives.

## What We Must Not Overclaim

- Not proven as a mature replacement for Codex CLI, Claude Code, ForgeCode, Pi, Oh My Pi, or OpenCode.
- Not proven with broad real Glass/Zed desktop UI automation.
- Not proven on arbitrary ACP clients.
- Not proven with arbitrary real ACP-attached MCP transports; current live-loop proof uses runtime metadata/executors and fake-server coverage.
- Not proven with a large live-model/live-client replay corpus.
- Not proven that current optimized prompt artifacts improve real coding quality; current artifact evidence is mixed.
- Not a free-form self-rewriting agent. Optimization is intentionally limited to policy artifacts and active pointers.

## Release Risks

1. The worktree remains very dirty, with many untracked source/test/doc files from prior iterations. Merge hygiene still needs an explicit packaging pass before any commit.
2. `.mcp.json` was removed from git tracking and replaced by `.mcp.example.json`; local ignored `.mcp.json` still exists. This is the right direction, but staging should be reviewed before commit.
3. ACP consumer validation is launch-target and protocol-level, not desktop UI proof.
4. The optimizer loop is structurally strong but data-poor. The next quality jump requires real session collection and replay evaluation over actual failures.
5. GEPA prompt artifacts need richer manifest metadata and promotion lineage, not just `system` content.

## Recommended Next Wave

1. Package the branch: decide which untracked files are product source, tests, docs, local artifacts, or ignored experiments.
2. Add real ACP `/run` dogfood traces from Glass and/or Zed with file edits, terminal verification, cancellation, and rollback.
3. Start a real replay corpus from successful and failed ACP sessions, with redaction review and hidden-holdout protection.
4. Run edit strategy ablations against real local-model and master-model outputs per codebase profile.
5. Improve GEPA artifact manifests so every promoted candidate carries candidate id, source evidence bundle ids, eval scorecard ids, promotion decision id, rollback checkpoint, model profile, and codebase profile.
6. Only after the corpus is non-trivial, enable autonomous optimization scheduling in dry-run mode first.
