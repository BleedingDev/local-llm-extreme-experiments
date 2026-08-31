# BleedingAgent Next Live Tool Loop Inventory

Scope: Wave 2 audit for `.codex/plans/bleeding-agent-next-live-tool-loop.plan.md` todo `next-tools-runtime-inventory`, with Wave 2B updates for `next-tools-model-facing-surface` and Wave 2C updates for MCP runtime policy, taxonomy, trace feedback, and tests. This inventory covers the current ACP built-in tool loop, MCP runtime scaffolding, side-effect policy, execution bridge, rendered/model-facing contracts, trace feedback, and tests.

## Current Status Refresh

The early audit sections below are retained as historical evidence. A later single-owner runtime
follow-up closed the main ACP-facing MCP live-loop gap: `BleedingAcpAgent.runLiveMcpToolCall()` now
routes model-facing MCP calls through rendered contracts, YOLO/Safe permission policy, ACP
tool-call updates, runtime failure taxonomy, bounded results, optimizer tool metrics, and feedback
records. Focused tests cover successful read-only fake-MCP calls, Safe denial for write tools,
malformed arguments, retry exhaustion, oversized-result feedback, ACP update status, and telemetry
persistence; lower-level MCP runtime tests continue to cover network/process policy, timeout,
cancellation, and retry mechanics.

Remaining boundary: BleedingAgent still should not claim arbitrary ACP-attached MCP tools are fully
discoverable and productized for every consumer. Tool-schema discovery from real ACP-attached MCP
servers, broad built-in-tool canonicalization, and large real-session evidence remain follow-on
work.

## Historical Executive Summary (Pre Live MCP Follow-Up)

- Built-in ACP tools are live today, but they are hardcoded inside `BleedingAcpAgent` as procedural steps, not exposed to the model through canonical tool specs or rendered tool contracts.
- MCP tooling has a strong isolated runtime substrate: metadata normalization, side-effect classification, rendered-contract preparation, explicit model-facing MCP contracts, argument validation, result bounding, permission decisions, timeout/cancellation handling, retry-budget enforcement, trace-shaped results, normalized failure codes, and optimizer feedback records are implemented in `src/mcp/runtime-tools.ts`.
- Earlier Wave 2 audit found that the live ACP loop did not call the MCP runtime substrate. That specific gap is superseded for model-facing MCP calls by `BleedingAcpAgent.runLiveMcpToolCall()`. The remaining boundary is broader: arbitrary ACP-attached MCP server discovery and productized schema import are still follow-on work.
- Trace and optimizer feedback capture versions and edit-contract lineage. Current live MCP calls now carry rendered-contract/policy/failure evidence through ACP updates, metrics, and feedback records; built-in generic tool calls still need the same per-call canonical tool IDs, rendered IDs, side-effect class, policy action, permission status, redaction status, truncation, and normalized failure-class indexing as first-class HALO dimensions.
- Replay fixtures cover the target taxonomy synthetically, including MCP lineage. Current ACP coding runs can now emit `replay-capture.json` records, but promotion-quality learning still depends on a larger real-session corpus and stronger attribution coverage.

## Live Built-In ACP Tool Loop

Implemented:

- ACP initialization advertises MCP transport capability (`http`, `sse`) and session capabilities through the facade in `src/acp-agent.ts` using metadata from `src/acp/surface.ts`.
- New/resumed/loaded sessions store `mcpServers` and pin an optimizer policy through `src/acp/session.ts`.
- `/mcp` is a visibility command only. It prints attached server metadata from `session.mcpServers` through `src/acp/slash-router.ts`.
- Planning mode runs a fixed built-in sequence through `runAcpTool`: knowledge load, context scout/build, interview, PRD, DAG, self-eval, policy optimize, and knowledge codify in `src/acp/planning-runner.ts`.
- Coding mode runs a fixed built-in sequence: context selection, ACP file reads, edit strategy resolution, model-generated patch, edit preview/write, post-apply checks, terminal verification, repair, rollback, telemetry, artifacts, and background optimization inspection in `src/acp/coding-runner.ts`.
- File reads use `readClientFile`, which calls `acp.fs.readTextFile` through `runAcpTool` and falls back to local `readFileSync` if the ACP client cannot read but the local path exists in `src/acp/workspace-io.ts`.
- Writes use `writeClientFileWithPermission`, emit an ACP `tool_call`, request permission in Safe mode, and call `connection.writeTextFile` in `src/acp/workspace-io.ts`.
- Terminal commands use `runTerminalCommand`, emit an ACP `tool_call`, request permission in Safe mode, call `connection.createTerminal`, bound ACP terminal output, and surface exit/output in `src/acp/terminal.ts`.
- Generic built-in tool UI and telemetry go through `runAcpTool`, which emits `tool_call` and `tool_call_update`, wraps execution in `RunTelemetry.measureToolCall`, and records raw output in `src/acp/tool-runner.ts`.

Scaffolded or partial:

- Built-in tool calls are measured with `toolName`, `namespace`, `descriptionVersion`, argument bytes/hash, result size/kind, duration, retry count, and error name through `RunTelemetry.measureToolCall` in `src/telemetry.ts:311-407`.
- The session optimizer pin includes model/profile/policy/tool-version lineage and is attached to telemetry spans through `src/acp/session.ts` and `src/telemetry.ts:440-468`.
- User-facing tool updates intentionally hide optimizer internals, covered by tests in `tests/bag.test.ts:760-788`.

Gaps for the next implementation lane:

- Built-in live tools are not represented as `CanonicalToolSpec` or `RenderedToolContract` objects before model use. `runAcpTool`, `writeClientFileWithPermission`, and `runTerminalCommand` carry only ad hoc `toolName` and `descriptionVersion` fields.
- Built-ins are not exposed as model-callable tools. The model is prompted to return JSON plans or edit payloads via `chatText`, not native/json tool calls. `src/llm.ts:10-22` exposes only `messages`, `maxTokens`, `temperature`, and `json`; request construction only sets `response_format` for JSON in `src/llm.ts:79-99`.
- Built-in read/search/think tool output has no explicit result-size policy before `rawOutput`; `RunTelemetry` only truncates span previews to 4096 chars in `src/telemetry.ts:679-685`.
- Built-in write/process policy is binary YOLO/Safe. It does not use a shared side-effect class model, network/process/write risk flags, retry policy, timeout policy, or structured denial taxonomy.
- `runTerminalCommand` has `outputByteLimit: 80_000`, but no canonical result budget object, omitted-byte count, or result-style contract is persisted for optimizer feedback.

## MCP Runtime Substrate

Implemented:

- MCP metadata types cover tool names, descriptions, schemas, annotations, examples, result style, server metadata, policy, normalized tool, explicit model-facing contract, callable tool, execution request, permission request, metrics, trace, result, and optimizer feedback records in `src/mcp/runtime-tools.ts`.
- Side-effect policy classification uses MCP annotations and name/description patterns for read, write, network, and process risk in `src/mcp/runtime-tools.ts:312-479`.
- Policy output includes `sideEffectLevel`, `requiresConfirmation`, write/network/process risk flags, Safe action, YOLO action, argument byte budget, result byte budget, and reasons in `src/mcp/runtime-tools.ts:48-61` and `src/mcp/runtime-tools.ts:469-478`.
- MCP tools normalize into `CanonicalToolSpec` with sanitized namespace/name, stable hashed canonical ID, input/output schemas, result style, side-effect level, confirmation requirement, and examples in `src/mcp/runtime-tools.ts:481-531`.
- Server tool lists normalize through `normalizeMcpServerTools` in `src/mcp/runtime-tools.ts:533-537`.
- MCP rendered contracts are prepared through the shared optimizer renderer in `prepareMcpRenderedToolContracts`, returning canonical specs, rendered contracts, model-facing contracts, model-facing lookup maps, policy maps, and result budgets in `src/mcp/runtime-tools.ts`.
- MCP model-facing contracts now wrap rendered contracts with tool-call-safe names, stable model-facing IDs, canonical/rendered/model/policy lineage, result style/version, truncation wording, structured-error guidance, result examples, prompt fragments, and side-effect policy summaries in `src/mcp/runtime-tools.ts`.
- Runtime argument validation rejects non-object args, oversized args, missing required fields, invalid primitive types, and unknown fields when `additionalProperties === false` in `src/mcp/runtime-tools.ts:716-806`.
- Result bounding converts arbitrary executor output to JSON, truncates oversized serialized results, records original/omitted bytes, and returns a structured truncation object where possible in `src/mcp/runtime-tools.ts:596-671`.
- Runtime policy decisions support Safe/YOLO mode, confirmation, default denial when no handler exists, permission handler errors, blocked decisions, denial classes, per-call/default timeout, cancellation, and retry-budget exhaustion in `src/mcp/runtime-tools.ts`.
- The execution bridge maps callable tools by model-facing ID/name, canonical ID, rendered ID, rendered name, runtime name, server-scoped MCP tool name, and original MCP tool name, then executes via an injected executor with timeout/cancellation signal propagation and retry-count propagation in `src/mcp/runtime-tools.ts`.
- MCP trace-shaped results include model-facing/canonical/rendered lineage, policy action, permission status, side-effect level, latency, argument bytes, argument-shape hash, redaction status, result sizes, omitted bytes, truncation, retry count, normalized failure code/class, and follow-up behavior in `src/mcp/runtime-tools.ts`.
- Optimizer feedback records can be produced from MCP results, with severity, lineage, bounded feedback text, policy fields, latency, argument shape, redaction status, size fields, retry count, failure code/class, follow-up behavior, and truncation flag in `src/mcp/runtime-tools.ts`.

Scaffolded or partial:

- Result feedback still carries a compatibility `redacted: false` flag alongside `redactionStatus`; there is no actual MCP result redaction path yet.
- The bridge receives an injected `executor`, but no production ACP/MCP client executor is wired to session MCP servers.

Historical gaps, with current boundary:

- Superseded for the fake-MCP/model-facing live loop: `BleedingAcpAgent.runLiveMcpToolCall()` now routes through the MCP runtime bridge and feedback helpers. The still-open product gap is discovering and importing real ACP-attached MCP server tool schemas into that surface.
- ACP `McpServer` session metadata likely does not include per-tool schemas. The current normalization accepts `McpServerMetadata` with a `tools` array, but `BagAcpSession.mcpServers` stores SDK `McpServer[]` from session params in `src/acp/session.ts` and `/mcp` only lists server connection fields through `src/acp/slash-router.ts`.
- There is no live discovery call that turns attached ACP MCP servers into `McpServerMetadata.tools`.
- MCP runtime can now produce model-facing MCP contracts, but there is still no live ACP model-facing tool selection step that combines approved built-ins and attached MCP tools under context and side-effect budgets.
- The current live MCP bridge covers the modeled fake-MCP path. It is not yet a broad productized bridge for arbitrary consumer-attached MCP servers.
- The current live MCP path adapts runtime permission policy to ACP updates and Safe/YOLO behavior. The remaining permission gap is unified built-in/MCP policy display and risk classification across every tool family.
- Current live MCP calls persist optimizer metrics and feedback evidence. Remaining hardening is per-call HALO indexing parity for built-ins and MCP calls, plus larger real replay extraction before GEPA can learn robust tool guidance.

## Rendered Tool Contracts And Registry

Implemented:

- The shared optimizer types define `CanonicalToolSpec` with namespace/name/schema/result style/side effect/confirmation/examples in `src/optimizer/types.ts:145-166`.
- `RenderedToolContract` includes rendered ID, canonical ID/version, rendered version, model profile, optional policy ID, renderer/version, rendered name, description, input schema, result style/version, prompt fragments, and examples in `src/optimizer/types.ts:168-188`.
- `renderToolContract` creates deterministic rendered IDs, stable ordered schemas, model-specific text fallback descriptions/fragments, example selection, renderer lineage, and policy/tool versions in `src/optimizer/tool-renderer.ts:196-221`.
- `renderToolContracts` sorts canonical specs deterministically in `src/optimizer/tool-renderer.ts:223-226`.
- `selectRenderedToolContracts` can prefer a complete promoted rendered-contract set from registry records and otherwise falls back to freshly rendered contracts in `src/optimizer/tool-renderer.ts:248-272`.
- Seed policy records carry canonical/rendered/result/verification/edit version pins in `src/optimizer/registry.ts:282-321` and `src/optimizer/registry.ts:324-364`; `resolveOptimizerPolicy` projects those into `ResolvedOptimizerPolicy` in `src/optimizer/policy-resolver.ts:333-359`.

Scaffolded or partial:

- Edit strategy contracts reuse the same renderer through `src/edit-strategy/contract-renderer.ts`, and live coding uses rendered edit contracts in `src/acp/edit-routing.ts` and model prompts in `src/acp/coding-generation.ts`.
- Generic built-in tool contracts are not seeded as `canonical_tool_spec` or `rendered_tool_contract` records. Seed registry records are model profiles, codebase profile, and policies only in `src/optimizer/registry.ts:366-367`.

Gaps for the next implementation lane:

- Define canonical specs for built-ins such as `acp.fs.readTextFile`, `acp.fs.writeTextFile`, `acp.terminal.create`, context scout/build, and maintenance/status tools where appropriate.
- Decide which built-ins are model-callable versus internal-only. Internal planning steps may not belong in the live model-facing surface.
- Materialize or select rendered contracts for approved built-ins and MCP tools from the active optimizer policy before each turn.
- MCP model-facing contracts now include truncation wording and structured error/truncated result examples. Built-in tool contracts and live ACP prompt injection still need equivalent treatment.

## Failure Taxonomy

Implemented in MCP runtime:

- Missing server and unknown tool: `missing_server` and `unknown_tool` are separated before execution, including server-scoped lookup for duplicated tool names.
- Malformed or schema-invalid arguments: status `invalid_arguments` now carries normalized codes `malformed_arguments` for non-object/oversized argument payloads and `schema_mismatch` for missing required fields, invalid primitive types, and unknown fields.
- Permission/policy denial: `permission_denied`, `policy_blocked`, and `permission_error` codes/classes are emitted with policy action, permission status, and side-effect level.
- Timeout/cancellation: runtime timeout emits status/code `timeout`; external abort emits status/code `cancelled`.
- Oversized successful output: success with `failureCode: "oversized_output"`, `metrics.truncated = true`, omitted bytes, and structured truncation object.
- Retry exhaustion/runtime exception: configured retry budgets emit `retry_exhausted`; ordinary executor failures retain class `execution_error` while carrying normalized code `runtime_exception`; stale-context-like executor errors map to `stale_context`.

Implemented in built-in ACP/replay scaffolds:

- `ToolCallMetric` records success/failure, error/errorName, retry count, argument bytes/hash, result bytes/kind in `src/types.ts:164-183`.
- Replay capture schema has statuses `succeeded`, `failed`, `malformed_args`, `permission_denied`, `timed_out`, and `truncated` plus optional `errorCode` in `src/replay/capture.ts:92-111`.
- Synthetic replay scenarios cover malformed arguments, oversized output, permission denial, retry behavior, truncation visibility, MCP call, and terminal verification enforcement in `src/replay/tool-call-scenarios.ts:29-37` and `src/replay/tool-call-scenarios.ts:149-680`.

Gaps against the plan taxonomy:

- No unified enum spans both built-ins and MCP; the normalized taxonomy is MCP-runtime-local until live ACP wiring adopts it.
- Built-in write/terminal permission rejections are normalized through `src/acp/permission-outcomes.ts` from the `src/acp/workspace-io.ts` and `src/acp/terminal.ts` boundaries.
- MCP runtime accepts `retryCount` and enforces `maxRetryCount`, but it still does not schedule retries itself. The caller must do retry orchestration.
- Built-in terminal has ACP terminal output bound but no explicit timeout control through the shared MCP runtime policy.

## Policy Enforcement

Implemented:

- Session YOLO/Safe state is initialized from `config.policy.requirePermissions` and toggled by config option or slash commands in `src/acp/session.ts` and `src/acp/slash-router.ts`.
- Built-in file writes and terminal commands request ACP permission in Safe mode and bypass in YOLO mode in `src/acp/workspace-io.ts` and `src/acp/terminal.ts`.
- MCP policy has side-effect class, confirmation flag, write/network/process risk flags, mode-specific Safe/YOLO actions, argument budget, result budget, and policy reasons in `src/mcp/runtime-tools.ts:426-479`.
- MCP process tools remain confirmation-bound even in YOLO mode via `yoloActionFor` in `src/mcp/runtime-tools.ts:410-411`.
- MCP runtime bridge enforces argument/result byte budgets, missing-server/tool lookup failures, permission decisions, per-call/default timeout, cancellation, and configured retry exhaustion before returning structured trace/feedback evidence.

Gaps:

- Built-ins and MCP do not share one policy resolver or policy decision object.
- Built-in tools do not classify `read`, `write`, `network`, and `process` through the same side-effect model used by MCP.
- Built-in terminal process execution has no policy distinction between safe verification and arbitrary process/network risk.
- MCP policy classification is heuristic and metadata-derived. It needs an approval/filtering layer before exposing tools to a model, especially because names/descriptions can understate risk.
- Timeout, cancellation, retry budget, and output-size budget are enforced in MCP runtime and used by the current live MCP path. Built-ins still use separate policies and need parity through the same shared policy/result-budget model.

## Trace, Metrics, And Feedback

Implemented:

- `RunTelemetry.measureToolCall` records tool metrics and HALO-style TOOL spans with tool name/namespace, description version, retry count, argument hash, argument bytes, output preview/bytes/kind, error type/message, duration, and session optimizer attributes in `src/telemetry.ts:311-407`.
- `recordHaloSpan` writes spans to configured JSONL with root optimizer attributes on every span in `src/telemetry.ts:471-522`.
- Trace indexing keeps model/profile/policy/version lineage and edit-contract dimensions in `src/trace-store.ts:6-46` and extracts them from span attributes in `src/trace-store.ts:874-938`.
- Trace analysis clusters failures and latency by observation kind and span name, preserving optimizer dimensions and input hashes in `src/trace-analysis.ts:176-242`.
- Candidate evidence bundles can convert trace failures, latency clusters, eval results, scorecards, and selected span excerpts into bounded, redacted optimizer evidence in `src/optimizer/evidence.ts:153-185` and `src/optimizer/evidence.ts:282-380`.
- MCP runtime can create feedback records directly from tool results in `src/mcp/runtime-tools.ts`, including argument-shape hash, redaction status, latency, normalized failure code, and follow-up behavior.

Gaps:

- `RunTelemetry.measureToolCall` does not accept rendered contract ID, canonical tool ID, side-effect class, policy action, permission status, redaction status, result-truncated flag, omitted bytes, or failure class.
- Trace indexing captures tool version lineage only through session optimizer attributes. It does not index per-call canonical tool IDs or rendered contract IDs for generic built-in/MCP tools.
- MCP runtime traces are returned as plain objects from the bridge with model-facing lineage and normalized trace feedback. They are not written as HALO spans by the live agent.
- MCP optimizer feedback records are not passed into `buildCandidateEvidenceBundle`, GEPA feedback, or saved as optimizer-visible artifacts.
- Planning/coding JSON `trace` arrays from `src/acp/planning-runner.ts` and `src/acp/coding-runner.ts` are artifact-local events, not normalized replay captures or indexed trace spans.
- Built-in tool outputs may appear as raw ACP `rawOutput`; there is no shared redaction or artifact-ref path for large/sensitive results.

## Tests And Fixtures

Covered:

- MCP normalization, policy classification, rendered/model-facing contract preparation, bridge success through model-facing names, malformed args, schema mismatch, missing server, unknown tool, oversized result bounding/truncation feedback, Safe denial, permission-approved writes/process, network Safe/YOLO behavior, process confirmation in YOLO, retry-count propagation, retry exhaustion, runtime exception feedback, timeout, cancellation, and trace/feedback lineage are covered in `tests/mcp-runtime-tools.test.ts`.
- Normal ACP command surface and hidden optimizer internals are covered in `tests/bag.test.ts:790-825`.
- `/mcp` visibility with a fake attached server is covered in `tests/bag.test.ts:1718-1752`.
- Replay tool-call taxonomy and holdout MCP lineage are covered in `tests/replay-tool-call-scenarios.test.ts:10-190`.
- Existing docs explicitly state that MCP metadata and runtime bridge pieces exist, but arbitrary MCP tools are not proxied into the live model loop in `docs/bleeding-agent.md:228-231` and `docs/bleeding-agent.md:270-272`.

Missing tests for implementation:

- Fake-model live loop where model output chooses a built-in read tool through a rendered contract.
- Fake-model live loop where model output chooses a fake MCP read tool from attached server metadata and receives bounded result content.
- Fake MCP write/network/process tools through actual ACP Safe permission and YOLO policy.
- Malformed built-in and MCP model tool calls with normalized failure codes and repair/follow-up behavior.
- Missing MCP server/tool and tool schema mismatch failures.
- Oversized built-in read/search/terminal outputs with omitted-byte counts and artifact refs.
- Timeout and retry exhaustion for built-in and MCP calls.
- Trace assertions that HALO spans contain rendered contract ID, argument shape/hash, redaction status, latency, retry count, result size, failure class, and policy action.
- Replay extraction from real live tool spans rather than manual synthetic captures.

## Exact Live-Loop Integration Gaps

1. Tool inventory builder: MCP-only model-facing contracts can be built from normalized MCP metadata, but no function builds the approved live tool set from built-in specs plus attached MCP metadata.
2. MCP discovery adapter: no live code converts ACP `McpServer[]` into `McpServerMetadata.tools`.
3. Contract selection: MCP runtime can call `renderToolContracts` and derive model-facing MCP contracts, but no live ACP code selects those contracts or does equivalent rendering for built-in runtime tools.
4. Model request surface: `LlmRouter.chatText` has no `tools`, `tool_choice`, function-call parser, or loop for tool call/result messages.
5. Execution dispatcher: no unified dispatcher routes model tool calls to built-ins or MCP bridge while applying one policy object.
6. ACP permission adapter: no adapter maps `McpRuntimePermissionRequest` to `connection.requestPermission`.
7. Failure taxonomy: no shared normalized live error enum covers built-ins and MCP.
8. Output policy: MCP runtime has argument/result budgets and truncation wording/examples, but no shared budget, redaction, and artifact-ref system applies to all built-in and MCP tools.
9. Trace writer: no common tool-call span writer persists live MCP bridge traces or richer built-in tool decisions.
10. Optimizer feedback: no live bridge from failed/truncated tool calls to candidate evidence or GEPA feedback.
11. Replay capture: no live capture extractor serializes real ACP tool-call spans into `AcpReplayToolCallRecord`.

## Implementation Risks

- Policy bypass risk: wiring MCP execution directly to ACP session servers without a shared side-effect policy would bypass the existing Safe/YOLO guardrails.
- Context explosion risk: dumping all MCP schemas/examples into prompts would exceed budgets and degrade tool choice.
- Trace blind spot risk: if MCP calls only return bridge objects but are not written as HALO spans, GEPA and replay will not see the failures.
- Taxonomy drift risk: keeping built-in string errors and MCP structured errors separate will make replay and optimizer feedback inconsistent.
- Overexposure risk: treating all attached MCP tools as model-visible without an allow/filter layer can expose network/process/destructive tools too eagerly.
- False safety risk: MCP side-effect classification is heuristic; destructive tools can be mislabeled if server metadata is incomplete or misleading.

## Implementation-Ready Status

This inventory is specific enough to feed implementation. The next lane can start with a narrow adapter layer:

1. Define canonical built-in runtime specs and an approved live-tool inventory builder that can merge built-ins with the MCP model-facing contracts.
2. Add MCP server tool discovery or fixture metadata adapter.
3. Render/select contracts for approved built-ins and wire the existing MCP model-facing contracts into the session optimizer pin.
4. Extend `LlmRouter` and the ACP turn loop with a bounded tool-call/result loop.
5. Route all live tool calls through one dispatcher that emits normalized policy decisions, results, HALO spans, replay records, and optimizer feedback.
