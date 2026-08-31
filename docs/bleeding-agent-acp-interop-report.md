# BleedingAgent ACP Interop Report

## Scope

This report covers the deterministic headless ACP harness used for protocol-level compatibility evidence plus the local named-consumer launch-target validation run on this host. It does not claim desktop UI rendering parity for any named ACP consumer.

## Covered ACP Flows

| Flow | Evidence | Coverage |
| --- | --- | --- |
| Initialize | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Records `initialize` protocol calls, sends ACP protocol version, client identity, and explicit headless consumer capabilities, and captures the agent response in the transcript. |
| Session create | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Creates a session with `cwd` and empty MCP server list, records `session/new`, and captures `available_commands_update` streamed updates. |
| Session resume | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Runner performs a default `session/list` plus `session/resume` check after creation. Tests verify the offline transcript records resume without recreating side effects. |
| Session cancel | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Runner supports `--cancel-after-ms` to send `session/cancel` during a prompt. Tests verify the transcript records cancel and existing runtime tests cover prompt reuse after cancellation. |
| Prompt send | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Runner sends `/yolo` when enabled and then `/run <task>`, with timeout/error recording. Offline tests record chat, metrics, and traces prompts. |
| Streamed updates | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Headless client records every `session/update` notification as transcript data. Tests assert streamed agent messages, command updates, diff content, terminal content, metrics, and trace text. |
| Filesystem read/write | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Client implements `readTextFile` with optional line/limit slicing and `writeTextFile` with parent directory creation. Transcript records path and byte counts. |
| Permission handling | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | YOLO auto-selects `allow_always` or `allow_once`. Safe/non-YOLO auto-selects reject options when present or returns cancelled. Offline tests cover rejected write and terminal outcomes. |
| Terminal execution | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Client supports `terminal/create`, `terminal/output`, `terminal/wait_for_exit`, `terminal/kill`, and `terminal/release`. The runner can use real subprocesses or deterministic `--terminal-mode stub`. |
| Transcript capture | `scripts/bag_acp_run.ts`; `tests/bag.test.ts` | Transcript contains protocol calls, streamed session updates, permission outcomes, filesystem operations, terminal lifecycle events, and agent stderr. Summary counts are written beside the raw transcript. |

## Regression Scenario Matrix

`scripts/bag_acp_run.ts` exports `HEADLESS_ACP_REGRESSION_SCENARIOS` so the transcript expectations stay named and reusable. `tests/bag.test.ts` runs the same scenario IDs through the offline headless harness.

| Scenario | Headless evidence | Required signal | Side-effect boundary |
| --- | --- | --- | --- |
| `greeting-no-side-effect` | Chat-mode prompt transcript | `agent_message_chunk` | No ACP filesystem, terminal, or permission calls. |
| `read-only-report` | `/plan` transcript with stubbed report progress | `current_mode_update`, `plan`, `tool_call`, `tool_call_update` | No ACP writes, terminals, or permission prompts. |
| `coding-run` | `/run` transcript with bounded coding progress | `current_mode_update`, `plan` | Run mode is explicit and visible before coding work. |
| `edit-preview-write` | Internal ACP edit preview/write transcript | `tool_call`, `tool_call_update`, `fs_write` | Preview is emitted before the final whole-file ACP write. |
| `terminal-verification` | Internal ACP terminal verification transcript | `terminal_create`, `terminal_output`, `terminal_exit` | Terminal lifecycle is exercised only when terminal capability is present. |
| `rejected-permission` | Safe-mode write rejection transcript | `permission`, failed `tool_call_update` | Rejection fails closed and skips `fs_write`. |
| `cancellation` | Active `/run` prompt cancelled by `session/cancel` | cancelled prompt response and current-mode update | Session remains reusable after cancellation. |
| `metrics-traces` | `/metrics` plus `/traces` transcript | compact `agent_message_chunk` telemetry text | No ACP writes or terminals. |
| `maintenance-isolation` | `/maintenance status` transcript | hidden command surface plus maintenance plan/tool progress | Maintenance is an explicit inspection path, not a normal command suggestion. |

## Capability Profiles

The headless runner now publishes explicit consumer capability profiles in transcript JSON under `consumerCapabilities`. These profiles separate ACP initialize capabilities from harness-level expectations such as rich tool content and artifact locations.

| Profile | ACP initialize capabilities | Rich tool content | Artifact links | Slash commands | Permissions | Unsupported |
| --- | --- | --- | --- | --- | --- | --- |
| `minimal` | `fs.readTextFile=false`, `fs.writeTextFile=false`, `terminal=false` | Diff and terminal content disabled; text fallback expected. | File-location references accepted; resource-link blocks are not claimed. | `available_commands_update` and text slash prompts accepted. | `requestPermission` callback exists, but side-effect paths should fail before permission because filesystem and terminal are unsupported. | Images, generic resources, NES, provider configuration/model selection, and session fork. |
| `capable` | `fs.readTextFile=true`, `fs.writeTextFile=true`, `terminal=true` | Diff content and terminal content expected, with text fallback still allowed. | File-location references accepted; resource-link blocks are not claimed. | `available_commands_update` and text slash prompts accepted. | YOLO auto-allows allow options; Safe/non-YOLO auto-selects reject options when present or cancels. | Images, generic resources, NES, provider configuration/model selection, and session fork. |

The capable profile is the default for full coding transcripts. The minimal profile is a regression fixture for clients that can display conversation and command updates but cannot expose filesystem or terminal methods.

The terminal implementation has two modes:

- `real`: spawns the requested command in the requested cwd and captures bounded stdout/stderr.
- `stub`: records the requested command and immediately returns exit code `0` with deterministic output.

Filesystem operations resolve relative paths against the session workdir. Absolute paths are passed through because ACP clients may expose absolute workspace paths; consumer-specific sandbox behavior is outside this harness.

Rich diff and terminal content are verified as ACP update payloads emitted by the agent, not as desktop rendering behavior. Artifact links mean ACP `locations` pointing at concrete files; this lane does not claim resource-link rendering.

## Unsupported Or Degraded Features

| Feature | Current behavior | Fallback expectation |
| --- | --- | --- |
| Desktop UI parity | Not validated in this lane. The harness verifies ACP protocol payloads, not editor rendering. | Add a named fixture only after local consumer validation with screenshots/transcript evidence. |
| `session/load` | Agent returns defensive load behavior but does not advertise full transcript replay. | Use `session/resume` for session continuity; treat replay extraction as separate work. |
| Image or audio prompt content | Agent prompt capabilities advertise text/embedded context, not image or audio. Headless profiles mark images unsupported. | Clients should send text prompts or embedded text resources. |
| Generic resources/resource links | Headless profiles accept file-location artifact references but do not claim generic resource-link rendering. | Surface artifact paths as text/file locations. |
| NES and provider/model APIs | Unstable provider, model-selection, NES, and fork-session APIs are not part of the compatibility claim. | Configure provider/model outside ACP before starting `bag acp`; use new ACP sessions after config changes. |
| ACP-attached MCP tools | Attached MCP server metadata is visible through `/mcp`, but arbitrary ACP-attached MCP tools are not generally proxied into the live model loop. | Use built-in ACP coding tools and document MCP runtime work as a separate integration layer. |
| File write granularity | Final ACP transport writes whole file content through `fs/write_text_file`. | Model-facing edit strategies can be richer, but consumers should expect whole-file commit transport today. |
| Consumers without filesystem/terminal support | Minimal profile fails closed before side effects and emits text fallback updates. | Display text fallback, failed tool updates, and artifact paths; do not assume diff/terminal widgets. |

## Consumer-Agnostic Boundary

The harness exercises the ACP contract through the TypeScript SDK instead of scripting one editor. Real consumers can differ in visual diff rendering, permission UI, terminal presentation, artifact-link handling, and filesystem sandboxing. Those differences should be added as named consumer fixtures only after local validation.

Real consumer validation boundary for this lane: local launch-target validation was performed, not desktop rendering automation. The verifier checked the installed Glass and Zed app bundles, parsed the local Zed/Glass-compatible `agent_servers` settings, spawned the same configured BleedingAgent ACP command, completed `initialize`, `session/new`, and a `/chat` prompt, and confirmed the greeting flow produced no filesystem, terminal, or permission side effects.

Local evidence from 2026-05-01:

| Consumer | Local signal | Result |
| --- | --- | --- |
| Glass | `/Applications/Glass.app`, bundle id `dev.glass.local`, version `0.1.0`; shares the local Zed-compatible `agent_servers` settings path used by this Glass fork. | Installed and covered by launch-target validation. |
| Zed | `/Applications/Zed.app`, bundle id `dev.zed.Zed`, version `1.0.0`; `~/.config/zed/settings.json` contains `agent_servers.bleeding-agent`. | Installed and covered by launch-target validation. |
| Launch command | `node /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/dist/index.js acp` from the local `agent_servers` entry. | Spawned successfully and completed ACP handshake. |
| No-side-effect greeting | `/chat Ahoj, co umis?` through the same launch target. | `fsRead=0`, `fsWrite=0`, `terminalCreate=0`, `permission=0`, `stopReason=end_turn`. |

Evidence artifact: `.bag/acp-consumer-fixtures/local-consumer-validation-latest.json`.

This is enough to claim that the installed named consumers have a working local launch configuration and that the configured ACP server target handshakes correctly. It is not enough to claim pixel-level UI parity, permission-dialog appearance, diff rendering quality, terminal widget rendering, or arbitrary ACP client support.

The compatibility claim should remain narrow: BleedingAgent can run as an ACP stdio backend and pass deterministic protocol-level transcripts for sessions, prompts, updates, filesystem operations, permissions, terminal lifecycle, cancellation, metrics, traces, and maintenance isolation. It should not claim universal ACP UI parity.

## Verification

Verified on 2026-05-01:

- `npm run build`
- `npm run acp:verify-consumers -- --timeout-ms 45000 --out .bag/acp-consumer-fixtures/local-consumer-validation-latest.json`
- `bun test tests/bag.test.ts --test-name-pattern "ACP regression scenario|headless ACP client fixture|offline headless ACP consumer smoke transcript|ACP consumer capabilities|minimal and capable ACP client transcripts"`
- `bun test tests/bag.test.ts`
