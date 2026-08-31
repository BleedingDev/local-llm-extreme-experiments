# BleedingAgent

BleedingAgent is an ACP coding-agent backend and self-evolving optimization harness. The `bag`
command is the operator entrypoint for launching ACP, running diagnostics, and inspecting
optimization artifacts; it is not the product boundary.

The intended runtime is:

- `GPT-5.5` as master planner, interviewer, PRD author, critic, and final judge.
- Local MLX `Qwen3.6-35B-A3B-RotorQuant-MLX-3bit` as the executor/scout pool through an OpenAI-compatible `mlx_lm.server`.
- Ax as the agent/RLM runtime layer, with bounded `maxBatchedLlmQueryConcurrency` and `maxSubAgentCalls`.

## Product Flow

1. An ACP-compatible client starts `bag acp` and negotiates client capabilities.
2. BleedingAgent creates an ACP session with Auto/Chat/Plan/Run modes and a coding-focused slash
   command surface.
3. Auto decides whether a prompt should stay chat-only, run a read-only plan/report flow, or execute
   a coding run with edits and verification.
4. Coding and planning turns persist traces, metrics, artifacts, self-evaluation, and replayable
   evidence under `.bag/`.
5. The optimization harness uses accumulated evidence to propose, evaluate, promote, monitor, and
   roll back scoped policy changes per model/codebase profile.

The operator CLI also keeps the older `interview` -> `prd` -> `dag` pipeline available as a
deterministic planning utility and maintenance surface.

The design borrows the interview -> PRD -> DAG shape from
`/Users/satan/code/coding-agent-zcp/platform`, especially the structured interview state,
PRD section contract, and PRD Beads graph model.

## Commands

```bash
npm run build
npm run typecheck
npm test
npm run bag -- doctor
npm run bag -- run "build the next coding-agent slice"
npm run bag -- metrics
npm run bag -- optimize
npm run bag -- self-optimize
npm run bag -- self-optimize --apply
npm run bag -- acp-settings
npm run bag -- acp
```

After build, the package exposes:

```bash
bag doctor
bag run "task"
```

## Monitoring And Self-Improvement

Every `bag run` writes:

- `.bag/runs/<run-id>/manifest.json`
- `.bag/runs/<run-id>/self-evaluation.json`
- `.bag/runs/<run-id>/optimization.json`
- `.bag/telemetry/events.jsonl`
- `.bag/telemetry/metrics.json`
- `.bag/telemetry/spans.jsonl`
- `.bag/knowledge.md`

The first self-evaluation is deterministic so it works offline. It scores step success,
artifact completeness, step latency, LLM-call reliability, and tool-call reliability. LLM calls are
recorded with role, model, endpoint, duration, HTTP status, and token usage when the provider
returns usage data.

Tool calls are recorded separately with tool name, namespace, description version, argument size,
argument hash, latency, retry count, result kind/size, and error class/message. This is the data
needed to improve tool descriptions, JSON schemas, retry policy, timeout defaults, and routing
between master/local executors.

In addition to metrics, BleedingAgent writes HALO-inspired OpenInference-shaped spans to
`.bag/telemetry/spans.jsonl`. Every ACP/coding run now records an `agent.run` root span plus step,
LLM, and tool spans with `trace_id`, `span_id`, `parent_span_id`, status, observation kind, model
name, token counts, tool name, input hash, and capped input/output previews. The shape intentionally
matches the useful part of `context-labs/HALO`: trace data is a flat JSONL span stream that can be
indexed, queried, clustered, and handed to an optimizer without coupling the agent runtime to a
specific editor.

BleedingAgent also builds a sidecar trace index next to the span file:

- `.bag/telemetry/spans.jsonl.index.jsonl`
- `.bag/telemetry/spans.jsonl.index.meta.json`

The internal trace store supports the HALO-style bounded operations that matter for agent
self-improvement: dataset overview, trace query/count filters, full trace view for small traces,
selected span view for surgical inspection, and substring search inside one trace. Large traces are
not dumped blindly; payloads are capped and oversized traces return a summary with top span names so
the optimizer can search or fetch selected spans instead.

`bag metrics` prints an operator summary. `bag metrics --json` prints the raw persisted metrics.

`bag self-optimize` reads accumulated telemetry and writes a concrete candidate under
`.bag/optimizations/<id>.json` plus a markdown report. A candidate can recommend safe config
changes, tool guidance, and failure-specific action items. `bag self-optimize --apply` or
`bag apply-optimization <id>` only writes safe local artifacts: `bag.config.json` for policy
changes and `.bag/tool-guidance.md` for learned tool-call guidance. It does not modify project
source files.

The self-optimizer now combines two views:

- aggregate metrics from `.bag/telemetry/metrics.json` for pass rates, p95 latencies, and
  concurrency recommendations;
- indexed HALO-style spans from `.bag/telemetry/spans.jsonl` for repeated failure clusters,
  repeated tool input hashes, model/tool observation kinds, trace counts, sample trace IDs, and slow
  trace groups.

Each candidate now includes eval-gated improvement proposals. Applying a candidate writes safe local
artifacts only: `.bag/tool-guidance.md`, `.bag/self-improvement-plan.md`, and optionally
`bag.config.json`. It does not edit project source files. The next layer is to let the master model
or a dedicated RLM inspect the indexed trace tools and synthesize prompt/tool-schema patch artifacts
under the same eval gates.

## ACP Integration

BleedingAgent implements the Agent Client Protocol through the official
`@agentclientprotocol/sdk` package. Run it as an ACP stdio agent:

```bash
bag acp
```

For any ACP client, `bag acp-settings` prints the generic launch command and named setup examples.
Use `bag acp-settings zed` when you only want the Zed settings object:

```bash
npm run bag -- acp-settings
npm run bag -- acp-settings zed
```

The ACP agent exposes:

- `initialize` with protocol version `1`, embedded text context support, session list/resume/close,
  and no authentication requirement.
- `session/new` and `session/resume` with Auto, Chat, Plan, and Run modes. Auto is the default.
- `session/prompt` in Auto mode asks the configured model router to choose `chat`, `plan`, or
  `run` from the prompt semantics. It does not use natural-language keyword lists.
- `session/prompt` in Chat mode as forced no-side-effects conversation and command discovery.
- `session/prompt` in Plan/Run modes as the visible BleedingAgent flow: knowledge load, context
  scout, repo context build, interview, PRD generation, DAG generation, self-evaluation, policy
  optimization, and knowledge codification.
- `session/update` plan entries and detailed ACP tool call lifecycle updates for every phase.

This gives compatible ACP clients the coding-agent shell, session UI, plan display, and detailed
tool-call timeline while BleedingAgent focuses on model routing, telemetry, evaluation, and
optimization. Named consumers such as Zed and Glass are setup examples, not special product modes.

Auto mode is the default. A prompt like "generate a codebase status report" should route to Plan
because repository reads are useful, while a prompt asking to change files or run verification
should route to Run. Use `/chat` only when you explicitly want forced no-side-effects chat.
BleedingAgent intentionally does not use hardcoded natural-language keyword lists to decide
whether a message should touch the project.

When Auto routes a single prompt into Plan or Run, the visible ACP mode is switched for that turn
and then restored to Auto after the turn finishes. The same restoration happens for `/plan <task>`
or `/run <task>` when the previous mode was Auto. Commands without a task, or manual ACP mode
changes, remain persistent.

Run mode is the full coding-agent path. Plan mode is available when you only want
interview/PRD/DAG artifacts without edits. Run mode performs:

1. Model-guided file selection from repo context and scout findings.
2. ACP `fs/read_text_file` reads for selected files.
3. Policy-selected model-facing edit contracts.
4. ACP edit preview and `request_permission` before every write when Safe mode is active. The final
   ACP transport currently writes complete file content through `fs/write_text_file`.
5. ACP `fs/write_text_file` writes after approval.
6. ACP terminal execution for verification commands after permission.
7. Up to two repair rounds when verification fails: failed output is fed back into the model,
   new edits are previewed, permission is requested again, and verification is rerun.
8. Trace persistence under `.bag/runs/<run-id>/coding-trace.json`.

The trace captures file-selection evidence, file hashes and byte sizes, edit hashes, permission
outcomes, terminal exit status/output, LLM metrics, tool metrics, and self-evaluation artifacts.
Those traces feed `bag metrics`, `bag optimize`, and `bag self-optimize` so failed edits,
permission rejections, bad command choices, slow tools, and model failures can be optimized over
time.

ACP status updates are intentionally compact in normal consumers. File reads report the file and
byte count, edit previews show rich diff content when the client supports the write/edit path and
fall back to text status otherwise, terminal verification embeds the client terminal when terminal
capability is available, and final messages link the trace and manifest artifacts.
Full raw tool outputs remain attached to ACP tool updates for operators and tests, but optimizer
policy ids, model profile ids, and rendered contract internals are kept out of ordinary user-facing
status text.

Prompt cancellation aborts the active turn and leaves the session reusable. Coding turns write
`coding-trace.json`, `cancellation.json`, and `manifest.json` when a started run is cancelled.
Planning turns persist completed intermediate artifacts as they finish, then write
`planning-trace.json`, `cancellation.json`, and `manifest.json` on cancellation. Terminal
verification cancellation kills and releases the ACP terminal when the client exposes those
operations. Repeated `session/resume` or defensive `session/load` calls for an existing session
return the current session state instead of recreating the session or replaying side effects.

YOLO mode is the default. File writes and terminal commands run without ACP permission prompts
unless the session is switched to safe mode. Use `/safe` to require approval again and `/yolo` to
return to no-approval mode. The ACP session config also exposes a `YOLO Mode` boolean. In Safe
mode, rejected edits and rejected terminal commands produce failed tool-call updates, no write or
terminal side effect is performed, and the rejection is captured in telemetry as
`permission_rejected`.

ACP slash commands advertised through `available_commands_update`:

- `/run <task>`: switch to coding-agent mode and run a task.
- `/plan <task>`: switch to planning-only mode and run a task.
- `/chat`: force no-side-effects chat mode.
- `/auto`: return to model-routed auto mode.
- `/yolo`: disable permission prompts.
- `/safe`: require permission prompts.
- `/skills`: list locally installed skills visible to the agent.
- `/mcp`: show MCP servers attached to the ACP session.
- `/metrics`: show telemetry/artifact locations and current mode.
- `/traces`: show the HALO-style trace dataset overview and recent failing trace IDs.

Maintenance/admin optimization controls are intentionally hidden from `available_commands_update`
so normal ACP users see a coding-focused command surface. Operators can still use
`/maintenance status`, `/maintenance eval`, `/maintenance optimize`, `/maintenance promote <id>`,
and `/maintenance rollback [checkpoint]`. These flows run as maintenance-scoped plan
inspections: status and eval are read-only, optimize computes a bounded report from existing
metrics, promote is a dry-run readiness inspection for candidate lineage and new-session-only
promotion safety, and rollback inspects checkpoint readiness without mutating the active pointer.

MCP servers passed by the ACP client are stored on the session and surfaced through `/mcp`.
BleedingAgent advertises HTTP and SSE MCP capability, plus stdio session metadata is accepted by
the SDK. The current implementation does not yet proxy arbitrary MCP tools into the model loop;
that is the next integration layer after the ACP coding path.

### ACP Compatibility And Named Consumers

The current validation is protocol-level ACP transcript testing plus local launch-target validation
for the installed Glass/Zed-compatible setup, not broad desktop UI automation. The smoke tests
instantiate `BleedingAcpAgent` with a fake ACP client and cover the consumer-visible protocol
behavior any compatible client needs. Glass and Zed are named tested setup examples, not the product
boundary.

| Flow | ACP contract coverage | Named consumer fixtures | Tested behavior |
| --- | --- | --- | --- |
| Session start | offline headless ACP transcript | Glass/Zed launch target locally validated | `initialize`, `session/new`, Auto mode, YOLO config, and `available_commands_update`. |
| Greeting | offline headless ACP transcript | Glass/Zed setup documented | Chat emits capability help without reads, writes, terminals, or optimizer internals. |
| Plan/report | offline headless ACP transcript | Glass/Zed setup documented | Auto and `/plan` switch into read-only plan behavior and restore Auto after temporary routing. |
| Edit run | offline headless ACP transcript | Glass/Zed setup documented | ACP read, edit preview, write, edit lifecycle spans, trace/artifact references. |
| Terminal verification | offline headless ACP transcript | Glass/Zed setup documented | ACP terminal creation, terminal content embedding when supported, exit/output capture, and cleanup on cancel. |
| Permissions | offline headless ACP transcript | Glass/Zed setup documented | YOLO bypasses prompts; Safe prompts; rejected writes/commands are failed, traceable outcomes. |
| Slash commands | offline headless ACP transcript | Glass/Zed setup documented | Normal command surface is run/plan/chat/auto/YOLO/Safe/skills/MCP/metrics/traces only. |
| Cancellation | offline headless ACP transcript | Glass/Zed setup documented | Prompt-level run/plan cancellation writes cancellation artifacts, clears the pending prompt, keeps the session reusable, and active terminal verification can be aborted, killed, released, and surfaced as cancelled. |
| Trace/artifacts | offline headless ACP transcript | Glass/Zed setup documented | `/metrics`, `/traces`, final trace paths, and raw tool outputs remain available. |

Recommended Zed setup:

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

Use `npm run bag -- acp-settings` to print the project-specific generic snippet and named examples.
Any ACP client should launch the same stdio command, `bag acp`, from the repository root. The local
Glass/Zed-compatible setup on this host uses `node dist/index.js acp`; `npm run build` refreshes that
launch target. Keep the working directory set to the project root so relative file paths and `.bag`
artifacts resolve predictably.

Known limitations: desktop rendering automation for named consumers has not been run in this lane;
local named-consumer validation currently means launch-target handshake plus protocol behavior.
MCP session metadata and runtime bridge pieces are present, but arbitrary ACP-attached MCP tools are
not yet generally proxied into the live model loop.
`session/load` is intentionally not advertised as a full transcript replay capability, but defensive
load calls are idempotent and do not restart side effects. Session fork, provider switching, and NES
APIs are still explicitly unsupported.

## Toolchain

- TypeScript source uses extensionless imports.
- `tsconfig.json` uses bundler-style module resolution instead of `NodeNext`.
- Rspack builds the executable `dist/index.js`.
- Bun Test owns tests through `npm test`.
- If BleedingAgent is later split into reusable packages, those package builds should move to RsLib.

## Model Setup

Start the local executor:

```bash
.venv/bin/mlx_lm.server \
  --model majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit \
  --host 127.0.0.1 \
  --port 18082 \
  --max-tokens 512 \
  --chat-template-args '{"enable_thinking":false}' \
  --prefill-step-size 2048 \
  --decode-concurrency 24 \
  --prompt-concurrency 24
```

Set `OPENAI_API_KEY` for the master model. Without it, `bag` still runs deterministic fallback
flows so the CLI, telemetry, and artifact contracts can be tested offline.

## Policy

Default local executor concurrency is `16`, with a hard practical cap of `24`. That matches the
measured sweet spot from this machine: roughly 16-24 concurrent calls produce the best aggregate
throughput, while 32 concurrent longer generations lose throughput and latency quality.
