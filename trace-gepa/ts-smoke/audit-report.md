# Source-adapter audit (Phase-2 Build Agent #5)

Date: 2026-05-01
Repo: /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx

## Verdict

- `acp-session-jsonl` **does NOT parse Claude Code transcripts** (`~/.claude/projects/<id>/<sid>.jsonl`).
  The existing detector requires JSON-RPC ACP envelopes (`jsonrpc:"2.0"`, `method` like `session/`, `fs/`, `terminal/`, or `sessionUpdate` payloads). Claude Code 2.1.126 transcripts use raw `type:user|assistant|attachment|system|permission-mode|last-prompt|ai-title|queue-operation|file-history-snapshot|summary` records with `uuid`/`parentUuid` lineage. Detection fails outright; canonicalization produced 0 spans and a single `unknown_source_shape` diagnostic.
- `codex-session-jsonl` **does parse Codex sessions** but with partial coverage. Detection succeeds; for `rollout-2026-04-03T23-30-43-019d542e-c1fb-7ed1-b8a2-2d3f44cdc7c3.jsonl` first-50 sample → 33/50 spans + 17 `unsupported_record` diagnostics. Documented but NOT modified per task constraints.

## Action taken

Created NEW adapter `cc-session-jsonl-v2` (additive, no rewrites of existing files):

- `src/source-adapters/cc-session-v2.ts` — detector helper `isCcSessionV2Record`, self-contained `canonicalizeCcSessionV2()` that produces the same `CanonicalSourceRecord[]` / `HaloSpan` shape as canonical.ts.
- `src/source-adapters/boundary.ts` — added type `cc-session-jsonl-v2`, registered detector entry, added `CC_V2_RECORD_TYPES` set, added detector function. The linter then refactored to import `isCcSessionV2Record` from the v2 module (single source of truth). Diff is additive only.

Mapped CC event types (event_kind / observation_kind):
- `user` → `user_message` / AGENT (or `tool_result` / TOOL when `content[].type=="tool_result"`, status_code=ERROR when `is_error:true`)
- `assistant` → `assistant_message` (LLM) | `assistant_thinking` (LLM) | `tool_call` (TOOL), one event per `content[]` item
- `attachment` → `attachment_<sub>` / CHAIN, including `skill_listing`, `task_reminder`, `queued_command`, `diagnostics`, `deferred_tools_delta`, `edited_text_file`
- `system` → `system_message` / CHAIN
- `permission-mode` → `permission_mode` / CHAIN
- `last-prompt` → `last_prompt` / CHAIN
- `ai-title` → `ai_title` / CHAIN
- `queue-operation` → `queue_operation` / CHAIN
- `file-history-snapshot` → `file_history_snapshot` / CHAIN
- `summary` → `summary` / CHAIN

Lineage: `uuid` → `lineage.id`, `parentUuid` → `lineage.parentId`, `sessionId` → `lineage.sessionId`. `isSidechain` propagated as `source.cc.is_sidechain` attribute. Tool calls/results emit `tool.call_id`, `tool.name`, `input.value`/`output.value`. `tool_result.is_error:true` produces `STATUS_CODE_ERROR`.

## Smoke results

| File | Adapter | Sample | detect.ok | spans | diag |
|---|---|---|---|---|---|
| cc405b87...jsonl (CC 2.1.126) | acp-session-jsonl (legacy) | 50 | false | 0 | 1 unknown_source_shape |
| cc405b87...jsonl (CC 2.1.126) | cc-session-jsonl-v2 (new) | 50 | true | **50** | **0** |
| rollout-019d542e (Codex 0.118.0) | codex-session-jsonl | 50 | true | 33 | 17 unsupported_record |

CC v2 event_kind coverage on the 50-record sample:
`permission_mode:3, file_history_snapshot:2, user_message:2, attachment_deferred_tools_delta:1, attachment_skill_listing:1, ai_title:3, assistant_thinking:9, tool_call:10, tool_result:9, last_prompt:2, assistant_message:2, system_message:2, queue_operation:2, attachment_queued_command:1, attachment_task_reminder:1`

Observation kinds: CHAIN:18, AGENT:2, LLM:11, TOOL:19 — sensible distribution.

Whole-file CC tally (138 attachments, 689 user, 1141 assistant, plus controls) covers all known v2 types — no surprises beyond the seen subtypes.

Whole-file Codex `event_msg.payload.type` tally surfaced unhandled subtypes the existing adapter drops as `unsupported_record`: `task_started`, `user_message`, `agent_message`, `token_count`, `exec_command_end`, `patch_apply_end`, `task_complete`, `collab_agent_spawn_end`, `collab_waiting_end`, `context_compacted`. `response_item.payload.type=reasoning` and `custom_tool_call`/`custom_tool_call_output` are also dropped. **Documented; not silently rewritten.** A future codex-session-v2 should map these (notably `exec_command_end` which carries the bash-exit-code signal, and `user_message` which is the canonical user turn for Codex).

## Validation

- `tsc -p tsconfig.json --noEmit` — clean.
- `tsc -p tsconfig.test.json --noEmit` — clean.
- `bun test tests/source-adapters` — 27/27 pass (no regression).
- `bun test tests` — 333 pass / 1 unrelated 5s-timeout flake in replay-runner-integration.

## Files

- `src/source-adapters/cc-session-v2.ts` (new)
- `src/source-adapters/boundary.ts` (additive: new union member, new detector, new record-type set; no removals)
- `trace-gepa/ts-smoke/audit-adapters.ts` (smoke harness)
- `trace-gepa/ts-smoke/audit-report.md` (this file)

## Caller-integration note

`canonicalizeSourceRecords()` in `canonical.ts` was NOT modified (per constraint). Callers wanting CC-v2 spans should call `canonicalizeCcSessionV2()` directly (or branch on `source.sourceType === "cc-session-jsonl-v2"`). The `audit-adapters.ts` smoke harness shows the dispatch pattern.
