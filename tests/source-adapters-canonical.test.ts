import { describe, expect, test } from "bun:test";
import {
  canonicalizeSourceRecord,
  canonicalizeSourceRecords,
  canonicalSourceRecordsToJsonl,
} from "../src/source-adapters/canonical";
import type { SourceMetadata } from "../src/source-adapters/boundary";

const now = "2026-04-30T00:00:00.000Z";

const codexSource: SourceMetadata = {
  sourceType: "codex-session-jsonl",
  path: "/repo/.codex/session.jsonl",
  sessionId: "codex-session-a",
  schemaVersion: "0.99.0",
  inspectedRecordCount: 3,
  detectedSignals: ["codex-session-meta", "codex-record-type-envelope"],
};

const acpSource: SourceMetadata = {
  sourceType: "acp-session-jsonl",
  path: "/repo/acp.jsonl",
  sessionId: "acp-session-a",
  schemaVersion: "1",
  inspectedRecordCount: 2,
  detectedSignals: ["acp-jsonrpc-method-or-session-update", "acp-session-lineage"],
};

const piSource: SourceMetadata = {
  sourceType: "pi-session-jsonl",
  path: "/repo/pi.jsonl",
  sessionId: "pi-session-a",
  schemaVersion: "3",
  inspectedRecordCount: 4,
  detectedSignals: ["pi-session-header", "pi-tree-entry-envelope"],
};

const ccSource: SourceMetadata = {
  sourceType: "cc-session-jsonl-v2",
  path: "/repo/cc-session.jsonl",
  sessionId: "cc-session-a",
  schemaVersion: "claude-sonnet-4-7",
  inspectedRecordCount: 2,
  detectedSignals: ["cc-transcript-type-envelope", "cc-uuid-lineage"],
};

describe("source adapter canonical output", () => {
  test("converts Codex messages and tool calls to canonical Halo spans", () => {
    const output = canonicalizeSourceRecords({
      source: codexSource,
      records: [
        {
          type: "response_item",
          timestamp: now,
          payload: {
            type: "message",
            role: "assistant",
            content: `Answer with sk-testsecretvalue1234567890 ${"x".repeat(80)}`,
          },
        },
        {
          type: "response_item",
          timestamp: now,
          payload: {
            type: "function_call",
            call_id: "call-read",
            name: "read_file",
            arguments: "{\"path\":\"/repo/src/index.ts\"}",
          },
        },
      ],
      redactionOptions: { maxTextExcerptChars: 24 },
    });

    expect(output.diagnostics).toEqual([]);
    expect(output.records).toHaveLength(2);

    const message = output.records[0]?.span;
    expect(message).toMatchObject({
      trace_id: expect.any(String),
      span_id: expect.any(String),
      parent_span_id: "",
      trace_state: "",
      name: "source.codex-session-jsonl.message",
      kind: "SPAN_KIND_INTERNAL",
      start_time: now,
      end_time: now,
      status: {
        code: "STATUS_CODE_OK",
        message: "",
      },
      scope: {
        name: "bag.source-adapters",
        version: "canonical.v1",
      },
      resource: {
        attributes: {
          "service.name": "bleeding-agent-source-adapter",
          "source.adapter.type": "codex-session-jsonl",
          "source.adapter.session_id": "codex-session-a",
        },
      },
      attributes: {
        "openinference.span.kind": "LLM",
        "inference.observation_kind": "LLM",
        "source.adapter.event_kind": "message",
        "source.adapter.type": "codex-session-jsonl",
        "source.record.index": 0,
        "message.role": "assistant",
      },
    });
    expect(JSON.stringify(message?.attributes)).not.toContain("sk-testsecretvalue1234567890");
    expect(message?.attributes["source.redaction.redaction_kinds"]).toEqual(["openai_api_key"]);
    expect(JSON.stringify(message?.attributes)).toContain("[TRUNCATED:");

    const tool = output.records[1]?.span;
    expect(tool?.name).toBe("source.codex-session-jsonl.tool_call");
    expect(tool?.kind).toBe("SPAN_KIND_CLIENT");
    expect(tool?.attributes["openinference.span.kind"]).toBe("TOOL");
    expect(tool?.attributes["tool.name"]).toBe("read_file");
    expect(tool?.attributes["tool.call_id"]).toBe("call-read");
    expect(output.records[1]?.lineage.toolCallIds).toEqual(["call-read"]);
    expect(output.records[1]?.lineage.toolNames).toEqual(["read_file"]);

    expect(canonicalSourceRecordsToJsonl(output.records)).toContain("\"trace_id\"");
  });

  test("converts ACP tool updates with lineage, source metadata, and failed status", () => {
    const output = canonicalizeSourceRecord({
      source: acpSource,
      recordIndex: 4,
      line: 12,
      record: {
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId: "acp-session-a",
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId: "tool-1",
            title: "Run npm test",
            kind: "execute",
            status: "failed",
            rawInput: {
              command: "npm",
              args: ["test"],
              token: "ghp_abcdefghijklmnopqrstuvwxyz123456",
            },
            rawOutput: {
              error: "exit 1",
            },
          },
        },
      },
    });

    expect(output.diagnostics).toEqual([]);
    expect(output.records).toHaveLength(1);
    const span = output.records[0]?.span;
    expect(span).toMatchObject({
      name: "source.acp-session-jsonl.tool_call_update",
      kind: "SPAN_KIND_CLIENT",
      status: {
        code: "STATUS_CODE_ERROR",
        message: "exit 1",
      },
      attributes: {
        "source.record.index": 4,
        "source.record.line": 12,
        "source.lineage.session_id": "acp-session-a",
        "source.lineage.tool_call_ids": ["tool-1"],
        "source.acp.session_update": "tool_call_update",
        "tool.call_id": "tool-1",
        "tool.status": "failed",
      },
    });
    expect(JSON.stringify(span?.attributes)).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");
    expect(JSON.stringify(span?.attributes)).toContain("[REDACTED:secret_field]");
  });

  test("converts Pi-ish messages, tool results, model changes, and compactions", () => {
    const output = canonicalizeSourceRecords({
      source: piSource,
      records: [
        {
          type: "assistant",
          id: "msg-a",
          parentId: "root",
          timestamp: now,
          content: [{ type: "text", text: "done" }],
        },
        {
          type: "tool_result",
          id: "result-a",
          parentId: "msg-a",
          tool_call_id: "call-a",
          status: "error",
          error: "file missing",
        },
        {
          type: "model_change",
          id: "model-a",
          parentId: "result-a",
          from: "local-small",
          to: "master-large",
        },
        {
          type: "compaction",
          id: "compact-a",
          parentId: "model-a",
          summary: "compressed context",
        },
      ],
    });

    expect(output.diagnostics).toEqual([]);
    expect(output.records.map((record) => record.span.attributes["source.adapter.event_kind"])).toEqual([
      "message",
      "tool_result",
      "model_change",
      "compaction",
    ]);
    expect(output.records[0]?.span.attributes["openinference.span.kind"]).toBe("LLM");
    expect(output.records[1]?.span.status).toEqual({
      code: "STATUS_CODE_ERROR",
      message: "file missing",
    });
    expect(output.records[1]?.lineage).toMatchObject({
      id: "result-a",
      parentId: "msg-a",
      sessionId: "pi-session-a",
      toolCallIds: ["call-a"],
    });
    expect(output.records[2]?.span.attributes["inference.llm.model_name"]).toBe("master-large");
    expect(output.records[3]?.span.name).toBe("source.pi-session-jsonl.compaction");
  });

  test("routes CC session v2 records through the generic canonicalizer", () => {
    const output = canonicalizeSourceRecord({
      source: ccSource,
      recordIndex: 9,
      line: 20,
      record: {
        type: "assistant",
        uuid: "cc-msg-assistant",
        parentUuid: "cc-msg-user",
        sessionId: "cc-session-a",
        timestamp: now,
        message: {
          role: "assistant",
          model: "claude-sonnet-4-7",
          content: [
            { type: "thinking", thinking: "Use the repo search tool, not raw memory." },
            { type: "text", text: "I will inspect the focused file." },
            {
              type: "tool_use",
              id: "toolu-read",
              name: "Read",
              input: {
                file_path: "/repo/src/source-adapters/canonical.ts",
                token: "ghp_abcdefghijklmnopqrstuvwxyz123456",
              },
            },
          ],
        },
      },
      redactionOptions: { maxTextExcerptChars: 64 },
    });

    expect(output.diagnostics).toEqual([]);
    expect(output.records.map((record) => record.span.attributes["source.adapter.event_kind"])).toEqual([
      "assistant_thinking",
      "assistant_message",
      "tool_call",
    ]);
    expect(output.records[0]?.span.scope).toEqual({
      name: "bag.source-adapters",
      version: "canonical.cc-v2",
    });
    expect(output.records[2]?.span).toMatchObject({
      name: "source.cc-session-jsonl-v2.tool_call",
      kind: "SPAN_KIND_CLIENT",
      attributes: {
        "source.adapter.type": "cc-session-jsonl-v2",
        "source.record.index": 9,
        "source.record.line": 20,
        "source.record.sub_event_index": 2,
        "source.lineage.id": "cc-msg-assistant",
        "source.lineage.parent_id": "cc-msg-user",
        "source.lineage.session_id": "cc-session-a",
        "tool.name": "Read",
        "tool.call_id": "toolu-read",
      },
    });
    expect(output.records[2]?.lineage).toMatchObject({
      id: "cc-msg-assistant",
      parentId: "cc-msg-user",
      sessionId: "cc-session-a",
      toolCallIds: ["toolu-read"],
      toolNames: ["Read"],
    });
    expect(JSON.stringify(output.records)).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");
    expect(JSON.stringify(output.records)).toContain("[REDACTED:secret_field]");
  });

  test("normalizes CC failure evidence across tool, edit, subagent, permission, file-history, and error shapes", () => {
    const output = canonicalizeSourceRecords({
      source: ccSource,
      records: [
        {
          type: "user",
          uuid: "cc-correction",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: "No, that is wrong. You changed the wrong file.",
          },
        },
        {
          type: "assistant",
          uuid: "cc-subagent",
          parentUuid: "cc-correction",
          sessionId: "cc-session-a",
          isSidechain: true,
          timestamp: now,
          message: {
            role: "assistant",
            model: "claude-sonnet-4-7",
            content: [
              { type: "text", text: "Subagent checking failures." },
              { type: "tool_use", id: "toolu-sub-bash", name: "Bash", input: { command: "missing-command" } },
            ],
          },
        },
        {
          type: "user",
          uuid: "cc-result-bash",
          parentUuid: "cc-subagent",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu-sub-bash",
                is_error: true,
                content: "zsh:1: command not found: missing-command",
              },
            ],
          },
        },
        {
          type: "user",
          uuid: "cc-result-exit",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu-exit", is_error: true, content: "Process exited with code 2" }],
          },
        },
        {
          type: "user",
          uuid: "cc-result-skill",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu-skill", is_error: true, content: "No such skill: repo-migration" }],
          },
        },
        {
          type: "user",
          uuid: "cc-result-edit-ambiguous",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu-edit", is_error: true, content: "old_string is not unique; found 3 matches" }],
          },
        },
        {
          type: "user",
          uuid: "cc-result-edit-read",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu-edit-read", is_error: true, content: "File has not been read yet. Read before editing." }],
          },
        },
        {
          type: "user",
          uuid: "cc-result-timeout",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu-timeout", is_error: true, content: "Command timed out after 1000ms" }],
          },
        },
        {
          type: "user",
          uuid: "cc-result-cancel",
          sessionId: "cc-session-a",
          timestamp: now,
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "toolu-cancel", is_error: true, content: "Tool execution cancelled by user" }],
          },
        },
        {
          type: "permission-mode",
          sessionId: "cc-session-a",
          timestamp: now,
          permissionMode: "acceptEdits",
        },
        {
          type: "file-history-snapshot",
          sessionId: "cc-session-a",
          timestamp: now,
          messageId: "cc-msg",
          snapshot: {
            trackedFileBackups: {
              "src/a.ts": { hash: "sha256:a" },
              "src/b.ts": { hash: "sha256:b" },
            },
          },
        },
        {
          type: "error",
          uuid: "cc-error",
          sessionId: "cc-session-a",
          timestamp: now,
          error: "Unhandled adapter parse error",
          code: "adapter_parse",
        },
      ],
    });

    expect(output.diagnostics).toEqual([]);
    expect(output.records.map((record) => record.span.attributes["source.adapter.event_kind"])).toContain("subagent_tool_call");
    expect(output.records.map((record) => record.span.attributes["source.adapter.event_kind"])).toContain("permission_mode");
    expect(output.records.map((record) => record.span.attributes["source.adapter.event_kind"])).toContain("file_history_snapshot");
    expect(output.records.find((record) =>
      record.span.attributes["source.adapter.event_kind"] === "file_history_snapshot")
      ?.span.attributes["source.cc.snapshot_file_count"]).toBe(2);
    expect(output.records
      .map((record) => record.span.attributes["source.failure.kind"])
      .filter(Boolean)).toEqual([
        "user_correction",
        "command_not_found",
        "bash_nonzero",
        "hallucinated_skill",
        "non_unique_edit_string",
        "edit_before_read",
        "timeout",
        "cancellation",
        "generic_error",
      ]);
    expect(output.records.filter((record) => record.span.attributes["source.failure.kind"]).every((record) =>
      record.span.attributes["source.baseline.role"] === "observed_baseline" &&
      record.span.attributes["source.baseline.gold"] === false)).toBe(true);
  });

  test("normalizes Codex and BAG/ACP tool failure equivalents as observed baselines", () => {
    const codex = canonicalizeSourceRecord({
      source: codexSource,
      record: {
        type: "response_item",
        timestamp: now,
        payload: {
          type: "function_call_output",
          id: "codex-out",
          call_id: "call-bash",
          status: "failed",
          output: "Process exited with code 127: command not found",
        },
      },
    });
    const acp = canonicalizeSourceRecord({
      source: acpSource,
      record: {
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId: "acp-session-a",
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId: "bag-tool-1",
            title: "run skill",
            kind: "execute",
            status: "failed",
            rawOutput: {
              error: "No such skill: release-auditor",
            },
          },
        },
      },
    });

    expect(codex.records[0]?.span.attributes).toMatchObject({
      "source.failure.kind": "command_not_found",
      "source.failure.error_code": "command_not_found",
      "source.baseline.role": "observed_baseline",
      "source.baseline.gold": false,
    });
    expect(codex.records[0]?.span.status.code).toBe("STATUS_CODE_ERROR");
    expect(acp.records[0]?.span.attributes).toMatchObject({
      "source.failure.kind": "hallucinated_skill",
      "source.failure.error_code": "hallucinated_skill",
      "source.baseline.role": "observed_baseline",
      "source.baseline.gold": false,
    });
  });

  test("emits unsupported diagnostics instead of catch-all spans", () => {
    const output = canonicalizeSourceRecord({
      source: codexSource,
      recordIndex: 2,
      record: {
        type: "response_item",
        payload: {
          type: "unknown_future_payload",
          value: "do not guess",
        },
      },
    });

    expect(output.records).toEqual([]);
    expect(output.diagnostics).toEqual([
      {
        code: "unsupported_record",
        message: "No canonical mapping for codex-session-jsonl record.",
        sourceType: "codex-session-jsonl",
        recordIndex: 2,
        recordType: "response_item",
      },
    ]);
  });
});
