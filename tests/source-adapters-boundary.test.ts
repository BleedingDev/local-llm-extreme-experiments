import { describe, expect, test } from "bun:test";
import {
  detectSourceJsonl,
  detectSourceRecords,
  sourceAdapters,
  type SourceAdapterType,
} from "../src/source-adapters/boundary";

const now = "2026-04-30T00:00:00.000Z";

const spanRecord = {
  trace_id: "trace-a",
  span_id: "span-a",
  parent_span_id: "root",
  trace_state: "",
  name: "tool.workspace.repo.read",
  kind: "SPAN_KIND_INTERNAL",
  start_time: now,
  end_time: now,
  status: {
    code: "STATUS_CODE_OK",
    message: "",
  },
  resource: {
    attributes: {
      "service.name": "bleeding-agent",
    },
  },
  scope: {
    name: "bleeding-agent",
    version: "1.0.0",
  },
  attributes: {
    "openinference.span.kind": "TOOL",
  },
};

const expectSource = (jsonl: string, sourceType: SourceAdapterType) => {
  const result = detectSourceJsonl(jsonl, { path: `/tmp/${sourceType}.jsonl` });
  expect(result.ok).toBe(true);
  if (!result.ok) {
    throw new Error("expected supported source");
  }
  expect(result.source.sourceType).toBe(sourceType);
  expect(result.source.path).toBe(`/tmp/${sourceType}.jsonl`);
  expect(result.source.recordCountEstimate).toEqual({ value: 2, kind: "exact" });
  expect(result.source.inspectedRecordCount).toBe(2);
  expect(result.source.detectedSignals.length).toBeGreaterThan(0);
  return result.source;
};

describe("source adapter boundary detection", () => {
  test("detects native spans JSONL with span lineage metadata", () => {
    const source = expectSource(
      `${JSON.stringify(spanRecord)}\n${JSON.stringify({ ...spanRecord, span_id: "span-b" })}\n`,
      "spans-jsonl",
    );

    expect(source.sessionId).toBe("trace-a");
    expect(source.schemaVersion).toBe("bleeding-agent@1.0.0");
  });

  test("detects ACP JSON-RPC session transcripts and preserves session metadata", () => {
    const source = expectSource(
      [
        {
          jsonrpc: "2.0",
          id: 1,
          method: "session/new",
          params: {
            cwd: "/repo",
            protocolVersion: 1,
          },
        },
        {
          jsonrpc: "2.0",
          method: "session/update",
          params: {
            sessionId: "acp-session-a",
            update: {
              sessionUpdate: "tool_call",
            },
          },
        },
      ].map((record) => JSON.stringify(record)).join("\n"),
      "acp-session-jsonl",
    );

    expect(source.sessionId).toBe("acp-session-a");
    expect(source.schemaVersion).toBe("1");
  });

  test("detects Codex session JSONL and extracts session id and CLI version", () => {
    const source = expectSource(
      [
        {
          type: "session_meta",
          timestamp: now,
          payload: {
            id: "codex-session-a",
            cwd: "/repo",
            cli_version: "0.99.0",
            source: "codex",
            model_provider: "openai",
          },
        },
        {
          type: "response_item",
          timestamp: now,
          payload: {
            type: "message",
            role: "user",
            content: "hello",
          },
        },
      ].map((record) => JSON.stringify(record)).join("\n"),
      "codex-session-jsonl",
    );

    expect(source.sessionId).toBe("codex-session-a");
    expect(source.schemaVersion).toBe("0.99.0");
  });

  test("detects optional Pi session JSONL without accepting generic chat logs", () => {
    const source = expectSource(
      [
        {
          type: "session",
          version: 3,
          id: "pi-session-a",
          timestamp: now,
          cwd: "/repo",
        },
        {
          type: "assistant",
          id: "entry-a",
          parentId: null,
          role: "assistant",
          content: [{ type: "text", text: "done" }],
        },
      ].map((record) => JSON.stringify(record)).join("\n"),
      "pi-session-jsonl",
    );

    expect(source.sessionId).toBe("pi-session-a");
    expect(source.schemaVersion).toBe("3");
  });

  test("fails closed for unknown object logs instead of treating them as sessions", () => {
    const result = detectSourceJsonl([
      JSON.stringify({ level: "info", message: "hello", timestamp: now }),
      JSON.stringify({ level: "info", message: "still not a supported source" }),
    ].join("\n"));

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual([
      {
        code: "unknown_source_shape",
        message: "JSONL records do not match the explicit boundary for known source adapter.",
      },
    ]);
    expect(result.recordCountEstimate).toEqual({ value: 2, kind: "exact" });
  });

  test("returns explicit diagnostics for malformed and non-object JSONL", () => {
    const malformed = detectSourceJsonl(`${JSON.stringify(spanRecord)}\n{"trace_id":\n`);
    expect(malformed.ok).toBe(false);
    expect(malformed.diagnostics[0]?.code).toBe("malformed_jsonl");
    expect(malformed.diagnostics[0]?.line).toBe(2);

    const nonObject = detectSourceJsonl(`${JSON.stringify(spanRecord)}\n42\n`);
    expect(nonObject.ok).toBe(false);
    expect(nonObject.diagnostics).toEqual([
      {
        code: "non_object_record",
        message: "Source detection only accepts JSON object records.",
        recordIndex: 1,
      },
    ]);
  });

  test("adapter-specific detection rejects the wrong source kind", () => {
    const codexAdapter = sourceAdapters.find((adapter) => adapter.sourceType === "codex-session-jsonl");
    if (codexAdapter == null) {
      throw new Error("missing Codex adapter boundary");
    }

    const result = codexAdapter.detect([spanRecord]);
    expect(result.ok).toBe(false);
    expect(result.diagnostics[0]?.code).toBe("unknown_source_shape");
    expect(result.diagnostics[0]?.message).toContain("codex-session-jsonl");
  });

  test("records sampled count estimates when inspection is capped", () => {
    const result = detectSourceJsonl(
      [
        JSON.stringify(spanRecord),
        JSON.stringify({ ...spanRecord, span_id: "span-b" }),
        JSON.stringify({ ...spanRecord, span_id: "span-c" }),
      ].join("\n"),
      { maxInspectionRecords: 2 },
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      throw new Error("expected supported source");
    }
    expect(result.source.recordCountEstimate).toEqual({ value: 3, kind: "sample" });
    expect(result.source.inspectedRecordCount).toBe(2);
  });

  test("direct record detection rejects empty sources", () => {
    const result = detectSourceRecords([]);
    expect(result.ok).toBe(false);
    expect(result.diagnostics[0]?.code).toBe("empty_source");
  });
});
