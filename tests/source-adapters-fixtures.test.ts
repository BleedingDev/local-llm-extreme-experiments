import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  canonicalizeSourceRecord,
  canonicalSourceRecordsToJsonl,
} from "../src/source-adapters/canonical";
import {
  detectSourceFile,
  streamSourceDirectory,
  streamSourceFile,
} from "../src/source-adapters/streaming";

const now = "2026-04-30T00:00:00.000Z";

const tempDir = async () => mkdtemp(join(tmpdir(), "bleeding-agent-source-fixtures-"));

const writeJsonl = async (path: string, records: readonly unknown[]) => {
  await writeFile(path, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
};

describe("source adapter synthetic fixtures", () => {
  test("runs a Codex session fixture through detection, streaming, redaction, and canonical output", async () => {
    const dir = await tempDir();
    const path = join(dir, "codex-session.jsonl");
    await writeJsonl(path, [
      {
        type: "session_meta",
        timestamp: now,
        payload: {
          id: "codex-session-fixture",
          cwd: "/repo",
          cli_version: "0.99.0",
          source: "codex",
          model_provider: "openai",
          model: "gpt-5.5",
          branch: "feature/source-adapters",
        },
      },
      {
        type: "turn_context",
        timestamp: now,
        payload: {
          id: "turn-1",
          cwd: "/repo",
          model: "qwen-local-executor",
          approval_policy: "never",
          sandbox_policy: "danger-full-access",
          branch: "feature/source-adapters",
        },
      },
      {
        type: "response_item",
        timestamp: now,
        payload: {
          type: "message",
          id: "msg-user",
          parentId: "turn-1",
          role: "user",
          content: "Investigate the failing source adapter test with Bearer abcdefghijklmnopqrstuvwx.",
        },
      },
      {
        type: "response_item",
        timestamp: now,
        payload: {
          type: "message",
          id: "msg-assistant",
          parentId: "msg-user",
          role: "assistant",
          content: "I will inspect the fixture and preserve branch/session lineage.",
        },
      },
      {
        type: "response_item",
        timestamp: now,
        payload: {
          type: "function_call",
          id: "fc-read",
          parentId: "msg-assistant",
          call_id: "call-read",
          name: "read_file",
          arguments: "{\"path\":\"tests/source-adapters-fixtures.test.ts\"}",
        },
      },
      {
        type: "response_item",
        timestamp: now,
        payload: {
          type: "function_call_output",
          id: "out-read",
          parentId: "fc-read",
          call_id: "call-read",
          status: "failed",
          output: "ENOENT while reading fixture; github_pat_abcdefghijklmnopqrstuvwxyz123456",
        },
      },
      {
        type: "event_msg",
        timestamp: now,
        payload: {
          level: "error",
          type: "tool_failure",
          error: "read_file failed: ENOENT",
        },
      },
    ]);

    const detection = await detectSourceFile(path);
    expect(detection.ok).toBe(true);
    if (!detection.ok) {
      throw new Error("expected Codex source detection");
    }
    expect(detection.source).toMatchObject({
      sourceType: "codex-session-jsonl",
      sessionId: "codex-session-fixture",
      schemaVersion: "0.99.0",
    });

    const streamed = [];
    for await (const item of streamSourceFile(path)) {
      streamed.push(item);
    }
    const streamedRecords = streamed.filter((item) => item.kind === "record");
    expect(streamedRecords).toHaveLength(7);
    expect(streamedRecords[2]).toMatchObject({
      line: 3,
      recordIndex: 2,
      source: {
        sourceType: "codex-session-jsonl",
        sessionId: "codex-session-fixture",
      },
    });

    const redactedStream = [];
    for await (const item of streamSourceFile(path, { redact: true, redactionOptions: { maxTextExcerptChars: 48 } })) {
      redactedStream.push(item);
    }
    const redactedSerialized = JSON.stringify(redactedStream);
    expect(redactedSerialized).not.toContain("Bearer abcdefghijklmnopqrstuvwx");
    expect(redactedSerialized).not.toContain("github_pat_abcdefghijklmnopqrstuvwxyz123456");
    expect(redactedStream.some((item) =>
      item.kind === "record" && item.redacted?.redaction.redactionKinds.includes("authorization"))).toBe(true);
    expect(redactedStream.some((item) =>
      item.kind === "record" && item.redacted?.redaction.redactionKinds.includes("github_token"))).toBe(true);

    const canonical = streamedRecords.flatMap((item) =>
      canonicalizeSourceRecord({
        source: item.source,
        record: item.record,
        recordIndex: item.recordIndex,
        line: item.line,
        redactionOptions: { maxTextExcerptChars: 64 },
      }).records);

    expect(canonical.map((record) => record.span.attributes["source.adapter.event_kind"])).toEqual([
      "session",
      "turn_context",
      "message",
      "message",
      "tool_call",
      "tool_result",
      "error",
    ]);
    expect(canonical[4]?.lineage).toMatchObject({
      id: "fc-read",
      parentId: "msg-assistant",
      sessionId: "codex-session-fixture",
      toolCallIds: ["call-read"],
      toolNames: ["read_file"],
    });
    expect(canonical[5]?.span.attributes["tool.status"]).toBe("failed");
    expect(canonical[6]?.span.status).toEqual({
      code: "STATUS_CODE_ERROR",
      message: "read_file failed: ENOENT",
    });
    expect(JSON.stringify(canonical)).not.toContain("github_pat_abcdefghijklmnopqrstuvwxyz123456");
    expect(JSON.stringify(canonical)).toContain("feature/source-adapters");
    expect(canonicalSourceRecordsToJsonl(canonical)).toContain("\"source.adapter.type\":\"codex-session-jsonl\"");
  });

  test("keeps Pi session tree lineage for tool results, model changes, and compactions", async () => {
    const dir = await tempDir();
    const nested = join(dir, "sessions");
    await mkdir(nested);
    const path = join(nested, "pi-session.jsonl");
    await writeJsonl(path, [
      {
        type: "session",
        version: 3,
        id: "pi-session-fixture",
        timestamp: now,
        cwd: "/repo",
        branch: "feature/pi-source-adapter",
      },
      {
        type: "user",
        id: "node-user",
        parentId: null,
        timestamp: now,
        role: "user",
        content: [{ type: "text", text: "Run the coding task on feature/pi-source-adapter." }],
      },
      {
        type: "assistant",
        id: "node-assistant",
        parentId: "node-user",
        timestamp: now,
        role: "assistant",
        content: [{ type: "text", text: "I will edit the repository." }],
      },
      {
        type: "tool_use",
        id: "node-tool",
        parentId: "node-assistant",
        timestamp: now,
        tool_call_id: "call-write",
        name: "repo.write",
        input: {
          path: "src/source-adapters/boundary.ts",
          token: "ghp_abcdefghijklmnopqrstuvwxyz123456",
        },
      },
      {
        type: "tool_result",
        id: "node-result",
        parentId: "node-tool",
        timestamp: now,
        tool_call_id: "call-write",
        status: "error",
        error: "permission denied",
        output: "write failed",
      },
      {
        type: "model_change",
        id: "node-model",
        parentId: "node-result",
        timestamp: now,
        from: "qwen-local-executor",
        to: "gpt-5.5-master",
      },
      {
        type: "compaction",
        id: "node-compact",
        parentId: "node-model",
        timestamp: now,
        before: "long session context",
        summary: "kept branch/session lineage and tool failure context",
        branch: "feature/pi-source-adapter",
      },
    ]);

    const detection = await detectSourceFile(path);
    expect(detection.ok).toBe(true);
    if (!detection.ok) {
      throw new Error("expected Pi source detection");
    }
    expect(detection.source).toMatchObject({
      sourceType: "pi-session-jsonl",
      sessionId: "pi-session-fixture",
      schemaVersion: "3",
    });

    const streamed = [];
    for await (const item of streamSourceDirectory(dir, { redact: true, redactionOptions: { maxTextExcerptChars: 80 } })) {
      streamed.push(item);
    }
    expect(streamed.filter((item) => item.kind === "diagnostic")).toEqual([]);
    const records = streamed.filter((item) => item.kind === "record");
    expect(records).toHaveLength(7);
    expect(JSON.stringify(records)).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");
    expect(JSON.stringify(records)).toContain("[REDACTED:secret_field]");

    const outputs = records.map((item) =>
      canonicalizeSourceRecord({
        source: item.source,
        record: item.record,
        recordIndex: item.recordIndex,
        line: item.line,
        redactionOptions: { maxTextExcerptChars: 80 },
      }));
    expect(outputs.flatMap((output) => output.diagnostics)).toEqual([]);
    const canonical = outputs.flatMap((output) => output.records);

    expect(canonical.map((record) => record.span.attributes["source.adapter.event_kind"])).toEqual([
      "session",
      "message",
      "message",
      "tool_call",
      "tool_result",
      "model_change",
      "compaction",
    ]);
    expect(canonical[4]?.span.status).toEqual({
      code: "STATUS_CODE_ERROR",
      message: "permission denied",
    });
    expect(canonical[4]?.lineage).toMatchObject({
      id: "node-result",
      parentId: "node-tool",
      sessionId: "pi-session-fixture",
      toolCallIds: ["call-write"],
    });
    expect(canonical[4]?.span.parent_span_id).not.toBe("");
    expect(canonical[5]?.span.attributes).toMatchObject({
      "inference.llm.model_name": "gpt-5.5-master",
      "source.pi.previous_model": "qwen-local-executor",
    });
    expect(canonical[6]?.span.name).toBe("source.pi-session-jsonl.compaction");
    expect(JSON.stringify(canonical)).toContain("feature/pi-source-adapter");
  });

  test("runs a CC session v2 fixture through generic streaming and canonical output", async () => {
    const dir = await tempDir();
    const path = join(dir, "cc-session.jsonl");
    await writeJsonl(path, [
      {
        type: "user",
        uuid: "cc-user",
        sessionId: "cc-session-fixture",
        timestamp: now,
        cwd: "/repo",
        permissionMode: "acceptEdits",
        message: {
          role: "user",
          content: "Inspect the source adapter and keep Authorization: Bearer abcdefghijklmnopqrstuvwx private.",
        },
      },
      {
        type: "assistant",
        uuid: "cc-assistant",
        parentUuid: "cc-user",
        sessionId: "cc-session-fixture",
        timestamp: now,
        message: {
          role: "assistant",
          model: "claude-sonnet-4-7",
          content: [
            { type: "text", text: "I will inspect the canonical source adapter path." },
            {
              type: "tool_use",
              id: "toolu-read",
              name: "Read",
              input: {
                file_path: "src/source-adapters/canonical.ts",
                token: "ghp_abcdefghijklmnopqrstuvwxyz123456",
              },
            },
          ],
        },
      },
      {
        type: "user",
        uuid: "cc-tool-result",
        parentUuid: "cc-assistant",
        sessionId: "cc-session-fixture",
        timestamp: now,
        message: {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu-read",
              is_error: true,
              content: "ENOENT reading src/source-adapters/canonical.ts",
            },
          ],
        },
      },
    ]);

    const detection = await detectSourceFile(path);
    expect(detection.ok).toBe(true);
    if (!detection.ok) {
      throw new Error("expected CC source detection");
    }
    expect(detection.source).toMatchObject({
      sourceType: "cc-session-jsonl-v2",
      sessionId: "cc-session-fixture",
      schemaVersion: "claude-sonnet-4-7",
    });

    const streamed = [];
    for await (const item of streamSourceFile(path, { redact: true, redactionOptions: { maxTextExcerptChars: 80 } })) {
      streamed.push(item);
    }
    const records = streamed.filter((item) => item.kind === "record");
    expect(records).toHaveLength(3);
    expect(JSON.stringify(records)).not.toContain("Bearer abcdefghijklmnopqrstuvwx");
    expect(JSON.stringify(records)).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");

    const canonical = records.flatMap((item) =>
      canonicalizeSourceRecord({
        source: item.source,
        record: item.record,
        recordIndex: item.recordIndex,
        line: item.line,
        redactionOptions: { maxTextExcerptChars: 80 },
      }).records);

    expect(canonical.map((record) => record.span.attributes["source.adapter.event_kind"])).toEqual([
      "user_message",
      "assistant_message",
      "tool_call",
      "tool_result",
    ]);
    expect(canonical[2]?.lineage).toMatchObject({
      id: "cc-assistant",
      parentId: "cc-user",
      sessionId: "cc-session-fixture",
      toolCallIds: ["toolu-read"],
      toolNames: ["Read"],
    });
    expect(canonical[2]?.span.attributes).toMatchObject({
      "source.record.index": 1,
      "source.record.line": 2,
      "source.record.sub_event_index": 1,
    });
    expect(canonical[3]?.span.status).toEqual({
      code: "STATUS_CODE_ERROR",
      message: "ENOENT reading src/source-adapters/canonical.ts",
    });
    expect(JSON.stringify(canonical)).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");
    expect(canonicalSourceRecordsToJsonl(canonical)).toContain("\"source.adapter.type\":\"cc-session-jsonl-v2\"");
  });
});
