import { describe, expect, test } from "bun:test";
import {
  redactSourceRecord,
  redactSourceRecords,
  withDangerousUnredactedSourceRecordContent,
  withFullSourceRecordContent,
} from "../src/source-adapters/redaction";
import type { SourceMetadata } from "../src/source-adapters/boundary";

const source: SourceMetadata = {
  sourceType: "codex-session-jsonl",
  path: "/repo/.codex/session.jsonl",
  sessionId: "session-a",
  schemaVersion: "0.99.0",
  inspectedRecordCount: 1,
  detectedSignals: ["codex-session-meta", "codex-record-type-envelope"],
};

describe("source adapter redaction", () => {
  test("redacts nested secret-looking values by default", () => {
    const result = redactSourceRecord({
      type: "response_item",
      payload: {
        role: "user",
        content: "Use OPENAI_API_KEY=sk-testsecretvalue1234567890 and Authorization: Bearer abcdefghijklmnop",
        headers: {
          authorization: "Bearer should-not-leak-123456",
          nested: {
            githubToken: "ghp_abcdefghijklmnopqrstuvwxyz123456",
          },
        },
      },
    });

    const serialized = JSON.stringify(result.record);
    expect(serialized).not.toContain("sk-testsecretvalue1234567890");
    expect(serialized).not.toContain("should-not-leak");
    expect(serialized).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");
    expect(serialized).toContain("[REDACTED:openai_api_key]");
    expect(serialized).toContain("[REDACTED:authorization]");
    expect(serialized).toContain("[REDACTED:secret_field]");
    expect(result.redaction.secretReplacementCount).toBeGreaterThanOrEqual(3);
    expect(result.redaction.redactionKinds).toContain("openai_api_key");
    expect(result.redaction.redactionKinds).toContain("authorization");
    expect(result.redaction.redactionKinds).toContain("secret_field");
  });

  test("caps text excerpts while preserving structural role fields", () => {
    const longText = "x".repeat(80);
    const result = redactSourceRecord({
      type: "response_item",
      payload: {
        role: "assistant",
        content: longText,
      },
    }, { maxTextExcerptChars: 16 });

    expect(result.record).toEqual({
      type: "response_item",
      payload: {
        role: "assistant",
        content: "x".repeat(16) + "...[TRUNCATED:64_chars]",
      },
    });
    expect(result.redaction.truncatedStringCount).toBe(1);
  });

  test("preserves lineage, source metadata, roles, and tool-call identifiers", () => {
    const result = redactSourceRecord({
      type: "response_item",
      id: "entry-a",
      parentId: "entry-root",
      payload: {
        role: "assistant",
        message: {
          tool_calls: [
            {
              id: "call-read",
              type: "function",
              function: {
                name: "read_file",
                arguments: "{\"path\":\"/repo/src/index.ts\",\"token\":\"sk-testsecretvalue1234567890\"}",
              },
            },
          ],
        },
      },
    }, { source });

    expect(result.lineage).toMatchObject({
      source,
      id: "entry-a",
      parentId: "entry-root",
      sessionId: "session-a",
      role: "assistant",
      toolCallIds: ["call-read"],
      toolNames: ["read_file"],
    });
    expect(JSON.stringify(result.record)).toContain("\"role\":\"assistant\"");
    expect(JSON.stringify(result.record)).toContain("\"id\":\"call-read\"");
    expect(JSON.stringify(result.record)).toContain("\"name\":\"read_file\"");
    expect(JSON.stringify(result.record)).not.toContain("sk-testsecretvalue1234567890");
  });

  test("keeps full content only through explicit opt-in while still redacting secrets", () => {
    const result = redactSourceRecord({
      type: "response_item",
      payload: {
        role: "assistant",
        content: `${"full-content ".repeat(20)}sk-testsecretvalue1234567890`,
      },
    }, withFullSourceRecordContent({ maxTextExcerptChars: 24 }));

    const serialized = JSON.stringify(result.record);
    expect(serialized).toContain("full-content full-content full-content");
    expect(serialized).not.toContain("...[TRUNCATED:");
    expect(serialized).not.toContain("sk-testsecretvalue1234567890");
    expect(serialized).toContain("[REDACTED:openai_api_key]");
    expect(result.redaction.fullContentIncluded).toBe(true);
    expect(result.redaction.secretRedactionDisabled).toBe(false);
  });

  test("only the dangerous helper disables secret redaction", () => {
    const result = redactSourceRecord({
      type: "response_item",
      payload: {
        role: "user",
        content: "sk-testsecretvalue1234567890",
      },
    }, withDangerousUnredactedSourceRecordContent());

    expect(JSON.stringify(result.record)).toContain("sk-testsecretvalue1234567890");
    expect(result.redaction.secretReplacementCount).toBe(0);
    expect(result.redaction.secretRedactionDisabled).toBe(true);
  });

  test("redacts batches with shared source metadata", () => {
    const results = redactSourceRecords([
      {
        type: "session_meta",
        payload: {
          id: "session-a",
          role: "system",
        },
      },
    ], { source });

    expect(results).toHaveLength(1);
    expect(results[0]?.lineage.source).toBe(source);
    expect(results[0]?.lineage.sessionId).toBe("session-a");
  });
});
