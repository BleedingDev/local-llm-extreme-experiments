import { describe, expect, test } from "bun:test";
import { canonicalizeSourceRecords } from "../src/source-adapters/canonical";
import { exportCanonicalSourceRecordsToReplayCase } from "../src/source-adapters/replay";
import type { SourceMetadata } from "../src/source-adapters/boundary";

const now = "2026-04-30T00:00:00.000Z";

const ccSource: SourceMetadata = {
  sourceType: "cc-session-jsonl-v2",
  path: "/private/tmp/cc-session.jsonl",
  sessionId: "cc-session-replay",
  schemaVersion: "claude-sonnet-4-7",
  inspectedRecordCount: 3,
  detectedSignals: ["cc-transcript-type-envelope", "cc-uuid-lineage"],
};

describe("source adapter replay export", () => {
  test("exports redacted canonical spans to split-safe replay cases with source lineage", () => {
    const canonical = canonicalizeSourceRecords({
      source: ccSource,
      records: [
        {
          type: "user",
          uuid: "cc-user",
          sessionId: "cc-session-replay",
          timestamp: now,
          message: {
            role: "user",
            content: "Run the verification command and report the failure.",
          },
        },
        {
          type: "assistant",
          uuid: "cc-assistant",
          parentUuid: "cc-user",
          sessionId: "cc-session-replay",
          timestamp: now,
          message: {
            role: "assistant",
            model: "claude-sonnet-4-7",
            content: [
              {
                type: "tool_use",
                id: "toolu-bash",
                name: "Bash",
                input: {
                  command: "npm test",
                  token: "ghp_abcdefghijklmnopqrstuvwxyz123456",
                },
              },
            ],
          },
        },
        {
          type: "user",
          uuid: "cc-result",
          parentUuid: "cc-assistant",
          sessionId: "cc-session-replay",
          timestamp: now,
          message: {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu-bash",
                is_error: true,
                content: "Process exited with code 1",
              },
            ],
          },
        },
      ],
      redactionOptions: { maxTextExcerptChars: 80 },
    });

    const exported = exportCanonicalSourceRecordsToReplayCase(canonical.records, {
      captureId: "capture.source-adapter.cc-bash",
      createdAt: now,
      defaultSplitHint: "dev",
      minimumDistinctSessionCount: 50,
    });

    expect(exported.capture.source).toMatchObject({
      sourceType: "cc-session-jsonl-v2",
      sessionId: "cc-session-replay",
    });
    expect(exported.capture.redactionStatus).toBe("redacted");
    expect(exported.replayCase).toMatchObject({
      evalCaseId: expect.stringMatching(/^replay\.eval\.source-adapter\./),
      split: "dev",
      captureId: "capture.source-adapter.cc-bash",
      sourceSessionId: "cc-session-replay",
      oracle: {
        strength: "weak",
      },
    });
    expect(exported.replayCase.oracle.expectedBehavior.summary).toContain("observed");
    expect(exported.replayCase.oracle.expectedBehavior.summary).not.toContain("golden");
    expect(exported.replayCase.tags).toEqual(expect.arrayContaining([
      "source-adapter",
      "observed-baseline",
      "bash_nonzero",
    ]));
    expect(exported.replayCase.observedFailures).toEqual([
      expect.objectContaining({
        failureKind: "terminal_command",
        status: "failed",
        errorCode: "bash_nonzero",
      }),
    ]);
    expect(exported.replayCase.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "span",
      traceId: canonical.records[2]?.span.trace_id,
      spanId: canonical.records[2]?.span.span_id,
      redactionStatus: "hash_only",
    }));
    expect(JSON.stringify(exported)).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");
    expect(exported.blocker).toMatchObject({
      code: "insufficient_safe_sessions",
      distinctSessionCount: 1,
      requiredDistinctSessionCount: 50,
    });
  });
});
