import { describe, expect, test } from "bun:test";
import {
  AcpReplayCaptureSchema,
  extractReplayEvalCaseSkeleton,
  groupAcpReplayRecords,
  replayCaptureToJson,
} from "../src/replay";

const now = "2026-05-01T00:00:00.000Z";

describe("ACP replay capture and extraction", () => {
  test("captures a read-only routing scenario and extracts split/oracle/source metadata", () => {
    const capture = AcpReplayCaptureSchema.parse({
      captureId: "capture.route.read-only",
      createdAt: now,
      source: {
        sourceType: "acp-session-jsonl",
        path: ".bag/sessions/route.jsonl",
        sessionId: "session.route.1",
        traceIds: ["trace.route.1"],
      },
      redactionStatus: "redacted",
      records: [
        {
          recordId: "record.prompt.route",
          recordKind: "prompt",
          promptRole: "user",
          content: "Tell me which release file lists the ship checklist. Do not edit files.",
          contentRedactionStatus: "redacted",
          traceRefs: [{ traceId: "trace.route.1", spanId: "span.prompt" }],
        },
        {
          recordId: "record.route.read-only",
          recordKind: "mode_route",
          promptRecordId: "record.prompt.route",
          parentRecordIds: ["record.prompt.route"],
          requestedMode: "auto",
          selectedMode: "read_only",
          sideEffectPolicy: "read_only",
          reason: "Question can be answered from repository facts without mutation.",
          traceRefs: [{ traceId: "trace.route.1", spanId: "span.route" }],
        },
        {
          recordId: "record.file.release",
          recordKind: "file_read",
          parentRecordIds: ["record.route.read-only"],
          path: "docs/release.md",
          status: "succeeded",
          contentHash: "sha256:release-doc",
          redactionStatus: "hash_only",
          ranges: [{ startLine: 0, endLine: 3 }],
          traceRefs: [{ traceId: "trace.route.1", spanId: "span.file" }],
        },
      ],
    });

    const groups = groupAcpReplayRecords(capture);
    expect(groups.prompts).toHaveLength(1);
    expect(groups.modeRoutes[0]?.selectedMode).toBe("read_only");
    expect(replayCaptureToJson(capture)).toContain("\"schemaVersion\": \"acp-replay-capture.v1\"");

    const skeleton = extractReplayEvalCaseSkeleton({
      capture,
      metadata: {
        evalCaseId: "replay.eval.route.read-only",
        title: "Read-only prompt routes without side effects",
        split: "train",
        splitRationale: "Synthetic first-slice route fixture is visible training data.",
        oracleStrength: "medium",
        expectedBehavior: {
          summary: "The agent should answer from docs/release.md and avoid workspace mutation.",
          assertions: [
            {
              assertionId: "assert.route.no-edits",
              assertionKind: "no_forbidden_path_changed",
              description: "Read-only routing must not write source files.",
              severity: "critical",
              paths: ["docs/release.md"],
            },
          ],
        },
        tags: ["replay", "routing", "read-only"],
      },
    });

    expect(skeleton.splitAssignment).toMatchObject({
      split: "train",
      assignedBy: "manual",
    });
    expect(skeleton.routing).toMatchObject({
      promptRecordIds: ["record.prompt.route"],
      routingRecordIds: ["record.route.read-only"],
      selectedMode: "read_only",
      sideEffectPolicy: "read_only",
    });
    expect(skeleton.oracle.strength).toBe("medium");
    expect(skeleton.redaction.status).toBe("redacted");
    expect(skeleton.sourceTraceIds).toEqual(["trace.route.1"]);
    expect(skeleton.sourceRefs.some((sourceRef) => sourceRef.recordId === "record.file.release")).toBe(true);
  });

  test("extracts edit and tool failure skeleton details with conservative redaction review", () => {
    const capture = AcpReplayCaptureSchema.parse({
      captureId: "capture.failure.edit-tool",
      createdAt: now,
      source: {
        sourceType: "spans-jsonl",
        sessionId: "session.failure.1",
        traceIds: ["trace.failure.1"],
      },
      redactionStatus: "raw_local_only",
      records: [
        {
          recordId: "record.prompt.failure",
          recordKind: "prompt",
          promptRole: "user",
          content: "Patch src/example.ts with the requested behavior.",
          contentRedactionStatus: "raw_local_only",
          traceRefs: [{ traceId: "trace.failure.1", spanId: "span.prompt" }],
        },
        {
          recordId: "record.route.mutate",
          recordKind: "mode_route",
          promptRecordId: "record.prompt.failure",
          parentRecordIds: ["record.prompt.failure"],
          selectedMode: "mutating",
          sideEffectPolicy: "write_allowed",
          traceRefs: [{ traceId: "trace.failure.1", spanId: "span.route" }],
        },
        {
          recordId: "record.edit.parse-failed",
          recordKind: "edit_attempt",
          parentRecordIds: ["record.route.mutate"],
          artifactRefs: [".bag/artifacts/edit-attempt.json"],
          traceRefs: [{ traceId: "trace.failure.1", spanId: "span.edit" }],
          attempt: {
            editAttemptId: "edit.attempt.parse-failure",
            runId: "run.acp.failure",
            traceId: "trace.failure.1",
            modelProfileId: "model.qwen36.local",
            codebaseProfileId: "codebase.bleeding-agent",
            policyId: "policy.qwen36.bleeding-agent",
            editStrategyId: "edit.unified-diff.v1",
            editStrategyFamily: "unified_diff",
            targetFiles: ["src/example.ts"],
            readSnapshotRefs: [
              {
                snapshotId: "snapshot.example.before",
                path: "src/example.ts",
                contentHash: "sha256:before",
                wholeFileSeen: true,
              },
            ],
            phaseResults: [
              {
                phase: "parse",
                status: "failed",
                errorCode: "parse_error",
                artifactRefs: [".bag/artifacts/parse-error.txt"],
              },
            ],
            parseErrorCode: "parse_error",
            verificationStatus: "not_run",
            redactionStatus: "redacted",
            artifactRefs: [".bag/artifacts/edit-attempt.json"],
            createdAt: now,
          },
        },
        {
          recordId: "record.tool.apply-malformed",
          recordKind: "tool_call",
          parentRecordIds: ["record.edit.parse-failed"],
          toolCallId: "tool.call.apply-malformed",
          namespace: "acp",
          name: "apply_patch",
          status: "malformed_args",
          args: { patch: "@@ malformed" },
          result: { error: "expected unified diff header" },
          resultStyle: "structured_error",
          retryCount: 1,
          redactionStatus: "redacted",
          errorCode: "invalid_patch",
          traceRefs: [{ traceId: "trace.failure.1", spanId: "span.tool" }],
        },
      ],
    });

    const skeleton = extractReplayEvalCaseSkeleton({
      capture,
      metadata: {
        evalCaseId: "replay.eval.failure.edit-tool",
        title: "Malformed edit attempt reports recoverable tool failure",
        split: "dev",
        oracleStrength: "strong",
        expectedBehavior: {
          summary:
            "The replay should preserve the parse failure and malformed tool call so a later runner can test fallback behavior.",
          assertions: [
            {
              assertionId: "assert.parse-failure-visible",
              assertionKind: "json_pointer_equals",
              description: "The skeleton exposes the edit parse failure.",
              artifact: "telemetry",
              pointer: "/observedFailures/0/errorCode",
              expected: "parse_error",
            },
          ],
        },
        tags: ["replay", "edit-failure", "tool-failure"],
      },
    });

    expect(skeleton.redaction).toMatchObject({
      status: "needs_review",
      needsReview: true,
      needsReviewRecordIds: ["record.prompt.failure"],
    });
    expect(skeleton.routing.selectedMode).toBe("mutating");
    expect(skeleton.observedFailures.map((failure) => failure.failureKind)).toEqual([
      "edit_attempt",
      "tool_call",
    ]);
    expect(skeleton.observedFailures[0]).toMatchObject({
      recordId: "record.edit.parse-failed",
      phase: "parse",
      errorCode: "parse_error",
    });
    expect(skeleton.observedFailures[1]).toMatchObject({
      recordId: "record.tool.apply-malformed",
      status: "malformed_args",
      errorCode: "invalid_patch",
    });
    expect(skeleton.split).toBe("dev");
    expect(skeleton.sourceTraceIds).toEqual(["trace.failure.1"]);
  });
});
