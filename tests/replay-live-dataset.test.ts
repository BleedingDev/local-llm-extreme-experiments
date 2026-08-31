import { describe, expect, test } from "bun:test";
import {
  extractReplayDatasetCaseFromCapture,
  redactAcpReplayCaptureForLocalSafeUse,
  selectReplayCasesForOptimizerInput,
  type AcpReplayCapture,
} from "../src/replay";

const createdAt = "2026-05-01T12:00:00.000Z";

const liveCapture = (overrides: Partial<AcpReplayCapture> = {}): AcpReplayCapture => ({
  captureId: "capture.live.redaction",
  schemaVersion: "acp-replay-capture.v1",
  createdAt,
  source: {
    sourceType: "manual",
    path: "/Users/satan/project/.bag/runs/run-1/replay-capture.json",
    sessionId: "session.live.redaction",
    traceIds: ["trace.live.redaction"],
  },
  context: {
    modelRole: "executor",
    provider: "local-openai-compatible",
    policyId: "policy.live.redaction",
    modelProfileId: "model.live.redaction",
    codebaseProfileId: "codebase.live.redaction",
    modelServerId: "server.live.redaction",
    modelServerProfileId: "server-profile.live.redaction",
    canonicalToolVersion: "canonical-tools.live.redaction",
    renderedToolVersion: "rendered-tools.live.redaction",
    resultStyleVersion: "result-style.live.redaction",
    verificationPolicyVersion: "verification.live.redaction",
  },
  defaultSplitHint: "dev",
  redactionStatus: "raw_local_only",
  records: [
    {
      recordId: "record.live.prompt",
      recordKind: "prompt",
      promptRole: "user",
      content: "Please inspect this. OPENAI_API_KEY=sk-abcdefghijklmnop and keep it private.",
      contentRedactionStatus: "raw_local_only",
    },
    {
      recordId: "record.live.route",
      recordKind: "mode_route",
      promptRecordId: "record.live.prompt",
      parentRecordIds: ["record.live.prompt"],
      requestedMode: "auto",
      selectedMode: "chat",
      restoredMode: "auto",
      sideEffectPolicy: "no_side_effects",
      reason: "Conversation only.",
    },
    {
      recordId: "record.live.file",
      recordKind: "file_read",
      parentRecordIds: ["record.live.route"],
      path: "/Users/satan/project/src/secret.ts",
      status: "succeeded",
      contentHash: "sha256:before",
      excerpt: "const token = 'ghp_abcdefghijklmnopqrstuvwxyz';\n".repeat(20),
      redactionStatus: "raw_local_only",
    },
    {
      recordId: "record.live.tool",
      recordKind: "tool_call",
      parentRecordIds: ["record.live.route"],
      toolCallId: "tool.live.write",
      namespace: "mcp.workspace",
      name: "mcp.workspace.write_file",
      status: "permission_denied",
      args: { path: "/Users/satan/project/out.txt", token: "ghp_abcdefghijklmnopqrstuvwxyz" },
      result: { error: "Permission denied for sk-abcdefghijklmnop" },
      resultStyle: "structured_error",
      retryCount: 0,
      redactionStatus: "raw_local_only",
      errorCode: "permission_denied",
    },
    {
      recordId: "record.live.terminal",
      recordKind: "terminal_command",
      parentRecordIds: ["record.live.tool"],
      commandId: "terminal.live.typecheck",
      command: ["npm", "run", "typecheck"],
      cwd: "/Users/satan/project",
      status: "failed",
      exitCode: 1,
      signal: null,
      stdoutArtifactRef: "artifact:stdout",
      stderrArtifactRef: "artifact:stderr",
      redactionStatus: "raw_local_only",
      errorCode: "verifier_error",
    },
    {
      recordId: "record.live.artifact",
      recordKind: "artifact_ref",
      artifactRef: "artifact:trace",
      artifactKind: "trace",
      path: "/Users/satan/private/outside/trace.json",
      contentHash: "sha256:trace",
      redactionStatus: "raw_local_only",
    },
  ],
  ...overrides,
});

describe("live replay dataset extraction", () => {
  test("redacts live ACP captures into optimizer-safe records while preserving roles and lineage", () => {
    const result = redactAcpReplayCaptureForLocalSafeUse(liveCapture(), {
      rootPath: "/Users/satan/project",
      maxTextExcerptChars: 48,
    });
    const serialized = JSON.stringify(result.capture);

    expect(result.capture.redactionStatus).toBe("redacted");
    expect(result.report.secretReplacementCount).toBeGreaterThan(0);
    expect(result.report.hashOnlyRecordCount).toBeGreaterThan(0);
    expect(serialized).not.toContain("sk-abcdefghijklmnop");
    expect(serialized).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz");
    expect(result.capture.records[0]).toMatchObject({
      recordKind: "prompt",
      promptRole: "user",
      contentRedactionStatus: "redacted",
    });
    expect(result.capture.records[2]).toMatchObject({
      recordKind: "file_read",
      path: "src/secret.ts",
      redactionStatus: "hash_only",
    });
    expect(serialized).toContain("path:sha256:");
  });

  test("extracts a redacted live capture into a replay case and keeps holdout out of optimizer input", () => {
    const datasetCase = extractReplayDatasetCaseFromCapture(liveCapture(), {
      sourcePath: ".bag/runs/run-1/replay-capture.json",
      redaction: { rootPath: "/Users/satan/project" },
    });
    const selection = selectReplayCasesForOptimizerInput([datasetCase.replayCase], "optimization_selection");

    expect(datasetCase.replayCase).toMatchObject({
      evalCaseId: "replay.eval.live.capture.live.redaction",
      split: "dev",
      sourceSessionId: "session.live.redaction",
    });
    expect(datasetCase.replayCase.observedFailures.map((failure) => failure.failureKind)).toEqual(
      expect.arrayContaining(["tool_call", "terminal_command"]),
    );
    expect(datasetCase.replayCase.redaction.needsReview).toBe(false);
    expect(selection.selectedEvalCaseIds).toEqual(["replay.eval.live.capture.live.redaction"]);

    const holdoutCase = extractReplayDatasetCaseFromCapture(liveCapture({ defaultSplitHint: "holdout" }), {
      redaction: { rootPath: "/Users/satan/project" },
    });
    const holdoutSelection = selectReplayCasesForOptimizerInput([holdoutCase.replayCase], "optimization_selection");
    expect(holdoutCase.replayCase.split).toBe("holdout");
    expect(holdoutSelection.selectedEvalCaseIds).toEqual([]);
    expect(holdoutSelection.hiddenHoldoutEvalCaseIds).toEqual(["replay.eval.live.capture.live.redaction"]);
  });

  test("explicit raw-local opt-in remains excluded from optimizer input", () => {
    const datasetCase = extractReplayDatasetCaseFromCapture(liveCapture(), {
      redaction: { includeRawLocalContent: true },
    });
    const selection = selectReplayCasesForOptimizerInput([datasetCase.replayCase], "proposer_prompt");

    expect(datasetCase.capture.redactionStatus).toBe("raw_local_only");
    expect(datasetCase.redactionReport.rawLocalContentRetained).toBe(true);
    expect(datasetCase.replayCase.redaction.needsReview).toBe(true);
    expect(selection.selectedEvalCaseIds).toEqual([]);
    expect(selection.rejectedCases[0]?.reasons.join("\n")).toContain("redaction status needs_review");
  });
});
