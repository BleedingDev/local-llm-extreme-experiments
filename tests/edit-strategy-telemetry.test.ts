import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultConfig } from "../src/config";
import type { EditAttemptContract } from "../src/edit-strategy/types";
import type { EditApplyResult } from "../src/edit-strategy/apply-layer";
import {
  editAttemptFromAcpWrite,
  finalizeEditAttemptLifecycle,
} from "../src/acp/edit-telemetry";
import type { CodingEditOperation, CodingFileSnapshot } from "../src/acp/coding-types";
import type { BagAcpSession } from "../src/acp/session";
import { RunTelemetry } from "../src/telemetry";
import { analyzeHaloSpans, readHaloSpans, renderTraceAnalysisMarkdown } from "../src/trace-analysis";
import { TraceStore } from "../src/trace-store";

const optimizerPin = (cwd: string) => ({
  modelRole: "local" as const,
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  policyId: "policy.qwen36.bleeding-agent",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
  editStrategyVersion: "edit-strategy.v1",
  renderedEditContractVersion: "rendered-edit-contract.v1",
  editFallbackPolicyVersion: "edit-fallback.v1",
  editRepairPolicyVersion: "edit-repair.v1",
  editVerifierPolicyVersion: "edit-verifier.v1",
  editObjectiveSetId: "edit-objectives.default.v1",
  source: "seed" as const,
  registryRoot: join(cwd, ".bag", "optimizer"),
});

const appliedButBrokenAttempt: EditAttemptContract = {
  schemaVersion: "edit-attempt.v1",
  editAttemptId: "edit.attempt.applied-broken",
  runId: "run-edit-telemetry",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  policyId: "policy.qwen36.bleeding-agent",
  editStrategyId: "edit.apply-patch.v1",
  editStrategyFamily: "apply_patch",
  canonicalEditToolSpecId: "edit-tool.apply-patch.v1",
  renderedEditToolContractId: "rendered-edit.apply-patch.qwen36",
  renderedEditContractVersion: "rendered-edit-contract.v1",
  targetFiles: ["src/example.ts"],
  readSnapshotRefs: [
    {
      snapshotId: "snapshot.src-example",
      path: "src/example.ts",
      contentHash: "sha256:before",
      wholeFileSeen: true,
    },
  ],
  inputContentHashes: { "src/example.ts": "sha256:before" },
  outputContentHashes: { "src/example.ts": "sha256:after" },
  targetContentHashes: [
    {
      path: "src/example.ts",
      beforeHash: "sha256:before",
      afterHash: "sha256:after",
      readSnapshotId: "snapshot.src-example",
      writeArtifactRef: "artifact://edit/write-result",
    },
  ],
  phaseResults: [
    { phase: "generation", status: "passed", durationMs: 80 },
    { phase: "parse", status: "passed", durationMs: 4 },
    { phase: "validate", status: "passed", durationMs: 3 },
    { phase: "preview", status: "passed", durationMs: 9 },
    { phase: "stale_context_check", status: "passed", durationMs: 2 },
    { phase: "apply", status: "passed", durationMs: 12 },
    { phase: "write", status: "passed", durationMs: 6 },
    { phase: "post_apply_consistency", status: "failed", errorCode: "post_apply_syntax_failure", durationMs: 35 },
    { phase: "verify", status: "failed", errorCode: "verifier_error", durationMs: 120 },
    { phase: "self_check", status: "passed", durationMs: 50 },
    {
      phase: "repair",
      status: "passed",
      durationMs: 30,
      artifactRefs: ["artifact://repair/round-1"],
    },
    {
      phase: "rollback",
      status: "skipped",
      durationMs: 2,
      artifactRefs: ["artifact://rollback/not-attempted"],
      attributes: { status: "not_attempted" },
    },
  ],
  staleContextStatus: "fresh",
  permissionStatus: "bypassed_yolo",
  verificationStatus: "failed",
  postApplyConsistencyStatus: "inconsistent",
  selfDetectedRegressionStatus: "confirmed",
  selfDetectedRegressionEvidenceRefs: ["artifact://verify/typecheck.log"],
  selfDetectedRegressionEvidence: [
    {
      evidenceRef: "artifact://verify/typecheck.log",
      evidenceKind: "verification",
      phase: "verify",
      status: "confirmed",
      artifactRefs: ["artifact://verify/typecheck.log"],
    },
  ],
  repairAttemptCount: 1,
  repairAttemptRefs: [
    {
      repairRound: 1,
      triggerPhase: "verify",
      status: "passed",
      artifactRefs: ["artifact://repair/round-1"],
    },
  ],
  rollbackStatus: "not_attempted",
  fallbackFromStrategyId: "edit.apply-patch.v1",
  fallbackToStrategyId: "edit.whole-file.acp-write.v1",
  fallbackPath: [
    {
      fromStrategyId: "edit.apply-patch.v1",
      toStrategyId: "edit.whole-file.acp-write.v1",
      trigger: "verification_failed",
      status: "passed",
      artifactRefs: ["artifact://edit/fallback-route.json"],
    },
  ],
  tokenUsage: {
    promptTokens: 100,
    completionTokens: 20,
    totalTokens: 120,
  },
  latencyMs: 321,
  changedFileCount: 1,
  changedLineCount: 4,
  protectedPathTouched: false,
  redactionStatus: "redacted",
  artifactRefs: ["artifact://edit/diff.patch", "artifact://verify/typecheck.log"],
  createdAt: "2026-04-30T12:00:00.000Z",
  completedAt: "2026-04-30T12:00:00.321Z",
};

const session = (cwd: string): BagAcpSession => ({
  id: "session.edit.telemetry",
  cwd,
  yolo: true,
  optimizerPin: {
    telemetry: optimizerPin(cwd),
  },
} as BagAcpSession);

const fileSnapshot = (content = "value=1\n"): CodingFileSnapshot => ({
  kind: "existing",
  path: "/repo/src/example.txt",
  relativePath: "src/example.txt",
  content,
  hash: "before-hash",
});

const edit = (overrides: Partial<CodingEditOperation> = {}): CodingEditOperation => ({
  reason: "test edit",
  editInput: {
    strategyFamily: "hash_range",
    payload: {
      operations: [{
        path: "src/example.txt",
        startLine: 1,
        endLine: 1,
        expectedContentHash: "sha256:expected",
        replacement: "value=2\n",
      }],
    },
  },
  targetFiles: ["src/example.txt"],
  editStrategyId: "edit.hash-range.experimental.v1",
  editStrategyFamily: "hash_range",
  renderedEditToolContractId: "rendered.edit.hash-range.test",
  ...overrides,
});

const failedApply = (errorCode: EditApplyResult["errorCode"], protectedPathTouched = false): EditApplyResult => ({
  strategyFamily: "hash_range",
  status: "failed",
  changedFiles: [],
  errorCode,
  errorMessage: errorCode,
  previewDiff: "",
  protectedPathTouched,
});

describe("edit strategy telemetry", () => {
  test("records applied-but-broken edit attempts as first-class trace dimensions", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-edit-telemetry-"));
    const config = defaultConfig();
    mkdirSync(join(cwd, ".bag", "telemetry"), { recursive: true });
    const telemetry = new RunTelemetry(config, "run-edit-telemetry", cwd, optimizerPin(cwd));

    const parsed = telemetry.recordEditAttempt(appliedButBrokenAttempt);

    const events = readFileSync(join(cwd, ".bag", "telemetry", "events.jsonl"), "utf8");
    const spans = readHaloSpans(config, cwd);
    const report = analyzeHaloSpans(spans);
    const overview = TraceStore.open(config, cwd).getOverview();
    const brokenEdits = TraceStore.open(config, cwd).queryTraces({
      editPostApplyConsistencyStatus: "inconsistent",
      editStrategyId: "edit.apply-patch.v1",
    });

    expect(parsed.editAttemptId).toBe("edit.attempt.applied-broken");
    expect(events).toContain('"type":"edit.attempt"');
    expect(events).toContain('"postApplyConsistencyStatus":"inconsistent"');
    const editSpan = spans.find((span) => span.attributes["edit.attempt_id"] === "edit.attempt.applied-broken");
    expect(editSpan?.status.code).toBe("STATUS_CODE_ERROR");
    expect(editSpan?.attributes["inference.observation_kind"]).toBe("EDIT");
    expect(editSpan?.attributes["edit.post_apply_consistency_status"]).toBe("inconsistent");
    expect(editSpan?.attributes["edit.self_detected_regression_status"]).toBe("confirmed");
    expect(editSpan?.attributes["edit.self_detected_regression_evidence_count"]).toBe(2);
    expect(editSpan?.attributes["edit.fallback_to_strategy_id"]).toBe("edit.whole-file.acp-write.v1");
    expect(editSpan?.attributes["edit.fallback_path_length"]).toBe(1);
    expect(editSpan?.attributes["edit.phase.verify.error_code"]).toBe("verifier_error");
    expect(editSpan?.attributes["edit.phase.rollback.status"]).toBe("skipped");
    expect(editSpan?.attributes["edit.target_hash.before_count"]).toBe(1);
    expect(editSpan?.attributes["edit.target_hash.after_count"]).toBe(1);
    expect(editSpan?.attributes["edit.required_phase_coverage_status"]).toBe("complete");
    expect(editSpan?.attributes["edit.capture_status"]).toBe("complete");
    expect(editSpan?.attributes["edit.token_count.total"]).toBe(120);
    expect(report.errorSpanCount).toBe(1);
    expect(report.optimizerDimensions.editStrategyVersions).toContain("edit-strategy.v1");
    expect(report.optimizerDimensions.editStrategyIds).toContain("edit.apply-patch.v1");
    expect(report.optimizerDimensions.editStrategyFamilies).toContain("apply_patch");
    expect(report.optimizerDimensions.renderedEditToolContractIds).toContain("rendered-edit.apply-patch.qwen36");
    expect(renderTraceAnalysisMarkdown(report)).toContain("editStrategyId: edit.apply-patch.v1");
    expect(overview.editStrategyVersions).toContain("edit-strategy.v1");
    expect(overview.editStrategyIds).toContain("edit.apply-patch.v1");
    expect(overview.editPostApplyConsistencyStatuses).toContain("inconsistent");
    expect(overview.editSelfDetectedRegressionStatuses).toContain("confirmed");
    expect(overview.editRedactionStatuses).toContain("redacted");
    expect(brokenEdits.total).toBe(1);
  });

  test("maps ACP preview parse, stale, and protected failures into phase-level edit evidence", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-edit-lifecycle-failures-"));
    const parseAttempt = editAttemptFromAcpWrite({
      session: session(cwd),
      editStartedAt: "2026-04-30T12:00:00.000Z",
      edit: edit({
        editInput: {
          strategyFamily: "apply_patch",
          payload: { patch: "not a patch" },
        },
        editStrategyId: "edit.apply-patch.v1",
        editStrategyFamily: "apply_patch",
      }),
      targetFiles: ["src/example.txt"],
      fileSnapshots: [fileSnapshot()],
      applyResult: {
        strategyFamily: "apply_patch",
        status: "failed",
        changedFiles: [],
        errorCode: "parse_error",
        errorMessage: "bad patch",
        previewDiff: "",
        protectedPathTouched: false,
      },
      writeResults: [],
    });
    const staleAttempt = editAttemptFromAcpWrite({
      session: session(cwd),
      editStartedAt: "2026-04-30T12:00:00.000Z",
      edit: edit(),
      targetFiles: ["src/example.txt"],
      fileSnapshots: [fileSnapshot()],
      applyResult: failedApply("hash_mismatch"),
      writeResults: [],
    });
    const protectedAttempt = editAttemptFromAcpWrite({
      session: session(cwd),
      editStartedAt: "2026-04-30T12:00:00.000Z",
      edit: edit({
        editStrategyId: "edit.whole-file.acp-write.v1",
        editStrategyFamily: "whole_file",
        editInput: {
          strategyFamily: "whole_file",
          payload: {
            path: "package-lock.json",
            content: "{}\n",
          },
        },
        targetFiles: ["package-lock.json"],
      }),
      targetFiles: ["package-lock.json"],
      fileSnapshots: [fileSnapshot()],
      applyResult: {
        ...failedApply("protected_path_violation", true),
        strategyFamily: "whole_file",
      },
      writeResults: [],
    });

    expect(parseAttempt).toMatchObject({
      parseErrorCode: "parse_error",
      staleContextStatus: "not_checked",
    });
    expect(parseAttempt.applyErrorCode).toBeUndefined();
    expect(parseAttempt.phaseResults.find((phase) => phase.phase === "parse")).toMatchObject({
      status: "failed",
      errorCode: "parse_error",
    });
    expect(parseAttempt.phaseResults.find((phase) => phase.phase === "apply")).toMatchObject({
      status: "not_started",
    });
    expect(staleAttempt).toMatchObject({
      applyErrorCode: "hash_mismatch",
      staleContextStatus: "stale",
    });
    expect(staleAttempt.phaseResults.find((phase) => phase.phase === "stale_context_check")).toMatchObject({
      status: "failed",
      errorCode: "hash_mismatch",
    });
    expect(protectedAttempt).toMatchObject({
      applyErrorCode: "protected_path_violation",
      protectedPathTouched: true,
    });
    expect(protectedAttempt.phaseResults.find((phase) => phase.phase === "apply")).toMatchObject({
      status: "failed",
      errorCode: "protected_path_violation",
    });
  });

  test("finalizes applied-but-broken lifecycle evidence with verification, repair, and rollback phases", () => {
    const finalized = finalizeEditAttemptLifecycle({
      attempt: appliedButBrokenAttempt,
      postApplyChecks: [{
        path: "src/example.ts",
        status: "inconsistent",
        expectedHash: "sha256:after",
        actualHash: "sha256:corrupt",
        reason: "post-write self-check found changed content",
        errorCode: "post_apply_behavior_failure",
      }],
      commandResults: [{
        command: "bun",
        args: ["test"],
        reason: "verification",
        exitCode: 1,
        signal: null,
        output: "syntax error",
      }],
      rollbackResults: [{
        path: "/repo/src/example.ts",
        ok: true,
        reason: "rollback",
        editStrategyId: "edit.rollback.acp-write.v1",
        editStatus: "rollback_applied",
      }],
      artifactRefs: ["artifact://verify/bun-test.log"],
    });

    expect(finalized).toMatchObject({
      postApplyConsistencyStatus: "inconsistent",
      verificationStatus: "failed",
      selfDetectedRegressionStatus: "confirmed",
      rollbackStatus: "succeeded",
    });
    expect(finalized.phaseResults.find((phase) => phase.phase === "post_apply_consistency")).toMatchObject({
      status: "failed",
      errorCode: "post_apply_behavior_failure",
    });
    expect(finalized.phaseResults.find((phase) => phase.phase === "verify")).toMatchObject({
      status: "failed",
      errorCode: "verifier_error",
    });
    expect(finalized.phaseResults.find((phase) => phase.phase === "repair")).toMatchObject({
      status: "passed",
    });
    expect(finalized.phaseResults.find((phase) => phase.phase === "rollback")).toMatchObject({
      status: "passed",
    });
  });

  test("records create-target evidence without requiring an existing snapshot or rollback on skipped verification", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-edit-greenfield-telemetry-"));
    const attempt = editAttemptFromAcpWrite({
      session: session(cwd),
      editStartedAt: "2026-04-30T12:00:00.000Z",
      edit: edit({
        editInput: {
          strategyFamily: "whole_file",
          payload: {
            path: "answer.py",
            content: "print('ok')\n",
          },
        },
        targetFiles: ["answer.py"],
        editStrategyId: "edit.whole-file.acp-write.v1",
        editStrategyFamily: "whole_file",
      }),
      targetFiles: ["answer.py"],
      fileSnapshots: [{
        kind: "create",
        path: join(cwd, "answer.py"),
        relativePath: "answer.py",
        content: "",
        hash: "empty",
      }],
      applyResult: {
        strategyFamily: "whole_file",
        status: "applied",
        changedFiles: [{
          path: "answer.py",
          afterContent: "print('ok')\n",
          changeKind: "added",
        }],
        previewDiff: "",
        protectedPathTouched: false,
      },
      writeResults: [{
        path: join(cwd, "answer.py"),
        ok: true,
        reason: "create answer",
        editStrategyId: "edit.whole-file.acp-write.v1",
        editStatus: "applied",
        oldHash: "empty",
        newHash: "new",
        newContent: "print('ok')\n",
      }],
    });

    expect(attempt.readSnapshotRefs).toEqual([]);
    expect(attempt.inputContentHashes).toEqual({});
    expect(attempt.targetContentHashes).toEqual([
      expect.objectContaining({
        path: "answer.py",
        afterHash: expect.stringMatching(/^sha256:/),
      }),
    ]);
    expect(attempt.targetContentHashes?.[0]).not.toHaveProperty("beforeHash");

    const finalized = finalizeEditAttemptLifecycle({
      attempt,
      postApplyChecks: [{
        path: "answer.py",
        status: "consistent",
        expectedHash: attempt.outputContentHashes["answer.py"],
        actualHash: attempt.outputContentHashes["answer.py"],
        reason: "client file content matches the written edit hash",
      }],
      commandResults: [],
      rollbackResults: [],
      artifactRefs: ["artifact://command-results.json"],
    });

    expect(finalized).toMatchObject({
      verificationStatus: "skipped",
      selfDetectedRegressionStatus: "none",
      rollbackStatus: "not_needed",
    });
    expect(finalized.phaseResults.find((phase) => phase.phase === "verify")).toMatchObject({
      status: "skipped",
      attributes: { commandCount: 0 },
    });
  });
});
