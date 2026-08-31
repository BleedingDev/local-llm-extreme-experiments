import { describe, expect, test } from "bun:test";
import {
  EditAttemptContractSchema,
  EditPhaseResultSchema,
  EditReadRangeSchema,
  EditTokenUsageSchema,
  editAttemptCaptureIssues,
  missingRequiredEditAttemptPhases,
} from "../src/edit-strategy/types";

const now = "2026-04-30T00:00:00.000Z";

const baseAttempt = {
  editAttemptId: "edit.attempt.1",
  runId: "run.acp.1",
  traceId: "trace.1",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  policyId: "policy.qwen36.bleeding-agent",
  editStrategyId: "edit.whole-file.acp-write.v1",
  editStrategyFamily: "whole_file",
  canonicalEditToolSpecId: "tool.edit.whole-file",
  renderedEditToolContractId: "tool.edit.whole-file.qwen36",
  targetFiles: ["src/example.ts"],
  readSnapshotRefs: [
    {
      snapshotId: "snapshot.src-example",
      path: "src/example.ts",
      contentHash: "sha256:before",
      wholeFileSeen: true,
    },
  ],
  inputContentHashes: {
    "src/example.ts": "sha256:before",
  },
  outputContentHashes: {
    "src/example.ts": "sha256:after",
  },
  phaseResults: [
    {
      phase: "generation",
      status: "passed",
      durationMs: 42,
    },
    {
      phase: "post_apply_consistency",
      status: "failed",
      errorCode: "post_apply_behavior_failure",
      artifactRefs: [".bag/artifacts/typecheck.txt"],
    },
  ],
  staleContextStatus: "fresh",
  permissionStatus: "bypassed_yolo",
  verificationStatus: "failed",
  postApplyConsistencyStatus: "inconsistent",
  selfDetectedRegressionStatus: "confirmed",
  selfDetectedRegressionEvidenceRefs: [".bag/artifacts/typecheck.txt"],
  repairAttemptCount: 1,
  rollbackStatus: "not_attempted",
  tokenUsage: {
    promptTokens: 100,
    completionTokens: 40,
    totalTokens: 140,
  },
  latencyMs: 1200,
  changedFileCount: 1,
  changedLineCount: 4,
  redactionStatus: "raw_local_only",
  artifactRefs: [".bag/artifacts/edit-attempt.json"],
  createdAt: now,
};

describe("edit strategy attempt contract", () => {
  test("parses an applied-but-broken edit attempt with replay metadata", () => {
    const attempt = EditAttemptContractSchema.parse(baseAttempt);

    expect(attempt.schemaVersion).toBe("edit-attempt.v1");
    expect(attempt.editStrategyFamily).toBe("whole_file");
    expect(attempt.postApplyConsistencyStatus).toBe("inconsistent");
    expect(attempt.phaseResults[1]?.errorCode).toBe("post_apply_behavior_failure");
    expect(attempt.readSnapshotRefs[0]?.wholeFileSeen).toBe(true);
  });

  test("captures complete real-attempt evidence for strategy phases hashes fallback repair rollback and self-check", () => {
    const attempt = EditAttemptContractSchema.parse({
      ...baseAttempt,
      editStrategyId: "edit.apply-patch.v1",
      editStrategyFamily: "apply_patch",
      canonicalEditToolSpecId: "edit.apply-patch.v1",
      renderedEditToolContractId: "rendered.edit.apply-patch.v1.model.qwen36.local",
      renderedEditContractVersion: "rendered-edit-contract.v1",
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
        { phase: "generation", status: "passed" },
        { phase: "parse", status: "passed" },
        { phase: "validate", status: "passed" },
        { phase: "apply", status: "passed" },
        { phase: "write", status: "passed" },
        {
          phase: "post_apply_consistency",
          status: "failed",
          errorCode: "post_apply_behavior_failure",
          artifactRefs: ["artifact://verify/post-apply"],
        },
        {
          phase: "verify",
          status: "failed",
          errorCode: "verifier_error",
          artifactRefs: ["artifact://verify/typecheck"],
        },
        {
          phase: "self_check",
          status: "failed",
          errorCode: "self_detected_regression",
          artifactRefs: ["artifact://self-check/report"],
        },
        { phase: "repair", status: "passed", artifactRefs: ["artifact://repair/round-1"] },
        { phase: "rollback", status: "passed", artifactRefs: ["artifact://rollback/result"] },
      ],
      selfDetectedRegressionEvidenceRefs: [],
      selfDetectedRegressionEvidence: [
        {
          evidenceRef: "artifact://self-check/report",
          evidenceKind: "model_self_check",
          phase: "self_check",
          status: "confirmed",
          artifactRefs: ["artifact://self-check/report"],
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
      rollbackStatus: "succeeded",
      fallbackFromStrategyId: "edit.unified-diff.v1",
      fallbackToStrategyId: "edit.apply-patch.v1",
      fallbackPath: [
        {
          fromStrategyId: "edit.unified-diff.v1",
          toStrategyId: "edit.apply-patch.v1",
          trigger: "apply_failed",
          status: "passed",
          artifactRefs: ["artifact://fallback/route"],
        },
      ],
    });

    expect(missingRequiredEditAttemptPhases(attempt.phaseResults)).toEqual([]);
    expect(editAttemptCaptureIssues(attempt)).toEqual([]);
  });

  test("fills conservative defaults for not-yet-checked phases", () => {
    const attempt = EditAttemptContractSchema.parse({
      editAttemptId: "edit.attempt.minimal",
      modelProfileId: "model.qwen36.local",
      codebaseProfileId: "codebase.bleeding-agent",
      policyId: "policy.qwen36.bleeding-agent",
      editStrategyId: "edit.exact-replace.v1",
      editStrategyFamily: "exact_replace",
      createdAt: now,
    });

    expect(attempt.targetFiles).toEqual([]);
    expect(attempt.tokenUsage.totalTokens).toBe(0);
    expect(attempt.staleContextStatus).toBe("not_checked");
    expect(attempt.permissionStatus).toBe("not_required");
    expect(attempt.verificationStatus).toBe("not_run");
    expect(attempt.rollbackStatus).toBe("not_needed");
  });

  test("surfaces partial real-attempt capture gaps without rejecting legacy attempts", () => {
    const attempt = EditAttemptContractSchema.parse({
      ...baseAttempt,
      targetFiles: ["src/example.ts"],
      inputContentHashes: { "src/example.ts": "sha256:before" },
      outputContentHashes: {},
      changedFileCount: 1,
      fallbackFromStrategyId: "edit.unified-diff.v1",
      fallbackToStrategyId: "edit.whole-file.acp-write.v1",
      renderedEditContractVersion: undefined,
    });

    const issues = editAttemptCaptureIssues(attempt);

    expect(issues).toContain("rendered_edit_contract_version");
    expect(issues).toContain("phase.parse");
    expect(issues).toContain("phase.validate");
    expect(issues).toContain("phase.apply");
    expect(issues).toContain("phase.write");
    expect(issues).toContain("phase.verify");
    expect(issues).toContain("target_hash.after.src/example.ts");
    expect(issues).toContain("fallback_path");
    expect(issues).toContain("phase.repair");
    expect(issues).toContain("phase.rollback");
  });

  test("requires evidence for confirmed self-detected regressions", () => {
    const result = EditAttemptContractSchema.safeParse({
      ...baseAttempt,
      selfDetectedRegressionEvidenceRefs: [],
    });

    expect(result.success).toBe(false);
  });

  test("accepts structured self-detected regression evidence without duplicate evidence refs", () => {
    const result = EditAttemptContractSchema.safeParse({
      ...baseAttempt,
      selfDetectedRegressionEvidenceRefs: [],
      selfDetectedRegressionEvidence: [
        {
          evidenceRef: "artifact://self-check/report",
          evidenceKind: "model_self_check",
          status: "confirmed",
        },
      ],
    });

    expect(result.success).toBe(true);
  });

  test("requires fallback source when a fallback target is recorded", () => {
    const result = EditAttemptContractSchema.safeParse({
      ...baseAttempt,
      fallbackToStrategyId: "edit.whole-file.acp-write.v1",
    });

    expect(result.success).toBe(false);
  });

  test("rejects inconsistent target hashes fallback paths and repair references", () => {
    const mismatchedHash = EditAttemptContractSchema.safeParse({
      ...baseAttempt,
      targetContentHashes: [{ path: "src/example.ts", beforeHash: "sha256:different" }],
    });
    const mismatchedFallback = EditAttemptContractSchema.safeParse({
      ...baseAttempt,
      fallbackFromStrategyId: "edit.unified-diff.v1",
      fallbackToStrategyId: "edit.whole-file.acp-write.v1",
      fallbackPath: [
        {
          fromStrategyId: "edit.unified-diff.v1",
          toStrategyId: "edit.exact-replace.v1",
          trigger: "apply_failed",
        },
      ],
    });
    const mismatchedRepairCount = EditAttemptContractSchema.safeParse({
      ...baseAttempt,
      repairAttemptCount: 1,
      repairAttemptRefs: [{ repairRound: 2, status: "failed" }],
    });

    expect(mismatchedHash.success).toBe(false);
    expect(mismatchedFallback.success).toBe(false);
    expect(mismatchedRepairCount.success).toBe(false);
  });

  test("requires failed phase results to carry stable error codes", () => {
    const result = EditPhaseResultSchema.safeParse({
      phase: "apply",
      status: "failed",
    });

    expect(result.success).toBe(false);
  });

  test("rejects error codes on non-failed phase results", () => {
    const result = EditPhaseResultSchema.safeParse({
      phase: "apply",
      status: "passed",
      errorCode: "partial_apply",
    });

    expect(result.success).toBe(false);
  });

  test("rejects invalid read ranges and inconsistent token totals", () => {
    expect(EditReadRangeSchema.safeParse({ startLine: 10, endLine: 9 }).success).toBe(false);
    expect(
      EditTokenUsageSchema.safeParse({
        promptTokens: 10,
        completionTokens: 5,
        totalTokens: 99,
      }).success,
    ).toBe(false);
  });
});
