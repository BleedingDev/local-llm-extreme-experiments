import { describe, expect, test } from "bun:test";
import {
  editFailureReplayCaptures,
  editFailureReplayScenarioSkeletons,
  editFailureReplayScenarios,
  extractEditFailureReplayScenarioSkeletons,
  visibleEditFailureReplayScenariosForOptimization,
} from "../src/replay";

describe("edit failure replay scenarios", () => {
  test("defines the edit failure scenario pack with redaction-safe captures", () => {
    expect(editFailureReplayScenarios.map((scenario) => scenario.scenarioKind)).toEqual([
      "parse_failure",
      "apply_failure",
      "stale_context",
      "protected_path",
      "fallback_success_after_primary_failure",
      "applied_but_broken_file",
      "applied_but_broken_verification_failure",
      "promotion_veto",
      "self_detected_regression",
    ]);
    expect(editFailureReplayCaptures).toHaveLength(9);
    expect(editFailureReplayScenarios.every((scenario) => scenario.capture.redactionStatus === "redacted")).toBe(true);
    expect(editFailureReplayScenarios.every((scenario) => scenario.capture.defaultSplitHint === scenario.split))
      .toBe(true);
    expect(editFailureReplayScenarios.every((scenario) => scenario.metadata.expectedBehavior.assertions.length > 0))
      .toBe(true);
    expect(editFailureReplayScenarios.every((scenario) => scenario.metadata.sourceRefs.every(
      (sourceRef) => sourceRef.redactionStatus !== "raw_local_only",
    ))).toBe(true);
  });

  test("extracts observed failures for every edit failure kind", () => {
    const failuresByKind = new Map(editFailureReplayScenarioSkeletons.map((skeleton) => {
      const scenario = editFailureReplayScenarios.find((candidate) => candidate.metadata.evalCaseId === skeleton.evalCaseId);
      return [scenario?.scenarioKind, skeleton.observedFailures];
    }));

    expect(failuresByKind.get("parse_failure")?.[0]).toMatchObject({
      failureKind: "edit_attempt",
      phase: "parse",
      errorCode: "parse_error",
    });
    expect(failuresByKind.get("apply_failure")?.[0]).toMatchObject({
      failureKind: "edit_attempt",
      phase: "apply",
      errorCode: "exact_match_not_found",
    });
    expect(failuresByKind.get("stale_context")?.[0]).toMatchObject({
      failureKind: "edit_attempt",
      phase: "stale_context_check",
      errorCode: "anchor_stale",
    });
    expect(failuresByKind.get("protected_path")?.[0]).toMatchObject({
      failureKind: "edit_attempt",
      phase: "permission",
      errorCode: "protected_path_violation",
    });
    expect(failuresByKind.get("applied_but_broken_verification_failure")?.map((failure) => failure.failureKind))
      .toEqual(["edit_attempt", "terminal_command"]);
    expect(failuresByKind.get("applied_but_broken_file")?.[0]).toMatchObject({
      failureKind: "edit_attempt",
      phase: "post_apply_consistency",
      errorCode: "post_apply_syntax_failure",
    });
    expect(failuresByKind.get("self_detected_regression")?.[0]).toMatchObject({
      failureKind: "edit_attempt",
      phase: "self_check",
      errorCode: "self_detected_regression",
    });
    expect(failuresByKind.get("promotion_veto")?.map((failure) => failure.errorCode))
      .toEqual(["post_apply_behavior_failure", "promotion_veto"]);
  });

  test("preserves primary failure evidence when fallback succeeds", () => {
    const scenario = editFailureReplayScenarios.find(
      (candidate) => candidate.scenarioKind === "fallback_success_after_primary_failure",
    );
    expect(scenario).toBeDefined();
    if (scenario == null) {
      throw new Error("fallback success scenario is missing");
    }

    const editAttempts = scenario.capture.records.filter((record) => record.recordKind === "edit_attempt");
    expect(editAttempts).toHaveLength(2);
    expect(editAttempts[0]?.attempt.phaseResults.map((phase) => [phase.phase, phase.status, phase.errorCode]))
      .toContainEqual(["apply", "failed", "hunk_context_mismatch"]);
    expect(editAttempts[1]?.attempt.verificationStatus).toBe("passed");

    const skeleton = extractEditFailureReplayScenarioSkeletons([scenario])[0];
    expect(skeleton).toBeDefined();
    if (skeleton == null) {
      throw new Error("fallback success skeleton is missing");
    }

    expect(skeleton.observedFailures).toEqual([
      expect.objectContaining({
        recordId: "record.replay.edit-failure.fallback-success.primary",
        phase: "apply",
        errorCode: "hunk_context_mismatch",
      }),
    ]);
    expect(skeleton.oracle.expectedBehavior.notes).toContain(
      "Preserve primary failure evidence even when the final edit result succeeds via fallback.",
    );
  });

  test("keeps hidden holdout scenarios out of optimization-visible edit failures", () => {
    const visibleScenarios = visibleEditFailureReplayScenariosForOptimization();
    expect(visibleScenarios.map((scenario) => scenario.scenarioKind)).not.toContain("self_detected_regression");
    expect(visibleScenarios.map((scenario) => scenario.scenarioKind)).not.toContain("promotion_veto");
    expect(visibleScenarios.every((scenario) => scenario.split !== "holdout")).toBe(true);
    expect(visibleScenarios.every((scenario) => scenario.optimizationAllowed)).toBe(true);

    const holdoutSkeleton = editFailureReplayScenarioSkeletons.find(
      (skeleton) => skeleton.evalCaseId === "replay.eval.edit-failure.self-detected-regression",
    );
    expect(holdoutSkeleton).toBeDefined();
    if (holdoutSkeleton == null) {
      throw new Error("self-detected regression skeleton is missing");
    }

    expect(holdoutSkeleton.split).toBe("holdout");
    expect(holdoutSkeleton.splitAssignment).toMatchObject({
      split: "holdout",
      assignedBy: "manual",
    });
    expect(holdoutSkeleton.observedFailures[0]).toMatchObject({
      phase: "self_check",
      errorCode: "self_detected_regression",
    });

    const promotionVetoSkeleton = editFailureReplayScenarioSkeletons.find(
      (skeleton) => skeleton.evalCaseId === "replay.eval.edit-failure.promotion-veto",
    );
    expect(promotionVetoSkeleton).toBeDefined();
    if (promotionVetoSkeleton == null) {
      throw new Error("promotion veto skeleton is missing");
    }

    expect(promotionVetoSkeleton.split).toBe("holdout");
    expect(promotionVetoSkeleton.observedFailures.map((failure) => failure.errorCode))
      .toEqual(["post_apply_behavior_failure", "promotion_veto"]);
    expect(promotionVetoSkeleton.oracle.expectedBehavior.notes).toContain(
      "This is a fixture-only replay case, not evidence of live ACP extraction.",
    );
  });

  test("preserves source refs, fixture workspace, oracle, and routing fields through extraction", () => {
    const verificationScenario = editFailureReplayScenarios.find(
      (scenario) => scenario.scenarioKind === "applied_but_broken_verification_failure",
    );
    expect(verificationScenario).toBeDefined();
    if (verificationScenario == null) {
      throw new Error("verification failure scenario is missing");
    }

    const skeleton = extractEditFailureReplayScenarioSkeletons([verificationScenario])[0];
    expect(skeleton).toBeDefined();
    if (skeleton == null) {
      throw new Error("verification failure skeleton is missing");
    }

    expect(skeleton).toMatchObject({
      evalCaseId: "replay.eval.edit-failure.verification-failure",
      split: "dev",
      captureId: "capture.replay.edit-failure.verification-failure",
      sourceSessionId: "session.replay.edit-failure.verification-failure",
      routing: {
        requestedMode: "auto",
        selectedMode: "mutating",
        sideEffectPolicy: "write_allowed",
      },
      oracle: {
        strength: "strong",
      },
    });
    expect(skeleton.fixtureWorkspace).toMatchObject({
      fixtureWorkspaceId: "fixture.replay.edit-failure.verification-failure",
      protectedPaths: [],
      verificationCommands: [["npm", "run", "typecheck"]],
    });
    expect(skeleton.oracle.expectedBehavior.assertions.map((assertion) => assertion.assertionId)).toContain(
      "assert.edit.verify.command-exit",
    );
    expect(skeleton.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "fixture",
      path: "synthetic://replay/edit-failure/replay.edit-failure.verification-failure/stderr",
      redactionStatus: "hash_only",
    }));
    expect(skeleton.sourceTraceIds).toEqual(["trace.replay.edit-failure.verification-failure"]);
  });
});
