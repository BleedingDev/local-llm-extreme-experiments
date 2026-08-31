import { describe, expect, test } from "bun:test";
import {
  extractRoutingReplayScenarioSkeletons,
  routingReplayCaptures,
  routingReplayScenarioSkeletons,
  routingReplayScenarios,
  visibleRoutingReplayScenariosForOptimization,
} from "../src/replay";

describe("routing replay scenarios", () => {
  test("defines the routing scenario pack with redaction-safe captures", () => {
    expect(routingReplayScenarios.map((scenario) => scenario.scenarioKind)).toEqual([
      "greeting_no_side_effect",
      "user_correction",
      "read_only_report",
      "mutation_request",
      "auto_temporary_restoration",
      "yolo_safe_behavior",
      "cancellation",
    ]);
    expect(routingReplayCaptures).toHaveLength(7);
    expect(routingReplayScenarios.every((scenario) => scenario.capture.redactionStatus === "redacted")).toBe(true);
    expect(routingReplayScenarios.every((scenario) => scenario.metadata.expectedBehavior.summary.length > 0))
      .toBe(true);
    expect(routingReplayScenarios.every((scenario) => scenario.metadata.sourceRefs.every(
      (sourceRef) => sourceRef.redactionStatus !== "raw_local_only",
    ))).toBe(true);
  });

  test("preserves scenario metadata, source refs, and oracle fields through extraction", () => {
    const autoScenario = routingReplayScenarios.find(
      (scenario) => scenario.scenarioKind === "auto_temporary_restoration",
    );
    expect(autoScenario).toBeDefined();
    if (autoScenario == null) {
      throw new Error("auto restoration scenario is missing");
    }

    const skeleton = extractRoutingReplayScenarioSkeletons([autoScenario])[0];
    expect(skeleton).toBeDefined();
    if (skeleton == null) {
      throw new Error("auto restoration skeleton is missing");
    }

    expect(skeleton).toMatchObject({
      evalCaseId: "replay.eval.routing.auto-restoration",
      split: "dev",
      captureId: "capture.replay.routing.auto-restoration",
      sourceSessionId: "session.replay.routing.auto-restoration",
      routing: {
        requestedMode: "auto",
        selectedMode: "auto",
        restoredMode: "safe",
        sideEffectPolicy: "write_allowed",
      },
      oracle: {
        strength: "strong",
      },
    });
    expect(skeleton.oracle.expectedBehavior.assertions.map((assertion) => assertion.assertionId)).toContain(
      "assert.routing.auto.restored",
    );
    expect(skeleton.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "artifact",
      artifactRef: "artifact:auto-mode-restoration",
      redactionStatus: "hash_only",
    }));
    expect(skeleton.sourceTraceIds).toEqual(["trace.replay.routing.auto-restoration"]);
  });

  test("keeps hidden holdout cancellation out of optimization-visible scenarios", () => {
    const visibleScenarios = visibleRoutingReplayScenariosForOptimization();
    expect(visibleScenarios.map((scenario) => scenario.scenarioKind)).not.toContain("cancellation");
    expect(visibleScenarios.every((scenario) => scenario.split !== "holdout")).toBe(true);
    expect(visibleScenarios.every((scenario) => scenario.optimizationAllowed)).toBe(true);

    const cancellationSkeleton = routingReplayScenarioSkeletons.find(
      (skeleton) => skeleton.evalCaseId === "replay.eval.routing.cancellation",
    );
    expect(cancellationSkeleton).toBeDefined();
    if (cancellationSkeleton == null) {
      throw new Error("cancellation skeleton is missing");
    }

    expect(cancellationSkeleton.split).toBe("holdout");
    expect(cancellationSkeleton.observedFailures).toEqual([
      expect.objectContaining({
        failureKind: "terminal_command",
        recordId: "record.replay.routing.cancellation.command",
        status: "timed_out",
        errorCode: "user_cancelled",
      }),
    ]);
  });

  test("preserves Safe/Yolo permission denial as an observed routing failure", () => {
    const safeScenario = routingReplayScenarios.find(
      (scenario) => scenario.scenarioKind === "yolo_safe_behavior",
    );
    expect(safeScenario).toBeDefined();
    if (safeScenario == null) {
      throw new Error("Safe/Yolo scenario is missing");
    }
    expect(safeScenario.capture.records
      .filter((record) => record.recordKind === "mode_route")
      .map((record) => record.selectedMode)).toEqual(["safe", "yolo"]);

    const safeSkeleton = routingReplayScenarioSkeletons.find(
      (skeleton) => skeleton.evalCaseId === "replay.eval.routing.yolo-safe-behavior",
    );
    expect(safeSkeleton).toBeDefined();
    if (safeSkeleton == null) {
      throw new Error("Safe/Yolo skeleton is missing");
    }

    expect(safeSkeleton.routing).toMatchObject({
      requestedMode: "safe",
      selectedMode: "safe",
      sideEffectPolicy: "no_side_effects",
    });
    expect(safeSkeleton.observedFailures).toEqual([
      expect.objectContaining({
        failureKind: "tool_call",
        recordId: "record.replay.routing.yolo-safe-behavior.tool-denied",
        status: "permission_denied",
        errorCode: "safe_mode_terminal_denied",
      }),
    ]);
  });

  test("extracts accepted user corrections as replay failure evidence", () => {
    const correctionScenario = routingReplayScenarios.find(
      (scenario) => scenario.scenarioKind === "user_correction",
    );
    expect(correctionScenario).toBeDefined();
    if (correctionScenario == null) {
      throw new Error("user correction scenario is missing");
    }

    const prompts = correctionScenario.capture.records.filter((record) => record.recordKind === "prompt");
    expect(prompts.map((record) => record.promptEvent)).toEqual(["message", "user_correction"]);

    const correctionSkeleton = routingReplayScenarioSkeletons.find(
      (skeleton) => skeleton.evalCaseId === "replay.eval.routing.user-correction",
    );
    expect(correctionSkeleton).toBeDefined();
    if (correctionSkeleton == null) {
      throw new Error("user correction skeleton is missing");
    }

    expect(correctionSkeleton.split).toBe("dev");
    expect(correctionSkeleton.observedFailures).toEqual([
      expect.objectContaining({
        failureKind: "user_correction",
        recordId: "record.replay.routing.user-correction.correction",
        status: "accepted",
        errorCode: "user_correction",
      }),
    ]);
    expect(correctionSkeleton.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "record",
      recordId: "record.replay.routing.user-correction.correction",
      redactionStatus: "redacted",
    }));
  });
});
