import { describe, expect, test } from "bun:test";
import { runEditStrategyAblation } from "../src/eval-harness/edit-strategy-ablation";
import {
  EditStrategyRouterDecisionSchema,
  routeEditStrategy,
  taskShapeBucketFor,
  type EditStrategyHistoricalMetric,
} from "../src/optimizer/edit-policy-router";

const policy = {
  modelProfileId: "model.synthetic.edit-strategy",
  codebaseProfileId: "codebase.synthetic.edit-strategy",
  policyId: "policy.synthetic.edit-strategy",
  editStrategyVersion: "edit-strategy.v1",
  renderedEditContractVersion: "rendered-edit-contract.v1",
  editFallbackPolicyVersion: "edit-fallback.v1",
  editRepairPolicyVersion: "edit-repair.v1",
  editVerifierPolicyVersion: "edit-verifier.v1",
  editObjectiveSetId: "edit-objectives.default.v1",
};

const metric = (
  strategyFamily: EditStrategyHistoricalMetric["strategyFamily"],
  overrides: Partial<EditStrategyHistoricalMetric> = {},
): EditStrategyHistoricalMetric => ({
  metricId: `metric.test.${strategyFamily.replaceAll("_", "-")}`,
  source: "trace",
  trainingAllowed: true,
  modelProfileId: policy.modelProfileId,
  codebaseProfileId: policy.codebaseProfileId,
  strategyFamily,
  sampleCount: 20,
  taskPassRate: 0.8,
  expectedOutcomeMatchRate: 0.8,
  parsePassRate: 0.9,
  applyAcceptedRate: 0.9,
  coverageRate: 1,
  averageScore: 0.8,
  protectedPathTouchRate: 0,
  staleRejectionRate: 0,
  appliedButBrokenRate: 0,
  ...overrides,
});

describe("edit policy router", () => {
  test("routes from visible ablation evidence and emits explicit fallback rules", () => {
    const report = runEditStrategyAblation({ createdAt: "2026-04-30T00:00:00.000Z" });
    const decision = routeEditStrategy({
      resolvedPolicy: policy,
      ablationReports: [report],
      taskShape: {
        targetFileCount: 1,
        largestTargetFileBytes: 1200,
        totalTargetFileBytes: 1200,
        verifierStrength: "strong",
        protectedPathRisk: "medium",
      },
    });

    expect(EditStrategyRouterDecisionSchema.parse(decision).schemaVersion).toBe("edit-policy-router.v1");
    expect(decision.evidenceUsed.some((evidence) => evidence.source === "ablation")).toBe(true);
    expect(decision.degraded).toBe(false);
    expect(decision.candidates.find((candidate) => candidate.strategyFamily === "ast_structured")?.eligible)
      .toBe(false);
    expect(decision.fallbackRules.map((rule) => rule.trigger)).toEqual([
      "parse_failed",
      "apply_failed",
      "stale_context",
      "protected_path_violation",
      "post_apply_inconsistent",
      "verification_failed",
      "self_detected_regression",
      "context_budget_exceeded",
    ]);
    expect(decision.fallbackRules.find((rule) => rule.trigger === "protected_path_violation")).toMatchObject({
      action: "abort",
    });
  });

  test("ignores holdout ablation reports as optimization evidence", () => {
    const holdoutReport = runEditStrategyAblation({
      splits: ["holdout"],
      includeHoldout: true,
      createdAt: "2026-04-30T00:00:00.000Z",
    });
    const decision = routeEditStrategy({
      resolvedPolicy: policy,
      ablationReports: [holdoutReport],
      taskShape: { targetFileCount: 1 },
    });

    expect(decision.evidenceUsed).toEqual([]);
    expect(decision.warnings).toEqual([
      "ignored non-optimization ablation report ablation.edit-strategy.visible because it used hidden holdout cases",
    ]);
  });

  test("uses task shape constraints instead of textual or language keyword routing", () => {
    const decision = routeEditStrategy({
      resolvedPolicy: policy,
      historicalMetrics: [
        metric("exact_replace", { taskPassRate: 0.99, expectedOutcomeMatchRate: 0.99, averageScore: 0.99 }),
        metric("apply_patch", { taskPassRate: 0.72, expectedOutcomeMatchRate: 0.8, averageScore: 0.72 }),
      ],
      taskShape: {
        targetFileCount: 2,
        estimatedChangedFileCount: 2,
        requiresMultiFileConsistency: true,
        largestTargetFileBytes: 2000,
        totalTargetFileBytes: 4000,
      },
    });

    expect(decision.selectedStrategyFamily).toBe("apply_patch");
    expect(decision.candidates.find((candidate) => candidate.strategyFamily === "exact_replace")).toMatchObject({
      eligible: false,
      blockers: ["task requires multi-file consistency but strategy is single-file"],
    });
  });

  test("blocks whole-file output when budget evidence says a localized strategy is needed", () => {
    const decision = routeEditStrategy({
      resolvedPolicy: policy,
      historicalMetrics: [
        metric("whole_file", { taskPassRate: 0.98, expectedOutcomeMatchRate: 0.98, averageScore: 0.98 }),
        metric("apply_patch", { taskPassRate: 0.78, expectedOutcomeMatchRate: 0.82, averageScore: 0.78 }),
      ],
      taskShape: {
        targetFileCount: 1,
        largestTargetFileBytes: 80_000,
        totalTargetFileBytes: 80_000,
        outputBudgetTokens: 2_000,
      },
    });

    expect(decision.selectedStrategyFamily).toBe("apply_patch");
    expect(decision.candidates.find((candidate) => candidate.strategyFamily === "whole_file")?.blockers)
      .toContain("whole-file output exceeds configured output budget");
  });

  test("treats protected path and applied-but-broken metrics as selection risks", () => {
    const protectedDecision = routeEditStrategy({
      resolvedPolicy: policy,
      historicalMetrics: [
        metric("whole_file", {
          taskPassRate: 0.95,
          expectedOutcomeMatchRate: 0.95,
          averageScore: 0.95,
          protectedPathTouchRate: 0.2,
        }),
        metric("exact_replace", { taskPassRate: 0.8, expectedOutcomeMatchRate: 0.85, averageScore: 0.8 }),
      ],
      taskShape: {
        targetFileCount: 1,
        protectedPathRisk: "high",
      },
    });

    expect(protectedDecision.selectedStrategyFamily).toBe("exact_replace");
    expect(protectedDecision.candidates.find((candidate) => candidate.strategyFamily === "whole_file")?.blockers)
      .toContain("measured protected-path touch rate is incompatible with high protected-path risk");

    const noVerifierDecision = routeEditStrategy({
      resolvedPolicy: policy,
      historicalMetrics: [
        metric("apply_patch", {
          taskPassRate: 0.95,
          expectedOutcomeMatchRate: 0.95,
          averageScore: 0.95,
          appliedButBrokenRate: 0.25,
        }),
        metric("exact_replace", { taskPassRate: 0.78, expectedOutcomeMatchRate: 0.82, averageScore: 0.78 }),
      ],
      taskShape: {
        targetFileCount: 1,
        verifierStrength: "none",
      },
    });

    expect(noVerifierDecision.selectedStrategyFamily).toBe("exact_replace");
    expect(noVerifierDecision.candidates.find((candidate) => candidate.strategyFamily === "apply_patch")?.blockers)
      .toContain("measured applied-but-broken risk requires a verifier");
  });

  test("uses task-shape-specific metrics before generic or mismatched winners", () => {
    const singleSmallShape = {
      targetFileCount: 1,
      estimatedChangedFileCount: 1,
      largestTargetFileBytes: 800,
      totalTargetFileBytes: 800,
      verifierStrength: "basic" as const,
      protectedPathRisk: "medium" as const,
      staleContextRisk: "medium" as const,
      requiresMultiFileConsistency: false,
    };
    const multiLargeShape = {
      targetFileCount: 3,
      estimatedChangedFileCount: 3,
      largestTargetFileBytes: 60_000,
      totalTargetFileBytes: 120_000,
      outputBudgetTokens: 2_000,
      verifierStrength: "basic" as const,
      protectedPathRisk: "medium" as const,
      staleContextRisk: "medium" as const,
      requiresMultiFileConsistency: true,
    };

    const metrics = [
      metric("exact_replace", {
        metricId: "metric.single-small.exact-replace",
        taskShapeBucket: taskShapeBucketFor(singleSmallShape),
        taskPassRate: 0.99,
        expectedOutcomeMatchRate: 0.99,
        averageScore: 0.99,
      }),
      metric("apply_patch", {
        metricId: "metric.multi-large.apply-patch",
        taskShapeBucket: taskShapeBucketFor(multiLargeShape),
        taskPassRate: 0.9,
        expectedOutcomeMatchRate: 0.92,
        averageScore: 0.9,
      }),
      metric("whole_file", {
        metricId: "metric.generic.whole-file",
        taskPassRate: 0.7,
        expectedOutcomeMatchRate: 0.72,
        averageScore: 0.7,
      }),
    ];

    const singleDecision = routeEditStrategy({
      resolvedPolicy: policy,
      historicalMetrics: metrics,
      taskShape: singleSmallShape,
    });
    const multiDecision = routeEditStrategy({
      resolvedPolicy: policy,
      historicalMetrics: metrics,
      taskShape: multiLargeShape,
    });

    expect(singleDecision.selectedStrategyFamily).toBe("exact_replace");
    expect(singleDecision.evidenceUsed).toContainEqual(expect.objectContaining({
      metricId: "metric.single-small.exact-replace",
      taskShapeBucketId: expect.any(String),
    }));
    expect(multiDecision.selectedStrategyFamily).toBe("apply_patch");
    expect(multiDecision.evidenceUsed).toContainEqual(expect.objectContaining({
      metricId: "metric.multi-large.apply-patch",
      taskShapeBucketId: expect.any(String),
    }));
    expect(multiDecision.evidenceUsed.map((evidence) => evidence.metricId)).not.toContain("metric.single-small.exact-replace");
  });
});
