import { describe, expect, test } from "bun:test";
import {
  assessGepaEvidenceReadiness,
  planGepaDryRunScheduler,
} from "../src/optimizer/gepa-operations";
import { editFailureReplayScenarioSkeletons } from "../src/replay/edit-failure-scenarios";
import { toolCallReplayScenarioSkeletons } from "../src/replay/tool-call-scenarios";
import type { TraceAnalysisReport, TraceOptimizerDimensions } from "../src/trace-analysis";

const now = "2026-05-01T00:00:00.000Z";

const emptyDimensions = (): TraceOptimizerDimensions => ({
  modelProfileIds: ["model.qwen36.local"],
  codebaseProfileIds: ["codebase.bleeding-agent"],
  policyIds: ["policy.qwen36.bleeding-agent"],
  canonicalToolVersions: ["canonical-tools.v1"],
  renderedToolVersions: ["rendered-tools.v1"],
  resultStyleVersions: ["result-style.v1"],
  verificationPolicyVersions: ["verification.v1"],
  editStrategyVersions: [],
  renderedEditContractVersions: [],
  editFallbackPolicyVersions: [],
  editRepairPolicyVersions: [],
  editVerifierPolicyVersions: [],
  editObjectiveSetIds: [],
  editStrategyIds: [],
  editStrategyFamilies: [],
  canonicalEditToolSpecIds: [],
  renderedEditToolContractIds: [],
});

const traceReport = (): TraceAnalysisReport => ({
  spanCount: 12,
  traceCount: 3,
  errorSpanCount: 2,
  observationKinds: { TOOL: 2 },
  optimizerDimensions: emptyDimensions(),
  failureClusters: [
    {
      name: "mcp.workspace.write_file",
      observationKind: "TOOL",
      count: 2,
      traces: ["trace-a", "trace-b"],
      messages: ["permission denied"],
      inputHashes: ["hash-a"],
      optimizerDimensions: emptyDimensions(),
    },
  ],
  latencyClusters: [],
});

const liveReplayCases = () => [
  {
    ...editFailureReplayScenarioSkeletons[0]!,
    evalCaseId: "replay.eval.live.edit.failure",
    tags: [...editFailureReplayScenarioSkeletons[0]!.tags, "live"],
  },
  {
    ...toolCallReplayScenarioSkeletons[0]!,
    evalCaseId: "replay.eval.live.tool.failure",
    tags: [...toolCallReplayScenarioSkeletons[0]!.tags, "live"],
  },
];

describe("GEPA operation readiness", () => {
  test("keeps candidate generation blocked until real replay and failure thresholds are met", () => {
    const readiness = assessGepaEvidenceReadiness({});

    expect(readiness.candidateGenerationReady).toBe(false);
    expect(readiness.autoPromotionReady).toBe(false);
    expect(readiness.blockedGateIds).toEqual(expect.arrayContaining([
      "real-replay-cases",
      "visible-replay-cases",
      "repeated-failure-clusters",
      "edit-failure-volume",
      "tool-failure-volume",
      "metric-observation-volume",
      "post-promotion-monitor-window",
    ]));
  });

  test("does not treat synthetic replay fixtures as real replay evidence", () => {
    const readiness = assessGepaEvidenceReadiness({
      replayCases: [
        editFailureReplayScenarioSkeletons[0]!,
        toolCallReplayScenarioSkeletons[0]!,
      ],
      traceReports: [traceReport()],
      metricObservationCount: 4,
    });

    expect(readiness.counts.visibleReplayCaseCount).toBe(2);
    expect(readiness.counts.realReplayCaseCount).toBe(0);
    expect(readiness.candidateGenerationReady).toBe(false);
    expect(readiness.blockedGateIds).toContain("real-replay-cases");
  });

  test("separates candidate generation readiness from auto-promotion monitoring readiness", () => {
    const candidateReady = assessGepaEvidenceReadiness({
      replayCases: liveReplayCases(),
      traceReports: [traceReport()],
      metricObservationCount: 4,
      userCorrectionSignalCount: 1,
    });

    expect(candidateReady.candidateGenerationReady).toBe(true);
    expect(candidateReady.autoPromotionReady).toBe(false);
    expect(candidateReady.blockedGateIds).toEqual(["post-promotion-monitor-window"]);

    const autoReady = assessGepaEvidenceReadiness({
      replayCases: liveReplayCases(),
      traceReports: [traceReport()],
      metricObservationCount: 4,
      userCorrectionSignalCount: 1,
      postPromotionMonitoringWindow: {
        startedAt: now,
        endedAt: "2026-05-01T05:00:00.000Z",
      },
    });

    expect(autoReady.candidateGenerationReady).toBe(true);
    expect(autoReady.autoPromotionReady).toBe(true);
    expect(autoReady.blockedGateIds).toEqual([]);
  });

  test("blocks auto-promotion when post-promotion regressions exceed the budget", () => {
    const readiness = assessGepaEvidenceReadiness({
      replayCases: liveReplayCases(),
      traceReports: [traceReport()],
      metricObservationCount: 4,
      postPromotionMonitoringWindow: {
        startedAt: now,
        endedAt: "2026-05-01T05:00:00.000Z",
      },
      postPromotionRegressionSignalCount: 1,
    });

    expect(readiness.candidateGenerationReady).toBe(true);
    expect(readiness.autoPromotionReady).toBe(false);
    expect(readiness.blockedGateIds).toEqual(["post-promotion-regression-budget"]);
  });

  test("can require scorecard, hidden holdout, and profile stability coverage", () => {
    const readiness = assessGepaEvidenceReadiness({
      replayCases: liveReplayCases(),
      traceReports: [traceReport()],
      metricObservationCount: 4,
      profileMismatchCount: 1,
      thresholds: {
        minScorecardCount: 1,
        minHiddenHoldoutScorecardCount: 1,
        maxProfileMismatchCount: 0,
      },
    });

    expect(readiness.blockedGateIds).toEqual(expect.arrayContaining([
      "scorecard-coverage",
      "hidden-holdout-coverage",
      "profile-stability",
    ]));
    expect(readiness.candidateGenerationReady).toBe(false);
  });

  test("plans dry-run scheduler outcomes without allowing actual promotion", () => {
    const blocked = planGepaDryRunScheduler({
      generatedAt: now,
      readiness: assessGepaEvidenceReadiness({}),
    });

    expect(blocked.decision).toBe("needs_more_evidence");
    expect(blocked.actualPromotionAllowed).toBe(false);
    expect(blocked.rolloutPlan.find((stage) => stage.stage === "operator_approval")).toMatchObject({
      status: "future",
    });
  });
});
