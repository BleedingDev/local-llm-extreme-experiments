import { z } from "zod";
import type { EditStrategyAblationReport } from "../eval-harness/edit-strategy-ablation";
import type { EvalRunResult, EvalScorecard } from "../eval-harness/types";
import type { ReplayEvalCaseSkeleton } from "../replay/extraction";
import type { TraceAnalysisReport } from "../trace-analysis";
import type { CandidateEvidenceBundle } from "./evidence";
import {
  OptimizerArtifactLineageDecisionSchema,
  type OptimizerArtifactLineageDecision,
} from "./artifact-lineage";

export const GepaOperationThresholdsSchema = z.object({
  minRealReplayCases: z.number().int().nonnegative().default(2),
  minVisibleReplayCases: z.number().int().nonnegative().default(2),
  repeatedFailureMinCount: z.number().int().positive().default(2),
  minRepeatedFailureClusters: z.number().int().nonnegative().default(1),
  minEditFailureSignals: z.number().int().nonnegative().default(1),
  minToolFailureSignals: z.number().int().nonnegative().default(1),
  minUserCorrectionSignals: z.number().int().nonnegative().default(0),
  minMetricObservationCount: z.number().int().nonnegative().default(4),
  minScorecardCount: z.number().int().nonnegative().default(0),
  minHiddenHoldoutScorecardCount: z.number().int().nonnegative().default(0),
  maxProfileMismatchCount: z.number().int().nonnegative().default(0),
  minPostPromotionMonitoringWindowMs: z.number().int().nonnegative().default(4 * 60 * 60 * 1000),
  maxPostPromotionRegressionSignals: z.number().int().nonnegative().default(0),
}).strict();
export type GepaOperationThresholds = z.infer<typeof GepaOperationThresholdsSchema>;

export const GepaOperationEvidenceCountsSchema = z.object({
  realReplayCaseCount: z.number().int().nonnegative(),
  visibleReplayCaseCount: z.number().int().nonnegative(),
  repeatedFailureClusterCount: z.number().int().nonnegative(),
  editFailureSignalCount: z.number().int().nonnegative(),
  toolFailureSignalCount: z.number().int().nonnegative(),
  userCorrectionSignalCount: z.number().int().nonnegative(),
  metricObservationCount: z.number().int().nonnegative(),
  scorecardCount: z.number().int().nonnegative(),
  hiddenHoldoutScorecardCount: z.number().int().nonnegative(),
  profileMismatchCount: z.number().int().nonnegative(),
  postPromotionMonitoringWindowMs: z.number().int().nonnegative(),
  postPromotionRegressionSignalCount: z.number().int().nonnegative(),
}).strict();
export type GepaOperationEvidenceCounts = z.infer<typeof GepaOperationEvidenceCountsSchema>;

export const GepaOperationReadinessGateSchema = z.object({
  gateId: z.enum([
    "real-replay-cases",
    "visible-replay-cases",
    "repeated-failure-clusters",
    "edit-failure-volume",
    "tool-failure-volume",
    "user-correction-volume",
    "metric-observation-volume",
    "scorecard-coverage",
    "hidden-holdout-coverage",
    "profile-stability",
    "post-promotion-monitor-window",
    "post-promotion-regression-budget",
  ]),
  phase: z.enum(["candidate_generation", "auto_promotion"]),
  passed: z.boolean(),
  actual: z.number().nonnegative(),
  required: z.number().nonnegative(),
  message: z.string().min(1),
}).strict();
export type GepaOperationReadinessGate = z.infer<typeof GepaOperationReadinessGateSchema>;

export const GepaOperationReadinessSchema = z.object({
  thresholds: GepaOperationThresholdsSchema,
  counts: GepaOperationEvidenceCountsSchema,
  candidateGenerationReady: z.boolean(),
  autoPromotionReady: z.boolean(),
  gates: z.array(GepaOperationReadinessGateSchema),
  blockedGateIds: z.array(z.string()),
  notes: z.array(z.string()).default([]),
}).strict();
export type GepaOperationReadiness = z.infer<typeof GepaOperationReadinessSchema>;

export const GepaDryRunSchedulerDecisionSchema = z.object({
  schemaVersion: z.literal("gepa-dry-run-scheduler.v1"),
  generatedAt: z.string().datetime({ offset: true }),
  dryRunOnly: z.literal(true),
  decision: z.enum(["would_promote", "would_reject", "would_quarantine", "needs_more_evidence"]),
  candidateGenerationReady: z.boolean(),
  autoPromotionReady: z.boolean(),
  actualPromotionAllowed: z.literal(false),
  reasons: z.array(z.string().min(1)).default([]),
  blockedGateIds: z.array(z.string()).default([]),
  lineageDecisionIds: z.array(z.string()).default([]),
  rolloutPlan: z.array(z.object({
    stage: z.enum([
      "observe",
      "generate_candidate",
      "evaluate_visible",
      "evaluate_holdout",
      "operator_approval",
      "promote_new_sessions",
      "monitor",
      "rollback",
    ]),
    status: z.enum(["ready", "blocked", "future"]),
    notes: z.string().min(1),
  }).strict()).default([]),
}).strict();
export type GepaDryRunSchedulerDecision = z.infer<typeof GepaDryRunSchedulerDecisionSchema>;

export type AssessGepaEvidenceReadinessInput = {
  replayCases?: readonly ReplayEvalCaseSkeleton[];
  traceReports?: readonly TraceAnalysisReport[];
  evalRunResults?: readonly EvalRunResult[];
  evalScorecards?: readonly EvalScorecard[];
  evidenceBundles?: readonly CandidateEvidenceBundle[];
  editAblationReports?: readonly EditStrategyAblationReport[];
  userCorrectionSignalCount?: number;
  metricObservationCount?: number;
  postPromotionMonitoringWindow?: {
    startedAt: string;
    endedAt: string;
  };
  postPromotionRegressionSignalCount?: number;
  profileMismatchCount?: number;
  thresholds?: Partial<GepaOperationThresholds>;
};

export type PlanGepaDryRunSchedulerInput = {
  readiness: GepaOperationReadiness;
  lineageDecisions?: readonly OptimizerArtifactLineageDecision[];
  generatedAt?: string;
};

export const assessGepaEvidenceReadiness = (
  input: AssessGepaEvidenceReadinessInput,
): GepaOperationReadiness => {
  const thresholds = GepaOperationThresholdsSchema.parse(input.thresholds ?? {});
  const counts = GepaOperationEvidenceCountsSchema.parse({
    realReplayCaseCount: countRealReplayCases(input.replayCases ?? []),
    visibleReplayCaseCount: countVisibleReplayCases(input.replayCases ?? []),
    repeatedFailureClusterCount: countRepeatedFailureClusters(input.traceReports ?? [], thresholds.repeatedFailureMinCount),
    editFailureSignalCount: countEditFailureSignals(input),
    toolFailureSignalCount: countToolFailureSignals(input),
    userCorrectionSignalCount: Math.max(0, Math.floor(input.userCorrectionSignalCount ?? 0)),
    metricObservationCount: Math.max(0, Math.floor(input.metricObservationCount ?? inferMetricObservationCount(input))),
    scorecardCount: (input.evalScorecards ?? []).length,
    hiddenHoldoutScorecardCount: (input.evalScorecards ?? []).filter((scorecard) => scorecard.split === "holdout").length,
    profileMismatchCount: Math.max(0, Math.floor(input.profileMismatchCount ?? 0)),
    postPromotionMonitoringWindowMs: postPromotionWindowMs(input.postPromotionMonitoringWindow),
    postPromotionRegressionSignalCount: Math.max(0, Math.floor(input.postPromotionRegressionSignalCount ?? 0)),
  });

  const gates = [
    minGate("real-replay-cases", "candidate_generation", counts.realReplayCaseCount, thresholds.minRealReplayCases),
    minGate("visible-replay-cases", "candidate_generation", counts.visibleReplayCaseCount, thresholds.minVisibleReplayCases),
    minGate(
      "repeated-failure-clusters",
      "candidate_generation",
      counts.repeatedFailureClusterCount,
      thresholds.minRepeatedFailureClusters,
    ),
    minGate("edit-failure-volume", "candidate_generation", counts.editFailureSignalCount, thresholds.minEditFailureSignals),
    minGate("tool-failure-volume", "candidate_generation", counts.toolFailureSignalCount, thresholds.minToolFailureSignals),
    minGate(
      "user-correction-volume",
      "candidate_generation",
      counts.userCorrectionSignalCount,
      thresholds.minUserCorrectionSignals,
    ),
    minGate(
      "metric-observation-volume",
      "candidate_generation",
      counts.metricObservationCount,
      thresholds.minMetricObservationCount,
    ),
    minGate("scorecard-coverage", "candidate_generation", counts.scorecardCount, thresholds.minScorecardCount),
    minGate(
      "hidden-holdout-coverage",
      "candidate_generation",
      counts.hiddenHoldoutScorecardCount,
      thresholds.minHiddenHoldoutScorecardCount,
    ),
    maxGate("profile-stability", "candidate_generation", counts.profileMismatchCount, thresholds.maxProfileMismatchCount),
    minGate(
      "post-promotion-monitor-window",
      "auto_promotion",
      counts.postPromotionMonitoringWindowMs,
      thresholds.minPostPromotionMonitoringWindowMs,
    ),
    maxGate(
      "post-promotion-regression-budget",
      "auto_promotion",
      counts.postPromotionRegressionSignalCount,
      thresholds.maxPostPromotionRegressionSignals,
    ),
  ];
  const candidateGenerationReady = gates
    .filter((gate) => gate.phase === "candidate_generation")
    .every((gate) => gate.passed);
  const autoPromotionReady = candidateGenerationReady &&
    gates.filter((gate) => gate.phase === "auto_promotion").every((gate) => gate.passed);
  const blockedGateIds = gates
    .filter((gate) => !gate.passed)
    .map((gate) => gate.gateId);

  return GepaOperationReadinessSchema.parse({
    thresholds,
    counts,
    candidateGenerationReady,
    autoPromotionReady,
    gates,
    blockedGateIds,
    notes: [
      "Candidate generation readiness is separate from auto-promotion readiness.",
      "Hidden holdout replay cases are counted by replay enforcement, not as optimizer-visible evidence.",
      "GEPA candidates remain scoped optimizer artifacts; runtime source rewriting is out of scope.",
    ],
  });
};

export const planGepaDryRunScheduler = (
  input: PlanGepaDryRunSchedulerInput,
): GepaDryRunSchedulerDecision => {
  const readiness = GepaOperationReadinessSchema.parse(input.readiness);
  const lineageDecisions = (input.lineageDecisions ?? []).map((decision) =>
    OptimizerArtifactLineageDecisionSchema.parse(decision));
  const lineageRejects = lineageDecisions.filter((decision) => decision.decision === "reject");
  const lineageQuarantines = lineageDecisions.filter((decision) => decision.decision === "quarantine");
  const lineageNeedsEvidence = lineageDecisions.filter((decision) => decision.decision === "needs_more_evidence");
  const decision = !readiness.candidateGenerationReady || lineageNeedsEvidence.length > 0
    ? "needs_more_evidence"
    : lineageQuarantines.length > 0
      ? "would_quarantine"
      : lineageRejects.length > 0
        ? "would_reject"
        : lineageDecisions.length > 0 && lineageDecisions.every((candidateDecision) => candidateDecision.promotionAllowed)
          ? "would_promote"
          : "needs_more_evidence";
  return GepaDryRunSchedulerDecisionSchema.parse({
    schemaVersion: "gepa-dry-run-scheduler.v1",
    generatedAt: input.generatedAt ?? new Date().toISOString(),
    dryRunOnly: true,
    decision,
    candidateGenerationReady: readiness.candidateGenerationReady,
    autoPromotionReady: readiness.autoPromotionReady,
    actualPromotionAllowed: false,
    reasons: schedulerReasons(readiness, lineageDecisions, decision),
    blockedGateIds: readiness.blockedGateIds,
    lineageDecisionIds: lineageDecisions.map((lineage) => lineage.lineageManifestId),
    rolloutPlan: rolloutPlan(readiness, decision),
  });
};

const countRealReplayCases = (replayCases: readonly ReplayEvalCaseSkeleton[]): number =>
  replayCases.filter(isRealReplayCase).length;

const isRealReplayCase = (replayCase: ReplayEvalCaseSkeleton): boolean =>
  replayCase.tags.includes("live") ||
  (replayCase.sourceSessionId != null && !replayCase.sourceSessionId.startsWith("session.replay."));

const countVisibleReplayCases = (replayCases: readonly ReplayEvalCaseSkeleton[]): number =>
  replayCases.filter((replayCase) =>
    replayCase.split !== "holdout" &&
    !replayCase.redaction.needsReview &&
    replayCase.redaction.status !== "raw_local_only"
  ).length;

const countRepeatedFailureClusters = (
  traceReports: readonly TraceAnalysisReport[],
  repeatedFailureMinCount: number,
): number =>
  traceReports
    .flatMap((report) => report.failureClusters)
    .filter((cluster) => cluster.count >= repeatedFailureMinCount)
    .length;

const countEditFailureSignals = (input: AssessGepaEvidenceReadinessInput): number => {
  const replaySignals = (input.replayCases ?? []).flatMap((replayCase) =>
    replayCase.observedFailures.filter((failure) => failure.failureKind === "edit_attempt")
  ).length;
  const ablationSignals = (input.editAblationReports ?? []).flatMap((report) =>
    report.probeResults.filter((probe) =>
      probe.status !== "passed" ||
      !probe.expectedOutcomeMatched ||
      probe.postApplyConsistencyStatus === "inconsistent" ||
      probe.verificationStatus === "failed" ||
      probe.verificationStatus === "error" ||
      probe.selfDetectedRegressionStatus === "confirmed"
    )
  ).length;
  const evidenceSignals = (input.evidenceBundles ?? []).flatMap((bundle) =>
    bundle.observations.filter((observation) =>
      observation.lineage.editStrategyIds.length > 0 ||
      observation.lineage.editStrategyFamilies.length > 0 ||
      observation.lineage.renderedEditToolContractIds.length > 0
    )
  ).length;
  return replaySignals + ablationSignals + evidenceSignals;
};

const countToolFailureSignals = (input: AssessGepaEvidenceReadinessInput): number => {
  const replaySignals = (input.replayCases ?? []).flatMap((replayCase) =>
    replayCase.observedFailures.filter((failure) => failure.failureKind === "tool_call")
  ).length;
  const traceSignals = (input.traceReports ?? []).flatMap((report) =>
    report.failureClusters.filter((cluster) =>
      cluster.observationKind.toLowerCase().includes("tool") ||
      cluster.name.toLowerCase().includes("tool")
    )
  ).length;
  const evidenceSignals = (input.evidenceBundles ?? []).flatMap((bundle) =>
    bundle.observations.filter((observation) => observation.toolNames.length > 0)
  ).length;
  return replaySignals + traceSignals + evidenceSignals;
};

const inferMetricObservationCount = (input: AssessGepaEvidenceReadinessInput): number =>
  (input.evalRunResults ?? []).length +
  (input.evalScorecards ?? []).length +
  (input.evalRunResults ?? []).reduce((total, run) => total + run.objectiveMetrics.length, 0) +
  (input.editAblationReports ?? []).reduce((total, report) => total + report.probeResults.length, 0);

const postPromotionWindowMs = (
  window: AssessGepaEvidenceReadinessInput["postPromotionMonitoringWindow"],
): number => {
  if (window == null) {
    return 0;
  }
  const startedAt = Date.parse(window.startedAt);
  const endedAt = Date.parse(window.endedAt);
  if (!Number.isFinite(startedAt) || !Number.isFinite(endedAt)) {
    return 0;
  }
  return Math.max(0, endedAt - startedAt);
};

const minGate = (
  gateId: GepaOperationReadinessGate["gateId"],
  phase: GepaOperationReadinessGate["phase"],
  actual: number,
  required: number,
): GepaOperationReadinessGate =>
  GepaOperationReadinessGateSchema.parse({
    gateId,
    phase,
    actual,
    required,
    passed: actual >= required,
    message: actual >= required
      ? `${gateId} passed with ${actual} >= ${required}.`
      : `${gateId} blocked with ${actual} < ${required}.`,
  });

const maxGate = (
  gateId: GepaOperationReadinessGate["gateId"],
  phase: GepaOperationReadinessGate["phase"],
  actual: number,
  required: number,
): GepaOperationReadinessGate =>
  GepaOperationReadinessGateSchema.parse({
    gateId,
    phase,
    actual,
    required,
    passed: actual <= required,
    message: actual <= required
      ? `${gateId} passed with ${actual} <= ${required}.`
      : `${gateId} blocked with ${actual} > ${required}.`,
  });

const schedulerReasons = (
  readiness: GepaOperationReadiness,
  lineageDecisions: readonly OptimizerArtifactLineageDecision[],
  decision: GepaDryRunSchedulerDecision["decision"],
): string[] => [
  `dry-run decision: ${decision}`,
  readiness.candidateGenerationReady
    ? "candidate generation readiness gates passed"
    : `candidate generation blocked by: ${readiness.blockedGateIds.join(", ")}`,
  readiness.autoPromotionReady
    ? "auto-promotion readiness gates passed, but this scheduler is dry-run only"
    : "auto-promotion remains disabled until operator approval and monitoring windows are satisfied",
  ...(lineageDecisions.length === 0
    ? ["no candidate lineage decisions supplied"]
    : lineageDecisions.map((lineage) => `${lineage.candidatePatchId}: ${lineage.decision}`)),
];

const rolloutPlan = (
  readiness: GepaOperationReadiness,
  decision: GepaDryRunSchedulerDecision["decision"],
): GepaDryRunSchedulerDecision["rolloutPlan"] => [
  {
    stage: "observe",
    status: readiness.candidateGenerationReady ? "ready" : "blocked",
    notes: "Collect real replay, scorecard, profile, and trace evidence before candidate generation.",
  },
  {
    stage: "generate_candidate",
    status: readiness.candidateGenerationReady ? "ready" : "blocked",
    notes: "Generate scoped optimizer artifacts only; do not rewrite runtime source.",
  },
  {
    stage: "evaluate_visible",
    status: readiness.candidateGenerationReady ? "ready" : "blocked",
    notes: "Evaluate train/dev visible splits and trace-mined scorecards.",
  },
  {
    stage: "evaluate_holdout",
    status: decision === "would_promote" ? "ready" : "blocked",
    notes: "Hidden holdout must stay out of candidate generation and be used only for final gating.",
  },
  {
    stage: "operator_approval",
    status: "future",
    notes: "Actual promotion requires explicit operator approval; dry-run output is advisory.",
  },
  {
    stage: "promote_new_sessions",
    status: "future",
    notes: "Promotion applies only to new sessions with session pinning.",
  },
  {
    stage: "monitor",
    status: "future",
    notes: "Monitor post-promotion scorecards and live traces within a bounded regression budget.",
  },
  {
    stage: "rollback",
    status: "future",
    notes: "Rollback restores the previous active pointer from checkpoint if regressions exceed budget.",
  },
];
