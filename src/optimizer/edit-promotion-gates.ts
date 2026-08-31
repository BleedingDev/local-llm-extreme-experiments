import { z } from "zod";
import type { BagConfig } from "../types";
import {
  EditStrategyAblationReportSchema,
  type EditStrategyAblationProbeResult,
  type EditStrategyAblationReport,
} from "../eval-harness/edit-strategy-ablation";
import {
  EvalScorecardSchema,
  EvalSplitSchema,
  type EvalScorecard,
} from "../eval-harness/types";
import { EditStrategyFamilySchema, type EditStrategyFamily } from "../edit-strategy/types";
import {
  evaluateRealAcpStabilityPromotionVetoes,
  type RealAcpStabilityScorecard,
  type RealAcpStabilityVetoThresholds,
} from "../replay/real-acp-scorecard";
import {
  promoteCandidatePatch,
  type CandidatePromotionResult,
} from "./promotion";
import {
  CandidatePatchSchema,
  type CandidatePatch,
} from "./types";
import {
  CandidateValidationResultSchema,
  type CandidateValidationResult,
} from "./validator";
import type { OptimizerArtifactLineageDecision } from "./artifact-lineage";

const EDIT_PROMOTION_GATES_SCHEMA_VERSION = "edit-promotion-gates.v1";

export const EditPromotionThresholdsSchema = z.object({
  minAggregateScore: z.number().min(0).max(1).default(1),
  maxLatencyMs: z.number().positive().optional(),
  maxTokenCount: z.number().positive().optional(),
}).strict();
export type EditPromotionThresholds = z.infer<typeof EditPromotionThresholdsSchema>;

export const EditPromotionGateResultSchema = z.object({
  gateId: z.enum([
    "visible-train-dev-evals",
    "hidden-holdout-eval",
    "hidden-holdout-not-training-input",
    "critical-protected-path-veto",
    "post-apply-consistency-veto",
    "stability-scorecard-veto",
    "latency-cost-constraints",
  ]),
  passed: z.boolean(),
  blocking: z.boolean().default(true),
  message: z.string().min(1),
  scorecardIds: z.array(z.string()).default([]),
  editAblationReportIds: z.array(z.string()).default([]),
}).strict();
export type EditPromotionGateResult = z.infer<typeof EditPromotionGateResultSchema>;

export const EditPromotionGateDecisionSchema = z.object({
  schemaVersion: z.literal(EDIT_PROMOTION_GATES_SCHEMA_VERSION),
  passed: z.boolean(),
  candidatePatchId: z.string().min(1),
  visibleScorecardIds: z.array(z.string()).default([]),
  holdoutScorecardIds: z.array(z.string()).default([]),
  editAblationReportIds: z.array(z.string()).default([]),
  gateResults: z.array(EditPromotionGateResultSchema),
}).strict();
export type EditPromotionGateDecision = z.infer<typeof EditPromotionGateDecisionSchema>;

export type EvaluateEditPromotionGatesInput = {
  candidate: CandidatePatch;
  visibleEvalScorecards: readonly EvalScorecard[];
  holdoutEvalScorecards: readonly EvalScorecard[];
  editAblationReports?: readonly EditStrategyAblationReport[];
  candidateStrategyFamilies?: readonly EditStrategyFamily[];
  realAcpStability?: {
    baseline: RealAcpStabilityScorecard;
    candidate: RealAcpStabilityScorecard;
    thresholds?: RealAcpStabilityVetoThresholds;
  };
  thresholds?: Partial<EditPromotionThresholds>;
};

export const EditStrategyPromotionResultSchema = z.object({
  promoted: z.boolean(),
  candidatePatchId: z.string().min(1),
  gateDecision: EditPromotionGateDecisionSchema,
  promotion: z.unknown().optional(),
});
export type EditStrategyPromotionResult = z.infer<typeof EditStrategyPromotionResultSchema> & {
  promotion?: CandidatePromotionResult | undefined;
};

export type PromoteEditStrategyCandidateInput = EvaluateEditPromotionGatesInput & {
  config: BagConfig;
  cwd?: string;
  validation: CandidateValidationResult;
  candidateEval: EvalScorecard;
  lineageDecision?: OptimizerArtifactLineageDecision;
  decidedAt?: string;
  decisionId?: string;
};

export const evaluateEditPromotionGates = (
  input: EvaluateEditPromotionGatesInput,
): EditPromotionGateDecision => {
  const candidate = CandidatePatchSchema.parse(input.candidate);
  const visibleEvalScorecards = input.visibleEvalScorecards.map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const holdoutEvalScorecards = input.holdoutEvalScorecards.map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const editAblationReports = (input.editAblationReports ?? []).map((report) => EditStrategyAblationReportSchema.parse(report));
  const thresholds = EditPromotionThresholdsSchema.parse(input.thresholds ?? {});
  const families = new Set(input.candidateStrategyFamilies ?? []);
  const gateResults = [
    visibleTrainDevEvalGate(visibleEvalScorecards, thresholds),
    hiddenHoldoutEvalGate(holdoutEvalScorecards, thresholds),
    hiddenHoldoutTrainingGate(editAblationReports),
    criticalProtectedPathGate(editAblationReports, families),
    postApplyConsistencyGate(editAblationReports, families),
    stabilityScorecardGate(input.realAcpStability),
    latencyCostGate([...visibleEvalScorecards, ...holdoutEvalScorecards], thresholds),
  ];

  return EditPromotionGateDecisionSchema.parse({
    schemaVersion: EDIT_PROMOTION_GATES_SCHEMA_VERSION,
    candidatePatchId: candidate.candidatePatchId,
    passed: gateResults.every((gate) => gate.passed || !gate.blocking),
    visibleScorecardIds: visibleEvalScorecards.map((scorecard) => scorecard.scorecardId),
    holdoutScorecardIds: holdoutEvalScorecards.map((scorecard) => scorecard.scorecardId),
    editAblationReportIds: editAblationReports.map((report) => report.ablationRunId),
    gateResults,
  });
};

export const promoteEditStrategyCandidate = (
  input: PromoteEditStrategyCandidateInput,
): EditStrategyPromotionResult => {
  const validation = CandidateValidationResultSchema.parse(input.validation);
  const candidateEval = EvalScorecardSchema.parse(input.candidateEval);
  const gateDecision = evaluateEditPromotionGates(input);
  if (!gateDecision.passed) {
    return EditStrategyPromotionResultSchema.parse({
      promoted: false,
      candidatePatchId: input.candidate.candidatePatchId,
      gateDecision,
    }) as EditStrategyPromotionResult;
  }

  const promotion = promoteCandidatePatch({
    config: input.config,
    ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
    candidate: input.candidate,
    validation,
    candidateEval,
    ...(input.lineageDecision === undefined ? {} : { lineageDecision: input.lineageDecision }),
    ...(input.decidedAt === undefined ? {} : { decidedAt: input.decidedAt }),
    ...(input.decisionId === undefined ? {} : { decisionId: input.decisionId }),
  });

  return EditStrategyPromotionResultSchema.parse({
    promoted: promotion.promoted,
    candidatePatchId: input.candidate.candidatePatchId,
    gateDecision,
    promotion,
  }) as EditStrategyPromotionResult;
};

const visibleTrainDevEvalGate = (
  scorecards: readonly EvalScorecard[],
  thresholds: EditPromotionThresholds,
): EditPromotionGateResult => {
  const visibleScorecards = scorecards.filter((scorecard) => scorecard.split !== "holdout");
  const splits = new Set(visibleScorecards.map((scorecard) => scorecard.split));
  const missingSplits = ["train", "dev"].filter((split) => !splits.has(EvalSplitSchema.parse(split)));
  const failing = visibleScorecards.filter((scorecard) => !scorecardPassed(scorecard, thresholds));
  const passed = missingSplits.length === 0 && failing.length === 0;
  return EditPromotionGateResultSchema.parse({
    gateId: "visible-train-dev-evals",
    passed,
    message: passed
      ? "Visible train/dev evals passed candidate score and critical regression gates."
      : `Visible train/dev eval gate failed${missingSplits.length > 0 ? `; missing splits: ${missingSplits.join(", ")}` : ""}${failing.length > 0 ? `; failing scorecards: ${failing.map((scorecard) => scorecard.scorecardId).join(", ")}` : ""}.`,
    scorecardIds: visibleScorecards.map((scorecard) => scorecard.scorecardId),
  });
};

const hiddenHoldoutEvalGate = (
  scorecards: readonly EvalScorecard[],
  thresholds: EditPromotionThresholds,
): EditPromotionGateResult => {
  const holdoutScorecards = scorecards.filter((scorecard) => scorecard.split === "holdout");
  const failing = holdoutScorecards.filter((scorecard) => !scorecardPassed(scorecard, thresholds));
  const passed = holdoutScorecards.length > 0 && failing.length === 0;
  return EditPromotionGateResultSchema.parse({
    gateId: "hidden-holdout-eval",
    passed,
    message: passed
      ? "Hidden holdout evals passed without promotion-blocking regressions."
      : holdoutScorecards.length === 0
        ? "Hidden holdout eval gate failed; no holdout scorecard was supplied."
        : `Hidden holdout eval gate failed; failing scorecards: ${failing.map((scorecard) => scorecard.scorecardId).join(", ")}.`,
    scorecardIds: holdoutScorecards.map((scorecard) => scorecard.scorecardId),
  });
};

const hiddenHoldoutTrainingGate = (
  reports: readonly EditStrategyAblationReport[],
): EditPromotionGateResult => {
  const leaked = reports.filter((report) => report.hiddenHoldoutUsed && report.optimizationAllowed);
  return EditPromotionGateResultSchema.parse({
    gateId: "hidden-holdout-not-training-input",
    passed: leaked.length === 0,
    message: leaked.length === 0
      ? "No hidden holdout edit ablation report was marked as optimization/training input."
      : `Hidden holdout leakage detected in optimization reports: ${leaked.map((report) => report.ablationRunId).join(", ")}.`,
    editAblationReportIds: reports.map((report) => report.ablationRunId),
  });
};

const criticalProtectedPathGate = (
  reports: readonly EditStrategyAblationReport[],
  families: ReadonlySet<EditStrategyFamily>,
): EditPromotionGateResult => {
  const offenders = relevantProbeResults(reports, families).filter((probe) => probe.protectedPathTouched);
  return EditPromotionGateResultSchema.parse({
    gateId: "critical-protected-path-veto",
    passed: offenders.length === 0,
    message: offenders.length === 0
      ? "No relevant edit ablation probe touched protected paths."
      : `Protected-path veto from probes: ${offenders.map((probe) => probe.probeId).join(", ")}.`,
    editAblationReportIds: reports.map((report) => report.ablationRunId),
  });
};

const postApplyConsistencyGate = (
  reports: readonly EditStrategyAblationReport[],
  families: ReadonlySet<EditStrategyFamily>,
): EditPromotionGateResult => {
  const offenders = relevantProbeResults(reports, families).filter((probe) =>
    probe.postApplyConsistencyStatus === "inconsistent" ||
    probe.verificationStatus === "failed" ||
    probe.verificationStatus === "error" ||
    probe.selfDetectedRegressionStatus === "confirmed",
  );
  return EditPromotionGateResultSchema.parse({
    gateId: "post-apply-consistency-veto",
    passed: offenders.length === 0,
    message: offenders.length === 0
      ? "No relevant edit ablation probe showed applied-but-broken or self-detected regression behavior."
      : `Post-apply consistency veto from probes: ${offenders.map((probe) => probe.probeId).join(", ")}.`,
    editAblationReportIds: reports.map((report) => report.ablationRunId),
  });
};

const stabilityScorecardGate = (
  stability: EvaluateEditPromotionGatesInput["realAcpStability"],
): EditPromotionGateResult => {
  if (stability === undefined) {
    return EditPromotionGateResultSchema.parse({
      gateId: "stability-scorecard-veto",
      passed: true,
      blocking: false,
      message: "No real ACP stability scorecard was supplied; gate is informational only.",
    });
  }
  const vetoes = evaluateRealAcpStabilityPromotionVetoes(stability);
  const failed = vetoes.filter((veto) => !veto.passed);
  return EditPromotionGateResultSchema.parse({
    gateId: "stability-scorecard-veto",
    passed: failed.length === 0,
    blocking: true,
    message: failed.length === 0
      ? "Real ACP stability scorecard vetoes passed."
      : `Real ACP stability veto failed: ${failed.map((veto) => veto.message).join(" ")}`,
    scorecardIds: [stability.baseline.scorecardId, stability.candidate.scorecardId],
  });
};

const latencyCostGate = (
  scorecards: readonly EvalScorecard[],
  thresholds: EditPromotionThresholds,
): EditPromotionGateResult => {
  const metrics = scorecards.flatMap((scorecard) => scorecard.objectiveMetrics);
  const latencyOffenders = thresholds.maxLatencyMs === undefined
    ? []
    : metrics.filter((metric) => metric.unit === "ms" && metric.value > thresholds.maxLatencyMs!);
  const tokenOffenders = thresholds.maxTokenCount === undefined
    ? []
    : metrics.filter((metric) => metric.unit === "tokens" && metric.value > thresholds.maxTokenCount!);
  const offenders = [...latencyOffenders, ...tokenOffenders];
  return EditPromotionGateResultSchema.parse({
    gateId: "latency-cost-constraints",
    passed: offenders.length === 0,
    message: offenders.length === 0
      ? "Latency and token-cost constraints passed."
      : `Latency/cost constraints failed: ${offenders.map((metric) => `${metric.metricId}=${metric.value}${metric.unit}`).join(", ")}.`,
    scorecardIds: scorecards.map((scorecard) => scorecard.scorecardId),
  });
};

const scorecardPassed = (scorecard: EvalScorecard, thresholds: EditPromotionThresholds): boolean =>
  scorecard.passed &&
  !scorecard.criticalRegressionVeto.vetoed &&
  scorecard.aggregateScore + 1e-9 >= thresholds.minAggregateScore;

const relevantProbeResults = (
  reports: readonly EditStrategyAblationReport[],
  families: ReadonlySet<EditStrategyFamily>,
): EditStrategyAblationProbeResult[] =>
  reports.flatMap((report) =>
    report.probeResults.filter((probe) =>
      families.size === 0 || families.has(EditStrategyFamilySchema.parse(probe.strategyFamily)),
    ),
  );
