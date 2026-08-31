import { z } from "zod";
import {
  EditStrategyFamilySchema,
} from "../edit-strategy/types";
import {
  parseCanonicalEditStrategyDefinitions,
  type CanonicalEditStrategyDefinition,
} from "../edit-strategy/taxonomy";
import {
  EditStrategyAblationReportSchema,
  type EditStrategyAblationFamilySummary,
  type EditStrategyAblationReport,
} from "../eval-harness/edit-strategy-ablation";
import { OptimizerIdSchema } from "./types";
import type { ResolvedOptimizerPolicy } from "./policy-resolver";

const ROUTER_SCHEMA_VERSION = "edit-policy-router.v1";
const WHOLE_FILE_BASELINE_ID = "edit.whole-file.acp-write.v1";

export const EditVerifierStrengthSchema = z.enum(["none", "basic", "strong"]);
export type EditVerifierStrength = z.infer<typeof EditVerifierStrengthSchema>;

export const EditRiskLevelSchema = z.enum(["low", "medium", "high"]);
export type EditRiskLevel = z.infer<typeof EditRiskLevelSchema>;

export const EditTaskShapeSchema = z.object({
  targetFileCount: z.number().int().nonnegative().default(1),
  estimatedChangedFileCount: z.number().int().nonnegative().optional(),
  largestTargetFileBytes: z.number().int().nonnegative().default(0),
  totalTargetFileBytes: z.number().int().nonnegative().default(0),
  contextBudgetTokens: z.number().int().positive().optional(),
  outputBudgetTokens: z.number().int().positive().optional(),
  verifierStrength: EditVerifierStrengthSchema.default("basic"),
  protectedPathRisk: EditRiskLevelSchema.default("medium"),
  staleContextRisk: EditRiskLevelSchema.default("medium"),
  requiresMultiFileConsistency: z.boolean().default(false),
}).strict();
export type EditTaskShape = z.infer<typeof EditTaskShapeSchema>;

export const EditTaskShapeBucketSchema = z.object({
  targetFileCount: z.enum(["zero", "single", "multi"]),
  changedFileCount: z.enum(["unknown", "single", "multi"]),
  fileSize: z.enum(["empty", "small", "large"]),
  budgetFit: z.enum(["fits", "output_constrained", "context_constrained", "output_and_context_constrained"]),
  verifierStrength: EditVerifierStrengthSchema,
  protectedPathRisk: EditRiskLevelSchema,
  staleContextRisk: EditRiskLevelSchema,
  requiresMultiFileConsistency: z.boolean(),
}).strict();
export type EditTaskShapeBucket = z.infer<typeof EditTaskShapeBucketSchema>;

export const EditStrategyHistoricalMetricSchema = z.object({
  metricId: OptimizerIdSchema,
  source: z.enum(["trace", "ablation", "manual"]),
  trainingAllowed: z.boolean().default(true),
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  strategyFamily: EditStrategyFamilySchema,
  strategyId: OptimizerIdSchema.optional(),
  taskShapeBucket: EditTaskShapeBucketSchema.optional(),
  sampleCount: z.number().int().nonnegative(),
  taskPassRate: z.number().min(0).max(1),
  expectedOutcomeMatchRate: z.number().min(0).max(1),
  parsePassRate: z.number().min(0).max(1),
  applyAcceptedRate: z.number().min(0).max(1),
  coverageRate: z.number().min(0).max(1).default(1),
  averageScore: z.number().min(0).max(1),
  protectedPathTouchRate: z.number().min(0).max(1).default(0),
  staleRejectionRate: z.number().min(0).max(1).default(0),
  appliedButBrokenRate: z.number().min(0).max(1).default(0),
  averageLatencyMs: z.number().nonnegative().optional(),
}).strict();
export type EditStrategyHistoricalMetric = z.infer<typeof EditStrategyHistoricalMetricSchema>;

export const EditStrategyRouterPolicySchema = z.object({
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  editStrategyVersion: z.string().min(1),
  renderedEditContractVersion: z.string().min(1),
  editFallbackPolicyVersion: z.string().min(1),
  editRepairPolicyVersion: z.string().min(1),
  editVerifierPolicyVersion: z.string().min(1),
  editObjectiveSetId: OptimizerIdSchema,
}).passthrough();

export const EditStrategyRouterInputSchema = z.object({
  resolvedPolicy: EditStrategyRouterPolicySchema,
  taskShape: EditTaskShapeSchema.optional(),
  definitions: z.array(z.any()).optional(),
  ablationReports: z.array(EditStrategyAblationReportSchema).default([]),
  historicalMetrics: z.array(EditStrategyHistoricalMetricSchema).default([]),
  includeFutureGated: z.boolean().default(false),
  minSampleCount: z.number().int().positive().default(1),
}).strict();
export type EditStrategyRouterInput = Omit<z.input<typeof EditStrategyRouterInputSchema>, "resolvedPolicy"> & {
  resolvedPolicy: Pick<
    ResolvedOptimizerPolicy,
    | "modelProfileId"
    | "codebaseProfileId"
    | "policyId"
    | "editStrategyVersion"
    | "renderedEditContractVersion"
    | "editFallbackPolicyVersion"
    | "editRepairPolicyVersion"
    | "editVerifierPolicyVersion"
    | "editObjectiveSetId"
  >;
};

export const EditStrategyRouteCandidateSchema = z.object({
  strategyId: OptimizerIdSchema,
  strategyFamily: EditStrategyFamilySchema,
  eligible: z.boolean(),
  rankingScore: z.number().min(0).max(1),
  measurementScore: z.number().min(0).max(1),
  evidenceConfidence: z.number().min(0).max(1),
  sampleCount: z.number().int().nonnegative(),
  reasons: z.array(z.string()).default([]),
  blockers: z.array(z.string()).default([]),
  fallbackEligible: z.boolean(),
  supportsMultiFile: z.boolean(),
  supportsPartialRead: z.boolean(),
  requiresWholeFileOutput: z.boolean(),
  deterministicApply: z.boolean(),
}).strict();
export type EditStrategyRouteCandidate = z.infer<typeof EditStrategyRouteCandidateSchema>;

export const EditStrategyFallbackRuleSchema = z.object({
  trigger: z.enum([
    "parse_failed",
    "apply_failed",
    "stale_context",
    "protected_path_violation",
    "post_apply_inconsistent",
    "verification_failed",
    "self_detected_regression",
    "context_budget_exceeded",
  ]),
  action: z.enum(["try_next_strategy", "repair_then_try_next", "abort"]),
  fromStrategyId: OptimizerIdSchema,
  toStrategyId: OptimizerIdSchema.optional(),
  reason: z.string().min(1),
}).strict();
export type EditStrategyFallbackRule = z.infer<typeof EditStrategyFallbackRuleSchema>;

export const EditStrategyRouterDecisionSchema = z.object({
  schemaVersion: z.literal(ROUTER_SCHEMA_VERSION),
  policyId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  editStrategyVersion: z.string().min(1),
  renderedEditContractVersion: z.string().min(1),
  selectedStrategyId: OptimizerIdSchema,
  selectedStrategyFamily: EditStrategyFamilySchema,
  degraded: z.boolean(),
  candidates: z.array(EditStrategyRouteCandidateSchema).min(1),
  fallbackRules: z.array(EditStrategyFallbackRuleSchema).default([]),
  evidenceUsed: z.array(z.object({
    source: z.enum(["ablation", "trace", "manual"]),
    metricId: OptimizerIdSchema,
    strategyFamily: EditStrategyFamilySchema,
    taskShapeBucketId: z.string().min(1).optional(),
    sampleCount: z.number().int().nonnegative(),
  }).strict()).default([]),
  warnings: z.array(z.string()).default([]),
}).strict();
export type EditStrategyRouterDecision = z.infer<typeof EditStrategyRouterDecisionSchema>;

type ParsedRouterInput = Omit<z.output<typeof EditStrategyRouterInputSchema>, "taskShape"> & {
  taskShape: EditTaskShape;
};

type CandidateMeasurement = {
  measurementScore: number;
  evidenceConfidence: number;
  sampleCount: number;
  taskPassRate: number;
  expectedOutcomeMatchRate: number;
  parsePassRate: number;
  applyAcceptedRate: number;
  coverageRate: number;
  averageScore: number;
  protectedPathTouchRate: number;
  staleRejectionRate: number;
  appliedButBrokenRate: number;
  evidenceUsed: EditStrategyRouterDecision["evidenceUsed"];
};

export const routeEditStrategy = (rawInput: EditStrategyRouterInput): EditStrategyRouterDecision => {
  const parsedInput = EditStrategyRouterInputSchema.parse(rawInput);
  const input: ParsedRouterInput = {
    ...parsedInput,
    taskShape: EditTaskShapeSchema.parse(parsedInput.taskShape ?? {}),
  };
  const definitions = parseCanonicalEditStrategyDefinitions(input.definitions);
  const metrics = metricsForPolicy(input);
  const warnings = holdoutWarnings(input.ablationReports);
  const candidates = definitions
    .map((definition) => routeCandidate({
      definition,
      metrics: metrics.filter((metric) => metric.strategyFamily === definition.family),
      input,
    }))
    .sort(compareCandidates);

  const selectable = candidates.find((candidate) => candidate.eligible);
  const baseline = candidates.find((candidate) => candidate.strategyId === WHOLE_FILE_BASELINE_ID);
  const selected = selectable ?? baseline ?? candidates[0];
  if (selected === undefined) {
    throw new Error("edit policy router requires at least one canonical edit strategy definition");
  }

  const selectedWithDegradedReason = selected.eligible
    ? selected
    : {
        ...selected,
        reasons: [...selected.reasons, "degraded fallback selected because no eligible measured strategy was available"],
      };
  const rankedCandidates = candidates.map((candidate) =>
    candidate.strategyId === selected.strategyId ? selectedWithDegradedReason : candidate,
  );

  return EditStrategyRouterDecisionSchema.parse({
    schemaVersion: ROUTER_SCHEMA_VERSION,
    policyId: input.resolvedPolicy.policyId,
    modelProfileId: input.resolvedPolicy.modelProfileId,
    codebaseProfileId: input.resolvedPolicy.codebaseProfileId,
    editStrategyVersion: input.resolvedPolicy.editStrategyVersion,
    renderedEditContractVersion: input.resolvedPolicy.renderedEditContractVersion,
    selectedStrategyId: selected.strategyId,
    selectedStrategyFamily: selected.strategyFamily,
    degraded: !selected.eligible,
    candidates: rankedCandidates,
    fallbackRules: fallbackRulesFor(selected, rankedCandidates),
    evidenceUsed: evidenceUsedForRoute(definitions, metrics, input),
    warnings,
  });
};

const routeCandidate = (input: {
  definition: CanonicalEditStrategyDefinition;
  metrics: readonly EditStrategyHistoricalMetric[];
  input: ParsedRouterInput;
}): EditStrategyRouteCandidate => {
  const measurement = combineMeasurements(input.metrics, input.input.minSampleCount, input.input.taskShape);
  const blockers = candidateBlockers(input.definition, input.input.taskShape, measurement, input.input.includeFutureGated);
  const reasons = candidateReasons(input.definition, measurement, input.input.taskShape, blockers);
  const capabilityScore = capabilityScoreFor(input.definition, input.input.taskShape);
  const riskPenalty =
    measurement.protectedPathTouchRate * protectedPathRiskWeight(input.input.taskShape.protectedPathRisk) +
    measurement.appliedButBrokenRate * appliedButBrokenRiskWeight(input.input.taskShape.verifierStrength) +
    measurement.staleRejectionRate * staleContextRiskWeight(input.input.taskShape.staleContextRisk);
  const measuredScore = measurement.evidenceConfidence === 0
    ? unmeasuredScore(input.definition, input.input.taskShape)
    : measurement.measurementScore;
  const rankingScore = clamp01((measuredScore * 0.72 + capabilityScore * 0.28) - riskPenalty);

  return EditStrategyRouteCandidateSchema.parse({
    strategyId: input.definition.strategyId,
    strategyFamily: input.definition.family,
    eligible: blockers.length === 0,
    rankingScore,
    measurementScore: measuredScore,
    evidenceConfidence: measurement.evidenceConfidence,
    sampleCount: measurement.sampleCount,
    reasons,
    blockers,
    fallbackEligible: blockers.length === 0 && input.definition.futureGate === "none",
    supportsMultiFile: input.definition.supportsMultiFile,
    supportsPartialRead: input.definition.supportsPartialRead,
    requiresWholeFileOutput: input.definition.requiresWholeFileOutput,
    deterministicApply: input.definition.deterministicApply,
  });
};

const metricsForPolicy = (input: ParsedRouterInput): EditStrategyHistoricalMetric[] => {
  const fromAblationReports = input.ablationReports.flatMap((report) =>
    report.optimizationAllowed
      ? metricsFromAblationReport(report, input.resolvedPolicy.modelProfileId, input.resolvedPolicy.codebaseProfileId)
      : [],
  );
  const directMetrics = input.historicalMetrics.filter((metric) =>
    metric.trainingAllowed &&
    metric.modelProfileId === input.resolvedPolicy.modelProfileId &&
    metric.codebaseProfileId === input.resolvedPolicy.codebaseProfileId
  );
  return [...fromAblationReports, ...directMetrics];
};

const metricsFromAblationReport = (
  report: EditStrategyAblationReport,
  modelProfileId: string,
  codebaseProfileId: string,
): EditStrategyHistoricalMetric[] =>
  report.familySummaries
    .filter((summary) => summary.modelProfileId === modelProfileId && summary.codebaseProfileId === codebaseProfileId)
    .map((summary) => metricFromAblationSummary(report, summary));

const metricFromAblationSummary = (
  report: EditStrategyAblationReport,
  summary: EditStrategyAblationFamilySummary,
): EditStrategyHistoricalMetric =>
  EditStrategyHistoricalMetricSchema.parse({
    metricId: `metric.${idPart(report.ablationRunId)}.${idPart(summary.summaryId)}`,
    source: "ablation",
    trainingAllowed: report.optimizationAllowed,
    modelProfileId: summary.modelProfileId,
    codebaseProfileId: summary.codebaseProfileId,
    strategyFamily: summary.strategyFamily,
    sampleCount: summary.probeCount,
    taskPassRate: summary.taskPassRate,
    expectedOutcomeMatchRate: summary.expectedOutcomeMatchRate,
    parsePassRate: summary.parsePassRate,
    applyAcceptedRate: summary.applyAcceptedRate,
    coverageRate: summary.coverageRate,
    averageScore: summary.averageScore,
    protectedPathTouchRate: ratio(summary.protectedPathTouchCount, summary.probeCount),
    staleRejectionRate: ratio(summary.staleRejectionCount, summary.probeCount),
    appliedButBrokenRate: ratio(summary.appliedButBrokenCount, summary.probeCount),
  });

const evidenceUsedForRoute = (
  definitions: readonly CanonicalEditStrategyDefinition[],
  metrics: readonly EditStrategyHistoricalMetric[],
  input: ParsedRouterInput,
): EditStrategyRouterDecision["evidenceUsed"] =>
  uniqueEvidence(definitions.flatMap((definition) =>
    combineMeasurements(
      metrics.filter((metric) => metric.strategyFamily === definition.family),
      input.minSampleCount,
      input.taskShape,
    ).evidenceUsed
  ));

const combineMeasurements = (
  metrics: readonly EditStrategyHistoricalMetric[],
  minSampleCount: number,
  taskShape: EditTaskShape,
): CandidateMeasurement => {
  const usableForPolicy = metrics.filter((metric) => metric.trainingAllowed && metric.sampleCount >= minSampleCount);
  const exactTaskShape = taskShapeBucketFor(taskShape);
  const exactShapeMetrics = usableForPolicy.filter((metric) =>
    metric.taskShapeBucket !== undefined && taskShapeBucketMatches(metric.taskShapeBucket, exactTaskShape)
  );
  const genericMetrics = usableForPolicy.filter((metric) => metric.taskShapeBucket === undefined);
  const usable = exactShapeMetrics.length > 0 ? exactShapeMetrics : genericMetrics;
  if (usable.length === 0) {
    return {
      measurementScore: 0,
      evidenceConfidence: 0,
      sampleCount: 0,
      taskPassRate: 0,
      expectedOutcomeMatchRate: 0,
      parsePassRate: 0,
      applyAcceptedRate: 0,
      coverageRate: 0,
      averageScore: 0,
      protectedPathTouchRate: 0,
      staleRejectionRate: 0,
      appliedButBrokenRate: 0,
      evidenceUsed: [],
    };
  }

  const sampleCount = usable.reduce((sum, metric) => sum + metric.sampleCount, 0);
  const taskPassRate = weightedAverage(usable, (metric) => metric.taskPassRate);
  const expectedOutcomeMatchRate = weightedAverage(usable, (metric) => metric.expectedOutcomeMatchRate);
  const parsePassRate = weightedAverage(usable, (metric) => metric.parsePassRate);
  const applyAcceptedRate = weightedAverage(usable, (metric) => metric.applyAcceptedRate);
  const coverageRate = weightedAverage(usable, (metric) => metric.coverageRate);
  const averageScore = weightedAverage(usable, (metric) => metric.averageScore);
  const protectedPathTouchRate = weightedAverage(usable, (metric) => metric.protectedPathTouchRate);
  const staleRejectionRate = weightedAverage(usable, (metric) => metric.staleRejectionRate);
  const appliedButBrokenRate = weightedAverage(usable, (metric) => metric.appliedButBrokenRate);
  const measurementScore = clamp01(
    taskPassRate * 0.32 +
      expectedOutcomeMatchRate * 0.24 +
      applyAcceptedRate * 0.14 +
      parsePassRate * 0.1 +
      averageScore * 0.12 +
      coverageRate * 0.08 -
      protectedPathTouchRate * 0.2 -
      appliedButBrokenRate * 0.18,
  );

  return {
    measurementScore,
    evidenceConfidence: Math.min(1, sampleCount / Math.max(1, minSampleCount * 5)),
    sampleCount,
    taskPassRate,
    expectedOutcomeMatchRate,
    parsePassRate,
    applyAcceptedRate,
    coverageRate,
    averageScore,
    protectedPathTouchRate,
    staleRejectionRate,
    appliedButBrokenRate,
    evidenceUsed: usable.map((metric) => ({
      source: metric.source,
      metricId: metric.metricId,
      strategyFamily: metric.strategyFamily,
      ...(metric.taskShapeBucket === undefined ? {} : { taskShapeBucketId: taskShapeBucketId(metric.taskShapeBucket) }),
      sampleCount: metric.sampleCount,
    })),
  };
};

const candidateBlockers = (
  definition: CanonicalEditStrategyDefinition,
  taskShape: EditTaskShape,
  measurement: CandidateMeasurement,
  includeFutureGated: boolean,
): string[] => {
  const blockers: string[] = [];
  if (definition.futureGate !== "none" && !includeFutureGated) {
    blockers.push(`future gate not enabled: ${definition.futureGate}`);
  }
  if ((taskShape.requiresMultiFileConsistency || taskShape.targetFileCount > 1) && !definition.supportsMultiFile) {
    blockers.push("task requires multi-file consistency but strategy is single-file");
  }
  if (definition.requiresWholeFileOutput && outputBudgetExceeded(taskShape)) {
    blockers.push("whole-file output exceeds configured output budget");
  }
  if (definition.requiresWholeFileOutput && contextBudgetExceeded(taskShape)) {
    blockers.push("whole-file strategy exceeds configured context budget");
  }
  if (taskShape.protectedPathRisk === "high" && measurement.protectedPathTouchRate > 0) {
    blockers.push("measured protected-path touch rate is incompatible with high protected-path risk");
  }
  if (taskShape.verifierStrength === "none" && measurement.appliedButBrokenRate > 0) {
    blockers.push("measured applied-but-broken risk requires a verifier");
  }
  return blockers;
};

const candidateReasons = (
  definition: CanonicalEditStrategyDefinition,
  measurement: CandidateMeasurement,
  taskShape: EditTaskShape,
  blockers: readonly string[],
): string[] => {
  const reasons = [
    measurement.sampleCount > 0
      ? `measured locally with ${measurement.sampleCount} samples`
      : "no local measurement yet; ranked conservatively",
    definition.supportsPartialRead
      ? "supports partial-read/localized edit workflows"
      : "requires whole-file output",
    definition.deterministicApply
      ? "deterministic apply path returns structured errors"
      : "non-deterministic apply path requires stronger verification",
  ];
  if (taskShape.requiresMultiFileConsistency || taskShape.targetFileCount > 1) {
    reasons.push(definition.supportsMultiFile ? "compatible with multi-file task shape" : "not compatible with multi-file task shape");
  }
  if (blockers.length > 0) {
    reasons.push(`blocked: ${blockers.join("; ")}`);
  }
  return reasons;
};

const capabilityScoreFor = (definition: CanonicalEditStrategyDefinition, taskShape: EditTaskShape): number => {
  let score = 0.5;
  if (definition.deterministicApply) {
    score += 0.12;
  }
  if (definition.supportsPartialRead && taskShape.largestTargetFileBytes > 16_384) {
    score += 0.1;
  }
  if (definition.supportsMultiFile && (taskShape.targetFileCount > 1 || taskShape.requiresMultiFileConsistency)) {
    score += 0.12;
  }
  if (!definition.requiresWholeFileOutput && outputBudgetExceeded(taskShape)) {
    score += 0.12;
  }
  if (definition.requiresWholeFileOutput && taskShape.largestTargetFileBytes <= 16_384) {
    score += 0.04;
  }
  return clamp01(score);
};

const fallbackRulesFor = (
  selected: EditStrategyRouteCandidate,
  candidates: readonly EditStrategyRouteCandidate[],
): EditStrategyFallbackRule[] => {
  const next = candidates.find((candidate) =>
    candidate.strategyId !== selected.strategyId && candidate.fallbackEligible,
  );
  const tryNext = (trigger: EditStrategyFallbackRule["trigger"], reason: string): EditStrategyFallbackRule => ({
    trigger,
    action: next === undefined ? "abort" : "try_next_strategy",
    fromStrategyId: selected.strategyId,
    ...(next === undefined ? {} : { toStrategyId: next.strategyId }),
    reason,
  });
  const repairNext = (trigger: EditStrategyFallbackRule["trigger"], reason: string): EditStrategyFallbackRule => ({
    trigger,
    action: next === undefined ? "abort" : "repair_then_try_next",
    fromStrategyId: selected.strategyId,
    ...(next === undefined ? {} : { toStrategyId: next.strategyId }),
    reason,
  });

  return [
    tryNext("parse_failed", "Parser failure is a strategy/contract signal; try the next measured fallback if available."),
    tryNext("apply_failed", "Apply failure should not be hidden; fall back only through the measured fallback order."),
    tryNext("stale_context", "Stale context requires a fresh read before trying the next strategy."),
    {
      trigger: "protected_path_violation",
      action: "abort",
      fromStrategyId: selected.strategyId,
      reason: "Protected path touches are critical policy violations and must not be auto-masked by fallback.",
    },
    repairNext("post_apply_inconsistent", "Applied-but-broken edits require repair before any fallback strategy is attempted."),
    repairNext("verification_failed", "Verification failure is final-quality signal; repair first, then try the next measured fallback."),
    repairNext("self_detected_regression", "Evidence-bound self-detected regressions should enter repair before fallback."),
    tryNext("context_budget_exceeded", "Budget overrun should move to the next eligible non-whole-file strategy if available."),
  ].map((rule) => EditStrategyFallbackRuleSchema.parse(rule));
};

const holdoutWarnings = (reports: readonly EditStrategyAblationReport[]): string[] =>
  reports
    .filter((report) => !report.optimizationAllowed || report.hiddenHoldoutUsed)
    .map((report) => `ignored non-optimization ablation report ${report.ablationRunId} because it used hidden holdout cases`);

const compareCandidates = (left: EditStrategyRouteCandidate, right: EditStrategyRouteCandidate): number => {
  const eligibilityDelta = Number(right.eligible) - Number(left.eligible);
  if (eligibilityDelta !== 0) {
    return eligibilityDelta;
  }
  const scoreDelta = right.rankingScore - left.rankingScore;
  if (Math.abs(scoreDelta) > Number.EPSILON) {
    return scoreDelta;
  }
  const confidenceDelta = right.evidenceConfidence - left.evidenceConfidence;
  if (Math.abs(confidenceDelta) > Number.EPSILON) {
    return confidenceDelta;
  }
  return left.strategyId.localeCompare(right.strategyId);
};

const outputBudgetExceeded = (taskShape: EditTaskShape): boolean =>
  taskShape.outputBudgetTokens !== undefined &&
  taskShape.largestTargetFileBytes > taskShape.outputBudgetTokens * 4;

const contextBudgetExceeded = (taskShape: EditTaskShape): boolean =>
  taskShape.contextBudgetTokens !== undefined &&
  taskShape.totalTargetFileBytes > taskShape.contextBudgetTokens * 4;

const unmeasuredScore = (definition: CanonicalEditStrategyDefinition, taskShape: EditTaskShape): number => {
  const wholeFilePenalty = definition.requiresWholeFileOutput && (
    outputBudgetExceeded(taskShape) ||
    contextBudgetExceeded(taskShape) ||
    taskShape.largestTargetFileBytes > 16_384
  )
    ? 0.12
    : 0;
  const localizedBonus = definition.supportsPartialRead && !definition.requiresWholeFileOutput ? 0.04 : 0;
  return clamp01(0.22 + capabilityScoreFor(definition, taskShape) * 0.18 + localizedBonus - wholeFilePenalty);
};

export const taskShapeBucketFor = (rawTaskShape: EditTaskShape): EditTaskShapeBucket => {
  const taskShape = EditTaskShapeSchema.parse(rawTaskShape);
  const outputConstrained = outputBudgetExceeded(taskShape);
  const contextConstrained = contextBudgetExceeded(taskShape);
  return EditTaskShapeBucketSchema.parse({
    targetFileCount: taskShape.targetFileCount === 0 ? "zero" : taskShape.targetFileCount === 1 ? "single" : "multi",
    changedFileCount: taskShape.estimatedChangedFileCount === undefined
      ? "unknown"
      : taskShape.estimatedChangedFileCount <= 1
        ? "single"
        : "multi",
    fileSize: taskShape.largestTargetFileBytes === 0
      ? "empty"
      : taskShape.largestTargetFileBytes > 16_384
        ? "large"
        : "small",
    budgetFit: outputConstrained && contextConstrained
      ? "output_and_context_constrained"
      : outputConstrained
        ? "output_constrained"
        : contextConstrained
          ? "context_constrained"
          : "fits",
    verifierStrength: taskShape.verifierStrength,
    protectedPathRisk: taskShape.protectedPathRisk,
    staleContextRisk: taskShape.staleContextRisk,
    requiresMultiFileConsistency: taskShape.requiresMultiFileConsistency,
  });
};

const taskShapeBucketMatches = (left: EditTaskShapeBucket, right: EditTaskShapeBucket): boolean =>
  taskShapeBucketId(left) === taskShapeBucketId(right);

const taskShapeBucketId = (bucket: EditTaskShapeBucket): string =>
  [
    bucket.targetFileCount,
    bucket.changedFileCount,
    bucket.fileSize,
    bucket.budgetFit,
    `verify-${bucket.verifierStrength}`,
    `protected-${bucket.protectedPathRisk}`,
    `stale-${bucket.staleContextRisk}`,
    bucket.requiresMultiFileConsistency ? "multi-consistency" : "independent",
  ].join(".");

const protectedPathRiskWeight = (risk: EditRiskLevel): number => {
  switch (risk) {
    case "high":
      return 0.35;
    case "medium":
      return 0.22;
    case "low":
      return 0.1;
  }
};

const appliedButBrokenRiskWeight = (verifierStrength: EditVerifierStrength): number => {
  switch (verifierStrength) {
    case "none":
      return 0.4;
    case "basic":
      return 0.24;
    case "strong":
      return 0.14;
  }
};

const staleContextRiskWeight = (risk: EditRiskLevel): number => {
  switch (risk) {
    case "high":
      return 0.18;
    case "medium":
      return 0.1;
    case "low":
      return 0.04;
  }
};

const weightedAverage = (
  metrics: readonly EditStrategyHistoricalMetric[],
  value: (metric: EditStrategyHistoricalMetric) => number,
): number => {
  const total = metrics.reduce((sum, metric) => sum + metric.sampleCount, 0);
  if (total === 0) {
    return 0;
  }
  return metrics.reduce((sum, metric) => sum + value(metric) * metric.sampleCount, 0) / total;
};

const uniqueEvidence = (
  evidence: readonly EditStrategyRouterDecision["evidenceUsed"][number][],
): EditStrategyRouterDecision["evidenceUsed"] => {
  const byId = new Map<string, EditStrategyRouterDecision["evidenceUsed"][number]>();
  for (const item of evidence) {
    byId.set(`${item.source}:${item.metricId}`, item);
  }
  return [...byId.values()].sort((left, right) => left.metricId.localeCompare(right.metricId));
};

const ratio = (numerator: number, denominator: number): number => denominator === 0 ? 0 : numerator / denominator;
const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

const idPart = (value: string): string => {
  const sanitized = value.replace(/_/gu, "-").replace(/[^A-Za-z0-9._:-]/gu, "-");
  return /^[A-Za-z0-9]/u.test(sanitized) ? sanitized : `id.${sanitized}`;
};
