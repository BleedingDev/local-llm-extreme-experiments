import { z } from "zod";
import type { BagConfig } from "../types";
import { createEvalScorecard } from "../eval-harness/scorer";
import {
  runEvalComparison,
  type EvalRunExecutor,
} from "../eval-harness/runner";
import type {
  ComparisonRunMetadata,
  EvalCase,
  EvalScorecard,
  EvalSplit,
} from "../eval-harness/types";
import { EvalScorecardSchema } from "../eval-harness/types";
import type { EditStrategyAblationReport } from "../eval-harness/edit-strategy-ablation";
import {
  createReplayProposerPromptCases,
  selectReplayCasesForOptimizerInput,
  selectReplayRunResultsForGepaFeedback,
  type ReplayOptimizationSelection,
  type ReplayProposerPromptCase,
} from "../replay/enforcement";
import {
  runReplayEvalComparison,
  type ReplayEvalScenario,
  type ReplayPolicyExecutor,
  type ReplayRunnableCase,
} from "../replay/runner";
import type { ReplayEvalCaseSkeleton } from "../replay/extraction";
import type { TraceAnalysisReport } from "../trace-analysis";
import {
  buildCandidateEvidenceBundle,
  CandidateEvidenceBundleSchema,
  type CandidateEvidenceBundle,
} from "./evidence";
import {
  buildGepaFeedbackBundle,
  GepaFeedbackBundleSchema,
  type BuildGepaFeedbackInput,
  type GepaFeedbackBundle,
} from "./gepa-feedback";
import {
  generateCandidatePatches,
  CandidateGenerationResultSchema,
  type CandidateGenerationResult,
} from "./candidates";
import {
  materializeCandidateArtifacts,
  type CandidateArtifactManifest,
} from "./materialization";
import {
  hashRegistryContent,
  loadActiveOptimizerPointer,
} from "./registry";
import {
  promoteCandidatePatch,
  type CandidatePromotionResult,
} from "./promotion";
import {
  CandidatePatchSchema,
  type CandidatePatch,
  type OptimizerRegistryRecord,
} from "./types";
import {
  validateCandidatePatch,
  CandidateValidationResultSchema,
} from "./validator";

const GEPA_LOOP_SCHEMA_VERSION = "gepa-closed-loop.v1";
const DEFAULT_CREATED_AT = "2026-05-01T00:00:00.000Z";

export const GepaEvaluationThresholdsSchema = z.object({
  maxLatencyMs: z.number().positive().optional(),
  maxTokenCount: z.number().positive().optional(),
}).strict();
export type GepaEvaluationThresholds = z.infer<typeof GepaEvaluationThresholdsSchema>;

export const GepaLoopDiagnosticSchema = z.object({
  severity: z.enum(["info", "warning", "error"]),
  reason: z.string().min(1),
  evalCaseIds: z.array(z.string()).default([]),
  candidatePatchId: z.string().optional(),
}).strict();
export type GepaLoopDiagnostic = z.infer<typeof GepaLoopDiagnosticSchema>;

export const GepaCandidatePreviewSchema = z.object({
  schemaVersion: z.literal(GEPA_LOOP_SCHEMA_VERSION),
  previewId: z.string().min(1),
  candidatePatchId: z.string().min(1),
  createdAt: z.string().min(1),
  validation: CandidateValidationResultSchema,
  baseHashes: z.object({
    expected: z.record(z.string(), z.string()),
    actual: z.record(z.string(), z.string()),
  }).strict(),
  rationale: z.string().min(1),
  affectedPolicyDimensions: z.array(z.string()).default([]),
  rollback: z.object({
    activePointerBeforePromotion: z.unknown().optional(),
    promotionCheckpointPath: z.string().optional(),
    rollbackSupported: z.boolean(),
    metadata: z.record(z.string(), z.string()).default({}),
  }).strict(),
  diagnostics: z.array(GepaLoopDiagnosticSchema).default([]),
  artifactManifest: z.unknown().optional(),
}).strict();
export type GepaCandidatePreview = z.infer<typeof GepaCandidatePreviewSchema> & {
  artifactManifest?: CandidateArtifactManifest | undefined;
};

export type OperatorSafeGepaFeedbackInput = {
  feedbackBundleId?: string;
  traceReports?: readonly TraceAnalysisReport[];
  evidenceBundles?: readonly CandidateEvidenceBundle[];
  replayCases?: readonly ReplayEvalCaseSkeleton[];
  evalRunResults?: BuildGepaFeedbackInput["evalRunResults"];
  evalScorecards?: BuildGepaFeedbackInput["evalScorecards"];
  editAblationReports?: readonly EditStrategyAblationReport[];
  testOutputs?: BuildGepaFeedbackInput["testOutputs"];
  truncationMistakes?: BuildGepaFeedbackInput["truncationMistakes"];
  llmCritiques?: BuildGepaFeedbackInput["llmCritiques"];
  limits?: BuildGepaFeedbackInput["limits"];
  createdAt?: string;
};

export type OperatorSafeGepaFeedbackBundle = {
  feedbackBundle: GepaFeedbackBundle;
  proposerReplayCases: ReplayProposerPromptCase[];
  replaySelection?: ReplayOptimizationSelection | undefined;
  excludedHoldoutEvalCaseIds: string[];
  diagnostics: GepaLoopDiagnostic[];
};

export type MaterializeGepaCandidatePreviewInput = {
  config?: BagConfig;
  cwd?: string;
  candidate: CandidatePatch;
  evidence: CandidateEvidenceBundle;
  records: readonly OptimizerRegistryRecord[];
  expectedBaseHashes?: Readonly<Record<string, string>>;
  actualBaseHashes?: Readonly<Record<string, string>>;
  requiredEvalGateIds?: readonly string[];
  candidateRoot?: string;
  createdAt?: string;
  writeArtifacts?: boolean;
  promotionCheckpointPath?: string;
  rollbackMetadata?: Record<string, string>;
};

export type RunGepaCandidateEvaluationInput = {
  candidate: CandidatePatch;
  baseline: ComparisonRunMetadata;
  candidateMetadata: ComparisonRunMetadata;
  replayCases?: readonly (ReplayRunnableCase | ReplayEvalCaseSkeleton | ReplayEvalScenario)[];
  baselineReplayPolicy?: ReplayPolicyExecutor;
  candidateReplayPolicy?: ReplayPolicyExecutor;
  curatedEvalCases?: readonly EvalCase[];
  baselineCuratedExecutor?: EvalRunExecutor;
  candidateCuratedExecutor?: EvalRunExecutor;
  includeHoldoutFinal?: boolean;
  evalSuiteId?: string;
  timeoutMs?: number;
  baseDir?: string;
  createdAt?: string;
  thresholds?: Partial<GepaEvaluationThresholds>;
  signal?: AbortSignal;
};

export type GepaEvaluationGate = {
  gateId: "train-dev-visible" | "hidden-holdout-final" | "critical-regression-veto" | "latency-cost-veto";
  passed: boolean;
  blocking: boolean;
  message: string;
  scorecardIds: string[];
};

export type GepaCandidateEvaluationResult = {
  visibleScorecards: EvalScorecard[];
  holdoutScorecards: EvalScorecard[];
  allScorecards: EvalScorecard[];
  gates: GepaEvaluationGate[];
  passed: boolean;
  promotionScorecard?: EvalScorecard | undefined;
};

export type EvaluateGepaPromotionGatesInput = {
  visibleScorecards: readonly EvalScorecard[];
  holdoutScorecards: readonly EvalScorecard[];
  includeHoldoutFinal?: boolean;
  thresholds?: Partial<GepaEvaluationThresholds>;
};

export type PromoteGepaCandidateInput = {
  config: BagConfig;
  cwd?: string;
  candidate: CandidatePatch;
  validation: z.infer<typeof CandidateValidationResultSchema>;
  evaluation: GepaCandidateEvaluationResult;
  decidedAt?: string;
  decisionId?: string;
};

export const buildOperatorSafeGepaFeedbackBundle = (
  input: OperatorSafeGepaFeedbackInput,
): OperatorSafeGepaFeedbackBundle => {
  const createdAt = input.createdAt ?? DEFAULT_CREATED_AT;
  const replayCases = input.replayCases ?? [];
  const replaySelection = replayCases.length === 0
    ? undefined
    : selectReplayCasesForOptimizerInput(replayCases, "proposer_prompt");
  const proposerReplayCases = replayCases.length === 0 ? [] : createReplayProposerPromptCases(replayCases);
  const traceEvidenceBundles = (input.traceReports ?? []).map((report, index) =>
    buildCandidateEvidenceBundle({
      evidenceBundleId: stableId("evidence", "trace-report", String(index)),
      createdAt,
      traceFailures: report.failureClusters,
      traceLatencies: report.latencyClusters,
    })
  );
  const visibleEvalRunResults = replayCases.length === 0
    ? (input.evalRunResults ?? []).filter((run) => run.split !== "holdout")
    : selectReplayRunResultsForGepaFeedback(input.evalRunResults ?? [], replayCases);
  const visibleEvalScorecards = (input.evalScorecards ?? []).filter((scorecard) => scorecard.split !== "holdout");
  const visibleAblations = (input.editAblationReports ?? []).filter((report) =>
    report.optimizationAllowed && !report.hiddenHoldoutUsed
  );
  const excludedHoldoutEvalCaseIds = uniqueSorted([
    ...(input.evalRunResults ?? []).filter((run) => run.split === "holdout").map((run) => run.evalCaseId),
    ...(input.evalScorecards ?? []).filter((scorecard) => scorecard.split === "holdout")
      .flatMap((scorecard) => scorecard.runResults.map((run) => run.evalCaseId)),
    ...(input.editAblationReports ?? []).flatMap((report) => report.hiddenHoldoutEvalCaseIds),
    ...(replaySelection?.hiddenHoldoutEvalCaseIds ?? []),
  ]);

  const feedbackBundle = buildGepaFeedbackBundle({
    ...(input.feedbackBundleId === undefined ? {} : { feedbackBundleId: input.feedbackBundleId }),
    evalRunResults: visibleEvalRunResults,
    evalScorecards: visibleEvalScorecards,
    evidenceBundles: [
      ...traceEvidenceBundles,
      ...(input.evidenceBundles ?? []).map((bundle) => CandidateEvidenceBundleSchema.parse(bundle)),
    ],
    editAblationReports: visibleAblations,
    ...(input.testOutputs === undefined ? {} : { testOutputs: input.testOutputs }),
    ...(input.truncationMistakes === undefined ? {} : { truncationMistakes: input.truncationMistakes }),
    ...(input.llmCritiques === undefined ? {} : { llmCritiques: input.llmCritiques }),
    ...(input.limits === undefined ? {} : { limits: input.limits }),
  });

  return {
    feedbackBundle: GepaFeedbackBundleSchema.parse(feedbackBundle),
    proposerReplayCases,
    replaySelection,
    excludedHoldoutEvalCaseIds,
    diagnostics: [
      ...(excludedHoldoutEvalCaseIds.length === 0
        ? []
        : [{
            severity: "info" as const,
            reason: "hidden holdout evidence was excluded from GEPA proposer input",
            evalCaseIds: excludedHoldoutEvalCaseIds,
          }]),
      ...(replaySelection?.rejectedCases ?? []).map((rejection) => ({
        severity: "warning" as const,
        reason: `replay case rejected for proposer input: ${rejection.reasons.join("; ")}`,
        evalCaseIds: [rejection.evalCaseId],
      })),
    ],
  };
};

export const proposeDeterministicGepaCandidates = (
  input: {
    evidence: CandidateEvidenceBundle;
    createdAt?: string;
    maxCandidates?: number;
  },
): CandidateGenerationResult =>
  CandidateGenerationResultSchema.parse(generateCandidatePatches(input));

export const materializeGepaCandidatePreview = (
  input: MaterializeGepaCandidatePreviewInput,
): GepaCandidatePreview => {
  const candidate = CandidatePatchSchema.parse(input.candidate);
  const evidence = CandidateEvidenceBundleSchema.parse(input.evidence);
  const expectedBaseHashes = input.expectedBaseHashes ?? baseHashesFromRecords(input.records);
  const actualBaseHashes = input.actualBaseHashes ?? expectedBaseHashes;
  const validation = validateCandidatePatch({
    candidate,
    records: input.records,
    expectedBaseHashes,
    actualBaseHashes,
    ...(input.requiredEvalGateIds === undefined ? {} : { requiredEvalGateIds: input.requiredEvalGateIds }),
  });
  const activePointerBeforePromotion = input.config == null
    ? undefined
    : loadActiveOptimizerPointer(input.config, input.cwd).pointer;
  const preview = GepaCandidatePreviewSchema.parse({
    schemaVersion: GEPA_LOOP_SCHEMA_VERSION,
    previewId: stableId("gepa-preview", candidate.candidatePatchId),
    candidatePatchId: candidate.candidatePatchId,
    createdAt: input.createdAt ?? DEFAULT_CREATED_AT,
    validation,
    baseHashes: {
      expected: expectedBaseHashes,
      actual: actualBaseHashes,
    },
    rationale: candidate.rationale,
    affectedPolicyDimensions: affectedPolicyDimensions(candidate),
    rollback: {
      ...(activePointerBeforePromotion === undefined ? {} : { activePointerBeforePromotion }),
      ...(input.promotionCheckpointPath === undefined ? {} : { promotionCheckpointPath: input.promotionCheckpointPath }),
      rollbackSupported: input.config != null,
      metadata: input.rollbackMetadata ?? {},
    },
    diagnostics: validation.issues.map((issue) => ({
      severity: issue.severity === "error" ? "error" : "warning",
      reason: `${issue.code}: ${issue.message}`,
      candidatePatchId: candidate.candidatePatchId,
    })),
  }) as GepaCandidatePreview;

  if (input.writeArtifacts === false) {
    return preview;
  }

  const manifest = materializeCandidateArtifacts({
    ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
    ...(input.candidateRoot === undefined ? {} : { candidateRoot: input.candidateRoot }),
    candidate,
    evidence,
    validation,
    createdAt: preview.createdAt,
    reportMarkdown: renderPreviewReport(preview, candidate),
  });
  return GepaCandidatePreviewSchema.parse({
    ...preview,
    artifactManifest: manifest,
  }) as GepaCandidatePreview;
};

export const runGepaCandidateEvaluation = async (
  input: RunGepaCandidateEvaluationInput,
): Promise<GepaCandidateEvaluationResult> => {
  const createdAt = input.createdAt ?? DEFAULT_CREATED_AT;
  const visibleScorecards: EvalScorecard[] = [];
  const holdoutScorecards: EvalScorecard[] = [];

  const visibleReplay = await runReplayEvalComparison({
    ...(input.replayCases === undefined ? {} : { replayCases: input.replayCases }),
    includeHoldout: false,
    baseline: input.baseline,
    candidate: input.candidateMetadata,
    ...(input.baselineReplayPolicy === undefined ? {} : { baselinePolicy: input.baselineReplayPolicy }),
    ...(input.candidateReplayPolicy === undefined ? {} : { candidatePolicy: input.candidateReplayPolicy }),
    candidatePatchId: input.candidate.candidatePatchId,
    evalSuiteId: input.evalSuiteId ?? "suite.gepa.replay",
    scorecardIdPrefix: stableId("scorecard", input.candidate.candidatePatchId, "replay"),
    ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
    ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
    ...(input.signal === undefined ? {} : { signal: input.signal }),
    createdAt,
  });
  try {
    visibleScorecards.push(...visibleReplay.scorecards);
  } finally {
    await visibleReplay.cleanup();
  }

  visibleScorecards.push(...await runCuratedComparisons({
    candidate: input.candidate,
    evalCases: (input.curatedEvalCases ?? []).filter((evalCase) => evalCase.split !== "holdout"),
    baseline: input.baseline,
    candidateMetadata: input.candidateMetadata,
    ...(input.baselineCuratedExecutor === undefined ? {} : { baselineExecutor: input.baselineCuratedExecutor }),
    ...(input.candidateCuratedExecutor === undefined ? {} : { candidateExecutor: input.candidateCuratedExecutor }),
    evalSuiteId: input.evalSuiteId ?? "suite.gepa.curated",
    scorecardIdPrefix: stableId("scorecard", input.candidate.candidatePatchId, "curated"),
    ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
    ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
    ...(input.signal === undefined ? {} : { signal: input.signal }),
    createdAt,
  }));

  const visibleGate = trainDevGate(visibleScorecards);
  if (input.includeHoldoutFinal === true && visibleGate.passed) {
    const holdoutReplayCases = (input.replayCases ?? []).filter((replayCase) => splitForReplayInput(replayCase) === "holdout");
    if (holdoutReplayCases.length > 0) {
      const holdoutReplay = await runReplayEvalComparison({
        replayCases: holdoutReplayCases,
        includeHoldout: true,
        baseline: input.baseline,
        candidate: input.candidateMetadata,
        ...(input.baselineReplayPolicy === undefined ? {} : { baselinePolicy: input.baselineReplayPolicy }),
        ...(input.candidateReplayPolicy === undefined ? {} : { candidatePolicy: input.candidateReplayPolicy }),
        candidatePatchId: input.candidate.candidatePatchId,
        evalSuiteId: input.evalSuiteId ?? "suite.gepa.replay",
        scorecardIdPrefix: stableId("scorecard", input.candidate.candidatePatchId, "replay-holdout"),
        ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
        ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
        ...(input.signal === undefined ? {} : { signal: input.signal }),
        createdAt,
      });
      try {
        holdoutScorecards.push(...holdoutReplay.scorecards.filter((scorecard) => scorecard.split === "holdout"));
      } finally {
        await holdoutReplay.cleanup();
      }
    }
    holdoutScorecards.push(...await runCuratedComparisons({
      candidate: input.candidate,
      evalCases: (input.curatedEvalCases ?? []).filter((evalCase) => evalCase.split === "holdout"),
      baseline: input.baseline,
      candidateMetadata: input.candidateMetadata,
      ...(input.baselineCuratedExecutor === undefined ? {} : { baselineExecutor: input.baselineCuratedExecutor }),
      ...(input.candidateCuratedExecutor === undefined ? {} : { candidateExecutor: input.candidateCuratedExecutor }),
      evalSuiteId: input.evalSuiteId ?? "suite.gepa.curated",
      scorecardIdPrefix: stableId("scorecard", input.candidate.candidatePatchId, "curated-holdout"),
      ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
      ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
      ...(input.signal === undefined ? {} : { signal: input.signal }),
      createdAt,
    }));
  }

  return evaluateGepaPromotionGates({
    visibleScorecards,
    holdoutScorecards,
    ...(input.includeHoldoutFinal === undefined ? {} : { includeHoldoutFinal: input.includeHoldoutFinal }),
    ...(input.thresholds === undefined ? {} : { thresholds: input.thresholds }),
  });
};

export const evaluateGepaPromotionGates = (
  input: EvaluateGepaPromotionGatesInput,
): GepaCandidateEvaluationResult => {
  const visibleScorecards = input.visibleScorecards.map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const holdoutScorecards = input.holdoutScorecards.map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const thresholds = GepaEvaluationThresholdsSchema.parse(input.thresholds ?? {});
  const allScorecards = [...visibleScorecards, ...holdoutScorecards];
  const gates = [
    trainDevGate(visibleScorecards),
    hiddenHoldoutGate(holdoutScorecards, input.includeHoldoutFinal === true),
    criticalRegressionGate(allScorecards),
    latencyCostGate(allScorecards, thresholds),
  ];
  return {
    visibleScorecards,
    holdoutScorecards,
    allScorecards,
    gates,
    passed: gates.every((gate) => gate.passed || !gate.blocking),
    promotionScorecard: holdoutScorecards.at(-1) ?? visibleScorecards.at(-1),
  };
};

export const promoteGepaCandidate = (
  input: PromoteGepaCandidateInput,
): CandidatePromotionResult => {
  const candidate = CandidatePatchSchema.parse(input.candidate);
  const validation = CandidateValidationResultSchema.parse(input.validation);
  const candidateEval = input.evaluation.promotionScorecard ?? input.evaluation.allScorecards.at(-1);
  if (candidateEval == null) {
    throw new Error("GEPA promotion requires at least one candidate eval scorecard");
  }
  const failedGateMessages = input.evaluation.gates
    .filter((gate) => gate.blocking && !gate.passed)
    .map((gate) => `${gate.gateId}: ${gate.message}`);

  return promoteCandidatePatch({
    config: input.config,
    ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
    candidate,
    validation,
    candidateEval,
    promotionGatePassed: input.evaluation.passed,
    ...(failedGateMessages.length === 0 ? {} : { promotionGateReason: failedGateMessages.join("; ") }),
    ...(input.decidedAt === undefined ? {} : { decidedAt: input.decidedAt }),
    ...(input.decisionId === undefined ? {} : { decisionId: input.decisionId }),
  });
};

const runCuratedComparisons = async (input: {
  candidate: CandidatePatch;
  evalCases: readonly EvalCase[];
  baseline: ComparisonRunMetadata;
  candidateMetadata: ComparisonRunMetadata;
  baselineExecutor?: EvalRunExecutor;
  candidateExecutor?: EvalRunExecutor;
  evalSuiteId: string;
  scorecardIdPrefix: string;
  timeoutMs?: number;
  baseDir?: string;
  signal?: AbortSignal;
  createdAt: string;
}): Promise<EvalScorecard[]> => {
  const scorecards: EvalScorecard[] = [];
  const bySplit = new Map<EvalSplit, Awaited<ReturnType<typeof runEvalComparison>>[]>();

  for (const evalCase of input.evalCases) {
    const execution = await runEvalComparison({
      evalCase,
      context: input.candidateMetadata.context,
      baselineComparisonRunId: input.baseline.comparisonRunId,
      candidateComparisonRunId: input.candidateMetadata.comparisonRunId,
      ...(input.baselineExecutor === undefined ? {} : { baselineExecutor: input.baselineExecutor }),
      candidateExecutor: input.candidateExecutor ?? (async () => undefined),
      candidatePatchId: input.candidate.candidatePatchId,
      ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
      ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
    try {
      bySplit.set(evalCase.split, [...(bySplit.get(evalCase.split) ?? []), execution]);
    } finally {
      await execution.cleanup();
    }
  }

  for (const [split, executions] of [...bySplit.entries()].sort((left, right) => splitOrder(left[0]) - splitOrder(right[0]))) {
    scorecards.push(createEvalScorecard({
      scorecardId: `${input.scorecardIdPrefix}.${split}`,
      evalSuiteId: input.evalSuiteId,
      split,
      baseline: input.baseline,
      candidate: input.candidateMetadata,
      baselineResults: executions.map((execution) => execution.baseline.result),
      candidateResults: executions.map((execution) => execution.candidate.result),
      createdAt: input.createdAt,
    }));
  }

  return scorecards;
};

const trainDevGate = (scorecards: readonly EvalScorecard[]): GepaEvaluationGate => {
  const visible = scorecards.filter((scorecard) => scorecard.split !== "holdout");
  const splits = new Set(visible.map((scorecard) => scorecard.split));
  const missing = ["train", "dev"].filter((split) => !splits.has(split as EvalSplit));
  const failing = visible.filter((scorecard) => !scorecard.passed || scorecard.criticalRegressionVeto.vetoed);
  const passed = missing.length === 0 && failing.length === 0;
  return {
    gateId: "train-dev-visible",
    passed,
    blocking: true,
    message: passed
      ? "Visible train/dev scorecards passed."
      : `Visible train/dev gate failed${missing.length > 0 ? `; missing splits: ${missing.join(", ")}` : ""}${failing.length > 0 ? `; failing scorecards: ${failing.map((scorecard) => scorecard.scorecardId).join(", ")}` : ""}.`,
    scorecardIds: visible.map((scorecard) => scorecard.scorecardId),
  };
};

const hiddenHoldoutGate = (
  scorecards: readonly EvalScorecard[],
  required: boolean,
): GepaEvaluationGate => {
  const failing = scorecards.filter((scorecard) => !scorecard.passed || scorecard.criticalRegressionVeto.vetoed);
  const passed = !required || (scorecards.length > 0 && failing.length === 0);
  return {
    gateId: "hidden-holdout-final",
    passed,
    blocking: required,
    message: passed
      ? "Hidden holdout final check passed or was not requested."
      : scorecards.length === 0
        ? "Hidden holdout final check failed; no holdout scorecard was produced."
        : `Hidden holdout final check failed: ${failing.map((scorecard) => scorecard.scorecardId).join(", ")}.`,
    scorecardIds: scorecards.map((scorecard) => scorecard.scorecardId),
  };
};

const criticalRegressionGate = (scorecards: readonly EvalScorecard[]): GepaEvaluationGate => {
  const vetoed = scorecards.filter((scorecard) => scorecard.criticalRegressionVeto.vetoed);
  return {
    gateId: "critical-regression-veto",
    passed: vetoed.length === 0,
    blocking: true,
    message: vetoed.length === 0
      ? "No critical regression vetoes were raised."
      : `Critical regression vetoes raised by: ${vetoed.map((scorecard) => scorecard.scorecardId).join(", ")}.`,
    scorecardIds: vetoed.map((scorecard) => scorecard.scorecardId),
  };
};

const latencyCostGate = (
  scorecards: readonly EvalScorecard[],
  thresholds: GepaEvaluationThresholds,
): GepaEvaluationGate => {
  const metrics = scorecards.flatMap((scorecard) => scorecard.objectiveMetrics);
  const latencyOffenders = thresholds.maxLatencyMs === undefined
    ? []
    : metrics.filter((metric) => metric.unit === "ms" && metric.value > thresholds.maxLatencyMs!);
  const tokenOffenders = thresholds.maxTokenCount === undefined
    ? []
    : metrics.filter((metric) => metric.unit === "tokens" && metric.value > thresholds.maxTokenCount!);
  const offenders = [...latencyOffenders, ...tokenOffenders];
  return {
    gateId: "latency-cost-veto",
    passed: offenders.length === 0,
    blocking: true,
    message: offenders.length === 0
      ? "Latency and token-cost vetoes passed."
      : `Latency/token-cost veto failed: ${offenders.map((metric) => `${metric.metricId}=${metric.value}${metric.unit}`).join(", ")}.`,
    scorecardIds: scorecards.map((scorecard) => scorecard.scorecardId),
  };
};

const baseHashesFromRecords = (
  records: readonly OptimizerRegistryRecord[],
): Record<string, string> => {
  const entries: [string, string][] = records.map((record) => [
    recordPayloadId(record),
    record.contentHash ?? hashRegistryContent(record.payload),
  ]);
  return Object.fromEntries(entries.sort((left, right) => left[0].localeCompare(right[0])));
};

const recordPayloadId = (record: OptimizerRegistryRecord): string => {
  switch (record.recordKind) {
    case "model_profile":
      return record.payload.modelProfileId;
    case "codebase_profile":
      return record.payload.codebaseProfileId;
    case "model_codebase_policy":
      return record.payload.policyId;
    case "canonical_tool_spec":
      return record.payload.canonicalToolId;
    case "rendered_tool_contract":
      return record.payload.renderedToolId;
    case "candidate_patch":
      return record.payload.candidatePatchId;
    case "eval_result":
      return record.payload.evalResultId;
    case "promotion_decision":
      return record.payload.promotionDecisionId;
  }
};

const affectedPolicyDimensions = (candidate: CandidatePatch): string[] =>
  uniqueSorted(candidate.operations.map((operation) => operation.path.split("/")[1] ?? operation.path));

const renderPreviewReport = (preview: GepaCandidatePreview, candidate: CandidatePatch): string =>
  [
    `# GEPA Candidate Preview ${candidate.candidatePatchId}`,
    "",
    `Validation: ${preview.validation.valid ? "valid" : "invalid"}`,
    `Scope: ${candidate.scope.artifactKind} ${candidate.scope.artifactId}`,
    `Affected dimensions: ${preview.affectedPolicyDimensions.join(", ") || "none"}`,
    "",
    "## Base Hashes",
    "",
    ...Object.entries(preview.baseHashes.expected).map(([artifactId, hash]) => `- ${artifactId}: ${hash}`),
    "",
    "## Rationale",
    "",
    candidate.rationale,
    "",
    "## Rollback",
    "",
    `Rollback supported: ${preview.rollback.rollbackSupported ? "yes" : "no"}`,
    preview.rollback.promotionCheckpointPath == null ? "" : `Checkpoint: ${preview.rollback.promotionCheckpointPath}`,
    "",
  ].filter((line) => line !== "").join("\n");

const splitForReplayInput = (
  replayCase: ReplayRunnableCase | ReplayEvalCaseSkeleton | ReplayEvalScenario,
): EvalSplit => {
  if ("replayCase" in replayCase) {
    return replayCase.replayCase.split;
  }
  if ("capture" in replayCase && "metadata" in replayCase) {
    return replayCase.metadata.split ?? "dev";
  }
  return replayCase.split;
};

const splitOrder = (split: EvalSplit): number => {
  switch (split) {
    case "train":
      return 0;
    case "dev":
      return 1;
    case "holdout":
      return 2;
  }
};

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values.filter((value) => value.length > 0))].sort((left, right) => left.localeCompare(right));

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 180) || "gepa-loop.empty";
