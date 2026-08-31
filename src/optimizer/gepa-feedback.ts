import { z } from "zod";
import type { EditStrategyAblationReport, EditStrategyAblationProbeResult } from "../eval-harness/edit-strategy-ablation";
import type { EvalRunResult, EvalScorecard, ObjectiveMetric } from "../eval-harness/types";
import type { CandidateEvidenceBundle, CandidateEvidenceObservation } from "./evidence";
import { OptimizerIdSchema, OptimizerVersionSchema } from "./types";

const DEFAULT_MAX_RECORDS = 32;
const DEFAULT_MAX_TEXT_CHARS = 1_200;
const MAX_RECORDS = 200;
const MAX_TEXT_CHARS = 8_000;
const REDACTION = "[REDACTED_SECRET]";

export const GepaFeedbackRecordSchema = z.object({
  feedbackId: OptimizerIdSchema,
  source: z.enum(["eval_run", "eval_scorecard", "trace_evidence", "edit_ablation", "test_output", "truncation", "llm_critique"]),
  severity: z.enum(["info", "warning", "failure", "critical"]),
  objective: z.string().min(1),
  feedback: z.string().min(1),
  modelProfileId: OptimizerIdSchema.optional(),
  codebaseProfileId: OptimizerIdSchema.optional(),
  policyId: OptimizerIdSchema.optional(),
  traceIds: z.array(z.string()).default([]),
  spanIds: z.array(z.string()).default([]),
  evalCaseIds: z.array(OptimizerIdSchema).default([]),
  runResultIds: z.array(OptimizerIdSchema).default([]),
  scorecardIds: z.array(OptimizerIdSchema).default([]),
  metricIds: z.array(OptimizerIdSchema).default([]),
  editStrategyVersions: z.array(OptimizerVersionSchema).default([]),
  renderedEditContractVersions: z.array(OptimizerVersionSchema).default([]),
  editFallbackPolicyVersions: z.array(OptimizerVersionSchema).default([]),
  editRepairPolicyVersions: z.array(OptimizerVersionSchema).default([]),
  editVerifierPolicyVersions: z.array(OptimizerVersionSchema).default([]),
  editObjectiveSetIds: z.array(OptimizerIdSchema).default([]),
  editStrategyIds: z.array(OptimizerIdSchema).default([]),
  editStrategyFamilies: z.array(OptimizerIdSchema).default([]),
  canonicalEditToolSpecIds: z.array(OptimizerIdSchema).default([]),
  renderedEditToolContractIds: z.array(OptimizerIdSchema).default([]),
  higherIsBetter: z.boolean().optional(),
  redacted: z.boolean().default(false),
  truncated: z.boolean().default(false),
}).strict();
export type GepaFeedbackRecord = z.infer<typeof GepaFeedbackRecordSchema>;

export const GepaFeedbackBundleSchema = z.object({
  feedbackBundleId: OptimizerIdSchema,
  schemaVersion: z.literal("gepa-feedback.v1").default("gepa-feedback.v1"),
  records: z.array(GepaFeedbackRecordSchema),
  redactionCount: z.number().int().nonnegative().default(0),
  limits: z.object({
    maxRecords: z.number().int().positive(),
    maxTextChars: z.number().int().positive(),
  }).strict(),
}).strict();
export type GepaFeedbackBundle = z.infer<typeof GepaFeedbackBundleSchema>;

export type GepaTextFeedbackInput = {
  id: string;
  text: string;
  modelProfileId?: string;
  codebaseProfileId?: string;
  policyId?: string;
  traceIds?: readonly string[];
  spanIds?: readonly string[];
  evalCaseIds?: readonly string[];
};

export type BuildGepaFeedbackInput = {
  feedbackBundleId?: string;
  evalRunResults?: readonly EvalRunResult[];
  evalScorecards?: readonly EvalScorecard[];
  evidenceBundles?: readonly CandidateEvidenceBundle[];
  editAblationReports?: readonly EditStrategyAblationReport[];
  testOutputs?: readonly GepaTextFeedbackInput[];
  truncationMistakes?: readonly GepaTextFeedbackInput[];
  llmCritiques?: readonly GepaTextFeedbackInput[];
  limits?: Partial<GepaFeedbackBundle["limits"]>;
};

type Sanitized = {
  text: string;
  redacted: boolean;
  truncated: boolean;
};

export const buildGepaFeedbackBundle = (input: BuildGepaFeedbackInput): GepaFeedbackBundle => {
  const limits = resolveLimits(input.limits);
  const records = [
    ...(input.evalRunResults ?? []).flatMap((run) => recordFromEvalRun(run, limits)),
    ...(input.evalScorecards ?? []).flatMap((scorecard) => recordFromScorecard(scorecard, limits)),
    ...(input.evidenceBundles ?? []).flatMap((bundle) => recordsFromEvidenceBundle(bundle, limits)),
    ...(input.editAblationReports ?? []).flatMap((report) => recordsFromEditAblationReport(report, limits)),
    ...(input.testOutputs ?? []).map((feedback) => recordFromText("test_output", feedback, limits)),
    ...(input.truncationMistakes ?? []).map((feedback) => recordFromText("truncation", feedback, limits)),
    ...(input.llmCritiques ?? []).map((feedback) => recordFromText("llm_critique", feedback, limits)),
  ]
    .sort(compareFeedback)
    .slice(0, limits.maxRecords);

  return GepaFeedbackBundleSchema.parse({
    feedbackBundleId: input.feedbackBundleId ?? stableId("gepa-feedback", records.map((record) => record.feedbackId).join(".")),
    schemaVersion: "gepa-feedback.v1",
    records,
    redactionCount: records.filter((record) => record.redacted).length,
    limits,
  });
};

const recordFromEvalRun = (
  run: EvalRunResult,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord[] => {
  const failures = run.assertionResults.filter((assertion) => !assertion.passed);
  const assertionRecords = failures.map((assertion) => {
    const sanitized = sanitizeText(
      [
        `Eval case ${run.evalCaseId} ${run.status}.`,
        `Assertion ${assertion.assertionId} (${assertion.assertionKind}) failed.`,
        assertion.message ?? "",
        assertion.actual == null ? "" : `Actual: ${stringify(assertion.actual)}`,
      ].filter(Boolean).join("\n"),
      limits.maxTextChars,
    );
    return GepaFeedbackRecordSchema.parse({
      feedbackId: stableId("gepa", "eval-run", run.runResultId, assertion.assertionId),
      source: "eval_run",
      severity: assertion.severity === "critical" ? "critical" : "failure",
      objective: "Improve the candidate so deterministic eval assertions pass without causing regressions.",
      feedback: sanitized.text,
      modelProfileId: run.context.modelProfileId,
      codebaseProfileId: run.context.codebaseProfileId,
      policyId: run.context.policyId,
      evalCaseIds: [run.evalCaseId],
      runResultIds: [run.runResultId],
      redacted: sanitized.redacted,
      truncated: sanitized.truncated,
    });
  });

  const metricRecords = run.objectiveMetrics
    .filter((metric) => metric.delta != null && metric.delta < 0)
    .map((metric) => recordFromMetric(run, metric, limits));

  return [...assertionRecords, ...metricRecords];
};

const recordFromMetric = (
  run: EvalRunResult,
  metric: ObjectiveMetric,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord => {
  const sanitized = sanitizeText(
    `Metric ${metric.metricId} regressed. value=${metric.value}, delta=${metric.delta}, higherIsBetter=${metric.higherIsBetter}.`,
    limits.maxTextChars,
  );
  return GepaFeedbackRecordSchema.parse({
    feedbackId: stableId("gepa", "eval-metric", run.runResultId, metric.metricId),
    source: "eval_run",
    severity: "warning",
    objective: "Improve the objective metric in the correct direction while preserving correctness.",
    feedback: sanitized.text,
    modelProfileId: run.context.modelProfileId,
    codebaseProfileId: run.context.codebaseProfileId,
    policyId: run.context.policyId,
    evalCaseIds: [run.evalCaseId],
    runResultIds: [run.runResultId],
    metricIds: [metric.metricId],
    higherIsBetter: metric.higherIsBetter,
    redacted: sanitized.redacted,
    truncated: sanitized.truncated,
  });
};

const recordFromScorecard = (
  scorecard: EvalScorecard,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord[] =>
  scorecard.criticalRegressionVeto.regressions.map((regression) => {
    const sanitized = sanitizeText(`Critical regression ${regression.regressionId}: ${regression.reason}`, limits.maxTextChars);
    return GepaFeedbackRecordSchema.parse({
      feedbackId: stableId("gepa", "scorecard", scorecard.scorecardId, regression.regressionId),
      source: "eval_scorecard",
      severity: "critical",
      objective: "Eliminate critical regressions before a candidate can be promoted.",
      feedback: sanitized.text,
      modelProfileId: scorecard.candidate.context.modelProfileId,
      codebaseProfileId: scorecard.candidate.context.codebaseProfileId,
      policyId: scorecard.candidate.context.policyId,
      evalCaseIds: [regression.evalCaseId],
      scorecardIds: [scorecard.scorecardId],
      metricIds: regression.metricId == null ? [] : [regression.metricId],
      redacted: sanitized.redacted,
      truncated: sanitized.truncated,
    });
  });

const recordsFromEvidenceBundle = (
  bundle: CandidateEvidenceBundle,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord[] =>
  bundle.observations.map((observation) => recordFromEvidenceObservation(bundle.evidenceBundleId, observation, limits));

const recordsFromEditAblationReport = (
  report: EditStrategyAblationReport,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord[] => {
  if (!report.optimizationAllowed) {
    return [];
  }
  const candidateRuns = new Map(report.candidateResults.map((run) => [run.runResultId, run]));
  return report.probeResults
    .filter((probe) => shouldEmitEditAblationFeedback(probe))
    .map((probe) => recordFromEditAblationProbe(report, probe, candidateRuns.get(probe.evalRunResultId), limits));
};

const shouldEmitEditAblationFeedback = (probe: EditStrategyAblationProbeResult): boolean =>
  probe.status !== "passed" ||
  !probe.expectedOutcomeMatched ||
  probe.protectedPathTouched ||
  probe.postApplyConsistencyStatus === "inconsistent" ||
  probe.verificationStatus === "failed" ||
  probe.verificationStatus === "error" ||
  probe.selfDetectedRegressionStatus === "confirmed";

const recordFromEditAblationProbe = (
  report: EditStrategyAblationReport,
  probe: EditStrategyAblationProbeResult,
  run: EvalRunResult | undefined,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord => {
  const sanitized = sanitizeText(
    [
      `Edit ablation ${report.ablationRunId} found ${probe.strategyFamily} ${probe.status} on ${probe.editEvalCaseId}.`,
      `parse=${probe.parseStatus}, apply=${probe.applyStatus}, expectedOutcomeMatched=${probe.expectedOutcomeMatched}`,
      probe.errorCode == null ? "" : `errorCode=${probe.errorCode}`,
      `postApplyConsistency=${probe.postApplyConsistencyStatus}, verification=${probe.verificationStatus}, selfDetected=${probe.selfDetectedRegressionStatus}`,
      `protectedPathTouched=${probe.protectedPathTouched}, changedFiles=${probe.changedFiles.join(", ") || "none"}`,
    ].filter(Boolean).join("\n"),
    limits.maxTextChars,
  );
  const severity = probe.protectedPathTouched ||
    probe.postApplyConsistencyStatus === "inconsistent" ||
    probe.verificationStatus === "failed" ||
    probe.selfDetectedRegressionStatus === "confirmed"
    ? "critical"
    : probe.status === "failed" ? "failure" : "warning";

  return GepaFeedbackRecordSchema.parse({
    feedbackId: stableId("gepa", "edit-ablation", report.ablationRunId, probe.probeResultId),
    source: "edit_ablation",
    severity,
    objective:
      "Optimize edit strategy policy, rendered edit contracts, fallback order, repair policy, rollback behavior, and verifier enforcement without changing runtime source code.",
    feedback: sanitized.text,
    modelProfileId: probe.modelProfileId,
    codebaseProfileId: probe.codebaseProfileId,
    ...(run?.context.policyId == null ? {} : { policyId: run.context.policyId }),
    evalCaseIds: [probe.editEvalCaseId],
    runResultIds: [probe.evalRunResultId],
    metricIds: probe.objectiveMetrics.map((metric) => metric.metricId),
    editStrategyFamilies: [probe.strategyFamily],
    redacted: sanitized.redacted,
    truncated: sanitized.truncated,
  });
};

const recordFromEvidenceObservation = (
  bundleId: string,
  observation: CandidateEvidenceObservation,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord => {
  const sanitized = sanitizeText(
    [
      observation.title,
      ...observation.excerpts.map((excerpt) => excerpt.text),
      observation.inputHashes.length > 0 ? `Input hashes: ${observation.inputHashes.join(", ")}` : "",
      observation.argumentHashes.length > 0 ? `Argument hashes: ${observation.argumentHashes.join(", ")}` : "",
    ].filter(Boolean).join("\n"),
    limits.maxTextChars,
  );
  return GepaFeedbackRecordSchema.parse({
    feedbackId: stableId("gepa", "evidence", bundleId, observation.observationId),
    source: "trace_evidence",
    severity: observation.severity === "critical" ? "critical" : observation.severity === "high" ? "failure" : "warning",
    objective: "Use trace evidence to revise prompts, tool guidance, or policy without changing runtime source code.",
    feedback: sanitized.text,
    modelProfileId: singleValue(observation.lineage.modelProfileIds),
    codebaseProfileId: singleValue(observation.lineage.codebaseProfileIds),
    policyId: singleValue(observation.lineage.policyIds),
    traceIds: observation.traceIds,
    spanIds: observation.spanIds,
    evalCaseIds: observation.evalCaseIds,
    runResultIds: observation.runResultIds,
    scorecardIds: observation.scorecardIds,
    editStrategyVersions: observation.lineage.editStrategyVersions,
    renderedEditContractVersions: observation.lineage.renderedEditContractVersions,
    editFallbackPolicyVersions: observation.lineage.editFallbackPolicyVersions,
    editRepairPolicyVersions: observation.lineage.editRepairPolicyVersions,
    editVerifierPolicyVersions: observation.lineage.editVerifierPolicyVersions,
    editObjectiveSetIds: observation.lineage.editObjectiveSetIds,
    editStrategyIds: observation.lineage.editStrategyIds,
    editStrategyFamilies: observation.lineage.editStrategyFamilies,
    canonicalEditToolSpecIds: observation.lineage.canonicalEditToolSpecIds,
    renderedEditToolContractIds: observation.lineage.renderedEditToolContractIds,
    redacted: sanitized.redacted || observation.excerpts.some((excerpt) => excerpt.redacted),
    truncated: sanitized.truncated || observation.excerpts.some((excerpt) => excerpt.truncated),
  });
};

const recordFromText = (
  source: "test_output" | "truncation" | "llm_critique",
  input: GepaTextFeedbackInput,
  limits: GepaFeedbackBundle["limits"],
): GepaFeedbackRecord => {
  const sanitized = sanitizeText(input.text, limits.maxTextChars);
  return GepaFeedbackRecordSchema.parse({
    feedbackId: stableId("gepa", source, input.id),
    source,
    severity: source === "llm_critique" ? "warning" : "failure",
    objective: sourceObjective(source),
    feedback: sanitized.text,
    ...(input.modelProfileId == null ? {} : { modelProfileId: input.modelProfileId }),
    ...(input.codebaseProfileId == null ? {} : { codebaseProfileId: input.codebaseProfileId }),
    ...(input.policyId == null ? {} : { policyId: input.policyId }),
    traceIds: [...(input.traceIds ?? [])].sort(),
    spanIds: [...(input.spanIds ?? [])].sort(),
    evalCaseIds: [...(input.evalCaseIds ?? [])].sort(),
    redacted: sanitized.redacted,
    truncated: sanitized.truncated,
  });
};

const sourceObjective = (source: "test_output" | "truncation" | "llm_critique"): string => {
  switch (source) {
    case "test_output":
      return "Repair the behavior described by test/typecheck output while preserving passing behavior.";
    case "truncation":
      return "Reduce context loss and preserve tail facts needed for correct coding-agent behavior.";
    case "llm_critique":
      return "Incorporate critique into prompt/tool policy candidates without overfitting hidden evals.";
  }
};

const sanitizeText = (value: string, maxChars: number): Sanitized => {
  let redacted = false;
  let text = value.replace(
    /\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}\b/g,
    () => {
      redacted = true;
      return REDACTION;
    },
  );
  text = text.replace(
    /\b[A-Za-z0-9_.:-]*(?:api[_-]?key|token|secret|password|credential)[A-Za-z0-9_.:-]*\s*[:=]\s*["']?[^"',\s]{8,}["']?/gi,
    (match) => {
      redacted = true;
      const separator = match.includes("=") ? "=" : ":";
      return `${match.slice(0, match.indexOf(separator) + 1)}${REDACTION}`;
    },
  );
  const truncated = text.length > maxChars;
  if (truncated) {
    text = `${text.slice(0, maxChars)}[truncated:${text.length - maxChars} chars]`;
  }
  return { text, redacted, truncated };
};

const resolveLimits = (limits: BuildGepaFeedbackInput["limits"]): GepaFeedbackBundle["limits"] => ({
  maxRecords: boundedInteger(limits?.maxRecords, DEFAULT_MAX_RECORDS, 1, MAX_RECORDS),
  maxTextChars: boundedInteger(limits?.maxTextChars, DEFAULT_MAX_TEXT_CHARS, 1, MAX_TEXT_CHARS),
});

const compareFeedback = (left: GepaFeedbackRecord, right: GepaFeedbackRecord): number => {
  const severity = severityRank(right.severity) - severityRank(left.severity);
  if (severity !== 0) {
    return severity;
  }
  const source = left.source.localeCompare(right.source);
  return source === 0 ? left.feedbackId.localeCompare(right.feedbackId) : source;
};

const severityRank = (severity: GepaFeedbackRecord["severity"]): number => {
  switch (severity) {
    case "critical":
      return 4;
    case "failure":
      return 3;
    case "warning":
      return 2;
    case "info":
      return 1;
  }
};

const singleValue = (values: readonly string[]): string | undefined =>
  values.length === 1 ? values[0] : undefined;

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 180) || "gepa-feedback.empty";

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};

const stringify = (value: unknown): string => {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};
