import { z } from "zod";
import type { EvalRunResult, EvalScorecard } from "../eval-harness/types";
import type { TraceFailureCluster, TraceLatencyCluster, TraceOptimizerDimensions } from "../trace-analysis";
import { OptimizerIdSchema, OptimizerVersionSchema } from "./types";

const DEFAULT_MAX_OBSERVATIONS = 8;
const DEFAULT_MAX_EXCERPTS_PER_OBSERVATION = 4;
const DEFAULT_MAX_EXCERPT_CHARS = 240;
const MAX_OBSERVATIONS = 50;
const MAX_EXCERPTS_PER_OBSERVATION = 20;
const MAX_EXCERPT_CHARS = 2_000;
const REDACTION = "[REDACTED_SECRET]";

export const CandidateEvidenceSeveritySchema = z.enum(["low", "medium", "high", "critical"]);
export type CandidateEvidenceSeverity = z.infer<typeof CandidateEvidenceSeveritySchema>;

export const CandidateEvidenceLineageSchema = z.object({
  modelProfileIds: z.array(OptimizerIdSchema).default([]),
  codebaseProfileIds: z.array(OptimizerIdSchema).default([]),
  policyIds: z.array(OptimizerIdSchema).default([]),
  canonicalToolVersions: z.array(OptimizerVersionSchema).default([]),
  renderedToolVersions: z.array(OptimizerVersionSchema).default([]),
  resultStyleVersions: z.array(OptimizerVersionSchema).default([]),
  verificationPolicyVersions: z.array(OptimizerVersionSchema).default([]),
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
}).strict();
export type CandidateEvidenceLineage = z.infer<typeof CandidateEvidenceLineageSchema>;

export const CandidateEvidenceExcerptSchema = z.object({
  excerptId: OptimizerIdSchema,
  source: z.enum(["trace", "eval", "span"]),
  text: z.string(),
  redacted: z.boolean().default(false),
  originalChars: z.number().int().nonnegative(),
  truncated: z.boolean().default(false),
  traceId: z.string().min(1).optional(),
  spanId: z.string().min(1).optional(),
  evalCaseId: OptimizerIdSchema.optional(),
  runResultId: OptimizerIdSchema.optional(),
  scorecardId: OptimizerIdSchema.optional(),
}).strict();
export type CandidateEvidenceExcerpt = z.infer<typeof CandidateEvidenceExcerptSchema>;

export const CandidateEvidenceObservationSchema = z.object({
  observationId: OptimizerIdSchema,
  source: z.enum(["trace_failure", "trace_latency", "eval_run", "eval_scorecard", "span_excerpt"]),
  severity: CandidateEvidenceSeveritySchema,
  title: z.string().min(1),
  count: z.number().int().nonnegative().default(1),
  traceIds: z.array(z.string().min(1)).default([]),
  spanIds: z.array(z.string().min(1)).default([]),
  evalCaseIds: z.array(OptimizerIdSchema).default([]),
  runResultIds: z.array(OptimizerIdSchema).default([]),
  scorecardIds: z.array(OptimizerIdSchema).default([]),
  inputHashes: z.array(z.string().min(1)).default([]),
  argumentHashes: z.array(z.string().min(1)).default([]),
  toolNames: z.array(OptimizerIdSchema).default([]),
  lineage: CandidateEvidenceLineageSchema,
  excerpts: z.array(CandidateEvidenceExcerptSchema).default([]),
}).strict();
export type CandidateEvidenceObservation = z.infer<typeof CandidateEvidenceObservationSchema>;

export const CandidateEvidenceBundleSchema = z.object({
  evidenceBundleId: OptimizerIdSchema,
  schemaVersion: OptimizerVersionSchema.default("candidate-evidence.v1"),
  createdAt: z.string(),
  lineage: CandidateEvidenceLineageSchema,
  observations: z.array(CandidateEvidenceObservationSchema).default([]),
  sourceTraceIds: z.array(z.string().min(1)).default([]),
  sourceSpanIds: z.array(z.string().min(1)).default([]),
  sourceEvalCaseIds: z.array(OptimizerIdSchema).default([]),
  sourceRunResultIds: z.array(OptimizerIdSchema).default([]),
  sourceScorecardIds: z.array(OptimizerIdSchema).default([]),
  redactionCount: z.number().int().nonnegative().default(0),
  limits: z.object({
    maxObservations: z.number().int().positive(),
    maxExcerptsPerObservation: z.number().int().nonnegative(),
    maxExcerptChars: z.number().int().nonnegative(),
  }).strict(),
}).strict();
export type CandidateEvidenceBundle = z.infer<typeof CandidateEvidenceBundleSchema>;

export type CandidateEvidenceSpanExcerptInput = {
  traceId: string;
  spanId: string;
  title?: string;
  text: string;
  lineage?: Partial<CandidateEvidenceLineage>;
  argumentHash?: string;
  toolName?: string;
};

export type BuildCandidateEvidenceBundleInput = {
  evidenceBundleId?: string;
  createdAt?: string;
  traceFailures?: readonly TraceFailureCluster[];
  traceLatencies?: readonly TraceLatencyCluster[];
  evalScorecards?: readonly EvalScorecard[];
  evalRunResults?: readonly EvalRunResult[];
  selectedSpanExcerpts?: readonly CandidateEvidenceSpanExcerptInput[];
  limits?: Partial<CandidateEvidenceBundle["limits"]>;
};

type SanitizedText = {
  text: string;
  redacted: boolean;
  originalChars: number;
  truncated: boolean;
};

const emptyLineage = (): CandidateEvidenceLineage => ({
  modelProfileIds: [],
  codebaseProfileIds: [],
  policyIds: [],
  canonicalToolVersions: [],
  renderedToolVersions: [],
  resultStyleVersions: [],
  verificationPolicyVersions: [],
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

const resolveLimits = (
  limits: BuildCandidateEvidenceBundleInput["limits"],
): CandidateEvidenceBundle["limits"] => ({
  maxObservations: boundedInteger(limits?.maxObservations, DEFAULT_MAX_OBSERVATIONS, 1, MAX_OBSERVATIONS),
  maxExcerptsPerObservation: boundedInteger(
    limits?.maxExcerptsPerObservation,
    DEFAULT_MAX_EXCERPTS_PER_OBSERVATION,
    0,
    MAX_EXCERPTS_PER_OBSERVATION,
  ),
  maxExcerptChars: boundedInteger(limits?.maxExcerptChars, DEFAULT_MAX_EXCERPT_CHARS, 0, MAX_EXCERPT_CHARS),
});

export const buildCandidateEvidenceBundle = (
  input: BuildCandidateEvidenceBundleInput,
): CandidateEvidenceBundle => {
  const limits = resolveLimits(input.limits);
  const observations = [
    ...(input.traceFailures ?? []).map((cluster) => observationFromTraceFailure(cluster, limits)),
    ...(input.traceLatencies ?? []).map((cluster) => observationFromTraceLatency(cluster, limits)),
    ...(input.evalRunResults ?? []).map((run) => observationFromEvalRun(run, limits)),
    ...(input.evalScorecards ?? []).map((scorecard) => observationFromEvalScorecard(scorecard, limits)),
    ...(input.selectedSpanExcerpts ?? []).map((span, index) => observationFromSpanExcerpt(span, index, limits)),
  ]
    .sort(compareObservations)
    .slice(0, limits.maxObservations);

  const bundle: CandidateEvidenceBundle = {
    evidenceBundleId: input.evidenceBundleId ?? stableEvidenceBundleId(observations),
    schemaVersion: "candidate-evidence.v1",
    createdAt: input.createdAt ?? new Date().toISOString(),
    lineage: mergeLineage(observations.map((observation) => observation.lineage)),
    observations,
    sourceTraceIds: uniqueSorted(observations.flatMap((observation) => observation.traceIds)),
    sourceSpanIds: uniqueSorted(observations.flatMap((observation) => observation.spanIds)),
    sourceEvalCaseIds: uniqueSorted(observations.flatMap((observation) => observation.evalCaseIds)),
    sourceRunResultIds: uniqueSorted(observations.flatMap((observation) => observation.runResultIds)),
    sourceScorecardIds: uniqueSorted(observations.flatMap((observation) => observation.scorecardIds)),
    redactionCount: observations.reduce(
      (count, observation) => count + observation.excerpts.filter((excerpt) => excerpt.redacted).length,
      0,
    ),
    limits,
  };

  return CandidateEvidenceBundleSchema.parse(bundle);
};

const observationFromTraceFailure = (
  cluster: TraceFailureCluster,
  limits: CandidateEvidenceBundle["limits"],
): CandidateEvidenceObservation => {
  const title = `Trace failure: ${cluster.observationKind} ${cluster.name}`;
  return CandidateEvidenceObservationSchema.parse({
    observationId: stableId("trace-failure", cluster.observationKind, cluster.name),
    source: "trace_failure",
    severity: cluster.count >= 3 ? "high" : "medium",
    title,
    count: cluster.count,
    traceIds: uniqueSorted(cluster.traces),
    inputHashes: uniqueSorted(cluster.inputHashes),
    lineage: lineageFromTraceDimensions(cluster.optimizerDimensions),
    excerpts: buildExcerpts("trace", title, cluster.messages, limits),
  });
};

const observationFromTraceLatency = (
  cluster: TraceLatencyCluster,
  limits: CandidateEvidenceBundle["limits"],
): CandidateEvidenceObservation => {
  const title = `Trace latency: ${cluster.observationKind} ${cluster.name}`;
  return CandidateEvidenceObservationSchema.parse({
    observationId: stableId("trace-latency", cluster.observationKind, cluster.name),
    source: "trace_latency",
    severity: cluster.p95Ms >= 60_000 ? "high" : "medium",
    title,
    count: cluster.count,
    lineage: lineageFromTraceDimensions(cluster.optimizerDimensions),
    excerpts: buildExcerpts(
      "trace",
      title,
      [`count=${cluster.count}`, `p50=${cluster.p50Ms}ms`, `p95=${cluster.p95Ms}ms`],
      limits,
    ),
  });
};

const observationFromEvalRun = (
  run: EvalRunResult,
  limits: CandidateEvidenceBundle["limits"],
): CandidateEvidenceObservation => {
  const failedAssertions = run.assertionResults.filter((assertion) => !assertion.passed);
  const messages = failedAssertions.flatMap((assertion) => [
    `${assertion.assertionId}: ${assertion.message ?? "assertion failed"}`,
    stringifyJsonValue(assertion.actual),
  ]);
  return CandidateEvidenceObservationSchema.parse({
    observationId: stableId("eval-run", run.runResultId),
    source: "eval_run",
    severity: evalRunSeverity(run),
    title: `Eval ${run.runRole} run ${run.status}: ${run.evalCaseId}`,
    count: 1,
    evalCaseIds: [run.evalCaseId],
    runResultIds: [run.runResultId],
    lineage: lineageFromEvalContext(run.context),
    excerpts: buildExcerpts("eval", `eval-run-${run.runResultId}`, messages, limits, {
      evalCaseId: run.evalCaseId,
      runResultId: run.runResultId,
    }),
  });
};

const observationFromEvalScorecard = (
  scorecard: EvalScorecard,
  limits: CandidateEvidenceBundle["limits"],
): CandidateEvidenceObservation => {
  const regressions = scorecard.criticalRegressionVeto.regressions;
  const messages = [
    `aggregateScore=${scorecard.aggregateScore}`,
    `passed=${scorecard.passed}`,
    ...regressions.map((regression) => `${regression.regressionId}: ${regression.reason}`),
  ];
  return CandidateEvidenceObservationSchema.parse({
    observationId: stableId("eval-scorecard", scorecard.scorecardId),
    source: "eval_scorecard",
    severity: scorecard.criticalRegressionVeto.vetoed ? "critical" : scorecard.passed ? "low" : "medium",
    title: `Eval scorecard ${scorecard.passed ? "passed" : "failed"}: ${scorecard.scorecardId}`,
    count: scorecard.runResults.length,
    evalCaseIds: uniqueSorted(scorecard.runResults.map((run) => run.evalCaseId)),
    runResultIds: uniqueSorted(scorecard.runResults.map((run) => run.runResultId)),
    scorecardIds: [scorecard.scorecardId],
    lineage: mergeLineage([
      lineageFromEvalContext(scorecard.baseline.context),
      lineageFromEvalContext(scorecard.candidate.context),
      ...scorecard.runResults.map((run) => lineageFromEvalContext(run.context)),
    ]),
    excerpts: buildExcerpts("eval", `scorecard-${scorecard.scorecardId}`, messages, limits, {
      scorecardId: scorecard.scorecardId,
    }),
  });
};

const observationFromSpanExcerpt = (
  span: CandidateEvidenceSpanExcerptInput,
  index: number,
  limits: CandidateEvidenceBundle["limits"],
): CandidateEvidenceObservation =>
  CandidateEvidenceObservationSchema.parse({
    observationId: stableId("span", span.traceId, span.spanId, String(index)),
    source: "span_excerpt",
    severity: "medium",
    title: span.title ?? `Selected span ${span.spanId}`,
    count: 1,
    traceIds: [span.traceId],
    spanIds: [span.spanId],
    argumentHashes: span.argumentHash == null ? [] : [span.argumentHash],
    toolNames: span.toolName == null ? [] : [span.toolName],
    lineage: CandidateEvidenceLineageSchema.parse({ ...emptyLineage(), ...span.lineage }),
    excerpts: buildExcerpts("span", `span-${span.spanId}`, [span.text], limits, {
      traceId: span.traceId,
      spanId: span.spanId,
    }),
  });

const buildExcerpts = (
  source: CandidateEvidenceExcerpt["source"],
  sourceId: string,
  texts: readonly string[],
  limits: CandidateEvidenceBundle["limits"],
  refs: Partial<Pick<CandidateEvidenceExcerpt, "traceId" | "spanId" | "evalCaseId" | "runResultId" | "scorecardId">> = {},
): CandidateEvidenceExcerpt[] =>
  texts
    .filter((text) => text.trim().length > 0)
    .slice(0, limits.maxExcerptsPerObservation)
    .map((text, index) => {
      const sanitized = sanitizeExcerpt(text, limits.maxExcerptChars);
      return CandidateEvidenceExcerptSchema.parse({
        excerptId: stableId(source, sourceId, String(index)),
        source,
        text: sanitized.text,
        redacted: sanitized.redacted,
        originalChars: sanitized.originalChars,
        truncated: sanitized.truncated,
        ...refs,
      });
    });

const sanitizeExcerpt = (value: string, maxChars: number): SanitizedText => {
  const originalChars = value.length;
  let redacted = false;
  let text = value.replace(
    /\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}\b/g,
    () => {
      redacted = true;
      return REDACTION;
    },
  );
  text = text.replace(
    /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
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

  const truncated = maxChars > 0 && text.length > maxChars;
  if (truncated) {
    text = `${text.slice(0, maxChars)}[truncated:${text.length - maxChars} chars]`;
  }

  return { text, redacted, originalChars, truncated };
};

const lineageFromTraceDimensions = (dimensions: TraceOptimizerDimensions): CandidateEvidenceLineage =>
  CandidateEvidenceLineageSchema.parse({
    modelProfileIds: uniqueSorted(dimensions.modelProfileIds),
    codebaseProfileIds: uniqueSorted(dimensions.codebaseProfileIds),
    policyIds: uniqueSorted(dimensions.policyIds),
    canonicalToolVersions: uniqueSorted(dimensions.canonicalToolVersions),
    renderedToolVersions: uniqueSorted(dimensions.renderedToolVersions),
    resultStyleVersions: uniqueSorted(dimensions.resultStyleVersions),
    verificationPolicyVersions: uniqueSorted(dimensions.verificationPolicyVersions),
    editStrategyVersions: uniqueSorted(dimensions.editStrategyVersions),
    renderedEditContractVersions: uniqueSorted(dimensions.renderedEditContractVersions),
    editFallbackPolicyVersions: uniqueSorted(dimensions.editFallbackPolicyVersions),
    editRepairPolicyVersions: uniqueSorted(dimensions.editRepairPolicyVersions),
    editVerifierPolicyVersions: uniqueSorted(dimensions.editVerifierPolicyVersions),
    editObjectiveSetIds: uniqueSorted(dimensions.editObjectiveSetIds),
    editStrategyIds: uniqueSorted(dimensions.editStrategyIds),
    editStrategyFamilies: uniqueSorted(dimensions.editStrategyFamilies),
    canonicalEditToolSpecIds: uniqueSorted(dimensions.canonicalEditToolSpecIds),
    renderedEditToolContractIds: uniqueSorted(dimensions.renderedEditToolContractIds),
  });

const lineageFromEvalContext = (context: EvalRunResult["context"]): CandidateEvidenceLineage =>
  CandidateEvidenceLineageSchema.parse({
    modelProfileIds: [context.modelProfileId],
    codebaseProfileIds: [context.codebaseProfileId],
    policyIds: [context.policyId],
    canonicalToolVersions: [context.canonicalToolVersion],
    renderedToolVersions: [context.renderedToolVersion],
    resultStyleVersions: [context.resultStyleVersion],
    verificationPolicyVersions: [context.verificationPolicyVersion],
  });

const mergeLineage = (lineages: readonly CandidateEvidenceLineage[]): CandidateEvidenceLineage =>
  CandidateEvidenceLineageSchema.parse({
    modelProfileIds: uniqueSorted(lineages.flatMap((lineage) => lineage.modelProfileIds)),
    codebaseProfileIds: uniqueSorted(lineages.flatMap((lineage) => lineage.codebaseProfileIds)),
    policyIds: uniqueSorted(lineages.flatMap((lineage) => lineage.policyIds)),
    canonicalToolVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.canonicalToolVersions)),
    renderedToolVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.renderedToolVersions)),
    resultStyleVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.resultStyleVersions)),
    verificationPolicyVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.verificationPolicyVersions)),
    editStrategyVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.editStrategyVersions)),
    renderedEditContractVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.renderedEditContractVersions)),
    editFallbackPolicyVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.editFallbackPolicyVersions)),
    editRepairPolicyVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.editRepairPolicyVersions)),
    editVerifierPolicyVersions: uniqueSorted(lineages.flatMap((lineage) => lineage.editVerifierPolicyVersions)),
    editObjectiveSetIds: uniqueSorted(lineages.flatMap((lineage) => lineage.editObjectiveSetIds)),
    editStrategyIds: uniqueSorted(lineages.flatMap((lineage) => lineage.editStrategyIds)),
    editStrategyFamilies: uniqueSorted(lineages.flatMap((lineage) => lineage.editStrategyFamilies)),
    canonicalEditToolSpecIds: uniqueSorted(lineages.flatMap((lineage) => lineage.canonicalEditToolSpecIds)),
    renderedEditToolContractIds: uniqueSorted(lineages.flatMap((lineage) => lineage.renderedEditToolContractIds)),
  });

const evalRunSeverity = (run: EvalRunResult): CandidateEvidenceSeverity => {
  if (run.assertionResults.some((assertion) => !assertion.passed && assertion.severity === "critical")) {
    return "critical";
  }
  if (run.status === "passed") {
    return "low";
  }
  return run.status === "timeout" || run.status === "error" ? "high" : "medium";
};

const compareObservations = (left: CandidateEvidenceObservation, right: CandidateEvidenceObservation): number => {
  const severity = severityRank(right.severity) - severityRank(left.severity);
  if (severity !== 0) {
    return severity;
  }
  const source = left.source.localeCompare(right.source);
  return source === 0 ? left.observationId.localeCompare(right.observationId) : source;
};

const severityRank = (severity: CandidateEvidenceSeverity): number => {
  switch (severity) {
    case "critical":
      return 4;
    case "high":
      return 3;
    case "medium":
      return 2;
    case "low":
      return 1;
  }
};

const stableEvidenceBundleId = (observations: readonly CandidateEvidenceObservation[]): string =>
  stableId("evidence", observations.map((observation) => observation.observationId).join("."));

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 160) || "evidence.empty";

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values.filter((value) => value.length > 0))].sort((left, right) => left.localeCompare(right));

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};

const stringifyJsonValue = (value: unknown): string =>
  value == null ? "" : typeof value === "string" ? value : JSON.stringify(value);
