import { z } from "zod";
import {
  CandidateGenerationDiagnosticSchema,
  CandidateGenerationResultSchema,
  generateCandidatePatches,
  type CandidateGenerationDiagnostic,
  type CandidateGenerationResult,
} from "./candidates";
import {
  CandidateEvidenceBundleSchema,
  CandidateEvidenceLineageSchema,
  CandidateEvidenceObservationSchema,
  type CandidateEvidenceBundle,
  type CandidateEvidenceLineage,
  type CandidateEvidenceObservation,
} from "./evidence";
import { GepaFeedbackBundleSchema, type GepaFeedbackBundle, type GepaFeedbackRecord } from "./gepa-feedback";
import {
  CandidateValidationResultSchema,
  validateCandidatePatch,
  type CandidateValidationResult,
} from "./validator";
import {
  CandidatePatchSchema,
  CandidateScopeSchema,
  JsonValueSchema,
  OptimizerIdSchema,
  type CandidatePatch,
  type CandidateScope,
  type OptimizerRegistryRecord,
} from "./types";

const DEFAULT_MAX_ITERATIONS = 3;
const DEFAULT_MAX_FEEDBACK_RECORDS_PER_ITERATION = 8;
const DEFAULT_MAX_CANDIDATES_PER_ITERATION = 4;
const DEFAULT_MAX_TOTAL_CANDIDATES = 12;
const MAX_ITERATIONS = 100;
const MAX_FEEDBACK_RECORDS_PER_ITERATION = 50;
const MAX_CANDIDATES_PER_ITERATION = 50;
const MAX_TOTAL_CANDIDATES = 500;

export const GepaRunnerDiagnosticSchema = z.object({
  iteration: z.number().int().nonnegative().optional(),
  feedbackId: OptimizerIdSchema.optional(),
  candidatePatchId: OptimizerIdSchema.optional(),
  severity: z.enum(["info", "warning", "error"]),
  reason: z.string().min(1),
}).strict();
export type GepaRunnerDiagnostic = z.infer<typeof GepaRunnerDiagnosticSchema>;

export const GepaRunnerIterationSchema = z.object({
  iteration: z.number().int().nonnegative(),
  feedbackIds: z.array(OptimizerIdSchema),
  evidenceBundleId: OptimizerIdSchema,
  candidatePatchIds: z.array(OptimizerIdSchema),
  diagnostics: z.array(GepaRunnerDiagnosticSchema).default([]),
}).strict();
export type GepaRunnerIteration = z.infer<typeof GepaRunnerIterationSchema>;

export const GepaRunnerStateSchema = z.object({
  runId: OptimizerIdSchema,
  schemaVersion: z.literal("gepa-runner.v1").default("gepa-runner.v1"),
  feedbackBundleId: OptimizerIdSchema,
  createdAt: z.string(),
  updatedAt: z.string(),
  iterationCount: z.number().int().nonnegative(),
  processedFeedbackIds: z.array(OptimizerIdSchema).default([]),
  iterations: z.array(GepaRunnerIterationSchema).default([]),
  candidates: z.array(CandidatePatchSchema).default([]),
  validations: z.array(CandidateValidationResultSchema).default([]),
  diagnostics: z.array(GepaRunnerDiagnosticSchema).default([]),
  exhausted: z.boolean().default(false),
}).strict();
export type GepaRunnerState = z.infer<typeof GepaRunnerStateSchema>;

export type GepaCandidateProposerInput = {
  iteration: number;
  createdAt: string;
  feedbackBundle: GepaFeedbackBundle;
  feedbackRecords: readonly GepaFeedbackRecord[];
  evidence: CandidateEvidenceBundle;
  maxCandidates: number;
};

export type GepaCandidateProposer = (input: GepaCandidateProposerInput) => CandidateGenerationResult;

export const GepaLlmProposerRequestSchema = z.object({
  schemaVersion: z.literal("gepa-llm-proposer-request.v1"),
  iteration: z.number().int().nonnegative(),
  evidenceBundleId: OptimizerIdSchema,
  maxCandidates: z.number().int().positive(),
  allowedScopes: z.array(CandidateScopeSchema),
  prompt: z.string().min(1),
  responseSchemaName: z.literal("CandidateGenerationResult"),
}).strict();
export type GepaLlmProposerRequest = z.infer<typeof GepaLlmProposerRequestSchema>;

const GepaLlmProposerResponseSchema = z.object({
  evidenceBundleId: OptimizerIdSchema,
  candidates: z.array(JsonValueSchema).default([]),
  diagnostics: z.array(CandidateGenerationDiagnosticSchema).default([]),
}).strict();

export type GepaLlmProposerClient = (request: GepaLlmProposerRequest) => unknown;

export type CreateLlmBackedGepaProposerInput = {
  client: GepaLlmProposerClient;
  fallbackProposer?: GepaCandidateProposer;
};

export type RunGepaOptimizerInput = {
  feedbackBundle: GepaFeedbackBundle;
  initialState?: GepaRunnerState;
  createdAt?: string;
  runId?: string;
  maxIterations?: number;
  maxFeedbackRecordsPerIteration?: number;
  maxCandidatesPerIteration?: number;
  maxTotalCandidates?: number;
  proposer?: GepaCandidateProposer;
  records?: readonly OptimizerRegistryRecord[];
  expectedBaseHashes?: Readonly<Record<string, string>>;
  actualBaseHashes?: Readonly<Record<string, string>>;
  requiredEvalGateIds?: readonly string[];
};

type RunnerLimits = {
  maxIterations: number;
  maxFeedbackRecordsPerIteration: number;
  maxCandidatesPerIteration: number;
  maxTotalCandidates: number;
};

export const runGepaOptimizer = (input: RunGepaOptimizerInput): GepaRunnerState => {
  const feedbackBundle = GepaFeedbackBundleSchema.parse(input.feedbackBundle);
  const createdAt = input.createdAt ?? new Date().toISOString();
  const limits = resolveLimits(input);
  const startingState = input.initialState == null
    ? emptyState(input.runId ?? stableId("gepa-run", feedbackBundle.feedbackBundleId), feedbackBundle.feedbackBundleId, createdAt)
    : GepaRunnerStateSchema.parse(input.initialState);

  const processedFeedbackIds = new Set(startingState.processedFeedbackIds);
  const candidates = [...startingState.candidates];
  const validations = [...startingState.validations];
  const diagnostics = [...startingState.diagnostics];
  const iterations = [...startingState.iterations];

  const remainingRecords = [...feedbackBundle.records]
    .filter((record) => !processedFeedbackIds.has(record.feedbackId))
    .sort(compareFeedbackRecords);

  if (remainingRecords.length === 0 && feedbackBundle.records.length === 0) {
    diagnostics.push({
      severity: "warning",
      reason: "no feedback records available for GEPA optimization",
    });
  }

  let nextIteration = startingState.iterationCount;
  while (
    nextIteration - startingState.iterationCount < limits.maxIterations &&
    remainingRecords.length > 0 &&
    candidates.length < limits.maxTotalCandidates
  ) {
    const batch = remainingRecords.splice(0, limits.maxFeedbackRecordsPerIteration);
    const remainingCandidateBudget = limits.maxTotalCandidates - candidates.length;
    const maxCandidates = Math.min(limits.maxCandidatesPerIteration, remainingCandidateBudget);
    const evidence = buildEvidenceForFeedbackBatch({
      feedbackBundle,
      records: batch,
      iteration: nextIteration,
      createdAt,
    });
    const generation = (input.proposer ?? defaultProposer)({
      iteration: nextIteration,
      createdAt,
      feedbackBundle,
      feedbackRecords: batch,
      evidence,
      maxCandidates,
    });
    const parsedGeneration = CandidateGenerationResultSchema.parse(generation);
    const iterationDiagnostics = [
      ...evidence.observations.flatMap((observation) => diagnosticsForObservation(observation, nextIteration)),
      ...parsedGeneration.diagnostics.map((diagnostic) => generationDiagnosticToRunnerDiagnostic(diagnostic, nextIteration)),
    ];

    const acceptedCandidates: CandidatePatch[] = [];
    for (const candidate of parsedGeneration.candidates) {
      const scopeIssue = autonomousCandidateScopeIssue(candidate);
      if (scopeIssue !== undefined) {
        iterationDiagnostics.push({
          iteration: nextIteration,
          candidatePatchId: candidate.candidatePatchId,
          severity: "warning",
          reason: `candidate rejected by autonomous GEPA scope: ${scopeIssue}`,
        });
        continue;
      }
      if (acceptedCandidates.length >= maxCandidates) {
        iterationDiagnostics.push({
          iteration: nextIteration,
          severity: "info",
          reason: `candidate cap reached at ${maxCandidates}`,
        });
        break;
      }
      acceptedCandidates.push(candidate);
    }
    if (acceptedCandidates.length === 0) {
      iterationDiagnostics.push({
        iteration: nextIteration,
        severity: "warning",
        reason: "GEPA iteration produced no candidate patches",
      });
    }

    for (const candidate of acceptedCandidates) {
      candidates.push(candidate);
      const validation = maybeValidateCandidate(input, candidate);
      if (validation == null) {
        iterationDiagnostics.push({
          iteration: nextIteration,
          candidatePatchId: candidate.candidatePatchId,
          severity: "warning",
          reason: "candidate validation skipped because registry records or expected base hashes were not supplied",
        });
      } else {
        validations.push(validation);
      }
    }

    for (const record of batch) {
      processedFeedbackIds.add(record.feedbackId);
    }

    iterations.push(GepaRunnerIterationSchema.parse({
      iteration: nextIteration,
      feedbackIds: batch.map((record) => record.feedbackId),
      evidenceBundleId: evidence.evidenceBundleId,
      candidatePatchIds: acceptedCandidates.map((candidate) => candidate.candidatePatchId),
      diagnostics: iterationDiagnostics,
    }));
    diagnostics.push(...iterationDiagnostics);
    nextIteration += 1;
  }

  if (remainingRecords.length > 0 && nextIteration - startingState.iterationCount >= limits.maxIterations) {
    diagnostics.push({
      iteration: nextIteration,
      severity: "info",
      reason: `iteration cap reached at ${limits.maxIterations}`,
    });
  }
  if (remainingRecords.length > 0 && candidates.length >= limits.maxTotalCandidates) {
    diagnostics.push({
      iteration: nextIteration,
      severity: "info",
      reason: `total candidate cap reached at ${limits.maxTotalCandidates}`,
    });
  }

  return GepaRunnerStateSchema.parse({
    ...startingState,
    updatedAt: createdAt,
    iterationCount: nextIteration,
    processedFeedbackIds: [...processedFeedbackIds].sort((left, right) => left.localeCompare(right)),
    iterations,
    candidates,
    validations,
    diagnostics,
    exhausted: remainingRecords.length === 0,
  });
};

const defaultProposer: GepaCandidateProposer = (input) =>
  generateCandidatePatches({
    evidence: input.evidence,
    createdAt: input.createdAt,
    maxCandidates: input.maxCandidates,
  });

export const createLlmBackedGepaProposer = (
  input: CreateLlmBackedGepaProposerInput,
): GepaCandidateProposer => {
  const fallbackProposer = input.fallbackProposer ?? defaultProposer;
  return (proposerInput) => {
    const fallback = CandidateGenerationResultSchema.parse(fallbackProposer(proposerInput));
    const allowedScopes = uniqueScopes(fallback.candidates.map((candidate) => candidate.scope));
    const request = GepaLlmProposerRequestSchema.parse({
      schemaVersion: "gepa-llm-proposer-request.v1",
      iteration: proposerInput.iteration,
      evidenceBundleId: proposerInput.evidence.evidenceBundleId,
      maxCandidates: proposerInput.maxCandidates,
      allowedScopes,
      prompt: renderLlmProposerPrompt(proposerInput, allowedScopes),
      responseSchemaName: "CandidateGenerationResult",
    });

    const rawResponse = input.client(request);
    const parsedResponse = GepaLlmProposerResponseSchema.safeParse(rawResponse);
    if (!parsedResponse.success) {
      return CandidateGenerationResultSchema.parse({
        ...fallback,
        diagnostics: [
          ...fallback.diagnostics,
          {
            severity: "warning",
            reason: `LLM proposer response failed schema validation; deterministic fallback used: ${zodSummary(parsedResponse.error)}`,
          },
        ],
      });
    }

    const diagnostics: CandidateGenerationDiagnostic[] = [
      ...fallback.diagnostics,
      ...parsedResponse.data.diagnostics,
    ];
    if (parsedResponse.data.evidenceBundleId !== proposerInput.evidence.evidenceBundleId) {
      return CandidateGenerationResultSchema.parse({
        ...fallback,
        diagnostics: [
          ...diagnostics,
          {
            severity: "warning",
            reason: `LLM proposer response evidenceBundleId ${parsedResponse.data.evidenceBundleId} did not match ${proposerInput.evidence.evidenceBundleId}; deterministic fallback used`,
          },
        ],
      });
    }

    const acceptedCandidates: CandidatePatch[] = [];
    for (const [index, candidateLike] of parsedResponse.data.candidates.entries()) {
      const parsedCandidate = CandidatePatchSchema.safeParse(candidateLike);
      if (!parsedCandidate.success) {
        diagnostics.push({
          severity: "warning",
          reason: `LLM proposer candidate ${index} failed candidate schema validation: ${zodSummary(parsedCandidate.error)}`,
        });
        continue;
      }

      const scopeIssue = candidateScopeIssue(parsedCandidate.data, allowedScopes);
      if (scopeIssue !== undefined) {
        diagnostics.push({
          severity: "warning",
          reason: `LLM proposer candidate ${parsedCandidate.data.candidatePatchId} rejected by scope restrictions: ${scopeIssue}`,
        });
        continue;
      }

      acceptedCandidates.push(parsedCandidate.data);
      if (acceptedCandidates.length >= proposerInput.maxCandidates) {
        break;
      }
    }

    if (acceptedCandidates.length === 0) {
      return CandidateGenerationResultSchema.parse({
        ...fallback,
        diagnostics: [
          ...diagnostics,
          {
            severity: "warning",
            reason: "LLM proposer produced no schema-valid in-scope candidates; deterministic fallback used",
          },
        ],
      });
    }

    return CandidateGenerationResultSchema.parse({
      evidenceBundleId: proposerInput.evidence.evidenceBundleId,
      candidates: acceptedCandidates,
      diagnostics,
    });
  };
};

const emptyState = (runId: string, feedbackBundleId: string, createdAt: string): GepaRunnerState =>
  GepaRunnerStateSchema.parse({
    runId,
    schemaVersion: "gepa-runner.v1",
    feedbackBundleId,
    createdAt,
    updatedAt: createdAt,
    iterationCount: 0,
    processedFeedbackIds: [],
    iterations: [],
    candidates: [],
    validations: [],
    diagnostics: [],
    exhausted: false,
  });

const buildEvidenceForFeedbackBatch = (input: {
  feedbackBundle: GepaFeedbackBundle;
  records: readonly GepaFeedbackRecord[];
  iteration: number;
  createdAt: string;
}): CandidateEvidenceBundle => {
  const observations = input.records.map((record) => observationFromFeedback(record));
  return CandidateEvidenceBundleSchema.parse({
    evidenceBundleId: stableId(input.feedbackBundle.feedbackBundleId, "iter", String(input.iteration)),
    schemaVersion: "candidate-evidence.v1",
    createdAt: input.createdAt,
    lineage: mergeLineage(observations.map((observation) => observation.lineage)),
    observations,
    sourceTraceIds: uniqueSorted(observations.flatMap((observation) => observation.traceIds)),
    sourceSpanIds: uniqueSorted(observations.flatMap((observation) => observation.spanIds)),
    sourceEvalCaseIds: uniqueSorted(observations.flatMap((observation) => observation.evalCaseIds)),
    sourceRunResultIds: uniqueSorted(observations.flatMap((observation) => observation.runResultIds)),
    sourceScorecardIds: uniqueSorted(observations.flatMap((observation) => observation.scorecardIds)),
    redactionCount: input.records.filter((record) => record.redacted).length,
    limits: {
      maxObservations: input.records.length || 1,
      maxExcerptsPerObservation: 1,
      maxExcerptChars: Math.max(...input.records.map((record) => record.feedback.length), 1),
    },
  });
};

const observationFromFeedback = (record: GepaFeedbackRecord): CandidateEvidenceObservation => {
  const lineage = CandidateEvidenceLineageSchema.parse({
    modelProfileIds: record.modelProfileId == null ? [] : [record.modelProfileId],
    codebaseProfileIds: record.codebaseProfileId == null ? [] : [record.codebaseProfileId],
    policyIds: record.policyId == null ? [] : [record.policyId],
    canonicalToolVersions: [],
    renderedToolVersions: [],
    resultStyleVersions: [],
    verificationPolicyVersions: [],
    editStrategyVersions: record.editStrategyVersions,
    renderedEditContractVersions: record.renderedEditContractVersions,
    editFallbackPolicyVersions: record.editFallbackPolicyVersions,
    editRepairPolicyVersions: record.editRepairPolicyVersions,
    editVerifierPolicyVersions: record.editVerifierPolicyVersions,
    editObjectiveSetIds: record.editObjectiveSetIds,
    editStrategyIds: record.editStrategyIds,
    editStrategyFamilies: record.editStrategyFamilies,
    canonicalEditToolSpecIds: record.canonicalEditToolSpecIds,
    renderedEditToolContractIds: record.renderedEditToolContractIds,
  });
  return CandidateEvidenceObservationSchema.parse({
    observationId: stableId("gepa-feedback", record.feedbackId),
    source: evidenceSourceForFeedback(record.source),
    severity: evidenceSeverityForFeedback(record.severity),
    title: `${record.source}: ${record.objective}`,
    count: 1,
    traceIds: record.traceIds,
    spanIds: record.spanIds,
    evalCaseIds: record.evalCaseIds,
    runResultIds: record.runResultIds,
    scorecardIds: record.scorecardIds,
    inputHashes: [],
    argumentHashes: [],
    toolNames: [],
    lineage,
    excerpts: [
      {
        excerptId: stableId("gepa-excerpt", record.feedbackId),
        source: record.traceIds.length > 0 || record.spanIds.length > 0 ? "trace" : "eval",
        text: record.feedback,
        redacted: record.redacted,
        originalChars: record.feedback.length,
        truncated: record.truncated,
        ...(record.traceIds[0] == null ? {} : { traceId: record.traceIds[0] }),
        ...(record.spanIds[0] == null ? {} : { spanId: record.spanIds[0] }),
        ...(record.evalCaseIds[0] == null ? {} : { evalCaseId: record.evalCaseIds[0] }),
        ...(record.runResultIds[0] == null ? {} : { runResultId: record.runResultIds[0] }),
        ...(record.scorecardIds[0] == null ? {} : { scorecardId: record.scorecardIds[0] }),
      },
    ],
  });
};

const evidenceSourceForFeedback = (
  source: GepaFeedbackRecord["source"],
): CandidateEvidenceObservation["source"] => {
  switch (source) {
    case "eval_run":
    case "edit_ablation":
    case "test_output":
      return "eval_run";
    case "eval_scorecard":
      return "eval_scorecard";
    case "trace_evidence":
      return "trace_failure";
    case "truncation":
    case "llm_critique":
      return "span_excerpt";
  }
};

const evidenceSeverityForFeedback = (
  severity: GepaFeedbackRecord["severity"],
): CandidateEvidenceObservation["severity"] => {
  switch (severity) {
    case "critical":
      return "critical";
    case "failure":
      return "high";
    case "warning":
      return "medium";
    case "info":
      return "low";
  }
};

const diagnosticsForObservation = (
  observation: CandidateEvidenceObservation,
  iteration: number,
): GepaRunnerDiagnostic[] => {
  if (
    observation.lineage.modelProfileIds.length === 1 &&
    observation.lineage.codebaseProfileIds.length === 1 &&
    observation.lineage.policyIds.length === 1
  ) {
    return [];
  }
  return [{
    iteration,
    severity: "warning",
    reason: `feedback has missing or ambiguous lineage: ${observation.observationId}`,
  }];
};

const generationDiagnosticToRunnerDiagnostic = (
  diagnostic: CandidateGenerationDiagnostic,
  iteration: number,
): GepaRunnerDiagnostic => GepaRunnerDiagnosticSchema.parse({
  iteration,
  severity: diagnostic.severity,
  reason: diagnostic.reason,
  ...(diagnostic.observationId == null ? {} : { feedbackId: stableId("observation", diagnostic.observationId) }),
});

const maybeValidateCandidate = (
  input: RunGepaOptimizerInput,
  candidate: CandidatePatch,
): CandidateValidationResult | undefined => {
  if (input.records == null || input.expectedBaseHashes == null) {
    return undefined;
  }
  return validateCandidatePatch({
    candidate,
    records: input.records,
    expectedBaseHashes: input.expectedBaseHashes,
    ...(input.actualBaseHashes == null ? {} : { actualBaseHashes: input.actualBaseHashes }),
    ...(input.requiredEvalGateIds == null ? {} : { requiredEvalGateIds: input.requiredEvalGateIds }),
  });
};

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

const compareFeedbackRecords = (left: GepaFeedbackRecord, right: GepaFeedbackRecord): number => {
  const severity = feedbackSeverityRank(right.severity) - feedbackSeverityRank(left.severity);
  if (severity !== 0) {
    return severity;
  }
  const source = left.source.localeCompare(right.source);
  return source === 0 ? left.feedbackId.localeCompare(right.feedbackId) : source;
};

const feedbackSeverityRank = (severity: GepaFeedbackRecord["severity"]): number => {
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

const resolveLimits = (input: RunGepaOptimizerInput): RunnerLimits => ({
  maxIterations: boundedInteger(input.maxIterations, DEFAULT_MAX_ITERATIONS, 1, MAX_ITERATIONS),
  maxFeedbackRecordsPerIteration: boundedInteger(
    input.maxFeedbackRecordsPerIteration,
    DEFAULT_MAX_FEEDBACK_RECORDS_PER_ITERATION,
    1,
    MAX_FEEDBACK_RECORDS_PER_ITERATION,
  ),
  maxCandidatesPerIteration: boundedInteger(
    input.maxCandidatesPerIteration,
    DEFAULT_MAX_CANDIDATES_PER_ITERATION,
    1,
    MAX_CANDIDATES_PER_ITERATION,
  ),
  maxTotalCandidates: boundedInteger(input.maxTotalCandidates, DEFAULT_MAX_TOTAL_CANDIDATES, 1, MAX_TOTAL_CANDIDATES),
});

const renderLlmProposerPrompt = (
  input: GepaCandidateProposerInput,
  allowedScopes: readonly CandidateScope[],
): string => [
  "Propose GEPA optimizer candidate patches as JSON.",
  `Evidence bundle: ${input.evidence.evidenceBundleId}`,
  `Iteration: ${input.iteration}`,
  `Max candidates: ${input.maxCandidates}`,
  "Allowed scopes:",
  ...allowedScopes.map((scope) =>
    `- ${scope.artifactKind} ${scope.artifactId}: ${scope.allowedJsonPointers.join(", ")}`
  ),
  "Observations:",
  ...input.evidence.observations.map((observation) => {
    const excerpt = observation.excerpts[0]?.text ?? "";
    return `- ${observation.observationId} ${observation.severity} ${observation.source}: ${observation.title}${excerpt.length === 0 ? "" : ` | ${excerpt}`}`;
  }),
  "Return an object with evidenceBundleId, candidates, and diagnostics.",
].join("\n");

const uniqueScopes = (scopes: readonly CandidateScope[]): CandidateScope[] => {
  const byScope = new Map<string, Set<string>>();
  for (const scope of scopes) {
    const key = `${scope.artifactKind}\0${scope.artifactId}`;
    byScope.set(key, new Set([...(byScope.get(key) ?? []), ...scope.allowedJsonPointers]));
  }
  return [...byScope.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, allowedJsonPointers]) => {
      const [artifactKind, artifactId] = key.split("\0");
      return CandidateScopeSchema.parse({
        artifactKind,
        artifactId,
        allowedJsonPointers: [...allowedJsonPointers].sort((left, right) => left.localeCompare(right)),
      });
    });
};

const candidateScopeIssue = (
  candidate: CandidatePatch,
  allowedScopes: readonly CandidateScope[],
): string | undefined => {
  const matchingScopes = allowedScopes.filter((scope) =>
    scope.artifactKind === candidate.scope.artifactKind && scope.artifactId === candidate.scope.artifactId
  );
  if (matchingScopes.length === 0) {
    return `artifact ${candidate.scope.artifactKind} ${candidate.scope.artifactId} is not allowed`;
  }

  const allowedPaths = matchingScopes.flatMap((scope) => scope.allowedJsonPointers);
  for (const scopedPath of candidate.scope.allowedJsonPointers) {
    if (!isPathAllowedBy(scopedPath, allowedPaths)) {
      return `declared allowed path ${scopedPath} is outside deterministic proposer scope`;
    }
  }
  for (const operation of candidate.operations) {
    if (!isPathAllowedBy(operation.path, allowedPaths)) {
      return `operation path ${operation.path} is outside deterministic proposer scope`;
    }
  }
  return undefined;
};

const isPathAllowedBy = (path: string, allowedPaths: readonly string[]): boolean =>
  allowedPaths.length > 0 &&
  allowedPaths.some((allowedPath) => path === allowedPath || path.startsWith(`${allowedPath}/`));

const AUTONOMOUS_POLICY_PATHS = [
  "/canonicalToolVersion",
  "/renderedToolVersion",
  "/resultStyleVersion",
  "/verificationPolicyVersion",
  "/editStrategyVersion",
  "/renderedEditContractVersion",
  "/editFallbackPolicyVersion",
  "/editRepairPolicyVersion",
  "/editVerifierPolicyVersion",
  "/editObjectiveSetId",
  "/verificationGates",
] as const;

const AUTONOMOUS_RENDERED_TOOL_PATHS = [
  "/description",
  "/inputSchema",
  "/resultStyle",
  "/promptFragments",
  "/examples",
] as const;

const autonomousCandidateScopeIssue = (candidate: CandidatePatch): string | undefined => {
  const allowedPaths = autonomousAllowedPaths(candidate.scope.artifactKind);
  if (allowedPaths.length === 0) {
    return `artifact kind ${candidate.scope.artifactKind} is outside prompt/tool/edit/verification policy scope`;
  }

  for (const scopedPath of candidate.scope.allowedJsonPointers) {
    if (!isPathAllowedBy(scopedPath, allowedPaths)) {
      return `declared allowed path ${scopedPath} is outside prompt/tool/edit/verification policy scope`;
    }
  }
  for (const operation of candidate.operations) {
    if (!isPathAllowedBy(operation.path, allowedPaths)) {
      return `operation path ${operation.path} is outside prompt/tool/edit/verification policy scope`;
    }
  }
  return undefined;
};

const autonomousAllowedPaths = (artifactKind: CandidateScope["artifactKind"]): readonly string[] => {
  switch (artifactKind) {
    case "model_codebase_policy":
      return AUTONOMOUS_POLICY_PATHS;
    case "rendered_tool_contract":
      return AUTONOMOUS_RENDERED_TOOL_PATHS;
    case "model_profile":
    case "codebase_profile":
    case "canonical_tool_spec":
      return [];
  }
};

const zodSummary = (error: z.ZodError): string =>
  error.issues.map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`).join("; ");

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values.filter((value) => value.length > 0))].sort((left, right) => left.localeCompare(right));

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 180) || "gepa-runner.empty";

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};
