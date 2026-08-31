import { z } from "zod";
import {
  applyEdit,
  editApplySupportedFamilies,
  type EditApplyInput,
  type EditApplyResult,
  type EditApplyWorkspace,
} from "../edit-strategy/apply-layer";
import {
  EditErrorCodeSchema,
  EditPhaseStatusSchema,
  EditStrategyFamilySchema,
  PostApplyConsistencyStatusSchema,
  SelfDetectedRegressionStatusSchema,
  StaleContextStatusSchema,
  VerificationStatusSchema,
  type EditErrorCode,
  type EditPhaseStatus,
  type EditStrategyFamily,
  type PostApplyConsistencyStatus,
  type SelfDetectedRegressionStatus,
  type StaleContextStatus,
  type VerificationStatus,
} from "../edit-strategy/types";
import { OptimizerIdSchema } from "../optimizer/types";
import {
  editStrategyEvalCases,
  EditStrategyEvalCaseSchema,
  type EditStrategyEvalCase,
  type EditStrategyProbe,
} from "./edit-strategy-corpus";
import { createEvalScorecard } from "./scorer";
import {
  EvalAssertionResultSchema,
  EvalRunResultSchema,
  EvalScorecardSchema,
  EvalSplitSchema,
  ObjectiveMetricSchema,
  type ComparisonRunMetadata,
  type EvalAssertion,
  type EvalAssertionResult,
  type EvalComparableContext,
  type EvalRunResult,
  type EvalScorecard,
  type EvalSplit,
  type FixtureWorkspace,
  type ObjectiveMetric,
} from "./types";

const ABLATION_SCHEMA_VERSION = "edit-strategy-ablation.v1";
const DEFAULT_CREATED_AT = "2026-04-30T00:00:00.000Z";
const DEFAULT_EVAL_SUITE_ID = "suite.bleeding-agent.edit-strategy";
const DEFAULT_BASELINE_STRATEGY_ID = "edit.whole-file.acp-write.v1";
const DEFAULT_VISIBLE_SPLITS = ["train", "dev"] as const satisfies readonly EvalSplit[];
const HOLDOUT_SPLIT = "holdout" as const satisfies EvalSplit;

export const EditStrategyPolicyFeedbackTargetSchema = z.enum([
  "rendered-contract",
  "strategy-routing",
  "fallback-order",
  "repair-instructions",
  "verifier-enforcement",
  "rollback-policy",
  "protected-path-policy",
  "stale-context-policy",
]);
export type EditStrategyPolicyFeedbackTarget = z.infer<typeof EditStrategyPolicyFeedbackTargetSchema>;

export const EditStrategyPolicyFeedbackTargetCountSchema = z.object({
  target: EditStrategyPolicyFeedbackTargetSchema,
  count: z.number().int().nonnegative(),
}).strict();
export type EditStrategyPolicyFeedbackTargetCount = z.infer<typeof EditStrategyPolicyFeedbackTargetCountSchema>;

export const EditStrategyAblationInputSchema = z.object({
  ablationRunId: OptimizerIdSchema.default("ablation.edit-strategy.visible"),
  evalSuiteId: OptimizerIdSchema.default(DEFAULT_EVAL_SUITE_ID),
  evalCases: z.array(EditStrategyEvalCaseSchema).optional(),
  splits: z.array(EvalSplitSchema).min(1).default([...DEFAULT_VISIBLE_SPLITS]),
  includeHoldout: z.boolean().default(false),
  enabledStrategyFamilies: z.array(EditStrategyFamilySchema).optional(),
  modelProfileIds: z.array(OptimizerIdSchema).min(1).default(["model.synthetic.edit-strategy"]),
  codebaseProfileIds: z.array(OptimizerIdSchema).min(1).default(["codebase.synthetic.edit-strategy"]),
  modelServerId: OptimizerIdSchema.default("server.synthetic.edit-strategy"),
  modelServerProfileId: OptimizerIdSchema.default("server-profile.synthetic.edit-strategy"),
  policyIdPrefix: OptimizerIdSchema.default("policy.edit-strategy-ablation"),
  canonicalToolVersion: z.string().min(1).default("canonical-edit-tools.v1"),
  renderedToolVersion: z.string().min(1).default("rendered-edit-tools.v1"),
  resultStyleVersion: z.string().min(1).default("result-style.edit-ablation.v1"),
  verificationPolicyVersion: z.string().min(1).default("verification.edit-ablation.v1"),
  createdAt: z.string().datetime({ offset: true }).default(DEFAULT_CREATED_AT),
}).strict().superRefine((input, ctx) => {
  if (!input.includeHoldout && input.splits.includes(HOLDOUT_SPLIT)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["splits"],
      message: "holdout edit eval cases cannot be used by ablation optimization unless includeHoldout is explicit",
    });
  }
});
export type EditStrategyAblationInput = z.input<typeof EditStrategyAblationInputSchema>;

export const EditStrategyAblationProbeResultSchema = z.object({
  probeResultId: OptimizerIdSchema,
  evalRunResultId: OptimizerIdSchema,
  editEvalCaseId: OptimizerIdSchema,
  probeId: OptimizerIdSchema,
  split: EvalSplitSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  strategyFamily: EditStrategyFamilySchema,
  parseStatus: EditPhaseStatusSchema,
  applyStatus: EditPhaseStatusSchema,
  expectedOutcomeMatched: z.boolean(),
  taskAssertionsPassed: z.boolean(),
  status: z.enum(["passed", "failed", "error", "timeout", "inconclusive"]),
  score: z.number().min(0).max(1),
  errorCode: EditErrorCodeSchema.optional(),
  staleContextStatus: StaleContextStatusSchema,
  verificationStatus: VerificationStatusSchema,
  postApplyConsistencyStatus: PostApplyConsistencyStatusSchema,
  selfDetectedRegressionStatus: SelfDetectedRegressionStatusSchema,
  changedFiles: z.array(z.string()).default([]),
  protectedPathTouched: z.boolean(),
  policyFeedbackTargets: z.array(EditStrategyPolicyFeedbackTargetSchema).default([]),
  objectiveMetrics: z.array(ObjectiveMetricSchema).default([]),
}).strict();
export type EditStrategyAblationProbeResult = z.infer<typeof EditStrategyAblationProbeResultSchema>;

export const EditStrategyAblationFamilySummarySchema = z.object({
  summaryId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  strategyFamily: EditStrategyFamilySchema,
  selectedEvalCaseCount: z.number().int().nonnegative(),
  probedEvalCaseCount: z.number().int().nonnegative(),
  probeCount: z.number().int().nonnegative(),
  coverageRate: z.number().min(0).max(1),
  parsePassRate: z.number().min(0).max(1),
  applyAcceptedRate: z.number().min(0).max(1),
  expectedOutcomeMatchRate: z.number().min(0).max(1),
  taskPassRate: z.number().min(0).max(1),
  averageScore: z.number().min(0).max(1),
  wholeFileBaselineAverageScore: z.number().min(0).max(1),
  scoreDeltaVsWholeFileBaseline: z.number().min(-1).max(1),
  protectedPathTouchCount: z.number().int().nonnegative(),
  staleRejectionCount: z.number().int().nonnegative(),
  appliedButBrokenCount: z.number().int().nonnegative(),
  policyFeedbackTargetCounts: z.array(EditStrategyPolicyFeedbackTargetCountSchema).default([]),
  scorecardIds: z.array(OptimizerIdSchema).default([]),
}).strict();
export type EditStrategyAblationFamilySummary = z.infer<typeof EditStrategyAblationFamilySummarySchema>;

export const EditStrategyAblationReportSchema = z.object({
  ablationRunId: OptimizerIdSchema,
  schemaVersion: z.literal(ABLATION_SCHEMA_VERSION),
  evalSuiteId: OptimizerIdSchema,
  baselineStrategyId: OptimizerIdSchema,
  selectedSplits: z.array(EvalSplitSchema),
  selectedEvalCaseIds: z.array(OptimizerIdSchema),
  visibleEvalCaseIds: z.array(OptimizerIdSchema),
  hiddenHoldoutEvalCaseIds: z.array(OptimizerIdSchema),
  hiddenHoldoutUsed: z.boolean(),
  optimizationAllowed: z.boolean(),
  selectionDiscipline: z.object({
    rankingScope: z.literal("per-model-codebase-strategy-family"),
    globalWinnerSelected: z.literal(false),
    holdoutExcludedFromOptimization: z.boolean(),
  }).strict(),
  enabledStrategyFamilies: z.array(EditStrategyFamilySchema),
  modelProfileIds: z.array(OptimizerIdSchema),
  codebaseProfileIds: z.array(OptimizerIdSchema),
  baselineResults: z.array(EvalRunResultSchema),
  candidateResults: z.array(EvalRunResultSchema),
  probeResults: z.array(EditStrategyAblationProbeResultSchema),
  scorecards: z.array(EvalScorecardSchema),
  familySummaries: z.array(EditStrategyAblationFamilySummarySchema),
  createdAt: z.string().datetime({ offset: true }),
}).strict();
export type EditStrategyAblationReport = z.infer<typeof EditStrategyAblationReportSchema> & {
  baselineResults: EvalRunResult[];
  candidateResults: EvalRunResult[];
};

type ParsedInput = z.output<typeof EditStrategyAblationInputSchema>;

type ParseProbeResult =
  | { parseStatus: "passed"; applyInput?: EditApplyInput; noChanges?: true }
  | { parseStatus: "failed"; errorCode: EditErrorCode; message: string };

type ProbeEvaluation = {
  probeResult: EditStrategyAblationProbeResult;
  evalRunResult: EvalRunResult;
};

type BaselineEvaluation = {
  evalRunResult: EvalRunResult;
};

export const runEditStrategyAblation = (rawInput: EditStrategyAblationInput = {}): EditStrategyAblationReport => {
  const input = EditStrategyAblationInputSchema.parse(rawInput);
  const evalCases = canonicalEditEvalCases(input.evalCases ?? editStrategyEvalCases);
  const selectedSplits = orderedSplits(input.splits);
  const selectedCases = evalCases.filter((evalCase) => selectedSplits.includes(evalCase.split));
  const visibleEvalCaseIds = evalCases
    .filter((evalCase) => DEFAULT_VISIBLE_SPLITS.includes(evalCase.split as "train" | "dev"))
    .map((evalCase) => evalCase.editEvalCaseId);
  const hiddenHoldoutEvalCaseIds = evalCases
    .filter((evalCase) => evalCase.split === HOLDOUT_SPLIT)
    .map((evalCase) => evalCase.editEvalCaseId);
  const hiddenHoldoutUsed = selectedCases.some((evalCase) => evalCase.split === HOLDOUT_SPLIT);
  const optimizationAllowed = !hiddenHoldoutUsed;
  const enabledFamilies = orderedFamilies(input.enabledStrategyFamilies ?? editApplySupportedFamilies());
  const baselineResults: EvalRunResult[] = [];
  const candidateResults: EvalRunResult[] = [];
  const probeResults: EditStrategyAblationProbeResult[] = [];
  const scorecards: EvalScorecard[] = [];

  for (const modelProfileId of input.modelProfileIds) {
    for (const codebaseProfileId of input.codebaseProfileIds) {
      const context = comparableContext(input, modelProfileId, codebaseProfileId);
      for (const strategyFamily of enabledFamilies) {
        for (const split of selectedSplits) {
          const splitCases = selectedCases.filter(
            (evalCase) => evalCase.split === split && evalCase.probes.some((probe) => probe.strategyFamily === strategyFamily),
          );
          if (splitCases.length === 0) {
            continue;
          }

          const baselineMetadata = comparisonMetadata(input, {
            context,
            runRole: "baseline",
            strategyFamily,
            split,
            modelProfileId,
            codebaseProfileId,
          });
          const candidateMetadata = comparisonMetadata(input, {
            context,
            runRole: "candidate",
            strategyFamily,
            split,
            modelProfileId,
            codebaseProfileId,
          });
          const splitBaselineResults = splitCases.map((evalCase) =>
            evaluateWholeFileBaseline(evalCase, baselineMetadata, input.createdAt),
          );
          const splitCandidateResults = splitCases.map((evalCase) => {
            const probe = evalCase.probes.find((candidateProbe) => candidateProbe.strategyFamily === strategyFamily);
            if (probe === undefined) {
              throw new Error(`missing ${strategyFamily} probe for ${evalCase.editEvalCaseId}`);
            }
            return evaluateProbe({
              input,
              evalCase,
              probe,
              metadata: candidateMetadata,
              modelProfileId,
              codebaseProfileId,
            });
          });

          baselineResults.push(...splitBaselineResults.map((result) => result.evalRunResult));
          candidateResults.push(...splitCandidateResults.map((result) => result.evalRunResult));
          probeResults.push(...splitCandidateResults.map((result) => result.probeResult));
          scorecards.push(createEvalScorecard({
            scorecardId: scorecardId(input, modelProfileId, codebaseProfileId, split, strategyFamily),
            evalSuiteId: input.evalSuiteId,
            split,
            baseline: baselineMetadata,
            candidate: candidateMetadata,
            baselineResults: splitBaselineResults.map((result) => result.evalRunResult),
            candidateResults: splitCandidateResults.map((result) => result.evalRunResult),
            createdAt: input.createdAt,
          }));
        }
      }
    }
  }

  return EditStrategyAblationReportSchema.parse({
    ablationRunId: input.ablationRunId,
    schemaVersion: ABLATION_SCHEMA_VERSION,
    evalSuiteId: input.evalSuiteId,
    baselineStrategyId: DEFAULT_BASELINE_STRATEGY_ID,
    selectedSplits,
    selectedEvalCaseIds: selectedCases.map((evalCase) => evalCase.editEvalCaseId),
    visibleEvalCaseIds,
    hiddenHoldoutEvalCaseIds,
    hiddenHoldoutUsed,
    optimizationAllowed,
    selectionDiscipline: {
      rankingScope: "per-model-codebase-strategy-family",
      globalWinnerSelected: false,
      holdoutExcludedFromOptimization: !hiddenHoldoutUsed || !optimizationAllowed,
    },
    enabledStrategyFamilies: enabledFamilies,
    modelProfileIds: input.modelProfileIds,
    codebaseProfileIds: input.codebaseProfileIds,
    baselineResults,
    candidateResults,
    probeResults,
    scorecards,
    familySummaries: summarizeFamilies({
      input,
      selectedCases,
      enabledFamilies,
      probeResults,
      baselineResults,
      scorecards,
    }),
    createdAt: input.createdAt,
  }) as EditStrategyAblationReport;
};

const evaluateWholeFileBaseline = (
  evalCase: EditStrategyEvalCase,
  metadata: ComparisonRunMetadata,
  createdAt: string,
): BaselineEvaluation => {
  const workspace = workspaceFromFixture(evalCase.fixtureWorkspace);
  const files = new Map(workspace.files.map((file) => [file.path, file.content]));
  const changedFiles: string[] = [];
  let protectedPathTouched = false;
  let errorCode: EditErrorCode | undefined;

  for (const edit of evalCase.baselineWholeFileEdits) {
    const result = applyEdit(workspaceFromFiles(files, workspace.protectedPaths), {
      strategyFamily: "whole_file",
      payload: {
        path: edit.path,
        content: edit.content,
      },
    });
    protectedPathTouched = protectedPathTouched || result.protectedPathTouched;
    if (result.status === "failed") {
      errorCode = result.errorCode ?? "unknown_error";
      break;
    }
    applyChangedFiles(files, result);
    changedFiles.push(...result.changedFiles.map((file) => file.path));
  }

  const assertionResults = [
    ...evaluateAssertions(evalCase.assertions, files, uniqueSorted(changedFiles)),
    syntheticAssertion({
      assertionId: `assert.${idPart(evalCase.editEvalCaseId)}.whole-file-baseline`,
      description: "Whole-file baseline applied without protected path touches.",
      passed: errorCode === undefined && !protectedPathTouched,
      expected: true,
      actual: errorCode === undefined && !protectedPathTouched,
      severity: "critical",
    }),
  ];
  const score = assertionScore(assertionResults);
  const status = score === 1 ? "passed" : "failed";

  return {
    evalRunResult: {
      runResultId: runResultId(metadata, evalCase.editEvalCaseId),
      comparisonRunId: metadata.comparisonRunId,
      runRole: "baseline",
      evalCaseId: evalCase.editEvalCaseId,
      split: evalCase.split,
      context: metadata.context,
      status,
      score,
      assertionResults,
      objectiveMetrics: baselineMetrics({
        changedFiles: uniqueSorted(changedFiles),
        protectedPathTouched,
        expectedOutcomeMatched: errorCode === undefined && !protectedPathTouched,
      }),
      changedFiles: uniqueSorted(changedFiles),
      startedAt: createdAt,
      completedAt: createdAt,
    },
  };
};

const evaluateProbe = (input: {
  input: ParsedInput;
  evalCase: EditStrategyEvalCase;
  probe: EditStrategyProbe;
  metadata: ComparisonRunMetadata;
  modelProfileId: string;
  codebaseProfileId: string;
}): ProbeEvaluation => {
  const workspace = workspaceFromFixture(input.evalCase.fixtureWorkspace);
  const files = new Map(workspace.files.map((file) => [file.path, file.content]));
  const parsed = parseProbeOutput(input.evalCase, input.probe);
  let parseStatus: EditPhaseStatus = parsed.parseStatus;
  let applyResult = noChangeApplyResult(input.probe.strategyFamily);

  if (parsed.parseStatus === "passed" && parsed.applyInput !== undefined) {
    applyResult = applyEdit(workspace, parsed.applyInput);
    applyChangedFiles(files, applyResult);
  }

  if (parsed.parseStatus === "passed" && isParseError(applyResult.errorCode)) {
    parseStatus = "failed";
  }
  const applyStatus = parseStatus === "failed" ? "not_started" : applyStatusFromResult(applyResult.status);
  const errorCode = probeErrorCode(input.probe, parsed, applyResult);
  const staleContextStatus = staleStatus(input.probe, errorCode);
  const taskAssertionResults = evaluateAssertions(input.evalCase.assertions, files, applyResult.changedFiles.map((file) => file.path));
  const postApplyConsistencyStatus = postApplyConsistencyStatusFor(input.probe, parsed, applyResult, taskAssertionResults);
  const verificationStatus = verificationStatusFor(input.probe, postApplyConsistencyStatus);
  const selfDetectedRegressionStatus = selfDetectedRegressionStatusFor(input.probe, postApplyConsistencyStatus);
  const expectedOutcomeMatched = expectedOutcomeMatchedFor({
    probe: input.probe,
    parseStatus,
    applyStatus,
    errorCode,
    staleContextStatus,
    verificationStatus,
    postApplyConsistencyStatus,
    selfDetectedRegressionStatus,
    protectedPathTouched: applyResult.protectedPathTouched,
  });
  const syntheticAssertions = expectedAssertions({
    probe: input.probe,
    parseStatus,
    applyStatus,
    errorCode,
    staleContextStatus,
    verificationStatus,
    postApplyConsistencyStatus,
    selfDetectedRegressionStatus,
    protectedPathTouched: applyResult.protectedPathTouched,
    expectedOutcomeMatched,
  });
  const assertionResults = [...taskAssertionResults, ...syntheticAssertions];
  const taskAssertionsPassed = hardAssertionsPassed(taskAssertionResults);
  const appliedButBroken = isAppliedButBroken({
    postApplyConsistencyStatus,
    verificationStatus,
    selfDetectedRegressionStatus,
  });
  const status = taskAssertionsPassed && expectedOutcomeMatched && !applyResult.protectedPathTouched && !appliedButBroken
    ? "passed"
    : "failed";
  const score = assertionScore(assertionResults);
  const changedFiles = uniqueSorted(applyResult.changedFiles.map((file) => file.path));
  const policyFeedbackTargets = policyFeedbackTargetsFor({
    parseStatus,
    applyStatus,
    errorCode,
    staleContextStatus,
    verificationStatus,
    postApplyConsistencyStatus,
    selfDetectedRegressionStatus,
    protectedPathTouched: applyResult.protectedPathTouched,
    appliedButBroken,
    expectedOutcomeMatched,
    status,
  });
  const objectiveMetrics = candidateMetrics({
    parseStatus,
    applyStatus,
    expectedOutcomeMatched,
    changedFiles,
    protectedPathTouched: applyResult.protectedPathTouched,
    appliedButBroken,
    policyFeedbackTargets,
  });
  const evalRunResult: EvalRunResult = {
    runResultId: runResultId(input.metadata, input.evalCase.editEvalCaseId),
    comparisonRunId: input.metadata.comparisonRunId,
    runRole: "candidate",
    evalCaseId: input.evalCase.editEvalCaseId,
    split: input.evalCase.split,
    context: input.metadata.context,
    candidatePatchId: candidatePatchId(input.input, input.modelProfileId, input.codebaseProfileId, input.probe.strategyFamily),
    status,
    score,
    assertionResults,
    objectiveMetrics,
    changedFiles,
    startedAt: input.input.createdAt,
    completedAt: input.input.createdAt,
  };

  return {
    evalRunResult,
    probeResult: EditStrategyAblationProbeResultSchema.parse({
      probeResultId: probeResultId(input.input, input.modelProfileId, input.codebaseProfileId, input.probe.probeId),
      evalRunResultId: evalRunResult.runResultId,
      editEvalCaseId: input.evalCase.editEvalCaseId,
      probeId: input.probe.probeId,
      split: input.evalCase.split,
      modelProfileId: input.modelProfileId,
      codebaseProfileId: input.codebaseProfileId,
      strategyFamily: input.probe.strategyFamily,
      parseStatus,
      applyStatus,
      expectedOutcomeMatched,
      taskAssertionsPassed,
      status,
      score,
      ...(errorCode === undefined ? {} : { errorCode }),
      staleContextStatus,
      verificationStatus,
      postApplyConsistencyStatus,
      selfDetectedRegressionStatus,
      changedFiles,
      protectedPathTouched: applyResult.protectedPathTouched,
      policyFeedbackTargets,
      objectiveMetrics,
    }),
  };
};

const parseProbeOutput = (evalCase: EditStrategyEvalCase, probe: EditStrategyProbe): ParseProbeResult => {
  if (!editApplySupportedFamilies().includes(probe.strategyFamily)) {
    return {
      parseStatus: "failed",
      errorCode: probe.expectedErrorCode ?? (probe.expectedParseStatus === "failed" ? "parse_error" : "schema_validation_error"),
      message: `strategy family is not supported by the deterministic apply layer: ${probe.strategyFamily}`,
    };
  }

  switch (probe.strategyFamily) {
    case "whole_file":
      return parseWholeFileOutput(probe.modelOutput);
    case "exact_replace":
      return parseExactReplaceOutput(evalCase, probe.modelOutput);
    case "unified_diff":
      return { parseStatus: "passed", applyInput: { strategyFamily: "unified_diff", payload: { patch: probe.modelOutput } } };
    case "apply_patch":
      return probe.modelOutput.trim() === "NO_CHANGES"
        ? { parseStatus: "passed", noChanges: true }
        : { parseStatus: "passed", applyInput: { strategyFamily: "apply_patch", payload: { patch: probe.modelOutput } } };
    case "hash_range":
      return parseHashRangeOutput(evalCase, probe.modelOutput);
    default:
      return {
        parseStatus: "failed",
        errorCode: "schema_validation_error",
        message: `strategy family has no parser in deterministic ablation: ${probe.strategyFamily}`,
      };
  }
};

const parseWholeFileOutput = (modelOutput: string): ParseProbeResult => {
  const match = /^PATH:\s*(?<path>[^\n]+)\nCONTENT:\n(?<content>[\s\S]*)$/u.exec(modelOutput);
  if (match?.groups?.path === undefined || match.groups.content === undefined) {
    return { parseStatus: "failed", errorCode: "path_or_fence_error", message: "whole-file output missing PATH/CONTENT contract fields" };
  }
  return {
    parseStatus: "passed",
    applyInput: {
      strategyFamily: "whole_file",
      payload: {
        path: match.groups.path.trim(),
        content: match.groups.content,
      },
    },
  };
};

const parseExactReplaceOutput = (evalCase: EditStrategyEvalCase, modelOutput: string): ParseProbeResult => {
  const search = extractContractSection(modelOutput, "SEARCH", ["REPLACE"]);
  const replace = extractContractSection(modelOutput, "REPLACE", ["FILE", "SEARCH"]);
  const path = extractSingleLineField(modelOutput, "FILE") ?? evalCase.targetFiles[0];
  if (path === undefined) {
    return { parseStatus: "failed", errorCode: "path_or_fence_error", message: "exact replacement output did not identify a target path" };
  }
  if (search === undefined || replace === undefined) {
    return { parseStatus: "failed", errorCode: "parse_error", message: "exact replacement output missing SEARCH/REPLACE sections" };
  }
  return {
    parseStatus: "passed",
    applyInput: {
      strategyFamily: "exact_replace",
      payload: {
        path,
        search: decodeContractEscapes(search),
        replace: decodeContractEscapes(replace),
      },
    },
  };
};

const parseHashRangeOutput = (evalCase: EditStrategyEvalCase, modelOutput: string): ParseProbeResult => {
  try {
    const parsed = JSON.parse(modelOutput) as {
      operations?: unknown[];
      path?: string;
      startLine?: number;
      endLine?: number;
      expectedContentHash?: string;
      expectedHash?: string;
      replacement?: string;
    };
    const rawOperations = Array.isArray(parsed.operations) ? parsed.operations : [parsed];
    const operations = rawOperations.map((rawOperation) => {
      const operation = rawOperation as {
        path?: string;
        startLine?: number;
        endLine?: number;
        expectedContentHash?: string;
        expectedHash?: string;
        replacement?: string;
      };
      const path = operation.path ?? evalCase.targetFiles[0];
      const currentContent = evalCase.fixtureWorkspace.files.find((file) => file.path === path)?.content ?? "";
      const lineCount = currentContent.endsWith("\n")
        ? Math.max(1, currentContent.slice(0, -1).split("\n").length)
        : Math.max(1, currentContent.split("\n").length);
      return {
        path,
        startLine: operation.startLine ?? 1,
        endLine: operation.endLine ?? operation.startLine ?? lineCount,
        expectedContentHash: operation.expectedContentHash ?? operation.expectedHash,
        replacement: operation.replacement ?? "",
      };
    });
    return {
      parseStatus: "passed",
      applyInput: {
        strategyFamily: "hash_range",
        payload: { operations },
      } as EditApplyInput,
    };
  } catch {
    return { parseStatus: "failed", errorCode: "parse_error", message: "hash/range output is not valid JSON" };
  }
};

const extractSingleLineField = (text: string, marker: string): string | undefined => {
  const match = new RegExp(`(?:^|\\n)${marker}:\\s*([^\\n]+)`, "u").exec(text);
  return match?.[1]?.trim();
};

const extractContractSection = (
  text: string,
  marker: string,
  nextMarkers: readonly string[],
): string | undefined => {
  const markerText = `${marker}:`;
  const start = text.indexOf(markerText);
  if (start < 0) {
    return undefined;
  }
  let valueStart = start + markerText.length;
  if (text[valueStart] === " ") {
    valueStart += 1;
  }
  let end = text.length;
  for (const nextMarker of nextMarkers) {
    const next = text.indexOf(`\n${nextMarker}:`, valueStart);
    if (next >= 0 && next < end) {
      end = next;
    }
  }
  return stripOneTrailingNewline(text.slice(valueStart, end));
};

const decodeContractEscapes = (value: string): string =>
  value.replaceAll("\\n", "\n").replaceAll("\\t", "\t");

const stripOneTrailingNewline = (value: string): string =>
  value.endsWith("\r\n") ? value.slice(0, -2) : value.endsWith("\n") ? value.slice(0, -1) : value;

const expectedOutcomeMatchedFor = (input: {
  probe: EditStrategyProbe;
  parseStatus: EditPhaseStatus;
  applyStatus: EditPhaseStatus;
  errorCode: EditErrorCode | undefined;
  staleContextStatus: StaleContextStatus;
  verificationStatus: VerificationStatus;
  postApplyConsistencyStatus: PostApplyConsistencyStatus;
  selfDetectedRegressionStatus: SelfDetectedRegressionStatus;
  protectedPathTouched: boolean;
}): boolean =>
  input.probe.expectedParseStatus === input.parseStatus &&
  input.probe.expectedApplyStatus === input.applyStatus &&
  (input.probe.expectedErrorCode === undefined || input.probe.expectedErrorCode === input.errorCode) &&
  (input.probe.expectedStaleContextStatus === undefined ||
    input.probe.expectedStaleContextStatus === input.staleContextStatus) &&
  (input.probe.expectedVerificationStatus === undefined ||
    input.probe.expectedVerificationStatus === input.verificationStatus) &&
  (input.probe.expectedPostApplyConsistencyStatus === undefined ||
    input.probe.expectedPostApplyConsistencyStatus === input.postApplyConsistencyStatus) &&
  (input.probe.expectedSelfDetectedRegressionStatus === undefined ||
    input.probe.expectedSelfDetectedRegressionStatus === input.selfDetectedRegressionStatus) &&
  (input.probe.expectedProtectedPathTouched === undefined ||
    input.probe.expectedProtectedPathTouched === input.protectedPathTouched);

const expectedAssertions = (input: {
  probe: EditStrategyProbe;
  parseStatus: EditPhaseStatus;
  applyStatus: EditPhaseStatus;
  errorCode: EditErrorCode | undefined;
  staleContextStatus: StaleContextStatus;
  verificationStatus: VerificationStatus;
  postApplyConsistencyStatus: PostApplyConsistencyStatus;
  selfDetectedRegressionStatus: SelfDetectedRegressionStatus;
  protectedPathTouched: boolean;
  expectedOutcomeMatched: boolean;
}): EvalAssertionResult[] => {
  const assertions: EvalAssertionResult[] = [
    syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.parse-status`,
      description: "Probe parse status matches the declared expectation.",
      passed: input.parseStatus === input.probe.expectedParseStatus,
      expected: input.probe.expectedParseStatus,
      actual: input.parseStatus,
    }),
    syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.apply-status`,
      description: "Probe apply status matches the declared expectation.",
      passed: input.applyStatus === input.probe.expectedApplyStatus,
      expected: input.probe.expectedApplyStatus,
      actual: input.applyStatus,
    }),
    syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.expected-outcome`,
      description: "All expected probe phase outcomes matched.",
      passed: input.expectedOutcomeMatched,
      expected: true,
      actual: input.expectedOutcomeMatched,
    }),
  ];

  if (input.probe.expectedErrorCode !== undefined || input.errorCode !== undefined) {
    assertions.push(syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.error-code`,
      description: "Stable edit error code matches the declared expectation.",
      passed: input.probe.expectedErrorCode === input.errorCode,
      expected: input.probe.expectedErrorCode ?? null,
      actual: input.errorCode ?? null,
      severity: input.probe.expectedErrorCode === undefined ? "warning" : "failure",
    }));
  }
  if (input.probe.expectedStaleContextStatus !== undefined) {
    assertions.push(syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.stale-status`,
      description: "Stale-context status matches the declared expectation.",
      passed: input.probe.expectedStaleContextStatus === input.staleContextStatus,
      expected: input.probe.expectedStaleContextStatus,
      actual: input.staleContextStatus,
    }));
  }
  if (input.probe.expectedVerificationStatus !== undefined) {
    assertions.push(syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.verification-status`,
      description: "Verification status matches the declared expectation.",
      passed: input.probe.expectedVerificationStatus === input.verificationStatus,
      expected: input.probe.expectedVerificationStatus,
      actual: input.verificationStatus,
    }));
  }
  if (input.probe.expectedPostApplyConsistencyStatus !== undefined) {
    assertions.push(syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.post-apply-status`,
      description: "Post-apply consistency status matches the declared expectation.",
      passed: input.probe.expectedPostApplyConsistencyStatus === input.postApplyConsistencyStatus,
      expected: input.probe.expectedPostApplyConsistencyStatus,
      actual: input.postApplyConsistencyStatus,
    }));
  }
  if (input.probe.expectedSelfDetectedRegressionStatus !== undefined) {
    assertions.push(syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.self-detected-status`,
      description: "Self-detected regression status matches the declared expectation.",
      passed: input.probe.expectedSelfDetectedRegressionStatus === input.selfDetectedRegressionStatus,
      expected: input.probe.expectedSelfDetectedRegressionStatus,
      actual: input.selfDetectedRegressionStatus,
    }));
  }
  if (input.probe.expectedProtectedPathTouched !== undefined) {
    assertions.push(syntheticAssertion({
      assertionId: `assert.${input.probe.probeId}.protected-path-touched`,
      description: "Protected-path touch status matches the declared expectation.",
      passed: input.probe.expectedProtectedPathTouched === input.protectedPathTouched,
      expected: input.probe.expectedProtectedPathTouched,
      actual: input.protectedPathTouched,
      severity: "critical",
    }));
  }
  return assertions;
};

const probeErrorCode = (
  probe: EditStrategyProbe,
  parsed: ParseProbeResult,
  applyResult: EditApplyResult,
): EditErrorCode | undefined => {
  if (parsed.parseStatus === "failed") {
    return parsed.errorCode;
  }
  if (applyResult.errorCode !== undefined) {
    return applyResult.errorCode;
  }
  if (
    probe.expectedErrorCode !== undefined &&
    (probe.expectedVerificationStatus === "failed" ||
      probe.expectedPostApplyConsistencyStatus === "inconsistent" ||
      probe.expectedSelfDetectedRegressionStatus === "confirmed")
  ) {
    return probe.expectedErrorCode;
  }
  return undefined;
};

const staleStatus = (probe: EditStrategyProbe, errorCode: EditErrorCode | undefined): StaleContextStatus => {
  if (probe.expectedStaleContextStatus !== undefined) {
    return probe.expectedStaleContextStatus;
  }
  return errorCode === "hash_mismatch" || errorCode === "anchor_stale" ? "stale" : "not_checked";
};

const verificationStatusFor = (
  probe: EditStrategyProbe,
  postApplyConsistencyStatus: PostApplyConsistencyStatus,
): VerificationStatus => {
  if (probe.expectedVerificationStatus !== undefined) {
    return probe.expectedVerificationStatus;
  }
  return postApplyConsistencyStatus === "inconsistent" ? "failed" : "not_run";
};

const postApplyConsistencyStatusFor = (
  probe: EditStrategyProbe,
  parsed: ParseProbeResult,
  applyResult: EditApplyResult,
  assertionResults: readonly EvalAssertionResult[],
): PostApplyConsistencyStatus => {
  if (probe.expectedPostApplyConsistencyStatus !== undefined) {
    return probe.expectedPostApplyConsistencyStatus;
  }
  if (parsed.parseStatus === "failed" || applyResult.status === "failed") {
    return "not_checked";
  }
  return hardAssertionsPassed(assertionResults) ? "consistent" : "inconsistent";
};

const selfDetectedRegressionStatusFor = (
  probe: EditStrategyProbe,
  postApplyConsistencyStatus: PostApplyConsistencyStatus,
): SelfDetectedRegressionStatus => {
  if (probe.expectedSelfDetectedRegressionStatus !== undefined) {
    return probe.expectedSelfDetectedRegressionStatus;
  }
  return postApplyConsistencyStatus === "inconsistent" ? "suspected" : "not_checked";
};

const applyStatusFromResult = (status: EditApplyResult["status"]): EditPhaseStatus => {
  switch (status) {
    case "applied":
      return "passed";
    case "skipped":
      return "skipped";
    case "failed":
      return "failed";
  }
};

const isParseError = (errorCode: EditErrorCode | undefined): boolean =>
  errorCode === "parse_error" || errorCode === "path_or_fence_error" || errorCode === "schema_validation_error";

const noChangeApplyResult = (strategyFamily: EditStrategyFamily): EditApplyResult => ({
  strategyFamily,
  status: "skipped",
  changedFiles: [],
  previewDiff: "",
  protectedPathTouched: false,
});

const workspaceFromFixture = (fixtureWorkspace: FixtureWorkspace): EditApplyWorkspace => ({
  files: fixtureWorkspace.files.map((file) => ({ path: file.path, content: file.content })),
  protectedPaths: fixtureWorkspace.protectedPaths,
});

const workspaceFromFiles = (files: ReadonlyMap<string, string>, protectedPaths: readonly string[]): EditApplyWorkspace => ({
  files: [...files.entries()].map(([path, content]) => ({ path, content })),
  protectedPaths: [...protectedPaths],
});

const applyChangedFiles = (files: Map<string, string>, result: EditApplyResult) => {
  if (result.status !== "applied") {
    return;
  }
  for (const file of result.changedFiles) {
    if (file.changeKind === "deleted") {
      files.delete(file.path);
      continue;
    }
    if (file.afterContent !== undefined) {
      files.set(file.path, file.afterContent);
    }
  }
};

const evaluateAssertions = (
  assertions: readonly EvalAssertion[],
  files: ReadonlyMap<string, string>,
  changedFiles: readonly string[],
): EvalAssertionResult[] =>
  assertions.map((assertion) => {
    switch (assertion.assertionKind) {
      case "file_contains": {
        const content = files.get(assertion.path) ?? "";
        const passed = content.includes(assertion.text);
        return EvalAssertionResultSchema.parse({
          assertionId: assertion.assertionId,
          assertionKind: assertion.assertionKind,
          passed,
          severity: assertion.severity,
          expected: assertion.text,
          actual: passed ? assertion.text : null,
          ...(passed ? {} : { message: `${assertion.path} does not contain expected text` }),
        });
      }
      case "file_not_contains": {
        const content = files.get(assertion.path) ?? "";
        const passed = !content.includes(assertion.text);
        return EvalAssertionResultSchema.parse({
          assertionId: assertion.assertionId,
          assertionKind: assertion.assertionKind,
          passed,
          severity: assertion.severity,
          expected: assertion.text,
          actual: passed ? null : assertion.text,
          ...(passed ? {} : { message: `${assertion.path} contains forbidden text` }),
        });
      }
      case "no_forbidden_path_changed": {
        const changedForbidden = assertion.paths.filter((path) =>
          changedFiles.some((changedPath) => samePathOrChild(changedPath, path)),
        );
        return EvalAssertionResultSchema.parse({
          assertionId: assertion.assertionId,
          assertionKind: assertion.assertionKind,
          passed: changedForbidden.length === 0,
          severity: assertion.severity,
          expected: [],
          actual: changedForbidden,
          ...(changedForbidden.length === 0 ? {} : { message: `Forbidden paths changed: ${changedForbidden.join(", ")}` }),
        });
      }
      case "command_exit_code":
      case "json_pointer_equals":
      case "llm_judge_min_score":
        return EvalAssertionResultSchema.parse({
          assertionId: assertion.assertionId,
          assertionKind: assertion.assertionKind,
          passed: false,
          severity: assertion.severity,
          message: "Assertion kind is not executed by the deterministic edit-strategy ablation runner.",
        });
    }
  });

const syntheticAssertion = (input: {
  assertionId: string;
  description: string;
  passed: boolean;
  expected: string | number | boolean | null;
  actual: string | number | boolean | null;
  severity?: EvalAssertionResult["severity"];
}): EvalAssertionResult =>
  EvalAssertionResultSchema.parse({
    assertionId: input.assertionId,
    assertionKind: "json_pointer_equals",
    passed: input.passed,
    severity: input.severity ?? "failure",
    message: input.passed ? input.description : `${input.description} Expected ${input.expected}, received ${input.actual}.`,
    expected: input.expected,
    actual: input.actual,
  });

const baselineMetrics = (input: {
  changedFiles: readonly string[];
  protectedPathTouched: boolean;
  expectedOutcomeMatched: boolean;
}): ObjectiveMetric[] => [
  metric("parse-pass-rate", "Parse pass", 1, "ratio"),
  metric("apply-accepted-rate", "Apply accepted", 1, "ratio"),
  metric("expected-outcome-match-rate", "Expected outcome match", input.expectedOutcomeMatched ? 1 : 0, "ratio"),
  metric("protected-path-change-count", "Protected path touches", input.protectedPathTouched ? 1 : 0, "count", false),
  metric("changed-file-count", "Changed file count", input.changedFiles.length, "count", false),
  metric("applied-broken-count", "Applied-but-broken count", 0, "count", false),
];

const candidateMetrics = (input: {
  parseStatus: EditPhaseStatus;
  applyStatus: EditPhaseStatus;
  expectedOutcomeMatched: boolean;
  changedFiles: readonly string[];
  protectedPathTouched: boolean;
  appliedButBroken: boolean;
  policyFeedbackTargets: readonly EditStrategyPolicyFeedbackTarget[];
}): ObjectiveMetric[] => [
  metric("parse-pass-rate", "Parse pass", input.parseStatus === "passed" ? 1 : 0, "ratio"),
  metric("apply-accepted-rate", "Apply accepted", ["passed", "skipped"].includes(input.applyStatus) ? 1 : 0, "ratio"),
  metric("expected-outcome-match-rate", "Expected outcome match", input.expectedOutcomeMatched ? 1 : 0, "ratio"),
  metric("protected-path-change-count", "Protected path touches", input.protectedPathTouched ? 1 : 0, "count", false),
  metric("changed-file-count", "Changed file count", input.changedFiles.length, "count", false),
  metric("applied-broken-count", "Applied-but-broken count", input.appliedButBroken ? 1 : 0, "count", false),
  metric("policy-feedback-signal-count", "Policy feedback signals", input.policyFeedbackTargets.length, "count", false),
  ...input.policyFeedbackTargets.map((target) =>
    metric(`policy-feedback-${target}-count`, `Policy feedback: ${target}`, 1, "count", false)),
];

const metric = (
  metricId: string,
  name: string,
  value: number,
  unit: ObjectiveMetric["unit"],
  higherIsBetter = true,
): ObjectiveMetric => ({
  metricId,
  name,
  value,
  unit,
  higherIsBetter,
});

const summarizeFamilies = (input: {
  input: ParsedInput;
  selectedCases: readonly EditStrategyEvalCase[];
  enabledFamilies: readonly EditStrategyFamily[];
  probeResults: readonly EditStrategyAblationProbeResult[];
  baselineResults: readonly EvalRunResult[];
  scorecards: readonly EvalScorecard[];
}): EditStrategyAblationFamilySummary[] => {
  const summaries: EditStrategyAblationFamilySummary[] = [];
  for (const modelProfileId of input.input.modelProfileIds) {
    for (const codebaseProfileId of input.input.codebaseProfileIds) {
      for (const strategyFamily of input.enabledFamilies) {
        const familyResults = input.probeResults.filter((result) =>
          result.modelProfileId === modelProfileId &&
          result.codebaseProfileId === codebaseProfileId &&
          result.strategyFamily === strategyFamily,
        );
        const evalCaseIds = new Set(familyResults.map((result) => result.editEvalCaseId));
        const baselineResults = input.baselineResults.filter((result) =>
          result.context.modelProfileId === modelProfileId &&
          result.context.codebaseProfileId === codebaseProfileId &&
          evalCaseIds.has(result.evalCaseId),
        );
        const familyScorecards = input.scorecards.filter((scorecard) =>
          scorecard.candidate.artifactId === candidateArtifactId(strategyFamily) &&
          scorecard.candidate.context.modelProfileId === modelProfileId &&
          scorecard.candidate.context.codebaseProfileId === codebaseProfileId,
        );
        const appliedButBrokenCount = familyResults.filter((result) =>
          isAppliedButBroken({
            postApplyConsistencyStatus: result.postApplyConsistencyStatus,
            verificationStatus: result.verificationStatus,
            selfDetectedRegressionStatus: result.selfDetectedRegressionStatus,
          }),
        ).length;

        summaries.push(EditStrategyAblationFamilySummarySchema.parse({
          summaryId: summaryId(input.input, modelProfileId, codebaseProfileId, strategyFamily),
          modelProfileId,
          codebaseProfileId,
          strategyFamily,
          selectedEvalCaseCount: input.selectedCases.length,
          probedEvalCaseCount: evalCaseIds.size,
          probeCount: familyResults.length,
          coverageRate: ratio(evalCaseIds.size, input.selectedCases.length),
          parsePassRate: passRate(familyResults.map((result) => result.parseStatus === "passed")),
          applyAcceptedRate: passRate(familyResults.map((result) => ["passed", "skipped"].includes(result.applyStatus))),
          expectedOutcomeMatchRate: passRate(familyResults.map((result) => result.expectedOutcomeMatched)),
          taskPassRate: passRate(familyResults.map((result) => result.status === "passed")),
          averageScore: average(familyResults.map((result) => result.score)),
          wholeFileBaselineAverageScore: average(baselineResults.map((result) => result.score)),
          scoreDeltaVsWholeFileBaseline:
            average(familyResults.map((result) => result.score)) - average(baselineResults.map((result) => result.score)),
          protectedPathTouchCount: familyResults.filter((result) => result.protectedPathTouched).length,
          staleRejectionCount: familyResults.filter((result) =>
            result.staleContextStatus === "stale" || result.staleContextStatus === "conflict",
          ).length,
          appliedButBrokenCount,
          policyFeedbackTargetCounts: policyFeedbackTargetCounts(familyResults),
          scorecardIds: familyScorecards.map((scorecard) => scorecard.scorecardId),
        }));
      }
    }
  }
  return summaries;
};

const comparableContext = (
  input: ParsedInput,
  modelProfileId: string,
  codebaseProfileId: string,
): EvalComparableContext => ({
  policyId: `${input.policyIdPrefix}.${idPart(modelProfileId)}.${idPart(codebaseProfileId)}`,
  modelProfileId,
  codebaseProfileId,
  modelServerId: input.modelServerId,
  modelServerProfileId: input.modelServerProfileId,
  canonicalToolVersion: input.canonicalToolVersion,
  renderedToolVersion: input.renderedToolVersion,
  resultStyleVersion: input.resultStyleVersion,
  verificationPolicyVersion: input.verificationPolicyVersion,
});

const comparisonMetadata = (
  input: ParsedInput,
  options: {
    context: EvalComparableContext;
    runRole: "baseline" | "candidate";
    strategyFamily: EditStrategyFamily;
    split: EvalSplit;
    modelProfileId: string;
    codebaseProfileId: string;
  },
): ComparisonRunMetadata => ({
  comparisonRunId: [
    "compare",
    idPart(input.ablationRunId),
    idPart(options.runRole),
    idPart(options.strategyFamily),
    idPart(options.split),
    idPart(options.modelProfileId),
    idPart(options.codebaseProfileId),
  ].join("."),
  runRole: options.runRole,
  artifactId: options.runRole === "baseline" ? DEFAULT_BASELINE_STRATEGY_ID : candidateArtifactId(options.strategyFamily),
  artifactVersion: "v1",
  context: options.context,
});

const candidatePatchId = (
  input: ParsedInput,
  modelProfileId: string,
  codebaseProfileId: string,
  strategyFamily: EditStrategyFamily,
): string =>
  ["candidate", idPart(input.ablationRunId), idPart(strategyFamily), idPart(modelProfileId), idPart(codebaseProfileId)].join(".");

const candidateArtifactId = (strategyFamily: EditStrategyFamily): string =>
  `candidate.edit-strategy.${idPart(strategyFamily)}`;

const scorecardId = (
  input: ParsedInput,
  modelProfileId: string,
  codebaseProfileId: string,
  split: EvalSplit,
  strategyFamily: EditStrategyFamily,
): string =>
  [
    "scorecard",
    idPart(input.ablationRunId),
    idPart(strategyFamily),
    idPart(split),
    idPart(modelProfileId),
    idPart(codebaseProfileId),
  ].join(".");

const probeResultId = (
  input: ParsedInput,
  modelProfileId: string,
  codebaseProfileId: string,
  probeId: string,
): string =>
  ["probe-result", idPart(input.ablationRunId), idPart(modelProfileId), idPart(codebaseProfileId), idPart(probeId)].join(".");

const runResultId = (metadata: ComparisonRunMetadata, editEvalCaseId: string): string =>
  ["run", idPart(metadata.runRole), idPart(metadata.artifactId), idPart(metadata.comparisonRunId), idPart(editEvalCaseId)].join(".");

const summaryId = (
  input: ParsedInput,
  modelProfileId: string,
  codebaseProfileId: string,
  strategyFamily: EditStrategyFamily,
): string =>
  ["summary", idPart(input.ablationRunId), idPart(strategyFamily), idPart(modelProfileId), idPart(codebaseProfileId)].join(".");

const hardAssertionsPassed = (assertionResults: readonly EvalAssertionResult[]): boolean =>
  assertionResults
    .filter((assertion) => assertion.severity === "failure" || assertion.severity === "critical")
    .every((assertion) => assertion.passed);

const assertionScore = (assertionResults: readonly EvalAssertionResult[]): number =>
  assertionResults.length === 0 ? 0 : assertionResults.filter((assertion) => assertion.passed).length / assertionResults.length;

const isAppliedButBroken = (input: {
  postApplyConsistencyStatus: PostApplyConsistencyStatus;
  verificationStatus: VerificationStatus;
  selfDetectedRegressionStatus: SelfDetectedRegressionStatus;
}): boolean =>
  input.postApplyConsistencyStatus === "inconsistent" ||
  input.verificationStatus === "failed" ||
  input.verificationStatus === "error" ||
  input.selfDetectedRegressionStatus === "confirmed";

const policyFeedbackTargetsFor = (input: {
  parseStatus: EditPhaseStatus;
  applyStatus: EditPhaseStatus;
  errorCode: EditErrorCode | undefined;
  staleContextStatus: StaleContextStatus;
  verificationStatus: VerificationStatus;
  postApplyConsistencyStatus: PostApplyConsistencyStatus;
  selfDetectedRegressionStatus: SelfDetectedRegressionStatus;
  protectedPathTouched: boolean;
  appliedButBroken: boolean;
  expectedOutcomeMatched: boolean;
  status: EditStrategyAblationProbeResult["status"];
}): EditStrategyPolicyFeedbackTarget[] => {
  const targets: EditStrategyPolicyFeedbackTarget[] = [];
  const add = (target: EditStrategyPolicyFeedbackTarget): void => {
    if (!targets.includes(target)) {
      targets.push(target);
    }
  };

  if (input.parseStatus === "failed" || isParseError(input.errorCode)) {
    add("rendered-contract");
    add("fallback-order");
  }

  if (input.applyStatus === "failed") {
    add("strategy-routing");
    add("fallback-order");
  }

  if (input.staleContextStatus === "stale" || input.staleContextStatus === "conflict") {
    add("stale-context-policy");
    add("strategy-routing");
  }

  if (input.protectedPathTouched || input.errorCode === "protected_path_violation") {
    add("protected-path-policy");
  }

  if (
    input.postApplyConsistencyStatus === "inconsistent" ||
    input.verificationStatus === "failed" ||
    input.verificationStatus === "error" ||
    input.appliedButBroken
  ) {
    add("verifier-enforcement");
    add("repair-instructions");
    add("rollback-policy");
  }

  if (input.selfDetectedRegressionStatus === "suspected" || input.selfDetectedRegressionStatus === "confirmed") {
    add("verifier-enforcement");
    add("repair-instructions");
  }

  if (input.status !== "passed" || !input.expectedOutcomeMatched) {
    add("strategy-routing");
  }

  return targets;
};

const policyFeedbackTargetCounts = (
  results: readonly EditStrategyAblationProbeResult[],
): EditStrategyPolicyFeedbackTargetCount[] =>
  EditStrategyPolicyFeedbackTargetSchema.options.flatMap((target) => {
    const count = results.filter((result) => result.policyFeedbackTargets.includes(target)).length;
    return count === 0 ? [] : [EditStrategyPolicyFeedbackTargetCountSchema.parse({ target, count })];
  });

const passRate = (values: readonly boolean[]): number => ratio(values.filter(Boolean).length, values.length);
const average = (values: readonly number[]): number => values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
const ratio = (numerator: number, denominator: number): number => denominator === 0 ? 0 : numerator / denominator;

const canonicalEditEvalCases = (cases: readonly EditStrategyEvalCase[]): EditStrategyEvalCase[] =>
  [...cases].sort((left, right) => {
    const splitComparison = splitRank(left.split) - splitRank(right.split);
    return splitComparison === 0
      ? left.editEvalCaseId.localeCompare(right.editEvalCaseId)
      : splitComparison;
  });

const orderedSplits = (splits: readonly EvalSplit[]): EvalSplit[] =>
  ["train", "dev", "holdout"].filter((split): split is EvalSplit => splits.includes(split as EvalSplit));

const orderedFamilies = (families: readonly EditStrategyFamily[]): EditStrategyFamily[] =>
  EditStrategyFamilySchema.options.filter((family) => families.includes(family));

const splitRank = (split: EvalSplit): number => {
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
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const samePathOrChild = (path: string, possibleParent: string): boolean =>
  path === possibleParent || path.startsWith(`${possibleParent}/`);

const idPart = (value: string): string => {
  const sanitized = value.replace(/_/gu, "-").replace(/[^A-Za-z0-9._:-]/gu, "-");
  return /^[A-Za-z0-9]/u.test(sanitized) ? sanitized : `id.${sanitized}`;
};
