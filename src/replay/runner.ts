import {
  createEvalScorecard,
} from "../eval-harness/scorer";
import {
  runEvalComparison,
  type EvalCommandResult,
  type EvalComparisonExecution,
  type EvalRunExecutorOutput,
} from "../eval-harness/runner";
import type {
  ComparisonRunMetadata,
  EvalCase,
  EvalRunResult,
  EvalScorecard,
  EvalSplit,
  FixtureWorkspace,
} from "../eval-harness/types";
import { JsonValueSchema, type JsonValue } from "../optimizer/types";
import type { AcpReplayCapture } from "./capture";
import { editFailureReplayScenarios } from "./edit-failure-scenarios";
import { selectReplayCasesForOptimizerInput } from "./enforcement";
import {
  ReplayEvalCaseSkeletonSchema,
  extractReplayEvalCaseSkeleton,
  type ReplayEvalCaseSkeleton,
  type ReplayExtractionMetadata,
} from "./extraction";
import { routingReplayScenarios } from "./routing-scenarios";
import { toolCallReplayScenarios } from "./tool-call-scenarios";

const REPLAY_EVAL_CASE_SCHEMA_VERSION = "eval-case.v1";
const REPLAY_RUNNER_CREATED_AT = "2026-05-01T00:00:00.000Z";

export type ReplayEvalScenario = {
  scenarioId: string;
  split: EvalSplit;
  optimizationAllowed: boolean;
  capture: AcpReplayCapture;
  metadata: ReplayExtractionMetadata;
};

export type ReplayRunnableCase = {
  replayCase: ReplayEvalCaseSkeleton;
  capture?: AcpReplayCapture;
  scenarioId?: string;
};

export type ReplayPolicyExecutionInput = {
  replayCase: ReplayEvalCaseSkeleton;
  evalCase: EvalCase;
  workspacePath: string;
  context: {
    signal: AbortSignal;
    timeoutMs: number;
  };
  runRole: "baseline" | "candidate";
};

export type ReplayPolicyExecutor = (
  input: ReplayPolicyExecutionInput,
) => Promise<void | EvalRunExecutorOutput>;

export type RunReplayEvalComparisonInput = {
  replayCases?: readonly (ReplayRunnableCase | ReplayEvalCaseSkeleton | ReplayEvalScenario)[];
  includeHoldout?: boolean;
  baseline: ComparisonRunMetadata;
  candidate: ComparisonRunMetadata;
  baselinePolicy?: ReplayPolicyExecutor;
  candidatePolicy?: ReplayPolicyExecutor;
  candidatePatchId?: string;
  evalSuiteId?: string;
  scorecardIdPrefix?: string;
  timeoutMs?: number;
  baseDir?: string;
  signal?: AbortSignal;
  createdAt?: string;
};

export type ReplayEvalComparisonResult = {
  replayCases: ReplayEvalCaseSkeleton[];
  evalCases: EvalCase[];
  executions: EvalComparisonExecution[];
  baselineResults: EvalComparisonExecution["baseline"]["result"][];
  candidateResults: EvalComparisonExecution["candidate"]["result"][];
  scorecards: EvalScorecard[];
  passed: boolean;
  cleanup: () => Promise<void>;
};

export const replayEvalScenarios: ReplayEvalScenario[] = [
  ...routingReplayScenarios,
  ...editFailureReplayScenarios,
  ...toolCallReplayScenarios,
];

export const replayEvalCaseSkeletons: ReplayEvalCaseSkeleton[] =
  replayEvalScenarios.map((scenario) => extractReplayEvalCaseSkeleton(scenario));

export const visibleReplayEvalCaseSkeletonsForOptimization = (): ReplayEvalCaseSkeleton[] =>
  selectReplayCasesForOptimizerInput(replayEvalCaseSkeletons, "optimization_selection").selectedCases;

export const replayEvalCaseFromSkeleton = (
  replayCaseInput: ReplayEvalCaseSkeleton,
): EvalCase => {
  const replayCase = ReplayEvalCaseSkeletonSchema.parse(replayCaseInput);
  return {
    evalCaseId: replayCase.evalCaseId,
    schemaVersion: REPLAY_EVAL_CASE_SCHEMA_VERSION,
    split: replayCase.split,
    title: replayCase.title,
    task: replayCase.task,
    fixtureWorkspace: replayCase.fixtureWorkspace ?? defaultReplayFixtureWorkspace(replayCase),
    assertions: replayCase.oracle.expectedBehavior.assertions,
    tags: ["replay", ...replayCase.tags],
    timeoutMs: replayCase.timeoutMs,
  };
};

export const createReplayEvalExecutor = (
  replayCaseInput: ReplayRunnableCase | ReplayEvalCaseSkeleton | ReplayEvalScenario,
  policy?: ReplayPolicyExecutor,
  runRole: "baseline" | "candidate" = "candidate",
) => {
  const runnableCase = replayRunnableCase(replayCaseInput);
  return async (
    workspacePath: string,
    evalCase: EvalCase,
    context: ReplayPolicyExecutionInput["context"],
  ): Promise<EvalRunExecutorOutput> => {
    const defaultOutput: EvalRunExecutorOutput = {
      telemetry: replayTelemetryForCase(runnableCase.replayCase),
      commandResults: runnableCase.capture == null
        ? []
        : terminalCommandResultsFromReplayCapture(runnableCase.capture, workspacePath),
    };
    const policyOutput = await policy?.({
      replayCase: runnableCase.replayCase,
      evalCase,
      workspacePath,
      context,
      runRole,
    });
    return mergeExecutorOutputs(defaultOutput, policyOutput);
  };
};

export const runReplayEvalComparison = async (
  input: RunReplayEvalComparisonInput,
): Promise<ReplayEvalComparisonResult> => {
  const runnableCases = selectRunnableCases(input.replayCases, input.includeHoldout ?? false);
  const evalCases = runnableCases.map((runnableCase) => replayEvalCaseFromSkeleton(runnableCase.replayCase));
  const executions: EvalComparisonExecution[] = [];

  for (const [index, runnableCase] of runnableCases.entries()) {
    const execution = await runEvalComparison({
      evalCase: evalCases[index] as EvalCase,
      context: input.candidate.context,
      baselineComparisonRunId: input.baseline.comparisonRunId,
      candidateComparisonRunId: input.candidate.comparisonRunId,
      baselineExecutor: createReplayEvalExecutor(runnableCase, input.baselinePolicy, "baseline"),
      candidateExecutor: createReplayEvalExecutor(runnableCase, input.candidatePolicy, "candidate"),
      ...(input.candidatePatchId === undefined ? {} : { candidatePatchId: input.candidatePatchId }),
      ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
      ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
    executions.push(execution);
  }

  const baselineResults = executions.map((execution) => execution.baseline.result);
  const candidateResults = executions.map((execution) => execution.candidate.result);
  const scorecards = createReplayScorecards({
    baseline: input.baseline,
    candidate: input.candidate,
    baselineResults,
    candidateResults,
    evalSuiteId: input.evalSuiteId ?? "suite.replay-eval-corpus",
    scorecardIdPrefix: input.scorecardIdPrefix ?? "scorecard.replay-eval-corpus",
    createdAt: input.createdAt ?? REPLAY_RUNNER_CREATED_AT,
  });

  return {
    replayCases: runnableCases.map((runnableCase) => runnableCase.replayCase),
    evalCases,
    executions,
    baselineResults,
    candidateResults,
    scorecards,
    passed: scorecards.every((scorecard) => scorecard.passed),
    cleanup: async () => {
      await Promise.all(executions.map((execution) => execution.cleanup()));
    },
  };
};

const selectRunnableCases = (
  cases: readonly (ReplayRunnableCase | ReplayEvalCaseSkeleton | ReplayEvalScenario)[] | undefined,
  includeHoldout: boolean,
): ReplayRunnableCase[] => {
  const selected = cases == null
    ? defaultRunnableReplayScenarios()
    : cases.map((replayCase) => replayRunnableCase(replayCase));
  const filtered = includeHoldout
    ? selected
    : selected.filter((runnableCase) => runnableCase.replayCase.split !== "holdout");
  return filtered.sort((left, right) => left.replayCase.evalCaseId.localeCompare(right.replayCase.evalCaseId));
};

const defaultRunnableReplayScenarios = (): ReplayRunnableCase[] => {
  const visibleIds = new Set(visibleReplayEvalCaseSkeletonsForOptimization().map((replayCase) => replayCase.evalCaseId));
  return replayEvalScenarios
    .map((scenario) => replayRunnableCase(scenario))
    .filter((runnableCase) => visibleIds.has(runnableCase.replayCase.evalCaseId));
};

const replayRunnableCase = (
  input: ReplayRunnableCase | ReplayEvalCaseSkeleton | ReplayEvalScenario,
): ReplayRunnableCase => {
  if ("replayCase" in input) {
    return {
      replayCase: ReplayEvalCaseSkeletonSchema.parse(input.replayCase),
      ...(input.capture === undefined ? {} : { capture: input.capture }),
      ...(input.scenarioId === undefined ? {} : { scenarioId: input.scenarioId }),
    };
  }
  if ("capture" in input && "metadata" in input) {
    return {
      replayCase: extractReplayEvalCaseSkeleton({
        capture: input.capture,
        metadata: input.metadata,
      }),
      capture: input.capture,
      scenarioId: input.scenarioId,
    };
  }
  return {
    replayCase: ReplayEvalCaseSkeletonSchema.parse(input),
  };
};

const defaultReplayFixtureWorkspace = (replayCase: ReplayEvalCaseSkeleton): FixtureWorkspace => ({
  fixtureWorkspaceId: `fixture.${replayCase.evalCaseId}`,
  name: replayCase.title,
  description: "Synthetic replay workspace used for metadata-only offline replay assertions.",
  rootFingerprint: `sha256:${replayCase.evalCaseId}.replay-root`,
  files: [
    {
      path: "REPLAY_CASE.txt",
      content: `Replay case: ${replayCase.evalCaseId}\n`,
      executable: false,
    },
  ],
  protectedPaths: ["REPLAY_CASE.txt"],
  setupCommands: [],
  verificationCommands: [],
});

const replayTelemetryForCase = (replayCase: ReplayEvalCaseSkeleton): JsonValue =>
  JsonValueSchema.parse({
    replayCaseId: replayCase.evalCaseId,
    replaySchemaVersion: replayCase.schemaVersion,
    split: replayCase.split,
    splitAssignment: replayCase.splitAssignment,
    title: replayCase.title,
    task: replayCase.task,
    captureId: replayCase.captureId,
    sourceSessionId: replayCase.sourceSessionId ?? null,
    sourceTraceIds: replayCase.sourceTraceIds,
    sourceRefs: replayCase.sourceRefs,
    redaction: replayCase.redaction,
    oracle: replayCase.oracle,
    routing: replayCase.routing,
    observedFailures: replayCase.observedFailures,
    tags: replayCase.tags,
  });

const terminalCommandResultsFromReplayCapture = (
  capture: AcpReplayCapture,
  workspacePath: string,
): EvalCommandResult[] =>
  capture.records.flatMap((record): EvalCommandResult[] => {
    if (record.recordKind !== "terminal_command") {
      return [];
    }
    return [{
      commandId: record.commandId,
      command: record.command,
      cwd: record.cwd ?? workspacePath,
      exitCode: record.exitCode ?? (record.status === "succeeded" ? 0 : null),
      signal: record.signal == null ? null : record.signal as NodeJS.Signals,
      stdout: record.stdoutArtifactRef == null ? "" : `[artifact:${record.stdoutArtifactRef}]`,
      stderr: record.stderrArtifactRef == null ? "" : `[artifact:${record.stderrArtifactRef}]`,
      timedOut: record.status === "timed_out",
      durationMs: 0,
      startedAt: capture.createdAt,
      completedAt: capture.createdAt,
    }];
  });

const mergeExecutorOutputs = (
  defaultOutput: EvalRunExecutorOutput,
  policyOutput: void | EvalRunExecutorOutput,
): EvalRunExecutorOutput => {
  if (policyOutput == null) {
    return defaultOutput;
  }
  const telemetry = mergeJsonObjects(defaultOutput.telemetry, policyOutput.telemetry);
  const commandResults = [
    ...(defaultOutput.commandResults ?? []),
    ...(policyOutput.commandResults ?? []),
  ];
  return {
    ...(telemetry === undefined ? {} : { telemetry }),
    ...(commandResults.length === 0 ? {} : { commandResults }),
  };
};

const mergeJsonObjects = (
  left: JsonValue | undefined,
  right: JsonValue | undefined,
): JsonValue | undefined => {
  if (right === undefined) {
    return left;
  }
  if (left === undefined) {
    return right;
  }
  if (isJsonObject(left) && isJsonObject(right)) {
    return {
      ...left,
      ...right,
    };
  }
  return right;
};

const createReplayScorecards = (input: {
  baseline: ComparisonRunMetadata;
  candidate: ComparisonRunMetadata;
  baselineResults: readonly EvalRunResult[];
  candidateResults: readonly EvalRunResult[];
  evalSuiteId: string;
  scorecardIdPrefix: string;
  createdAt: string;
}): EvalScorecard[] => {
  const splits = [...new Set([
    ...input.baselineResults.map((result) => result.split),
    ...input.candidateResults.map((result) => result.split),
  ])].sort((left, right) => splitOrder(left) - splitOrder(right));

  return splits.map((split) => createEvalScorecard({
    scorecardId: `${input.scorecardIdPrefix}.${split}`,
    evalSuiteId: input.evalSuiteId,
    split,
    baseline: input.baseline,
    candidate: input.candidate,
    baselineResults: input.baselineResults.filter((result) => result.split === split),
    candidateResults: input.candidateResults.filter((result) => result.split === split),
    createdAt: input.createdAt,
  }));
};

const isJsonObject = (value: JsonValue | undefined): value is Record<string, JsonValue> =>
  value != null && typeof value === "object" && !Array.isArray(value);

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
