import { relative, resolve, sep } from "node:path";
import type { JsonValue } from "../optimizer/types";
import {
  RealAcpEditStrategyRecordSchema,
  RealAcpRepairRecordSchema,
  RealAcpRollbackRecordSchema,
  RealAcpRouteRecordSchema,
  RealAcpTaskOutcomeStatusSchema,
  RealAcpTerminalRecordSchema,
  RealAcpToolRecordSchema,
  RealAcpVerifierRecordSchema,
  type RealAcpEditStrategyRecord,
  type RealAcpExecutionMode,
  type RealAcpExecutorTaskInput,
  type RealAcpExecutorTaskOutput,
  type RealAcpHeadlessExecutor,
  type RealAcpLineageRecord,
  type RealAcpRepairRecord,
  type RealAcpRollbackRecord,
  type RealAcpRouteRecord,
  type RealAcpRunMetadata,
  type RealAcpTaskOutcomeStatus,
  type RealAcpTerminalRecord,
  type RealAcpToolRecord,
  type RealAcpVerifierRecord,
} from "./real-acp-runner";
import type { RealAcpCorpusTask } from "./real-acp-task-pack";

export type RealAcpHeadlessRunnerStatus =
  | RealAcpTaskOutcomeStatus
  | "success"
  | "succeeded"
  | "failure"
  | "cancel"
  | "canceled"
  | "aborted"
  | "skip";

export type RealAcpHeadlessRunnerInput = {
  task: {
    taskId: string;
    title: string;
    userPrompt: string;
    correctionPrompts: readonly string[];
    primaryLabel: RealAcpCorpusTask["primaryLabel"];
    labels: readonly RealAcpCorpusTask["primaryLabel"][];
    timeoutMs: number;
  };
  workspace: {
    workspacePath: string;
    workspaceId: string;
    kind: RealAcpCorpusTask["workspace"]["kind"];
    allowedPathPrefixes: readonly string[];
    protectedPaths: readonly string[];
    materializedFilePaths: readonly string[];
  };
  expectedOutcome: RealAcpCorpusTask["expectedOutcome"];
  run: {
    executionMode: RealAcpExecutionMode;
    dryRun: boolean;
    metadata: RealAcpRunMetadata;
  };
  context: {
    signal: AbortSignal;
    timeoutMs: number;
  };
};

export type RealAcpHeadlessRunnerOutput = {
  status: RealAcpHeadlessRunnerStatus;
  route?: RealAcpRouteRecord;
  editStrategy?: RealAcpEditStrategyRecord;
  toolCalls?: readonly RealAcpToolRecord[];
  terminalCommands?: readonly RealAcpTerminalRecord[];
  verifier?: RealAcpVerifierRecord;
  repair?: RealAcpRepairRecord;
  rollback?: RealAcpRollbackRecord;
  corrections?: RealAcpExecutorTaskOutput["corrections"];
  telemetry?: JsonValue;
  failureReason?: string;
  skipReason?: string;
  lineage?: Partial<Omit<RealAcpLineageRecord, "taskId" | "runResultId" | "sourceTaskPackId">>;
};

export type RealAcpHeadlessTaskRunner = (
  input: RealAcpHeadlessRunnerInput,
) => Promise<RealAcpHeadlessRunnerOutput>;

export type RealAcpHeadlessExecutorOptions = {
  executorId?: string;
  executorVersion?: string;
  currentRepoPath?: string;
  allowedWorkspaceRoot?: string;
  runTask: RealAcpHeadlessTaskRunner;
};

const DEFAULT_EXECUTOR_ID = "real-acp.executor.headless.injected";
const DEFAULT_EXECUTOR_VERSION = "headless-adapter.v1";

export const createRealAcpHeadlessExecutor = (
  options: RealAcpHeadlessExecutorOptions,
): RealAcpHeadlessExecutor => ({
  executorId: options.executorId ?? DEFAULT_EXECUTOR_ID,
  executorVersion: options.executorVersion ?? DEFAULT_EXECUTOR_VERSION,
  kind: "headless_acp",
  executeTask: async (input) => executeHeadlessTask(input, options),
});

const executeHeadlessTask = async (
  input: RealAcpExecutorTaskInput,
  options: RealAcpHeadlessExecutorOptions,
): Promise<RealAcpExecutorTaskOutput> => {
  const currentRepoPath = resolve(options.currentRepoPath ?? process.cwd());
  assertWorkspaceOutsideCurrentRepo(input.workspacePath, currentRepoPath, options.allowedWorkspaceRoot);

  if (input.context.signal.aborted) {
    return cancelledOutput(input.task, {
      reason: "headless ACP execution cancelled before start",
      telemetry: {
        cancellation: {
          status: "cancelled",
          phase: "before_start",
        },
      },
    });
  }

  try {
    const output = await options.runTask(toRunnerInput(input));
    return normalizeRunnerOutput(input.task, output);
  } catch (error) {
    if (isAbortLike(error) || input.context.signal.aborted) {
      return cancelledOutput(input.task, {
        reason: errorMessage(error) || "headless ACP execution cancelled",
        telemetry: {
          cancellation: {
            status: "cancelled",
            phase: "runner",
          },
        },
      });
    }
    return errorOutput(input.task, error);
  }
};

const toRunnerInput = (input: RealAcpExecutorTaskInput): RealAcpHeadlessRunnerInput => ({
  task: {
    taskId: input.task.taskId,
    title: input.task.title,
    userPrompt: input.task.userPrompt,
    correctionPrompts: input.task.correctionPrompts,
    primaryLabel: input.task.primaryLabel,
    labels: input.task.labels,
    timeoutMs: input.task.timeoutMs,
  },
  workspace: {
    workspacePath: input.workspacePath,
    workspaceId: input.task.workspace.workspaceId,
    kind: input.task.workspace.kind,
    allowedPathPrefixes: input.task.workspace.allowedPathPrefixes,
    protectedPaths: input.task.workspace.protectedPaths,
    materializedFilePaths: input.task.workspace.files.map((file) => file.path),
  },
  expectedOutcome: input.task.expectedOutcome,
  run: {
    executionMode: input.executionMode,
    dryRun: input.dryRun,
    metadata: input.metadata,
  },
  context: input.context,
});

const normalizeRunnerOutput = (
  task: RealAcpCorpusTask,
  output: RealAcpHeadlessRunnerOutput,
): RealAcpExecutorTaskOutput => {
  const status = normalizeStatus(output.status);
  const terminalCommands = normalizeTerminalCommands(output.terminalCommands);
  const telemetry = output.telemetry ?? telemetryForStatus(status, output);
  const normalized: RealAcpExecutorTaskOutput = {
    status,
    route: output.route === undefined
      ? defaultRoute(task, status)
      : RealAcpRouteRecordSchema.parse(output.route),
    editStrategy: output.editStrategy === undefined
      ? defaultEditStrategy(task, status)
      : RealAcpEditStrategyRecordSchema.parse(output.editStrategy),
    toolCalls: normalizeToolCalls(output.toolCalls),
    terminalCommands,
    verifier: output.verifier === undefined
      ? defaultVerifier(task, terminalCommands, status)
      : RealAcpVerifierRecordSchema.parse(output.verifier),
    repair: output.repair === undefined
      ? defaultRepair(status)
      : RealAcpRepairRecordSchema.parse(output.repair),
    rollback: output.rollback === undefined
      ? defaultRollback(status)
      : RealAcpRollbackRecordSchema.parse(output.rollback),
    corrections: output.corrections ?? [],
    telemetry,
    ...(output.failureReason === undefined && status !== "failed" && status !== "error"
      ? {}
      : { failureReason: output.failureReason ?? defaultFailureReason(status) }),
    ...(output.skipReason === undefined && status !== "skipped"
      ? {}
      : { skipReason: output.skipReason ?? "headless ACP task skipped" }),
    ...(output.lineage === undefined ? {} : { lineage: output.lineage }),
  };
  return normalized;
};

const normalizeStatus = (status: RealAcpHeadlessRunnerStatus): RealAcpTaskOutcomeStatus => {
  switch (status) {
    case "success":
    case "succeeded":
      return "passed";
    case "failure":
      return "failed";
    case "skip":
      return "skipped";
    case "cancel":
    case "canceled":
    case "aborted":
      return "cancelled";
    default:
      return RealAcpTaskOutcomeStatusSchema.parse(status);
  }
};

const normalizeToolCalls = (
  toolCalls: readonly RealAcpToolRecord[] | undefined,
): RealAcpToolRecord[] =>
  (toolCalls ?? []).map((toolCall) => RealAcpToolRecordSchema.parse(toolCall));

const normalizeTerminalCommands = (
  terminalCommands: readonly RealAcpTerminalRecord[] | undefined,
): RealAcpTerminalRecord[] =>
  (terminalCommands ?? []).map((command) => RealAcpTerminalRecordSchema.parse(command));

const cancelledOutput = (
  task: RealAcpCorpusTask,
  input: { reason: string; telemetry: JsonValue },
): RealAcpExecutorTaskOutput => ({
  status: "cancelled",
  route: defaultRoute(task, "cancelled"),
  editStrategy: noEditStrategy("headless ACP execution cancelled"),
  toolCalls: [],
  terminalCommands: [],
  verifier: defaultVerifier(task, [], "cancelled"),
  repair: { attempted: false, status: "skipped", reason: "headless ACP execution cancelled" },
  rollback: { attempted: false, status: "skipped", reason: "headless ACP execution cancelled" },
  corrections: [],
  telemetry: input.telemetry,
  failureReason: input.reason,
});

const errorOutput = (
  task: RealAcpCorpusTask,
  error: unknown,
): RealAcpExecutorTaskOutput => ({
  status: "error",
  route: defaultRoute(task, "error"),
  editStrategy: noEditStrategy("headless ACP runner error"),
  toolCalls: [],
  terminalCommands: [],
  verifier: defaultVerifier(task, [], "error"),
  repair: { attempted: false, status: "skipped", reason: "headless ACP runner error" },
  rollback: { attempted: false, status: "skipped", reason: "headless ACP runner error" },
  corrections: [],
  telemetry: {
    error: errorMessage(error),
  },
  failureReason: errorMessage(error),
});

const defaultRoute = (
  task: RealAcpCorpusTask,
  status: RealAcpTaskOutcomeStatus,
): RealAcpRouteRecord => RealAcpRouteRecordSchema.parse({
  routeId: `route.${safeId(task.taskId)}.headless`,
  selectedMode: status === "cancelled" ? "cancelled" : "coding",
  reason: `Headless ACP adapter route for ${task.primaryLabel}`,
  confidence: 1,
});

const defaultEditStrategy = (
  task: RealAcpCorpusTask,
  status: RealAcpTaskOutcomeStatus,
): RealAcpEditStrategyRecord => {
  if (status === "skipped" || status === "cancelled" || task.expectedOutcome.mutation === "no_change") {
    return noEditStrategy(`headless ACP ${status} output`);
  }
  return RealAcpEditStrategyRecordSchema.parse({
    strategyId: "edit.headless-acp.runner.v1",
    family: "diff",
    selectedBy: "executor",
    reason: `Headless ACP runner selected edit behavior for ${task.expectedOutcome.mutation}`,
  });
};

const noEditStrategy = (reason: string): RealAcpEditStrategyRecord => RealAcpEditStrategyRecordSchema.parse({
  strategyId: "edit.none.v1",
  family: "none",
  selectedBy: "not_applicable",
  reason,
});

const defaultVerifier = (
  task: RealAcpCorpusTask,
  terminalCommands: readonly RealAcpTerminalRecord[],
  status: RealAcpTaskOutcomeStatus,
): RealAcpVerifierRecord => RealAcpVerifierRecordSchema.parse({
  status: verifierStatusFor(status),
  policy: task.expectedOutcome.verification.policy,
  commandIds: terminalCommands.map((command) => command.commandId),
  ...(task.expectedOutcome.verification.skipReason === undefined
    ? {}
    : { skipReason: task.expectedOutcome.verification.skipReason }),
});

const verifierStatusFor = (
  status: RealAcpTaskOutcomeStatus,
): RealAcpVerifierRecord["status"] => {
  switch (status) {
    case "passed":
      return "passed";
    case "failed":
      return "failed";
    case "skipped":
      return "skipped";
    case "cancelled":
    case "error":
      return "not_run";
  }
};

const defaultRepair = (status: RealAcpTaskOutcomeStatus): RealAcpRepairRecord =>
  RealAcpRepairRecordSchema.parse(status === "failed"
    ? { attempted: false, status: "skipped", reason: "headless ACP runner did not report repair" }
    : { attempted: false, status: "not_needed" });

const defaultRollback = (status: RealAcpTaskOutcomeStatus): RealAcpRollbackRecord =>
  RealAcpRollbackRecordSchema.parse(status === "failed" || status === "error" || status === "cancelled"
    ? { attempted: false, status: "skipped", reason: "headless ACP runner did not report rollback" }
    : { attempted: false, status: "not_needed" });

const telemetryForStatus = (
  status: RealAcpTaskOutcomeStatus,
  output: RealAcpHeadlessRunnerOutput,
): JsonValue => {
  if (status === "skipped") {
    return { verificationStatus: "skipped", skipReason: output.skipReason ?? "headless ACP task skipped" };
  }
  if (status === "cancelled") {
    return { cancellation: { status: "cancelled" } };
  }
  return {};
};

const defaultFailureReason = (status: RealAcpTaskOutcomeStatus): string => {
  switch (status) {
    case "failed":
      return "headless ACP task failed";
    case "error":
      return "headless ACP task errored";
    case "passed":
    case "skipped":
    case "cancelled":
      return "headless ACP task did not pass";
  }
};

const assertWorkspaceOutsideCurrentRepo = (
  workspacePath: string,
  currentRepoPath: string,
  allowedWorkspaceRoot?: string,
): void => {
  if (allowedWorkspaceRoot !== undefined && isInsideOrEqual(workspacePath, resolve(allowedWorkspaceRoot))) {
    return;
  }
  if (isInsideOrEqual(workspacePath, currentRepoPath)) {
    throw new Error("headless ACP executor refuses to run against the current repository workspace");
  }
};

const isInsideOrEqual = (candidatePath: string, rootPath: string): boolean => {
  const relativePath = relative(resolve(rootPath), resolve(candidatePath));
  return relativePath === "" || (!relativePath.startsWith("..") && !relativePath.includes(`..${sep}`));
};

const isAbortLike = (error: unknown): boolean => {
  if (!(error instanceof Error)) {
    return false;
  }
  return error.name === "AbortError" || /aborted|cancelled|canceled/i.test(error.message);
};

const safeId = (value: string): string => value.replace(/[^A-Za-z0-9._:-]+/g, "-");

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);
