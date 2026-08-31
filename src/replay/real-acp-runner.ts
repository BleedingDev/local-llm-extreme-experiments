import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { tmpdir } from "node:os";
import { z } from "zod";
import type { EvalAssertion, EvalSplit } from "../eval-harness/types";
import { JsonValueSchema, OptimizerIdSchema, type JsonValue } from "../optimizer/types";
import {
  RealAcpCorpusTaskSchema,
  RealAcpTaskPackSchema,
  realAcpCodingCorpusTaskPack,
  type RealAcpCorpusTask,
  type RealAcpTaskPack,
} from "./real-acp-task-pack";

const REAL_ACP_RUN_SCHEMA_VERSION = "real-acp-corpus-run.v1" as const;
const REAL_ACP_TASK_RESULT_SCHEMA_VERSION = "real-acp-task-result.v1" as const;
const DEFAULT_CREATED_AT = "2026-05-04T00:00:00.000Z";
const HIDDEN_SPLIT: EvalSplit = "holdout";
const SAFE_OUTPUT_ROOT = join(".bag", "replay-corpus", "real-acp-runs");

const IsoTimestampSchema = z.string().datetime({ offset: true });
const RelativeSafePathSchema = z.string().min(1).regex(/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$)).+$/);

export const RealAcpCorpusRunPurposeSchema = z.enum([
  "optimizer_input",
  "development_eval",
  "holdout_final",
]);
export type RealAcpCorpusRunPurpose = z.infer<typeof RealAcpCorpusRunPurposeSchema>;

export const RealAcpExecutionModeSchema = z.enum([
  "dry_run",
  "headless_acp",
  "real_consumer",
]);
export type RealAcpExecutionMode = z.infer<typeof RealAcpExecutionModeSchema>;

export const RealAcpTaskOutcomeStatusSchema = z.enum([
  "passed",
  "failed",
  "skipped",
  "cancelled",
  "error",
]);
export type RealAcpTaskOutcomeStatus = z.infer<typeof RealAcpTaskOutcomeStatusSchema>;

export const RealAcpRunModelMetadataSchema = z.object({
  modelProfileId: OptimizerIdSchema,
  provider: z.string().min(1),
  model: z.string().min(1),
  modelRole: z.string().min(1),
  contextWindowTokens: z.number().int().positive(),
  toolCallingMode: z.enum(["native", "json", "text", "disabled"]),
}).strict();
export type RealAcpRunModelMetadata = z.infer<typeof RealAcpRunModelMetadataSchema>;

export const RealAcpRunCodebaseMetadataSchema = z.object({
  codebaseProfileId: OptimizerIdSchema,
  rootFingerprint: z.string().min(1),
  languageSummary: z.string().min(1),
  testRiskTier: OptimizerIdSchema,
  protectedPathPolicy: z.string().min(1),
}).strict();
export type RealAcpRunCodebaseMetadata = z.infer<typeof RealAcpRunCodebaseMetadataSchema>;

export const RealAcpRunClientMetadataSchema = z.object({
  clientProfileId: OptimizerIdSchema,
  clientName: z.string().min(1),
  clientVersion: z.string().min(1),
  transport: z.enum(["stdio", "http", "websocket", "in_process", "simulated"]),
  acpConsumerCapabilities: z.record(z.string(), JsonValueSchema),
}).strict();
export type RealAcpRunClientMetadata = z.infer<typeof RealAcpRunClientMetadataSchema>;

export const RealAcpRunProfileMetadataSchema = z.object({
  policyId: OptimizerIdSchema,
  optimizerProfileId: OptimizerIdSchema,
  verificationPolicyVersion: z.string().min(1),
  resultStyleVersion: z.string().min(1),
  canonicalToolVersion: z.string().min(1),
  renderedToolVersion: z.string().min(1),
}).strict();
export type RealAcpRunProfileMetadata = z.infer<typeof RealAcpRunProfileMetadataSchema>;

export const RealAcpRunMetadataSchema = z.object({
  model: RealAcpRunModelMetadataSchema,
  codebase: RealAcpRunCodebaseMetadataSchema,
  client: RealAcpRunClientMetadataSchema,
  profile: RealAcpRunProfileMetadataSchema,
}).strict();
export type RealAcpRunMetadata = z.infer<typeof RealAcpRunMetadataSchema>;

export const RealAcpRouteRecordSchema = z.object({
  routeId: OptimizerIdSchema,
  selectedMode: z.enum(["coding", "planning", "maintenance", "read_only", "cancelled"]),
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1).optional(),
}).strict();
export type RealAcpRouteRecord = z.infer<typeof RealAcpRouteRecordSchema>;

export const RealAcpEditStrategyRecordSchema = z.object({
  strategyId: OptimizerIdSchema,
  family: z.enum(["whole_file", "diff", "search_replace", "none"]),
  selectedBy: z.enum(["optimizer_policy", "deterministic_gate", "executor", "not_applicable"]),
  fallbackStrategyId: OptimizerIdSchema.optional(),
  reason: z.string().min(1).optional(),
}).strict();
export type RealAcpEditStrategyRecord = z.infer<typeof RealAcpEditStrategyRecordSchema>;

export const RealAcpToolRecordSchema = z.object({
  toolCallId: OptimizerIdSchema,
  namespace: OptimizerIdSchema.optional(),
  name: OptimizerIdSchema,
  status: z.enum(["succeeded", "failed", "skipped", "blocked"]),
  sideEffectLevel: z.enum(["none", "read", "write", "network", "process"]),
  errorCode: OptimizerIdSchema.optional(),
}).strict();
export type RealAcpToolRecord = z.infer<typeof RealAcpToolRecordSchema>;

export const RealAcpTerminalRecordSchema = z.object({
  commandId: OptimizerIdSchema,
  command: z.array(z.string().min(1)).min(1),
  status: z.enum(["succeeded", "failed", "skipped", "timed_out"]),
  exitCode: z.number().int().nullable(),
  durationMs: z.number().int().nonnegative(),
}).strict();
export type RealAcpTerminalRecord = z.infer<typeof RealAcpTerminalRecordSchema>;

export const RealAcpVerifierRecordSchema = z.object({
  status: z.enum(["passed", "failed", "skipped", "not_run"]),
  policy: z.enum(["required", "allowed_to_skip", "must_skip", "expected_to_fail_before_repair"]),
  commandIds: z.array(OptimizerIdSchema).default([]),
  skipReason: z.string().min(1).optional(),
}).strict();
export type RealAcpVerifierRecord = z.infer<typeof RealAcpVerifierRecordSchema>;

export const RealAcpRepairRecordSchema = z.object({
  attempted: z.boolean(),
  status: z.enum(["not_needed", "succeeded", "failed", "skipped"]),
  reason: z.string().min(1).optional(),
}).strict();
export type RealAcpRepairRecord = z.infer<typeof RealAcpRepairRecordSchema>;

export const RealAcpRollbackRecordSchema = z.object({
  attempted: z.boolean(),
  status: z.enum(["not_needed", "succeeded", "failed", "skipped"]),
  reason: z.string().min(1).optional(),
}).strict();
export type RealAcpRollbackRecord = z.infer<typeof RealAcpRollbackRecordSchema>;

export const RealAcpCorrectionRecordSchema = z.object({
  correctionId: OptimizerIdSchema,
  promptIndex: z.number().int().nonnegative(),
  applied: z.boolean(),
  scopeChanged: z.boolean().default(false),
}).strict();
export type RealAcpCorrectionRecord = z.infer<typeof RealAcpCorrectionRecordSchema>;

export const RealAcpLineageRecordSchema = z.object({
  taskId: OptimizerIdSchema,
  runResultId: OptimizerIdSchema,
  parentRunResultId: OptimizerIdSchema.optional(),
  correctionOfRunResultId: OptimizerIdSchema.optional(),
  repairOfRunResultId: OptimizerIdSchema.optional(),
  rollbackOfRunResultId: OptimizerIdSchema.optional(),
  sourceTaskPackId: OptimizerIdSchema,
}).strict();
export type RealAcpLineageRecord = z.infer<typeof RealAcpLineageRecordSchema>;

export const RealAcpChangedFileRecordSchema = z.object({
  path: RelativeSafePathSchema,
  changeKind: z.enum(["added", "modified", "deleted"]),
  beforeHash: z.string().optional(),
  afterHash: z.string().optional(),
}).strict();
export type RealAcpChangedFileRecord = z.infer<typeof RealAcpChangedFileRecordSchema>;

export const RealAcpTaskRedactionSchema = z.object({
  rawLocalStatus: z.literal("raw_local_only"),
  optimizerSafe: z.boolean(),
  excludedFromOptimizerReasons: z.array(z.string().min(1)).default([]),
}).strict();
export type RealAcpTaskRedaction = z.infer<typeof RealAcpTaskRedactionSchema>;

export const RealAcpTaskRunResultSchema = z.object({
  schemaVersion: z.literal(REAL_ACP_TASK_RESULT_SCHEMA_VERSION),
  runResultId: OptimizerIdSchema,
  taskId: OptimizerIdSchema,
  split: z.enum(["train", "dev", "holdout"]),
  optimizationAllowed: z.boolean(),
  status: RealAcpTaskOutcomeStatusSchema,
  startedAt: IsoTimestampSchema,
  completedAt: IsoTimestampSchema,
  workspaceFingerprintBefore: z.string().min(1),
  workspaceFingerprintAfter: z.string().min(1),
  changedFiles: z.array(RealAcpChangedFileRecordSchema).default([]),
  route: RealAcpRouteRecordSchema,
  editStrategy: RealAcpEditStrategyRecordSchema,
  toolCalls: z.array(RealAcpToolRecordSchema).default([]),
  terminalCommands: z.array(RealAcpTerminalRecordSchema).default([]),
  verifier: RealAcpVerifierRecordSchema,
  repair: RealAcpRepairRecordSchema,
  rollback: RealAcpRollbackRecordSchema,
  corrections: z.array(RealAcpCorrectionRecordSchema).default([]),
  lineage: RealAcpLineageRecordSchema,
  telemetry: JsonValueSchema,
  redaction: RealAcpTaskRedactionSchema,
  failureReason: z.string().min(1).optional(),
  skipReason: z.string().min(1).optional(),
}).strict();
export type RealAcpTaskRunResult = z.infer<typeof RealAcpTaskRunResultSchema>;

export const RealAcpRunRedactionHandoffSchema = z.object({
  rawLocal: z.object({
    status: z.literal("raw_local_only"),
    containsWorkspaceSnapshots: z.literal(true),
    containsExecutorTelemetry: z.literal(true),
    storageGuidance: z.string().min(1),
  }).strict(),
  optimizerSafe: z.object({
    status: z.enum(["prepared", "blocked"]),
    includedTaskResultIds: z.array(OptimizerIdSchema),
    excludedTaskResultIds: z.array(OptimizerIdSchema),
    redactedFields: z.array(z.string().min(1)),
    nextSteps: z.array(z.string().min(1)),
  }).strict(),
}).strict();
export type RealAcpRunRedactionHandoff = z.infer<typeof RealAcpRunRedactionHandoffSchema>;

export const RealAcpCorpusRunManifestSchema = z.object({
  schemaVersion: z.literal(REAL_ACP_RUN_SCHEMA_VERSION),
  runId: OptimizerIdSchema,
  taskPackId: OptimizerIdSchema,
  createdAt: IsoTimestampSchema,
  executionMode: RealAcpExecutionModeSchema,
  dryRun: z.boolean(),
  purpose: RealAcpCorpusRunPurposeSchema,
  executor: z.object({
    executorId: OptimizerIdSchema,
    executorVersion: z.string().min(1),
    kind: z.enum(["simulated", "headless_acp", "real_consumer"]),
  }).strict(),
  metadata: RealAcpRunMetadataSchema,
  safety: z.object({
    workspaceIsolation: z.literal("per_task_materialized_fixture"),
    currentRepoMutationRefused: z.literal(true),
    realConsumerMutationAllowed: z.boolean(),
  }).strict(),
  splitPolicy: z.object({
    includeHoldout: z.boolean(),
    visibleOptimizationSplits: z.array(z.enum(["train", "dev", "holdout"])),
    hiddenSplits: z.array(z.enum(["train", "dev", "holdout"])),
    optimizerLeakageRefused: z.boolean(),
  }).strict(),
  taskResults: z.array(RealAcpTaskRunResultSchema),
  redactionHandoff: RealAcpRunRedactionHandoffSchema,
  summary: z.object({
    total: z.number().int().nonnegative(),
    passed: z.number().int().nonnegative(),
    failed: z.number().int().nonnegative(),
    skipped: z.number().int().nonnegative(),
    cancelled: z.number().int().nonnegative(),
    error: z.number().int().nonnegative(),
    holdout: z.number().int().nonnegative(),
  }).strict(),
  manifestPath: z.string().min(1).optional(),
}).strict();
export type RealAcpCorpusRunManifest = z.infer<typeof RealAcpCorpusRunManifestSchema>;

export type RealAcpExecutorTaskInput = {
  task: RealAcpCorpusTask;
  workspacePath: string;
  executionMode: RealAcpExecutionMode;
  dryRun: boolean;
  metadata: RealAcpRunMetadata;
  context: {
    signal: AbortSignal;
    timeoutMs: number;
  };
};

export type RealAcpExecutorTaskOutput = {
  status: RealAcpTaskOutcomeStatus;
  route: RealAcpRouteRecord;
  editStrategy: RealAcpEditStrategyRecord;
  toolCalls?: readonly RealAcpToolRecord[];
  terminalCommands?: readonly RealAcpTerminalRecord[];
  verifier: RealAcpVerifierRecord;
  repair: RealAcpRepairRecord;
  rollback: RealAcpRollbackRecord;
  corrections?: readonly RealAcpCorrectionRecord[];
  telemetry?: JsonValue;
  failureReason?: string;
  skipReason?: string;
  lineage?: Partial<Omit<RealAcpLineageRecord, "taskId" | "runResultId" | "sourceTaskPackId">>;
};

export type RealAcpHeadlessExecutor = {
  executorId: string;
  executorVersion: string;
  kind: "simulated" | "headless_acp" | "real_consumer";
  executeTask: (input: RealAcpExecutorTaskInput) => Promise<RealAcpExecutorTaskOutput>;
};

export type RunRealAcpCorpusInput = {
  runId: string;
  metadata: RealAcpRunMetadata;
  executor: RealAcpHeadlessExecutor;
  taskPack?: RealAcpTaskPack;
  taskIds?: readonly string[];
  purpose?: RealAcpCorpusRunPurpose;
  executionMode?: RealAcpExecutionMode;
  includeHoldout?: boolean;
  workspaceBaseDir?: string;
  outputDir?: string;
  currentRepoPath?: string;
  createdAt?: string;
  signal?: AbortSignal;
};

export type SimulatedRealAcpExecutorOptions = {
  failTaskIds?: readonly string[];
  skipTaskIds?: readonly string[];
};

export const runRealAcpCorpus = async (
  input: RunRealAcpCorpusInput,
): Promise<RealAcpCorpusRunManifest> => {
  const taskPack = RealAcpTaskPackSchema.parse(input.taskPack ?? realAcpCodingCorpusTaskPack);
  const metadata = RealAcpRunMetadataSchema.parse(input.metadata);
  const purpose = RealAcpCorpusRunPurposeSchema.parse(input.purpose ?? "development_eval");
  const executionMode = RealAcpExecutionModeSchema.parse(input.executionMode ?? "dry_run");
  const includeHoldout = input.includeHoldout ?? false;
  const createdAt = input.createdAt ?? DEFAULT_CREATED_AT;
  const currentRepoPath = resolve(input.currentRepoPath ?? process.cwd());
  const selectedTasks = selectRealAcpCorpusTasks({
    taskPack,
    purpose,
    includeHoldout,
    ...(input.taskIds === undefined ? {} : { taskIds: input.taskIds }),
  });

  const workspaceBaseDir = resolve(input.workspaceBaseDir ?? tmpdir());
  assertWorkspaceBaseIsSafe(workspaceBaseDir, currentRepoPath);

  const taskResults: RealAcpTaskRunResult[] = [];
  for (const task of selectedTasks) {
    taskResults.push(await runRealAcpTask({
      runId: input.runId,
      taskPackId: taskPack.taskPackId,
      task,
      metadata,
      executor: input.executor,
      executionMode,
      dryRun: executionMode === "dry_run",
      workspaceBaseDir,
      currentRepoPath,
      createdAt,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    }));
  }

  const manifestWithoutPath = RealAcpCorpusRunManifestSchema.omit({ manifestPath: true }).parse({
    schemaVersion: REAL_ACP_RUN_SCHEMA_VERSION,
    runId: input.runId,
    taskPackId: taskPack.taskPackId,
    createdAt,
    executionMode,
    dryRun: executionMode === "dry_run",
    purpose,
    executor: {
      executorId: input.executor.executorId,
      executorVersion: input.executor.executorVersion,
      kind: input.executor.kind,
    },
    metadata,
    safety: {
      workspaceIsolation: "per_task_materialized_fixture",
      currentRepoMutationRefused: true,
      realConsumerMutationAllowed: executionMode === "real_consumer" && input.executor.kind === "real_consumer",
    },
    splitPolicy: {
      includeHoldout,
      visibleOptimizationSplits: taskPack.splitPolicy.visibleOptimizationSplits,
      hiddenSplits: taskPack.splitPolicy.hiddenSplits,
      optimizerLeakageRefused: true,
    },
    taskResults,
    redactionHandoff: realAcpRunRedactionHandoff(taskResults),
    summary: summarizeTaskResults(taskResults),
  });

  if (input.outputDir === undefined) {
    return RealAcpCorpusRunManifestSchema.parse(manifestWithoutPath);
  }

  const outputDir = resolve(input.outputDir);
  assertOutputDirIsSafe(outputDir, currentRepoPath);
  await mkdir(outputDir, { recursive: true });
  const manifestPath = join(outputDir, `${safeId(input.runId)}.manifest.json`);
  const manifest = RealAcpCorpusRunManifestSchema.parse({
    ...manifestWithoutPath,
    manifestPath,
  });
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return manifest;
};

export const selectRealAcpCorpusTasks = (input: {
  taskPack?: RealAcpTaskPack;
  taskIds?: readonly string[];
  purpose?: RealAcpCorpusRunPurpose;
  includeHoldout?: boolean;
}): RealAcpCorpusTask[] => {
  const taskPack = RealAcpTaskPackSchema.parse(input.taskPack ?? realAcpCodingCorpusTaskPack);
  const purpose = RealAcpCorpusRunPurposeSchema.parse(input.purpose ?? "development_eval");
  const includeHoldout = input.includeHoldout ?? false;
  const taskIds = input.taskIds == null ? undefined : new Set(input.taskIds);
  const selected = taskPack.tasks.filter((task) => taskIds === undefined || taskIds.has(task.taskId));
  if (taskIds !== undefined && selected.length !== taskIds.size) {
    const found = new Set(selected.map((task) => task.taskId));
    const missing = [...taskIds].filter((taskId) => !found.has(taskId));
    throw new Error(`unknown real ACP task ids: ${missing.join(", ")}`);
  }

  const filtered = includeHoldout
    ? selected
    : selected.filter((task) => task.split !== HIDDEN_SPLIT);
  if (purpose === "optimizer_input") {
    const hidden = filtered.filter((task) => task.split === HIDDEN_SPLIT || !task.optimizationAllowed);
    if (includeHoldout || hidden.length > 0) {
      throw new Error(`hidden holdout optimizer leakage refused: ${hidden.map((task) => task.taskId).join(", ")}`);
    }
  }
  return filtered.sort((left, right) => left.taskId.localeCompare(right.taskId));
};

export const realAcpRunRedactionHandoff = (
  taskResults: readonly RealAcpTaskRunResult[],
): RealAcpRunRedactionHandoff => {
  const includedTaskResultIds = taskResults
    .filter((result) => result.redaction.optimizerSafe)
    .map((result) => result.runResultId)
    .sort((left, right) => left.localeCompare(right));
  const excludedTaskResultIds = taskResults
    .filter((result) => !result.redaction.optimizerSafe)
    .map((result) => result.runResultId)
    .sort((left, right) => left.localeCompare(right));

  return RealAcpRunRedactionHandoffSchema.parse({
    rawLocal: {
      status: "raw_local_only",
      containsWorkspaceSnapshots: true,
      containsExecutorTelemetry: true,
      storageGuidance: "Keep full manifests local until a redacted artifact index is built.",
    },
    optimizerSafe: {
      status: "prepared",
      includedTaskResultIds,
      excludedTaskResultIds,
      redactedFields: [
        "workspacePath",
        "workspaceSnapshots",
        "terminal stdout/stderr",
        "executor raw telemetry",
        "holdout task results",
      ],
      nextSteps: [
        "Build full redacted artifact index for optimizer-safe task summaries.",
        "Hash or omit raw local workspace contents before exporting beyond local storage.",
      ],
    },
  });
};

export const assertRealAcpRunManifestSafeForOptimizerInput = (
  manifestInput: RealAcpCorpusRunManifest,
): RealAcpCorpusRunManifest => {
  const manifest = RealAcpCorpusRunManifestSchema.parse(manifestInput);
  const blocked = manifest.taskResults.filter((result) => !result.redaction.optimizerSafe);
  if (blocked.length > 0) {
    throw new Error(`real ACP optimizer input rejected hidden or raw-local task results (${blocked.map((result) => result.taskId).join(", ")})`);
  }
  return manifest;
};

export const createSimulatedRealAcpExecutor = (
  options: SimulatedRealAcpExecutorOptions = {},
): RealAcpHeadlessExecutor => {
  const failTaskIds = new Set(options.failTaskIds ?? []);
  const skipTaskIds = new Set(options.skipTaskIds ?? []);

  return {
    executorId: "real-acp.executor.simulated",
    executorVersion: "simulated.v1",
    kind: "simulated",
    executeTask: async (input) => {
      if (skipTaskIds.has(input.task.taskId)) {
        return simulatedSkippedOutput(input.task);
      }
      const output = await simulatedSuccessfulOutput(input.task, input.workspacePath);
      if (!failTaskIds.has(input.task.taskId)) {
        return output;
      }
      return {
        ...output,
        status: "failed",
        failureReason: "simulated executor forced failure",
        verifier: {
          ...output.verifier,
          status: output.verifier.status === "skipped" ? "skipped" : "failed",
        },
      };
    },
  };
};

const runRealAcpTask = async (input: {
  runId: string;
  taskPackId: string;
  task: RealAcpCorpusTask;
  metadata: RealAcpRunMetadata;
  executor: RealAcpHeadlessExecutor;
  executionMode: RealAcpExecutionMode;
  dryRun: boolean;
  workspaceBaseDir: string;
  currentRepoPath: string;
  createdAt: string;
  signal?: AbortSignal;
}): Promise<RealAcpTaskRunResult> => {
  const task = RealAcpCorpusTaskSchema.parse(input.task);
  const runResultId = `${input.runId}.${safeId(task.taskId)}`;
  const materialized = await materializeRealAcpWorkspace(task, input.workspaceBaseDir);
  try {
    assertNotInsideCurrentRepo(materialized.workspacePath, input.currentRepoPath);
    const before = await snapshotDirectory(materialized.workspacePath);
    const executorOutput = await input.executor.executeTask({
      task,
      workspacePath: materialized.workspacePath,
      executionMode: input.executionMode,
      dryRun: input.dryRun,
      metadata: input.metadata,
      context: {
        signal: input.signal ?? new AbortController().signal,
        timeoutMs: task.timeoutMs,
      },
    });
    const after = await snapshotDirectory(materialized.workspacePath);
    const changedFiles = detectSnapshotChanges(before, after);
    return RealAcpTaskRunResultSchema.parse({
      schemaVersion: REAL_ACP_TASK_RESULT_SCHEMA_VERSION,
      runResultId,
      taskId: task.taskId,
      split: task.split,
      optimizationAllowed: task.optimizationAllowed,
      status: executorOutput.status,
      startedAt: input.createdAt,
      completedAt: input.createdAt,
      workspaceFingerprintBefore: fingerprintSnapshot(before),
      workspaceFingerprintAfter: fingerprintSnapshot(after),
      changedFiles,
      route: executorOutput.route,
      editStrategy: executorOutput.editStrategy,
      toolCalls: executorOutput.toolCalls ?? [],
      terminalCommands: executorOutput.terminalCommands ?? [],
      verifier: executorOutput.verifier,
      repair: executorOutput.repair,
      rollback: executorOutput.rollback,
      corrections: executorOutput.corrections ?? [],
      lineage: {
        taskId: task.taskId,
        runResultId,
        sourceTaskPackId: input.taskPackId,
        ...(executorOutput.lineage ?? {}),
      },
      telemetry: executorOutput.telemetry ?? {},
      redaction: taskRedaction(task),
      ...(executorOutput.failureReason === undefined ? {} : { failureReason: executorOutput.failureReason }),
      ...(executorOutput.skipReason === undefined ? {} : { skipReason: executorOutput.skipReason }),
    });
  } catch (error) {
    return RealAcpTaskRunResultSchema.parse({
      schemaVersion: REAL_ACP_TASK_RESULT_SCHEMA_VERSION,
      runResultId,
      taskId: task.taskId,
      split: task.split,
      optimizationAllowed: task.optimizationAllowed,
      status: "error",
      startedAt: input.createdAt,
      completedAt: input.createdAt,
      workspaceFingerprintBefore: "sha256:error",
      workspaceFingerprintAfter: "sha256:error",
      changedFiles: [],
      route: defaultRoute(task),
      editStrategy: noEditStrategy("executor error"),
      toolCalls: [],
      terminalCommands: [],
      verifier: verifierForTask(task, [], "not_run"),
      repair: { attempted: false, status: "skipped", reason: "executor error" },
      rollback: { attempted: false, status: "skipped", reason: "executor error" },
      corrections: [],
      lineage: {
        taskId: task.taskId,
        runResultId,
        sourceTaskPackId: input.taskPackId,
      },
      telemetry: { error: errorMessage(error) },
      redaction: taskRedaction(task),
      failureReason: errorMessage(error),
    });
  } finally {
    await materialized.cleanup();
  }
};

const simulatedSuccessfulOutput = async (
  task: RealAcpCorpusTask,
  workspacePath: string,
): Promise<RealAcpExecutorTaskOutput> => {
  const telemetry = telemetryFromAssertions(task.expectedOutcome.assertions);
  const terminalCommands = terminalRecordsForTask(task);
  const verifier = verifierForTask(task, terminalCommands, verifierStatusForTask(task));

  if (task.primaryLabel !== "cancellation" && task.expectedOutcome.mutation !== "rollback_to_original") {
    await applySimulatedTextAssertions(task, workspacePath);
  }

  return {
    status: defaultStatusForTask(task),
    route: defaultRoute(task),
    editStrategy: editStrategyForTask(task),
    toolCalls: toolRecordsForTask(task),
    terminalCommands,
    verifier,
    repair: repairForTask(task),
    rollback: rollbackForTask(task),
    corrections: correctionRecordsForTask(task),
    telemetry,
    ...(defaultStatusForTask(task) === "failed" ? { failureReason: "simulated expected failed outcome" } : {}),
    ...(defaultStatusForTask(task) === "skipped" ? { skipReason: task.expectedOutcome.verification.skipReason ?? "simulated skip" } : {}),
  };
};

const simulatedSkippedOutput = (task: RealAcpCorpusTask): RealAcpExecutorTaskOutput => ({
  status: "skipped",
  route: defaultRoute(task),
  editStrategy: noEditStrategy("simulated skip"),
  toolCalls: [],
  terminalCommands: [],
  verifier: verifierForTask(task, [], "skipped"),
  repair: { attempted: false, status: "skipped", reason: "simulated skip" },
  rollback: { attempted: false, status: "skipped", reason: "simulated skip" },
  corrections: [],
  telemetry: {
    verificationStatus: "skipped",
    skipReason: "simulated executor forced skip",
  },
  skipReason: "simulated executor forced skip",
});

const materializeRealAcpWorkspace = async (
  task: RealAcpCorpusTask,
  baseDir: string,
): Promise<{ workspacePath: string; cleanup: () => Promise<void> }> => {
  await mkdir(baseDir, { recursive: true });
  const materializedRoot = await mkdtemp(join(baseDir, "real-acp-run-"));
  const workspacePath = join(materializedRoot, "workspace");
  await mkdir(workspacePath, { recursive: true });
  for (const file of task.workspace.files) {
    const targetPath = resolveSafeWorkspacePath(workspacePath, file.path);
    await mkdir(dirname(targetPath), { recursive: true });
    await writeFile(targetPath, file.content, "utf8");
  }
  return {
    workspacePath,
    cleanup: async () => {
      await rm(materializedRoot, { recursive: true, force: true });
    },
  };
};

type SnapshotFile = {
  path: string;
  hash: string;
  content: string;
};

const snapshotDirectory = async (rootInput: string): Promise<SnapshotFile[]> => {
  const root = resolve(rootInput);
  const files: SnapshotFile[] = [];
  const visit = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const absolutePath = join(directory, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === ".bag") {
          continue;
        }
        await visit(absolutePath);
        continue;
      }
      if (!entry.isFile()) {
        continue;
      }
      const content = await readFile(absolutePath, "utf8");
      files.push({
        path: toPosix(relative(root, absolutePath)),
        hash: sha256(content),
        content,
      });
    }
  };
  await visit(root);
  return files.sort((left, right) => left.path.localeCompare(right.path));
};

const detectSnapshotChanges = (
  before: readonly SnapshotFile[],
  after: readonly SnapshotFile[],
): RealAcpChangedFileRecord[] => {
  const beforeByPath = new Map(before.map((file) => [file.path, file]));
  const afterByPath = new Map(after.map((file) => [file.path, file]));
  const paths = [...new Set([...beforeByPath.keys(), ...afterByPath.keys()])]
    .sort((left, right) => left.localeCompare(right));
  return paths.flatMap((path): RealAcpChangedFileRecord[] => {
    const beforeFile = beforeByPath.get(path);
    const afterFile = afterByPath.get(path);
    if (beforeFile === undefined && afterFile !== undefined) {
      return [{ path, changeKind: "added", afterHash: afterFile.hash }];
    }
    if (beforeFile !== undefined && afterFile === undefined) {
      return [{ path, changeKind: "deleted", beforeHash: beforeFile.hash }];
    }
    if (beforeFile !== undefined && afterFile !== undefined && beforeFile.hash !== afterFile.hash) {
      return [{ path, changeKind: "modified", beforeHash: beforeFile.hash, afterHash: afterFile.hash }];
    }
    return [];
  });
};

const fingerprintSnapshot = (snapshot: readonly SnapshotFile[]): string =>
  sha256(JSON.stringify(snapshot.map((file) => ({ path: file.path, hash: file.hash }))));

const applySimulatedTextAssertions = async (
  task: RealAcpCorpusTask,
  workspacePath: string,
): Promise<void> => {
  for (const assertion of task.expectedOutcome.assertions) {
    if (assertion.assertionKind !== "file_contains") {
      continue;
    }
    if (!task.expectedOutcome.expectedChangedPaths.includes(assertion.path)) {
      continue;
    }
    const targetPath = resolveSafeWorkspacePath(workspacePath, assertion.path);
    await mkdir(dirname(targetPath), { recursive: true });
    let content = "";
    try {
      content = await readFile(targetPath, "utf8");
    } catch {
      content = "";
    }
    if (!content.includes(assertion.text)) {
      await writeFile(targetPath, `${content}${content.endsWith("\n") || content.length === 0 ? "" : "\n"}${assertion.text}\n`, "utf8");
    }
  }
};

const telemetryFromAssertions = (
  assertions: readonly EvalAssertion[],
): JsonValue => {
  const telemetry: Record<string, JsonValue> = {};
  for (const assertion of assertions) {
    if (assertion.assertionKind === "json_pointer_equals" && assertion.artifact === "telemetry") {
      setJsonPointer(telemetry, assertion.pointer, assertion.expected);
    }
  }
  return telemetry;
};

const terminalRecordsForTask = (task: RealAcpCorpusTask): RealAcpTerminalRecord[] =>
  task.expectedOutcome.assertions.flatMap((assertion): RealAcpTerminalRecord[] => {
    if (assertion.assertionKind !== "command_exit_code") {
      return [];
    }
    const command = task.expectedOutcome.verification.commands[0] ?? ["true"];
    return [RealAcpTerminalRecordSchema.parse({
      commandId: assertion.commandId,
      command,
      status: assertion.expectedExitCode === 0 ? "succeeded" : "failed",
      exitCode: assertion.expectedExitCode,
      durationMs: 0,
    })];
  });

const toolRecordsForTask = (task: RealAcpCorpusTask): RealAcpToolRecord[] => {
  if (task.primaryLabel === "mcp_tool_failure") {
    return [
      {
        toolCallId: `tool.${safeId(task.taskId)}.mcp-failed`,
        namespace: "bag.acp",
        name: "fs.write",
        status: "failed",
        sideEffectLevel: "write",
        errorCode: "mcp_tool_failed",
      },
      {
        toolCallId: `tool.${safeId(task.taskId)}.fallback-write`,
        namespace: "bag.acp",
        name: "fs.write",
        status: "succeeded",
        sideEffectLevel: "write",
      },
    ];
  }
  if (task.expectedOutcome.mutation === "no_change") {
    return [];
  }
  return [{
    toolCallId: `tool.${safeId(task.taskId)}.workspace-write`,
    namespace: "bag.acp",
    name: "fs.write",
    status: "succeeded",
    sideEffectLevel: "write",
  }];
};

const verifierForTask = (
  task: RealAcpCorpusTask,
  terminalCommands: readonly RealAcpTerminalRecord[],
  status: RealAcpVerifierRecord["status"],
): RealAcpVerifierRecord => RealAcpVerifierRecordSchema.parse({
  status,
  policy: task.expectedOutcome.verification.policy,
  commandIds: terminalCommands.map((command) => command.commandId),
  ...(task.expectedOutcome.verification.skipReason === undefined
    ? {}
    : { skipReason: task.expectedOutcome.verification.skipReason }),
});

const verifierStatusForTask = (task: RealAcpCorpusTask): RealAcpVerifierRecord["status"] => {
  switch (task.expectedOutcome.verification.policy) {
    case "must_skip":
    case "allowed_to_skip":
      return task.expectedOutcome.verification.commands.length === 0 ? "skipped" : "passed";
    case "expected_to_fail_before_repair":
      return "failed";
    case "required":
      return "passed";
  }
};

const defaultStatusForTask = (task: RealAcpCorpusTask): RealAcpTaskOutcomeStatus => {
  switch (task.primaryLabel) {
    case "cancellation":
      return "cancelled";
    case "applied_but_broken":
      return "failed";
    case "verifier_skip":
      return "skipped";
    default:
      return "passed";
  }
};

const defaultRoute = (task: RealAcpCorpusTask): RealAcpRouteRecord => ({
  routeId: `route.${safeId(task.taskId)}`,
  selectedMode: task.primaryLabel === "cancellation" ? "cancelled" : "coding",
  reason: `Simulated route for ${task.primaryLabel}`,
  confidence: 1,
});

const editStrategyForTask = (task: RealAcpCorpusTask): RealAcpEditStrategyRecord => {
  if (task.expectedOutcome.mutation === "no_change") {
    return noEditStrategy("no final mutation expected");
  }
  return {
    strategyId: "edit.whole-file.acp-write.v1",
    family: "whole_file",
    selectedBy: "optimizer_policy",
    reason: `Selected for ${task.expectedOutcome.mutation}`,
  };
};

const noEditStrategy = (reason: string): RealAcpEditStrategyRecord => ({
  strategyId: "edit.none.v1",
  family: "none",
  selectedBy: "not_applicable",
  reason,
});

const repairForTask = (task: RealAcpCorpusTask): RealAcpRepairRecord =>
  task.primaryLabel === "applied_but_broken"
    ? { attempted: true, status: "failed", reason: "simulated applied-but-broken verifier failure" }
    : { attempted: false, status: "not_needed" };

const rollbackForTask = (task: RealAcpCorpusTask): RealAcpRollbackRecord =>
  task.expectedOutcome.mutation === "rollback_to_original"
    ? { attempted: true, status: "succeeded", reason: "simulated rollback after failed verifier" }
    : { attempted: false, status: "not_needed" };

const correctionRecordsForTask = (task: RealAcpCorpusTask): RealAcpCorrectionRecord[] =>
  task.correctionPrompts.map((_, index) => ({
    correctionId: `correction.${safeId(task.taskId)}.${index}`,
    promptIndex: index,
    applied: true,
    scopeChanged: true,
  }));

const taskRedaction = (task: RealAcpCorpusTask): RealAcpTaskRedaction => {
  const excludedFromOptimizerReasons: string[] = [];
  if (task.split === HIDDEN_SPLIT) {
    excludedFromOptimizerReasons.push("hidden holdout split");
  }
  if (!task.optimizationAllowed) {
    excludedFromOptimizerReasons.push("task is not optimizer-allowed");
  }
  return {
    rawLocalStatus: "raw_local_only",
    optimizerSafe: excludedFromOptimizerReasons.length === 0,
    excludedFromOptimizerReasons,
  };
};

const summarizeTaskResults = (
  taskResults: readonly RealAcpTaskRunResult[],
): RealAcpCorpusRunManifest["summary"] => ({
  total: taskResults.length,
  passed: taskResults.filter((result) => result.status === "passed").length,
  failed: taskResults.filter((result) => result.status === "failed").length,
  skipped: taskResults.filter((result) => result.status === "skipped").length,
  cancelled: taskResults.filter((result) => result.status === "cancelled").length,
  error: taskResults.filter((result) => result.status === "error").length,
  holdout: taskResults.filter((result) => result.split === HIDDEN_SPLIT).length,
});

const setJsonPointer = (
  target: Record<string, JsonValue>,
  pointer: string,
  value: JsonValue,
): void => {
  const parts = pointer.split("/").slice(1).map((part) => part.replaceAll("~1", "/").replaceAll("~0", "~"));
  let cursor: Record<string, JsonValue> = target;
  for (const [index, part] of parts.entries()) {
    if (index === parts.length - 1) {
      cursor[part] = value;
      return;
    }
    const nextPart = parts[index + 1];
    const existing = cursor[part];
    if (existing == null || typeof existing !== "object" || Array.isArray(existing)) {
      cursor[part] = nextPart != null && /^\d+$/.test(nextPart) ? [] : {};
    }
    const next = cursor[part];
    if (Array.isArray(next)) {
      const arrayIndex = Number(nextPart);
      if (!Number.isInteger(arrayIndex)) {
        throw new Error(`unsupported non-numeric array pointer segment: ${String(nextPart)}`);
      }
      if (next[arrayIndex] == null) {
        next[arrayIndex] = {};
      }
      const arrayValue = next[arrayIndex];
      if (arrayValue == null || typeof arrayValue !== "object" || Array.isArray(arrayValue)) {
        throw new Error(`unsupported scalar array pointer parent: ${pointer}`);
      }
      cursor = arrayValue;
      continue;
    }
    cursor = next as Record<string, JsonValue>;
  }
};

const resolveSafeWorkspacePath = (workspacePath: string, relativePath: string): string => {
  const normalizedRelativePath = RelativeSafePathSchema.parse(relativePath);
  const root = resolve(workspacePath);
  const target = resolve(root, normalizedRelativePath);
  if (!isInsideOrEqual(target, root)) {
    throw new Error(`workspace path escapes fixture root: ${relativePath}`);
  }
  return target;
};

const assertWorkspaceBaseIsSafe = (workspaceBaseDir: string, currentRepoPath: string): void => {
  if (isInsideOrEqual(workspaceBaseDir, currentRepoPath) && !isInsideOrEqual(workspaceBaseDir, resolve(currentRepoPath, SAFE_OUTPUT_ROOT))) {
    throw new Error("real ACP corpus runner refuses to materialize executor workspaces inside the current repo");
  }
};

const assertOutputDirIsSafe = (outputDir: string, currentRepoPath: string): void => {
  if (isInsideOrEqual(outputDir, currentRepoPath) && !isInsideOrEqual(outputDir, resolve(currentRepoPath, SAFE_OUTPUT_ROOT))) {
    throw new Error(`real ACP corpus output inside repo must stay under ${SAFE_OUTPUT_ROOT}`);
  }
};

const assertNotInsideCurrentRepo = (workspacePath: string, currentRepoPath: string): void => {
  if (isInsideOrEqual(workspacePath, currentRepoPath) && !isInsideOrEqual(workspacePath, resolve(currentRepoPath, SAFE_OUTPUT_ROOT))) {
    throw new Error("real ACP corpus runner refuses to pass the current repo path to an executor");
  }
};

const isInsideOrEqual = (candidatePath: string, rootPath: string): boolean => {
  const relativePath = relative(resolve(rootPath), resolve(candidatePath));
  return relativePath === "" || (!relativePath.startsWith("..") && !relativePath.includes(`..${sep}`));
};

const toPosix = (path: string): string => path.split(sep).join("/");

const safeId = (value: string): string => value.replace(/[^A-Za-z0-9._:-]+/g, "-");

const sha256 = (value: string): string =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);
