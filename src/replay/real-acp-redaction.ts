import { createHash } from "node:crypto";
import { z } from "zod";
import { EvalSplitSchema } from "../eval-harness/types";
import { JsonValueSchema, OptimizerIdSchema, type JsonValue } from "../optimizer/types";
import {
  RealAcpCorpusRunManifestSchema,
  RealAcpCorpusRunPurposeSchema,
  type RealAcpCorpusRunManifest,
  type RealAcpCorpusRunPurpose,
  type RealAcpTaskRunResult,
} from "./real-acp-runner";
import {
  RealAcpTaskLabelSchema,
  RealAcpTaskPackSchema,
  type RealAcpCorpusTask,
  type RealAcpTaskPack,
} from "./real-acp-task-pack";

const REAL_ACP_REPLAY_CASE_SCHEMA_VERSION = "real-acp-replay-case.v1" as const;
const REAL_ACP_REPLAY_EXPORT_SCHEMA_VERSION = "real-acp-replay-export.v1" as const;
const HIDDEN_SPLIT = "holdout" as const;
const RAW_TEXT_FIELD_PATTERN =
  /(?:content|fileContent|workspaceSnapshot|snapshot|stdout|stderr|terminalOutput|output|transcript|diff|patch|raw|body)$/i;

export const RealAcpReplayExportStatusSchema = z.enum([
  "optimizer_safe",
  "evaluation_only",
]);
export type RealAcpReplayExportStatus = z.infer<typeof RealAcpReplayExportStatusSchema>;

export const RealAcpReplayRedactionSummarySchema = z.object({
  status: z.enum(["redacted", "hash_only", "omitted"]),
  redactedFields: z.array(z.string().min(1)),
  secretReplacementCount: z.number().int().nonnegative(),
  pathHashCount: z.number().int().nonnegative(),
  omittedRawFieldCount: z.number().int().nonnegative(),
}).strict();
export type RealAcpReplayRedactionSummary = z.infer<typeof RealAcpReplayRedactionSummarySchema>;

export const RealAcpReplaySourceRefSchema = z.object({
  sourceKind: z.enum([
    "manifest",
    "task_pack",
    "task_result",
    "changed_file",
    "tool_call",
    "terminal_command",
  ]),
  refId: OptimizerIdSchema.optional(),
  artifactRef: z.string().min(1).optional(),
  path: z.string().min(1).optional(),
  contentHash: z.string().min(1).optional(),
  redactionStatus: z.enum(["redacted", "hash_only", "omitted"]),
}).strict();
export type RealAcpReplaySourceRef = z.infer<typeof RealAcpReplaySourceRefSchema>;

export const RealAcpReplayLineageSchema = z.object({
  runId: OptimizerIdSchema,
  taskPackId: OptimizerIdSchema,
  taskId: OptimizerIdSchema,
  runResultId: OptimizerIdSchema,
  sourceTaskPackId: OptimizerIdSchema,
  parentRunResultId: OptimizerIdSchema.optional(),
  correctionOfRunResultId: OptimizerIdSchema.optional(),
  repairOfRunResultId: OptimizerIdSchema.optional(),
  rollbackOfRunResultId: OptimizerIdSchema.optional(),
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  optimizerProfileId: OptimizerIdSchema,
  verificationPolicyVersion: z.string().min(1),
  resultStyleVersion: z.string().min(1),
  canonicalToolVersion: z.string().min(1),
  renderedToolVersion: z.string().min(1),
}).strict();
export type RealAcpReplayLineage = z.infer<typeof RealAcpReplayLineageSchema>;

export const RealAcpReplayAssertionSummarySchema = z.object({
  assertionId: OptimizerIdSchema,
  assertionKind: z.enum([
    "file_contains",
    "file_not_contains",
    "command_exit_code",
    "no_forbidden_path_changed",
    "json_pointer_equals",
    "llm_judge_min_score",
  ]),
  severity: z.enum(["info", "warning", "failure", "critical"]),
  description: z.string().min(1),
  path: z.string().min(1).optional(),
  commandId: OptimizerIdSchema.optional(),
  artifact: z.enum(["result", "telemetry", "scorecard"]).optional(),
  pointer: z.string().min(1).optional(),
  expectedHash: z.string().min(1).optional(),
  rubricId: OptimizerIdSchema.optional(),
  minimumScore: z.number().min(0).max(1).optional(),
}).strict();
export type RealAcpReplayAssertionSummary = z.infer<typeof RealAcpReplayAssertionSummarySchema>;

export const RealAcpReplayCaseRecordSchema = z.object({
  schemaVersion: z.literal(REAL_ACP_REPLAY_CASE_SCHEMA_VERSION),
  replayCaseId: OptimizerIdSchema,
  evalCaseId: OptimizerIdSchema,
  split: EvalSplitSchema,
  optimizerInputAllowed: z.boolean(),
  optimizerExclusionReasons: z.array(z.string().min(1)),
  title: z.string().min(1),
  taskSummary: z.string().min(1),
  labels: z.array(RealAcpTaskLabelSchema),
  sourceRefs: z.array(RealAcpReplaySourceRefSchema).min(1),
  lineage: RealAcpReplayLineageSchema,
  workspace: z.object({
    workspaceId: OptimizerIdSchema.optional(),
    kind: z.enum(["fixture", "greenfield"]).optional(),
    fileCount: z.number().int().nonnegative().optional(),
    allowedPathPrefixes: z.array(z.string().min(1)).default([]),
    protectedPaths: z.array(z.string().min(1)).default([]),
    rootFingerprintBefore: z.string().min(1),
    rootFingerprintAfter: z.string().min(1),
  }).strict(),
  expectedOutcome: z.object({
    mutation: z.enum([
      "edit_existing",
      "create_files",
      "no_change",
      "rollback_to_original",
      "detect_without_final_success",
      "unknown",
    ]),
    expectedChangedPaths: z.array(z.string().min(1)).default([]),
    expectedNoChangePaths: z.array(z.string().min(1)).default([]),
    verifierPolicy: z.enum([
      "required",
      "allowed_to_skip",
      "must_skip",
      "expected_to_fail_before_repair",
      "unknown",
    ]),
    assertionSummaries: z.array(RealAcpReplayAssertionSummarySchema).default([]),
  }).strict(),
  outcome: z.object({
    status: z.enum(["passed", "failed", "skipped", "cancelled", "error"]),
    passed: z.boolean(),
    failureReason: z.string().min(1).optional(),
    skipReason: z.string().min(1).optional(),
    verifierStatus: z.enum(["passed", "failed", "skipped", "not_run"]),
    routeSelectedMode: z.enum(["coding", "planning", "maintenance", "read_only", "cancelled"]),
    editStrategyFamily: z.enum(["whole_file", "diff", "search_replace", "none"]),
    repairStatus: z.enum(["not_needed", "succeeded", "failed", "skipped"]),
    rollbackStatus: z.enum(["not_needed", "succeeded", "failed", "skipped"]),
    correctionCount: z.number().int().nonnegative(),
  }).strict(),
  evidence: z.object({
    changedFiles: z.array(z.object({
      path: z.string().min(1),
      changeKind: z.enum(["added", "modified", "deleted"]),
      beforeHash: z.string().min(1).optional(),
      afterHash: z.string().min(1).optional(),
    }).strict()).default([]),
    toolCalls: z.array(z.object({
      toolCallId: OptimizerIdSchema,
      namespace: OptimizerIdSchema.optional(),
      name: OptimizerIdSchema,
      status: z.enum(["succeeded", "failed", "skipped", "blocked"]),
      sideEffectLevel: z.enum(["none", "read", "write", "network", "process"]),
      errorCode: OptimizerIdSchema.optional(),
    }).strict()).default([]),
    terminalCommands: z.array(z.object({
      commandId: OptimizerIdSchema,
      commandHash: z.string().min(1),
      commandPreview: z.array(z.string().min(1)),
      status: z.enum(["succeeded", "failed", "skipped", "timed_out"]),
      exitCode: z.number().int().nullable(),
      durationMs: z.number().int().nonnegative(),
    }).strict()).default([]),
    telemetry: JsonValueSchema,
  }).strict(),
  redaction: RealAcpReplayRedactionSummarySchema,
}).strict();
export type RealAcpReplayCaseRecord = z.infer<typeof RealAcpReplayCaseRecordSchema>;

export const RealAcpReplayOptimizationSelectionSchema = z.object({
  selectedReplayCaseIds: z.array(OptimizerIdSchema),
  selectedTaskResultIds: z.array(OptimizerIdSchema),
  hiddenHoldoutReplayCaseIds: z.array(OptimizerIdSchema),
  excludedReplayCaseIds: z.array(OptimizerIdSchema),
  rejectionReasons: z.record(OptimizerIdSchema, z.array(z.string().min(1))),
}).strict();
export type RealAcpReplayOptimizationSelection = z.infer<typeof RealAcpReplayOptimizationSelectionSchema>;

export const RealAcpReplayExportManifestSchema = z.object({
  schemaVersion: z.literal(REAL_ACP_REPLAY_EXPORT_SCHEMA_VERSION),
  exportId: OptimizerIdSchema,
  sourceRunId: OptimizerIdSchema,
  sourceTaskPackId: OptimizerIdSchema,
  createdAt: z.string().datetime({ offset: true }),
  purpose: RealAcpCorpusRunPurposeSchema,
  status: RealAcpReplayExportStatusSchema,
  includeHoldout: z.boolean(),
  optimizerInputAllowed: z.boolean(),
  sourceMetadata: z.object({
    executionMode: z.enum(["dry_run", "headless_acp", "real_consumer"]),
    dryRun: z.boolean(),
    modelProfileId: OptimizerIdSchema,
    codebaseProfileId: OptimizerIdSchema,
    clientProfileId: OptimizerIdSchema,
    policyId: OptimizerIdSchema,
    optimizerProfileId: OptimizerIdSchema,
  }).strict(),
  cases: z.array(RealAcpReplayCaseRecordSchema),
  optimizerSelection: RealAcpReplayOptimizationSelectionSchema,
  summary: z.object({
    totalSourceTaskResults: z.number().int().nonnegative(),
    exportedCases: z.number().int().nonnegative(),
    optimizerVisibleCases: z.number().int().nonnegative(),
    hiddenHoldoutCases: z.number().int().nonnegative(),
    failedCases: z.number().int().nonnegative(),
    skippedCases: z.number().int().nonnegative(),
    redactedCases: z.number().int().nonnegative(),
  }).strict(),
}).strict();
export type RealAcpReplayExportManifest = z.infer<typeof RealAcpReplayExportManifestSchema>;

export type CreateRealAcpReplayExportInput = {
  manifest: RealAcpCorpusRunManifest;
  taskPack?: RealAcpTaskPack;
  purpose?: RealAcpCorpusRunPurpose;
  status?: RealAcpReplayExportStatus;
  includeHoldout?: boolean;
  exportId?: string;
};

type RedactionAccumulator = {
  redactedFields: Set<string>;
  secretReplacementCount: number;
  pathHashCount: number;
  omittedRawFieldCount: number;
};

export const createRealAcpReplayExportManifest = (
  input: CreateRealAcpReplayExportInput,
): RealAcpReplayExportManifest => {
  const manifest = RealAcpCorpusRunManifestSchema.parse(input.manifest);
  const taskPack = input.taskPack === undefined ? undefined : RealAcpTaskPackSchema.parse(input.taskPack);
  const tasksById = new Map((taskPack?.tasks ?? []).map((task) => [task.taskId, task]));
  const purpose = RealAcpCorpusRunPurposeSchema.parse(input.purpose ?? manifest.purpose);
  const status = RealAcpReplayExportStatusSchema.parse(input.status ?? defaultExportStatus(purpose));
  const includeHoldout = input.includeHoldout ?? purpose === "holdout_final";

  assertHoldoutExportPolicy({ purpose, status, includeHoldout });

  const sourceCases = manifest.taskResults
    .filter((result) => includeHoldout || result.split !== HIDDEN_SPLIT)
    .map((result) => {
      const task = tasksById.get(result.taskId);
      return realAcpTaskResultToReplayCase({
        manifest,
        result,
        ...(task === undefined ? {} : { task }),
      });
    });
  const cases = purpose === "optimizer_input"
    ? sourceCases.filter((replayCase) => replayCase.optimizerInputAllowed)
    : sourceCases;
  const optimizerSelection = selectRealAcpReplayCasesForOptimizerInput(cases);
  const optimizerInputAllowed = status === "optimizer_safe" && optimizerSelection.excludedReplayCaseIds.length === 0;

  return RealAcpReplayExportManifestSchema.parse({
    schemaVersion: REAL_ACP_REPLAY_EXPORT_SCHEMA_VERSION,
    exportId: input.exportId ?? `real-acp-replay-export.${stableId(manifest.runId)}.${purpose}`,
    sourceRunId: manifest.runId,
    sourceTaskPackId: manifest.taskPackId,
    createdAt: manifest.createdAt,
    purpose,
    status,
    includeHoldout,
    optimizerInputAllowed,
    sourceMetadata: {
      executionMode: manifest.executionMode,
      dryRun: manifest.dryRun,
      modelProfileId: manifest.metadata.model.modelProfileId,
      codebaseProfileId: manifest.metadata.codebase.codebaseProfileId,
      clientProfileId: manifest.metadata.client.clientProfileId,
      policyId: manifest.metadata.profile.policyId,
      optimizerProfileId: manifest.metadata.profile.optimizerProfileId,
    },
    cases,
    optimizerSelection,
    summary: {
      totalSourceTaskResults: manifest.taskResults.length,
      exportedCases: cases.length,
      optimizerVisibleCases: optimizerSelection.selectedReplayCaseIds.length,
      hiddenHoldoutCases: cases.filter((replayCase) => replayCase.split === HIDDEN_SPLIT).length,
      failedCases: cases.filter((replayCase) => replayCase.outcome.status === "failed" || replayCase.outcome.status === "error").length,
      skippedCases: cases.filter((replayCase) => replayCase.outcome.status === "skipped" || replayCase.outcome.status === "cancelled").length,
      redactedCases: cases.filter((replayCase) => replayCase.redaction.status === "redacted").length,
    },
  });
};

export const realAcpTaskResultToReplayCase = (input: {
  manifest: RealAcpCorpusRunManifest;
  result: RealAcpTaskRunResult;
  task?: RealAcpCorpusTask;
}): RealAcpReplayCaseRecord => {
  const manifest = RealAcpCorpusRunManifestSchema.parse(input.manifest);
  const result = manifest.taskResults.find((taskResult) => taskResult.runResultId === input.result.runResultId)
    ?? input.result;
  const task = input.task === undefined ? undefined : input.task;
  const accumulator = createRedactionAccumulator();
  const changedFiles = result.changedFiles.map((file) => ({
    path: sanitizePath(file.path, accumulator, "changedFiles.path"),
    changeKind: file.changeKind,
    ...(file.beforeHash === undefined ? {} : { beforeHash: file.beforeHash }),
    ...(file.afterHash === undefined ? {} : { afterHash: file.afterHash }),
  }));
  const terminalCommands = result.terminalCommands.map((command) => {
    const redactedCommand = command.command.map((part) => redactText(part, accumulator, "terminalCommands.command"));
    return {
      commandId: command.commandId,
      commandHash: sha256(JSON.stringify(command.command)),
      commandPreview: redactedCommand,
      status: command.status,
      exitCode: command.exitCode,
      durationMs: command.durationMs,
    };
  });
  const optimizerExclusionReasons = optimizerExclusionReasonsForResult(result);

  return RealAcpReplayCaseRecordSchema.parse({
    schemaVersion: REAL_ACP_REPLAY_CASE_SCHEMA_VERSION,
    replayCaseId: `real-acp.replay.${stableId(result.runResultId)}`,
    evalCaseId: `real-acp.eval.${stableId(result.runResultId)}`,
    split: result.split,
    optimizerInputAllowed: optimizerExclusionReasons.length === 0,
    optimizerExclusionReasons,
    title: redactText(task?.title ?? result.taskId, accumulator, "title"),
    taskSummary: redactText(task?.userPrompt ?? `Real ACP task result ${result.taskId}`, accumulator, "taskSummary"),
    labels: task?.labels ?? [],
    sourceRefs: sourceRefsForResult(manifest, result, changedFiles),
    lineage: {
      runId: manifest.runId,
      taskPackId: manifest.taskPackId,
      taskId: result.taskId,
      runResultId: result.runResultId,
      sourceTaskPackId: result.lineage.sourceTaskPackId,
      ...(result.lineage.parentRunResultId === undefined ? {} : { parentRunResultId: result.lineage.parentRunResultId }),
      ...(result.lineage.correctionOfRunResultId === undefined ? {} : { correctionOfRunResultId: result.lineage.correctionOfRunResultId }),
      ...(result.lineage.repairOfRunResultId === undefined ? {} : { repairOfRunResultId: result.lineage.repairOfRunResultId }),
      ...(result.lineage.rollbackOfRunResultId === undefined ? {} : { rollbackOfRunResultId: result.lineage.rollbackOfRunResultId }),
      modelProfileId: manifest.metadata.model.modelProfileId,
      codebaseProfileId: manifest.metadata.codebase.codebaseProfileId,
      clientProfileId: manifest.metadata.client.clientProfileId,
      policyId: manifest.metadata.profile.policyId,
      optimizerProfileId: manifest.metadata.profile.optimizerProfileId,
      verificationPolicyVersion: manifest.metadata.profile.verificationPolicyVersion,
      resultStyleVersion: manifest.metadata.profile.resultStyleVersion,
      canonicalToolVersion: manifest.metadata.profile.canonicalToolVersion,
      renderedToolVersion: manifest.metadata.profile.renderedToolVersion,
    },
    workspace: {
      ...(task === undefined ? {} : {
        workspaceId: task.workspace.workspaceId,
        kind: task.workspace.kind,
        fileCount: task.workspace.files.length,
        allowedPathPrefixes: task.workspace.allowedPathPrefixes.map((path) =>
          sanitizePath(path, accumulator, "workspace.allowedPathPrefixes")),
        protectedPaths: task.workspace.protectedPaths.map((path) =>
          sanitizePath(path, accumulator, "workspace.protectedPaths")),
      }),
      rootFingerprintBefore: result.workspaceFingerprintBefore,
      rootFingerprintAfter: result.workspaceFingerprintAfter,
    },
    expectedOutcome: expectedOutcomeForTask(task, accumulator),
    outcome: {
      status: result.status,
      passed: result.status === "passed",
      ...(result.failureReason === undefined ? {} : { failureReason: redactText(result.failureReason, accumulator, "failureReason") }),
      ...(result.skipReason === undefined ? {} : { skipReason: redactText(result.skipReason, accumulator, "skipReason") }),
      verifierStatus: result.verifier.status,
      routeSelectedMode: result.route.selectedMode,
      editStrategyFamily: result.editStrategy.family,
      repairStatus: result.repair.status,
      rollbackStatus: result.rollback.status,
      correctionCount: result.corrections.length,
    },
    evidence: {
      changedFiles,
      toolCalls: result.toolCalls.map((toolCall) => ({
        toolCallId: toolCall.toolCallId,
        ...(toolCall.namespace === undefined ? {} : { namespace: toolCall.namespace }),
        name: toolCall.name,
        status: toolCall.status,
        sideEffectLevel: toolCall.sideEffectLevel,
        ...(toolCall.errorCode === undefined ? {} : { errorCode: toolCall.errorCode }),
      })),
      terminalCommands,
      telemetry: redactJsonValue(result.telemetry, accumulator, "telemetry"),
    },
    redaction: redactionSummary(accumulator),
  });
};

export const selectRealAcpReplayCasesForOptimizerInput = (
  replayCases: readonly RealAcpReplayCaseRecord[],
): RealAcpReplayOptimizationSelection => {
  const parsed = replayCases
    .map((replayCase) => RealAcpReplayCaseRecordSchema.parse(replayCase))
    .sort((left, right) => splitOrder(left.split) - splitOrder(right.split) || left.replayCaseId.localeCompare(right.replayCaseId));
  const selected = parsed.filter((replayCase) => replayCase.optimizerInputAllowed);
  const excluded = parsed.filter((replayCase) => !replayCase.optimizerInputAllowed);
  return RealAcpReplayOptimizationSelectionSchema.parse({
    selectedReplayCaseIds: selected.map((replayCase) => replayCase.replayCaseId),
    selectedTaskResultIds: selected.map((replayCase) => replayCase.lineage.runResultId),
    hiddenHoldoutReplayCaseIds: parsed
      .filter((replayCase) => replayCase.split === HIDDEN_SPLIT)
      .map((replayCase) => replayCase.replayCaseId),
    excludedReplayCaseIds: excluded.map((replayCase) => replayCase.replayCaseId),
    rejectionReasons: Object.fromEntries(excluded.map((replayCase) => [
      replayCase.replayCaseId,
      replayCase.optimizerExclusionReasons,
    ])),
  });
};

export const assertRealAcpReplayExportSafeForOptimizerInput = (
  exportManifestInput: RealAcpReplayExportManifest,
): RealAcpReplayExportManifest => {
  const exportManifest = RealAcpReplayExportManifestSchema.parse(exportManifestInput);
  if (exportManifest.purpose !== "optimizer_input" || exportManifest.status !== "optimizer_safe") {
    throw new Error(`real ACP replay export is not optimizer input: ${exportManifest.purpose}/${exportManifest.status}`);
  }
  if (exportManifest.optimizerSelection.excludedReplayCaseIds.length > 0) {
    throw new Error(`real ACP replay optimizer input rejected unsafe cases (${exportManifest.optimizerSelection.excludedReplayCaseIds.join(", ")})`);
  }
  if (exportManifest.optimizerSelection.hiddenHoldoutReplayCaseIds.length > 0) {
    throw new Error(`real ACP replay optimizer input rejected hidden holdout cases (${exportManifest.optimizerSelection.hiddenHoldoutReplayCaseIds.join(", ")})`);
  }
  return exportManifest;
};

const defaultExportStatus = (purpose: RealAcpCorpusRunPurpose): RealAcpReplayExportStatus =>
  purpose === "optimizer_input" ? "optimizer_safe" : "evaluation_only";

const assertHoldoutExportPolicy = (input: {
  purpose: RealAcpCorpusRunPurpose;
  status: RealAcpReplayExportStatus;
  includeHoldout: boolean;
}): void => {
  if (input.purpose === "optimizer_input" && input.includeHoldout) {
    throw new Error("real ACP replay optimizer input must exclude hidden holdout cases");
  }
  if (input.includeHoldout && (input.purpose !== "holdout_final" || input.status !== "evaluation_only")) {
    throw new Error("real ACP replay holdout export requires purpose holdout_final and status evaluation_only");
  }
  if (input.status === "optimizer_safe" && input.purpose !== "optimizer_input") {
    throw new Error("real ACP replay optimizer_safe export status is reserved for optimizer_input purpose");
  }
};

const optimizerExclusionReasonsForResult = (result: RealAcpTaskRunResult): string[] => {
  const reasons = [...result.redaction.excludedFromOptimizerReasons];
  if (result.split === HIDDEN_SPLIT && !reasons.includes("hidden holdout split")) {
    reasons.push("hidden holdout split");
  }
  if (!result.optimizationAllowed && !reasons.includes("task is not optimizer-allowed")) {
    reasons.push("task is not optimizer-allowed");
  }
  if (!result.redaction.optimizerSafe && reasons.length === 0) {
    reasons.push("task result redaction is not optimizer-safe");
  }
  return uniqueSorted(reasons);
};

const sourceRefsForResult = (
  manifest: RealAcpCorpusRunManifest,
  result: RealAcpTaskRunResult,
  changedFiles: readonly { path: string; beforeHash?: string; afterHash?: string }[],
): RealAcpReplaySourceRef[] => [
  {
    sourceKind: "manifest",
    refId: manifest.runId,
    artifactRef: `real-acp-run:${manifest.runId}`,
    redactionStatus: "redacted",
  },
  {
    sourceKind: "task_pack",
    refId: manifest.taskPackId,
    artifactRef: `real-acp-task-pack:${manifest.taskPackId}`,
    redactionStatus: "redacted",
  },
  {
    sourceKind: "task_result",
    refId: result.runResultId,
    artifactRef: `real-acp-task-result:${result.runResultId}`,
    redactionStatus: "redacted",
  },
  ...changedFiles.map((file) => ({
    sourceKind: "changed_file" as const,
    refId: result.runResultId,
    path: file.path,
    contentHash: file.afterHash ?? file.beforeHash,
    redactionStatus: "hash_only" as const,
  })),
  ...result.toolCalls.map((toolCall) => ({
    sourceKind: "tool_call" as const,
    refId: toolCall.toolCallId,
    redactionStatus: "redacted" as const,
  })),
  ...result.terminalCommands.map((command) => ({
    sourceKind: "terminal_command" as const,
    refId: command.commandId,
    contentHash: sha256(JSON.stringify(command.command)),
    redactionStatus: "hash_only" as const,
  })),
];

const expectedOutcomeForTask = (
  task: RealAcpCorpusTask | undefined,
  accumulator: RedactionAccumulator,
): RealAcpReplayCaseRecord["expectedOutcome"] => {
  if (task === undefined) {
    return {
      mutation: "unknown",
      expectedChangedPaths: [],
      expectedNoChangePaths: [],
      verifierPolicy: "unknown",
      assertionSummaries: [],
    };
  }
  return {
    mutation: task.expectedOutcome.mutation,
    expectedChangedPaths: task.expectedOutcome.expectedChangedPaths.map((path) =>
      sanitizePath(path, accumulator, "expectedOutcome.expectedChangedPaths")),
    expectedNoChangePaths: task.expectedOutcome.expectedNoChangePaths.map((path) =>
      sanitizePath(path, accumulator, "expectedOutcome.expectedNoChangePaths")),
    verifierPolicy: task.expectedOutcome.verification.policy,
    assertionSummaries: task.expectedOutcome.assertions.map((assertion) => {
      const base = {
        assertionId: assertion.assertionId,
        assertionKind: assertion.assertionKind,
        severity: assertion.severity,
        description: redactText(assertion.description, accumulator, "expectedOutcome.assertions.description"),
      };
      switch (assertion.assertionKind) {
        case "file_contains":
        case "file_not_contains":
          accumulator.redactedFields.add("expectedOutcome.assertions.text");
          accumulator.omittedRawFieldCount += 1;
          return {
            ...base,
            path: sanitizePath(assertion.path, accumulator, "expectedOutcome.assertions.path"),
            expectedHash: sha256(assertion.text),
          };
        case "command_exit_code":
          return { ...base, commandId: assertion.commandId };
        case "no_forbidden_path_changed":
          return {
            ...base,
            expectedHash: sha256(JSON.stringify(assertion.paths)),
          };
        case "json_pointer_equals":
          return {
            ...base,
            artifact: assertion.artifact,
            pointer: assertion.pointer,
            expectedHash: sha256(JSON.stringify(assertion.expected)),
          };
        case "llm_judge_min_score":
          return {
            ...base,
            rubricId: assertion.rubricId,
            minimumScore: assertion.minimumScore,
          };
      }
    }),
  };
};

const redactJsonValue = (
  value: JsonValue,
  accumulator: RedactionAccumulator,
  path: string,
): JsonValue => {
  if (typeof value === "string") {
    return redactText(value, accumulator, path);
  }
  if (Array.isArray(value)) {
    return value.map((item, index) => redactJsonValue(item, accumulator, `${path}.${index}`));
  }
  if (value !== null && typeof value === "object") {
    const redacted: Record<string, JsonValue> = {};
    for (const [key, child] of Object.entries(value)) {
      const childPath = `${path}.${key}`;
      if (RAW_TEXT_FIELD_PATTERN.test(key)) {
        accumulator.redactedFields.add(childPath);
        accumulator.omittedRawFieldCount += 1;
        redacted[key] = redactedRawField(child);
        continue;
      }
      redacted[key] = redactJsonValue(child, accumulator, childPath);
    }
    return redacted;
  }
  return value;
};

const redactText = (
  value: string,
  accumulator: RedactionAccumulator,
  field: string,
): string => {
  let output = value;
  output = output.replace(/github_pat_[A-Za-z0-9_]+/g, () => secretReplacement(accumulator, field));
  output = output.replace(/gh[pousr]_[A-Za-z0-9_]{20,}/g, () => secretReplacement(accumulator, field));
  output = output.replace(/sk-[A-Za-z0-9_-]{20,}/g, () => secretReplacement(accumulator, field));
  output = output.replace(/xox[baprs]-[A-Za-z0-9-]{20,}/g, () => secretReplacement(accumulator, field));
  output = output.replace(/\bAKIA[0-9A-Z]{16}\b/g, () => secretReplacement(accumulator, field));
  output = output.replace(
    /\b(?:api[_-]?key|token|secret|password|authorization)\s*[:=]\s*["']?[^"'\s,;]+/gi,
    (match) => {
      const separator = match.includes(":") ? ":" : "=";
      return `${match.slice(0, match.toLowerCase().indexOf(separator) + 1)}${secretReplacement(accumulator, field)}`;
    },
  );
  output = output.replace(/Bearer\s+[A-Za-z0-9._~+/=-]{20,}/gi, () => `Bearer ${secretReplacement(accumulator, field)}`);
  output = output.replace(/(?:\/Users|\/private\/tmp|\/tmp|\/var\/folders)\/[^\s"'`,;)]+/g, (match) =>
    sanitizePath(match, accumulator, field));
  output = output.replace(/[A-Za-z]:\\[^\s"'`,;)]+/g, (match) => sanitizePath(match, accumulator, field));
  if (output !== value) {
    accumulator.redactedFields.add(field);
  }
  return output;
};

const sanitizePath = (
  value: string,
  accumulator: RedactionAccumulator,
  field: string,
): string => {
  if (isLocalPath(value)) {
    accumulator.redactedFields.add(field);
    accumulator.pathHashCount += 1;
    return `path:sha256:${sha256(value).slice(0, 24)}`;
  }
  return redactSecretOnly(value, accumulator, field);
};

const redactSecretOnly = (
  value: string,
  accumulator: RedactionAccumulator,
  field: string,
): string => {
  const redacted = redactTextWithoutPaths(value, accumulator, field);
  if (redacted !== value) {
    accumulator.redactedFields.add(field);
  }
  return redacted;
};

const redactTextWithoutPaths = (
  value: string,
  accumulator: RedactionAccumulator,
  field: string,
): string => {
  let output = value;
  output = output.replace(/github_pat_[A-Za-z0-9_]+/g, () => secretReplacement(accumulator, field));
  output = output.replace(/gh[pousr]_[A-Za-z0-9_]{20,}/g, () => secretReplacement(accumulator, field));
  output = output.replace(/sk-[A-Za-z0-9_-]{20,}/g, () => secretReplacement(accumulator, field));
  return output;
};

const redactedRawField = (value: JsonValue): string =>
  `omitted:sha256:${sha256(JSON.stringify(value)).slice(0, 24)}`;

const secretReplacement = (
  accumulator: RedactionAccumulator,
  field: string,
): string => {
  accumulator.redactedFields.add(field);
  accumulator.secretReplacementCount += 1;
  return "[secret:redacted]";
};

const redactionSummary = (
  accumulator: RedactionAccumulator,
): RealAcpReplayRedactionSummary => {
  const redactedFields = [...accumulator.redactedFields].sort((left, right) => left.localeCompare(right));
  return RealAcpReplayRedactionSummarySchema.parse({
    status: redactedFields.length > 0 ? "redacted" : "hash_only",
    redactedFields,
    secretReplacementCount: accumulator.secretReplacementCount,
    pathHashCount: accumulator.pathHashCount,
    omittedRawFieldCount: accumulator.omittedRawFieldCount,
  });
};

const createRedactionAccumulator = (): RedactionAccumulator => ({
  redactedFields: new Set<string>(),
  secretReplacementCount: 0,
  pathHashCount: 0,
  omittedRawFieldCount: 0,
});

const isLocalPath = (value: string): boolean =>
  value.startsWith("/") || /^[A-Za-z]:\\/.test(value) || value.includes("\\Users\\");

const splitOrder = (split: "train" | "dev" | "holdout"): number => {
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

const stableId = (value: string): string => {
  const sanitized = value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "");
  return sanitized.length > 0 ? sanitized : sha256(value).slice(0, 12);
};

const sha256 = (value: string): string =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;
