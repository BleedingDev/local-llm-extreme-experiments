import { readdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { z } from "zod";
import { codingProgressClassFromTelemetry } from "../acp/coding-progress-diagnostics";
import { EvalSplitSchema } from "../eval-harness/types";
import { OptimizerIdSchema, type JsonValue } from "../optimizer/types";
import {
  RealAcpCorpusRunManifestSchema,
  type RealAcpCorpusRunManifest,
  type RealAcpTaskRunResult,
} from "./real-acp-runner";
import {
  realAcpCodingCorpusTaskPack,
  RealAcpTaskPackSchema,
  type RealAcpCorpusTask,
  type RealAcpTaskPack,
} from "./real-acp-task-pack";
import {
  RealAcpReplayCaseRecordSchema,
  RealAcpReplayExportManifestSchema,
  type RealAcpReplayCaseRecord,
} from "./real-acp-redaction";
import {
  noWriteValidationInputFromRealAcpTaskRunResult,
  NoWriteChangedFileSchema,
  NoWriteExpectedMutationSchema,
  NoWriteExpectedSideEffectSchema,
  NoWriteValidationInputSchema,
  type NoWriteExpectedMutation,
  type NoWriteExpectedSideEffect,
  type NoWriteValidationInput,
} from "./no-write-validation";

const NO_WRITE_REPLAY_SLICE_SCHEMA_VERSION = "no-write-replay-slice.v1" as const;
const DEFAULT_REPLAY_CORPUS_ROOT = join(".bag", "replay-corpus");
const HIDDEN_SPLIT = "holdout" as const;

const NoWriteReplaySliceCaseSchema = z.object({
  schemaVersion: z.literal(NO_WRITE_REPLAY_SLICE_SCHEMA_VERSION),
  sliceCaseId: OptimizerIdSchema,
  sourceKind: z.enum(["real_acp_task_run", "real_acp_replay_case"]),
  runId: OptimizerIdSchema,
  runResultId: OptimizerIdSchema,
  sourceCompletedAt: z.string().datetime({ offset: true }).optional(),
  taskId: OptimizerIdSchema,
  taskPackId: OptimizerIdSchema,
  split: EvalSplitSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  stopReason: z.string().min(1).optional(),
  editStrategyFamily: z.string().min(1),
  expectedMutation: NoWriteExpectedMutationSchema,
  expectedSideEffectLevel: NoWriteExpectedSideEffectSchema,
  fileWrites: z.object({
    changedFiles: z.array(NoWriteChangedFileSchema),
    fsWriteCount: z.number().int().nonnegative(),
  }).strict(),
  terminalActivity: z.object({
    terminalCreateCount: z.number().int().nonnegative(),
    terminalExitCount: z.number().int().nonnegative(),
    terminalCommandCount: z.number().int().nonnegative(),
  }).strict(),
  evidenceRefs: z.array(z.string().min(1)),
  validationInput: NoWriteValidationInputSchema,
}).strict();
export type NoWriteReplaySliceCase = z.infer<typeof NoWriteReplaySliceCaseSchema>;

const NoWriteReplaySliceStatusSchema = z.object({
  totalRecordsSeen: z.number().int().nonnegative(),
  includedCases: z.number().int().nonnegative(),
  skippedHiddenHoldout: z.number().int().nonnegative(),
  skippedUnsafeOrExcluded: z.number().int().nonnegative(),
  skippedDuplicate: z.number().int().nonnegative(),
}).strict();
export type NoWriteReplaySliceStatus = z.infer<typeof NoWriteReplaySliceStatusSchema>;

const NoWriteReplaySliceSchema = z.object({
  schemaVersion: z.literal(NO_WRITE_REPLAY_SLICE_SCHEMA_VERSION),
  sliceId: OptimizerIdSchema,
  status: NoWriteReplaySliceStatusSchema,
  cases: z.array(NoWriteReplaySliceCaseSchema),
}).strict();
export type NoWriteReplaySlice = z.infer<typeof NoWriteReplaySliceSchema>;

export type BuildNoWriteReplaySliceInput = {
  sliceId?: string;
  manifests?: readonly RealAcpCorpusRunManifest[];
  replayCases?: readonly RealAcpReplayCaseRecord[];
  taskPack?: RealAcpTaskPack;
  latestPerTaskProfile?: boolean;
};

export type BuildNoWriteReplaySliceFromCorpusInput = {
  corpusRoot?: string;
  sliceId?: string;
  taskPack?: RealAcpTaskPack;
};

export const buildNoWriteReplaySlice = (
  input: BuildNoWriteReplaySliceInput = {},
): NoWriteReplaySlice => {
  const taskPack = RealAcpTaskPackSchema.parse(input.taskPack ?? realAcpCodingCorpusTaskPack);
  const taskById = new Map(taskPack.tasks.map((task) => [task.taskId, task]));
  const cases: NoWriteReplaySliceCase[] = [];
  const seenRunResultIds = new Set<string>();
  const status: NoWriteReplaySliceStatus = {
    totalRecordsSeen: 0,
    includedCases: 0,
    skippedHiddenHoldout: 0,
    skippedUnsafeOrExcluded: 0,
    skippedDuplicate: 0,
  };

  for (const manifestInput of input.manifests ?? []) {
    const manifest = RealAcpCorpusRunManifestSchema.parse(manifestInput);
    for (const result of manifest.taskResults) {
      status.totalRecordsSeen += 1;
      if (isHiddenSplit(result.split)) {
        status.skippedHiddenHoldout += 1;
        continue;
      }
      if (!result.optimizationAllowed || !result.redaction.optimizerSafe) {
        status.skippedUnsafeOrExcluded += 1;
        continue;
      }
      if (seenRunResultIds.has(result.runResultId)) {
        status.skippedDuplicate += 1;
        continue;
      }
      seenRunResultIds.add(result.runResultId);
      cases.push(caseFromTaskRunResult({
        manifest,
        result,
        task: taskById.get(result.taskId),
      }));
    }
  }

  for (const replayCaseInput of input.replayCases ?? []) {
    const replayCase = RealAcpReplayCaseRecordSchema.parse(replayCaseInput);
    status.totalRecordsSeen += 1;
    if (isHiddenSplit(replayCase.split)) {
      status.skippedHiddenHoldout += 1;
      continue;
    }
    if (!replayCase.optimizerInputAllowed || replayCase.optimizerExclusionReasons.length > 0) {
      status.skippedUnsafeOrExcluded += 1;
      continue;
    }
    if (seenRunResultIds.has(replayCase.lineage.runResultId)) {
      status.skippedDuplicate += 1;
      continue;
    }
    seenRunResultIds.add(replayCase.lineage.runResultId);
    cases.push(caseFromReplayCase(replayCase));
  }

  const currentCases = input.latestPerTaskProfile === true
    ? latestCasesByTaskProfile(cases, status)
    : cases;
  const sortedCases = currentCases.sort((left, right) =>
    left.runId.localeCompare(right.runId) ||
    left.taskId.localeCompare(right.taskId) ||
    left.runResultId.localeCompare(right.runResultId));

  return NoWriteReplaySliceSchema.parse({
    schemaVersion: NO_WRITE_REPLAY_SLICE_SCHEMA_VERSION,
    sliceId: input.sliceId ?? "no-write-replay-slice.visible-acp",
    status: {
      ...status,
      includedCases: sortedCases.length,
    },
    cases: sortedCases,
  });
};

export const buildNoWriteReplaySliceFromCorpus = async (
  input: BuildNoWriteReplaySliceFromCorpusInput = {},
): Promise<NoWriteReplaySlice> => {
  const corpusRoot = resolve(input.corpusRoot ?? DEFAULT_REPLAY_CORPUS_ROOT);
  const manifests: RealAcpCorpusRunManifest[] = [];
  const replayCases: RealAcpReplayCaseRecord[] = [];

  for (const runDir of await safeReadDir(join(corpusRoot, "real-acp-runs"))) {
    const runPath = join(corpusRoot, "real-acp-runs", runDir);
    for (const fileName of await safeReadDir(runPath)) {
      const path = join(runPath, fileName);
      if (fileName.endsWith(".manifest.json")) {
        manifests.push(RealAcpCorpusRunManifestSchema.parse(await readJson(path)));
      } else if (fileName.endsWith(".replay-export.json")) {
        const replayExport = RealAcpReplayExportManifestSchema.parse(await readJson(path));
        replayCases.push(...replayExport.cases);
      }
    }
  }

  return buildNoWriteReplaySlice({
    ...(input.sliceId === undefined ? {} : { sliceId: input.sliceId }),
    manifests,
    replayCases,
    ...(input.taskPack === undefined ? {} : { taskPack: input.taskPack }),
  });
};

export const noWriteValidationInputsFromReplaySlice = (
  slice: NoWriteReplaySlice,
): NoWriteValidationInput[] =>
  NoWriteReplaySliceSchema.parse(slice).cases.map((sliceCase) => sliceCase.validationInput);

const caseFromTaskRunResult = (input: {
  manifest: RealAcpCorpusRunManifest;
  result: RealAcpTaskRunResult;
  task: RealAcpCorpusTask | undefined;
}): NoWriteReplaySliceCase => {
  const expectedMutation = expectedMutationForTask(input.task);
  const expectedSideEffectLevel = expectedSideEffectForMutation(expectedMutation);
  const evidenceRefs = evidenceRefsForTaskRun(input.manifest, input.result);
  const validationInput = noWriteValidationInputFromRealAcpTaskRunResult({
    result: input.result,
    expectedMutation,
    expectedSideEffect: expectedSideEffectLevel,
    evidenceRefs,
  });

  return NoWriteReplaySliceCaseSchema.parse({
    schemaVersion: NO_WRITE_REPLAY_SLICE_SCHEMA_VERSION,
    sliceCaseId: `no-write-slice.${stableId(input.result.runResultId)}`,
    sourceKind: "real_acp_task_run",
    runId: input.manifest.runId,
    runResultId: input.result.runResultId,
    sourceCompletedAt: input.result.completedAt,
    taskId: input.result.taskId,
    taskPackId: input.manifest.taskPackId,
    split: input.result.split,
    modelProfileId: input.manifest.metadata.model.modelProfileId,
    codebaseProfileId: input.manifest.metadata.codebase.codebaseProfileId,
    ...(validationInput.stopReason === undefined ? {} : { stopReason: validationInput.stopReason }),
    editStrategyFamily: validationInput.editStrategyFamily,
    expectedMutation,
    expectedSideEffectLevel,
    fileWrites: {
      changedFiles: validationInput.changedFiles,
      fsWriteCount: validationInput.fsWriteCount,
    },
    terminalActivity: {
      terminalCreateCount: validationInput.terminalCreateCount,
      terminalExitCount: validationInput.terminalExitCount,
      terminalCommandCount: validationInput.terminalCommandCount,
    },
    evidenceRefs: validationInput.evidenceRefs,
    validationInput,
  });
};

const caseFromReplayCase = (replayCase: RealAcpReplayCaseRecord): NoWriteReplaySliceCase => {
  const validationInput = validationInputFromReplayCase(replayCase);
  return NoWriteReplaySliceCaseSchema.parse({
    schemaVersion: NO_WRITE_REPLAY_SLICE_SCHEMA_VERSION,
    sliceCaseId: `no-write-slice.${stableId(replayCase.lineage.runResultId)}`,
    sourceKind: "real_acp_replay_case",
    runId: replayCase.lineage.runId,
    runResultId: replayCase.lineage.runResultId,
    taskId: replayCase.lineage.taskId,
    taskPackId: replayCase.lineage.taskPackId,
    split: replayCase.split,
    modelProfileId: replayCase.lineage.modelProfileId,
    codebaseProfileId: replayCase.lineage.codebaseProfileId,
    ...(validationInput.stopReason === undefined ? {} : { stopReason: validationInput.stopReason }),
    editStrategyFamily: validationInput.editStrategyFamily,
    expectedMutation: validationInput.expectedMutation,
    expectedSideEffectLevel: validationInput.expectedSideEffect,
    fileWrites: {
      changedFiles: validationInput.changedFiles,
      fsWriteCount: validationInput.fsWriteCount,
    },
    terminalActivity: {
      terminalCreateCount: validationInput.terminalCreateCount,
      terminalExitCount: validationInput.terminalExitCount,
      terminalCommandCount: validationInput.terminalCommandCount,
    },
    evidenceRefs: validationInput.evidenceRefs,
    validationInput,
  });
};

const validationInputFromReplayCase = (
  replayCase: RealAcpReplayCaseRecord,
): NoWriteValidationInput => {
  const counts = objectAt(objectAt(replayCase.evidence.telemetry, "headlessAcp"), "counts");
  const expectedMutation = NoWriteExpectedMutationSchema.parse(replayCase.expectedOutcome.mutation);
  const expectedSideEffect = expectedSideEffectForMutation(expectedMutation);
  const skipReason = replayCase.outcome.skipReason;
  const verifierSkippedJustification = skipReason === undefined
    ? undefined
    : {
      present: replayCase.outcome.verifierStatus === "skipped",
      reason: skipReason,
      policy: replayCase.expectedOutcome.verifierPolicy === "unknown"
        ? undefined
        : replayCase.expectedOutcome.verifierPolicy,
    };

  return NoWriteValidationInputSchema.parse({
    recordId: replayCase.lineage.runResultId,
    taskId: replayCase.lineage.taskId,
    routeSelectedMode: replayCase.outcome.routeSelectedMode,
    expectedMutation,
    expectedSideEffect,
    changedFiles: replayCase.evidence.changedFiles,
    fsWriteCount: numberAt(counts, "fsWrite") ?? replayCase.evidence.toolCalls.filter((tool) => tool.sideEffectLevel === "write").length,
    terminalCreateCount: numberAt(counts, "terminalCreate") ?? replayCase.evidence.terminalCommands.length,
    terminalExitCount: numberAt(counts, "terminalExit") ??
      replayCase.evidence.terminalCommands.filter((command) => command.exitCode !== null).length,
    terminalCommandCount: replayCase.evidence.terminalCommands.length,
    ...(stringAt(objectAt(replayCase.evidence.telemetry, "headlessAcp"), "stopReason") === undefined
      ? {}
      : { stopReason: stringAt(objectAt(replayCase.evidence.telemetry, "headlessAcp"), "stopReason") }),
    editStrategyFamily: replayCase.outcome.editStrategyFamily,
    ...(codingProgressClassFromTelemetry(replayCase.evidence.telemetry) === undefined
      ? {}
      : { codingProgressClass: codingProgressClassFromTelemetry(replayCase.evidence.telemetry) }),
    verifierStatus: replayCase.outcome.verifierStatus,
    ...(verifierSkippedJustification === undefined ? {} : { verifierSkippedJustification }),
    evidenceRefs: evidenceRefsForReplayCase(replayCase),
  });
};

const expectedMutationForTask = (task: RealAcpCorpusTask | undefined): NoWriteExpectedMutation =>
  task === undefined ? "unknown" : NoWriteExpectedMutationSchema.parse(task.expectedOutcome.mutation);

const expectedSideEffectForMutation = (
  mutation: NoWriteExpectedMutation,
): NoWriteExpectedSideEffect => {
  switch (mutation) {
    case "edit_existing":
    case "create_files":
    case "rollback_to_original":
    case "detect_without_final_success":
      return "mutation";
    case "no_change":
      return "read";
    case "unknown":
      return "unknown";
  }
};

const evidenceRefsForTaskRun = (
  manifest: RealAcpCorpusRunManifest,
  result: RealAcpTaskRunResult,
): string[] => uniqueStrings([
  `real-acp-run:${manifest.runId}`,
  `real-acp-task-pack:${manifest.taskPackId}`,
  `real-acp-task-result:${result.runResultId}`,
  `real-acp-model-profile:${manifest.metadata.model.modelProfileId}`,
  `real-acp-codebase-profile:${manifest.metadata.codebase.codebaseProfileId}`,
  ...result.toolCalls.map((tool) => `real-acp-tool-call:${tool.toolCallId}`),
  ...result.terminalCommands.map((command) => `real-acp-terminal-command:${command.commandId}`),
]);

const evidenceRefsForReplayCase = (
  replayCase: RealAcpReplayCaseRecord,
): string[] => uniqueStrings([
  `real-acp-replay-case:${replayCase.replayCaseId}`,
  `real-acp-run:${replayCase.lineage.runId}`,
  `real-acp-task-pack:${replayCase.lineage.taskPackId}`,
  `real-acp-task-result:${replayCase.lineage.runResultId}`,
  `real-acp-model-profile:${replayCase.lineage.modelProfileId}`,
  `real-acp-codebase-profile:${replayCase.lineage.codebaseProfileId}`,
  ...replayCase.sourceRefs.flatMap((ref) => [
    ref.artifactRef,
    ref.refId === undefined ? undefined : `real-acp-source-ref:${ref.sourceKind}:${ref.refId}`,
  ]),
]);

const latestCasesByTaskProfile = (
  cases: readonly NoWriteReplaySliceCase[],
  status: NoWriteReplaySliceStatus,
): NoWriteReplaySliceCase[] => {
  const byKey = new Map<string, NoWriteReplaySliceCase>();
  for (const candidate of cases) {
    const key = [
      candidate.taskPackId,
      candidate.taskId,
      candidate.modelProfileId,
      candidate.codebaseProfileId,
    ].join("|");
    const current = byKey.get(key);
    if (current === undefined || compareCaseFreshness(candidate, current) > 0) {
      if (current !== undefined) {
        status.skippedDuplicate += 1;
      }
      byKey.set(key, candidate);
    } else {
      status.skippedDuplicate += 1;
    }
  }
  return [...byKey.values()];
};

const compareCaseFreshness = (
  left: NoWriteReplaySliceCase,
  right: NoWriteReplaySliceCase,
): number => {
  const leftTime = freshnessMillis(left);
  const rightTime = freshnessMillis(right);
  if (leftTime !== rightTime) return leftTime - rightTime;
  return left.runId.localeCompare(right.runId) ||
    left.runResultId.localeCompare(right.runResultId);
};

const freshnessMillis = (candidate: NoWriteReplaySliceCase): number =>
  Math.max(
    timestampMillis(candidate.sourceCompletedAt),
    timestampFromRunId(candidate.runId),
  );

const timestampMillis = (value: string | undefined): number => {
  if (value === undefined) return Number.NEGATIVE_INFINITY;
  const millis = Date.parse(value);
  return Number.isFinite(millis) ? millis : Number.NEGATIVE_INFINITY;
};

const timestampFromRunId = (runId: string): number => {
  const match = /(?:^|[^0-9])([0-9]{4})([0-9]{2})([0-9]{2})(?:[^0-9]|$)/.exec(runId);
  if (match === null) return Number.NEGATIVE_INFINITY;
  const [, year, month, day] = match;
  return Date.UTC(Number(year), Number(month) - 1, Number(day));
};

const safeReadDir = async (path: string): Promise<string[]> => {
  try {
    return (await readdir(path, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() || entry.isFile())
      .map((entry) => entry.name)
      .sort((left, right) => left.localeCompare(right));
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
};

const readJson = async (path: string): Promise<unknown> =>
  JSON.parse(await readFile(path, "utf8"));

const isHiddenSplit = (split: string): boolean => split === HIDDEN_SPLIT;

const stableId = (value: string): string =>
  value.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "unknown";

const uniqueStrings = (values: readonly (string | undefined)[]): string[] =>
  [...new Set(values.filter((value): value is string => value !== undefined && value.length > 0))]
    .sort((left, right) => left.localeCompare(right));

const objectAt = (value: JsonValue | undefined, key: string): Record<string, JsonValue> | undefined => {
  if (value == null || Array.isArray(value) || typeof value !== "object") return undefined;
  const child = value[key];
  if (child == null || Array.isArray(child) || typeof child !== "object") return undefined;
  return child;
};

const numberAt = (value: Record<string, JsonValue> | undefined, key: string): number | undefined => {
  const child = value?.[key];
  return typeof child === "number" && Number.isFinite(child) ? child : undefined;
};

const stringAt = (value: Record<string, JsonValue> | undefined, key: string): string | undefined => {
  const child = value?.[key];
  return typeof child === "string" && child.length > 0 ? child : undefined;
};
