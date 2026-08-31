import { createHash } from "node:crypto";
import { z } from "zod";
import { RedactionStatusSchema } from "../edit-strategy/types";
import { EvalSplitSchema, type EvalSplit } from "../eval-harness/types";
import { OptimizerIdSchema, type JsonValue } from "../optimizer/types";
import {
  RealAcpCorpusRunManifestSchema,
  RealAcpTaskOutcomeStatusSchema,
  type RealAcpCorpusRunManifest,
  type RealAcpTaskRunResult,
} from "./real-acp-runner";
import {
  RealAcpTaskPackSchema,
  realAcpCodingCorpusTaskPack,
  type RealAcpCorpusTask,
  type RealAcpTaskPack,
} from "./real-acp-task-pack";
import {
  ReplayEvalCaseSkeletonSchema,
  type ReplayEvalCaseSkeleton,
} from "./extraction";

const INDEX_SCHEMA_VERSION = "real-acp-replay-corpus-index.v1" as const;
const SPLIT_ORDER: Record<EvalSplit, number> = { train: 0, dev: 1, holdout: 2 };

const RealAcpReplayCorpusIndexSourceRefSchema = z.object({
  sourceKind: z.enum([
    "real_acp_run_manifest",
    "real_acp_task_result",
    "redacted_replay_case",
    "capture",
    "record",
    "trace",
    "span",
    "artifact",
    "fixture",
  ]),
  runId: OptimizerIdSchema.optional(),
  runResultId: OptimizerIdSchema.optional(),
  taskId: OptimizerIdSchema.optional(),
  taskPackId: OptimizerIdSchema.optional(),
  replayCaseId: OptimizerIdSchema.optional(),
  captureId: OptimizerIdSchema.optional(),
  recordId: OptimizerIdSchema.optional(),
  traceId: z.string().min(1).optional(),
  spanId: z.string().min(1).optional(),
  artifactRef: z.string().min(1).optional(),
  path: z.string().min(1).optional(),
  redactionStatus: RedactionStatusSchema.optional(),
}).strict();
export type RealAcpReplayCorpusIndexSourceRef = z.infer<typeof RealAcpReplayCorpusIndexSourceRefSchema>;

const RealAcpReplayCorpusIndexSafetyLabelSchema = z.enum([
  "visible_optimizer_safe",
  "hidden_holdout",
  "raw_local_only",
  "needs_review",
  "optimizer_allowed",
  "optimizer_blocked",
]);
export type RealAcpReplayCorpusIndexSafetyLabel = z.infer<typeof RealAcpReplayCorpusIndexSafetyLabelSchema>;

const RealAcpReplayCorpusIndexReproductionSchema = z.object({
  command: z.array(z.string().min(1)).min(1),
  sourceArtifactPath: z.string().min(1).optional(),
  cwd: z.string().min(1).optional(),
  notes: z.array(z.string().min(1)).default([]),
}).strict();
export type RealAcpReplayCorpusIndexReproduction = z.infer<typeof RealAcpReplayCorpusIndexReproductionSchema>;

export const RealAcpReplayCorpusIndexRecordSchema = z.object({
  schemaVersion: z.literal(INDEX_SCHEMA_VERSION),
  indexRecordId: OptimizerIdSchema,
  sourceKind: z.enum(["real_acp_run_result", "redacted_replay_case"]),
  runId: OptimizerIdSchema.optional(),
  runResultId: OptimizerIdSchema.optional(),
  taskId: OptimizerIdSchema.optional(),
  taskPackId: OptimizerIdSchema.optional(),
  replayCaseId: OptimizerIdSchema.optional(),
  captureId: OptimizerIdSchema.optional(),
  title: z.string().min(1).optional(),
  split: EvalSplitSchema,
  status: RealAcpTaskOutcomeStatusSchema.optional(),
  scores: z.record(z.string(), z.number().finite()).default({}),
  sourceRefs: z.array(RealAcpReplayCorpusIndexSourceRefSchema).min(1),
  labels: z.object({
    safety: z.array(RealAcpReplayCorpusIndexSafetyLabelSchema).default([]),
    task: z.array(OptimizerIdSchema).default([]),
  }).strict(),
  identities: z.object({
    modelProfileId: OptimizerIdSchema.optional(),
    provider: z.string().min(1).optional(),
    model: z.string().min(1).optional(),
    codebaseProfileId: OptimizerIdSchema.optional(),
    clientProfileId: OptimizerIdSchema.optional(),
    policyId: OptimizerIdSchema.optional(),
    optimizerProfileId: OptimizerIdSchema.optional(),
  }).strict(),
  reproduction: RealAcpReplayCorpusIndexReproductionSchema,
  safety: z.object({
    hiddenHoldout: z.boolean(),
    optimizerSafe: z.boolean(),
    optimizerInputAllowed: z.boolean(),
    rawLocalOnly: z.boolean(),
    optimizationAllowed: z.boolean().optional(),
    redactionStatus: RedactionStatusSchema.optional(),
    excludedFromOptimizerReasons: z.array(z.string().min(1)).default([]),
  }).strict(),
}).strict().superRefine((record, ctx) => {
  if (record.split === "holdout" && !record.safety.hiddenHoldout) {
    ctx.addIssue({
      code: "custom",
      path: ["safety", "hiddenHoldout"],
      message: "holdout records must be flagged as hidden holdout",
    });
  }
  if (record.split !== "holdout" && record.safety.hiddenHoldout) {
    ctx.addIssue({
      code: "custom",
      path: ["safety", "hiddenHoldout"],
      message: "only holdout records may be flagged as hidden holdout",
    });
  }
  if (record.safety.hiddenHoldout && record.safety.optimizerInputAllowed) {
    ctx.addIssue({
      code: "custom",
      path: ["safety", "optimizerInputAllowed"],
      message: "hidden holdout records cannot be optimizer input",
    });
  }
  if (record.safety.optimizerInputAllowed && !record.safety.optimizerSafe) {
    ctx.addIssue({
      code: "custom",
      path: ["safety", "optimizerInputAllowed"],
      message: "optimizer input records must be optimizer safe",
    });
  }
});
export type RealAcpReplayCorpusIndexRecord = z.infer<typeof RealAcpReplayCorpusIndexRecordSchema>;

export type BuildRealAcpReplayCorpusIndexInput = {
  runManifests?: readonly RealAcpCorpusRunManifest[];
  replayCases?: readonly ReplayEvalCaseSkeleton[];
  taskPacks?: readonly RealAcpTaskPack[];
  reproductionCommand?: readonly string[];
  reproductionCwd?: string;
};

export const buildRealAcpReplayCorpusIndex = (
  input: BuildRealAcpReplayCorpusIndexInput,
): RealAcpReplayCorpusIndexRecord[] => {
  const taskById = taskMap(input.taskPacks);
  const records = [
    ...(input.runManifests ?? []).flatMap((manifestInput) =>
      recordsFromRunManifest(
        RealAcpCorpusRunManifestSchema.parse(manifestInput),
        taskById,
        input,
      )),
    ...(input.replayCases ?? []).map((caseInput) =>
      recordFromReplayCase(
        ReplayEvalCaseSkeletonSchema.parse(caseInput),
        input,
      )),
  ];
  assertNoDuplicateIndexInputs(records);
  return records
    .sort(compareIndexRecords)
    .map((record) => RealAcpReplayCorpusIndexRecordSchema.parse(record));
};

export const serializeRealAcpReplayCorpusIndexJsonl = (
  recordsInput: readonly RealAcpReplayCorpusIndexRecord[],
): string => {
  const records = recordsInput
    .map((record) => RealAcpReplayCorpusIndexRecordSchema.parse(record))
    .sort(compareIndexRecords);
  assertNoDuplicateIndexInputs(records);
  return records.map((record) => JSON.stringify(record)).join("\n") + (records.length === 0 ? "" : "\n");
};

const recordsFromRunManifest = (
  manifest: RealAcpCorpusRunManifest,
  taskById: ReadonlyMap<string, RealAcpCorpusTask>,
  input: Pick<BuildRealAcpReplayCorpusIndexInput, "reproductionCommand" | "reproductionCwd">,
): RealAcpReplayCorpusIndexRecord[] =>
  manifest.taskResults.map((result) => {
    const task = taskById.get(result.taskId);
    const replayCaseId = replayCaseIdFromTelemetry(result.telemetry);
    const hiddenHoldout = result.split === "holdout";
    const optimizerSafe = result.redaction.optimizerSafe && !hiddenHoldout;
    const optimizerInputAllowed = optimizerSafe && result.optimizationAllowed;
    const sourceArtifactPath = manifest.manifestPath;
    return RealAcpReplayCorpusIndexRecordSchema.parse({
      schemaVersion: INDEX_SCHEMA_VERSION,
      indexRecordId: optimizerId(`real-acp-index.run.${manifest.runId}.${result.runResultId}`),
      sourceKind: "real_acp_run_result",
      runId: manifest.runId,
      runResultId: result.runResultId,
      taskId: result.taskId,
      taskPackId: manifest.taskPackId,
      ...(replayCaseId === undefined ? {} : { replayCaseId }),
      ...(task === undefined ? {} : { title: task.title }),
      split: result.split,
      status: result.status,
      scores: {
        outcomeScore: outcomeScore(result.status),
        ...collectScoreFields(result.telemetry, "telemetry"),
      },
      sourceRefs: [
        {
          sourceKind: "real_acp_run_manifest",
          runId: manifest.runId,
          taskPackId: manifest.taskPackId,
          ...(sourceArtifactPath === undefined ? {} : { path: sourceArtifactPath }),
        },
        {
          sourceKind: "real_acp_task_result",
          runId: manifest.runId,
          runResultId: result.runResultId,
          taskId: result.taskId,
          taskPackId: manifest.taskPackId,
        },
      ],
      labels: {
        safety: safetyLabels({
          hiddenHoldout,
          optimizerSafe,
          optimizerInputAllowed,
          rawLocalOnly: result.redaction.rawLocalStatus === "raw_local_only",
        }),
        task: task?.labels ?? [],
      },
      identities: {
        modelProfileId: manifest.metadata.model.modelProfileId,
        provider: manifest.metadata.model.provider,
        model: manifest.metadata.model.model,
        codebaseProfileId: manifest.metadata.codebase.codebaseProfileId,
        clientProfileId: manifest.metadata.client.clientProfileId,
        policyId: manifest.metadata.profile.policyId,
        optimizerProfileId: manifest.metadata.profile.optimizerProfileId,
      },
      reproduction: reproductionMetadata({
        command: input.reproductionCommand,
        cwd: input.reproductionCwd,
        sourceArtifactPath,
        fallbackCommand: sourceArtifactPath === undefined
          ? ["bag", "replay", "real-acp-run", "--run-id", manifest.runId]
          : ["bag", "replay", "real-acp-run", "--manifest", sourceArtifactPath],
      }),
      safety: {
        hiddenHoldout,
        optimizerSafe,
        optimizerInputAllowed,
        rawLocalOnly: result.redaction.rawLocalStatus === "raw_local_only",
        optimizationAllowed: result.optimizationAllowed,
        excludedFromOptimizerReasons: [
          ...result.redaction.excludedFromOptimizerReasons,
          ...(hiddenHoldout && !result.redaction.excludedFromOptimizerReasons.includes("hidden holdout split")
            ? ["hidden holdout split"]
            : []),
        ],
      },
    });
  });

const recordFromReplayCase = (
  replayCase: ReplayEvalCaseSkeleton,
  input: Pick<BuildRealAcpReplayCorpusIndexInput, "reproductionCommand" | "reproductionCwd">,
): RealAcpReplayCorpusIndexRecord => {
  const hiddenHoldout = replayCase.split === "holdout";
  const rawLocalOnly = replayCase.redaction.status === "raw_local_only";
  const needsReview = replayCase.redaction.status === "needs_review" || replayCase.redaction.needsReview;
  const optimizerSafe = !hiddenHoldout && !rawLocalOnly && !needsReview;
  const optimizerInputAllowed = optimizerSafe;
  const sourceArtifactPath = replayCase.sourceRefs.find((sourceRef) => sourceRef.path !== undefined)?.path;
  return RealAcpReplayCorpusIndexRecordSchema.parse({
    schemaVersion: INDEX_SCHEMA_VERSION,
    indexRecordId: optimizerId(`real-acp-index.case.${replayCase.evalCaseId}`),
    sourceKind: "redacted_replay_case",
    replayCaseId: replayCase.evalCaseId,
    captureId: replayCase.captureId,
    title: replayCase.title,
    split: replayCase.split,
    scores: {},
    sourceRefs: [
      {
        sourceKind: "redacted_replay_case",
        replayCaseId: replayCase.evalCaseId,
        captureId: replayCase.captureId,
        redactionStatus: replayCase.redaction.status,
      },
      ...replayCase.sourceRefs.map((sourceRef) => RealAcpReplayCorpusIndexSourceRefSchema.parse({
        sourceKind: sourceRef.sourceKind,
        ...(sourceRef.captureId === undefined ? {} : { captureId: sourceRef.captureId }),
        ...(sourceRef.recordId === undefined ? {} : { recordId: sourceRef.recordId }),
        ...(sourceRef.traceId === undefined ? {} : { traceId: sourceRef.traceId }),
        ...(sourceRef.spanId === undefined ? {} : { spanId: sourceRef.spanId }),
        ...(sourceRef.artifactRef === undefined ? {} : { artifactRef: sourceRef.artifactRef }),
        ...(sourceRef.path === undefined ? {} : { path: sourceRef.path }),
        ...(sourceRef.redactionStatus === undefined ? {} : { redactionStatus: sourceRef.redactionStatus }),
      })),
    ],
    labels: {
      safety: safetyLabels({
        hiddenHoldout,
        optimizerSafe,
        optimizerInputAllowed,
        rawLocalOnly,
        needsReview,
      }),
      task: replayCase.tags,
    },
    identities: {},
    reproduction: reproductionMetadata({
      command: input.reproductionCommand,
      cwd: input.reproductionCwd,
      sourceArtifactPath,
      fallbackCommand: sourceArtifactPath === undefined
        ? ["bag", "replay", "case", "--case-id", replayCase.evalCaseId]
        : ["bag", "replay", "case", "--case", sourceArtifactPath],
    }),
    safety: {
      hiddenHoldout,
      optimizerSafe,
      optimizerInputAllowed,
      rawLocalOnly,
      redactionStatus: replayCase.redaction.status,
      excludedFromOptimizerReasons: [
        ...(hiddenHoldout ? ["hidden holdout split"] : []),
        ...(rawLocalOnly ? ["raw local only redaction"] : []),
        ...(needsReview ? ["redaction needs review"] : []),
      ],
    },
  });
};

const taskMap = (
  taskPacks: readonly RealAcpTaskPack[] | undefined,
): ReadonlyMap<string, RealAcpCorpusTask> => {
  const packs = [
    realAcpCodingCorpusTaskPack,
    ...(taskPacks ?? []),
  ].map((pack) => RealAcpTaskPackSchema.parse(pack));
  return new Map(packs.flatMap((pack) => pack.tasks.map((task) => [task.taskId, task] as const)));
};

const safetyLabels = (input: {
  hiddenHoldout: boolean;
  optimizerSafe: boolean;
  optimizerInputAllowed: boolean;
  rawLocalOnly: boolean;
  needsReview?: boolean;
}): RealAcpReplayCorpusIndexSafetyLabel[] => {
  const labels: RealAcpReplayCorpusIndexSafetyLabel[] = [];
  if (input.hiddenHoldout) labels.push("hidden_holdout");
  if (input.rawLocalOnly) labels.push("raw_local_only");
  if (input.needsReview === true) labels.push("needs_review");
  if (input.optimizerSafe && !input.hiddenHoldout) labels.push("visible_optimizer_safe");
  labels.push(input.optimizerInputAllowed ? "optimizer_allowed" : "optimizer_blocked");
  return uniqueSorted(labels) as RealAcpReplayCorpusIndexSafetyLabel[];
};

const reproductionMetadata = (input: {
  command: readonly string[] | undefined;
  fallbackCommand: readonly string[];
  sourceArtifactPath: string | undefined;
  cwd: string | undefined;
}): RealAcpReplayCorpusIndexReproduction =>
  RealAcpReplayCorpusIndexReproductionSchema.parse({
    command: [...(input.command ?? input.fallbackCommand)],
    ...(input.sourceArtifactPath === undefined ? {} : { sourceArtifactPath: input.sourceArtifactPath }),
    ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
    notes: input.command === undefined
      ? ["Derived replay command metadata; verify the command against the runner entrypoint before publishing."]
      : [],
  });

const outcomeScore = (status: RealAcpTaskRunResult["status"]): number =>
  status === "passed" ? 1 : 0;

const collectScoreFields = (
  value: JsonValue,
  prefix: string,
  depth = 0,
): Record<string, number> => {
  if (depth > 5 || value == null || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  const scores: Record<string, number> = {};
  for (const [key, nested] of Object.entries(value)) {
    const path = `${prefix}.${key}`;
    if (typeof nested === "number" && Number.isFinite(nested) && /score/i.test(key)) {
      scores[path] = nested;
      continue;
    }
    Object.assign(scores, collectScoreFields(nested, path, depth + 1));
  }
  return scores;
};

const replayCaseIdFromTelemetry = (telemetry: JsonValue): string | undefined => {
  if (telemetry == null || typeof telemetry !== "object" || Array.isArray(telemetry)) {
    return undefined;
  }
  const value = telemetry.replayCaseId;
  return typeof value === "string" && value.length > 0
    ? OptimizerIdSchema.parse(value)
    : undefined;
};

const assertNoDuplicateIndexInputs = (
  records: readonly RealAcpReplayCorpusIndexRecord[],
): void => {
  assertUnique(records.flatMap((record) =>
    record.runResultId === undefined ? [] : [`${record.sourceKind}:${record.runResultId}`]), "runResultId");
  assertUnique(records.flatMap((record) =>
    record.replayCaseId === undefined ? [] : [`${record.sourceKind}:${record.replayCaseId}`]), "replayCaseId");
  assertUnique(records.map((record) => record.indexRecordId), "indexRecordId");
};

const assertUnique = (values: readonly string[], label: string): void => {
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      duplicates.add(value);
    }
    seen.add(value);
  }
  if (duplicates.size > 0) {
    throw new Error(`duplicate real ACP replay corpus index ${label}: ${[...duplicates].sort().join(", ")}`);
  }
};

const compareIndexRecords = (
  left: RealAcpReplayCorpusIndexRecord,
  right: RealAcpReplayCorpusIndexRecord,
): number =>
  SPLIT_ORDER[left.split] - SPLIT_ORDER[right.split]
  || left.sourceKind.localeCompare(right.sourceKind)
  || (left.runId ?? "").localeCompare(right.runId ?? "")
  || (left.replayCaseId ?? "").localeCompare(right.replayCaseId ?? "")
  || (left.runResultId ?? "").localeCompare(right.runResultId ?? "")
  || left.indexRecordId.localeCompare(right.indexRecordId);

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const optimizerId = (value: string): string => {
  const sanitized = value.replace(/[^A-Za-z0-9._:-]+/g, ".")
    .replace(/^[^A-Za-z0-9]+/, "")
    .replace(/[^A-Za-z0-9]+$/, "");
  return OptimizerIdSchema.parse(sanitized.length > 0 ? sanitized : `id.${stableId(value)}`);
};

const stableId = (value: string): string =>
  createHash("sha256").update(value).digest("hex").slice(0, 16);
