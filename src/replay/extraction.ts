import { z } from "zod";
import {
  AcpReplayCaptureSchema,
  AcpReplayModeSchema,
  groupAcpReplayRecords,
  redactionStatusForReplayRecord,
  type AcpReplayCapture,
  type AcpReplayCaptureInput,
  type AcpReplayRecord,
} from "./capture";
import { RedactionStatusSchema, type RedactionStatus } from "../edit-strategy/types";
import {
  EvalAssertionSchema,
  EvalSplitSchema,
  FixtureWorkspaceSchema,
  type EvalAssertion,
  type EvalSplit,
  type FixtureWorkspace,
} from "../eval-harness/types";
import { OptimizerIdSchema } from "../optimizer/types";

const ReplayCaseSchemaVersion = "replay-eval-case.v1" as const;

export const ReplayOracleStrengthSchema = z.enum([
  "none",
  "weak",
  "medium",
  "strong",
  "golden",
]);
export type ReplayOracleStrength = z.infer<typeof ReplayOracleStrengthSchema>;

export const ReplaySourceRefSchema = z.object({
  sourceKind: z.enum(["capture", "record", "trace", "span", "artifact", "fixture"]),
  captureId: OptimizerIdSchema.optional(),
  recordId: OptimizerIdSchema.optional(),
  traceId: z.string().min(1).optional(),
  spanId: z.string().min(1).optional(),
  artifactRef: z.string().min(1).optional(),
  path: z.string().min(1).optional(),
  redactionStatus: RedactionStatusSchema.optional(),
}).strict();
export type ReplaySourceRef = z.infer<typeof ReplaySourceRefSchema>;

export const ReplayRedactionSummarySchema = z.object({
  status: RedactionStatusSchema,
  needsReview: z.boolean(),
  needsReviewRecordIds: z.array(OptimizerIdSchema).default([]),
  recordStatuses: z.array(z.object({
    recordId: OptimizerIdSchema,
    status: RedactionStatusSchema,
  }).strict()).default([]),
}).strict();
export type ReplayRedactionSummary = z.infer<typeof ReplayRedactionSummarySchema>;

export const ReplaySplitAssignmentSchema = z.object({
  split: EvalSplitSchema,
  assignedBy: z.enum(["manual", "capture_hint", "default"]).default("manual"),
  rationale: z.string().min(1).optional(),
}).strict();
export type ReplaySplitAssignment = z.infer<typeof ReplaySplitAssignmentSchema>;

export const ReplayExpectedBehaviorSchema = z.object({
  summary: z.string().min(1),
  assertions: z.array(EvalAssertionSchema).default([]),
  notes: z.array(z.string().min(1)).default([]),
}).strict();
export type ReplayExpectedBehavior = z.infer<typeof ReplayExpectedBehaviorSchema>;

export const ReplayObservedFailureSchema = z.object({
  failureKind: z.enum(["file_read", "edit_attempt", "tool_call", "terminal_command", "user_correction"]),
  recordId: OptimizerIdSchema,
  status: z.string().min(1),
  errorCode: z.string().min(1).optional(),
  phase: z.string().min(1).optional(),
  artifactRefs: z.array(z.string().min(1)).default([]),
}).strict();
export type ReplayObservedFailure = z.infer<typeof ReplayObservedFailureSchema>;

export const ReplayRoutingSummarySchema = z.object({
  promptRecordIds: z.array(OptimizerIdSchema),
  routingRecordIds: z.array(OptimizerIdSchema),
  requestedMode: AcpReplayModeSchema.optional(),
  selectedMode: AcpReplayModeSchema.optional(),
  restoredMode: AcpReplayModeSchema.optional(),
  sideEffectPolicy: z.enum([
    "no_side_effects",
    "read_only",
    "write_allowed",
    "terminal_allowed",
    "unknown",
  ]).optional(),
}).strict();
export type ReplayRoutingSummary = z.infer<typeof ReplayRoutingSummarySchema>;

export const ReplayEvalCaseSkeletonSchema = z.object({
  evalCaseId: OptimizerIdSchema,
  schemaVersion: z.literal(ReplayCaseSchemaVersion).default(ReplayCaseSchemaVersion),
  split: EvalSplitSchema,
  splitAssignment: ReplaySplitAssignmentSchema,
  title: z.string().min(1),
  task: z.string().min(1),
  captureId: OptimizerIdSchema,
  sourceSessionId: z.string().min(1).optional(),
  sourceTraceIds: z.array(z.string().min(1)).default([]),
  sourceRefs: z.array(ReplaySourceRefSchema).min(1),
  redaction: ReplayRedactionSummarySchema,
  oracle: z.object({
    strength: ReplayOracleStrengthSchema,
    expectedBehavior: ReplayExpectedBehaviorSchema,
  }).strict(),
  routing: ReplayRoutingSummarySchema,
  observedFailures: z.array(ReplayObservedFailureSchema).default([]),
  fixtureWorkspace: FixtureWorkspaceSchema.optional(),
  tags: z.array(OptimizerIdSchema).default([]),
  timeoutMs: z.number().int().positive(),
}).strict();
export type ReplayEvalCaseSkeleton = z.infer<typeof ReplayEvalCaseSkeletonSchema>;

export const ReplayExtractionMetadataSchema = z.object({
  evalCaseId: OptimizerIdSchema,
  title: z.string().min(1),
  task: z.string().min(1).optional(),
  split: EvalSplitSchema.optional(),
  splitRationale: z.string().min(1).optional(),
  oracleStrength: ReplayOracleStrengthSchema.default("weak"),
  expectedBehavior: ReplayExpectedBehaviorSchema,
  sourceRefs: z.array(ReplaySourceRefSchema).default([]),
  fixtureWorkspace: FixtureWorkspaceSchema.optional(),
  tags: z.array(OptimizerIdSchema).default([]),
  timeoutMs: z.number().int().positive().default(120000),
}).strict();
export type ReplayExtractionMetadata = z.infer<typeof ReplayExtractionMetadataSchema>;
export type ReplayExtractionMetadataInput = z.input<typeof ReplayExtractionMetadataSchema>;

export type ExtractReplayEvalCaseSkeletonInput = {
  capture: AcpReplayCaptureInput;
  metadata: ReplayExtractionMetadataInput;
};

export const extractReplayEvalCaseSkeleton = (
  input: ExtractReplayEvalCaseSkeletonInput,
): ReplayEvalCaseSkeleton => {
  const capture = AcpReplayCaptureSchema.parse(input.capture);
  const metadata = ReplayExtractionMetadataSchema.parse(input.metadata);
  const groups = groupAcpReplayRecords(capture);
  const primaryUserPrompt = groups.prompts.find((prompt) => prompt.promptRole === "user") ?? groups.prompts[0];
  const splitAssignment = assignSplit(metadata, capture);
  const routingRecord = groups.modeRoutes[0];
  const skeleton = cleanObject({
    evalCaseId: metadata.evalCaseId,
    schemaVersion: ReplayCaseSchemaVersion,
    split: splitAssignment.split,
    splitAssignment,
    title: metadata.title,
    task: metadata.task ?? primaryUserPrompt?.content,
    captureId: capture.captureId,
    sourceSessionId: capture.source.sessionId,
    sourceTraceIds: collectTraceIds(capture),
    sourceRefs: [
      captureSourceRef(capture),
      ...recordSourceRefs(capture.records),
      ...metadata.sourceRefs,
    ],
    redaction: summarizeReplayRedaction(capture),
    oracle: {
      strength: metadata.oracleStrength,
      expectedBehavior: metadata.expectedBehavior,
    },
    routing: cleanObject({
      promptRecordIds: groups.prompts.map((prompt) => prompt.recordId),
      routingRecordIds: groups.modeRoutes.map((route) => route.recordId),
      requestedMode: routingRecord?.requestedMode,
      selectedMode: routingRecord?.selectedMode,
      restoredMode: routingRecord?.restoredMode,
      sideEffectPolicy: routingRecord?.sideEffectPolicy,
    }),
    observedFailures: collectObservedFailures(capture.records),
    fixtureWorkspace: metadata.fixtureWorkspace,
    tags: metadata.tags,
    timeoutMs: metadata.timeoutMs,
  });

  return ReplayEvalCaseSkeletonSchema.parse(skeleton);
};

export const summarizeReplayRedaction = (
  captureInput: AcpReplayCaptureInput,
): ReplayRedactionSummary => {
  const capture = AcpReplayCaptureSchema.parse(captureInput);
  const recordStatuses = capture.records.flatMap((record) => {
    const status = redactionStatusForReplayRecord(record);
    return status == null ? [] : [{ recordId: record.recordId, status }];
  });
  const statuses = [capture.redactionStatus, ...recordStatuses.map((record) => record.status)];
  const needsReviewRecordIds = recordStatuses
    .filter((record) => record.status === "raw_local_only" || record.status === "needs_review")
    .map((record) => record.recordId)
    .sort((left, right) => left.localeCompare(right));
  const status = aggregateRedactionStatus(statuses);
  return ReplayRedactionSummarySchema.parse({
    status,
    needsReview: status === "needs_review",
    needsReviewRecordIds,
    recordStatuses,
  });
};

const assignSplit = (
  metadata: ReplayExtractionMetadata,
  capture: AcpReplayCapture,
): ReplaySplitAssignment => {
  if (metadata.split != null) {
    return ReplaySplitAssignmentSchema.parse(cleanObject({
      split: metadata.split,
      assignedBy: "manual",
      rationale: metadata.splitRationale,
    }));
  }

  if (capture.defaultSplitHint != null) {
    return ReplaySplitAssignmentSchema.parse({
      split: capture.defaultSplitHint,
      assignedBy: "capture_hint",
    });
  }

  return ReplaySplitAssignmentSchema.parse({
    split: "dev",
    assignedBy: "default",
    rationale: "No split metadata was supplied; defaulting replay extraction to visible dev.",
  });
};

const collectTraceIds = (capture: AcpReplayCapture): string[] =>
  uniqueSorted([
    ...capture.source.traceIds,
    ...capture.records.flatMap((record) => record.traceRefs.map((traceRef) => traceRef.traceId)),
    ...capture.records.flatMap((record) =>
      record.recordKind === "edit_attempt" && record.attempt.traceId != null ? [record.attempt.traceId] : []),
  ]);

const captureSourceRef = (capture: AcpReplayCapture): ReplaySourceRef =>
  ReplaySourceRefSchema.parse(cleanObject({
    sourceKind: "capture",
    captureId: capture.captureId,
    path: capture.source.path,
    redactionStatus: capture.redactionStatus,
  }));

const recordSourceRefs = (records: readonly AcpReplayRecord[]): ReplaySourceRef[] =>
  records.flatMap((record) => {
    const redactionStatus = redactionStatusForReplayRecord(record);
    const base = ReplaySourceRefSchema.parse(cleanObject({
      sourceKind: "record",
      recordId: record.recordId,
      redactionStatus,
    }));
    const traceRefs = record.traceRefs.map((traceRef) =>
      ReplaySourceRefSchema.parse(cleanObject({
        sourceKind: traceRef.spanId == null ? "trace" : "span",
        recordId: record.recordId,
        traceId: traceRef.traceId,
        spanId: traceRef.spanId,
        redactionStatus,
      })));
    const artifactRefs = record.artifactRefs.map((artifactRef) =>
      ReplaySourceRefSchema.parse(cleanObject({
        sourceKind: "artifact",
        recordId: record.recordId,
        artifactRef,
        redactionStatus,
      })));
    return [base, ...traceRefs, ...artifactRefs];
  });

const collectObservedFailures = (records: readonly AcpReplayRecord[]): ReplayObservedFailure[] =>
  records.flatMap((record) => {
    switch (record.recordKind) {
      case "prompt":
        return record.promptEvent === "user_correction"
          ? [ReplayObservedFailureSchema.parse(cleanObject({
              failureKind: "user_correction",
              recordId: record.recordId,
              status: "accepted",
              errorCode: "user_correction",
              artifactRefs: record.artifactRefs,
            }))]
          : [];
      case "file_read":
        return record.status === "failed"
          ? [ReplayObservedFailureSchema.parse(cleanObject({
              failureKind: "file_read",
              recordId: record.recordId,
              status: record.status,
              errorCode: record.errorCode,
              artifactRefs: record.artifactRefs,
            }))]
          : [];
      case "edit_attempt":
        return record.attempt.phaseResults
          .filter((phase) => phase.status === "failed")
          .map((phase) => ReplayObservedFailureSchema.parse(cleanObject({
            failureKind: "edit_attempt",
            recordId: record.recordId,
            status: phase.status,
            phase: phase.phase,
            errorCode: phase.errorCode ?? record.attempt.parseErrorCode ?? record.attempt.applyErrorCode,
            artifactRefs: [...record.artifactRefs, ...phase.artifactRefs],
          })));
      case "tool_call":
        return record.status === "succeeded"
          ? []
          : [ReplayObservedFailureSchema.parse(cleanObject({
              failureKind: "tool_call",
              recordId: record.recordId,
              status: record.status,
              errorCode: record.errorCode,
              artifactRefs: record.artifactRefs,
            }))];
      case "terminal_command":
        return record.status === "succeeded"
          ? []
          : [ReplayObservedFailureSchema.parse(cleanObject({
              failureKind: "terminal_command",
              recordId: record.recordId,
              status: record.status,
              errorCode: record.errorCode,
              artifactRefs: [
                ...record.artifactRefs,
                ...optionalArray(record.stdoutArtifactRef),
                ...optionalArray(record.stderrArtifactRef),
              ],
            }))];
      case "mode_route":
      case "artifact_ref":
        return [];
    }
  });

const aggregateRedactionStatus = (statuses: readonly RedactionStatus[]): RedactionStatus => {
  if (statuses.some((status) => status === "raw_local_only" || status === "needs_review")) {
    return "needs_review";
  }
  if (statuses.includes("redacted")) {
    return "redacted";
  }
  if (statuses.includes("hash_only")) {
    return "hash_only";
  }
  if (statuses.includes("omitted")) {
    return "omitted";
  }
  return "needs_review";
};

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const optionalArray = (value: string | undefined): string[] => value == null ? [] : [value];

const cleanObject = <T extends Record<string, unknown>>(value: T): T => {
  const entries = Object.entries(value).filter(([, entryValue]) => entryValue !== undefined);
  return Object.fromEntries(entries) as T;
};

export type ReplayExtractionFixtureInput = {
  fixtureWorkspace?: FixtureWorkspace;
  assertions?: EvalAssertion[];
  split?: EvalSplit;
};
