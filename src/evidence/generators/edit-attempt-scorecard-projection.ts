import { createHash } from "node:crypto";
import { z } from "zod";
import {
  EDIT_ATTEMPT_RECORD_SCHEMA_VERSION,
  EditAttemptRecordFinalOutcomeSchema,
  EditAttemptRecordSchema,
  type EditAttemptRecord,
} from "../../acp/edit-attempt-record";
import { countBy, uniqueSorted } from "./artifacts";

export const EDIT_ATTEMPT_SCORECARD_PROJECTION_SCHEMA_VERSION = "evidence.edit-attempt-scorecard-projection.v1";

const EditAttemptScorecardFailureSignalSchema = z.enum([
  "no_write",
  "stale_context",
  "protected_path",
  "syntax_breakage",
  "applied_but_broken",
  "self_detected_regression",
  "verifier_mismatch",
  "preview_failed",
  "apply_failed",
  "write_failed",
  "verifier_failed",
  "verifier_skipped",
  "repair_failed",
  "rolled_back",
  "rollback_failed",
]);
export type EditAttemptScorecardFailureSignal = z.infer<typeof EditAttemptScorecardFailureSignalSchema>;

const EditAttemptScorecardSourceRecordSchema = z.object({
  editAttemptRecordId: z.string().min(1),
  editAttemptId: z.string().min(1).optional(),
  runId: z.string().min(1).optional(),
  traceId: z.string().min(1).optional(),
  evidenceRefs: z.array(z.string().min(1)),
}).strict();
export type EditAttemptScorecardSourceRecord = z.infer<typeof EditAttemptScorecardSourceRecordSchema>;

const EditAttemptScorecardDimensionsSchema = z.object({
  modelProfileId: z.string().min(1),
  codebaseProfileId: z.string().min(1),
  policyId: z.string().min(1),
  editStrategyId: z.string().min(1),
  finalOutcome: EditAttemptRecordFinalOutcomeSchema,
  failureSignals: z.array(EditAttemptScorecardFailureSignalSchema),
}).strict();
export type EditAttemptScorecardDimensions = z.infer<typeof EditAttemptScorecardDimensionsSchema>;

const EditAttemptScorecardGroupSchema = z.object({
  groupId: z.string().min(1),
  dimensions: EditAttemptScorecardDimensionsSchema,
  attemptCount: z.number().int().nonnegative(),
  targetPaths: z.array(z.string().min(1)),
  verificationStatuses: z.record(z.string(), z.number().int().nonnegative()),
  repairOutcomes: z.record(z.string(), z.number().int().nonnegative()),
  rollbackOutcomes: z.record(z.string(), z.number().int().nonnegative()),
  evidenceRefs: z.array(z.string().min(1)),
  sourceRecords: z.array(EditAttemptScorecardSourceRecordSchema),
}).strict();
export type EditAttemptScorecardGroup = z.infer<typeof EditAttemptScorecardGroupSchema>;

export const EditAttemptScorecardProjectionSchema = z.object({
  schemaVersion: z.literal(EDIT_ATTEMPT_SCORECARD_PROJECTION_SCHEMA_VERSION),
  projectionId: z.string().min(1),
  graphId: z.string().min(1).optional(),
  generatedAt: z.string().min(1).optional(),
  sourceRecordSchemaVersion: z.literal(EDIT_ATTEMPT_RECORD_SCHEMA_VERSION),
  sourceBasis: z.literal("edit_attempt_records"),
  sourceRecordCount: z.number().int().nonnegative(),
  totals: z.object({
    byModelProfile: z.record(z.string(), z.number().int().nonnegative()),
    byCodebaseProfile: z.record(z.string(), z.number().int().nonnegative()),
    byPolicy: z.record(z.string(), z.number().int().nonnegative()),
    byEditStrategy: z.record(z.string(), z.number().int().nonnegative()),
    byFinalOutcome: z.record(z.string(), z.number().int().nonnegative()),
    byFailureSignal: z.record(z.string(), z.number().int().nonnegative()),
    byVerificationStatus: z.record(z.string(), z.number().int().nonnegative()),
  }).strict(),
  evidenceRefs: z.array(z.string().min(1)),
  groups: z.array(EditAttemptScorecardGroupSchema),
}).strict();
export type EditAttemptScorecardProjection = z.infer<typeof EditAttemptScorecardProjectionSchema>;

export type ProjectEditAttemptRecordsToScorecardInput = {
  records: readonly unknown[];
  projectionId?: string;
  graphId?: string;
  generatedAt?: string;
};

export const projectEditAttemptRecordsToScorecard = (
  input: ProjectEditAttemptRecordsToScorecardInput,
): EditAttemptScorecardProjection => {
  const records = input.records
    .map((record) => EditAttemptRecordSchema.parse(record))
    .sort(compareRecords);
  const groups = [...groupRecords(records).values()]
    .map(groupFromRecords)
    .sort(compareGroups);
  const failureSignals = records.flatMap(failureSignalsForRecord);
  const evidenceRefs = uniqueSorted(records.flatMap(evidenceRefsForRecord));
  const projectionId = input.projectionId ?? stableId("edit-attempt-scorecard", [
    EDIT_ATTEMPT_SCORECARD_PROJECTION_SCHEMA_VERSION,
    ...records.map(recordProjectionKey),
  ]);

  return EditAttemptScorecardProjectionSchema.parse({
    schemaVersion: EDIT_ATTEMPT_SCORECARD_PROJECTION_SCHEMA_VERSION,
    projectionId,
    ...(input.graphId === undefined ? {} : { graphId: input.graphId }),
    ...(input.generatedAt === undefined ? {} : { generatedAt: input.generatedAt }),
    sourceRecordSchemaVersion: EDIT_ATTEMPT_RECORD_SCHEMA_VERSION,
    sourceBasis: "edit_attempt_records",
    sourceRecordCount: records.length,
    totals: {
      byModelProfile: countBy(records, (record) => record.modelProfileId),
      byCodebaseProfile: countBy(records, (record) => record.codebaseProfileId),
      byPolicy: countBy(records, (record) => record.policyId),
      byEditStrategy: countBy(records, (record) => record.editStrategyId),
      byFinalOutcome: countBy(records, (record) => record.finalOutcome),
      byFailureSignal: countBy(failureSignals, (signal) => signal),
      byVerificationStatus: countBy(records, (record) => record.verificationStatus),
    },
    evidenceRefs,
    groups,
  });
};

export const failureSignalsForEditAttemptRecord = (
  recordInput: unknown,
): EditAttemptScorecardFailureSignal[] => failureSignalsForRecord(EditAttemptRecordSchema.parse(recordInput));

const groupFromRecords = (records: readonly EditAttemptRecord[]): EditAttemptScorecardGroup => {
  const [first] = records;
  if (first === undefined) {
    throw new Error("edit attempt scorecard groups require at least one record");
  }
  const dimensions: EditAttemptScorecardDimensions = {
    modelProfileId: first.modelProfileId,
    codebaseProfileId: first.codebaseProfileId,
    policyId: first.policyId,
    editStrategyId: first.editStrategyId,
    finalOutcome: first.finalOutcome,
    failureSignals: failureSignalsForRecord(first),
  };
  return {
    groupId: stableId("edit-attempt-scorecard.group", groupDimensionKey(dimensions)),
    dimensions,
    attemptCount: records.length,
    targetPaths: uniqueSorted(records.flatMap((record) => record.targetPaths)),
    verificationStatuses: countBy(records, (record) => record.verificationStatus),
    repairOutcomes: countBy(records, (record) => record.repairOutcome),
    rollbackOutcomes: countBy(records, (record) => record.rollbackOutcome),
    evidenceRefs: uniqueSorted(records.flatMap(evidenceRefsForRecord)),
    sourceRecords: records.map(sourceRecordFromRecord),
  };
};

const groupRecords = (records: readonly EditAttemptRecord[]): Map<string, EditAttemptRecord[]> => {
  const groups = new Map<string, EditAttemptRecord[]>();
  for (const record of records) {
    const dimensions: EditAttemptScorecardDimensions = {
      modelProfileId: record.modelProfileId,
      codebaseProfileId: record.codebaseProfileId,
      policyId: record.policyId,
      editStrategyId: record.editStrategyId,
      finalOutcome: record.finalOutcome,
      failureSignals: failureSignalsForRecord(record),
    };
    const key = groupDimensionKey(dimensions).join("\0");
    const current = groups.get(key);
    if (current === undefined) {
      groups.set(key, [record]);
    } else {
      current.push(record);
    }
  }
  return groups;
};

const failureSignalsForRecord = (record: EditAttemptRecord): EditAttemptScorecardFailureSignal[] => {
  const signals = new Set<EditAttemptScorecardFailureSignal>();
  if (record.finalOutcome === "no_write") signals.add("no_write");
  if (record.signals.staleContext.status === "stale" || record.signals.staleContext.status === "conflict") {
    signals.add("stale_context");
  }
  if (record.signals.protectedPath.touched || record.signals.protectedPath.blocked) {
    signals.add("protected_path");
  }
  if (record.signals.syntaxBreakage.detected || record.finalOutcome === "syntax_breakage") {
    signals.add("syntax_breakage");
  }
  if (
    record.signals.appliedButBroken.detected ||
    record.signals.appliedButBroken.status === "inconsistent" ||
    record.finalOutcome === "applied_but_broken"
  ) {
    signals.add("applied_but_broken");
  }
  if (record.signals.selfDetectedRegression.status === "suspected" || record.signals.selfDetectedRegression.status === "confirmed") {
    signals.add("self_detected_regression");
  }
  if (record.signals.verifierMismatch.detected || record.finalOutcome === "verifier_mismatch") {
    signals.add("verifier_mismatch");
  }
  if (record.phases.preview.status === "failed" || record.finalOutcome === "preview_failed") {
    signals.add("preview_failed");
  }
  if (record.phases.apply.status === "failed" || record.finalOutcome === "apply_failed") {
    signals.add("apply_failed");
  }
  if (record.phases.write.status === "failed" || record.finalOutcome === "write_failed") {
    signals.add("write_failed");
  }
  if (record.phases.verify.status === "failed" || record.verificationStatus === "failed" || record.verificationStatus === "error") {
    signals.add("verifier_failed");
  }
  if (record.phases.verify.status === "skipped" || record.verificationStatus === "skipped") {
    signals.add("verifier_skipped");
  }
  if (record.repairOutcome === "failed" || record.repairOutcome === "partial" || record.finalOutcome === "repair_failed") {
    signals.add("repair_failed");
  }
  if (record.rollbackOutcome === "succeeded" || record.finalOutcome === "rolled_back") {
    signals.add("rolled_back");
  }
  if (record.rollbackOutcome === "failed" || record.rollbackOutcome === "partial" || record.finalOutcome === "rollback_failed") {
    signals.add("rollback_failed");
  }
  return [...signals].sort((left, right) => left.localeCompare(right));
};

const evidenceRefsForRecord = (record: EditAttemptRecord): string[] => uniqueSorted([
  ...record.artifactRefs,
  ...Object.values(record.phases).flatMap((phase) => phase.artifactRefs),
  ...record.signals.staleContext.evidenceRefs,
  ...record.signals.protectedPath.evidenceRefs,
  ...record.signals.syntaxBreakage.evidenceRefs,
  ...record.signals.appliedButBroken.evidenceRefs,
  ...record.signals.selfDetectedRegression.evidenceRefs,
  ...record.signals.verifierMismatch.evidenceRefs,
]);

const sourceRecordFromRecord = (record: EditAttemptRecord): EditAttemptScorecardSourceRecord => ({
  editAttemptRecordId: record.editAttemptRecordId,
  ...(record.editAttemptId === undefined ? {} : { editAttemptId: record.editAttemptId }),
  ...(record.runId === undefined ? {} : { runId: record.runId }),
  ...(record.traceId === undefined ? {} : { traceId: record.traceId }),
  evidenceRefs: evidenceRefsForRecord(record),
});

const groupDimensionKey = (dimensions: EditAttemptScorecardDimensions): string[] => [
  dimensions.modelProfileId,
  dimensions.codebaseProfileId,
  dimensions.policyId,
  dimensions.editStrategyId,
  dimensions.finalOutcome,
  dimensions.failureSignals.join("+") || "none",
];

const recordProjectionKey = (record: EditAttemptRecord): string => [
  record.editAttemptRecordId,
  record.editAttemptId ?? "",
  record.runId ?? "",
  record.traceId ?? "",
  record.verificationStatus,
  record.repairOutcome,
  record.rollbackOutcome,
  ...groupDimensionKey({
    modelProfileId: record.modelProfileId,
    codebaseProfileId: record.codebaseProfileId,
    policyId: record.policyId,
    editStrategyId: record.editStrategyId,
    finalOutcome: record.finalOutcome,
    failureSignals: failureSignalsForRecord(record),
  }),
  ...record.targetPaths,
  ...evidenceRefsForRecord(record),
].join("\0");

const stableId = (prefix: string, parts: readonly string[]): string =>
  `${prefix}.${createHash("sha256").update(parts.join("\0")).digest("hex").slice(0, 16)}`;

const compareRecords = (left: EditAttemptRecord, right: EditAttemptRecord): number =>
  recordSortKey(left).localeCompare(recordSortKey(right));

const recordSortKey = (record: EditAttemptRecord): string =>
  [record.editAttemptRecordId, record.editAttemptId ?? "", record.runId ?? "", record.traceId ?? ""].join("\0");

const compareGroups = (left: EditAttemptScorecardGroup, right: EditAttemptScorecardGroup): number =>
  groupDimensionKey(left.dimensions).join("\0").localeCompare(groupDimensionKey(right.dimensions).join("\0"));
