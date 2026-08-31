import { createHash } from "node:crypto";
import { z } from "zod";

export const PROMOTION_OPERATOR_APPROVAL_SCHEMA_VERSION = "optimizer-operator-approval.v1";
export const PROMOTION_ROLLBACK_CHECKPOINT_PROOF_SCHEMA_VERSION = "optimizer-rollback-checkpoint-proof.v1";
export const PROMOTION_MONITOR_WINDOW_SCHEMA_VERSION = "optimizer-post-promotion-monitor-window.v1";

const ArtifactRefSchema = z.object({
  path: z.string().min(1),
  sha256: z.string().min(1).optional(),
}).strict();

const SourceGraphRefSchema = z.object({
  graphId: z.string().min(1),
  selectionHash: z.string().min(1),
  planSetHash: z.string().min(1).optional(),
  snapshotPath: z.string().min(1),
  snapshotSha256: z.string().min(1).optional(),
}).strict();

const PromotionEvidenceBaseSchema = z.object({
  graphId: z.string().min(1),
  selectionHash: z.string().min(1),
  planSetHash: z.string().min(1).optional(),
  evidenceEpochId: z.string().min(1),
  sourceGraph: SourceGraphRefSchema,
  releaseProofRef: ArtifactRefSchema,
  candidatePatchId: z.string().min(1),
  promotionDecisionId: z.string().min(1).optional(),
  generatedAt: z.string().min(1),
}).strict();

export const OperatorApprovalEvidenceRecordSchema = PromotionEvidenceBaseSchema.extend({
  schemaVersion: z.literal(PROMOTION_OPERATOR_APPROVAL_SCHEMA_VERSION),
  approvalId: z.string().min(1),
  approvalKind: z.enum(["promotion", "rollback_drill", "monitor_window"]).default("promotion"),
  approved: z.boolean(),
  approvedBy: z.string().min(1),
  approvedAt: z.string().min(1),
  expiresAt: z.string().min(1).optional(),
  notes: z.array(z.string().min(1)).default([]),
}).strict();
export type OperatorApprovalEvidenceRecord = z.infer<typeof OperatorApprovalEvidenceRecordSchema>;

export const RollbackCheckpointProofRecordSchema = PromotionEvidenceBaseSchema.extend({
  schemaVersion: z.literal(PROMOTION_ROLLBACK_CHECKPOINT_PROOF_SCHEMA_VERSION),
  checkpointProofId: z.string().min(1),
  checkpointPath: z.string().min(1),
  checkpointSha256: z.string().min(1),
  checkpointCreatedAt: z.string().min(1),
  previousPointerHash: z.string().min(1).optional(),
  restoreMode: z.enum(["dry_run", "request", "perform"]).default("dry_run"),
  restorable: z.boolean(),
  rollbackCommand: z.array(z.string().min(1)).default([]),
}).strict();
export type RollbackCheckpointProofRecord = z.infer<typeof RollbackCheckpointProofRecordSchema>;

const MonitorSignalSchema = z.object({
  signalId: z.string().min(1),
  severity: z.enum(["info", "warning", "failure", "critical"]),
  source: z.string().min(1),
  reason: z.string().min(1),
}).strict();

export const PostPromotionMonitorWindowEvidenceRecordSchema = PromotionEvidenceBaseSchema.extend({
  schemaVersion: z.literal(PROMOTION_MONITOR_WINDOW_SCHEMA_VERSION),
  monitorWindowId: z.string().min(1),
  promotedPolicyId: z.string().min(1),
  startedAt: z.string().min(1),
  completedAt: z.string().min(1),
  requiredWindowMs: z.number().int().positive(),
  observedWindowMs: z.number().int().nonnegative(),
  regressionDetected: z.boolean(),
  rollbackRequested: z.boolean(),
  rolledBack: z.boolean(),
  checkpointPath: z.string().min(1).optional(),
  signals: z.array(MonitorSignalSchema).default([]),
}).strict();
export type PostPromotionMonitorWindowEvidenceRecord = z.infer<typeof PostPromotionMonitorWindowEvidenceRecordSchema>;

export type PromotionEvidenceContext = {
  graphId: string;
  selectionHash?: string | undefined;
  planSetHash?: string | undefined;
  evidenceEpochId?: string | undefined;
  snapshotPath?: string | undefined;
  snapshotSha256?: string | undefined;
  releaseProofPath: string;
  releaseProofSha256?: string | undefined;
  candidatePatchId?: string | undefined;
  promotionDecisionId?: string | undefined;
  generatedAt?: string | undefined;
  now?: string | undefined;
};

export type PromotionEvidenceContractInputs = {
  operatorApproval?: OperatorApprovalEvidenceRecord | undefined;
  rollbackCheckpointProof?: RollbackCheckpointProofRecord | undefined;
  monitorWindow?: PostPromotionMonitorWindowEvidenceRecord | undefined;
};

export type PromotionEvidenceContractStatus = {
  passed: boolean;
  blockingReasons: string[];
  candidatePatchId?: string | undefined;
  promotionDecisionId?: string | undefined;
};

type ContractRecord =
  | OperatorApprovalEvidenceRecord
  | RollbackCheckpointProofRecord
  | PostPromotionMonitorWindowEvidenceRecord;

export const evaluatePromotionEvidenceContracts = (
  input: PromotionEvidenceContractInputs & { context: PromotionEvidenceContext },
): PromotionEvidenceContractStatus => {
  const blockingReasons: string[] = [];
  const records = [
    ["operator approval", input.operatorApproval],
    ["rollback checkpoint proof", input.rollbackCheckpointProof],
    ["post-promotion monitor-window proof", input.monitorWindow],
  ] as const;

  for (const [label, record] of records) {
    if (record === undefined) {
      blockingReasons.push(`missing ${label} evidence`);
      continue;
    }
    blockingReasons.push(...recordBindingBlockers(label, record, input.context));
  }

  if (input.operatorApproval !== undefined) {
    if (!input.operatorApproval.approved) {
      blockingReasons.push("operator approval evidence is not approved");
    }
    if (input.operatorApproval.approvalKind !== "promotion") {
      blockingReasons.push(`operator approval kind ${input.operatorApproval.approvalKind} is not promotion`);
    }
    if (isBefore(input.operatorApproval.expiresAt, input.context.now)) {
      blockingReasons.push("operator approval evidence is expired");
    }
  }

  if (input.rollbackCheckpointProof !== undefined) {
    if (!input.rollbackCheckpointProof.restorable) {
      blockingReasons.push("rollback checkpoint proof is not restorable");
    }
  }

  if (input.monitorWindow !== undefined) {
    if (input.monitorWindow.observedWindowMs < input.monitorWindow.requiredWindowMs) {
      blockingReasons.push(
        `post-promotion monitor window observed ${input.monitorWindow.observedWindowMs}ms but requires ${input.monitorWindow.requiredWindowMs}ms`,
      );
    }
    if (input.monitorWindow.regressionDetected) {
      blockingReasons.push("post-promotion monitor window detected regressions");
    }
    if (input.monitorWindow.signals.some((signal) => signal.severity === "failure" || signal.severity === "critical")) {
      blockingReasons.push("post-promotion monitor window contains failure or critical signals");
    }
  }

  blockingReasons.push(...crossRecordBlockers(records.map(([, record]) => record).filter(isRecord)));

  return {
    passed: blockingReasons.length === 0,
    blockingReasons: uniqueSorted(blockingReasons),
    ...commonCandidateAndDecision(records.map(([, record]) => record).filter(isRecord)),
  };
};

export const stablePromotionEvidenceHash = (value: unknown): string =>
  createHash("sha256").update(JSON.stringify(stableValue(value))).digest("hex");

const recordBindingBlockers = (
  label: string,
  record: ContractRecord,
  context: PromotionEvidenceContext,
): string[] => {
  const blockers: string[] = [];
  const expectedEpoch = context.evidenceEpochId ?? (
    context.selectionHash === undefined ? undefined : `evidence-epoch.${context.graphId}.${context.selectionHash}`
  );

  if (record.graphId !== context.graphId) {
    blockers.push(`${label} targets graph ${record.graphId}, not ${context.graphId}`);
  }
  if (context.selectionHash !== undefined && record.selectionHash !== context.selectionHash) {
    blockers.push(`${label} targets selection ${record.selectionHash}, not ${context.selectionHash}`);
  }
  if (context.planSetHash !== undefined && record.planSetHash !== context.planSetHash) {
    blockers.push(`${label} targets plan set ${record.planSetHash ?? "missing"}, not ${context.planSetHash}`);
  }
  if (expectedEpoch !== undefined && record.evidenceEpochId !== expectedEpoch) {
    blockers.push(`${label} targets evidence epoch ${record.evidenceEpochId}, not ${expectedEpoch}`);
  }
  if (record.sourceGraph.graphId !== record.graphId || record.sourceGraph.selectionHash !== record.selectionHash) {
    blockers.push(`${label} sourceGraph does not match its top-level graph binding`);
  }
  if (context.snapshotPath !== undefined && record.sourceGraph.snapshotPath !== context.snapshotPath) {
    blockers.push(`${label} references snapshot ${record.sourceGraph.snapshotPath}, not ${context.snapshotPath}`);
  }
  if (
    context.snapshotSha256 !== undefined &&
    record.sourceGraph.snapshotSha256 !== undefined &&
    record.sourceGraph.snapshotSha256 !== context.snapshotSha256
  ) {
    blockers.push(`${label} snapshot hash does not match current graph snapshot`);
  }
  if (record.releaseProofRef.path !== context.releaseProofPath) {
    blockers.push(`${label} references release proof ${record.releaseProofRef.path}, not ${context.releaseProofPath}`);
  }
  if (
    context.releaseProofSha256 !== undefined &&
    record.releaseProofRef.sha256 !== undefined &&
    record.releaseProofRef.sha256 !== context.releaseProofSha256
  ) {
    blockers.push(`${label} release proof hash does not match current release proof`);
  }
  if (context.candidatePatchId !== undefined && record.candidatePatchId !== context.candidatePatchId) {
    blockers.push(`${label} targets candidate ${record.candidatePatchId}, not ${context.candidatePatchId}`);
  }
  if (
    context.promotionDecisionId !== undefined &&
    record.promotionDecisionId !== undefined &&
    record.promotionDecisionId !== context.promotionDecisionId
  ) {
    blockers.push(`${label} targets promotion decision ${record.promotionDecisionId}, not ${context.promotionDecisionId}`);
  }
  if (isBefore(record.generatedAt, context.generatedAt)) {
    blockers.push(`${label} was generated before the current graph snapshot`);
  }
  return blockers;
};

const crossRecordBlockers = (records: readonly ContractRecord[]): string[] => {
  const blockers: string[] = [];
  const candidateIds = uniqueSorted(records.map((record) => record.candidatePatchId));
  const promotionDecisionIds = uniqueSorted(records.map((record) => record.promotionDecisionId ?? ""));
  const epochIds = uniqueSorted(records.map((record) => record.evidenceEpochId));
  if (candidateIds.length > 1) {
    blockers.push(`promotion evidence candidate mismatch: ${candidateIds.join(", ")}`);
  }
  if (promotionDecisionIds.length > 1) {
    blockers.push(`promotion evidence decision mismatch: ${promotionDecisionIds.join(", ")}`);
  }
  if (epochIds.length > 1) {
    blockers.push(`promotion evidence epoch mismatch: ${epochIds.join(", ")}`);
  }
  return blockers;
};

const commonCandidateAndDecision = (
  records: readonly ContractRecord[],
): Pick<PromotionEvidenceContractStatus, "candidatePatchId" | "promotionDecisionId"> => {
  const candidateIds = uniqueSorted(records.map((record) => record.candidatePatchId));
  const promotionDecisionIds = uniqueSorted(records.map((record) => record.promotionDecisionId ?? ""));
  return {
    ...(candidateIds.length === 1 ? { candidatePatchId: candidateIds[0] } : {}),
    ...(promotionDecisionIds.length === 1 ? { promotionDecisionId: promotionDecisionIds[0] } : {}),
  };
};

const isRecord = (record: ContractRecord | undefined): record is ContractRecord => record !== undefined;

const isBefore = (left: string | undefined, right: string | undefined): boolean => {
  if (left === undefined || right === undefined) return false;
  const leftMs = Date.parse(left);
  const rightMs = Date.parse(right);
  if (Number.isNaN(leftMs) || Number.isNaN(rightMs)) return false;
  return leftMs < rightMs;
};

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values.filter((value) => value !== ""))].sort((left, right) => left.localeCompare(right));

const stableValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(stableValue);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
};
