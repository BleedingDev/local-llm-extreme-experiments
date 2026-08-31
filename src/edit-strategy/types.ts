import { z } from "zod";
import { JsonValueSchema, OptimizerIdSchema } from "../optimizer/types";

export const EditStrategyFamilySchema = z.enum([
  "whole_file",
  "exact_replace",
  "multi_exact_replace",
  "fenced_diff",
  "unified_diff",
  "apply_patch",
  "hash_range",
  "apply_model",
  "architect_editor",
  "ast_structured",
  "range_native",
  "custom",
]);
export type EditStrategyFamily = z.infer<typeof EditStrategyFamilySchema>;

export const EditAttemptPhaseSchema = z.enum([
  "generation",
  "parse",
  "validate",
  "preview",
  "stale_context_check",
  "permission",
  "apply",
  "write",
  "post_apply_consistency",
  "verify",
  "self_check",
  "repair",
  "rollback",
  "fallback",
]);
export type EditAttemptPhase = z.infer<typeof EditAttemptPhaseSchema>;

export const REAL_EDIT_ATTEMPT_REQUIRED_PHASES = [
  "parse",
  "validate",
  "apply",
  "write",
  "post_apply_consistency",
  "verify",
] as const satisfies readonly EditAttemptPhase[];

export const EditPhaseStatusSchema = z.enum([
  "not_started",
  "skipped",
  "passed",
  "warning",
  "failed",
  "inconclusive",
]);
export type EditPhaseStatus = z.infer<typeof EditPhaseStatusSchema>;

export const EditErrorCodeSchema = z.enum([
  "parse_error",
  "path_or_fence_error",
  "schema_validation_error",
  "exact_match_not_found",
  "exact_match_ambiguous",
  "overlapping_edits",
  "hunk_context_mismatch",
  "anchor_not_found",
  "anchor_stale",
  "anchor_ambiguous",
  "hash_mismatch",
  "range_out_of_bounds",
  "partial_apply",
  "scope_violation",
  "protected_path_violation",
  "permission_rejected",
  "acp_write_failed",
  "truncation_induced_error",
  "post_apply_syntax_failure",
  "post_apply_behavior_failure",
  "self_detected_regression",
  "fallback_masked_failure",
  "rollback_failed",
  "anti_pattern_detected",
  "verifier_error",
  "unknown_error",
]);
export type EditErrorCode = z.infer<typeof EditErrorCodeSchema>;

export const StaleContextStatusSchema = z.enum([
  "not_checked",
  "fresh",
  "stale",
  "conflict",
  "inconclusive",
]);
export type StaleContextStatus = z.infer<typeof StaleContextStatusSchema>;

export const PermissionStatusSchema = z.enum([
  "not_required",
  "requested",
  "approved",
  "rejected",
  "bypassed_yolo",
  "failed",
]);
export type PermissionStatus = z.infer<typeof PermissionStatusSchema>;

export const VerificationStatusSchema = z.enum([
  "not_run",
  "passed",
  "failed",
  "error",
  "skipped",
  "inconclusive",
]);
export type VerificationStatus = z.infer<typeof VerificationStatusSchema>;

export const PostApplyConsistencyStatusSchema = z.enum([
  "not_checked",
  "consistent",
  "inconsistent",
  "pre_existing_failure",
  "inconclusive",
]);
export type PostApplyConsistencyStatus = z.infer<typeof PostApplyConsistencyStatusSchema>;

export const SelfDetectedRegressionStatusSchema = z.enum([
  "not_checked",
  "none",
  "suspected",
  "confirmed",
  "inconclusive",
]);
export type SelfDetectedRegressionStatus = z.infer<typeof SelfDetectedRegressionStatusSchema>;

export const RollbackStatusSchema = z.enum([
  "not_needed",
  "not_attempted",
  "succeeded",
  "failed",
  "partial",
]);
export type RollbackStatus = z.infer<typeof RollbackStatusSchema>;

export const RedactionStatusSchema = z.enum([
  "raw_local_only",
  "redacted",
  "hash_only",
  "omitted",
  "needs_review",
]);
export type RedactionStatus = z.infer<typeof RedactionStatusSchema>;

export const EditFallbackTriggerSchema = z.enum([
  "parse_failed",
  "apply_failed",
  "protected_path_violation",
  "post_apply_inconsistent",
  "verification_failed",
  "self_detected_regression",
  "context_budget_exceeded",
  "unknown",
]);
export type EditFallbackTrigger = z.infer<typeof EditFallbackTriggerSchema>;

export const EditReadRangeSchema = z.object({
  startLine: z.number().int().nonnegative(),
  endLine: z.number().int().nonnegative(),
}).strict().refine((range) => range.endLine >= range.startLine, {
  message: "endLine must be greater than or equal to startLine",
});
export type EditReadRange = z.infer<typeof EditReadRangeSchema>;

export const EditReadSnapshotRefSchema = z.object({
  snapshotId: OptimizerIdSchema,
  path: z.string().min(1),
  contentHash: z.string().min(1),
  wholeFileSeen: z.boolean().default(false),
  ranges: z.array(EditReadRangeSchema).default([]),
  artifactRef: z.string().min(1).optional(),
}).strict();
export type EditReadSnapshotRef = z.infer<typeof EditReadSnapshotRefSchema>;

export const EditTokenUsageSchema = z.object({
  promptTokens: z.number().int().nonnegative().default(0),
  completionTokens: z.number().int().nonnegative().default(0),
  totalTokens: z.number().int().nonnegative().default(0),
}).strict().superRefine((usage, ctx) => {
  const computedTotal = usage.promptTokens + usage.completionTokens;
  if (usage.totalTokens !== 0 && usage.totalTokens !== computedTotal) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "totalTokens must equal promptTokens + completionTokens when provided",
      path: ["totalTokens"],
    });
  }
});
export type EditTokenUsage = z.infer<typeof EditTokenUsageSchema>;

export const EditPhaseResultSchema = z.object({
  phase: EditAttemptPhaseSchema,
  status: EditPhaseStatusSchema,
  errorCode: EditErrorCodeSchema.optional(),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  durationMs: z.number().nonnegative().optional(),
  message: z.string().optional(),
  artifactRefs: z.array(z.string().min(1)).default([]),
  attributes: z.record(z.string(), JsonValueSchema).default({}),
}).strict().superRefine((result, ctx) => {
  if (result.status === "failed" && result.errorCode === undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "failed edit phases must include an errorCode",
      path: ["errorCode"],
    });
  }
  if (result.status !== "failed" && result.errorCode !== undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "errorCode is only valid for failed phases",
      path: ["errorCode"],
    });
  }
});
export type EditPhaseResult = z.infer<typeof EditPhaseResultSchema>;

export const EditTargetContentHashSchema = z.object({
  path: z.string().min(1),
  beforeHash: z.string().min(1).optional(),
  afterHash: z.string().min(1).optional(),
  readSnapshotId: OptimizerIdSchema.optional(),
  writeArtifactRef: z.string().min(1).optional(),
  hashAlgorithm: z.string().min(1).default("sha256"),
}).strict().superRefine((target, ctx) => {
  if (target.beforeHash === undefined && target.afterHash === undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "target content hash evidence needs beforeHash or afterHash",
      path: ["beforeHash"],
    });
  }
});
export type EditTargetContentHash = z.infer<typeof EditTargetContentHashSchema>;

export const EditFallbackPathStepSchema = z.object({
  fromStrategyId: OptimizerIdSchema,
  toStrategyId: OptimizerIdSchema,
  trigger: EditFallbackTriggerSchema.default("unknown"),
  status: EditPhaseStatusSchema.default("passed"),
  attemptId: OptimizerIdSchema.optional(),
  artifactRefs: z.array(z.string().min(1)).default([]),
}).strict().refine((step) => step.fromStrategyId !== step.toStrategyId, {
  message: "fallback path step must change strategy",
  path: ["toStrategyId"],
});
export type EditFallbackPathStep = z.infer<typeof EditFallbackPathStepSchema>;

export const EditRepairAttemptRefSchema = z.object({
  repairAttemptId: OptimizerIdSchema.optional(),
  parentAttemptId: OptimizerIdSchema.optional(),
  repairRound: z.number().int().positive(),
  triggerPhase: EditAttemptPhaseSchema.optional(),
  status: EditPhaseStatusSchema,
  artifactRefs: z.array(z.string().min(1)).default([]),
}).strict();
export type EditRepairAttemptRef = z.infer<typeof EditRepairAttemptRefSchema>;

export const EditSelfDetectedRegressionEvidenceSchema = z.object({
  evidenceRef: z.string().min(1),
  evidenceKind: z.enum([
    "model_self_check",
    "post_apply_consistency",
    "verification",
    "diff_review",
    "other",
  ]),
  phase: EditAttemptPhaseSchema.default("self_check"),
  status: SelfDetectedRegressionStatusSchema.default("confirmed"),
  artifactRefs: z.array(z.string().min(1)).default([]),
  summary: z.string().min(1).optional(),
}).strict();
export type EditSelfDetectedRegressionEvidence = z.infer<typeof EditSelfDetectedRegressionEvidenceSchema>;

export const EditAttemptContractSchema = z.object({
  schemaVersion: z.literal("edit-attempt.v1").default("edit-attempt.v1"),
  editAttemptId: OptimizerIdSchema,
  runId: z.string().min(1).optional(),
  traceId: z.string().min(1).optional(),
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  editStrategyId: OptimizerIdSchema,
  editStrategyFamily: EditStrategyFamilySchema,
  canonicalEditToolSpecId: OptimizerIdSchema.optional(),
  renderedEditToolContractId: OptimizerIdSchema.optional(),
  renderedEditContractVersion: OptimizerIdSchema.optional(),
  taskShape: z.record(z.string(), JsonValueSchema).default({}),
  targetFiles: z.array(z.string().min(1)).default([]),
  readSnapshotRefs: z.array(EditReadSnapshotRefSchema).default([]),
  inputContentHashes: z.record(z.string(), z.string().min(1)).default({}),
  outputContentHashes: z.record(z.string(), z.string().min(1)).default({}),
  targetContentHashes: z.array(EditTargetContentHashSchema).optional(),
  phaseResults: z.array(EditPhaseResultSchema).default([]),
  parseErrorCode: EditErrorCodeSchema.optional(),
  applyErrorCode: EditErrorCodeSchema.optional(),
  staleContextStatus: StaleContextStatusSchema.default("not_checked"),
  permissionStatus: PermissionStatusSchema.default("not_required"),
  verificationStatus: VerificationStatusSchema.default("not_run"),
  postApplyConsistencyStatus: PostApplyConsistencyStatusSchema.default("not_checked"),
  selfDetectedRegressionStatus: SelfDetectedRegressionStatusSchema.default("not_checked"),
  selfDetectedRegressionEvidenceRefs: z.array(z.string().min(1)).default([]),
  selfDetectedRegressionEvidence: z.array(EditSelfDetectedRegressionEvidenceSchema).optional(),
  repairAttemptCount: z.number().int().nonnegative().default(0),
  repairAttemptRefs: z.array(EditRepairAttemptRefSchema).optional(),
  rollbackStatus: RollbackStatusSchema.default("not_needed"),
  fallbackFromStrategyId: OptimizerIdSchema.optional(),
  fallbackToStrategyId: OptimizerIdSchema.optional(),
  fallbackPath: z.array(EditFallbackPathStepSchema).optional(),
  tokenUsage: EditTokenUsageSchema.default({
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  }),
  latencyMs: z.number().nonnegative().optional(),
  changedFileCount: z.number().int().nonnegative().default(0),
  changedLineCount: z.number().int().nonnegative().default(0),
  protectedPathTouched: z.boolean().default(false),
  redactionStatus: RedactionStatusSchema.default("raw_local_only"),
  artifactRefs: z.array(z.string().min(1)).default([]),
  createdAt: z.string(),
  completedAt: z.string().optional(),
}).strict().superRefine((attempt, ctx) => {
  if (attempt.fallbackToStrategyId !== undefined && attempt.fallbackFromStrategyId === undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "fallbackToStrategyId requires fallbackFromStrategyId",
      path: ["fallbackFromStrategyId"],
    });
  }
  if (attempt.fallbackPath !== undefined && attempt.fallbackPath.length > 0) {
    const firstStep = attempt.fallbackPath[0];
    const lastStep = attempt.fallbackPath[attempt.fallbackPath.length - 1];
    if (attempt.fallbackFromStrategyId === undefined) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "fallbackPath requires fallbackFromStrategyId",
        path: ["fallbackFromStrategyId"],
      });
    }
    if (attempt.fallbackToStrategyId === undefined) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "fallbackPath requires fallbackToStrategyId",
        path: ["fallbackToStrategyId"],
      });
    }
    if (
      firstStep !== undefined &&
      attempt.fallbackFromStrategyId !== undefined &&
      firstStep.fromStrategyId !== attempt.fallbackFromStrategyId
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "fallbackPath first step must match fallbackFromStrategyId",
        path: ["fallbackPath", 0, "fromStrategyId"],
      });
    }
    if (
      lastStep !== undefined &&
      attempt.fallbackToStrategyId !== undefined &&
      lastStep.toStrategyId !== attempt.fallbackToStrategyId
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "fallbackPath last step must match fallbackToStrategyId",
        path: ["fallbackPath", attempt.fallbackPath.length - 1, "toStrategyId"],
      });
    }
  }
  if (
    attempt.selfDetectedRegressionStatus === "confirmed" &&
    attempt.selfDetectedRegressionEvidenceRefs.length === 0 &&
    (attempt.selfDetectedRegressionEvidence?.length ?? 0) === 0
  ) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "confirmed self-detected regressions require evidence refs",
      path: ["selfDetectedRegressionEvidenceRefs"],
    });
  }
  if (
    attempt.postApplyConsistencyStatus === "inconsistent" &&
    attempt.verificationStatus === "not_run" &&
    attempt.selfDetectedRegressionStatus === "not_checked"
  ) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "inconsistent post-apply status needs verification or self-check evidence",
      path: ["postApplyConsistencyStatus"],
    });
  }
  if (attempt.targetContentHashes !== undefined) {
    const seenPaths = new Set<string>();
    for (const [index, target] of attempt.targetContentHashes.entries()) {
      if (seenPaths.has(target.path)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "targetContentHashes cannot contain duplicate paths",
          path: ["targetContentHashes", index, "path"],
        });
      }
      seenPaths.add(target.path);
      if (attempt.targetFiles.length > 0 && !attempt.targetFiles.includes(target.path)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "targetContentHashes path must be listed in targetFiles",
          path: ["targetContentHashes", index, "path"],
        });
      }
      const inputHash = attempt.inputContentHashes[target.path];
      if (inputHash !== undefined && target.beforeHash !== undefined && inputHash !== target.beforeHash) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "target beforeHash must match inputContentHashes for the same path",
          path: ["targetContentHashes", index, "beforeHash"],
        });
      }
      const outputHash = attempt.outputContentHashes[target.path];
      if (outputHash !== undefined && target.afterHash !== undefined && outputHash !== target.afterHash) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "target afterHash must match outputContentHashes for the same path",
          path: ["targetContentHashes", index, "afterHash"],
        });
      }
      const readSnapshot = attempt.readSnapshotRefs.find((snapshot) => snapshot.path === target.path);
      if (
        readSnapshot !== undefined &&
        target.beforeHash !== undefined &&
        readSnapshot.contentHash !== target.beforeHash
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "target beforeHash must match read snapshot contentHash",
          path: ["targetContentHashes", index, "beforeHash"],
        });
      }
    }
  }
  if (attempt.repairAttemptRefs !== undefined) {
    const maxRepairRound = Math.max(0, ...attempt.repairAttemptRefs.map((repair) => repair.repairRound));
    if (maxRepairRound > attempt.repairAttemptCount) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "repairAttemptCount must cover the highest repairAttemptRefs round",
        path: ["repairAttemptCount"],
      });
    }
  }
});
export type EditAttemptContract = z.infer<typeof EditAttemptContractSchema>;

export type EditAttemptTargetHashRow = {
  path: string;
  beforeHash?: string;
  afterHash?: string;
  source: "target_content_hashes" | "content_hash_maps";
};

export const editAttemptTargetHashRows = (attempt: EditAttemptContract): EditAttemptTargetHashRow[] => {
  if (attempt.targetContentHashes !== undefined && attempt.targetContentHashes.length > 0) {
    return attempt.targetContentHashes.map((target) => {
      const row: EditAttemptTargetHashRow = {
        path: target.path,
        source: "target_content_hashes",
      };
      if (target.beforeHash !== undefined) {
        row.beforeHash = target.beforeHash;
      }
      if (target.afterHash !== undefined) {
        row.afterHash = target.afterHash;
      }
      return row;
    });
  }

  const paths = new Set([
    ...attempt.targetFiles,
    ...Object.keys(attempt.inputContentHashes),
    ...Object.keys(attempt.outputContentHashes),
  ]);
  return [...paths].map((path) => {
    const row: EditAttemptTargetHashRow = {
      path,
      source: "content_hash_maps",
    };
    const beforeHash = attempt.inputContentHashes[path];
    const afterHash = attempt.outputContentHashes[path];
    if (beforeHash !== undefined) {
      row.beforeHash = beforeHash;
    }
    if (afterHash !== undefined) {
      row.afterHash = afterHash;
    }
    return row;
  });
};

export const missingRequiredEditAttemptPhases = (
  phaseResults: readonly EditPhaseResult[],
): EditAttemptPhase[] => {
  const phases = new Set(phaseResults.map((phase) => phase.phase));
  return REAL_EDIT_ATTEMPT_REQUIRED_PHASES.filter((phase) => !phases.has(phase));
};

export const editAttemptCaptureIssues = (attempt: EditAttemptContract): string[] => {
  const issues = [
    attempt.canonicalEditToolSpecId === undefined ? "canonical_edit_tool_spec_id" : undefined,
    attempt.renderedEditToolContractId === undefined ? "rendered_edit_tool_contract_id" : undefined,
    attempt.renderedEditContractVersion === undefined ? "rendered_edit_contract_version" : undefined,
    ...missingRequiredEditAttemptPhases(attempt.phaseResults).map((phase) => `phase.${phase}`),
  ];
  const targetHashRows = editAttemptTargetHashRows(attempt);
  for (const target of attempt.targetFiles) {
    const row = targetHashRows.find((candidate) => candidate.path === target);
    const targetHasExistingSnapshot =
      attempt.inputContentHashes[target] !== undefined ||
      attempt.readSnapshotRefs.some((snapshot) => snapshot.path === target);
    if (targetHasExistingSnapshot && row?.beforeHash === undefined) {
      issues.push(`target_hash.before.${target}`);
    }
    if (attempt.changedFileCount > 0 && row?.afterHash === undefined) {
      issues.push(`target_hash.after.${target}`);
    }
  }
  if (
    (attempt.fallbackFromStrategyId !== undefined || attempt.fallbackToStrategyId !== undefined) &&
    (attempt.fallbackPath?.length ?? 0) === 0
  ) {
    issues.push("fallback_path");
  }
  if (
    attempt.repairAttemptCount > 0 &&
    !attempt.phaseResults.some((phase) => phase.phase === "repair")
  ) {
    issues.push("phase.repair");
  }
  if (
    attempt.rollbackStatus !== "not_needed" &&
    !attempt.phaseResults.some((phase) => phase.phase === "rollback")
  ) {
    issues.push("phase.rollback");
  }
  if (
    attempt.selfDetectedRegressionStatus === "confirmed" &&
    attempt.selfDetectedRegressionEvidenceRefs.length === 0 &&
    (attempt.selfDetectedRegressionEvidence?.length ?? 0) === 0
  ) {
    issues.push("self_detected_regression_evidence");
  }
  return issues.filter((issue): issue is string => issue !== undefined);
};
