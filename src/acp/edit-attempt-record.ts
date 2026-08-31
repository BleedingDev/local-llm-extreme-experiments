import { z } from "zod";
import {
  EditErrorCodeSchema,
  PostApplyConsistencyStatusSchema,
  RollbackStatusSchema,
  SelfDetectedRegressionStatusSchema,
  StaleContextStatusSchema,
  VerificationStatusSchema,
} from "../edit-strategy/types";
import {
  JsonValueSchema,
  OptimizerIdSchema,
} from "../optimizer/types";

export const EDIT_ATTEMPT_RECORD_SCHEMA_VERSION = "acp.edit-attempt-record.v1";

export const EditAttemptRecordPhaseNameSchema = z.enum([
  "preview",
  "apply",
  "write",
  "verify",
  "repair",
  "rollback",
]);
export type EditAttemptRecordPhaseName = z.infer<typeof EditAttemptRecordPhaseNameSchema>;

export const EditAttemptRecordPhaseStatusSchema = z.enum([
  "not_started",
  "skipped",
  "passed",
  "warning",
  "failed",
  "inconclusive",
]);
export type EditAttemptRecordPhaseStatus = z.infer<typeof EditAttemptRecordPhaseStatusSchema>;

export const EditAttemptRecordFinalOutcomeSchema = z.enum([
  "success",
  "no_write",
  "stale_context_rejected",
  "protected_path_rejected",
  "syntax_breakage",
  "applied_but_broken",
  "self_detected_regression",
  "verifier_mismatch",
  "preview_failed",
  "apply_failed",
  "write_failed",
  "repair_failed",
  "rolled_back",
  "rollback_failed",
  "failed",
]);
export type EditAttemptRecordFinalOutcome = z.infer<typeof EditAttemptRecordFinalOutcomeSchema>;

export const EditAttemptRecordHashSchema = z.object({
  path: z.string().min(1),
  beforeHash: z.string().min(1).optional(),
  afterHash: z.string().min(1).optional(),
  hashAlgorithm: z.string().min(1).default("sha256"),
}).strict().superRefine((target, ctx) => {
  if (target.beforeHash === undefined && target.afterHash === undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "target hash records require beforeHash or afterHash",
      path: ["beforeHash"],
    });
  }
});
export type EditAttemptRecordHash = z.infer<typeof EditAttemptRecordHashSchema>;

export const EditAttemptRecordPhaseSchema = z.object({
  status: EditAttemptRecordPhaseStatusSchema.default("not_started"),
  errorCode: EditErrorCodeSchema.optional(),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  durationMs: z.number().nonnegative().optional(),
  message: z.string().min(1).optional(),
  skipJustification: z.string().min(1).optional(),
  artifactRefs: z.array(z.string().min(1)).default([]),
  attributes: z.record(z.string(), JsonValueSchema).default({}),
}).strict().superRefine((phase, ctx) => {
  if (phase.status === "failed" && phase.errorCode === undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "failed edit-attempt record phases require an errorCode",
      path: ["errorCode"],
    });
  }
  if (phase.status !== "failed" && phase.errorCode !== undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "errorCode is only valid for failed edit-attempt record phases",
      path: ["errorCode"],
    });
  }
});
export type EditAttemptRecordPhase = z.infer<typeof EditAttemptRecordPhaseSchema>;

const defaultPhase = (status: EditAttemptRecordPhaseStatus = "not_started"): EditAttemptRecordPhase => ({
  status,
  artifactRefs: [],
  attributes: {},
});

const EditAttemptRecordPhasesSchema = z.object({
  preview: EditAttemptRecordPhaseSchema.default(defaultPhase()),
  apply: EditAttemptRecordPhaseSchema.default(defaultPhase()),
  write: EditAttemptRecordPhaseSchema.default(defaultPhase()),
  verify: EditAttemptRecordPhaseSchema.default(defaultPhase()),
  repair: EditAttemptRecordPhaseSchema.default({
    ...defaultPhase("skipped"),
    skipJustification: "repair was not needed",
  }),
  rollback: EditAttemptRecordPhaseSchema.default({
    ...defaultPhase("skipped"),
    skipJustification: "rollback was not needed",
  }),
}).strict().superRefine((phases, ctx) => {
  if (phases.verify.status === "skipped" && phases.verify.skipJustification === undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "skipped verifier phases require a skipJustification",
      path: ["verify", "skipJustification"],
    });
  }
});
export type EditAttemptRecordPhases = z.infer<typeof EditAttemptRecordPhasesSchema>;

const BooleanSignalSchema = z.object({
  detected: z.boolean().default(false),
  errorCode: EditErrorCodeSchema.optional(),
  message: z.string().min(1).optional(),
  evidenceRefs: z.array(z.string().min(1)).default([]),
}).strict();

const VerifierMismatchSignalSchema = z.object({
  detected: z.boolean().default(false),
  expected: z.string().min(1).optional(),
  actual: z.string().min(1).optional(),
  message: z.string().min(1).optional(),
  evidenceRefs: z.array(z.string().min(1)).default([]),
}).strict();

const defaultBooleanSignal = (): z.infer<typeof BooleanSignalSchema> => ({
  detected: false,
  evidenceRefs: [],
});

const defaultVerifierMismatchSignal = (): z.infer<typeof VerifierMismatchSignalSchema> => ({
  detected: false,
  evidenceRefs: [],
});

export const EditAttemptRecordSignalsSchema = z.object({
  staleContext: z.object({
    status: StaleContextStatusSchema.default("not_checked"),
    errorCode: EditErrorCodeSchema.optional(),
    message: z.string().min(1).optional(),
    evidenceRefs: z.array(z.string().min(1)).default([]),
  }).strict().default({
    status: "not_checked",
    evidenceRefs: [],
  }),
  protectedPath: z.object({
    touched: z.boolean().default(false),
    blocked: z.boolean().default(false),
    paths: z.array(z.string().min(1)).default([]),
    errorCode: EditErrorCodeSchema.optional(),
    message: z.string().min(1).optional(),
    evidenceRefs: z.array(z.string().min(1)).default([]),
  }).strict().default({
    touched: false,
    blocked: false,
    paths: [],
    evidenceRefs: [],
  }),
  syntaxBreakage: BooleanSignalSchema.default(defaultBooleanSignal()),
  appliedButBroken: z.object({
    detected: z.boolean().default(false),
    status: PostApplyConsistencyStatusSchema.default("not_checked"),
    message: z.string().min(1).optional(),
    evidenceRefs: z.array(z.string().min(1)).default([]),
  }).strict().default({
    detected: false,
    status: "not_checked",
    evidenceRefs: [],
  }),
  selfDetectedRegression: z.object({
    status: SelfDetectedRegressionStatusSchema.default("not_checked"),
    message: z.string().min(1).optional(),
    evidenceRefs: z.array(z.string().min(1)).default([]),
  }).strict().default({
    status: "not_checked",
    evidenceRefs: [],
  }),
  verifierMismatch: VerifierMismatchSignalSchema.default(defaultVerifierMismatchSignal()),
}).strict();
export type EditAttemptRecordSignals = z.infer<typeof EditAttemptRecordSignalsSchema>;

const defaultSignals = (): EditAttemptRecordSignals => ({
  staleContext: {
    status: "not_checked",
    evidenceRefs: [],
  },
  protectedPath: {
    touched: false,
    blocked: false,
    paths: [],
    evidenceRefs: [],
  },
  syntaxBreakage: defaultBooleanSignal(),
  appliedButBroken: {
    detected: false,
    status: "not_checked",
    evidenceRefs: [],
  },
  selfDetectedRegression: {
    status: "not_checked",
    evidenceRefs: [],
  },
  verifierMismatch: defaultVerifierMismatchSignal(),
});

export const EditAttemptRecordSchema = z.object({
  schemaVersion: z.literal(EDIT_ATTEMPT_RECORD_SCHEMA_VERSION).default(EDIT_ATTEMPT_RECORD_SCHEMA_VERSION),
  editAttemptRecordId: OptimizerIdSchema,
  editAttemptId: OptimizerIdSchema.optional(),
  runId: z.string().min(1).optional(),
  traceId: z.string().min(1).optional(),
  editStrategyId: OptimizerIdSchema,
  renderedEditToolContractId: OptimizerIdSchema.optional(),
  renderedEditContractVersion: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  targetPaths: z.array(z.string().min(1)).default([]),
  targetHashes: z.array(EditAttemptRecordHashSchema).default([]),
  phases: EditAttemptRecordPhasesSchema.default({
    preview: defaultPhase(),
    apply: defaultPhase(),
    write: defaultPhase(),
    verify: defaultPhase(),
    repair: {
      ...defaultPhase("skipped"),
      skipJustification: "repair was not needed",
    },
    rollback: {
      ...defaultPhase("skipped"),
      skipJustification: "rollback was not needed",
    },
  }),
  staleContextStatus: StaleContextStatusSchema.default("not_checked"),
  verificationStatus: VerificationStatusSchema.default("not_run"),
  repairOutcome: z.enum(["not_needed", "not_attempted", "succeeded", "failed", "partial"]).default("not_needed"),
  rollbackOutcome: RollbackStatusSchema.default("not_needed"),
  signals: EditAttemptRecordSignalsSchema.default(defaultSignals()),
  finalOutcome: EditAttemptRecordFinalOutcomeSchema,
  artifactRefs: z.array(z.string().min(1)).default([]),
  createdAt: z.string(),
  completedAt: z.string().optional(),
}).strict().superRefine((record, ctx) => {
  const targetPaths = new Set(record.targetPaths);
  const seenHashes = new Set<string>();
  for (const [index, hash] of record.targetHashes.entries()) {
    if (seenHashes.has(hash.path)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "targetHashes cannot contain duplicate paths",
        path: ["targetHashes", index, "path"],
      });
    }
    seenHashes.add(hash.path);
    if (targetPaths.size > 0 && !targetPaths.has(hash.path)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "targetHashes path must be listed in targetPaths",
        path: ["targetHashes", index, "path"],
      });
    }
  }
  if (record.staleContextStatus !== record.signals.staleContext.status) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "staleContextStatus must mirror signals.staleContext.status",
      path: ["staleContextStatus"],
    });
  }
  if (record.finalOutcome === "no_write" && record.targetHashes.some((hash) => hash.afterHash !== undefined)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "no_write final outcome cannot include after hashes",
      path: ["finalOutcome"],
    });
  }
});
export type EditAttemptRecord = z.infer<typeof EditAttemptRecordSchema>;
export type EditAttemptRecordInput = Omit<z.input<typeof EditAttemptRecordSchema>, "finalOutcome"> & {
  finalOutcome?: EditAttemptRecordFinalOutcome;
};

export const editAttemptRecordTargetHash = (input: {
  path: string;
  beforeHash?: string;
  afterHash?: string;
  hashAlgorithm?: string;
}): EditAttemptRecordHash => EditAttemptRecordHashSchema.parse(input);

export const createEditAttemptRecord = (input: EditAttemptRecordInput): EditAttemptRecord => {
  const signals = EditAttemptRecordSignalsSchema.parse(input.signals ?? {});
  const targetHashes = (input.targetHashes ?? []).map((hash) => EditAttemptRecordHashSchema.parse(hash));
  const phases = EditAttemptRecordPhasesSchema.parse(input.phases ?? {});
  const repairOutcome = input.repairOutcome ?? repairOutcomeForPhase(phases.repair);
  const rollbackOutcome = input.rollbackOutcome ?? rollbackOutcomeForPhase(phases.rollback);
  return EditAttemptRecordSchema.parse({
    ...input,
    targetHashes,
    staleContextStatus: input.staleContextStatus ?? signals.staleContext.status,
    verificationStatus: input.verificationStatus ?? verificationStatusForPhase(phases.verify),
    repairOutcome,
    rollbackOutcome,
    signals,
    finalOutcome: input.finalOutcome ?? classifyEditAttemptRecordOutcome({
      targetHashes,
      phases,
      signals,
      repairOutcome,
      rollbackOutcome,
    }),
  });
};

export const classifyEditAttemptRecordOutcome = (input: {
  targetHashes: readonly EditAttemptRecordHash[];
  phases: EditAttemptRecordPhases;
  signals: EditAttemptRecordSignals;
  repairOutcome?: EditAttemptRecord["repairOutcome"];
  rollbackOutcome?: EditAttemptRecord["rollbackOutcome"];
}): EditAttemptRecordFinalOutcome => {
  const repairOutcome = input.repairOutcome ?? repairOutcomeForPhase(input.phases.repair);
  const rollbackOutcome = input.rollbackOutcome ?? rollbackOutcomeForPhase(input.phases.rollback);
  if (rollbackOutcome === "succeeded") return "rolled_back";
  if (rollbackOutcome === "failed" || rollbackOutcome === "partial") return "rollback_failed";
  if (input.signals.protectedPath.blocked || input.signals.protectedPath.errorCode === "protected_path_violation") {
    return "protected_path_rejected";
  }
  if (input.signals.staleContext.status === "stale" || input.signals.staleContext.status === "conflict") {
    return "stale_context_rejected";
  }
  if (input.signals.syntaxBreakage.detected) return "syntax_breakage";
  if (input.signals.appliedButBroken.detected || input.signals.appliedButBroken.status === "inconsistent") {
    return "applied_but_broken";
  }
  if (
    input.signals.selfDetectedRegression.status === "confirmed" ||
    input.signals.selfDetectedRegression.status === "suspected"
  ) {
    return "self_detected_regression";
  }
  if (input.signals.verifierMismatch.detected) return "verifier_mismatch";
  if (input.phases.preview.status === "failed") return "preview_failed";
  if (input.phases.apply.status === "failed") return "apply_failed";
  if (input.phases.write.status === "failed") return "write_failed";
  if (repairOutcome === "failed" || repairOutcome === "partial") return "repair_failed";
  if (!input.targetHashes.some((hash) => hash.afterHash !== undefined)) return "no_write";
  return "success";
};

const verificationStatusForPhase = (
  phase: z.input<typeof EditAttemptRecordPhaseSchema> | undefined,
): EditAttemptRecord["verificationStatus"] => {
  switch (phase?.status) {
    case "passed":
      return "passed";
    case "failed":
      return "failed";
    case "skipped":
      return "skipped";
    case "inconclusive":
    case "warning":
      return "inconclusive";
    case "not_started":
    case undefined:
      return "not_run";
  }
};

const repairOutcomeForPhase = (
  phase: z.input<typeof EditAttemptRecordPhaseSchema> | undefined,
): EditAttemptRecord["repairOutcome"] => {
  switch (phase?.status) {
    case "passed":
      return "succeeded";
    case "failed":
      return "failed";
    case "warning":
    case "inconclusive":
      return "partial";
    case "not_started":
      return "not_attempted";
    case "skipped":
    case undefined:
      return "not_needed";
  }
};

const rollbackOutcomeForPhase = (
  phase: z.input<typeof EditAttemptRecordPhaseSchema> | undefined,
): EditAttemptRecord["rollbackOutcome"] => {
  switch (phase?.status) {
    case "passed":
      return "succeeded";
    case "failed":
      return "failed";
    case "warning":
    case "inconclusive":
      return "partial";
    case "not_started":
      return "not_attempted";
    case "skipped":
    case undefined:
      return "not_needed";
  }
};
