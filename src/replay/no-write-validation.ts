import { z } from "zod";
import {
  CodingProgressClassSchema,
  codingProgressClassFromTelemetry,
} from "../acp/coding-progress-diagnostics";
import { OptimizerIdSchema, type JsonValue } from "../optimizer/types";
import { RealAcpTaskRunResultSchema, type RealAcpTaskRunResult } from "./real-acp-runner";

const NO_WRITE_VALIDATION_SCHEMA_VERSION = "no-write-validation.v1" as const;

export const NoWriteExpectedMutationSchema = z.enum([
  "edit_existing",
  "create_files",
  "rollback_to_original",
  "detect_without_final_success",
  "no_change",
  "unknown",
]);
export type NoWriteExpectedMutation = z.infer<typeof NoWriteExpectedMutationSchema>;

export const NoWriteExpectedSideEffectSchema = z.enum([
  "none",
  "read",
  "write",
  "terminal",
  "mutation",
  "unknown",
]);
export type NoWriteExpectedSideEffect = z.infer<typeof NoWriteExpectedSideEffectSchema>;

export const NoWriteRouteModeSchema = z.enum([
  "coding",
  "planning",
  "maintenance",
  "read_only",
  "chat",
  "mutating",
  "auto",
  "safe",
  "yolo",
  "cancelled",
  "unknown",
]);
export type NoWriteRouteMode = z.infer<typeof NoWriteRouteModeSchema>;

export const NoWriteVerifierStatusSchema = z.enum([
  "passed",
  "failed",
  "skipped",
  "not_run",
  "error",
  "inconclusive",
  "unknown",
]);
export type NoWriteVerifierStatus = z.infer<typeof NoWriteVerifierStatusSchema>;

export const NoWriteChangedFileSchema = z.union([
  z.string().min(1),
  z.object({
    path: z.string().min(1),
    changeKind: z.string().min(1).optional(),
  }).passthrough(),
]);
export type NoWriteChangedFile = z.infer<typeof NoWriteChangedFileSchema>;

export const NoWriteVerifierSkippedJustificationSchema = z.object({
  present: z.boolean(),
  reason: z.string().min(1).optional(),
  policy: z.enum(["allowed_to_skip", "must_skip", "required", "expected_to_fail_before_repair", "unknown"]).optional(),
}).strict();
export type NoWriteVerifierSkippedJustification = z.infer<typeof NoWriteVerifierSkippedJustificationSchema>;

export const NoWriteValidationInputSchema = z.object({
  recordId: OptimizerIdSchema.optional(),
  taskId: OptimizerIdSchema.optional(),
  routeSelectedMode: NoWriteRouteModeSchema.default("unknown"),
  expectedMutation: NoWriteExpectedMutationSchema.default("unknown"),
  expectedSideEffect: NoWriteExpectedSideEffectSchema.default("unknown"),
  changedFiles: z.array(NoWriteChangedFileSchema).default([]),
  fsWriteCount: z.number().int().nonnegative().default(0),
  terminalCreateCount: z.number().int().nonnegative().default(0),
  terminalExitCount: z.number().int().nonnegative().default(0),
  terminalCommandCount: z.number().int().nonnegative().default(0),
  stopReason: z.string().min(1).optional(),
  editStrategyFamily: z.string().min(1).default("unknown"),
  codingProgressClass: CodingProgressClassSchema.optional(),
  verifierStatus: NoWriteVerifierStatusSchema.default("unknown"),
  verifierSkippedJustification: NoWriteVerifierSkippedJustificationSchema.optional(),
  evidenceRefs: z.array(z.string().min(1)).default([]),
}).strict();
export type NoWriteValidationInput = z.infer<typeof NoWriteValidationInputSchema>;

export const NoWriteValidationClassificationSchema = z.enum([
  "mutation_progress_missing",
  "read_only_legitimate",
  "verifier_skip_justified",
  "write_or_terminal_progress",
  "mutation_expectation_unknown",
  "no_model",
  "model_error",
  "empty_edits",
  "parse_rejected",
  "fallback_empty",
  "executor_failed",
  "permission_rejected",
  "client_write_failed",
  "verifier_missing",
  "verifier_failed",
  "verified_edit",
  "structured_impossibility",
]);
export type NoWriteValidationClassification = z.infer<typeof NoWriteValidationClassificationSchema>;

export const NoWriteValidationSeveritySchema = z.enum(["pass", "warn", "block"]);
export type NoWriteValidationSeverity = z.infer<typeof NoWriteValidationSeveritySchema>;

export const NoWriteValidationObservedSchema = z.object({
  routeSelectedMode: NoWriteRouteModeSchema,
  expectedMutation: NoWriteExpectedMutationSchema,
  expectedSideEffect: NoWriteExpectedSideEffectSchema,
  changedFileCount: z.number().int().nonnegative(),
  fsWriteCount: z.number().int().nonnegative(),
  terminalCreateCount: z.number().int().nonnegative(),
  terminalExitCount: z.number().int().nonnegative(),
  terminalCommandCount: z.number().int().nonnegative(),
  stopReason: z.string().min(1).optional(),
  editStrategyFamily: z.string().min(1),
  codingProgressClass: CodingProgressClassSchema.optional(),
  verifierStatus: NoWriteVerifierStatusSchema,
  verifierSkippedJustificationPresent: z.boolean(),
}).strict();
export type NoWriteValidationObserved = z.infer<typeof NoWriteValidationObservedSchema>;

export const NoWriteValidationResultSchema = z.object({
  schemaVersion: z.literal(NO_WRITE_VALIDATION_SCHEMA_VERSION),
  recordId: OptimizerIdSchema.optional(),
  taskId: OptimizerIdSchema.optional(),
  passed: z.boolean(),
  severity: NoWriteValidationSeveritySchema,
  classification: NoWriteValidationClassificationSchema,
  reasons: z.array(z.string().min(1)),
  evidenceRefs: z.array(z.string().min(1)).default([]),
  missingProgressSignals: z.array(z.enum([
    "changed_files",
    "fs_write",
    "terminal_create",
    "terminal_exit",
  ])).default([]),
  observed: NoWriteValidationObservedSchema,
}).strict();
export type NoWriteValidationResult = z.infer<typeof NoWriteValidationResultSchema>;

export type NoWriteRealAcpTaskRunInput = {
  result: RealAcpTaskRunResult;
  expectedMutation?: NoWriteExpectedMutation;
  expectedSideEffect?: NoWriteExpectedSideEffect;
  evidenceRefs?: readonly string[];
};

export const validateNoWriteProgress = (
  inputValue: NoWriteValidationInput,
): NoWriteValidationResult => {
  const input = NoWriteValidationInputSchema.parse(inputValue);
  const changedFileCount = input.changedFiles.length;
  const verifierSkippedJustificationPresent = verifierSkipIsJustified(input);
  const observed = NoWriteValidationObservedSchema.parse({
    routeSelectedMode: input.routeSelectedMode,
    expectedMutation: input.expectedMutation,
    expectedSideEffect: input.expectedSideEffect,
    changedFileCount,
    fsWriteCount: input.fsWriteCount,
    terminalCreateCount: input.terminalCreateCount,
    terminalExitCount: input.terminalExitCount,
    terminalCommandCount: input.terminalCommandCount,
    ...(input.stopReason === undefined ? {} : { stopReason: input.stopReason }),
    editStrategyFamily: input.editStrategyFamily,
    ...(input.codingProgressClass === undefined ? {} : { codingProgressClass: input.codingProgressClass }),
    verifierStatus: input.verifierStatus,
    verifierSkippedJustificationPresent,
  });
  const progressSignals = progressSignalCount(input, changedFileCount);
  const mutationExpectation = mutationExpectationFor(input);

  if (mutationExpectation === "read_only") {
    return resultFor(input, observed, {
      passed: true,
      severity: "pass",
      classification: "read_only_legitimate",
      reasons: ["No file mutation was expected for this task."],
    });
  }

  if (progressSignals > 0) {
    return resultFor(input, observed, {
      passed: true,
      severity: "pass",
      classification: "write_or_terminal_progress",
      reasons: ["The task recorded write or terminal progress."],
    });
  }

  if (mutationExpectation === "expected" && input.codingProgressClass === "structured_impossibility") {
    return resultFor(input, observed, {
      passed: true,
      severity: "warn",
      classification: "structured_impossibility",
      reasons: ["No write or terminal progress was recorded, but the coding run returned a structured impossibility."],
      missingProgressSignals: missingProgressSignals(input, changedFileCount),
    });
  }

  if (mutationExpectation === "expected" && verifierSkippedJustificationPresent) {
    return resultFor(input, observed, {
      passed: true,
      severity: "warn",
      classification: "verifier_skip_justified",
      reasons: ["No write or terminal progress was recorded, but verifier skip was explicit and justified."],
      missingProgressSignals: missingProgressSignals(input, changedFileCount),
    });
  }

  if (mutationExpectation === "expected" && codingRouteRequiresProgress(input.routeSelectedMode)) {
    const preciseClass = input.codingProgressClass === undefined ||
        input.codingProgressClass === "verified_edit" ||
        input.codingProgressClass === "structured_impossibility"
      ? "mutation_progress_missing"
      : input.codingProgressClass;
    return resultFor(input, observed, {
      passed: false,
      severity: "block",
      classification: preciseClass,
      reasons: [
        preciseClass === "mutation_progress_missing"
          ? "Mutation was expected on a coding route, but no changed files, fsWrite, terminal create, or terminal exit were recorded."
          : `Mutation was expected on a coding route, but coding progress stopped at ${preciseClass} before mutating progress was recorded.`,
      ],
      missingProgressSignals: missingProgressSignals(input, changedFileCount),
    });
  }

  return resultFor(input, observed, {
    passed: true,
    severity: "warn",
    classification: "mutation_expectation_unknown",
    reasons: ["Mutation expectation was unknown, so no-write validation is informational."],
    missingProgressSignals: missingProgressSignals(input, changedFileCount),
  });
};

export const noWriteValidationInputFromRealAcpTaskRunResult = (
  input: NoWriteRealAcpTaskRunInput,
): NoWriteValidationInput => {
  const result = RealAcpTaskRunResultSchema.parse(input.result);
  const headlessAcp = objectAt(result.telemetry, "headlessAcp");
  const counts = objectAt(headlessAcp, "counts");
  const fsWriteCount = numberAt(counts, "fsWrite") ?? result.toolCalls.filter((tool) => tool.sideEffectLevel === "write").length;
  const terminalCreateCount = numberAt(counts, "terminalCreate") ?? result.terminalCommands.length;
  const terminalExitCount = numberAt(counts, "terminalExit") ??
    result.terminalCommands.filter((command) => command.exitCode !== undefined).length;
  const stopReason = stringAt(headlessAcp, "stopReason");
  const transcriptPath = stringAt(headlessAcp, "transcriptPath");
  const skipReason = result.verifier.skipReason ?? result.skipReason;
  const verifierSkippedJustification = skipReason === undefined
    ? undefined
    : {
      present: result.verifier.status === "skipped",
      reason: skipReason,
      policy: result.verifier.policy,
    } satisfies NoWriteVerifierSkippedJustification;

  return NoWriteValidationInputSchema.parse({
    recordId: result.runResultId,
    taskId: result.taskId,
    routeSelectedMode: result.route.selectedMode,
    expectedMutation: input.expectedMutation ?? "unknown",
    expectedSideEffect: input.expectedSideEffect ?? expectedSideEffectForMutation(input.expectedMutation),
    changedFiles: result.changedFiles,
    fsWriteCount,
    terminalCreateCount,
    terminalExitCount,
    terminalCommandCount: result.terminalCommands.length,
    ...(stopReason === undefined ? {} : { stopReason }),
    editStrategyFamily: result.editStrategy.family,
    ...(codingProgressClassFromTelemetry(result.telemetry) === undefined
      ? {}
      : { codingProgressClass: codingProgressClassFromTelemetry(result.telemetry) }),
    verifierStatus: result.verifier.status,
    ...(verifierSkippedJustification === undefined ? {} : { verifierSkippedJustification }),
    evidenceRefs: [
      `real-acp-task-result:${result.runResultId}`,
      ...(transcriptPath === undefined ? [] : [transcriptPath]),
      ...(input.evidenceRefs ?? []),
    ],
  });
};

const expectedSideEffectForMutation = (
  mutation: NoWriteExpectedMutation | undefined,
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
    case undefined:
      return "unknown";
  }
};

const resultFor = (
  input: NoWriteValidationInput,
  observed: NoWriteValidationObserved,
  result: {
    passed: boolean;
    severity: NoWriteValidationSeverity;
    classification: NoWriteValidationClassification;
    reasons: readonly string[];
    missingProgressSignals?: readonly NoWriteValidationResult["missingProgressSignals"][number][];
  },
): NoWriteValidationResult =>
  NoWriteValidationResultSchema.parse({
    schemaVersion: NO_WRITE_VALIDATION_SCHEMA_VERSION,
    ...(input.recordId === undefined ? {} : { recordId: input.recordId }),
    ...(input.taskId === undefined ? {} : { taskId: input.taskId }),
    passed: result.passed,
    severity: result.severity,
    classification: result.classification,
    reasons: result.reasons,
    evidenceRefs: input.evidenceRefs,
    missingProgressSignals: result.missingProgressSignals ?? [],
    observed,
  });

const mutationExpectationFor = (input: NoWriteValidationInput): "expected" | "read_only" | "unknown" => {
  if (input.expectedMutation === "no_change" || input.expectedSideEffect === "none" || input.expectedSideEffect === "read") {
    return "read_only";
  }
  if (
    input.expectedMutation === "edit_existing" ||
    input.expectedMutation === "create_files" ||
    input.expectedMutation === "rollback_to_original" ||
    input.expectedMutation === "detect_without_final_success" ||
    input.expectedSideEffect === "write" ||
    input.expectedSideEffect === "terminal" ||
    input.expectedSideEffect === "mutation"
  ) {
    return "expected";
  }
  return "unknown";
};

const codingRouteRequiresProgress = (routeSelectedMode: NoWriteRouteMode): boolean =>
  routeSelectedMode === "coding" ||
  routeSelectedMode === "mutating" ||
  routeSelectedMode === "auto" ||
  routeSelectedMode === "safe" ||
  routeSelectedMode === "yolo";

const progressSignalCount = (input: NoWriteValidationInput, changedFileCount: number): number =>
  Number(changedFileCount > 0) +
  Number(input.fsWriteCount > 0) +
  Number(input.terminalCreateCount > 0) +
  Number(input.terminalExitCount > 0) +
  Number(input.terminalCommandCount > 0);

const verifierSkipIsJustified = (input: NoWriteValidationInput): boolean =>
  input.verifierStatus === "skipped" &&
  input.verifierSkippedJustification?.present === true;

const missingProgressSignals = (
  input: NoWriteValidationInput,
  changedFileCount: number,
): NoWriteValidationResult["missingProgressSignals"] => [
  ...(changedFileCount === 0 ? ["changed_files" as const] : []),
  ...(input.fsWriteCount === 0 ? ["fs_write" as const] : []),
  ...(input.terminalCreateCount === 0 ? ["terminal_create" as const] : []),
  ...(input.terminalExitCount === 0 ? ["terminal_exit" as const] : []),
];

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
