import { z } from "zod";
import type { ToolCallMetric } from "../types";
import type {
  CodingEditResult,
  CodingPatch,
  PostApplyConsistencyCheck,
} from "./coding-types";
import type { TerminalCommandResult } from "./terminal";

export const CodingProgressClassSchema = z.enum([
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
export type CodingProgressClass = z.infer<typeof CodingProgressClassSchema>;

export const CodingStructuredImpossibilitySchema = z.object({
  reason: z.string().min(1),
  evidenceRefs: z.array(z.string().min(1)).default([]),
}).strict();
export type CodingStructuredImpossibility = z.infer<typeof CodingStructuredImpossibilitySchema>;

export const CodingGenerationDiagnosticSchema = z.object({
  modelAvailable: z.boolean().optional(),
  modelRole: z.enum(["master", "local"]).optional(),
  modelError: z.string().min(1).optional(),
  rawEditCount: z.number().int().nonnegative().optional(),
  rawCommandCount: z.number().int().nonnegative().optional(),
  structuredImpossibility: CodingStructuredImpossibilitySchema.optional(),
}).strict();
export type CodingGenerationDiagnostic = z.infer<typeof CodingGenerationDiagnosticSchema>;

export const CodingProgressDiagnosticSchema = z.object({
  schemaVersion: z.literal("coding-progress-diagnostics.v1"),
  runId: z.string().min(1),
  classifiedAt: z.string().datetime({ offset: true }),
  progressClass: CodingProgressClassSchema,
  terminal: z.enum(["pre_verifier", "final"]),
  mutatingProgress: z.object({
    changedFileCount: z.number().int().nonnegative(),
    successfulWriteCount: z.number().int().nonnegative(),
    failedWriteCount: z.number().int().nonnegative(),
    terminalCommandCount: z.number().int().nonnegative(),
    terminalExitCount: z.number().int().nonnegative(),
  }).strict(),
  generation: z.object({
    modelAvailable: z.boolean().optional(),
    modelRole: z.enum(["master", "local"]).optional(),
    modelError: z.string().min(1).optional(),
    editCount: z.number().int().nonnegative(),
    rawEditCount: z.number().int().nonnegative().optional(),
    parseFailureCount: z.number().int().nonnegative(),
    fallbackAttempted: z.boolean(),
    fallbackEditCount: z.number().int().nonnegative(),
    fallbackParseFailureCount: z.number().int().nonnegative(),
    structuredImpossibility: CodingStructuredImpossibilitySchema.optional(),
  }).strict(),
  verifier: z.object({
    plannedCommandCount: z.number().int().nonnegative(),
    executedCommandCount: z.number().int().nonnegative(),
    failedCommandCount: z.number().int().nonnegative(),
    missing: z.boolean(),
  }).strict(),
  failureSignals: z.array(z.string().min(1)).default([]),
  evidenceRefs: z.array(z.string().min(1)).default([]),
  reason: z.string().min(1),
}).strict();
export type CodingProgressDiagnostic = z.infer<typeof CodingProgressDiagnosticSchema>;

export type ClassifyCodingProgressInput = {
  runId: string;
  patch: CodingPatch;
  fallbackPatch?: CodingPatch;
  editResults?: readonly CodingEditResult[];
  postApplyChecks?: readonly PostApplyConsistencyCheck[];
  plannedCommands?: readonly unknown[];
  commandResults?: readonly TerminalCommandResult[];
  toolMetrics?: readonly ToolCallMetric[];
  terminal?: "pre_verifier" | "final";
  evidenceRefs?: readonly string[];
};

export const classifyCodingProgress = (
  input: ClassifyCodingProgressInput,
): CodingProgressDiagnostic => {
  const editResults = input.editResults ?? [];
  const commandResults = input.commandResults ?? [];
  const plannedCommandCount = input.plannedCommands?.length ?? commandResults.length;
  const failedCommandCount = commandResults.filter((command) => command.exitCode !== 0).length;
  const successfulWrites = editResults.filter((result) => result.ok);
  const failedWrites = editResults.filter((result) => !result.ok);
  const changedFileCount = new Set(successfulWrites.map((result) => normalizePath(result.path))).size;
  const failureSignals = new Set<string>();
  const generation = generationFor(input.patch);
  const fallbackGeneration = input.fallbackPatch === undefined ? undefined : generationFor(input.fallbackPatch);
  const fallbackAttempted = input.fallbackPatch !== undefined;
  const modelError = generation.modelError ?? fallbackGeneration?.modelError;
  const structuredImpossibility =
    generation.structuredImpossibility ?? fallbackGeneration?.structuredImpossibility;

  const progressClass = (() : CodingProgressClass => {
    if (generation.modelAvailable === false) {
      failureSignals.add("generation.no_model");
      return "no_model";
    }
    if (modelError !== undefined) {
      failureSignals.add("generation.model_error");
      return "model_error";
    }
    if (structuredImpossibility !== undefined) {
      failureSignals.add("generation.structured_impossibility");
      return "structured_impossibility";
    }
    if (fallbackAttempted && (input.fallbackPatch?.edits.length ?? 0) === 0) {
      failureSignals.add("fallback.empty");
      return "fallback_empty";
    }
    if (editResults.some(isPermissionRejected)) {
      failureSignals.add("write.permission_rejected");
      return "permission_rejected";
    }
    if (editResults.some(isClientWriteFailed)) {
      failureSignals.add("write.client_write_failed");
      return "client_write_failed";
    }
    if (failedWrites.length > 0) {
      failureSignals.add("edit.executor_failed");
      return "executor_failed";
    }
    if (input.postApplyChecks?.some((check) => check.status === "inconsistent") === true) {
      failureSignals.add("post_apply.inconsistent");
      return "verifier_failed";
    }
    if (failedCommandCount > 0) {
      failureSignals.add("verifier.failed");
      return "verifier_failed";
    }
    if (successfulWrites.length > 0) {
      if (plannedCommandCount === 0 && commandResults.length === 0) {
        failureSignals.add("verifier.missing");
        return "verifier_missing";
      }
      return "verified_edit";
    }
    if (input.patch.parseFailures.length > 0) {
      failureSignals.add("generation.parse_rejected");
      return "parse_rejected";
    }
    if (input.patch.edits.length === 0) {
      failureSignals.add("generation.empty_edits");
      return "empty_edits";
    }
    failureSignals.add("edit.executor_failed");
    return "executor_failed";
  })();

  const reason = reasonFor(progressClass, {
    editCount: input.patch.edits.length,
    parseFailureCount: input.patch.parseFailures.length,
    fallbackAttempted,
    fallbackEditCount: input.fallbackPatch?.edits.length ?? 0,
    successfulWriteCount: successfulWrites.length,
    failedWriteCount: failedWrites.length,
    plannedCommandCount,
    executedCommandCount: commandResults.length,
    failedCommandCount,
    ...(structuredImpossibility?.reason === undefined ? {} : { structuredImpossibilityReason: structuredImpossibility.reason }),
    ...(modelError === undefined ? {} : { modelError }),
  });

  return CodingProgressDiagnosticSchema.parse({
    schemaVersion: "coding-progress-diagnostics.v1",
    runId: input.runId,
    classifiedAt: new Date().toISOString(),
    progressClass,
    terminal: input.terminal ?? (commandResults.length > 0 ? "final" : "pre_verifier"),
    mutatingProgress: {
      changedFileCount,
      successfulWriteCount: successfulWrites.length,
      failedWriteCount: failedWrites.length,
      terminalCommandCount: commandResults.length,
      terminalExitCount: commandResults.filter((command) => command.exitCode !== null).length,
    },
    generation: {
      modelAvailable: generation.modelAvailable,
      modelRole: generation.modelRole,
      modelError,
      editCount: input.patch.edits.length,
      rawEditCount: generation.rawEditCount,
      parseFailureCount: input.patch.parseFailures.length,
      fallbackAttempted,
      fallbackEditCount: input.fallbackPatch?.edits.length ?? 0,
      fallbackParseFailureCount: input.fallbackPatch?.parseFailures.length ?? 0,
      structuredImpossibility,
    },
    verifier: {
      plannedCommandCount,
      executedCommandCount: commandResults.length,
      failedCommandCount,
      missing: plannedCommandCount === 0 && commandResults.length === 0,
    },
    failureSignals: [...failureSignals].sort(),
    evidenceRefs: [...new Set(input.evidenceRefs ?? [])],
    reason,
  });
};

export const codingProgressClassFromTelemetry = (value: unknown): CodingProgressClass | undefined => {
  const candidate = findCodingProgressDiagnostic(value);
  return candidate?.progressClass;
};

const findCodingProgressDiagnostic = (value: unknown): CodingProgressDiagnostic | undefined => {
  const parsed = CodingProgressDiagnosticSchema.safeParse(value);
  if (parsed.success) return parsed.data;
  if (Array.isArray(value)) {
    for (const entry of value) {
      const found = findCodingProgressDiagnostic(entry);
      if (found !== undefined) return found;
    }
    return undefined;
  }
  if (value == null || typeof value !== "object") return undefined;
  const object = value as Record<string, unknown>;
  for (const key of ["codingProgressDiagnostic", "codingProgressDiagnostics", "codingProgress"]) {
    const found = findCodingProgressDiagnostic(object[key]);
    if (found !== undefined) return found;
  }
  for (const child of Object.values(object)) {
    const found = findCodingProgressDiagnostic(child);
    if (found !== undefined) return found;
  }
  return undefined;
};

const generationFor = (patch: CodingPatch): CodingGenerationDiagnostic => {
  const generation = patch.generation === undefined
    ? {}
    : CodingGenerationDiagnosticSchema.parse(patch.generation);
  return {
    ...generation,
    structuredImpossibility: patch.structuredImpossibility ?? generation.structuredImpossibility,
  };
};

const normalizePath = (path: string): string => path.replaceAll("\\", "/");

const isPermissionRejected = (result: CodingEditResult): boolean =>
  result.errorCode === "permission_rejected" ||
  result.reason.toLowerCase().includes("permission rejected");

const isClientWriteFailed = (result: CodingEditResult): boolean => {
  if (result.ok || isPermissionRejected(result)) return false;
  const text = `${result.errorCode ?? ""} ${result.reason}`.toLowerCase();
  return text.includes("acp_write_failed") ||
    text.includes("write_text_file") ||
    text.includes("writetextfile") ||
    text.includes("client") ||
    text.includes("fs/write");
};

const reasonFor = (
  progressClass: CodingProgressClass,
  input: {
    editCount: number;
    parseFailureCount: number;
    fallbackAttempted: boolean;
    fallbackEditCount: number;
    successfulWriteCount: number;
    failedWriteCount: number;
    plannedCommandCount: number;
    executedCommandCount: number;
    failedCommandCount: number;
    structuredImpossibilityReason?: string;
    modelError?: string;
  },
): string => {
  switch (progressClass) {
    case "no_model":
      return "No master or local model was available to generate coding edits.";
    case "model_error":
      return `The model call failed before usable edits were produced: ${input.modelError ?? "unknown model error"}.`;
    case "empty_edits":
      return "The model returned a coding patch with no edit operations.";
    case "parse_rejected":
      return `All generated edit operations were rejected by the selected edit contract parser (${input.parseFailureCount} parse failure(s)).`;
    case "fallback_empty":
      return `Fallback edit generation was attempted but produced no usable edits (${input.fallbackEditCount} fallback edit(s)).`;
    case "executor_failed":
      return `The edit executor or preview path failed before a successful client write (${input.failedWriteCount} failed result(s)).`;
    case "permission_rejected":
      return "The ACP permission flow rejected the requested file write.";
    case "client_write_failed":
      return "The ACP client write call failed after an edit was previewed.";
    case "verifier_missing":
      return "The run wrote files but had no verifier command to establish mutating-task success.";
    case "verifier_failed":
      return `The run reached verification, but ${input.failedCommandCount} verifier command(s) failed.`;
    case "verified_edit":
      return `The run applied ${input.successfulWriteCount} write(s) and verifier evidence did not fail.`;
    case "structured_impossibility":
      return input.structuredImpossibilityReason ?? "The model returned a structured impossibility instead of edits.";
  }
};
