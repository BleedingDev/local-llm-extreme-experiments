import { z } from "zod";
import {
  NoWriteValidationInputSchema,
  NoWriteValidationResultSchema,
  validateNoWriteProgress,
  type NoWriteValidationInput,
  type NoWriteValidationResult,
} from "../replay/no-write-validation";

const NO_WRITE_GATE_SCHEMA_VERSION = "no-write-promotion-gate.v1" as const;

export const NoWritePromotionGateStatusSchema = z.enum(["pass", "block", "warn"]);
export type NoWritePromotionGateStatus = z.infer<typeof NoWritePromotionGateStatusSchema>;

export const NoWritePromotionGateDecisionSchema = z.object({
  schemaVersion: z.literal(NO_WRITE_GATE_SCHEMA_VERSION),
  gateId: z.literal("acp-no-write-progress"),
  status: NoWritePromotionGateStatusSchema,
  passed: z.boolean(),
  blocking: z.boolean(),
  reasons: z.array(z.string().min(1)),
  evidenceRefs: z.array(z.string().min(1)).default([]),
  checkedRecordIds: z.array(z.string().min(1)).default([]),
  blockedRecordIds: z.array(z.string().min(1)).default([]),
  warnedRecordIds: z.array(z.string().min(1)).default([]),
  resultCounts: z.object({
    total: z.number().int().nonnegative(),
    passed: z.number().int().nonnegative(),
    blocked: z.number().int().nonnegative(),
    warned: z.number().int().nonnegative(),
  }).strict(),
  validationResults: z.array(NoWriteValidationResultSchema).default([]),
}).strict();
export type NoWritePromotionGateDecision = z.infer<typeof NoWritePromotionGateDecisionSchema>;

export type EvaluateNoWritePromotionGateInput = {
  cases?: readonly NoWriteValidationInput[];
  validationResults?: readonly NoWriteValidationResult[];
  requireEvidence?: boolean;
};

export const evaluateNoWritePromotionGate = (
  input: EvaluateNoWritePromotionGateInput,
): NoWritePromotionGateDecision => {
  const caseResults = (input.cases ?? []).map((candidate) =>
    validateNoWriteProgress(NoWriteValidationInputSchema.parse(candidate)),
  );
  const suppliedResults = (input.validationResults ?? []).map((result) =>
    NoWriteValidationResultSchema.parse(result),
  );
  const validationResults = [...caseResults, ...suppliedResults];

  if (validationResults.length === 0) {
    const status: NoWritePromotionGateStatus = input.requireEvidence === true ? "block" : "warn";
    return NoWritePromotionGateDecisionSchema.parse({
      schemaVersion: NO_WRITE_GATE_SCHEMA_VERSION,
      gateId: "acp-no-write-progress",
      status,
      passed: status !== "block",
      blocking: status === "block",
      reasons: [input.requireEvidence === true
        ? "No no-write ACP validation evidence was supplied."
        : "No no-write ACP validation evidence was supplied; gate is informational."],
      resultCounts: {
        total: 0,
        passed: 0,
        blocked: 0,
        warned: 0,
      },
      validationResults,
    });
  }

  const blocked = validationResults.filter((result) => result.severity === "block" || !result.passed);
  const warned = validationResults.filter((result) => result.severity === "warn" && result.passed);
  const status: NoWritePromotionGateStatus = blocked.length > 0 ? "block" : warned.length > 0 ? "warn" : "pass";
  const checkedRecordIds = validationResults.map(recordKeyFor);
  const blockedRecordIds = blocked.map(recordKeyFor);
  const warnedRecordIds = warned.map(recordKeyFor);

  return NoWritePromotionGateDecisionSchema.parse({
    schemaVersion: NO_WRITE_GATE_SCHEMA_VERSION,
    gateId: "acp-no-write-progress",
    status,
    passed: blocked.length === 0,
    blocking: blocked.length > 0,
    reasons: gateReasons(validationResults.length, blocked, warned),
    evidenceRefs: [...new Set(validationResults.flatMap((result) => result.evidenceRefs))],
    checkedRecordIds,
    blockedRecordIds,
    warnedRecordIds,
    resultCounts: {
      total: validationResults.length,
      passed: validationResults.filter((result) => result.passed).length,
      blocked: blocked.length,
      warned: warned.length,
    },
    validationResults,
  });
};

const gateReasons = (
  total: number,
  blocked: readonly NoWriteValidationResult[],
  warned: readonly NoWriteValidationResult[],
): string[] => {
  if (blocked.length > 0) {
    return [
      `Blocked ${blocked.length}/${total} no-write validation cases with missing mutation progress.`,
      ...blocked.map((result) => `${recordKeyFor(result)}: ${result.reasons.join(" ")}`),
    ];
  }
  if (warned.length > 0) {
    return [
      `Warned on ${warned.length}/${total} no-write validation cases with justified verifier skips or unknown expectations.`,
      ...warned.map((result) => `${recordKeyFor(result)}: ${result.reasons.join(" ")}`),
    ];
  }
  return [`Passed ${total}/${total} no-write validation cases.`];
};

const recordKeyFor = (result: NoWriteValidationResult): string =>
  result.recordId ?? result.taskId ?? `classification:${result.classification}`;
