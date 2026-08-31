import { z } from "zod";
import {
  CandidatePatchSchema,
  type CandidatePatch,
  type CandidatePatchOperation,
  type OptimizerRegistryRecord,
} from "./types";

const DEFAULT_MAX_OPERATIONS = 20;
const DEFAULT_MAX_PATCH_BYTES = 16_384;
const SECRET_PATTERN =
  /\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}\b|\b[A-Za-z0-9_.:-]*(?:api[_-]?key|token|secret|password|credential)[A-Za-z0-9_.:-]*\s*[:=]\s*["']?[^"',\s]{8,}["']?/i;

export const CandidateValidationIssueSchema = z.object({
  severity: z.enum(["error", "warning"]),
  code: z.enum([
    "schema_invalid",
    "target_missing",
    "base_hash_missing",
    "base_hash_mismatch",
    "scope_violation",
    "patch_too_large",
    "secret_like_value",
    "required_eval_gate_missing",
  ]),
  message: z.string().min(1),
  path: z.string().optional(),
}).strict();
export type CandidateValidationIssue = z.infer<typeof CandidateValidationIssueSchema>;

export const CandidateValidationResultSchema = z.object({
  candidatePatchId: z.string().min(1).optional(),
  valid: z.boolean(),
  issues: z.array(CandidateValidationIssueSchema),
}).strict();
export type CandidateValidationResult = z.infer<typeof CandidateValidationResultSchema>;

export type CandidateValidationInput = {
  candidate: unknown;
  records: readonly OptimizerRegistryRecord[];
  expectedBaseHashes?: Readonly<Record<string, string>>;
  actualBaseHashes?: Readonly<Record<string, string>>;
  requiredEvalGateIds?: readonly string[];
  maxOperations?: number;
  maxPatchBytes?: number;
};

export const validateCandidatePatch = (input: CandidateValidationInput): CandidateValidationResult => {
  const parsed = CandidatePatchSchema.safeParse(input.candidate);
  if (!parsed.success) {
    return CandidateValidationResultSchema.parse({
      valid: false,
      issues: [
        {
          severity: "error",
          code: "schema_invalid",
          message: parsed.error.issues.map((issue) => issue.message).join("; "),
        },
      ],
    });
  }

  const candidate = parsed.data;
  const issues: CandidateValidationIssue[] = [];
  const target = findTargetRecord(candidate, input.records);
  const maxOperations = boundedInteger(input.maxOperations, DEFAULT_MAX_OPERATIONS, 1, DEFAULT_MAX_OPERATIONS);
  const maxPatchBytes = boundedInteger(input.maxPatchBytes, DEFAULT_MAX_PATCH_BYTES, 1, DEFAULT_MAX_PATCH_BYTES);
  const patchBytes = Buffer.byteLength(JSON.stringify(candidate.operations));

  if (target == null) {
    issues.push({
      severity: "error",
      code: "target_missing",
      message: `target artifact not found: ${candidate.scope.artifactKind} ${candidate.scope.artifactId}`,
    });
  }

  if (candidate.operations.length > maxOperations || patchBytes > maxPatchBytes) {
    issues.push({
      severity: "error",
      code: "patch_too_large",
      message: `candidate patch exceeds caps: operations=${candidate.operations.length}/${maxOperations}, bytes=${patchBytes}/${maxPatchBytes}`,
    });
  }

  for (const operation of candidate.operations) {
    if (!isPathAllowed(operation.path, candidate.scope.allowedJsonPointers)) {
      issues.push({
        severity: "error",
        code: "scope_violation",
        path: operation.path,
        message: `operation path is outside candidate scope: ${operation.path}`,
      });
    }
    if (operationContainsSecret(operation)) {
      issues.push({
        severity: "error",
        code: "secret_like_value",
        path: operation.path,
        message: `operation value contains secret-looking content: ${operation.path}`,
      });
    }
  }

  const expectedHash = input.expectedBaseHashes?.[candidate.scope.artifactId];
  const actualHash = input.actualBaseHashes?.[candidate.scope.artifactId];
  if (expectedHash == null) {
    issues.push({
      severity: "error",
      code: "base_hash_missing",
      message: `missing expected base hash for ${candidate.scope.artifactId}`,
    });
  } else if (actualHash != null && actualHash !== expectedHash) {
    issues.push({
      severity: "error",
      code: "base_hash_mismatch",
      message: `base hash mismatch for ${candidate.scope.artifactId}`,
    });
  }

  const missingGateIds = missingRequiredEvalGates(candidate, target, input.requiredEvalGateIds ?? []);
  for (const gateId of missingGateIds) {
    issues.push({
      severity: "error",
      code: "required_eval_gate_missing",
      message: `required eval gate missing: ${gateId}`,
    });
  }

  return CandidateValidationResultSchema.parse({
    candidatePatchId: candidate.candidatePatchId,
    valid: issues.every((issue) => issue.severity !== "error"),
    issues,
  });
};

const findTargetRecord = (
  candidate: CandidatePatch,
  records: readonly OptimizerRegistryRecord[],
): OptimizerRegistryRecord | undefined =>
  records.find((record) => record.recordKind === candidate.scope.artifactKind && recordPayloadId(record) === candidate.scope.artifactId);

const recordPayloadId = (record: OptimizerRegistryRecord): string => {
  switch (record.recordKind) {
    case "model_profile":
      return record.payload.modelProfileId;
    case "codebase_profile":
      return record.payload.codebaseProfileId;
    case "model_codebase_policy":
      return record.payload.policyId;
    case "canonical_tool_spec":
      return record.payload.canonicalToolId;
    case "rendered_tool_contract":
      return record.payload.renderedToolId;
    case "candidate_patch":
      return record.payload.candidatePatchId;
    case "eval_result":
      return record.payload.evalResultId;
    case "promotion_decision":
      return record.payload.promotionDecisionId;
  }
};

const isPathAllowed = (path: string, allowedJsonPointers: readonly string[]): boolean =>
  allowedJsonPointers.length > 0 &&
  allowedJsonPointers.some((allowedPath) => path === allowedPath || path.startsWith(`${allowedPath}/`));

const operationContainsSecret = (operation: CandidatePatchOperation): boolean => {
  if (operation.op === "remove") {
    return false;
  }
  return SECRET_PATTERN.test(JSON.stringify(operation.value));
};

const missingRequiredEvalGates = (
  candidate: CandidatePatch,
  target: OptimizerRegistryRecord | undefined,
  requiredEvalGateIds: readonly string[],
): string[] => {
  if (requiredEvalGateIds.length === 0) {
    return [];
  }

  const candidateGateIds = candidate.operations.flatMap((operation) => {
    if (operation.op === "remove" || !operation.path.startsWith("/verificationGates")) {
      return [];
    }
    const value = operation.value;
    if (value != null && typeof value === "object" && !Array.isArray(value) && typeof value.gateId === "string") {
      return [value.gateId];
    }
    return [];
  });
  const targetGateIds =
    target?.recordKind === "model_codebase_policy"
      ? target.payload.verificationGates.map((gate) => gate.gateId)
      : [];
  const present = new Set([...candidateGateIds, ...targetGateIds]);
  return requiredEvalGateIds.filter((gateId) => !present.has(gateId));
};

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};
