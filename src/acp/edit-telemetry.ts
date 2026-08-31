import { createRunId } from "../artifacts";
import type { EditApplyResult } from "../edit-strategy/apply-layer";
import type {
  EditAttemptContract,
  EditErrorCode,
  EditReadSnapshotRef,
  EditTargetContentHash,
  EditPhaseResult,
  PostApplyConsistencyStatus,
  RollbackStatus,
  SelfDetectedRegressionStatus,
  VerificationStatus,
} from "../edit-strategy/types";
import { RunTelemetry } from "../telemetry";
import type { TerminalCommandResult } from "./terminal";
import type {
  CodingEditOperation,
  CodingEditResult,
  CodingFileSnapshot,
  LiveEditContext,
  PostApplyConsistencyCheck,
} from "./coding-types";
import type { BagAcpSession } from "./session";
import { sha256 } from "./surface";

export const inputContentHashesFor = (
  targetFiles: readonly string[],
  fileSnapshots: readonly CodingFileSnapshot[],
  applyResult: EditApplyResult,
): Record<string, string> => {
  const files = new Set([
    ...targetFiles,
    ...applyResult.changedFiles.map((file) => file.path),
  ]);
  return Object.fromEntries(
    [...files].flatMap((path) => {
      const snapshot = fileSnapshots.find((file) => file.relativePath === path);
      if (snapshot === undefined || snapshot.kind === "create") {
        return [];
      }
      return [[path, `sha256:${snapshot.hash}`]];
    }),
  );
};

export const outputContentHashesFor = (
  applyResult: EditApplyResult,
  _writeResults: readonly CodingEditResult[],
): Record<string, string> =>
  Object.fromEntries(applyResult.changedFiles.flatMap((file) =>
    file.afterContent === undefined ? [] : [[file.path, `sha256:${sha256(file.afterContent)}`] as const],
  ));

export const readSnapshotRefsFor = (
  targetFiles: readonly string[],
  fileSnapshots: readonly CodingFileSnapshot[],
): EditReadSnapshotRef[] => {
  const targets = new Set(targetFiles);
  return fileSnapshots
    .filter((file) => file.kind === "existing" && (targets.size === 0 || targets.has(file.relativePath)))
    .map((file) => ({
      snapshotId: `snapshot.${sha256(file.relativePath).slice(0, 16)}`,
      path: file.relativePath,
      contentHash: `sha256:${file.hash}`,
      wholeFileSeen: true,
      ranges: file.content.length === 0
        ? []
        : [{
            startLine: 0,
            endLine: Math.max(0, file.content.split(/\r?\n/).length - 1),
          }],
    }));
};

export const targetContentHashesFor = (
  targetFiles: readonly string[],
  fileSnapshots: readonly CodingFileSnapshot[],
  applyResult: EditApplyResult,
): EditTargetContentHash[] => {
  const inputHashes = inputContentHashesFor(targetFiles, fileSnapshots, applyResult);
  const outputHashes = outputContentHashesFor(applyResult, []);
  const readSnapshotsByPath = new Map(readSnapshotRefsFor(targetFiles, fileSnapshots).map((snapshot) => [snapshot.path, snapshot]));
  const paths = new Set([
    ...targetFiles,
    ...Object.keys(inputHashes),
    ...Object.keys(outputHashes),
    ...applyResult.changedFiles.map((file) => file.path),
  ]);
  return [...paths].flatMap((path): EditTargetContentHash[] => {
    const beforeHash = inputHashes[path];
    const afterHash = outputHashes[path];
    if (beforeHash === undefined && afterHash === undefined) {
      return [];
    }
    const readSnapshot = readSnapshotsByPath.get(path);
    return [{
      path,
      ...(beforeHash === undefined ? {} : { beforeHash }),
      ...(afterHash === undefined ? {} : { afterHash }),
      ...(readSnapshot === undefined ? {} : { readSnapshotId: readSnapshot.snapshotId }),
      hashAlgorithm: "sha256",
    }];
  });
};

export const editAttemptFromParseFailure = (input: {
  session: BagAcpSession;
  editContext: LiveEditContext;
  parseFailure: string;
}): EditAttemptContract => {
  const now = new Date().toISOString();
  const phaseResults: EditPhaseResult[] = [
    { phase: "generation", status: "passed", artifactRefs: [], attributes: {} },
    {
      phase: "parse",
      status: "failed",
      errorCode: "schema_validation_error",
      message: input.parseFailure,
      artifactRefs: [],
      attributes: {
        selectedStrategyId: input.editContext.decision.selectedStrategyId,
        renderedEditToolContractId: input.editContext.renderedContract.renderedToolId,
      },
    },
    { phase: "validate", status: "not_started", artifactRefs: [], attributes: {} },
    { phase: "preview", status: "not_started", artifactRefs: [], attributes: {} },
    { phase: "apply", status: "not_started", artifactRefs: [], attributes: {} },
    { phase: "permission", status: "not_started", artifactRefs: [], attributes: {} },
    { phase: "write", status: "not_started", artifactRefs: [], attributes: {} },
  ];
  return {
    schemaVersion: "edit-attempt.v1",
    editAttemptId: `edit.${createRunId()}`,
    modelProfileId: input.session.optimizerPin.telemetry.modelProfileId,
    codebaseProfileId: input.session.optimizerPin.telemetry.codebaseProfileId,
    policyId: input.session.optimizerPin.telemetry.policyId,
    editStrategyId: input.editContext.decision.selectedStrategyId,
    editStrategyFamily: input.editContext.decision.selectedStrategyFamily,
    canonicalEditToolSpecId: input.editContext.decision.selectedStrategyId,
    renderedEditToolContractId: input.editContext.renderedContract.renderedToolId,
    taskShape: input.editContext.taskShape as Record<string, number | string | boolean>,
    targetFiles: [],
    readSnapshotRefs: [],
    inputContentHashes: {},
    outputContentHashes: {},
    phaseResults,
    parseErrorCode: "schema_validation_error",
    staleContextStatus: "not_checked",
    permissionStatus: "not_required",
    verificationStatus: "not_run",
    postApplyConsistencyStatus: "not_checked",
    selfDetectedRegressionStatus: "not_checked",
    selfDetectedRegressionEvidenceRefs: [],
    repairAttemptCount: 0,
    rollbackStatus: "not_needed",
    tokenUsage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    changedFileCount: 0,
    changedLineCount: 0,
    protectedPathTouched: false,
    redactionStatus: "raw_local_only",
    artifactRefs: [],
    createdAt: now,
    completedAt: now,
  };
};

export const editAttemptFromAcpWrite = (input: {
  session: BagAcpSession;
  editStartedAt: string;
  edit: CodingEditOperation;
  targetFiles: string[];
  fileSnapshots: readonly CodingFileSnapshot[];
  applyResult: EditApplyResult;
  writeResults: readonly CodingEditResult[];
}): EditAttemptContract => {
  const failedWrite = input.writeResults.find((result) => !result.ok);
  const writeStatus = input.writeResults.length === 0
    ? input.applyResult.status === "skipped"
      ? "skipped"
      : "not_started"
    : failedWrite === undefined
      ? "passed"
      : "failed";
  const writeErrorCode: EditErrorCode | undefined =
    writeStatus === "failed" && failedWrite?.reason.includes("permission rejected")
      ? "permission_rejected"
      : writeStatus === "failed"
        ? "acp_write_failed"
        : undefined;
  const applyErrorCode: EditErrorCode | undefined =
    input.applyResult.status === "failed" ? input.applyResult.errorCode ?? "unknown_error" : undefined;
  const parseErrorCode = parseErrorCodeForApplyResult(input.applyResult);
  const staleContextStatus = staleContextStatusForApplyResult(input.edit, input.applyResult);
  const staleContextErrorCode = staleContextStatus === "stale" || staleContextStatus === "conflict"
    ? applyErrorCode
    : undefined;
  const parsePhaseStatus: EditPhaseResult["status"] = parseErrorCode === undefined ? "passed" : "failed";
  const validatePhaseStatus: EditPhaseResult["status"] = parseErrorCode !== undefined
    ? "not_started"
    : input.applyResult.status === "failed"
      ? "failed"
      : "passed";
  const previewPhaseStatus: EditPhaseResult["status"] = parseErrorCode !== undefined || input.applyResult.status === "failed"
    ? "failed"
    : "passed";
  const stalePhaseStatus: EditPhaseResult["status"] =
    staleContextStatus === "stale" || staleContextStatus === "conflict"
      ? "failed"
      : staleContextStatus === "fresh"
        ? "passed"
        : staleContextStatus === "inconclusive"
          ? "inconclusive"
          : "skipped";
  const applyPhaseStatus: EditPhaseResult["status"] = parseErrorCode !== undefined
    ? "not_started"
    : input.applyResult.status === "failed"
      ? "failed"
      : input.applyResult.status === "skipped"
        ? "skipped"
        : "passed";
  const permissionPhaseStatus: EditPhaseResult["status"] = input.session.yolo
    ? "skipped"
    : writeStatus === "not_started"
      ? "not_started"
      : writeErrorCode === "permission_rejected"
        ? "failed"
        : "passed";
  const inputContentHashes = inputContentHashesFor(input.targetFiles, input.fileSnapshots, input.applyResult);
  const outputContentHashes = outputContentHashesFor(input.applyResult, input.writeResults);
  const readSnapshotRefs = readSnapshotRefsFor(input.targetFiles, input.fileSnapshots);
  const targetContentHashes = targetContentHashesFor(input.targetFiles, input.fileSnapshots, input.applyResult);
  const phaseResults: EditPhaseResult[] = [
    { phase: "generation", status: "passed", artifactRefs: [], attributes: {} },
    { phase: "parse", status: parsePhaseStatus, artifactRefs: [], attributes: {}, ...(parseErrorCode === undefined ? {} : { errorCode: parseErrorCode }) },
    { phase: "validate", status: validatePhaseStatus, artifactRefs: [], attributes: {}, ...(validatePhaseStatus === "failed" && applyErrorCode !== undefined ? { errorCode: applyErrorCode } : {}) },
    { phase: "preview", status: previewPhaseStatus, artifactRefs: [], attributes: {}, ...(previewPhaseStatus === "failed" && applyErrorCode !== undefined ? { errorCode: applyErrorCode } : {}) },
    { phase: "stale_context_check", status: stalePhaseStatus, artifactRefs: [], attributes: { status: staleContextStatus }, ...(stalePhaseStatus === "failed" && staleContextErrorCode !== undefined ? { errorCode: staleContextErrorCode } : {}) },
    { phase: "apply", status: applyPhaseStatus, artifactRefs: [], attributes: {}, ...(applyPhaseStatus === "failed" && applyErrorCode !== undefined ? { errorCode: applyErrorCode } : {}) },
    { phase: "permission", status: permissionPhaseStatus, artifactRefs: [], attributes: {}, ...(writeErrorCode === "permission_rejected" ? { errorCode: writeErrorCode } : {}) },
    { phase: "write", status: writeStatus, artifactRefs: [], attributes: {}, ...(writeErrorCode === undefined ? {} : { errorCode: writeErrorCode }) },
    ...(input.edit.fallbackFromStrategyId === undefined
      ? []
      : [{
          phase: "fallback" as const,
          status: "passed" as const,
          artifactRefs: [],
          attributes: {
            trigger: input.edit.fallbackTrigger ?? "unknown",
            fromStrategyId: input.edit.fallbackFromStrategyId,
            toStrategyId: input.edit.fallbackToStrategyId ?? input.edit.editStrategyId,
          },
        }]),
    ...(input.edit.repairRound === undefined
      ? []
      : [{
          phase: "repair" as const,
          status: "passed" as const,
          artifactRefs: [],
          attributes: {
            repairRound: input.edit.repairRound,
          },
        }]),
  ];
  return {
    schemaVersion: "edit-attempt.v1",
    editAttemptId: `edit.${createRunId()}`,
    modelProfileId: input.session.optimizerPin.telemetry.modelProfileId,
    codebaseProfileId: input.session.optimizerPin.telemetry.codebaseProfileId,
    policyId: input.session.optimizerPin.telemetry.policyId,
    editStrategyId: input.edit.editStrategyId,
    editStrategyFamily: input.edit.editStrategyFamily,
    canonicalEditToolSpecId: input.edit.editStrategyId,
    renderedEditToolContractId: input.edit.renderedEditToolContractId,
    taskShape: {},
    targetFiles: input.targetFiles,
    readSnapshotRefs,
    inputContentHashes,
    outputContentHashes,
    ...(targetContentHashes.length === 0 ? {} : { targetContentHashes }),
    phaseResults,
    ...(parseErrorCode === undefined ? {} : { parseErrorCode }),
    ...(applyErrorCode === undefined || parseErrorCode !== undefined ? {} : { applyErrorCode }),
    staleContextStatus,
    permissionStatus: input.session.yolo
      ? "bypassed_yolo"
      : writeErrorCode === "permission_rejected"
        ? "rejected"
        : writeStatus === "failed"
          ? "failed"
          : input.writeResults.length === 0
            ? "not_required"
            : "approved",
    verificationStatus: "not_run",
    postApplyConsistencyStatus: "not_checked",
    selfDetectedRegressionStatus: "not_checked",
    selfDetectedRegressionEvidenceRefs: [],
    repairAttemptCount: input.edit.repairRound ?? 0,
    rollbackStatus: "not_needed",
    ...(input.edit.fallbackFromStrategyId === undefined ? {} : { fallbackFromStrategyId: input.edit.fallbackFromStrategyId }),
    ...(input.edit.fallbackToStrategyId === undefined ? {} : { fallbackToStrategyId: input.edit.fallbackToStrategyId }),
    tokenUsage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    changedFileCount: input.applyResult.status === "applied" ? input.applyResult.changedFiles.length : 0,
    changedLineCount: 0,
    protectedPathTouched: input.applyResult.protectedPathTouched,
    redactionStatus: "raw_local_only",
    artifactRefs: [],
    createdAt: input.editStartedAt,
    completedAt: new Date().toISOString(),
  };
};

const parseErrorCodeForApplyResult = (applyResult: EditApplyResult): EditErrorCode | undefined => {
  if (applyResult.status !== "failed") {
    return undefined;
  }
  switch (applyResult.errorCode) {
    case "parse_error":
    case "path_or_fence_error":
    case "schema_validation_error":
    case "truncation_induced_error":
      return applyResult.errorCode;
    default:
      return undefined;
  }
};

const staleContextStatusForApplyResult = (
  edit: CodingEditOperation,
  applyResult: EditApplyResult,
): "not_checked" | "fresh" | "stale" | "conflict" | "inconclusive" => {
  if (applyResult.errorCode === "hash_mismatch" || applyResult.errorCode === "anchor_stale") {
    return "stale";
  }
  if (applyResult.errorCode === "anchor_ambiguous") {
    return "conflict";
  }
  return editInputCarriesStaleGuard(edit) ? "fresh" : "not_checked";
};

const editInputCarriesStaleGuard = (edit: CodingEditOperation): boolean => {
  switch (edit.editInput.strategyFamily) {
    case "whole_file":
      return edit.editInput.payload.baseContentHash !== undefined;
    case "exact_replace":
      return edit.editInput.payload.expectedContentHash !== undefined;
    case "hash_range":
      return edit.editInput.payload.operations.some((operation) => operation.expectedContentHash !== undefined);
    case "apply_patch":
    case "unified_diff":
      return false;
  }
};

export const recordFinalEditLifecycleTelemetry = (input: {
  telemetry: RunTelemetry;
  editResults: readonly CodingEditResult[];
  postApplyChecks: readonly PostApplyConsistencyCheck[];
  commandResults: readonly TerminalCommandResult[];
  rollbackResults: readonly CodingEditResult[];
  artifactRefs: readonly string[];
}): EditAttemptContract[] => {
  const attempts = uniqueEditAttemptsFromResults(input.editResults);
  const finalized = attempts.map((attempt) => finalizeEditAttemptLifecycle({
    attempt,
    postApplyChecks: input.postApplyChecks,
    commandResults: input.commandResults,
    rollbackResults: input.rollbackResults,
    artifactRefs: input.artifactRefs,
  }));
  for (const attempt of finalized) {
    input.telemetry.recordEditAttempt(attempt);
  }
  return finalized;
};

export const uniqueEditAttemptsFromResults = (results: readonly CodingEditResult[]): EditAttemptContract[] => {
  const attempts = new Map<string, EditAttemptContract>();
  for (const result of results) {
    if (result.editAttempt !== undefined) {
      attempts.set(result.editAttempt.editAttemptId, result.editAttempt);
    }
  }
  return [...attempts.values()];
};

export const finalizeEditAttemptLifecycle = (input: {
  attempt: EditAttemptContract;
  postApplyChecks: readonly PostApplyConsistencyCheck[];
  commandResults: readonly TerminalCommandResult[];
  rollbackResults: readonly CodingEditResult[];
  artifactRefs: readonly string[];
}): EditAttemptContract => {
  const postApplyChecks = latestPostApplyChecksForAttempt(input.attempt, input.postApplyChecks);
  const postApplyConsistencyStatus = postApplyStatusForAttempt(input.attempt, postApplyChecks);
  const verificationStatus = Object.keys(input.attempt.outputContentHashes).length === 0
    ? "skipped"
    : verificationStatusForCommands(input.commandResults);
  const rollbackStatus = rollbackStatusForAttempt({
    attempt: input.attempt,
    postApplyConsistencyStatus,
    verificationStatus,
    rollbackResults: input.rollbackResults,
  });
  const selfDetectedRegressionStatus = selfDetectedRegressionStatusForLifecycle(
    postApplyConsistencyStatus,
    verificationStatus,
  );
  const selfDetectedRegressionEvidenceRefs = selfDetectedRegressionStatus === "confirmed"
    ? [
        ...postApplyChecks
          .filter((check) => check.status === "inconsistent")
          .map((check) => `post-apply:${check.path}`),
        ...input.artifactRefs,
      ]
    : [];
  const lifecyclePhases: EditPhaseResult[] = [
    postApplyPhase(postApplyConsistencyStatus, postApplyChecks),
    verificationPhase(verificationStatus, input.commandResults),
    selfCheckPhase(selfDetectedRegressionStatus, selfDetectedRegressionEvidenceRefs),
    rollbackPhase(rollbackStatus, input.rollbackResults),
  ];
  return {
    ...input.attempt,
    phaseResults: replaceEditPhaseResults(input.attempt.phaseResults, lifecyclePhases),
    verificationStatus,
    postApplyConsistencyStatus,
    selfDetectedRegressionStatus,
    selfDetectedRegressionEvidenceRefs,
    rollbackStatus,
    artifactRefs: [...new Set([...input.attempt.artifactRefs, ...input.artifactRefs])],
    completedAt: new Date().toISOString(),
  };
};

export const latestPostApplyChecksForAttempt = (
  attempt: EditAttemptContract,
  checks: readonly PostApplyConsistencyCheck[],
): PostApplyConsistencyCheck[] => {
  const latestByPath = new Map<string, PostApplyConsistencyCheck>();
  const targetFiles = new Set(attempt.targetFiles);
  for (const check of checks) {
    if (targetFiles.size === 0 || targetFiles.has(check.path)) {
      latestByPath.set(check.path, check);
    }
  }
  return [...latestByPath.values()];
};

export const postApplyStatusForAttempt = (
  attempt: EditAttemptContract,
  checks: readonly PostApplyConsistencyCheck[],
): PostApplyConsistencyStatus => {
  if (Object.keys(attempt.outputContentHashes).length === 0 || checks.length === 0) {
    return "not_checked";
  }
  if (checks.some((check) => check.status === "inconsistent")) {
    return "inconsistent";
  }
  if (checks.some((check) => check.status === "inconclusive")) {
    return "inconclusive";
  }
  return "consistent";
};

export const verificationStatusForCommands = (commands: readonly TerminalCommandResult[]): VerificationStatus => {
  if (commands.length === 0) {
    return "skipped";
  }
  if (commands.some((command) => command.exitCode === null)) {
    return "error";
  }
  return commands.some((command) => command.exitCode !== 0) ? "failed" : "passed";
};

export const rollbackStatusForAttempt = (input: {
  attempt: EditAttemptContract;
  postApplyConsistencyStatus: PostApplyConsistencyStatus;
  verificationStatus: VerificationStatus;
  rollbackResults: readonly CodingEditResult[];
}): RollbackStatus => {
  const rollbackNeeded =
    Object.keys(input.attempt.outputContentHashes).length > 0 &&
    (
      input.postApplyConsistencyStatus === "inconsistent" ||
      input.verificationStatus === "failed" ||
      input.verificationStatus === "error"
    );
  if (!rollbackNeeded) {
    return "not_needed";
  }
  const targetFiles = new Set(input.attempt.targetFiles);
  const relevantResults = input.rollbackResults.filter((result) => {
    const normalizedPath = result.path.replaceAll("\\", "/");
    return targetFiles.size === 0 ||
      targetFiles.has(normalizedPath) ||
      [...targetFiles].some((target) => normalizedPath.endsWith(`/${target}`));
  });
  if (relevantResults.length === 0) {
    return "not_attempted";
  }
  const successful = relevantResults.filter((result) => result.ok).length;
  if (successful === relevantResults.length) {
    return "succeeded";
  }
  return successful === 0 ? "failed" : "partial";
};

export const selfDetectedRegressionStatusForLifecycle = (
  postApplyConsistencyStatus: PostApplyConsistencyStatus,
  verificationStatus: VerificationStatus,
): SelfDetectedRegressionStatus => {
  if (postApplyConsistencyStatus === "inconsistent") {
    return "confirmed";
  }
  if (verificationStatus === "failed" || verificationStatus === "error") {
    return "suspected";
  }
  if (postApplyConsistencyStatus === "inconclusive" || verificationStatus === "inconclusive") {
    return "inconclusive";
  }
  if (postApplyConsistencyStatus === "not_checked" && verificationStatus === "skipped") {
    return "not_checked";
  }
  return "none";
};

export const postApplyPhase = (
  status: PostApplyConsistencyStatus,
  checks: readonly PostApplyConsistencyCheck[],
): EditPhaseResult => {
  if (status === "not_checked" || status === "pre_existing_failure") {
    return { phase: "post_apply_consistency", status: "skipped", artifactRefs: [], attributes: { status } };
  }
  if (status === "inconsistent") {
    return {
      phase: "post_apply_consistency",
      status: "failed",
      errorCode: "post_apply_behavior_failure",
      artifactRefs: [],
      attributes: { checks: checks.map((check) => ({ path: check.path, status: check.status })) },
    };
  }
  return {
    phase: "post_apply_consistency",
    status: status === "consistent" ? "passed" : "inconclusive",
    artifactRefs: [],
    attributes: { checks: checks.map((check) => ({ path: check.path, status: check.status })) },
  };
};

export const verificationPhase = (
  status: VerificationStatus,
  commands: readonly TerminalCommandResult[],
): EditPhaseResult => {
  if (status === "failed" || status === "error") {
    return {
      phase: "verify",
      status: "failed",
      errorCode: "verifier_error",
      artifactRefs: [],
      attributes: {
        failedCommands: commands
          .filter((command) => command.exitCode !== 0)
          .map((command) => [command.command, ...command.args].join(" ")),
      },
    };
  }
  return {
    phase: "verify",
    status: status === "passed" ? "passed" : status === "skipped" ? "skipped" : "inconclusive",
    artifactRefs: [],
    attributes: {
      commandCount: commands.length,
      ...(status === "skipped" ? { skipReason: "no_verification_commands" } : {}),
    },
  };
};

export const selfCheckPhase = (
  status: SelfDetectedRegressionStatus,
  evidenceRefs: readonly string[],
): EditPhaseResult => {
  if (status === "confirmed") {
    return {
      phase: "self_check",
      status: "failed",
      errorCode: "self_detected_regression",
      artifactRefs: [...evidenceRefs],
      attributes: { status },
    };
  }
  return {
    phase: "self_check",
    status: status === "none" ? "passed" : status === "not_checked" ? "skipped" : status === "suspected" ? "warning" : "inconclusive",
    artifactRefs: [...evidenceRefs],
    attributes: { status },
  };
};

export const rollbackPhase = (status: RollbackStatus, results: readonly CodingEditResult[]): EditPhaseResult => {
  if (status === "failed" || status === "partial") {
    return {
      phase: "rollback",
      status: "failed",
      errorCode: "rollback_failed",
      artifactRefs: [],
      attributes: {
        status,
        resultCount: results.length,
        failedPaths: results.filter((result) => !result.ok).map((result) => result.path),
      },
    };
  }
  return {
    phase: "rollback",
    status: status === "succeeded" ? "passed" : "skipped",
    artifactRefs: [],
    attributes: { status, resultCount: results.length },
  };
};

export const replaceEditPhaseResults = (
  existing: readonly EditPhaseResult[],
  replacements: readonly EditPhaseResult[],
): EditPhaseResult[] => {
  const replacedPhases = new Set(replacements.map((phase) => phase.phase));
  return [
    ...existing.filter((phase) => !replacedPhases.has(phase.phase)),
    ...replacements,
  ];
};
