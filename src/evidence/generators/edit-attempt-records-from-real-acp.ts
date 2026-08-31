import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  createEditAttemptRecord,
  editAttemptRecordTargetHash,
  type EditAttemptRecord,
  type EditAttemptRecordPhase,
} from "../../acp/edit-attempt-record";
import type { JsonValue } from "../../optimizer/types";
import {
  realAcpCodingCorpusTaskPack,
  type RealAcpCorpusTask,
} from "../../replay/real-acp-task-pack";
import {
  RealAcpCorpusRunManifestSchema,
  type RealAcpCorpusRunManifest,
  type RealAcpRepairRecord,
  type RealAcpRollbackRecord,
  type RealAcpTaskRunResult,
  type RealAcpVerifierRecord,
} from "../../replay/real-acp-runner";

const DEFAULT_REPLAY_CORPUS_ROOT = join(".bag", "replay-corpus");

export type BuildEditAttemptRecordsFromRealAcpCorpusInput = {
  cwd: string;
  corpusRoot?: string;
};

export const buildEditAttemptRecordsFromRealAcpCorpus = (
  input: BuildEditAttemptRecordsFromRealAcpCorpusInput,
): EditAttemptRecord[] => {
  const manifests = readRealAcpManifests(input.cwd, input.corpusRoot ?? DEFAULT_REPLAY_CORPUS_ROOT);
  return buildEditAttemptRecordsFromRealAcpManifests(manifests);
};

export const buildEditAttemptRecordsFromRealAcpManifests = (
  manifests: readonly RealAcpCorpusRunManifest[],
): EditAttemptRecord[] => {
  const taskById = new Map(realAcpCodingCorpusTaskPack.tasks.map((task) => [task.taskId, task]));
  return manifests
    .flatMap((manifest) =>
      manifest.taskResults
        .filter((result) => result.split !== "holdout" && result.optimizationAllowed && result.redaction.optimizerSafe)
        .map((result) => editAttemptRecordFromTaskResult({
          manifest,
          result,
          task: taskById.get(result.taskId),
        }))
    )
    .sort((left, right) => left.editAttemptRecordId.localeCompare(right.editAttemptRecordId));
};

const editAttemptRecordFromTaskResult = (input: {
  manifest: RealAcpCorpusRunManifest;
  result: RealAcpTaskRunResult;
  task: RealAcpCorpusTask | undefined;
}): EditAttemptRecord => {
  const { manifest, result, task } = input;
  const targetPaths = uniqueSorted([
    ...(task?.expectedOutcome.expectedChangedPaths ?? []),
    ...result.changedFiles.map((file) => file.path),
  ]);
  const hasWriteProgress = result.changedFiles.length > 0 || fsWriteCount(result) > 0;
  const evidenceRefs = evidenceRefsForTaskRun(manifest, result);
  const transcriptPath = stringAt(objectAt(result.telemetry, "headlessAcp"), "transcriptPath");
  const artifactRefs = uniqueSorted([
    manifest.manifestPath,
    ...(transcriptPath === undefined ? [] : [transcriptPath]),
  ]);

  return createEditAttemptRecord({
    editAttemptRecordId: `edit-attempt-record.${safeId(result.runResultId)}`,
    editAttemptId: `edit-attempt.${safeId(result.runResultId)}`,
    runId: manifest.runId,
    traceId: `trace.${safeId(result.runResultId)}`,
    editStrategyId: result.editStrategy.strategyId,
    renderedEditToolContractId: `rendered-tool.${manifest.metadata.profile.renderedToolVersion}`,
    renderedEditContractVersion: manifest.metadata.profile.renderedToolVersion,
    modelProfileId: manifest.metadata.model.modelProfileId,
    codebaseProfileId: manifest.metadata.codebase.codebaseProfileId,
    policyId: manifest.metadata.profile.policyId,
    targetPaths,
    targetHashes: result.changedFiles.map((file) =>
      editAttemptRecordTargetHash({
        path: file.path,
        ...(file.beforeHash === undefined ? {} : { beforeHash: file.beforeHash }),
        ...(file.afterHash === undefined ? {} : { afterHash: file.afterHash }),
      })
    ),
    phases: {
      preview: previewPhase(result, artifactRefs),
      apply: applyLikePhase("apply", hasWriteProgress, artifactRefs),
      write: applyLikePhase("write", hasWriteProgress, artifactRefs),
      verify: verifyPhase(result.verifier, artifactRefs),
      repair: repairPhase(result.repair, artifactRefs),
      rollback: rollbackPhase(result.rollback, artifactRefs),
    },
    verificationStatus: verifierStatus(result.verifier),
    repairOutcome: repairOutcome(result.repair),
    rollbackOutcome: rollbackOutcome(result.rollback),
    signals: {
      staleContext: {
        status: "not_checked",
        evidenceRefs: [],
      },
      protectedPath: protectedPathSignal(result, task, evidenceRefs),
      syntaxBreakage: {
        detected: false,
        evidenceRefs: [],
      },
      appliedButBroken: {
        detected: result.changedFiles.length > 0 &&
          result.verifier.status === "failed" &&
          task?.primaryLabel === "applied_but_broken",
        status: result.changedFiles.length > 0 && result.verifier.status === "failed"
          ? "inconsistent"
          : "not_checked",
        evidenceRefs: result.verifier.status === "failed" ? evidenceRefs : [],
      },
      selfDetectedRegression: {
        status: "not_checked",
        evidenceRefs: [],
      },
      verifierMismatch: {
        detected: result.changedFiles.length > 0 && result.verifier.status === "failed",
        message: result.verifier.status === "failed" ? result.failureReason ?? "Verifier failed after edit progress." : undefined,
        evidenceRefs: result.verifier.status === "failed" ? evidenceRefs : [],
      },
    },
    artifactRefs,
    createdAt: result.startedAt,
    completedAt: result.completedAt,
  });
};

const previewPhase = (
  result: RealAcpTaskRunResult,
  artifactRefs: readonly string[],
): EditAttemptRecordPhase => {
  if (result.route.selectedMode === "cancelled") {
    return {
      status: "skipped",
      skipJustification: "ACP task was cancelled before a stable edit preview.",
      artifactRefs: [...artifactRefs],
      attributes: {},
    };
  }
  if (result.toolCalls.some((tool) => tool.sideEffectLevel === "read") || result.route.selectedMode === "coding") {
    return {
      status: "passed",
      artifactRefs: [...artifactRefs],
      attributes: {},
    };
  }
  return {
    status: "skipped",
    skipJustification: "No explicit edit preview was observed in the ACP task result.",
    artifactRefs: [...artifactRefs],
    attributes: {},
  };
};

const applyLikePhase = (
  phase: "apply" | "write",
  hasWriteProgress: boolean,
  artifactRefs: readonly string[],
): EditAttemptRecordPhase => {
  if (hasWriteProgress) {
    return {
      status: "passed",
      artifactRefs: [...artifactRefs],
      attributes: {},
    };
  }
  return {
    status: "skipped",
    skipJustification: `No ${phase} progress was observed in the ACP task result.`,
    artifactRefs: [...artifactRefs],
    attributes: {},
  };
};

const verifyPhase = (
  verifier: RealAcpVerifierRecord,
  artifactRefs: readonly string[],
): EditAttemptRecordPhase => {
  switch (verifier.status) {
    case "passed":
      return { status: "passed", artifactRefs: [...artifactRefs], attributes: { policy: verifier.policy } };
    case "failed":
      return {
        status: "failed",
        errorCode: "verifier_error",
        message: "Verifier failed for this ACP task result.",
        artifactRefs: [...artifactRefs],
        attributes: { policy: verifier.policy },
      };
    case "skipped":
      return {
        status: "skipped",
        skipJustification: verifier.skipReason ?? `Verifier policy ${verifier.policy} skipped execution.`,
        artifactRefs: [...artifactRefs],
        attributes: { policy: verifier.policy },
      };
    case "not_run":
      return { status: "not_started", artifactRefs: [...artifactRefs], attributes: { policy: verifier.policy } };
  }
};

const repairPhase = (
  repair: RealAcpRepairRecord,
  artifactRefs: readonly string[],
): EditAttemptRecordPhase => {
  if (!repair.attempted || repair.status === "not_needed") {
    return {
      status: "skipped",
      skipJustification: repair.reason ?? "Repair was not needed or not attempted.",
      artifactRefs: [...artifactRefs],
      attributes: {},
    };
  }
  if (repair.status === "succeeded") return { status: "passed", artifactRefs: [...artifactRefs], attributes: {} };
  if (repair.status === "failed") {
    return {
      status: "failed",
      errorCode: "unknown_error",
      message: repair.reason ?? "Repair failed.",
      artifactRefs: [...artifactRefs],
      attributes: {},
    };
  }
  return {
    status: "skipped",
    skipJustification: repair.reason ?? "Repair was skipped.",
    artifactRefs: [...artifactRefs],
    attributes: {},
  };
};

const rollbackPhase = (
  rollback: RealAcpRollbackRecord,
  artifactRefs: readonly string[],
): EditAttemptRecordPhase => {
  if (!rollback.attempted || rollback.status === "not_needed") {
    return {
      status: "skipped",
      skipJustification: rollback.reason ?? "Rollback was not needed or not attempted.",
      artifactRefs: [...artifactRefs],
      attributes: {},
    };
  }
  if (rollback.status === "succeeded") return { status: "passed", artifactRefs: [...artifactRefs], attributes: {} };
  if (rollback.status === "failed") {
    return {
      status: "failed",
      errorCode: "rollback_failed",
      message: rollback.reason ?? "Rollback failed.",
      artifactRefs: [...artifactRefs],
      attributes: {},
    };
  }
  return {
    status: "skipped",
    skipJustification: rollback.reason ?? "Rollback was skipped.",
    artifactRefs: [...artifactRefs],
    attributes: {},
  };
};

const protectedPathSignal = (
  result: RealAcpTaskRunResult,
  task: RealAcpCorpusTask | undefined,
  evidenceRefs: readonly string[],
): EditAttemptRecord["signals"]["protectedPath"] => {
  const protectedPaths = new Set(task?.workspace.protectedPaths ?? []);
  const touchedPaths = result.changedFiles
    .map((file) => file.path)
    .filter((path) => protectedPaths.has(path));
  return {
    touched: touchedPaths.length > 0,
    blocked: touchedPaths.length > 0,
    paths: touchedPaths,
    ...(touchedPaths.length === 0 ? {} : { errorCode: "protected_path_violation" }),
    evidenceRefs: touchedPaths.length === 0 ? [] : [...evidenceRefs],
  };
};

const verifierStatus = (verifier: RealAcpVerifierRecord): EditAttemptRecord["verificationStatus"] => {
  switch (verifier.status) {
    case "passed":
      return "passed";
    case "failed":
      return "failed";
    case "skipped":
      return "skipped";
    case "not_run":
      return "not_run";
  }
};

const repairOutcome = (repair: RealAcpRepairRecord): EditAttemptRecord["repairOutcome"] => {
  if (!repair.attempted) return repair.status === "not_needed" ? "not_needed" : "not_attempted";
  if (repair.status === "succeeded") return "succeeded";
  if (repair.status === "failed") return "failed";
  if (repair.status === "skipped") return "not_attempted";
  return "not_needed";
};

const rollbackOutcome = (rollback: RealAcpRollbackRecord): EditAttemptRecord["rollbackOutcome"] => {
  if (!rollback.attempted) return rollback.status === "not_needed" ? "not_needed" : "not_attempted";
  if (rollback.status === "succeeded") return "succeeded";
  if (rollback.status === "failed") return "failed";
  if (rollback.status === "skipped") return "not_attempted";
  return "not_needed";
};

const readRealAcpManifests = (cwd: string, corpusRoot: string): RealAcpCorpusRunManifest[] => {
  const runsRoot = join(cwd, corpusRoot, "real-acp-runs");
  try {
    return readdirSync(runsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .flatMap((entry) => {
        const runDir = join(runsRoot, entry.name);
        return readdirSync(runDir, { withFileTypes: true })
          .filter((file) => file.isFile() && file.name.endsWith(".manifest.json"))
          .map((file) => join(runDir, file.name));
      })
      .sort()
      .map((path) => RealAcpCorpusRunManifestSchema.parse(JSON.parse(readFileSync(path, "utf8")) as unknown));
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
};

const fsWriteCount = (result: RealAcpTaskRunResult): number => {
  const counts = objectAt(objectAt(result.telemetry, "headlessAcp"), "counts");
  const fromTelemetry = numberAt(counts, "fsWrite");
  return fromTelemetry ?? result.toolCalls.filter((tool) => tool.sideEffectLevel === "write").length;
};

const evidenceRefsForTaskRun = (
  manifest: RealAcpCorpusRunManifest,
  result: RealAcpTaskRunResult,
): string[] => uniqueSorted([
  `real-acp-run:${manifest.runId}`,
  `real-acp-task-pack:${manifest.taskPackId}`,
  `real-acp-task-result:${result.runResultId}`,
  `real-acp-model-profile:${manifest.metadata.model.modelProfileId}`,
  `real-acp-codebase-profile:${manifest.metadata.codebase.codebaseProfileId}`,
  ...result.toolCalls.map((tool) => `real-acp-tool-call:${tool.toolCallId}`),
  ...result.terminalCommands.map((command) => `real-acp-terminal-command:${command.commandId}`),
]);

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

const safeId = (value: string): string =>
  value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "") || "unknown";

const uniqueSorted = (values: readonly (string | undefined)[]): string[] =>
  [...new Set(values.filter((value): value is string => value !== undefined && value.length > 0))]
    .sort((left, right) => left.localeCompare(right));
