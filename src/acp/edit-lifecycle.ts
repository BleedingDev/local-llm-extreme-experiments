import { applyEdit, type EditApplyResult } from "../edit-strategy/apply-layer";
import type { EditErrorCode, EditStrategyFamily } from "../edit-strategy/types";
import type { RunTelemetry } from "../telemetry";
import type { AcpToolInput } from "./tool-runner";
import type { BagAcpSession } from "./session";
import { artifactLocation, sha256 } from "./surface";
import type {
  AcpWriteClientFileInput,
  AcpWriteClientFileResult,
} from "./workspace-io";
import {
  targetFilesForEditInput,
  type CodingEditOperation,
  type CodingEditResult,
  type CodingFileSnapshot,
  type PostApplyConsistencyCheck,
} from "./coding-types";
import { editAttemptFromAcpWrite } from "./edit-telemetry";

export type AcpEditLifecycleDeps = {
  runAcpTool: <T>(input: AcpToolInput) => Promise<T>;
  readClientFile: (input: {
    sessionId: string;
    telemetry: RunTelemetry;
    path: string;
    signal?: AbortSignal;
  }) => Promise<string>;
  writeClientFileWithPermission: (input: AcpWriteClientFileInput) => Promise<AcpWriteClientFileResult>;
  absoluteSessionPath: (session: BagAcpSession, path: string) => string;
  sessionRelativePath: (session: BagAcpSession, path: string) => string;
};

export const failedEditApplyResult = (
  strategyFamily: EditStrategyFamily,
  errorCode: EditErrorCode,
  errorMessage: string,
): EditApplyResult => ({
  strategyFamily,
  status: "failed",
  changedFiles: [],
  errorCode,
  errorMessage,
  previewDiff: "",
  protectedPathTouched: errorCode === "protected_path_violation",
});

export function previewAndWriteClientEdit(deps: AcpEditLifecycleDeps, input: {
  session: BagAcpSession;
  telemetry: RunTelemetry;
  fileSnapshots: CodingFileSnapshot[];
  edit: CodingEditOperation;
  signal?: AbortSignal;
}): Promise<CodingEditResult[]>;
export function previewAndWriteClientEdit(deps: AcpEditLifecycleDeps, input: {
  session: BagAcpSession;
  telemetry: RunTelemetry;
  path: string;
  oldContent: string;
  newContent: string;
  reason: string;
}): Promise<CodingEditResult>;
export async function previewAndWriteClientEdit(deps: AcpEditLifecycleDeps, input: {
  session: BagAcpSession;
  telemetry: RunTelemetry;
  fileSnapshots?: CodingFileSnapshot[];
  edit?: CodingEditOperation;
  path?: string;
  oldContent?: string;
  newContent?: string;
  reason?: string;
  signal?: AbortSignal;
}): Promise<CodingEditResult[] | CodingEditResult> {
  if (input.edit !== undefined && input.fileSnapshots !== undefined) {
    return previewAndWriteLiveEdit(deps, {
      session: input.session,
      telemetry: input.telemetry,
      fileSnapshots: input.fileSnapshots,
      edit: input.edit,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
  }

  const path = input.path ?? "";
  const oldContent = input.oldContent ?? "";
  const newContent = input.newContent ?? "";
  const reason = input.reason ?? "Model-proposed code edit.";
  const relativePath = deps.sessionRelativePath(input.session, path);
  const editStrategyId = "edit.whole-file.acp-write.v1";
  const edit: CodingEditOperation = {
    reason,
    editInput: {
      strategyFamily: "whole_file",
      payload: {
        path: relativePath,
        content: newContent,
        intent: reason,
      },
    },
    targetFiles: [relativePath],
    editStrategyId,
    editStrategyFamily: "whole_file",
    renderedEditToolContractId: [
      "rendered",
      editStrategyId,
      input.session.optimizerPin.telemetry.modelProfileId,
      input.session.optimizerPin.telemetry.renderedEditContractVersion,
    ].join("."),
  };
  const results = await previewAndWriteLiveEdit(deps, {
    session: input.session,
    telemetry: input.telemetry,
    fileSnapshots: [{ kind: "existing", path, relativePath, content: oldContent, hash: sha256(oldContent) }],
    edit,
  });
  return results[0] ?? {
    path,
    ok: false,
    reason: "edit preview produced no result",
    editStrategyId,
    editStatus: "failed",
    errorCode: "unknown_error",
  };
}

export const updateFileSnapshotsFromEditResult = (
  deps: Pick<AcpEditLifecycleDeps, "absoluteSessionPath" | "sessionRelativePath">,
  session: BagAcpSession,
  fileSnapshots: CodingFileSnapshot[],
  result: CodingEditResult,
): void => {
  if (!result.ok || result.newContent === undefined) {
    return;
  }
  const path = deps.absoluteSessionPath(session, result.path);
  const relativePath = deps.sessionRelativePath(session, path);
  const index = fileSnapshots.findIndex((file) => file.relativePath === relativePath);
  const existing = index >= 0 ? fileSnapshots[index] : undefined;
  if (existing !== undefined && index >= 0) {
    fileSnapshots[index] = {
      kind: existing.kind,
      path: existing.path,
      relativePath: existing.relativePath,
      content: result.newContent,
      hash: sha256(result.newContent),
    };
  } else {
    fileSnapshots.push({
      kind: "create",
      path,
      relativePath,
      content: result.newContent,
      hash: sha256(result.newContent),
    });
  }
};

export const checkPostApplyConsistency = async (
  deps: Pick<AcpEditLifecycleDeps, "absoluteSessionPath" | "sessionRelativePath" | "readClientFile">,
  input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editResults: readonly CodingEditResult[];
  },
): Promise<PostApplyConsistencyCheck[]> => {
  const latestExpectedByPath = new Map<string, CodingEditResult>();
  for (const result of input.editResults) {
    if (!result.ok || result.newHash === undefined) {
      continue;
    }
    const absolutePath = deps.absoluteSessionPath(input.session, result.path);
    latestExpectedByPath.set(deps.sessionRelativePath(input.session, absolutePath), {
      ...result,
      path: absolutePath,
    });
  }

  const checks: PostApplyConsistencyCheck[] = [];
  for (const [relativePath, result] of latestExpectedByPath) {
    const expectedHash = result.newHash;
    if (expectedHash === undefined) {
      continue;
    }
    try {
      const content = await deps.readClientFile({
        sessionId: input.session.id,
        telemetry: input.telemetry,
        path: result.path,
      });
      const actualHash = sha256(content);
      checks.push({
        path: relativePath,
        status: actualHash === result.newHash ? "consistent" : "inconsistent",
        expectedHash,
        actualHash,
        reason: actualHash === result.newHash
          ? "client file content matches the written edit hash"
          : "client file content does not match the written edit hash",
        ...(actualHash === result.newHash ? {} : { errorCode: "post_apply_behavior_failure" as const }),
      });
    } catch (error) {
      checks.push({
        path: relativePath,
        status: "inconclusive",
        expectedHash,
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }
  return checks;
};

export const hasPostApplyInconsistency = (checks: readonly PostApplyConsistencyCheck[]): boolean =>
  checks.some((check) => check.status === "inconsistent");

export const rollbackLiveEdits = async (
  deps: Pick<AcpEditLifecycleDeps, "absoluteSessionPath" | "sessionRelativePath" | "writeClientFileWithPermission">,
  input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    baselineFileSnapshots: readonly CodingFileSnapshot[];
    currentFileSnapshots: readonly CodingFileSnapshot[];
    editResults: readonly CodingEditResult[];
  },
): Promise<CodingEditResult[]> => {
  const changedPaths = new Set(
    input.editResults
      .filter((result) => result.ok)
      .map((result) => deps.sessionRelativePath(input.session, deps.absoluteSessionPath(input.session, result.path))),
  );
  const results: CodingEditResult[] = [];
  for (const relativePath of changedPaths) {
    const baseline = input.baselineFileSnapshots.find((file) => file.relativePath === relativePath);
    const current = input.currentFileSnapshots.find((file) => file.relativePath === relativePath);
    const absolutePath = deps.absoluteSessionPath(input.session, relativePath);
    if (baseline === undefined) {
      results.push({
        path: absolutePath,
        ok: false,
        reason: "rollback skipped because no baseline snapshot exists for the changed file",
        editStrategyId: "edit.rollback.acp-write.v1",
        editStatus: "rollback_failed",
        errorCode: "rollback_failed",
        ...(current?.hash === undefined ? {} : { oldHash: current.hash }),
      });
      continue;
    }

    if (baseline.kind === "create") {
      results.push({
        path: absolutePath,
        ok: false,
        reason: "rollback skipped: baseline was a create-from-scratch snapshot; keeping latest content",
        editStrategyId: "edit.rollback.acp-write.v1",
        editStatus: "rollback_skipped",
        errorCode: "rollback_skipped_greenfield",
        ...(current?.hash === undefined ? {} : { oldHash: current.hash }),
      });
      continue;
    }

    const writeResult = await deps.writeClientFileWithPermission({
      sessionId: input.session.id,
      telemetry: input.telemetry,
      path: absolutePath,
      oldContent: current?.content ?? "",
      newContent: baseline.content,
      reason: "Rollback after unrepaired verification or post-apply consistency failure.",
      editStrategyId: "edit.rollback.acp-write.v1",
      editStrategyFamily: "whole_file",
      renderedEditContractVersion: input.session.optimizerPin.telemetry.renderedEditContractVersion,
    });
    results.push({
      ...writeResult,
      editStrategyId: "edit.rollback.acp-write.v1",
      editStatus: writeResult.ok ? "rollback_applied" : "rollback_failed",
      ...(writeResult.ok ? { newContent: baseline.content } : { errorCode: "rollback_failed" as const }),
    });
  }
  return results;
};

export const previewAndWriteLiveEdit = async (
  deps: AcpEditLifecycleDeps,
  input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    fileSnapshots: CodingFileSnapshot[];
    edit: CodingEditOperation;
    signal?: AbortSignal;
  },
): Promise<CodingEditResult[]> => {
  const editStartedAt = new Date().toISOString();
  const targetFiles = input.edit.targetFiles.length > 0
    ? input.edit.targetFiles
    : targetFilesForEditInput(input.edit.editInput, input.fileSnapshots);
  const primaryTarget = targetFiles[0] ?? input.fileSnapshots[0]?.relativePath ?? "workspace";
  const primarySnapshot = input.fileSnapshots.find((file) => file.relativePath === primaryTarget);
  const primaryAbsolutePath = deps.absoluteSessionPath(input.session, primaryTarget);
  const applyResult = await deps.runAcpTool<EditApplyResult>({
    sessionId: input.session.id,
    telemetry: input.telemetry,
    title: `Preview edit strategy (${input.edit.editStrategyFamily})`,
    toolName: "bag.edit.preview",
    kind: "edit",
    rawInput: {
      editStrategyId: input.edit.editStrategyId,
      editStrategyFamily: input.edit.editStrategyFamily,
      renderedEditToolContractId: input.edit.renderedEditToolContractId,
      renderedEditContractVersion: input.session.optimizerPin.telemetry.renderedEditContractVersion,
      targetFiles,
      reason: input.edit.reason,
      editInput: input.edit.editInput,
    },
    locations: targetFiles.map((path) => artifactLocation(deps.absoluteSessionPath(input.session, path))),
    ...(input.signal === undefined ? {} : { signal: input.signal }),
    fn: async () => {
      try {
        return applyEdit(
          {
            files: input.fileSnapshots.map((file) => ({ path: file.relativePath, content: file.content })),
            protectedPaths: input.session.optimizerPin.resolvedPolicy.codebaseProfile.protectedPaths,
          },
          input.edit.editInput,
        );
      } catch (error) {
        return failedEditApplyResult(
          input.edit.editStrategyFamily,
          "schema_validation_error",
          error instanceof Error ? error.message : String(error),
        );
      }
    },
  });

  if (applyResult.status !== "applied") {
    const editAttempt = editAttemptFromAcpWrite({
      session: input.session,
      editStartedAt,
      edit: input.edit,
      targetFiles,
      fileSnapshots: input.fileSnapshots,
      applyResult,
      writeResults: [],
    });
    const result: CodingEditResult = {
      path: primaryAbsolutePath,
      ok: applyResult.status === "skipped",
      reason: applyResult.status === "skipped"
        ? "edit skipped because the strategy produced no file changes"
        : applyResult.errorMessage ?? applyResult.errorCode ?? "edit preview failed",
      editStrategyId: input.edit.editStrategyId,
      editStatus: applyResult.status,
      ...(primarySnapshot?.hash === undefined ? {} : { oldHash: primarySnapshot.hash, newHash: primarySnapshot.hash }),
      ...(applyResult.errorCode === undefined ? {} : { errorCode: applyResult.errorCode }),
      editAttempt,
    };
    input.telemetry.recordEditAttempt(editAttempt);
    return [result];
  }

  const writeResults: CodingEditResult[] = [];
  for (const changedFile of applyResult.changedFiles) {
    const absolutePath = deps.absoluteSessionPath(input.session, changedFile.path);
    const snapshot = input.fileSnapshots.find((file) => file.relativePath === changedFile.path);
    const oldContent = snapshot?.content ?? changedFile.beforeContent ?? "";
    if (changedFile.afterContent === undefined) {
      writeResults.push({
        path: absolutePath,
        ok: false,
        reason: "delete edits are not supported by the ACP write path yet",
        editStrategyId: input.edit.editStrategyId,
        editStatus: applyResult.status,
        errorCode: "partial_apply",
        ...(snapshot?.hash === undefined ? {} : { oldHash: snapshot.hash }),
      });
      continue;
    }
    const writeResult = await deps.writeClientFileWithPermission({
      sessionId: input.session.id,
      telemetry: input.telemetry,
      path: absolutePath,
      oldContent,
      newContent: changedFile.afterContent,
      reason: input.edit.reason,
      editStrategyId: input.edit.editStrategyId,
      editStrategyFamily: input.edit.editStrategyFamily,
      renderedEditContractVersion: input.session.optimizerPin.telemetry.renderedEditContractVersion,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
    writeResults.push({
      ...writeResult,
      editStrategyId: input.edit.editStrategyId,
      editStatus: applyResult.status,
      newContent: changedFile.afterContent,
    });
  }
  const editAttempt = editAttemptFromAcpWrite({
    session: input.session,
    editStartedAt,
    edit: input.edit,
    targetFiles,
    fileSnapshots: input.fileSnapshots,
    applyResult,
    writeResults,
  });
  input.telemetry.recordEditAttempt(editAttempt);
  return writeResults.length > 0 ? writeResults.map((result) => ({ ...result, editAttempt })) : [{
    path: primaryAbsolutePath,
    ok: false,
    reason: "edit apply reported success but produced no writable file changes",
    editStrategyId: input.edit.editStrategyId,
    editStatus: applyResult.status,
    errorCode: "unknown_error",
    ...(primarySnapshot?.hash === undefined ? {} : { oldHash: primarySnapshot.hash }),
    editAttempt,
  }];
};
