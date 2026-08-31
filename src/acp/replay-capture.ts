import {
  AcpReplayCaptureSchema,
  type AcpReplayCapture,
  type AcpReplayMode,
  type AcpReplayRecord,
} from "../replay";
import type { ToolCallMetric } from "../types";
import type { EditAttemptContract } from "../edit-strategy/types";
import type { BagAcpSession } from "./session";
import { replaySafeId } from "./surface";
import type { TerminalCommandResult } from "./terminal";
import type { CodingFileSnapshot } from "./coding-types";
import type { CodingProgressDiagnostic } from "./coding-progress-diagnostics";

export type BuildCodingReplayCaptureInput = {
  session: BagAcpSession;
  runId: string;
  task: string;
  tracePath: string;
  fileSnapshots: readonly CodingFileSnapshot[];
  editAttempts: readonly EditAttemptContract[];
  toolMetrics: readonly ToolCallMetric[];
  commandResults: readonly TerminalCommandResult[];
  artifactRefs: readonly string[];
  codingProgressDiagnostic?: CodingProgressDiagnostic;
};

export const buildCodingReplayCapture = (input: BuildCodingReplayCaptureInput): AcpReplayCapture => {
  const runId = replaySafeId(input.runId);
  const promptRecordId = `record.${runId}.prompt`;
  const routeRecordId = `record.${runId}.route`;
  const traceId = `trace.${runId}`;
  const records: AcpReplayRecord[] = [
    {
      recordId: promptRecordId,
      recordKind: "prompt",
      parentRecordIds: [],
      artifactRefs: [],
      promptRole: "user",
      promptEvent: "message",
      content: input.task,
      contentRedactionStatus: "raw_local_only",
      traceRefs: [{ traceId }],
    },
    {
      recordId: routeRecordId,
      recordKind: "mode_route",
      promptRecordId,
      parentRecordIds: [promptRecordId],
      artifactRefs: [],
      requestedMode: replayModeForSessionMode(input.session.mode),
      selectedMode: "mutating",
      ...(input.session.mode === "auto" ? { restoredMode: "auto" as const } : {}),
      sideEffectPolicy: input.session.yolo ? "terminal_allowed" : "write_allowed",
      reason: "ACP coding run selected the mutating file-edit and verification path.",
      traceRefs: [{ traceId }],
    },
  ];

  const fileReadRecordIds = new Map<string, string>();
  for (const [index, file] of input.fileSnapshots.entries()) {
    const recordId = `record.${runId}.file.${index}`;
    fileReadRecordIds.set(file.relativePath, recordId);
    records.push({
      recordId,
      recordKind: "file_read",
      parentRecordIds: [routeRecordId],
      artifactRefs: [],
      path: file.relativePath,
      status: file.kind === "create" ? "omitted" : "succeeded",
      ...(file.kind === "create" ? {} : { contentHash: `sha256:${file.hash}` }),
      redactionStatus: "hash_only",
      ranges: file.content.length === 0 ? [] : [{
        startLine: 0,
        endLine: Math.max(0, file.content.split(/\r?\n/).length - 1),
      }],
      ...(file.kind === "create" ? { errorCode: "planned_create_target" } : {}),
      traceRefs: [{ traceId }],
    });
  }

  for (const [index, attempt] of input.editAttempts.entries()) {
    const parentRecordIds = [...new Set([
      routeRecordId,
      ...attempt.targetFiles.flatMap((path) => {
        const recordId = fileReadRecordIds.get(path);
        return recordId === undefined ? [] : [recordId];
      }),
    ])];
    records.push({
      recordId: `record.${runId}.edit.${index}`,
      recordKind: "edit_attempt",
      parentRecordIds,
      attempt,
      artifactRefs: attempt.artifactRefs,
      traceRefs: [{ traceId }],
    });
  }

  for (const [index, command] of input.commandResults.entries()) {
    const commandText = [command.command, ...command.args];
    records.push({
      recordId: `record.${runId}.terminal.${index}`,
      recordKind: "terminal_command",
      parentRecordIds: [routeRecordId],
      artifactRefs: [],
      commandId: `terminal.${runId}.${index}`,
      command: commandText,
      cwd: input.session.cwd,
      status: command.signal == null ? command.exitCode === 0 ? "succeeded" : "failed" : "timed_out",
      exitCode: command.exitCode,
      signal: command.signal,
      redactionStatus: "raw_local_only",
      ...(command.exitCode === 0 ? {} : { errorCode: "verifier_error" }),
      traceRefs: [{ traceId }],
    });
  }

  for (const [index, metric] of input.toolMetrics.entries()) {
    records.push({
      recordId: `record.${runId}.tool.${index}`,
      recordKind: "tool_call",
      parentRecordIds: [routeRecordId],
      artifactRefs: [],
      toolCallId: `tool.${runId}.${index}`,
      ...(metric.namespace === undefined ? {} : { namespace: replaySafeId(metric.namespace) }),
      name: replaySafeId(metric.toolName),
      status: replayToolStatusForMetric(metric),
      args: {
        argumentHash: metric.argumentHash,
        argumentBytes: metric.argumentBytes,
        descriptionVersion: metric.descriptionVersion ?? null,
      },
      result: metric.ok
        ? {
            resultBytes: metric.resultBytes ?? null,
            resultKind: metric.resultKind,
          }
        : {
            error: metric.error ?? "tool failed",
            errorName: metric.errorName ?? null,
          },
      resultStyle: "json",
      retryCount: metric.retryCount,
      redactionStatus: "hash_only",
      ...(metric.ok ? {} : { errorCode: replayToolErrorCodeForMetric(metric) }),
      traceRefs: [{ traceId }],
    });
  }

  for (const [index, artifactRef] of input.artifactRefs.entries()) {
    records.push({
      recordId: `record.${runId}.artifact.${index}`,
      recordKind: "artifact_ref",
      parentRecordIds: [routeRecordId],
      artifactRefs: [artifactRef],
      artifactRef,
      artifactKind: artifactRef === input.tracePath ? "trace" : "other",
      path: artifactRef,
      redactionStatus: "raw_local_only",
      traceRefs: [{ traceId }],
    });
  }

  return AcpReplayCaptureSchema.parse({
    captureId: `capture.${runId}`,
    createdAt: new Date().toISOString(),
    source: {
      sourceType: "acp-session-jsonl",
      path: input.tracePath,
      sessionId: input.session.id,
      traceIds: [traceId],
    },
    context: {
      modelRole: input.session.optimizerPin.telemetry.modelRole,
      provider: input.session.optimizerPin.telemetry.provider,
      providerConfigRole: input.session.optimizerPin.telemetry.providerConfigRole,
      policyId: input.session.optimizerPin.telemetry.policyId,
      modelProfileId: input.session.optimizerPin.telemetry.modelProfileId,
      codebaseProfileId: input.session.optimizerPin.telemetry.codebaseProfileId,
      modelServerId: input.session.optimizerPin.telemetry.modelServerId,
      modelServerProfileId: input.session.optimizerPin.telemetry.modelServerProfileId,
      canonicalToolVersion: input.session.optimizerPin.telemetry.canonicalToolVersion,
      renderedToolVersion: input.session.optimizerPin.telemetry.renderedToolVersion,
      resultStyleVersion: input.session.optimizerPin.telemetry.resultStyleVersion,
      verificationPolicyVersion: input.session.optimizerPin.telemetry.verificationPolicyVersion,
      acpConsumerCapabilities: {
        source: input.session.clientCapabilities.source,
        fsReadTextFile: input.session.clientCapabilities.fsReadTextFile,
        fsWriteTextFile: input.session.clientCapabilities.fsWriteTextFile,
        terminal: input.session.clientCapabilities.terminal,
        richDiffContent: input.session.clientCapabilities.richDiffContent,
        richTerminalContent: input.session.clientCapabilities.richTerminalContent,
      },
    },
    redactionStatus: "raw_local_only",
    records,
  });
};

export const replayToolStatusForMetric = (metric: ToolCallMetric): Extract<
  AcpReplayRecord,
  { recordKind: "tool_call" }
>["status"] => {
  if (metric.ok) {
    return metric.resultBytes !== undefined && metric.resultBytes > 64_000 ? "truncated" : "succeeded";
  }
  const text = `${metric.error ?? ""} ${metric.errorName ?? ""}`.toLowerCase();
  if (text.includes("permission") || text.includes("rejected")) {
    return "permission_denied";
  }
  if (text.includes("timeout") || text.includes("timed out")) {
    return "timed_out";
  }
  if (text.includes("schema") || text.includes("argument") || text.includes("malformed")) {
    return "malformed_args";
  }
  return "failed";
};

export const replayToolErrorCodeForMetric = (metric: ToolCallMetric): string => {
  const status = replayToolStatusForMetric(metric);
  switch (status) {
    case "permission_denied":
      return "permission_denied";
    case "timed_out":
      return "timeout";
    case "malformed_args":
      return "malformed_arguments";
    case "failed":
    case "succeeded":
    case "truncated":
      return metric.errorName ?? "runtime_exception";
  }
};

export const replayModeForSessionMode = (mode: BagAcpSession["mode"]): AcpReplayMode => {
  switch (mode) {
    case "auto":
      return "auto";
    case "chat":
      return "chat";
    case "plan":
      return "read_only";
    case "run":
      return "mutating";
  }
};
