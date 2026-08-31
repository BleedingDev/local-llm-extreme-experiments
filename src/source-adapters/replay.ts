import { createHash } from "node:crypto";
import type { HaloSpan } from "../telemetry";
import {
  AcpReplayCaptureSchema,
  extractReplayEvalCaseSkeleton,
  redactAcpReplayCaptureForLocalSafeUse,
  type AcpReplayCapture,
  type AcpReplayCaptureInput,
  type ReplayEvalCaseSkeleton,
  type ReplayExtractionMetadataInput,
  type ReplayRedactionOptions,
  type ReplayRedactionReport,
} from "../replay";
import type { JsonValue } from "../optimizer/types";
import type { EvalSplit } from "../eval-harness/types";
import type { SourceMetadata } from "./boundary";
import type { CanonicalSourceRecord } from "./canonical";
import type { SourceAdapterFailureKind } from "./failures";

type AcpReplayRecordInput = AcpReplayCaptureInput["records"][number];

export type SourceAdapterReplayExportOptions = {
  captureId?: string;
  createdAt?: string;
  defaultSplitHint?: EvalSplit;
  metadata?: Partial<ReplayExtractionMetadataInput>;
  redaction?: ReplayRedactionOptions;
  minimumDistinctSessionCount?: number;
};

export type SourceAdapterReplayExportBlocker = {
  code: "insufficient_safe_sessions";
  message: string;
  distinctSessionCount: number;
  requiredDistinctSessionCount: number;
};

export type SourceAdapterReplayExportCase = {
  capture: AcpReplayCapture;
  replayCase: ReplayEvalCaseSkeleton;
  redactionReport: ReplayRedactionReport;
  blocker?: SourceAdapterReplayExportBlocker;
};

export const exportCanonicalSourceRecordsToReplayCase = (
  records: readonly CanonicalSourceRecord[],
  options: SourceAdapterReplayExportOptions = {},
): SourceAdapterReplayExportCase => {
  const captureInput = canonicalSourceRecordsToReplayCapture(records, options);
  const redacted = redactAcpReplayCaptureForLocalSafeUse(captureInput, options.redaction);
  const metadata = sourceAdapterReplayMetadata(redacted.capture, records, options.metadata);
  return {
    capture: redacted.capture,
    replayCase: extractReplayEvalCaseSkeleton({
      capture: redacted.capture,
      metadata,
    }),
    redactionReport: redacted.report,
    ...(sourceAdapterReplayExportBlocker(records, options.minimumDistinctSessionCount) ?? {}),
  };
};

export const canonicalSourceRecordsToReplayCapture = (
  records: readonly CanonicalSourceRecord[],
  options: SourceAdapterReplayExportOptions = {},
): AcpReplayCapture => {
  const source = commonSource(records);
  const captureId = optimizerId(options.captureId ?? `capture.source-adapter.${stableId([
    source.sourceType,
    source.sessionId,
    source.path,
    records.map((record) => record.span.span_id).join("."),
  ].join("|"))}`);
  const replayRecords = replayRecordsFromCanonical(records, captureId);
  const promptRecords = replayRecords.filter((record) => record.recordKind === "prompt");
  const recordsWithPrompt = promptRecords.length > 0
    ? replayRecords
    : [syntheticPromptRecord(captureId, records[0]?.span), ...replayRecords];
  const promptRecordId = recordsWithPrompt.find((record) => record.recordKind === "prompt")?.recordId;
  const linkedRecords = promptRecordId == null
    ? recordsWithPrompt
    : recordsWithPrompt.map((record): AcpReplayRecordInput =>
        record.recordKind === "mode_route"
          ? { ...record, promptRecordId, parentRecordIds: [promptRecordId] }
          : record);

  return AcpReplayCaptureSchema.parse({
    captureId,
    createdAt: options.createdAt ?? records[0]?.span.start_time ?? "1970-01-01T00:00:00.000Z",
    source: {
      sourceType: source.sourceType,
      path: source.path,
      sessionId: source.sessionId,
      traceIds: uniqueSorted(records.map((record) => record.span.trace_id)),
    },
    defaultSplitHint: options.defaultSplitHint ?? "dev",
    redactionStatus: "redacted",
    records: linkedRecords,
  } satisfies AcpReplayCaptureInput);
};

export const sourceAdapterReplayExportBlocker = (
  records: readonly CanonicalSourceRecord[],
  minimumDistinctSessionCount = 50,
): { blocker: SourceAdapterReplayExportBlocker } | undefined => {
  const sessions = new Set(records.map((record) =>
    record.lineage.sessionId ?? record.source.sessionId ?? record.span.trace_id));
  if (sessions.size >= minimumDistinctSessionCount) return undefined;
  return {
    blocker: {
      code: "insufficient_safe_sessions",
      message:
        `Only ${sessions.size} redacted source sessions are available; ${minimumDistinctSessionCount} are required before claiming a 50-session export.`,
      distinctSessionCount: sessions.size,
      requiredDistinctSessionCount: minimumDistinctSessionCount,
    },
  };
};

const replayRecordsFromCanonical = (
  records: readonly CanonicalSourceRecord[],
  captureId: string,
): AcpReplayRecordInput[] =>
  records.flatMap((record, index) => {
    const span = record.span;
    const eventKind = stringAttr(span, "source.adapter.event_kind");
    const recordId = optimizerId(`record.${captureId}.${index}.${eventKind ?? "span"}`);
    const traceRefs = [{
      traceId: span.trace_id,
      spanId: span.span_id,
      ...(span.parent_span_id.length === 0 ? {} : { parentSpanId: span.parent_span_id }),
    }];
    const base = {
      recordId,
      traceRefs,
      artifactRefs: sourceArtifactRefs(span),
    };
    const role = stringAttr(span, "message.role");

    if (eventKind?.includes("user_message") || eventKind === "message" && role === "user") {
      return [{
        ...base,
        recordKind: "prompt",
        promptRole: "user",
        promptEvent: failureKind(span) === "user_correction" ? "user_correction" : "message",
        content: textAttr(span, "input.value") ?? "Observed user message was redacted.",
        contentRedactionStatus: "redacted",
      } satisfies AcpReplayRecordInput];
    }
    if (eventKind === "system_message" || role === "system") {
      return [{
        ...base,
        recordKind: "prompt",
        promptRole: "system",
        content: textAttr(span, "output.value") ?? textAttr(span, "input.value") ?? "Observed system message was redacted.",
        contentRedactionStatus: "redacted",
      } satisfies AcpReplayRecordInput];
    }
    if (eventKind?.includes("assistant_message") || eventKind === "message" && role === "assistant") {
      return [{
        ...base,
        recordKind: "prompt",
        promptRole: "assistant",
        content: textAttr(span, "output.value") ?? "Observed assistant message was redacted.",
        contentRedactionStatus: "redacted",
      } satisfies AcpReplayRecordInput];
    }
    if (eventKind === "permission_mode") {
      return modeRouteRecord(span, base);
    }
    if (isTerminalSpan(span)) {
      return [terminalRecord(span, base)];
    }
    if (isFileReadSpan(span)) {
      return [fileReadRecord(span, base)];
    }
    if (span.attributes["openinference.span.kind"] === "TOOL" || eventKind?.includes("tool") === true) {
      return [toolCallRecord(span, base)];
    }
    if (span.status.code === "STATUS_CODE_ERROR") {
      return [toolCallRecord(span, base)];
    }
    return [];
  });

const modeRouteRecord = (
  span: HaloSpan,
  base: Pick<AcpReplayRecordInput, "recordId" | "traceRefs" | "artifactRefs">,
): AcpReplayRecordInput => {
  const permissionMode = lower(stringAttr(span, "source.cc.permission_mode"));
  const selectedMode = permissionMode === "bypasspermissions" || permissionMode === "yolo"
    ? "yolo"
    : permissionMode === "acceptedits" || permissionMode === "default"
      ? "mutating"
      : permissionMode === "readonly" || permissionMode === "plan"
        ? "read_only"
        : "unknown";
  return {
    ...base,
    recordKind: "mode_route",
    promptRecordId: "record.synthetic.prompt",
    parentRecordIds: ["record.synthetic.prompt"],
    selectedMode,
    sideEffectPolicy: selectedMode === "read_only" ? "read_only" : selectedMode === "mutating" ? "write_allowed" : "unknown",
    reason: "Observed source transcript permission-mode routing state.",
  };
};

const fileReadRecord = (
  span: HaloSpan,
  base: Pick<AcpReplayRecordInput, "recordId" | "traceRefs" | "artifactRefs">,
): AcpReplayRecordInput => {
  const input = objectAttr(span, "input.value");
  const path = stringFrom(input?.file_path ?? input?.path) ?? "redacted-path";
  const failed = span.status.code === "STATUS_CODE_ERROR";
  return {
    ...base,
    recordKind: "file_read",
    path,
    status: failed ? "failed" : "succeeded",
    contentHash: `sha256:${stableId(JSON.stringify(span.attributes["output.value"] ?? span.span_id))}`,
    excerpt: failed ? undefined : textAttr(span, "output.value"),
    redactionStatus: failed ? "redacted" : "hash_only",
    errorCode: failed ? failureErrorCode(span) : undefined,
  };
};

const terminalRecord = (
  span: HaloSpan,
  base: Pick<AcpReplayRecordInput, "recordId" | "traceRefs" | "artifactRefs">,
): AcpReplayRecordInput => {
  const input = objectAttr(span, "input.value");
  const commandText = stringFrom(input?.command ?? input?.cmd ?? span.attributes["input.value"]) ?? stringAttr(span, "tool.name") ?? "command";
  const timedOut = failureKind(span) === "timeout";
  const failed = span.status.code === "STATUS_CODE_ERROR";
  return {
    ...base,
    recordKind: "terminal_command",
    commandId: optimizerId(`terminal.${base.recordId}`),
    command: commandText.split(/\s+/).filter((part) => part.length > 0),
    cwd: stringFrom(input?.cwd),
    status: timedOut ? "timed_out" : failed ? "failed" : "succeeded",
    exitCode: failed ? null : 0,
    redactionStatus: "hash_only",
    errorCode: failed ? failureErrorCode(span) : undefined,
  };
};

const toolCallRecord = (
  span: HaloSpan,
  base: Pick<AcpReplayRecordInput, "recordId" | "traceRefs" | "artifactRefs">,
): AcpReplayRecordInput => {
  const failed = span.status.code === "STATUS_CODE_ERROR";
  const kind = failureKind(span);
  return {
    ...base,
    recordKind: "tool_call",
    toolCallId: optimizerId(stringAttr(span, "tool.call_id") ?? `tool.${base.recordId}`),
    namespace: namespaceForTool(span),
    name: optimizerId(stringAttr(span, "tool.name") ?? "unknown_tool"),
    status: kind === "timeout" ? "timed_out" : kind === "permission_denied" ? "permission_denied" : failed ? "failed" : "succeeded",
    args: jsonAttr(span, "input.value"),
    result: jsonAttr(span, "output.value"),
    resultStyle: typeof span.attributes["output.value"] === "object" ? "json" : "text",
    retryCount: 0,
    redactionStatus: "redacted",
    errorCode: failed ? failureErrorCode(span) : undefined,
  };
};

const syntheticPromptRecord = (captureId: string, span: HaloSpan | undefined): AcpReplayRecordInput => ({
  recordId: optimizerId(`record.${captureId}.synthetic.prompt`),
  recordKind: "prompt",
  promptRole: "user",
  content: "Replay the observed source-adapter session while preserving failure evidence and source lineage.",
  contentRedactionStatus: "redacted",
  traceRefs: span == null ? [] : [{ traceId: span.trace_id, spanId: span.span_id }],
});

const sourceAdapterReplayMetadata = (
  capture: AcpReplayCapture,
  records: readonly CanonicalSourceRecord[],
  overrides: Partial<ReplayExtractionMetadataInput> | undefined,
): ReplayExtractionMetadataInput => {
  const failures = records
    .map((record) => failureKind(record.span))
    .filter((kind): kind is SourceAdapterFailureKind => kind != null);
  const sourceType = capture.source.sourceType;
  return {
    evalCaseId: overrides?.evalCaseId ?? optimizerId(`replay.eval.source-adapter.${stableId(capture.captureId)}`),
    title: overrides?.title ?? (failures.length === 0
      ? "Observed source-adapter replay"
      : `Observed source-adapter failure replay: ${uniqueSorted(failures).join(", ")}`),
    task: overrides?.task ?? firstPromptContent(capture) ?? "Replay observed source-adapter behavior.",
    split: overrides?.split ?? capture.defaultSplitHint ?? "dev",
    splitRationale: overrides?.splitRationale
      ?? "Source-adapter export defaults to a visible dev split unless the caller explicitly assigns train or hidden holdout.",
    oracleStrength: overrides?.oracleStrength ?? "weak",
    expectedBehavior: overrides?.expectedBehavior ?? {
      summary:
        "Preserve observed Claude/Codex/BAG baseline behavior and failure evidence for comparison; observed traces are evidence, not infallible gold.",
      assertions: [],
      notes: [
        "Generated from redacted canonical source-adapter spans.",
        "Claude/Codex/BAG outputs are treated as observed baselines rather than golden expected behavior.",
      ],
    },
    sourceRefs: overrides?.sourceRefs ?? [{
      sourceKind: "capture",
      captureId: capture.captureId,
      path: capture.source.path,
      redactionStatus: capture.redactionStatus,
    }],
    tags: overrides?.tags ?? ["replay", "source-adapter", "observed-baseline", sourceType, ...uniqueSorted(failures)],
    timeoutMs: overrides?.timeoutMs ?? 120000,
    ...(overrides?.fixtureWorkspace === undefined ? {} : { fixtureWorkspace: overrides.fixtureWorkspace }),
  };
};

const commonSource = (records: readonly CanonicalSourceRecord[]): SourceMetadata => {
  const source = records[0]?.source;
  if (source == null) {
    return {
      sourceType: "spans-jsonl",
      inspectedRecordCount: 0,
      detectedSignals: ["canonical-span-export"],
    };
  }
  return source;
};

const sourceArtifactRefs = (span: HaloSpan): string[] => {
  const path = stringAttr(span, "source.adapter.path");
  return path == null ? [] : [`source:${stableId(path)}`];
};

const namespaceForTool = (span: HaloSpan): string | undefined => {
  const sourceType = stringAttr(span, "source.adapter.type");
  if (sourceType == null) return undefined;
  return optimizerId(sourceType.replace(/-session-jsonl(?:-v2)?$/, ""));
};

const isTerminalSpan = (span: HaloSpan): boolean => {
  const toolName = lower(stringAttr(span, "tool.name"));
  const phase = stringAttr(span, "source.failure.phase");
  return phase === "terminal" || toolName === "bash" || toolName === "exec_command" || toolName === "shell";
};

const isFileReadSpan = (span: HaloSpan): boolean => {
  const toolName = lower(stringAttr(span, "tool.name"));
  return toolName === "read" || toolName === "read_file" || toolName === "view";
};

const firstPromptContent = (capture: AcpReplayCapture): string | undefined =>
  capture.records.find((record): record is Extract<AcpReplayCapture["records"][number], { recordKind: "prompt" }> =>
    record.recordKind === "prompt" && record.promptRole === "user")?.content;

const failureKind = (span: HaloSpan): SourceAdapterFailureKind | undefined =>
  stringAttr(span, "source.failure.kind") as SourceAdapterFailureKind | undefined;

const failureErrorCode = (span: HaloSpan): string | undefined =>
  stringAttr(span, "source.failure.error_code") ?? (span.status.code === "STATUS_CODE_ERROR" ? "generic_error" : undefined);

const stringAttr = (span: HaloSpan, key: string): string | undefined => stringFrom(span.attributes[key]);

const textAttr = (span: HaloSpan, key: string): string | undefined => {
  const value = span.attributes[key];
  if (typeof value === "string" && value.length > 0) return value;
  if (value == null) return undefined;
  return JSON.stringify(value);
};

const objectAttr = (span: HaloSpan, key: string): Record<string, unknown> | undefined => {
  const value = span.attributes[key];
  return isObject(value) ? value : undefined;
};

const jsonAttr = (span: HaloSpan, key: string): JsonValue | undefined => toJsonValue(span.attributes[key]);

const toJsonValue = (value: unknown): JsonValue | undefined => {
  if (value == null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value as JsonValue;
  }
  if (Array.isArray(value)) {
    return value.map((item) => toJsonValue(item) ?? null);
  }
  if (isObject(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, nested]) => [key, toJsonValue(nested) ?? null]));
  }
  return String(value);
};

const stringFrom = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const lower = (value: string | undefined): string | undefined => value?.toLowerCase();

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value != null && !Array.isArray(value);

const optimizerId = (value: string): string => {
  const sanitized = value.replace(/[^A-Za-z0-9._:-]+/g, ".").replace(/^[^A-Za-z0-9]+/, "").replace(/[^A-Za-z0-9]+$/, "");
  return sanitized.length > 0 ? sanitized : `id.${stableId(value)}`;
};

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const stableId = (value: string): string =>
  createHash("sha256").update(value).digest("hex").slice(0, 16);
