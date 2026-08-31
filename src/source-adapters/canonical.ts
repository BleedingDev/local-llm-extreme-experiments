import { createHash } from "node:crypto";
import type { HaloSpan } from "../telemetry";
import type { SourceMetadata } from "./boundary";
import { canonicalizeCcSessionV2 } from "./cc-session-v2";
import {
  redactSourceRecord,
  type RedactedSourceRecord,
  type SourceAdapterRedactionOptions,
  type SourceRecordLineage,
} from "./redaction";
import {
  classifySourceAdapterFailure,
  sourceAdapterFailureAttributes,
} from "./failures";

export type SourceAdapterCanonicalDiagnosticCode =
  | "invalid_native_span"
  | "non_object_record"
  | "unsupported_record"
  | "unsupported_source_type";

export type SourceAdapterCanonicalDiagnostic = {
  code: SourceAdapterCanonicalDiagnosticCode;
  message: string;
  sourceType?: SourceMetadata["sourceType"];
  recordIndex?: number;
  line?: number;
  recordType?: string;
  eventKind?: string;
};

export type CanonicalSourceRecordKind = "span";

export type CanonicalSourceRecord = {
  kind: CanonicalSourceRecordKind;
  source: SourceMetadata;
  lineage: SourceRecordLineage;
  redaction: RedactedSourceRecord["redaction"];
  span: HaloSpan;
};

export type CanonicalSourceAdapterOutput = {
  records: CanonicalSourceRecord[];
  diagnostics: SourceAdapterCanonicalDiagnostic[];
};

export type CanonicalizeSourceRecordInput = {
  source: SourceMetadata;
  record: unknown;
  recordIndex?: number;
  line?: number;
  redactionOptions?: Omit<SourceAdapterRedactionOptions, "source">;
};

export type CanonicalizeSourceRecordsInput = {
  source: SourceMetadata;
  records: readonly unknown[];
  redactionOptions?: Omit<SourceAdapterRedactionOptions, "source">;
};

type JsonObject = Record<string, unknown>;

type SourceEvent = {
  eventKind: string;
  name: string;
  observationKind: "AGENT" | "CHAIN" | "LLM" | "TOOL";
  statusCode?: HaloSpan["status"]["code"] | undefined;
  statusMessage?: string | undefined;
  timestamp?: string | undefined;
  attributes: Record<string, unknown>;
};

const CANONICAL_SCOPE_NAME = "bag.source-adapters";
const CANONICAL_SCOPE_VERSION = "canonical.v1";
const DEFAULT_TIMESTAMP = "1970-01-01T00:00:00.000Z";

export const canonicalizeSourceRecords = (
  input: CanonicalizeSourceRecordsInput,
): CanonicalSourceAdapterOutput => {
  const outputs = input.records.map((record, index) =>
    canonicalizeSourceRecord(cleanObject({
      source: input.source,
      record,
      recordIndex: index,
      redactionOptions: input.redactionOptions,
    }) as CanonicalizeSourceRecordInput));
  return {
    records: outputs.flatMap((output) => output.records),
    diagnostics: outputs.flatMap((output) => output.diagnostics),
  };
};

export const canonicalizeSourceRecord = (
  input: CanonicalizeSourceRecordInput,
): CanonicalSourceAdapterOutput => {
  if (!isObject(input.record)) {
    return {
      records: [],
      diagnostics: [{
        code: "non_object_record",
        message: "Canonical source adapter output requires JSON object records.",
        sourceType: input.source.sourceType,
        ...(input.recordIndex == null ? {} : { recordIndex: input.recordIndex }),
        ...(input.line == null ? {} : { line: input.line }),
      }],
    };
  }

  if (input.source.sourceType === "spans-jsonl") {
    return canonicalizeNativeSpan(input);
  }

  if (input.source.sourceType === "cc-session-jsonl-v2") {
    return canonicalizeCcSessionV2({
      source: input.source,
      records: [input.record],
      recordIndexOffset: input.recordIndex ?? 0,
      ...(input.line === undefined ? {} : { lineNumbers: [input.line] }),
      ...(input.redactionOptions === undefined ? {} : { redactionOptions: input.redactionOptions }),
    });
  }

  const redacted = redactSourceRecord(input.record, {
    ...input.redactionOptions,
    source: input.source,
  });
  const redactedObject = asObject(redacted.record);
  if (redactedObject == null) {
    return {
      records: [],
      diagnostics: [{
        code: "non_object_record",
        message: "Redacted source record was not a JSON object.",
        sourceType: input.source.sourceType,
        ...(input.recordIndex == null ? {} : { recordIndex: input.recordIndex }),
        ...(input.line == null ? {} : { line: input.line }),
      }],
    };
  }
  const event = eventFromSourceRecord(input.source, redactedObject);

  if (event == null) {
    const recordType = stringValue(redactedObject.type) ?? stringValue(asObject(redactedObject.payload)?.type)
      ?? stringValue(asObject(asObject(redactedObject.params)?.update)?.sessionUpdate);
    return {
      records: [],
      diagnostics: [cleanObject({
        code: "unsupported_record",
        message: `No canonical mapping for ${input.source.sourceType} record.`,
        sourceType: input.source.sourceType,
        recordIndex: input.recordIndex,
        line: input.line,
        recordType,
      }) as SourceAdapterCanonicalDiagnostic],
    };
  }

  return {
    records: [{
      kind: "span",
      source: input.source,
      lineage: lineageForEvent(redacted.lineage, event),
      redaction: redacted.redaction,
      span: spanFromEvent(cleanObject({
        source: input.source,
        redacted,
        event,
        recordIndex: input.recordIndex,
        line: input.line,
      }) as {
        source: SourceMetadata;
        redacted: RedactedSourceRecord;
        event: SourceEvent;
        recordIndex?: number;
        line?: number;
      }),
    }],
    diagnostics: [],
  };
};

export const canonicalSourceRecordsToJsonl = (records: readonly CanonicalSourceRecord[]): string =>
  records.map((record) => JSON.stringify(record.span)).join("\n") + (records.length === 0 ? "" : "\n");

const canonicalizeNativeSpan = (input: CanonicalizeSourceRecordInput): CanonicalSourceAdapterOutput => {
  if (!isHaloSpan(input.record)) {
    return {
      records: [],
      diagnostics: [{
        code: "invalid_native_span",
        message: "Native spans source record does not satisfy the canonical HaloSpan shape.",
        sourceType: input.source.sourceType,
        ...(input.recordIndex == null ? {} : { recordIndex: input.recordIndex }),
        ...(input.line == null ? {} : { line: input.line }),
      }],
    };
  }

  const redacted = redactSourceRecord(input.record, {
    ...input.redactionOptions,
    source: input.source,
  });
  const event: SourceEvent = {
    eventKind: "native_span",
    name: input.record.name,
    observationKind: observationKind(input.record),
    attributes: {},
  };
  const lineage = lineageForEvent(redacted.lineage, event);
  const redactedSpan = redacted.record as HaloSpan;
  return {
    records: [{
      kind: "span",
      source: input.source,
      lineage,
      redaction: redacted.redaction,
      span: {
        ...redactedSpan,
        resource: {
          attributes: {
            ...redactedSpan.resource.attributes,
            ...sourceResourceAttributes(input.source),
          },
        },
        attributes: cleanAttributes({
          ...redactedSpan.attributes,
          ...sourceAttributes(input.source, lineage, redacted.redaction, input.recordIndex, input.line),
          "source.adapter.event_kind": "native_span",
        }),
      },
    }],
    diagnostics: [],
  };
};

const eventFromSourceRecord = (source: SourceMetadata, record: JsonObject): SourceEvent | undefined => {
  switch (source.sourceType) {
    case "acp-session-jsonl":
      return acpEvent(record);
    case "codex-session-jsonl":
      return codexEvent(record);
    case "pi-session-jsonl":
      return piEvent(record);
    case "cc-session-jsonl-v2":
      return undefined;
    case "spans-jsonl":
      return undefined;
  }
};

const acpEvent = (record: JsonObject): SourceEvent | undefined => {
  const message = asObject(record.message) ?? record;
  const method = stringValue(message.method);
  const params = asObject(message.params);
  const update = asObject(params?.update) ?? asObject(record.update);
  const sessionUpdate = stringValue(update?.sessionUpdate) ?? stringValue(record.sessionUpdate);
  const timestamp = timestampFrom(record);

  if (sessionUpdate != null) {
    if (sessionUpdate === "tool_call" || sessionUpdate === "tool_call_update") {
      const status = statusFromToolUpdate(update);
      return {
        eventKind: sessionUpdate,
        name: `source.acp-session-jsonl.${sessionUpdate}`,
        observationKind: "TOOL",
        statusCode: status.code,
        statusMessage: status.message,
        timestamp,
        attributes: cleanAttributes({
          "source.acp.session_update": sessionUpdate,
          "tool.name": stringValue(update?.title),
          "tool.call_id": stringValue(update?.toolCallId),
          "tool.kind": stringValue(update?.kind),
          "tool.status": stringValue(update?.status),
          "input.value": update?.rawInput,
          "output.value": update?.rawOutput,
        }),
      };
    }

    if (sessionUpdate === "agent_message_chunk") {
      return {
        eventKind: sessionUpdate,
        name: "source.acp-session-jsonl.message",
        observationKind: "LLM",
        timestamp,
        attributes: cleanAttributes({
          "source.acp.session_update": sessionUpdate,
          "message.role": "assistant",
          "output.value": update?.content,
        }),
      };
    }

    if (sessionUpdate === "plan" || sessionUpdate === "current_mode_update" || sessionUpdate === "available_commands_update") {
      return {
        eventKind: sessionUpdate,
        name: `source.acp-session-jsonl.${sessionUpdate}`,
        observationKind: "CHAIN",
        timestamp,
        attributes: cleanAttributes({
          "source.acp.session_update": sessionUpdate,
          "output.value": update,
        }),
      };
    }
  }

  if (method === "initialize" || method === "session/new" || method === "session/load") {
    return {
      eventKind: "session",
      name: "source.acp-session-jsonl.session",
      observationKind: "CHAIN",
      timestamp,
      attributes: cleanAttributes({
        "source.acp.method": method,
        "input.value": params,
      }),
    };
  }

  if (method != null && (method.startsWith("fs/") || method.startsWith("terminal/"))) {
    return {
      eventKind: "tool_rpc",
      name: `source.acp-session-jsonl.${method.replaceAll("/", ".")}`,
      observationKind: "TOOL",
      timestamp,
      attributes: cleanAttributes({
        "source.acp.method": method,
        "input.value": params,
      }),
    };
  }

  return undefined;
};

const codexEvent = (record: JsonObject): SourceEvent | undefined => {
  const payload = asObject(record.payload);
  const type = stringValue(record.type);
  const payloadType = stringValue(payload?.type);
  const timestamp = timestampFrom(record) ?? timestampFrom(payload);

  if (type === "session_meta") {
    return {
      eventKind: "session",
      name: "source.codex-session-jsonl.session",
      observationKind: "CHAIN",
      timestamp,
      attributes: cleanAttributes({
        "source.codex.record_type": type,
        "source.codex.cwd": payload?.cwd,
        "llm.model_provider": payload?.model_provider,
        "inference.llm.model_name": payload?.model,
        "source.codex.cli_version": payload?.cli_version,
      }),
    };
  }

  if (type === "turn_context") {
    return {
      eventKind: "turn_context",
      name: "source.codex-session-jsonl.turn_context",
      observationKind: "CHAIN",
      timestamp,
      attributes: cleanAttributes({
        "source.codex.record_type": type,
        "inference.cwd": payload?.cwd,
        "inference.llm.model_name": payload?.model,
        "source.codex.approval_policy": payload?.approval_policy,
        "source.codex.sandbox_policy": payload?.sandbox_policy,
      }),
    };
  }

  if (type === "event_msg") {
    const error = payload?.error ?? payload?.message;
    if (error != null || stringValue(payload?.level) === "error") {
      return {
        eventKind: "error",
        name: "source.codex-session-jsonl.error",
        observationKind: "CHAIN",
        statusCode: "STATUS_CODE_ERROR",
        statusMessage: firstLine(error),
        timestamp,
        attributes: cleanAttributes({
          "source.codex.record_type": type,
          "error.message": error,
          "error.type": payload?.type,
          "output.value": payload,
        }),
      };
    }
    return undefined;
  }

  if (type !== "response_item" || payload == null) {
    return undefined;
  }

  if (payloadType === "message") {
    const role = stringValue(payload.role);
    return {
      eventKind: "message",
      name: "source.codex-session-jsonl.message",
      observationKind: role === "assistant" ? "LLM" : "AGENT",
      timestamp,
      attributes: cleanAttributes({
        "message.role": role,
        "input.value": role === "assistant" ? undefined : payload.content,
        "output.value": role === "assistant" ? payload.content : undefined,
      }),
    };
  }

  if (payloadType === "function_call" || payloadType === "tool_call") {
    return {
      eventKind: "tool_call",
      name: "source.codex-session-jsonl.tool_call",
      observationKind: "TOOL",
      timestamp,
      attributes: cleanAttributes({
        "tool.name": payload.name ?? asObject(payload.function)?.name,
        "tool.call_id": payload.call_id ?? payload.id,
        "input.value": payload.arguments ?? payload.input ?? asObject(payload.function)?.arguments,
      }),
    };
  }

  if (payloadType === "function_call_output" || payloadType === "tool_result") {
    return {
      eventKind: "tool_result",
      name: "source.codex-session-jsonl.tool_result",
      observationKind: "TOOL",
      timestamp,
      attributes: cleanAttributes({
        "tool.call_id": payload.call_id ?? payload.id,
        "tool.status": payload.status,
        "output.value": payload.output ?? payload.content,
      }),
    };
  }

  return undefined;
};

const piEvent = (record: JsonObject): SourceEvent | undefined => {
  const type = stringValue(record.type);
  const timestamp = timestampFrom(record);

  if (type === "session") {
    return {
      eventKind: "session",
      name: "source.pi-session-jsonl.session",
      observationKind: "CHAIN",
      timestamp,
      attributes: cleanAttributes({
        "source.pi.record_type": type,
        "inference.cwd": record.cwd,
        "source.pi.version": record.version,
      }),
    };
  }

  if (type === "user" || type === "assistant" || type === "system") {
    return {
      eventKind: "message",
      name: "source.pi-session-jsonl.message",
      observationKind: type === "assistant" ? "LLM" : "AGENT",
      timestamp,
      attributes: cleanAttributes({
        "message.role": stringValue(record.role) ?? type,
        "input.value": type === "assistant" ? undefined : record.content ?? record.message,
        "output.value": type === "assistant" ? record.content ?? record.message : undefined,
      }),
    };
  }

  if (type === "tool_use") {
    return {
      eventKind: "tool_call",
      name: "source.pi-session-jsonl.tool_call",
      observationKind: "TOOL",
      timestamp,
      attributes: cleanAttributes({
        "tool.name": record.name ?? asObject(record.function)?.name,
        "tool.call_id": record.tool_call_id ?? record.call_id ?? record.id,
        "input.value": record.input ?? record.arguments ?? asObject(record.function)?.arguments,
      }),
    };
  }

  if (type === "tool_result") {
    return {
      eventKind: "tool_result",
      name: "source.pi-session-jsonl.tool_result",
      observationKind: "TOOL",
      statusCode: stringValue(record.status) === "error" ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK",
      statusMessage: stringValue(record.error) ?? "",
      timestamp,
      attributes: cleanAttributes({
        "tool.call_id": record.tool_call_id ?? record.call_id ?? record.id,
        "tool.status": record.status,
        "output.value": record.output ?? record.content,
        "error.message": record.error,
      }),
    };
  }

  if (type === "model_change") {
    return {
      eventKind: "model_change",
      name: "source.pi-session-jsonl.model_change",
      observationKind: "CHAIN",
      timestamp,
      attributes: cleanAttributes({
        "inference.llm.model_name": record.model ?? record.to,
        "source.pi.previous_model": record.previousModel ?? record.from,
      }),
    };
  }

  if (type === "compaction" || type === "compaction_start" || type === "compaction_end") {
    return {
      eventKind: "compaction",
      name: `source.pi-session-jsonl.${type}`,
      observationKind: "CHAIN",
      timestamp,
      attributes: cleanAttributes({
        "source.pi.record_type": type,
        "input.value": record.before,
        "output.value": record.after ?? record.summary ?? record.content,
      }),
    };
  }

  return undefined;
};

const spanFromEvent = (input: {
  source: SourceMetadata;
  redacted: RedactedSourceRecord;
  event: SourceEvent;
  recordIndex?: number;
  line?: number;
}): HaloSpan => {
  const lineage = lineageForEvent(input.redacted.lineage, input.event);
  const timestamp = input.event.timestamp ?? DEFAULT_TIMESTAMP;
  const traceId = lineage.traceId ?? stableId("trace", input.source.sourceType, input.source.sessionId, input.source.path);
  const spanId = lineage.spanId
    ?? stableId("span", input.source.sourceType, input.source.sessionId, lineage.id, input.recordIndex, input.event.eventKind);
  const parentSpanId = lineage.parentSpanId
    ?? (lineage.parentId == null
      ? ""
      : stableId("span", input.source.sourceType, input.source.sessionId, lineage.parentId));
  const classification = classifySourceAdapterFailure({
    sourceType: input.source.sourceType,
    eventKind: input.event.eventKind,
    observationKind: input.event.observationKind,
    statusCode: input.event.statusCode,
    statusMessage: input.event.statusMessage,
    attributes: input.event.attributes,
  });

  return {
    trace_id: traceId,
    span_id: spanId,
    parent_span_id: parentSpanId,
    trace_state: "",
    name: input.event.name,
    kind: input.event.observationKind === "TOOL" ? "SPAN_KIND_CLIENT" : "SPAN_KIND_INTERNAL",
    start_time: timestamp,
    end_time: timestamp,
    status: {
      code: input.event.statusCode ?? classification?.statusCode ?? "STATUS_CODE_OK",
      message: input.event.statusMessage ?? classification?.statusMessage ?? "",
    },
    resource: {
      attributes: sourceResourceAttributes(input.source),
    },
    scope: {
      name: CANONICAL_SCOPE_NAME,
      version: CANONICAL_SCOPE_VERSION,
    },
    attributes: cleanAttributes({
      "openinference.span.kind": input.event.observationKind,
      "inference.observation_kind": input.event.observationKind,
      "inference.export.schema_version": 1,
      "source.adapter.event_kind": input.event.eventKind,
      ...sourceAttributes(input.source, lineage, input.redacted.redaction, input.recordIndex, input.line),
      ...input.event.attributes,
      ...sourceAdapterFailureAttributes(classification),
      "source.record.redacted": input.redacted.record,
    }),
  };
};

const lineageForEvent = (lineage: SourceRecordLineage, event: SourceEvent): SourceRecordLineage => {
  const toolCallId = stringValue(event.attributes["tool.call_id"]);
  const toolName = stringValue(event.attributes["tool.name"]);
  return {
    ...lineage,
    toolCallIds: toolCallId == null ? lineage.toolCallIds : uniqueSorted([toolCallId]),
    toolNames: toolName == null ? lineage.toolNames : uniqueSorted([...lineage.toolNames, toolName]),
  };
};

const sourceResourceAttributes = (source: SourceMetadata): Record<string, unknown> =>
  cleanAttributes({
    "service.name": "bleeding-agent-source-adapter",
    "telemetry.sdk.language": "typescript",
    "source.adapter.type": source.sourceType,
    "source.adapter.path": source.path,
    "source.adapter.session_id": source.sessionId,
    "source.adapter.schema_version": source.schemaVersion,
  });

const sourceAttributes = (
  source: SourceMetadata,
  lineage: SourceRecordLineage,
  redaction: RedactedSourceRecord["redaction"],
  recordIndex: number | undefined,
  line: number | undefined,
): Record<string, unknown> =>
  cleanAttributes({
    "source.adapter.type": source.sourceType,
    "source.adapter.path": source.path,
    "source.adapter.session_id": source.sessionId,
    "source.adapter.schema_version": source.schemaVersion,
    "source.adapter.detected_signals": source.detectedSignals,
    "source.record.index": recordIndex,
    "source.record.line": line,
    "source.lineage.id": lineage.id,
    "source.lineage.parent_id": lineage.parentId,
    "source.lineage.session_id": lineage.sessionId,
    "source.lineage.tool_call_ids": lineage.toolCallIds,
    "source.lineage.tool_names": lineage.toolNames,
    "source.redaction.secret_replacement_count": redaction.secretReplacementCount,
    "source.redaction.redaction_kinds": redaction.redactionKinds,
    "source.redaction.truncated_string_count": redaction.truncatedStringCount,
    "source.redaction.truncated_array_count": redaction.truncatedArrayCount,
    "source.redaction.truncated_depth_count": redaction.truncatedDepthCount,
    "source.redaction.full_content_included": redaction.fullContentIncluded,
    "source.redaction.secret_redaction_disabled": redaction.secretRedactionDisabled,
  });

const statusFromToolUpdate = (update: JsonObject | undefined): {
  code: HaloSpan["status"]["code"];
  message: string;
} => {
  const status = stringValue(update?.status);
  if (status === "failed" || status === "error") {
    return {
      code: "STATUS_CODE_ERROR",
      message: firstLine(asObject(update?.rawOutput)?.error ?? update?.rawOutput ?? "tool failed"),
    };
  }
  return {
    code: "STATUS_CODE_OK",
    message: "",
  };
};

const cleanAttributes = (attributes: Record<string, unknown>): Record<string, unknown> => {
  return cleanObject(attributes);
};

const cleanObject = <T extends Record<string, unknown>>(attributes: T): Record<string, unknown> => {
  const clean: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (value !== undefined) {
      clean[key] = value;
    }
  }
  return clean;
};

const isHaloSpan = (value: unknown): value is HaloSpan => {
  if (!isObject(value)) {
    return false;
  }
  return typeof value.trace_id === "string"
    && typeof value.span_id === "string"
    && typeof value.parent_span_id === "string"
    && typeof value.trace_state === "string"
    && typeof value.name === "string"
    && (value.kind === "SPAN_KIND_INTERNAL" || value.kind === "SPAN_KIND_CLIENT")
    && typeof value.start_time === "string"
    && typeof value.end_time === "string"
    && isObject(value.status)
    && (value.status.code === "STATUS_CODE_OK" || value.status.code === "STATUS_CODE_ERROR")
    && typeof value.status.message === "string"
    && isObject(value.resource)
    && isObject(value.resource.attributes)
    && isObject(value.scope)
    && typeof value.scope.name === "string"
    && typeof value.scope.version === "string"
    && isObject(value.attributes);
};

const observationKind = (span: HaloSpan): SourceEvent["observationKind"] => {
  const kind = span.attributes["inference.observation_kind"] ?? span.attributes["openinference.span.kind"];
  return kind === "AGENT" || kind === "CHAIN" || kind === "LLM" || kind === "TOOL" ? kind : "CHAIN";
};

const timestampFrom = (record: JsonObject | undefined): string | undefined => {
  const value = record?.timestamp ?? record?.time ?? record?.created_at ?? record?.createdAt;
  return typeof value === "string" && value.length > 0 ? value : undefined;
};

const firstLine = (value: unknown): string => {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return (text ?? "").split(/\r?\n/, 1)[0] ?? "";
};

const stableId = (...parts: readonly unknown[]): string =>
  createHash("sha256")
    .update(parts.map((part) => part == null ? "" : String(part)).join("\0"))
    .digest("hex")
    .slice(0, 32);

const stringValue = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const uniqueSorted = (values: readonly string[]): string[] => [...new Set(values)].sort();

const asObject = (value: unknown): JsonObject | undefined => isObject(value) ? value : undefined;

const isObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value != null && !Array.isArray(value);
