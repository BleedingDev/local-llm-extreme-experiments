import { createHash } from "node:crypto";
import type { HaloSpan } from "../telemetry";
import type {
  SourceAdapterType,
  SourceDetectionDiagnostic,
  SourceDetectionOptions,
  SourceDetectionResult,
  SourceMetadata,
  SourceRecordCountEstimate,
} from "./boundary";
import {
  redactSourceRecord,
  type RedactedSourceRecord,
  type SourceAdapterRedactionOptions,
  type SourceRecordLineage,
} from "./redaction";
import type {
  CanonicalSourceAdapterOutput,
  CanonicalSourceRecord,
  SourceAdapterCanonicalDiagnostic,
} from "./canonical";
import {
  classifySourceAdapterFailure,
  sourceAdapterFailureAttributes,
} from "./failures";

// New CC adapter type. Registered in boundary.ts so detection succeeds
// before the legacy "acp-session-jsonl" detector. We deliberately keep it
// separate (no removals) so existing behaviour is preserved.
export const CC_SESSION_V2_TYPE: SourceAdapterType = "cc-session-jsonl-v2" as SourceAdapterType;

const CC_RECORD_TYPES = new Set([
  "user",
  "assistant",
  "attachment",
  "system",
  "permission-mode",
  "last-prompt",
  "ai-title",
  "queue-operation",
  "file-history-snapshot",
  "summary",
  "error",
]);

const CANONICAL_SCOPE_NAME = "bag.source-adapters";
const CANONICAL_SCOPE_VERSION = "canonical.cc-v2";
const DEFAULT_TIMESTAMP = "1970-01-01T00:00:00.000Z";

type JsonObject = Record<string, unknown>;

type CcSourceEvent = {
  eventKind: string;
  name: string;
  observationKind: "AGENT" | "CHAIN" | "LLM" | "TOOL";
  statusCode?: HaloSpan["status"]["code"] | undefined;
  statusMessage?: string | undefined;
  timestamp?: string | undefined;
  attributes: Record<string, unknown>;
};

export const isCcSessionV2Record = (record: unknown): boolean => {
  if (!isObject(record)) return false;
  const t = stringValue(record.type);
  if (t == null || !CC_RECORD_TYPES.has(t)) return false;
  if (t === "user" || t === "assistant") {
    return isObject(record.message) && (typeof record.uuid === "string" || typeof record.sessionId === "string");
  }
  if (t === "attachment") {
    return isObject(record.attachment);
  }
  return typeof record.sessionId === "string"
    || typeof record.timestamp === "string"
    || typeof record.messageId === "string"
    || typeof record.uuid === "string"
    || typeof record.permissionMode === "string"
    || typeof record.title === "string"
    || isObject(record.snapshot);
};

export const detectCcSessionV2 = (
  records: readonly unknown[],
  options: SourceDetectionOptions = {},
  recordCountEstimate?: SourceRecordCountEstimate,
): SourceDetectionResult => {
  const inspected = records.slice(0, options.maxInspectionRecords ?? 32);
  const estimate: SourceRecordCountEstimate = recordCountEstimate
    ?? { value: records.length, kind: "exact" };
  if (inspected.length === 0) {
    return {
      ok: false,
      diagnostics: [{ code: "empty_source", message: "Empty CC sample." }],
      inspectedRecordCount: 0,
      ...(options.path != null ? { path: options.path } : {}),
      recordCountEstimate: estimate,
    };
  }
  if (!inspected.every((r) => isObject(r))) {
    return {
      ok: false,
      diagnostics: [{ code: "non_object_record", message: "Non-object record." }],
      inspectedRecordCount: inspected.length,
      ...(options.path != null ? { path: options.path } : {}),
      recordCountEstimate: estimate,
    };
  }
  if (!inspected.every(isCcSessionV2Record)) {
    return {
      ok: false,
      diagnostics: [{ code: "unknown_source_shape", message: "Records do not match cc-session-jsonl-v2 shape." }],
      inspectedRecordCount: inspected.length,
      ...(options.path != null ? { path: options.path } : {}),
      recordCountEstimate: estimate,
    };
  }
  const objs = inspected as JsonObject[];
  const sessionId = firstString(objs, [["sessionId"]]);
  const schemaVersion = firstString(objs, [["version"], ["message", "model"]]);
  const meta: SourceMetadata = {
    sourceType: CC_SESSION_V2_TYPE,
    inspectedRecordCount: inspected.length,
    detectedSignals: ["cc-transcript-type-envelope", "cc-uuid-lineage"],
    recordCountEstimate: estimate,
    ...(options.path != null ? { path: options.path } : {}),
    ...(sessionId != null ? { sessionId } : {}),
    ...(schemaVersion != null ? { schemaVersion } : {}),
  };
  const diagnostics: SourceDetectionDiagnostic[] = [];
  return { ok: true, source: meta, diagnostics };
};

export type CanonicalizeCcSessionV2Input = {
  source: SourceMetadata;
  records: readonly unknown[];
  recordIndexOffset?: number;
  lineNumbers?: readonly (number | undefined)[];
  redactionOptions?: Omit<SourceAdapterRedactionOptions, "source">;
};

export const canonicalizeCcSessionV2 = (input: CanonicalizeCcSessionV2Input): CanonicalSourceAdapterOutput => {
  const out: CanonicalSourceRecord[] = [];
  const diagnostics: SourceAdapterCanonicalDiagnostic[] = [];

  for (let index = 0; index < input.records.length; index += 1) {
    const recordIndex = (input.recordIndexOffset ?? 0) + index;
    const line = input.lineNumbers?.[index];
    const record = input.records[index];
    if (!isObject(record)) {
      diagnostics.push({
        code: "non_object_record",
        message: "Canonical CC v2 requires JSON objects.",
        sourceType: input.source.sourceType,
        recordIndex,
        ...(line === undefined ? {} : { line }),
      });
      continue;
    }
    const redacted = redactSourceRecord(record, { ...input.redactionOptions, source: input.source });
    const redactedObj = asObject(redacted.record);
    if (redactedObj == null) {
      diagnostics.push({
        code: "non_object_record",
        message: "Redacted CC record was not an object.",
        sourceType: input.source.sourceType,
        recordIndex,
        ...(line === undefined ? {} : { line }),
      });
      continue;
    }

    const events = ccEventsFromRecord(redactedObj);
    if (events.length === 0) {
      const recordType = stringValue(redactedObj.type);
      const diag: SourceAdapterCanonicalDiagnostic = {
        code: "unsupported_record",
        message: `No canonical mapping for cc-session-jsonl-v2 record.`,
        sourceType: input.source.sourceType,
        recordIndex,
        ...(line === undefined ? {} : { line }),
      };
      if (recordType != null) diag.recordType = recordType;
      diagnostics.push(diag);
      continue;
    }
    for (let evIdx = 0; evIdx < events.length; evIdx += 1) {
      const ev = events[evIdx]!;
      const lineage = ccLineage(redacted.lineage, redactedObj, ev);
      out.push({
        kind: "span",
        source: input.source,
        lineage,
        redaction: redacted.redaction,
        span: spanFromEvent({
          source: input.source,
          redacted,
          event: ev,
          lineage,
          recordIndex,
          ...(line === undefined ? {} : { line }),
          subEventIndex: evIdx,
        }),
      });
    }
  }

  return { records: out, diagnostics };
};

// Map a CC raw record into 0..N canonical events. A single assistant record can
// hold thinking + text + multiple tool_use; user records can hold tool_results.
const ccEventsFromRecord = (record: JsonObject): CcSourceEvent[] => {
  const t = stringValue(record.type);
  const ts = stringValue(record.timestamp);
  if (t == null) return [];
  const sidechain = record.isSidechain === true;

  if (t === "user") {
    const message = asObject(record.message);
    const content = message?.content;
    const events: CcSourceEvent[] = [];
    if (typeof content === "string") {
      events.push({
        eventKind: sidechain ? "subagent_user_message" : "user_message",
        name: "source.cc-session-jsonl-v2.user_message",
        observationKind: "AGENT",
        timestamp: ts,
        attributes: cleanAttributes({
          "message.role": "user",
          "input.value": content,
          "source.cc.is_sidechain": sidechain,
          "source.cc.uuid": record.uuid,
          "source.cc.parent_uuid": record.parentUuid,
          "source.cc.cwd": record.cwd,
          "source.cc.permission_mode": record.permissionMode,
        }),
      });
    } else if (Array.isArray(content)) {
      for (const item of content) {
        if (!isObject(item)) continue;
        const itemType = stringValue(item.type);
        if (itemType === "tool_result") {
          const isErr = item.is_error === true;
          events.push({
            eventKind: sidechain ? "subagent_tool_result" : "tool_result",
            name: "source.cc-session-jsonl-v2.tool_result",
            observationKind: "TOOL",
            statusCode: isErr ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK",
            statusMessage: isErr ? firstLine(item.content) : "",
            timestamp: ts,
            attributes: cleanAttributes({
              "tool.call_id": item.tool_use_id,
              "tool.is_error": isErr,
              "output.value": item.content,
              "source.cc.is_sidechain": sidechain,
              "source.cc.uuid": record.uuid,
              "source.cc.parent_uuid": record.parentUuid,
            }),
          });
        } else if (itemType === "text") {
          events.push({
            eventKind: sidechain ? "subagent_user_message" : "user_message",
            name: "source.cc-session-jsonl-v2.user_message",
            observationKind: "AGENT",
            timestamp: ts,
            attributes: cleanAttributes({
              "message.role": "user",
              "input.value": item.text,
              "source.cc.is_sidechain": sidechain,
              "source.cc.uuid": record.uuid,
              "source.cc.parent_uuid": record.parentUuid,
            }),
          });
        }
      }
    }
    if (events.length === 0) {
      events.push({
        eventKind: sidechain ? "subagent_user_message" : "user_message",
        name: "source.cc-session-jsonl-v2.user_message",
        observationKind: "AGENT",
        timestamp: ts,
        attributes: cleanAttributes({
          "message.role": "user",
          "input.value": content,
          "source.cc.uuid": record.uuid,
          "source.cc.parent_uuid": record.parentUuid,
        }),
      });
    }
    return events;
  }

  if (t === "assistant") {
    const message = asObject(record.message);
    const content = message?.content;
    const events: CcSourceEvent[] = [];
    const model = stringValue(message?.model);
    if (typeof content === "string") {
      events.push({
        eventKind: sidechain ? "subagent_assistant_message" : "assistant_message",
        name: "source.cc-session-jsonl-v2.assistant_message",
        observationKind: "LLM",
        timestamp: ts,
        attributes: cleanAttributes({
          "message.role": "assistant",
          "output.value": content,
          "inference.llm.model_name": model,
          "source.cc.is_sidechain": sidechain,
          "source.cc.uuid": record.uuid,
          "source.cc.parent_uuid": record.parentUuid,
        }),
      });
    } else if (Array.isArray(content)) {
      for (const item of content) {
        if (!isObject(item)) continue;
        const itemType = stringValue(item.type);
        if (itemType === "text") {
          events.push({
            eventKind: sidechain ? "subagent_assistant_message" : "assistant_message",
            name: "source.cc-session-jsonl-v2.assistant_message",
            observationKind: "LLM",
            timestamp: ts,
            attributes: cleanAttributes({
              "message.role": "assistant",
              "output.value": item.text,
              "inference.llm.model_name": model,
              "source.cc.is_sidechain": sidechain,
              "source.cc.uuid": record.uuid,
              "source.cc.parent_uuid": record.parentUuid,
            }),
          });
        } else if (itemType === "thinking") {
          events.push({
            eventKind: sidechain ? "subagent_assistant_thinking" : "assistant_thinking",
            name: "source.cc-session-jsonl-v2.assistant_thinking",
            observationKind: "LLM",
            timestamp: ts,
            attributes: cleanAttributes({
              "message.role": "assistant",
              "source.cc.thinking": item.thinking,
              "inference.llm.model_name": model,
              "source.cc.is_sidechain": sidechain,
              "source.cc.uuid": record.uuid,
              "source.cc.parent_uuid": record.parentUuid,
            }),
          });
        } else if (itemType === "tool_use") {
          events.push({
            eventKind: sidechain ? "subagent_tool_call" : "tool_call",
            name: "source.cc-session-jsonl-v2.tool_call",
            observationKind: "TOOL",
            timestamp: ts,
            attributes: cleanAttributes({
              "tool.name": item.name,
              "tool.call_id": item.id,
              "input.value": item.input,
              "inference.llm.model_name": model,
              "source.cc.is_sidechain": sidechain,
              "source.cc.uuid": record.uuid,
              "source.cc.parent_uuid": record.parentUuid,
            }),
          });
        }
      }
    }
    return events;
  }

  if (t === "attachment") {
    const att = asObject(record.attachment);
    const subType = stringValue(att?.type) ?? "unknown";
    return [{
      eventKind: `attachment_${subType}`,
      name: `source.cc-session-jsonl-v2.attachment.${subType}`,
      observationKind: "CHAIN",
      timestamp: ts,
      attributes: cleanAttributes({
        "source.cc.attachment_type": subType,
        "output.value": att,
        "source.cc.uuid": record.uuid,
        "source.cc.parent_uuid": record.parentUuid,
      }),
    }];
  }

  if (t === "system") {
    return [{
      eventKind: "system_message",
      name: "source.cc-session-jsonl-v2.system",
      observationKind: "CHAIN",
      timestamp: ts,
      attributes: cleanAttributes({
        "message.role": "system",
        "output.value": record.content ?? record.message ?? record.text ?? record.subtype,
        "source.cc.subtype": record.subtype,
        "source.cc.uuid": record.uuid,
      }),
    }];
  }

  if (t === "permission-mode") {
    return [{
      eventKind: "permission_mode",
      name: "source.cc-session-jsonl-v2.permission_mode",
      observationKind: "CHAIN",
      timestamp: ts,
      attributes: cleanAttributes({
        "source.cc.permission_mode": record.permissionMode,
      }),
    }];
  }

  if (t === "last-prompt") {
    return [{
      eventKind: "last_prompt",
      name: "source.cc-session-jsonl-v2.last_prompt",
      observationKind: "CHAIN",
      timestamp: ts,
      attributes: cleanAttributes({
        "source.cc.prompt": record.prompt ?? record.lastPrompt ?? record.text,
        "source.cc.uuid": record.uuid,
      }),
    }];
  }

  if (t === "ai-title") {
    return [{
      eventKind: "ai_title",
      name: "source.cc-session-jsonl-v2.ai_title",
      observationKind: "CHAIN",
      timestamp: ts,
      attributes: cleanAttributes({
        "source.cc.title": record.title ?? record.aiTitle,
      }),
    }];
  }

  if (t === "queue-operation") {
    return [{
      eventKind: "queue_operation",
      name: "source.cc-session-jsonl-v2.queue_operation",
      observationKind: "CHAIN",
      timestamp: ts,
      attributes: cleanAttributes({
        "source.cc.queue_operation": record.operation ?? record.kind,
        "output.value": record,
      }),
    }];
  }

  if (t === "file-history-snapshot") {
    const snapshot = asObject(record.snapshot);
    return [{
      eventKind: "file_history_snapshot",
      name: "source.cc-session-jsonl-v2.file_history_snapshot",
      observationKind: "CHAIN",
      timestamp: ts ?? stringValue(snapshot?.timestamp),
      attributes: cleanAttributes({
        "source.cc.message_id": record.messageId ?? snapshot?.messageId,
        "source.cc.is_snapshot_update": record.isSnapshotUpdate,
        "source.cc.snapshot_file_count": Object.keys(asObject(snapshot?.trackedFileBackups) ?? {}).length,
      }),
    }];
  }

  if (t === "summary") {
    return [{
      eventKind: "summary",
      name: "source.cc-session-jsonl-v2.summary",
      observationKind: "CHAIN",
      timestamp: ts,
      attributes: cleanAttributes({
        "output.value": record.summary ?? record.content ?? record.text,
      }),
    }];
  }

  if (t === "error") {
    const error = record.error ?? record.message ?? record.content ?? record.text;
    return [{
      eventKind: "error",
      name: "source.cc-session-jsonl-v2.error",
      observationKind: "CHAIN",
      statusCode: "STATUS_CODE_ERROR",
      statusMessage: firstLine(error),
      timestamp: ts,
      attributes: cleanAttributes({
        "error.message": error,
        "error.type": record.errorType ?? record.subtype ?? record.code,
        "output.value": record,
        "source.cc.uuid": record.uuid,
        "source.cc.parent_uuid": record.parentUuid,
      }),
    }];
  }

  return [];
};

const ccLineage = (
  base: SourceRecordLineage,
  record: JsonObject,
  event: CcSourceEvent,
): SourceRecordLineage => {
  const id = stringValue(record.uuid) ?? base.id;
  const parentId = stringValue(record.parentUuid) ?? base.parentId;
  const sessionId = stringValue(record.sessionId) ?? base.sessionId;
  const toolCallId = stringValue(event.attributes["tool.call_id"]);
  const toolName = stringValue(event.attributes["tool.name"]);
  return {
    ...base,
    ...(id != null ? { id } : {}),
    ...(parentId != null ? { parentId } : {}),
    ...(sessionId != null ? { sessionId } : {}),
    toolCallIds: toolCallId == null ? base.toolCallIds : uniqueSorted([...base.toolCallIds, toolCallId]),
    toolNames: toolName == null ? base.toolNames : uniqueSorted([...base.toolNames, toolName]),
  };
};

const spanFromEvent = (input: {
  source: SourceMetadata;
  redacted: RedactedSourceRecord;
  event: CcSourceEvent;
  lineage: SourceRecordLineage;
  recordIndex?: number;
  line?: number;
  subEventIndex?: number;
}): HaloSpan => {
  const ts = input.event.timestamp ?? DEFAULT_TIMESTAMP;
  const traceId = input.lineage.traceId
    ?? stableId("trace", input.source.sourceType, input.source.sessionId, input.source.path);
  const spanId = stableId(
    "span",
    input.source.sourceType,
    input.source.sessionId,
    input.lineage.id,
    input.recordIndex,
    input.subEventIndex,
    input.event.eventKind,
  );
  const parentSpanId = input.lineage.parentId == null
    ? ""
    : stableId("span", input.source.sourceType, input.source.sessionId, input.lineage.parentId);
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
    start_time: ts,
    end_time: ts,
    status: {
      code: input.event.statusCode ?? classification?.statusCode ?? "STATUS_CODE_OK",
      message: input.event.statusMessage ?? classification?.statusMessage ?? "",
    },
    resource: {
      attributes: cleanAttributes({
        "service.name": "bleeding-agent-source-adapter",
        "telemetry.sdk.language": "typescript",
        "source.adapter.type": input.source.sourceType,
        "source.adapter.path": input.source.path,
        "source.adapter.session_id": input.source.sessionId,
        "source.adapter.schema_version": input.source.schemaVersion,
      }),
    },
    scope: { name: CANONICAL_SCOPE_NAME, version: CANONICAL_SCOPE_VERSION },
    attributes: cleanAttributes({
      "openinference.span.kind": input.event.observationKind,
      "inference.observation_kind": input.event.observationKind,
      "inference.export.schema_version": 1,
      "source.adapter.event_kind": input.event.eventKind,
      "source.adapter.type": input.source.sourceType,
      "source.adapter.path": input.source.path,
      "source.adapter.session_id": input.source.sessionId,
      "source.adapter.schema_version": input.source.schemaVersion,
      "source.adapter.detected_signals": input.source.detectedSignals,
      "source.record.index": input.recordIndex,
      "source.record.line": input.line,
      "source.record.sub_event_index": input.subEventIndex,
      "source.lineage.id": input.lineage.id,
      "source.lineage.parent_id": input.lineage.parentId,
      "source.lineage.session_id": input.lineage.sessionId,
      "source.lineage.tool_call_ids": input.lineage.toolCallIds,
      "source.lineage.tool_names": input.lineage.toolNames,
      "source.redaction.secret_replacement_count": input.redacted.redaction.secretReplacementCount,
      "source.redaction.redaction_kinds": input.redacted.redaction.redactionKinds,
      "source.redaction.truncated_string_count": input.redacted.redaction.truncatedStringCount,
      "source.redaction.truncated_array_count": input.redacted.redaction.truncatedArrayCount,
      "source.redaction.truncated_depth_count": input.redacted.redaction.truncatedDepthCount,
      "source.redaction.full_content_included": input.redacted.redaction.fullContentIncluded,
      "source.redaction.secret_redaction_disabled": input.redacted.redaction.secretRedactionDisabled,
      ...input.event.attributes,
      ...sourceAdapterFailureAttributes(classification),
      "source.record.redacted": input.redacted.record,
    }),
  };
};

const cleanAttributes = (attributes: Record<string, unknown>): Record<string, unknown> => {
  const clean: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(attributes)) {
    if (v !== undefined) clean[k] = v;
  }
  return clean;
};

const firstString = (records: readonly JsonObject[], paths: readonly string[][]): string | undefined => {
  for (const record of records) {
    for (const path of paths) {
      let cur: unknown = record;
      for (const seg of path) {
        if (!isObject(cur)) { cur = undefined; break; }
        cur = (cur as JsonObject)[seg];
      }
      if (typeof cur === "string" && cur.length > 0) return cur;
    }
  }
  return undefined;
};

const stableId = (...parts: readonly unknown[]): string =>
  createHash("sha256")
    .update(parts.map((p) => p == null ? "" : String(p)).join("\0"))
    .digest("hex")
    .slice(0, 32);

const stringValue = (v: unknown): string | undefined =>
  typeof v === "string" && v.length > 0 ? v : undefined;

const asObject = (v: unknown): JsonObject | undefined => isObject(v) ? v : undefined;

const isObject = (v: unknown): v is JsonObject =>
  typeof v === "object" && v != null && !Array.isArray(v);

const uniqueSorted = (xs: readonly string[]): string[] => [...new Set(xs)].sort();

const firstLine = (value: unknown): string => {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return (text ?? "").split(/\r?\n/, 1)[0] ?? "";
};
