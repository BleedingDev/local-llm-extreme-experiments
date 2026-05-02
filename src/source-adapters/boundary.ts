import { isCcSessionV2Record } from "./cc-session-v2";

export type SourceAdapterType =
  | "spans-jsonl"
  | "acp-session-jsonl"
  | "codex-session-jsonl"
  | "pi-session-jsonl"
  | "cc-session-jsonl-v2";

export type SourceDetectionDiagnosticCode =
  | "empty_source"
  | "malformed_jsonl"
  | "non_object_record"
  | "unknown_source_shape"
  | "ambiguous_source_shape";

export type SourceDetectionDiagnostic = {
  code: SourceDetectionDiagnosticCode;
  message: string;
  line?: number;
  recordIndex?: number;
};

export type SourceRecordCountEstimate = {
  value: number;
  kind: "exact" | "sample";
};

export type SourceMetadata = {
  sourceType: SourceAdapterType;
  path?: string;
  sessionId?: string;
  schemaVersion?: string;
  recordCountEstimate?: SourceRecordCountEstimate;
  inspectedRecordCount: number;
  detectedSignals: string[];
};

export type SupportedSourceDetection = {
  ok: true;
  source: SourceMetadata;
  diagnostics: SourceDetectionDiagnostic[];
};

export type UnsupportedSourceDetection = {
  ok: false;
  diagnostics: SourceDetectionDiagnostic[];
  path?: string;
  recordCountEstimate?: SourceRecordCountEstimate;
  inspectedRecordCount: number;
};

export type SourceDetectionResult = SupportedSourceDetection | UnsupportedSourceDetection;

export type SourceDetectionOptions = {
  path?: string;
  maxInspectionRecords?: number;
};

export type SourceAdapterBoundary = {
  sourceType: SourceAdapterType;
  detect(records: readonly unknown[], options?: SourceDetectionOptions): SourceDetectionResult;
};

type JsonObject = Record<string, unknown>;

const DEFAULT_MAX_INSPECTION_RECORDS = 32;

const ACP_METHOD_PREFIXES = ["session/", "fs/", "terminal/"] as const;
const ACP_METHODS = new Set(["initialize", "shutdown"]);
const CODEX_RECORD_TYPES = new Set(["session_meta", "turn_context", "response_item", "event_msg"]);
const PI_RECORD_TYPES = new Set([
  "session",
  "user",
  "assistant",
  "tool_use",
  "tool_result",
  "system",
  "custom",
  "model_change",
  "compaction",
  "compaction_start",
  "compaction_end",
]);

export const sourceAdapters: readonly SourceAdapterBoundary[] = [
  { sourceType: "spans-jsonl", detect: (records, options) => detectSourceRecords(records, options, "spans-jsonl") },
  { sourceType: "acp-session-jsonl", detect: (records, options) => detectSourceRecords(records, options, "acp-session-jsonl") },
  { sourceType: "codex-session-jsonl", detect: (records, options) => detectSourceRecords(records, options, "codex-session-jsonl") },
  { sourceType: "pi-session-jsonl", detect: (records, options) => detectSourceRecords(records, options, "pi-session-jsonl") },
  { sourceType: "cc-session-jsonl-v2", detect: (records, options) => detectSourceRecords(records, options, "cc-session-jsonl-v2") },
];

export const detectSourceJsonl = (jsonl: string, options: SourceDetectionOptions = {}): SourceDetectionResult => {
  const parsed = parseJsonlSample(jsonl, options);
  if (parsed.diagnostics.length > 0) {
    return unsupported(parsed.diagnostics, options, parsed.recordCountEstimate, parsed.records.length);
  }
  return detectSourceRecords(parsed.records, {
    ...options,
    maxInspectionRecords: parsed.records.length,
  }, undefined, parsed.recordCountEstimate);
};

export const detectSourceRecords = (
  records: readonly unknown[],
  options: SourceDetectionOptions = {},
  requiredSourceType?: SourceAdapterType,
  recordCountEstimate: SourceRecordCountEstimate = { value: records.length, kind: "exact" },
): SourceDetectionResult => {
  const maxInspectionRecords = options.maxInspectionRecords ?? DEFAULT_MAX_INSPECTION_RECORDS;
  const inspected = records.slice(0, maxInspectionRecords);
  if (inspected.length === 0) {
    return unsupported([
      {
        code: "empty_source",
        message: "Source detection requires at least one non-blank JSONL record.",
      },
    ], options, recordCountEstimate, 0);
  }

  const nonObjectIndex = inspected.findIndex((record) => !isObject(record));
  if (nonObjectIndex >= 0) {
    return unsupported([
      {
        code: "non_object_record",
        message: "Source detection only accepts JSON object records.",
        recordIndex: nonObjectIndex,
      },
    ], options, recordCountEstimate, inspected.length);
  }

  const objects = inspected as JsonObject[];
  const matches = sourceDetectors
    .map((detector) => detector(objects))
    .filter((match): match is SourceDetectorMatch => match != null)
    .filter((match) => requiredSourceType == null || match.sourceType === requiredSourceType);

  if (matches.length === 1) {
    const match = matches[0];
    if (match == null) {
      return unsupported([
        {
          code: "unknown_source_shape",
          message: "Source detector produced no match.",
        },
      ], options, recordCountEstimate, inspected.length);
    }
    return {
      ok: true,
      source: metadata(match, options, recordCountEstimate, inspected.length),
      diagnostics: [],
    };
  }

  if (matches.length > 1) {
    return unsupported([
      {
        code: "ambiguous_source_shape",
        message: `Source matched multiple adapter boundaries: ${matches.map((match) => match.sourceType).join(", ")}.`,
      },
    ], options, recordCountEstimate, inspected.length);
  }

  const requested = requiredSourceType == null ? "known source adapter" : requiredSourceType;
  return unsupported([
    {
      code: "unknown_source_shape",
      message: `JSONL records do not match the explicit boundary for ${requested}.`,
    },
  ], options, recordCountEstimate, inspected.length);
};

type ParsedSample = {
  records: unknown[];
  diagnostics: SourceDetectionDiagnostic[];
  recordCountEstimate: SourceRecordCountEstimate;
};

const parseJsonlSample = (jsonl: string, options: SourceDetectionOptions): ParsedSample => {
  const maxInspectionRecords = options.maxInspectionRecords ?? DEFAULT_MAX_INSPECTION_RECORDS;
  const records: unknown[] = [];
  const diagnostics: SourceDetectionDiagnostic[] = [];
  const lines = jsonl.split(/\r?\n/);
  let nonBlankLineCount = 0;

  for (const [index, line] of lines.entries()) {
    if (line.trim() === "") {
      continue;
    }
    nonBlankLineCount += 1;
    if (records.length >= maxInspectionRecords) {
      continue;
    }
    try {
      records.push(JSON.parse(line) as unknown);
    } catch (error) {
      diagnostics.push({
        code: "malformed_jsonl",
        message: `Malformed JSONL at line ${index + 1}: ${error instanceof Error ? error.message : String(error)}`,
        line: index + 1,
      });
      break;
    }
  }

  return {
    records,
    diagnostics,
    recordCountEstimate: {
      value: nonBlankLineCount,
      kind: records.length === nonBlankLineCount ? "exact" : "sample",
    },
  };
};

type SourceDetectorMatch = {
  sourceType: SourceAdapterType;
  sessionId?: string | undefined;
  schemaVersion?: string | undefined;
  detectedSignals: string[];
};

const sourceDetectors: ReadonlyArray<(records: readonly JsonObject[]) => SourceDetectorMatch | undefined> = [
  (records) => {
    if (!records.every(isHaloSpanRecord)) {
      return undefined;
    }
    const first = records[0];
    return {
      sourceType: "spans-jsonl",
      sessionId: stringValue(first?.trace_id),
      schemaVersion: spanSchemaVersion(first),
      detectedSignals: ["halo-span-required-fields", "openinference-status-resource-scope"],
    };
  },
  (records) => {
    const acpRecords = records.filter(isAcpSessionRecord);
    if (acpRecords.length === 0 || acpRecords.length !== records.length) {
      return undefined;
    }
    return {
      sourceType: "acp-session-jsonl",
      sessionId: firstString(records, [
        ["sessionId"],
        ["params", "sessionId"],
        ["params", "session_id"],
        ["message", "params", "sessionId"],
      ]),
      schemaVersion: firstString(records, [["protocolVersion"], ["version"], ["params", "protocolVersion"]]),
      detectedSignals: ["acp-jsonrpc-method-or-session-update", "acp-session-lineage"],
    };
  },
  (records) => {
    const meta = records.find(isCodexSessionMeta);
    if (meta == null || !records.every(isCodexSessionRecord)) {
      return undefined;
    }
    return {
      sourceType: "codex-session-jsonl",
      sessionId: stringValue(objectValue(meta.payload)?.id),
      schemaVersion: firstString(records, [["payload", "cli_version"]]),
      detectedSignals: ["codex-session-meta", "codex-record-type-envelope"],
    };
  },
  (records) => {
    const header = records.find(isPiSessionHeader);
    if (header == null || !records.every(isPiSessionRecord)) {
      return undefined;
    }
    return {
      sourceType: "pi-session-jsonl",
      sessionId: stringValue(header.id),
      schemaVersion: numberOrStringValue(header.version),
      detectedSignals: ["pi-session-header", "pi-tree-entry-envelope"],
    };
  },
  (records) => {
    if (!records.every(isCcSessionV2Record)) {
      return undefined;
    }
    return {
      sourceType: "cc-session-jsonl-v2",
      sessionId: firstString(records, [["sessionId"]]),
      schemaVersion: firstString(records, [["version"], ["message", "model"]]),
      detectedSignals: ["cc-transcript-type-envelope", "cc-uuid-lineage"],
    };
  },
];

const metadata = (
  match: SourceDetectorMatch,
  options: SourceDetectionOptions,
  recordCountEstimate: SourceRecordCountEstimate,
  inspectedRecordCount: number,
): SourceMetadata => {
  const source: SourceMetadata = {
    sourceType: match.sourceType,
    recordCountEstimate,
    inspectedRecordCount,
    detectedSignals: match.detectedSignals,
  };
  if (options.path != null) {
    source.path = options.path;
  }
  if (match.sessionId != null) {
    source.sessionId = match.sessionId;
  }
  if (match.schemaVersion != null) {
    source.schemaVersion = match.schemaVersion;
  }
  return source;
};

const unsupported = (
  diagnostics: SourceDetectionDiagnostic[],
  options: SourceDetectionOptions,
  recordCountEstimate: SourceRecordCountEstimate | undefined,
  inspectedRecordCount: number,
): UnsupportedSourceDetection => {
  const result: UnsupportedSourceDetection = {
    ok: false,
    diagnostics,
    inspectedRecordCount,
  };
  if (options.path != null) {
    result.path = options.path;
  }
  if (recordCountEstimate != null) {
    result.recordCountEstimate = recordCountEstimate;
  }
  return result;
};

const isHaloSpanRecord = (record: JsonObject): boolean =>
  typeof record.trace_id === "string"
  && typeof record.span_id === "string"
  && typeof record.parent_span_id === "string"
  && typeof record.trace_state === "string"
  && typeof record.name === "string"
  && typeof record.start_time === "string"
  && typeof record.end_time === "string"
  && isObject(record.status)
  && typeof record.status.code === "string"
  && isObject(record.resource)
  && isObject(record.scope)
  && isObject(record.attributes);

const isAcpSessionRecord = (record: JsonObject): boolean => {
  const candidate = isObject(record.message) ? record.message : record;
  const method = stringValue(candidate.method);
  const params = objectValue(candidate.params);
  const update = objectValue(params?.update);
  return (
    candidate.jsonrpc === "2.0"
    && method != null
    && (ACP_METHODS.has(method) || ACP_METHOD_PREFIXES.some((prefix) => method.startsWith(prefix)))
  ) || (
    stringValue(record.sessionId) != null
    && stringValue(record.sessionUpdate) != null
  ) || (
    stringValue(params?.sessionId) != null
    && stringValue(update?.sessionUpdate) != null
  );
};

const isCodexSessionMeta = (record: JsonObject): boolean => {
  const payload = objectValue(record.payload);
  return record.type === "session_meta"
    && typeof payload?.id === "string"
    && typeof payload.cwd === "string"
    && typeof payload.cli_version === "string"
    && (payload.source === "codex" || typeof payload.model_provider === "string" || typeof payload.originator === "string");
};

const isCodexSessionRecord = (record: JsonObject): boolean => {
  if (typeof record.type !== "string" || !CODEX_RECORD_TYPES.has(record.type)) {
    return false;
  }
  if (record.type === "session_meta") {
    return isCodexSessionMeta(record);
  }
  return isObject(record.payload);
};

const isPiSessionHeader = (record: JsonObject): boolean =>
  record.type === "session"
  && (record.version === 1 || record.version === 2 || record.version === 3)
  && typeof record.id === "string"
  && typeof record.timestamp === "string"
  && typeof record.cwd === "string";

const isPiSessionRecord = (record: JsonObject): boolean => {
  if (isPiSessionHeader(record)) {
    return true;
  }
  if (typeof record.type !== "string" || !PI_RECORD_TYPES.has(record.type)) {
    return false;
  }
  const hasTreeId = typeof record.id === "string";
  const hasParent = record.parentId == null || typeof record.parentId === "string";
  if (!hasTreeId || !hasParent) {
    return false;
  }

  if (record.type === "user" || record.type === "assistant" || record.type === "system" || record.type === "custom") {
    return Array.isArray(record.content) || isObject(record.message) || typeof record.role === "string";
  }

  if (record.type === "tool_use") {
    return typeof record.name === "string"
      || isObject(record.function)
      || record.input != null
      || record.arguments != null
      || typeof record.tool_call_id === "string"
      || typeof record.call_id === "string";
  }

  if (record.type === "tool_result") {
    return typeof record.tool_call_id === "string"
      || typeof record.call_id === "string"
      || record.output != null
      || record.content != null
      || typeof record.error === "string";
  }

  if (record.type === "model_change") {
    return typeof record.model === "string"
      || typeof record.to === "string"
      || typeof record.previousModel === "string"
      || typeof record.from === "string";
  }

  if (record.type === "compaction" || record.type === "compaction_start" || record.type === "compaction_end") {
    return record.before != null || record.after != null || record.summary != null || record.content != null;
  }

  return false;
};

const spanSchemaVersion = (record: JsonObject | undefined): string | undefined => {
  const scope = objectValue(record?.scope);
  const attributes = objectValue(record?.attributes);
  return firstPresent([
    numberOrStringValue(attributes?.["telemetry.schema_version"]),
    numberOrStringValue(attributes?.["schema.version"]),
    scope == null ? undefined : `${stringValue(scope.name) ?? "unknown"}@${stringValue(scope.version) ?? "unknown"}`,
  ]);
};

const firstString = (records: readonly JsonObject[], paths: readonly string[][]): string | undefined => {
  for (const record of records) {
    for (const path of paths) {
      const value = valueAtPath(record, path);
      if (typeof value === "string" && value.length > 0) {
        return value;
      }
      if (typeof value === "number") {
        return String(value);
      }
    }
  }
  return undefined;
};

const valueAtPath = (record: JsonObject, path: readonly string[]): unknown => {
  let current: unknown = record;
  for (const segment of path) {
    if (!isObject(current)) {
      return undefined;
    }
    current = current[segment];
  }
  return current;
};

const firstPresent = (values: ReadonlyArray<string | undefined>): string | undefined =>
  values.find((value) => value != null && value.length > 0);

const numberOrStringValue = (value: unknown): string | undefined => {
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  if (typeof value === "number") {
    return String(value);
  }
  return undefined;
};

const stringValue = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const objectValue = (value: unknown): JsonObject | undefined => isObject(value) ? value : undefined;

const isObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value != null && !Array.isArray(value);
