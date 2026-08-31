import type { SourceMetadata } from "./boundary";

export type SourceAdapterRedactionKind =
  | "authorization"
  | "aws_access_key"
  | "github_token"
  | "openai_api_key"
  | "private_key"
  | "secret_field"
  | "slack_token";

export type SourceAdapterRedactionOptions = {
  source?: SourceMetadata;
  maxTextExcerptChars?: number;
  maxArrayItems?: number;
  maxObjectDepth?: number;
  includeFullContent?: boolean;
  dangerouslyDisableSecretRedaction?: boolean;
};

export type SourceAdapterRedactionMetadata = {
  secretReplacementCount: number;
  redactionKinds: SourceAdapterRedactionKind[];
  truncatedStringCount: number;
  truncatedArrayCount: number;
  truncatedDepthCount: number;
  fullContentIncluded: boolean;
  secretRedactionDisabled: boolean;
};

export type SourceRecordLineage = {
  source?: SourceMetadata;
  id?: string;
  parentId?: string;
  sessionId?: string;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  role?: string;
  toolCallIds: string[];
  toolNames: string[];
};

export type RedactedSourceRecord = {
  record: unknown;
  lineage: SourceRecordLineage;
  redaction: SourceAdapterRedactionMetadata;
};

type JsonObject = Record<string, unknown>;

type RedactionAccumulator = {
  secretReplacementCount: number;
  redactionKinds: Set<SourceAdapterRedactionKind>;
  truncatedStringCount: number;
  truncatedArrayCount: number;
  truncatedDepthCount: number;
};

type SecretPattern = {
  kind: SourceAdapterRedactionKind;
  pattern: RegExp;
};

const DEFAULT_MAX_TEXT_EXCERPT_CHARS = 240;
const DEFAULT_MAX_ARRAY_ITEMS = 64;
const DEFAULT_MAX_OBJECT_DEPTH = 24;

const SECRET_PATTERNS: readonly SecretPattern[] = [
  { kind: "private_key", pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g },
  { kind: "authorization", pattern: /\bBearer\s+[A-Za-z0-9._~+/=-]{12,}/g },
  { kind: "openai_api_key", pattern: /\bsk-[A-Za-z0-9_-]{12,}\b/g },
  { kind: "github_token", pattern: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b/g },
  { kind: "github_token", pattern: /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g },
  { kind: "aws_access_key", pattern: /\bAKIA[0-9A-Z]{16}\b/g },
  { kind: "slack_token", pattern: /\bxox[baprs]-[A-Za-z0-9-]{20,}\b/g },
];

const SECRET_FIELD_PATTERN =
  /(^|[_\-.])(api[_\-.]?key|auth(?:orization)?|credential|password|passwd|private[_\-.]?key|secret|token)([_\-.]|$)/i;

const STRUCTURAL_STRING_FIELDS = new Set([
  "id",
  "method",
  "name",
  "parentId",
  "parent_span_id",
  "role",
  "schemaVersion",
  "sessionId",
  "session_id",
  "sourceType",
  "span_id",
  "trace_id",
  "type",
]);

export const withFullSourceRecordContent = (
  options: Omit<SourceAdapterRedactionOptions, "includeFullContent" | "dangerouslyDisableSecretRedaction"> = {},
): SourceAdapterRedactionOptions => ({
  ...options,
  includeFullContent: true,
});

export const withDangerousUnredactedSourceRecordContent = (
  options: Omit<SourceAdapterRedactionOptions, "includeFullContent" | "dangerouslyDisableSecretRedaction"> = {},
): SourceAdapterRedactionOptions => ({
  ...options,
  includeFullContent: true,
  dangerouslyDisableSecretRedaction: true,
});

export const redactSourceRecords = (
  records: readonly unknown[],
  options: SourceAdapterRedactionOptions = {},
): RedactedSourceRecord[] =>
  records.map((record) => redactSourceRecord(record, options));

export const redactSourceRecord = (
  record: unknown,
  options: SourceAdapterRedactionOptions = {},
): RedactedSourceRecord => {
  const accumulator = emptyAccumulator();
  const redactedRecord = redactValue(record, [], options, accumulator, 0);
  return {
    record: redactedRecord,
    lineage: extractSourceRecordLineage(record, options.source),
    redaction: redactionMetadata(accumulator, options),
  };
};

export const extractSourceRecordLineage = (record: unknown, source?: SourceMetadata): SourceRecordLineage => {
  const lineage: SourceRecordLineage = {
    toolCallIds: [],
    toolNames: [],
  };
  if (source != null) {
    lineage.source = source;
  }

  const id = firstStringAt(record, [["id"], ["payload", "id"], ["params", "id"], ["message", "id"]]);
  const parentId = firstStringAt(record, [["parentId"], ["payload", "parentId"], ["params", "parentId"]]);
  const codexSessionMetaId = firstStringAt(record, [["type"]]) === "session_meta"
    ? firstStringAt(record, [["payload", "id"]])
    : undefined;
  const sessionId = firstStringAt(record, [
    ["sessionId"],
    ["session_id"],
    ["params", "sessionId"],
    ["params", "session_id"],
    ["payload", "sessionId"],
  ]) ?? codexSessionMetaId;
  const traceId = firstStringAt(record, [["trace_id"], ["traceId"], ["payload", "trace_id"], ["attributes", "trace_id"]]);
  const spanId = firstStringAt(record, [["span_id"], ["spanId"], ["payload", "span_id"], ["attributes", "span_id"]]);
  const parentSpanId = firstStringAt(record, [["parent_span_id"], ["parentSpanId"], ["payload", "parent_span_id"]]);
  const role = firstStringAt(record, [["role"], ["payload", "role"], ["message", "role"], ["payload", "message", "role"]]);

  if (id != null) {
    lineage.id = id;
  }
  if (parentId != null) {
    lineage.parentId = parentId;
  }
  const resolvedSessionId = sessionId ?? source?.sessionId;
  if (resolvedSessionId != null) {
    lineage.sessionId = resolvedSessionId;
  }
  if (traceId != null) {
    lineage.traceId = traceId;
  }
  if (spanId != null) {
    lineage.spanId = spanId;
  }
  if (parentSpanId != null) {
    lineage.parentSpanId = parentSpanId;
  }
  if (role != null) {
    lineage.role = role;
  }

  collectToolCalls(record, lineage);
  lineage.toolCallIds = uniqueSorted(lineage.toolCallIds);
  lineage.toolNames = uniqueSorted(lineage.toolNames);
  return lineage;
};

const redactValue = (
  value: unknown,
  path: readonly string[],
  options: SourceAdapterRedactionOptions,
  accumulator: RedactionAccumulator,
  depth: number,
): unknown => {
  const maxObjectDepth = options.maxObjectDepth ?? DEFAULT_MAX_OBJECT_DEPTH;
  if (depth > maxObjectDepth) {
    accumulator.truncatedDepthCount += 1;
    return "[TRUNCATED:object_depth]";
  }

  const parentKey = path.at(-1);
  if (typeof value === "string") {
    return redactString(value, parentKey, options, accumulator);
  }
  if (value == null || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    return redactArray(value, path, options, accumulator, depth);
  }
  if (isObject(value)) {
    return redactObject(value, path, options, accumulator, depth);
  }
  return String(value);
};

const redactArray = (
  value: readonly unknown[],
  path: readonly string[],
  options: SourceAdapterRedactionOptions,
  accumulator: RedactionAccumulator,
  depth: number,
): unknown[] => {
  const maxArrayItems = options.maxArrayItems ?? DEFAULT_MAX_ARRAY_ITEMS;
  const visibleItems = value.slice(0, maxArrayItems);
  const redacted = visibleItems.map((item, index) =>
    redactValue(item, [...path, String(index)], options, accumulator, depth + 1));
  if (value.length > visibleItems.length) {
    accumulator.truncatedArrayCount += 1;
    redacted.push({ __truncatedItems: value.length - visibleItems.length });
  }
  return redacted;
};

const redactObject = (
  value: JsonObject,
  path: readonly string[],
  options: SourceAdapterRedactionOptions,
  accumulator: RedactionAccumulator,
  depth: number,
): JsonObject => {
  const output: JsonObject = {};
  for (const [key, nested] of Object.entries(value)) {
    if (isSecretFieldKey(key) && !options.dangerouslyDisableSecretRedaction) {
      output[key] = redactSecretField(nested, accumulator);
      continue;
    }
    output[key] = redactValue(nested, [...path, key], options, accumulator, depth + 1);
  }
  return output;
};

const redactString = (
  value: string,
  key: string | undefined,
  options: SourceAdapterRedactionOptions,
  accumulator: RedactionAccumulator,
): string => {
  let next = value;
  if (!options.dangerouslyDisableSecretRedaction) {
    for (const rule of SECRET_PATTERNS) {
      next = next.replace(rule.pattern, () => {
        accumulator.secretReplacementCount += 1;
        accumulator.redactionKinds.add(rule.kind);
        return `[REDACTED:${rule.kind}]`;
      });
    }
  }

  if (options.includeFullContent || (key != null && STRUCTURAL_STRING_FIELDS.has(key))) {
    return next;
  }

  const maxTextExcerptChars = options.maxTextExcerptChars ?? DEFAULT_MAX_TEXT_EXCERPT_CHARS;
  if (next.length <= maxTextExcerptChars) {
    return next;
  }
  accumulator.truncatedStringCount += 1;
  return `${next.slice(0, maxTextExcerptChars)}...[TRUNCATED:${next.length - maxTextExcerptChars}_chars]`;
};

const redactSecretField = (value: unknown, accumulator: RedactionAccumulator): unknown => {
  accumulator.secretReplacementCount += 1;
  accumulator.redactionKinds.add("secret_field");
  if (Array.isArray(value)) {
    return value.map(() => "[REDACTED:secret_field]");
  }
  if (isObject(value)) {
    return "[REDACTED:secret_field]";
  }
  return "[REDACTED:secret_field]";
};

const redactionMetadata = (
  accumulator: RedactionAccumulator,
  options: SourceAdapterRedactionOptions,
): SourceAdapterRedactionMetadata => ({
  secretReplacementCount: accumulator.secretReplacementCount,
  redactionKinds: [...accumulator.redactionKinds].sort(),
  truncatedStringCount: accumulator.truncatedStringCount,
  truncatedArrayCount: accumulator.truncatedArrayCount,
  truncatedDepthCount: accumulator.truncatedDepthCount,
  fullContentIncluded: options.includeFullContent === true,
  secretRedactionDisabled: options.dangerouslyDisableSecretRedaction === true,
});

const emptyAccumulator = (): RedactionAccumulator => ({
  secretReplacementCount: 0,
  redactionKinds: new Set(),
  truncatedStringCount: 0,
  truncatedArrayCount: 0,
  truncatedDepthCount: 0,
});

const collectToolCalls = (value: unknown, lineage: SourceRecordLineage): void => {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectToolCalls(item, lineage);
    }
    return;
  }
  if (!isObject(value)) {
    return;
  }

  const type = stringValue(value.type);
  const id = stringValue(value.id) ?? stringValue(value.tool_call_id) ?? stringValue(value.call_id);
  const name = stringValue(value.name) ?? stringValue(objectValue(value.function)?.name);
  const looksLikeToolCall = type === "tool_call"
    || type === "function"
    || value.tool_call_id != null
    || value.call_id != null
    || objectValue(value.function) != null;

  if (looksLikeToolCall) {
    if (id != null) {
      lineage.toolCallIds.push(id);
    }
    if (name != null) {
      lineage.toolNames.push(name);
    }
  }

  for (const nested of Object.values(value)) {
    collectToolCalls(nested, lineage);
  }
};

const firstStringAt = (record: unknown, paths: readonly (readonly string[])[]): string | undefined => {
  for (const path of paths) {
    const value = valueAtPath(record, path);
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
    if (typeof value === "number") {
      return String(value);
    }
  }
  return undefined;
};

const valueAtPath = (record: unknown, path: readonly string[]): unknown => {
  let current: unknown = record;
  for (const segment of path) {
    if (!isObject(current)) {
      return undefined;
    }
    current = current[segment];
  }
  return current;
};

const uniqueSorted = (values: readonly string[]): string[] => [...new Set(values)].sort();

const isSecretFieldKey = (key: string): boolean => SECRET_FIELD_PATTERN.test(key);

const stringValue = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const objectValue = (value: unknown): JsonObject | undefined => isObject(value) ? value : undefined;

const isObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value != null && !Array.isArray(value);
