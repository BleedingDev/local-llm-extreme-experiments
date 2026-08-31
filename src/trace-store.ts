import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type { HaloSpan } from "./telemetry";
import type { BagConfig } from "./types";

export type TraceIndexRow = {
  traceId: string;
  byteOffsets: number[];
  byteLengths: number[];
  rawJsonlBytes: number;
  spanCount: number;
  errorSpanCount: number;
  startTime: string;
  endTime: string;
  hasErrors: boolean;
  serviceNames: string[];
  modelNames: string[];
  agentNames: string[];
  spanNames: string[];
  observationKinds: string[];
  modelProfileIds: string[];
  codebaseProfileIds: string[];
  policyIds: string[];
  canonicalToolVersions: string[];
  renderedToolVersions: string[];
  resultStyleVersions: string[];
  verificationPolicyVersions: string[];
  editStrategyVersions: string[];
  renderedEditContractVersions: string[];
  editFallbackPolicyVersions: string[];
  editRepairPolicyVersions: string[];
  editVerifierPolicyVersions: string[];
  editObjectiveSetIds: string[];
  editStrategyIds: string[];
  editStrategyFamilies: string[];
  canonicalEditToolSpecIds: string[];
  renderedEditToolContractIds: string[];
  editVerificationStatuses: string[];
  editPostApplyConsistencyStatuses: string[];
  editSelfDetectedRegressionStatuses: string[];
  editRollbackStatuses: string[];
  editRedactionStatuses: string[];
  totalInputTokens: number;
  totalOutputTokens: number;
  projectId?: string;
};

export type TraceIndexMeta = {
  schemaVersion: 3;
  sourcePath: string;
  sourceSize: number;
  sourceBytes: number;
  sourceMtimeMs: number;
  sourceMtimeNs: string;
  sourceCtimeNs: string;
  sourceDev: string;
  sourceIno: string;
  rawJsonlBytes: number;
  parsedBytes: number;
  corruptBytes: number;
  lineCount: number;
  blankLineCount: number;
  parsedLineCount: number;
  corruptLineCount: number;
  parseErrorCount: number;
  traceCount: number;
  spanCount: number;
  builtAt: string;
};

export type TraceFilters = {
  hasErrors?: boolean;
  serviceName?: string;
  modelName?: string;
  agentName?: string;
  observationKind?: string;
  projectId?: string;
  modelProfileId?: string;
  codebaseProfileId?: string;
  policyId?: string;
  canonicalToolVersion?: string;
  renderedToolVersion?: string;
  resultStyleVersion?: string;
  verificationPolicyVersion?: string;
  editStrategyVersion?: string;
  renderedEditContractVersion?: string;
  editFallbackPolicyVersion?: string;
  editRepairPolicyVersion?: string;
  editVerifierPolicyVersion?: string;
  editObjectiveSetId?: string;
  editStrategyId?: string;
  editStrategyFamily?: string;
  canonicalEditToolSpecId?: string;
  renderedEditToolContractId?: string;
  editVerificationStatus?: string;
  editPostApplyConsistencyStatus?: string;
  editSelfDetectedRegressionStatus?: string;
  editRollbackStatus?: string;
  editRedactionStatus?: string;
  minSpans?: number;
  maxSpans?: number;
};

export type TraceSummary = Omit<TraceIndexRow, "byteOffsets" | "byteLengths">;

export type DatasetOverview = {
  traceCount: number;
  spanCount: number;
  sourcePath: string;
  sourceBytes: number;
  sourceMtimeMs: number;
  sourceMtimeNs: string;
  indexBuiltAt: string;
  rawJsonlBytes: number;
  parsedBytes: number;
  corruptBytes: number;
  lineCount: number;
  blankLineCount: number;
  parsedLineCount: number;
  corruptLineCount: number;
  parseErrorCount: number;
  errorTraceCount: number;
  errorSpanCount: number;
  services: string[];
  models: string[];
  agents: string[];
  observationKinds: string[];
  modelProfileIds: string[];
  codebaseProfileIds: string[];
  policyIds: string[];
  canonicalToolVersions: string[];
  renderedToolVersions: string[];
  resultStyleVersions: string[];
  verificationPolicyVersions: string[];
  editStrategyVersions: string[];
  renderedEditContractVersions: string[];
  editFallbackPolicyVersions: string[];
  editRepairPolicyVersions: string[];
  editVerifierPolicyVersions: string[];
  editObjectiveSetIds: string[];
  editStrategyIds: string[];
  editStrategyFamilies: string[];
  canonicalEditToolSpecIds: string[];
  renderedEditToolContractIds: string[];
  editVerificationStatuses: string[];
  editPostApplyConsistencyStatuses: string[];
  editSelfDetectedRegressionStatuses: string[];
  editRollbackStatuses: string[];
  editRedactionStatuses: string[];
  totalInputTokens: number;
  totalOutputTokens: number;
  sampleTraceIds: string[];
};

export type TraceView = {
  traceId: string;
  spanCount: number;
  spans: HaloSpan[];
  requestedSpanCount?: number;
  returnedCount?: number;
  missingSpanIds?: string[];
  omittedSpanCount?: number;
  hasMore?: boolean;
  truncated?: boolean;
  limits?: TraceViewLimits;
  oversized?: {
    reason: string;
    charBudget: number;
    estimatedChars: number;
    spanCount: number;
    errorSpanCount: number;
    topSpanNames: Array<{ name: string; count: number }>;
  };
};

export type TraceViewLimits = {
  maxRequested: number;
  maxReturned: number;
  attrCap: number;
  charBudget: number;
};

export type TraceSearchResult = {
  traceId: string;
  pattern: string;
  mode: "literal" | "regex";
  flags: string;
  matches: HaloSpan[];
  matchCount: number;
  returnedCount: number;
  hasMore: boolean;
  truncated: boolean;
  contexts: TraceSearchContext[];
  limits: TraceSearchLimits;
  error?: TraceSearchError;
};

export type TraceSpanSearchResult = Omit<TraceSearchResult, "limits"> & {
  limits: TraceSpanSearchLimits;
  requestedSpanCount: number;
  searchedSpanCount: number;
  missingSpanIds: string[];
  omittedSpanCount: number;
};

export type TraceQueryResult = {
  total: number;
  traces: TraceSummary[];
};

export type TraceSearchOptions = {
  mode?: "literal" | "regex";
  flags?: string;
  limit?: number;
  maxMatchCount?: number;
  maxPatternLength?: number;
  contextChars?: number;
};

export type TraceSpanSearchOptions = TraceSearchOptions & {
  maxSpanIds?: number;
  maxSearchBytes?: number;
  attrCap?: number;
};

export type TraceViewSpansOptions = {
  maxRequested?: number;
  limit?: number;
  attrCap?: number;
  charBudget?: number;
};

export type TraceSearchContext = {
  spanId: string;
  spanName: string;
  index: number;
  match: string;
  before: string;
  after: string;
};

export type TraceSearchLimits = {
  maxPatternLength: number;
  maxReturned: number;
  maxMatchCount: number;
  maxSpanSearchChars: number;
  contextChars: number;
};

export type TraceSpanSearchLimits = TraceSearchLimits & {
  maxSpanIds: number;
  maxSearchBytes: number;
  attrCap: number;
};

export type TraceSearchError = {
  code: "pattern_too_long" | "invalid_regex" | "unsupported_regex_flags" | "unsafe_regex";
  message: string;
};

const SCHEMA_VERSION = 3;
const DISCOVERY_ATTR_CAP = 4096;
const VIEW_SPANS_ATTR_CAP = 16_384;
const TRACE_CHAR_BUDGET = 150_000;
const VIEW_SPANS_CHAR_BUDGET = 150_000;
const DEFAULT_VIEW_SPANS_REQUEST_LIMIT = 100;
const MAX_VIEW_SPANS_REQUEST_LIMIT = 500;
const DEFAULT_VIEW_SPANS_RETURN_LIMIT = 100;
const MAX_VIEW_SPANS_RETURN_LIMIT = 100;
const DEFAULT_SEARCH_PATTERN_LIMIT = 512;
const DEFAULT_SEARCH_RETURN_LIMIT = 100;
const MAX_SEARCH_RETURN_LIMIT = 100;
const DEFAULT_SEARCH_COUNT_LIMIT = 1_000;
const MAX_SEARCH_COUNT_LIMIT = 1_000;
const MAX_SEARCH_CONTEXT_CHARS = 200;
const MAX_SPAN_SEARCH_CHARS = 200_000;
const DEFAULT_SPAN_SEARCH_ID_LIMIT = 100;
const MAX_SPAN_SEARCH_ID_LIMIT = 500;
const DEFAULT_SPAN_SEARCH_BYTES = 200_000;
const MAX_SPAN_SEARCH_BYTES = 1_000_000;
const REGEX_FLAG_ALLOWLIST = /^[imsu]*$/;
const NOISY_PROJECTION_SUMMARY_PREFIX = "trace.sanitized.openinference_flat_projection";
const NOISY_PROJECTION_KEY_SAMPLE_LIMIT = 20;
const NOISY_PROJECTION_MARKER_CAP = 256;

const telemetryTracePath = (config: BagConfig, cwd: string): string => resolve(cwd, config.telemetry.spans);
const indexPath = (tracePath: string): string => `${tracePath}.index.jsonl`;
const metaPath = (tracePath: string): string => `${tracePath}.index.meta.json`;

const emptyTraceIndexMeta = (tracePath: string): TraceIndexMeta => ({
  schemaVersion: SCHEMA_VERSION,
  sourcePath: tracePath,
  sourceSize: 0,
  sourceBytes: 0,
  sourceMtimeMs: 0,
  sourceMtimeNs: "0",
  sourceCtimeNs: "0",
  sourceDev: "0",
  sourceIno: "0",
  rawJsonlBytes: 0,
  parsedBytes: 0,
  corruptBytes: 0,
  lineCount: 0,
  blankLineCount: 0,
  parsedLineCount: 0,
  corruptLineCount: 0,
  parseErrorCount: 0,
  traceCount: 0,
  spanCount: 0,
  builtAt: new Date().toISOString(),
});

const sourceIndexMetadata = (
  tracePath: string,
  stat: { size: number; mtimeMs: number },
  highResolutionStat: { mtimeNs: bigint; ctimeNs: bigint; dev: bigint; ino: bigint },
): Pick<
  TraceIndexMeta,
  "sourcePath" | "sourceSize" | "sourceBytes" | "sourceMtimeMs" | "sourceMtimeNs" | "sourceCtimeNs" | "sourceDev" | "sourceIno"
> => ({
  sourcePath: tracePath,
  sourceSize: stat.size,
  sourceBytes: stat.size,
  sourceMtimeMs: stat.mtimeMs,
  sourceMtimeNs: highResolutionStat.mtimeNs.toString(),
  sourceCtimeNs: highResolutionStat.ctimeNs.toString(),
  sourceDev: highResolutionStat.dev.toString(),
  sourceIno: highResolutionStat.ino.toString(),
});

const hasCompleteTraceSizingMeta = (meta: TraceIndexMeta): boolean =>
  typeof meta.sourceBytes === "number" &&
  typeof meta.sourceMtimeNs === "string" &&
  typeof meta.sourceCtimeNs === "string" &&
  typeof meta.sourceDev === "string" &&
  typeof meta.sourceIno === "string" &&
  typeof meta.rawJsonlBytes === "number" &&
  typeof meta.parsedBytes === "number" &&
  typeof meta.corruptBytes === "number" &&
  typeof meta.lineCount === "number" &&
  typeof meta.blankLineCount === "number" &&
  typeof meta.parsedLineCount === "number" &&
  typeof meta.corruptLineCount === "number" &&
  typeof meta.parseErrorCount === "number";

const observationKind = (span: HaloSpan): string => {
  const kind = span.attributes["inference.observation_kind"] ?? span.attributes["openinference.span.kind"];
  return typeof kind === "string" && kind.length > 0 ? kind : "SPAN";
};

const stringAttr = (value: unknown): string | undefined => (typeof value === "string" && value.length > 0 ? value : undefined);

const numberAttr = (value: unknown): number => (typeof value === "number" && Number.isFinite(value) ? value : 0);

const addMaybe = (set: Set<string>, value: unknown): void => {
  if (typeof value === "string" && value.length > 0) {
    set.add(value);
  }
};

const toSummary = (row: TraceIndexRow): TraceSummary => {
  const { byteOffsets: _byteOffsets, byteLengths: _byteLengths, ...summary } = row;
  return summary;
};

const matchesFilters = (row: TraceIndexRow, filters: TraceFilters = {}): boolean => {
  if (filters.hasErrors != null && row.hasErrors !== filters.hasErrors) {
    return false;
  }
  if (filters.serviceName != null && !row.serviceNames.includes(filters.serviceName)) {
    return false;
  }
  if (filters.modelName != null && !row.modelNames.includes(filters.modelName)) {
    return false;
  }
  if (filters.agentName != null && !row.agentNames.includes(filters.agentName)) {
    return false;
  }
  if (filters.observationKind != null && !row.observationKinds.includes(filters.observationKind)) {
    return false;
  }
  if (filters.projectId != null && row.projectId !== filters.projectId) {
    return false;
  }
  if (filters.modelProfileId != null && !row.modelProfileIds.includes(filters.modelProfileId)) {
    return false;
  }
  if (filters.codebaseProfileId != null && !row.codebaseProfileIds.includes(filters.codebaseProfileId)) {
    return false;
  }
  if (filters.policyId != null && !row.policyIds.includes(filters.policyId)) {
    return false;
  }
  if (filters.canonicalToolVersion != null && !row.canonicalToolVersions.includes(filters.canonicalToolVersion)) {
    return false;
  }
  if (filters.renderedToolVersion != null && !row.renderedToolVersions.includes(filters.renderedToolVersion)) {
    return false;
  }
  if (filters.resultStyleVersion != null && !row.resultStyleVersions.includes(filters.resultStyleVersion)) {
    return false;
  }
  if (
    filters.verificationPolicyVersion != null &&
    !row.verificationPolicyVersions.includes(filters.verificationPolicyVersion)
  ) {
    return false;
  }
  if (filters.editStrategyVersion != null && !row.editStrategyVersions.includes(filters.editStrategyVersion)) {
    return false;
  }
  if (
    filters.renderedEditContractVersion != null &&
    !row.renderedEditContractVersions.includes(filters.renderedEditContractVersion)
  ) {
    return false;
  }
  if (
    filters.editFallbackPolicyVersion != null &&
    !row.editFallbackPolicyVersions.includes(filters.editFallbackPolicyVersion)
  ) {
    return false;
  }
  if (
    filters.editRepairPolicyVersion != null &&
    !row.editRepairPolicyVersions.includes(filters.editRepairPolicyVersion)
  ) {
    return false;
  }
  if (
    filters.editVerifierPolicyVersion != null &&
    !row.editVerifierPolicyVersions.includes(filters.editVerifierPolicyVersion)
  ) {
    return false;
  }
  if (filters.editObjectiveSetId != null && !row.editObjectiveSetIds.includes(filters.editObjectiveSetId)) {
    return false;
  }
  if (filters.editStrategyId != null && !row.editStrategyIds.includes(filters.editStrategyId)) {
    return false;
  }
  if (filters.editStrategyFamily != null && !row.editStrategyFamilies.includes(filters.editStrategyFamily)) {
    return false;
  }
  if (
    filters.canonicalEditToolSpecId != null &&
    !row.canonicalEditToolSpecIds.includes(filters.canonicalEditToolSpecId)
  ) {
    return false;
  }
  if (
    filters.renderedEditToolContractId != null &&
    !row.renderedEditToolContractIds.includes(filters.renderedEditToolContractId)
  ) {
    return false;
  }
  if (
    filters.editVerificationStatus != null &&
    !row.editVerificationStatuses.includes(filters.editVerificationStatus)
  ) {
    return false;
  }
  if (
    filters.editPostApplyConsistencyStatus != null &&
    !row.editPostApplyConsistencyStatuses.includes(filters.editPostApplyConsistencyStatus)
  ) {
    return false;
  }
  if (
    filters.editSelfDetectedRegressionStatus != null &&
    !row.editSelfDetectedRegressionStatuses.includes(filters.editSelfDetectedRegressionStatus)
  ) {
    return false;
  }
  if (filters.editRollbackStatus != null && !row.editRollbackStatuses.includes(filters.editRollbackStatus)) {
    return false;
  }
  if (filters.editRedactionStatus != null && !row.editRedactionStatuses.includes(filters.editRedactionStatus)) {
    return false;
  }
  if (filters.minSpans != null && row.spanCount < filters.minSpans) {
    return false;
  }
  if (filters.maxSpans != null && row.spanCount > filters.maxSpans) {
    return false;
  }
  return true;
};

const truncateAttributes = (value: unknown, cap: number): unknown => {
  if (typeof value === "string") {
    if (value.length <= cap) {
      return value;
    }
    return `${value.slice(0, cap)}[BleedingAgent trace-store truncated: original ${value.length} chars]`;
  }
  if (Array.isArray(value)) {
    return value.map((item) => truncateAttributes(item, cap));
  }
  if (value != null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, nested]) => [key, truncateAttributes(nested, cap)]),
    );
  }
  return value;
};

const approximateAttributeBytes = (value: unknown): number => {
  try {
    return Buffer.byteLength(JSON.stringify(value));
  } catch {
    return Buffer.byteLength(String(value));
  }
};

const noisyProjectionGroup = (key: string): string | undefined => {
  if (/^llm\.input_messages\.\d+\./.test(key)) {
    return "llm.input_messages";
  }
  if (/^llm\.output_messages\.\d+\./.test(key)) {
    return "llm.output_messages";
  }
  if (/^llm\.tools\.\d+\./.test(key)) {
    return "llm.tools";
  }
  if (/^retrieval\.documents\.\d+\./.test(key)) {
    return "retrieval.documents";
  }
  if (/^embedding\.embeddings\.\d+\./.test(key)) {
    return "embedding.embeddings";
  }
  return undefined;
};

const isNoisyProjectionMarker = (key: string, value: unknown): boolean => {
  if (typeof value !== "string" && typeof value !== "number" && typeof value !== "boolean") {
    return false;
  }
  return (
    /\.message\.role$/.test(key) ||
    /\.message_content\.type$/.test(key) ||
    /\.tool_call\.id$/.test(key) ||
    /\.tool_call\.function\.name$/.test(key) ||
    /\.document\.id$/.test(key)
  );
};

const sanitizeTraceAttributes = (attributes: Record<string, unknown>, cap: number): Record<string, unknown> => {
  const sanitized: Record<string, unknown> = {};
  const omittedKeys: string[] = [];
  const groups = new Set<string>();
  let omittedCount = 0;
  let omittedBytes = 0;

  for (const [key, value] of Object.entries(attributes)) {
    const group = noisyProjectionGroup(key);
    if (group == null || isNoisyProjectionMarker(key, value)) {
      sanitized[key] = truncateAttributes(value, group == null ? cap : Math.min(cap, NOISY_PROJECTION_MARKER_CAP));
      continue;
    }
    groups.add(group);
    omittedCount += 1;
    omittedBytes += approximateAttributeBytes(value);
    if (omittedKeys.length < NOISY_PROJECTION_KEY_SAMPLE_LIMIT) {
      omittedKeys.push(key);
    }
  }

  if (omittedCount > 0) {
    sanitized[`${NOISY_PROJECTION_SUMMARY_PREFIX}.count`] = omittedCount;
    sanitized[`${NOISY_PROJECTION_SUMMARY_PREFIX}.bytes`] = omittedBytes;
    sanitized[`${NOISY_PROJECTION_SUMMARY_PREFIX}.groups`] = [...groups].sort();
    sanitized[`${NOISY_PROJECTION_SUMMARY_PREFIX}.keys`] = omittedKeys.sort();
    if (omittedCount > omittedKeys.length) {
      sanitized[`${NOISY_PROJECTION_SUMMARY_PREFIX}.omitted_key_count`] = omittedCount - omittedKeys.length;
    }
  }

  return sanitized;
};

const sanitizeSpanForOutput = (span: HaloSpan, cap: number): HaloSpan => ({
  ...span,
  resource: {
    attributes: truncateAttributes(span.resource.attributes, cap) as Record<string, unknown>,
  },
  attributes: sanitizeTraceAttributes(span.attributes, cap),
});

const estimateTraceChars = (spans: HaloSpan[], attrCap: number): number =>
  spans.reduce((sum, span) => sum + JSON.stringify(sanitizeSpanForOutput(span, attrCap)).length, 0);

const topSpanNames = (spans: HaloSpan[]): Array<{ name: string; count: number }> =>
  [...spans.reduce((map, span) => map.set(span.name, (map.get(span.name) ?? 0) + 1), new Map<string, number>())]
    .map(([name, count]) => ({ name, count }))
    .sort((left, right) => right.count - left.count)
    .slice(0, 10);

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};

const traceSearchLimits = (options: TraceSearchOptions): TraceSearchLimits => {
  const maxReturned = boundedInteger(options.limit, DEFAULT_SEARCH_RETURN_LIMIT, 0, MAX_SEARCH_RETURN_LIMIT);
  return {
    maxPatternLength: boundedInteger(
      options.maxPatternLength,
      DEFAULT_SEARCH_PATTERN_LIMIT,
      1,
      DEFAULT_SEARCH_PATTERN_LIMIT,
    ),
    maxReturned,
    maxMatchCount: Math.max(
      maxReturned,
      boundedInteger(options.maxMatchCount, DEFAULT_SEARCH_COUNT_LIMIT, 1, MAX_SEARCH_COUNT_LIMIT),
    ),
    maxSpanSearchChars: MAX_SPAN_SEARCH_CHARS,
    contextChars: boundedInteger(options.contextChars, 0, 0, MAX_SEARCH_CONTEXT_CHARS),
  };
};

const traceSpanSearchLimits = (options: TraceSpanSearchOptions): TraceSpanSearchLimits => ({
  ...traceSearchLimits(options),
  maxSpanIds: boundedInteger(options.maxSpanIds, DEFAULT_SPAN_SEARCH_ID_LIMIT, 1, MAX_SPAN_SEARCH_ID_LIMIT),
  maxSearchBytes: boundedInteger(options.maxSearchBytes, DEFAULT_SPAN_SEARCH_BYTES, 1, MAX_SPAN_SEARCH_BYTES),
  attrCap: boundedInteger(options.attrCap, VIEW_SPANS_ATTR_CAP, 0, VIEW_SPANS_ATTR_CAP),
});

const traceViewLimits = (options: TraceViewSpansOptions): TraceViewLimits => ({
  maxRequested: boundedInteger(
    options.maxRequested,
    DEFAULT_VIEW_SPANS_REQUEST_LIMIT,
    0,
    MAX_VIEW_SPANS_REQUEST_LIMIT,
  ),
  maxReturned: boundedInteger(options.limit, DEFAULT_VIEW_SPANS_RETURN_LIMIT, 0, MAX_VIEW_SPANS_RETURN_LIMIT),
  attrCap: boundedInteger(options.attrCap, VIEW_SPANS_ATTR_CAP, 0, VIEW_SPANS_ATTR_CAP),
  charBudget: boundedInteger(options.charBudget, VIEW_SPANS_CHAR_BUDGET, 1, VIEW_SPANS_CHAR_BUDGET),
});

const emptySearchResult = (
  traceId: string,
  pattern: string,
  mode: "literal" | "regex",
  flags: string,
  limits: TraceSearchLimits,
  error?: TraceSearchError,
): TraceSearchResult => ({
  traceId,
  pattern,
  mode,
  flags,
  matches: [],
  matchCount: 0,
  returnedCount: 0,
  hasMore: false,
  truncated: false,
  contexts: [],
  limits,
  ...(error == null ? {} : { error }),
});

const emptySpanSearchResult = (
  traceId: string,
  pattern: string,
  mode: "literal" | "regex",
  flags: string,
  limits: TraceSpanSearchLimits,
  requestedSpanCount: number,
  missingSpanIds: string[],
  omittedSpanCount: number,
  error?: TraceSearchError,
): TraceSpanSearchResult => ({
  ...emptySearchResult(traceId, pattern, mode, flags, limits, error),
  limits,
  requestedSpanCount,
  searchedSpanCount: 0,
  missingSpanIds,
  omittedSpanCount,
  hasMore: omittedSpanCount > 0,
  truncated: omittedSpanCount > 0,
});

const uniqueRegexFlags = (flags: string): string => [...new Set(flags.split(""))].join("");

const hasUnsafeRegexShape = (pattern: string): boolean => {
  const withoutEscapes = pattern.replace(/\\./g, "");
  return /\((?:[^()]|\([^()]*\))*[+*{][^)]*\)\s*(?:[+*?]|\{\d*,?\d*\})/.test(withoutEscapes);
};

const literalContext = (raw: string, pattern: string, chars: number): Omit<TraceSearchContext, "spanId" | "spanName"> | undefined => {
  if (chars <= 0) {
    return undefined;
  }
  const index = raw.indexOf(pattern);
  if (index < 0) {
    return undefined;
  }
  return {
    index,
    match: pattern,
    before: raw.slice(Math.max(0, index - chars), index),
    after: raw.slice(index + pattern.length, index + pattern.length + chars),
  };
};

const regexContext = (
  raw: string,
  regex: RegExp,
  chars: number,
): Omit<TraceSearchContext, "spanId" | "spanName"> | undefined => {
  if (chars <= 0) {
    return undefined;
  }
  regex.lastIndex = 0;
  const match = regex.exec(raw);
  if (match == null || match.index < 0) {
    return undefined;
  }
  return {
    index: match.index,
    match: match[0],
    before: raw.slice(Math.max(0, match.index - chars), match.index),
    after: raw.slice(match.index + match[0].length, match.index + match[0].length + chars),
  };
};

export const buildTraceIndex = (input: {
  config: BagConfig;
  cwd?: string;
  force?: boolean;
}): {
  tracePath: string;
  indexPath: string;
  metaPath: string;
  meta: TraceIndexMeta;
  rows: TraceIndexRow[];
} => {
  const cwd = input.cwd ?? process.cwd();
  const tracePath = telemetryTracePath(input.config, cwd);
  const index = indexPath(tracePath);
  const meta = metaPath(tracePath);
  if (!existsSync(tracePath)) {
    const emptyMeta = emptyTraceIndexMeta(tracePath);
    return { tracePath, indexPath: index, metaPath: meta, meta: emptyMeta, rows: [] };
  }

  const stat = statSync(tracePath);
  const highResolutionStat = statSync(tracePath, { bigint: true });
  const sourceMeta = sourceIndexMetadata(tracePath, stat, highResolutionStat);
  if (!input.force && existsSync(index) && existsSync(meta)) {
    const cached = JSON.parse(readFileSync(meta, "utf8")) as TraceIndexMeta;
    if (
      cached.schemaVersion === SCHEMA_VERSION &&
      hasCompleteTraceSizingMeta(cached) &&
      cached.sourceSize === stat.size &&
      cached.sourceBytes === stat.size &&
      cached.sourceMtimeMs === stat.mtimeMs &&
      cached.sourceMtimeNs === sourceMeta.sourceMtimeNs &&
      cached.sourceCtimeNs === sourceMeta.sourceCtimeNs &&
      cached.sourceDev === sourceMeta.sourceDev &&
      cached.sourceIno === sourceMeta.sourceIno
    ) {
      return {
        tracePath,
        indexPath: index,
        metaPath: meta,
        meta: cached,
        rows: readIndexRows(index),
      };
    }
  }

  const rows = new Map<string, TraceIndexRow>();
  const content = readFileSync(tracePath, "utf8");
  let offset = 0;
  let spanCount = 0;
  let parsedBytes = 0;
  let corruptBytes = 0;
  let lineCount = 0;
  let blankLineCount = 0;
  let parsedLineCount = 0;
  let corruptLineCount = 0;
  let parseErrorCount = 0;
  for (const rawLine of content.split(/(?<=\n)/)) {
    const lineLength = Buffer.byteLength(rawLine);
    const stripped = rawLine.trimEnd();
    if (stripped.length === 0) {
      if (lineLength > 0) {
        lineCount += 1;
        blankLineCount += 1;
      }
      offset += lineLength;
      continue;
    }
    lineCount += 1;
    try {
      const span = JSON.parse(stripped) as HaloSpan;
      const existing = rows.get(span.trace_id);
      const projectId = stringAttr(span.attributes["inference.project_id"]);
      const row: TraceIndexRow =
        existing ??
        {
          traceId: span.trace_id,
          byteOffsets: [],
          byteLengths: [],
          rawJsonlBytes: 0,
          spanCount: 0,
          errorSpanCount: 0,
          startTime: "",
          endTime: "",
          hasErrors: false,
          serviceNames: [],
          modelNames: [],
          agentNames: [],
          spanNames: [],
          observationKinds: [],
          modelProfileIds: [],
          codebaseProfileIds: [],
          policyIds: [],
          canonicalToolVersions: [],
          renderedToolVersions: [],
          resultStyleVersions: [],
          verificationPolicyVersions: [],
          editStrategyVersions: [],
          renderedEditContractVersions: [],
          editFallbackPolicyVersions: [],
          editRepairPolicyVersions: [],
          editVerifierPolicyVersions: [],
          editObjectiveSetIds: [],
          editStrategyIds: [],
          editStrategyFamilies: [],
          canonicalEditToolSpecIds: [],
          renderedEditToolContractIds: [],
          editVerificationStatuses: [],
          editPostApplyConsistencyStatuses: [],
          editSelfDetectedRegressionStatuses: [],
          editRollbackStatuses: [],
          editRedactionStatuses: [],
          totalInputTokens: 0,
          totalOutputTokens: 0,
          ...(projectId == null ? {} : { projectId }),
        };
      const services = new Set(row.serviceNames);
      const models = new Set(row.modelNames);
      const agents = new Set(row.agentNames);
      const names = new Set(row.spanNames);
      const kinds = new Set(row.observationKinds);
      const modelProfileIds = new Set(row.modelProfileIds);
      const codebaseProfileIds = new Set(row.codebaseProfileIds);
      const policyIds = new Set(row.policyIds);
      const canonicalToolVersions = new Set(row.canonicalToolVersions);
      const renderedToolVersions = new Set(row.renderedToolVersions);
      const resultStyleVersions = new Set(row.resultStyleVersions);
      const verificationPolicyVersions = new Set(row.verificationPolicyVersions);
      const editStrategyVersions = new Set(row.editStrategyVersions);
      const renderedEditContractVersions = new Set(row.renderedEditContractVersions);
      const editFallbackPolicyVersions = new Set(row.editFallbackPolicyVersions);
      const editRepairPolicyVersions = new Set(row.editRepairPolicyVersions);
      const editVerifierPolicyVersions = new Set(row.editVerifierPolicyVersions);
      const editObjectiveSetIds = new Set(row.editObjectiveSetIds);
      const editStrategyIds = new Set(row.editStrategyIds);
      const editStrategyFamilies = new Set(row.editStrategyFamilies);
      const canonicalEditToolSpecIds = new Set(row.canonicalEditToolSpecIds);
      const renderedEditToolContractIds = new Set(row.renderedEditToolContractIds);
      const editVerificationStatuses = new Set(row.editVerificationStatuses);
      const editPostApplyConsistencyStatuses = new Set(row.editPostApplyConsistencyStatuses);
      const editSelfDetectedRegressionStatuses = new Set(row.editSelfDetectedRegressionStatuses);
      const editRollbackStatuses = new Set(row.editRollbackStatuses);
      const editRedactionStatuses = new Set(row.editRedactionStatuses);
      addMaybe(services, span.resource.attributes["service.name"]);
      addMaybe(models, span.attributes["inference.llm.model_name"] ?? span.attributes["llm.model_name"]);
      addMaybe(agents, span.attributes["inference.agent_name"]);
      addMaybe(modelProfileIds, span.attributes["optimizer.model_profile_id"]);
      addMaybe(codebaseProfileIds, span.attributes["optimizer.codebase_profile_id"]);
      addMaybe(policyIds, span.attributes["optimizer.policy_id"]);
      addMaybe(canonicalToolVersions, span.attributes["optimizer.canonical_tool_version"]);
      addMaybe(renderedToolVersions, span.attributes["optimizer.rendered_tool_version"]);
      addMaybe(resultStyleVersions, span.attributes["optimizer.result_style_version"]);
      addMaybe(verificationPolicyVersions, span.attributes["optimizer.verification_policy_version"]);
      addMaybe(editStrategyVersions, span.attributes["optimizer.edit_strategy_version"]);
      addMaybe(renderedEditContractVersions, span.attributes["optimizer.rendered_edit_contract_version"]);
      addMaybe(editFallbackPolicyVersions, span.attributes["optimizer.edit_fallback_policy_version"]);
      addMaybe(editRepairPolicyVersions, span.attributes["optimizer.edit_repair_policy_version"]);
      addMaybe(editVerifierPolicyVersions, span.attributes["optimizer.edit_verifier_policy_version"]);
      addMaybe(editObjectiveSetIds, span.attributes["optimizer.edit_objective_set_id"]);
      addMaybe(editStrategyIds, span.attributes["edit.strategy_id"]);
      addMaybe(editStrategyFamilies, span.attributes["edit.strategy_family"]);
      addMaybe(canonicalEditToolSpecIds, span.attributes["edit.canonical_tool_spec_id"]);
      addMaybe(renderedEditToolContractIds, span.attributes["edit.rendered_tool_contract_id"]);
      addMaybe(editVerificationStatuses, span.attributes["edit.verification_status"]);
      addMaybe(editPostApplyConsistencyStatuses, span.attributes["edit.post_apply_consistency_status"]);
      addMaybe(editSelfDetectedRegressionStatuses, span.attributes["edit.self_detected_regression_status"]);
      addMaybe(editRollbackStatuses, span.attributes["edit.rollback_status"]);
      addMaybe(editRedactionStatuses, span.attributes["edit.redaction_status"]);
      names.add(span.name);
      kinds.add(observationKind(span));
      row.byteOffsets.push(offset);
      row.byteLengths.push(Buffer.byteLength(stripped));
      row.rawJsonlBytes += lineLength;
      row.spanCount += 1;
      row.errorSpanCount += span.status.code === "STATUS_CODE_ERROR" ? 1 : 0;
      row.hasErrors ||= span.status.code === "STATUS_CODE_ERROR";
      row.startTime = row.startTime === "" || span.start_time < row.startTime ? span.start_time : row.startTime;
      row.endTime = row.endTime === "" || span.end_time > row.endTime ? span.end_time : row.endTime;
      row.serviceNames = [...services].sort();
      row.modelNames = [...models].sort();
      row.agentNames = [...agents].sort();
      row.spanNames = [...names].sort();
      row.observationKinds = [...kinds].sort();
      row.modelProfileIds = [...modelProfileIds].sort();
      row.codebaseProfileIds = [...codebaseProfileIds].sort();
      row.policyIds = [...policyIds].sort();
      row.canonicalToolVersions = [...canonicalToolVersions].sort();
      row.renderedToolVersions = [...renderedToolVersions].sort();
      row.resultStyleVersions = [...resultStyleVersions].sort();
      row.verificationPolicyVersions = [...verificationPolicyVersions].sort();
      row.editStrategyVersions = [...editStrategyVersions].sort();
      row.renderedEditContractVersions = [...renderedEditContractVersions].sort();
      row.editFallbackPolicyVersions = [...editFallbackPolicyVersions].sort();
      row.editRepairPolicyVersions = [...editRepairPolicyVersions].sort();
      row.editVerifierPolicyVersions = [...editVerifierPolicyVersions].sort();
      row.editObjectiveSetIds = [...editObjectiveSetIds].sort();
      row.editStrategyIds = [...editStrategyIds].sort();
      row.editStrategyFamilies = [...editStrategyFamilies].sort();
      row.canonicalEditToolSpecIds = [...canonicalEditToolSpecIds].sort();
      row.renderedEditToolContractIds = [...renderedEditToolContractIds].sort();
      row.editVerificationStatuses = [...editVerificationStatuses].sort();
      row.editPostApplyConsistencyStatuses = [...editPostApplyConsistencyStatuses].sort();
      row.editSelfDetectedRegressionStatuses = [...editSelfDetectedRegressionStatuses].sort();
      row.editRollbackStatuses = [...editRollbackStatuses].sort();
      row.editRedactionStatuses = [...editRedactionStatuses].sort();
      row.totalInputTokens += numberAttr(span.attributes["inference.llm.input_tokens"]);
      row.totalOutputTokens += numberAttr(span.attributes["inference.llm.output_tokens"]);
      rows.set(span.trace_id, row);
      spanCount += 1;
      parsedLineCount += 1;
      parsedBytes += lineLength;
    } catch {
      corruptLineCount += 1;
      parseErrorCount += 1;
      corruptBytes += lineLength;
    }
    offset += lineLength;
  }

  const sortedRows = [...rows.values()].sort((left, right) => left.startTime.localeCompare(right.startTime));
  const nextMeta: TraceIndexMeta = {
    schemaVersion: SCHEMA_VERSION,
    ...sourceMeta,
    rawJsonlBytes: stat.size,
    parsedBytes,
    corruptBytes,
    lineCount,
    blankLineCount,
    parsedLineCount,
    corruptLineCount,
    parseErrorCount,
    traceCount: sortedRows.length,
    spanCount,
    builtAt: new Date().toISOString(),
  };
  mkdirSync(dirname(index), { recursive: true });
  writeFileSync(index, sortedRows.map((row) => JSON.stringify(row)).join("\n") + (sortedRows.length > 0 ? "\n" : ""));
  writeFileSync(meta, `${JSON.stringify(nextMeta, null, 2)}\n`);
  return { tracePath, indexPath: index, metaPath: meta, meta: nextMeta, rows: sortedRows };
};

export const readIndexRows = (path: string): TraceIndexRow[] => {
  if (!existsSync(path)) {
    return [];
  }
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .flatMap((line) => {
      try {
        const row = JSON.parse(line) as TraceIndexRow;
        if (typeof row.rawJsonlBytes !== "number") {
          row.rawJsonlBytes = row.byteLengths.reduce((sum, length) => sum + length, 0);
        }
        row.editStrategyVersions ??= [];
        row.renderedEditContractVersions ??= [];
        row.editFallbackPolicyVersions ??= [];
        row.editRepairPolicyVersions ??= [];
        row.editVerifierPolicyVersions ??= [];
        row.editObjectiveSetIds ??= [];
        row.editStrategyIds ??= [];
        row.editStrategyFamilies ??= [];
        row.canonicalEditToolSpecIds ??= [];
        row.renderedEditToolContractIds ??= [];
        row.editVerificationStatuses ??= [];
        row.editPostApplyConsistencyStatuses ??= [];
        row.editSelfDetectedRegressionStatuses ??= [];
        row.editRollbackStatuses ??= [];
        row.editRedactionStatuses ??= [];
        return [row];
      } catch {
        return [];
      }
    });
};

export class TraceStore {
  private readonly source: Buffer;
  private readonly meta: TraceIndexMeta;
  private readonly rowsById: Map<string, TraceIndexRow>;

  constructor(
    tracePath: string,
    rows: TraceIndexRow[],
    meta: TraceIndexMeta = emptyTraceIndexMeta(tracePath),
  ) {
    this.source = existsSync(tracePath) ? readFileSync(tracePath) : Buffer.from("");
    this.meta = meta;
    this.rowsById = new Map(rows.map((row) => [row.traceId, row]));
  }

  static open(config: BagConfig, cwd = process.cwd()): TraceStore {
    const index = buildTraceIndex({ config, cwd });
    return new TraceStore(index.tracePath, index.rows, index.meta);
  }

  getOverview(filters: TraceFilters = {}): DatasetOverview {
    const rows = this.matchingRows(filters);
    const services = new Set<string>();
    const models = new Set<string>();
    const agents = new Set<string>();
    const kinds = new Set<string>();
    const modelProfileIds = new Set<string>();
    const codebaseProfileIds = new Set<string>();
    const policyIds = new Set<string>();
    const canonicalToolVersions = new Set<string>();
    const renderedToolVersions = new Set<string>();
    const resultStyleVersions = new Set<string>();
    const verificationPolicyVersions = new Set<string>();
    const editStrategyVersions = new Set<string>();
    const renderedEditContractVersions = new Set<string>();
    const editFallbackPolicyVersions = new Set<string>();
    const editRepairPolicyVersions = new Set<string>();
    const editVerifierPolicyVersions = new Set<string>();
    const editObjectiveSetIds = new Set<string>();
    const editStrategyIds = new Set<string>();
    const editStrategyFamilies = new Set<string>();
    const canonicalEditToolSpecIds = new Set<string>();
    const renderedEditToolContractIds = new Set<string>();
    const editVerificationStatuses = new Set<string>();
    const editPostApplyConsistencyStatuses = new Set<string>();
    const editSelfDetectedRegressionStatuses = new Set<string>();
    const editRollbackStatuses = new Set<string>();
    const editRedactionStatuses = new Set<string>();
    for (const row of rows) {
      row.serviceNames.forEach((value) => services.add(value));
      row.modelNames.forEach((value) => models.add(value));
      row.agentNames.forEach((value) => agents.add(value));
      row.observationKinds.forEach((value) => kinds.add(value));
      row.modelProfileIds.forEach((value) => modelProfileIds.add(value));
      row.codebaseProfileIds.forEach((value) => codebaseProfileIds.add(value));
      row.policyIds.forEach((value) => policyIds.add(value));
      row.canonicalToolVersions.forEach((value) => canonicalToolVersions.add(value));
      row.renderedToolVersions.forEach((value) => renderedToolVersions.add(value));
      row.resultStyleVersions.forEach((value) => resultStyleVersions.add(value));
      row.verificationPolicyVersions.forEach((value) => verificationPolicyVersions.add(value));
      row.editStrategyVersions.forEach((value) => editStrategyVersions.add(value));
      row.renderedEditContractVersions.forEach((value) => renderedEditContractVersions.add(value));
      row.editFallbackPolicyVersions.forEach((value) => editFallbackPolicyVersions.add(value));
      row.editRepairPolicyVersions.forEach((value) => editRepairPolicyVersions.add(value));
      row.editVerifierPolicyVersions.forEach((value) => editVerifierPolicyVersions.add(value));
      row.editObjectiveSetIds.forEach((value) => editObjectiveSetIds.add(value));
      row.editStrategyIds.forEach((value) => editStrategyIds.add(value));
      row.editStrategyFamilies.forEach((value) => editStrategyFamilies.add(value));
      row.canonicalEditToolSpecIds.forEach((value) => canonicalEditToolSpecIds.add(value));
      row.renderedEditToolContractIds.forEach((value) => renderedEditToolContractIds.add(value));
      row.editVerificationStatuses.forEach((value) => editVerificationStatuses.add(value));
      row.editPostApplyConsistencyStatuses.forEach((value) => editPostApplyConsistencyStatuses.add(value));
      row.editSelfDetectedRegressionStatuses.forEach((value) => editSelfDetectedRegressionStatuses.add(value));
      row.editRollbackStatuses.forEach((value) => editRollbackStatuses.add(value));
      row.editRedactionStatuses.forEach((value) => editRedactionStatuses.add(value));
    }
    return {
      traceCount: rows.length,
      spanCount: rows.reduce((sum, row) => sum + row.spanCount, 0),
      sourcePath: this.meta.sourcePath,
      sourceBytes: this.meta.sourceBytes,
      sourceMtimeMs: this.meta.sourceMtimeMs,
      sourceMtimeNs: this.meta.sourceMtimeNs,
      indexBuiltAt: this.meta.builtAt,
      rawJsonlBytes: rows.reduce((sum, row) => sum + row.rawJsonlBytes, 0),
      parsedBytes: rows.reduce((sum, row) => sum + row.rawJsonlBytes, 0),
      corruptBytes: this.meta.corruptBytes,
      lineCount: this.meta.lineCount,
      blankLineCount: this.meta.blankLineCount,
      parsedLineCount: rows.reduce((sum, row) => sum + row.spanCount, 0),
      corruptLineCount: this.meta.corruptLineCount,
      parseErrorCount: this.meta.parseErrorCount,
      errorTraceCount: rows.filter((row) => row.hasErrors).length,
      errorSpanCount: rows.reduce((sum, row) => sum + row.errorSpanCount, 0),
      services: [...services].sort(),
      models: [...models].sort(),
      agents: [...agents].sort(),
      observationKinds: [...kinds].sort(),
      modelProfileIds: [...modelProfileIds].sort(),
      codebaseProfileIds: [...codebaseProfileIds].sort(),
      policyIds: [...policyIds].sort(),
      canonicalToolVersions: [...canonicalToolVersions].sort(),
      renderedToolVersions: [...renderedToolVersions].sort(),
      resultStyleVersions: [...resultStyleVersions].sort(),
      verificationPolicyVersions: [...verificationPolicyVersions].sort(),
      editStrategyVersions: [...editStrategyVersions].sort(),
      renderedEditContractVersions: [...renderedEditContractVersions].sort(),
      editFallbackPolicyVersions: [...editFallbackPolicyVersions].sort(),
      editRepairPolicyVersions: [...editRepairPolicyVersions].sort(),
      editVerifierPolicyVersions: [...editVerifierPolicyVersions].sort(),
      editObjectiveSetIds: [...editObjectiveSetIds].sort(),
      editStrategyIds: [...editStrategyIds].sort(),
      editStrategyFamilies: [...editStrategyFamilies].sort(),
      canonicalEditToolSpecIds: [...canonicalEditToolSpecIds].sort(),
      renderedEditToolContractIds: [...renderedEditToolContractIds].sort(),
      editVerificationStatuses: [...editVerificationStatuses].sort(),
      editPostApplyConsistencyStatuses: [...editPostApplyConsistencyStatuses].sort(),
      editSelfDetectedRegressionStatuses: [...editSelfDetectedRegressionStatuses].sort(),
      editRollbackStatuses: [...editRollbackStatuses].sort(),
      editRedactionStatuses: [...editRedactionStatuses].sort(),
      totalInputTokens: rows.reduce((sum, row) => sum + row.totalInputTokens, 0),
      totalOutputTokens: rows.reduce((sum, row) => sum + row.totalOutputTokens, 0),
      sampleTraceIds: rows.slice(0, 20).map((row) => row.traceId),
    };
  }

  queryTraces(filters: TraceFilters = {}, limit = 20, offset = 0): TraceQueryResult {
    const rows = this.matchingRows(filters);
    return {
      total: rows.length,
      traces: rows.slice(offset, offset + limit).map(toSummary),
    };
  }

  countTraces(filters: TraceFilters = {}): number {
    return this.matchingRows(filters).length;
  }

  viewTrace(traceId: string): TraceView {
    const spans = this.readTraceSpans(traceId);
    const estimatedChars = estimateTraceChars(spans, DISCOVERY_ATTR_CAP);
    if (estimatedChars > TRACE_CHAR_BUDGET) {
      return {
        traceId,
        spanCount: spans.length,
        spans: [],
        oversized: {
          reason: "trace exceeds bounded view budget; use searchTrace or viewSpans",
          charBudget: TRACE_CHAR_BUDGET,
          estimatedChars,
          spanCount: spans.length,
          errorSpanCount: spans.filter((span) => span.status.code === "STATUS_CODE_ERROR").length,
          topSpanNames: topSpanNames(spans),
        },
      };
    }
    return {
      traceId,
      spanCount: spans.length,
      spans: spans.map((span) => sanitizeSpanForOutput(span, DISCOVERY_ATTR_CAP)),
    };
  }

  viewSpans(traceId: string, spanIds: string[], options: TraceViewSpansOptions = {}): TraceView {
    const limits = traceViewLimits(options);
    const requestedSpanIds = [...new Set(spanIds)];
    const cappedSpanIds = requestedSpanIds.slice(0, limits.maxRequested);
    const wanted = new Set(cappedSpanIds);
    const selectedSpans = this.readTraceSpans(traceId).filter((span) => wanted.has(span.span_id));
    const foundSpanIds = new Set(selectedSpans.map((span) => span.span_id));
    const missingSpanIds = cappedSpanIds.filter((spanId) => !foundSpanIds.has(spanId));
    const spans: HaloSpan[] = [];
    let estimatedChars = 0;
    for (const span of selectedSpans) {
      if (spans.length >= limits.maxReturned) {
        break;
      }
      const capped = sanitizeSpanForOutput(span, limits.attrCap);
      const nextEstimatedChars = estimatedChars + JSON.stringify(capped).length;
      if (nextEstimatedChars > limits.charBudget) {
        break;
      }
      spans.push(capped);
      estimatedChars = nextEstimatedChars;
    }
    const omittedSpanCount = requestedSpanIds.length - cappedSpanIds.length + selectedSpans.length - spans.length;
    const hasMore = omittedSpanCount > 0;
    return {
      traceId,
      spanCount: selectedSpans.length,
      spans,
      requestedSpanCount: requestedSpanIds.length,
      returnedCount: spans.length,
      missingSpanIds,
      omittedSpanCount,
      hasMore,
      truncated: hasMore,
      limits,
    };
  }

  searchTrace(traceId: string, pattern: string, options: TraceSearchOptions = {}): TraceSearchResult {
    const mode = options.mode ?? "literal";
    const limits = traceSearchLimits(options);
    const flags = mode === "regex" ? uniqueRegexFlags(options.flags ?? "") : "";
    const row = this.rowsById.get(traceId);
    if (row == null || pattern.length === 0) {
      return emptySearchResult(traceId, pattern, mode, flags, limits);
    }
    if (pattern.length > limits.maxPatternLength) {
      return emptySearchResult(traceId, pattern, mode, flags, limits, {
        code: "pattern_too_long",
        message: `search pattern exceeds ${limits.maxPatternLength} characters`,
      });
    }
    let regex: RegExp | undefined;
    if (mode === "regex") {
      if (!REGEX_FLAG_ALLOWLIST.test(flags)) {
        return emptySearchResult(traceId, pattern, mode, flags, limits, {
          code: "unsupported_regex_flags",
          message: "regex flags must be drawn from i, m, s, and u",
        });
      }
      if (hasUnsafeRegexShape(pattern)) {
        return emptySearchResult(traceId, pattern, mode, flags, limits, {
          code: "unsafe_regex",
          message: "regex contains a nested quantified expression that is not allowed for bounded trace search",
        });
      }
      try {
        regex = new RegExp(pattern, flags);
      } catch (error) {
        return emptySearchResult(traceId, pattern, mode, flags, limits, {
          code: "invalid_regex",
          message: error instanceof Error ? error.message : "invalid regex pattern",
        });
      }
    }
    const matches: HaloSpan[] = [];
    const contexts: TraceSearchContext[] = [];
    let matchCount = 0;
    let truncated = false;
    for (let index = 0; index < row.byteOffsets.length; index += 1) {
      const offset = row.byteOffsets[index] ?? 0;
      const length = row.byteLengths[index] ?? 0;
      if (length > limits.maxSpanSearchChars) {
        truncated = true;
      }
      const raw = this.source.subarray(offset, offset + Math.min(length, limits.maxSpanSearchChars)).toString("utf8");
      const found = regex == null ? raw.includes(pattern) : regex.test(raw);
      if (regex != null) {
        regex.lastIndex = 0;
      }
      if (!found) {
        continue;
      }
      matchCount += 1;
      if (matches.length < limits.maxReturned) {
        const parsed = JSON.parse(this.source.subarray(offset, offset + length).toString("utf8")) as HaloSpan;
        matches.push(sanitizeSpanForOutput(parsed, DISCOVERY_ATTR_CAP));
        const context = regex == null ? literalContext(raw, pattern, limits.contextChars) : regexContext(raw, regex, limits.contextChars);
        if (context != null) {
          contexts.push({ spanId: parsed.span_id, spanName: parsed.name, ...context });
        }
      }
      if (matchCount >= limits.maxMatchCount) {
        truncated = index < row.byteOffsets.length - 1 || matchCount > matches.length;
        break;
      }
    }
    const hasMore = matchCount > matches.length || truncated;
    return {
      traceId,
      pattern,
      mode,
      flags,
      matches,
      matchCount,
      returnedCount: matches.length,
      hasMore,
      contexts,
      limits,
      truncated: hasMore,
    };
  }

  searchSpan(
    traceId: string,
    spanIds: string | string[],
    pattern: string,
    options: TraceSpanSearchOptions = {},
  ): TraceSpanSearchResult {
    const requestedSpanIds = [...new Set(Array.isArray(spanIds) ? spanIds : [spanIds])];
    const mode = options.mode ?? "literal";
    const limits = traceSpanSearchLimits(options);
    const cappedSpanIds = requestedSpanIds.slice(0, limits.maxSpanIds);
    const flags = mode === "regex" ? uniqueRegexFlags(options.flags ?? "") : "";
    const row = this.rowsById.get(traceId);
    if (row == null || pattern.length === 0) {
      return emptySpanSearchResult(
        traceId,
        pattern,
        mode,
        flags,
        limits,
        requestedSpanIds.length,
        row == null ? cappedSpanIds : [],
        requestedSpanIds.length - cappedSpanIds.length,
      );
    }
    if (pattern.length > limits.maxPatternLength) {
      return emptySpanSearchResult(
        traceId,
        pattern,
        mode,
        flags,
        limits,
        requestedSpanIds.length,
        [],
        requestedSpanIds.length - cappedSpanIds.length,
        {
          code: "pattern_too_long",
          message: `search pattern exceeds ${limits.maxPatternLength} characters`,
        },
      );
    }
    let regex: RegExp | undefined;
    if (mode === "regex") {
      if (!REGEX_FLAG_ALLOWLIST.test(flags)) {
        return emptySpanSearchResult(
          traceId,
          pattern,
          mode,
          flags,
          limits,
          requestedSpanIds.length,
          [],
          requestedSpanIds.length - cappedSpanIds.length,
          {
            code: "unsupported_regex_flags",
            message: "regex flags must be drawn from i, m, s, and u",
          },
        );
      }
      if (hasUnsafeRegexShape(pattern)) {
        return emptySpanSearchResult(
          traceId,
          pattern,
          mode,
          flags,
          limits,
          requestedSpanIds.length,
          [],
          requestedSpanIds.length - cappedSpanIds.length,
          {
            code: "unsafe_regex",
            message: "regex contains a nested quantified expression that is not allowed for bounded trace search",
          },
        );
      }
      try {
        regex = new RegExp(pattern, flags);
      } catch (error) {
        return emptySpanSearchResult(
          traceId,
          pattern,
          mode,
          flags,
          limits,
          requestedSpanIds.length,
          [],
          requestedSpanIds.length - cappedSpanIds.length,
          {
            code: "invalid_regex",
            message: error instanceof Error ? error.message : "invalid regex pattern",
          },
        );
      }
    }

    const wanted = new Set(cappedSpanIds);
    const selected: Array<{ span: HaloSpan; offset: number; length: number }> = [];
    for (let index = 0; index < row.byteOffsets.length; index += 1) {
      const offset = row.byteOffsets[index] ?? 0;
      const length = row.byteLengths[index] ?? 0;
      try {
        const span = JSON.parse(this.source.subarray(offset, offset + length).toString("utf8")) as HaloSpan;
        if (wanted.has(span.span_id)) {
          selected.push({ span, offset, length });
        }
      } catch {
        // Ignore corrupt rows while preserving bounded, best-effort trace inspection.
      }
    }

    const foundSpanIds = new Set(selected.map(({ span }) => span.span_id));
    const missingSpanIds = cappedSpanIds.filter((spanId) => !foundSpanIds.has(spanId));
    const matches: HaloSpan[] = [];
    const contexts: TraceSearchContext[] = [];
    let matchCount = 0;
    let searchedSpanCount = 0;
    let searchedBytes = 0;
    let truncated = requestedSpanIds.length > cappedSpanIds.length;

    for (const selectedSpan of selected) {
      if (searchedBytes >= limits.maxSearchBytes) {
        truncated = true;
        break;
      }
      const remainingBytes = limits.maxSearchBytes - searchedBytes;
      const searchableBytes = Math.min(selectedSpan.length, limits.maxSpanSearchChars, remainingBytes);
      if (searchableBytes < selectedSpan.length) {
        truncated = true;
      }
      searchedBytes += searchableBytes;
      searchedSpanCount += 1;
      const raw = this.source.subarray(selectedSpan.offset, selectedSpan.offset + searchableBytes).toString("utf8");
      const found = regex == null ? raw.includes(pattern) : regex.test(raw);
      if (regex != null) {
        regex.lastIndex = 0;
      }
      if (!found) {
        continue;
      }
      matchCount += 1;
      if (matches.length < limits.maxReturned) {
        matches.push(sanitizeSpanForOutput(selectedSpan.span, limits.attrCap));
        const context =
          regex == null ? literalContext(raw, pattern, limits.contextChars) : regexContext(raw, regex, limits.contextChars);
        if (context != null) {
          contexts.push({ spanId: selectedSpan.span.span_id, spanName: selectedSpan.span.name, ...context });
        }
      }
      if (matchCount >= limits.maxMatchCount) {
        truncated = true;
        break;
      }
    }

    const omittedSpanCount = requestedSpanIds.length - cappedSpanIds.length + selected.length - searchedSpanCount;
    const hasMore = matchCount > matches.length || truncated || omittedSpanCount > 0;
    return {
      traceId,
      pattern,
      mode,
      flags,
      matches,
      matchCount,
      returnedCount: matches.length,
      hasMore,
      contexts,
      limits,
      truncated: hasMore,
      requestedSpanCount: requestedSpanIds.length,
      searchedSpanCount,
      missingSpanIds,
      omittedSpanCount,
    };
  }

  private matchingRows(filters: TraceFilters): TraceIndexRow[] {
    return [...this.rowsById.values()].filter((row) => matchesFilters(row, filters));
  }

  private readTraceSpans(traceId: string): HaloSpan[] {
    const row = this.rowsById.get(traceId);
    if (row == null) {
      return [];
    }
    return row.byteOffsets.flatMap((offset, index) => {
      const length = row.byteLengths[index] ?? 0;
      try {
        return [JSON.parse(this.source.subarray(offset, offset + length).toString("utf8")) as HaloSpan];
      } catch {
        return [];
      }
    });
  }
}
