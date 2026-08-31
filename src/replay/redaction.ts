import { createHash } from "node:crypto";
import { isAbsolute, relative } from "node:path";
import {
  redactSourceRecord,
  type SourceAdapterRedactionKind,
  type SourceAdapterRedactionMetadata,
} from "../source-adapters/redaction";
import {
  EditAttemptContractSchema,
  type EditAttemptContract,
  type RedactionStatus,
} from "../edit-strategy/types";
import { JsonValueSchema, type JsonValue } from "../optimizer/types";
import {
  AcpReplayCaptureSchema,
  type AcpReplayCapture,
  type AcpReplayCaptureInput,
  type AcpReplayRecord,
} from "./capture";

export type ReplayPathRedactionMode = "relative_or_hash" | "hash" | "preserve";

export type ReplayRedactionOptions = {
  rootPath?: string;
  maxTextExcerptChars?: number;
  includeRawLocalContent?: boolean;
  pathMode?: ReplayPathRedactionMode;
  hashOnlyFileReadContent?: boolean;
};

export type ReplayRedactionReport = {
  captureId: string;
  redactionStatus: RedactionStatus;
  recordsProcessed: number;
  rawLocalContentRetained: boolean;
  pathMode: ReplayPathRedactionMode;
  secretReplacementCount: number;
  redactionKinds: SourceAdapterRedactionKind[];
  truncatedStringCount: number;
  truncatedArrayCount: number;
  truncatedDepthCount: number;
  pathHashCount: number;
  hashOnlyRecordCount: number;
};

export type RedactedReplayCaptureResult = {
  capture: AcpReplayCapture;
  report: ReplayRedactionReport;
};

type ReplayRedactionAccumulator = {
  metadata: SourceAdapterRedactionMetadata[];
  pathHashCount: number;
  hashOnlyRecordCount: number;
};

const DEFAULT_MAX_TEXT_EXCERPT_CHARS = 240;

export const redactAcpReplayCaptureForLocalSafeUse = (
  captureInput: AcpReplayCaptureInput,
  options: ReplayRedactionOptions = {},
): RedactedReplayCaptureResult => {
  const capture = AcpReplayCaptureSchema.parse(captureInput);
  const includeRaw = options.includeRawLocalContent === true;
  const accumulator: ReplayRedactionAccumulator = {
    metadata: [],
    pathHashCount: 0,
    hashOnlyRecordCount: 0,
  };
  const redactionOptions = normalizeReplayRedactionOptions(options);
  const records = capture.records.map((record) =>
    redactReplayRecord(record, redactionOptions, accumulator));
  const redactionStatus = includeRaw ? "raw_local_only" : aggregateReplayRecordStatus(records);
  const redacted = AcpReplayCaptureSchema.parse({
    ...capture,
    redactionStatus,
    source: {
      ...capture.source,
      ...(capture.source.path === undefined ? {} : {
        path: sanitizeReplayPath(capture.source.path, redactionOptions, accumulator),
      }),
    },
    records,
  });
  return {
    capture: redacted,
    report: replayRedactionReport(redacted, redactionOptions, accumulator),
  };
};

export const sanitizeReplayPath = (
  value: string,
  options: Required<Pick<ReplayRedactionOptions, "pathMode">> & Pick<ReplayRedactionOptions, "rootPath">,
  accumulator?: Pick<ReplayRedactionAccumulator, "pathHashCount">,
): string => {
  if (options.pathMode === "preserve") {
    return value;
  }
  if (options.pathMode === "hash") {
    accumulator && (accumulator.pathHashCount += 1);
    return `path:sha256:${sha256(value).slice(0, 24)}`;
  }
  if (options.rootPath != null && isAbsolute(value)) {
    const rel = relative(options.rootPath, value);
    if (!rel.startsWith("..") && !isAbsolute(rel) && rel !== "") {
      return rel;
    }
  }
  if (isAbsolute(value)) {
    accumulator && (accumulator.pathHashCount += 1);
    return `path:sha256:${sha256(value).slice(0, 24)}`;
  }
  return value;
};

const normalizeReplayRedactionOptions = (
  options: ReplayRedactionOptions,
): Required<Pick<ReplayRedactionOptions, "maxTextExcerptChars" | "includeRawLocalContent" | "pathMode" | "hashOnlyFileReadContent">>
  & Pick<ReplayRedactionOptions, "rootPath"> => ({
    ...(options.rootPath === undefined ? {} : { rootPath: options.rootPath }),
    maxTextExcerptChars: options.maxTextExcerptChars ?? DEFAULT_MAX_TEXT_EXCERPT_CHARS,
    includeRawLocalContent: options.includeRawLocalContent === true,
    pathMode: options.pathMode ?? "relative_or_hash",
    hashOnlyFileReadContent: options.hashOnlyFileReadContent ?? true,
  });

const redactReplayRecord = (
  record: AcpReplayRecord,
  options: ReturnType<typeof normalizeReplayRedactionOptions>,
  accumulator: ReplayRedactionAccumulator,
): AcpReplayRecord => {
  if (options.includeRawLocalContent) {
    return record;
  }
  switch (record.recordKind) {
    case "prompt":
      return {
        ...record,
        content: redactText(record.content, options, accumulator),
        contentRedactionStatus: "redacted",
      };
    case "file_read": {
      const hashOnly = options.hashOnlyFileReadContent && record.contentHash !== undefined;
      if (hashOnly) accumulator.hashOnlyRecordCount += 1;
      return {
        ...record,
        path: sanitizeReplayPath(record.path, options, accumulator),
        ...(hashOnly ? { excerpt: undefined } : { excerpt: redactOptionalText(record.excerpt, options, accumulator) }),
        redactionStatus: hashOnly ? "hash_only" : "redacted",
      };
    }
    case "edit_attempt":
      return {
        ...record,
        attempt: redactEditAttempt(record.attempt, options, accumulator),
      };
    case "tool_call":
      return {
        ...record,
        args: redactOptionalJson(record.args, options, accumulator),
        result: redactOptionalJson(record.result, options, accumulator),
        redactionStatus: "redacted",
      };
    case "terminal_command":
      accumulator.hashOnlyRecordCount += 1;
      return {
        ...record,
        cwd: record.cwd === undefined ? undefined : sanitizeReplayPath(record.cwd, options, accumulator),
        redactionStatus: "hash_only",
      };
    case "artifact_ref":
      accumulator.hashOnlyRecordCount += 1;
      return {
        ...record,
        path: record.path === undefined ? undefined : sanitizeReplayPath(record.path, options, accumulator),
        redactionStatus: record.contentHash == null ? "redacted" : "hash_only",
      };
    case "mode_route":
      return record;
  }
};

const redactEditAttempt = (
  attempt: EditAttemptContract,
  options: ReturnType<typeof normalizeReplayRedactionOptions>,
  accumulator: ReplayRedactionAccumulator,
): EditAttemptContract => {
  const sanitizeRecord = (value: Record<string, string>): Record<string, string> =>
    Object.fromEntries(Object.entries(value).map(([key, nested]) => [
      sanitizeReplayPath(key, options, accumulator),
      nested,
    ]));
  return EditAttemptContractSchema.parse({
    ...attempt,
    targetFiles: attempt.targetFiles.map((path) => sanitizeReplayPath(path, options, accumulator)),
    readSnapshotRefs: attempt.readSnapshotRefs.map((snapshot) => ({
      ...snapshot,
      path: sanitizeReplayPath(snapshot.path, options, accumulator),
    })),
    inputContentHashes: sanitizeRecord(attempt.inputContentHashes),
    outputContentHashes: sanitizeRecord(attempt.outputContentHashes),
    targetContentHashes: attempt.targetContentHashes?.map((target) => ({
      ...target,
      path: sanitizeReplayPath(target.path, options, accumulator),
    })),
    phaseResults: attempt.phaseResults.map((phase) => ({
      ...phase,
      message: redactOptionalText(phase.message, options, accumulator),
      attributes: redactJson(phase.attributes, options, accumulator) as Record<string, JsonValue>,
    })),
    redactionStatus: "redacted",
  });
};

const redactText = (
  value: string,
  options: ReturnType<typeof normalizeReplayRedactionOptions>,
  accumulator: ReplayRedactionAccumulator,
): string => {
  const redacted = redactSourceRecord({ value }, {
    maxTextExcerptChars: options.maxTextExcerptChars,
    includeFullContent: false,
  });
  accumulator.metadata.push(redacted.redaction);
  const record = redacted.record;
  if (isRecord(record) && typeof record.value === "string") {
    return record.value;
  }
  return "";
};

const redactOptionalText = (
  value: string | undefined,
  options: ReturnType<typeof normalizeReplayRedactionOptions>,
  accumulator: ReplayRedactionAccumulator,
): string | undefined => value === undefined ? undefined : redactText(value, options, accumulator);

const redactJson = (
  value: JsonValue,
  options: ReturnType<typeof normalizeReplayRedactionOptions>,
  accumulator: ReplayRedactionAccumulator,
): JsonValue => {
  const redacted = redactSourceRecord(value, {
    maxTextExcerptChars: options.maxTextExcerptChars,
    includeFullContent: false,
  });
  accumulator.metadata.push(redacted.redaction);
  return JsonValueSchema.parse(redacted.record);
};

const redactOptionalJson = (
  value: JsonValue | undefined,
  options: ReturnType<typeof normalizeReplayRedactionOptions>,
  accumulator: ReplayRedactionAccumulator,
): JsonValue | undefined => value === undefined ? undefined : redactJson(value, options, accumulator);

const replayRedactionReport = (
  capture: AcpReplayCapture,
  options: ReturnType<typeof normalizeReplayRedactionOptions>,
  accumulator: ReplayRedactionAccumulator,
): ReplayRedactionReport => ({
  captureId: capture.captureId,
  redactionStatus: capture.redactionStatus,
  recordsProcessed: capture.records.length,
  rawLocalContentRetained: options.includeRawLocalContent,
  pathMode: options.pathMode,
  secretReplacementCount: sum(accumulator.metadata.map((metadata) => metadata.secretReplacementCount)),
  redactionKinds: uniqueSorted(accumulator.metadata.flatMap((metadata) => metadata.redactionKinds)),
  truncatedStringCount: sum(accumulator.metadata.map((metadata) => metadata.truncatedStringCount)),
  truncatedArrayCount: sum(accumulator.metadata.map((metadata) => metadata.truncatedArrayCount)),
  truncatedDepthCount: sum(accumulator.metadata.map((metadata) => metadata.truncatedDepthCount)),
  pathHashCount: accumulator.pathHashCount,
  hashOnlyRecordCount: accumulator.hashOnlyRecordCount,
});

const aggregateReplayRecordStatus = (records: readonly AcpReplayRecord[]): RedactionStatus => {
  const statuses = records.map((record): RedactionStatus | undefined => {
    switch (record.recordKind) {
      case "prompt":
        return record.contentRedactionStatus;
      case "file_read":
      case "tool_call":
      case "terminal_command":
      case "artifact_ref":
        return record.redactionStatus;
      case "edit_attempt":
        return record.attempt.redactionStatus;
      case "mode_route":
        return undefined;
    }
  }).filter((status): status is RedactionStatus => status !== undefined);
  if (statuses.includes("redacted")) return "redacted";
  if (statuses.includes("hash_only")) return "hash_only";
  if (statuses.includes("omitted")) return "omitted";
  return "redacted";
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === "object" && !Array.isArray(value);

const sha256 = (value: string): string => createHash("sha256").update(value).digest("hex");

const uniqueSorted = <T extends string>(values: readonly T[]): T[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const sum = (values: readonly number[]): number => values.reduce((total, value) => total + value, 0);
