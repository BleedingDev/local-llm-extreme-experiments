import { createReadStream } from "node:fs";
import { opendir, stat } from "node:fs/promises";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import {
  detectSourceRecords,
  type SourceDetectionDiagnostic,
  type SourceDetectionResult,
  type SourceMetadata,
} from "./boundary";
import {
  redactSourceRecord,
  type RedactedSourceRecord,
  type SourceAdapterRedactionOptions,
} from "./redaction";

export type SourceAdapterStreamingDiagnosticCode =
  | "directory_entry_skipped"
  | "file_too_large"
  | "malformed_jsonl"
  | "max_files_reached"
  | "max_records_reached"
  | "non_object_record"
  | "source_detection_failed";

export type SourceAdapterStreamingDiagnostic = {
  code: SourceAdapterStreamingDiagnosticCode;
  message: string;
  path?: string;
  line?: number;
  recordIndex?: number;
  limit?: number;
  cause?: SourceDetectionDiagnostic;
};

export type StreamedJsonlRecord = {
  kind: "record";
  path: string;
  line: number;
  recordIndex: number;
  record: unknown;
};

export type StreamedSourceRecord = {
  kind: "record";
  path: string;
  line: number;
  recordIndex: number;
  source: SourceMetadata;
  record: unknown;
  redacted?: RedactedSourceRecord;
};

export type StreamedDiagnostic = {
  kind: "diagnostic";
  diagnostic: SourceAdapterStreamingDiagnostic;
};

export type StreamedJsonlItem = StreamedJsonlRecord | StreamedDiagnostic;
export type StreamedSourceItem = StreamedSourceRecord | StreamedDiagnostic;

export type SourceFileDiscovery = {
  kind: "file";
  path: string;
  detection: SourceDetectionResult;
};

export type SourceFileDiscoveryItem = SourceFileDiscovery | StreamedDiagnostic;

export type StreamJsonlFileOptions = {
  maxFileBytes?: number;
  maxRecords?: number;
  requireObjectRecords?: boolean;
};

export type SourceDetectionFileOptions = {
  maxFileBytes?: number;
  maxInspectionRecords?: number;
};

export type StreamSourceFileOptions = StreamJsonlFileOptions & SourceDetectionFileOptions & {
  redact?: boolean;
  redactionOptions?: Omit<SourceAdapterRedactionOptions, "source">;
};

export type DiscoverSourceFilesOptions = SourceDetectionFileOptions & {
  recursive?: boolean;
  maxFiles?: number;
  includeUnsupported?: boolean;
};

export type StreamSourceDirectoryOptions = StreamSourceFileOptions & DiscoverSourceFilesOptions;

const DEFAULT_MAX_FILE_BYTES = 512 * 1024 * 1024;
const DEFAULT_MAX_INSPECTION_RECORDS = 32;
const JSONL_EXTENSIONS = new Set([".jsonl", ".ndjson", ".log"]);

export async function* streamJsonlFile(
  path: string,
  options: StreamJsonlFileOptions = {},
): AsyncGenerator<StreamedJsonlItem> {
  const fileTooLarge = await maxFileSizeDiagnostic(path, options.maxFileBytes);
  if (fileTooLarge != null) {
    yield { kind: "diagnostic", diagnostic: fileTooLarge };
    return;
  }

  let recordIndex = 0;
  for await (const line of readLines(path)) {
    if (line.text.trim() === "") {
      continue;
    }

    let record: unknown;
    try {
      record = JSON.parse(line.text) as unknown;
    } catch (error) {
      yield {
        kind: "diagnostic",
        diagnostic: {
          code: "malformed_jsonl",
          message: `Malformed JSONL at line ${line.number}: ${error instanceof Error ? error.message : String(error)}`,
          path,
          line: line.number,
        },
      };
      continue;
    }

    if ((options.requireObjectRecords ?? true) && !isObject(record)) {
      yield {
        kind: "diagnostic",
        diagnostic: {
          code: "non_object_record",
          message: "Source adapter streaming only accepts JSON object records.",
          path,
          line: line.number,
          recordIndex,
        },
      };
      recordIndex += 1;
      continue;
    }

    if (options.maxRecords != null && recordIndex >= options.maxRecords) {
      yield {
        kind: "diagnostic",
        diagnostic: {
          code: "max_records_reached",
          message: `Stopped reading ${path} after maxRecords=${options.maxRecords}.`,
          path,
          limit: options.maxRecords,
        },
      };
      return;
    }

    yield {
      kind: "record",
      path,
      line: line.number,
      recordIndex,
      record,
    };
    recordIndex += 1;
  }
}

export const detectSourceFile = async (
  path: string,
  options: SourceDetectionFileOptions = {},
): Promise<SourceDetectionResult> => {
  const fileTooLarge = await maxFileSizeDiagnostic(path, options.maxFileBytes);
  if (fileTooLarge != null) {
    return {
      ok: false,
      path,
      inspectedRecordCount: 0,
      diagnostics: [{
        code: "unknown_source_shape",
        message: fileTooLarge.message,
      }],
    };
  }

  const maxInspectionRecords = options.maxInspectionRecords ?? DEFAULT_MAX_INSPECTION_RECORDS;
  const records: unknown[] = [];
  let inspectedLineCount = 0;
  const streamOptions: StreamJsonlFileOptions = {
    maxRecords: maxInspectionRecords,
    requireObjectRecords: false,
  };
  if (options.maxFileBytes != null) {
    streamOptions.maxFileBytes = options.maxFileBytes;
  }

  for await (const item of streamJsonlFile(path, streamOptions)) {
    if (item.kind === "diagnostic") {
      return detectionFailure(path, inspectedLineCount, item.diagnostic);
    }
    inspectedLineCount += 1;
    records.push(item.record);
    if (records.length >= maxInspectionRecords) {
      break;
    }
  }

  return detectSourceRecords(records, { path, maxInspectionRecords }, undefined, {
    value: records.length,
    kind: "sample",
  });
};

export async function* streamSourceFile(
  path: string,
  options: StreamSourceFileOptions = {},
): AsyncGenerator<StreamedSourceItem> {
  const detection = await detectSourceFile(path, options);
  if (!detection.ok) {
    for (const cause of detection.diagnostics) {
      const diagnostic: SourceAdapterStreamingDiagnostic = {
        code: "source_detection_failed",
        message: cause.message,
        path,
        cause,
      };
      if (cause.line != null) {
        diagnostic.line = cause.line;
      }
      if (cause.recordIndex != null) {
        diagnostic.recordIndex = cause.recordIndex;
      }
      yield {
        kind: "diagnostic",
        diagnostic,
      };
    }
    return;
  }

  for await (const item of streamJsonlFile(path, options)) {
    if (item.kind === "diagnostic") {
      yield item;
      continue;
    }

    let record = item.record;
    let redacted: RedactedSourceRecord | undefined;
    if (options.redact === true) {
      redacted = redactSourceRecord(item.record, {
        ...options.redactionOptions,
        source: detection.source,
      });
      record = redacted.record;
    }

    const output: StreamedSourceRecord = {
      ...item,
      record,
      source: detection.source,
    };
    if (redacted != null) {
      output.redacted = redacted;
    }
    yield output;
  }
}

export async function* discoverSourceFiles(
  rootPath: string,
  options: DiscoverSourceFilesOptions = {},
): AsyncGenerator<SourceFileDiscoveryItem> {
  let yieldedFiles = 0;
  for await (const path of walkCandidateFiles(rootPath, options.recursive ?? true)) {
    if (options.maxFiles != null && yieldedFiles >= options.maxFiles) {
      yield {
        kind: "diagnostic",
        diagnostic: {
          code: "max_files_reached",
          message: `Stopped directory discovery after maxFiles=${options.maxFiles}.`,
          path: rootPath,
          limit: options.maxFiles,
        },
      };
      return;
    }

    const detection = await detectSourceFile(path, options);
    if (detection.ok || options.includeUnsupported === true) {
      yield { kind: "file", path, detection };
      yieldedFiles += 1;
      continue;
    }

    yield {
      kind: "diagnostic",
      diagnostic: {
        code: "source_detection_failed",
        message: detection.diagnostics.map((diagnostic) => diagnostic.message).join("; "),
        path,
      },
    };
  }
}

export async function* streamSourceDirectory(
  rootPath: string,
  options: StreamSourceDirectoryOptions = {},
): AsyncGenerator<StreamedSourceItem> {
  for await (const item of discoverSourceFiles(rootPath, options)) {
    if (item.kind === "diagnostic") {
      yield item;
      continue;
    }
    if (!item.detection.ok) {
      continue;
    }
    for await (const sourceItem of streamSourceFile(item.path, options)) {
      yield sourceItem;
    }
  }
}

async function* readLines(path: string): AsyncGenerator<{ number: number; text: string }> {
  const stream = createReadStream(path, { encoding: "utf8" });
  const lines = createInterface({
    input: stream,
    crlfDelay: Infinity,
  });

  let number = 0;
  try {
    for await (const text of lines) {
      number += 1;
      yield { number, text };
    }
  } finally {
    lines.close();
    stream.destroy();
  }
}

async function* walkCandidateFiles(rootPath: string, recursive: boolean): AsyncGenerator<string> {
  const resolvedRoot = resolve(rootPath);
  const rootStats = await stat(resolvedRoot);
  if (rootStats.isFile()) {
    if (isCandidateJsonlFile(resolvedRoot)) {
      yield resolvedRoot;
    }
    return;
  }

  const directory = await opendir(resolvedRoot);
  try {
    for await (const entry of directory) {
      const entryPath = resolve(resolvedRoot, entry.name);
      if (entry.isFile()) {
        if (isCandidateJsonlFile(entryPath)) {
          yield entryPath;
        }
        continue;
      }
      if (recursive && entry.isDirectory()) {
        yield* walkCandidateFiles(entryPath, true);
      }
    }
  } finally {
    try {
      const closeResult = directory.close();
      if (closeResult != null) {
        await closeResult;
      }
    } catch {
      // Directory handles are auto-closed when their async iterator completes.
    }
  }
}

const maxFileSizeDiagnostic = async (
  path: string,
  maxFileBytes = DEFAULT_MAX_FILE_BYTES,
): Promise<SourceAdapterStreamingDiagnostic | undefined> => {
  const fileStats = await stat(path);
  if (fileStats.size <= maxFileBytes) {
    return undefined;
  }
  return {
    code: "file_too_large",
    message: `Skipped ${path} because size ${fileStats.size} exceeds maxFileBytes=${maxFileBytes}.`,
    path,
    limit: maxFileBytes,
  };
};

const detectionFailure = (
  path: string,
  inspectedRecordCount: number,
  diagnostic: SourceAdapterStreamingDiagnostic,
): SourceDetectionResult => {
  const sourceDiagnostic: SourceDetectionDiagnostic = {
    code: diagnostic.code === "malformed_jsonl" ? "malformed_jsonl" : "non_object_record",
    message: diagnostic.message,
  };
  if (diagnostic.line != null) {
    sourceDiagnostic.line = diagnostic.line;
  }
  if (diagnostic.recordIndex != null) {
    sourceDiagnostic.recordIndex = diagnostic.recordIndex;
  }

  return {
    ok: false,
    path,
    inspectedRecordCount,
    diagnostics: [sourceDiagnostic],
  };
};

const isCandidateJsonlFile = (path: string): boolean => {
  const normalized = path.toLowerCase();
  for (const extension of JSONL_EXTENSIONS) {
    if (normalized.endsWith(extension)) {
      return true;
    }
  }
  return false;
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value != null && !Array.isArray(value);
