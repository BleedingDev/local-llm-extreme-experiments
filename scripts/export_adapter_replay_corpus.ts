#!/usr/bin/env -S node --loader=tsx
import { createHash } from "node:crypto";
import { mkdir, readdir, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, extname, join, relative, resolve } from "node:path";
import process from "node:process";
import { canonicalizeSourceRecord } from "../src/source-adapters/canonical";
import { detectSourceFile, streamSourceFile } from "../src/source-adapters/streaming";
import { exportCanonicalSourceRecordsToReplayCase } from "../src/source-adapters/replay";
import type { EvalSplit } from "../src/eval-harness/types";
import type { ReplayRedactionReport } from "../src/replay";
import type { SourceAdapterCanonicalDiagnostic, CanonicalSourceRecord } from "../src/source-adapters/canonical";
import type { SourceAdapterType, SourceMetadata } from "../src/source-adapters/boundary";
import type { SourceAdapterStreamingDiagnostic } from "../src/source-adapters/streaming";

export type AdapterReplayExportOptions = {
  roots: string[];
  outDir: string;
  limit: number;
  minDistinctSessions: number;
  maxCandidateFiles: number;
  maxFileBytes: number;
  maxRecordsPerSession: number;
  maxTextExcerptChars: number;
  splitPattern: readonly EvalSplit[];
  rootPath: string;
};

export type AdapterReplayExportedSession = {
  sessionKey: string;
  sourceType: SourceAdapterType;
  sourceSessionId?: string;
  sourcePathHash: string;
  capturePath: string;
  replayCasePath: string;
  split: EvalSplit;
  captureId: string;
  evalCaseId: string;
  recordCount: number;
  replayRecordCount: number;
  canonicalDiagnosticCount: number;
  redactionReport: ReplayRedactionReport;
};

export type AdapterReplayRejectedSession = {
  pathHash: string;
  sourceType?: SourceAdapterType;
  sourceSessionId?: string;
  reason: string;
  diagnosticCount: number;
  canonicalRecordCount: number;
};

export type AdapterReplayExportManifest = {
  schemaVersion: "adapter-replay-export-manifest.v1";
  createdAt: string;
  status: "complete" | "blocked";
  blocker?: {
    code: "insufficient_safe_sessions";
    message: string;
    safeDistinctSessionCount: number;
    requiredDistinctSessionCount: number;
  };
  reproductionCommand: string;
  options: {
    roots: string[];
    outDir: string;
    limit: number;
    minDistinctSessions: number;
    maxCandidateFiles: number;
    maxFileBytes: number;
    maxRecordsPerSession: number;
    maxTextExcerptChars: number;
    splitPattern: EvalSplit[];
  };
  discovery: {
    candidateFileCount: number;
    detectedSourceFileCount: number;
    exportedSessionCount: number;
    rejectedSessionCount: number;
    distinctSessionCount: number;
  };
  counts: {
    bySourceKind: Record<string, number>;
    bySplit: Record<string, number>;
    redaction: {
      recordsProcessed: number;
      secretReplacementCount: number;
      truncatedStringCount: number;
      truncatedArrayCount: number;
      truncatedDepthCount: number;
      pathHashCount: number;
      hashOnlyRecordCount: number;
      rawLocalContentRetained: boolean;
      redactionKinds: string[];
      statuses: Record<string, number>;
    };
  };
  exportedSessions: AdapterReplayExportedSession[];
  rejectedSessions: AdapterReplayRejectedSession[];
};

type CandidateFile = {
  path: string;
  mtimeMs: number;
};

type CliArgs = {
  roots: string[];
  outDir?: string;
  limit?: number;
  minDistinctSessions?: number;
  maxCandidateFiles?: number;
  maxFileBytes?: number;
  maxRecordsPerSession?: number;
  maxTextExcerptChars?: number;
};

const DEFAULT_OUT_DIR = ".bag/replay-corpus/source-adapters/adapter-replay-export";
const DEFAULT_LIMIT = 50;
const DEFAULT_MIN_DISTINCT_SESSIONS = 50;
const DEFAULT_MAX_CANDIDATE_FILES = 5000;
const DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024;
const DEFAULT_MAX_RECORDS_PER_SESSION = 500;
const DEFAULT_MAX_TEXT_EXCERPT_CHARS = 240;
const DEFAULT_SPLIT_PATTERN: readonly EvalSplit[] = [
  "train",
  "train",
  "train",
  "dev",
  "holdout",
];
const JSONL_EXTENSIONS = new Set([".jsonl", ".ndjson", ".log"]);

export const defaultAdapterReplayRoots = (): string[] => [
  join(homedir(), ".codex", "sessions"),
  join(homedir(), ".codex", "archived_sessions"),
  join(homedir(), ".claude", "projects"),
  join(homedir(), ".claude", "transcripts"),
];

export const parseAdapterReplayExportArgs = (argv: readonly string[]): AdapterReplayExportOptions => {
  const parsed: CliArgs = { roots: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--root") {
      parsed.roots.push(requiredValue(argv, ++index, arg));
    } else if (arg === "--roots") {
      parsed.roots.push(...requiredValue(argv, ++index, arg).split(",").map((value) => value.trim()).filter(Boolean));
    } else if (arg === "--out") {
      parsed.outDir = requiredValue(argv, ++index, arg);
    } else if (arg === "--limit") {
      parsed.limit = positiveInteger(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--min-distinct-sessions") {
      parsed.minDistinctSessions = positiveInteger(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--max-candidate-files") {
      parsed.maxCandidateFiles = positiveInteger(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--max-file-mb") {
      parsed.maxFileBytes = positiveInteger(requiredValue(argv, ++index, arg), arg) * 1024 * 1024;
    } else if (arg === "--max-records-per-session") {
      parsed.maxRecordsPerSession = positiveInteger(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--max-text-excerpt-chars") {
      parsed.maxTextExcerptChars = positiveInteger(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  const roots = parsed.roots.length > 0 ? parsed.roots : defaultAdapterReplayRoots();
  return {
    roots: roots.map((root) => resolve(root)),
    outDir: resolve(parsed.outDir ?? DEFAULT_OUT_DIR),
    limit: parsed.limit ?? DEFAULT_LIMIT,
    minDistinctSessions: parsed.minDistinctSessions ?? DEFAULT_MIN_DISTINCT_SESSIONS,
    maxCandidateFiles: parsed.maxCandidateFiles ?? DEFAULT_MAX_CANDIDATE_FILES,
    maxFileBytes: parsed.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES,
    maxRecordsPerSession: parsed.maxRecordsPerSession ?? DEFAULT_MAX_RECORDS_PER_SESSION,
    maxTextExcerptChars: parsed.maxTextExcerptChars ?? DEFAULT_MAX_TEXT_EXCERPT_CHARS,
    splitPattern: DEFAULT_SPLIT_PATTERN,
    rootPath: process.cwd(),
  };
};

export const exportAdapterReplayCorpus = async (
  options: AdapterReplayExportOptions,
  now: string = new Date().toISOString(),
): Promise<AdapterReplayExportManifest> => {
  const candidates = await discoverCandidateSourceFiles(options.roots, options.maxCandidateFiles);
  const exported: AdapterReplayExportedSession[] = [];
  const rejected: AdapterReplayRejectedSession[] = [];
  const seenSessionKeys = new Set<string>();
  let detectedSourceFileCount = 0;

  for (const candidate of candidates) {
    if (exported.length >= options.limit) break;
    const detection = await detectSourceFile(candidate.path, {
      maxFileBytes: options.maxFileBytes,
      maxInspectionRecords: 32,
    });
    if (!detection.ok) {
      rejected.push({
        pathHash: pathHash(candidate.path),
        reason: sanitizedReason(candidate.path, detection.diagnostics.map((diagnostic) => diagnostic.message).join("; ")),
        diagnosticCount: detection.diagnostics.length,
        canonicalRecordCount: 0,
      });
      continue;
    }
    detectedSourceFileCount += 1;

    const sessionKey = sessionKeyForSource(detection.source, candidate.path);
    if (seenSessionKeys.has(sessionKey)) {
      rejected.push({
        pathHash: pathHash(candidate.path),
        sourceType: detection.source.sourceType,
        ...(detection.source.sessionId === undefined ? {} : { sourceSessionId: detection.source.sessionId }),
        reason: "duplicate_session",
        diagnosticCount: 0,
        canonicalRecordCount: 0,
      });
      continue;
    }

    const loaded = await canonicalizeDetectedFile(candidate.path, detection.source, options);
    if (loaded.records.length === 0) {
      rejected.push({
        pathHash: pathHash(candidate.path),
        sourceType: detection.source.sourceType,
        ...(detection.source.sessionId === undefined ? {} : { sourceSessionId: detection.source.sessionId }),
        reason: sanitizedReason(
          candidate.path,
          loaded.diagnostics.map((diagnostic) => diagnostic.message).join("; ") || "no_canonical_records",
        ),
        diagnosticCount: loaded.diagnostics.length,
        canonicalRecordCount: 0,
      });
      continue;
    }
    if (loaded.streamDiagnostics.length > 0) {
      rejected.push({
        pathHash: pathHash(candidate.path),
        sourceType: detection.source.sourceType,
        ...(detection.source.sessionId === undefined ? {} : { sourceSessionId: detection.source.sessionId }),
        reason: sanitizedReason(candidate.path, loaded.streamDiagnostics.map((diagnostic) => diagnostic.message).join("; ")),
        diagnosticCount: loaded.streamDiagnostics.length,
        canonicalRecordCount: loaded.records.length,
      });
      continue;
    }

    const split = options.splitPattern[exported.length % options.splitPattern.length] ?? "dev";
    const sourceSlug = stableSlug(sessionKey);
    const exportedCase = exportCanonicalSourceRecordsToReplayCase(loaded.records, {
      captureId: `capture.source-adapter.${sourceSlug}`,
      createdAt: loaded.records[0]?.span.start_time ?? now,
      defaultSplitHint: split,
      minimumDistinctSessionCount: 1,
      redaction: {
        rootPath: options.rootPath,
        pathMode: "hash",
        maxTextExcerptChars: options.maxTextExcerptChars,
        hashOnlyFileReadContent: true,
      },
      metadata: {
        evalCaseId: `replay.eval.source-adapter.${sourceSlug}`,
        split,
        splitRationale:
          "Deterministic adapter replay corpus split assignment; observed source behavior is baseline evidence, not gold.",
      },
    });
    if (exportedCase.blocker != null) {
      rejected.push({
        pathHash: pathHash(candidate.path),
        sourceType: detection.source.sourceType,
        ...(detection.source.sessionId === undefined ? {} : { sourceSessionId: detection.source.sessionId }),
        reason: exportedCase.blocker.message,
        diagnosticCount: loaded.diagnostics.length,
        canonicalRecordCount: loaded.records.length,
      });
      continue;
    }

    seenSessionKeys.add(sessionKey);
    const capturePath = join(options.outDir, "captures", `${sourceSlug}.capture.json`);
    const replayCasePath = join(options.outDir, "cases", `${sourceSlug}.case.json`);
    await writeJson(capturePath, exportedCase.capture);
    await writeJson(replayCasePath, exportedCase.replayCase);
    exported.push({
      sessionKey,
      sourceType: detection.source.sourceType,
      ...(detection.source.sessionId === undefined ? {} : { sourceSessionId: detection.source.sessionId }),
      sourcePathHash: pathHash(candidate.path),
      capturePath: relative(options.rootPath, capturePath),
      replayCasePath: relative(options.rootPath, replayCasePath),
      split,
      captureId: exportedCase.capture.captureId,
      evalCaseId: exportedCase.replayCase.evalCaseId,
      recordCount: loaded.records.length,
      replayRecordCount: exportedCase.capture.records.length,
      canonicalDiagnosticCount: loaded.diagnostics.length,
      redactionReport: exportedCase.redactionReport,
    });
  }

  const status = exported.length >= options.minDistinctSessions ? "complete" : "blocked";
  const manifest: AdapterReplayExportManifest = {
    schemaVersion: "adapter-replay-export-manifest.v1",
    createdAt: now,
    status,
    ...(status === "complete" ? {} : {
      blocker: {
        code: "insufficient_safe_sessions",
        message:
          `Exported ${exported.length} safe distinct adapter replay sessions; ${options.minDistinctSessions} are required.`,
        safeDistinctSessionCount: exported.length,
        requiredDistinctSessionCount: options.minDistinctSessions,
      },
    }),
    reproductionCommand: reproductionCommand(options),
    options: {
      roots: options.roots,
      outDir: relative(options.rootPath, options.outDir),
      limit: options.limit,
      minDistinctSessions: options.minDistinctSessions,
      maxCandidateFiles: options.maxCandidateFiles,
      maxFileBytes: options.maxFileBytes,
      maxRecordsPerSession: options.maxRecordsPerSession,
      maxTextExcerptChars: options.maxTextExcerptChars,
      splitPattern: [...options.splitPattern],
    },
    discovery: {
      candidateFileCount: candidates.length,
      detectedSourceFileCount,
      exportedSessionCount: exported.length,
      rejectedSessionCount: rejected.length,
      distinctSessionCount: exported.length,
    },
    counts: exportCounts(exported),
    exportedSessions: exported,
    rejectedSessions: rejected,
  };

  await writeJson(join(options.outDir, "manifest.json"), manifest);
  return manifest;
};

export const discoverCandidateSourceFiles = async (
  roots: readonly string[],
  maxCandidateFiles: number,
): Promise<CandidateFile[]> => {
  const candidates: CandidateFile[] = [];
  for (const root of roots) {
    await walkCandidateFiles(root, candidates, maxCandidateFiles);
    if (candidates.length >= maxCandidateFiles) break;
  }
  return candidates
    .sort((left, right) => right.mtimeMs - left.mtimeMs || left.path.localeCompare(right.path))
    .slice(0, maxCandidateFiles);
};

const walkCandidateFiles = async (
  root: string,
  candidates: CandidateFile[],
  maxCandidateFiles: number,
): Promise<void> => {
  if (candidates.length >= maxCandidateFiles) return;
  let info;
  try {
    info = await stat(root);
  } catch {
    return;
  }
  if (info.isFile()) {
    if (JSONL_EXTENSIONS.has(extname(root)) && !isOutputPath(root)) {
      candidates.push({ path: root, mtimeMs: info.mtimeMs });
    }
    return;
  }
  if (!info.isDirectory() || shouldSkipDirectory(root)) return;

  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (candidates.length >= maxCandidateFiles) return;
    const full = join(root, entry.name);
    if (entry.isDirectory()) {
      await walkCandidateFiles(full, candidates, maxCandidateFiles);
    } else if (entry.isFile() && JSONL_EXTENSIONS.has(extname(entry.name)) && !isOutputPath(full)) {
      const fileInfo = await stat(full);
      candidates.push({ path: full, mtimeMs: fileInfo.mtimeMs });
    }
  }
};

const canonicalizeDetectedFile = async (
  path: string,
  source: SourceMetadata,
  options: AdapterReplayExportOptions,
): Promise<{
  records: CanonicalSourceRecord[];
  diagnostics: SourceAdapterCanonicalDiagnostic[];
  streamDiagnostics: SourceAdapterStreamingDiagnostic[];
}> => {
  const records: CanonicalSourceRecord[] = [];
  const diagnostics: SourceAdapterCanonicalDiagnostic[] = [];
  const streamDiagnostics: SourceAdapterStreamingDiagnostic[] = [];
  for await (const item of streamSourceFile(path, {
    maxFileBytes: options.maxFileBytes,
    maxRecords: options.maxRecordsPerSession,
    redact: true,
    redactionOptions: { maxTextExcerptChars: options.maxTextExcerptChars },
  })) {
    if (item.kind === "diagnostic") {
      streamDiagnostics.push(item.diagnostic);
      continue;
    }
    const canonical = canonicalizeSourceRecord({
      source,
      record: item.record,
      recordIndex: item.recordIndex,
      line: item.line,
      redactionOptions: { maxTextExcerptChars: options.maxTextExcerptChars },
    });
    records.push(...canonical.records);
    diagnostics.push(...canonical.diagnostics);
  }
  return { records, diagnostics, streamDiagnostics };
};

const exportCounts = (exported: readonly AdapterReplayExportedSession[]): AdapterReplayExportManifest["counts"] => {
  const redactionKinds = new Set<string>();
  const statuses = new Map<string, number>();
  const bySourceKind = new Map<string, number>();
  const bySplit = new Map<string, number>();
  const redaction = {
    recordsProcessed: 0,
    secretReplacementCount: 0,
    truncatedStringCount: 0,
    truncatedArrayCount: 0,
    truncatedDepthCount: 0,
    pathHashCount: 0,
    hashOnlyRecordCount: 0,
    rawLocalContentRetained: false,
  };

  for (const session of exported) {
    increment(bySourceKind, session.sourceType);
    increment(bySplit, session.split);
    const report = session.redactionReport;
    redaction.recordsProcessed += report.recordsProcessed;
    redaction.secretReplacementCount += report.secretReplacementCount;
    redaction.truncatedStringCount += report.truncatedStringCount;
    redaction.truncatedArrayCount += report.truncatedArrayCount;
    redaction.truncatedDepthCount += report.truncatedDepthCount;
    redaction.pathHashCount += report.pathHashCount;
    redaction.hashOnlyRecordCount += report.hashOnlyRecordCount;
    redaction.rawLocalContentRetained ||= report.rawLocalContentRetained;
    increment(statuses, report.redactionStatus);
    for (const kind of report.redactionKinds) redactionKinds.add(kind);
  }

  return {
    bySourceKind: sortedObject(bySourceKind),
    bySplit: sortedObject(bySplit),
    redaction: {
      ...redaction,
      redactionKinds: [...redactionKinds].sort(),
      statuses: sortedObject(statuses),
    },
  };
};

const reproductionCommand = (options: AdapterReplayExportOptions): string => [
  "npx tsx scripts/export_adapter_replay_corpus.ts",
  ...options.roots.flatMap((root) => ["--root", shellQuote(root)]),
  "--out",
  shellQuote(relative(options.rootPath, options.outDir)),
  "--limit",
  String(options.limit),
  "--min-distinct-sessions",
  String(options.minDistinctSessions),
  "--max-candidate-files",
  String(options.maxCandidateFiles),
  "--max-file-mb",
  String(Math.ceil(options.maxFileBytes / 1024 / 1024)),
  "--max-records-per-session",
  String(options.maxRecordsPerSession),
  "--max-text-excerpt-chars",
  String(options.maxTextExcerptChars),
].join(" ");

const writeJson = async (path: string, value: unknown): Promise<void> => {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
};

const sessionKeyForSource = (source: SourceMetadata, path: string): string =>
  `${source.sourceType}:${source.sessionId ?? pathHash(path)}`;

const shouldSkipDirectory = (path: string): boolean => {
  const name = basename(path);
  return name === ".git" || name === "node_modules" || name === ".venv" || name === "vendor_imports";
};

const isOutputPath = (path: string): boolean => path.includes(`${join(".bag", "replay-corpus")}`);

const increment = (map: Map<string, number>, key: string): void => {
  map.set(key, (map.get(key) ?? 0) + 1);
};

const sortedObject = (map: Map<string, number>): Record<string, number> =>
  Object.fromEntries([...map.entries()].sort((left, right) => left[0].localeCompare(right[0])));

const stableSlug = (value: string): string => createHash("sha256").update(value).digest("hex").slice(0, 16);

const pathHash = (value: string): string => `sha256:${stableSlug(value)}`;

const sanitizedReason = (path: string, reason: string): string =>
  reason.split(path).join(pathHash(path));

const requiredValue = (argv: readonly string[], index: number, flag: string): string => {
  const value = argv[index];
  if (value == null || value.startsWith("--")) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

const positiveInteger = (value: string, flag: string): number => {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return parsed;
};

const shellQuote = (value: string): string =>
  /^[A-Za-z0-9_./:=+-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`;

const printUsage = (): void => {
  process.stdout.write(`usage: tsx scripts/export_adapter_replay_corpus.ts [--root PATH ...] [--out PATH] [--limit N]

Exports redacted adapter replay captures/cases under an ignored local corpus path.
Default roots: ${defaultAdapterReplayRoots().join(", ")}
`);
};

const main = async (): Promise<void> => {
  const options = parseAdapterReplayExportArgs(process.argv.slice(2));
  const manifest = await exportAdapterReplayCorpus(options);
  process.stdout.write(`${JSON.stringify({
    status: manifest.status,
    exportedSessionCount: manifest.discovery.exportedSessionCount,
    rejectedSessionCount: manifest.discovery.rejectedSessionCount,
    manifestPath: relative(options.rootPath, join(options.outDir, "manifest.json")),
  }, null, 2)}\n`);
  if (manifest.status === "blocked") {
    process.exitCode = 1;
  }
};

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  });
}
