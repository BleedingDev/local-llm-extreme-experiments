import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";

export type EvidenceArtifactKind = "json" | "jsonl" | "markdown";

export type EvidenceArtifactRef = {
  path: string;
  absolutePath: string;
  kind: EvidenceArtifactKind;
  required: boolean;
  exists: boolean;
  bytes?: number;
  sha256?: string;
};

export type EvidenceCheckSeverity = "info" | "warning" | "blocking";

export type EvidenceCheck = {
  checkId: string;
  passed: boolean;
  severity: EvidenceCheckSeverity;
  message: string;
  path?: string;
};

export type EvidenceWriteIntent = {
  path: string;
  action: "none" | "would_write";
  reason: string;
};

export type ArtifactRead<T> = {
  artifact: EvidenceArtifactRef;
  checks: EvidenceCheck[];
  value?: T | undefined;
};

export const EvidenceIndexRecordSchema = z.object({
  schemaVersion: z.literal("local-evidence-index.v1"),
  recordKind: z.string().min(1),
  evidenceId: z.string().min(1).optional(),
  sliceId: z.string().min(1).optional(),
  title: z.string().min(1).optional(),
  path: z.string().min(1).optional(),
  pathKind: z.string().min(1).optional(),
  family: z.string().min(1).optional(),
  roles: z.array(z.string().min(1)).optional(),
  parentEvidenceIds: z.array(z.string().min(1)).optional(),
  memberEvidenceIds: z.array(z.string().min(1)).optional(),
  quality: z.record(z.string(), z.unknown()).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
}).passthrough().refine((record) => record.evidenceId !== undefined || record.sliceId !== undefined, {
  message: "Expected evidenceId or sliceId",
});
export type EvidenceIndexRecord = z.infer<typeof EvidenceIndexRecordSchema>;

export const ScorecardSuiteSchema = z.object({
  schemaVersion: z.literal("local-evidence-scorecard-suite.v1"),
  scorecardSuiteId: z.string().min(1),
  graphId: z.string().min(1),
  generatedAt: z.string().min(1),
  sourceIndex: z.string().min(1),
  scorecards: z.array(z.object({
    scorecardId: z.string().min(1),
    jsonPath: z.string().min(1),
    markdownPath: z.string().min(1),
    primaryUse: z.string().min(1).optional(),
  }).passthrough()).min(1),
  promotionGateInputs: z.array(z.string().min(1)).optional(),
  optimizerReadySlices: z.array(z.string().min(1)).optional(),
  caveats: z.array(z.string().min(1)).optional(),
}).passthrough();
export type ScorecardSuite = z.infer<typeof ScorecardSuiteSchema>;

export const OptimizerGateSuiteSchema = z.object({
  schemaVersion: z.literal("local-evidence-optimizer-gate-suite.v1"),
  optimizerGateSuiteId: z.string().min(1),
  graphId: z.string().min(1),
  generatedAt: z.string().min(1),
  sourceEvidenceIndex: z.string().min(1),
  sourceScorecardSuite: z.string().min(1),
  contracts: z.array(z.object({
    contractId: z.string().min(1),
    jsonPath: z.string().min(1),
    markdownPath: z.string().min(1),
    primaryUse: z.string().min(1).optional(),
  }).passthrough()).min(1),
  currentDecision: z.object({
    candidateGeneration: z.string().min(1),
    autoPromotion: z.string().min(1),
    promotionReady: z.boolean(),
    blockingReasons: z.array(z.string().min(1)).default([]),
  }).passthrough(),
  mustFailClosedOn: z.array(z.string().min(1)).default([]),
}).passthrough();
export type OptimizerGateSuite = z.infer<typeof OptimizerGateSuiteSchema>;

export const ReleaseProofSchema = z.object({
  schemaVersion: z.literal("local-evidence-release-proof.v1"),
  releaseProofId: z.string().min(1),
  graphId: z.string().min(1),
  selectionHash: z.string().min(1),
  generatedAt: z.string().min(1),
  proofMode: z.enum(["current_graph", "historical"]).optional(),
  sourceGraph: z.object({
    graphId: z.string().min(1),
    selectionHash: z.string().min(1),
    planSetHash: z.string().min(1).optional(),
    snapshotPath: z.string().min(1),
    generatedAt: z.string().min(1).optional(),
    dependencyOverlay: z.array(z.object({
      source: z.string().min(1),
      target: z.string().min(1),
    })).default([]),
    selectedPlanPaths: z.array(z.string().min(1)).default([]),
  }).optional(),
  commandOutputs: z.record(z.string(), z.object({
    ok: z.boolean(),
    summary: z.string().min(1),
    artifactCount: z.number().int().nonnegative(),
    checkCount: z.number().int().nonnegative(),
    blockingChecks: z.array(z.string().min(1)).default([]),
  })).optional(),
  artifactHashes: z.array(z.object({
    path: z.string().min(1),
    kind: z.string().min(1),
    bytes: z.number().int().nonnegative(),
    sha256: z.string().min(1),
  })).default([]),
  historicalProof: z.object({
    releaseProofId: z.string().min(1),
    graphId: z.string().min(1),
    selectionHash: z.string().min(1),
    path: z.string().min(1),
    staleForCurrentGraph: z.boolean(),
  }).optional(),
  validation: z.record(z.string(), z.string().min(1)),
  optimizerDecision: z.object({
    candidateGeneration: z.string().min(1),
    autoPromotion: z.string().min(1),
    promotionReady: z.boolean(),
    blockingReasons: z.array(z.string().min(1)).default([]),
  }).passthrough(),
  primaryOutputs: z.array(z.string().min(1)).default([]),
  nextExecutionFrontier: z.array(z.string().min(1)).default([]),
}).passthrough();
export type ReleaseProof = z.infer<typeof ReleaseProofSchema>;

export type EvidenceIndexPayload = {
  schemaVersion: "evidence-command.index.v1";
  sourcePath: string;
  recordCount: number;
  recordKinds: Record<string, number>;
  families: Record<string, number>;
  evidenceIds: string[];
  missingReferencedSourceIds: string[];
};

export type ScorecardsPayload = {
  schemaVersion: "evidence-command.scorecards.v1";
  suiteId: string;
  graphId: string;
  generatedAt: string;
  scorecardCount: number;
  scorecards: Array<{
    scorecardId: string;
    jsonPath: string;
    markdownPath: string;
    primaryUse?: string;
  }>;
  promotionGateInputCount: number;
  optimizerReadySliceCount: number;
  editAttemptProjection?: {
    projectionId: string;
    sourceRecordCount: number;
    groupCount: number;
    outputPath: string;
    byFinalOutcome: Record<string, number>;
    byFailureSignal: Record<string, number>;
  };
};

export type OptimizerGatesPayload = {
  schemaVersion: "evidence-command.optimizer-gates.v1";
  suiteId: string;
  graphId: string;
  generatedAt: string;
  contractCount: number;
  contracts: Array<{
    contractId: string;
    jsonPath: string;
    markdownPath: string;
    primaryUse?: string;
  }>;
  candidateGeneration: string;
  autoPromotion: string;
  promotionReady: boolean;
  blockingReasons: string[];
  mustFailClosedOn: string[];
};

export type ReleaseProofPayload = {
  schemaVersion: "evidence-command.release-proof.v1";
  releaseProofId: string;
  graphId: string;
  selectionHash: string;
  generatedAt: string;
  proofMode: "current_graph" | "historical";
  sourceGraph?: {
    graphId: string;
    selectionHash: string;
    planSetHash?: string;
    snapshotPath: string;
    generatedAt?: string;
    dependencyOverlay: Array<{ source: string; target: string }>;
    selectedPlanPaths: string[];
  };
  commandOutputs?: Record<string, {
    ok: boolean;
    summary: string;
    artifactCount: number;
    checkCount: number;
    blockingChecks: string[];
  }>;
  artifactHashes: Array<{
    path: string;
    kind: string;
    bytes: number;
    sha256: string;
  }>;
  historicalProof?: {
    releaseProofId: string;
    graphId: string;
    selectionHash: string;
    path: string;
    staleForCurrentGraph: boolean;
  };
  validation: Record<string, string>;
  validationPassed: boolean;
  candidateGeneration: string;
  autoPromotion: string;
  promotionReady: boolean;
  blockingReasons: string[];
  primaryOutputs: string[];
  nextExecutionFrontier: string[];
};

export type CanonicalEpochArtifactClassification =
  | "current"
  | "candidate_input"
  | "historical"
  | "stale";

export type CanonicalEpochArtifact = {
  path: string;
  kind: EvidenceArtifactKind;
  classification: CanonicalEpochArtifactClassification;
  reason: string;
  graphId?: string;
  selectionHash?: string;
  generatedAt?: string;
};

export type CanonicalEpochPayload = {
  schemaVersion: "evidence-command.epoch.v1";
  epochId: string;
  graphId: string;
  selectionHash: string;
  generatedAt: string;
  sourceGraph: {
    graphId: string;
    selectionHash: string;
    planSetHash?: string;
    snapshotPath: string;
    generatedAt?: string;
    selectedPlanPaths: string[];
  };
  planCount: number;
  graphInventory: Array<{
    graphId: string;
    selectionHash: string;
    snapshotPath: string;
    classification: "current" | "historical";
    generatedAt?: string;
  }>;
  runIds: Array<{
    runId: string;
    path: string;
    classification: "candidate_input" | "historical";
  }>;
  artifacts: CanonicalEpochArtifact[];
  driftChecks: Array<{
    checkId: string;
    status: "passed" | "failed" | "warning";
    message: string;
    path?: string;
  }>;
  driftStatus: "passed" | "blocked";
  promotionReady: false;
  currentEvidencePaths: string[];
  candidateInputPaths: string[];
  historicalContextPaths: string[];
  stalePaths: string[];
};

export type ValidatePayload = {
  schemaVersion: "evidence-command.validate.v1";
  indexRecords: number;
  epochId?: string;
  epochDriftStatus?: "passed" | "blocked";
  scorecards: number;
  optimizerContracts: number;
  releaseProofValidationPassed: boolean;
  promotionReady: boolean;
  blockingReasons: string[];
};

export const evidencePath = (cwd: string, relativePath: string): string => resolve(cwd, relativePath);

export const artifactRef = (
  cwd: string,
  path: string,
  kind: EvidenceArtifactKind,
  required = true,
): EvidenceArtifactRef => {
  const absolutePath = evidencePath(cwd, path);
  if (!existsSync(absolutePath)) {
    return {
      path,
      absolutePath,
      kind,
      required,
      exists: false,
    };
  }
  const stats = statSync(absolutePath);
  const content = readFileSync(absolutePath);
  return {
    path,
    absolutePath,
    kind,
    required,
    exists: true,
    bytes: stats.size,
    sha256: createHash("sha256").update(content).digest("hex"),
  };
};

export const readJsonArtifact = <T>(
  cwd: string,
  path: string,
  schema: z.ZodType<T>,
  checkId: string,
): ArtifactRead<T> => {
  const artifact = artifactRef(cwd, path, "json");
  if (!artifact.exists) {
    return {
      artifact,
      checks: [blocking(checkId, `Missing required JSON artifact: ${path}`, path)],
    };
  }

  try {
    const parsed = JSON.parse(readFileSync(artifact.absolutePath, "utf8")) as unknown;
    const value = schema.parse(parsed);
    return {
      artifact,
      value,
      checks: [passed(checkId, `Validated JSON artifact: ${path}`, path)],
    };
  } catch (error) {
    return {
      artifact,
      checks: [blocking(checkId, `Invalid JSON artifact ${path}: ${errorMessage(error)}`, path)],
    };
  }
};

export const readJsonlArtifact = <T>(
  cwd: string,
  path: string,
  schema: z.ZodType<T>,
  checkId: string,
): ArtifactRead<T[]> => {
  const artifact = artifactRef(cwd, path, "jsonl");
  if (!artifact.exists) {
    return {
      artifact,
      checks: [blocking(checkId, `Missing required JSONL artifact: ${path}`, path)],
    };
  }

  const records: T[] = [];
  const checks: EvidenceCheck[] = [];
  const lines = readFileSync(artifact.absolutePath, "utf8").split(/\r?\n/);
  for (const [index, line] of lines.entries()) {
    if (line.trim() === "") continue;
    try {
      records.push(schema.parse(JSON.parse(line) as unknown));
    } catch (error) {
      checks.push(blocking(`${checkId}.line-${index + 1}`, `Invalid JSONL record ${path}:${index + 1}: ${errorMessage(error)}`, path));
    }
  }

  if (checks.length === 0) {
    checks.push(passed(checkId, `Validated ${records.length} JSONL records: ${path}`, path));
  }
  return {
    artifact,
    value: checks.some((check) => !check.passed && check.severity === "blocking") ? undefined : records,
    checks,
  };
};

export const markdownArtifact = (cwd: string, path: string, checkId: string): ArtifactRead<undefined> => {
  const artifact = artifactRef(cwd, path, "markdown");
  return {
    artifact,
    checks: [
      artifact.exists
        ? passed(checkId, `Found markdown artifact: ${path}`, path)
        : blocking(checkId, `Missing required markdown artifact: ${path}`, path),
    ],
  };
};

export const passed = (checkId: string, message: string, path?: string): EvidenceCheck => ({
  checkId,
  passed: true,
  severity: "info",
  message,
  ...(path === undefined ? {} : { path }),
});

export const warning = (checkId: string, message: string, path?: string): EvidenceCheck => ({
  checkId,
  passed: false,
  severity: "warning",
  message,
  ...(path === undefined ? {} : { path }),
});

export const blocking = (checkId: string, message: string, path?: string): EvidenceCheck => ({
  checkId,
  passed: false,
  severity: "blocking",
  message,
  ...(path === undefined ? {} : { path }),
});

export const hasBlockingFailure = (checks: readonly EvidenceCheck[]): boolean =>
  checks.some((check) => !check.passed && check.severity === "blocking");

export const countBy = <T>(values: readonly T[], keyFor: (value: T) => string | undefined): Record<string, number> => {
  const counts: Record<string, number> = {};
  for (const value of values) {
    const key = keyFor(value);
    if (key === undefined || key === "") continue;
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
};

export const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values.filter((value) => value !== ""))].sort((left, right) => left.localeCompare(right));

export const noWriteIntent = (path: string, reason: string): EvidenceWriteIntent => ({
  path,
  action: "none",
  reason,
});

export const wouldWriteIntent = (path: string, reason: string): EvidenceWriteIntent => ({
  path,
  action: "would_write",
  reason,
});

const errorMessage = (error: unknown): string => error instanceof Error ? error.message : String(error);
