import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";
import { z } from "zod";
import type { BagConfig } from "../types";
import { optimizerRegistryCheckpointsDir, stableRegistryJson } from "./registry";
import { GepaRunnerStateSchema, type GepaRunnerState } from "./gepa-runner";

const GEPA_CHECKPOINT_SCHEMA_VERSION = "gepa-checkpoint.v1";
const GEPA_CHECKPOINT_KIND = "gepa-runner-state";
const GEPA_CHECKPOINT_DIR = "gepa-runs";

export const GepaCheckpointDiagnosticSchema = z.object({
  path: z.string(),
  kind: z.enum(["read_error", "parse_error", "validation_error"]),
  message: z.string(),
}).strict();
export type GepaCheckpointDiagnostic = z.infer<typeof GepaCheckpointDiagnosticSchema>;

export const GepaRunnerCheckpointSchema = z.object({
  schemaVersion: z.literal(GEPA_CHECKPOINT_SCHEMA_VERSION),
  checkpointKind: z.literal(GEPA_CHECKPOINT_KIND),
  checkpointId: z.string().min(1),
  runId: z.string().min(1),
  feedbackBundleId: z.string().min(1),
  savedAt: z.string().min(1),
  runnerCreatedAt: z.string().min(1),
  runnerUpdatedAt: z.string().min(1),
  iterationCount: z.number().int().nonnegative(),
  candidateCount: z.number().int().nonnegative(),
  validationCount: z.number().int().nonnegative(),
  diagnosticCount: z.number().int().nonnegative(),
  exhausted: z.boolean(),
  stateContentHash: z.string().min(1),
  state: GepaRunnerStateSchema,
}).strict();
export type GepaRunnerCheckpoint = z.infer<typeof GepaRunnerCheckpointSchema>;

export type GepaCheckpointSummary = Omit<GepaRunnerCheckpoint, "state"> & {
  path: string;
};

export interface SaveGepaRunnerCheckpointInput {
  config: BagConfig;
  state: GepaRunnerState;
  cwd?: string;
  savedAt?: string;
}

export interface SaveGepaRunnerCheckpointResult {
  path: string;
  checkpoint: GepaRunnerCheckpoint;
}

export interface LoadGepaRunnerCheckpointResult {
  path: string;
  checkpoint?: GepaRunnerCheckpoint;
  state?: GepaRunnerState;
  errors: GepaCheckpointDiagnostic[];
}

export interface ListGepaRunnerCheckpointsInput {
  config: BagConfig;
  cwd?: string;
  runId?: string;
}

export interface ListGepaRunnerCheckpointsResult {
  dir: string;
  checkpoints: GepaCheckpointSummary[];
  invalidCheckpoints: GepaCheckpointDiagnostic[];
}

export const gepaRunnerCheckpointsRoot = (config: BagConfig, cwd = process.cwd()): string =>
  join(optimizerRegistryCheckpointsDir(config, cwd), GEPA_CHECKPOINT_DIR);

export const gepaRunnerCheckpointRunDir = (config: BagConfig, runId: string, cwd = process.cwd()): string =>
  join(gepaRunnerCheckpointsRoot(config, cwd), safePathSegment(runId));

export const saveGepaRunnerCheckpoint = (
  input: SaveGepaRunnerCheckpointInput,
): SaveGepaRunnerCheckpointResult => {
  const state = GepaRunnerStateSchema.parse(input.state);
  const savedAt = input.savedAt ?? new Date().toISOString();
  const checkpoint = checkpointForState(state, savedAt);
  const runDir = gepaRunnerCheckpointRunDir(input.config, state.runId, input.cwd);
  const path = join(runDir, `${safePathSegment(checkpoint.savedAt)}.${safePathSegment(checkpoint.checkpointId)}.json`);
  atomicWriteJson(path, checkpoint);
  return { path, checkpoint };
};

export const loadGepaRunnerCheckpoint = (path: string): LoadGepaRunnerCheckpointResult => {
  const loaded = loadJsonFile(path);
  if ("error" in loaded) {
    return {
      path,
      errors: [loaded.error],
    };
  }

  const parsed = GepaRunnerCheckpointSchema.safeParse(loaded.value);
  if (!parsed.success) {
    return {
      path,
      errors: [{
        path,
        kind: "validation_error",
        message: zodErrorMessage(parsed.error),
      }],
    };
  }

  const expectedHash = hashStableValue(parsed.data.state);
  if (parsed.data.stateContentHash !== expectedHash) {
    return {
      path,
      errors: [{
        path,
        kind: "validation_error",
        message: `stateContentHash mismatch: expected ${expectedHash}`,
      }],
    };
  }

  return {
    path,
    checkpoint: parsed.data,
    state: parsed.data.state,
    errors: [],
  };
};

export const listGepaRunnerCheckpoints = (
  input: ListGepaRunnerCheckpointsInput,
): ListGepaRunnerCheckpointsResult => {
  const root = gepaRunnerCheckpointsRoot(input.config, input.cwd);
  const files = input.runId == null
    ? listJsonFiles(root)
    : listJsonFiles(gepaRunnerCheckpointRunDir(input.config, input.runId, input.cwd));
  const checkpoints: GepaCheckpointSummary[] = [];
  const invalidCheckpoints: GepaCheckpointDiagnostic[] = [];

  for (const file of files) {
    const loaded = loadGepaRunnerCheckpoint(file);
    if (loaded.checkpoint == null) {
      invalidCheckpoints.push(...loaded.errors);
      continue;
    }
    if (input.runId != null && loaded.checkpoint.runId !== input.runId) {
      invalidCheckpoints.push({
        path: file,
        kind: "validation_error",
        message: `checkpoint runId ${loaded.checkpoint.runId} does not match requested runId ${input.runId}`,
      });
      continue;
    }
    const { state: _state, ...summary } = loaded.checkpoint;
    checkpoints.push({ ...summary, path: file });
  }

  checkpoints.sort(compareCheckpointSummaries);
  return { dir: root, checkpoints, invalidCheckpoints };
};

export const latestGepaRunnerCheckpoint = (
  input: ListGepaRunnerCheckpointsInput,
): GepaCheckpointSummary | undefined =>
  listGepaRunnerCheckpoints(input).checkpoints.at(-1);

export const loadLatestGepaRunnerCheckpoint = (
  input: ListGepaRunnerCheckpointsInput,
): LoadGepaRunnerCheckpointResult | undefined => {
  const latest = latestGepaRunnerCheckpoint(input);
  return latest == null ? undefined : loadGepaRunnerCheckpoint(latest.path);
};

const checkpointForState = (state: GepaRunnerState, savedAt: string): GepaRunnerCheckpoint => {
  const stateContentHash = hashStableValue(state);
  return GepaRunnerCheckpointSchema.parse({
    schemaVersion: GEPA_CHECKPOINT_SCHEMA_VERSION,
    checkpointKind: GEPA_CHECKPOINT_KIND,
    checkpointId: stableId("gepa-checkpoint", state.runId, savedAt, state.updatedAt, stateContentHash),
    runId: state.runId,
    feedbackBundleId: state.feedbackBundleId,
    savedAt,
    runnerCreatedAt: state.createdAt,
    runnerUpdatedAt: state.updatedAt,
    iterationCount: state.iterationCount,
    candidateCount: state.candidates.length,
    validationCount: state.validations.length,
    diagnosticCount: state.diagnostics.length,
    exhausted: state.exhausted,
    stateContentHash,
    state,
  });
};

const atomicWriteJson = (path: string, value: unknown): void => {
  mkdirSync(dirname(path), { recursive: true });
  const tempPath = join(dirname(path), `.${basename(path)}.${process.pid}.${Date.now()}.tmp`);
  try {
    writeFileSync(tempPath, `${JSON.stringify(stableValue(value), null, 2)}\n`, { flag: "wx" });
    renameSync(tempPath, path);
  } catch (error) {
    rmSync(tempPath, { force: true });
    throw error;
  }
};

const loadJsonFile = (path: string): { value: unknown } | { error: GepaCheckpointDiagnostic } => {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (error) {
    return {
      error: {
        path,
        kind: "read_error",
        message: error instanceof Error ? error.message : String(error),
      },
    };
  }

  try {
    return { value: JSON.parse(raw) as unknown };
  } catch (error) {
    return {
      error: {
        path,
        kind: "parse_error",
        message: error instanceof Error ? error.message : String(error),
      },
    };
  }
};

const listJsonFiles = (dir: string): string[] => {
  if (!existsSync(dir)) {
    return [];
  }
  return readdirSync(dir, { withFileTypes: true })
    .flatMap((entry) => {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) {
        return listJsonFiles(path);
      }
      return entry.isFile() && entry.name.endsWith(".json") ? [path] : [];
    })
    .sort((left, right) => left.localeCompare(right));
};

const compareCheckpointSummaries = (left: GepaCheckpointSummary, right: GepaCheckpointSummary): number =>
  left.runId.localeCompare(right.runId)
    || left.savedAt.localeCompare(right.savedAt)
    || left.runnerUpdatedAt.localeCompare(right.runnerUpdatedAt)
    || left.checkpointId.localeCompare(right.checkpointId)
    || left.path.localeCompare(right.path);

const hashStableValue = (value: unknown): string =>
  `sha256:${createHash("sha256").update(stableRegistryJson(value)).digest("hex")}`;

const stableValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((entry) => stableValue(entry));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, stableValue(entry)]),
    );
  }
  return value;
};

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 180) || "gepa-checkpoint.empty";

const safePathSegment = (value: string): string =>
  value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 160) || "gepa-checkpoint";

const zodErrorMessage = (error: z.ZodError): string =>
  error.issues.map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`).join("; ");
