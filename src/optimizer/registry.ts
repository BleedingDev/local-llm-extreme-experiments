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
import { basename, dirname, join, resolve } from "node:path";
import { z } from "zod";
import { resolveModelRoleConfig } from "../config";
import type { BagConfig } from "../types";
import { MODEL_RUNTIME_ROLES, type ModelRuntimeRole } from "../types";
import {
  OptimizerIdSchema,
  OptimizerRegistryRecordSchema,
  type OptimizerRegistryRecord,
} from "./types";
import { generateCodebaseProfile } from "./codebase-profile";

const OPTIMIZER_DIR = "optimizer";
const RECORDS_DIR = "records";
const ACTIVE_POINTER_FILE = "active.json";
const CANDIDATES_DIR = "candidates";
const CHECKPOINTS_DIR = "checkpoints";
const DEFAULT_CREATED_AT = "1970-01-01T00:00:00.000Z";
const SCHEMA_VERSION = "optimizer-schema.v1";
const RECORD_VERSION = "record.v1";
const CANONICAL_TOOL_VERSION = "canonical-tools.v1";
const RENDERED_TOOL_VERSION = "rendered-tools.v1";
const RESULT_STYLE_VERSION = "result-style.v1";
const VERIFICATION_POLICY_VERSION = "verification.v1";
const EDIT_STRATEGY_VERSION = "edit-strategy.v1";
const RENDERED_EDIT_CONTRACT_VERSION = "rendered-edit-contract.v1";
const EDIT_FALLBACK_POLICY_VERSION = "edit-fallback.v1";
const EDIT_REPAIR_POLICY_VERSION = "edit-repair.v1";
const EDIT_VERIFIER_POLICY_VERSION = "edit-verifier.v1";
const EDIT_OBJECTIVE_SET_ID = "edit-objectives.default.v1";
const ACTIVE_POINTER_SCHEMA_VERSION = "optimizer-active.v1";
const SEEDED_MODEL_ROLES: ModelRuntimeRole[] = [...MODEL_RUNTIME_ROLES];

export const optimizerRegistryRoot = (config: BagConfig, cwd = process.cwd()): string =>
  resolve(cwd, config.artifactDir, OPTIMIZER_DIR);

export const optimizerRegistryRecordsDir = (config: BagConfig, cwd = process.cwd()): string =>
  join(optimizerRegistryRoot(config, cwd), RECORDS_DIR);

export const optimizerRegistryRecordPath = (
  config: BagConfig,
  record: Pick<OptimizerRegistryRecord, "recordKind" | "registryRecordId">,
  cwd = process.cwd(),
): string => join(optimizerRegistryRecordsDir(config, cwd), record.recordKind, `${record.registryRecordId}.json`);

export const optimizerRegistryActivePointerPath = (config: BagConfig, cwd = process.cwd()): string =>
  join(optimizerRegistryRoot(config, cwd), ACTIVE_POINTER_FILE);

export const optimizerRegistryCandidatesDir = (config: BagConfig, cwd = process.cwd()): string =>
  join(optimizerRegistryRoot(config, cwd), CANDIDATES_DIR);

export const optimizerRegistryCheckpointsDir = (config: BagConfig, cwd = process.cwd()): string =>
  join(optimizerRegistryRoot(config, cwd), CHECKPOINTS_DIR);

export type RegistryArtifactErrorKind = "read_error" | "parse_error" | "validation_error";

export interface RegistryArtifactError {
  path: string;
  kind: RegistryArtifactErrorKind;
  message: string;
}

export const ActiveOptimizerPointerSchema = z.object({
  schemaVersion: z.literal(ACTIVE_POINTER_SCHEMA_VERSION).default(ACTIVE_POINTER_SCHEMA_VERSION),
  activeModelProfileId: OptimizerIdSchema.optional(),
  activeCodebaseProfileId: OptimizerIdSchema.optional(),
  activeCodebaseRootFingerprint: z.string().min(1).optional(),
  activePolicyId: OptimizerIdSchema.optional(),
  promotedAt: z.string().optional(),
  contentHash: z.string().optional(),
}).strict();

export type ActiveOptimizerPointer = z.infer<typeof ActiveOptimizerPointerSchema>;

export interface ActiveOptimizerPointerLoadResult {
  pointer?: ActiveOptimizerPointer;
  errors: RegistryArtifactError[];
}

export interface OptimizerRegistryLoadResult {
  root: string;
  records: OptimizerRegistryRecord[];
  seedRecords: OptimizerRegistryRecord[];
  persistedRecords: OptimizerRegistryRecord[];
  invalidRecords: RegistryArtifactError[];
  activePointer?: ActiveOptimizerPointer;
  errors: RegistryArtifactError[];
}

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

export const stableRegistryJson = (value: unknown): string => JSON.stringify(stableValue(value));

export const hashRegistryContent = (value: unknown): string =>
  `sha256:${createHash("sha256").update(stableRegistryJson(value)).digest("hex")}`;

const withoutContentHash = <T extends object>(value: T): Omit<T, "contentHash"> => {
  const copy = { ...(value as Record<string, unknown>) };
  delete copy.contentHash;
  return copy as Omit<T, "contentHash">;
};

const withRecordContentHash = (record: OptimizerRegistryRecord): OptimizerRegistryRecord =>
  OptimizerRegistryRecordSchema.parse({
    ...withoutContentHash(record),
    contentHash: hashRegistryContent(withoutContentHash(record)),
  });

const withPointerContentHash = (pointer: ActiveOptimizerPointer): ActiveOptimizerPointer =>
  ActiveOptimizerPointerSchema.parse({
    ...withoutContentHash(pointer),
    contentHash: hashRegistryContent(withoutContentHash(pointer)),
  });

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

const loadJsonFile = (path: string): { value: unknown } | { error: RegistryArtifactError } => {
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

const zodErrorMessage = (error: z.ZodError): string =>
  error.issues.map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`).join("; ");

const modelIdFor = (role: ModelRuntimeRole, model: string): string =>
  `model.${role}.${hashRegistryContent(model).slice("sha256:".length, "sha256:".length + 12)}`;

const codebaseIdFor = (cwd: string): string =>
  `codebase.${hashRegistryContent(resolve(cwd)).slice("sha256:".length, "sha256:".length + 12)}`;

const policyIdFor = (modelProfileId: string, codebaseProfileId: string): string =>
  `policy.${hashRegistryContent({ codebaseProfileId, modelProfileId }).slice("sha256:".length, "sha256:".length + 12)}`;

const record = (input: Omit<OptimizerRegistryRecord, "schemaVersion" | "recordVersion" | "createdAt" | "updatedAt">): OptimizerRegistryRecord =>
  withRecordContentHash(OptimizerRegistryRecordSchema.parse({
    ...input,
    schemaVersion: SCHEMA_VERSION,
    recordVersion: RECORD_VERSION,
    createdAt: DEFAULT_CREATED_AT,
    updatedAt: DEFAULT_CREATED_AT,
  }));

const displayRoleName = (role: ModelRuntimeRole): string =>
  role.split("_").map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`).join(" ");

const modelProfileForRole = (config: BagConfig, role: ModelRuntimeRole): Extract<OptimizerRegistryRecord, { recordKind: "model_profile" }> => {
  const roleConfig = resolveModelRoleConfig(config, role);
  const modelProfileId = modelIdFor(role, roleConfig.model);
  const usesMasterConfig = roleConfig.providerConfigRole === "master";
  return record({
    registryRecordId: `registry.${modelProfileId}`,
    recordKind: "model_profile",
    status: "active",
    labels: [
      "seed",
      role,
      `provider-config:${roleConfig.providerConfigRole}`,
      `endpoint:${roleConfig.endpointKind}`,
      `server:${roleConfig.modelServerId}`,
      `server-profile:${roleConfig.modelServerProfileId}`,
    ],
    payload: {
      modelProfileId,
      displayName: `${displayRoleName(role)} ${roleConfig.model}`,
      modelRole: role,
      providerConfigRole: roleConfig.providerConfigRole,
      ...(roleConfig.fallbackModelRole === undefined ? {} : { fallbackModelRole: roleConfig.fallbackModelRole }),
      provider: roleConfig.provider,
      model: roleConfig.model,
      baseUrl: roleConfig.baseUrl,
      endpointKind: roleConfig.endpointKind,
      modelServerId: roleConfig.modelServerId,
      modelServerProfileId: roleConfig.modelServerProfileId,
      providerDiscoverySource: roleConfig.providerDiscoverySource,
      contextWindowTokens: roleConfig.contextWindowTokens,
      contextWindowSource: roleConfig.contextWindowSource,
      maxOutputTokens: roleConfig.maxOutputTokens,
      defaultTemperature: roleConfig.temperature,
      toolCallingMode: usesMasterConfig ? "native" : "json",
      structuredOutputMode: "json_schema",
      supportsParallelToolCalls: false,
      promptStyle: usesMasterConfig ? "system_user" : "chatml",
      resultStyleVersion: RESULT_STYLE_VERSION,
      verificationPolicyVersion: VERIFICATION_POLICY_VERSION,
    },
  }) as Extract<OptimizerRegistryRecord, { recordKind: "model_profile" }>;
};

const modelPolicyForRole = (
  modelProfileId: string,
  codebaseProfileId: string,
  codebaseRootFingerprint: string,
  role: ModelRuntimeRole,
): Extract<OptimizerRegistryRecord, { recordKind: "model_codebase_policy" }> => {
  const policyId = policyIdFor(modelProfileId, codebaseProfileId);
  return record({
    registryRecordId: `registry.${policyId}`,
    recordKind: "model_codebase_policy",
    status: "active",
    labels: ["seed", role],
    payload: {
      policyId,
      modelProfileId,
      codebaseProfileId,
      codebaseRootFingerprint,
      status: "promoted",
      canonicalToolVersion: CANONICAL_TOOL_VERSION,
      renderedToolVersion: RENDERED_TOOL_VERSION,
      resultStyleVersion: RESULT_STYLE_VERSION,
      verificationPolicyVersion: VERIFICATION_POLICY_VERSION,
      editStrategyVersion: EDIT_STRATEGY_VERSION,
      renderedEditContractVersion: RENDERED_EDIT_CONTRACT_VERSION,
      editFallbackPolicyVersion: EDIT_FALLBACK_POLICY_VERSION,
      editRepairPolicyVersion: EDIT_REPAIR_POLICY_VERSION,
      editVerifierPolicyVersion: EDIT_VERIFIER_POLICY_VERSION,
      editObjectiveSetId: EDIT_OBJECTIVE_SET_ID,
      candidateScopes: [],
      verificationGates: [
        {
          gateId: "typecheck",
          commandId: "typecheck",
          comparator: "eq",
          threshold: 0,
          required: true,
        },
        {
          gateId: "test",
          commandId: "test",
          comparator: "eq",
          threshold: 0,
          required: true,
        },
      ],
      maxConcurrentEvaluations: 1,
      riskTolerance: "low",
    },
  }) as Extract<OptimizerRegistryRecord, { recordKind: "model_codebase_policy" }>;
};

export const seedOptimizerRegistry = (config: BagConfig, cwd = process.cwd()): OptimizerRegistryRecord[] => {
  const resolvedCwd = resolve(cwd);
  const codebaseProfileId = codebaseIdFor(resolvedCwd);
  const modelProfiles = SEEDED_MODEL_ROLES.map((role) => modelProfileForRole(config, role));
  const generatedCodebaseProfile = generateCodebaseProfile({
    cwd: resolvedCwd,
    codebaseProfileId,
    displayName: basename(resolvedCwd),
    protectedPathDefaults: [config.artifactDir],
    verificationPolicyVersion: VERIFICATION_POLICY_VERSION,
  }).profile;

  const codebase = record({
    registryRecordId: `registry.${codebaseProfileId}`,
    recordKind: "codebase_profile",
    status: "active",
    labels: ["seed", "codebase"],
    payload: {
      ...generatedCodebaseProfile,
      codebaseProfileId,
      displayName: basename(resolvedCwd),
      verificationPolicyVersion: VERIFICATION_POLICY_VERSION,
    },
  });

  const policies = modelProfiles.map((modelProfile) =>
    modelPolicyForRole(
      modelProfile.payload.modelProfileId,
      codebaseProfileId,
      generatedCodebaseProfile.rootFingerprint,
      modelProfile.payload.modelRole ?? "local",
    )
  );

  return [...modelProfiles, codebase, ...policies];
};

export const loadPersistedOptimizerRegistryRecords = (
  config: BagConfig,
  cwd = process.cwd(),
): { records: OptimizerRegistryRecord[]; invalidRecords: RegistryArtifactError[] } => {
  const invalidRecords: RegistryArtifactError[] = [];
  const records: OptimizerRegistryRecord[] = [];

  for (const path of listJsonFiles(optimizerRegistryRecordsDir(config, cwd))) {
    const loaded = loadJsonFile(path);
    if ("error" in loaded) {
      invalidRecords.push(loaded.error);
      continue;
    }

    const parsed = OptimizerRegistryRecordSchema.safeParse(loaded.value);
    if (!parsed.success) {
      invalidRecords.push({
        path,
        kind: "validation_error",
        message: zodErrorMessage(parsed.error),
      });
      continue;
    }

    records.push(parsed.data);
  }

  return { records, invalidRecords };
};

export const mergeRegistryRecords = (
  seedRecords: OptimizerRegistryRecord[],
  persistedRecords: OptimizerRegistryRecord[],
): OptimizerRegistryRecord[] => {
  const recordsById = new Map<string, OptimizerRegistryRecord>();
  for (const registryRecord of seedRecords) {
    recordsById.set(registryRecord.registryRecordId, registryRecord);
  }
  for (const registryRecord of persistedRecords) {
    recordsById.set(registryRecord.registryRecordId, registryRecord);
  }
  return [...recordsById.values()].sort((left, right) => left.registryRecordId.localeCompare(right.registryRecordId));
};

export const loadActiveOptimizerPointer = (
  config: BagConfig,
  cwd = process.cwd(),
): ActiveOptimizerPointerLoadResult => {
  const path = optimizerRegistryActivePointerPath(config, cwd);
  if (!existsSync(path)) {
    return { errors: [] };
  }

  const loaded = loadJsonFile(path);
  if ("error" in loaded) {
    return { errors: [loaded.error] };
  }

  const parsed = ActiveOptimizerPointerSchema.safeParse(loaded.value);
  if (!parsed.success) {
    return {
      errors: [
        {
          path,
          kind: "validation_error",
          message: zodErrorMessage(parsed.error),
        },
      ],
    };
  }

  return { pointer: parsed.data, errors: [] };
};

export const loadOptimizerRegistry = (config: BagConfig, cwd = process.cwd()): OptimizerRegistryLoadResult => {
  const seedRecords = seedOptimizerRegistry(config, cwd);
  const persisted = loadPersistedOptimizerRegistryRecords(config, cwd);
  const activePointer = loadActiveOptimizerPointer(config, cwd);
  const records = mergeRegistryRecords(seedRecords, persisted.records);
  const errors = [...persisted.invalidRecords, ...activePointer.errors];
  return {
    root: optimizerRegistryRoot(config, cwd),
    records,
    seedRecords,
    persistedRecords: persisted.records,
    invalidRecords: persisted.invalidRecords,
    ...(activePointer.pointer === undefined ? {} : { activePointer: activePointer.pointer }),
    errors,
  };
};

export const saveOptimizerRegistryRecord = (
  config: BagConfig,
  recordToSave: OptimizerRegistryRecord,
  cwd = process.cwd(),
): OptimizerRegistryRecord => {
  const recordWithHash = withRecordContentHash(recordToSave);
  atomicWriteJson(optimizerRegistryRecordPath(config, recordWithHash, cwd), recordWithHash);
  return recordWithHash;
};

export const saveActiveOptimizerPointer = (
  config: BagConfig,
  pointer: ActiveOptimizerPointer,
  cwd = process.cwd(),
): ActiveOptimizerPointer => {
  const pointerWithHash = withPointerContentHash(pointer);
  atomicWriteJson(optimizerRegistryActivePointerPath(config, cwd), pointerWithHash);
  return pointerWithHash;
};

export const promoteActiveOptimizerPointer = (
  config: BagConfig,
  pointer: Omit<ActiveOptimizerPointer, "schemaVersion" | "contentHash">,
  cwd = process.cwd(),
): ActiveOptimizerPointer =>
  saveActiveOptimizerPointer(
    config,
    ActiveOptimizerPointerSchema.parse({
      schemaVersion: ACTIVE_POINTER_SCHEMA_VERSION,
      ...pointer,
    }),
    cwd,
  );
