import type {
  ActiveOptimizerPointer,
  OptimizerRegistryLoadResult,
} from "./registry";
import type { ModelEndpointKind, ModelProvider, ModelProviderConfigRole, ModelRuntimeRole } from "../types";
import type {
  CodebaseProfile,
  ModelCodebasePolicy,
  ModelProfile,
  OptimizerRegistryRecord,
} from "./types";

export type OptimizerModelRole = ModelRuntimeRole;

type ModelProfileRecord = Extract<OptimizerRegistryRecord, { recordKind: "model_profile" }>;
type CodebaseProfileRecord = Extract<OptimizerRegistryRecord, { recordKind: "codebase_profile" }>;
type PolicyRecord = Extract<OptimizerRegistryRecord, { recordKind: "model_codebase_policy" }>;

export interface ResolveOptimizerPolicyInput {
  records: OptimizerRegistryRecord[];
  seedRecords: OptimizerRegistryRecord[];
  persistedRecords?: OptimizerRegistryRecord[];
  activePointer?: ActiveOptimizerPointer;
  modelRole?: OptimizerModelRole;
  modelName?: string;
  modelProfileId?: string;
  codebaseProfileId?: string;
  codebaseRootFingerprint?: string;
}

export interface ResolvedOptimizerPolicy {
  source: "active_pointer" | "registry" | "seed";
  modelProfile: ModelProfile;
  codebaseProfile: CodebaseProfile;
  policy: ModelCodebasePolicy;
  modelRole?: OptimizerModelRole;
  providerConfigRole?: ModelProviderConfigRole;
  fallbackModelRole?: OptimizerModelRole;
  provider: ModelProvider;
  model: string;
  baseUrl?: string;
  endpointKind: ModelEndpointKind;
  modelServerId?: string;
  modelServerProfileId?: string;
  contextWindowTokens: number;
  contextWindowSource?: ModelProfile["contextWindowSource"];
  maxOutputTokens: number;
  providerDiscoverySource?: ModelProfile["providerDiscoverySource"];
  modelProfileId: string;
  codebaseProfileId: string;
  codebaseRootFingerprint: string;
  policyId: string;
  canonicalToolVersion: string;
  renderedToolVersion: string;
  resultStyleVersion: string;
  verificationPolicyVersion: string;
  editStrategyVersion: string;
  renderedEditContractVersion: string;
  editFallbackPolicyVersion: string;
  editRepairPolicyVersion: string;
  editVerifierPolicyVersion: string;
  editObjectiveSetId: string;
  recordIds: {
    modelProfileRecordId: string;
    codebaseProfileRecordId: string;
    policyRecordId: string;
  };
}

interface ActiveSet {
  modelRecord: ModelProfileRecord;
  codebaseRecord: CodebaseProfileRecord;
  policyRecord: PolicyRecord;
}

export const isSelectableProfileRecord = (record: OptimizerRegistryRecord): boolean =>
  (record.recordKind === "model_profile" || record.recordKind === "codebase_profile") &&
  (record.status === "active" || record.status === "promoted");

export const isSelectablePolicyRecord = (record: OptimizerRegistryRecord): boolean =>
  record.recordKind === "model_codebase_policy" &&
  (record.status === "active" || record.status === "promoted") &&
  record.payload.status === "promoted";

const modelRecords = (records: OptimizerRegistryRecord[]): ModelProfileRecord[] =>
  records.filter((record): record is ModelProfileRecord => record.recordKind === "model_profile");

const codebaseRecords = (records: OptimizerRegistryRecord[]): CodebaseProfileRecord[] =>
  records.filter((record): record is CodebaseProfileRecord => record.recordKind === "codebase_profile");

const policyRecords = (records: OptimizerRegistryRecord[]): PolicyRecord[] =>
  records.filter((record): record is PolicyRecord => record.recordKind === "model_codebase_policy");

const roleFromRecord = (record: ModelProfileRecord): OptimizerModelRole | undefined => {
  if (record.payload.modelRole !== undefined) {
    return record.payload.modelRole;
  }
  const labelRole = record.labels.find((label): label is OptimizerModelRole =>
    label === "local" ||
    label === "master" ||
    label === "planner" ||
    label === "executor" ||
    label === "verifier" ||
    label === "critic" ||
    label === "summarizer" ||
    label === "fast_scout" ||
    label === "local_batch_executor"
  );
  return labelRole;
};

const recordMatchesRole = (record: ModelProfileRecord, role: OptimizerModelRole): boolean =>
  roleFromRecord(record) === role;

const recordStatusRank = (record: OptimizerRegistryRecord): number => {
  switch (record.status) {
    case "promoted":
      return 3;
    case "active":
      return 2;
    case "draft":
      return 1;
    case "rejected":
    case "retired":
      return 0;
  }
};

const compareStringsDescending = (left: string, right: string): number => right.localeCompare(left);

const comparePolicyRecordPreference = (
  left: PolicyRecord,
  right: PolicyRecord,
  persistedRecordIds: ReadonlySet<string>,
): number => {
  const statusDelta = recordStatusRank(right) - recordStatusRank(left);
  if (statusDelta !== 0) {
    return statusDelta;
  }

  const persistedDelta = Number(persistedRecordIds.has(right.registryRecordId)) - Number(persistedRecordIds.has(left.registryRecordId));
  if (persistedDelta !== 0) {
    return persistedDelta;
  }

  const updatedAtDelta = compareStringsDescending(left.updatedAt, right.updatedAt);
  if (updatedAtDelta !== 0) {
    return updatedAtDelta;
  }

  return left.registryRecordId.localeCompare(right.registryRecordId);
};

const compareProfileRecordPreference = (
  left: ModelProfileRecord | CodebaseProfileRecord,
  right: ModelProfileRecord | CodebaseProfileRecord,
  persistedRecordIds: ReadonlySet<string>,
): number => {
  const statusDelta = recordStatusRank(right) - recordStatusRank(left);
  if (statusDelta !== 0) {
    return statusDelta;
  }

  const persistedDelta = Number(persistedRecordIds.has(right.registryRecordId)) - Number(persistedRecordIds.has(left.registryRecordId));
  if (persistedDelta !== 0) {
    return persistedDelta;
  }

  const updatedAtDelta = compareStringsDescending(left.updatedAt, right.updatedAt);
  if (updatedAtDelta !== 0) {
    return updatedAtDelta;
  }

  return left.registryRecordId.localeCompare(right.registryRecordId);
};

const findSeedModelRecord = (input: ResolveOptimizerPolicyInput): ModelProfileRecord => {
  const candidates = modelRecords(input.seedRecords).filter((record) => isSelectableProfileRecord(record));

  const exact = input.modelProfileId === undefined
    ? undefined
    : candidates.find((record) => record.payload.modelProfileId === input.modelProfileId);
  if (exact !== undefined) {
    return exact;
  }

  const role = input.modelRole ?? "local";
  const byRole = candidates.find((record) =>
    recordMatchesRole(record, role) &&
    (input.modelName === undefined || record.payload.model === input.modelName)
  );
  if (byRole !== undefined) {
    return byRole;
  }

  const byName = input.modelName === undefined
    ? undefined
    : candidates.find((record) => record.payload.model === input.modelName);
  if (byName !== undefined) {
    return byName;
  }

  const fallback = candidates[0];
  if (fallback === undefined) {
    throw new Error("optimizer policy resolver requires at least one selectable seed model profile");
  }
  return fallback;
};

const findSeedCodebaseRecord = (input: ResolveOptimizerPolicyInput): CodebaseProfileRecord => {
  const candidates = codebaseRecords(input.seedRecords).filter((record) => isSelectableProfileRecord(record));

  const exact = input.codebaseProfileId === undefined
    ? undefined
    : candidates.find((record) =>
      record.payload.codebaseProfileId === input.codebaseProfileId &&
      (input.codebaseRootFingerprint === undefined || record.payload.rootFingerprint === input.codebaseRootFingerprint)
    );
  if (exact !== undefined) {
    return exact;
  }

  const fallback = candidates[0];
  if (fallback === undefined) {
    throw new Error("optimizer policy resolver requires at least one selectable seed codebase profile");
  }
  return fallback;
};

const findSeedPolicyRecord = (
  seedRecords: OptimizerRegistryRecord[],
  modelProfileId: string,
  codebaseProfileId: string,
): PolicyRecord => {
  const policy = policyRecords(seedRecords).find((record) =>
    isSelectablePolicyRecord(record) &&
    record.payload.modelProfileId === modelProfileId &&
    record.payload.codebaseProfileId === codebaseProfileId
  );
  if (policy === undefined) {
    throw new Error("optimizer policy resolver requires a selectable seed policy for the current model and codebase");
  }
  return policy;
};

const modelMatchesCurrent = (
  record: ModelProfileRecord,
  seedModel: ModelProfileRecord,
  input: ResolveOptimizerPolicyInput,
): boolean => {
  if (!isSelectableProfileRecord(record)) {
    return false;
  }
  if (input.modelProfileId !== undefined) {
    return record.payload.modelProfileId === input.modelProfileId;
  }
  const modelName = input.modelName ?? seedModel.payload.model;
  const requestedRole = input.modelRole ?? roleFromRecord(seedModel) ?? "local";
  const recordRole = roleFromRecord(record);
  if (recordRole !== undefined && recordRole !== requestedRole) {
    return false;
  }
  if (recordRole === undefined && requestedRole !== "local" && requestedRole !== "master") {
    return false;
  }
  return record.payload.model === modelName;
};

const codebaseMatchesCurrent = (
  record: CodebaseProfileRecord,
  seedCodebase: CodebaseProfileRecord,
  input: ResolveOptimizerPolicyInput,
): boolean => {
  if (!isSelectableProfileRecord(record)) {
    return false;
  }
  if (input.codebaseProfileId !== undefined) {
    return record.payload.codebaseProfileId === input.codebaseProfileId &&
      (input.codebaseRootFingerprint === undefined || record.payload.rootFingerprint === input.codebaseRootFingerprint);
  }
  return record.payload.rootFingerprint === seedCodebase.payload.rootFingerprint;
};

const policyMatchesCodebase = (policyRecord: PolicyRecord, codebaseRecord: CodebaseProfileRecord): boolean =>
  policyRecord.payload.codebaseProfileId === codebaseRecord.payload.codebaseProfileId &&
  (
    policyRecord.payload.codebaseRootFingerprint === undefined ||
    policyRecord.payload.codebaseRootFingerprint === codebaseRecord.payload.rootFingerprint
  );

const findRecordByPayloadId = <T extends ModelProfileRecord | CodebaseProfileRecord | PolicyRecord>(
  records: T[],
  payloadId: string,
): T | undefined => records.find((record) => {
  switch (record.recordKind) {
    case "model_profile":
      return record.payload.modelProfileId === payloadId;
    case "codebase_profile":
      return record.payload.codebaseProfileId === payloadId;
    case "model_codebase_policy":
      return record.payload.policyId === payloadId;
  }
});

const resolvePointerActiveSet = (
  input: ResolveOptimizerPolicyInput,
  seedModel: ModelProfileRecord,
  seedCodebase: CodebaseProfileRecord,
): ActiveSet | undefined => {
  const pointer = input.activePointer;
  if (
    pointer?.activeModelProfileId === undefined ||
    pointer.activeCodebaseProfileId === undefined ||
    pointer.activePolicyId === undefined
  ) {
    return undefined;
  }

  const modelRecord = findRecordByPayloadId(modelRecords(input.records), pointer.activeModelProfileId);
  const codebaseRecord = findRecordByPayloadId(codebaseRecords(input.records), pointer.activeCodebaseProfileId);
  const policyRecord = findRecordByPayloadId(policyRecords(input.records), pointer.activePolicyId);
  if (modelRecord === undefined || codebaseRecord === undefined || policyRecord === undefined) {
    return undefined;
  }

  if (
    !modelMatchesCurrent(modelRecord, seedModel, input) ||
    !codebaseMatchesCurrent(codebaseRecord, seedCodebase, input) ||
    !isSelectablePolicyRecord(policyRecord)
  ) {
    return undefined;
  }

  if (
    policyRecord.payload.modelProfileId !== modelRecord.payload.modelProfileId ||
    !policyMatchesCodebase(policyRecord, codebaseRecord)
  ) {
    return undefined;
  }

  if (
    pointer.activeCodebaseRootFingerprint !== undefined &&
    pointer.activeCodebaseRootFingerprint !== codebaseRecord.payload.rootFingerprint
  ) {
    return undefined;
  }

  return { modelRecord, codebaseRecord, policyRecord };
};

const resolveRegistryActiveSet = (
  input: ResolveOptimizerPolicyInput,
  seedModel: ModelProfileRecord,
  seedCodebase: CodebaseProfileRecord,
  persistedRecordIds: ReadonlySet<string>,
): ActiveSet | undefined => {
  const modelsById = new Map<string, ModelProfileRecord>();
  for (const record of modelRecords(input.records)
    .filter((modelRecord) => modelMatchesCurrent(modelRecord, seedModel, input))
    .sort((left, right) => compareProfileRecordPreference(left, right, persistedRecordIds))) {
    if (!modelsById.has(record.payload.modelProfileId)) {
      modelsById.set(record.payload.modelProfileId, record);
    }
  }

  const codebasesById = new Map<string, CodebaseProfileRecord>();
  for (const record of codebaseRecords(input.records)
    .filter((codebaseRecord) => codebaseMatchesCurrent(codebaseRecord, seedCodebase, input))
    .sort((left, right) => compareProfileRecordPreference(left, right, persistedRecordIds))) {
    if (!codebasesById.has(record.payload.codebaseProfileId)) {
      codebasesById.set(record.payload.codebaseProfileId, record);
    }
  }

  const policyRecord = policyRecords(input.records)
    .filter((record) =>
      isSelectablePolicyRecord(record) &&
      modelsById.has(record.payload.modelProfileId) &&
      policyCodebaseRecord(record, codebasesById) !== undefined
    )
    .sort((left, right) => comparePolicyRecordPreference(left, right, persistedRecordIds))[0];

  if (policyRecord === undefined) {
    return undefined;
  }

  const modelRecord = modelsById.get(policyRecord.payload.modelProfileId);
  const codebaseRecord = codebasesById.get(policyRecord.payload.codebaseProfileId);
  if (modelRecord === undefined || codebaseRecord === undefined) {
    return undefined;
  }
  if (!policyMatchesCodebase(policyRecord, codebaseRecord)) {
    return undefined;
  }

  return { modelRecord, codebaseRecord, policyRecord };
};

const policyCodebaseRecord = (
  policyRecord: PolicyRecord,
  codebasesById: ReadonlyMap<string, CodebaseProfileRecord>,
): CodebaseProfileRecord | undefined => {
  const codebaseRecord = codebasesById.get(policyRecord.payload.codebaseProfileId);
  if (codebaseRecord === undefined) {
    return undefined;
  }
  return policyMatchesCodebase(policyRecord, codebaseRecord) ? codebaseRecord : undefined;
};

const toResolution = (
  activeSet: ActiveSet,
  source: ResolvedOptimizerPolicy["source"],
): ResolvedOptimizerPolicy => ({
  source,
  modelProfile: activeSet.modelRecord.payload,
  codebaseProfile: activeSet.codebaseRecord.payload,
  policy: activeSet.policyRecord.payload,
  ...(activeSet.modelRecord.payload.modelRole === undefined ? {} : { modelRole: activeSet.modelRecord.payload.modelRole }),
  ...(activeSet.modelRecord.payload.providerConfigRole === undefined
    ? {}
    : { providerConfigRole: activeSet.modelRecord.payload.providerConfigRole }),
  ...(activeSet.modelRecord.payload.fallbackModelRole === undefined
    ? {}
    : { fallbackModelRole: activeSet.modelRecord.payload.fallbackModelRole }),
  provider: activeSet.modelRecord.payload.provider,
  model: activeSet.modelRecord.payload.model,
  ...(activeSet.modelRecord.payload.baseUrl === undefined ? {} : { baseUrl: activeSet.modelRecord.payload.baseUrl }),
  endpointKind: activeSet.modelRecord.payload.endpointKind,
  ...(activeSet.modelRecord.payload.modelServerId === undefined
    ? {}
    : { modelServerId: activeSet.modelRecord.payload.modelServerId }),
  ...(activeSet.modelRecord.payload.modelServerProfileId === undefined
    ? {}
    : { modelServerProfileId: activeSet.modelRecord.payload.modelServerProfileId }),
  contextWindowTokens: activeSet.modelRecord.payload.contextWindowTokens,
  ...(activeSet.modelRecord.payload.contextWindowSource === undefined
    ? {}
    : { contextWindowSource: activeSet.modelRecord.payload.contextWindowSource }),
  maxOutputTokens: activeSet.modelRecord.payload.maxOutputTokens,
  ...(activeSet.modelRecord.payload.providerDiscoverySource === undefined
    ? {}
    : { providerDiscoverySource: activeSet.modelRecord.payload.providerDiscoverySource }),
  modelProfileId: activeSet.modelRecord.payload.modelProfileId,
  codebaseProfileId: activeSet.codebaseRecord.payload.codebaseProfileId,
  codebaseRootFingerprint: activeSet.codebaseRecord.payload.rootFingerprint,
  policyId: activeSet.policyRecord.payload.policyId,
  canonicalToolVersion: activeSet.policyRecord.payload.canonicalToolVersion,
  renderedToolVersion: activeSet.policyRecord.payload.renderedToolVersion,
  resultStyleVersion: activeSet.policyRecord.payload.resultStyleVersion,
  verificationPolicyVersion: activeSet.policyRecord.payload.verificationPolicyVersion,
  editStrategyVersion: activeSet.policyRecord.payload.editStrategyVersion,
  renderedEditContractVersion: activeSet.policyRecord.payload.renderedEditContractVersion,
  editFallbackPolicyVersion: activeSet.policyRecord.payload.editFallbackPolicyVersion,
  editRepairPolicyVersion: activeSet.policyRecord.payload.editRepairPolicyVersion,
  editVerifierPolicyVersion: activeSet.policyRecord.payload.editVerifierPolicyVersion,
  editObjectiveSetId: activeSet.policyRecord.payload.editObjectiveSetId,
  recordIds: {
    modelProfileRecordId: activeSet.modelRecord.registryRecordId,
    codebaseProfileRecordId: activeSet.codebaseRecord.registryRecordId,
    policyRecordId: activeSet.policyRecord.registryRecordId,
  },
});

export const resolveOptimizerPolicy = (input: ResolveOptimizerPolicyInput): ResolvedOptimizerPolicy => {
  const seedModel = findSeedModelRecord(input);
  const seedCodebase = findSeedCodebaseRecord(input);
  const seedPolicy = findSeedPolicyRecord(
    input.seedRecords,
    seedModel.payload.modelProfileId,
    seedCodebase.payload.codebaseProfileId,
  );
  const seedActiveSet = {
    modelRecord: seedModel,
    codebaseRecord: seedCodebase,
    policyRecord: seedPolicy,
  };
  const persistedRecordIds = new Set((input.persistedRecords ?? []).map((record) => record.registryRecordId));

  const pointerActiveSet = resolvePointerActiveSet(input, seedModel, seedCodebase);
  if (pointerActiveSet !== undefined) {
    return toResolution(pointerActiveSet, "active_pointer");
  }

  const registryActiveSet = resolveRegistryActiveSet(input, seedModel, seedCodebase, persistedRecordIds);
  if (registryActiveSet !== undefined) {
    const source = registryActiveSet.policyRecord.registryRecordId === seedPolicy.registryRecordId ? "seed" : "registry";
    return toResolution(registryActiveSet, source);
  }

  return toResolution(seedActiveSet, "seed");
};

export const resolveLoadedOptimizerPolicy = (
  registry: OptimizerRegistryLoadResult,
  input: Omit<ResolveOptimizerPolicyInput, "records" | "seedRecords" | "persistedRecords" | "activePointer"> = {},
): ResolvedOptimizerPolicy =>
  resolveOptimizerPolicy({
    ...input,
    records: registry.records,
    seedRecords: registry.seedRecords,
    persistedRecords: registry.persistedRecords,
    ...(registry.activePointer === undefined ? {} : { activePointer: registry.activePointer }),
  });
