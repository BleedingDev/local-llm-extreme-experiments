import { describe, expect, test } from "bun:test";
import {
  canonicalEditStrategyToToolSpec,
  canonicalEditToolSpecs,
  renderEditToolContract,
  renderEditToolContracts,
  selectRenderedEditToolContracts,
} from "../src/edit-strategy/contract-renderer";
import {
  initialExperimentalEditStrategyIds,
  parseCanonicalEditStrategyDefinitions,
  type CanonicalEditStrategyDefinition,
} from "../src/edit-strategy/taxonomy";
import type { ResolvedOptimizerPolicy } from "../src/optimizer/policy-resolver";
import type { ModelProfile, OptimizerRegistryRecord, RenderedToolContract } from "../src/optimizer/types";

const now = "2026-04-30T00:00:00.000Z";

const nativeModelProfile: ModelProfile = {
  modelProfileId: "model.native",
  displayName: "Native Tool Model",
  provider: "openai",
  model: "native-tool-model",
  endpointKind: "responses",
  contextWindowTokens: 128000,
  maxOutputTokens: 4096,
  defaultTemperature: 0.1,
  toolCallingMode: "native",
  structuredOutputMode: "json_schema",
  supportsParallelToolCalls: true,
  promptStyle: "system_user",
  resultStyleVersion: "result-style.model",
  verificationPolicyVersion: "verification.model",
};

const textModelProfile: ModelProfile = {
  ...nativeModelProfile,
  modelProfileId: "model.text",
  displayName: "Text Tool Model",
  model: "text-tool-model",
  toolCallingMode: "text",
  structuredOutputMode: "text",
  supportsParallelToolCalls: false,
  promptStyle: "plain_text",
};

const resolvedPolicy = (modelProfile: ModelProfile): ResolvedOptimizerPolicy => ({
  source: "seed",
  modelProfile,
  codebaseProfile: {
    codebaseProfileId: "codebase.test",
    displayName: "Test Codebase",
    rootFingerprint: "sha256:test",
    languages: ["typescript"],
    packageManagers: ["npm"],
    sourceRoots: ["src"],
    testCommands: [],
    typecheckCommands: [],
    lintCommands: [],
    protectedPaths: [],
    conventions: [],
    verificationPolicyVersion: "verification.codebase",
  },
  policy: {
    policyId: "policy.test",
    modelProfileId: modelProfile.modelProfileId,
    codebaseProfileId: "codebase.test",
    status: "promoted",
    canonicalToolVersion: "canonical-tools.policy",
    renderedToolVersion: "rendered-tools.policy",
    resultStyleVersion: "result-style.policy",
    verificationPolicyVersion: "verification.policy",
    editStrategyVersion: "edit-strategy.policy",
    renderedEditContractVersion: "rendered-edit-contract.policy",
    editFallbackPolicyVersion: "edit-fallback.policy",
    editRepairPolicyVersion: "edit-repair.policy",
    editVerifierPolicyVersion: "edit-verifier.policy",
    editObjectiveSetId: "edit-objectives.policy",
    candidateScopes: [],
    verificationGates: [],
    maxConcurrentEvaluations: 1,
    riskTolerance: "low",
  },
  modelProfileId: modelProfile.modelProfileId,
  codebaseProfileId: "codebase.test",
  policyId: "policy.test",
  canonicalToolVersion: "canonical-tools.policy",
  renderedToolVersion: "rendered-tools.policy",
  resultStyleVersion: "result-style.policy",
  verificationPolicyVersion: "verification.policy",
  editStrategyVersion: "edit-strategy.policy",
  renderedEditContractVersion: "rendered-edit-contract.policy",
  editFallbackPolicyVersion: "edit-fallback.policy",
  editRepairPolicyVersion: "edit-repair.policy",
  editVerifierPolicyVersion: "edit-verifier.policy",
  editObjectiveSetId: "edit-objectives.policy",
  recordIds: {
    modelProfileRecordId: `registry.${modelProfile.modelProfileId}`,
    codebaseProfileRecordId: "registry.codebase.test",
    policyRecordId: "registry.policy.test",
  },
});

const recordFor = (
  contract: RenderedToolContract,
  overrides: Partial<OptimizerRegistryRecord> = {},
): OptimizerRegistryRecord => ({
  registryRecordId: `registry.${contract.renderedToolId}`,
  recordKind: "rendered_tool_contract",
  schemaVersion: "optimizer-schema.v1",
  recordVersion: "record.v1",
  status: "promoted",
  createdAt: now,
  updatedAt: now,
  labels: [],
  payload: contract,
  ...overrides,
});

const definition = (strategyId: string): CanonicalEditStrategyDefinition => {
  const match = parseCanonicalEditStrategyDefinitions().find((entry) => entry.strategyId === strategyId);
  if (match === undefined) {
    throw new Error(`missing test strategy ${strategyId}`);
  }
  return match;
};

describe("edit strategy contract renderer", () => {
  test("turns canonical edit strategies into optimizer tool specs", () => {
    const spec = canonicalEditStrategyToToolSpec(definition("edit.apply-patch.v1"));

    expect(spec.canonicalToolId).toBe("edit.apply-patch.v1");
    expect(spec.namespace).toBe("edit");
    expect(spec.name).toBe("apply_patch");
    expect(spec.sideEffectLevel).toBe("write");
    expect(spec.resultStyle).toBe("structured_error");
    expect(spec.description).toContain("Selection is optimizer-controlled");
    expect(spec.inputSchema.properties).toHaveProperty("patch");
    expect(spec.outputSchema?.properties).toHaveProperty("errorCode");
  });

  test("renders initial edit contracts with edit-specific policy versions", () => {
    const contracts = renderEditToolContracts({
      resolvedPolicy: resolvedPolicy(nativeModelProfile),
      initialExperimentOnly: true,
    });
    const contractIds = contracts.map((contract) => contract.canonicalToolId).sort((left, right) => left.localeCompare(right));

    expect(contractIds).toEqual(initialExperimentalEditStrategyIds());
    expect(contracts.every((contract) => contract.canonicalToolVersion === "edit-strategy.policy")).toBe(true);
    expect(contracts.every((contract) => contract.renderedToolVersion === "rendered-edit-contract.policy")).toBe(true);
    expect(contracts.every((contract) => contract.policyId === "policy.test")).toBe(true);
    expect(contracts.every((contract) => contract.description.includes("post-apply consistency"))).toBe(true);
    expect(contracts.every((contract) => contract.promptFragments.some((fragment) => fragment.includes("Fallback policy"))))
      .toBe(true);
    expect(JSON.stringify(contracts)).not.toContain("best strategy");
    expect(JSON.stringify(contracts)).not.toContain("preferred strategy");
  });

  test("excludes future-gated edit strategies by default", () => {
    const specs = canonicalEditToolSpecs();
    const withFuture = canonicalEditToolSpecs(parseCanonicalEditStrategyDefinitions(), { includeFutureGated: true });

    expect(specs.some((spec) => spec.canonicalToolId.includes(".future."))).toBe(false);
    expect(withFuture.some((spec) => spec.canonicalToolId === "edit.range-native.future.v1")).toBe(true);
  });

  test("uses compact text fallback contracts for text-only models", () => {
    const contract = renderEditToolContract(definition("edit.exact-replace.v1"), resolvedPolicy(textModelProfile));

    expect(contract.description).toContain("Arguments:");
    expect(contract.promptFragments.some((fragment) => fragment.includes("Text contract"))).toBe(true);
    expect(contract.promptFragments.some((fragment) => fragment.includes("Repair policy"))).toBe(true);
    expect(contract.inputSchema.properties).toHaveProperty("search");
    expect(contract.inputSchema.properties).toHaveProperty("replace");
  });

  test("selects complete promoted edit contracts and falls back when incomplete", () => {
    const policy = resolvedPolicy(nativeModelProfile);
    const definitions = [definition("edit.apply-patch.v1"), definition("edit.exact-replace.v1")];
    const fresh = renderEditToolContracts({ resolvedPolicy: policy, definitions });
    const promoted = fresh.map((contract) => ({
      ...contract,
      renderedToolId: `${contract.renderedToolId}.promoted`,
      description: `Promoted ${contract.description}`,
    }));

    const selected = selectRenderedEditToolContracts({
      resolvedPolicy: policy,
      definitions,
      records: promoted.map((contract) => recordFor(contract)),
    });
    expect(selected.map((contract) => contract.renderedToolId)).toEqual(
      promoted.map((contract) => contract.renderedToolId),
    );
    expect(selected.every((contract) => contract.description.startsWith("Promoted"))).toBe(true);

    const partial = selectRenderedEditToolContracts({
      resolvedPolicy: policy,
      definitions,
      records: [recordFor(promoted[0]!)],
    });
    expect(partial.map((contract) => contract.renderedToolId)).toEqual(
      fresh.map((contract) => contract.renderedToolId),
    );
  });
});
