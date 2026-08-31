import { describe, expect, test } from "bun:test";
import {
  renderToolContract,
  renderToolContracts,
  selectRenderedToolContracts,
} from "../src/optimizer/tool-renderer";
import type { ResolvedOptimizerPolicy } from "../src/optimizer/policy-resolver";
import type {
  CanonicalToolSpec,
  ModelProfile,
  OptimizerRegistryRecord,
  RenderedToolContract,
} from "../src/optimizer/types";

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

const resolvedPolicy = (modelProfile: ModelProfile, versions = {}): ResolvedOptimizerPolicy => ({
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
    candidateScopes: [],
    verificationGates: [],
    maxConcurrentEvaluations: 1,
    riskTolerance: "low",
    ...versions,
  },
  modelProfileId: modelProfile.modelProfileId,
  codebaseProfileId: "codebase.test",
  policyId: "policy.test",
  canonicalToolVersion: "canonical-tools.policy",
  renderedToolVersion: "rendered-tools.policy",
  resultStyleVersion: "result-style.policy",
  verificationPolicyVersion: "verification.policy",
  recordIds: {
    modelProfileRecordId: `registry.${modelProfile.modelProfileId}`,
    codebaseProfileRecordId: "registry.codebase.test",
    policyRecordId: "registry.policy.test",
  },
  ...versions,
});

const canonicalReadTool: CanonicalToolSpec = {
  canonicalToolId: "tool.repo.read",
  canonicalToolVersion: "canonical-tools.source",
  namespace: "repo",
  name: "read",
  title: "Read repository file",
  description: "Read a UTF-8 text file from the repository.",
  inputSchema: {
    required: ["query", "path"],
    type: "object",
    properties: {
      query: {
        anyOf: [{ type: "string" }, { type: "null" }],
        description: "Optional search term to highlight.",
      },
      path: {
        description: "Repository-relative path.",
        type: "string",
      },
      mode: {
        enum: ["full", "summary"],
        type: "string",
      },
    },
  },
  outputSchema: {
    type: "object",
    properties: {
      content: { type: "string" },
    },
  },
  resultStyle: "json",
  sideEffectLevel: "read",
  requiresConfirmation: false,
  examples: [
    {
      name: "read package",
      input: {
        path: "package.json",
        query: null,
      },
      output: {
        content: "{...}",
      },
    },
  ],
};

const canonicalWriteTool: CanonicalToolSpec = {
  ...canonicalReadTool,
  canonicalToolId: "tool.repo.write",
  namespace: "repo",
  name: "write",
  title: "Write repository file",
  description: "Write a UTF-8 text file in the repository.",
  resultStyle: "text",
  sideEffectLevel: "write",
  requiresConfirmation: true,
};

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

describe("optimizer tool renderer", () => {
  test("renders deterministic IDs, order, and stable object schema ordering", () => {
    const policy = resolvedPolicy(nativeModelProfile);
    const first = renderToolContracts({
      canonicalToolSpecs: [canonicalWriteTool, canonicalReadTool],
      resolvedPolicy: policy,
    });
    const second = renderToolContracts({
      canonicalToolSpecs: [canonicalWriteTool, canonicalReadTool],
      resolvedPolicy: policy,
    });

    expect(first).toEqual(second);
    expect(first.map((contract) => contract.name)).toEqual(["repo_read", "repo_write"]);
    expect(first.map((contract) => contract.renderedToolId)).toEqual(second.map((contract) => contract.renderedToolId));
    expect(Object.keys(first[0]!.inputSchema)).toEqual(["properties", "required", "type"]);
    expect(Object.keys(first[0]!.inputSchema.properties as object)).toEqual(["mode", "path", "query"]);
  });

  test("does not mutate canonical specs while rendering", () => {
    const before = JSON.parse(JSON.stringify(canonicalReadTool));

    renderToolContract(canonicalReadTool, resolvedPolicy(textModelProfile));

    expect(canonicalReadTool).toEqual(before);
    expect(canonicalReadTool.inputSchema.properties).toHaveProperty("query.anyOf");
    expect(canonicalReadTool.inputSchema.properties).toHaveProperty("mode.enum");
  });

  test("preserves full object schema for native/json models and compacts text fallback contracts", () => {
    const nativeContract = renderToolContract(canonicalReadTool, resolvedPolicy(nativeModelProfile));
    const textContract = renderToolContract(canonicalReadTool, resolvedPolicy(textModelProfile));

    expect(nativeContract.inputSchema.properties).toHaveProperty("query.anyOf");
    expect(nativeContract.inputSchema.properties).toHaveProperty("mode.enum");
    expect(nativeContract.promptFragments).toEqual([]);

    expect(textContract.inputSchema.properties).not.toHaveProperty("query.anyOf");
    expect(textContract.inputSchema.properties).not.toHaveProperty("mode.enum");
    expect(textContract.description).toContain("Arguments:");
    expect(textContract.promptFragments[0]).toContain("Text contract");
    expect(textContract.examples).toHaveLength(1);
  });

  test("propagates policy versions and model/profile identifiers", () => {
    const policy = resolvedPolicy(nativeModelProfile, {
      canonicalToolVersion: "canonical-tools.v9",
      renderedToolVersion: "rendered-tools.v9",
      resultStyleVersion: "result-style.v9",
      verificationPolicyVersion: "verification.v9",
    });
    const contract = renderToolContract(canonicalReadTool, policy);

    expect(contract.modelProfileId).toBe("model.native");
    expect(contract.policyId).toBe("policy.test");
    expect(contract.canonicalToolVersion).toBe("canonical-tools.v9");
    expect(contract.renderedToolVersion).toBe("rendered-tools.v9");
    expect(contract.resultStyleVersion).toBe("result-style.v9");
    expect(contract.canonicalToolVersion).not.toBe(canonicalReadTool.canonicalToolVersion);
  });

  test("prefers a complete promoted rendered contract set and falls back to fresh rendering otherwise", () => {
    const policy = resolvedPolicy(nativeModelProfile);
    const fresh = renderToolContracts({
      canonicalToolSpecs: [canonicalReadTool, canonicalWriteTool],
      resolvedPolicy: policy,
    });
    const promoted = fresh.map((contract) => ({
      ...contract,
      renderedToolId: `${contract.renderedToolId}.promoted`,
      description: `Promoted ${contract.description}`,
    }));
    const selected = selectRenderedToolContracts({
      canonicalToolSpecs: [canonicalReadTool, canonicalWriteTool],
      resolvedPolicy: policy,
      records: promoted.map((contract) => recordFor(contract)),
    });

    expect(selected).toEqual(promoted);

    const noPromoted = selectRenderedToolContracts({
      canonicalToolSpecs: [canonicalReadTool, canonicalWriteTool],
      resolvedPolicy: policy,
      records: promoted.map((contract) => recordFor(contract, { status: "active" })),
    });
    expect(noPromoted).toEqual(fresh);

    const partialPromoted = selectRenderedToolContracts({
      canonicalToolSpecs: [canonicalReadTool, canonicalWriteTool],
      resolvedPolicy: policy,
      records: [recordFor(promoted[0]!)],
    });
    expect(partialPromoted).toEqual(fresh);
  });
});
