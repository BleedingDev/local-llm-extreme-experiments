import { describe, expect, test } from "bun:test";
import {
  CandidatePatchSchema,
  CodebaseProfileSchema,
  EvalResultSchema,
  ModelCodebasePolicySchema,
  ModelProfileSchema,
  OptimizerRegistryRecordSchema,
  PromotionDecisionSchema,
  RenderedToolContractSchema,
  CanonicalToolSpecSchema,
} from "../src/optimizer/types";

const now = "2026-04-30T00:00:00.000Z";

const modelProfile = {
  modelProfileId: "model.qwen36.local",
  displayName: "Qwen 3.6 Local",
  provider: "openai-compatible",
  model: "majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit",
  contextWindowTokens: 131072,
  maxOutputTokens: 4096,
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
};

const codebaseProfile = {
  codebaseProfileId: "codebase.bleeding-agent",
  displayName: "BleedingAgent",
  rootFingerprint: "sha256:repo",
  languages: ["typescript", "shell"],
  packageManagers: ["npm", "bun"],
  primaryPackageManager: "npm",
  generatedDirs: ["dist"],
  ignoredDirs: ["node_modules", ".bag"],
  typecheckCommands: [
    {
      commandId: "typecheck",
      command: ["npm", "run", "typecheck"],
    },
  ],
  testCommands: [
    {
      commandId: "test",
      command: ["npm", "test"],
    },
  ],
  testRiskTiers: [
    {
      tierId: "risk.test",
      description: "Run unit tests before promotion.",
      commandIds: ["test"],
    },
  ],
  protectedPaths: [".bag", ".git", "node_modules"],
  conventions: ["package-manager.npm", "source-root.src"],
  knownFailures: [
    {
      failureId: "known-failure.test.flaky",
      source: "verifier",
      commandId: "test",
      summary: "Historical flaky verifier failure.",
      lastExitCode: 1,
    },
  ],
  acpClientQuirks: [
    {
      quirkId: "acp.client.terminal-create.optional",
      affectedCapability: "terminal/create",
      behavior: "Terminal capability may be absent.",
    },
  ],
  verificationPolicyVersion: "verification.v1",
};

const policy = {
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  codebaseRootFingerprint: "sha256:repo",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
  editStrategyVersion: "edit-strategy.v1",
  renderedEditContractVersion: "rendered-edit-contract.v1",
  editFallbackPolicyVersion: "edit-fallback.v1",
  editRepairPolicyVersion: "edit-repair.v1",
  editVerifierPolicyVersion: "edit-verifier.v1",
  editObjectiveSetId: "edit-objectives.default.v1",
  candidateScopes: [
    {
      artifactKind: "rendered_tool_contract",
      artifactId: "tool.repo-read.qwen36",
      allowedJsonPointers: ["/description", "/promptFragments/0"],
    },
  ],
  verificationGates: [
    {
      gateId: "tests-pass",
      commandId: "test",
      comparator: "eq",
      threshold: 0,
    },
  ],
};

const canonicalToolSpec = {
  canonicalToolId: "tool.repo-read",
  canonicalToolVersion: "canonical-tools.v1",
  namespace: "repo",
  name: "read",
  title: "Read repository file",
  description: "Read a UTF-8 text file from the repository.",
  inputSchema: {
    type: "object",
    properties: {
      path: { type: "string" },
    },
    required: ["path"],
  },
};

const renderedToolContract = {
  renderedToolId: "tool.repo-read.qwen36",
  canonicalToolId: "tool.repo-read",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  modelProfileId: "model.qwen36.local",
  policyId: "policy.qwen36.bleeding-agent",
  renderer: "renderer.default",
  rendererVersion: "renderer.v1",
  name: "repo_read",
  description: "Read one repository file by relative path.",
  inputSchema: canonicalToolSpec.inputSchema,
  resultStyleVersion: "result-style.v1",
  promptFragments: ["Return concise file contents with path context."],
};

const candidatePatch = {
  candidatePatchId: "candidate.tool-rendering.1",
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  codebaseRootFingerprint: "sha256:repo",
  scope: {
    artifactKind: "rendered_tool_contract",
    artifactId: "tool.repo-read.qwen36",
    allowedJsonPointers: ["/description"],
  },
  operations: [
    {
      op: "replace",
      path: "/description",
      value: "Read exactly one repository file by relative path.",
    },
  ],
  rationale: "Clarify single-file behavior for this model.",
  createdAt: now,
};

const evalResult = {
  evalResultId: "eval.candidate.1",
  candidatePatchId: "candidate.tool-rendering.1",
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  codebaseRootFingerprint: "sha256:repo",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
  status: "passed",
  score: 0.91,
  metrics: [
    {
      metricId: "tool-call-success-rate",
      value: 0.99,
      unit: "ratio",
    },
  ],
  startedAt: now,
  completedAt: now,
};

const promotionDecision = {
  promotionDecisionId: "promotion.candidate.1",
  decision: "promote",
  policyId: "policy.qwen36.bleeding-agent",
  candidatePatchId: "candidate.tool-rendering.1",
  evalResultId: "eval.candidate.1",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  codebaseRootFingerprint: "sha256:repo",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
  reason: "Candidate passed deterministic gates.",
  decidedAt: now,
};

describe("optimizer foundation schemas", () => {
  test("parse representative profile, policy, tool, candidate, eval, and promotion objects", () => {
    expect(ModelProfileSchema.parse(modelProfile).toolCallingMode).toBe("json");
    const parsedCodebaseProfile = CodebaseProfileSchema.parse(codebaseProfile);
    expect(parsedCodebaseProfile.sourceRoots).toEqual(["src"]);
    expect(parsedCodebaseProfile.generatedDirs).toEqual(["dist"]);
    expect(parsedCodebaseProfile.testRiskTiers[0]?.tierId).toBe("risk.test");
    expect(parsedCodebaseProfile.acpClientQuirks[0]?.quirkId).toBe("acp.client.terminal-create.optional");
    expect(ModelCodebasePolicySchema.parse(policy).riskTolerance).toBe("low");
    expect(ModelCodebasePolicySchema.parse(policy).editObjectiveSetId).toBe("edit-objectives.default.v1");
    expect(CanonicalToolSpecSchema.parse(canonicalToolSpec).sideEffectLevel).toBe("read");
    expect(RenderedToolContractSchema.parse(renderedToolContract).resultStyle).toBe("text");
    expect(CandidatePatchSchema.parse(candidatePatch).operations).toHaveLength(1);
    expect(EvalResultSchema.parse(evalResult).metrics[0]?.higherIsBetter).toBe(true);
    expect(PromotionDecisionSchema.parse(promotionDecision).appliesToNewSessionsOnly).toBe(true);
  });

  test("parses typed registry records around payloads", () => {
    const record = OptimizerRegistryRecordSchema.parse({
      registryRecordId: "registry.model.qwen36.local",
      recordKind: "model_profile",
      schemaVersion: "optimizer-schema.v1",
      recordVersion: "record.v1",
      createdAt: now,
      updatedAt: now,
      payload: modelProfile,
    });

    expect(record.recordKind).toBe("model_profile");
    expect(record.payload.modelProfileId).toBe("model.qwen36.local");
  });

  test("rejects source-file candidate scopes", () => {
    const result = CandidatePatchSchema.safeParse({
      ...candidatePatch,
      scope: {
        artifactKind: "source_file",
        artifactId: "src/config.ts",
      },
    });

    expect(result.success).toBe(false);
  });

  test("rejects malformed patch operations", () => {
    const result = CandidatePatchSchema.safeParse({
      ...candidatePatch,
      operations: [
        {
          op: "replace",
          path: "description",
        },
      ],
    });

    expect(result.success).toBe(false);
  });
});
