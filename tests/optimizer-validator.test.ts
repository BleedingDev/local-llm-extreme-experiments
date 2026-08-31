import { describe, expect, test } from "bun:test";
import { validateCandidatePatch } from "../src/optimizer/validator";
import type { CandidatePatch, OptimizerRegistryRecord } from "../src/optimizer/types";

const now = "2026-04-30T00:00:00.000Z";

const policyRecord: OptimizerRegistryRecord = {
  registryRecordId: "registry.policy.qwen36.bleeding-agent",
  recordKind: "model_codebase_policy",
  schemaVersion: "optimizer-schema.v1",
  recordVersion: "record.v1",
  status: "active",
  createdAt: now,
  updatedAt: now,
  contentHash: "sha256:policy",
  payload: {
    policyId: "policy.qwen36.bleeding-agent",
    modelProfileId: "model.qwen36.local",
    codebaseProfileId: "codebase.bleeding-agent",
    canonicalToolVersion: "canonical-tools.v1",
    renderedToolVersion: "rendered-tools.v1",
    resultStyleVersion: "result-style.v1",
    verificationPolicyVersion: "verification.v1",
    verificationGates: [
      {
        gateId: "tests-pass",
        commandId: "test",
        comparator: "eq",
        threshold: 0,
        required: true,
      },
    ],
  },
};

const toolRecord: OptimizerRegistryRecord = {
  registryRecordId: "registry.rendered-tool.repo-write",
  recordKind: "rendered_tool_contract",
  schemaVersion: "optimizer-schema.v1",
  recordVersion: "record.v1",
  status: "active",
  createdAt: now,
  updatedAt: now,
  contentHash: "sha256:tool",
  payload: {
    renderedToolId: "tool.repo-write.qwen36",
    canonicalToolId: "tool.repo-write",
    canonicalToolVersion: "canonical-tools.v1",
    renderedToolVersion: "rendered-tools.v1",
    modelProfileId: "model.qwen36.local",
    policyId: "policy.qwen36.bleeding-agent",
    renderer: "renderer.default",
    rendererVersion: "renderer.v1",
    name: "repo_write",
    description: "Write one repository file.",
    inputSchema: {
      type: "object",
    },
    resultStyleVersion: "result-style.v1",
    promptFragments: [],
  },
};

const candidatePatch = (overrides: Partial<CandidatePatch> = {}): CandidatePatch => ({
  candidatePatchId: "candidate.validator.tool",
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  scope: {
    artifactKind: "rendered_tool_contract",
    artifactId: "tool.repo-write.qwen36",
    allowedJsonPointers: ["/description", "/promptFragments/0"],
  },
  operations: [
    {
      op: "add",
      path: "/promptFragments/0",
      value: "Validate required path before calling.",
    },
  ],
  rationale: "Tighten rendered tool guidance.",
  createdAt: now,
  sourceTraceIds: ["trace-tool"],
  ...overrides,
});

describe("candidate patch validator", () => {
  test("accepts parseable scoped candidates with existing target and matching base hash", () => {
    const result = validateCandidatePatch({
      candidate: candidatePatch(),
      records: [toolRecord, policyRecord],
      expectedBaseHashes: {
        "tool.repo-write.qwen36": "sha256:tool",
      },
      actualBaseHashes: {
        "tool.repo-write.qwen36": "sha256:tool",
      },
    });

    expect(result).toEqual({
      candidatePatchId: "candidate.validator.tool",
      valid: true,
      issues: [],
    });
  });

  test("rejects missing targets and missing base hashes", () => {
    const result = validateCandidatePatch({
      candidate: candidatePatch(),
      records: [],
    });

    expect(result.valid).toBe(false);
    expect(result.issues.map((issue) => issue.code)).toEqual([
      "target_missing",
      "base_hash_missing",
    ]);
  });

  test("rejects operations outside declared scope and secret-looking values", () => {
    const result = validateCandidatePatch({
      candidate: candidatePatch({
        operations: [
          {
            op: "replace",
            path: "/inputSchema/properties/token",
            value: "api_key=should-not-appear-123456789",
          },
        ],
      }),
      records: [toolRecord],
      expectedBaseHashes: {
        "tool.repo-write.qwen36": "sha256:tool",
      },
      actualBaseHashes: {
        "tool.repo-write.qwen36": "sha256:tool",
      },
    });

    expect(result.valid).toBe(false);
    expect(result.issues.map((issue) => issue.code)).toEqual([
      "scope_violation",
      "secret_like_value",
    ]);
  });

  test("rejects base hash mismatches and oversized patch operations", () => {
    const result = validateCandidatePatch({
      candidate: candidatePatch(),
      records: [toolRecord],
      expectedBaseHashes: {
        "tool.repo-write.qwen36": "sha256:old",
      },
      actualBaseHashes: {
        "tool.repo-write.qwen36": "sha256:new",
      },
      maxPatchBytes: 4,
    });

    expect(result.valid).toBe(false);
    expect(result.issues.map((issue) => issue.code)).toEqual([
      "patch_too_large",
      "base_hash_mismatch",
    ]);
  });

  test("enforces required eval gates from target policy or candidate operations", () => {
    const policyCandidate = candidatePatch({
      candidatePatchId: "candidate.validator.policy",
      scope: {
        artifactKind: "model_codebase_policy",
        artifactId: "policy.qwen36.bleeding-agent",
        allowedJsonPointers: ["/verificationGates/0"],
      },
      operations: [
        {
          op: "add",
          path: "/verificationGates/0",
          value: {
            gateId: "tool-success-rate",
            metric: "tool-call-success-rate",
            comparator: "gte",
            threshold: 0.95,
            required: true,
          },
        },
      ],
    });

    const result = validateCandidatePatch({
      candidate: policyCandidate,
      records: [policyRecord],
      expectedBaseHashes: {
        "policy.qwen36.bleeding-agent": "sha256:policy",
      },
      actualBaseHashes: {
        "policy.qwen36.bleeding-agent": "sha256:policy",
      },
      requiredEvalGateIds: ["tests-pass", "tool-success-rate", "holdout-pass"],
    });

    expect(result.valid).toBe(false);
    expect(result.issues).toEqual([
      {
        severity: "error",
        code: "required_eval_gate_missing",
        message: "required eval gate missing: holdout-pass",
      },
    ]);
  });

  test("reports malformed candidate schemas before deeper checks", () => {
    const result = validateCandidatePatch({
      candidate: {
        candidatePatchId: "",
      },
      records: [toolRecord],
    });

    expect(result.valid).toBe(false);
    expect(result.issues[0]?.code).toBe("schema_invalid");
  });
});
