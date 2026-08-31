import { describe, expect, test } from "bun:test";
import { buildCandidateEvidenceBundle, type CandidateEvidenceBundle } from "../src/optimizer/evidence";
import { generateCandidatePatches } from "../src/optimizer/candidates";
import { CandidatePatchSchema } from "../src/optimizer/types";

const now = "2026-04-30T00:00:00.000Z";

const lineage = {
  modelProfileIds: ["model.qwen36.local"],
  codebaseProfileIds: ["codebase.bleeding-agent"],
  policyIds: ["policy.qwen36.bleeding-agent"],
  canonicalToolVersions: ["canonical-tools.v1"],
  renderedToolVersions: ["rendered-tools.v1"],
  resultStyleVersions: ["result-style.v1"],
  verificationPolicyVersions: ["verification.v1"],
};

const editLineage = {
  ...lineage,
  editStrategyVersions: ["edit-strategy.v1"],
  renderedEditContractVersions: ["rendered-edit-contract.v1"],
  editFallbackPolicyVersions: ["edit-fallback.v1"],
  editRepairPolicyVersions: ["edit-repair.v1"],
  editVerifierPolicyVersions: ["edit-verifier.v1"],
  editObjectiveSetIds: ["edit-objectives.default.v1"],
  editStrategyIds: ["edit.apply-patch.v1"],
  editStrategyFamilies: ["apply_patch"],
  canonicalEditToolSpecIds: ["edit.apply-patch.v1"],
  renderedEditToolContractIds: ["rendered.edit.apply-patch.v1.model.qwen36.local"],
};

const toolEvidence = (): CandidateEvidenceBundle =>
  buildCandidateEvidenceBundle({
    evidenceBundleId: "evidence.generator.tool",
    createdAt: now,
    selectedSpanExcerpts: [
      {
        traceId: "trace-tool",
        spanId: "span-tool",
        title: "repo_write failed",
        text: "missing required path",
        argumentHash: "arg-hash-1",
        toolName: "tool.repo-write.qwen36",
        lineage,
      },
    ],
  });

const evalEvidence = (): CandidateEvidenceBundle =>
  buildCandidateEvidenceBundle({
    evidenceBundleId: "evidence.generator.eval",
    createdAt: now,
    evalRunResults: [
      {
        runResultId: "run.generator.candidate",
        comparisonRunId: "compare.generator.candidate",
        runRole: "candidate",
        evalCaseId: "eval.small-edit",
        split: "dev",
        context: {
          policyId: "policy.qwen36.bleeding-agent",
          modelProfileId: "model.qwen36.local",
          codebaseProfileId: "codebase.bleeding-agent",
          modelServerId: "server.local-mlx",
          modelServerProfileId: "server-profile.qwen36.rotorquant",
          canonicalToolVersion: "canonical-tools.v1",
          renderedToolVersion: "rendered-tools.v1",
          resultStyleVersion: "result-style.v1",
          verificationPolicyVersion: "verification.v1",
        },
        status: "failed",
        score: 0,
        assertionResults: [
          {
            assertionId: "assert.output",
            assertionKind: "file_contains",
            passed: false,
            severity: "critical",
            message: "expected file output missing",
          },
        ],
        objectiveMetrics: [],
        changedFiles: [],
        startedAt: now,
        completedAt: now,
      },
    ],
  });

describe("candidate patch generator", () => {
  test("generates deterministic parseable tool guidance candidates", () => {
    const first = generateCandidatePatches({
      evidence: toolEvidence(),
      createdAt: now,
    });
    const second = generateCandidatePatches({
      evidence: toolEvidence(),
      createdAt: now,
    });

    expect(first).toEqual(second);
    expect(first.diagnostics).toEqual([]);
    expect(first.candidates).toHaveLength(1);
    expect(CandidatePatchSchema.parse(first.candidates[0])).toEqual(first.candidates[0]);
    expect(first.candidates[0]?.scope).toEqual({
      artifactKind: "rendered_tool_contract",
      artifactId: "tool.repo-write.qwen36",
      allowedJsonPointers: [
        "/description",
        "/inputSchema",
        "/resultStyle",
        "/promptFragments/0",
        "/examples/0/expectedResultShape",
      ],
    });
    expect(first.candidates[0]?.sourceTraceIds).toEqual(["trace-tool"]);
  });

  test("skips ambiguous lineage instead of inventing ids", () => {
    const ambiguous = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.generator.ambiguous",
      createdAt: now,
      selectedSpanExcerpts: [
        {
          traceId: "trace-ambiguous",
          spanId: "span-ambiguous",
          text: "tool failed",
          toolName: "tool.repo-write.qwen36",
          lineage: {
            ...lineage,
            policyIds: ["policy.one", "policy.two"],
          },
        },
      ],
    });

    const result = generateCandidatePatches({
      evidence: ambiguous,
      createdAt: now,
    });

    expect(result.candidates).toEqual([]);
    expect(result.diagnostics).toEqual([
      {
        observationId: "span.trace-ambiguous.span-ambiguous.0",
        severity: "warning",
        reason: "skipped observation with missing or ambiguous policy/model/codebase lineage",
      },
    ]);
  });

  test("maps eval failures to model-codebase policy verification gates", () => {
    const result = generateCandidatePatches({
      evidence: evalEvidence(),
      createdAt: now,
    });

    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]?.scope.artifactKind).toBe("model_codebase_policy");
    expect(result.candidates[0]?.scope.artifactId).toBe("policy.qwen36.bleeding-agent");
    expect(result.candidates[0]?.operations[0]).toMatchObject({
      op: "add",
      path: "/verificationGates/0",
      value: {
        metric: "aggregate-score",
        comparator: "gte",
        threshold: 1,
        required: true,
      },
    });
  });

  test("maps edit evidence to rendered edit contracts before generic tool guidance", () => {
    const evidence = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.generator.edit-contract",
      createdAt: now,
      selectedSpanExcerpts: [
        {
          traceId: "trace-edit",
          spanId: "span-edit",
          title: "edit apply_patch parsed but verification failed",
          text: "postApplyConsistency=inconsistent verification=failed",
          toolName: "tool.generic-write",
          lineage: editLineage,
        },
      ],
    });
    const result = generateCandidatePatches({ evidence, createdAt: now });

    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]?.scope).toEqual({
      artifactKind: "rendered_tool_contract",
      artifactId: "rendered.edit.apply-patch.v1.model.qwen36.local",
      allowedJsonPointers: [
        "/description",
        "/inputSchema",
        "/resultStyle",
        "/promptFragments/0",
        "/examples/0",
        "/examples/0/expectedResultShape",
      ],
    });
    expect(JSON.stringify(result.candidates[0]?.operations)).toContain("applied-but-broken");
  });

  test("maps edit family evidence without a rendered contract to edit policy dimensions", () => {
    const evidence = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.generator.edit-policy",
      createdAt: now,
      selectedSpanExcerpts: [
        {
          traceId: "trace-edit-policy",
          spanId: "span-edit-policy",
          title: "edit exact_replace failed repeated snippet",
          text: "exact_match_ambiguous",
          lineage: {
            ...editLineage,
            renderedEditToolContractIds: [],
          },
        },
      ],
    });
    const result = generateCandidatePatches({ evidence, createdAt: now });

    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]?.scope).toEqual({
      artifactKind: "model_codebase_policy",
      artifactId: "policy.qwen36.bleeding-agent",
      allowedJsonPointers: [
        "/editStrategyVersion",
        "/editFallbackPolicyVersion",
        "/editRepairPolicyVersion",
        "/editVerifierPolicyVersion",
        "/editObjectiveSetId",
        "/verificationGates/0",
      ],
    });
    expect(result.candidates[0]?.operations.map((operation) => operation.path)).toEqual([
      "/editStrategyVersion",
      "/editFallbackPolicyVersion",
      "/editRepairPolicyVersion",
      "/editVerifierPolicyVersion",
      "/editObjectiveSetId",
      "/verificationGates/0",
    ]);
    expect(result.candidates[0]?.operations.at(-1)).toMatchObject({
      op: "add",
      path: "/verificationGates/0",
      value: {
        metric: "edit-final-consistency-score",
        required: true,
      },
    });
  });

  test("caps generated candidates and reports skipped overflow", () => {
    const evidence = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.generator.cap",
      createdAt: now,
      selectedSpanExcerpts: [
        {
          traceId: "trace-a",
          spanId: "span-a",
          text: "first failure",
          toolName: "tool.a",
          lineage,
        },
        {
          traceId: "trace-b",
          spanId: "span-b",
          text: "second failure",
          toolName: "tool.b",
          lineage,
        },
      ],
    });

    const result = generateCandidatePatches({
      evidence,
      createdAt: now,
      maxCandidates: 1,
    });

    expect(result.candidates).toHaveLength(1);
    expect(result.diagnostics).toContainEqual({
      severity: "info",
      reason: "candidate cap reached at 1",
    });
  });

  test("maps truncation and result-shape evidence to result style policy scope", () => {
    const evidence = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.generator.truncation",
      createdAt: now,
      selectedSpanExcerpts: [
        {
          traceId: "trace-result-style",
          spanId: "span-result-style",
          title: "tool output truncation caused parser failure",
          text: "result style lost structured tail fields; parse success regressed",
          lineage,
        },
      ],
    });

    const result = generateCandidatePatches({ evidence, createdAt: now });

    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]?.scope).toEqual({
      artifactKind: "model_codebase_policy",
      artifactId: "policy.qwen36.bleeding-agent",
      allowedJsonPointers: ["/resultStyleVersion", "/verificationGates/0"],
    });
    expect(result.candidates[0]?.operations.map((operation) => operation.path)).toEqual([
      "/resultStyleVersion",
      "/verificationGates/0",
    ]);
  });
});
