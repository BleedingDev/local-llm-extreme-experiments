import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { buildCandidateEvidenceBundle } from "../src/optimizer/evidence";
import { materializeCandidateArtifacts } from "../src/optimizer/materialization";
import type { CandidatePatch, PromotionDecision } from "../src/optimizer/types";
import type { CandidateValidationResult } from "../src/optimizer/validator";

const now = "2026-04-30T00:00:00.000Z";

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.materialization.tool",
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  scope: {
    artifactKind: "rendered_tool_contract",
    artifactId: "tool.repo-write.qwen36",
    allowedJsonPointers: ["/promptFragments/0"],
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
};

const evidence = buildCandidateEvidenceBundle({
  evidenceBundleId: "evidence.materialization",
  createdAt: now,
  selectedSpanExcerpts: [
    {
      traceId: "trace-tool",
      spanId: "span-tool",
      text: "missing required path",
      toolName: "tool.repo-write.qwen36",
      lineage: {
        modelProfileIds: ["model.qwen36.local"],
        codebaseProfileIds: ["codebase.bleeding-agent"],
        policyIds: ["policy.qwen36.bleeding-agent"],
        canonicalToolVersions: ["canonical-tools.v1"],
        renderedToolVersions: ["rendered-tools.v1"],
        resultStyleVersions: ["result-style.v1"],
        verificationPolicyVersions: ["verification.v1"],
      },
    },
  ],
});

const validation: CandidateValidationResult = {
  candidatePatchId: "candidate.materialization.tool",
  valid: true,
  issues: [],
};

const decision: PromotionDecision = {
  promotionDecisionId: "decision.materialization.hold",
  decision: "hold",
  policyId: "policy.qwen36.bleeding-agent",
  candidatePatchId: "candidate.materialization.tool",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
  reason: "Waiting for candidate eval.",
  decidedAt: now,
};

describe("candidate artifact materialization", () => {
  test("writes patch, evidence, validation, decision, report, and manifest artifacts", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-candidate-materialization-"));
    try {
      const manifest = materializeCandidateArtifacts({
        cwd,
        candidate,
        evidence,
        validation,
        decision,
        createdAt: now,
      });

      expect(manifest.candidatePatchId).toBe(candidate.candidatePatchId);
      expect(manifest.files).toMatchObject({
        patch: "patch.json",
        evidence: "evidence.json",
        validation: "validation.json",
        decision: "decision.json",
        report: "report.md",
      });
      expect(JSON.parse(readFileSync(join(manifest.artifactDir, "patch.json"), "utf8"))).toEqual(candidate);
      expect(JSON.parse(readFileSync(join(manifest.artifactDir, "evidence.json"), "utf8")).evidenceBundleId)
        .toBe("evidence.materialization");
      expect(JSON.parse(readFileSync(join(manifest.artifactDir, "validation.json"), "utf8"))).toEqual(validation);
      expect(JSON.parse(readFileSync(join(manifest.artifactDir, "decision.json"), "utf8")).decision).toBe("hold");
      expect(JSON.parse(readFileSync(join(manifest.artifactDir, "manifest.json"), "utf8"))).toEqual(manifest);
      expect(readFileSync(join(manifest.artifactDir, "report.md"), "utf8")).toContain(
        "# Candidate candidate.materialization.tool",
      );
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  test("uses custom report content and keeps optional eval artifacts absent when not supplied", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-candidate-materialization-custom-"));
    try {
      const manifest = materializeCandidateArtifacts({
        cwd,
        candidate,
        evidence,
        validation,
        reportMarkdown: "# Custom Report\n",
        createdAt: now,
      });

      expect(manifest.files.baselineEval).toBeUndefined();
      expect(manifest.files.candidateEval).toBeUndefined();
      expect(manifest.files.decision).toBeUndefined();
      expect(readFileSync(join(manifest.artifactDir, "report.md"), "utf8")).toBe("# Custom Report\n");
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});
