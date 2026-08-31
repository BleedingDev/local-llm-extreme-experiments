import { describe, expect, test } from "bun:test";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type { ComparisonRunMetadata, EvalComparableContext, EvalRunResult } from "../src/eval-harness/types";
import {
  buildCandidateEvidenceBundle,
  CandidateEvidenceBundleSchema,
  type CandidateEvidenceLineage,
} from "../src/optimizer/evidence";
import type { TraceFailureCluster, TraceOptimizerDimensions } from "../src/trace-analysis";

const now = "2026-04-30T00:00:00.000Z";

const context: EvalComparableContext = {
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  modelServerId: "server.local-mlx",
  modelServerProfileId: "server-profile.qwen36.rotorquant",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
};

const baselineMetadata: ComparisonRunMetadata = {
  comparisonRunId: "compare.evidence.baseline",
  runRole: "baseline",
  artifactId: "policy.qwen36.bleeding-agent.baseline",
  artifactVersion: "policy.v1",
  context,
};

const candidateMetadata: ComparisonRunMetadata = {
  comparisonRunId: "compare.evidence.candidate",
  runRole: "candidate",
  artifactId: "candidate.evidence",
  artifactVersion: "candidate.v1",
  context,
};

const dimensions: TraceOptimizerDimensions = {
  modelProfileIds: ["model.qwen36.local"],
  codebaseProfileIds: ["codebase.bleeding-agent"],
  policyIds: ["policy.qwen36.bleeding-agent"],
  canonicalToolVersions: ["canonical-tools.v1"],
  renderedToolVersions: ["rendered-tools.v1"],
  resultStyleVersions: ["result-style.v1"],
  verificationPolicyVersions: ["verification.v1"],
  editStrategyVersions: ["edit-strategy.v1"],
  renderedEditContractVersions: ["rendered-edit-contract.v1"],
  editFallbackPolicyVersions: ["edit-fallback.v1"],
  editRepairPolicyVersions: ["edit-repair.v1"],
  editVerifierPolicyVersions: ["edit-verifier.v1"],
  editObjectiveSetIds: ["edit-objectives.default.v1"],
  editStrategyIds: ["edit.apply-patch.v1"],
  editStrategyFamilies: ["apply_patch"],
  canonicalEditToolSpecIds: ["edit-tool.apply-patch.v1"],
  renderedEditToolContractIds: ["rendered-edit.apply-patch.qwen36"],
};

const traceFailure = (overrides: Partial<TraceFailureCluster> = {}): TraceFailureCluster => ({
  name: "repo_write",
  observationKind: "TOOL",
  count: 3,
  traces: ["trace-b", "trace-a", "trace-a"],
  messages: [
    "missing required path",
    "Authorization: Bearer sk-test-secret-value-1234567890",
    "api_key=local-secret-value-1234567890",
  ],
  inputHashes: ["hash-b", "hash-a"],
  optimizerDimensions: dimensions,
  ...overrides,
});

const makeRun = (input: {
  runRole: "baseline" | "candidate";
  status?: EvalRunResult["status"];
  passed?: boolean;
}): EvalRunResult => {
  const metadata = input.runRole === "baseline" ? baselineMetadata : candidateMetadata;
  const passed = input.passed ?? true;
  return {
    runResultId: `run.evidence.${input.runRole}`,
    comparisonRunId: metadata.comparisonRunId,
    runRole: input.runRole,
    evalCaseId: "eval.small-edit",
    split: "dev",
    context,
    status: input.status ?? (passed ? "passed" : "failed"),
    score: passed ? 1 : 0,
    assertionResults: [
      {
        assertionId: "assert.file",
        assertionKind: "file_contains",
        passed,
        severity: passed ? "failure" : "critical",
        message: passed ? "ok" : "secret=should-not-leak-123456789",
      },
    ],
    objectiveMetrics: [],
    changedFiles: [],
    startedAt: now,
    completedAt: now,
  };
};

const partialLineage = (): Partial<CandidateEvidenceLineage> => ({
  modelProfileIds: ["model.qwen36.local"],
  codebaseProfileIds: ["codebase.bleeding-agent"],
  policyIds: ["policy.qwen36.bleeding-agent"],
  canonicalToolVersions: ["canonical-tools.v1"],
  renderedToolVersions: ["rendered-tools.v1"],
  resultStyleVersions: ["result-style.v1"],
  verificationPolicyVersions: ["verification.v1"],
});

describe("candidate evidence bundles", () => {
  test("builds deterministic evidence with stable ordering and ids", () => {
    const first = buildCandidateEvidenceBundle({
      createdAt: now,
      traceFailures: [
        traceFailure({ name: "late_failure", count: 1 }),
        traceFailure(),
      ],
    });
    const second = buildCandidateEvidenceBundle({
      createdAt: now,
      traceFailures: [
        traceFailure(),
        traceFailure({ name: "late_failure", count: 1 }),
      ],
    });

    expect(first).toEqual(second);
    expect(first.observations.map((observation) => observation.observationId)).toEqual([
      "trace-failure.tool.repo_write",
      "trace-failure.tool.late_failure",
    ]);
    expect(CandidateEvidenceBundleSchema.parse(first)).toEqual(first);
  });

  test("caps observations and excerpts while redacting secret-looking content", () => {
    const bundle = buildCandidateEvidenceBundle({
      createdAt: now,
      traceFailures: [
        traceFailure({
          messages: [
            "plain oversized excerpt ".repeat(20),
            "Authorization: Bearer sk-test-secret-value-1234567890",
            "api_key=local-secret-value-1234567890",
          ],
        }),
      ],
      selectedSpanExcerpts: [
        {
          traceId: "trace-c",
          spanId: "span-c",
          text: "password=local-super-secret-value with a long payload that should be clipped",
          lineage: partialLineage(),
        },
      ],
      limits: {
        maxObservations: 1,
        maxExcerptsPerObservation: 2,
        maxExcerptChars: 32,
      },
    });

    expect(bundle.observations).toHaveLength(1);
    expect(bundle.observations[0]?.excerpts).toHaveLength(2);
    expect(JSON.stringify(bundle)).not.toContain("sk-test-secret-value");
    expect(JSON.stringify(bundle)).not.toContain("local-secret-value");
    expect(JSON.stringify(bundle)).toContain("[REDACTED_SECRET]");
    expect(bundle.redactionCount).toBeGreaterThan(0);
    expect(bundle.observations[0]?.excerpts.some((excerpt) => excerpt.truncated)).toBe(true);
  });

  test("preserves trace, span, eval, profile, policy, and tool-version lineage", () => {
    const bundle = buildCandidateEvidenceBundle({
      createdAt: now,
      traceFailures: [traceFailure()],
      selectedSpanExcerpts: [
        {
          traceId: "trace-span",
          spanId: "span-tool-call",
          text: "tool failed with missing path",
          argumentHash: "arg-hash-1",
          toolName: "repo_write",
          lineage: partialLineage(),
        },
      ],
    });

    expect(bundle.sourceTraceIds).toEqual(["trace-a", "trace-b", "trace-span"]);
    expect(bundle.sourceSpanIds).toEqual(["span-tool-call"]);
    expect(bundle.lineage.modelProfileIds).toEqual(["model.qwen36.local"]);
    expect(bundle.lineage.codebaseProfileIds).toEqual(["codebase.bleeding-agent"]);
    expect(bundle.lineage.policyIds).toEqual(["policy.qwen36.bleeding-agent"]);
    expect(bundle.lineage.canonicalToolVersions).toEqual(["canonical-tools.v1"]);
    expect(bundle.lineage.renderedToolVersions).toEqual(["rendered-tools.v1"]);
    expect(bundle.lineage.resultStyleVersions).toEqual(["result-style.v1"]);
    expect(bundle.lineage.verificationPolicyVersions).toEqual(["verification.v1"]);
    expect(bundle.lineage.editStrategyVersions).toEqual(["edit-strategy.v1"]);
    expect(bundle.lineage.renderedEditContractVersions).toEqual(["rendered-edit-contract.v1"]);
    expect(bundle.lineage.editStrategyIds).toEqual(["edit.apply-patch.v1"]);
    expect(bundle.lineage.editStrategyFamilies).toEqual(["apply_patch"]);
    expect(bundle.lineage.renderedEditToolContractIds).toEqual(["rendered-edit.apply-patch.qwen36"]);
    expect(bundle.observations.find((observation) => observation.source === "span_excerpt")?.argumentHashes)
      .toEqual(["arg-hash-1"]);
  });

  test("combines trace failures with eval scorecard and run evidence", () => {
    const baselineRun = makeRun({ runRole: "baseline" });
    const candidateRun = makeRun({ runRole: "candidate", passed: false });
    const scorecard = createEvalScorecard({
      scorecardId: "scorecard.evidence",
      evalSuiteId: "suite.bleeding-agent.core",
      split: "dev",
      baseline: baselineMetadata,
      candidate: candidateMetadata,
      baselineResults: [baselineRun],
      candidateResults: [candidateRun],
      createdAt: now,
    });

    const bundle = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.bundle.test",
      createdAt: now,
      traceFailures: [traceFailure()],
      evalRunResults: [candidateRun],
      evalScorecards: [scorecard],
    });

    expect(bundle.sourceTraceIds).toEqual(["trace-a", "trace-b"]);
    expect(bundle.sourceEvalCaseIds).toEqual(["eval.small-edit"]);
    expect(bundle.sourceRunResultIds).toEqual(["run.evidence.baseline", "run.evidence.candidate"]);
    expect(bundle.sourceScorecardIds).toEqual(["scorecard.evidence"]);
    expect(bundle.observations.map((observation) => observation.source)).toContain("trace_failure");
    expect(bundle.observations.map((observation) => observation.source)).toContain("eval_scorecard");
    expect(bundle.observations.map((observation) => observation.source)).toContain("eval_run");
  });
});
