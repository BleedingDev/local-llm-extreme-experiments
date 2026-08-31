import { describe, expect, test } from "bun:test";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import { runEditStrategyAblation } from "../src/eval-harness/edit-strategy-ablation";
import type { ComparisonRunMetadata, EvalComparableContext, EvalRunResult } from "../src/eval-harness/types";
import { buildCandidateEvidenceBundle } from "../src/optimizer/evidence";
import { buildGepaFeedbackBundle, GepaFeedbackBundleSchema } from "../src/optimizer/gepa-feedback";

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
  comparisonRunId: "compare.gepa.baseline",
  runRole: "baseline",
  artifactId: "policy.qwen36.bleeding-agent.baseline",
  artifactVersion: "policy.v1",
  context,
};

const candidateMetadata: ComparisonRunMetadata = {
  comparisonRunId: "compare.gepa.candidate",
  runRole: "candidate",
  artifactId: "candidate.gepa",
  artifactVersion: "candidate.v1",
  context,
};

const makeRun = (input: {
  runRole: "baseline" | "candidate";
  passed?: boolean;
  metricDelta?: number;
}): EvalRunResult => {
  const metadata = input.runRole === "baseline" ? baselineMetadata : candidateMetadata;
  const passed = input.passed ?? true;
  return {
    runResultId: `run.gepa.${input.runRole}`,
    comparisonRunId: metadata.comparisonRunId,
    runRole: input.runRole,
    evalCaseId: "eval.small-edit",
    split: "dev",
    context,
    status: passed ? "passed" : "failed",
    score: passed ? 1 : 0,
    assertionResults: [
      {
        assertionId: "assert.output",
        assertionKind: "file_contains",
        passed,
        severity: passed ? "failure" : "critical",
        message: passed ? "ok" : "expected output missing",
        actual: passed ? undefined : "secret=should-not-leak-123456789",
      },
    ],
    objectiveMetrics: input.metricDelta == null
      ? []
      : [
          {
            metricId: "latency-ms",
            name: "Latency",
            value: 120,
            unit: "ms",
            higherIsBetter: false,
            delta: input.metricDelta,
          },
        ],
    changedFiles: [],
    startedAt: now,
    completedAt: now,
  };
};

describe("GEPA feedback records", () => {
  test("converts eval failures and critical scorecard regressions into textual feedback", () => {
    const baselineRun = makeRun({ runRole: "baseline" });
    const candidateRun = makeRun({ runRole: "candidate", passed: false });
    const scorecard = createEvalScorecard({
      scorecardId: "scorecard.gepa.failure",
      evalSuiteId: "suite.bleeding-agent.core",
      split: "dev",
      baseline: baselineMetadata,
      candidate: candidateMetadata,
      baselineResults: [baselineRun],
      candidateResults: [candidateRun],
      createdAt: now,
    });

    const bundle = buildGepaFeedbackBundle({
      feedbackBundleId: "gepa.bundle.eval",
      evalRunResults: [candidateRun],
      evalScorecards: [scorecard],
    });

    expect(bundle.records.map((record) => record.source)).toEqual([
      "eval_run",
      "eval_scorecard",
      "eval_scorecard",
    ]);
    expect(bundle.records[0]).toMatchObject({
      severity: "critical",
      modelProfileId: "model.qwen36.local",
      codebaseProfileId: "codebase.bleeding-agent",
      policyId: "policy.qwen36.bleeding-agent",
      evalCaseIds: ["eval.small-edit"],
      runResultIds: ["run.gepa.candidate"],
    });
    expect(JSON.stringify(bundle)).not.toContain("should-not-leak");
    expect(bundle.redactionCount).toBeGreaterThan(0);
    expect(GepaFeedbackBundleSchema.parse(bundle)).toEqual(bundle);
  });

  test("converts trace evidence into bounded lineage-preserving feedback", () => {
    const evidence = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.gepa.trace",
      selectedSpanExcerpts: [
        {
          traceId: "trace-tool",
          spanId: "span-tool",
          title: "repo_write failed",
          text: "missing required path",
          argumentHash: "arg-hash-1",
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

    const bundle = buildGepaFeedbackBundle({
      evidenceBundles: [evidence],
      limits: { maxTextChars: 48 },
    });

    expect(bundle.records).toHaveLength(1);
    expect(bundle.records[0]).toMatchObject({
      source: "trace_evidence",
      severity: "warning",
      traceIds: ["trace-tool"],
      spanIds: ["span-tool"],
      modelProfileId: "model.qwen36.local",
      policyId: "policy.qwen36.bleeding-agent",
    });
    expect(bundle.records[0]?.feedback.length).toBeGreaterThan(0);
    expect(bundle.records[0]?.truncated).toBe(true);
  });

  test("captures test output, truncation mistakes, LLM critiques, and metric direction", () => {
    const run = makeRun({ runRole: "candidate", passed: true, metricDelta: -42 });
    const bundle = buildGepaFeedbackBundle({
      evalRunResults: [run],
      testOutputs: [
        {
          id: "typecheck",
          text: "TS2322: type mismatch",
          modelProfileId: "model.qwen36.local",
          codebaseProfileId: "codebase.bleeding-agent",
          policyId: "policy.qwen36.bleeding-agent",
          evalCaseIds: ["eval.small-edit"],
        },
      ],
      truncationMistakes: [
        {
          id: "tail-facts",
          text: "lost final instruction after context compaction",
          traceIds: ["trace-tail"],
        },
      ],
      llmCritiques: [
        {
          id: "judge-note",
          text: "candidate overfit train fixture; keep holdout hidden",
        },
      ],
    });

    expect(bundle.records.map((record) => record.source)).toEqual([
      "test_output",
      "truncation",
      "eval_run",
      "llm_critique",
    ]);
    expect(bundle.records.find((record) => record.metricIds.includes("latency-ms"))).toMatchObject({
      higherIsBetter: false,
      severity: "warning",
    });
    expect(bundle.records.find((record) => record.source === "truncation")?.traceIds).toEqual(["trace-tail"]);
  });

  test("turns edit ablation failures into edit-specific GEPA feedback without holdout leakage", () => {
    const visibleReport = runEditStrategyAblation({ createdAt: now });
    const holdoutReport = runEditStrategyAblation({
      splits: ["holdout"],
      includeHoldout: true,
      createdAt: now,
    });
    const bundle = buildGepaFeedbackBundle({
      editAblationReports: [visibleReport, holdoutReport],
    });

    const editRecords = bundle.records.filter((record) => record.source === "edit_ablation");
    expect(editRecords.length).toBeGreaterThan(0);
    expect(editRecords.every((record) => record.evalCaseIds.every((id) => !holdoutReport.selectedEvalCaseIds.includes(id))))
      .toBe(true);
    expect(editRecords.find((record) => record.editStrategyFamilies.includes("apply_patch"))).toMatchObject({
      severity: "critical",
      objective:
        "Optimize edit strategy policy, rendered edit contracts, fallback order, repair policy, rollback behavior, and verifier enforcement without changing runtime source code.",
    });
    expect(JSON.stringify(editRecords)).toContain("postApplyConsistency=inconsistent");
  });

  test("is deterministic and caps feedback record count", () => {
    const first = buildGepaFeedbackBundle({
      feedbackBundleId: "gepa.bundle.cap",
      testOutputs: [
        { id: "b", text: "second" },
        { id: "a", text: "first" },
      ],
      limits: { maxRecords: 1 },
    });
    const second = buildGepaFeedbackBundle({
      feedbackBundleId: "gepa.bundle.cap",
      testOutputs: [
        { id: "a", text: "first" },
        { id: "b", text: "second" },
      ],
      limits: { maxRecords: 1 },
    });

    expect(first).toEqual(second);
    expect(first.records).toHaveLength(1);
    expect(first.records[0]?.feedbackId).toBe("gepa.test_output.a");
  });
});
