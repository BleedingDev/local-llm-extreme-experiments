import { describe, expect, test } from "bun:test";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type {
  ComparisonRunMetadata,
  EvalComparableContext,
  EvalRunResult,
  ObjectiveMetric,
} from "../src/eval-harness/types";

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
  comparisonRunId: "compare.scorer.baseline",
  runRole: "baseline",
  artifactId: "policy.qwen36.bleeding-agent.baseline",
  artifactVersion: "policy.v1",
  context,
};

const candidateMetadata: ComparisonRunMetadata = {
  comparisonRunId: "compare.scorer.candidate",
  runRole: "candidate",
  artifactId: "candidate.eval-scorer",
  artifactVersion: "candidate.v1",
  context,
};

const makeRun = (input: {
  runRole: "baseline" | "candidate";
  evalCaseId?: string;
  status?: EvalRunResult["status"];
  assertions?: EvalRunResult["assertionResults"];
  objectiveMetrics?: ObjectiveMetric[];
  runContext?: EvalComparableContext;
}): EvalRunResult => {
  const metadata = input.runRole === "baseline" ? baselineMetadata : candidateMetadata;
  const assertions = input.assertions ?? [
    {
      assertionId: "assert.scorer.default",
      assertionKind: "file_contains",
      passed: true,
      severity: "failure",
    },
  ];
  return {
    runResultId: `run.${input.runRole}.${input.evalCaseId ?? "eval.scorer"}`,
    comparisonRunId: metadata.comparisonRunId,
    runRole: input.runRole,
    evalCaseId: input.evalCaseId ?? "eval.scorer",
    split: "dev",
    context: input.runContext ?? metadata.context,
    status: input.status ?? "passed",
    score: assertions.filter((assertion) => assertion.passed).length / assertions.length,
    assertionResults: assertions,
    objectiveMetrics: input.objectiveMetrics ?? [],
    changedFiles: [],
    startedAt: now,
    completedAt: now,
  };
};

const createScorecard = (input: {
  baselineRun: EvalRunResult;
  candidateRun: EvalRunResult;
  baseline?: ComparisonRunMetadata;
  candidate?: ComparisonRunMetadata;
}) =>
  createEvalScorecard({
    scorecardId: "scorecard.eval-scorer",
    evalSuiteId: "suite.bleeding-agent.core",
    split: "dev",
    baseline: input.baseline ?? baselineMetadata,
    candidate: input.candidate ?? candidateMetadata,
    baselineResults: [input.baselineRun],
    candidateResults: [input.candidateRun],
    createdAt: now,
  });

describe("eval harness scorer", () => {
  test("passes a candidate with deterministic improvement", () => {
    const baselineRun = makeRun({
      runRole: "baseline",
      status: "failed",
      assertions: [
        {
          assertionId: "assert.output",
          assertionKind: "file_contains",
          passed: false,
          severity: "failure",
        },
      ],
    });
    const candidateRun = makeRun({
      runRole: "candidate",
      assertions: [
        {
          assertionId: "assert.output",
          assertionKind: "file_contains",
          passed: true,
          severity: "failure",
        },
      ],
    });

    const scorecard = createScorecard({ baselineRun, candidateRun });

    expect(scorecard.passed).toBe(true);
    expect(scorecard.criticalRegressionVeto.vetoed).toBe(false);
    expect(scorecard.objectiveMetrics.find((metric) => metric.metricId === "aggregate-score-delta")?.delta)
      .toBeGreaterThan(0);
  });

  test("vetoes promotion for critical regressions", () => {
    const baselineRun = makeRun({
      runRole: "baseline",
      assertions: [
        {
          assertionId: "assert.protected-path",
          assertionKind: "no_forbidden_path_changed",
          passed: true,
          severity: "critical",
        },
      ],
    });
    const candidateRun = makeRun({
      runRole: "candidate",
      status: "failed",
      assertions: [
        {
          assertionId: "assert.protected-path",
          assertionKind: "no_forbidden_path_changed",
          passed: false,
          severity: "critical",
          message: "Forbidden paths changed: package.json",
        },
      ],
    });

    const scorecard = createScorecard({ baselineRun, candidateRun });

    expect(scorecard.passed).toBe(false);
    expect(scorecard.criticalRegressionVeto.vetoed).toBe(true);
    expect(scorecard.criticalRegressionVeto.regressions.map((regression) => regression.assertionId))
      .toContain("assert.protected-path");
  });

  test("rejects context mismatches through scorecard schema validation", () => {
    const baselineRun = makeRun({ runRole: "baseline" });
    const candidateRun = makeRun({ runRole: "candidate" });
    const mismatchedCandidate: ComparisonRunMetadata = {
      ...candidateMetadata,
      context: {
        ...context,
        modelProfileId: "model.other.local",
      },
    };

    expect(() =>
      createScorecard({
        baselineRun,
        candidateRun,
        candidate: mismatchedCandidate,
      }),
    ).toThrow();
  });

  test("handles objective metric direction when lower values are better", () => {
    const latencyBaseline: ObjectiveMetric = {
      metricId: "latency-ms",
      name: "Latency",
      value: 100,
      unit: "ms",
      higherIsBetter: false,
    };
    const latencyCandidate: ObjectiveMetric = {
      ...latencyBaseline,
      value: 80,
    };
    const baselineRun = makeRun({
      runRole: "baseline",
      objectiveMetrics: [latencyBaseline],
    });
    const candidateRun = makeRun({
      runRole: "candidate",
      objectiveMetrics: [latencyCandidate],
    });

    const scorecard = createScorecard({ baselineRun, candidateRun });
    const latency = scorecard.objectiveMetrics.find((metric) => metric.metricId === "latency-ms.eval.scorer");
    const aggregateDelta = scorecard.objectiveMetrics.find((metric) => metric.metricId === "aggregate-score-delta");

    expect(scorecard.passed).toBe(true);
    expect(latency?.delta).toBe(20);
    expect(latency?.higherIsBetter).toBe(false);
    expect(aggregateDelta?.delta).toBeGreaterThan(0);
  });
});
