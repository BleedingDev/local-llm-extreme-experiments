import { describe, expect, test } from "bun:test";
import {
  CriticalRegressionVetoSchema,
  EvalCaseSchema,
  EvalRunResultSchema,
  EvalScorecardSchema,
} from "../src/eval-harness/types";

const now = "2026-04-30T00:00:00.000Z";

const context = {
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

const evalCase = {
  evalCaseId: "eval.small-edit",
  schemaVersion: "eval-schema.v1",
  split: "dev",
  title: "Small edit preserves tests",
  task: "Update the greeting implementation and run verification.",
  fixtureWorkspace: {
    fixtureWorkspaceId: "fixture.small-edit",
    name: "Small TypeScript package",
    rootFingerprint: "sha256:fixture",
    files: [
      {
        path: "src/greeting.ts",
        content: "export const greeting = () => 'hello';\n",
      },
      {
        path: "package.json",
        content: "{\"scripts\":{\"test\":\"bun test\"}}\n",
      },
    ],
    protectedPaths: ["package.json"],
  },
  assertions: [
    {
      assertionId: "assert.greeting-updated",
      assertionKind: "file_contains",
      description: "Greeting implementation was updated.",
      path: "src/greeting.ts",
      text: "hello, world",
    },
    {
      assertionId: "assert.package-unchanged",
      assertionKind: "no_forbidden_path_changed",
      description: "Package metadata is not modified.",
      severity: "critical",
      paths: ["package.json"],
    },
  ],
  tags: ["small-edit", "verification"],
  timeoutMs: 120000,
};

const baselineRun = {
  runResultId: "run.baseline.small-edit",
  comparisonRunId: "compare.baseline",
  runRole: "baseline",
  evalCaseId: "eval.small-edit",
  split: "dev",
  context,
  status: "passed",
  score: 0.8,
  assertionResults: [
    {
      assertionId: "assert.greeting-updated",
      assertionKind: "file_contains",
      passed: true,
    },
    {
      assertionId: "assert.package-unchanged",
      assertionKind: "no_forbidden_path_changed",
      severity: "critical",
      passed: true,
    },
  ],
  objectiveMetrics: [
    {
      metricId: "tests-pass",
      name: "Verification pass rate",
      value: 1,
      unit: "ratio",
    },
  ],
  changedFiles: ["src/greeting.ts"],
  telemetryArtifactPath: ".bag/evals/baseline.jsonl",
  startedAt: now,
  completedAt: now,
};

const candidateRun = {
  ...baselineRun,
  runResultId: "run.candidate.small-edit",
  comparisonRunId: "compare.candidate",
  runRole: "candidate",
  score: 0.92,
  objectiveMetrics: [
    {
      metricId: "tests-pass",
      name: "Verification pass rate",
      value: 1,
      unit: "ratio",
      baselineValue: 1,
      candidateValue: 1,
      delta: 0,
    },
  ],
  telemetryArtifactPath: ".bag/evals/candidate.jsonl",
};

const scorecard = {
  scorecardId: "scorecard.small-edit",
  schemaVersion: "eval-schema.v1",
  evalSuiteId: "suite.bleeding-agent.core",
  split: "dev",
  baseline: {
    comparisonRunId: "compare.baseline",
    runRole: "baseline",
    artifactId: "policy.qwen36.bleeding-agent.baseline",
    artifactVersion: "policy.v1",
    context,
  },
  candidate: {
    comparisonRunId: "compare.candidate",
    runRole: "candidate",
    artifactId: "candidate.tool-rendering.1",
    artifactVersion: "candidate.v1",
    context,
  },
  runResults: [baselineRun, candidateRun],
  objectiveMetrics: [
    {
      metricId: "aggregate-score-delta",
      name: "Aggregate score delta",
      value: 0.12,
      unit: "score",
      baselineValue: 0.8,
      candidateValue: 0.92,
      delta: 0.12,
    },
  ],
  aggregateScore: 0.92,
  passed: true,
  criticalRegressionVeto: {
    vetoed: false,
    regressions: [],
  },
  createdAt: now,
};

describe("eval harness schemas", () => {
  test("parse representative eval case, run result, and scorecard", () => {
    expect(EvalCaseSchema.parse(evalCase).fixtureWorkspace.protectedPaths).toEqual(["package.json"]);
    expect(EvalRunResultSchema.parse(candidateRun).objectiveMetrics[0]?.higherIsBetter).toBe(true);
    expect(EvalScorecardSchema.parse(scorecard).criticalRegressionVeto.vetoed).toBe(false);
  });

  test("rejects cross-model comparison metadata", () => {
    const result = EvalScorecardSchema.safeParse({
      ...scorecard,
      candidate: {
        ...scorecard.candidate,
        context: {
          ...context,
          modelProfileId: "model.other.local",
        },
      },
    });

    expect(result.success).toBe(false);
  });

  test("rejects cross-server-profile comparison metadata", () => {
    const result = EvalScorecardSchema.safeParse({
      ...scorecard,
      candidate: {
        ...scorecard.candidate,
        context: {
          ...context,
          modelServerProfileId: "server-profile.qwen36.other-runtime",
        },
      },
    });

    expect(result.success).toBe(false);
  });

  test("rejects missing critical comparable context fields", () => {
    const incompleteContext: Record<string, string> = { ...context };
    delete incompleteContext.verificationPolicyVersion;
    const result = EvalRunResultSchema.safeParse({
      ...candidateRun,
      context: incompleteContext,
    });

    expect(result.success).toBe(false);
  });

  test("enforces critical regression veto semantics", () => {
    const regression = {
      regressionId: "regression.package-mutated",
      evalCaseId: "eval.small-edit",
      assertionId: "assert.package-unchanged",
      reason: "Candidate modified a protected package manifest.",
      baselineStatus: "passed",
      candidateStatus: "failed",
      blocksPromotion: true,
    };

    expect(CriticalRegressionVetoSchema.parse({
      vetoed: true,
      regressions: [regression],
    }).vetoed).toBe(true);

    expect(CriticalRegressionVetoSchema.safeParse({
      vetoed: false,
      regressions: [regression],
    }).success).toBe(false);

    expect(EvalScorecardSchema.safeParse({
      ...scorecard,
      passed: false,
      criticalRegressionVeto: {
        vetoed: false,
        regressions: [regression],
      },
    }).success).toBe(false);
  });
});
