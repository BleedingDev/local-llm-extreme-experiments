import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type {
  ComparisonRunMetadata,
  EvalCase,
  EvalComparableContext,
  EvalRunResult,
  EvalSplit,
} from "../src/eval-harness/types";
import { defaultConfig } from "../src/config";
import { routingReplayScenarios } from "../src/replay";
import {
  buildOperatorSafeGepaFeedbackBundle,
  evaluateGepaPromotionGates,
  materializeGepaCandidatePreview,
  proposeDeterministicGepaCandidates,
  runGepaCandidateEvaluation,
} from "../src/optimizer/gepa-loop";
import { buildCandidateEvidenceBundle } from "../src/optimizer/evidence";
import type { CandidatePatch, OptimizerRegistryRecord } from "../src/optimizer/types";

const now = "2026-05-01T00:00:00.000Z";

const context: EvalComparableContext = {
  policyId: "policy.gepa.loop",
  modelProfileId: "model.gepa.loop",
  codebaseProfileId: "codebase.gepa.loop",
  modelServerId: "server.gepa.loop",
  modelServerProfileId: "server-profile.gepa.loop",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
};

const baseline: ComparisonRunMetadata = {
  comparisonRunId: "compare.gepa.loop.baseline",
  runRole: "baseline",
  artifactId: "policy.gepa.loop.baseline",
  artifactVersion: "policy.v1",
  context,
};

const candidateMetadata: ComparisonRunMetadata = {
  comparisonRunId: "compare.gepa.loop.candidate",
  runRole: "candidate",
  artifactId: "candidate.gepa.loop",
  artifactVersion: "candidate.v1",
  context,
};

const makeRun = (input: { evalCaseId: string; split: "dev" | "holdout"; passed: boolean }): EvalRunResult => ({
  runResultId: `run.${input.evalCaseId}.${input.split}`,
  comparisonRunId: candidateMetadata.comparisonRunId,
  runRole: "candidate",
  evalCaseId: input.evalCaseId,
  split: input.split,
  context,
  status: input.passed ? "passed" : "failed",
  score: input.passed ? 1 : 0,
  assertionResults: [
    {
      assertionId: `assert.${input.evalCaseId}`,
      assertionKind: "file_contains",
      passed: input.passed,
      severity: "critical",
      message: input.passed ? "ok" : "missing expected output",
    },
  ],
  objectiveMetrics: [],
  changedFiles: [],
  startedAt: now,
  completedAt: now,
});

const policyRecord: OptimizerRegistryRecord = {
  registryRecordId: "registry.policy.gepa.loop",
  recordKind: "model_codebase_policy",
  schemaVersion: "optimizer-schema.v1",
  recordVersion: "record.v1",
  status: "active",
  createdAt: now,
  updatedAt: now,
  contentHash: "sha256:policy",
  payload: {
    policyId: "policy.gepa.loop",
    modelProfileId: "model.gepa.loop",
    codebaseProfileId: "codebase.gepa.loop",
    canonicalToolVersion: "canonical-tools.v1",
    renderedToolVersion: "rendered-tools.v1",
    resultStyleVersion: "result-style.v1",
    verificationPolicyVersion: "verification.v1",
    candidateScopes: [],
    verificationGates: [],
    maxConcurrentEvaluations: 1,
    riskTolerance: "low",
    status: "draft",
  },
};

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.gepa.loop.policy",
  policyId: "policy.gepa.loop",
  modelProfileId: "model.gepa.loop",
  codebaseProfileId: "codebase.gepa.loop",
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: "policy.gepa.loop",
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
  rationale: "Add a reliability gate from bounded GEPA feedback.",
  createdAt: now,
  sourceTraceIds: ["trace-gepa"],
};

const evalCase = (split: "train" | "dev" | "holdout"): EvalCase => ({
  evalCaseId: `eval.gepa.loop.${split}`,
  schemaVersion: "eval-case.v1",
  split,
  title: `GEPA loop ${split}`,
  task: "Keep the fixture output intact.",
  fixtureWorkspace: {
    fixtureWorkspaceId: `fixture.gepa.loop.${split}`,
    name: `GEPA loop ${split}`,
    rootFingerprint: `sha256:${split}`,
    files: [
      {
        path: "result.txt",
        content: "ok\n",
        executable: false,
      },
    ],
    protectedPaths: [],
    setupCommands: [],
    verificationCommands: [],
  },
  assertions: [
    {
      assertionId: `assert.gepa.loop.${split}`,
      assertionKind: "file_contains",
      path: "result.txt",
      text: "ok",
      severity: "critical",
      description: "Fixture output remains correct.",
    },
  ],
  tags: ["gepa-loop"],
  timeoutMs: 1_000,
});

const metricRun = (
  split: EvalSplit,
  runRole: "baseline" | "candidate",
  latencyMs: number,
  tokenCount: number,
): EvalRunResult => {
  const metadata = runRole === "baseline" ? baseline : candidateMetadata;
  return {
    runResultId: `run.gepa.loop.${split}.${runRole}.metrics`,
    comparisonRunId: metadata.comparisonRunId,
    runRole,
    evalCaseId: `eval.gepa.loop.${split}.metrics`,
    split,
    context,
    status: "passed",
    score: 1,
    assertionResults: [
      {
        assertionId: `assert.gepa.loop.${split}.metrics`,
        assertionKind: "file_contains",
        passed: true,
        severity: "critical",
        message: "ok",
      },
    ],
    objectiveMetrics: [
      {
        metricId: "latency-ms",
        name: "Latency",
        value: latencyMs,
        unit: "ms",
        higherIsBetter: false,
      },
      {
        metricId: "token-count",
        name: "Token count",
        value: tokenCount,
        unit: "tokens",
        higherIsBetter: false,
      },
    ],
    changedFiles: [],
    startedAt: now,
    completedAt: now,
  };
};

const metricScorecard = (split: EvalSplit, latencyMs: number, tokenCount: number) =>
  createEvalScorecard({
    scorecardId: `scorecard.gepa.loop.${split}.metrics`,
    evalSuiteId: "suite.gepa.loop.metrics",
    split,
    baseline,
    candidate: candidateMetadata,
    baselineResults: [metricRun(split, "baseline", 10, 10)],
    candidateResults: [metricRun(split, "candidate", latencyMs, tokenCount)],
    createdAt: now,
  });

describe("GEPA closed-loop core helpers", () => {
  test("builds bounded proposer-safe feedback without leaking holdout eval results", () => {
    const result = buildOperatorSafeGepaFeedbackBundle({
      feedbackBundleId: "gepa.loop.feedback",
      evalRunResults: [
        makeRun({ evalCaseId: "eval.visible", split: "dev", passed: false }),
        makeRun({ evalCaseId: "eval.hidden", split: "holdout", passed: false }),
      ],
      testOutputs: [
        {
          id: "typecheck",
          text: "token=should-not-leak-123456789 TS2322: bad shape",
          modelProfileId: context.modelProfileId,
          codebaseProfileId: context.codebaseProfileId,
          policyId: context.policyId,
        },
      ],
      limits: {
        maxTextChars: 80,
      },
      createdAt: now,
    });

    expect(result.excludedHoldoutEvalCaseIds).toEqual(["eval.hidden"]);
    expect(result.feedbackBundle.records.flatMap((record) => record.evalCaseIds)).not.toContain("eval.hidden");
    expect(JSON.stringify(result.feedbackBundle)).not.toContain("should-not-leak");
    expect(result.feedbackBundle.redactionCount).toBeGreaterThan(0);
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      reason: "hidden holdout evidence was excluded from GEPA proposer input",
      evalCaseIds: ["eval.hidden"],
    }));
  });

  test("materializes candidate preview artifacts with validation, base hashes, dimensions, and rollback metadata", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-gepa-loop-preview-"));
    const evidence = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.gepa.loop",
      createdAt: now,
      selectedSpanExcerpts: [
        {
          traceId: "trace-gepa",
          spanId: "span-gepa",
          text: "tool call failed",
          lineage: {
            modelProfileIds: [context.modelProfileId],
            codebaseProfileIds: [context.codebaseProfileId],
            policyIds: [context.policyId],
          },
        },
      ],
    });

    try {
      const preview = materializeGepaCandidatePreview({
        config: defaultConfig(),
        cwd,
        candidate,
        evidence,
        records: [policyRecord],
        createdAt: now,
        rollbackMetadata: {
          operator: "test",
        },
      });

      expect(preview.validation.valid).toBe(true);
      expect(preview.baseHashes.expected["policy.gepa.loop"]).toBe("sha256:policy");
      expect(preview.affectedPolicyDimensions).toEqual(["verificationGates"]);
      expect(preview.rollback).toMatchObject({
        rollbackSupported: true,
        metadata: {
          operator: "test",
        },
      });
      expect(preview.artifactManifest?.files).toMatchObject({
        patch: "patch.json",
        validation: "validation.json",
        report: "report.md",
      });
      expect(readFileSync(join(preview.artifactManifest!.artifactDir, "report.md"), "utf8")).toContain(
        "GEPA Candidate Preview candidate.gepa.loop.policy",
      );
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  test("runs baseline-vs-candidate train/dev gates and hidden holdout final checks", async () => {
    const baseDir = mkdtempSync(join(tmpdir(), "bag-gepa-loop-eval-"));
    const replayScenario = routingReplayScenarios.find((scenario) =>
      scenario.scenarioKind === "greeting_no_side_effect"
    );
    expect(replayScenario).toBeDefined();
    if (replayScenario == null) {
      throw new Error("greeting replay scenario missing");
    }

    try {
      const evaluation = await runGepaCandidateEvaluation({
        candidate,
        baseline,
        candidateMetadata,
        replayCases: [replayScenario],
        curatedEvalCases: [evalCase("train"), evalCase("dev"), evalCase("holdout")],
        includeHoldoutFinal: true,
        baseDir,
        createdAt: now,
      });

      expect(evaluation.visibleScorecards.map((scorecard) => scorecard.split).sort()).toEqual(["dev", "train", "train"]);
      expect(evaluation.visibleScorecards.some((scorecard) => scorecard.evalSuiteId === "suite.gepa.replay")).toBe(true);
      expect(evaluation.holdoutScorecards.map((scorecard) => scorecard.split)).toEqual(["holdout"]);
      expect(evaluation.gates).toContainEqual(expect.objectContaining({
        gateId: "train-dev-visible",
        passed: true,
      }));
      expect(evaluation.gates).toContainEqual(expect.objectContaining({
        gateId: "hidden-holdout-final",
        passed: true,
      }));
      expect(evaluation.passed).toBe(true);
      expect(evaluation.promotionScorecard?.split).toBe("holdout");
    } finally {
      rmSync(baseDir, { recursive: true, force: true });
    }
  });

  test("vetoes GEPA promotion gates on latency and token-cost regressions", () => {
    const evaluation = evaluateGepaPromotionGates({
      visibleScorecards: [
        metricScorecard("train", 80, 100),
        metricScorecard("dev", 180, 100),
      ],
      holdoutScorecards: [metricScorecard("holdout", 80, 300)],
      includeHoldoutFinal: true,
      thresholds: {
        maxLatencyMs: 100,
        maxTokenCount: 200,
      },
    });

    expect(evaluation.passed).toBe(false);
    expect(evaluation.gates).toContainEqual(expect.objectContaining({
      gateId: "latency-cost-veto",
      passed: false,
      blocking: true,
    }));
    expect(evaluation.gates.find((gate) => gate.gateId === "latency-cost-veto")?.message).toContain("latency-ms");
    expect(evaluation.gates.find((gate) => gate.gateId === "latency-cost-veto")?.message).toContain("token-count");
  });

  test("keeps deterministic candidate proposal available for offline operation", () => {
    const evidence = buildCandidateEvidenceBundle({
      evidenceBundleId: "evidence.gepa.loop.propose",
      createdAt: now,
      selectedSpanExcerpts: [
        {
          traceId: "trace-gepa",
          spanId: "span-gepa",
          text: "edit verification failed",
          lineage: {
            modelProfileIds: [context.modelProfileId],
            codebaseProfileIds: [context.codebaseProfileId],
            policyIds: [context.policyId],
            editStrategyFamilies: ["apply_patch"],
          },
        },
      ],
    });

    const proposed = proposeDeterministicGepaCandidates({
      evidence,
      createdAt: now,
      maxCandidates: 1,
    });

    expect(proposed.candidates).toHaveLength(1);
    expect(proposed.candidates[0]?.policyId).toBe(context.policyId);
  });
});
