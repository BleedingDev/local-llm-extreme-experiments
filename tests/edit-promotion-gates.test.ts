import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/config";
import { runEditStrategyAblation } from "../src/eval-harness/edit-strategy-ablation";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type { ComparisonRunMetadata, EvalComparableContext, EvalRunResult, EvalScorecard, EvalSplit } from "../src/eval-harness/types";
import type { RealAcpStabilityScorecard } from "../src/replay";
import {
  evaluateEditPromotionGates,
  promoteEditStrategyCandidate,
} from "../src/optimizer/edit-promotion-gates";
import type { OptimizerArtifactLineageDecision } from "../src/optimizer/artifact-lineage";
import type { CandidatePatch } from "../src/optimizer/types";
import type { CandidateValidationResult } from "../src/optimizer/validator";

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

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.edit-promotion.policy",
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: "policy.qwen36.bleeding-agent",
    allowedJsonPointers: ["/editStrategyVersion"],
  },
  operations: [
    {
      op: "replace",
      path: "/editStrategyVersion",
      value: "edit-strategy.gepa.pass",
    },
  ],
  rationale: "Promote measured edit policy candidate.",
  createdAt: now,
  sourceTraceIds: ["trace-edit"],
};

const validation: CandidateValidationResult = {
  candidatePatchId: candidate.candidatePatchId,
  valid: true,
  issues: [],
};

const passingLineageDecision = (): OptimizerArtifactLineageDecision => ({
  schemaVersion: "optimizer-artifact-lineage.v1",
  lineageManifestId: `lineage.${candidate.candidatePatchId}`,
  candidatePatchId: candidate.candidatePatchId,
  promotionAllowed: true,
  decision: "would_promote",
  gates: [],
  blockingGateIds: [],
  report: "lineage gates passed",
});

const metadata = (split: EvalSplit, role: "baseline" | "candidate"): ComparisonRunMetadata => ({
  comparisonRunId: `compare.edit-promotion.${split}.${role}`,
  runRole: role,
  artifactId: role === "baseline" ? "policy.edit-promotion.baseline" : `candidate.edit-promotion.${split}`,
  artifactVersion: "policy.v1",
  context,
});

const run = (
  split: EvalSplit,
  role: "baseline" | "candidate",
  passed = true,
  latencyMs?: number,
  tokenCount?: number,
): EvalRunResult => ({
  runResultId: `run.edit-promotion.${split}.${role}`,
  comparisonRunId: metadata(split, role).comparisonRunId,
  runRole: role,
  evalCaseId: `eval.edit-promotion.${split}`,
  split,
  context,
  status: passed ? "passed" : "failed",
  score: passed ? 1 : 0,
  assertionResults: [
    {
      assertionId: "assert.edit-promotion.output",
      assertionKind: "file_contains",
      passed,
      severity: "critical",
      message: passed ? "ok" : "missing output",
    },
  ],
  objectiveMetrics: [
    ...(latencyMs === undefined
      ? []
      : [
        {
          metricId: "latency-ms",
          name: "Latency",
          value: latencyMs,
          unit: "ms",
          higherIsBetter: false,
        },
      ]),
    ...(tokenCount === undefined
      ? []
      : [
        {
          metricId: "token-count",
          name: "Token count",
          value: tokenCount,
          unit: "tokens",
          higherIsBetter: false,
        },
      ]),
  ],
  changedFiles: [],
  startedAt: now,
  completedAt: now,
});

const scorecard = (split: EvalSplit, passed = true, latencyMs?: number, tokenCount?: number): EvalScorecard =>
  createEvalScorecard({
    scorecardId: [
      "scorecard.edit-promotion",
      split,
      passed ? "pass" : "fail",
      latencyMs == null ? undefined : "latency",
      tokenCount == null ? undefined : "tokens",
    ].filter(Boolean).join("."),
    evalSuiteId: "suite.bleeding-agent.edit-strategy",
    split,
    baseline: metadata(split, "baseline"),
    candidate: metadata(split, "candidate"),
    baselineResults: [run(
      split,
      "baseline",
      true,
      latencyMs == null ? undefined : 10,
      tokenCount == null ? undefined : 10,
    )],
    candidateResults: [run(split, "candidate", passed, latencyMs, tokenCount)],
    createdAt: now,
  });

const stabilitySummary = (count: number, rate: number) => ({ count, rate });

const stabilityScorecard = (
  scorecardId: string,
  appliedButBrokenRate = 0,
): RealAcpStabilityScorecard => ({
  schemaVersion: "real-acp-stability-scorecard.v1",
  scorecardId,
  createdAt: now,
  runIds: [`run.${scorecardId}`],
  taskCount: 1,
  aggregate: {
    passed: stabilitySummary(appliedButBrokenRate > 0 ? 0 : 1, appliedButBrokenRate > 0 ? 0 : 1),
    failed: stabilitySummary(appliedButBrokenRate > 0 ? 1 : 0, appliedButBrokenRate > 0 ? 1 : 0),
    cancelled: stabilitySummary(0, 0),
    errored: stabilitySummary(0, 0),
    appliedButBroken: stabilitySummary(appliedButBrokenRate > 0 ? 1 : 0, appliedButBrokenRate),
    wobbled: stabilitySummary(0, 0),
    protectedPathTouched: stabilitySummary(0, 0),
    repairAttempted: stabilitySummary(0, 0),
    repairFailed: stabilitySummary(0, 0),
    rollbackAttempted: stabilitySummary(0, 0),
    rollbackFailed: stabilitySummary(0, 0),
    fallbackUsed: stabilitySummary(0, 0),
  },
  taskRecords: [],
  groupSummaries: [],
});

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-edit-promotion-"));
  try {
    writePromotionReadyGateSuite(cwd);
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const writePromotionReadyGateSuite = (cwd: string): void => {
  mkdirSync(join(cwd, ".bag", "evidence", "optimizer"), { recursive: true });
  writeFileSync(
    join(cwd, ".bag", "evidence", "optimizer", "index.json"),
    `${JSON.stringify({
      schemaVersion: "local-evidence-optimizer-gate-suite.v1",
      optimizerGateSuiteId: "optimizer-gate-suite.edit-promotion-test",
      graphId: "self-evolving-runtime-gates-v1",
      generatedAt: now,
      sourceEvidenceIndex: ".bag/evidence/index.jsonl",
      sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
      contracts: [
        {
          contractId: "optimizer-runtime-readiness.edit-promotion-test",
          jsonPath: ".bag/evidence/optimizer/runtime-readiness.json",
          markdownPath: "docs/local-evidence-optimizer-runtime-readiness.md",
          primaryUse: "edit promotion test runtime readiness",
        },
      ],
      currentDecision: {
        candidateGeneration: "allowed_as_scoped_dry_run",
        autoPromotion: "allowed",
        promotionReady: true,
        blockingReasons: [],
      },
      mustFailClosedOn: [
        "missing optimizer gate suite",
        "invalid optimizer gate suite",
        "blocking optimizer gate suite decision",
      ],
      policySeparation: {
        dimensions: ["modelProfileId", "codebaseProfileId", "modelCodebasePolicyId"],
        principle: "Promotion applies only to the exact evaluated model/codebase policy tuple.",
      },
    }, null, 2)}\n`,
    "utf8",
  );
};

describe("edit strategy promotion gates", () => {
  test("promotes only after visible train/dev and hidden holdout gates pass", () => {
    withTempCwd((cwd) => {
      const train = scorecard("train");
      const dev = scorecard("dev");
      const holdout = scorecard("holdout");
      const result = promoteEditStrategyCandidate({
        config: defaultConfig(),
        cwd,
        candidate,
        validation,
        candidateEval: dev,
        visibleEvalScorecards: [train, dev],
        holdoutEvalScorecards: [holdout],
        lineageDecision: passingLineageDecision(),
        decidedAt: now,
      });

      expect(result.promoted).toBe(true);
      expect(result.gateDecision.passed).toBe(true);
      expect(result.promotion?.checkpointPath).toBeString();
      expect(result.promotion?.promoted).toBe(true);
    });
  });

  test("blocks promotion when hidden holdout is missing or leaked into optimization input", () => {
    const train = scorecard("train");
    const dev = scorecard("dev");
    const missingHoldout = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [train, dev],
      holdoutEvalScorecards: [],
    });

    expect(missingHoldout.passed).toBe(false);
    expect(missingHoldout.gateResults.find((gate) => gate.gateId === "hidden-holdout-eval")).toMatchObject({
      passed: false,
    });

    const holdoutReport = runEditStrategyAblation({
      splits: ["holdout"],
      includeHoldout: true,
      createdAt: now,
    });
    const leaked = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [train, dev],
      holdoutEvalScorecards: [scorecard("holdout")],
      editAblationReports: [{ ...holdoutReport, optimizationAllowed: true }],
    });

    expect(leaked.passed).toBe(false);
    expect(leaked.gateResults.find((gate) => gate.gateId === "hidden-holdout-not-training-input")).toMatchObject({
      passed: false,
    });
  });

  test("vetoes protected path and applied-but-broken edit ablation signals", () => {
    const train = scorecard("train");
    const dev = scorecard("dev");
    const holdout = scorecard("holdout");
    const visibleReport = runEditStrategyAblation({ createdAt: now });
    const appliedBroken = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [train, dev],
      holdoutEvalScorecards: [holdout],
      editAblationReports: [visibleReport],
      candidateStrategyFamilies: ["apply_patch"],
    });

    expect(appliedBroken.passed).toBe(false);
    expect(appliedBroken.gateResults.find((gate) => gate.gateId === "post-apply-consistency-veto")).toMatchObject({
      passed: false,
    });

    const holdoutReport = runEditStrategyAblation({
      splits: ["holdout"],
      includeHoldout: true,
      createdAt: now,
    });
    const protectedPath = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [train, dev],
      holdoutEvalScorecards: [holdout],
      editAblationReports: [holdoutReport],
      candidateStrategyFamilies: ["whole_file"],
    });

    expect(protectedPath.passed).toBe(false);
    expect(protectedPath.gateResults.find((gate) => gate.gateId === "critical-protected-path-veto")).toMatchObject({
      passed: false,
    });
  });

  test("scopes ablation vetoes to the candidate strategy family instead of choosing a global winner", () => {
    const train = scorecard("train");
    const dev = scorecard("dev");
    const holdout = scorecard("holdout");
    const visibleReport = runEditStrategyAblation({ createdAt: now });
    const hashRangeCandidate = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [train, dev],
      holdoutEvalScorecards: [holdout],
      editAblationReports: [visibleReport],
      candidateStrategyFamilies: ["hash_range"],
    });

    expect(hashRangeCandidate.passed).toBe(true);
    expect(hashRangeCandidate.gateResults.find((gate) => gate.gateId === "post-apply-consistency-veto"))
      .toMatchObject({
        passed: true,
      });
    expect(hashRangeCandidate.gateResults.find((gate) => gate.gateId === "critical-protected-path-veto"))
      .toMatchObject({
        passed: true,
      });
  });

  test("enforces latency and score thresholds before promotion", () => {
    const train = scorecard("train", true, 120);
    const dev = scorecard("dev", true, 140);
    const holdout = scorecard("holdout", true, 130);
    const decision = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [train, dev],
      holdoutEvalScorecards: [holdout],
      thresholds: {
        maxLatencyMs: 100,
      },
    });

    expect(decision.passed).toBe(false);
    expect(decision.gateResults.find((gate) => gate.gateId === "latency-cost-constraints")).toMatchObject({
      passed: false,
    });

    const scoreFailure = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [scorecard("train", true), scorecard("dev", false)],
      holdoutEvalScorecards: [scorecard("holdout", true)],
    });
    expect(scoreFailure.gateResults.find((gate) => gate.gateId === "visible-train-dev-evals")).toMatchObject({
      passed: false,
    });

    const tokenCostFailure = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [scorecard("train", true, undefined, 250), scorecard("dev", true, undefined, 275)],
      holdoutEvalScorecards: [scorecard("holdout", true, undefined, 260)],
      thresholds: {
        maxTokenCount: 200,
      },
    });
    expect(tokenCostFailure.gateResults.find((gate) => gate.gateId === "latency-cost-constraints")).toMatchObject({
      passed: false,
    });
  });

  test("blocks promotion when real ACP stability scorecard regresses", () => {
    const decision = evaluateEditPromotionGates({
      candidate,
      visibleEvalScorecards: [scorecard("train"), scorecard("dev")],
      holdoutEvalScorecards: [scorecard("holdout")],
      realAcpStability: {
        baseline: stabilityScorecard("real-acp-stability.baseline", 0),
        candidate: stabilityScorecard("real-acp-stability.candidate", 1),
      },
    });

    expect(decision.passed).toBe(false);
    expect(decision.gateResults.find((gate) => gate.gateId === "stability-scorecard-veto")).toMatchObject({
      passed: false,
      blocking: true,
    });
  });
});
