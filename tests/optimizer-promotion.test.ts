import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/config";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type { ComparisonRunMetadata, EvalComparableContext, EvalRunResult } from "../src/eval-harness/types";
import { loadActiveOptimizerPointer, loadOptimizerRegistry, saveActiveOptimizerPointer } from "../src/optimizer/registry";
import { promoteGepaCandidate } from "../src/optimizer/gepa-loop";
import {
  monitorPostPromotionRollback,
  promoteCandidatePatch,
  rollbackOptimizerPromotion,
} from "../src/optimizer/promotion";
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

const baselineMetadata: ComparisonRunMetadata = {
  comparisonRunId: "compare.promotion.baseline",
  runRole: "baseline",
  artifactId: "policy.qwen36.bleeding-agent.baseline",
  artifactVersion: "policy.v1",
  context,
};

const candidateMetadata: ComparisonRunMetadata = {
  comparisonRunId: "compare.promotion.candidate",
  runRole: "candidate",
  artifactId: "candidate.promotion",
  artifactVersion: "candidate.v1",
  context,
};

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.promotion.tool",
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  codebaseRootFingerprint: "sha256:profile",
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

const validation: CandidateValidationResult = {
  candidatePatchId: "candidate.promotion.tool",
  valid: true,
  issues: [],
};

const passingLineageDecision = (
  candidatePatchId = candidate.candidatePatchId,
): OptimizerArtifactLineageDecision => ({
  schemaVersion: "optimizer-artifact-lineage.v1",
  lineageManifestId: `lineage.${candidatePatchId}`,
  candidatePatchId,
  promotionAllowed: true,
  decision: "would_promote",
  gates: [],
  blockingGateIds: [],
  report: "lineage gates passed",
});

const makeRun = (runRole: "baseline" | "candidate", passed: boolean): EvalRunResult => {
  const metadata = runRole === "baseline" ? baselineMetadata : candidateMetadata;
  return {
    runResultId: `run.promotion.${runRole}`,
    comparisonRunId: metadata.comparisonRunId,
    runRole,
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
        severity: "critical",
        message: passed ? "ok" : "missing output",
      },
    ],
    objectiveMetrics: [],
    changedFiles: [],
    startedAt: now,
    completedAt: now,
  };
};

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-promotion-"));
  try {
    writePromotionReadyGateSuite(cwd);
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const withTempCwdWithoutGateSuite = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-promotion-no-gate-"));
  try {
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const writePromotionReadyGateSuite = (cwd: string): void => {
  const path = join(cwd, ".bag", "evidence", "optimizer", "index.json");
  mkdirSync(join(cwd, ".bag", "evidence", "optimizer"), { recursive: true });
  writeFileSync(
    path,
    `${JSON.stringify({
      schemaVersion: "local-evidence-optimizer-gate-suite.v1",
      optimizerGateSuiteId: "optimizer-gate-suite.promotion-test",
      graphId: "self-evolving-runtime-gates-v1",
      generatedAt: now,
      sourceEvidenceIndex: ".bag/evidence/index.jsonl",
      sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
      contracts: [
        {
          contractId: "optimizer-runtime-readiness.promotion-test",
          jsonPath: ".bag/evidence/optimizer/runtime-readiness.json",
          markdownPath: "docs/local-evidence-optimizer-runtime-readiness.md",
          primaryUse: "promotion test runtime readiness",
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

describe("candidate promotion", () => {
  test("fails closed when runtime optimizer gate suite is missing", () => {
    withTempCwdWithoutGateSuite((cwd) => {
      const config = defaultConfig();
      const evalScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.missing-gate",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", true)],
        createdAt: now,
      });

      const result = promoteCandidatePatch({
        config,
        cwd,
        candidate,
        validation,
        candidateEval: evalScorecard,
        lineageDecision: passingLineageDecision(),
        decidedAt: now,
      });

      expect(result.promoted).toBe(false);
      expect(result.decision.decision).toBe("reject");
      expect(result.decision.reason).toContain("runtime optimizer promotions fail closed");
      expect(loadActiveOptimizerPointer(config, cwd).pointer).toBeUndefined();
    });
  });

  test("promotes only validation-passing and eval-passing candidates and writes registry records", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const evalScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.pass",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", true)],
        createdAt: now,
      });

      const result = promoteCandidatePatch({
        config,
        cwd,
        candidate,
        validation,
        candidateEval: evalScorecard,
        lineageDecision: passingLineageDecision(),
        decidedAt: now,
      });

      expect(result.promoted).toBe(true);
      expect(result.activePointer).toMatchObject({
        activeModelProfileId: "model.qwen36.local",
        activeCodebaseProfileId: "codebase.bleeding-agent",
        activeCodebaseRootFingerprint: "sha256:profile",
        activePolicyId: "policy.qwen36.bleeding-agent",
        promotedAt: now,
      });
      expect(result.decision.codebaseRootFingerprint).toBe("sha256:profile");

      const registry = loadOptimizerRegistry(config, cwd);
      expect(registry.persistedRecords.map((record) => record.recordKind).sort()).toEqual([
        "candidate_patch",
        "promotion_decision",
      ]);
      expect(loadActiveOptimizerPointer(config, cwd).pointer?.activePolicyId).toBe(
        "policy.qwen36.bleeding-agent",
      );
    });
  });

  test("rejects failed eval candidates without changing active pointer", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const evalScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.fail",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", false)],
        createdAt: now,
      });

      const result = promoteCandidatePatch({
        config,
        cwd,
        candidate,
        validation,
        candidateEval: evalScorecard,
        lineageDecision: passingLineageDecision(),
        decidedAt: now,
      });

      expect(result.promoted).toBe(false);
      expect(result.decision.decision).toBe("reject");
      expect(loadActiveOptimizerPointer(config, cwd).pointer).toBeUndefined();
    });
  });

  test("rejects profile-mismatched promotion evidence without changing active pointer", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const evalScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.profile-mismatch",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", true)],
        createdAt: now,
      });

      const result = promoteCandidatePatch({
        config,
        cwd,
        candidate: {
          ...candidate,
          codebaseProfileId: "codebase.other",
          codebaseRootFingerprint: "sha256:other",
        },
        validation: {
          ...validation,
          candidatePatchId: "candidate.promotion.profile-mismatch",
        },
        candidateEval: evalScorecard,
        decidedAt: now,
      });

      expect(result.promoted).toBe(false);
      expect(result.decision.decision).toBe("reject");
      expect(result.decision.reason).toContain("codebase profile gate failed");
      expect(loadActiveOptimizerPointer(config, cwd).pointer).toBeUndefined();
    });
  });

  test("GEPA promotion orchestration rejects aggregate gate failures without changing active pointer", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const evalScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.aggregate-gate",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", true)],
        createdAt: now,
      });

      const result = promoteGepaCandidate({
        config,
        cwd,
        candidate,
        validation,
        evaluation: {
          visibleScorecards: [evalScorecard],
          holdoutScorecards: [],
          allScorecards: [evalScorecard],
          promotionScorecard: evalScorecard,
          passed: false,
          gates: [
            {
              gateId: "hidden-holdout-final",
              passed: false,
              blocking: true,
              message: "Hidden holdout final check failed; no holdout scorecard was produced.",
              scorecardIds: [],
            },
          ],
        },
        decidedAt: now,
      });

      expect(result.promoted).toBe(false);
      expect(result.decision.decision).toBe("reject");
      expect(result.decision.reason).toContain("aggregate promotion gates failed");
      expect(loadActiveOptimizerPointer(config, cwd).pointer).toBeUndefined();
    });
  });

  test("rolls back to previous active pointer checkpoint", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const previous = saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: "model.previous",
          activeCodebaseProfileId: "codebase.previous",
          activePolicyId: "policy.previous",
          promotedAt: "2026-04-29T00:00:00.000Z",
        },
        cwd,
      );
      const evalScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.rollback",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", true)],
        createdAt: now,
      });

      const result = promoteCandidatePatch({
        config,
        cwd,
        candidate,
        validation,
        candidateEval: evalScorecard,
        lineageDecision: passingLineageDecision(),
        decidedAt: now,
      });
      expect(loadActiveOptimizerPointer(config, cwd).pointer?.activePolicyId).toBe(
        "policy.qwen36.bleeding-agent",
      );

      const rolledBack = rollbackOptimizerPromotion({
        config,
        cwd,
        checkpointPath: result.checkpointPath,
      });

      expect(rolledBack).toMatchObject({
        activeModelProfileId: previous.activeModelProfileId,
        activeCodebaseProfileId: previous.activeCodebaseProfileId,
        activePolicyId: previous.activePolicyId,
        promotedAt: previous.promotedAt,
      });
      expect(loadActiveOptimizerPointer(config, cwd).pointer?.activePolicyId).toBe("policy.previous");
    });
  });

  test("monitors post-promotion scorecards and rolls back deterministic regressions", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const previous = saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: "model.previous",
          activeCodebaseProfileId: "codebase.previous",
          activePolicyId: "policy.previous",
          promotedAt: "2026-04-29T00:00:00.000Z",
        },
        cwd,
      );
      const passingScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.monitor-pass",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", true)],
        createdAt: now,
      });
      const promotion = promoteCandidatePatch({
        config,
        cwd,
        candidate,
        validation,
        candidateEval: passingScorecard,
        lineageDecision: passingLineageDecision(),
        decidedAt: now,
      });
      const regressionScorecard = createEvalScorecard({
        scorecardId: "scorecard.promotion.monitor-regression",
        evalSuiteId: "suite.bleeding-agent.core",
        split: "dev",
        baseline: baselineMetadata,
        candidate: candidateMetadata,
        baselineResults: [makeRun("baseline", true)],
        candidateResults: [makeRun("candidate", false)],
        createdAt: now,
      });

      const monitor = monitorPostPromotionRollback({
        config,
        cwd,
        promotion,
        evalScorecards: [regressionScorecard],
      });

      expect(monitor.regressionDetected).toBe(true);
      expect(monitor.rollbackRequested).toBe(true);
      expect(monitor.rolledBack).toBe(true);
      expect(monitor.signals).toContainEqual(expect.objectContaining({
        source: "eval_scorecard",
        severity: "critical",
        scorecardId: "scorecard.promotion.monitor-regression",
      }));
      expect(monitor.rollbackPointer).toMatchObject({
        activeModelProfileId: previous.activeModelProfileId,
        activeCodebaseProfileId: previous.activeCodebaseProfileId,
        activePolicyId: previous.activePolicyId,
      });
      expect(loadActiveOptimizerPointer(config, cwd).pointer?.activePolicyId).toBe("policy.previous");
    });
  });
});
