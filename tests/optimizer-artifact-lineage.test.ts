import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/config";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type { ComparisonRunMetadata, EvalComparableContext, EvalRunResult, EvalScorecard, EvalSplit } from "../src/eval-harness/types";
import {
  assessOptimizerArtifactLineage,
  buildOptimizerArtifactLineageManifest,
} from "../src/optimizer/artifact-lineage";
import { buildCandidateEvidenceBundle } from "../src/optimizer/evidence";
import {
  assessFrozenCandidateVisibleEvaluation,
  buildFrozenCandidateRecord,
  buildHoldoutAggregateProof,
} from "../src/optimizer/frozen-candidate";
import { promoteCandidatePatch } from "../src/optimizer/promotion";
import type { CandidatePatch, PromotionDecision } from "../src/optimizer/types";
import type { CandidateValidationResult } from "../src/optimizer/validator";

const now = "2026-05-04T00:00:00.000Z";

const context: EvalComparableContext = {
  policyId: "policy.artifact-lineage.candidate",
  modelProfileId: "model.artifact-lineage",
  codebaseProfileId: "codebase.artifact-lineage",
  modelServerId: "server.artifact-lineage",
  modelServerProfileId: "server-profile.artifact-lineage",
  canonicalToolVersion: "canonical.v1",
  renderedToolVersion: "rendered.v1",
  resultStyleVersion: "result.v1",
  verificationPolicyVersion: "verification.v1",
};

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.artifact-lineage",
  policyId: "policy.artifact-lineage.candidate",
  baselinePolicyId: "policy.artifact-lineage.baseline",
  candidatePolicyId: "policy.artifact-lineage.candidate",
  modelProfileId: "model.artifact-lineage",
  codebaseProfileId: "codebase.artifact-lineage",
  clientProfileId: "client.artifact-lineage",
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: "policy.artifact-lineage.candidate",
    allowedJsonPointers: ["/resultStyleVersion"],
  },
  operations: [
    {
      op: "replace",
      path: "/resultStyleVersion",
      value: "result.v2",
    },
  ],
  rationale: "Improve result style with measured evidence.",
  createdAt: now,
  sourceTraceIds: ["trace.artifact-lineage"],
};

const validation: CandidateValidationResult = {
  candidatePatchId: candidate.candidatePatchId,
  valid: true,
  issues: [],
};

const metadata = (split: EvalSplit, role: "baseline" | "candidate"): ComparisonRunMetadata => ({
  comparisonRunId: `compare.artifact-lineage.${split}.${role}`,
  runRole: role,
  artifactId: role === "baseline" ? "policy.artifact-lineage.baseline" : "policy.artifact-lineage.candidate",
  artifactVersion: "policy.v1",
  context,
});

const run = (split: EvalSplit, role: "baseline" | "candidate"): EvalRunResult => ({
  runResultId: `run.artifact-lineage.${split}.${role}`,
  comparisonRunId: metadata(split, role).comparisonRunId,
  runRole: role,
  evalCaseId: `eval.artifact-lineage.${split}`,
  split,
  context: metadata(split, role).context,
  status: "passed",
  score: 1,
  assertionResults: [
    {
      assertionId: "assert.artifact-lineage",
      assertionKind: "file_contains",
      passed: true,
      severity: "critical",
    },
  ],
  objectiveMetrics: [],
  changedFiles: [],
  startedAt: now,
  completedAt: now,
});

const scorecard = (split: EvalSplit): EvalScorecard =>
  createEvalScorecard({
    scorecardId: `scorecard.artifact-lineage.${split}`,
    evalSuiteId: "suite.artifact-lineage",
    split,
    baseline: metadata(split, "baseline"),
    candidate: metadata(split, "candidate"),
    baselineResults: [run(split, "baseline")],
    candidateResults: [run(split, "candidate")],
    createdAt: now,
  });

const evidence = () => buildCandidateEvidenceBundle({
  evidenceBundleId: "evidence.artifact-lineage",
  createdAt: now,
  selectedSpanExcerpts: [
    {
      traceId: "trace.artifact-lineage",
      spanId: "span.artifact-lineage",
      text: "Measured artifact-lineage evidence.",
      lineage: {
        modelProfileIds: ["model.artifact-lineage"],
        codebaseProfileIds: ["codebase.artifact-lineage"],
        policyIds: ["policy.artifact-lineage.candidate"],
      },
    },
  ],
});

const promotionDecision: PromotionDecision = {
  promotionDecisionId: "promotion.artifact-lineage",
  decision: "promote",
  policyId: "policy.artifact-lineage.candidate",
  candidatePatchId: candidate.candidatePatchId,
  evalResultId: "scorecard.artifact-lineage.dev",
  modelProfileId: "model.artifact-lineage",
  codebaseProfileId: "codebase.artifact-lineage",
  clientProfileId: "client.artifact-lineage",
  baselinePolicyId: "policy.artifact-lineage.baseline",
  candidatePolicyId: "policy.artifact-lineage.candidate",
  canonicalToolVersion: "canonical.v1",
  renderedToolVersion: "rendered.v1",
  resultStyleVersion: "result.v1",
  verificationPolicyVersion: "verification.v1",
  evidenceBundleIds: ["evidence.artifact-lineage"],
  scorecardIds: ["scorecard.artifact-lineage.train", "scorecard.artifact-lineage.dev", "scorecard.artifact-lineage.holdout"],
  rollbackCheckpointPath: ".bag/optimizer/checkpoints/checkpoint.json",
  reason: "promotion test",
  decidedAt: now,
  decidedBy: "deterministic_gate",
  appliesToNewSessionsOnly: true,
};

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "optimizer-artifact-lineage-"));
  try {
    writePromotionReadyGateSuite(cwd);
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const writePromotionReadyGateSuite = (cwd: string): void => {
  const dir = join(cwd, ".bag", "evidence", "optimizer");
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "index.json"),
    `${JSON.stringify({
      schemaVersion: "local-evidence-optimizer-gate-suite.v1",
      optimizerGateSuiteId: "optimizer-gate-suite.artifact-lineage-test",
      graphId: "self-evolving-runtime-gates-v1",
      generatedAt: now,
      sourceEvidenceIndex: ".bag/evidence/index.jsonl",
      sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
      contracts: [
        {
          contractId: "optimizer-runtime-readiness.artifact-lineage-test",
          jsonPath: ".bag/evidence/optimizer/runtime-readiness.json",
          markdownPath: "docs/local-evidence-optimizer-runtime-readiness.md",
          primaryUse: "artifact lineage promotion test runtime readiness",
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

describe("optimizer artifact lineage", () => {
  test("builds auditable lineage manifests with separated uplift classes", () => {
    const manifest = buildOptimizerArtifactLineageManifest({
      candidate,
      validation,
      visibleScorecards: [scorecard("train"), scorecard("dev")],
      holdoutScorecards: [scorecard("holdout")],
      evidenceBundles: [evidence()],
      promotionDecision,
      rollbackCheckpointPath: ".bag/optimizer/checkpoints/checkpoint.json",
    });
    const decision = assessOptimizerArtifactLineage(manifest);

    expect(manifest.upliftClasses.map((uplift) => uplift.upliftClass)).toEqual([
      "validation",
      "train_dev",
      "hidden_holdout",
      "full_eval",
      "live_rollout",
    ]);
    expect(decision.promotionAllowed).toBe(true);
    expect(decision.report).toContain("Promotion allowed: yes");
  });

  test("blocks weak artifacts missing evidence, holdout, rollback, and promotion lineage", () => {
    const { baselinePolicyId: _baselinePolicyId, ...candidateWithoutBaseline } = candidate;
    const manifest = buildOptimizerArtifactLineageManifest({
      candidate: candidateWithoutBaseline,
      validation,
      visibleScorecards: [scorecard("train"), scorecard("dev")],
    });
    const decision = assessOptimizerArtifactLineage(manifest);

    expect(decision.promotionAllowed).toBe(false);
    expect(decision.blockingGateIds).toContain("evidence-bundles-present");
    expect(decision.blockingGateIds).toContain("hidden-holdout-uplift-present");
    expect(decision.blockingGateIds).toContain("rollback-checkpoint-present");
    expect(decision.blockingGateIds).toContain("promotion-decision-present");
  });

  test("accepts aggregate-only hidden holdout proof without raw holdout scorecards", () => {
    const frozenCandidate = buildFrozenCandidateRecord({
      candidate,
      graphId: "blocker-closure-v1",
      selectionHash: "a49f7e68fb",
      epochId: "epoch.artifact-lineage",
      frozenAt: now,
      visibleInputBindings: [
        {
          bindingId: "binding.artifact-lineage.train",
          sourceKind: "eval_scorecard",
          sourceArtifactId: "scorecard.artifact-lineage.train",
          split: "train",
          contentHash: "sha256:train",
          optimizerInputAllowed: true,
        },
        {
          bindingId: "binding.artifact-lineage.dev",
          sourceKind: "eval_scorecard",
          sourceArtifactId: "scorecard.artifact-lineage.dev",
          split: "dev",
          contentHash: "sha256:dev",
          optimizerInputAllowed: true,
        },
      ],
    });
    const visibleEvaluation = assessFrozenCandidateVisibleEvaluation({
      frozenCandidate,
      visibleScorecards: [scorecard("train"), scorecard("dev")],
    });
    const holdoutProof = buildHoldoutAggregateProof({
      frozenCandidate,
      visibleEvaluation,
      holdoutScorecards: [scorecard("holdout")],
      createdAt: now,
    });
    const manifest = buildOptimizerArtifactLineageManifest({
      candidate,
      validation,
      visibleScorecards: [scorecard("train"), scorecard("dev")],
      holdoutAggregateProofs: [holdoutProof],
      evidenceBundles: [evidence()],
      promotionDecision,
      rollbackCheckpointPath: ".bag/optimizer/checkpoints/checkpoint.json",
    });
    const decision = assessOptimizerArtifactLineage(manifest);

    expect(manifest.holdoutAggregateProofIds).toEqual([holdoutProof.proofId]);
    expect(manifest.upliftClasses.find((uplift) => uplift.upliftClass === "hidden_holdout"))
      .toMatchObject({
        status: "passed",
        aggregateProofIds: [holdoutProof.proofId],
      });
    expect(decision.promotionAllowed).toBe(true);
    expect(decision.report).toContain("holdout aggregate proofs");
  });

  test("promotion rejects candidates when supplied lineage decision blocks them", () => {
    const weakManifest = buildOptimizerArtifactLineageManifest({
      candidate,
      validation,
      visibleScorecards: [scorecard("train"), scorecard("dev")],
    });
    const lineageDecision = assessOptimizerArtifactLineage(weakManifest);

    withTempCwd((cwd) => {
      const result = promoteCandidatePatch({
        config: defaultConfig(),
        cwd,
        candidate,
        validation,
        candidateEval: scorecard("dev"),
        lineageDecision,
        decidedAt: now,
      });

      expect(result.promoted).toBe(false);
      expect(result.decision.reason).toContain("artifact lineage gates failed");
      expect(result.decision.rollbackCheckpointPath).toBeString();
      expect(result.decision.scorecardIds).toContain("scorecard.artifact-lineage.dev");
    });
  });
});
