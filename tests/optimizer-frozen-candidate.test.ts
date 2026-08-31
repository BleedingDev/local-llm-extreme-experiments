import { describe, expect, test } from "bun:test";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type {
  ComparisonRunMetadata,
  EvalComparableContext,
  EvalRunResult,
  EvalScorecard,
  EvalSplit,
} from "../src/eval-harness/types";
import {
  assertFrozenCandidateNonLeakage,
  assessFrozenCandidateVisibleEvaluation,
  buildFrozenCandidateRecord,
  buildHoldoutAggregateProof,
} from "../src/optimizer/frozen-candidate";
import type { CandidatePatch } from "../src/optimizer/types";

const now = "2026-05-05T00:00:00.000Z";

const context: EvalComparableContext = {
  policyId: "policy.frozen-candidate.candidate",
  modelProfileId: "model.frozen-candidate",
  codebaseProfileId: "codebase.frozen-candidate",
  modelServerId: "server.frozen-candidate",
  modelServerProfileId: "server-profile.frozen-candidate",
  canonicalToolVersion: "canonical.v1",
  renderedToolVersion: "rendered.v1",
  resultStyleVersion: "result.v1",
  verificationPolicyVersion: "verification.v1",
};

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.frozen-candidate",
  policyId: context.policyId,
  baselinePolicyId: "policy.frozen-candidate.baseline",
  candidatePolicyId: context.policyId,
  modelProfileId: context.modelProfileId,
  codebaseProfileId: context.codebaseProfileId,
  clientProfileId: "client.frozen-candidate",
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: context.policyId,
    allowedJsonPointers: ["/resultStyleVersion"],
  },
  operations: [
    {
      op: "replace",
      path: "/resultStyleVersion",
      value: "result.v2",
    },
  ],
  rationale: "Freeze the exact candidate before hidden holdout.",
  createdAt: now,
  sourceTraceIds: ["trace.frozen-candidate"],
};

const metadata = (split: EvalSplit, role: "baseline" | "candidate"): ComparisonRunMetadata => ({
  comparisonRunId: `compare.frozen-candidate.${split}.${role}`,
  runRole: role,
  artifactId: role === "baseline" ? "policy.frozen-candidate.baseline" : context.policyId,
  artifactVersion: "policy.v1",
  context,
});

const run = (split: EvalSplit, role: "baseline" | "candidate", passed = true): EvalRunResult => ({
  runResultId: `run.frozen-candidate.${split}.${role}.${passed ? "pass" : "fail"}`,
  comparisonRunId: metadata(split, role).comparisonRunId,
  runRole: role,
  evalCaseId: `eval.frozen-candidate.${split}`,
  split,
  context,
  candidatePatchId: candidate.candidatePatchId,
  status: passed ? "passed" : "failed",
  score: passed ? 1 : 0,
  assertionResults: [
    {
      assertionId: "assert.frozen-candidate",
      assertionKind: "file_contains",
      passed,
      severity: "critical",
      message: passed ? "ok" : "empty_edits: no mutation evidence",
    },
  ],
  objectiveMetrics: [],
  changedFiles: [],
  startedAt: now,
  completedAt: now,
});

const scorecard = (split: EvalSplit, candidatePassed = true): EvalScorecard =>
  createEvalScorecard({
    scorecardId: `scorecard.frozen-candidate.${split}.${candidatePassed ? "pass" : "fail"}`,
    evalSuiteId: "suite.frozen-candidate",
    split,
    baseline: metadata(split, "baseline"),
    candidate: metadata(split, "candidate"),
    baselineResults: [run(split, "baseline", true)],
    candidateResults: [run(split, "candidate", candidatePassed)],
    createdAt: now,
  });

const frozenCandidate = () => buildFrozenCandidateRecord({
  candidate,
  graphId: "blocker-closure-v1",
  selectionHash: "a49f7e68fb",
  epochId: "epoch.2026-05-05",
  frozenAt: now,
  promptFragments: [
    {
      fragmentId: "prompt.frozen-candidate.policy",
      content: "visible prompt fragment only",
    },
  ],
  visibleInputBindings: [
    {
      bindingId: "binding.frozen-candidate.train",
      sourceKind: "eval_scorecard",
      sourceArtifactId: "scorecard.frozen-candidate.train.pass",
      split: "train",
      contentHash: "sha256:train-visible",
      optimizerInputAllowed: true,
      includedEvalCaseIds: ["eval.frozen-candidate.train"],
    },
    {
      bindingId: "binding.frozen-candidate.dev",
      sourceKind: "eval_scorecard",
      sourceArtifactId: "scorecard.frozen-candidate.dev.pass",
      split: "dev",
      contentHash: "sha256:dev-visible",
      optimizerInputAllowed: true,
      includedEvalCaseIds: ["eval.frozen-candidate.dev"],
    },
  ],
});

describe("optimizer frozen candidate holdout protocol", () => {
  test("freezes candidate identity with visible input bindings and prompt fragment hashes", () => {
    const frozen = frozenCandidate();

    expect(frozen.status).toBe("frozen");
    expect(frozen.graphId).toBe("blocker-closure-v1");
    expect(frozen.selectionHash).toBe("a49f7e68fb");
    expect(frozen.candidatePatchId).toBe(candidate.candidatePatchId);
    expect(frozen.promptFragments[0]?.contentHash).toStartWith("sha256:");
    expect(JSON.stringify(frozen)).not.toContain("visible prompt fragment only");
  });

  test("blocks holdout aggregate proof when visible candidate quality is negative", () => {
    const frozen = frozenCandidate();
    const visibleEvaluation = assessFrozenCandidateVisibleEvaluation({
      frozenCandidate: frozen,
      visibleScorecards: [
        scorecard("train"),
        scorecard("dev", false),
      ],
    });
    const proof = buildHoldoutAggregateProof({
      frozenCandidate: frozen,
      visibleEvaluation,
      createdAt: now,
    });

    expect(visibleEvaluation.readyForHoldout).toBe(false);
    expect(visibleEvaluation.blocker).toContain("Visible scorecards failed");
    expect(proof.status).toBe("blocked");
    expect(proof.blockedReason).toContain("visible evaluation did not qualify");
    expect(proof.sourceScorecardIds).toEqual([]);
    expect(proof.aggregateOnly).toBe(true);
    expect(proof.optimizerInputAllowed).toBe(false);
  });

  test("exports aggregate-only hidden holdout proof after visible train/dev pass", () => {
    const frozen = frozenCandidate();
    const visibleEvaluation = assessFrozenCandidateVisibleEvaluation({
      frozenCandidate: frozen,
      visibleScorecards: [
        scorecard("train"),
        scorecard("dev"),
      ],
    });
    const proof = buildHoldoutAggregateProof({
      frozenCandidate: frozen,
      visibleEvaluation,
      holdoutScorecards: [scorecard("holdout")],
      sourceReplayExportIds: ["real-acp-replay-export.hidden"],
      sourceRunIds: ["real-acp-run.hidden"],
      hiddenHoldoutCaseCount: 3,
      createdAt: now,
    });
    const assessment = assertFrozenCandidateNonLeakage({
      frozenCandidate: frozen,
      holdoutAggregateProofs: [proof],
    });

    expect(visibleEvaluation.readyForHoldout).toBe(true);
    expect(proof.status).toBe("passed");
    expect(proof.metrics).toMatchObject({
      scorecardCount: 1,
      passedScorecardCount: 1,
      failedScorecardCount: 0,
      hiddenHoldoutCaseCount: 3,
    });
    expect(assessment.passed).toBe(true);
    expect(JSON.stringify(proof)).not.toContain("eval.frozen-candidate.holdout");
    expect(JSON.stringify(proof)).not.toContain("empty_edits");
  });

  test("rejects non-holdout scorecards as hidden holdout proof inputs", () => {
    const frozen = frozenCandidate();
    const visibleEvaluation = assessFrozenCandidateVisibleEvaluation({
      frozenCandidate: frozen,
      visibleScorecards: [
        scorecard("train"),
        scorecard("dev"),
      ],
    });

    expect(() => buildHoldoutAggregateProof({
      frozenCandidate: frozen,
      visibleEvaluation,
      holdoutScorecards: [scorecard("dev")],
      createdAt: now,
    })).toThrow(/rejected non-holdout/);
  });
});
