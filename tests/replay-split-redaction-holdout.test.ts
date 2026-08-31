import { describe, expect, test } from "bun:test";
import {
  assertReplayCasesAllowedForOptimizerInput,
  assertReplayRunResultsAllowedForGepaFeedback,
  createReplayProposerPromptCases,
  replayEvalCaseSkeletons,
  routingReplayScenarioSkeletons,
  selectReplayCasesForOptimizerInput,
  selectReplayRunResultsForGepaFeedback,
} from "../src/replay";
import type { EvalRunResult } from "../src/eval-harness/types";

const now = "2026-05-01T00:00:00.000Z";
const context = {
  policyId: "policy.replay.test",
  modelProfileId: "model.replay.test",
  codebaseProfileId: "codebase.replay.test",
  modelServerId: "server.replay.test",
  modelServerProfileId: "server-profile.replay.test",
  canonicalToolVersion: "canonical-tools.replay.test",
  renderedToolVersion: "rendered-tools.replay.test",
  resultStyleVersion: "result-style.replay.test",
  verificationPolicyVersion: "verification.replay.test",
};

const makeRun = (evalCaseId: string): EvalRunResult => ({
  runResultId: `run.candidate.${evalCaseId}`,
  comparisonRunId: "compare.replay.gepa.candidate",
  runRole: "candidate",
  evalCaseId,
  split: evalCaseId.includes("cancellation") ? "holdout" : "train",
  context,
  status: "failed",
  score: 0,
  assertionResults: [
    {
      assertionId: `assert.${evalCaseId}.failed`,
      assertionKind: "json_pointer_equals",
      passed: false,
      severity: "failure",
    },
  ],
  objectiveMetrics: [],
  changedFiles: [],
  startedAt: now,
  completedAt: now,
});

describe("replay split, redaction, and holdout enforcement", () => {
  test("selects only metadata-visible redacted train/dev replay cases for optimization", () => {
    const selection = selectReplayCasesForOptimizerInput(replayEvalCaseSkeletons, "optimization_selection");

    expect(selection.selectedCases.length).toBeGreaterThan(0);
    expect(selection.selectedCases.every((replayCase) => replayCase.split !== "holdout")).toBe(true);
    expect(selection.selectedCases.every((replayCase) => replayCase.redaction.status !== "needs_review")).toBe(true);
    expect(selection.hiddenHoldoutEvalCaseIds).toEqual([
      "replay.eval.edit-failure.promotion-veto",
      "replay.eval.edit-failure.self-detected-regression",
      "replay.eval.routing.cancellation",
      "replay.eval.tool-call.mcp-call",
    ]);
    expect(selection.rejectedCases.map((rejection) => rejection.evalCaseId)).toEqual(
      selection.hiddenHoldoutEvalCaseIds,
    );
  });

  test("rejects holdout and needs-review replay cases for proposer and GEPA input", () => {
    const holdoutCase = replayEvalCaseSkeletons.find(
      (replayCase) => replayCase.evalCaseId === "replay.eval.routing.cancellation",
    );
    expect(holdoutCase).toBeDefined();
    if (holdoutCase == null) {
      throw new Error("holdout case missing");
    }

    expect(() => assertReplayCasesAllowedForOptimizerInput([holdoutCase], "proposer_prompt"))
      .toThrow(/hidden from optimizer input/);

    const unsafeCase = {
      ...routingReplayScenarioSkeletons[0]!,
      evalCaseId: "replay.eval.routing.unsafe-redaction",
      redaction: {
        status: "needs_review" as const,
        needsReview: true,
        needsReviewRecordIds: ["record.replay.routing.greeting-no-side-effect.prompt"],
        recordStatuses: [
          {
            recordId: "record.replay.routing.greeting-no-side-effect.prompt",
            status: "raw_local_only" as const,
          },
        ],
      },
      sourceRefs: [
        {
          sourceKind: "record" as const,
          recordId: "record.replay.routing.greeting-no-side-effect.prompt",
          redactionStatus: "raw_local_only" as const,
        },
      ],
    };

    expect(() => assertReplayCasesAllowedForOptimizerInput([unsafeCase], "gepa_feedback"))
      .toThrow(/redaction status needs_review/);
  });

  test("builds proposer prompt inputs without hidden holdout cases", () => {
    const promptCases = createReplayProposerPromptCases(replayEvalCaseSkeletons);

    expect(promptCases.length).toBeGreaterThan(0);
    expect(promptCases.every((promptCase) => promptCase.split === "train" || promptCase.split === "dev")).toBe(true);
    expect(promptCases.map((promptCase) => promptCase.evalCaseId)).not.toContain(
      "replay.eval.tool-call.mcp-call",
    );
    expect(promptCases[0]).toEqual(expect.objectContaining({
      evalCaseId: expect.any(String),
      expectedBehaviorSummary: expect.any(String),
      routing: expect.any(Object),
    }));
  });

  test("filters and asserts replay run results before GEPA feedback consumption", () => {
    const visibleRun = makeRun("replay.eval.routing.greeting-no-side-effect");
    const holdoutRun = makeRun("replay.eval.routing.cancellation");

    expect(selectReplayRunResultsForGepaFeedback([visibleRun, holdoutRun], replayEvalCaseSkeletons)
      .map((run) => run.evalCaseId)).toEqual(["replay.eval.routing.greeting-no-side-effect"]);
    expect(() => assertReplayRunResultsAllowedForGepaFeedback([visibleRun, holdoutRun], replayEvalCaseSkeletons))
      .toThrow(/replay\.eval\.routing\.cancellation/);
  });
});
