import { describe, expect, test } from "bun:test";
import {
  EditStrategyAblationReportSchema,
  runEditStrategyAblation,
} from "../src/eval-harness/edit-strategy-ablation";

const now = "2026-04-30T00:00:00.000Z";

describe("edit strategy ablation runner", () => {
  test("runs visible-split ablations without leaking holdout cases into optimization", () => {
    const report = runEditStrategyAblation({ createdAt: now });

    expect(EditStrategyAblationReportSchema.parse(report).ablationRunId).toBe("ablation.edit-strategy.visible");
    expect(report.selectedSplits).toEqual(["train", "dev"]);
    expect(report.hiddenHoldoutUsed).toBe(false);
    expect(report.optimizationAllowed).toBe(true);
    expect(report.selectionDiscipline).toEqual({
      rankingScope: "per-model-codebase-strategy-family",
      globalWinnerSelected: false,
      holdoutExcludedFromOptimization: true,
    });
    expect(report.selectedEvalCaseIds).not.toContain("edit-eval.stale-read-detection");
    expect(report.hiddenHoldoutEvalCaseIds).toEqual([
      "edit-eval.formatting-sensitive-json",
      "edit-eval.protected-path-veto",
      "edit-eval.stale-read-detection",
    ]);
    expect(report.scorecards.length).toBeGreaterThan(0);
    expect(report.familySummaries.map((summary) => summary.strategyFamily)).toEqual([
      "whole_file",
      "exact_replace",
      "unified_diff",
      "apply_patch",
      "hash_range",
    ]);
  });

  test("requires an explicit holdout opt-in and keeps holdout runs marked as non-optimization input", () => {
    expect(() => runEditStrategyAblation({ splits: ["holdout"], createdAt: now })).toThrow(/holdout/);

    const report = runEditStrategyAblation({
      splits: ["holdout"],
      includeHoldout: true,
      createdAt: now,
    });

    expect(report.selectedSplits).toEqual(["holdout"]);
    expect(report.hiddenHoldoutUsed).toBe(true);
    expect(report.optimizationAllowed).toBe(false);
    expect(report.selectionDiscipline).toMatchObject({
      globalWinnerSelected: false,
      holdoutExcludedFromOptimization: true,
    });
    expect(report.selectedEvalCaseIds).toEqual([
      "edit-eval.formatting-sensitive-json",
      "edit-eval.protected-path-veto",
      "edit-eval.stale-read-detection",
    ]);
  });

  test("captures strategy-specific parse, apply, stale, and applied-but-broken outcomes", () => {
    const report = runEditStrategyAblation({ createdAt: now });
    const probe = (probeId: string) => {
      const result = report.probeResults.find((candidate) => candidate.probeId === probeId);
      expect(result).toBeDefined();
      return result!;
    };

    expect(probe("probe.malformed.unified-diff")).toMatchObject({
      parseStatus: "failed",
      applyStatus: "not_started",
      errorCode: "parse_error",
      expectedOutcomeMatched: true,
      status: "passed",
    });
    expect(probe("probe.malformed.unified-diff").policyFeedbackTargets).toEqual([
      "rendered-contract",
      "fallback-order",
    ]);
    expect(probe("probe.repeated.exact-ambiguous")).toMatchObject({
      applyStatus: "failed",
      errorCode: "exact_match_ambiguous",
      expectedOutcomeMatched: true,
      status: "failed",
    });
    expect(probe("probe.repeated.exact-ambiguous").policyFeedbackTargets).toEqual([
      "strategy-routing",
      "fallback-order",
    ]);
    expect(probe("probe.no-op.apply-patch")).toMatchObject({
      applyStatus: "skipped",
      expectedOutcomeMatched: true,
      status: "passed",
    });
    expect(probe("probe.applied-broken.apply-patch")).toMatchObject({
      applyStatus: "passed",
      postApplyConsistencyStatus: "inconsistent",
      verificationStatus: "failed",
      errorCode: "post_apply_syntax_failure",
      status: "failed",
    });
    expect(probe("probe.applied-broken.apply-patch").policyFeedbackTargets).toEqual([
      "verifier-enforcement",
      "repair-instructions",
      "rollback-policy",
      "strategy-routing",
    ]);
    expect(probe("probe.applied-broken.apply-patch").objectiveMetrics.map((metric) => metric.metricId))
      .toContain("policy-feedback-verifier-enforcement-count");
  });

  test("compares candidate families against the whole-file baseline without selecting a global winner", () => {
    const report = runEditStrategyAblation({ createdAt: now });
    const exactReplace = report.familySummaries.find((summary) => summary.strategyFamily === "exact_replace");
    const applyPatch = report.familySummaries.find((summary) => summary.strategyFamily === "apply_patch");
    const hashRange = report.familySummaries.find((summary) => summary.strategyFamily === "hash_range");

    expect(exactReplace).toMatchObject({
      probedEvalCaseCount: 2,
      expectedOutcomeMatchRate: 1,
      wholeFileBaselineAverageScore: 1,
    });
    expect(exactReplace!.scoreDeltaVsWholeFileBaseline).toBeLessThan(0);
    expect(applyPatch!.appliedButBrokenCount).toBe(1);
    expect(applyPatch!.policyFeedbackTargetCounts).toContainEqual({
      target: "verifier-enforcement",
      count: 1,
    });
    expect(hashRange).toMatchObject({
      probedEvalCaseCount: 1,
      taskPassRate: 1,
    });
    expect(report.familySummaries.every((summary) => summary.scorecardIds.length > 0 || summary.probeCount === 0))
      .toBe(true);
  });
});
