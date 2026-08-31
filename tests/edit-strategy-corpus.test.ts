import { describe, expect, test } from "bun:test";
import {
  editStrategyEvalCaseIds,
  editStrategyEvalCases,
  editStrategyEvalCasesByFailureMode,
  editStrategyEvalCasesByStrategyFamily,
  editStrategyFixtureWorkspaces,
} from "../src/eval-harness/edit-strategy-corpus";
import {
  EditStrategyEvalCaseSchema,
  EditStrategyEvalFailureModeSchema,
} from "../src/eval-harness/edit-strategy-corpus";
import { initialExperimentalEditStrategyIds, parseCanonicalEditStrategyDefinitions } from "../src/edit-strategy/taxonomy";
import type { EvalAssertion } from "../src/eval-harness/types";

const pathBearingAssertionPaths = (assertion: EvalAssertion): string[] => {
  switch (assertion.assertionKind) {
    case "file_contains":
    case "file_not_contains":
      return [assertion.path];
    case "no_forbidden_path_changed":
      return assertion.paths;
    case "command_exit_code":
    case "json_pointer_equals":
    case "llm_judge_min_score":
      return [];
  }
};

const expectSafeRelativePath = (path: string) => {
  expect(path).not.toBe("");
  expect(path.startsWith("/")).toBe(false);
  expect(/^[A-Za-z]:[\\/]/.test(path)).toBe(false);
  expect(path.includes("\\")).toBe(false);
  expect(path.split("/")).not.toContain("..");
};

describe("edit strategy eval corpus", () => {
  test("exports every edit-strategy fixture id exactly once", () => {
    expect(editStrategyEvalCases.map((evalCase) => evalCase.editEvalCaseId)).toEqual(editStrategyEvalCaseIds);
    expect(new Set(editStrategyEvalCases.map((evalCase) => evalCase.editEvalCaseId)).size).toBe(
      editStrategyEvalCaseIds.length,
    );
  });

  test("parses every edit-strategy eval case and workspace", () => {
    for (const evalCase of editStrategyEvalCases) {
      expect(EditStrategyEvalCaseSchema.parse(evalCase).editEvalCaseId).toBe(evalCase.editEvalCaseId);
    }

    expect(editStrategyFixtureWorkspaces.map((workspace) => workspace.fixtureWorkspaceId)).toEqual(
      editStrategyEvalCases.map((evalCase) => evalCase.fixtureWorkspace.fixtureWorkspaceId),
    );
  });

  test("covers every required edit failure mode with at least one fixture", () => {
    const byFailureMode = editStrategyEvalCasesByFailureMode();

    for (const failureMode of EditStrategyEvalFailureModeSchema.options) {
      expect(byFailureMode[failureMode].length).toBeGreaterThan(0);
    }
  });

  test("covers initial experimental strategy families without choosing a winner", () => {
    const definitions = parseCanonicalEditStrategyDefinitions();
    const initialFamilies = new Set(
      initialExperimentalEditStrategyIds()
        .map((id) => definitions.find((definition) => definition.strategyId === id)?.family)
        .filter((family): family is NonNullable<typeof family> => family != null),
    );
    const byFamily = editStrategyEvalCasesByStrategyFamily();

    for (const family of initialFamilies) {
      expect(byFamily[family].length).toBeGreaterThan(0);
    }

    expect(editStrategyEvalCases.some((evalCase) => evalCase.probes.length > 1)).toBe(true);
    expect(JSON.stringify(editStrategyEvalCases)).not.toContain("preferredStrategy");
    expect(JSON.stringify(editStrategyEvalCases)).not.toContain("bestStrategy");
  });

  test("keeps protected and expected changed paths disjoint and safe", () => {
    for (const evalCase of editStrategyEvalCases) {
      const expectedChangedPaths = new Set(evalCase.expectedChangedFiles);
      const protectedPaths = new Set(evalCase.fixtureWorkspace.protectedPaths);
      const allPaths = [
        ...evalCase.fixtureWorkspace.files.map((file) => file.path),
        ...evalCase.fixtureWorkspace.protectedPaths,
        ...evalCase.targetFiles,
        ...evalCase.expectedChangedFiles,
        ...evalCase.forbiddenChangedFiles,
        ...evalCase.assertions.flatMap(pathBearingAssertionPaths),
      ];

      for (const path of allPaths) {
        expectSafeRelativePath(path);
      }

      for (const forbiddenPath of evalCase.forbiddenChangedFiles) {
        expect(expectedChangedPaths.has(forbiddenPath)).toBe(false);
      }

      for (const protectedPath of protectedPaths) {
        expect(evalCase.fixtureWorkspace.files.some((file) => file.path === protectedPath)).toBe(true);
      }
    }
  });

  test("contains probes for malformed, stale, protected, and applied-but-broken outcomes", () => {
    const probes = editStrategyEvalCases.flatMap((evalCase) => evalCase.probes);

    expect(probes.some((probe) => probe.expectedParseStatus === "failed")).toBe(true);
    expect(probes.some((probe) => probe.expectedStaleContextStatus === "stale")).toBe(true);
    expect(probes.some((probe) => probe.expectedProtectedPathTouched === true)).toBe(true);
    expect(probes.some((probe) => probe.expectedPostApplyConsistencyStatus === "inconsistent")).toBe(true);
    expect(probes.some((probe) => probe.expectedSelfDetectedRegressionStatus === "confirmed")).toBe(true);
    expect(probes.some((probe) => probe.expectedApplyStatus === "skipped")).toBe(true);
  });
});
