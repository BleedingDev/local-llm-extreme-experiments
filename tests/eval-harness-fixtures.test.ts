import { describe, expect, test } from "bun:test";
import {
  evalFixtureIds,
  evalFixtures,
  fixtureWorkspaces,
} from "../src/eval-harness/fixtures";
import {
  EvalCaseSchema,
  FixtureWorkspaceSchema,
  type EvalAssertion,
} from "../src/eval-harness/types";

const expectedFixtureIds = [
  "eval.chat-no-side-effect",
  "eval.read-only-report",
  "eval.small-edit",
  "eval.verification-repair",
  "eval.tool-failure-recovery",
  "eval.truncation-behavior",
  "eval.schema-shape-reliability",
] as const;

const expectedProtectedPathsByFixtureId = {
  "eval.chat-no-side-effect": ["README.md", "docs/release.md"],
  "eval.read-only-report": ["incidents/summary.md", "services.json"],
  "eval.small-edit": ["package.json"],
  "eval.verification-repair": ["tests/math.test.mjs"],
  "eval.tool-failure-recovery": ["scripts/read-primary.mjs", "data/fallback.txt"],
  "eval.truncation-behavior": ["logs/events.log"],
  "eval.schema-shape-reliability": [
    "schema/expected-result.schema.json",
    "input/request.json",
  ],
} as const;

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

describe("eval harness fixtures", () => {
  test("exports every built-in fixture id exactly once", () => {
    expect(evalFixtureIds).toEqual(expectedFixtureIds);
    expect(evalFixtures.map((fixture) => fixture.evalCaseId)).toEqual(expectedFixtureIds);
    expect(new Set(evalFixtures.map((fixture) => fixture.evalCaseId)).size).toBe(
      expectedFixtureIds.length,
    );
  });

  test("parses exported eval cases and workspaces", () => {
    for (const fixture of evalFixtures) {
      expect(EvalCaseSchema.parse(fixture).evalCaseId).toBe(fixture.evalCaseId);
      expect(FixtureWorkspaceSchema.parse(fixture.fixtureWorkspace).fixtureWorkspaceId).toBe(
        fixture.fixtureWorkspace.fixtureWorkspaceId,
      );
    }

    expect(fixtureWorkspaces.map((workspace) => workspace.fixtureWorkspaceId)).toEqual(
      evalFixtures.map((fixture) => fixture.fixtureWorkspace.fixtureWorkspaceId),
    );
  });

  test("includes protected paths for fixtures that should avoid collateral edits", () => {
    for (const fixture of evalFixtures) {
      const expectedProtectedPaths =
        expectedProtectedPathsByFixtureId[
          fixture.evalCaseId as keyof typeof expectedProtectedPathsByFixtureId
        ];
      const filePaths = new Set(fixture.fixtureWorkspace.files.map((file) => file.path));

      expect(expectedProtectedPaths.length).toBeGreaterThan(0);
      expect(fixture.fixtureWorkspace.protectedPaths).toEqual([...expectedProtectedPaths]);

      for (const protectedPath of fixture.fixtureWorkspace.protectedPaths) {
        expect(filePaths.has(protectedPath)).toBe(true);
      }
    }
  });

  test("uses only safe relative fixture paths", () => {
    for (const fixture of evalFixtures) {
      const paths = [
        ...fixture.fixtureWorkspace.files.map((file) => file.path),
        ...fixture.fixtureWorkspace.protectedPaths,
        ...fixture.assertions.flatMap(pathBearingAssertionPaths),
      ];

      for (const path of paths) {
        expectSafeRelativePath(path);
      }
    }
  });
});
