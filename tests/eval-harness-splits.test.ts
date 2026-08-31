import { describe, expect, test } from "bun:test";
import { evalFixtures } from "../src/eval-harness/fixtures";
import {
  assertCandidateTrainingInputAllowed,
  createHoldoutEvalPack,
  createVisibleEvalPack,
  devEvalCases,
  groupEvalCasesBySplit,
  holdoutEvalCases,
  summarizeEvalSplits,
  trainEvalCases,
  validateCandidateTrainingInput,
  visibleEvalCases,
} from "../src/eval-harness/splits";

const trainIds = [
  "eval.chat-no-side-effect",
  "eval.small-edit",
] as const;

const devIds = [
  "eval.read-only-report",
  "eval.schema-shape-reliability",
  "eval.verification-repair",
] as const;

const holdoutIds = [
  "eval.tool-failure-recovery",
  "eval.truncation-behavior",
] as const;

describe("eval harness splits", () => {
  test("groups built-in fixtures into deterministic train, dev, and holdout splits", () => {
    const groups = groupEvalCasesBySplit(evalFixtures);

    expect(groups.train.map((evalCase) => evalCase.evalCaseId)).toEqual(trainIds);
    expect(groups.dev.map((evalCase) => evalCase.evalCaseId)).toEqual(devIds);
    expect(groups.holdout.map((evalCase) => evalCase.evalCaseId)).toEqual(holdoutIds);
  });

  test("summarizes splits with stable ids and tags", () => {
    expect(summarizeEvalSplits(evalFixtures)).toEqual([
      {
        split: "train",
        count: 2,
        evalCaseIds: [...trainIds],
        fixtureWorkspaceIds: [
          "fixture.chat-no-side-effect",
          "fixture.small-edit",
        ],
        tags: [
          "chat",
          "no-side-effect",
          "protected-paths",
          "read-only",
          "small-edit",
        ],
      },
      {
        split: "dev",
        count: 3,
        evalCaseIds: [...devIds],
        fixtureWorkspaceIds: [
          "fixture.read-only-report",
          "fixture.schema-shape-reliability",
          "fixture.verification-repair",
        ],
        tags: [
          "fact-gathering",
          "read-only",
          "reliability",
          "repair",
          "report",
          "schema",
          "structured-output",
          "tests",
          "verification",
        ],
      },
      {
        split: "holdout",
        count: 2,
        evalCaseIds: [...holdoutIds],
        fixtureWorkspaceIds: [
          "fixture.tool-failure-recovery",
          "fixture.truncation-behavior",
        ],
        tags: [
          "fallback",
          "long-context",
          "recovery",
          "tail-facts",
          "tool-failure",
          "truncation",
        ],
      },
    ]);
  });

  test("selects visible train and dev packs while excluding hidden holdout cases", () => {
    expect(trainEvalCases(evalFixtures).map((evalCase) => evalCase.evalCaseId)).toEqual(trainIds);
    expect(devEvalCases(evalFixtures).map((evalCase) => evalCase.evalCaseId)).toEqual(devIds);
    expect(visibleEvalCases(evalFixtures).map((evalCase) => evalCase.evalCaseId)).toEqual([
      ...trainIds,
      ...devIds,
    ]);
    expect(holdoutEvalCases(evalFixtures).map((evalCase) => evalCase.evalCaseId)).toEqual(
      holdoutIds,
    );

    expect(createVisibleEvalPack(evalFixtures)).toMatchObject({
      splits: ["train", "dev"],
      hidden: false,
      evalCaseIds: [...trainIds, ...devIds],
      fixtureWorkspaceIds: [
        "fixture.chat-no-side-effect",
        "fixture.small-edit",
        "fixture.read-only-report",
        "fixture.schema-shape-reliability",
        "fixture.verification-repair",
      ],
    });

    expect(createHoldoutEvalPack(evalFixtures)).toMatchObject({
      splits: ["holdout"],
      hidden: true,
      evalCaseIds: [...holdoutIds],
      fixtureWorkspaceIds: [
        "fixture.tool-failure-recovery",
        "fixture.truncation-behavior",
      ],
    });
  });

  test("vetoes candidate promotion when training input includes hidden holdout fixtures", () => {
    const validation = validateCandidateTrainingInput({
      evalCaseIds: [
        "eval.small-edit",
        "eval.tool-failure-recovery",
      ],
      fixtureWorkspaceIds: [
        "fixture.truncation-behavior",
      ],
    }, evalFixtures);

    expect(validation).toMatchObject({
      vetoed: true,
      blockedEvalCaseIds: ["eval.tool-failure-recovery"],
      blockedFixtureWorkspaceIds: ["fixture.truncation-behavior"],
      hiddenHoldoutEvalCaseIds: [...holdoutIds],
      hiddenHoldoutFixtureWorkspaceIds: [
        "fixture.tool-failure-recovery",
        "fixture.truncation-behavior",
      ],
      visibleEvalCaseIds: [...trainIds, ...devIds],
    });
    expect(validation.message).toContain("hidden holdout");
    expect(() =>
      assertCandidateTrainingInputAllowed({
        evalCaseIds: ["eval.truncation-behavior"],
      }, evalFixtures),
    ).toThrow(/candidate promotion vetoed/);
  });

  test("allows candidate promotion when training input only includes visible fixtures", () => {
    const validation = assertCandidateTrainingInputAllowed({
      evalCaseIds: [...trainIds, ...devIds],
      evalCases: visibleEvalCases(evalFixtures),
    }, evalFixtures);

    expect(validation.vetoed).toBe(false);
    expect(validation.blockedEvalCaseIds).toEqual([]);
    expect(validation.blockedFixtureWorkspaceIds).toEqual([]);
  });

  test("detects duplicate eval case ids before split selection", () => {
    const duplicate = {
      ...evalFixtures[0],
      split: "holdout" as const,
    };

    expect(() => groupEvalCasesBySplit([...evalFixtures, duplicate])).toThrow(
      /duplicate eval case id: eval\.chat-no-side-effect/,
    );
  });
});
