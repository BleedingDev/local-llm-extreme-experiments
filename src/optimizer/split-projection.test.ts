import { describe, expect, test } from "bun:test";
import {
  assertNoDuplicateStableIds,
  assertNoSplitLeakage,
  assertOptimizerSplitVisibilityAllowed,
  dedupeStableIds,
  findDuplicateStableIds,
  findSplitLeakage,
  projectStableIdsToOptimizerSplits,
  projectedIdsForOptimizerConsumer,
  validateOptimizerSplitVisibility,
} from "./split-projection";

const fixtureIds = [
  "task.delta",
  "task.alpha",
  "task.echo",
  "task.bravo",
  "task.charlie",
  "task.foxtrot",
  "task.golf",
  "task.hotel",
  "task.india",
] as const;

const projectionInput = {
  stableIds: fixtureIds,
  seed: "seed.local-evidence.v1",
  projectionVersion: "projection.v1",
  sourceId: "evidence.split.action-v2",
  ratios: { train: 1, dev: 1, hiddenHoldout: 1 },
} as const;

describe("optimizer split projection", () => {
  test("projects stable ids deterministically with seed and version metadata", () => {
    const projection = projectStableIdsToOptimizerSplits(projectionInput);
    const reordered = projectStableIdsToOptimizerSplits({
      ...projectionInput,
      stableIds: [...fixtureIds].reverse(),
    });

    expect(projection.projectionId).toBe("optimizer.split-projection.projection.v1.b5e5e0776d771704");
    expect(reordered).toEqual(projection);
    expect(projection.metadata).toMatchObject({
      projectionVersion: "projection.v1",
      seed: "seed.local-evidence.v1",
      sourceId: "evidence.split.action-v2",
      algorithm: "sha256-threshold.v1",
      ratios: { train: 1, dev: 1, hiddenHoldout: 1 },
      normalizedRatios: {
        train: 1 / 3,
        dev: 1 / 3,
        hiddenHoldout: 1 / 3,
      },
      stableIdCount: 9,
    });
    expect(projection.splits).toEqual({
      train: [
        "task.alpha",
        "task.echo",
        "task.golf",
        "task.hotel",
      ],
      dev: [
        "task.charlie",
        "task.india",
      ],
      "hidden-holdout": [
        "task.bravo",
        "task.delta",
        "task.foxtrot",
      ],
    });
    expect(projection.counts).toEqual({
      train: 4,
      dev: 2,
      "hidden-holdout": 3,
    });
    expect(projection.assignments.map((assignment) => assignment.stableId)).toEqual(
      [...fixtureIds].sort((left, right) => left.localeCompare(right)),
    );
    expect(projection.assignments.find((assignment) => assignment.stableId === "task.bravo"))
      .toMatchObject({
        split: "hidden-holdout",
        visibility: "hidden",
        ordinal: 1,
      });
  });

  test("rejects duplicate stable ids instead of silently resealing a split", () => {
    expect(() =>
      projectStableIdsToOptimizerSplits({
        ...projectionInput,
        stableIds: ["task.alpha", "task.beta", "task.alpha"],
      }),
    ).toThrow(/duplicate stable ids: task\.alpha/);
  });

  test("fails closed for hidden-holdout visibility misuse", () => {
    expect(validateOptimizerSplitVisibility("candidate-generation", ["train", "dev"]))
      .toMatchObject({
        allowed: true,
        blockedSplits: [],
      });
    expect(() =>
      assertOptimizerSplitVisibilityAllowed("candidate-generation", ["hidden-holdout"]),
    ).toThrow(/candidate-generation cannot read split labels: hidden-holdout/);
    expect(() =>
      assertOptimizerSplitVisibilityAllowed("development-evaluation", ["dev", "hidden-holdout"]),
    ).toThrow(/development-evaluation cannot read split labels: hidden-holdout/);
    expect(() =>
      assertOptimizerSplitVisibilityAllowed("unknown-consumer", ["train"]),
    ).toThrow(/unknown optimizer evidence consumer/);
    expect(() =>
      assertOptimizerSplitVisibilityAllowed("candidate-generation", ["public-dev"]),
    ).toThrow(/unknown optimizer split labels requested/);
    expect(assertOptimizerSplitVisibilityAllowed(
      "frozen-candidate-hidden-holdout-evaluator",
      ["hidden-holdout"],
    ).allowed).toBe(true);
  });

  test("selects only consumer-visible ids from a projection", () => {
    const projection = projectStableIdsToOptimizerSplits(projectionInput);

    expect(projectedIdsForOptimizerConsumer(projection, "candidate-generation")).toEqual([
      "task.alpha",
      "task.echo",
      "task.golf",
      "task.hotel",
      "task.charlie",
      "task.india",
    ]);
    expect(projectedIdsForOptimizerConsumer(projection, "frozen-candidate-hidden-holdout-evaluator"))
      .toEqual([
        "task.bravo",
        "task.delta",
        "task.foxtrot",
      ]);
    expect(() =>
      projectedIdsForOptimizerConsumer(projection, "prompt-drafting", ["train", "hidden-holdout"]),
    ).toThrow(/prompt-drafting cannot read split labels: hidden-holdout/);
  });

  test("provides dedup and leakage primitives for split fixtures", () => {
    expect(dedupeStableIds(["case.b", " case.a ", "case.b", "case.c"])).toEqual({
      stableIds: ["case.a", "case.b", "case.c"],
      duplicates: [{ stableId: "case.b", count: 2 }],
    });
    expect(findDuplicateStableIds(["case.a", "case.a", "case.b"])).toEqual([
      { stableId: "case.a", count: 2 },
    ]);
    expect(() => assertNoDuplicateStableIds(["case.a", "case.a"]))
      .toThrow(/duplicate stable ids detected: case\.a/);

    const leakage = findSplitLeakage({
      train: ["case.a", "case.b"],
      dev: ["case.c", "case.b"],
      "hidden-holdout": ["case.d", "case.a"],
    });

    expect(leakage).toEqual([
      {
        leftSplit: "train",
        rightSplit: "dev",
        overlapIds: ["case.b"],
        overlapCount: 1,
      },
      {
        leftSplit: "train",
        rightSplit: "hidden-holdout",
        overlapIds: ["case.a"],
        overlapCount: 1,
      },
    ]);
    expect(() => assertNoSplitLeakage({ train: ["case.a"], "hidden-holdout": ["case.a"] }))
      .toThrow(/split leakage detected: train\/hidden-holdout: case\.a/);
  });
});
