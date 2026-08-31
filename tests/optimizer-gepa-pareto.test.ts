import { describe, expect, test } from "bun:test";
import {
  buildGepaParetoFronts,
  compareGepaParetoCandidates,
  stableGepaObjectiveSetId,
  type GepaParetoCandidate,
  type GepaParetoObjective,
} from "../src/optimizer/gepa-pareto";

const objectives: GepaParetoObjective[] = [
  { metricId: "aggregate-score", higherIsBetter: true },
  { metricId: "latency-ms", higherIsBetter: false },
];

const candidate = (overrides: Partial<GepaParetoCandidate> & {
  candidatePatchId: string;
  score: number;
  latency: number;
}): GepaParetoCandidate => ({
  candidatePatchId: overrides.candidatePatchId,
  modelProfileId: overrides.modelProfileId ?? "model.qwen36.local",
  codebaseProfileId: overrides.codebaseProfileId ?? "codebase.bleeding-agent",
  evalPackId: overrides.evalPackId ?? "evalpack.coding.dev",
  objectiveSetId: overrides.objectiveSetId ?? stableGepaObjectiveSetId(objectives),
  policyId: overrides.policyId ?? "policy.qwen36.bleeding-agent",
  evalResultIds: overrides.evalResultIds ?? [`run.${overrides.candidatePatchId}`],
  scorecardIds: overrides.scorecardIds ?? [`scorecard.${overrides.candidatePatchId}`],
  sourceTraceIds: overrides.sourceTraceIds ?? [],
  createdAt: overrides.createdAt ?? "2026-04-30T00:00:00.000Z",
  metrics: overrides.metrics ?? [
    {
      metricId: "aggregate-score",
      name: "Aggregate score",
      value: overrides.score,
      unit: "score",
      higherIsBetter: true,
    },
    {
      metricId: "latency-ms",
      name: "Latency",
      value: overrides.latency,
      unit: "ms",
      higherIsBetter: false,
    },
  ],
});

describe("GEPA Pareto fronts", () => {
  test("removes dominated candidates while preserving non-dominated tradeoffs", () => {
    const fronts = buildGepaParetoFronts({
      candidates: [
        candidate({ candidatePatchId: "candidate.slow-best", score: 0.95, latency: 120 }),
        candidate({ candidatePatchId: "candidate.fast-good", score: 0.9, latency: 80 }),
        candidate({ candidatePatchId: "candidate.dominated", score: 0.85, latency: 110 }),
      ],
    });

    expect(fronts).toHaveLength(1);
    expect(fronts[0]?.dominatedCandidateIds).toEqual(["candidate.dominated"]);
    expect(fronts[0]?.candidates.map((entry) => entry.candidatePatchId)).toEqual([
      "candidate.slow-best",
      "candidate.fast-good",
    ]);
  });

  test("respects objective direction for lower-is-better metrics", () => {
    const fast = candidate({ candidatePatchId: "candidate.fast", score: 0.9, latency: 80 });
    const slow = candidate({ candidatePatchId: "candidate.slow", score: 0.9, latency: 120 });

    expect(compareGepaParetoCandidates(fast, slow, objectives)).toBe("left_dominates");
    expect(buildGepaParetoFronts({ candidates: [slow, fast] })[0]?.candidates.map((entry) => entry.candidatePatchId))
      .toEqual(["candidate.fast"]);
  });

  test("partitions by model profile, codebase profile, eval pack, and objective set", () => {
    const alternateObjectives: GepaParetoObjective[] = [
      { metricId: "tool-call-success-rate", higherIsBetter: true },
    ];
    const alternateObjectiveSetId = stableGepaObjectiveSetId(alternateObjectives);
    const fronts = buildGepaParetoFronts({
      candidates: [
        candidate({ candidatePatchId: "candidate.base", score: 0.9, latency: 90 }),
        candidate({ candidatePatchId: "candidate.other-model", modelProfileId: "model.gpt55.master", score: 0.9, latency: 90 }),
        candidate({ candidatePatchId: "candidate.other-codebase", codebaseProfileId: "codebase.other", score: 0.9, latency: 90 }),
        candidate({ candidatePatchId: "candidate.other-pack", evalPackId: "evalpack.coding.holdout", score: 0.9, latency: 90 }),
        candidate({
          candidatePatchId: "candidate.other-objectives",
          score: 0.9,
          latency: 90,
          objectiveSetId: alternateObjectiveSetId,
          metrics: [
            {
              metricId: "tool-call-success-rate",
              name: "Tool success",
              value: 0.99,
              unit: "ratio",
              higherIsBetter: true,
            },
          ],
        }),
      ],
    });

    expect(fronts).toHaveLength(5);
    expect(fronts.map((front) => front.key)).toEqual([
      expect.objectContaining({ codebaseProfileId: "codebase.bleeding-agent", evalPackId: "evalpack.coding.dev", modelProfileId: "model.gpt55.master" }),
      expect.objectContaining({ codebaseProfileId: "codebase.bleeding-agent", evalPackId: "evalpack.coding.dev", modelProfileId: "model.qwen36.local" }),
      expect.objectContaining({ codebaseProfileId: "codebase.bleeding-agent", evalPackId: "evalpack.coding.dev", modelProfileId: "model.qwen36.local" }),
      expect.objectContaining({ codebaseProfileId: "codebase.bleeding-agent", evalPackId: "evalpack.coding.holdout", modelProfileId: "model.qwen36.local" }),
      expect.objectContaining({ codebaseProfileId: "codebase.other", evalPackId: "evalpack.coding.dev", modelProfileId: "model.qwen36.local" }),
    ]);
  });

  test("preserves metric ties and candidate/eval lineage", () => {
    const fronts = buildGepaParetoFronts({
      candidates: [
        candidate({
          candidatePatchId: "candidate.tie.b",
          score: 0.92,
          latency: 100,
          evalResultIds: ["run.tie.b"],
          scorecardIds: ["scorecard.tie.b"],
          sourceTraceIds: ["trace.tie.b"],
        }),
        candidate({
          candidatePatchId: "candidate.tie.a",
          score: 0.92,
          latency: 100,
          evalResultIds: ["run.tie.a"],
          scorecardIds: ["scorecard.tie.a"],
          sourceTraceIds: ["trace.tie.a"],
        }),
      ],
    });

    expect(fronts[0]?.dominatedCandidateIds).toEqual([]);
    expect(fronts[0]?.candidates.map((entry) => entry.candidatePatchId)).toEqual([
      "candidate.tie.a",
      "candidate.tie.b",
    ]);
    expect(fronts[0]?.candidates[0]).toMatchObject({
      candidatePatchId: "candidate.tie.a",
      evalResultIds: ["run.tie.a"],
      scorecardIds: ["scorecard.tie.a"],
      sourceTraceIds: ["trace.tie.a"],
    });
  });

  test("returns deterministic fronts independent of input ordering", () => {
    const inputs = [
      candidate({ candidatePatchId: "candidate.c", score: 0.88, latency: 70 }),
      candidate({ candidatePatchId: "candidate.a", score: 0.94, latency: 100 }),
      candidate({ candidatePatchId: "candidate.b", score: 0.9, latency: 80 }),
      candidate({ candidatePatchId: "candidate.dominated", score: 0.8, latency: 130 }),
    ];

    const first = buildGepaParetoFronts({ candidates: inputs });
    const second = buildGepaParetoFronts({ candidates: [...inputs].reverse() });

    expect(first).toEqual(second);
    expect(first[0]?.candidates.map((entry) => entry.candidatePatchId)).toEqual([
      "candidate.a",
      "candidate.b",
      "candidate.c",
    ]);
  });
});
