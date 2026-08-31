import { createHash } from "node:crypto";
import { z } from "zod";
import { ObjectiveMetricSchema } from "../eval-harness/types";
import { OptimizerIdSchema } from "./types";

export const GepaParetoObjectiveSchema = z.object({
  metricId: OptimizerIdSchema,
  higherIsBetter: z.boolean(),
}).strict();
export type GepaParetoObjective = z.infer<typeof GepaParetoObjectiveSchema>;

export const GepaParetoCandidateSchema = z.object({
  candidatePatchId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  evalPackId: OptimizerIdSchema,
  objectiveSetId: OptimizerIdSchema.optional(),
  policyId: OptimizerIdSchema.optional(),
  evalResultIds: z.array(OptimizerIdSchema).default([]),
  scorecardIds: z.array(OptimizerIdSchema).default([]),
  sourceTraceIds: z.array(z.string()).default([]),
  metrics: z.array(ObjectiveMetricSchema).min(1),
  createdAt: z.string().optional(),
}).strict();
export type GepaParetoCandidate = z.infer<typeof GepaParetoCandidateSchema>;

export const GepaParetoCandidateEntrySchema = GepaParetoCandidateSchema.extend({
  objectiveSetId: OptimizerIdSchema,
  objectiveValues: z.array(z.object({
    metricId: OptimizerIdSchema,
    value: z.number().finite().optional(),
    higherIsBetter: z.boolean(),
  }).strict()),
}).strict();
export type GepaParetoCandidateEntry = z.infer<typeof GepaParetoCandidateEntrySchema>;

export const GepaParetoPartitionKeySchema = z.object({
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  evalPackId: OptimizerIdSchema,
  objectiveSetId: OptimizerIdSchema,
}).strict();
export type GepaParetoPartitionKey = z.infer<typeof GepaParetoPartitionKeySchema>;

export const GepaParetoFrontSchema = z.object({
  frontId: OptimizerIdSchema,
  key: GepaParetoPartitionKeySchema,
  objectives: z.array(GepaParetoObjectiveSchema).min(1),
  candidates: z.array(GepaParetoCandidateEntrySchema),
  dominatedCandidateIds: z.array(OptimizerIdSchema).default([]),
}).strict();
export type GepaParetoFront = z.infer<typeof GepaParetoFrontSchema>;

export type GepaParetoDominance = "left_dominates" | "right_dominates" | "tie" | "non_dominated";

export type BuildGepaParetoFrontsInput = {
  candidates: readonly GepaParetoCandidate[];
};

export const stableGepaObjectiveSetId = (objectives: readonly GepaParetoObjective[]): string => {
  const canonical = normalizeObjectives(objectives);
  const digest = createHash("sha256")
    .update(JSON.stringify(canonical))
    .digest("hex")
    .slice(0, 16);
  return `objective-set.${digest}`;
};

export const compareGepaParetoCandidates = (
  left: GepaParetoCandidate,
  right: GepaParetoCandidate,
  objectives: readonly GepaParetoObjective[],
): GepaParetoDominance => {
  const parsedLeft = candidateForComparison(left);
  const parsedRight = candidateForComparison(right);
  const canonicalObjectives = normalizeObjectives(objectives);

  let leftBetter = false;
  let rightBetter = false;

  for (const objective of canonicalObjectives) {
    const comparison = compareMetricValue(
      metricValue(parsedLeft, objective.metricId),
      metricValue(parsedRight, objective.metricId),
      objective.higherIsBetter,
    );
    if (comparison < 0) {
      leftBetter = true;
    } else if (comparison > 0) {
      rightBetter = true;
    }
  }

  if (leftBetter && !rightBetter) {
    return "left_dominates";
  }
  if (rightBetter && !leftBetter) {
    return "right_dominates";
  }
  if (!leftBetter && !rightBetter) {
    return "tie";
  }
  return "non_dominated";
};

export const buildGepaParetoFronts = (input: BuildGepaParetoFrontsInput): GepaParetoFront[] => {
  const candidates = input.candidates.map((candidate) => GepaParetoCandidateSchema.parse(candidate));
  const partitions = new Map<string, GepaParetoCandidate[]>();

  for (const candidate of candidates) {
    const objectiveSetId = resolveObjectiveSetId(candidate);
    const partitionKey = serializePartitionKey({
      modelProfileId: candidate.modelProfileId,
      codebaseProfileId: candidate.codebaseProfileId,
      evalPackId: candidate.evalPackId,
      objectiveSetId,
    });
    const existing = partitions.get(partitionKey);
    if (existing == null) {
      partitions.set(partitionKey, [candidate]);
    } else {
      existing.push(candidate);
    }
  }

  return [...partitions.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([serializedKey, partitionCandidates]) => {
      const key = deserializePartitionKey(serializedKey);
      const objectives = normalizeObjectives(partitionCandidates.flatMap((candidate) => objectivesFromMetrics(candidate.metrics)));
      const candidatesWithObjectiveSet = partitionCandidates.map((candidate) =>
        entryForCandidate(candidate, key.objectiveSetId, objectives)
      );
      const dominatedCandidateIds = dominatedEntries(candidatesWithObjectiveSet, objectives)
        .map((candidate) => candidate.candidatePatchId)
        .sort((left, right) => left.localeCompare(right));
      const frontCandidates = candidatesWithObjectiveSet
        .filter((candidate) => !dominatedCandidateIds.includes(candidate.candidatePatchId))
        .sort((left, right) => compareEntriesForObjectiveSet(left, right, objectives));

      return GepaParetoFrontSchema.parse({
        frontId: stableFrontId(key),
        key,
        objectives,
        candidates: frontCandidates,
        dominatedCandidateIds,
      });
    });
};

const dominatedEntries = (
  candidates: readonly GepaParetoCandidateEntry[],
  objectives: readonly GepaParetoObjective[],
): GepaParetoCandidateEntry[] =>
  candidates.filter((candidate) =>
    candidates.some((other) =>
      other.candidatePatchId !== candidate.candidatePatchId &&
      compareGepaParetoCandidates(other, candidate, objectives) === "left_dominates"
    )
  );

const entryForCandidate = (
  candidate: GepaParetoCandidate,
  objectiveSetId: string,
  objectives: readonly GepaParetoObjective[],
): GepaParetoCandidateEntry => {
  const parsed = GepaParetoCandidateSchema.parse(candidate);
  const objectiveValues = objectives.map((objective) => {
    const value = metricValue(parsed, objective.metricId);
    return value === undefined
      ? {
          metricId: objective.metricId,
          higherIsBetter: objective.higherIsBetter,
        }
      : {
          metricId: objective.metricId,
          value,
          higherIsBetter: objective.higherIsBetter,
        };
  });

  return GepaParetoCandidateEntrySchema.parse({
    ...parsed,
    objectiveSetId,
    objectiveValues,
  });
};

const compareEntriesForObjectiveSet = (
  left: GepaParetoCandidateEntry,
  right: GepaParetoCandidateEntry,
  objectives: readonly GepaParetoObjective[],
): number => {
  for (const objective of objectives) {
    const comparison = compareMetricValue(
      metricValue(left, objective.metricId),
      metricValue(right, objective.metricId),
      objective.higherIsBetter,
    );
    if (comparison !== 0) {
      return comparison;
    }
  }

  return left.candidatePatchId.localeCompare(right.candidatePatchId) ||
    left.evalResultIds.join("\u0000").localeCompare(right.evalResultIds.join("\u0000")) ||
    left.scorecardIds.join("\u0000").localeCompare(right.scorecardIds.join("\u0000"));
};

const compareMetricValue = (
  left: number | undefined,
  right: number | undefined,
  higherIsBetter: boolean,
): number => {
  if (left === undefined && right === undefined) {
    return 0;
  }
  if (left === undefined) {
    return 1;
  }
  if (right === undefined) {
    return -1;
  }
  if (left === right) {
    return 0;
  }
  return higherIsBetter
    ? right - left
    : left - right;
};

const metricValue = (candidate: GepaParetoCandidate, metricId: string): number | undefined =>
  candidate.metrics.find((metric) => metric.metricId === metricId)?.value;

const candidateForComparison = (candidate: GepaParetoCandidate): GepaParetoCandidate => {
  const input = {
    candidatePatchId: candidate.candidatePatchId,
    modelProfileId: candidate.modelProfileId,
    codebaseProfileId: candidate.codebaseProfileId,
    evalPackId: candidate.evalPackId,
    metrics: candidate.metrics,
    evalResultIds: candidate.evalResultIds,
    scorecardIds: candidate.scorecardIds,
    sourceTraceIds: candidate.sourceTraceIds,
    ...(candidate.objectiveSetId === undefined ? {} : { objectiveSetId: candidate.objectiveSetId }),
    ...(candidate.policyId === undefined ? {} : { policyId: candidate.policyId }),
    ...(candidate.createdAt === undefined ? {} : { createdAt: candidate.createdAt }),
  };
  return GepaParetoCandidateSchema.parse(input);
};

const resolveObjectiveSetId = (candidate: GepaParetoCandidate): string =>
  candidate.objectiveSetId ?? stableGepaObjectiveSetId(objectivesFromMetrics(candidate.metrics));

const objectivesFromMetrics = (metrics: readonly z.infer<typeof ObjectiveMetricSchema>[]): GepaParetoObjective[] =>
  normalizeObjectives(metrics.map((metric) => ({
    metricId: metric.metricId,
    higherIsBetter: metric.higherIsBetter,
  })));

const normalizeObjectives = (objectives: readonly GepaParetoObjective[]): GepaParetoObjective[] => {
  const byMetricId = new Map<string, GepaParetoObjective>();
  for (const objective of objectives) {
    const parsed = GepaParetoObjectiveSchema.parse(objective);
    const existing = byMetricId.get(parsed.metricId);
    if (existing != null && existing.higherIsBetter !== parsed.higherIsBetter) {
      throw new Error(`objective metric direction conflict for ${parsed.metricId}`);
    }
    byMetricId.set(parsed.metricId, parsed);
  }

  const normalized = [...byMetricId.values()].sort((left, right) =>
    left.metricId.localeCompare(right.metricId) || Number(right.higherIsBetter) - Number(left.higherIsBetter)
  );
  if (normalized.length === 0) {
    throw new Error("GEPA Pareto objective set cannot be empty");
  }
  return normalized;
};

const serializePartitionKey = (key: GepaParetoPartitionKey): string =>
  [
    key.modelProfileId,
    key.codebaseProfileId,
    key.evalPackId,
    key.objectiveSetId,
  ].join("\u0000");

const deserializePartitionKey = (serialized: string): GepaParetoPartitionKey => {
  const [modelProfileId, codebaseProfileId, evalPackId, objectiveSetId] = serialized.split("\u0000");
  return GepaParetoPartitionKeySchema.parse({
    modelProfileId,
    codebaseProfileId,
    evalPackId,
    objectiveSetId,
  });
};

const stableFrontId = (key: GepaParetoPartitionKey): string => {
  const digest = createHash("sha256")
    .update(JSON.stringify(key))
    .digest("hex")
    .slice(0, 16);
  return `pareto-front.${digest}`;
};
