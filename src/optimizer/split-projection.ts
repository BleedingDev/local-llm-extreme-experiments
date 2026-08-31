import { createHash } from "node:crypto";

export const OPTIMIZER_PROJECTED_SPLITS = ["train", "dev", "hidden-holdout"] as const;
export type OptimizerProjectedSplit = typeof OPTIMIZER_PROJECTED_SPLITS[number];

export const VISIBLE_OPTIMIZER_PROJECTED_SPLITS = ["train", "dev"] as const satisfies readonly OptimizerProjectedSplit[];
export const HIDDEN_OPTIMIZER_PROJECTED_SPLITS = ["hidden-holdout"] as const satisfies readonly OptimizerProjectedSplit[];

export const DEFAULT_SPLIT_PROJECTION_RATIOS = {
  train: 70,
  dev: 15,
  hiddenHoldout: 15,
} as const;

export type SplitProjectionRatios = {
  train: number;
  dev: number;
  hiddenHoldout: number;
};

export type SplitProjectionMetadata = {
  projectionVersion: string;
  seed: string;
  algorithm: "sha256-threshold.v1";
  sourceId?: string;
  ratios: SplitProjectionRatios;
  normalizedRatios: SplitProjectionRatios;
  splitLabels: readonly OptimizerProjectedSplit[];
  stableIdCount: number;
};

export type SplitProjectionAssignment = {
  stableId: string;
  split: OptimizerProjectedSplit;
  visibility: "visible" | "hidden";
  hash: string;
  score: number;
  ordinal: number;
};

export type SplitProjection = {
  projectionId: string;
  metadata: SplitProjectionMetadata;
  assignments: SplitProjectionAssignment[];
  splits: Record<OptimizerProjectedSplit, string[]>;
  counts: Record<OptimizerProjectedSplit, number>;
  visibleIds: string[];
  hiddenHoldoutIds: string[];
};

export type ProjectStableIdsInput = {
  stableIds: readonly string[];
  seed: string;
  projectionVersion: string;
  sourceId?: string;
  ratios?: Partial<SplitProjectionRatios>;
};

export const OPTIMIZER_EVIDENCE_CONSUMERS = [
  "candidate-generation",
  "prompt-drafting",
  "policy-synthesis",
  "retrieval",
  "failure-clustering",
  "training",
  "development-evaluation",
  "frozen-candidate-hidden-holdout-evaluator",
  "promotion-gate",
  "redacted-aggregate-audit-reporter",
  "post-promotion-monitoring",
  "live-rollout",
] as const;
export type OptimizerEvidenceConsumer = typeof OPTIMIZER_EVIDENCE_CONSUMERS[number];

export type SplitVisibilityCheck = {
  allowed: boolean;
  consumer: string;
  requestedSplits: OptimizerProjectedSplit[];
  allowedSplits: OptimizerProjectedSplit[];
  blockedSplits: OptimizerProjectedSplit[];
  unknownSplitLabels: string[];
  message?: string;
};

export type StableIdDuplicate = {
  stableId: string;
  count: number;
};

export type StableIdDedupResult = {
  stableIds: string[];
  duplicates: StableIdDuplicate[];
};

export type SplitLeakagePair = {
  leftSplit: OptimizerProjectedSplit;
  rightSplit: OptimizerProjectedSplit;
  overlapIds: string[];
  overlapCount: number;
};

export const projectStableIdsToOptimizerSplits = (input: ProjectStableIdsInput): SplitProjection => {
  const seed = normalizeMetadataField(input.seed, "seed");
  const projectionVersion = normalizeMetadataField(input.projectionVersion, "projectionVersion");
  const sourceId = input.sourceId == null ? undefined : normalizeMetadataField(input.sourceId, "sourceId");
  const stableIds = dedupeStableIds(input.stableIds);
  if (stableIds.duplicates.length > 0) {
    throw new Error(`split projection rejected duplicate stable ids: ${stableIds.duplicates.map((duplicate) => duplicate.stableId).join(", ")}`);
  }

  const normalizedStableIds = stableIds.stableIds;
  const ratios = resolveRatios(input.ratios);
  const normalizedRatios = normalizeRatios(ratios);
  const splits = emptySplitGroups();
  const assignments = normalizedStableIds.map((stableId, ordinal) => {
    const hash = projectionHash({ seed, projectionVersion, sourceId, stableId });
    const score = hashToUnitInterval(hash);
    const split = splitForScore(score, normalizedRatios);
    const assignment: SplitProjectionAssignment = {
      stableId,
      split,
      visibility: split === "hidden-holdout" ? "hidden" : "visible",
      hash,
      score,
      ordinal,
    };
    splits[split].push(stableId);
    return assignment;
  });

  const metadata: SplitProjectionMetadata = {
    projectionVersion,
    seed,
    algorithm: "sha256-threshold.v1",
    ...(sourceId == null ? {} : { sourceId }),
    ratios,
    normalizedRatios,
    splitLabels: OPTIMIZER_PROJECTED_SPLITS,
    stableIdCount: normalizedStableIds.length,
  };

  const counts = {
    train: splits.train.length,
    dev: splits.dev.length,
    "hidden-holdout": splits["hidden-holdout"].length,
  };

  return {
    projectionId: stableProjectionId(metadata, normalizedStableIds),
    metadata,
    assignments,
    splits,
    counts,
    visibleIds: [...splits.train, ...splits.dev],
    hiddenHoldoutIds: [...splits["hidden-holdout"]],
  };
};

export const allowedSplitsForOptimizerConsumer = (
  consumer: OptimizerEvidenceConsumer | string,
): OptimizerProjectedSplit[] => {
  switch (consumer) {
    case "candidate-generation":
    case "prompt-drafting":
    case "policy-synthesis":
    case "retrieval":
    case "failure-clustering":
    case "training":
    case "development-evaluation":
      return [...VISIBLE_OPTIMIZER_PROJECTED_SPLITS];
    case "frozen-candidate-hidden-holdout-evaluator":
      return [...HIDDEN_OPTIMIZER_PROJECTED_SPLITS];
    case "promotion-gate":
    case "redacted-aggregate-audit-reporter":
    case "post-promotion-monitoring":
    case "live-rollout":
      return [...OPTIMIZER_PROJECTED_SPLITS];
    default:
      return [];
  }
};

export const validateOptimizerSplitVisibility = (
  consumer: OptimizerEvidenceConsumer | string,
  requestedSplits: readonly string[],
): SplitVisibilityCheck => {
  const unknownSplitLabels = unknownSplits(requestedSplits);
  const normalizedRequestedSplits = orderedSplits(requestedSplits.filter(isOptimizerProjectedSplit));
  const allowedSplits = allowedSplitsForOptimizerConsumer(consumer);
  const allowed = new Set(allowedSplits);
  const blockedSplits = normalizedRequestedSplits.filter((split) => !allowed.has(split));
  const allowedRequest = blockedSplits.length === 0 &&
    unknownSplitLabels.length === 0 &&
    normalizedRequestedSplits.length > 0;
  return {
    allowed: allowedRequest,
    consumer,
    requestedSplits: normalizedRequestedSplits,
    allowedSplits,
    blockedSplits,
    unknownSplitLabels,
    ...(allowedRequest
      ? {}
      : { message: visibilityFailureMessage(consumer, allowedSplits, blockedSplits, unknownSplitLabels) }),
  };
};

export const assertOptimizerSplitVisibilityAllowed = (
  consumer: OptimizerEvidenceConsumer | string,
  requestedSplits: readonly string[],
): SplitVisibilityCheck => {
  const visibility = validateOptimizerSplitVisibility(consumer, requestedSplits);
  if (!visibility.allowed) {
    throw new Error(visibility.message ?? `optimizer evidence consumer ${consumer} requested no allowed split labels`);
  }
  return visibility;
};

export const projectedIdsForOptimizerConsumer = (
  projection: SplitProjection,
  consumer: OptimizerEvidenceConsumer | string,
  requestedSplits: readonly string[] = allowedSplitsForOptimizerConsumer(consumer),
): string[] => {
  const visibility = assertOptimizerSplitVisibilityAllowed(consumer, requestedSplits);
  return visibility.requestedSplits.flatMap((split) => projection.splits[split]);
};

export const dedupeStableIds = (stableIds: readonly string[]): StableIdDedupResult => {
  const counts = new Map<string, number>();
  for (const rawStableId of stableIds) {
    const stableId = normalizeStableId(rawStableId);
    counts.set(stableId, (counts.get(stableId) ?? 0) + 1);
  }

  const duplicates = [...counts.entries()]
    .filter(([, count]) => count > 1)
    .map(([stableId, count]) => ({ stableId, count }))
    .sort(compareDuplicates);

  return {
    stableIds: [...counts.keys()].sort(compareStableIds),
    duplicates,
  };
};

export const findDuplicateStableIds = (stableIds: readonly string[]): StableIdDuplicate[] =>
  dedupeStableIds(stableIds).duplicates;

export const assertNoDuplicateStableIds = (stableIds: readonly string[]): void => {
  const duplicates = findDuplicateStableIds(stableIds);
  if (duplicates.length > 0) {
    throw new Error(`duplicate stable ids detected: ${duplicates.map((duplicate) => duplicate.stableId).join(", ")}`);
  }
};

export const findSplitLeakage = (
  splits: Partial<Record<OptimizerProjectedSplit, readonly string[]>>,
): SplitLeakagePair[] => {
  const splitSets = new Map<OptimizerProjectedSplit, Set<string>>();
  for (const split of OPTIMIZER_PROJECTED_SPLITS) {
    splitSets.set(split, new Set(dedupeStableIds(splits[split] ?? []).stableIds));
  }

  const leakagePairs: SplitLeakagePair[] = [];
  for (let leftIndex = 0; leftIndex < OPTIMIZER_PROJECTED_SPLITS.length; leftIndex += 1) {
    const leftSplit = OPTIMIZER_PROJECTED_SPLITS[leftIndex];
    if (leftSplit == null) continue;
    const leftSet = splitSets.get(leftSplit) ?? new Set<string>();
    for (let rightIndex = leftIndex + 1; rightIndex < OPTIMIZER_PROJECTED_SPLITS.length; rightIndex += 1) {
      const rightSplit = OPTIMIZER_PROJECTED_SPLITS[rightIndex];
      if (rightSplit == null) continue;
      const rightSet = splitSets.get(rightSplit) ?? new Set<string>();
      const overlapIds = [...leftSet].filter((stableId) => rightSet.has(stableId)).sort(compareStableIds);
      if (overlapIds.length > 0) {
        leakagePairs.push({
          leftSplit,
          rightSplit,
          overlapIds,
          overlapCount: overlapIds.length,
        });
      }
    }
  }
  return leakagePairs;
};

export const assertNoSplitLeakage = (
  splits: Partial<Record<OptimizerProjectedSplit, readonly string[]>>,
): void => {
  const leakage = findSplitLeakage(splits);
  if (leakage.length > 0) {
    const details = leakage
      .map((pair) => `${pair.leftSplit}/${pair.rightSplit}: ${pair.overlapIds.join(", ")}`)
      .join("; ");
    throw new Error(`split leakage detected: ${details}`);
  }
};

const emptySplitGroups = (): Record<OptimizerProjectedSplit, string[]> => ({
  train: [],
  dev: [],
  "hidden-holdout": [],
});

const resolveRatios = (ratios: ProjectStableIdsInput["ratios"]): SplitProjectionRatios => ({
  train: ratioValue(ratios?.train, DEFAULT_SPLIT_PROJECTION_RATIOS.train, "train"),
  dev: ratioValue(ratios?.dev, DEFAULT_SPLIT_PROJECTION_RATIOS.dev, "dev"),
  hiddenHoldout: ratioValue(
    ratios?.hiddenHoldout,
    DEFAULT_SPLIT_PROJECTION_RATIOS.hiddenHoldout,
    "hiddenHoldout",
  ),
});

const ratioValue = (value: number | undefined, fallback: number, field: keyof SplitProjectionRatios): number => {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < 0) {
    throw new Error(`split projection ratio ${field} must be a non-negative finite number`);
  }
  return resolved;
};

const normalizeRatios = (ratios: SplitProjectionRatios): SplitProjectionRatios => {
  const total = ratios.train + ratios.dev + ratios.hiddenHoldout;
  if (total <= 0) {
    throw new Error("split projection ratios must include at least one non-zero bucket");
  }
  return {
    train: ratios.train / total,
    dev: ratios.dev / total,
    hiddenHoldout: ratios.hiddenHoldout / total,
  };
};

const splitForScore = (score: number, ratios: SplitProjectionRatios): OptimizerProjectedSplit => {
  if (score < ratios.train) return "train";
  if (score < ratios.train + ratios.dev) return "dev";
  return "hidden-holdout";
};

const projectionHash = (input: {
  seed: string;
  projectionVersion: string;
  sourceId: string | undefined;
  stableId: string;
}): string =>
  createHash("sha256")
    .update(JSON.stringify({
      algorithm: "sha256-threshold.v1",
      seed: input.seed,
      projectionVersion: input.projectionVersion,
      sourceId: input.sourceId ?? null,
      stableId: input.stableId,
    }))
    .digest("hex");

const hashToUnitInterval = (hash: string): number =>
  Number.parseInt(hash.slice(0, 13), 16) / 0x10000000000000;

const stableProjectionId = (metadata: SplitProjectionMetadata, stableIds: readonly string[]): string => {
  const digest = createHash("sha256")
    .update(JSON.stringify({
      metadata,
      stableIds,
    }))
    .digest("hex")
    .slice(0, 16);
  return `optimizer.split-projection.${sanitizeIdPart(metadata.projectionVersion)}.${digest}`;
};

const orderedSplits = (splits: readonly OptimizerProjectedSplit[]): OptimizerProjectedSplit[] => {
  const requested = new Set(splits);
  return OPTIMIZER_PROJECTED_SPLITS.filter((split) => requested.has(split));
};

const visibilityFailureMessage = (
  consumer: OptimizerEvidenceConsumer | string,
  allowedSplits: readonly OptimizerProjectedSplit[],
  blockedSplits: readonly OptimizerProjectedSplit[],
  unknownSplitLabels: readonly string[],
): string => {
  if (unknownSplitLabels.length > 0) {
    return `unknown optimizer split labels requested by ${consumer}: ${unknownSplitLabels.join(", ")}`;
  }
  if (allowedSplits.length === 0) {
    return `unknown optimizer evidence consumer cannot read split labels: ${consumer}`;
  }
  if (blockedSplits.length > 0) {
    return `optimizer evidence consumer ${consumer} cannot read split labels: ${blockedSplits.join(", ")}`;
  }
  return `optimizer evidence consumer ${consumer} requested no allowed split labels`;
};

const unknownSplits = (splits: readonly string[]): string[] =>
  [...new Set(splits.filter((split) => !isOptimizerProjectedSplit(split)))].sort(compareStableIds);

const isOptimizerProjectedSplit = (split: string): split is OptimizerProjectedSplit =>
  OPTIMIZER_PROJECTED_SPLITS.includes(split as OptimizerProjectedSplit);

const normalizeStableId = (stableId: string): string => {
  const normalized = stableId.trim();
  if (normalized.length === 0) {
    throw new Error("stable id must be a non-empty string");
  }
  return normalized;
};

const normalizeMetadataField = (value: string, field: string): string => {
  const normalized = value.trim();
  if (normalized.length === 0) {
    throw new Error(`split projection ${field} must be a non-empty string`);
  }
  return normalized;
};

const sanitizeIdPart = (value: string): string => {
  const sanitized = value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "");
  return sanitized.length > 0 ? sanitized : "unknown";
};

const compareStableIds = (left: string, right: string): number => left.localeCompare(right);

const compareDuplicates = (left: StableIdDuplicate, right: StableIdDuplicate): number =>
  left.stableId.localeCompare(right.stableId);
