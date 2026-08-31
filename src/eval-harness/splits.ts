import {
  EvalCaseSchema,
  type EvalCase,
  type EvalSplit,
} from "./types";

export const EVAL_SPLITS = ["train", "dev", "holdout"] as const satisfies readonly EvalSplit[];
export const VISIBLE_EVAL_SPLITS = ["train", "dev"] as const satisfies readonly EvalSplit[];
export const HIDDEN_EVAL_SPLITS = ["holdout"] as const satisfies readonly EvalSplit[];

const splitOrder = new Map<EvalSplit, number>(
  EVAL_SPLITS.map((split, index) => [split, index]),
);

export type EvalSplitGroups = Record<EvalSplit, EvalCase[]>;

export type EvalSplitSummary = {
  split: EvalSplit;
  count: number;
  evalCaseIds: string[];
  fixtureWorkspaceIds: string[];
  tags: string[];
};

export type EvalSplitPack = {
  splits: EvalSplit[];
  hidden: boolean;
  cases: EvalCase[];
  evalCaseIds: string[];
  fixtureWorkspaceIds: string[];
};

export type CandidateTrainingInput = {
  evalCaseIds?: readonly string[];
  fixtureWorkspaceIds?: readonly string[];
  evalCases?: readonly EvalCase[];
};

export type CandidateTrainingSplitValidation = {
  vetoed: boolean;
  blockedEvalCaseIds: string[];
  blockedFixtureWorkspaceIds: string[];
  hiddenHoldoutEvalCaseIds: string[];
  hiddenHoldoutFixtureWorkspaceIds: string[];
  visibleEvalCaseIds: string[];
  message?: string;
};

export const canonicalEvalCases = (cases: readonly EvalCase[]): EvalCase[] => {
  const parsedCases = cases.map((evalCase) => EvalCaseSchema.parse(evalCase));
  assertUniqueEvalCaseIds(parsedCases);
  return [...parsedCases].sort(compareEvalCases);
};

export const groupEvalCasesBySplit = (cases: readonly EvalCase[]): EvalSplitGroups => {
  const groups = emptySplitGroups();
  for (const evalCase of canonicalEvalCases(cases)) {
    groups[evalCase.split].push(evalCase);
  }
  return groups;
};

export const summarizeEvalSplits = (cases: readonly EvalCase[]): EvalSplitSummary[] =>
  EVAL_SPLITS.map((split) => {
    const splitCases = filterEvalCasesBySplit(cases, [split]);
    return {
      split,
      count: splitCases.length,
      evalCaseIds: splitCases.map((evalCase) => evalCase.evalCaseId),
      fixtureWorkspaceIds: splitCases.map((evalCase) => evalCase.fixtureWorkspace.fixtureWorkspaceId),
      tags: uniqueSorted(splitCases.flatMap((evalCase) => evalCase.tags)),
    };
  });

export const filterEvalCasesBySplit = (
  cases: readonly EvalCase[],
  splits: readonly EvalSplit[],
): EvalCase[] => {
  const requestedSplits = new Set(splits);
  return canonicalEvalCases(cases).filter((evalCase) => requestedSplits.has(evalCase.split));
};

export const trainEvalCases = (cases: readonly EvalCase[]): EvalCase[] =>
  filterEvalCasesBySplit(cases, ["train"]);

export const devEvalCases = (cases: readonly EvalCase[]): EvalCase[] =>
  filterEvalCasesBySplit(cases, ["dev"]);

export const holdoutEvalCases = (cases: readonly EvalCase[]): EvalCase[] =>
  filterEvalCasesBySplit(cases, ["holdout"]);

export const visibleEvalCases = (cases: readonly EvalCase[]): EvalCase[] =>
  filterEvalCasesBySplit(cases, VISIBLE_EVAL_SPLITS);

export const createEvalSplitPack = (
  cases: readonly EvalCase[],
  splits: readonly EvalSplit[],
): EvalSplitPack => {
  const splitCases = filterEvalCasesBySplit(cases, splits);
  const normalizedSplits = uniqueSplits(splits);
  return {
    splits: normalizedSplits,
    hidden: normalizedSplits.every((split) => HIDDEN_EVAL_SPLITS.includes(split as "holdout")),
    cases: splitCases,
    evalCaseIds: splitCases.map((evalCase) => evalCase.evalCaseId),
    fixtureWorkspaceIds: splitCases.map((evalCase) => evalCase.fixtureWorkspace.fixtureWorkspaceId),
  };
};

export const createVisibleEvalPack = (cases: readonly EvalCase[]): EvalSplitPack =>
  createEvalSplitPack(cases, VISIBLE_EVAL_SPLITS);

export const createHoldoutEvalPack = (cases: readonly EvalCase[]): EvalSplitPack =>
  createEvalSplitPack(cases, HIDDEN_EVAL_SPLITS);

export const validateCandidateTrainingInput = (
  input: CandidateTrainingInput,
  suiteCases: readonly EvalCase[],
): CandidateTrainingSplitValidation => {
  const canonicalCases = canonicalEvalCases(suiteCases);
  const holdoutCases = holdoutEvalCases(canonicalCases);
  const visibleCases = visibleEvalCases(canonicalCases);
  const holdoutEvalCaseIds = new Set(holdoutCases.map((evalCase) => evalCase.evalCaseId));
  const holdoutFixtureWorkspaceIds = new Set(
    holdoutCases.map((evalCase) => evalCase.fixtureWorkspace.fixtureWorkspaceId),
  );
  const inputEvalCaseIds = [
    ...(input.evalCaseIds ?? []),
    ...(input.evalCases ?? []).map((evalCase) => evalCase.evalCaseId),
  ];
  const inputFixtureWorkspaceIds = [
    ...(input.fixtureWorkspaceIds ?? []),
    ...(input.evalCases ?? []).map((evalCase) => evalCase.fixtureWorkspace.fixtureWorkspaceId),
  ];
  const blockedEvalCaseIds = uniqueSorted(
    inputEvalCaseIds.filter((evalCaseId) => holdoutEvalCaseIds.has(evalCaseId)),
  );
  const blockedFixtureWorkspaceIds = uniqueSorted(
    inputFixtureWorkspaceIds.filter((fixtureWorkspaceId) =>
      holdoutFixtureWorkspaceIds.has(fixtureWorkspaceId),
    ),
  );
  const vetoed = blockedEvalCaseIds.length > 0 || blockedFixtureWorkspaceIds.length > 0;

  return {
    vetoed,
    blockedEvalCaseIds,
    blockedFixtureWorkspaceIds,
    hiddenHoldoutEvalCaseIds: [...holdoutEvalCaseIds].sort((left, right) => left.localeCompare(right)),
    hiddenHoldoutFixtureWorkspaceIds: [...holdoutFixtureWorkspaceIds].sort((left, right) =>
      left.localeCompare(right),
    ),
    visibleEvalCaseIds: visibleCases.map((evalCase) => evalCase.evalCaseId),
    ...(vetoed
      ? {
          message:
            "candidate training input includes hidden holdout eval fixtures and must not be promoted",
        }
      : {}),
  };
};

export const assertCandidateTrainingInputAllowed = (
  input: CandidateTrainingInput,
  suiteCases: readonly EvalCase[],
): CandidateTrainingSplitValidation => {
  const validation = validateCandidateTrainingInput(input, suiteCases);
  if (validation.vetoed) {
    const blockedIds = [
      ...validation.blockedEvalCaseIds,
      ...validation.blockedFixtureWorkspaceIds,
    ].join(", ");
    throw new Error(`candidate promotion vetoed: hidden holdout fixtures were used for training (${blockedIds})`);
  }
  return validation;
};

const emptySplitGroups = (): EvalSplitGroups => ({
  train: [],
  dev: [],
  holdout: [],
});

const compareEvalCases = (left: EvalCase, right: EvalCase): number => {
  const splitComparison = (splitOrder.get(left.split) ?? 0) - (splitOrder.get(right.split) ?? 0);
  return splitComparison === 0
    ? left.evalCaseId.localeCompare(right.evalCaseId)
    : splitComparison;
};

const assertUniqueEvalCaseIds = (cases: readonly EvalCase[]) => {
  const seen = new Set<string>();
  for (const evalCase of cases) {
    if (seen.has(evalCase.evalCaseId)) {
      throw new Error(`duplicate eval case id: ${evalCase.evalCaseId}`);
    }
    seen.add(evalCase.evalCaseId);
  }
};

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const uniqueSplits = (splits: readonly EvalSplit[]): EvalSplit[] => {
  const requestedSplits = new Set(splits);
  return EVAL_SPLITS.filter((split) => requestedSplits.has(split));
};
