import {
  EvalRunResultSchema,
  EvalScorecardSchema,
  type ComparisonRunMetadata,
  type CriticalRegression,
  type EvalRunResult,
  type EvalScorecard,
  type EvalSplit,
  type ObjectiveMetric,
} from "./types";

const SCORECARD_SCHEMA_VERSION = "eval-scorecard.v1";
const AGGREGATE_WEIGHTS = {
  assertions: 0.5,
  status: 0.2,
  commandAssertions: 0.1,
  protectedPathAssertions: 0.1,
  objectiveMetrics: 0.1,
} as const;

export type CreateEvalScorecardInput = {
  scorecardId: string;
  evalSuiteId: string;
  split: EvalSplit;
  baseline: ComparisonRunMetadata;
  candidate: ComparisonRunMetadata;
  baselineResults: readonly EvalRunResult[];
  candidateResults: readonly EvalRunResult[];
  schemaVersion?: string;
  judgedMetrics?: readonly ObjectiveMetric[];
  createdAt?: string;
};

type RunPair = {
  evalCaseId: string;
  baseline: EvalRunResult;
  candidate: EvalRunResult;
};

type AggregateSignals = {
  assertionPassRate: number;
  statusPassRate: number;
  commandAssertionPassRate: number;
  protectedPathAssertionPassRate: number;
  objectiveMetricScore: number;
};

export const createEvalScorecard = (input: CreateEvalScorecardInput): EvalScorecard => {
  const baselineResults = input.baselineResults.map((result) => EvalRunResultSchema.parse(result));
  const candidateResults = input.candidateResults.map((result) => EvalRunResultSchema.parse(result));
  const pairs = pairRunResults(baselineResults, candidateResults);
  const baselineSignals = {
    ...aggregateSignals(pairs.map((pair) => pair.baseline)),
    objectiveMetricScore: baselineObjectiveMetricScore(pairs, input.judgedMetrics ?? []),
  };
  const candidateSignals = aggregateSignals(
    pairs.map((pair) => pair.candidate),
    pairs,
    input.judgedMetrics ?? [],
  );
  const baselineScore = aggregateScore(baselineSignals);
  const candidateScore = aggregateScore(candidateSignals);
  const regressions = criticalRegressions(pairs);
  const objectiveMetrics = [
    ...summaryObjectiveMetrics({
      baselineSignals,
      candidateSignals,
      baselineScore,
      candidateScore,
    }),
    ...objectiveMetricComparisons(pairs),
    ...judgedObjectiveMetrics(input.judgedMetrics ?? []),
  ];

  return EvalScorecardSchema.parse({
    scorecardId: input.scorecardId,
    schemaVersion: input.schemaVersion ?? SCORECARD_SCHEMA_VERSION,
    evalSuiteId: input.evalSuiteId,
    split: input.split,
    baseline: input.baseline,
    candidate: input.candidate,
    runResults: [...baselineResults, ...candidateResults],
    objectiveMetrics,
    aggregateScore: candidateScore,
    passed: regressions.length === 0 && candidateScore >= baselineScore,
    criticalRegressionVeto: {
      vetoed: regressions.length > 0,
      regressions,
    },
    createdAt: input.createdAt ?? new Date().toISOString(),
  });
};

const pairRunResults = (
  baselineResults: readonly EvalRunResult[],
  candidateResults: readonly EvalRunResult[],
): RunPair[] => {
  if (baselineResults.length === 0 || candidateResults.length === 0) {
    throw new Error("scorecards require at least one baseline and one candidate run result");
  }

  const baselineByCase = resultMapByEvalCase(baselineResults, "baseline");
  const candidateByCase = resultMapByEvalCase(candidateResults, "candidate");
  const evalCaseIds = [...baselineByCase.keys()].sort((left, right) => left.localeCompare(right));

  if (evalCaseIds.length !== candidateByCase.size) {
    throw new Error("baseline and candidate run results must cover the same eval cases");
  }

  return evalCaseIds.map((evalCaseId) => {
    const baseline = baselineByCase.get(evalCaseId);
    const candidate = candidateByCase.get(evalCaseId);
    if (baseline === undefined || candidate === undefined) {
      throw new Error(`missing paired run result for eval case ${evalCaseId}`);
    }
    return { evalCaseId, baseline, candidate };
  });
};

const resultMapByEvalCase = (
  results: readonly EvalRunResult[],
  expectedRole: "baseline" | "candidate",
): Map<string, EvalRunResult> => {
  const byCase = new Map<string, EvalRunResult>();
  for (const result of results) {
    if (result.runRole !== expectedRole) {
      throw new Error(`expected ${expectedRole} run result, received ${result.runRole}`);
    }
    if (byCase.has(result.evalCaseId)) {
      throw new Error(`duplicate ${expectedRole} run result for eval case ${result.evalCaseId}`);
    }
    byCase.set(result.evalCaseId, result);
  }
  return byCase;
};

const aggregateSignals = (
  results: readonly EvalRunResult[],
  pairs: readonly RunPair[] = [],
  judgedMetrics: readonly ObjectiveMetric[] = [],
): AggregateSignals => ({
  assertionPassRate: passRate(results.flatMap((result) => deterministicAssertions(result))),
  statusPassRate: passRate(results.map((result) => result.status === "passed")),
  commandAssertionPassRate: passRate(
    results.flatMap((result) =>
      deterministicAssertions(result)
        .filter((assertion) => assertion.assertionKind === "command_exit_code")
        .map((assertion) => assertion.passed),
    ),
    1,
  ),
  protectedPathAssertionPassRate: passRate(
    results.flatMap((result) =>
      deterministicAssertions(result)
        .filter((assertion) => assertion.assertionKind === "no_forbidden_path_changed")
        .map((assertion) => assertion.passed),
    ),
    1,
  ),
  objectiveMetricScore: objectiveMetricScore(pairs, judgedMetrics),
});

const aggregateScore = (signals: AggregateSignals): number =>
  clamp01(
    signals.assertionPassRate * AGGREGATE_WEIGHTS.assertions +
      signals.statusPassRate * AGGREGATE_WEIGHTS.status +
      signals.commandAssertionPassRate * AGGREGATE_WEIGHTS.commandAssertions +
      signals.protectedPathAssertionPassRate * AGGREGATE_WEIGHTS.protectedPathAssertions +
      signals.objectiveMetricScore * AGGREGATE_WEIGHTS.objectiveMetrics,
  );

const deterministicAssertions = (result: EvalRunResult): EvalRunResult["assertionResults"] =>
  result.assertionResults.filter((assertion) => assertion.assertionKind !== "llm_judge_min_score");

const passRate = (
  values: readonly boolean[] | readonly { passed: boolean }[],
  emptyValue = 0,
): number => {
  if (values.length === 0) {
    return emptyValue;
  }

  const passed = values.filter((value) => typeof value === "boolean" ? value : value.passed).length;
  return passed / values.length;
};

const objectiveMetricScore = (
  pairs: readonly RunPair[],
  judgedMetrics: readonly ObjectiveMetric[],
): number => {
  const scores = pairs.flatMap((pair) =>
    objectiveMetricPairs(pair).map(({ baseline, candidate }) =>
      objectiveComparisonScore(baseline.value, candidate.value, candidate.higherIsBetter),
    ),
  );
  scores.push(...judgedMetrics.map((metric) => normalizedMetricValue(metric)));
  return scores.length === 0 ? 1 : average(scores);
};

const objectiveMetricPairs = (pair: RunPair): { baseline: ObjectiveMetric; candidate: ObjectiveMetric }[] => {
  const baselineMetrics = new Map(pair.baseline.objectiveMetrics.map((metric) => [metric.metricId, metric]));
  return pair.candidate.objectiveMetrics.flatMap((candidate) => {
    const baseline = baselineMetrics.get(candidate.metricId);
    return baseline === undefined ? [] : [{ baseline, candidate }];
  });
};

const objectiveComparisonScore = (
  baselineValue: number,
  candidateValue: number,
  higherIsBetter = true,
): number => {
  const directionalDelta = higherIsBetter
    ? candidateValue - baselineValue
    : baselineValue - candidateValue;

  const scale = Math.max(Math.abs(baselineValue), 1);
  return clamp01(0.5 + (directionalDelta / scale) * 0.5);
};

const baselineObjectiveMetricScore = (
  pairs: readonly RunPair[],
  judgedMetrics: readonly ObjectiveMetric[],
): number => {
  const hasPairedMetrics = pairs.some((pair) => objectiveMetricPairs(pair).length > 0);
  return hasPairedMetrics || judgedMetrics.length > 0 ? 0.5 : 1;
};

const criticalRegressions = (pairs: readonly RunPair[]): CriticalRegression[] => {
  const regressions: CriticalRegression[] = [];
  for (const pair of pairs) {
    if (pair.baseline.status === "passed" && pair.candidate.status !== "passed") {
      regressions.push({
        regressionId: `regression.${pair.evalCaseId}.status`,
        evalCaseId: pair.evalCaseId,
        reason: `Candidate status regressed from ${pair.baseline.status} to ${pair.candidate.status}.`,
        baselineStatus: pair.baseline.status,
        candidateStatus: pair.candidate.status,
        blocksPromotion: true,
      });
    }

    const baselineAssertions = new Map(
      pair.baseline.assertionResults.map((assertion) => [assertion.assertionId, assertion]),
    );
    for (const candidateAssertion of pair.candidate.assertionResults) {
      const baselineAssertion = baselineAssertions.get(candidateAssertion.assertionId);
      if (baselineAssertion?.passed !== true || candidateAssertion.passed) {
        continue;
      }
      if (candidateAssertion.severity !== "critical" && candidateAssertion.assertionKind !== "no_forbidden_path_changed") {
        continue;
      }
      regressions.push({
        regressionId: `regression.${pair.evalCaseId}.${candidateAssertion.assertionId}`,
        evalCaseId: pair.evalCaseId,
        assertionId: candidateAssertion.assertionId,
        reason: candidateAssertion.message ?? `Candidate failed critical assertion ${candidateAssertion.assertionId}.`,
        baselineStatus: pair.baseline.status,
        candidateStatus: pair.candidate.status,
        blocksPromotion: true,
      });
    }

    for (const { baseline, candidate } of objectiveMetricPairs(pair)) {
      if (
        candidate.metricId === "protected-path-change-count" &&
        candidate.value > baseline.value &&
        candidate.value > 0
      ) {
        regressions.push({
          regressionId: `regression.${pair.evalCaseId}.${candidate.metricId}`,
          evalCaseId: pair.evalCaseId,
          metricId: candidate.metricId,
          reason: "Candidate introduced protected path changes.",
          baselineStatus: pair.baseline.status,
          candidateStatus: pair.candidate.status,
          blocksPromotion: true,
        });
      }
    }
  }

  return regressions;
};

const summaryObjectiveMetrics = (input: {
  baselineSignals: AggregateSignals;
  candidateSignals: AggregateSignals;
  baselineScore: number;
  candidateScore: number;
}): ObjectiveMetric[] => [
  comparisonMetric({
    metricId: "assertion-pass-rate",
    name: "Assertion pass rate",
    unit: "ratio",
    baselineValue: input.baselineSignals.assertionPassRate,
    candidateValue: input.candidateSignals.assertionPassRate,
  }),
  comparisonMetric({
    metricId: "run-status-pass-rate",
    name: "Run status pass rate",
    unit: "ratio",
    baselineValue: input.baselineSignals.statusPassRate,
    candidateValue: input.candidateSignals.statusPassRate,
  }),
  comparisonMetric({
    metricId: "command-assertion-pass-rate",
    name: "Command assertion pass rate",
    unit: "ratio",
    baselineValue: input.baselineSignals.commandAssertionPassRate,
    candidateValue: input.candidateSignals.commandAssertionPassRate,
  }),
  comparisonMetric({
    metricId: "protected-path-assertion-pass-rate",
    name: "Protected path assertion pass rate",
    unit: "ratio",
    baselineValue: input.baselineSignals.protectedPathAssertionPassRate,
    candidateValue: input.candidateSignals.protectedPathAssertionPassRate,
  }),
  comparisonMetric({
    metricId: "objective-metric-score",
    name: "Objective metric score",
    unit: "score",
    baselineValue: input.baselineSignals.objectiveMetricScore,
    candidateValue: input.candidateSignals.objectiveMetricScore,
  }),
  comparisonMetric({
    metricId: "aggregate-score-delta",
    name: "Aggregate score delta",
    unit: "score",
    baselineValue: input.baselineScore,
    candidateValue: input.candidateScore,
  }),
];

const objectiveMetricComparisons = (pairs: readonly RunPair[]): ObjectiveMetric[] =>
  pairs.flatMap((pair) =>
    objectiveMetricPairs(pair).map(({ baseline, candidate }) =>
      comparisonMetric({
        metricId: `${candidate.metricId}.${pair.evalCaseId}`,
        name: `${candidate.name} (${pair.evalCaseId})`,
        unit: candidate.unit,
        higherIsBetter: candidate.higherIsBetter,
        baselineValue: baseline.value,
        candidateValue: candidate.value,
        ...optionalThreshold(candidate.threshold ?? baseline.threshold),
      }),
    ),
  );

const judgedObjectiveMetrics = (metrics: readonly ObjectiveMetric[]): ObjectiveMetric[] =>
  metrics.map((metric) => ({
    ...metric,
    metricId: `judged.${metric.metricId}`,
    name: `Judged ${metric.name}`,
  }));

const comparisonMetric = (input: {
  metricId: string;
  name: string;
  unit: ObjectiveMetric["unit"];
  baselineValue: number;
  candidateValue: number;
  higherIsBetter?: boolean;
  threshold?: number;
}): ObjectiveMetric => {
  const higherIsBetter = input.higherIsBetter ?? true;
  const delta = higherIsBetter
    ? input.candidateValue - input.baselineValue
    : input.baselineValue - input.candidateValue;
  return {
    metricId: input.metricId,
    name: input.name,
    value: input.candidateValue,
    unit: input.unit,
    higherIsBetter,
    baselineValue: input.baselineValue,
    candidateValue: input.candidateValue,
    delta,
    ...(input.threshold === undefined ? {} : { threshold: input.threshold }),
  };
};

const optionalThreshold = (threshold: number | undefined): { threshold?: number } =>
  threshold === undefined ? {} : { threshold };

const normalizedMetricValue = (metric: ObjectiveMetric): number => {
  if (metric.unit === "score" || metric.unit === "ratio") {
    return clamp01(metric.value);
  }
  return metric.threshold === undefined
    ? 1
    : objectiveComparisonScore(metric.threshold, metric.value, metric.higherIsBetter);
};

const average = (values: readonly number[]): number =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));
