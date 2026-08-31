import { createHash } from "node:crypto";
import type { EvalSplit } from "../eval-harness/types";
import type { ReplayEvalCaseSkeleton, ReplayObservedFailure } from "../replay";
import type { SourceAdapterFailureKind } from "./failures";

export type AdapterBehaviorObservedSystem = "claude" | "codex" | "bag" | "unknown";

export type AdapterBehaviorComparisonStatus =
  | "needs_bag_run"
  | "compared";

export type AdapterBehaviorDimensionStatus =
  | "matches_observed"
  | "differs_from_observed"
  | "observed_only"
  | "needs_bag_run";

export type AdapterBehaviorBagOutcomeStatus =
  | "succeeded"
  | "failed"
  | "error"
  | "timeout"
  | "cancelled"
  | "permission_denied"
  | "inconclusive";

export type AdapterBehaviorFailureCategory =
  | ReplayObservedFailure["failureKind"]
  | "none";

export type AdapterBehaviorManifestSession = {
  sourceType: string;
  sourceSessionId?: string;
  split: EvalSplit;
  captureId: string;
  evalCaseId: string;
  replayCasePath?: string;
};

export type AdapterBehaviorManifest = {
  schemaVersion?: string;
  exportedSessions: readonly AdapterBehaviorManifestSession[];
};

export type AdapterBehaviorFailureObservation = {
  failureKind: AdapterBehaviorFailureCategory;
  errorCode?: string;
  status?: string;
};

export type BleedingAgentBehaviorOutcome = {
  evalCaseId: string;
  captureId?: string;
  policyId: string;
  status: AdapterBehaviorBagOutcomeStatus;
  failureKinds?: readonly string[];
  failures?: readonly AdapterBehaviorFailureObservation[];
  toolFailures?: number;
  terminalFailures?: number;
  editFailures?: number;
  fileReadFailures?: number;
  permissionEvents?: number;
  cancellationEvents?: number;
  sourceType?: string;
  sessionKind?: string;
  split?: EvalSplit;
  notes?: readonly string[];
};

export type AdapterBehaviorObservedBaseline = {
  role: "observed_baseline";
  gold: false;
  sourceSystem: AdapterBehaviorObservedSystem;
  sourceType: string;
  sessionKind: string;
  captureId: string;
  evalCaseId: string;
  split: EvalSplit;
  sourceSessionId?: string;
};

export type AdapterBehaviorBagRunPlaceholder = {
  policyId: string | null;
  status: AdapterBehaviorBagOutcomeStatus | "needs_bag_run";
  outcomeSource: "bleeding_agent_policy";
  notes: string[];
};

export type AdapterBehaviorDimension<T> = {
  observed: T;
  bag?: T;
  status: AdapterBehaviorDimensionStatus;
};

export type AdapterBehaviorComparisonDimensions = {
  failureKinds: AdapterBehaviorDimension<string[]>;
  failureCategories: AdapterBehaviorDimension<AdapterBehaviorFailureCategory[]>;
  toolFailures: AdapterBehaviorDimension<number>;
  terminalFailures: AdapterBehaviorDimension<number>;
  editFailures: AdapterBehaviorDimension<number>;
  fileReadFailures: AdapterBehaviorDimension<number>;
  permissionEvents: AdapterBehaviorDimension<number>;
  cancellationEvents: AdapterBehaviorDimension<number>;
  sourceKind: AdapterBehaviorDimension<string>;
  sessionKind: AdapterBehaviorDimension<string>;
  split: AdapterBehaviorDimension<EvalSplit>;
  policyOutcome: AdapterBehaviorDimension<string>;
};

export type AdapterBehaviorComparisonScorecard = {
  scorecardId: string;
  schemaVersion: "adapter-behavior-comparison-scorecard.v1";
  evalCaseId: string;
  captureId: string;
  split: EvalSplit;
  baseline: AdapterBehaviorObservedBaseline;
  bag: AdapterBehaviorBagRunPlaceholder;
  gold: false;
  comparisonStatus: AdapterBehaviorComparisonStatus;
  dimensions: AdapterBehaviorComparisonDimensions;
  alignment: {
    comparableDimensionCount: number;
    matchedDimensionCount: number;
    score?: number;
  };
  notes: string[];
};

export type CreateAdapterBehaviorComparisonScorecardsInput = {
  manifest?: AdapterBehaviorManifest;
  replayCases: readonly ReplayEvalCaseSkeleton[];
  bagOutcomes?: readonly BleedingAgentBehaviorOutcome[];
  defaultPolicyId?: string;
};

export type AdapterBehaviorComparisonSummary = {
  schemaVersion: "adapter-behavior-comparison-summary.v1";
  scorecardCount: number;
  needsBagRunCount: number;
  comparedCount: number;
  bySplit: Record<EvalSplit, number>;
  bySourceKind: Record<string, number>;
  byObservedSystem: Record<AdapterBehaviorObservedSystem, number>;
  byComparisonStatus: Record<AdapterBehaviorComparisonStatus, number>;
};

const KNOWN_SOURCE_FAILURE_KINDS: readonly SourceAdapterFailureKind[] = [
  "bash_nonzero",
  "cancellation",
  "command_not_found",
  "edit_before_read",
  "generic_error",
  "hallucinated_skill",
  "malformed_args",
  "non_unique_edit_string",
  "permission_denied",
  "timeout",
  "user_correction",
];

const KNOWN_FAILURE_KIND_SET = new Set<string>(KNOWN_SOURCE_FAILURE_KINDS);

export const createAdapterBehaviorComparisonScorecards = (
  input: CreateAdapterBehaviorComparisonScorecardsInput,
): AdapterBehaviorComparisonScorecard[] => {
  const manifestByEvalCaseId = new Map(
    (input.manifest?.exportedSessions ?? []).map((session) => [session.evalCaseId, session]),
  );
  const manifestByCaptureId = new Map(
    (input.manifest?.exportedSessions ?? []).map((session) => [session.captureId, session]),
  );
  const bagByEvalCaseId = new Map((input.bagOutcomes ?? []).map((outcome) => [outcome.evalCaseId, outcome]));
  const bagByCaptureId = new Map(
    (input.bagOutcomes ?? [])
      .filter((outcome): outcome is BleedingAgentBehaviorOutcome & { captureId: string } => outcome.captureId != null)
      .map((outcome) => [outcome.captureId, outcome]),
  );

  return [...input.replayCases]
    .sort((left, right) => left.evalCaseId.localeCompare(right.evalCaseId))
    .map((replayCase) => {
      const manifestSession = manifestByEvalCaseId.get(replayCase.evalCaseId)
        ?? manifestByCaptureId.get(replayCase.captureId);
      const bagOutcome = bagByEvalCaseId.get(replayCase.evalCaseId)
        ?? bagByCaptureId.get(replayCase.captureId);
      return createAdapterBehaviorComparisonScorecard({
        replayCase,
        manifestSession,
        bagOutcome,
        defaultPolicyId: input.defaultPolicyId,
      });
    });
};

export const summarizeAdapterBehaviorComparisonScorecards = (
  scorecards: readonly AdapterBehaviorComparisonScorecard[],
): AdapterBehaviorComparisonSummary => {
  const bySplit = baseSplitCounts();
  const bySourceKind: Record<string, number> = {};
  const byObservedSystem: Record<AdapterBehaviorObservedSystem, number> = {
    bag: 0,
    claude: 0,
    codex: 0,
    unknown: 0,
  };
  const byComparisonStatus: Record<AdapterBehaviorComparisonStatus, number> = {
    compared: 0,
    needs_bag_run: 0,
  };

  for (const scorecard of scorecards) {
    bySplit[scorecard.split] += 1;
    bySourceKind[scorecard.baseline.sourceType] = (bySourceKind[scorecard.baseline.sourceType] ?? 0) + 1;
    byObservedSystem[scorecard.baseline.sourceSystem] += 1;
    byComparisonStatus[scorecard.comparisonStatus] += 1;
  }

  return {
    schemaVersion: "adapter-behavior-comparison-summary.v1",
    scorecardCount: scorecards.length,
    needsBagRunCount: byComparisonStatus.needs_bag_run,
    comparedCount: byComparisonStatus.compared,
    bySplit,
    bySourceKind: sortedRecord(bySourceKind),
    byObservedSystem,
    byComparisonStatus,
  };
};

const createAdapterBehaviorComparisonScorecard = (input: {
  replayCase: ReplayEvalCaseSkeleton;
  manifestSession: AdapterBehaviorManifestSession | undefined;
  bagOutcome: BleedingAgentBehaviorOutcome | undefined;
  defaultPolicyId: string | undefined;
}): AdapterBehaviorComparisonScorecard => {
  const sourceType = input.manifestSession?.sourceType ?? inferSourceTypeFromTags(input.replayCase.tags);
  const sessionKind = sessionKindForSourceType(sourceType);
  const sourceSessionId = input.manifestSession?.sourceSessionId ?? input.replayCase.sourceSessionId;
  const baseline: AdapterBehaviorObservedBaseline = {
    role: "observed_baseline",
    gold: false,
    sourceSystem: observedSystemForSourceType(sourceType),
    sourceType,
    sessionKind,
    captureId: input.replayCase.captureId,
    evalCaseId: input.replayCase.evalCaseId,
    split: input.replayCase.split,
    ...(sourceSessionId === undefined ? {} : { sourceSessionId }),
  };
  const observed = summarizeObservedReplayCase(input.replayCase);
  const bag = bagPlaceholder(input.bagOutcome, input.defaultPolicyId);
  const dimensions = comparisonDimensions({
    observed,
    baseline,
    bagOutcome: input.bagOutcome,
    bagStatus: bag.status,
  });
  const alignment = alignmentFromDimensions(dimensions);
  const needsBagRun = input.bagOutcome == null;

  return {
    scorecardId: stableId("scorecard.adapter-behavior", input.replayCase.evalCaseId, input.replayCase.captureId),
    schemaVersion: "adapter-behavior-comparison-scorecard.v1",
    evalCaseId: input.replayCase.evalCaseId,
    captureId: input.replayCase.captureId,
    split: input.replayCase.split,
    baseline,
    bag,
    gold: false,
    comparisonStatus: needsBagRun ? "needs_bag_run" : "compared",
    dimensions,
    alignment,
    notes: [
      "External Claude/Codex traces are observed_baseline evidence with gold=false.",
      needsBagRun
        ? "No BleedingAgent policy outcome was supplied; this scorecard is queued as needs_bag_run."
        : "BleedingAgent policy outcome is compared for behavioral alignment only, not judged against a gold label.",
    ],
  };
};

const summarizeObservedReplayCase = (replayCase: ReplayEvalCaseSkeleton): {
  failureKinds: string[];
  failureCategories: AdapterBehaviorFailureCategory[];
  toolFailures: number;
  terminalFailures: number;
  editFailures: number;
  fileReadFailures: number;
  permissionEvents: number;
  cancellationEvents: number;
} => {
  const failureKinds = uniqueSorted([
    ...replayCase.observedFailures.flatMap((failure) => optionalArray(failure.errorCode)),
    ...replayCase.tags.filter((tag) => KNOWN_FAILURE_KIND_SET.has(tag)),
  ]);
  const failureCategories = categoriesFromFailures(replayCase.observedFailures);
  return {
    failureKinds,
    failureCategories,
    toolFailures: countFailures(replayCase.observedFailures, "tool_call"),
    terminalFailures: countFailures(replayCase.observedFailures, "terminal_command"),
    editFailures: countFailures(replayCase.observedFailures, "edit_attempt"),
    fileReadFailures: countFailures(replayCase.observedFailures, "file_read"),
    permissionEvents: eventCount(replayCase.observedFailures, "permission_denied", failureKinds),
    cancellationEvents: eventCount(replayCase.observedFailures, "cancellation", failureKinds),
  };
};

const comparisonDimensions = (input: {
  observed: ReturnType<typeof summarizeObservedReplayCase>;
  baseline: AdapterBehaviorObservedBaseline;
  bagOutcome: BleedingAgentBehaviorOutcome | undefined;
  bagStatus: AdapterBehaviorBagOutcomeStatus | "needs_bag_run";
}): AdapterBehaviorComparisonDimensions => ({
  failureKinds: compareArrays(input.observed.failureKinds, input.bagOutcome?.failureKinds),
  failureCategories: compareArrays(input.observed.failureCategories, bagFailureCategories(input.bagOutcome)),
  toolFailures: compareNumber(input.observed.toolFailures, input.bagOutcome?.toolFailures),
  terminalFailures: compareNumber(input.observed.terminalFailures, input.bagOutcome?.terminalFailures),
  editFailures: compareNumber(input.observed.editFailures, input.bagOutcome?.editFailures),
  fileReadFailures: compareNumber(input.observed.fileReadFailures, input.bagOutcome?.fileReadFailures),
  permissionEvents: compareNumber(input.observed.permissionEvents, input.bagOutcome?.permissionEvents),
  cancellationEvents: compareNumber(input.observed.cancellationEvents, input.bagOutcome?.cancellationEvents),
  sourceKind: compareString(input.baseline.sourceType, input.bagOutcome?.sourceType),
  sessionKind: compareString(input.baseline.sessionKind, input.bagOutcome?.sessionKind),
  split: compareSplit(input.baseline.split, input.bagOutcome?.split),
  policyOutcome: {
    observed: "observed_baseline",
    ...(input.bagStatus === "needs_bag_run" ? {} : { bag: input.bagStatus }),
    status: input.bagStatus === "needs_bag_run" ? "needs_bag_run" : "observed_only",
  },
});

const bagPlaceholder = (
  outcome: BleedingAgentBehaviorOutcome | undefined,
  defaultPolicyId: string | undefined,
): AdapterBehaviorBagRunPlaceholder => {
  if (outcome == null) {
    return {
      policyId: defaultPolicyId ?? null,
      status: "needs_bag_run",
      outcomeSource: "bleeding_agent_policy",
      notes: ["Run BleedingAgent on this replay case before comparing policy behavior."],
    };
  }
  return {
    policyId: outcome.policyId,
    status: outcome.status,
    outcomeSource: "bleeding_agent_policy",
    notes: [...(outcome.notes ?? [])],
  };
};

const compareArrays = <T extends string>(
  observed: readonly T[],
  bag: readonly T[] | readonly string[] | undefined,
): AdapterBehaviorDimension<T[]> => {
  if (bag == null) return { observed: [...observed], status: "needs_bag_run" };
  const normalizedBag = uniqueSorted([...bag]) as T[];
  const normalizedObserved = uniqueSorted([...observed]) as T[];
  return {
    observed: normalizedObserved,
    bag: normalizedBag,
    status: sameArray(normalizedObserved, normalizedBag) ? "matches_observed" : "differs_from_observed",
  };
};

const compareNumber = (observed: number, bag: number | undefined): AdapterBehaviorDimension<number> => {
  if (bag == null) return { observed, status: "needs_bag_run" };
  return {
    observed,
    bag,
    status: observed === bag ? "matches_observed" : "differs_from_observed",
  };
};

const compareString = (observed: string, bag: string | undefined): AdapterBehaviorDimension<string> => {
  if (bag == null) return { observed, status: "needs_bag_run" };
  return {
    observed,
    bag,
    status: observed === bag ? "matches_observed" : "differs_from_observed",
  };
};

const compareSplit = (observed: EvalSplit, bag: EvalSplit | undefined): AdapterBehaviorDimension<EvalSplit> => {
  if (bag == null) return { observed, status: "needs_bag_run" };
  return {
    observed,
    bag,
    status: observed === bag ? "matches_observed" : "differs_from_observed",
  };
};

const alignmentFromDimensions = (
  dimensions: AdapterBehaviorComparisonDimensions,
): AdapterBehaviorComparisonScorecard["alignment"] => {
  const entries = Object.values(dimensions);
  const comparable = entries.filter((dimension) =>
    dimension.status === "matches_observed" || dimension.status === "differs_from_observed");
  const matched = comparable.filter((dimension) => dimension.status === "matches_observed");
  if (comparable.length === 0) {
    return {
      comparableDimensionCount: 0,
      matchedDimensionCount: 0,
    };
  }
  return {
    comparableDimensionCount: comparable.length,
    matchedDimensionCount: matched.length,
    score: matched.length / comparable.length,
  };
};

const bagFailureCategories = (
  outcome: BleedingAgentBehaviorOutcome | undefined,
): AdapterBehaviorFailureCategory[] | undefined => {
  if (outcome == null) return undefined;
  if (outcome.failures != null) return categoriesFromFailureObservations(outcome.failures);
  if ((outcome.failureKinds ?? []).length === 0) return ["none"];
  return undefined;
};

const categoriesFromFailures = (
  failures: readonly ReplayObservedFailure[],
): AdapterBehaviorFailureCategory[] =>
  failures.length === 0 ? ["none"] : uniqueSorted(failures.map((failure) => failure.failureKind));

const categoriesFromFailureObservations = (
  failures: readonly AdapterBehaviorFailureObservation[],
): AdapterBehaviorFailureCategory[] =>
  failures.length === 0 ? ["none"] : uniqueSorted(failures.map((failure) => failure.failureKind));

const countFailures = (
  failures: readonly ReplayObservedFailure[],
  failureKind: ReplayObservedFailure["failureKind"],
): number => failures.filter((failure) => failure.failureKind === failureKind).length;

const eventCount = (
  failures: readonly ReplayObservedFailure[],
  kind: SourceAdapterFailureKind,
  failureKinds: readonly string[],
): number => {
  const fromFailures = failures.filter((failure) =>
    failure.errorCode === kind || failure.status === kind).length;
  return fromFailures + (failureKinds.includes(kind) && fromFailures === 0 ? 1 : 0);
};

const observedSystemForSourceType = (sourceType: string): AdapterBehaviorObservedSystem => {
  if (sourceType === "cc-session-jsonl-v2") return "claude";
  if (sourceType === "codex-session-jsonl") return "codex";
  if (sourceType === "acp-session-jsonl") return "bag";
  return "unknown";
};

const sessionKindForSourceType = (sourceType: string): string =>
  sourceType.replace(/-session-jsonl(?:-v2)?$/, "") || "unknown";

const inferSourceTypeFromTags = (tags: readonly string[]): string =>
  tags.find((tag) => tag.endsWith("-session-jsonl") || tag.endsWith("-session-jsonl-v2")) ?? "unknown";

const baseSplitCounts = (): Record<EvalSplit, number> => ({
  dev: 0,
  holdout: 0,
  train: 0,
});

const sameArray = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((value, index) => value === right[index]);

const uniqueSorted = <T extends string>(values: readonly T[]): T[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const optionalArray = (value: string | undefined): string[] => value == null ? [] : [value];

const sortedRecord = (record: Record<string, number>): Record<string, number> =>
  Object.fromEntries(Object.entries(record).sort(([left], [right]) => left.localeCompare(right)));

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .replace(/[^A-Za-z0-9._:-]+/g, ".")
    .replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "")
    || `scorecard.${createHash("sha256").update(parts.join("\0")).digest("hex").slice(0, 16)}`;
