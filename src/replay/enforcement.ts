import { z } from "zod";
import {
  EvalRunResultSchema,
  type EvalRunResult,
  type EvalSplit,
} from "../eval-harness/types";
import { OptimizerIdSchema } from "../optimizer/types";
import {
  ReplayEvalCaseSkeletonSchema,
  type ReplayEvalCaseSkeleton,
} from "./extraction";

const SAFE_OPTIMIZER_REDACTION_STATUSES = new Set(["redacted", "hash_only", "omitted"]);
const HIDDEN_REPLAY_SPLITS = new Set<EvalSplit>(["holdout"]);
const VISIBLE_REPLAY_OPTIMIZATION_SPLITS = new Set<EvalSplit>(["train", "dev"]);

export const ReplayOptimizerInputPurposeSchema = z.enum([
  "optimization_selection",
  "gepa_feedback",
  "proposer_prompt",
]);
export type ReplayOptimizerInputPurpose = z.infer<typeof ReplayOptimizerInputPurposeSchema>;

export const ReplayOptimizationRejectionSchema = z.object({
  evalCaseId: OptimizerIdSchema,
  split: z.enum(["train", "dev", "holdout"]),
  reasons: z.array(z.string().min(1)).min(1),
}).strict();
export type ReplayOptimizationRejection = z.infer<typeof ReplayOptimizationRejectionSchema>;

export const ReplayOptimizationSelectionSchema = z.object({
  purpose: ReplayOptimizerInputPurposeSchema,
  selectedCases: z.array(ReplayEvalCaseSkeletonSchema),
  selectedEvalCaseIds: z.array(OptimizerIdSchema),
  rejectedCases: z.array(ReplayOptimizationRejectionSchema),
  hiddenHoldoutEvalCaseIds: z.array(OptimizerIdSchema),
}).strict();
export type ReplayOptimizationSelection = z.infer<typeof ReplayOptimizationSelectionSchema>;

export type ReplayProposerPromptCase = {
  evalCaseId: string;
  split: Exclude<EvalSplit, "holdout">;
  title: string;
  task: string;
  oracleStrength: ReplayEvalCaseSkeleton["oracle"]["strength"];
  expectedBehaviorSummary: string;
  routing: ReplayEvalCaseSkeleton["routing"];
  observedFailureKinds: string[];
  tags: string[];
};

export const replayCaseOptimizationRejectionReasons = (
  replayCaseInput: ReplayEvalCaseSkeleton,
): string[] => {
  const replayCase = ReplayEvalCaseSkeletonSchema.parse(replayCaseInput);
  const reasons: string[] = [];
  if (!VISIBLE_REPLAY_OPTIMIZATION_SPLITS.has(replayCase.split)) {
    reasons.push(`split ${replayCase.split} is hidden from optimizer input`);
  }
  if (replayCase.redaction.needsReview || !SAFE_OPTIMIZER_REDACTION_STATUSES.has(replayCase.redaction.status)) {
    reasons.push(`redaction status ${replayCase.redaction.status} is not optimizer-safe`);
  }
  const unsafeSourceRefs = replayCase.sourceRefs.filter((sourceRef) =>
    sourceRef.redactionStatus === "raw_local_only" || sourceRef.redactionStatus === "needs_review",
  );
  if (unsafeSourceRefs.length > 0) {
    reasons.push("source refs include raw or needs-review content");
  }
  return reasons;
};

export const isReplayCaseAllowedForOptimizerInput = (
  replayCase: ReplayEvalCaseSkeleton,
): boolean => replayCaseOptimizationRejectionReasons(replayCase).length === 0;

export const selectReplayCasesForOptimizerInput = (
  replayCasesInput: readonly ReplayEvalCaseSkeleton[],
  purpose: ReplayOptimizerInputPurpose = "optimization_selection",
): ReplayOptimizationSelection => {
  const replayCases = canonicalReplayCases(replayCasesInput);
  const selectedCases: ReplayEvalCaseSkeleton[] = [];
  const rejectedCases: ReplayOptimizationRejection[] = [];
  const hiddenHoldoutEvalCaseIds: string[] = [];

  for (const replayCase of replayCases) {
    if (HIDDEN_REPLAY_SPLITS.has(replayCase.split)) {
      hiddenHoldoutEvalCaseIds.push(replayCase.evalCaseId);
    }
    const reasons = replayCaseOptimizationRejectionReasons(replayCase);
    if (reasons.length === 0) {
      selectedCases.push(replayCase);
    } else {
      rejectedCases.push(ReplayOptimizationRejectionSchema.parse({
        evalCaseId: replayCase.evalCaseId,
        split: replayCase.split,
        reasons,
      }));
    }
  }

  return ReplayOptimizationSelectionSchema.parse({
    purpose,
    selectedCases,
    selectedEvalCaseIds: selectedCases.map((replayCase) => replayCase.evalCaseId),
    rejectedCases,
    hiddenHoldoutEvalCaseIds: hiddenHoldoutEvalCaseIds.sort((left, right) => left.localeCompare(right)),
  });
};

export const assertReplayCasesAllowedForOptimizerInput = (
  replayCases: readonly ReplayEvalCaseSkeleton[],
  purpose: ReplayOptimizerInputPurpose,
): ReplayOptimizationSelection => {
  const selection = selectReplayCasesForOptimizerInput(replayCases, purpose);
  if (selection.rejectedCases.length > 0) {
    const blockedIds = selection.rejectedCases
      .map((rejection) => `${rejection.evalCaseId}: ${rejection.reasons.join("; ")}`)
      .join(", ");
    throw new Error(`replay ${purpose} rejected unsafe cases (${blockedIds})`);
  }
  return selection;
};

export const createReplayProposerPromptCases = (
  replayCases: readonly ReplayEvalCaseSkeleton[],
): ReplayProposerPromptCase[] =>
  selectReplayCasesForOptimizerInput(replayCases, "proposer_prompt").selectedCases.map((replayCase) => ({
    evalCaseId: replayCase.evalCaseId,
    split: replayCase.split as Exclude<EvalSplit, "holdout">,
    title: replayCase.title,
    task: replayCase.task,
    oracleStrength: replayCase.oracle.strength,
    expectedBehaviorSummary: replayCase.oracle.expectedBehavior.summary,
    routing: replayCase.routing,
    observedFailureKinds: [...new Set(replayCase.observedFailures.map((failure) => failure.failureKind))],
    tags: replayCase.tags,
  }));

export const selectReplayRunResultsForGepaFeedback = (
  runResultsInput: readonly EvalRunResult[],
  replayCasesInput: readonly ReplayEvalCaseSkeleton[],
): EvalRunResult[] => {
  const runResults = runResultsInput.map((result) => EvalRunResultSchema.parse(result));
  const allowedIds = new Set(
    selectReplayCasesForOptimizerInput(replayCasesInput, "gepa_feedback").selectedEvalCaseIds,
  );
  return runResults.filter((result) => allowedIds.has(result.evalCaseId));
};

export const assertReplayRunResultsAllowedForGepaFeedback = (
  runResultsInput: readonly EvalRunResult[],
  replayCasesInput: readonly ReplayEvalCaseSkeleton[],
): EvalRunResult[] => {
  const runResults = runResultsInput.map((result) => EvalRunResultSchema.parse(result));
  const allowedIds = new Set(
    selectReplayCasesForOptimizerInput(replayCasesInput, "gepa_feedback").selectedEvalCaseIds,
  );
  const blockedIds = [...new Set(runResults
    .filter((result) => !allowedIds.has(result.evalCaseId))
    .map((result) => result.evalCaseId))]
    .sort((left, right) => left.localeCompare(right));
  if (blockedIds.length > 0) {
    throw new Error(`replay GEPA feedback rejected hidden or unsafe eval cases (${blockedIds.join(", ")})`);
  }
  return runResults;
};

const canonicalReplayCases = (replayCases: readonly ReplayEvalCaseSkeleton[]): ReplayEvalCaseSkeleton[] => {
  const parsed = replayCases.map((replayCase) => ReplayEvalCaseSkeletonSchema.parse(replayCase));
  const seen = new Set<string>();
  for (const replayCase of parsed) {
    if (seen.has(replayCase.evalCaseId)) {
      throw new Error(`duplicate replay eval case id: ${replayCase.evalCaseId}`);
    }
    seen.add(replayCase.evalCaseId);
  }
  return parsed.sort((left, right) => {
    const splitComparison = splitOrder(left.split) - splitOrder(right.split);
    return splitComparison === 0
      ? left.evalCaseId.localeCompare(right.evalCaseId)
      : splitComparison;
  });
};

const splitOrder = (split: EvalSplit): number => {
  switch (split) {
    case "train":
      return 0;
    case "dev":
      return 1;
    case "holdout":
      return 2;
  }
};
