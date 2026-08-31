import { createHash } from "node:crypto";
import { z } from "zod";
import {
  EvalScorecardSchema,
  type EvalScorecard,
} from "../eval-harness/types";
import {
  CandidatePatchSchema,
  OptimizerIdSchema,
  type CandidatePatch,
} from "./types";

const FROZEN_CANDIDATE_SCHEMA_VERSION = "optimizer-frozen-candidate.v1" as const;
const HOLDOUT_AGGREGATE_PROOF_SCHEMA_VERSION = "optimizer-holdout-aggregate-proof.v1" as const;

export const FrozenCandidatePromptFragmentRefSchema = z.object({
  fragmentId: OptimizerIdSchema,
  contentHash: z.string().min(1),
}).strict();
export type FrozenCandidatePromptFragmentRef = z.infer<typeof FrozenCandidatePromptFragmentRefSchema>;

export const FrozenCandidateVisibleInputBindingSchema = z.object({
  bindingId: OptimizerIdSchema,
  sourceKind: z.enum([
    "evidence_bundle",
    "eval_scorecard",
    "replay_export",
    "scorecard_suite",
  ]),
  sourceArtifactId: OptimizerIdSchema,
  split: z.enum(["train", "dev"]),
  contentHash: z.string().min(1),
  optimizerInputAllowed: z.literal(true),
  includedEvalCaseIds: z.array(OptimizerIdSchema).default([]),
}).strict();
export type FrozenCandidateVisibleInputBinding = z.infer<typeof FrozenCandidateVisibleInputBindingSchema>;

export const FrozenCandidateRecordSchema = z.object({
  schemaVersion: z.literal(FROZEN_CANDIDATE_SCHEMA_VERSION),
  frozenCandidateId: OptimizerIdSchema,
  candidatePatchId: OptimizerIdSchema,
  candidateContentHash: z.string().min(1),
  graphId: OptimizerIdSchema,
  selectionHash: z.string().min(1),
  epochId: OptimizerIdSchema,
  frozenAt: z.string().datetime({ offset: true }),
  policyId: OptimizerIdSchema,
  baselinePolicyId: OptimizerIdSchema.optional(),
  candidatePolicyId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema.optional(),
  codebaseRootFingerprint: z.string().min(1).optional(),
  promptFragments: z.array(FrozenCandidatePromptFragmentRefSchema).default([]),
  visibleInputBindings: z.array(FrozenCandidateVisibleInputBindingSchema).min(1),
  status: z.literal("frozen"),
}).strict().superRefine((value, ctx) => {
  const visibleSplits = new Set(value.visibleInputBindings.map((binding) => binding.split));
  for (const split of ["train", "dev"] as const) {
    if (!visibleSplits.has(split)) {
      ctx.addIssue({
        code: "custom",
        path: ["visibleInputBindings"],
        message: `frozen candidate requires ${split} visible input binding`,
      });
    }
  }
});
export type FrozenCandidateRecord = z.infer<typeof FrozenCandidateRecordSchema>;

export const FrozenCandidateVisibleEvaluationGateSchema = z.object({
  gateId: z.enum([
    "frozen-candidate-match",
    "visible-train-dev-present",
    "visible-scorecards-pass",
    "visible-holdout-excluded",
  ]),
  passed: z.boolean(),
  blocking: z.literal(true),
  message: z.string().min(1),
  scorecardIds: z.array(OptimizerIdSchema).default([]),
}).strict();
export type FrozenCandidateVisibleEvaluationGate = z.infer<typeof FrozenCandidateVisibleEvaluationGateSchema>;

export const FrozenCandidateVisibleEvaluationSchema = z.object({
  frozenCandidateId: OptimizerIdSchema,
  candidatePatchId: OptimizerIdSchema,
  readyForHoldout: z.boolean(),
  gates: z.array(FrozenCandidateVisibleEvaluationGateSchema),
  visibleScorecardIds: z.array(OptimizerIdSchema),
  aggregateScore: z.number().min(0).max(1).optional(),
  blocker: z.string().min(1).optional(),
}).strict();
export type FrozenCandidateVisibleEvaluation = z.infer<typeof FrozenCandidateVisibleEvaluationSchema>;

export const HoldoutAggregateProofSchema = z.object({
  schemaVersion: z.literal(HOLDOUT_AGGREGATE_PROOF_SCHEMA_VERSION),
  proofId: OptimizerIdSchema,
  frozenCandidateId: OptimizerIdSchema,
  candidatePatchId: OptimizerIdSchema,
  graphId: OptimizerIdSchema,
  selectionHash: z.string().min(1),
  epochId: OptimizerIdSchema,
  createdAt: z.string().datetime({ offset: true }),
  purpose: z.literal("holdout_final"),
  status: z.enum(["passed", "failed", "blocked", "not_run"]),
  evaluationOnly: z.literal(true),
  aggregateOnly: z.literal(true),
  optimizerInputAllowed: z.literal(false),
  rawHoldoutContentIncluded: z.literal(false),
  sourceScorecardIds: z.array(OptimizerIdSchema).default([]),
  sourceReplayExportIds: z.array(OptimizerIdSchema).default([]),
  sourceRunIds: z.array(OptimizerIdSchema).default([]),
  metrics: z.object({
    scorecardCount: z.number().int().nonnegative(),
    passedScorecardCount: z.number().int().nonnegative(),
    failedScorecardCount: z.number().int().nonnegative(),
    candidateAggregateScore: z.number().min(0).max(1).optional(),
    criticalRegressionCount: z.number().int().nonnegative(),
    baselineRunCount: z.number().int().nonnegative(),
    candidateRunCount: z.number().int().nonnegative(),
    hiddenHoldoutCaseCount: z.number().int().nonnegative().optional(),
  }).strict(),
  blockedReason: z.string().min(1).optional(),
}).strict();
export type HoldoutAggregateProof = z.infer<typeof HoldoutAggregateProofSchema>;

export const FrozenCandidateNonLeakageAssessmentSchema = z.object({
  frozenCandidateId: OptimizerIdSchema,
  passed: z.boolean(),
  violations: z.array(z.string().min(1)),
  visibleInputBindingIds: z.array(OptimizerIdSchema),
  holdoutAggregateProofIds: z.array(OptimizerIdSchema),
}).strict();
export type FrozenCandidateNonLeakageAssessment = z.infer<typeof FrozenCandidateNonLeakageAssessmentSchema>;

export type BuildFrozenCandidateRecordInput = {
  candidate: CandidatePatch;
  graphId: string;
  selectionHash: string;
  epochId: string;
  frozenAt?: string;
  promptFragments?: readonly (FrozenCandidatePromptFragmentRef | { fragmentId: string; content: string })[];
  visibleInputBindings: readonly FrozenCandidateVisibleInputBinding[];
};

export type BuildHoldoutAggregateProofInput = {
  frozenCandidate: FrozenCandidateRecord;
  visibleEvaluation: FrozenCandidateVisibleEvaluation;
  holdoutScorecards?: readonly EvalScorecard[];
  sourceReplayExportIds?: readonly string[];
  sourceRunIds?: readonly string[];
  hiddenHoldoutCaseCount?: number;
  createdAt?: string;
  proofId?: string;
  blockedReason?: string;
};

export const buildFrozenCandidateRecord = (
  input: BuildFrozenCandidateRecordInput,
): FrozenCandidateRecord => {
  const candidate = CandidatePatchSchema.parse(input.candidate);
  const visibleInputBindings = input.visibleInputBindings.map((binding) =>
    FrozenCandidateVisibleInputBindingSchema.parse(binding)
  );
  const candidatePolicyId = candidate.candidatePolicyId ?? candidate.policyId;
  const candidateContentHash = sha256(stableJson(candidate));
  return FrozenCandidateRecordSchema.parse({
    schemaVersion: FROZEN_CANDIDATE_SCHEMA_VERSION,
    frozenCandidateId: stableId(
      "frozen-candidate",
      input.graphId,
      input.selectionHash,
      input.epochId,
      candidate.candidatePatchId,
      candidateContentHash.slice(0, 16),
    ),
    candidatePatchId: candidate.candidatePatchId,
    candidateContentHash,
    graphId: input.graphId,
    selectionHash: input.selectionHash,
    epochId: input.epochId,
    frozenAt: input.frozenAt ?? new Date().toISOString(),
    policyId: candidate.policyId,
    ...(candidate.baselinePolicyId === undefined ? {} : { baselinePolicyId: candidate.baselinePolicyId }),
    candidatePolicyId,
    modelProfileId: candidate.modelProfileId,
    codebaseProfileId: candidate.codebaseProfileId,
    ...(candidate.clientProfileId === undefined ? {} : { clientProfileId: candidate.clientProfileId }),
    ...(candidate.codebaseRootFingerprint === undefined ? {} : { codebaseRootFingerprint: candidate.codebaseRootFingerprint }),
    promptFragments: (input.promptFragments ?? []).map((fragment) =>
      "contentHash" in fragment
        ? FrozenCandidatePromptFragmentRefSchema.parse(fragment)
        : FrozenCandidatePromptFragmentRefSchema.parse({
            fragmentId: fragment.fragmentId,
            contentHash: sha256(fragment.content),
          })
    ),
    visibleInputBindings,
    status: "frozen",
  });
};

export const assessFrozenCandidateVisibleEvaluation = (input: {
  frozenCandidate: FrozenCandidateRecord;
  visibleScorecards: readonly EvalScorecard[];
}): FrozenCandidateVisibleEvaluation => {
  const frozenCandidate = FrozenCandidateRecordSchema.parse(input.frozenCandidate);
  const visibleScorecards = input.visibleScorecards.map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const scorecardIds = visibleScorecards.map((scorecard) => scorecard.scorecardId);
  const holdoutScorecards = visibleScorecards.filter((scorecard) => scorecard.split === "holdout");
  const visibleOnlyScorecards = visibleScorecards.filter((scorecard) => scorecard.split !== "holdout");
  const visibleSplits = new Set(visibleOnlyScorecards.map((scorecard) => scorecard.split));
  const missingSplits = ["train", "dev"].filter((split) => !visibleSplits.has(split as "train" | "dev"));
  const mismatched = visibleOnlyScorecards.filter((scorecard) => !scorecardMatchesFrozenCandidate(scorecard, frozenCandidate));
  const failing = visibleOnlyScorecards.filter((scorecard) => !scorecard.passed || scorecard.criticalRegressionVeto.vetoed);
  const gates = [
    visibleGate(
      "frozen-candidate-match",
      mismatched.length === 0,
      mismatched.length === 0
        ? "Visible scorecards match the frozen candidate profile."
        : `Visible scorecards do not match frozen candidate: ${mismatched.map((scorecard) => scorecard.scorecardId).join(", ")}.`,
      mismatched.map((scorecard) => scorecard.scorecardId),
    ),
    visibleGate(
      "visible-train-dev-present",
      missingSplits.length === 0,
      missingSplits.length === 0
        ? "Visible train/dev scorecards are present."
        : `Visible train/dev scorecards are missing splits: ${missingSplits.join(", ")}.`,
      scorecardIds,
    ),
    visibleGate(
      "visible-scorecards-pass",
      failing.length === 0,
      failing.length === 0
        ? "Visible scorecards passed."
        : `Visible scorecards failed: ${failing.map((scorecard) => scorecard.scorecardId).join(", ")}.`,
      failing.map((scorecard) => scorecard.scorecardId),
    ),
    visibleGate(
      "visible-holdout-excluded",
      holdoutScorecards.length === 0,
      holdoutScorecards.length === 0
        ? "Holdout scorecards are excluded from visible evaluation."
        : `Holdout scorecards were supplied to visible evaluation: ${holdoutScorecards.map((scorecard) => scorecard.scorecardId).join(", ")}.`,
      holdoutScorecards.map((scorecard) => scorecard.scorecardId),
    ),
  ];
  const readyForHoldout = gates.every((gate) => gate.passed);
  const aggregateScore = visibleOnlyScorecards.length === 0
    ? undefined
    : visibleOnlyScorecards.reduce((sum, scorecard) => sum + scorecard.aggregateScore, 0) / visibleOnlyScorecards.length;
  const blocker = gates.find((gate) => !gate.passed)?.message;
  return FrozenCandidateVisibleEvaluationSchema.parse({
    frozenCandidateId: frozenCandidate.frozenCandidateId,
    candidatePatchId: frozenCandidate.candidatePatchId,
    readyForHoldout,
    gates,
    visibleScorecardIds: visibleOnlyScorecards.map((scorecard) => scorecard.scorecardId),
    ...(aggregateScore === undefined ? {} : { aggregateScore }),
    ...(blocker === undefined ? {} : { blocker }),
  });
};

export const buildHoldoutAggregateProof = (
  input: BuildHoldoutAggregateProofInput,
): HoldoutAggregateProof => {
  const frozenCandidate = FrozenCandidateRecordSchema.parse(input.frozenCandidate);
  const visibleEvaluation = FrozenCandidateVisibleEvaluationSchema.parse(input.visibleEvaluation);
  const holdoutScorecards = (input.holdoutScorecards ?? []).map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const invalidScorecards = holdoutScorecards.filter((scorecard) =>
    scorecard.split !== "holdout" || !scorecardMatchesFrozenCandidate(scorecard, frozenCandidate)
  );
  if (invalidScorecards.length > 0) {
    throw new Error(`holdout aggregate proof rejected non-holdout or mismatched scorecards: ${invalidScorecards.map((scorecard) => scorecard.scorecardId).join(", ")}`);
  }

  const failedScorecards = holdoutScorecards.filter((scorecard) => !scorecard.passed || scorecard.criticalRegressionVeto.vetoed);
  const blockedReason = input.blockedReason ?? (
    visibleEvaluation.readyForHoldout
      ? undefined
      : `visible evaluation did not qualify frozen candidate for holdout: ${visibleEvaluation.blocker ?? "unknown blocker"}`
  );
  const status = blockedReason !== undefined
    ? "blocked"
    : holdoutScorecards.length === 0
      ? "not_run"
      : failedScorecards.length === 0
        ? "passed"
        : "failed";
  const candidateAggregateScore = holdoutScorecards.length === 0
    ? undefined
    : holdoutScorecards.reduce((sum, scorecard) => sum + scorecard.aggregateScore, 0) / holdoutScorecards.length;

  return HoldoutAggregateProofSchema.parse({
    schemaVersion: HOLDOUT_AGGREGATE_PROOF_SCHEMA_VERSION,
    proofId: input.proofId ?? stableId("holdout-proof", frozenCandidate.frozenCandidateId, status),
    frozenCandidateId: frozenCandidate.frozenCandidateId,
    candidatePatchId: frozenCandidate.candidatePatchId,
    graphId: frozenCandidate.graphId,
    selectionHash: frozenCandidate.selectionHash,
    epochId: frozenCandidate.epochId,
    createdAt: input.createdAt ?? new Date().toISOString(),
    purpose: "holdout_final",
    status,
    evaluationOnly: true,
    aggregateOnly: true,
    optimizerInputAllowed: false,
    rawHoldoutContentIncluded: false,
    sourceScorecardIds: holdoutScorecards.map((scorecard) => scorecard.scorecardId),
    sourceReplayExportIds: uniqueSorted(input.sourceReplayExportIds ?? []),
    sourceRunIds: uniqueSorted(input.sourceRunIds ?? []),
    metrics: {
      scorecardCount: holdoutScorecards.length,
      passedScorecardCount: holdoutScorecards.length - failedScorecards.length,
      failedScorecardCount: failedScorecards.length,
      ...(candidateAggregateScore === undefined ? {} : { candidateAggregateScore }),
      criticalRegressionCount: holdoutScorecards.reduce(
        (sum, scorecard) => sum + scorecard.criticalRegressionVeto.regressions.length,
        0,
      ),
      baselineRunCount: holdoutScorecards.reduce(
        (sum, scorecard) => sum + scorecard.runResults.filter((run) => run.runRole === "baseline").length,
        0,
      ),
      candidateRunCount: holdoutScorecards.reduce(
        (sum, scorecard) => sum + scorecard.runResults.filter((run) => run.runRole === "candidate").length,
        0,
      ),
      ...(input.hiddenHoldoutCaseCount === undefined ? {} : { hiddenHoldoutCaseCount: input.hiddenHoldoutCaseCount }),
    },
    ...(blockedReason === undefined ? {} : { blockedReason }),
  });
};

export const assessFrozenCandidateNonLeakage = (input: {
  frozenCandidate: FrozenCandidateRecord;
  holdoutAggregateProofs?: readonly HoldoutAggregateProof[];
}): FrozenCandidateNonLeakageAssessment => {
  const frozenCandidate = FrozenCandidateRecordSchema.parse(input.frozenCandidate);
  const holdoutAggregateProofs = (input.holdoutAggregateProofs ?? []).map((proof) =>
    HoldoutAggregateProofSchema.parse(proof)
  );
  const violations = holdoutAggregateProofs.flatMap((proof) => {
    const proofViolations: string[] = [];
    if (proof.frozenCandidateId !== frozenCandidate.frozenCandidateId) {
      proofViolations.push(`holdout proof ${proof.proofId} does not match frozen candidate`);
    }
    if (!proof.aggregateOnly || proof.rawHoldoutContentIncluded || proof.optimizerInputAllowed) {
      proofViolations.push(`holdout proof ${proof.proofId} is not aggregate-only evaluation evidence`);
    }
    return proofViolations;
  });
  return FrozenCandidateNonLeakageAssessmentSchema.parse({
    frozenCandidateId: frozenCandidate.frozenCandidateId,
    passed: violations.length === 0,
    violations,
    visibleInputBindingIds: frozenCandidate.visibleInputBindings.map((binding) => binding.bindingId),
    holdoutAggregateProofIds: holdoutAggregateProofs.map((proof) => proof.proofId),
  });
};

export const assertFrozenCandidateNonLeakage = (input: {
  frozenCandidate: FrozenCandidateRecord;
  holdoutAggregateProofs?: readonly HoldoutAggregateProof[];
}): FrozenCandidateNonLeakageAssessment => {
  const assessment = assessFrozenCandidateNonLeakage(input);
  if (!assessment.passed) {
    throw new Error(`frozen candidate holdout non-leakage failed: ${assessment.violations.join("; ")}`);
  }
  return assessment;
};

const scorecardMatchesFrozenCandidate = (
  scorecard: EvalScorecard,
  frozenCandidate: FrozenCandidateRecord,
): boolean =>
  scorecard.candidate.context.modelProfileId === frozenCandidate.modelProfileId &&
  scorecard.candidate.context.codebaseProfileId === frozenCandidate.codebaseProfileId &&
  scorecard.candidate.context.policyId === frozenCandidate.candidatePolicyId;

const visibleGate = (
  gateId: FrozenCandidateVisibleEvaluationGate["gateId"],
  passed: boolean,
  message: string,
  scorecardIds: readonly string[],
): FrozenCandidateVisibleEvaluationGate => FrozenCandidateVisibleEvaluationGateSchema.parse({
  gateId,
  passed,
  blocking: true,
  message,
  scorecardIds,
});

const stableJson = (value: unknown): string => {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableJson(item)).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
};

const sha256 = (value: string): string =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const stableId = (...parts: readonly string[]): string =>
  parts.join(".").toLowerCase().replace(/[^a-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 180);
