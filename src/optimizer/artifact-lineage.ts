import { z } from "zod";
import { EvalScorecardSchema, type EvalScorecard } from "../eval-harness/types";
import { CandidateEvidenceBundleSchema, type CandidateEvidenceBundle } from "./evidence";
import { HoldoutAggregateProofSchema, type HoldoutAggregateProof } from "./frozen-candidate";
import { CandidatePatchSchema, PromotionDecisionSchema, type CandidatePatch, type PromotionDecision } from "./types";
import { CandidateValidationResultSchema, type CandidateValidationResult } from "./validator";

const ARTIFACT_LINEAGE_SCHEMA_VERSION = "optimizer-artifact-lineage.v1" as const;

const OptimizerIdSchema = z.string().min(1).regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/);

export const OptimizerArtifactUpliftClassSchema = z.object({
  upliftClass: z.enum([
    "validation",
    "train_dev",
    "hidden_holdout",
    "full_eval",
    "live_rollout",
  ]),
  status: z.enum(["passed", "failed", "missing", "not_run", "inconclusive"]),
  scorecardIds: z.array(OptimizerIdSchema).default([]),
  evidenceBundleIds: z.array(OptimizerIdSchema).default([]),
  aggregateProofIds: z.array(OptimizerIdSchema).default([]),
  score: z.number().min(0).max(1).optional(),
  notes: z.array(z.string().min(1)).default([]),
}).strict();
export type OptimizerArtifactUpliftClass = z.infer<typeof OptimizerArtifactUpliftClassSchema>;

export const OptimizerArtifactLineageManifestSchema = z.object({
  schemaVersion: z.literal(ARTIFACT_LINEAGE_SCHEMA_VERSION),
  lineageManifestId: OptimizerIdSchema,
  candidatePatchId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema.optional(),
  baselinePolicyId: OptimizerIdSchema.optional(),
  candidatePolicyId: OptimizerIdSchema,
  codebaseRootFingerprint: z.string().min(1).optional(),
  evidenceBundleIds: z.array(OptimizerIdSchema).default([]),
  scorecardIds: z.array(OptimizerIdSchema).default([]),
  holdoutAggregateProofIds: z.array(OptimizerIdSchema).default([]),
  promotionDecisionId: OptimizerIdSchema.optional(),
  rollbackCheckpointPath: z.string().min(1).optional(),
  sourceTraceIds: z.array(z.string().min(1)).default([]),
  upliftClasses: z.array(OptimizerArtifactUpliftClassSchema).default([]),
  profile: z.object({
    candidateMatchesScorecards: z.boolean(),
    candidateMatchesEvidence: z.boolean(),
    candidateMatchesHoldoutProofs: z.boolean().default(true),
    hiddenHoldoutSeparated: z.boolean(),
    hiddenHoldoutAggregateOnly: z.boolean().default(true),
  }).strict(),
}).strict();
export type OptimizerArtifactLineageManifest = z.infer<typeof OptimizerArtifactLineageManifestSchema>;

export const OptimizerArtifactLineageGateSchema = z.object({
  gateId: z.enum([
    "candidate-id-present",
    "evidence-bundles-present",
    "scorecards-present",
    "profile-match",
    "baseline-candidate-policy-present",
    "hidden-holdout-separated",
    "rollback-checkpoint-present",
    "promotion-decision-present",
    "validation-uplift-present",
    "train-dev-uplift-present",
    "hidden-holdout-uplift-present",
    "weak-prompt-artifact-veto",
  ]),
  passed: z.boolean(),
  blocking: z.boolean().default(true),
  message: z.string().min(1),
}).strict();
export type OptimizerArtifactLineageGate = z.infer<typeof OptimizerArtifactLineageGateSchema>;

export const OptimizerArtifactLineageDecisionSchema = z.object({
  schemaVersion: z.literal(ARTIFACT_LINEAGE_SCHEMA_VERSION),
  lineageManifestId: OptimizerIdSchema,
  candidatePatchId: OptimizerIdSchema,
  promotionAllowed: z.boolean(),
  decision: z.enum(["would_promote", "reject", "quarantine", "needs_more_evidence"]),
  gates: z.array(OptimizerArtifactLineageGateSchema),
  blockingGateIds: z.array(OptimizerIdSchema).default([]),
  report: z.string().min(1),
}).strict();
export type OptimizerArtifactLineageDecision = z.infer<typeof OptimizerArtifactLineageDecisionSchema>;

export type BuildOptimizerArtifactLineageManifestInput = {
  candidate: CandidatePatch;
  validation?: CandidateValidationResult;
  visibleScorecards?: readonly EvalScorecard[];
  holdoutScorecards?: readonly EvalScorecard[];
  holdoutAggregateProofs?: readonly HoldoutAggregateProof[];
  fullEvalScorecards?: readonly EvalScorecard[];
  liveRolloutScorecards?: readonly EvalScorecard[];
  evidenceBundles?: readonly CandidateEvidenceBundle[];
  promotionDecision?: PromotionDecision;
  rollbackCheckpointPath?: string;
  clientProfileId?: string;
  baselinePolicyId?: string;
  candidatePolicyId?: string;
  lineageManifestId?: string;
};

export const buildOptimizerArtifactLineageManifest = (
  input: BuildOptimizerArtifactLineageManifestInput,
): OptimizerArtifactLineageManifest => {
  const candidate = CandidatePatchSchema.parse(input.candidate);
  const validation = input.validation === undefined ? undefined : CandidateValidationResultSchema.parse(input.validation);
  const visibleScorecards = (input.visibleScorecards ?? []).map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const holdoutScorecards = (input.holdoutScorecards ?? []).map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const holdoutAggregateProofs = (input.holdoutAggregateProofs ?? []).map((proof) => HoldoutAggregateProofSchema.parse(proof));
  const fullEvalScorecards = (input.fullEvalScorecards ?? []).map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const liveRolloutScorecards = (input.liveRolloutScorecards ?? []).map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const evidenceBundles = (input.evidenceBundles ?? []).map((bundle) => CandidateEvidenceBundleSchema.parse(bundle));
  const promotionDecision = input.promotionDecision === undefined
    ? undefined
    : PromotionDecisionSchema.parse(input.promotionDecision);
  const allScorecards = [...visibleScorecards, ...holdoutScorecards, ...fullEvalScorecards, ...liveRolloutScorecards];
  const scorecardIds = uniqueSorted([
    ...(candidate.scorecardIds ?? []),
    ...allScorecards.map((scorecard) => scorecard.scorecardId),
  ]);
  const evidenceBundleIds = uniqueSorted([
    ...(candidate.evidenceBundleIds ?? []),
    ...evidenceBundles.map((bundle) => bundle.evidenceBundleId),
  ]);
  const holdoutAggregateProofIds = uniqueSorted(holdoutAggregateProofs.map((proof) => proof.proofId));
  const candidatePolicyId = input.candidatePolicyId ?? candidate.candidatePolicyId ?? candidate.policyId;
  const clientProfileId = input.clientProfileId ?? candidate.clientProfileId;
  const baselinePolicyId = input.baselinePolicyId ?? candidate.baselinePolicyId;
  const rollbackCheckpointPath = input.rollbackCheckpointPath ?? candidate.rollbackCheckpointPath;
  return OptimizerArtifactLineageManifestSchema.parse({
    schemaVersion: ARTIFACT_LINEAGE_SCHEMA_VERSION,
    lineageManifestId: input.lineageManifestId ?? stableId("artifact-lineage", candidate.candidatePatchId),
    candidatePatchId: candidate.candidatePatchId,
    modelProfileId: candidate.modelProfileId,
    codebaseProfileId: candidate.codebaseProfileId,
    ...(clientProfileId === undefined ? {} : { clientProfileId }),
    ...(baselinePolicyId === undefined ? {} : { baselinePolicyId }),
    candidatePolicyId,
    ...(candidate.codebaseRootFingerprint === undefined ? {} : { codebaseRootFingerprint: candidate.codebaseRootFingerprint }),
    evidenceBundleIds,
    scorecardIds,
    holdoutAggregateProofIds,
    ...(promotionDecision?.promotionDecisionId === undefined ? {} : { promotionDecisionId: promotionDecision.promotionDecisionId }),
    ...(rollbackCheckpointPath === undefined ? {} : { rollbackCheckpointPath }),
    sourceTraceIds: candidate.sourceTraceIds,
    upliftClasses: upliftClasses({
      validation,
      visibleScorecards,
      holdoutScorecards,
      holdoutAggregateProofs,
      fullEvalScorecards,
      liveRolloutScorecards,
      evidenceBundleIds,
    }),
    profile: {
      candidateMatchesScorecards: allScorecards.every((scorecard) =>
        scorecard.candidate.context.modelProfileId === candidate.modelProfileId &&
        scorecard.candidate.context.codebaseProfileId === candidate.codebaseProfileId &&
        scorecard.candidate.context.policyId === candidatePolicyId),
      candidateMatchesEvidence: evidenceBundles.every((bundle) =>
        bundle.lineage.modelProfileIds.length === 0 || bundle.lineage.modelProfileIds.includes(candidate.modelProfileId)),
      candidateMatchesHoldoutProofs: holdoutAggregateProofs.every((proof) =>
        proof.candidatePatchId === candidate.candidatePatchId),
      hiddenHoldoutSeparated: holdoutScorecards.every((scorecard) => scorecard.split === "holdout") &&
        visibleScorecards.every((scorecard) => scorecard.split !== "holdout"),
      hiddenHoldoutAggregateOnly: holdoutAggregateProofs.every((proof) =>
        proof.purpose === "holdout_final" &&
        proof.evaluationOnly &&
        proof.aggregateOnly &&
        !proof.optimizerInputAllowed &&
        !proof.rawHoldoutContentIncluded),
    },
  });
};

export const assessOptimizerArtifactLineage = (
  manifestInput: OptimizerArtifactLineageManifest,
): OptimizerArtifactLineageDecision => {
  const manifest = OptimizerArtifactLineageManifestSchema.parse(manifestInput);
  const gates = [
    gate("candidate-id-present", manifest.candidatePatchId.length > 0, "Candidate id is present."),
    gate("evidence-bundles-present", manifest.evidenceBundleIds.length > 0, "At least one evidence bundle is attached."),
    gate("scorecards-present", manifest.scorecardIds.length > 0, "At least one scorecard is attached."),
    gate("profile-match", manifest.profile.candidateMatchesScorecards && manifest.profile.candidateMatchesEvidence && manifest.profile.candidateMatchesHoldoutProofs, "Candidate profile matches scorecards, evidence, and holdout aggregate proofs."),
    gate("baseline-candidate-policy-present", manifest.baselinePolicyId !== undefined && manifest.candidatePolicyId.length > 0, "Baseline and candidate policy ids are present."),
    gate("hidden-holdout-separated", manifest.profile.hiddenHoldoutSeparated && manifest.profile.hiddenHoldoutAggregateOnly, "Hidden holdout evidence is separated from visible evidence and aggregate-only."),
    gate("rollback-checkpoint-present", manifest.rollbackCheckpointPath !== undefined, "Rollback checkpoint path is present."),
    gate("promotion-decision-present", manifest.promotionDecisionId !== undefined, "Promotion decision id is present."),
    gate("validation-uplift-present", upliftStatus(manifest, "validation") !== "missing", "Validation uplift class is present."),
    gate("train-dev-uplift-present", upliftStatus(manifest, "train_dev") !== "missing", "Train/dev uplift class is present."),
    gate("hidden-holdout-uplift-present", upliftStatus(manifest, "hidden_holdout") !== "missing", "Hidden holdout uplift class is present."),
    gate(
      "weak-prompt-artifact-veto",
      manifest.evidenceBundleIds.length > 0 && manifest.scorecardIds.length > 0 && manifest.upliftClasses.length >= 3,
      "Artifact has evidence, scorecards, and multiple uplift classes.",
    ),
  ];
  const blockingGateIds = gates.filter((candidateGate) => candidateGate.blocking && !candidateGate.passed).map((candidateGate) => candidateGate.gateId);
  const promotionAllowed = blockingGateIds.length === 0;
  const decision = promotionAllowed
    ? "would_promote"
    : blockingGateIds.includes("profile-match") || blockingGateIds.includes("hidden-holdout-separated")
      ? "quarantine"
      : blockingGateIds.includes("rollback-checkpoint-present") || blockingGateIds.includes("promotion-decision-present")
        ? "needs_more_evidence"
        : "reject";
  return OptimizerArtifactLineageDecisionSchema.parse({
    schemaVersion: ARTIFACT_LINEAGE_SCHEMA_VERSION,
    lineageManifestId: manifest.lineageManifestId,
    candidatePatchId: manifest.candidatePatchId,
    promotionAllowed,
    decision,
    gates,
    blockingGateIds,
    report: renderOptimizerArtifactLineageReport({ manifest, gates, decision, promotionAllowed }),
  });
};

export const renderOptimizerArtifactLineageReport = (input: {
  manifest: OptimizerArtifactLineageManifest;
  gates: readonly OptimizerArtifactLineageGate[];
  decision: OptimizerArtifactLineageDecision["decision"];
  promotionAllowed: boolean;
}): string => [
  `# Optimizer Artifact Lineage`,
  ``,
  `Candidate: \`${input.manifest.candidatePatchId}\``,
  ``,
  `Decision: \`${input.decision}\``,
  ``,
  `Promotion allowed: ${input.promotionAllowed ? "yes" : "no"}`,
  ``,
  `## Evidence`,
  ``,
  `- evidence bundles: ${input.manifest.evidenceBundleIds.length === 0 ? "(none)" : input.manifest.evidenceBundleIds.map((id) => `\`${id}\``).join(", ")}`,
  `- scorecards: ${input.manifest.scorecardIds.length === 0 ? "(none)" : input.manifest.scorecardIds.map((id) => `\`${id}\``).join(", ")}`,
  `- holdout aggregate proofs: ${input.manifest.holdoutAggregateProofIds.length === 0 ? "(none)" : input.manifest.holdoutAggregateProofIds.map((id) => `\`${id}\``).join(", ")}`,
  `- rollback checkpoint: ${input.manifest.rollbackCheckpointPath ?? "(missing)"}`,
  ``,
  `## Uplift Classes`,
  ``,
  `| Class | Status | Score | Scorecards |`,
  `| --- | --- | ---: | --- |`,
  ...input.manifest.upliftClasses.map((uplift) =>
    `| ${uplift.upliftClass} | ${uplift.status} | ${uplift.score ?? 0} | ${uplift.scorecardIds.join(", ") || "-"} |`),
  ``,
  `## Gates`,
  ``,
  `| Gate | Passed | Blocking | Message |`,
  `| --- | --- | --- | --- |`,
  ...input.gates.map((candidateGate) =>
    `| ${candidateGate.gateId} | ${candidateGate.passed ? "yes" : "no"} | ${candidateGate.blocking ? "yes" : "no"} | ${candidateGate.message} |`),
  ``,
].join("\n");

const upliftClasses = (input: {
  validation: CandidateValidationResult | undefined;
  visibleScorecards: readonly EvalScorecard[];
  holdoutScorecards: readonly EvalScorecard[];
  holdoutAggregateProofs: readonly HoldoutAggregateProof[];
  fullEvalScorecards: readonly EvalScorecard[];
  liveRolloutScorecards: readonly EvalScorecard[];
  evidenceBundleIds: readonly string[];
}): OptimizerArtifactUpliftClass[] => [
  OptimizerArtifactUpliftClassSchema.parse({
    upliftClass: "validation",
    status: input.validation === undefined ? "missing" : input.validation.valid ? "passed" : "failed",
    evidenceBundleIds: input.evidenceBundleIds,
  }),
  scorecardUplift("train_dev", input.visibleScorecards),
  holdoutUplift(input.holdoutScorecards, input.holdoutAggregateProofs),
  scorecardUplift("full_eval", input.fullEvalScorecards),
  scorecardUplift("live_rollout", input.liveRolloutScorecards),
];

const scorecardUplift = (
  upliftClass: OptimizerArtifactUpliftClass["upliftClass"],
  scorecards: readonly EvalScorecard[],
): OptimizerArtifactUpliftClass => OptimizerArtifactUpliftClassSchema.parse({
  upliftClass,
  status: scorecards.length === 0 ? "missing" : scorecards.every((scorecard) => scorecard.passed) ? "passed" : "failed",
  scorecardIds: scorecards.map((scorecard) => scorecard.scorecardId),
  score: scorecards.length === 0
    ? undefined
    : scorecards.reduce((total, scorecard) => total + scorecard.aggregateScore, 0) / scorecards.length,
});

const holdoutUplift = (
  scorecards: readonly EvalScorecard[],
  proofs: readonly HoldoutAggregateProof[],
): OptimizerArtifactUpliftClass => {
  if (scorecards.length > 0) {
    return OptimizerArtifactUpliftClassSchema.parse({
      ...scorecardUplift("hidden_holdout", scorecards),
      aggregateProofIds: proofs.map((proof) => proof.proofId),
      notes: proofs.map((proof) => `aggregate proof ${proof.proofId}: ${proof.status}`),
    });
  }
  if (proofs.length === 0) {
    return scorecardUplift("hidden_holdout", []);
  }
  const failedProofs = proofs.filter((proof) => proof.status !== "passed");
  const scores = proofs
    .map((proof) => proof.metrics.candidateAggregateScore)
    .filter((score): score is number => score !== undefined);
  return OptimizerArtifactUpliftClassSchema.parse({
    upliftClass: "hidden_holdout",
    status: failedProofs.length === 0 ? "passed" : failedProofs.some((proof) => proof.status === "failed") ? "failed" : "not_run",
    scorecardIds: uniqueSorted(proofs.flatMap((proof) => proof.sourceScorecardIds)),
    aggregateProofIds: proofs.map((proof) => proof.proofId),
    score: scores.length === 0 ? undefined : scores.reduce((sum, score) => sum + score, 0) / scores.length,
    notes: proofs.map((proof) => `aggregate proof ${proof.proofId}: ${proof.status}`),
  });
};

const upliftStatus = (
  manifest: OptimizerArtifactLineageManifest,
  upliftClass: OptimizerArtifactUpliftClass["upliftClass"],
): OptimizerArtifactUpliftClass["status"] =>
  manifest.upliftClasses.find((uplift) => uplift.upliftClass === upliftClass)?.status ?? "missing";

const gate = (
  gateId: OptimizerArtifactLineageGate["gateId"],
  passed: boolean,
  message: string,
): OptimizerArtifactLineageGate => OptimizerArtifactLineageGateSchema.parse({
  gateId,
  passed,
  blocking: true,
  message: passed ? message : message.replace(/\.$/, " is missing or invalid."),
});

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const stableId = (...parts: readonly string[]): string =>
  parts.join(".").toLowerCase().replace(/[^a-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 160);
