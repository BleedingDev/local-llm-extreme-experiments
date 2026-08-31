import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { ReleaseProofSchema } from "../evidence/generators/artifacts";
import { RealAcpCorpusRunManifestSchema, type RealAcpCorpusRunManifest } from "../replay/real-acp-runner";
import {
  FrozenCandidateRecordSchema,
  HoldoutAggregateProofSchema,
  type FrozenCandidateRecord,
  type HoldoutAggregateProof,
} from "./frozen-candidate";
import { OptimizerGateSuiteSchema } from "./gate-suite";
import {
  OperatorApprovalEvidenceRecordSchema,
  PostPromotionMonitorWindowEvidenceRecordSchema,
  RollbackCheckpointProofRecordSchema,
  evaluatePromotionEvidenceContracts,
  type OperatorApprovalEvidenceRecord,
  type PostPromotionMonitorWindowEvidenceRecord,
  type RollbackCheckpointProofRecord,
} from "./promotion-evidence-contracts";

const CANONICAL_EPOCH_PATH = ".bag/evidence/canonical-epoch.json";
const RELEASE_PROOF_PATH = ".bag/evidence/release-proof.json";
const OPTIMIZER_GATE_SUITE_PATH = ".bag/evidence/optimizer/index.json";
const OPERATOR_APPROVAL_PATH = ".bag/evidence/optimizer/operator-approval.json";
const ROLLBACK_CHECKPOINT_PROOF_PATH = ".bag/evidence/optimizer/rollback-checkpoint-proof.json";
const MONITOR_WINDOW_PATH = ".bag/evidence/optimizer/post-promotion-monitor-window.json";
const FROZEN_CANDIDATE_PATH = ".bag/evidence/optimizer/frozen-candidate.json";
const HOLDOUT_PROOF_PATH = ".bag/evidence/optimizer/holdout-aggregate-proof.json";
const REAL_ACP_RUNS_ROOT = ".bag/replay-corpus/real-acp-runs";

export const PromotionWorkflowActionSchema = z.enum(["preview", "approve", "promote", "monitor", "rollback"]);
export type PromotionWorkflowAction = z.infer<typeof PromotionWorkflowActionSchema>;

export const CanonicalEpochPromotionWorkflowSchema = z.object({
  schemaVersion: z.literal("evidence-command.epoch.v1"),
  epochId: z.string().min(1),
  graphId: z.string().min(1),
  selectionHash: z.string().min(1),
  generatedAt: z.string().min(1),
  sourceGraph: z.object({
    graphId: z.string().min(1),
    selectionHash: z.string().min(1),
    planSetHash: z.string().min(1).optional(),
    snapshotPath: z.string().min(1),
    generatedAt: z.string().min(1).optional(),
    selectedPlanPaths: z.array(z.string().min(1)).default([]),
  }).passthrough(),
  driftStatus: z.enum(["passed", "blocked"]),
  promotionReady: z.boolean(),
  stalePaths: z.array(z.string().min(1)).default([]),
  currentEvidencePaths: z.array(z.string().min(1)).default([]),
  candidateInputPaths: z.array(z.string().min(1)).default([]),
}).passthrough();
export type CanonicalEpochPromotionWorkflow = z.infer<typeof CanonicalEpochPromotionWorkflowSchema>;

const StabilityTaskRecordSchema = z.object({
  taskId: z.string().min(1),
  runResultId: z.string().min(1),
  runId: z.string().min(1),
  status: z.enum(["passed", "failed", "skipped", "cancelled", "error"]),
  changedFileCount: z.number().int().nonnegative(),
  writeToolCallCount: z.number().int().nonnegative(),
  terminalCommandCount: z.number().int().nonnegative().optional(),
  verifierStatus: z.enum(["passed", "failed", "skipped", "not_run"]),
  codingProgressClass: z.string().min(1).optional(),
}).passthrough();

const StabilityScorecardSchema = z.object({
  schemaVersion: z.literal("real-acp-stability-scorecard.v1"),
  scorecardId: z.string().min(1),
  createdAt: z.string().min(1),
  runIds: z.array(z.string().min(1)),
  taskRecords: z.array(StabilityTaskRecordSchema).default([]),
}).passthrough();
type StabilityScorecard = z.infer<typeof StabilityScorecardSchema>;

export type PromotionWorkflowArtifacts = {
  canonicalEpoch?: CanonicalEpochPromotionWorkflow | undefined;
  releaseProof?: z.infer<typeof ReleaseProofSchema> | undefined;
  optimizerGateSuite?: z.infer<typeof OptimizerGateSuiteSchema> | undefined;
  frozenCandidate?: FrozenCandidateRecord | undefined;
  holdoutProof?: HoldoutAggregateProof | undefined;
  operatorApproval?: OperatorApprovalEvidenceRecord | undefined;
  rollbackCheckpointProof?: RollbackCheckpointProofRecord | undefined;
  monitorWindow?: PostPromotionMonitorWindowEvidenceRecord | undefined;
  realAcpManifests?: readonly RealAcpCorpusRunManifest[] | undefined;
  stabilityScorecards?: readonly StabilityScorecard[] | undefined;
};

export type PromotionWorkflowInput = {
  cwd?: string | undefined;
  action?: PromotionWorkflowAction | undefined;
  graphId: string;
  selectionHash: string;
  candidatePatchId?: string | undefined;
  promotionDecisionId?: string | undefined;
  now?: string | undefined;
  artifacts?: PromotionWorkflowArtifacts | undefined;
};

export type PromotionWorkflowBlocker = {
  blockerId: string;
  message: string;
  phase: "epoch" | "contracts" | "candidate" | "quality" | "consumer" | "monitor" | "action";
  path?: string | undefined;
};

export type PromotionWorkflowDecision = {
  schemaVersion: "optimizer-promotion-workflow.v1";
  action: PromotionWorkflowAction;
  graphId: string;
  selectionHash: string;
  evidenceEpochId: string;
  candidatePatchId?: string | undefined;
  promotionDecisionId?: string | undefined;
  promotionReady: boolean;
  actionAllowed: boolean;
  failClosed: boolean;
  phase: "blocked" | "ready";
  blockers: PromotionWorkflowBlocker[];
  consumedEvidence: {
    canonicalEpoch?: string | undefined;
    releaseProof?: string | undefined;
    optimizerGateSuite?: string | undefined;
    frozenCandidate?: string | undefined;
    holdoutProof?: string | undefined;
    operatorApproval?: string | undefined;
    rollbackCheckpointProof?: string | undefined;
    monitorWindow?: string | undefined;
    realAcpRunIds: string[];
    stabilityScorecardIds: string[];
  };
};

export type PromotionWorkflowCommandResult = {
  ok: boolean;
  exitCode: number;
  decision: PromotionWorkflowDecision;
};

export const evaluatePromotionWorkflow = (input: PromotionWorkflowInput): PromotionWorkflowDecision => {
  const action = PromotionWorkflowActionSchema.parse(input.action ?? "preview");
  const artifacts = input.artifacts ?? loadPromotionWorkflowArtifacts(input.cwd ?? process.cwd());
  const expectedEpochId = `evidence-epoch.${input.graphId}.${input.selectionHash}`;
  const blockers = uniqueBlockers([
    ...epochBlockers(input, artifacts.canonicalEpoch),
    ...releaseProofBlockers(input, artifacts.releaseProof),
    ...optimizerGateBlockers(input, artifacts.optimizerGateSuite),
    ...candidateBlockers(input, artifacts.frozenCandidate, artifacts.holdoutProof),
    ...promotionContractBlockers(input, artifacts),
    ...currentQualityBlockers(artifacts),
    ...realConsumerBlockers(artifacts),
  ]);
  const promotionReady = blockers.length === 0;
  const actionAllowed = action === "preview" || promotionReady;
  const actionBlockers = actionAllowed
    ? []
    : [{
      blockerId: `action.${action}.blocked`,
      phase: "action" as const,
      message: `${action} is blocked until promotionReady is true for ${input.graphId}/${input.selectionHash}.`,
    }];

  return {
    schemaVersion: "optimizer-promotion-workflow.v1",
    action,
    graphId: input.graphId,
    selectionHash: input.selectionHash,
    evidenceEpochId: expectedEpochId,
    ...(input.candidatePatchId === undefined ? {} : { candidatePatchId: input.candidatePatchId }),
    ...(input.promotionDecisionId === undefined ? {} : { promotionDecisionId: input.promotionDecisionId }),
    promotionReady,
    actionAllowed,
    failClosed: !promotionReady,
    phase: promotionReady ? "ready" : "blocked",
    blockers: uniqueBlockers([...blockers, ...actionBlockers]),
    consumedEvidence: {
      ...(artifacts.canonicalEpoch === undefined ? {} : { canonicalEpoch: artifacts.canonicalEpoch.epochId }),
      ...(artifacts.releaseProof === undefined ? {} : { releaseProof: artifacts.releaseProof.releaseProofId }),
      ...(artifacts.optimizerGateSuite === undefined ? {} : { optimizerGateSuite: artifacts.optimizerGateSuite.optimizerGateSuiteId }),
      ...(artifacts.frozenCandidate === undefined ? {} : { frozenCandidate: artifacts.frozenCandidate.frozenCandidateId }),
      ...(artifacts.holdoutProof === undefined ? {} : { holdoutProof: artifacts.holdoutProof.proofId }),
      ...(artifacts.operatorApproval === undefined ? {} : { operatorApproval: artifacts.operatorApproval.approvalId }),
      ...(artifacts.rollbackCheckpointProof === undefined ? {} : { rollbackCheckpointProof: artifacts.rollbackCheckpointProof.checkpointProofId }),
      ...(artifacts.monitorWindow === undefined ? {} : { monitorWindow: artifacts.monitorWindow.monitorWindowId }),
      realAcpRunIds: (artifacts.realAcpManifests ?? []).map((manifest) => manifest.runId).sort(),
      stabilityScorecardIds: (artifacts.stabilityScorecards ?? []).map((scorecard) => scorecard.scorecardId).sort(),
    },
  };
};

export const runPromotionWorkflowCommand = (input: PromotionWorkflowInput): PromotionWorkflowCommandResult => {
  const decision = evaluatePromotionWorkflow(input);
  const ok = decision.action === "preview" ? decision.promotionReady : decision.actionAllowed;
  return {
    ok,
    exitCode: ok ? 0 : 1,
    decision,
  };
};

export const loadPromotionWorkflowArtifacts = (cwd: string): PromotionWorkflowArtifacts => ({
  canonicalEpoch: readOptionalJson(cwd, CANONICAL_EPOCH_PATH, CanonicalEpochPromotionWorkflowSchema),
  releaseProof: readOptionalJson(cwd, RELEASE_PROOF_PATH, ReleaseProofSchema),
  optimizerGateSuite: readOptionalJson(cwd, OPTIMIZER_GATE_SUITE_PATH, OptimizerGateSuiteSchema),
  frozenCandidate: readOptionalJson(cwd, FROZEN_CANDIDATE_PATH, FrozenCandidateRecordSchema),
  holdoutProof: readOptionalJson(cwd, HOLDOUT_PROOF_PATH, HoldoutAggregateProofSchema),
  operatorApproval: readOptionalJson(cwd, OPERATOR_APPROVAL_PATH, OperatorApprovalEvidenceRecordSchema),
  rollbackCheckpointProof: readOptionalJson(cwd, ROLLBACK_CHECKPOINT_PROOF_PATH, RollbackCheckpointProofRecordSchema),
  monitorWindow: readOptionalJson(cwd, MONITOR_WINDOW_PATH, PostPromotionMonitorWindowEvidenceRecordSchema),
  realAcpManifests: readRealAcpManifests(cwd),
  stabilityScorecards: readStabilityScorecards(cwd),
});

const epochBlockers = (
  input: PromotionWorkflowInput,
  epoch: CanonicalEpochPromotionWorkflow | undefined,
): PromotionWorkflowBlocker[] => {
  if (epoch === undefined) {
    return [blocker("epoch.missing", "epoch", `missing canonical epoch artifact: ${CANONICAL_EPOCH_PATH}`, CANONICAL_EPOCH_PATH)];
  }
  const blockers: PromotionWorkflowBlocker[] = [];
  if (epoch.graphId !== input.graphId) {
    blockers.push(blocker("epoch.graph-mismatch", "epoch", `canonical epoch targets graph ${epoch.graphId}, not ${input.graphId}`, CANONICAL_EPOCH_PATH));
  }
  if (epoch.selectionHash !== input.selectionHash) {
    blockers.push(blocker("epoch.selection-mismatch", "epoch", `canonical epoch targets selection ${epoch.selectionHash}, not ${input.selectionHash}`, CANONICAL_EPOCH_PATH));
  }
  const expectedEpochId = `evidence-epoch.${input.graphId}.${input.selectionHash}`;
  if (epoch.epochId !== expectedEpochId) {
    blockers.push(blocker("epoch.id-mismatch", "epoch", `canonical epoch id ${epoch.epochId} does not match ${expectedEpochId}`, CANONICAL_EPOCH_PATH));
  }
  if (epoch.driftStatus !== "passed") {
    blockers.push(blocker("epoch.drift-blocked", "epoch", `canonical epoch driftStatus=${epoch.driftStatus}; stale current slots: ${epoch.stalePaths.join(", ") || "unknown"}`, CANONICAL_EPOCH_PATH));
  }
  if (epoch.promotionReady) {
    blockers.push(blocker("epoch.unexpected-promotion-ready", "epoch", "canonical epoch must not be the source of promotionReady=true", CANONICAL_EPOCH_PATH));
  }
  return blockers;
};

const releaseProofBlockers = (
  input: PromotionWorkflowInput,
  proof: z.infer<typeof ReleaseProofSchema> | undefined,
): PromotionWorkflowBlocker[] => {
  if (proof === undefined) {
    return [blocker("release-proof.missing", "epoch", `missing release proof artifact: ${RELEASE_PROOF_PATH}`, RELEASE_PROOF_PATH)];
  }
  const blockers: PromotionWorkflowBlocker[] = [];
  if (proof.proofMode !== "current_graph") {
    blockers.push(blocker("release-proof.not-current", "epoch", `release proof mode is ${proof.proofMode ?? "historical"}, not current_graph`, RELEASE_PROOF_PATH));
  }
  if (proof.graphId !== input.graphId) {
    blockers.push(blocker("release-proof.graph-mismatch", "epoch", `release proof targets graph ${proof.graphId}, not ${input.graphId}`, RELEASE_PROOF_PATH));
  }
  if (proof.selectionHash !== input.selectionHash) {
    blockers.push(blocker("release-proof.selection-mismatch", "epoch", `release proof targets selection ${proof.selectionHash}, not ${input.selectionHash}`, RELEASE_PROOF_PATH));
  }
  if (proof.sourceGraph?.snapshotPath === undefined) {
    blockers.push(blocker("release-proof.source-graph-missing", "epoch", "release proof is missing sourceGraph snapshot binding", RELEASE_PROOF_PATH));
  }
  for (const reason of proof.optimizerDecision.blockingReasons) {
    blockers.push(blocker(`release-proof.optimizer.${stableId(reason)}`, "epoch", `release proof optimizer blocker: ${reason}`, RELEASE_PROOF_PATH));
  }
  if (proof.optimizerDecision.promotionReady) {
    blockers.push(blocker("release-proof.unexpected-promotion-ready", "epoch", "release proof cannot be accepted while other workflow proofs are re-evaluated separately", RELEASE_PROOF_PATH));
  }
  return blockers;
};

const optimizerGateBlockers = (
  input: PromotionWorkflowInput,
  suite: z.infer<typeof OptimizerGateSuiteSchema> | undefined,
): PromotionWorkflowBlocker[] => {
  if (suite === undefined) {
    return [blocker("optimizer-gates.missing", "contracts", `missing optimizer gate suite: ${OPTIMIZER_GATE_SUITE_PATH}`, OPTIMIZER_GATE_SUITE_PATH)];
  }
  const blockers: PromotionWorkflowBlocker[] = [];
  if (suite.graphId !== input.graphId) {
    blockers.push(blocker("optimizer-gates.graph-mismatch", "contracts", `optimizer gate suite targets graph ${suite.graphId}, not ${input.graphId}`, OPTIMIZER_GATE_SUITE_PATH));
  }
  if (!suite.currentDecision.promotionReady) {
    blockers.push(blocker("optimizer-gates.promotion-ready-false", "contracts", "optimizer gate suite promotionReady=false", OPTIMIZER_GATE_SUITE_PATH));
  }
  if (suite.currentDecision.autoPromotion !== "allowed") {
    blockers.push(blocker("optimizer-gates.auto-promotion-blocked", "contracts", `optimizer gate suite autoPromotion=${suite.currentDecision.autoPromotion}`, OPTIMIZER_GATE_SUITE_PATH));
  }
  for (const reason of suite.currentDecision.blockingReasons) {
    blockers.push(blocker(`optimizer-gates.${stableId(reason)}`, "contracts", `optimizer gate blocker: ${reason}`, OPTIMIZER_GATE_SUITE_PATH));
  }
  return blockers;
};

const candidateBlockers = (
  input: PromotionWorkflowInput,
  frozenCandidate: FrozenCandidateRecord | undefined,
  holdoutProof: HoldoutAggregateProof | undefined,
): PromotionWorkflowBlocker[] => {
  const blockers: PromotionWorkflowBlocker[] = [];
  if (frozenCandidate === undefined) {
    blockers.push(blocker("frozen-candidate.missing", "candidate", `missing frozen candidate artifact: ${FROZEN_CANDIDATE_PATH}`, FROZEN_CANDIDATE_PATH));
  } else {
    blockers.push(...frozenCandidateBlockers(input, frozenCandidate));
  }
  if (holdoutProof === undefined) {
    blockers.push(blocker("holdout-proof.missing", "candidate", `missing hidden holdout aggregate proof: ${HOLDOUT_PROOF_PATH}`, HOLDOUT_PROOF_PATH));
  } else {
    blockers.push(...holdoutProofBlockers(input, frozenCandidate, holdoutProof));
  }
  return blockers;
};

const frozenCandidateBlockers = (
  input: PromotionWorkflowInput,
  frozenCandidate: FrozenCandidateRecord,
): PromotionWorkflowBlocker[] => {
  const blockers: PromotionWorkflowBlocker[] = [];
  if (frozenCandidate.graphId !== input.graphId) {
    blockers.push(blocker("frozen-candidate.graph-mismatch", "candidate", `frozen candidate targets graph ${frozenCandidate.graphId}, not ${input.graphId}`, FROZEN_CANDIDATE_PATH));
  }
  if (frozenCandidate.selectionHash !== input.selectionHash) {
    blockers.push(blocker("frozen-candidate.selection-mismatch", "candidate", `frozen candidate targets selection ${frozenCandidate.selectionHash}, not ${input.selectionHash}`, FROZEN_CANDIDATE_PATH));
  }
  if (frozenCandidate.epochId !== `evidence-epoch.${input.graphId}.${input.selectionHash}`) {
    blockers.push(blocker("frozen-candidate.epoch-mismatch", "candidate", `frozen candidate targets epoch ${frozenCandidate.epochId}`, FROZEN_CANDIDATE_PATH));
  }
  if (input.candidatePatchId !== undefined && frozenCandidate.candidatePatchId !== input.candidatePatchId) {
    blockers.push(blocker("frozen-candidate.candidate-mismatch", "candidate", `frozen candidate targets candidate ${frozenCandidate.candidatePatchId}, not ${input.candidatePatchId}`, FROZEN_CANDIDATE_PATH));
  }
  return blockers;
};

const holdoutProofBlockers = (
  input: PromotionWorkflowInput,
  frozenCandidate: FrozenCandidateRecord | undefined,
  proof: HoldoutAggregateProof,
): PromotionWorkflowBlocker[] => {
  const blockers: PromotionWorkflowBlocker[] = [];
  if (proof.graphId !== input.graphId) {
    blockers.push(blocker("holdout-proof.graph-mismatch", "candidate", `hidden holdout proof targets graph ${proof.graphId}, not ${input.graphId}`, HOLDOUT_PROOF_PATH));
  }
  if (proof.selectionHash !== input.selectionHash) {
    blockers.push(blocker("holdout-proof.selection-mismatch", "candidate", `hidden holdout proof targets selection ${proof.selectionHash}, not ${input.selectionHash}`, HOLDOUT_PROOF_PATH));
  }
  if (proof.epochId !== `evidence-epoch.${input.graphId}.${input.selectionHash}`) {
    blockers.push(blocker("holdout-proof.epoch-mismatch", "candidate", `hidden holdout proof targets epoch ${proof.epochId}`, HOLDOUT_PROOF_PATH));
  }
  if (input.candidatePatchId !== undefined && proof.candidatePatchId !== input.candidatePatchId) {
    blockers.push(blocker("holdout-proof.candidate-mismatch", "candidate", `hidden holdout proof targets candidate ${proof.candidatePatchId}, not ${input.candidatePatchId}`, HOLDOUT_PROOF_PATH));
  }
  if (frozenCandidate !== undefined && proof.frozenCandidateId !== frozenCandidate.frozenCandidateId) {
    blockers.push(blocker("holdout-proof.frozen-candidate-mismatch", "candidate", "hidden holdout proof does not match the frozen candidate", HOLDOUT_PROOF_PATH));
  }
  if (proof.status !== "passed") {
    blockers.push(blocker("holdout-proof.not-passed", "candidate", `hidden holdout proof status=${proof.status}${proof.blockedReason === undefined ? "" : `: ${proof.blockedReason}`}`, HOLDOUT_PROOF_PATH));
  }
  if (!proof.evaluationOnly || !proof.aggregateOnly || proof.optimizerInputAllowed || proof.rawHoldoutContentIncluded) {
    blockers.push(blocker("holdout-proof.leakage", "candidate", "hidden holdout proof must be aggregate-only evaluation evidence and never optimizer input", HOLDOUT_PROOF_PATH));
  }
  if (proof.metrics.scorecardCount === 0) {
    blockers.push(blocker("holdout-proof.empty", "candidate", "hidden holdout proof has no scorecards", HOLDOUT_PROOF_PATH));
  }
  return blockers;
};

const promotionContractBlockers = (
  input: PromotionWorkflowInput,
  artifacts: PromotionWorkflowArtifacts,
): PromotionWorkflowBlocker[] => {
  const context = {
    graphId: input.graphId,
    selectionHash: input.selectionHash,
    planSetHash: artifacts.canonicalEpoch?.sourceGraph.planSetHash,
    evidenceEpochId: `evidence-epoch.${input.graphId}.${input.selectionHash}`,
    snapshotPath: artifacts.canonicalEpoch?.sourceGraph.snapshotPath ?? artifacts.releaseProof?.sourceGraph?.snapshotPath,
    snapshotSha256: snapshotHash(input.cwd, artifacts.canonicalEpoch?.sourceGraph.snapshotPath ?? artifacts.releaseProof?.sourceGraph?.snapshotPath),
    releaseProofPath: RELEASE_PROOF_PATH,
    releaseProofSha256: fileSha256(input.cwd, RELEASE_PROOF_PATH),
    ...(input.candidatePatchId === undefined ? {} : { candidatePatchId: input.candidatePatchId }),
    ...(input.promotionDecisionId === undefined ? {} : { promotionDecisionId: input.promotionDecisionId }),
    generatedAt: artifacts.canonicalEpoch?.sourceGraph.generatedAt ?? artifacts.releaseProof?.sourceGraph?.generatedAt,
    now: input.now ?? new Date().toISOString(),
  };
  const status = evaluatePromotionEvidenceContracts({
    context,
    ...(artifacts.operatorApproval === undefined ? {} : { operatorApproval: artifacts.operatorApproval }),
    ...(artifacts.rollbackCheckpointProof === undefined ? {} : { rollbackCheckpointProof: artifacts.rollbackCheckpointProof }),
    ...(artifacts.monitorWindow === undefined ? {} : { monitorWindow: artifacts.monitorWindow }),
  });
  return status.blockingReasons.map((reason) =>
    blocker(`promotion-contracts.${stableId(reason)}`, contractPhase(reason), `promotion evidence contract blocker: ${reason}`)
  );
};

const currentQualityBlockers = (artifacts: PromotionWorkflowArtifacts): PromotionWorkflowBlocker[] => {
  const qualityRecords = [
    ...(artifacts.realAcpManifests ?? []).flatMap((manifest) => manifest.taskResults.map((task) => ({
      sourceId: manifest.runId,
      status: task.status,
      changedFileCount: task.changedFiles.length,
      writeToolCallCount: task.toolCalls.filter((tool) => tool.sideEffectLevel === "write" && tool.status === "succeeded").length,
      verifierStatus: task.verifier.status,
      codingProgressClass: codingProgressClassFromUnknown(task.telemetry),
      executionMode: manifest.executionMode,
    }))),
    ...(artifacts.stabilityScorecards ?? []).flatMap((scorecard) => scorecard.taskRecords.map((task) => ({
      sourceId: scorecard.scorecardId,
      status: task.status,
      changedFileCount: task.changedFileCount,
      writeToolCallCount: task.writeToolCallCount,
      verifierStatus: task.verifierStatus,
      codingProgressClass: task.codingProgressClass,
      executionMode: "stability_scorecard",
    }))),
  ];
  if (qualityRecords.length === 0) {
    return [blocker("quality.missing", "quality", "missing current live ACP quality evidence with edit and verifier results")];
  }
  const negativeClass = qualityRecords.find((record) =>
    record.codingProgressClass !== undefined &&
    record.codingProgressClass !== "verified_edit" &&
    record.codingProgressClass !== "structured_impossibility"
  );
  if (negativeClass !== undefined) {
    return [blocker(
      "quality.coding-progress-failed",
      "quality",
      `current quality evidence ${negativeClass.sourceId} has codingProgressClass=${negativeClass.codingProgressClass}`,
    )];
  }
  const passing = qualityRecords.some((record) =>
    record.status === "passed" &&
    record.verifierStatus === "passed" &&
    (record.changedFileCount > 0 || record.writeToolCallCount > 0 || record.codingProgressClass === "verified_edit")
  );
  return passing
    ? []
    : [blocker("quality.no-non-empty-edit-pass", "quality", "missing non-empty edit plus passing verifier evidence in current quality runs")];
};

const realConsumerBlockers = (artifacts: PromotionWorkflowArtifacts): PromotionWorkflowBlocker[] => {
  const realConsumerManifests = (artifacts.realAcpManifests ?? []).filter((manifest) => manifest.executionMode === "real_consumer");
  if (realConsumerManifests.length === 0) {
    return [blocker("consumer.real-consumer-missing", "consumer", "missing real_consumer ACP evidence with non-empty edit and verifier pass")];
  }
  const passing = realConsumerManifests.some((manifest) =>
    !manifest.dryRun &&
    manifest.safety.realConsumerMutationAllowed &&
    manifest.taskResults.some((task) =>
      task.status === "passed" &&
      task.verifier.status === "passed" &&
      (task.changedFiles.length > 0 || task.toolCalls.some((tool) => tool.sideEffectLevel === "write" && tool.status === "succeeded") ||
        codingProgressClassFromUnknown(task.telemetry) === "verified_edit")
    )
  );
  return passing
    ? []
    : [blocker("consumer.real-consumer-quality-failed", "consumer", "real_consumer evidence exists but lacks non-empty edit plus passing verifier proof")];
};

const contractPhase = (reason: string): PromotionWorkflowBlocker["phase"] =>
  reason.includes("monitor window") || reason.includes("monitor-window") ? "monitor" : "contracts";

const blocker = (
  blockerId: string,
  phase: PromotionWorkflowBlocker["phase"],
  message: string,
  path?: string,
): PromotionWorkflowBlocker => ({
  blockerId,
  phase,
  message,
  ...(path === undefined ? {} : { path }),
});

const readOptionalJson = <T>(cwd: string, path: string, schema: z.ZodType<T>): T | undefined => {
  const absolutePath = join(cwd, path);
  if (!existsSync(absolutePath)) return undefined;
  try {
    return schema.parse(JSON.parse(readFileSync(absolutePath, "utf8")) as unknown);
  } catch {
    return undefined;
  }
};

const readRealAcpManifests = (cwd: string): RealAcpCorpusRunManifest[] => {
  const root = join(cwd, REAL_ACP_RUNS_ROOT);
  if (!existsSync(root)) return [];
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .flatMap((entry) => {
      const runDir = join(root, entry.name);
      return readdirSync(runDir, { withFileTypes: true })
        .filter((file) => file.isFile() && file.name.endsWith(".manifest.json"))
        .map((file) => join(REAL_ACP_RUNS_ROOT, entry.name, file.name));
    })
    .sort()
    .map((path) => readOptionalJson(cwd, path, RealAcpCorpusRunManifestSchema))
    .filter((manifest): manifest is RealAcpCorpusRunManifest => manifest !== undefined);
};

const readStabilityScorecards = (cwd: string): StabilityScorecard[] => {
  const root = join(cwd, REAL_ACP_RUNS_ROOT);
  if (!existsSync(root)) return [];
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .flatMap((entry) => {
      const runDir = join(root, entry.name);
      return readdirSync(runDir, { withFileTypes: true })
        .filter((file) => file.isFile() && file.name.endsWith(".stability-scorecard.json"))
        .map((file) => join(REAL_ACP_RUNS_ROOT, entry.name, file.name));
    })
    .sort()
    .map((path) => readOptionalJson(cwd, path, StabilityScorecardSchema))
    .filter((scorecard): scorecard is StabilityScorecard => scorecard !== undefined);
};

const codingProgressClassFromUnknown = (value: unknown): string | undefined => {
  if (value === null || typeof value !== "object") return undefined;
  const progressClass = (value as { codingProgressDiagnostic?: { progressClass?: unknown } }).codingProgressDiagnostic?.progressClass;
  return typeof progressClass === "string" && progressClass.length > 0 ? progressClass : undefined;
};

const fileSha256 = (cwd: string | undefined, path: string | undefined): string | undefined => {
  if (cwd === undefined || path === undefined) return undefined;
  const absolutePath = join(cwd, path);
  if (!existsSync(absolutePath)) return undefined;
  return `sha256:${createHash("sha256").update(readFileSync(absolutePath)).digest("hex")}`;
};

const snapshotHash = (inputCwd: string | undefined, path: string | undefined): string | undefined =>
  fileSha256(inputCwd, path);

const uniqueBlockers = (blockers: readonly PromotionWorkflowBlocker[]): PromotionWorkflowBlocker[] => {
  const byId = new Map<string, PromotionWorkflowBlocker>();
  for (const item of blockers) {
    if (!byId.has(item.blockerId)) {
      byId.set(item.blockerId, item);
    }
  }
  return [...byId.values()].sort((left, right) => left.blockerId.localeCompare(right.blockerId));
};

const stableId = (value: string): string =>
  value.toLowerCase().replace(/[^a-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 100) || "empty";
