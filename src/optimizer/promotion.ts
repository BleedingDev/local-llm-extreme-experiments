import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import type { BagConfig } from "../types";
import {
  EvalRunResultSchema,
  EvalScorecardSchema,
  type EvalRunResult,
  type EvalScorecard,
} from "../eval-harness/types";
import {
  CandidateEvidenceBundleSchema,
  type CandidateEvidenceBundle,
  type CandidateEvidenceObservation,
} from "./evidence";
import {
  OptimizerArtifactLineageDecisionSchema,
  type OptimizerArtifactLineageDecision,
} from "./artifact-lineage";
import {
  loadActiveOptimizerPointer,
  loadOptimizerRegistry,
  optimizerRegistryCheckpointsDir,
  promoteActiveOptimizerPointer,
  saveOptimizerRegistryRecord,
  type ActiveOptimizerPointer,
} from "./registry";
import { evaluateOptimizerRuntimeReadiness } from "./runtime-readiness";
import {
  CandidatePatchSchema,
  PromotionDecisionSchema,
  type CandidatePatch,
  type OptimizerRegistryRecord,
  type PromotionDecision,
} from "./types";
import { CandidateValidationResultSchema, type CandidateValidationResult } from "./validator";

export const CandidatePromotionResultSchema = z.object({
  promoted: z.boolean(),
  candidatePatchId: z.string().min(1),
  decision: PromotionDecisionSchema,
  previousPointer: z.unknown().optional(),
  activePointer: z.unknown().optional(),
  checkpointPath: z.string().optional(),
  registryRecordIds: z.array(z.string()).default([]),
});
export type CandidatePromotionResult = z.infer<typeof CandidatePromotionResultSchema>;

export type PromoteCandidatePatchInput = {
  config: BagConfig;
  cwd?: string;
  candidate: CandidatePatch;
  validation: CandidateValidationResult;
  candidateEval: EvalScorecard;
  decidedAt?: string;
  decisionId?: string;
  promotionGatePassed?: boolean;
  promotionGateReason?: string;
  lineageDecision?: OptimizerArtifactLineageDecision;
};

export type RollbackOptimizerPromotionInput = {
  config: BagConfig;
  cwd?: string;
  checkpointPath?: string;
};

export type PostPromotionRollbackMode = "perform" | "request" | "disabled";

export const PostPromotionRegressionSignalSchema = z.object({
  signalId: z.string().min(1),
  source: z.enum(["trace_evidence", "eval_run", "eval_scorecard"]),
  severity: z.enum(["warning", "failure", "critical"]),
  policyId: z.string().min(1),
  candidatePatchId: z.string().min(1).optional(),
  scorecardId: z.string().min(1).optional(),
  runResultId: z.string().min(1).optional(),
  traceId: z.string().min(1).optional(),
  reason: z.string().min(1),
}).strict();
export type PostPromotionRegressionSignal = z.infer<typeof PostPromotionRegressionSignalSchema>;

export const PostPromotionMonitorResultSchema = z.object({
  promotedPolicyId: z.string().min(1),
  candidatePatchId: z.string().min(1).optional(),
  regressionDetected: z.boolean(),
  rollbackRequested: z.boolean(),
  rolledBack: z.boolean(),
  signals: z.array(PostPromotionRegressionSignalSchema),
  checkpointPath: z.string().optional(),
  previousPointer: z.unknown().optional(),
  rollbackPointer: z.unknown().optional(),
});
export type PostPromotionMonitorResult = z.infer<typeof PostPromotionMonitorResultSchema> & {
  previousPointer?: ActiveOptimizerPointer | undefined;
  rollbackPointer?: ActiveOptimizerPointer | undefined;
};

export type MonitorPostPromotionInput = {
  config: BagConfig;
  cwd?: string;
  promotion: CandidatePromotionResult;
  evalScorecards?: readonly EvalScorecard[];
  evalRunResults?: readonly EvalRunResult[];
  evidenceBundles?: readonly CandidateEvidenceBundle[];
  rollbackMode?: PostPromotionRollbackMode;
};

export const promoteCandidatePatch = (input: PromoteCandidatePatchInput): CandidatePromotionResult => {
  const cwd = input.cwd ?? process.cwd();
  const candidate = CandidatePatchSchema.parse(input.candidate);
  const validation = CandidateValidationResultSchema.parse(input.validation);
  const candidateEval = EvalScorecardSchema.parse(input.candidateEval);
  const decidedAt = input.decidedAt ?? new Date().toISOString();
  const previousPointer = loadActiveOptimizerPointer(input.config, cwd).pointer;
  const runtimeReadiness = evaluateOptimizerRuntimeReadiness({
    cwd,
    registry: loadOptimizerRegistry(input.config, cwd),
  });
  const aggregatePromotionGatePassed = runtimeReadiness.allowed && input.promotionGatePassed !== false;
  const aggregatePromotionGateReason = promotionGateReason([
    ...(runtimeReadiness.allowed ? [] : runtimeReadiness.reasons),
    ...(input.promotionGatePassed === false
      ? [input.promotionGateReason ?? "aggregate promotion gates failed"]
      : []),
  ]);
  const checkpointPath = writePromotionCheckpoint(input.config, cwd, candidate.candidatePatchId, previousPointer, decidedAt);
  const profileGate = promotionProfileGate(candidate, candidateEval);
  const lineageGate = promotionLineageGate(candidate, input.lineageDecision);
  const decision = PromotionDecisionSchema.parse({
    promotionDecisionId: input.decisionId ?? stableId("promotion", candidate.candidatePatchId),
    decision: validation.valid &&
        candidateEval.passed &&
        !candidateEval.criticalRegressionVeto.vetoed &&
        profileGate.passed &&
        aggregatePromotionGatePassed &&
        lineageGate.passed
      ? "promote"
      : "reject",
    policyId: candidate.policyId,
    candidatePatchId: candidate.candidatePatchId,
    evalResultId: candidateEval.scorecardId,
    modelProfileId: candidate.modelProfileId,
    codebaseProfileId: candidate.codebaseProfileId,
    ...(candidate.clientProfileId === undefined ? {} : { clientProfileId: candidate.clientProfileId }),
    ...(candidate.baselinePolicyId === undefined ? {} : { baselinePolicyId: candidate.baselinePolicyId }),
    candidatePolicyId: candidate.candidatePolicyId ?? candidate.policyId,
    ...(candidate.codebaseRootFingerprint === undefined ? {} : { codebaseRootFingerprint: candidate.codebaseRootFingerprint }),
    canonicalToolVersion: candidateEval.candidate.context.canonicalToolVersion,
    renderedToolVersion: candidateEval.candidate.context.renderedToolVersion,
    resultStyleVersion: candidateEval.candidate.context.resultStyleVersion,
    verificationPolicyVersion: candidateEval.candidate.context.verificationPolicyVersion,
    evidenceBundleIds: candidate.evidenceBundleIds ?? [],
    scorecardIds: [...new Set([...(candidate.scorecardIds ?? []), candidateEval.scorecardId])],
    rollbackCheckpointPath: checkpointPath,
    reason: promotionReason(
      validation,
      candidateEval,
      aggregatePromotionGatePassed,
      aggregatePromotionGateReason,
      profileGate.reason,
      lineageGate.reason,
    ),
    decidedAt,
    decidedBy: "deterministic_gate",
    appliesToNewSessionsOnly: true,
  });

  const candidateRecord = registryRecordForCandidate(candidate, decidedAt, decision.decision);
  const decisionRecord = registryRecordForDecision(decision, decidedAt);
  const savedCandidate = saveOptimizerRegistryRecord(input.config, candidateRecord, cwd);
  const savedDecision = saveOptimizerRegistryRecord(input.config, decisionRecord, cwd);

  if (decision.decision !== "promote") {
    return CandidatePromotionResultSchema.parse({
      promoted: false,
      candidatePatchId: candidate.candidatePatchId,
      decision,
      ...(previousPointer === undefined ? {} : { previousPointer }),
      checkpointPath,
      registryRecordIds: [savedCandidate.registryRecordId, savedDecision.registryRecordId],
    });
  }

  const activePointer = promoteActiveOptimizerPointer(
    input.config,
    {
      activeModelProfileId: candidate.modelProfileId,
      activeCodebaseProfileId: candidate.codebaseProfileId,
      ...(candidate.codebaseRootFingerprint === undefined ? {} : { activeCodebaseRootFingerprint: candidate.codebaseRootFingerprint }),
      activePolicyId: candidate.policyId,
      promotedAt: decidedAt,
    },
    cwd,
  );

  return CandidatePromotionResultSchema.parse({
    promoted: true,
    candidatePatchId: candidate.candidatePatchId,
    decision,
    ...(previousPointer === undefined ? {} : { previousPointer }),
    activePointer,
    checkpointPath,
    registryRecordIds: [savedCandidate.registryRecordId, savedDecision.registryRecordId],
  });
};

export const rollbackOptimizerPromotion = (
  input: RollbackOptimizerPromotionInput,
): ActiveOptimizerPointer | undefined => {
  const cwd = input.cwd ?? process.cwd();
  const checkpointPath = input.checkpointPath ?? latestCheckpointPath(input.config, cwd);
  if (checkpointPath == null) {
    return undefined;
  }
  const checkpoint = PromotionCheckpointSchema.parse(JSON.parse(readFileSync(checkpointPath, "utf8")) as unknown);
  if (checkpoint.previousPointer == null) {
    return undefined;
  }
  return promoteActiveOptimizerPointer(input.config, checkpoint.previousPointer, cwd);
};

export const monitorPostPromotionRollback = (
  input: MonitorPostPromotionInput,
): PostPromotionMonitorResult => {
  const cwd = input.cwd ?? process.cwd();
  const promotion = CandidatePromotionResultSchema.parse(input.promotion) as CandidatePromotionResult;
  const promotedPolicyId = promotion.decision.policyId;
  const candidatePatchId = promotion.decision.candidatePatchId;
  const scorecards = (input.evalScorecards ?? []).map((scorecard) => EvalScorecardSchema.parse(scorecard));
  const runResults = (input.evalRunResults ?? []).map((run) => EvalRunResultSchema.parse(run));
  const evidenceBundles = (input.evidenceBundles ?? []).map((bundle) => CandidateEvidenceBundleSchema.parse(bundle));
  const signals = [
    ...signalsFromScorecards(scorecards, promotedPolicyId, candidatePatchId),
    ...signalsFromRunResults(runResults, promotedPolicyId, candidatePatchId),
    ...signalsFromEvidenceBundles(evidenceBundles, promotedPolicyId, candidatePatchId),
  ];
  const regressionDetected = signals.some((signal) => signal.severity === "failure" || signal.severity === "critical");
  const rollbackMode = input.rollbackMode ?? "perform";
  const rollbackRequested = promotion.promoted && regressionDetected && rollbackMode !== "disabled";
  const rollbackPointer = rollbackRequested && rollbackMode === "perform"
    ? rollbackOptimizerPromotion({
      config: input.config,
      cwd,
      ...(promotion.checkpointPath === undefined ? {} : { checkpointPath: promotion.checkpointPath }),
    })
    : undefined;

  return PostPromotionMonitorResultSchema.parse({
    promotedPolicyId,
    ...(candidatePatchId === undefined ? {} : { candidatePatchId }),
    regressionDetected,
    rollbackRequested,
    rolledBack: rollbackPointer !== undefined,
    signals,
    ...(promotion.checkpointPath === undefined ? {} : { checkpointPath: promotion.checkpointPath }),
    ...(promotion.previousPointer === undefined ? {} : { previousPointer: promotion.previousPointer }),
    ...(rollbackPointer === undefined ? {} : { rollbackPointer }),
  }) as PostPromotionMonitorResult;
};

const PromotionCheckpointSchema = z.object({
  candidatePatchId: z.string().min(1),
  createdAt: z.string(),
  previousPointer: z.object({
    activeModelProfileId: z.string().optional(),
    activeCodebaseProfileId: z.string().optional(),
    activeCodebaseRootFingerprint: z.string().optional(),
    activePolicyId: z.string().optional(),
    promotedAt: z.string().optional(),
  }).strict().optional(),
}).strict();

const writePromotionCheckpoint = (
  config: BagConfig,
  cwd: string,
  candidatePatchId: string,
  previousPointer: ActiveOptimizerPointer | undefined,
  createdAt: string,
): string => {
  const checkpointsDir = optimizerRegistryCheckpointsDir(config, cwd);
  mkdirSync(checkpointsDir, { recursive: true });
  const path = join(checkpointsDir, `${safePathSegment(createdAt)}.${safePathSegment(candidatePatchId)}.json`);
  writeFileSync(
    path,
    `${JSON.stringify({
      candidatePatchId,
      createdAt,
      ...(previousPointer === undefined
        ? {}
        : {
            previousPointer: {
              ...(previousPointer.activeModelProfileId === undefined ? {} : { activeModelProfileId: previousPointer.activeModelProfileId }),
              ...(previousPointer.activeCodebaseProfileId === undefined ? {} : { activeCodebaseProfileId: previousPointer.activeCodebaseProfileId }),
              ...(previousPointer.activeCodebaseRootFingerprint === undefined
                ? {}
                : { activeCodebaseRootFingerprint: previousPointer.activeCodebaseRootFingerprint }),
              ...(previousPointer.activePolicyId === undefined ? {} : { activePolicyId: previousPointer.activePolicyId }),
              ...(previousPointer.promotedAt === undefined ? {} : { promotedAt: previousPointer.promotedAt }),
            },
          }),
    }, null, 2)}\n`,
    "utf8",
  );
  return path;
};

const latestCheckpointPath = (config: BagConfig, cwd: string): string | undefined => {
  const dir = optimizerRegistryCheckpointsDir(config, cwd);
  if (!existsSync(dir)) {
    return undefined;
  }
  const latest = readdirSync(dir)
    .filter((file) => file.endsWith(".json"))
    .sort((left, right) => left.localeCompare(right))
    .at(-1);
  return latest == null ? undefined : join(dir, latest);
};

const registryRecordForCandidate = (
  candidate: CandidatePatch,
  timestamp: string,
  decision: PromotionDecision["decision"],
): OptimizerRegistryRecord => ({
  registryRecordId: stableId("registry", candidate.candidatePatchId),
  recordKind: "candidate_patch",
  schemaVersion: "optimizer-schema.v1",
  recordVersion: "record.v1",
  status: decision === "promote" ? "promoted" : "rejected",
  createdAt: timestamp,
  updatedAt: timestamp,
  labels: ["candidate-promotion"],
  payload: candidate,
});

const registryRecordForDecision = (decision: PromotionDecision, timestamp: string): OptimizerRegistryRecord => ({
  registryRecordId: stableId("registry", decision.promotionDecisionId),
  recordKind: "promotion_decision",
  schemaVersion: "optimizer-schema.v1",
  recordVersion: "record.v1",
  status: decision.decision === "promote" ? "promoted" : "rejected",
  createdAt: timestamp,
  updatedAt: timestamp,
  labels: ["candidate-promotion"],
  payload: decision,
});

const promotionReason = (
  validation: CandidateValidationResult,
  candidateEval: EvalScorecard,
  promotionGatePassed: boolean | undefined,
  promotionGateReason: string | undefined,
  profileGateReason: string | undefined,
  lineageGateReason: string | undefined,
): string => {
  if (!validation.valid) {
    return `Candidate rejected because validation failed: ${validation.issues.map((issue) => issue.code).join(", ")}`;
  }
  if (!candidateEval.passed || candidateEval.criticalRegressionVeto.vetoed) {
    return "Candidate rejected because eval gates failed or critical regression vetoed promotion.";
  }
  if (profileGateReason !== undefined) {
    return `Candidate rejected because codebase profile gate failed: ${profileGateReason}`;
  }
  if (promotionGatePassed === false) {
    return promotionGateReason == null
      ? "Candidate rejected because aggregate promotion gates failed."
      : `Candidate rejected because aggregate promotion gates failed: ${promotionGateReason}`;
  }
  if (lineageGateReason !== undefined) {
    return `Candidate rejected because artifact lineage gates failed: ${lineageGateReason}`;
  }
  return "Candidate passed validation and candidate eval gates.";
};

const promotionLineageGate = (
  candidate: CandidatePatch,
  lineageDecisionInput: OptimizerArtifactLineageDecision | undefined,
): { passed: boolean; reason?: string } => {
  if (lineageDecisionInput === undefined) {
    return {
      passed: false,
      reason: "artifact lineage decision is required before active pointer update",
    };
  }
  const lineageDecision = OptimizerArtifactLineageDecisionSchema.parse(lineageDecisionInput);
  if (lineageDecision.candidatePatchId !== candidate.candidatePatchId) {
    return {
      passed: false,
      reason: `lineage candidatePatchId ${lineageDecision.candidatePatchId} does not match candidate ${candidate.candidatePatchId}`,
    };
  }
  if (!lineageDecision.promotionAllowed) {
    return {
      passed: false,
      reason: lineageDecision.blockingGateIds.join(", "),
    };
  }
  return { passed: true };
};

const promotionProfileGate = (
  candidate: CandidatePatch,
  candidateEval: EvalScorecard,
): { passed: boolean; reason?: string } => {
  const evalContext = candidateEval.candidate.context as EvalScorecard["candidate"]["context"] & {
    codebaseRootFingerprint?: string;
  };
  if (candidate.policyId !== evalContext.policyId) {
    return { passed: false, reason: `candidate policyId ${candidate.policyId} does not match eval policyId ${evalContext.policyId}` };
  }
  if (candidate.modelProfileId !== evalContext.modelProfileId) {
    return {
      passed: false,
      reason: `candidate modelProfileId ${candidate.modelProfileId} does not match eval modelProfileId ${evalContext.modelProfileId}`,
    };
  }
  if (candidate.codebaseProfileId !== evalContext.codebaseProfileId) {
    return {
      passed: false,
      reason: `candidate codebaseProfileId ${candidate.codebaseProfileId} does not match eval codebaseProfileId ${evalContext.codebaseProfileId}`,
    };
  }
  if (
    candidate.codebaseRootFingerprint !== undefined &&
    evalContext.codebaseRootFingerprint !== undefined &&
    candidate.codebaseRootFingerprint !== evalContext.codebaseRootFingerprint
  ) {
    return { passed: false, reason: "candidate codebase profile fingerprint does not match eval fingerprint" };
  }
  return { passed: true };
};

const promotionGateReason = (reasons: readonly string[]): string | undefined => {
  const unique = [...new Set(reasons.filter((reason) => reason.length > 0))];
  return unique.length === 0 ? undefined : unique.join("; ");
};

const signalsFromScorecards = (
  scorecards: readonly EvalScorecard[],
  promotedPolicyId: string,
  candidatePatchId: string | undefined,
): PostPromotionRegressionSignal[] =>
  scorecards
    .filter((scorecard) => scorecard.candidate.context.policyId === promotedPolicyId)
    .filter((scorecard) => !scorecard.passed || scorecard.criticalRegressionVeto.vetoed)
    .map((scorecard) => PostPromotionRegressionSignalSchema.parse({
      signalId: stableId("post-promotion", "scorecard", scorecard.scorecardId),
      source: "eval_scorecard",
      severity: scorecard.criticalRegressionVeto.vetoed ? "critical" : "failure",
      policyId: promotedPolicyId,
      ...(candidatePatchId === undefined ? {} : { candidatePatchId }),
      scorecardId: scorecard.scorecardId,
      reason: scorecard.criticalRegressionVeto.vetoed
        ? `critical regression veto in ${scorecard.scorecardId}: ${scorecard.criticalRegressionVeto.regressions.map((regression) => regression.reason).join("; ")}`
        : `scorecard ${scorecard.scorecardId} failed with aggregateScore=${scorecard.aggregateScore}`,
    }));

const signalsFromRunResults = (
  runResults: readonly EvalRunResult[],
  promotedPolicyId: string,
  candidatePatchId: string | undefined,
): PostPromotionRegressionSignal[] =>
  runResults
    .filter((run) => run.context.policyId === promotedPolicyId)
    .filter((run) => run.status !== "passed" || run.assertionResults.some((assertion) => !assertion.passed && assertion.severity === "critical"))
    .map((run) => PostPromotionRegressionSignalSchema.parse({
      signalId: stableId("post-promotion", "eval-run", run.runResultId),
      source: "eval_run",
      severity: run.assertionResults.some((assertion) => !assertion.passed && assertion.severity === "critical")
        ? "critical"
        : "failure",
      policyId: promotedPolicyId,
      ...(candidatePatchId === undefined ? {} : { candidatePatchId }),
      runResultId: run.runResultId,
      reason: `eval run ${run.runResultId} for ${run.evalCaseId} ended with status ${run.status}`,
    }));

const signalsFromEvidenceBundles = (
  evidenceBundles: readonly CandidateEvidenceBundle[],
  promotedPolicyId: string,
  candidatePatchId: string | undefined,
): PostPromotionRegressionSignal[] =>
  evidenceBundles.flatMap((bundle) =>
    bundle.observations
      .filter((observation) => observation.lineage.policyIds.includes(promotedPolicyId))
      .filter((observation) =>
        observation.source === "trace_failure" ||
        observation.source === "eval_run" ||
        observation.source === "eval_scorecard"
      )
      .filter((observation) => observation.severity === "high" || observation.severity === "critical")
      .map((observation) => signalFromObservation(observation, promotedPolicyId, candidatePatchId)),
  );

const signalFromObservation = (
  observation: CandidateEvidenceObservation,
  promotedPolicyId: string,
  candidatePatchId: string | undefined,
): PostPromotionRegressionSignal =>
  PostPromotionRegressionSignalSchema.parse({
    signalId: stableId("post-promotion", "evidence", observation.observationId),
    source: observation.source === "trace_failure" ? "trace_evidence" : observation.source,
    severity: observation.severity === "critical" ? "critical" : "failure",
    policyId: promotedPolicyId,
    ...(candidatePatchId === undefined ? {} : { candidatePatchId }),
    ...(observation.scorecardIds[0] === undefined ? {} : { scorecardId: observation.scorecardIds[0] }),
    ...(observation.runResultIds[0] === undefined ? {} : { runResultId: observation.runResultIds[0] }),
    ...(observation.traceIds[0] === undefined ? {} : { traceId: observation.traceIds[0] }),
    reason: `${observation.source} ${observation.observationId}: ${observation.title}`,
  });

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 160) || "promotion.empty";

const safePathSegment = (value: string): string =>
  value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 160) || "checkpoint";
