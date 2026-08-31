import { describe, expect, test } from "bun:test";
import {
  buildFrozenCandidateRecord,
  type FrozenCandidateRecord,
  type HoldoutAggregateProof,
} from "../src/optimizer/frozen-candidate";
import {
  evaluatePromotionWorkflow,
  runPromotionWorkflowCommand,
  type CanonicalEpochPromotionWorkflow,
  type PromotionWorkflowArtifacts,
} from "../src/optimizer/promotion-workflow";
import type { OperatorApprovalEvidenceRecord, PostPromotionMonitorWindowEvidenceRecord, RollbackCheckpointProofRecord } from "../src/optimizer/promotion-evidence-contracts";
import type { CandidatePatch } from "../src/optimizer/types";
import type { RealAcpCorpusRunManifest } from "../src/replay/real-acp-runner";

const graphId = "blocker-closure-v1";
const selectionHash = "a49f7e68fb";
const planSetHash = "d5e2f05f89";
const epochId = `evidence-epoch.${graphId}.${selectionHash}`;
const now = "2026-05-05T12:30:00.000Z";
const snapshotPath = ".codex/plan-graphs/blocker-closure-v1/snapshot.json";
const releaseProofPath = ".bag/evidence/release-proof.json";
const candidatePatchId = "candidate.workflow.good";
const promotionDecisionId = "promotion.workflow.good";

const candidate: CandidatePatch = {
  candidatePatchId,
  policyId: "policy.workflow.candidate",
  baselinePolicyId: "policy.workflow.baseline",
  candidatePolicyId: "policy.workflow.candidate",
  modelProfileId: "model.workflow",
  codebaseProfileId: "codebase.workflow",
  clientProfileId: "client.workflow",
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: "policy.workflow.candidate",
    allowedJsonPointers: ["/resultStyleVersion"],
  },
  operations: [{
    op: "replace",
    path: "/resultStyleVersion",
    value: "result.workflow.v2",
  }],
  rationale: "Workflow promotion test candidate.",
  createdAt: now,
  sourceTraceIds: ["trace.workflow"],
};

describe("optimizer monitored promotion workflow", () => {
  test("allows preview only when every epoch, contract, holdout, consumer, and quality proof is bound", () => {
    const decision = evaluatePromotionWorkflow({
      graphId,
      selectionHash,
      candidatePatchId,
      promotionDecisionId,
      now,
      artifacts: passingArtifacts(),
    });

    expect(decision.promotionReady).toBe(true);
    expect(decision.actionAllowed).toBe(true);
    expect(decision.failClosed).toBe(false);
    expect(decision.blockers).toEqual([]);
    expect(decision.consumedEvidence.realAcpRunIds).toEqual(["real-acp-run.workflow.real_consumer"]);
  });

  test("blocks wrong graph and stale current slots", () => {
    const artifacts = passingArtifacts({
      canonicalEpoch: {
        ...canonicalEpoch(),
        graphId: "other-graph",
        driftStatus: "blocked",
        stalePaths: [".bag/evidence/release-proof.json"],
      },
      releaseProof: {
        ...releaseProof(),
        graphId: "other-graph",
        selectionHash: "other-selection",
      },
    });

    const decision = evaluatePromotionWorkflow({
      graphId,
      selectionHash,
      candidatePatchId,
      promotionDecisionId,
      now,
      artifacts,
    });

    expect(messages(decision)).toContain("canonical epoch targets graph other-graph, not blocker-closure-v1");
    expect(messages(decision)).toContain("canonical epoch driftStatus=blocked; stale current slots: .bag/evidence/release-proof.json");
    expect(messages(decision)).toContain("release proof targets selection other-selection, not a49f7e68fb");
    expect(decision.promotionReady).toBe(false);
  });

  test("blocks wrong candidate bindings across frozen, holdout, and contract evidence", () => {
    const artifacts = passingArtifacts({
      frozenCandidate: frozenCandidate("candidate.workflow.other"),
      holdoutProof: holdoutProof("candidate.workflow.other"),
      operatorApproval: operatorApproval("candidate.workflow.other"),
    });

    const decision = evaluatePromotionWorkflow({
      graphId,
      selectionHash,
      candidatePatchId,
      promotionDecisionId,
      now,
      artifacts,
    });

    expect(messages(decision)).toContain("frozen candidate targets candidate candidate.workflow.other, not candidate.workflow.good");
    expect(messages(decision)).toContain("hidden holdout proof targets candidate candidate.workflow.other, not candidate.workflow.good");
    expect(messages(decision)).toContain("promotion evidence contract blocker: operator approval targets candidate candidate.workflow.other, not candidate.workflow.good");
    expect(decision.promotionReady).toBe(false);
    expect(decision.failClosed).toBe(true);
  });

  test("blocks missing approval, checkpoint, monitor, frozen candidate, and holdout evidence", () => {
    const decision = evaluatePromotionWorkflow({
      graphId,
      selectionHash,
      candidatePatchId,
      promotionDecisionId,
      now,
      artifacts: passingArtifacts({
        frozenCandidate: undefined,
        holdoutProof: undefined,
        operatorApproval: undefined,
        rollbackCheckpointProof: undefined,
        monitorWindow: undefined,
      }),
    });

    expect(messages(decision)).toContain("missing frozen candidate artifact: .bag/evidence/optimizer/frozen-candidate.json");
    expect(messages(decision)).toContain("missing hidden holdout aggregate proof: .bag/evidence/optimizer/holdout-aggregate-proof.json");
    expect(messages(decision)).toContain("promotion evidence contract blocker: missing operator approval evidence");
    expect(messages(decision)).toContain("promotion evidence contract blocker: missing rollback checkpoint proof evidence");
    expect(messages(decision)).toContain("promotion evidence contract blocker: missing post-promotion monitor-window proof evidence");
  });

  test("blocks current empty_edits quality and missing real_consumer proof", () => {
    const decision = evaluatePromotionWorkflow({
      graphId,
      selectionHash,
      candidatePatchId,
      promotionDecisionId,
      now,
      artifacts: passingArtifacts({
        realAcpManifests: [realAcpManifest({
          executionMode: "headless_acp",
          realConsumerMutationAllowed: false,
          status: "failed",
          verifierStatus: "failed",
          changedFiles: [],
          codingProgressClass: "empty_edits",
        })],
      }),
    });

    expect(messages(decision)).toContain("current quality evidence real-acp-run.workflow.headless_acp has codingProgressClass=empty_edits");
    expect(messages(decision)).toContain("missing real_consumer ACP evidence with non-empty edit and verifier pass");
    expect(decision.promotionReady).toBe(false);
  });

  test("blocks failed or short monitor windows", () => {
    const decision = evaluatePromotionWorkflow({
      graphId,
      selectionHash,
      candidatePatchId,
      promotionDecisionId,
      now,
      artifacts: passingArtifacts({
        monitorWindow: monitorWindow({
          observedWindowMs: 1_000,
          requiredWindowMs: 14_400_000,
          regressionDetected: true,
          signals: [{
            signalId: "signal.workflow.failure",
            severity: "failure",
            source: "eval_scorecard",
            reason: "post-promotion regression",
          }],
        }),
      }),
    });

    expect(messages(decision)).toContain("promotion evidence contract blocker: post-promotion monitor window observed 1000ms but requires 14400000ms");
    expect(messages(decision)).toContain("promotion evidence contract blocker: post-promotion monitor window detected regressions");
    expect(messages(decision)).toContain("promotion evidence contract blocker: post-promotion monitor window contains failure or critical signals");
  });

  test("approve, promote, monitor, and rollback actions fail closed when preview is red", () => {
    for (const action of ["approve", "promote", "monitor", "rollback"] as const) {
      const result = runPromotionWorkflowCommand({
        action,
        graphId,
        selectionHash,
        candidatePatchId,
        promotionDecisionId,
        now,
        artifacts: passingArtifacts({ operatorApproval: undefined }),
      });

      expect(result.ok).toBe(false);
      expect(result.exitCode).toBe(1);
      expect(messages(result.decision)).toContain(`${action} is blocked until promotionReady is true for blocker-closure-v1/a49f7e68fb.`);
    }
  });
});

const messages = (decision: ReturnType<typeof evaluatePromotionWorkflow>): string[] =>
  decision.blockers.map((blocker) => blocker.message);

const passingArtifacts = (overrides: Partial<PromotionWorkflowArtifacts> = {}): PromotionWorkflowArtifacts => ({
  canonicalEpoch: canonicalEpoch(),
  releaseProof: releaseProof(),
  optimizerGateSuite: optimizerGateSuite(),
  frozenCandidate: frozenCandidate(),
  holdoutProof: holdoutProof(),
  operatorApproval: operatorApproval(),
  rollbackCheckpointProof: rollbackCheckpointProof(),
  monitorWindow: monitorWindow(),
  realAcpManifests: [realAcpManifest()],
  stabilityScorecards: [],
  ...overrides,
});

const canonicalEpoch = (): CanonicalEpochPromotionWorkflow => ({
  schemaVersion: "evidence-command.epoch.v1",
  epochId,
  graphId,
  selectionHash,
  generatedAt: now,
  sourceGraph: {
    graphId,
    selectionHash,
    planSetHash,
    snapshotPath,
    generatedAt: "2026-05-05T12:22:33.090Z",
    selectedPlanPaths: [".codex/plans/blocker-closure-v1/06-monitored-promotion-workflow.plan.md"],
  },
  driftStatus: "passed",
  promotionReady: false,
  stalePaths: [],
  currentEvidencePaths: [CANONICAL_EPOCH_PATH],
  candidateInputPaths: [".bag/evidence/index.jsonl"],
});

const releaseProof = () => ({
  schemaVersion: "local-evidence-release-proof.v1" as const,
  releaseProofId: "release-proof.blocker-closure-v1",
  graphId,
  selectionHash,
  generatedAt: now,
  proofMode: "current_graph" as const,
  sourceGraph: {
    graphId,
    selectionHash,
    planSetHash,
    snapshotPath,
    generatedAt: "2026-05-05T12:22:33.090Z",
    dependencyOverlay: [],
    selectedPlanPaths: [".codex/plans/blocker-closure-v1/06-monitored-promotion-workflow.plan.md"],
  },
  commandOutputs: {},
  artifactHashes: [],
  validation: {
    planGraphSnapshot: "passed",
    evidenceIndexCommand: "passed",
    scorecardsCommand: "passed",
    optimizerGatesCommand: "passed",
  },
  optimizerDecision: {
    candidateGeneration: "allowed_as_scoped_dry_run",
    autoPromotion: "allowed",
    promotionReady: false,
    blockingReasons: [],
  },
  primaryOutputs: [releaseProofPath],
  nextExecutionFrontier: [],
});

const optimizerGateSuite = () => ({
  schemaVersion: "local-evidence-optimizer-gate-suite.v1" as const,
  optimizerGateSuiteId: "optimizer-gate-suite.blocker-closure-v1",
  graphId,
  generatedAt: now,
  sourceEvidenceIndex: ".bag/evidence/index.jsonl",
  sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
  contracts: [{
    contractId: "optimizer.workflow",
    jsonPath: ".bag/evidence/optimizer/workflow.json",
    markdownPath: "docs/optimizer-workflow.md",
    primaryUse: "promotion workflow test",
  }],
  currentDecision: {
    candidateGeneration: "allowed_as_scoped_dry_run",
    autoPromotion: "allowed" as const,
    promotionReady: true,
    blockingReasons: [],
  },
  mustFailClosedOn: ["missing operator approval evidence"],
  policySeparation: {
    dimensions: ["modelProfileId", "codebaseProfileId", "modelCodebasePolicyId"] as const,
    principle: "Promotion applies only to the exact evaluated tuple.",
  },
});

const frozenCandidate = (nextCandidatePatchId = candidatePatchId): FrozenCandidateRecord =>
  buildFrozenCandidateRecord({
    candidate: { ...candidate, candidatePatchId: nextCandidatePatchId },
    graphId,
    selectionHash,
    epochId,
    frozenAt: now,
    visibleInputBindings: [
      visibleBinding("train"),
      visibleBinding("dev"),
    ],
  });

const visibleBinding = (split: "train" | "dev") => ({
  bindingId: `binding.workflow.${split}`,
  sourceKind: "eval_scorecard" as const,
  sourceArtifactId: `scorecard.workflow.${split}`,
  split,
  contentHash: `sha256:${split}`,
  optimizerInputAllowed: true as const,
  includedEvalCaseIds: [`eval.workflow.${split}`],
});

const holdoutProof = (nextCandidatePatchId = candidatePatchId): HoldoutAggregateProof => ({
  schemaVersion: "optimizer-holdout-aggregate-proof.v1",
  proofId: "holdout-proof.workflow",
  frozenCandidateId: frozenCandidate(nextCandidatePatchId).frozenCandidateId,
  candidatePatchId: nextCandidatePatchId,
  graphId,
  selectionHash,
  epochId,
  createdAt: now,
  purpose: "holdout_final",
  status: "passed",
  evaluationOnly: true,
  aggregateOnly: true,
  optimizerInputAllowed: false,
  rawHoldoutContentIncluded: false,
  sourceScorecardIds: ["scorecard.workflow.holdout"],
  sourceReplayExportIds: ["replay-export.workflow.holdout"],
  sourceRunIds: ["real-acp-run.workflow.real-consumer"],
  metrics: {
    scorecardCount: 1,
    passedScorecardCount: 1,
    failedScorecardCount: 0,
    candidateAggregateScore: 1,
    criticalRegressionCount: 0,
    baselineRunCount: 1,
    candidateRunCount: 1,
    hiddenHoldoutCaseCount: 1,
  },
});

const sourceGraph = {
  graphId,
  selectionHash,
  planSetHash,
  snapshotPath,
};

const operatorApproval = (nextCandidatePatchId = candidatePatchId): OperatorApprovalEvidenceRecord => ({
  schemaVersion: "optimizer-operator-approval.v1",
  graphId,
  selectionHash,
  planSetHash,
  evidenceEpochId: epochId,
  sourceGraph,
  releaseProofRef: { path: releaseProofPath },
  candidatePatchId: nextCandidatePatchId,
  promotionDecisionId,
  generatedAt: now,
  approvalId: "approval.workflow",
  approvalKind: "promotion",
  approved: true,
  approvedBy: "operator.workflow",
  approvedAt: now,
  expiresAt: "2026-05-06T12:30:00.000Z",
  notes: [],
});

const rollbackCheckpointProof = (): RollbackCheckpointProofRecord => ({
  schemaVersion: "optimizer-rollback-checkpoint-proof.v1",
  graphId,
  selectionHash,
  planSetHash,
  evidenceEpochId: epochId,
  sourceGraph,
  releaseProofRef: { path: releaseProofPath },
  candidatePatchId,
  promotionDecisionId,
  generatedAt: now,
  checkpointProofId: "checkpoint-proof.workflow",
  checkpointPath: ".bag/optimizer/checkpoints/workflow.json",
  checkpointSha256: "sha256:checkpoint",
  checkpointCreatedAt: now,
  restoreMode: "dry_run",
  restorable: true,
  rollbackCommand: ["bag", "optimizer", "rollback"],
});

const monitorWindow = (
  overrides: Partial<PostPromotionMonitorWindowEvidenceRecord> = {},
): PostPromotionMonitorWindowEvidenceRecord => ({
  schemaVersion: "optimizer-post-promotion-monitor-window.v1",
  graphId,
  selectionHash,
  planSetHash,
  evidenceEpochId: epochId,
  sourceGraph,
  releaseProofRef: { path: releaseProofPath },
  candidatePatchId,
  promotionDecisionId,
  generatedAt: now,
  monitorWindowId: "monitor-window.workflow",
  promotedPolicyId: candidate.policyId,
  startedAt: "2026-05-05T08:00:00.000Z",
  completedAt: "2026-05-05T12:30:00.000Z",
  requiredWindowMs: 14_400_000,
  observedWindowMs: 16_200_000,
  regressionDetected: false,
  rollbackRequested: false,
  rolledBack: false,
  checkpointPath: ".bag/optimizer/checkpoints/workflow.json",
  signals: [],
  ...overrides,
});

const realAcpManifest = (input: {
  executionMode?: "headless_acp" | "real_consumer";
  realConsumerMutationAllowed?: boolean;
  status?: "passed" | "failed";
  verifierStatus?: "passed" | "failed";
  changedFiles?: RealAcpCorpusRunManifest["taskResults"][number]["changedFiles"];
  codingProgressClass?: string;
} = {}): RealAcpCorpusRunManifest => {
  const executionMode = input.executionMode ?? "real_consumer";
  const status = input.status ?? "passed";
  const verifierStatus = input.verifierStatus ?? "passed";
  const changedFiles = input.changedFiles ?? [{
    path: "src/workflow.ts",
    changeKind: "modified" as const,
    beforeHash: "sha256:before",
    afterHash: "sha256:after",
  }];
  return {
    schemaVersion: "real-acp-corpus-run.v1",
    runId: `real-acp-run.workflow.${executionMode}`,
    taskPackId: "real-acp-run-corpus.task-pack.v1",
    createdAt: now,
    executionMode,
    dryRun: false,
    purpose: "development_eval",
    executor: {
      executorId: "real-acp.executor.workflow",
      executorVersion: "workflow.v1",
      kind: executionMode,
    },
    metadata: {
      model: {
        modelProfileId: candidate.modelProfileId,
        provider: "openai-compatible",
        model: "workflow-model",
        modelRole: "local",
        contextWindowTokens: 128000,
        toolCallingMode: "native",
      },
      codebase: {
        codebaseProfileId: candidate.codebaseProfileId,
        rootFingerprint: "sha256:workflow",
        languageSummary: "fixture",
        testRiskTier: "risk.workflow",
        protectedPathPolicy: "fixture only",
      },
      client: {
        clientProfileId: "client.workflow",
        clientName: "workflow real consumer",
        clientVersion: "v1",
        transport: executionMode === "real_consumer" ? "stdio" : "in_process",
        acpConsumerCapabilities: {
          filesystem: true,
          terminal: true,
        },
      },
      profile: {
        policyId: candidate.policyId,
        optimizerProfileId: "optimizer.workflow",
        verificationPolicyVersion: "verification.workflow.v1",
        resultStyleVersion: "result.workflow.v1",
        canonicalToolVersion: "canonical.workflow.v1",
        renderedToolVersion: "rendered.workflow.v1",
      },
    },
    safety: {
      workspaceIsolation: "per_task_materialized_fixture",
      currentRepoMutationRefused: true,
      realConsumerMutationAllowed: input.realConsumerMutationAllowed ?? executionMode === "real_consumer",
    },
    splitPolicy: {
      includeHoldout: false,
      visibleOptimizationSplits: ["train", "dev"],
      hiddenSplits: ["holdout"],
      optimizerLeakageRefused: true,
    },
    taskResults: [{
      schemaVersion: "real-acp-task-result.v1",
      runResultId: `real-acp-run.workflow.${executionMode}.task`,
      taskId: "real-acp.task.workflow",
      split: "train",
      optimizationAllowed: true,
      status,
      startedAt: now,
      completedAt: now,
      workspaceFingerprintBefore: "sha256:before",
      workspaceFingerprintAfter: changedFiles.length > 0 ? "sha256:after" : "sha256:before",
      changedFiles,
      route: {
        routeId: "route.workflow",
        selectedMode: "coding",
        reason: "workflow test",
        confidence: 1,
      },
      editStrategy: {
        strategyId: "edit.workflow",
        family: changedFiles.length > 0 ? "whole_file" : "none",
        selectedBy: "executor",
      },
      toolCalls: changedFiles.length > 0
        ? [{
          toolCallId: "tool.workflow.write",
          namespace: "acp.fs",
          name: "writeTextFile",
          status: "succeeded",
          sideEffectLevel: "write",
        }]
        : [],
      terminalCommands: [],
      verifier: {
        status: verifierStatus,
        policy: "required",
        commandIds: [],
      },
      repair: {
        attempted: false,
        status: "not_needed",
      },
      rollback: {
        attempted: false,
        status: "not_needed",
      },
      corrections: [],
      lineage: {
        taskId: "real-acp.task.workflow",
        runResultId: `real-acp-run.workflow.${executionMode}.task`,
        sourceTaskPackId: "real-acp-run-corpus.task-pack.v1",
      },
      telemetry: {
        codingProgressDiagnostic: {
          progressClass: input.codingProgressClass ?? (changedFiles.length > 0 ? "verified_edit" : "empty_edits"),
        },
      },
      redaction: {
        rawLocalStatus: "raw_local_only",
        optimizerSafe: true,
        excludedFromOptimizerReasons: [],
      },
      ...(status === "passed" ? {} : { failureReason: "workflow fixture failure" }),
    }],
    redactionHandoff: {
      rawLocal: {
        status: "raw_local_only",
        containsWorkspaceSnapshots: true,
        containsExecutorTelemetry: true,
        storageGuidance: "local only",
      },
      optimizerSafe: {
        status: "prepared",
        includedTaskResultIds: [`real-acp-run.workflow.${executionMode}.task`],
        excludedTaskResultIds: [],
        redactedFields: [],
        nextSteps: [],
      },
    },
    summary: {
      total: 1,
      passed: status === "passed" ? 1 : 0,
      failed: status === "failed" ? 1 : 0,
      skipped: 0,
      cancelled: 0,
      error: 0,
      holdout: 0,
    },
    manifestPath: `.bag/replay-corpus/real-acp-runs/real-acp-run.workflow.${executionMode}/manifest.json`,
  };
};

const CANONICAL_EPOCH_PATH = ".bag/evidence/canonical-epoch.json";
