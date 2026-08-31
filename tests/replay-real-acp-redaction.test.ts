import { describe, expect, test } from "bun:test";
import {
  RealAcpCorpusRunManifestSchema,
  type RealAcpCorpusRunManifest,
  type RealAcpRunMetadata,
  type RealAcpTaskRunResult,
} from "../src/replay/real-acp-runner";
import {
  realAcpCodingCorpusTaskPack,
  type RealAcpCorpusTask,
} from "../src/replay/real-acp-task-pack";
import {
  RealAcpReplayExportManifestSchema,
  assertRealAcpReplayExportSafeForOptimizerInput,
  createRealAcpReplayExportManifest,
  realAcpTaskResultToReplayCase,
} from "../src/replay/real-acp-redaction";
import {
  assessFrozenCandidateVisibleEvaluation,
  buildFrozenCandidateRecord,
  buildHoldoutAggregateProof,
} from "../src/optimizer/frozen-candidate";
import type { CandidatePatch } from "../src/optimizer/types";

const createdAt = "2026-05-04T00:00:00.000Z";

const metadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.redaction",
    provider: "simulated",
    model: "simulated-acp-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.redaction",
    rootFingerprint: "sha256:redaction-fixture",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.redaction",
    protectedPathPolicy: "Only task fixture paths may be changed.",
  },
  client: {
    clientProfileId: "client.real-acp.redaction",
    clientName: "Simulated ACP harness",
    clientVersion: "v1",
    transport: "simulated",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.redaction",
    optimizerProfileId: "optimizer.real-acp.redaction",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
};

const visibleTask = realAcpCodingCorpusTaskPack.tasks.find((task) => task.split === "train");
const holdoutTask = realAcpCodingCorpusTaskPack.tasks.find((task) => task.split === "holdout");

if (visibleTask === undefined || holdoutTask === undefined) {
  throw new Error("real ACP fixture task pack must include visible and holdout tasks");
}

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.real-acp.redaction",
  policyId: metadata.profile.policyId,
  baselinePolicyId: "policy.real-acp.redaction.baseline",
  candidatePolicyId: metadata.profile.policyId,
  modelProfileId: metadata.model.modelProfileId,
  codebaseProfileId: metadata.codebase.codebaseProfileId,
  clientProfileId: metadata.client.clientProfileId,
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: metadata.profile.policyId,
    allowedJsonPointers: ["/resultStyleVersion"],
  },
  operations: [
    {
      op: "replace",
      path: "/resultStyleVersion",
      value: "result.real-acp.v2",
    },
  ],
  rationale: "Candidate bound to redacted real ACP aggregate proof.",
  createdAt,
  sourceTraceIds: [],
};

const resultForTask = (
  task: RealAcpCorpusTask,
  overrides: Partial<RealAcpTaskRunResult> = {},
): RealAcpTaskRunResult => ({
  schemaVersion: "real-acp-task-result.v1",
  runResultId: `real-acp-run.redaction.${task.taskId.replaceAll(".", "-")}`,
  taskId: task.taskId,
  split: task.split,
  optimizationAllowed: task.optimizationAllowed,
  status: "passed",
  startedAt: createdAt,
  completedAt: createdAt,
  workspaceFingerprintBefore: "sha256:before",
  workspaceFingerprintAfter: "sha256:after",
  changedFiles: [
    {
      path: task.expectedOutcome.expectedChangedPaths[0] ?? "src/example.ts",
      changeKind: "modified",
      beforeHash: "sha256:before-file",
      afterHash: "sha256:after-file",
    },
  ],
  route: {
    routeId: `route.${task.taskId.replaceAll(".", "-")}`,
    selectedMode: task.primaryLabel === "cancellation" ? "cancelled" : "coding",
    reason: "Simulated route",
    confidence: 1,
  },
  editStrategy: {
    strategyId: "edit.whole-file.acp-write.v1",
    family: task.expectedOutcome.mutation === "no_change" ? "none" : "whole_file",
    selectedBy: task.expectedOutcome.mutation === "no_change" ? "not_applicable" : "optimizer_policy",
  },
  toolCalls: [
    {
      toolCallId: `tool.${task.taskId.replaceAll(".", "-")}.write`,
      namespace: "bag.acp",
      name: "fs.write",
      status: "succeeded",
      sideEffectLevel: "write",
    },
  ],
  terminalCommands: [
    {
      commandId: `cmd.${task.taskId.replaceAll(".", "-")}`,
      command: ["bun", "test", "/Users/satan/private/project/tests/secret.test.ts", "API_KEY=sk-123456789012345678901234"],
      status: "succeeded",
      exitCode: 0,
      durationMs: 12,
    },
  ],
  verifier: {
    status: "passed",
    policy: task.expectedOutcome.verification.policy,
    commandIds: [`cmd.${task.taskId.replaceAll(".", "-")}`],
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
    taskId: task.taskId,
    runResultId: `real-acp-run.redaction.${task.taskId.replaceAll(".", "-")}`,
    sourceTaskPackId: realAcpCodingCorpusTaskPack.taskPackId,
  },
  telemetry: {
    cwd: "/Users/satan/private/project",
    stdout: "raw terminal output includes export const privateValue = 42;",
    nested: {
      message: "failed with ghp_abcdefghijklmnopqrstuvwxyz1234567890",
      token: "sk-123456789012345678901234",
    },
  },
  redaction: {
    rawLocalStatus: "raw_local_only",
    optimizerSafe: task.optimizationAllowed,
    excludedFromOptimizerReasons: task.split === "holdout" ? ["hidden holdout split"] : [],
  },
  ...overrides,
});

const manifestWithResults = (
  taskResults: readonly RealAcpTaskRunResult[],
  purpose: RealAcpCorpusRunManifest["purpose"] = "development_eval",
): RealAcpCorpusRunManifest => RealAcpCorpusRunManifestSchema.parse({
  schemaVersion: "real-acp-corpus-run.v1",
  runId: "real-acp-run.redaction",
  taskPackId: realAcpCodingCorpusTaskPack.taskPackId,
  createdAt,
  executionMode: "dry_run",
  dryRun: true,
  purpose,
  executor: {
    executorId: "real-acp.executor.simulated",
    executorVersion: "simulated.v1",
    kind: "simulated",
  },
  metadata,
  safety: {
    workspaceIsolation: "per_task_materialized_fixture",
    currentRepoMutationRefused: true,
    realConsumerMutationAllowed: false,
  },
  splitPolicy: {
    includeHoldout: taskResults.some((result) => result.split === "holdout"),
    visibleOptimizationSplits: ["train", "dev"],
    hiddenSplits: ["holdout"],
    optimizerLeakageRefused: true,
  },
  taskResults,
  redactionHandoff: {
    rawLocal: {
      status: "raw_local_only",
      containsWorkspaceSnapshots: true,
      containsExecutorTelemetry: true,
      storageGuidance: "Keep raw artifacts local.",
    },
    optimizerSafe: {
      status: "prepared",
      includedTaskResultIds: taskResults.filter((result) => result.redaction.optimizerSafe).map((result) => result.runResultId),
      excludedTaskResultIds: taskResults.filter((result) => !result.redaction.optimizerSafe).map((result) => result.runResultId),
      redactedFields: ["workspaceSnapshots", "terminal stdout/stderr"],
      nextSteps: ["Export redacted replay records."],
    },
  },
  summary: {
    total: taskResults.length,
    passed: taskResults.filter((result) => result.status === "passed").length,
    failed: taskResults.filter((result) => result.status === "failed").length,
    skipped: taskResults.filter((result) => result.status === "skipped").length,
    cancelled: taskResults.filter((result) => result.status === "cancelled").length,
    error: taskResults.filter((result) => result.status === "error").length,
    holdout: taskResults.filter((result) => result.split === "holdout").length,
  },
});

describe("real ACP redaction replay export", () => {
  test("converts task results to redacted replay case records with lineage and scorecard metadata", () => {
    const manifest = manifestWithResults([
      resultForTask(visibleTask, {
        status: "failed",
        failureReason: "Verifier failed under /Users/satan/private/project with token=ghp_abcdefghijklmnopqrstuvwxyz1234567890",
      }),
    ]);

    const replayCase = realAcpTaskResultToReplayCase({
      manifest,
      result: manifest.taskResults[0]!,
      task: visibleTask,
    });

    expect(replayCase).toMatchObject({
      split: "train",
      optimizerInputAllowed: true,
      labels: visibleTask.labels,
      lineage: {
        runId: manifest.runId,
        taskId: visibleTask.taskId,
        runResultId: manifest.taskResults[0]!.runResultId,
        modelProfileId: metadata.model.modelProfileId,
        codebaseProfileId: metadata.codebase.codebaseProfileId,
        clientProfileId: metadata.client.clientProfileId,
        policyId: metadata.profile.policyId,
        optimizerProfileId: metadata.profile.optimizerProfileId,
      },
      outcome: {
        status: "failed",
        verifierStatus: "passed",
      },
      expectedOutcome: {
        mutation: visibleTask.expectedOutcome.mutation,
        verifierPolicy: visibleTask.expectedOutcome.verification.policy,
      },
    });
    expect(replayCase.sourceRefs).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceKind: "manifest", refId: manifest.runId }),
      expect.objectContaining({ sourceKind: "task_result", refId: manifest.taskResults[0]!.runResultId }),
      expect.objectContaining({ sourceKind: "changed_file", redactionStatus: "hash_only" }),
      expect.objectContaining({ sourceKind: "terminal_command", redactionStatus: "hash_only" }),
    ]));
    expect(replayCase.expectedOutcome.assertionSummaries[0]).toHaveProperty("expectedHash");

    const serialized = JSON.stringify(replayCase);
    expect(serialized).not.toContain("/Users/satan/private/project");
    expect(serialized).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz1234567890");
    expect(serialized).not.toContain("sk-123456789012345678901234");
    expect(serialized).not.toContain("export const privateValue = 42");
    expect(replayCase.redaction.secretReplacementCount).toBeGreaterThan(0);
    expect(replayCase.redaction.pathHashCount).toBeGreaterThan(0);
    expect(replayCase.redaction.omittedRawFieldCount).toBeGreaterThan(0);
  });

  test("exports optimizer input without hidden holdout cases", () => {
    const manifest = manifestWithResults([
      resultForTask(visibleTask),
      resultForTask(holdoutTask),
    ], "development_eval");

    const exportManifest = createRealAcpReplayExportManifest({
      manifest,
      taskPack: realAcpCodingCorpusTaskPack,
      purpose: "optimizer_input",
    });

    expect(RealAcpReplayExportManifestSchema.parse(exportManifest)).toEqual(exportManifest);
    expect(exportManifest.status).toBe("optimizer_safe");
    expect(exportManifest.includeHoldout).toBe(false);
    expect(exportManifest.cases).toHaveLength(1);
    expect(exportManifest.cases.every((replayCase) => replayCase.split !== "holdout")).toBe(true);
    expect(exportManifest.optimizerSelection.hiddenHoldoutReplayCaseIds).toEqual([]);
    expect(exportManifest.optimizerSelection.excludedReplayCaseIds).toEqual([]);
    expect(assertRealAcpReplayExportSafeForOptimizerInput(exportManifest)).toEqual(exportManifest);
  });

  test("allows holdout only for explicit evaluation-only holdout final export", () => {
    const manifest = manifestWithResults([
      resultForTask(visibleTask),
      resultForTask(holdoutTask),
    ], "holdout_final");

    expect(() => createRealAcpReplayExportManifest({
      manifest,
      taskPack: realAcpCodingCorpusTaskPack,
      purpose: "optimizer_input",
      includeHoldout: true,
    })).toThrow(/optimizer input must exclude hidden holdout/);

    expect(() => createRealAcpReplayExportManifest({
      manifest,
      taskPack: realAcpCodingCorpusTaskPack,
      purpose: "development_eval",
      status: "evaluation_only",
      includeHoldout: true,
    })).toThrow(/holdout export requires purpose holdout_final/);

    const exportManifest = createRealAcpReplayExportManifest({
      manifest,
      taskPack: realAcpCodingCorpusTaskPack,
      purpose: "holdout_final",
      status: "evaluation_only",
      includeHoldout: true,
    });

    expect(exportManifest.cases.map((replayCase) => replayCase.split).sort()).toEqual(["holdout", "train"]);
    expect(exportManifest.optimizerSelection.hiddenHoldoutReplayCaseIds).toHaveLength(1);
    expect(exportManifest.optimizerSelection.excludedReplayCaseIds).toHaveLength(1);
    expect(exportManifest.cases.find((replayCase) => replayCase.split === "holdout"))
      .toMatchObject({
        optimizerInputAllowed: false,
        optimizerExclusionReasons: ["hidden holdout split", "task is not optimizer-allowed"],
      });
    expect(() => assertRealAcpReplayExportSafeForOptimizerInput(exportManifest))
      .toThrow(/not optimizer input/);
  });

  test("feeds holdout final export into aggregate-only frozen-candidate proof without raw task leakage", () => {
    const manifest = manifestWithResults([
      resultForTask(visibleTask),
      resultForTask(holdoutTask),
    ], "holdout_final");
    const exportManifest = createRealAcpReplayExportManifest({
      manifest,
      taskPack: realAcpCodingCorpusTaskPack,
      purpose: "holdout_final",
      status: "evaluation_only",
      includeHoldout: true,
    });
    const frozenCandidate = buildFrozenCandidateRecord({
      candidate,
      graphId: "blocker-closure-v1",
      selectionHash: "a49f7e68fb",
      epochId: "epoch.real-acp.redaction",
      frozenAt: createdAt,
      visibleInputBindings: [
        {
          bindingId: "binding.real-acp.redaction.train",
          sourceKind: "replay_export",
          sourceArtifactId: "real-acp-replay-export.visible.train",
          split: "train",
          contentHash: "sha256:visible-train",
          optimizerInputAllowed: true,
        },
        {
          bindingId: "binding.real-acp.redaction.dev",
          sourceKind: "replay_export",
          sourceArtifactId: "real-acp-replay-export.visible.dev",
          split: "dev",
          contentHash: "sha256:visible-dev",
          optimizerInputAllowed: true,
        },
      ],
    });
    const visibleEvaluation = assessFrozenCandidateVisibleEvaluation({
      frozenCandidate,
      visibleScorecards: [],
    });
    const proof = buildHoldoutAggregateProof({
      frozenCandidate,
      visibleEvaluation,
      sourceReplayExportIds: [exportManifest.exportId],
      sourceRunIds: [exportManifest.sourceRunId],
      hiddenHoldoutCaseCount: exportManifest.summary.hiddenHoldoutCases,
      createdAt,
    });

    expect(proof.status).toBe("blocked");
    expect(proof.aggregateOnly).toBe(true);
    expect(proof.evaluationOnly).toBe(true);
    expect(proof.optimizerInputAllowed).toBe(false);
    expect(proof.metrics.hiddenHoldoutCaseCount).toBe(1);
    expect(JSON.stringify(proof)).not.toContain(holdoutTask.userPrompt);
    expect(JSON.stringify(proof)).not.toContain(holdoutTask.expectedOutcome.assertions[0]?.description ?? "missing");
  });
});
