import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { evaluateNoWritePromotionGate } from "../optimizer/no-write-gate";
import type { RealAcpReplayCaseRecord, RealAcpReplayExportManifest } from "./real-acp-redaction";
import type { RealAcpCorpusRunManifest, RealAcpTaskRunResult } from "./real-acp-runner";
import {
  buildNoWriteReplaySlice,
  buildNoWriteReplaySliceFromCorpus,
  noWriteValidationInputsFromReplaySlice,
} from "./no-write-slice";

describe("no-write replay slice builder", () => {
  test("builds deterministic no-write cases from visible ACP task-run records", () => {
    const slice = buildNoWriteReplaySlice({
      manifests: [manifestFixture([
        taskResultFixture("real-acp.task.simple-edit-greeting", "train", true),
        taskResultFixture("real-acp.task.refactor-price-format", "holdout", false),
        taskResultFixture("real-acp.task.greenfield-slugify", "dev", false),
      ])],
    });

    expect(slice.status).toEqual({
      totalRecordsSeen: 3,
      includedCases: 1,
      skippedHiddenHoldout: 1,
      skippedUnsafeOrExcluded: 1,
      skippedDuplicate: 0,
    });
    expect(slice.cases).toHaveLength(1);
    expect(slice.cases[0]).toMatchObject({
      runId: "real-acp-run.visible-test",
      taskId: "real-acp.task.simple-edit-greeting",
      modelProfileId: "model.real-acp.local-headless",
      codebaseProfileId: "codebase.real-acp.bleeding-agent",
      stopReason: "end_turn",
      editStrategyFamily: "none",
      expectedMutation: "edit_existing",
      expectedSideEffectLevel: "mutation",
      fileWrites: {
        changedFiles: [],
        fsWriteCount: 0,
      },
      terminalActivity: {
        terminalCreateCount: 0,
        terminalExitCount: 0,
        terminalCommandCount: 0,
      },
    });
    expect(slice.cases[0]!.evidenceRefs).toContain("real-acp-model-profile:model.real-acp.local-headless");
    expect(slice.cases[0]!.evidenceRefs).toContain("real-acp-codebase-profile:codebase.real-acp.bleeding-agent");

    const gate = evaluateNoWritePromotionGate({
      cases: noWriteValidationInputsFromReplaySlice(slice),
      requireEvidence: true,
    });
    expect(gate.status).toBe("block");
    expect(gate.checkedRecordIds).toEqual(["real-acp-run.visible-test.real-acp.task.simple-edit-greeting"]);
  });

  test("builds from replay cases and keeps structured expected side effects", () => {
    const slice = buildNoWriteReplaySlice({
      replayCases: [
        replayCaseFixture("real-acp.task.cancellation-mid-edit", "train", true),
        replayCaseFixture("real-acp.task.refactor-price-format", "holdout", true),
        replayCaseFixture("real-acp.task.protected-path-doc", "dev", false),
      ],
    });

    expect(slice.status).toMatchObject({
      totalRecordsSeen: 3,
      includedCases: 1,
      skippedHiddenHoldout: 1,
      skippedUnsafeOrExcluded: 1,
    });
    expect(slice.cases[0]).toMatchObject({
      sourceKind: "real_acp_replay_case",
      taskId: "real-acp.task.cancellation-mid-edit",
      expectedMutation: "no_change",
      expectedSideEffectLevel: "read",
      stopReason: "cancelled",
      terminalActivity: {
        terminalCreateCount: 0,
        terminalExitCount: 0,
        terminalCommandCount: 0,
      },
    });
    expect(slice.cases[0]!.validationInput.verifierSkippedJustification).toEqual({
      present: true,
      reason: "structured cancellation policy",
      policy: "must_skip",
    });
  });

  test("can keep only the latest visible run per task/profile for current promotion gates", () => {
    const oldResult = taskResultFixture("real-acp.task.simple-edit-greeting", "train", true);
    const newResult: RealAcpTaskRunResult = {
      ...oldResult,
      runResultId: "real-acp-run.visible-test-20260505.real-acp.task.simple-edit-greeting",
      startedAt: "2026-05-04T00:00:00.000Z",
      completedAt: "2026-05-04T00:00:00.000Z",
      terminalCommands: [{
        commandId: "cmd.latest.no-edit-failure",
        command: ["sh", "-c", "exit 1"],
        status: "failed",
        exitCode: 1,
        durationMs: 0,
      }],
      verifier: {
        ...oldResult.verifier,
        commandIds: ["cmd.latest.no-edit-failure"],
      },
      telemetry: {
        headlessAcp: {
          stopReason: "end_turn",
          counts: {
            fsWrite: 0,
            terminalCreate: 1,
            terminalExit: 1,
          },
          transcriptPath: "/tmp/no-write-slice-transcript.json",
        },
      },
    };
    const oldManifest = manifestFixture([oldResult]);
    const newManifest: RealAcpCorpusRunManifest = {
      ...manifestFixture([newResult]),
      runId: "real-acp-run.headless-current-visible-20260505",
      createdAt: "2026-05-05T00:00:00.000Z",
    };
    const lexicallyNewerOldManifest: RealAcpCorpusRunManifest = {
      ...oldManifest,
      runId: "real-acp-run.headless-visible-20260504",
    };

    const slice = buildNoWriteReplaySlice({
      manifests: [lexicallyNewerOldManifest, newManifest],
      latestPerTaskProfile: true,
    });

    expect(slice.status).toMatchObject({
      totalRecordsSeen: 2,
      includedCases: 1,
      skippedDuplicate: 1,
    });
    expect(slice.cases.map((sliceCase) => sliceCase.runResultId)).toEqual([
      "real-acp-run.visible-test-20260505.real-acp.task.simple-edit-greeting",
    ]);
    expect(slice.cases.map((sliceCase) => sliceCase.runId)).toEqual([
      "real-acp-run.headless-current-visible-20260505",
    ]);
    expect(slice.cases[0]!.terminalActivity.terminalExitCount).toBe(1);
  });

  test("reads only real ACP manifest and replay-export files from a corpus root", async () => {
    const root = await mkdtemp(join(tmpdir(), "no-write-slice-"));
    const corpusRoot = join(root, ".bag", "replay-corpus");
    const runDir = join(corpusRoot, "real-acp-runs", "real-acp-run.visible-test");
    await mkdir(runDir, { recursive: true });
    await writeFile(
      join(runDir, "real-acp-run.visible-test.manifest.json"),
      `${JSON.stringify(manifestFixture([taskResultFixture("real-acp.task.simple-edit-greeting", "train", true)]), null, 2)}\n`,
      "utf8",
    );
    await writeFile(
      join(runDir, "real-acp-run.visible-test.replay-export.json"),
      `${JSON.stringify(replayExportFixture([replayCaseFixture("real-acp.task.simple-edit-greeting", "train", true)]), null, 2)}\n`,
      "utf8",
    );
    await writeFile(join(runDir, "transcript.json"), "{this is intentionally not json", "utf8");

    const slice = await buildNoWriteReplaySliceFromCorpus({ corpusRoot });

    expect(slice.status).toMatchObject({
      totalRecordsSeen: 2,
      includedCases: 1,
      skippedDuplicate: 1,
    });
    expect(slice.cases.map((sliceCase) => sliceCase.runResultId)).toEqual([
      "real-acp-run.visible-test.real-acp.task.simple-edit-greeting",
    ]);
  });
});

const manifestFixture = (taskResults: readonly RealAcpTaskRunResult[]): RealAcpCorpusRunManifest => ({
  schemaVersion: "real-acp-corpus-run.v1",
  runId: "real-acp-run.visible-test",
  taskPackId: "real-acp-run-corpus.task-pack.v1",
  createdAt: "2026-05-04T00:00:00.000Z",
  executionMode: "headless_acp",
  dryRun: false,
  purpose: "development_eval",
  executor: {
    executorId: "real-acp.executor.headless.injected",
    executorVersion: "headless-adapter.v1",
    kind: "headless_acp",
  },
  metadata: {
    model: {
      modelProfileId: "model.real-acp.local-headless",
      provider: "configured",
      model: "configured-runtime",
      modelRole: "local",
      contextWindowTokens: 128000,
      toolCallingMode: "native",
    },
    codebase: {
      codebaseProfileId: "codebase.real-acp.bleeding-agent",
      rootFingerprint: "sha256:fixture",
      languageSummary: "TypeScript fixture",
      testRiskTier: "risk.real-acp.fixture",
      protectedPathPolicy: "fixture paths only",
    },
    client: {
      clientProfileId: "client.real-acp.headless-capable",
      clientName: "bag-headless-acp-runner",
      clientVersion: "1.0.0",
      transport: "in_process",
      acpConsumerCapabilities: {},
    },
    profile: {
      policyId: "policy.real-acp.current",
      optimizerProfileId: "optimizer.real-acp.current",
      verificationPolicyVersion: "verification.real-acp.v1",
      resultStyleVersion: "result.real-acp.v1",
      canonicalToolVersion: "canonical.real-acp.v1",
      renderedToolVersion: "rendered.real-acp.v1",
    },
  },
  safety: {
    workspaceIsolation: "per_task_materialized_fixture",
    currentRepoMutationRefused: true,
    realConsumerMutationAllowed: false,
  },
  splitPolicy: {
    includeHoldout: false,
    visibleOptimizationSplits: ["train", "dev"],
    hiddenSplits: ["holdout"],
    optimizerLeakageRefused: true,
  },
  taskResults: [...taskResults],
  redactionHandoff: {
    rawLocal: {
      status: "raw_local_only",
      containsWorkspaceSnapshots: true,
      containsExecutorTelemetry: true,
      storageGuidance: "local fixture",
    },
    optimizerSafe: {
      status: "prepared",
      includedTaskResultIds: taskResults.filter((result) => result.redaction.optimizerSafe).map((result) => result.runResultId),
      excludedTaskResultIds: taskResults.filter((result) => !result.redaction.optimizerSafe).map((result) => result.runResultId),
      redactedFields: [],
      nextSteps: [],
    },
  },
  summary: {
    total: taskResults.length,
    passed: 0,
    failed: taskResults.length,
    skipped: 0,
    cancelled: 0,
    error: 0,
    holdout: taskResults.filter((result) => result.split === "holdout").length,
  },
});

const taskResultFixture = (
  taskId: string,
  split: RealAcpTaskRunResult["split"],
  optimizerSafe: boolean,
): RealAcpTaskRunResult => ({
  schemaVersion: "real-acp-task-result.v1",
  runResultId: `real-acp-run.visible-test.${taskId}`,
  taskId,
  split,
  optimizationAllowed: split !== "holdout" && optimizerSafe,
  status: "failed",
  startedAt: "2026-05-04T00:00:00.000Z",
  completedAt: "2026-05-04T00:00:00.000Z",
  workspaceFingerprintBefore: "sha256:before",
  workspaceFingerprintAfter: "sha256:after",
  changedFiles: [],
  route: {
    routeId: `route.${taskId}`,
    selectedMode: split === "holdout" ? "coding" : "coding",
    reason: "fixture route",
  },
  editStrategy: {
    strategyId: "edit.headless-acp.consumer.v1",
    family: "none",
    selectedBy: "not_applicable",
  },
  toolCalls: [
    {
      toolCallId: `tool.${taskId}.read`,
      namespace: "acp.fs",
      name: "readTextFile",
      status: "succeeded",
      sideEffectLevel: "read",
    },
  ],
  terminalCommands: [],
  verifier: {
    status: "failed",
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
    taskId,
    runResultId: `real-acp-run.visible-test.${taskId}`,
    sourceTaskPackId: "real-acp-run-corpus.task-pack.v1",
  },
  telemetry: {
    headlessAcp: {
      stopReason: "end_turn",
      counts: {
        fsWrite: 0,
        terminalCreate: 0,
        terminalExit: 0,
      },
      transcriptPath: "/tmp/no-write-slice-transcript.json",
    },
  },
  redaction: {
    rawLocalStatus: "raw_local_only",
    optimizerSafe,
    excludedFromOptimizerReasons: optimizerSafe ? [] : ["fixture exclusion"],
  },
});

const replayExportFixture = (cases: readonly RealAcpReplayCaseRecord[]): RealAcpReplayExportManifest => ({
  schemaVersion: "real-acp-replay-export.v1",
  exportId: "real-acp-replay-export.visible-test",
  sourceRunId: "real-acp-run.visible-test",
  sourceTaskPackId: "real-acp-run-corpus.task-pack.v1",
  createdAt: "2026-05-04T00:00:00.000Z",
  purpose: "development_eval",
  status: "optimizer_safe",
  includeHoldout: false,
  optimizerInputAllowed: true,
  sourceMetadata: {
    executionMode: "headless_acp",
    dryRun: false,
    modelProfileId: "model.real-acp.local-headless",
    codebaseProfileId: "codebase.real-acp.bleeding-agent",
    clientProfileId: "client.real-acp.headless-capable",
    policyId: "policy.real-acp.current",
    optimizerProfileId: "optimizer.real-acp.current",
  },
  cases: [...cases],
  optimizerSelection: {
    selectedReplayCaseIds: cases.map((replayCase) => replayCase.replayCaseId),
    selectedTaskResultIds: cases.map((replayCase) => replayCase.lineage.runResultId),
    hiddenHoldoutReplayCaseIds: [],
    excludedReplayCaseIds: [],
    rejectionReasons: {},
  },
  summary: {
    totalSourceTaskResults: cases.length,
    exportedCases: cases.length,
    optimizerVisibleCases: cases.length,
    hiddenHoldoutCases: 0,
    failedCases: cases.length,
    skippedCases: 0,
    redactedCases: cases.length,
  },
});

const replayCaseFixture = (
  taskId: string,
  split: RealAcpReplayCaseRecord["split"],
  optimizerInputAllowed: boolean,
): RealAcpReplayCaseRecord => ({
  schemaVersion: "real-acp-replay-case.v1",
  replayCaseId: `real-acp.replay.real-acp-run.visible-test.${taskId}`,
  evalCaseId: `real-acp.eval.real-acp-run.visible-test.${taskId}`,
  split,
  optimizerInputAllowed,
  optimizerExclusionReasons: optimizerInputAllowed ? [] : ["fixture exclusion"],
  title: taskId,
  taskSummary: taskId,
  labels: taskId === "real-acp.task.cancellation-mid-edit" ? ["cancellation"] : ["simple_edit"],
  sourceRefs: [
    {
      sourceKind: "manifest",
      refId: "real-acp-run.visible-test",
      artifactRef: "real-acp-run:real-acp-run.visible-test",
      redactionStatus: "redacted",
    },
    {
      sourceKind: "task_result",
      refId: `real-acp-run.visible-test.${taskId}`,
      artifactRef: `real-acp-task-result:real-acp-run.visible-test.${taskId}`,
      redactionStatus: "redacted",
    },
  ],
  lineage: {
    runId: "real-acp-run.visible-test",
    taskPackId: "real-acp-run-corpus.task-pack.v1",
    taskId,
    runResultId: `real-acp-run.visible-test.${taskId}`,
    sourceTaskPackId: "real-acp-run-corpus.task-pack.v1",
    modelProfileId: "model.real-acp.local-headless",
    codebaseProfileId: "codebase.real-acp.bleeding-agent",
    clientProfileId: "client.real-acp.headless-capable",
    policyId: "policy.real-acp.current",
    optimizerProfileId: "optimizer.real-acp.current",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
  workspace: {
    allowedPathPrefixes: [],
    protectedPaths: [],
    rootFingerprintBefore: "sha256:before",
    rootFingerprintAfter: "sha256:after",
  },
  expectedOutcome: {
    mutation: "no_change",
    expectedChangedPaths: [],
    expectedNoChangePaths: ["src/settings.ts"],
    verifierPolicy: "must_skip",
    assertionSummaries: [],
  },
  outcome: {
    status: "cancelled",
    passed: false,
    skipReason: "structured cancellation policy",
    verifierStatus: "skipped",
    routeSelectedMode: "cancelled",
    editStrategyFamily: "none",
    repairStatus: "not_needed",
    rollbackStatus: "not_needed",
    correctionCount: 0,
  },
  evidence: {
    changedFiles: [],
    toolCalls: [],
    terminalCommands: [],
    telemetry: {
      headlessAcp: {
        stopReason: "cancelled",
        counts: {
          fsWrite: 0,
          terminalCreate: 0,
          terminalExit: 0,
        },
      },
    },
  },
  redaction: {
    status: "redacted",
    redactedFields: [],
    secretReplacementCount: 0,
    pathHashCount: 0,
    omittedRawFieldCount: 0,
  },
});
