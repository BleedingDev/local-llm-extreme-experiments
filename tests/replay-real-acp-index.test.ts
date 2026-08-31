import { describe, expect, test } from "bun:test";
import {
  RealAcpReplayCorpusIndexRecordSchema,
  buildRealAcpReplayCorpusIndex,
  serializeRealAcpReplayCorpusIndexJsonl,
  type RealAcpReplayCorpusIndexRecord,
} from "../src/replay/real-acp-index";
import {
  realAcpCodingCorpusTaskPack,
  type RealAcpCorpusRunManifest,
  type RealAcpTaskRunResult,
  type RealAcpRunMetadata,
} from "../src/replay";
import type { RealAcpCorpusTask } from "../src/replay/real-acp-task-pack";
import type { ReplayEvalCaseSkeleton } from "../src/replay/extraction";

const createdAt = "2026-05-04T00:00:00.000Z";

const metadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.index",
    provider: "simulated",
    model: "simulated-acp-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.index",
    rootFingerprint: "sha256:index-fixture",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.index",
    protectedPathPolicy: "Fixture-only writes.",
  },
  client: {
    clientProfileId: "client.real-acp.index",
    clientName: "Index test ACP client",
    clientVersion: "v1",
    transport: "simulated",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.index",
    optimizerProfileId: "optimizer.real-acp.index",
    verificationPolicyVersion: "verification.real-acp.index",
    resultStyleVersion: "result.real-acp.index",
    canonicalToolVersion: "canonical.real-acp.index",
    renderedToolVersion: "rendered.real-acp.index",
  },
};

const visibleTask = realAcpCodingCorpusTaskPack.tasks.find((task) => task.split === "train");
const holdoutTask = realAcpCodingCorpusTaskPack.tasks.find((task) => task.split === "holdout");
if (visibleTask == null || holdoutTask == null) {
  throw new Error("real ACP task pack must include visible and holdout tasks");
}

describe("real ACP replay corpus index helpers", () => {
  test("builds stable real-run index records with source refs, labels, ids, scores, and reproduction metadata", () => {
    const visible = taskResult(visibleTask, {
      runResultId: "run.result.index.visible",
      status: "passed",
      telemetry: {
        score: 0.75,
        selfEvaluation: { score: 0.82 },
        replayCaseId: "replay.eval.index.linked",
      },
    });
    const hidden = taskResult(holdoutTask, {
      runResultId: "run.result.index.holdout",
      status: "failed",
      telemetry: { score: 0.1 },
    });

    const index = buildRealAcpReplayCorpusIndex({
      runManifests: [manifest([hidden, visible])],
      reproductionCommand: ["bun", "scripts/bag_acp_run.ts", "--manifest", "fixture"],
      reproductionCwd: "/repo",
    });

    expect(index.map((record) => record.runResultId)).toEqual([
      "run.result.index.visible",
      "run.result.index.holdout",
    ]);

    const visibleRecord = index[0]!;
    expect(visibleRecord).toMatchObject({
      sourceKind: "real_acp_run_result",
      runId: "real-acp-run.index",
      runResultId: "run.result.index.visible",
      taskId: visibleTask.taskId,
      taskPackId: realAcpCodingCorpusTaskPack.taskPackId,
      replayCaseId: "replay.eval.index.linked",
      split: "train",
      status: "passed",
      scores: {
        outcomeScore: 1,
        "telemetry.score": 0.75,
        "telemetry.selfEvaluation.score": 0.82,
      },
      identities: {
        modelProfileId: "model.real-acp.index",
        codebaseProfileId: "codebase.real-acp.index",
        clientProfileId: "client.real-acp.index",
        policyId: "policy.real-acp.index",
        optimizerProfileId: "optimizer.real-acp.index",
      },
      reproduction: {
        command: ["bun", "scripts/bag_acp_run.ts", "--manifest", "fixture"],
        cwd: "/repo",
        sourceArtifactPath: ".bag/replay-corpus/real-acp-runs/index.manifest.json",
      },
      safety: {
        hiddenHoldout: false,
        optimizerSafe: true,
        optimizerInputAllowed: true,
        rawLocalOnly: true,
      },
    });
    expect(visibleRecord.labels.task).toEqual(visibleTask.labels);
    expect(visibleRecord.labels.safety).toEqual([
      "optimizer_allowed",
      "raw_local_only",
      "visible_optimizer_safe",
    ]);
    expect(visibleRecord.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "real_acp_run_manifest",
      path: ".bag/replay-corpus/real-acp-runs/index.manifest.json",
    }));

    const holdoutRecord = index[1]!;
    expect(holdoutRecord.safety).toMatchObject({
      hiddenHoldout: true,
      optimizerSafe: false,
      optimizerInputAllowed: false,
    });
    expect(holdoutRecord.labels.safety).toContain("hidden_holdout");
    expect(holdoutRecord.safety.excludedFromOptimizerReasons).toContain("hidden holdout split");
  });

  test("indexes redacted replay case manifests and blocks holdout or unsafe redaction from optimizer input", () => {
    const trainCase = replayCase("replay.eval.index.case.train", "train", "redacted");
    const holdoutCase = replayCase("replay.eval.index.case.holdout", "holdout", "redacted");
    const needsReviewCase = replayCase("replay.eval.index.case.needs-review", "dev", "needs_review");

    const index = buildRealAcpReplayCorpusIndex({
      replayCases: [holdoutCase, needsReviewCase, trainCase],
    });

    expect(index.map((record) => record.replayCaseId)).toEqual([
      "replay.eval.index.case.train",
      "replay.eval.index.case.needs-review",
      "replay.eval.index.case.holdout",
    ]);
    expect(index[0]).toMatchObject({
      sourceKind: "redacted_replay_case",
      replayCaseId: "replay.eval.index.case.train",
      captureId: "capture.replay.eval.index.case.train",
      split: "train",
      labels: {
        task: ["source-adapter", "tool-call"],
      },
      safety: {
        redactionStatus: "redacted",
        hiddenHoldout: false,
        optimizerInputAllowed: true,
      },
    });
    expect(index[1]!.safety).toMatchObject({
      hiddenHoldout: false,
      optimizerSafe: false,
      optimizerInputAllowed: false,
      redactionStatus: "needs_review",
    });
    expect(index[1]!.labels.safety).toContain("needs_review");
    expect(index[2]!.safety).toMatchObject({
      hiddenHoldout: true,
      optimizerSafe: false,
      optimizerInputAllowed: false,
    });
  });

  test("rejects duplicate source identities", () => {
    const first = taskResult(visibleTask, { runResultId: "run.result.index.dup" });
    const second = taskResult(holdoutTask, { runResultId: "run.result.index.dup" });

    expect(() => buildRealAcpReplayCorpusIndex({
      runManifests: [manifest([first, second])],
    })).toThrow(/duplicate real ACP replay corpus index runResultId/);

    expect(() => buildRealAcpReplayCorpusIndex({
      replayCases: [
        replayCase("replay.eval.index.duplicate", "train", "redacted"),
        replayCase("replay.eval.index.duplicate", "dev", "redacted"),
      ],
    })).toThrow(/duplicate real ACP replay corpus index replayCaseId/);
  });

  test("schema validation enforces hidden holdout safety flags", () => {
    const [record] = buildRealAcpReplayCorpusIndex({
      replayCases: [replayCase("replay.eval.index.schema-holdout", "holdout", "redacted")],
    });
    expect(record).toBeDefined();
    const unsafe = {
      ...record!,
      safety: {
        ...record!.safety,
        hiddenHoldout: false,
        optimizerInputAllowed: true,
      },
    } satisfies RealAcpReplayCorpusIndexRecord;

    expect(() => RealAcpReplayCorpusIndexRecordSchema.parse(unsafe))
      .toThrow(/holdout records must be flagged as hidden holdout/);
  });

  test("serializes deterministic JSONL", () => {
    const index = buildRealAcpReplayCorpusIndex({
      replayCases: [
        replayCase("replay.eval.index.jsonl.dev", "dev", "redacted"),
        replayCase("replay.eval.index.jsonl.train", "train", "redacted"),
      ],
    });

    const jsonl = serializeRealAcpReplayCorpusIndexJsonl(index);
    const lines = jsonl.trim().split("\n").map((line) => JSON.parse(line));

    expect(lines.map((line) => line.replayCaseId)).toEqual([
      "replay.eval.index.jsonl.train",
      "replay.eval.index.jsonl.dev",
    ]);
    expect(jsonl.endsWith("\n")).toBe(true);
  });
});

const manifest = (taskResults: RealAcpTaskRunResult[]): RealAcpCorpusRunManifest => ({
  schemaVersion: "real-acp-corpus-run.v1",
  runId: "real-acp-run.index",
  taskPackId: realAcpCodingCorpusTaskPack.taskPackId,
  createdAt,
  executionMode: "dry_run",
  dryRun: true,
  purpose: "development_eval",
  executor: {
    executorId: "real-acp.executor.index",
    executorVersion: "index.v1",
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
      storageGuidance: "Keep local.",
    },
    optimizerSafe: {
      status: "prepared",
      includedTaskResultIds: taskResults.filter((result) => result.redaction.optimizerSafe).map((result) => result.runResultId),
      excludedTaskResultIds: taskResults.filter((result) => !result.redaction.optimizerSafe).map((result) => result.runResultId),
      redactedFields: ["workspaceSnapshots"],
      nextSteps: ["Build index."],
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
  manifestPath: ".bag/replay-corpus/real-acp-runs/index.manifest.json",
});

const taskResult = (
  task: RealAcpCorpusTask,
  overrides: Partial<Pick<RealAcpTaskRunResult, "runResultId" | "status" | "telemetry">> = {},
): RealAcpTaskRunResult => {
  const runResultId = overrides.runResultId ?? `run.result.index.${task.taskId}`;
  return {
    schemaVersion: "real-acp-task-result.v1",
    runResultId,
    taskId: task.taskId,
    split: task.split,
    optimizationAllowed: task.optimizationAllowed,
    status: overrides.status ?? "passed",
    startedAt: createdAt,
    completedAt: createdAt,
    workspaceFingerprintBefore: "sha256:before",
    workspaceFingerprintAfter: "sha256:after",
    changedFiles: [],
    route: {
      routeId: `route.${runResultId}`,
      selectedMode: "coding",
      reason: "Index test route.",
      confidence: 1,
    },
    editStrategy: {
      strategyId: "edit.index.whole-file",
      family: "whole_file",
      selectedBy: "optimizer_policy",
      reason: "Index test strategy.",
    },
    toolCalls: [],
    terminalCommands: [],
    verifier: {
      status: "passed",
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
      taskId: task.taskId,
      runResultId,
      sourceTaskPackId: realAcpCodingCorpusTaskPack.taskPackId,
    },
    telemetry: overrides.telemetry ?? {},
    redaction: {
      rawLocalStatus: "raw_local_only",
      optimizerSafe: task.split !== "holdout",
      excludedFromOptimizerReasons: task.split === "holdout" ? ["hidden holdout split"] : [],
    },
  };
};

const replayCase = (
  evalCaseId: string,
  split: "train" | "dev" | "holdout",
  redactionStatus: "redacted" | "needs_review",
): ReplayEvalCaseSkeleton => ({
  evalCaseId,
  schemaVersion: "replay-eval-case.v1",
  split,
  splitAssignment: {
    split,
    assignedBy: "manual",
    rationale: "Index test split.",
  },
  title: `Replay case ${evalCaseId}`,
  task: "Replay an indexed redacted case.",
  captureId: `capture.${evalCaseId}`,
  sourceTraceIds: [],
  sourceRefs: [{
    sourceKind: "capture",
    captureId: `capture.${evalCaseId}`,
    path: `.bag/replay-corpus/cases/${evalCaseId}.json`,
    redactionStatus,
  }],
  redaction: {
    status: redactionStatus,
    needsReview: redactionStatus === "needs_review",
    needsReviewRecordIds: redactionStatus === "needs_review" ? [`record.${evalCaseId}.needs-review`] : [],
    recordStatuses: [],
  },
  oracle: {
    strength: "weak",
    expectedBehavior: {
      summary: "Preserve observed behavior.",
      assertions: [],
      notes: [],
    },
  },
  routing: {
    promptRecordIds: [],
    routingRecordIds: [],
  },
  observedFailures: [],
  tags: ["source-adapter", "tool-call"],
  timeoutMs: 120000,
});
