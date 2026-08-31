import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  REAL_ACP_CORPUS_OUTPUT_ROOT,
  loadRealAcpRunMetadata,
  parseRunRealAcpCorpusArgs,
  planRealAcpCorpusRun,
  runPlannedRealAcpCorpus,
  runPlannedRealAcpCorpusWithArtifacts,
  type RealAcpCorpusRunPlan,
} from "../scripts/run_real_acp_corpus";
import { summarizeConsumerTrajectory, type RealAcpConsumerReadiness, type RealAcpRunMetadata } from "../src/replay";

const createdAt = "2026-05-04T00:00:00.000Z";

const metadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.script-test",
    provider: "simulated",
    model: "simulated-acp-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.script-fixture",
    rootFingerprint: "sha256:real-acp-script-fixture",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.script-fixture",
    protectedPathPolicy: "Only per-task materialized fixture paths may be changed.",
  },
  client: {
    clientProfileId: "client.real-acp.script-simulated",
    clientName: "Simulated ACP corpus script",
    clientVersion: "v1",
    transport: "simulated",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
      cancellation: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.script-test",
    optimizerProfileId: "optimizer.real-acp.script-test",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
};

const makeTempDir = (): Promise<string> => mkdtemp(join(tmpdir(), "run-real-acp-corpus-script-"));

const readyConsumerReadiness = (root: string): RealAcpConsumerReadiness => ({
  providerId: "real-acp.consumer.test-ready",
  consumerName: "stdio",
  status: "ready",
  blockers: [],
  launch: {
    command: "node",
    args: ["-e", "process.stdin.resume()"],
    cwd: root,
  },
  clientMetadata: {
    clientProfileId: "client.real-acp.test-ready",
    clientName: "Injected ready stdio ACP consumer",
    clientVersion: "test",
    transport: "stdio",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
      cancellation: true,
      permissions: true,
      transcript: true,
      desktopUiParity: false,
    },
  },
  capabilityEvidence: {
    source: "test injected readiness",
    desktopUiParity: false,
  },
});

const missingConsumerReadiness = (): RealAcpConsumerReadiness => ({
  providerId: "real-acp.consumer.test-missing-zed",
  consumerName: "Zed",
  status: "blocked",
  blockers: [
    "Zed app not found at /missing/Zed.app",
    "settings file not found: /missing/zed/settings.json",
  ],
  clientMetadata: {
    clientProfileId: "client.real-acp.test-missing-zed",
    clientName: "Missing Zed ACP consumer",
    clientVersion: "unknown",
    transport: "stdio",
    acpConsumerCapabilities: {},
  },
  capabilityEvidence: {
    source: "test missing readiness",
  },
});

describe("real ACP corpus launch script", () => {
  test("parses a safe dry-run plan by default under the real ACP corpus output root", async () => {
    const root = await makeTempDir();
    try {
      const options = parseRunRealAcpCorpusArgs(["--run-id", "real-acp-run.script.default"], root);
      const plan = planRealAcpCorpusRun(options);

      expect(plan).toMatchObject({
        status: "ready",
        runId: "real-acp-run.script.default",
        mode: "dry_run",
        purpose: "development_eval",
        includeHoldout: false,
        integrationBlockers: [],
        runnerInvocation: {
          api: "runRealAcpCorpus",
          executionMode: "dry_run",
          executorKind: "simulated",
          dryRun: true,
        },
        safety: {
          outputUnderSafeRoot: true,
          workspaceUnderSafeRoot: true,
          currentRepoMutationRefused: true,
          actualConsumerLaunch: false,
        },
      } satisfies Partial<RealAcpCorpusRunPlan>);
      expect(plan.outputDir).toBe(join(root, REAL_ACP_CORPUS_OUTPUT_ROOT, "real-acp-run.script.default"));
      expect(plan.workspaceBaseDir).toBe(join(plan.outputDir, "workspaces"));
      expect(plan.selectedTaskIds).toHaveLength(9);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("loads metadata and runs only the simulated dry-run corpus substrate", async () => {
    const root = await makeTempDir();
    try {
      const metadataPath = join(root, "real-acp-metadata.json");
      await writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
      const loadedMetadata = await loadRealAcpRunMetadata(metadataPath);
      const options = parseRunRealAcpCorpusArgs([
        "--metadata",
        metadataPath,
        "--run-id",
        "real-acp-run.script.execute",
        "--task-id",
        "real-acp.task.simple-edit-greeting",
      ], root);
      const plan = planRealAcpCorpusRun(options);
      const manifest = await runPlannedRealAcpCorpus(plan, loadedMetadata, createdAt);

      expect(manifest.executionMode).toBe("dry_run");
      expect(manifest.dryRun).toBe(true);
      expect(manifest.safety.currentRepoMutationRefused).toBe(true);
      expect(manifest.safety.realConsumerMutationAllowed).toBe(false);
      expect(manifest.summary.total).toBe(1);
      expect(manifest.taskResults[0]?.taskId).toBe("real-acp.task.simple-edit-greeting");
      expect(manifest.manifestPath).toBe(join(plan.outputDir, "real-acp-run.script.execute.manifest.json"));
      const written = JSON.parse(await readFile(manifest.manifestPath!, "utf8"));
      expect(written.metadata).toEqual(metadata);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("refuses include-holdout when the selected purpose is optimizer input", async () => {
    const root = await makeTempDir();
    try {
      const options = parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.holdout",
        "--purpose",
        "optimizer_input",
        "--include-holdout",
      ], root);

      expect(() => planRealAcpCorpusRun(options)).toThrow(/hidden holdout optimizer leakage refused/);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("blocks output and workspace directories outside the real ACP corpus output root", async () => {
    const root = await makeTempDir();
    try {
      const outputOutsideRoot = parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.bad-out",
        "--out-dir",
        ".bag/not-real-acp",
      ], root);
      expect(() => planRealAcpCorpusRun(outputOutsideRoot)).toThrow(/--out-dir must be under/);

      const workspaceOutsideRoot = parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.bad-workspace",
        "--workspace-base-dir",
        ".bag/not-real-acp/workspaces",
      ], root);
      expect(() => planRealAcpCorpusRun(workspaceOutsideRoot)).toThrow(/--workspace-base-dir must be under/);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("runs headless ACP mode through an injected transcript runner and writes export/index artifacts", async () => {
    const root = await makeTempDir();
    try {
      const options = parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.headless-execute",
        "--mode",
        "headless_acp",
        "--task-id",
        "real-acp.task.simple-edit-greeting",
      ], root);
      const plan = planRealAcpCorpusRun(options);
      expect(plan).toMatchObject({
        status: "ready",
        mode: "headless_acp",
        integrationBlockers: [],
        safety: {
          actualConsumerLaunch: true,
        },
      });

      const artifacts = await runPlannedRealAcpCorpusWithArtifacts(plan, {
        ...metadata,
        client: {
          ...metadata.client,
          clientProfileId: "client.real-acp.script-headless",
          clientName: "Injected transcript headless ACP",
          transport: "in_process",
        },
      }, createdAt, {
        runHeadlessTranscript: async (input) => {
          const sourcePath = join(input.workspace.workspacePath, "src/greeter.ts");
          await writeFile(sourcePath, "export const formatGreeting = (name: string): string => `Hello, ${name}!`;\n", "utf8");
          return {
            stopReason: "end_turn",
            trajectoryLength: 5,
            counts: {
              fsRead: 1,
              fsWrite: 1,
              terminalCreate: 1,
              terminalExit: 1,
              permission: 1,
              agentStderr: 0,
            },
            trajectory: [
              { kind: "fs_read", path: sourcePath, bytes: 42 },
              { kind: "permission", chosen: "allow_always" },
              { kind: "fs_write", path: sourcePath, bytes: 78 },
              { kind: "terminal_create", terminalId: "term-1", command: "bun", args: ["test", "tests/greeter.test.ts"] },
              { kind: "terminal_exit", terminalId: "term-1", exitCode: 0, signal: null },
            ],
            transcriptPath: join(input.workspace.workspacePath, ".bag", "headless-transcript.json"),
          };
        },
      });

      expect(artifacts.manifest.executionMode).toBe("headless_acp");
      expect(artifacts.manifest.dryRun).toBe(false);
      expect(artifacts.manifest.summary).toMatchObject({
        total: 1,
        passed: 1,
        failed: 0,
      });
      expect(artifacts.manifest.taskResults[0]).toMatchObject({
        status: "passed",
        route: {
          selectedMode: "coding",
        },
        verifier: {
          status: "passed",
        },
        repair: {
          attempted: false,
          status: "not_needed",
        },
      });
      expect(artifacts.manifest.taskResults[0]?.changedFiles).toEqual([
        expect.objectContaining({ path: "src/greeter.ts", changeKind: "modified" }),
      ]);
      expect(artifacts.manifest.taskResults[0]?.telemetry).toMatchObject({
        codingProgressDiagnostic: {
          progressClass: "verified_edit",
          mutatingProgress: {
            successfulWriteCount: 1,
            terminalCommandCount: 1,
          },
          verifier: {
            executedCommandCount: 1,
            failedCommandCount: 0,
          },
        },
        writeEvents: [
          expect.objectContaining({ kind: "fs_write", path: "src/greeter.ts" }),
        ],
      });
      expect(artifacts.replayExport.cases).toHaveLength(1);
      expect(artifacts.scorecard.aggregate.passed).toMatchObject({ count: 1, rate: 1 });
      expect(artifacts.scorecard.taskRecords[0]).toMatchObject({
        taskId: "real-acp.task.simple-edit-greeting",
        codingProgressClass: "verified_edit",
        changedFileCount: 1,
        writeToolCallCount: 1,
        terminalCommandCount: 1,
      });
      expect(artifacts.indexRecordCount).toBe(1);
      expect(JSON.parse(await readFile(artifacts.exportPath, "utf8")).sourceRunId)
        .toBe("real-acp-run.script.headless-execute");
      expect(JSON.parse(await readFile(artifacts.scorecardPath, "utf8")).scorecardId)
        .toBe("real-acp-stability.real-acp-run.script.headless-execute");
      expect(await readFile(artifacts.scorecardMarkdownPath, "utf8"))
        .toContain("real-acp.task.simple-edit-greeting");
      expect((await readFile(artifacts.indexPath, "utf8")).trim().split("\n")).toHaveLength(1);
      expect((await readFile(artifacts.rootIndexPath, "utf8")).trim().split("\n")).toHaveLength(1);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("headless ACP mode records a model/profile blocker instead of a generic no-edit result", async () => {
    const root = await makeTempDir();
    try {
      const options = parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.headless-blocked",
        "--mode",
        "headless_acp",
        "--task-id",
        "real-acp.task.simple-edit-greeting",
      ], root);
      const plan = planRealAcpCorpusRun(options);
      const artifacts = await runPlannedRealAcpCorpusWithArtifacts(plan, metadata, createdAt);
      const result = artifacts.manifest.taskResults[0]!;

      expect(result.status).toBe("error");
      expect(result.failureReason).toContain("model/profile prerequisites");
      expect(result.changedFiles).toEqual([]);
      expect(result.toolCalls).toEqual([]);
      expect(result.terminalCommands).toEqual([]);
      expect(result.telemetry).toMatchObject({
        codingProgressDiagnostic: {
          progressClass: "no_model",
          generation: {
            modelAvailable: false,
          },
        },
        headlessAcp: {
          blocked: true,
          blockerKind: "model_profile_prerequisite",
        },
      });
      expect(artifacts.scorecard.taskRecords[0]).toMatchObject({
        taskId: "real-acp.task.simple-edit-greeting",
        status: "error",
        strategyFamily: "none",
        codingProgressClass: "no_model",
      });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("blocks real consumer mode on named-consumer readiness instead of unconditional runner refusal", async () => {
    const root = await makeTempDir();
    try {
      const headlessPlan = planRealAcpCorpusRun(parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.headless",
        "--mode",
        "headless_acp",
      ], root));
      expect(headlessPlan.status).toBe("ready");
      expect(headlessPlan.runnerInvocation).toMatchObject({
        executionMode: "headless_acp",
        executorKind: "headless_acp",
        dryRun: false,
      });
      expect(headlessPlan.integrationBlockers).toEqual([]);

      const realConsumerPlan = planRealAcpCorpusRun(parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.real-consumer",
        "--mode",
        "real_consumer",
      ], root), {
        realConsumerReadinessProvider: missingConsumerReadiness,
      });
      expect(realConsumerPlan.status).toBe("blocked");
      expect(realConsumerPlan.runnerInvocation).toMatchObject({
        executionMode: "real_consumer",
        executorKind: "real_consumer",
        dryRun: false,
      });
      expect(realConsumerPlan.safety.actualConsumerLaunch).toBe(false);
      expect(realConsumerPlan.realConsumer?.protocolEvidence).toBe("acp_stdio_only_not_desktop_ui_parity");
      expect(realConsumerPlan.integrationBlockers.join("\n")).toContain("Zed real_consumer readiness blocked: Zed app not found");
      expect(realConsumerPlan.integrationBlockers.join("\n")).toContain("settings file not found");
      await expect(runPlannedRealAcpCorpus(realConsumerPlan, metadata, createdAt)).rejects.toThrow(/plan is blocked/);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("runs real consumer mode through an injected ready stdio protocol runner in isolated workspaces", async () => {
    const root = await makeTempDir();
    try {
      const options = parseRunRealAcpCorpusArgs([
        "--run-id",
        "real-acp-run.script.real-consumer-ready",
        "--mode",
        "real_consumer",
        "--consumer",
        "stdio",
        "--consumer-command",
        "node",
        "--consumer-arg",
        "-e",
        "--consumer-arg",
        "process.stdin.resume()",
        "--task-id",
        "real-acp.task.simple-edit-greeting",
      ], root);
      const readiness = readyConsumerReadiness(root);
      const plan = planRealAcpCorpusRun(options, {
        realConsumerReadinessProvider: () => readiness,
      });

      expect(plan).toMatchObject({
        status: "ready",
        mode: "real_consumer",
        integrationBlockers: [],
        safety: {
          actualConsumerLaunch: true,
        },
      });

      const artifacts = await runPlannedRealAcpCorpusWithArtifacts(plan, metadata, createdAt, {
        runRealConsumerProtocol: async (input) => {
          const sourcePath = join(input.workspacePath, "src/greeter.ts");
          await writeFile(sourcePath, "export const formatGreeting = (name: string): string => `Hello, ${name}!`;\n", "utf8");
          return summarizeConsumerTrajectory({
            stopReason: "end_turn",
            transcriptPath: input.transcriptPath,
            trajectory: [
              { kind: "protocol_call", at: createdAt, method: "initialize", phase: "response", payload: {} },
              { kind: "session_update", at: createdAt, update: { update: "tool_call" } },
              { kind: "permission", at: createdAt, chosen: "allow_always", toolCall: { name: "writeTextFile" } },
              { kind: "fs_write", at: createdAt, path: sourcePath, bytes: 78 },
              { kind: "terminal_create", at: createdAt, terminalId: "term-1", command: "bun", args: ["test", "tests/greeter.test.ts"] },
              { kind: "terminal_exit", at: createdAt, terminalId: "term-1", exitCode: 0, signal: null, outputBytes: 0 },
            ],
          });
        },
      });

      expect(artifacts.manifest.executionMode).toBe("real_consumer");
      expect(artifacts.manifest.metadata.client).toEqual(readiness.clientMetadata);
      expect(artifacts.manifest.safety.realConsumerMutationAllowed).toBe(true);
      expect(artifacts.manifest.summary).toMatchObject({ total: 1, passed: 1, failed: 0 });
      expect(artifacts.manifest.taskResults[0]).toMatchObject({
        status: "passed",
        changedFiles: [
          expect.objectContaining({ path: "src/greeter.ts", changeKind: "modified" }),
        ],
        telemetry: {
          realConsumer: {
            consumerName: "stdio",
            protocolBoundary: "ACP over stdio; this is not desktop UI rendering parity evidence.",
            counts: {
              fsWrite: 1,
              terminalExit: 1,
              permission: 1,
            },
          },
        },
      });
      expect(artifacts.scorecard.taskRecords[0]).toMatchObject({
        taskId: "real-acp.task.simple-edit-greeting",
        status: "passed",
        changedFileCount: 1,
      });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
