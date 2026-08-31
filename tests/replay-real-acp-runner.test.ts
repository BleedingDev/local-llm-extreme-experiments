import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  RealAcpCorpusRunManifestSchema,
  assertRealAcpRunManifestSafeForOptimizerInput,
  createSimulatedRealAcpExecutor,
  realAcpCodingCorpusTaskPack,
  runRealAcpCorpus,
  selectRealAcpCorpusTasks,
  type RealAcpRunMetadata,
} from "../src/replay";

const createdAt = "2026-05-04T00:00:00.000Z";

const metadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.test",
    provider: "simulated",
    model: "simulated-acp-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.fixture",
    rootFingerprint: "sha256:real-acp-fixture",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.fixture",
    protectedPathPolicy: "Only task fixture paths may be changed; protected paths must remain unchanged.",
  },
  client: {
    clientProfileId: "client.real-acp.simulated",
    clientName: "Simulated ACP harness",
    clientVersion: "v1",
    transport: "simulated",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
      cancellation: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.test",
    optimizerProfileId: "optimizer.real-acp.test",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
};

const makeTempBase = (): Promise<string> => mkdtemp(join(tmpdir(), "real-acp-runner-test-"));

describe("real ACP corpus runner", () => {
  test("runs visible task pack through the simulated executor and writes a manifest", async () => {
    const baseDir = await makeTempBase();
    const outputDir = await makeTempBase();

    try {
      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.test.visible",
        metadata,
        executor: createSimulatedRealAcpExecutor(),
        purpose: "development_eval",
        executionMode: "dry_run",
        workspaceBaseDir: baseDir,
        outputDir,
        createdAt,
      });

      expect(RealAcpCorpusRunManifestSchema.parse(manifest)).toEqual(manifest);
      expect(manifest.taskPackId).toBe(realAcpCodingCorpusTaskPack.taskPackId);
      expect(manifest.dryRun).toBe(true);
      expect(manifest.safety.currentRepoMutationRefused).toBe(true);
      expect(manifest.taskResults).toHaveLength(9);
      expect(manifest.summary).toEqual({
        total: 9,
        passed: 6,
        failed: 1,
        skipped: 1,
        cancelled: 1,
        error: 0,
        holdout: 0,
      });
      expect(manifest.taskResults.every((result) => result.split !== "holdout")).toBe(true);
      expect(manifest.taskResults.every((result) => result.redaction.optimizerSafe)).toBe(true);
      expect(manifest.redactionHandoff.rawLocal.status).toBe("raw_local_only");
      expect(manifest.redactionHandoff.optimizerSafe.includedTaskResultIds).toHaveLength(9);
      expect(manifest.redactionHandoff.optimizerSafe.excludedTaskResultIds).toEqual([]);
      expect(manifest.manifestPath).toBeDefined();

      const written = JSON.parse(await readFile(manifest.manifestPath!, "utf8"));
      expect(written.runId).toBe("real-acp-run.test.visible");
      expect(written.taskResults[0]).toEqual(expect.objectContaining({
        route: expect.any(Object),
        editStrategy: expect.any(Object),
        verifier: expect.any(Object),
        repair: expect.any(Object),
        rollback: expect.any(Object),
        lineage: expect.any(Object),
      }));
    } finally {
      await rm(baseDir, { recursive: true, force: true });
      await rm(outputDir, { recursive: true, force: true });
    }
  });

  test("records required model, codebase, client, and profile metadata", async () => {
    const baseDir = await makeTempBase();

    try {
      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.test.metadata",
        metadata,
        executor: createSimulatedRealAcpExecutor(),
        taskIds: ["real-acp.task.simple-edit-greeting"],
        workspaceBaseDir: baseDir,
        createdAt,
      });

      expect(manifest.metadata.model).toEqual(metadata.model);
      expect(manifest.metadata.codebase).toEqual(metadata.codebase);
      expect(manifest.metadata.client).toEqual(metadata.client);
      expect(manifest.metadata.profile).toEqual(metadata.profile);
      await expect(runRealAcpCorpus({
        runId: "real-acp-run.test.bad-metadata",
        metadata: {
          ...metadata,
          model: {
            ...metadata.model,
            contextWindowTokens: 0,
          },
        },
        executor: createSimulatedRealAcpExecutor(),
        taskIds: ["real-acp.task.simple-edit-greeting"],
        workspaceBaseDir: baseDir,
        createdAt,
      })).rejects.toThrow();
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("refuses hidden holdout leakage for optimizer input but can run holdout for evaluation", async () => {
    const baseDir = await makeTempBase();

    try {
      expect(() => selectRealAcpCorpusTasks({
        purpose: "optimizer_input",
        includeHoldout: true,
      })).toThrow(/hidden holdout optimizer leakage refused/);

      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.test.holdout",
        metadata,
        executor: createSimulatedRealAcpExecutor(),
        purpose: "holdout_final",
        includeHoldout: true,
        workspaceBaseDir: baseDir,
        createdAt,
      });

      expect(manifest.summary.holdout).toBe(3);
      expect(manifest.redactionHandoff.optimizerSafe.excludedTaskResultIds).toHaveLength(3);
      expect(manifest.taskResults.filter((result) => result.split === "holdout").every((result) =>
        !result.redaction.optimizerSafe && result.redaction.excludedFromOptimizerReasons.includes("hidden holdout split")
      )).toBe(true);
      expect(() => assertRealAcpRunManifestSafeForOptimizerInput(manifest))
        .toThrow(/optimizer input rejected hidden or raw-local task results/);
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("records failed and skipped task outcomes without running real mutations", async () => {
    const baseDir = await makeTempBase();

    try {
      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.test.outcomes",
        metadata,
        executor: createSimulatedRealAcpExecutor({
          failTaskIds: ["real-acp.task.simple-edit-greeting"],
          skipTaskIds: ["real-acp.task.cart-bugfix-fail-to-pass"],
        }),
        taskIds: [
          "real-acp.task.simple-edit-greeting",
          "real-acp.task.cart-bugfix-fail-to-pass",
        ],
        workspaceBaseDir: baseDir,
        createdAt,
      });

      expect(manifest.summary).toEqual({
        total: 2,
        passed: 0,
        failed: 1,
        skipped: 1,
        cancelled: 0,
        error: 0,
        holdout: 0,
      });
      expect(manifest.taskResults.find((result) => result.taskId === "real-acp.task.simple-edit-greeting"))
        .toEqual(expect.objectContaining({
          status: "failed",
          failureReason: "simulated executor forced failure",
        }));
      expect(manifest.taskResults.find((result) => result.taskId === "real-acp.task.cart-bugfix-fail-to-pass"))
        .toEqual(expect.objectContaining({
          status: "skipped",
          skipReason: "simulated executor forced skip",
          changedFiles: [],
        }));
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("excludes internal .bag runner artifacts from task changed-file evidence", async () => {
    const baseDir = await makeTempBase();

    try {
      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.test.internal-artifacts",
        metadata,
        executor: {
          executorId: "real-acp.executor.internal-artifacts",
          executorVersion: "v1",
          kind: "headless_acp",
          executeTask: async (input) => {
            await mkdir(`${input.workspacePath}/.bag/internal`, { recursive: true });
            await Bun.write(`${input.workspacePath}/.bag/internal/transcript.json`, "{}\n");
            await Bun.write(`${input.workspacePath}/src/greeter.ts`, "export const formatGreeting = (name: string): string => `Hello, ${name}!`;\n");
            return {
              status: "passed",
              route: {
                routeId: "route.internal-artifacts",
                selectedMode: "coding",
                reason: "test",
              },
              editStrategy: {
                strategyId: "edit.test",
                family: "whole_file",
                selectedBy: "executor",
              },
              verifier: {
                status: "passed",
                policy: "required",
                commandIds: [],
              },
              repair: { attempted: false, status: "not_needed" },
              rollback: { attempted: false, status: "not_needed" },
            };
          },
        },
        executionMode: "headless_acp",
        taskIds: ["real-acp.task.simple-edit-greeting"],
        workspaceBaseDir: baseDir,
        currentRepoPath: join(baseDir, "repo"),
        createdAt,
      });

      expect(manifest.taskResults[0]?.changedFiles).toEqual([
        expect.objectContaining({ path: "src/greeter.ts", changeKind: "modified" }),
      ]);
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("allows real_consumer execution only through materialized isolated workspaces", async () => {
    const baseDir = await makeTempBase();

    try {
      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.test.real-consumer-isolated",
        metadata,
        executor: {
          executorId: "real-acp.executor.test-real-consumer",
          executorVersion: "v1",
          kind: "real_consumer",
          executeTask: async (input) => {
            await writeFile(join(input.workspacePath, "src/greeter.ts"), "export const formatGreeting = (name: string): string => `Hello, ${name}!`;\n", "utf8");
            return {
              status: "passed",
              route: {
                routeId: "route.test-real-consumer",
                selectedMode: "coding",
                reason: "test real consumer executor",
              },
              editStrategy: {
                strategyId: "edit.test-real-consumer",
                family: "diff",
                selectedBy: "executor",
              },
              toolCalls: [{
                toolCallId: "tool.test-real-consumer.write",
                namespace: "acp.fs",
                name: "writeTextFile",
                status: "succeeded",
                sideEffectLevel: "write",
              }],
              terminalCommands: [],
              verifier: {
                status: "not_run",
                policy: "required",
                commandIds: [],
              },
              repair: { attempted: false, status: "not_needed" },
              rollback: { attempted: false, status: "not_needed" },
              telemetry: {
                realConsumer: {
                  protocolBoundary: "ACP over stdio; this is not desktop UI rendering parity evidence.",
                },
              },
            };
          },
        },
        executionMode: "real_consumer",
        taskIds: ["real-acp.task.simple-edit-greeting"],
        workspaceBaseDir: baseDir,
        currentRepoPath: join(baseDir, "repo"),
        createdAt,
      });

      expect(manifest.dryRun).toBe(false);
      expect(manifest.executor.kind).toBe("real_consumer");
      expect(manifest.safety.currentRepoMutationRefused).toBe(true);
      expect(manifest.safety.realConsumerMutationAllowed).toBe(true);
      expect(manifest.taskResults[0]?.changedFiles).toEqual([
        expect.objectContaining({ path: "src/greeter.ts", changeKind: "modified" }),
      ]);
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });
});
