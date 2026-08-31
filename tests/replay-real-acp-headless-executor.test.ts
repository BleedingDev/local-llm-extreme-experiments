import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  createRealAcpHeadlessExecutor,
  type RealAcpHeadlessRunnerInput,
} from "../src/replay/real-acp-headless-executor";
import {
  RealAcpTaskRunResultSchema,
  runRealAcpCorpus,
  type RealAcpExecutorTaskInput,
  type RealAcpRunMetadata,
} from "../src/replay/real-acp-runner";
import {
  realAcpCodingCorpusTaskPack,
  type RealAcpCorpusTask,
} from "../src/replay/real-acp-task-pack";

const createdAt = "2026-05-04T00:00:00.000Z";

const metadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.headless-test",
    provider: "injected",
    model: "headless-acp-test-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.headless-fixture",
    rootFingerprint: "sha256:real-acp-headless-fixture",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.headless-fixture",
    protectedPathPolicy: "Only materialized task fixture paths may be changed.",
  },
  client: {
    clientProfileId: "client.real-acp.headless-injected",
    clientName: "Injected headless ACP runner",
    clientVersion: "v1",
    transport: "in_process",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
      cancellation: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.headless-test",
    optimizerProfileId: "optimizer.real-acp.headless-test",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
};

const makeTempBase = (prefix: string): Promise<string> =>
  mkdtemp(join(tmpdir(), prefix));

const taskById = (taskId: string): RealAcpCorpusTask => {
  const task = realAcpCodingCorpusTaskPack.tasks.find((candidate) => candidate.taskId === taskId);
  if (task === undefined) {
    throw new Error(`missing fixture task ${taskId}`);
  }
  return task;
};

const materializeTaskWorkspace = async (
  task: RealAcpCorpusTask,
  baseDir: string,
): Promise<string> => {
  const workspacePath = join(baseDir, "workspace");
  await mkdir(workspacePath, { recursive: true });
  for (const file of task.workspace.files) {
    const target = join(workspacePath, file.path);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, file.content, "utf8");
  }
  return workspacePath;
};

const directExecutorInput = (
  task: RealAcpCorpusTask,
  workspacePath: string,
  signal = new AbortController().signal,
): RealAcpExecutorTaskInput => ({
  task,
  workspacePath,
  executionMode: "headless_acp",
  dryRun: false,
  metadata,
  context: {
    signal,
    timeoutMs: task.timeoutMs,
  },
});

describe("real ACP headless executor adapter", () => {
  test("maps materialized task requests and headless results into corpus lineage", async () => {
    const workspaceBaseDir = await makeTempBase("real-acp-headless-workspaces-");
    const currentRepoPath = await makeTempBase("real-acp-headless-current-repo-");
    const sentinelPath = join(currentRepoPath, "sentinel.txt");
    const requests: RealAcpHeadlessRunnerInput[] = [];

    await writeFile(sentinelPath, "do not mutate\n", "utf8");

    const executor = createRealAcpHeadlessExecutor({
      executorId: "real-acp.executor.headless.test",
      executorVersion: "headless-test.v1",
      currentRepoPath,
      runTask: async (input) => {
        requests.push(input);
        expect(input.task.taskId).toBe("real-acp.task.simple-edit-greeting");
        expect(input.task.userPrompt).toContain("Change formatGreeting");
        expect(input.workspace.workspacePath.startsWith(currentRepoPath)).toBe(false);
        expect(input.workspace.materializedFilePaths).toEqual([
          "src/greeter.ts",
          "tests/greeter.test.ts",
        ]);

        const sourcePath = join(input.workspace.workspacePath, "src/greeter.ts");
        expect(await readFile(sourcePath, "utf8")).toContain("Hi, ${name}.");
        await writeFile(sourcePath, "export const formatGreeting = (name: string): string => `Hello, ${name}!`;\n", "utf8");

        return {
          status: "succeeded",
          toolCalls: [{
            toolCallId: "tool.real-acp.headless.write-greeter",
            namespace: "bag.acp",
            name: "fs.write",
            status: "succeeded",
            sideEffectLevel: "write",
          }],
          terminalCommands: [{
            commandId: "cmd.real-acp.simple-edit.bun-test",
            command: ["bun", "test", "tests/greeter.test.ts"],
            status: "succeeded",
            exitCode: 0,
            durationMs: 7,
          }],
          telemetry: {
            headlessRunner: "injected-test",
          },
          lineage: {
            parentRunResultId: "real-acp.previous.simple-edit",
          },
        };
      },
    });

    try {
      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.headless.adapter",
        metadata,
        executor,
        executionMode: "headless_acp",
        taskIds: ["real-acp.task.simple-edit-greeting"],
        workspaceBaseDir,
        currentRepoPath,
        createdAt,
      });

      expect(requests).toHaveLength(1);
      expect(manifest.executor).toEqual({
        executorId: "real-acp.executor.headless.test",
        executorVersion: "headless-test.v1",
        kind: "headless_acp",
      });
      expect(manifest.summary).toEqual({
        total: 1,
        passed: 1,
        failed: 0,
        skipped: 0,
        cancelled: 0,
        error: 0,
        holdout: 0,
      });

      const result = RealAcpTaskRunResultSchema.parse(manifest.taskResults[0]);
      expect(result.status).toBe("passed");
      expect(result.changedFiles).toEqual([expect.objectContaining({
        path: "src/greeter.ts",
        changeKind: "modified",
      })]);
      expect(result.lineage).toEqual(expect.objectContaining({
        taskId: "real-acp.task.simple-edit-greeting",
        runResultId: "real-acp-run.headless.adapter.real-acp.task.simple-edit-greeting",
        sourceTaskPackId: realAcpCodingCorpusTaskPack.taskPackId,
        parentRunResultId: "real-acp.previous.simple-edit",
      }));
      expect(result.telemetry).toEqual({
        headlessRunner: "injected-test",
      });
      expect(await readFile(sentinelPath, "utf8")).toBe("do not mutate\n");
    } finally {
      await rm(workspaceBaseDir, { recursive: true, force: true });
      await rm(currentRepoPath, { recursive: true, force: true });
    }
  });

  test("normalizes failure, skip, and cancellation statuses from the injected runner", async () => {
    const currentRepoPath = await makeTempBase("real-acp-headless-status-repo-");
    const workspaceRoot = await makeTempBase("real-acp-headless-status-workspace-");

    const taskStatus = new Map<string, "failure" | "skip" | "canceled">([
      ["real-acp.task.simple-edit-greeting", "failure"],
      ["real-acp.task.cart-bugfix-fail-to-pass", "skip"],
      ["real-acp.task.cancellation-mid-edit", "canceled"],
    ]);

    const executor = createRealAcpHeadlessExecutor({
      currentRepoPath,
      runTask: async (input) => {
        const status = taskStatus.get(input.task.taskId);
        if (status === undefined) {
          throw new Error(`unexpected task ${input.task.taskId}`);
        }
        return {
          status,
          ...(status === "failure" ? { failureReason: "headless verifier failed" } : {}),
          ...(status === "skip" ? { skipReason: "headless verifier intentionally skipped" } : {}),
        };
      },
    });

    try {
      const manifest = await runRealAcpCorpus({
        runId: "real-acp-run.headless.statuses",
        metadata,
        executor,
        executionMode: "headless_acp",
        taskIds: [...taskStatus.keys()],
        workspaceBaseDir: workspaceRoot,
        currentRepoPath,
        createdAt,
      });

      expect(manifest.summary).toEqual({
        total: 3,
        passed: 0,
        failed: 1,
        skipped: 1,
        cancelled: 1,
        error: 0,
        holdout: 0,
      });
      expect(manifest.taskResults.find((result) => result.taskId === "real-acp.task.simple-edit-greeting"))
        .toEqual(expect.objectContaining({
          status: "failed",
          failureReason: "headless verifier failed",
          verifier: expect.objectContaining({ status: "failed" }),
        }));
      expect(manifest.taskResults.find((result) => result.taskId === "real-acp.task.cart-bugfix-fail-to-pass"))
        .toEqual(expect.objectContaining({
          status: "skipped",
          skipReason: "headless verifier intentionally skipped",
          verifier: expect.objectContaining({ status: "skipped" }),
        }));
      expect(manifest.taskResults.find((result) => result.taskId === "real-acp.task.cancellation-mid-edit"))
        .toEqual(expect.objectContaining({
          status: "cancelled",
          route: expect.objectContaining({ selectedMode: "cancelled" }),
          verifier: expect.objectContaining({ status: "not_run" }),
        }));
    } finally {
      await rm(currentRepoPath, { recursive: true, force: true });
      await rm(workspaceRoot, { recursive: true, force: true });
    }
  });

  test("refuses to pass the current repository path to the headless runner", async () => {
    const currentRepoPath = await makeTempBase("real-acp-headless-refuse-repo-");
    const task = taskById("real-acp.task.simple-edit-greeting");
    let called = false;

    const executor = createRealAcpHeadlessExecutor({
      currentRepoPath,
      runTask: async () => {
        called = true;
        return { status: "succeeded" };
      },
    });

    try {
      await expect(executor.executeTask(directExecutorInput(task, currentRepoPath)))
        .rejects.toThrow(/refuses to run against the current repository workspace/);
      expect(called).toBe(false);
    } finally {
      await rm(currentRepoPath, { recursive: true, force: true });
    }
  });

  test("allows an explicit isolated workspace root under the current repository artifact area", async () => {
    const currentRepoPath = await makeTempBase("real-acp-headless-artifact-repo-");
    const workspaceRoot = join(currentRepoPath, ".bag", "replay-corpus", "real-acp-runs", "run-1", "workspaces");
    const task = taskById("real-acp.task.simple-edit-greeting");
    const workspacePath = await materializeTaskWorkspace(task, workspaceRoot);
    let called = false;

    const executor = createRealAcpHeadlessExecutor({
      currentRepoPath,
      allowedWorkspaceRoot: workspaceRoot,
      runTask: async () => {
        called = true;
        return { status: "succeeded" };
      },
    });

    try {
      const output = await executor.executeTask(directExecutorInput(task, workspacePath));
      expect(output.status).toBe("passed");
      expect(called).toBe(true);
    } finally {
      await rm(currentRepoPath, { recursive: true, force: true });
    }
  });

  test("maps a pre-aborted task to cancelled without invoking the runner", async () => {
    const currentRepoPath = await makeTempBase("real-acp-headless-abort-repo-");
    const workspaceRoot = await makeTempBase("real-acp-headless-abort-workspace-");
    const task = taskById("real-acp.task.cancellation-mid-edit");
    const workspacePath = await materializeTaskWorkspace(task, workspaceRoot);
    const controller = new AbortController();
    let called = false;

    controller.abort();
    const executor = createRealAcpHeadlessExecutor({
      currentRepoPath,
      runTask: async () => {
        called = true;
        return { status: "succeeded" };
      },
    });

    try {
      const output = await executor.executeTask(directExecutorInput(task, workspacePath, controller.signal));
      expect(output.status).toBe("cancelled");
      expect(output.route.selectedMode).toBe("cancelled");
      expect(output.verifier.status).toBe("not_run");
      expect(called).toBe(false);
    } finally {
      await rm(currentRepoPath, { recursive: true, force: true });
      await rm(workspaceRoot, { recursive: true, force: true });
    }
  });
});
