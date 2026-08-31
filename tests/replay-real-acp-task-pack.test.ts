import { describe, expect, test } from "bun:test";
import {
  RealAcpTaskLabelSchema,
  deterministicRealAcpTaskSplit,
  realAcpCodingCorpusTaskPack,
  realAcpCorpusTasks,
  realAcpTaskLabelsCovered,
  realAcpTaskSplitDistribution,
  visibleRealAcpCorpusTasksForOptimization,
  type RealAcpCorpusTask,
} from "../src/replay/real-acp-task-pack";

const forbiddenBasenames = new Set([
  ".gitignore",
  "package.json",
  "package-lock.json",
  "pnpm-lock.yaml",
  "yarn.lock",
  "bun.lock",
  "bun.lockb",
]);

const collectTaskPaths = (task: RealAcpCorpusTask): string[] => [
  ...task.workspace.files.map((file) => file.path),
  ...task.workspace.allowedPathPrefixes,
  ...task.workspace.protectedPaths,
  ...task.expectedOutcome.expectedChangedPaths,
  ...task.expectedOutcome.expectedNoChangePaths,
  ...task.expectedOutcome.assertions.flatMap((assertion) => {
    if (assertion.assertionKind === "file_contains" || assertion.assertionKind === "file_not_contains") {
      return [assertion.path];
    }
    if (assertion.assertionKind === "no_forbidden_path_changed") {
      return assertion.paths;
    }
    return [];
  }),
];

describe("real ACP task pack", () => {
  test("defines a balanced label-complete corpus", () => {
    expect(realAcpCodingCorpusTaskPack.taskPackId).toBe("real-acp-run-corpus.task-pack.v1");
    expect(realAcpCorpusTasks).toHaveLength(12);
    expect(realAcpTaskLabelsCovered()).toEqual(RealAcpTaskLabelSchema.options);
    expect(realAcpCorpusTasks.map((task) => task.primaryLabel)).toEqual(RealAcpTaskLabelSchema.options);
    expect(realAcpCorpusTasks.every((task) => task.runTargets.includes("headless_acp"))).toBe(true);
    expect(realAcpCorpusTasks.every((task) => task.runTargets.includes("real_consumer"))).toBe(true);
  });

  test("uses unique task, workspace, and assertion ids", () => {
    const taskIds = realAcpCorpusTasks.map((task) => task.taskId);
    const workspaceIds = realAcpCorpusTasks.map((task) => task.workspace.workspaceId);
    const assertionIds = realAcpCorpusTasks.flatMap((task) =>
      task.expectedOutcome.assertions.map((assertion) => assertion.assertionId)
    );

    expect(new Set(taskIds).size).toBe(taskIds.length);
    expect(new Set(workspaceIds).size).toBe(workspaceIds.length);
    expect(new Set(assertionIds).size).toBe(assertionIds.length);
  });

  test("keeps all task paths fixture-safe and avoids package metadata", () => {
    for (const task of realAcpCorpusTasks) {
      const protectedPaths = new Set(task.workspace.protectedPaths);
      for (const path of collectTaskPaths(task)) {
        expect(path.startsWith("/"), `${task.taskId} uses absolute path ${path}`).toBe(false);
        expect(path.split("/"), `${task.taskId} uses parent traversal in ${path}`).not.toContain("..");
        expect(path.split("/").some((segment) => forbiddenBasenames.has(segment)), `${task.taskId} targets ${path}`)
          .toBe(false);
      }
      for (const changedPath of task.expectedOutcome.expectedChangedPaths) {
        expect(protectedPaths.has(changedPath), `${task.taskId} changes protected path ${changedPath}`).toBe(false);
      }
    }
  });

  test("requires model, codebase, client, and profile metadata for future runs", () => {
    expect(realAcpCodingCorpusTaskPack.runMetadataRequirements.model).toEqual([
      "modelProfileId",
      "provider",
      "model",
      "modelRole",
      "contextWindowTokens",
      "toolCallingMode",
    ]);
    expect(realAcpCodingCorpusTaskPack.runMetadataRequirements.codebase).toEqual([
      "codebaseProfileId",
      "rootFingerprint",
      "languageSummary",
      "testRiskTier",
      "protectedPathPolicy",
    ]);
    expect(realAcpCodingCorpusTaskPack.runMetadataRequirements.client).toEqual([
      "clientProfileId",
      "clientName",
      "clientVersion",
      "transport",
      "acpConsumerCapabilities",
    ]);
    expect(realAcpCodingCorpusTaskPack.runMetadataRequirements.profile).toEqual([
      "policyId",
      "optimizerProfileId",
      "verificationPolicyVersion",
      "resultStyleVersion",
      "canonicalToolVersion",
      "renderedToolVersion",
    ]);
  });

  test("uses deterministic split hints and keeps holdout out of optimizer input", () => {
    expect(realAcpTaskSplitDistribution()).toEqual({
      train: 6,
      dev: 3,
      holdout: 3,
    });

    for (const task of realAcpCorpusTasks) {
      expect(task.split).toBe(deterministicRealAcpTaskSplit(task.splitHint.seedOrdinal));
    }

    const visibleTasks = visibleRealAcpCorpusTasksForOptimization();
    expect(visibleTasks).toHaveLength(9);
    expect(visibleTasks.every((task) => task.split !== "holdout")).toBe(true);
    expect(visibleTasks.every((task) => task.optimizationAllowed)).toBe(true);
    expect(visibleTasks.map((task) => task.primaryLabel)).not.toContain("refactor");
    expect(visibleTasks.map((task) => task.primaryLabel)).not.toContain("rollback");
    expect(visibleTasks.map((task) => task.primaryLabel)).not.toContain("mcp_tool_failure");
  });
});
