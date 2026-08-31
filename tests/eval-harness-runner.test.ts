import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  detectChangedFiles,
  detectProtectedPathChanges,
  materializeFixtureWorkspace,
  runCommand,
  runEvalCase,
  snapshotWorkspace,
} from "../src/eval-harness/runner";
import type { EvalCase, EvalComparableContext, FixtureWorkspace } from "../src/eval-harness/types";

const context: EvalComparableContext = {
  policyId: "policy.qwen36.bleeding-agent",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  modelServerId: "server.local-mlx",
  modelServerProfileId: "server-profile.qwen36.rotorquant",
  canonicalToolVersion: "canonical-tools.v1",
  renderedToolVersion: "rendered-tools.v1",
  resultStyleVersion: "result-style.v1",
  verificationPolicyVersion: "verification.v1",
};

const fixtureWorkspace: FixtureWorkspace = {
  fixtureWorkspaceId: "fixture.runner-basic",
  name: "Runner basic fixture",
  rootFingerprint: "sha256:runner-basic",
  files: [
    {
      path: "src/banner.txt",
      content: "Welcome, PROJECT_NAME.\n",
    },
    {
      path: "package.json",
      content: "{\n  \"private\": true\n}\n",
    },
  ],
  protectedPaths: ["package.json"],
  setupCommands: [],
  verificationCommands: [],
};

const evalCase: EvalCase = {
  evalCaseId: "eval.runner-basic",
  schemaVersion: "eval-case.v1",
  split: "dev",
  title: "Runner basic eval",
  task: "Update src/banner.txt and leave package.json unchanged.",
  fixtureWorkspace,
  assertions: [
    {
      assertionId: "assert.banner-updated",
      assertionKind: "file_contains",
      description: "Banner names the project.",
      path: "src/banner.txt",
      text: "BleedingAgent",
      severity: "failure",
    },
    {
      assertionId: "assert.package-unchanged",
      assertionKind: "no_forbidden_path_changed",
      description: "Manifest remains untouched.",
      paths: ["package.json"],
      severity: "critical",
    },
  ],
  tags: ["runner"],
  timeoutMs: 5000,
};

const makeTempBase = (): Promise<string> => mkdtemp(join(tmpdir(), "bleeding-agent-runner-test-"));

describe("eval harness runner", () => {
  test("materializes fixture workspaces into a temp directory", async () => {
    const baseDir = await makeTempBase();
    try {
      const materialized = await materializeFixtureWorkspace(fixtureWorkspace, { baseDir });
      try {
        expect(materialized.workspacePath.startsWith(baseDir)).toBe(true);
        expect(await readFile(join(materialized.workspacePath, "src/banner.txt"), "utf8")).toBe(
          "Welcome, PROJECT_NAME.\n",
        );
        expect(await readFile(join(materialized.workspacePath, "package.json"), "utf8")).toContain(
          "\"private\": true",
        );
      } finally {
        await materialized.cleanup();
      }
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("rejects unsafe fixture paths before writing files", async () => {
    const baseDir = await makeTempBase();
    try {
      await expect(materializeFixtureWorkspace({
        ...fixtureWorkspace,
        files: [
          {
            path: "../outside.txt",
            content: "escape\n",
          },
        ],
      }, { baseDir })).rejects.toThrow();
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("detects changed files and protected path changes", async () => {
    const baseDir = await makeTempBase();
    try {
      const materialized = await materializeFixtureWorkspace(fixtureWorkspace, { baseDir });
      try {
        const before = await snapshotWorkspace(materialized.workspacePath);
        await writeFile(join(materialized.workspacePath, "src/banner.txt"), "Welcome, BleedingAgent.\n");
        await writeFile(join(materialized.workspacePath, "package.json"), "{\n  \"private\": false\n}\n");
        const after = await snapshotWorkspace(materialized.workspacePath);
        const changedFiles = detectChangedFiles(before, after);

        expect(changedFiles.map((file) => file.path)).toEqual(["package.json", "src/banner.txt"]);
        expect(detectProtectedPathChanges(changedFiles, ["package.json"]).map((file) => file.path)).toEqual([
          "package.json",
        ]);
      } finally {
        await materialized.cleanup();
      }
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("captures command exit codes and timeout state", async () => {
    const baseDir = await makeTempBase();
    try {
      const materialized = await materializeFixtureWorkspace(fixtureWorkspace, { baseDir });
      try {
        const failed = await runCommand({
          commandId: "verify.exit",
          command: ["node", "-e", "process.exit(7)"],
          cwd: materialized.workspacePath,
          timeoutMs: 2000,
        });
        expect(failed.exitCode).toBe(7);
        expect(failed.timedOut).toBe(false);

        const timedOut = await runCommand({
          commandId: "verify.timeout",
          command: ["node", "-e", "setTimeout(() => {}, 5000)"],
          cwd: materialized.workspacePath,
          timeoutMs: 50,
        });
        expect(timedOut.timedOut).toBe(true);
        expect(timedOut.exitCode).toBeNull();
      } finally {
        await materialized.cleanup();
      }
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("roundtrips an injected candidate executor into an EvalRunResult", async () => {
    const baseDir = await makeTempBase();
    try {
      const execution = await runEvalCase({
        evalCase,
        runRole: "candidate",
        context,
        comparisonRunId: "compare.runner-candidate",
        candidatePatchId: "candidate.runner-basic",
        baseDir,
        executor: async (workspacePath, receivedEvalCase) => {
          expect(receivedEvalCase.evalCaseId).toBe(evalCase.evalCaseId);
          await writeFile(join(workspacePath, "src/banner.txt"), "Welcome, BleedingAgent.\n");
        },
      });
      try {
        expect(execution.result.status).toBe("passed");
        expect(execution.result.score).toBe(1);
        expect(execution.result.changedFiles).toEqual(["src/banner.txt"]);
        expect(execution.result.telemetryArtifactPath).toMatch(/^\.bag\/evals\/run\.candidate\./);
        expect(execution.protectedPathChanges).toEqual([]);
      } finally {
        await execution.cleanup();
      }
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });
});
