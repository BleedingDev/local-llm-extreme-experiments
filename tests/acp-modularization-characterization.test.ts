import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BleedingAcpAgent } from "../src/acp-agent";
import {
  readClientFileThroughAgentForTest,
  replaceRunCodingTurnForTest,
  replaceRunPlanningTurnForTest,
  requireAgentSessionForTest,
  runTerminalCommandThroughAgentForTest,
  telemetryForAgentSession,
  writeClientFileThroughAgentForTest,
} from "./acp-agent-test-harness";

type PermissionOutcome =
  | { outcome: "cancelled" }
  | { outcome: "selected"; optionId: string };

type AcpUpdate = {
  update?: {
    sessionUpdate?: string;
    currentModeId?: string;
    status?: string;
    rawOutput?: Record<string, unknown>;
  };
};

const capableClientCapabilities = {
  fs: { readTextFile: true, writeTextFile: true },
  terminal: true,
};

const currentModeUpdates = (updates: unknown[]): string[] =>
  updates
    .map((item) => (item as AcpUpdate).update)
    .filter((update) => update?.sessionUpdate === "current_mode_update")
    .map((update) => update?.currentModeId)
    .filter((mode): mode is string => typeof mode === "string");

const failedToolUpdates = (updates: unknown[]): Array<NonNullable<AcpUpdate["update"]>> =>
  updates
    .map((item) => (item as AcpUpdate).update)
    .filter((update): update is NonNullable<AcpUpdate["update"]> =>
      update?.sessionUpdate === "tool_call_update" && update.status === "failed"
    );

const createHarness = (options: { terminalExitCode?: number; terminalOutput?: string } = {}) => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-acp-modularization-"));
  const updates: unknown[] = [];
  const reads: string[] = [];
  const writes: Array<{ path: string; content: string }> = [];
  const permissionRequests: unknown[] = [];
  const permissionOutcomes: PermissionOutcome[] = [];
  const terminalRuns: Array<{ command: string; args: string[]; cwd: string }> = [];
  const agent = new BleedingAcpAgent(
    {
      sessionUpdate: async (update: unknown) => {
        updates.push(update);
      },
      readTextFile: async (input: { path: string }) => {
        reads.push(input.path);
        return { content: `client read: ${input.path}\n` };
      },
      writeTextFile: async (input: { path: string; content: string }) => {
        writes.push(input);
        return {};
      },
      requestPermission: async (input: unknown) => {
        permissionRequests.push(input);
        return { outcome: permissionOutcomes.shift() ?? { outcome: "selected", optionId: "allow" } };
      },
      createTerminal: async (input: { command: string; args: string[]; cwd: string }) => {
        terminalRuns.push(input);
        return {
          id: `terminal-${terminalRuns.length}`,
          waitForExit: async () => ({ exitCode: options.terminalExitCode ?? 0, signal: null }),
          currentOutput: async () => ({ output: options.terminalOutput ?? "ok\n" }),
          kill: async () => ({}),
          release: async () => ({}),
        };
      },
    } as never,
    cwd,
  );
  return { agent, cwd, permissionOutcomes, permissionRequests, reads, terminalRuns, updates, writes };
};

describe("BleedingAgent ACP modularization characterization", () => {
  test("explicit /yolo after /safe restores no-prompt writes and terminals", async () => {
    const { agent, cwd, permissionRequests, terminalRuns, updates, writes } = createHarness();
    const session = await agent.newSession({ cwd, mcpServers: [], additionalDirectories: [] } as never);
    const telemetry = telemetryForAgentSession(agent, session.sessionId, cwd, "test-acp-yolo-after-safe");

    await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/safe" }] } as never);
    expect(requireAgentSessionForTest(agent, session.sessionId).yolo).toBe(false);
    await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/yolo" }] } as never);
    expect(requireAgentSessionForTest(agent, session.sessionId).yolo).toBe(true);

    await expect(writeClientFileThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      path: join(cwd, "example.ts"),
      oldContent: "export const value = 1;\n",
      newContent: "export const value = 2;\n",
      reason: "characterize yolo write after safe",
    })).resolves.toMatchObject({ ok: true });
    await expect(runTerminalCommandThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      command: "npm",
      args: ["test"],
      reason: "characterize yolo terminal after safe",
      cwd,
    })).resolves.toMatchObject({ exitCode: 0 });

    expect(permissionRequests).toHaveLength(0);
    expect(writes).toHaveLength(1);
    expect(terminalRuns).toHaveLength(1);
    expect(JSON.stringify(updates)).toContain("YOLO mode enabled");
  });

  test("temporary /run and /plan modes restore auto when runners throw or cancel", async () => {
    const { agent, cwd, updates } = createHarness();
    const session = await agent.newSession({ cwd, mcpServers: [], additionalDirectories: [] } as never);

    replaceRunCodingTurnForTest(agent, async () => {
      throw new Error("forced coding runner failure");
    });
    await expect(
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/run trigger failure" }] } as never),
    ).rejects.toThrow("forced coding runner failure");
    expect(requireAgentSessionForTest(agent, session.sessionId).mode).toBe("auto");

    replaceRunPlanningTurnForTest(agent, async (_session, _task, signal) => {
      await agent.cancel({ sessionId: session.sessionId });
      if (signal.aborted) {
        throw new Error("cancelled");
      }
    });
    await expect(
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/plan trigger cancel" }] } as never),
    ).resolves.toEqual({ stopReason: "cancelled" });
    expect(requireAgentSessionForTest(agent, session.sessionId).mode).toBe("auto");
    expect(currentModeUpdates(updates)).toEqual(["run", "auto", "plan", "auto"]);
  });

  test("runTerminalCommand returns and publishes failed updates for nonzero exits", async () => {
    const { agent, cwd, terminalRuns, updates } = createHarness({
      terminalExitCode: 7,
      terminalOutput: "verification failed\n",
    });
    const session = await agent.newSession({ cwd, mcpServers: [], additionalDirectories: [] } as never);
    const telemetry = telemetryForAgentSession(agent, session.sessionId, cwd, "test-acp-terminal-nonzero");

    const result = await runTerminalCommandThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      command: "npm",
      args: ["test"],
      reason: "characterize nonzero terminal exit",
      cwd,
    });

    expect(result).toEqual(expect.objectContaining({ exitCode: 7, output: "verification failed\n" }));
    expect(terminalRuns).toEqual([expect.objectContaining({ command: "npm", args: ["test"], cwd })]);
    expect(failedToolUpdates(updates)).toEqual([
      expect.objectContaining({ rawOutput: expect.objectContaining({ exitCode: 7 }) }),
    ]);
  });

  test("cancelled Safe permission prompts surface as cancelled write and terminal outcomes", async () => {
    const { agent, cwd, permissionOutcomes, permissionRequests, terminalRuns, updates, writes } = createHarness();
    const session = await agent.newSession({ cwd, mcpServers: [], additionalDirectories: [] } as never);
    const telemetry = telemetryForAgentSession(agent, session.sessionId, cwd, "test-acp-permission-cancelled");

    await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/safe" }] } as never);
    permissionOutcomes.push({ outcome: "cancelled" }, { outcome: "cancelled" });

    await expect(writeClientFileThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      path: join(cwd, "cancelled-write.ts"),
      oldContent: "",
      newContent: "export const value = 1;\n",
      reason: "characterize cancelled write permission",
    })).rejects.toThrow("cancelled");
    await expect(runTerminalCommandThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      command: "npm",
      args: ["test"],
      reason: "characterize cancelled terminal permission",
      cwd,
    })).rejects.toThrow("cancelled");

    expect(permissionRequests).toHaveLength(2);
    expect(writes).toHaveLength(0);
    expect(terminalRuns).toHaveLength(0);
    expect(failedToolUpdates(updates).map((update) => update.rawOutput?.outcome)).toEqual(["cancelled", "cancelled"]);
  });

  test("additionalDirectories flow through agent read, write, and terminal helpers", async () => {
    const { agent, cwd, reads, terminalRuns, writes } = createHarness();
    const extra = mkdtempSync(join(tmpdir(), "bag-acp-extra-root-"));
    await agent.initialize({ protocolVersion: 1, clientCapabilities: capableClientCapabilities } as never);
    const session = await agent.newSession({ cwd, mcpServers: [], additionalDirectories: [extra] } as never);
    const telemetry = telemetryForAgentSession(agent, session.sessionId, cwd, "test-acp-additional-directories");
    const extraReadPath = join(extra, "readme.md");
    const extraWritePath = join(extra, "generated.ts");

    await expect(readClientFileThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      path: extraReadPath,
    })).resolves.toBe(`client read: ${extraReadPath}\n`);
    await expect(writeClientFileThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      path: extraWritePath,
      oldContent: "",
      newContent: "export const generated = true;\n",
      reason: "characterize additional directory write",
    })).resolves.toMatchObject({ ok: true });
    await expect(runTerminalCommandThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      command: "pwd",
      args: [],
      reason: "characterize additional directory terminal",
      cwd: extra,
    })).resolves.toMatchObject({ exitCode: 0 });

    expect(reads).toEqual([extraReadPath]);
    expect(writes).toEqual([expect.objectContaining({ path: extraWritePath })]);
    expect(terminalRuns).toEqual([expect.objectContaining({ cwd: extra })]);
  });
});
