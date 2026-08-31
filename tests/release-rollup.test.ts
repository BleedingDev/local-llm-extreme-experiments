import { describe, expect, test } from "bun:test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { BleedingAcpAgent } from "../src/acp-agent";
import {
  checkPostApplyConsistencyThroughAgentForTest,
  previewAndWriteClientEditThroughAgentForTest,
  requireAgentSessionForTest,
  runTerminalCommandThroughAgentForTest,
  telemetryForAgentSession,
} from "./acp-agent-test-harness";

type SessionUpdate = {
  sessionId: string;
  update: Record<string, unknown>;
};

type PermissionRequest = {
  sessionId: string;
  toolCall: Record<string, unknown>;
  options?: Array<{ optionId: string; kind: string }>;
};

type WriteRequest = {
  sessionId: string;
  path: string;
  content: string;
};

const sha256 = (content: string): string => createHash("sha256").update(content).digest("hex");

const textPrompt = (text: string) => ({
  prompt: [{ type: "text", text }],
});

const agentMessages = (updates: SessionUpdate[]): string[] =>
  updates.flatMap((entry) => {
    if (entry.update.sessionUpdate !== "agent_message_chunk") {
      return [];
    }
    const content = entry.update.content as { text?: unknown } | undefined;
    return typeof content?.text === "string" ? [content.text] : [];
  });

const updatePayloads = (updates: SessionUpdate[], sessionUpdate: string): Record<string, unknown>[] =>
  updates.map((entry) => entry.update).filter((update) => update.sessionUpdate === sessionUpdate);

const createHarnessConnection = (workdir: string) => {
  const updates: SessionUpdate[] = [];
  const writes: WriteRequest[] = [];
  const permissionRequests: PermissionRequest[] = [];
  const terminals: Array<{ id: string; command: string; args: string[]; cwd: string }> = [];

  return {
    updates,
    writes,
    permissionRequests,
    terminals,

    async sessionUpdate(params: SessionUpdate): Promise<void> {
      updates.push(params);
    },

    async readTextFile(params: { path: string; line?: number; limit?: number }): Promise<{ content: string }> {
      const path = resolve(workdir, params.path);
      const raw = readFileSync(path, "utf8");
      if (typeof params.line !== "number" && typeof params.limit !== "number") {
        return { content: raw };
      }
      const start = Math.max(0, (params.line ?? 1) - 1);
      const end = params.limit == null ? undefined : start + params.limit;
      return { content: raw.split("\n").slice(start, end).join("\n") };
    },

    async writeTextFile(params: WriteRequest): Promise<Record<string, never>> {
      const path = resolve(workdir, params.path);
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, params.content);
      writes.push({ ...params, path });
      return {};
    },

    async requestPermission(params: PermissionRequest): Promise<{ outcome: { outcome: "selected"; optionId: string } }> {
      permissionRequests.push(params);
      const allow = params.options?.find((option) => option.kind.startsWith("allow")) ?? params.options?.[0];
      return { outcome: { outcome: "selected", optionId: allow?.optionId ?? "allow" } };
    },

    async createTerminal(params: {
      command: string;
      args?: string[];
      cwd?: string;
      outputByteLimit?: number;
    }): Promise<{
      id: string;
      waitForExit: () => Promise<{ exitCode: number | null; signal: string | null }>;
      currentOutput: () => Promise<{ output: string; truncated: boolean }>;
      release: () => Promise<void>;
      kill: () => Promise<void>;
    }> {
      const id = `term-${terminals.length + 1}`;
      const cwd = params.cwd ?? workdir;
      const args = params.args ?? [];
      const proc: ChildProcessWithoutNullStreams = spawn(params.command, args, { cwd, env: process.env });
      let output = "";
      let truncated = false;
      const append = (chunk: Buffer) => {
        output += chunk.toString("utf8");
        if (params.outputByteLimit != null && Buffer.byteLength(output) > params.outputByteLimit) {
          const buffer = Buffer.from(output, "utf8");
          output = buffer.slice(buffer.length - params.outputByteLimit).toString("utf8");
          truncated = true;
        }
      };
      proc.stdout.on("data", append);
      proc.stderr.on("data", append);
      const exitPromise = new Promise<{ exitCode: number | null; signal: string | null }>((resolveExit) => {
        proc.once("close", (exitCode, signal) => resolveExit({ exitCode, signal: signal == null ? null : String(signal) }));
      });
      terminals.push({ id, command: params.command, args, cwd });
      return {
        id,
        waitForExit: () => exitPromise,
        currentOutput: async () => ({ output, truncated }),
        release: async () => {},
        kill: async () => {
          proc.kill("SIGKILL");
        },
      };
    },
  };
};

describe("BleedingAgent release rollup dogfood harness", () => {
  test("runs deterministic ACP-style chat, maintenance, edit, terminal, trace, and optimize flow", async () => {
    const workdir = await mkdtemp(join(tmpdir(), "bleeding-agent-release-rollup-"));
    const targetPath = join(workdir, "src", "rollup-fixture.ts");
    mkdirSync(dirname(targetPath), { recursive: true });
    writeFileSync(targetPath, "export const releaseRollup = 'before';\n");

    const connection = createHarnessConnection(workdir);
    const agent = new BleedingAcpAgent(connection as never, workdir);

    try {
      const init = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {
          fs: {
            readTextFile: true,
            writeTextFile: true,
          },
          terminal: true,
        },
      } as never);
      expect(init.agentInfo.name).toBe("bleeding-agent");
      expect(init.agentCapabilities.sessionCapabilities?.resume).toEqual({});

      const sessionResponse = await agent.newSession({ cwd: workdir, additionalDirectories: [], mcpServers: [] } as never);
      const sessionId = sessionResponse.sessionId;
      const session = requireAgentSessionForTest(agent, sessionId);
      expect(session).toBeDefined();

      await agent.prompt({ sessionId, ...textPrompt("/chat") } as never);
      await agent.prompt({ sessionId, ...textPrompt("hello, show the user-facing ACP surface only") } as never);
      expect(agentMessages(connection.updates).join("\n")).toContain("Ahoj. Jsem BleedingAgent ACP coding agent");
      expect(connection.writes).toHaveLength(0);
      expect(connection.terminals).toHaveLength(0);

      await agent.prompt({ sessionId, ...textPrompt("/maintenance status") } as never);
      const planUpdates = updatePayloads(connection.updates, "plan");
      expect(planUpdates.some((update) => JSON.stringify(update).includes("Inspect optimizer registry"))).toBe(true);
      expect(agentMessages(connection.updates).join("\n")).toContain("Maintenance optimizer status:");

      await agent.prompt({ sessionId, ...textPrompt("/safe") } as never);
      const original = readFileSync(targetPath, "utf8");
      const telemetry = telemetryForAgentSession(agent, sessionId, workdir, "release-rollup-dogfood");
      const editResult = await previewAndWriteClientEditThroughAgentForTest(agent, {
        session,
        telemetry,
        path: targetPath,
        oldContent: original,
        newContent: "export const releaseRollup = 'after';\n",
        reason: "Release rollup dogfood fixture edit through ACP write boundary.",
      });
      expect(editResult).toMatchObject({
        ok: true,
        editStrategyId: "edit.whole-file.acp-write.v1",
        editStatus: "applied",
      });
      expect(connection.permissionRequests).toHaveLength(1);
      expect(connection.writes.map((write) => write.path)).toEqual([targetPath]);
      expect(readFileSync(targetPath, "utf8")).toBe("export const releaseRollup = 'after';\n");

      const consistency = await checkPostApplyConsistencyThroughAgentForTest(agent, {
        session,
        telemetry,
        editResults: [editResult],
      });
      expect(consistency).toContainEqual(expect.objectContaining({
        path: "src/rollup-fixture.ts",
        status: "consistent",
        actualHash: sha256("export const releaseRollup = 'after';\n"),
      }));

      const terminalResult = await runTerminalCommandThroughAgentForTest(agent, {
        sessionId,
        telemetry,
        command: process.execPath,
        args: ["-e", "console.log('release-rollup-verification')"],
        cwd: workdir,
        reason: "Release rollup dogfood terminal verification.",
      });
      expect(terminalResult).toMatchObject({
        exitCode: 0,
        output: expect.stringContaining("release-rollup-verification"),
      });
      expect(connection.permissionRequests).toHaveLength(2);
      expect(connection.terminals).toHaveLength(1);

      await agent.prompt({ sessionId, ...textPrompt("/traces") } as never);
      expect(agentMessages(connection.updates).join("\n")).toContain("HALO-style trace dataset:");
      expect(agentMessages(connection.updates).join("\n")).toContain("- spans:");

      await agent.prompt({ sessionId, ...textPrompt("/maintenance optimize") } as never);
      expect(agentMessages(connection.updates).join("\n")).toContain("Maintenance optimize report:");
      const completedToolUpdates = updatePayloads(connection.updates, "tool_call_update").filter(
        (update) => update.status === "completed",
      );
      expect(completedToolUpdates.some((update) => JSON.stringify(update.rawOutput).includes("bag.maintenance.optimize")))
        .toBe(true);
    } finally {
      await rm(workdir, { recursive: true, force: true });
    }
  });
});
