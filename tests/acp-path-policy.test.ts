import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BleedingAcpAgent } from "../src/acp-agent";
import { createAcpOptimizerSessionPin, createBagAcpSession } from "../src/acp/session";
import { acpClientCapabilityProfileFromInitialize } from "../src/acp/surface";
import { runTerminalCommand } from "../src/acp/terminal";
import { readClientFile, writeClientFileWithPermission } from "../src/acp/workspace-io";
import { defaultConfig } from "../src/config";
import { RunTelemetry } from "../src/telemetry";
import { requireAgentSessionForTest } from "./acp-agent-test-harness";

const capableClientCapabilities = {
  fs: { readTextFile: true, writeTextFile: true },
  terminal: true,
};

describe("BleedingAgent ACP path policy", () => {
  test("blocks ACP reads, writes, and terminals outside the session roots before client calls", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-path-root-"));
    const outside = mkdtempSync(join(tmpdir(), "bag-acp-path-outside-"));
    const outsideFile = join(outside, "secret.txt");
    writeFileSync(outsideFile, "secret\n");

    const reads: string[] = [];
    const writes: string[] = [];
    const terminals: string[] = [];
    const config = defaultConfig();
    const sessions = new Map();
    const moduleSession = createBagAcpSession({
      config,
      sessions,
      cwd,
      additionalDirectories: [],
      id: "bag-path-policy-test",
      mcpServers: [],
      clientCapabilities: acpClientCapabilityProfileFromInitialize(capableClientCapabilities, "test"),
      createOptimizerSessionPin: (resolvedCwd) => createAcpOptimizerSessionPin(config, resolvedCwd),
    });
    const telemetry = new RunTelemetry(config, "test-acp-path-policy", cwd, moduleSession.optimizerPin.telemetry);
    const moduleDeps = {
      connection: {
        sessionUpdate: async () => {},
        readTextFile: async (input: { path: string }) => {
          reads.push(input.path);
          return { content: "should not read\n" };
        },
        writeTextFile: async (input: { path: string }) => {
          writes.push(input.path);
          return {};
        },
        requestPermission: async () => ({ outcome: { outcome: "selected", optionId: "allow" } }),
        createTerminal: async (input: { cwd: string }) => {
          terminals.push(input.cwd);
          return {
            id: "terminal",
            waitForExit: async () => ({ exitCode: 0, signal: null }),
            currentOutput: async () => ({ output: "" }),
            kill: async () => ({}),
            release: async () => ({}),
          };
        },
      } as never,
      requireSession: () => moduleSession,
    };

    await expect(readClientFile({
      ...moduleDeps,
      runAcpTool: async <T>(input: { fn: () => Promise<unknown> }): Promise<T> => (await input.fn()) as T,
    }, { sessionId: moduleSession.id, telemetry, path: outsideFile }))
      .rejects.toThrow("escapes allowed roots");
    await expect(writeClientFileWithPermission({
      ...moduleDeps,
      runAcpTool: async <T>(input: { fn: () => Promise<unknown> }): Promise<T> => (await input.fn()) as T,
    }, {
      sessionId: moduleSession.id,
      telemetry,
      path: outsideFile,
      oldContent: "",
      newContent: "x\n",
      reason: "should be blocked",
    })).rejects.toThrow("escapes allowed roots");
    await expect(runTerminalCommand(moduleDeps, {
      sessionId: moduleSession.id,
      telemetry,
      command: "pwd",
      args: [],
      cwd: outside,
      reason: "should be blocked",
    })).rejects.toThrow("escapes allowed roots");

    expect(reads).toEqual([]);
    expect(writes).toEqual([]);
    expect(terminals).toEqual([]);
  });

  test("applies executor-concurrency session config to the live session state", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-config-"));
    const agent = new BleedingAcpAgent({ sessionUpdate: async () => {} } as never, cwd);
    await agent.initialize({ protocolVersion: 1, clientCapabilities: capableClientCapabilities } as never);
    const session = await agent.newSession({ cwd, mcpServers: [], additionalDirectories: [] } as never);

    const response = await agent.setSessionConfigOption({
      sessionId: session.sessionId,
      configId: "executor-concurrency",
      value: "20",
    } as never);
    const currentValue = response.configOptions
      ?.find((option) => option.id === "executor-concurrency")
      ?.currentValue;

    expect(currentValue).toBe("20");
    expect(requireAgentSessionForTest(agent, session.sessionId).executorConcurrency).toBe(20);
  });
});
