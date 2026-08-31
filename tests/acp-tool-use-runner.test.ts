import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { AgentSideConnection as AcpConnection, TerminalHandle } from "@agentclientprotocol/sdk";
import type { ChatWithToolsOptions, ChatWithToolsResult, LlmRouter } from "../src/llm";
import { createAcpOptimizerSessionPin, type BagAcpSession } from "../src/acp/session";
import { runLiveMcpToolCall } from "../src/acp/mcp-bridge";
import {
  runAcpAdaptiveCodingTurn,
  runAcpAutonomousToolUseTurn,
  runAcpDagDrivenToolUseTurn,
  type AcpToolUseRunnerDeps,
} from "../src/acp/tool-use-runner";
import { defaultConfig } from "../src/config";
import { connectMcpStdioRuntimeServer } from "../src/mcp/runtime-tools";
import { RunTelemetry } from "../src/telemetry";

const noMasterConfig = () => {
  const config = defaultConfig();
  return {
    ...config,
    master: {
      ...config.master,
      apiKeyEnv: "BAG_TEST_OPENAI_KEY_THAT_SHOULD_NOT_EXIST",
    },
  };
};

const sessionFor = (cwd: string): BagAcpSession => ({
  id: "bag-tool-use-test",
  cwd,
  additionalDirectories: [],
  executorConcurrency: 8,
  mode: "auto",
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
  pendingPrompt: null,
  title: "test",
  yolo: true,
  mcpServers: [],
  optimizerPin: { telemetry: {} } as never,
  clientCapabilities: {
    fsReadTextFile: true,
    fsWriteTextFile: true,
    terminal: true,
    richDiffContent: true,
    richTerminalContent: true,
    source: "test",
  },
});

const depsFor = (messages: string[]): AcpToolUseRunnerDeps => ({
  connection: { sessionUpdate: async () => {} } as unknown as AcpConnection,
  config: noMasterConfig(),
  agentMessage: async (_sessionId, text) => {
    messages.push(text);
  },
});

const controlledMcpFixturePath = (): string =>
  join(dirname(fileURLToPath(import.meta.url)), "fixtures", "controlled-mcp-server.mjs");

describe("ACP tool-use runner module", () => {
  test("autonomous, DAG, and adaptive runners preserve no-master diagnostics through injected ACP deps", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-tool-use-runner-"));
    const session = sessionFor(cwd);
    const messages: string[] = [];
    const deps = depsFor(messages);

    await runAcpAutonomousToolUseTurn(deps, { session, task: "fix a bug", signal: new AbortController().signal });
    await runAcpDagDrivenToolUseTurn(deps, { session, task: "fix a bug", signal: new AbortController().signal });
    await runAcpAdaptiveCodingTurn(deps, { session, task: "fix a bug", signal: new AbortController().signal });

    const text = messages.join("\n");
    expect(text).toContain("Autonomous tool-use turn started");
    expect(text).toContain("No master model is configured; cannot run autonomous mode.");
    expect(text).toContain("DAG-driven tool-use turn started");
    expect(text).toContain("No master model is configured; cannot run DAG-tools mode.");
    expect(text).toContain("Adaptive coding turn started");
    expect(text).toContain("No master model is configured; cannot run adaptive mode.");
  });
});

describe("ACP live MCP tool-use loop", () => {
  test("records ACP telemetry and lineage for a real stdio MCP read", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-stdio-mcp-"));
    const session = {
      ...sessionFor(cwd),
      yolo: false,
      optimizerPin: createAcpOptimizerSessionPin(defaultConfig(), cwd),
    };
    const updates: unknown[] = [];
    const connection = {
      sessionUpdate: async (update: unknown) => {
        updates.push(update);
      },
      requestPermission: async () => {
        throw new Error("read-only MCP call should not ask for permission");
      },
    } as unknown as AcpConnection;
    const telemetry = new RunTelemetry(defaultConfig(), "test-acp-stdio-mcp", cwd, session.optimizerPin.telemetry);
    const transport = await connectMcpStdioRuntimeServer({
      serverId: "controlled-fixture",
      name: "controlled-fixture",
      command: "node",
      args: [controlledMcpFixturePath()],
      startupTimeoutMs: 2_000,
      requestTimeoutMs: 2_000,
    });

    try {
      const result = await runLiveMcpToolCall({ connection }, {
        session,
        telemetry,
        server: transport.server,
        call: {
          callId: "call.acp.stdio.read",
          toolName: "read_note",
          arguments: { id: "alpha" },
        },
        executor: transport.executor,
      });

      expect(result).toMatchObject({
        ok: true,
        status: "success",
        result: {
          structuredContent: {
            id: "alpha",
            note: "controlled fixture note",
          },
        },
        trace: {
          serverId: "controlled-fixture",
          toolName: "read_note",
          permissionStatus: "not_required",
          sideEffectLevel: "read",
        },
      });
      expect(telemetry.toolMetrics).toHaveLength(1);
      expect(telemetry.toolMetrics[0]).toMatchObject({
        namespace: "mcp",
        ok: true,
        retryCount: 0,
        resultKind: "json",
      });
      expect(telemetry.toolMetrics[0]!.durationMs).toBeGreaterThanOrEqual(0);
      expect(telemetry.toolMetrics[0]!.resultBytes).toBeGreaterThan(0);
      expect(JSON.stringify(updates)).toContain('"sessionUpdate":"tool_call"');
      expect(JSON.stringify(updates)).toContain('"sessionUpdate":"tool_call_update"');
      expect(JSON.stringify(updates)).toContain(`"modelProfileId":"${session.optimizerPin.telemetry.modelProfileId}"`);
    } finally {
      await transport.close();
    }
  });

  test("renders fake MCP tools into the model tool list and executes them through ACP before normal bash completion", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-tool-use-mcp-loop-"));
    const session = {
      ...sessionFor(cwd),
      optimizerPin: createAcpOptimizerSessionPin(defaultConfig(), cwd),
    };
    const updates: unknown[] = [];
    const toolsSeen: string[][] = [];
    const messagesSeen: unknown[][] = [];
    const executorRequests: unknown[] = [];
    let terminalId = 0;
    let toolTurn = 0;
    const router: LlmRouter = {
      masterAvailable: true,
      localAvailable: async () => true,
      chatText: async (options) =>
        options.purpose === "pre-submit-self-check"
          ? JSON.stringify({ complete: true, missing: [] })
          : JSON.stringify({ probes: [] }),
      chatTextWithTools: async (options: ChatWithToolsOptions): Promise<ChatWithToolsResult> => {
        toolsSeen.push(options.tools.map((tool) => tool.function.name));
        messagesSeen.push([...options.messages]);
        const mcpToolName = options.tools.find((tool) => tool.function.name.startsWith("mcp_workspace_read_file_"))
          ?.function.name;
        if (toolTurn === 0) {
          toolTurn += 1;
          expect(mcpToolName).toBeDefined();
          return {
            finishReason: "tool_calls",
            textContent: "",
            toolCalls: [{
              id: "call-mcp-read",
              name: mcpToolName!,
              argumentsJson: JSON.stringify({ path: "README.md" }),
            }],
            promptTokens: 11,
            completionTokens: 3,
          };
        }
        return {
          finishReason: "tool_calls",
          textContent: "",
          toolCalls: [{
            id: "call-bash-submit",
            name: "bash",
            argumentsJson: JSON.stringify({ command: "echo BAG_TASK_COMPLETE" }),
          }],
          promptTokens: 7,
          completionTokens: 2,
        };
      },
    };
    const connection = {
      sessionUpdate: async (update: unknown) => {
        updates.push(update);
      },
      createTerminal: async () => {
        terminalId += 1;
        return {
          id: `terminal-${terminalId}`,
          waitForExit: async () => ({ exitCode: 0, signal: null }),
          currentOutput: async () => ({
            output: "BAG_TASK_COMPLETE\n",
            truncated: false,
            exitStatus: { exitCode: 0, signal: null },
          }),
          release: async () => {},
        } as TerminalHandle;
      },
    } as unknown as AcpConnection;
    const messages: string[] = [];
    const deps: AcpToolUseRunnerDeps = {
      connection,
      config: defaultConfig(),
      agentMessage: async (_sessionId, text) => {
        messages.push(text);
      },
      createRouter: () => router,
      mcpRuntimeServers: () => [{
        server: {
          serverId: "workspace",
          name: "workspace",
          displayName: "Workspace tools",
          tools: [{
            name: "read_file",
            description: "Read a file from the workspace.",
            inputSchema: {
              type: "object",
              properties: { path: { type: "string" } },
              required: ["path"],
              additionalProperties: false,
            },
            annotations: { readOnlyHint: true },
          }],
        },
        executor: async (request) => {
          executorRequests.push(request);
          return { content: `read ${request.arguments.path}` };
        },
      }],
    };

    await runAcpAutonomousToolUseTurn(deps, {
      session,
      task: "read through MCP, then finish",
      signal: new AbortController().signal,
    });

    expect(executorRequests).toHaveLength(1);
    expect(executorRequests[0]).toMatchObject({
      toolName: "read_file",
      arguments: { path: "README.md" },
    });
    expect(toolsSeen[0]).toEqual(expect.arrayContaining(["bash", expect.stringMatching(/^mcp_workspace_read_file_/)]));
    expect(JSON.stringify(messagesSeen[1])).toContain("call-mcp-read");
    expect(JSON.stringify(messagesSeen[1])).toContain("read README.md");
    expect(JSON.stringify(updates)).toContain('"sessionUpdate":"tool_call"');
    expect(JSON.stringify(updates)).toContain('"sessionUpdate":"tool_call_update"');
    expect(messages.join("\n")).toContain("Autonomous turn complete: submitted");
  });
});
