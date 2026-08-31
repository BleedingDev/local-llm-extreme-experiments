import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { runAutonomousCodingTurn } from "../src/autonomous-coding-turn";
import type {
  AcpTerminalClient,
  AutonomousToolResult,
} from "../src/autonomous-tools";
import type {
  CodebaseSearchBackend,
  CodebaseSearchHit,
} from "../src/codebase-index/colgrep-bridge";
import type {
  ChatOptions,
  ChatWithToolsOptions,
  ChatWithToolsResult,
  LlmRouter,
} from "../src/llm";

type ToolCallScript = {
  toolName: string;
  argumentsJson: string;
  textContent?: string;
};

const buildToolCallResponse = (script: ToolCallScript): ChatWithToolsResult => ({
  finishReason: "tool_calls",
  textContent: script.textContent ?? "",
  toolCalls: [
    {
      id: `tool-${Math.random().toString(36).slice(2, 10)}`,
      name: script.toolName,
      argumentsJson: script.argumentsJson,
    },
  ],
  promptTokens: 10,
  completionTokens: 5,
});

const createScriptedRouter = (
  toolScripts: ToolCallScript[],
  textScripts: string[],
): {
  router: LlmRouter;
  toolCallsTaken: ChatWithToolsOptions[];
  textCallsTaken: ChatOptions[];
} => {
  const remaining = [...toolScripts];
  const remainingText = [...textScripts];
  const toolCallsTaken: ChatWithToolsOptions[] = [];
  const textCallsTaken: ChatOptions[] = [];
  const router: LlmRouter = {
    masterAvailable: true,
    localAvailable: async () => true,
    chatText: async (options) => {
      textCallsTaken.push(options);
      const next = remainingText.shift();
      if (next === undefined) {
        // Default-accept (fail-open) when self-check has no scripted reply
        return JSON.stringify({ complete: true, missing: [] });
      }
      return next;
    },
    chatTextWithTools: async (options) => {
      toolCallsTaken.push(options);
      const next = remaining.shift();
      if (next === undefined) {
        throw new Error("scripted router: out of tool scripts");
      }
      return buildToolCallResponse(next);
    },
  };
  return { router, toolCallsTaken, textCallsTaken };
};

const createBashTerminalClient = (): AcpTerminalClient & { commandsSeen: string[] } => {
  const commandsSeen: string[] = [];
  let counter = 0;
  const exitsByTerminal = new Map<string, { exitCode: number | null; signal: string | null }>();
  const outputsByTerminal = new Map<string, string>();
  const client: AcpTerminalClient = {
    createTerminal: async (params) => {
      counter += 1;
      const terminalId = `term-${counter}`;
      const wrapped = params.args[1] ?? "";
      const userCommand = wrapped.replace(/^set -o pipefail;\s*/, "");
      commandsSeen.push(userCommand);
      const isSubmit = userCommand.trim() === "echo BAG_TASK_COMPLETE";
      const output = isSubmit ? "BAG_TASK_COMPLETE\n" : "ok\n";
      outputsByTerminal.set(terminalId, output);
      exitsByTerminal.set(terminalId, { exitCode: 0, signal: null });
      return { terminalId };
    },
    waitForTerminalExit: async ({ terminalId }) =>
      exitsByTerminal.get(terminalId) ?? { exitCode: 0, signal: null },
    terminalOutput: async ({ terminalId }) => ({
      output: outputsByTerminal.get(terminalId) ?? "",
      truncated: false,
      exitStatus: exitsByTerminal.get(terminalId) ?? null,
    }),
    releaseTerminal: async () => ({}),
  };
  return Object.assign(client, { commandsSeen });
};

const okBackend = (
  hits: CodebaseSearchHit[],
): CodebaseSearchBackend & { searchCalls: Array<{ query: string; topK?: number }> } => {
  const searchCalls: Array<{ query: string; topK?: number }> = [];
  const backend: CodebaseSearchBackend = {
    isAvailable: async () => true,
    ensureIndex: async () => ({ status: "fresh", durationMs: 0 }),
    search: async (input) => {
      const call: { query: string; topK?: number } = { query: input.query };
      if (input.topK !== undefined) call.topK = input.topK;
      searchCalls.push(call);
      return hits;
    },
    invalidate: async () => undefined,
  };
  return Object.assign(backend, { searchCalls });
};

const unavailableBackend = (): CodebaseSearchBackend => ({
  isAvailable: async () => false,
  ensureIndex: async () => ({ status: "skipped" }),
  search: async () => {
    throw new Error("must not be called when unavailable");
  },
  invalidate: async () => undefined,
});

describe("autonomous-coding-turn code_search dispatch", () => {
  // Each test runs in a clean env state for BAG_CODE_SEARCH
  const savedEnv = process.env.BAG_CODE_SEARCH;
  beforeEach(() => {
    delete process.env.BAG_CODE_SEARCH;
  });
  afterEach(() => {
    if (savedEnv === undefined) delete process.env.BAG_CODE_SEARCH;
    else process.env.BAG_CODE_SEARCH = savedEnv;
  });

  test("registers code_search in the tools surface by default", async () => {
    const { router, toolCallsTaken } = createScriptedRouter(
      [
        // First turn: do a code_search
        {
          toolName: "code_search",
          argumentsJson: JSON.stringify({ query: "where is the auth middleware" }),
        },
        // Second turn: submit
        { toolName: "bash", argumentsJson: JSON.stringify({ command: "echo BAG_TASK_COMPLETE" }) },
      ],
      [],
    );
    const backend = okBackend([
      {
        file: "src/auth.ts",
        lineRange: [12, 28],
        symbol: "authMiddleware",
        score: 0.93,
        snippet: "export const authMiddleware = ...",
      },
    ]);
    const result = await runAutonomousCodingTurn({
      router,
      client: createBashTerminalClient(),
      sessionId: "test-session",
      cwd: "/app",
      task: "Localize the auth middleware then submit.",
      config: {
        maxTurns: 4,
        codeSearchBackend: backend,
      },
    });
    expect(result.stopReason).toBe("submitted");
    // First chatTextWithTools call must have all three tool definitions
    const firstToolList = toolCallsTaken[0]?.tools.map((t) => t.function.name) ?? [];
    expect(firstToolList).toContain("bash");
    expect(firstToolList).toContain("view_image");
    expect(firstToolList).toContain("code_search");
    // Backend was actually invoked
    expect(backend.searchCalls).toHaveLength(1);
    expect(backend.searchCalls[0]?.query).toBe("where is the auth middleware");
    // The trace contains a code_search entry
    const codeSearchEntries = result.trace.filter((e) => e.kind === "code_search");
    expect(codeSearchEntries).toHaveLength(1);
    expect(codeSearchEntries[0]).toMatchObject({
      hitCount: 1,
      backendStatus: "available",
    });
    // The model's NEXT turn saw the rendered hits as a tool message
    const secondCallMessages = toolCallsTaken[1]?.messages ?? [];
    const toolReply = secondCallMessages.find(
      (m) => "role" in m && m.role === "tool",
    ) as { content: string } | undefined;
    expect(toolReply?.content).toContain("authMiddleware");
    expect(toolReply?.content).toContain("src/auth.ts:12-28");
  });

  test("returns a structured error (not crash) when backend unavailable", async () => {
    const { router } = createScriptedRouter(
      [
        {
          toolName: "code_search",
          argumentsJson: JSON.stringify({ query: "rate limit" }),
        },
        { toolName: "bash", argumentsJson: JSON.stringify({ command: "echo BAG_TASK_COMPLETE" }) },
      ],
      [],
    );
    const result = await runAutonomousCodingTurn({
      router,
      client: createBashTerminalClient(),
      sessionId: "test-session",
      cwd: "/app",
      task: "Investigate rate limiting; submit.",
      config: { maxTurns: 4, codeSearchBackend: unavailableBackend() },
    });
    expect(result.stopReason).toBe("submitted");
    // Trace shows backendStatus=unavailable
    const csEntries = result.trace.filter((e) => e.kind === "code_search") as Array<
      Extract<typeof result.trace[number], { kind: "code_search" }>
    >;
    expect(csEntries).toHaveLength(1);
    expect(csEntries[0]?.backendStatus).toBe("unavailable");
    // No format_error or abort was triggered
    expect(result.trace.find((e) => e.kind === "abort")).toBeUndefined();
  });

  test("removes code_search from tool surface when BAG_CODE_SEARCH=0", async () => {
    process.env.BAG_CODE_SEARCH = "0";
    const { router, toolCallsTaken } = createScriptedRouter(
      [{ toolName: "bash", argumentsJson: JSON.stringify({ command: "echo BAG_TASK_COMPLETE" }) }],
      [],
    );
    await runAutonomousCodingTurn({
      router,
      client: createBashTerminalClient(),
      sessionId: "test-session",
      cwd: "/app",
      task: "Trivial submit.",
      config: {
        maxTurns: 2,
        codeSearchBackend: okBackend([]),
      },
    });
    const tools = toolCallsTaken[0]?.tools.map((t) => t.function.name) ?? [];
    expect(tools).toContain("bash");
    expect(tools).toContain("view_image");
    expect(tools).not.toContain("code_search");
  });

  test("rejects malformed code_search args with a structured tool-result error", async () => {
    const { router, toolCallsTaken } = createScriptedRouter(
      [
        // Missing required `query` field
        { toolName: "code_search", argumentsJson: JSON.stringify({}) },
        { toolName: "bash", argumentsJson: JSON.stringify({ command: "echo BAG_TASK_COMPLETE" }) },
      ],
      [],
    );
    const result = await runAutonomousCodingTurn({
      router,
      client: createBashTerminalClient(),
      sessionId: "test-session",
      cwd: "/app",
      task: "Trivial.",
      config: { maxTurns: 4, codeSearchBackend: okBackend([]) },
    });
    expect(result.stopReason).toBe("submitted");
    // The model's second turn saw a structured error reply for the bad call
    const secondMessages = toolCallsTaken[1]?.messages ?? [];
    const toolReply = secondMessages.find((m) => "role" in m && m.role === "tool") as
      | { content: string }
      | undefined;
    expect(toolReply?.content).toContain("code_search error");
    expect(toolReply?.content).toContain("non-empty 'query'");
  });
});

// Touch types so isolated-modules treats imports correctly
export type _AutonomousToolResultUsed = AutonomousToolResult;
