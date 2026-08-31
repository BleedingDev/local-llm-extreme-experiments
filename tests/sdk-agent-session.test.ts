/**
 * Tests for `src/sdk/agent-session.ts`.
 *
 * Goals:
 *   - `createBagSession(...).run(...)` must yield generic `AgentEvent`s in
 *     order, ending with `session_ended` carrying the autonomous-turn stop
 *     reason.
 *   - The router/terminal-client mocks drive the loop deterministically so
 *     the test asserts both event kinds and event payloads (tool name,
 *     submitted output, etc.) without depending on a real LLM.
 *   - `cancel()` aborts the active run; the resulting stream surfaces an
 *     `abort` event followed by `session_ended` with stop_reason="cancelled".
 *   - `steer(message)` queues an interjection that becomes a `steer` event
 *     and is injected into the next chatTextWithTools call as a user
 *     message tagged with `[BAG steer]`.
 */

import { describe, expect, test } from "bun:test";

import {
  createBagSession,
  type AgentEvent,
} from "../src/sdk/agent-session";
import type { AcpTerminalClient } from "../src/autonomous-tools";
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

type ScriptedRouter = {
  router: LlmRouter;
  toolCallsTaken: ChatWithToolsOptions[];
  textCallsTaken: ChatOptions[];
};

const createScriptedRouter = (script: {
  toolScripts: ToolCallScript[];
  textScripts?: string[];
}): ScriptedRouter => {
  const toolScripts = [...script.toolScripts];
  const textScripts = [...(script.textScripts ?? [])];
  const toolCallsTaken: ChatWithToolsOptions[] = [];
  const textCallsTaken: ChatOptions[] = [];
  const router: LlmRouter = {
    masterAvailable: true,
    localAvailable: async () => true,
    chatText: async (options) => {
      textCallsTaken.push(options);
      const next = textScripts.shift();
      // Fallback for self-check: return "complete:true" so the gate accepts.
      return next ?? JSON.stringify({ complete: true, missing: [] });
    },
    chatTextWithTools: async (options) => {
      toolCallsTaken.push(options);
      const next = toolScripts.shift();
      if (next === undefined) {
        // Drain by returning end_turn so the loop exits cleanly.
        return {
          finishReason: "stop",
          textContent: "complete",
          toolCalls: [],
          promptTokens: 1,
          completionTokens: 1,
        };
      }
      return buildToolCallResponse(next);
    },
  };
  return { router, toolCallsTaken, textCallsTaken };
};

const createSubmitSentinelTerminalClient = (
  options: { sentinel?: string; nonSubmitOutput?: string } = {},
): AcpTerminalClient & { commandsSeen: string[] } => {
  const sentinel = options.sentinel ?? "BAG_TASK_COMPLETE";
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
      const isSubmit = userCommand.trim() === `echo ${sentinel}`;
      const output = isSubmit ? `${sentinel}\n` : options.nonSubmitOutput ?? "ok\n";
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

const collect = async (
  iterable: AsyncIterable<AgentEvent>,
): Promise<AgentEvent[]> => {
  const events: AgentEvent[] = [];
  for await (const ev of iterable) events.push(ev);
  return events;
};

describe("createBagSession — happy path", () => {
  test("yields session_started → user_message → assistant/tool events → session_ended(submitted)", async () => {
    const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
    const scripted = createScriptedRouter({
      toolScripts: [{ toolName: "bash", argumentsJson: submitArgs }],
    });
    const client = createSubmitSentinelTerminalClient();
    const session = createBagSession({
      router: scripted.router,
      cwd: "/app",
      sessionId: "test-sdk-1",
      client,
      config: { maxTurns: 4 },
    });
    const events = await collect(session.run("trivial submit task"));

    // First event must be session_started with the task echoed back.
    expect(events[0]).toMatchObject({
      kind: "session_started",
      task: "trivial submit task",
      cwd: "/app",
    });
    // The trace pushes a user_message before any assistant_message.
    const kindsInOrder = events.map((e) => e.kind);
    expect(kindsInOrder).toContain("user_message");
    expect(kindsInOrder).toContain("assistant_message");
    expect(kindsInOrder).toContain("tool_call");
    expect(kindsInOrder).toContain("tool_result");

    // Tool call carries the model's bash command verbatim.
    const toolCall = events.find(
      (e): e is Extract<AgentEvent, { kind: "tool_call" }> => e.kind === "tool_call",
    );
    expect(toolCall).toBeDefined();
    expect(toolCall?.tool).toBe("bash");
    expect(toolCall?.arguments_json).toBe(submitArgs);

    // Tool result reflects the submitted sentinel output.
    const toolResult = events.find(
      (e): e is Extract<AgentEvent, { kind: "tool_result" }> => e.kind === "tool_result",
    );
    expect(toolResult).toBeDefined();
    expect(toolResult?.submitted).toBe(true);
    expect(toolResult?.output).toBe("BAG_TASK_COMPLETE\n");

    // Final event must be session_ended with submitted stop reason.
    const final = events[events.length - 1];
    expect(final?.kind).toBe("session_ended");
    expect(final).toMatchObject({
      kind: "session_ended",
      stop_reason: "submitted",
      submitted_output: "BAG_TASK_COMPLETE\n",
    });
  });

  test("steer(message) injects [BAG steer] user message into the next chatTextWithTools call", async () => {
    const submit = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
    const scripted = createScriptedRouter({
      toolScripts: [{ toolName: "bash", argumentsJson: submit }],
    });
    const client = createSubmitSentinelTerminalClient();
    const session = createBagSession({
      router: scripted.router,
      cwd: "/app",
      client,
      config: { maxTurns: 4 },
    });

    // Pre-queue the interjection BEFORE `run()` starts iterating. The
    // wrapping router drains pending steer messages on every
    // `chatTextWithTools` call, so this lands on the first assistant turn.
    // Mid-flight steer (queued from inside the for-await loop) lands on
    // the next turn boundary; that path is harder to test deterministically
    // without a custom yield-point hook, so we cover the boundary case here.
    session.steer("focus on /app/main.py first");

    const events: AgentEvent[] = [];
    for await (const ev of session.run("explore then submit")) events.push(ev);

    // The first chatTextWithTools call must include a user message tagged
    // with [BAG steer].
    expect(scripted.toolCallsTaken.length).toBeGreaterThanOrEqual(1);
    const firstCall = scripted.toolCallsTaken[0];
    const matchingUserMessage = firstCall?.messages.find(
      (m) =>
        m.role === "user" &&
        typeof m.content === "string" &&
        m.content.includes("[BAG steer]") &&
        m.content.includes("focus on /app/main.py first"),
    );
    expect(matchingUserMessage).toBeDefined();
  });
});

describe("createBagSession — cancellation", () => {
  test("cancel() aborts an in-flight run and surfaces stop_reason=cancelled", async () => {
    // The router stalls indefinitely on chatTextWithTools so the loop is
    // hung waiting for the first assistant turn. cancel() must unblock it
    // (the loop checks input.signal at the start of each turn iteration).
    let resolveBlocker: () => void = () => {};
    const blocker = new Promise<void>((resolve) => {
      resolveBlocker = resolve;
    });
    const router: LlmRouter = {
      masterAvailable: true,
      localAvailable: async () => true,
      chatText: async () => JSON.stringify({ complete: true, missing: [] }),
      chatTextWithTools: async (): Promise<ChatWithToolsResult> => {
        // Return after the test releases the blocker. Once released, the loop
        // moves on, sees signal.aborted, and pushes an abort entry.
        await blocker;
        return {
          finishReason: "stop",
          textContent: "",
          toolCalls: [
            {
              id: "tool-late",
              name: "bash",
              argumentsJson: JSON.stringify({ command: "ls" }),
            },
          ],
          promptTokens: 1,
          completionTokens: 1,
        };
      },
    };
    const client = createSubmitSentinelTerminalClient();
    const session = createBagSession({
      router,
      cwd: "/app",
      client,
      config: { maxTurns: 4 },
    });
    const events: AgentEvent[] = [];
    const collector = (async () => {
      for await (const ev of session.run("never-completing task")) events.push(ev);
    })();
    // Cancel before the router resolves, then release the blocker so the
    // loop can observe the abort signal on the next iteration.
    session.cancel();
    resolveBlocker();
    await collector;

    const final = events[events.length - 1];
    expect(final?.kind).toBe("session_ended");
    expect(final).toMatchObject({ kind: "session_ended", stop_reason: "cancelled" });
    // The abort trace entry maps to an abort event.
    expect(events.some((e) => e.kind === "abort")).toBe(true);
  });
});

describe("createBagSession — concurrency guard", () => {
  test("calling run() while another run is active throws", async () => {
    const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
    const scripted = createScriptedRouter({
      toolScripts: [{ toolName: "bash", argumentsJson: submitArgs }],
    });
    const client = createSubmitSentinelTerminalClient();
    const session = createBagSession({
      router: scripted.router,
      cwd: "/app",
      client,
      config: { maxTurns: 4 },
    });
    const stream = session.run("first run");
    // Without consuming `stream`, a second run() must throw immediately —
    // the activeAbort latch was set the moment the first run() returned.
    expect(() => session.run("second run")).toThrow(
      /previous run is still in flight/,
    );
    // Drain so the bun test runner doesn't leak the underlying queue.
    for await (const _ev of stream) {
      void _ev;
    }
  });
});
