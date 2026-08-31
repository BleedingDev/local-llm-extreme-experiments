import { describe, expect, test } from "bun:test";
import { runAutonomousCodingTurn } from "../src/autonomous-coding-turn";
import type {
  AcpTerminalClient,
  AutonomousToolResult,
} from "../src/autonomous-tools";
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

type RouterScript = {
  toolScripts: ToolCallScript[];
  textScripts: string[];
};

type ScriptedRouter = {
  router: LlmRouter;
  toolCallsTaken: ChatWithToolsOptions[];
  textCallsTaken: ChatOptions[];
  remainingToolScripts: () => number;
  remainingTextScripts: () => number;
};

const createScriptedRouter = (script: RouterScript): ScriptedRouter => {
  const toolScripts = [...script.toolScripts];
  const textScripts = [...script.textScripts];
  const toolCallsTaken: ChatWithToolsOptions[] = [];
  const textCallsTaken: ChatOptions[] = [];
  const router: LlmRouter = {
    masterAvailable: true,
    localAvailable: async () => true,
    chatText: async (options) => {
      textCallsTaken.push(options);
      const next = textScripts.shift();
      if (next === undefined) {
        throw new Error("scripted router: no more chatText responses configured");
      }
      return next;
    },
    chatTextWithTools: async (options) => {
      toolCallsTaken.push(options);
      const next = toolScripts.shift();
      if (next === undefined) {
        throw new Error("scripted router: no more chatTextWithTools responses configured");
      }
      return buildToolCallResponse(next);
    },
  };
  return {
    router,
    toolCallsTaken,
    textCallsTaken,
    remainingToolScripts: () => toolScripts.length,
    remainingTextScripts: () => textScripts.length,
  };
};

const createSubmitSentinelTerminalClient = (
  options: { sentinel: string; nonSubmitOutput?: string } = { sentinel: "BAG_TASK_COMPLETE" },
): AcpTerminalClient & { commandsSeen: string[] } => {
  const commandsSeen: string[] = [];
  let counter = 0;
  const exitsByTerminal = new Map<string, { exitCode: number | null; signal: string | null }>();
  const outputsByTerminal = new Map<string, string>();
  const client: AcpTerminalClient = {
    createTerminal: async (params) => {
      counter += 1;
      const terminalId = `term-${counter}`;
      // The bash tool wraps user commands as `set -o pipefail; <command>` and
      // passes them as a single argv element after `-lc`, so we extract the
      // user-visible command from the second positional arg.
      const wrapped = params.args[1] ?? "";
      const userCommand = wrapped.replace(/^set -o pipefail;\s*/, "");
      commandsSeen.push(userCommand);
      const isSubmit = userCommand.trim() === `echo ${options.sentinel}`;
      const output = isSubmit
        ? `${options.sentinel}\n`
        : options.nonSubmitOutput ?? "ok\n";
      outputsByTerminal.set(terminalId, output);
      exitsByTerminal.set(terminalId, { exitCode: 0, signal: null });
      return { terminalId };
    },
    waitForTerminalExit: async ({ terminalId }) => {
      return exitsByTerminal.get(terminalId) ?? { exitCode: 0, signal: null };
    },
    terminalOutput: async ({ terminalId }) => {
      return {
        output: outputsByTerminal.get(terminalId) ?? "",
        truncated: false,
        exitStatus: exitsByTerminal.get(terminalId) ?? null,
      };
    },
    releaseTerminal: async () => ({}),
  };
  return Object.assign(client, { commandsSeen });
};

describe("pre-submit self-check gate", () => {
  test(
    "injects synthetic feedback and grants a second attempt when auditor flags missing requirements",
    async () => {
      const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
      const fixArgs = JSON.stringify({ command: "touch /app/report.jsonl" });
      const scripted = createScriptedRouter({
        toolScripts: [
          // First "submission" — just echoes the sentinel before doing the work.
          { toolName: "bash", argumentsJson: submitArgs, textContent: "" },
          // Self-check rejects → agent fixes the missing deliverable.
          { toolName: "bash", argumentsJson: fixArgs, textContent: "" },
          // Re-submits.
          { toolName: "bash", argumentsJson: submitArgs, textContent: "" },
        ],
        textScripts: [
          // First self-check call → reject.
          JSON.stringify({
            complete: false,
            missing: ["create /app/report.jsonl"],
          }),
          // Second self-check call → accept.
          JSON.stringify({ complete: true, missing: [] }),
        ],
      });
      const terminalClient = createSubmitSentinelTerminalClient({
        sentinel: "BAG_TASK_COMPLETE",
      });
      const instruction = [
        "Fix the bottle.py CRLF injection (steps 1-3).",
        "Then create /app/report.jsonl with vulnerability metadata (step 4).",
        "Run pytest -k hkey to verify.",
        "Echo BAG_TASK_COMPLETE when done.",
      ].join("\n");

      const result = await runAutonomousCodingTurn({
        router: scripted.router,
        client: terminalClient,
        sessionId: "test-session",
        cwd: "/app",
        task: instruction,
        config: { maxTurns: 8 },
      });

      // The turn should ultimately succeed via the second submission.
      expect(result.stopReason).toBe("submitted");
      // Two self-check calls were made (one reject, one accept).
      expect(scripted.textCallsTaken).toHaveLength(2);
      expect(scripted.remainingTextScripts()).toBe(0);
      // The model was asked for tool-calls three times (submit, fix, resubmit).
      expect(scripted.toolCallsTaken.length).toBeGreaterThanOrEqual(3);
      // Both submissions echoed the sentinel; the fix was the touch command.
      const userCommands = terminalClient.commandsSeen.map((c) => c.trim());
      expect(userCommands).toContain("echo BAG_TASK_COMPLETE");
      expect(userCommands).toContain("touch /app/report.jsonl");
      expect(
        userCommands.filter((c) => c === "echo BAG_TASK_COMPLETE").length,
      ).toBe(2);

      // The trace must contain the new `pre_submit_self_check` entries.
      const selfCheckEntries = result.trace.filter(
        (entry): entry is Extract<typeof entry, { kind: "pre_submit_self_check" }> =>
          entry.kind === "pre_submit_self_check",
      );
      expect(selfCheckEntries).toHaveLength(2);
      expect(selfCheckEntries[0]).toMatchObject({
        complete: false,
        missing: ["create /app/report.jsonl"],
      });
      expect(selfCheckEntries[1]).toMatchObject({ complete: true, missing: [] });

      // The synthetic feedback must have been injected into the second
      // tool-call request's messages (the call that produced the fix command).
      const secondToolCall = scripted.toolCallsTaken[1];
      expect(secondToolCall).toBeDefined();
      const messagesJson = JSON.stringify(secondToolCall?.messages ?? []);
      expect(messagesJson).toContain("[BAG pre-submit self-check]");
      expect(messagesJson).toContain("create /app/report.jsonl");
    },
  );

  test(
    "passes through when the auditor errors (fail-open)",
    async () => {
      const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
      const router: LlmRouter = {
        masterAvailable: true,
        localAvailable: async () => true,
        chatText: async () => {
          throw new Error("haiku unavailable");
        },
        chatTextWithTools: async (): Promise<ChatWithToolsResult> => ({
          finishReason: "tool_calls",
          textContent: "",
          toolCalls: [
            {
              id: "tool-only",
              name: "bash",
              argumentsJson: submitArgs,
            },
          ],
        }),
      };
      const terminalClient = createSubmitSentinelTerminalClient();
      const result = await runAutonomousCodingTurn({
        router,
        client: terminalClient,
        sessionId: "test-session",
        cwd: "/app",
        task: "Trivial task. Submit immediately.",
        config: { maxTurns: 4 },
      });
      // Auditor throw must not block the submission.
      expect(result.stopReason).toBe("submitted");
      // The fail-open path returns `complete: true, missing: []`, so a trace
      // entry IS recorded — but it must report no unmet requirements so the
      // outer loop accepted the submission.
      const selfCheckEntries = result.trace.filter(
        (entry): entry is Extract<typeof entry, { kind: "pre_submit_self_check" }> =>
          entry.kind === "pre_submit_self_check",
      );
      expect(selfCheckEntries).toHaveLength(1);
      expect(selfCheckEntries[0]).toMatchObject({ complete: true, missing: [] });
    },
  );

  test(
    "every successful submission emits exactly one approved self-check trace entry",
    async () => {
      // Path: agent submits, auditor approves on first call. The new
      // measurability contract requires that a `pre_submit_self_check`
      // entry is ALWAYS pushed — including when the auditor approves —
      // so audit tooling can distinguish "gate fired and approved"
      // from "gate never reached on this attempt".
      const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
      const scripted = createScriptedRouter({
        toolScripts: [
          { toolName: "bash", argumentsJson: submitArgs, textContent: "" },
        ],
        textScripts: [
          // Auditor approves immediately, no missing items.
          JSON.stringify({ complete: true, missing: [] }),
        ],
      });
      const terminalClient = createSubmitSentinelTerminalClient({
        sentinel: "BAG_TASK_COMPLETE",
      });
      const result = await runAutonomousCodingTurn({
        router: scripted.router,
        client: terminalClient,
        sessionId: "test-session",
        cwd: "/app",
        task: "Write 'hello' to /tmp/x.txt. Echo BAG_TASK_COMPLETE when done.",
        config: { maxTurns: 4 },
      });

      expect(result.stopReason).toBe("submitted");
      // Exactly ONE auditor LLM call must have been made for the single
      // successful submission — and exactly ONE trace entry must have
      // been emitted.
      expect(scripted.textCallsTaken).toHaveLength(1);
      const selfCheckEntries = result.trace.filter(
        (entry): entry is Extract<typeof entry, { kind: "pre_submit_self_check" }> =>
          entry.kind === "pre_submit_self_check",
      );
      expect(selfCheckEntries).toHaveLength(1);
      // The approved entry MUST have `gate_reached: true`, `complete:
      // true`, `missing: []`, and NO `error` field.
      const entry = selfCheckEntries[0];
      expect(entry).toMatchObject({
        kind: "pre_submit_self_check",
        gate_reached: true,
        complete: true,
        missing: [],
      });
      // `error` is the sentinel that distinguishes "gate ran to verdict"
      // from "gate ran but failed open"; on a clean approval it MUST be
      // absent.
      expect(
        (entry as { error?: string }).error,
      ).toBeUndefined();
    },
  );

  test(
    "auditor LLM throw still emits a trace entry with the error field set (fail-open with attribution)",
    async () => {
      // Same shape as the original fail-open test, but asserts the new
      // attribution contract: when the auditor's chatText throws, the
      // gate STILL emits a trace entry, AND that entry's `error` field
      // captures the throw message. This is what the lift analyzer uses
      // to subtract gate-runs that produced no real verdict from the
      // "gate fired" denominator.
      const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
      const router: LlmRouter = {
        masterAvailable: true,
        localAvailable: async () => true,
        chatText: async () => {
          throw new Error("haiku unavailable: ECONNREFUSED 127.0.0.1:11434");
        },
        chatTextWithTools: async (): Promise<ChatWithToolsResult> => ({
          finishReason: "tool_calls",
          textContent: "",
          toolCalls: [
            {
              id: "tool-only",
              name: "bash",
              argumentsJson: submitArgs,
            },
          ],
        }),
      };
      const terminalClient = createSubmitSentinelTerminalClient();
      const result = await runAutonomousCodingTurn({
        router,
        client: terminalClient,
        sessionId: "test-session",
        cwd: "/app",
        task: "Trivial task. Submit immediately.",
        config: { maxTurns: 4 },
      });
      expect(result.stopReason).toBe("submitted");
      const selfCheckEntries = result.trace.filter(
        (entry): entry is Extract<typeof entry, { kind: "pre_submit_self_check" }> =>
          entry.kind === "pre_submit_self_check",
      );
      expect(selfCheckEntries).toHaveLength(1);
      const entry = selfCheckEntries[0];
      expect(entry).toMatchObject({
        gate_reached: true,
        complete: true,
        missing: [],
      });
      // The `error` field MUST surface the auditor's throw.
      const errorField = (entry as { error?: string }).error;
      expect(errorField).toBeDefined();
      expect(errorField).toContain("haiku unavailable");
    },
  );
});

describe("pre-submit self-check — scratch hygiene failure modes 6 & 7", () => {
  test(
    "surfaces a /tmp/ scratch leak as a missing item (failure mode 6)",
    async () => {
      // Sequence: agent writes /tmp/build_log.txt, never cleans it up,
      // then submits. The auditor (scripted to flag mode 6) returns
      // `complete: false, missing: ['/tmp/build_log.txt …']`.
      const writeArgs = JSON.stringify({
        command: "pytest 2>&1 > /tmp/build_log.txt; cat /tmp/build_log.txt",
      });
      const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
      const cleanupArgs = JSON.stringify({ command: "rm -f /tmp/build_log.txt" });
      const scripted = createScriptedRouter({
        toolScripts: [
          { toolName: "bash", argumentsJson: writeArgs, textContent: "" },
          { toolName: "bash", argumentsJson: submitArgs, textContent: "" },
          { toolName: "bash", argumentsJson: cleanupArgs, textContent: "" },
          { toolName: "bash", argumentsJson: submitArgs, textContent: "" },
        ],
        textScripts: [
          // First self-check rejects with a mode-6 missing entry citing the leak path.
          JSON.stringify({
            complete: false,
            missing: [
              "SCRATCH-DIR POLLUTION: /tmp/build_log.txt was written and never cleaned up before submit",
            ],
          }),
          // Second self-check approves.
          JSON.stringify({ complete: true, missing: [] }),
        ],
      });
      const terminalClient = createSubmitSentinelTerminalClient();
      const result = await runAutonomousCodingTurn({
        router: scripted.router,
        client: terminalClient,
        sessionId: "test-session",
        cwd: "/app",
        task: "Run pytest and produce /app/result.json. Echo BAG_TASK_COMPLETE when done.",
        config: { maxTurns: 8 },
      });

      expect(result.stopReason).toBe("submitted");
      const entries = result.trace.filter(
        (e): e is Extract<typeof e, { kind: "pre_submit_self_check" }> =>
          e.kind === "pre_submit_self_check",
      );
      expect(entries.length).toBeGreaterThanOrEqual(2);
      expect(entries[0]).toMatchObject({ complete: false });
      expect(entries[0].missing.join("\n")).toContain("/tmp/build_log.txt");

      // The structured signal should have been part of the auditor prompt.
      expect(scripted.textCallsTaken.length).toBeGreaterThan(0);
      const firstAuditorMessages = JSON.stringify(
        scripted.textCallsTaken[0].messages ?? [],
      );
      expect(firstAuditorMessages).toContain("[Pre-submit hygiene scan]");
      expect(firstAuditorMessages).toContain("/tmp/build_log.txt");

      // Synthetic feedback was injected into the second tool-call prompt.
      const secondCallMessagesJson = JSON.stringify(
        scripted.toolCallsTaken[1]?.messages ?? [],
      );
      expect(secondCallMessagesJson).toContain("[BAG pre-submit self-check]");
      expect(secondCallMessagesJson).toContain("/tmp/build_log.txt");
    },
  );

  test(
    "surfaces an ignored Python Traceback as a missing item (failure mode 7)",
    async () => {
      // The agent runs pytest, gets a traceback with a non-zero exit, then
      // submits anyway. The auditor (mode 7) flags it.
      const failingArgs = JSON.stringify({
        command: "pytest -x tests/test_chelpers.py",
      });
      const submitArgs = JSON.stringify({ command: "echo BAG_TASK_COMPLETE" });
      const fixArgs = JSON.stringify({
        command: "sed -i 's/old/new/' src/chelpers.pyx && pytest -x tests/test_chelpers.py",
      });
      const scripted = createScriptedRouter({
        toolScripts: [
          // The failing pytest call returns a traceback.
          { toolName: "bash", argumentsJson: failingArgs, textContent: "" },
          // Agent submits despite the traceback.
          { toolName: "bash", argumentsJson: submitArgs, textContent: "" },
          // Auditor rejects → agent fixes and re-runs.
          { toolName: "bash", argumentsJson: fixArgs, textContent: "" },
          // Re-submit.
          { toolName: "bash", argumentsJson: submitArgs, textContent: "" },
        ],
        textScripts: [
          // First self-check rejects citing mode 7.
          JSON.stringify({
            complete: false,
            missing: [
              "IGNORED TRACEBACK: pytest -x tests/test_chelpers.py raised AttributeError module 'numpy' has no attribute 'array_api' and was never re-run successfully",
            ],
          }),
          // Second self-check approves.
          JSON.stringify({ complete: true, missing: [] }),
        ],
      });
      // Custom terminal client that returns a traceback for the failing
      // pytest command and "ok" for everything else.
      const tracebackOutput = [
        "============================ test session starts =============================",
        "Traceback (most recent call last):",
        "  File \"/app/src/chelpers.pyx\", line 12, in <module>",
        "    from numpy.array_api import asarray",
        "AttributeError: module 'numpy' has no attribute 'array_api'",
      ].join("\n");
      const commandsSeen: string[] = [];
      let counter = 0;
      const exits = new Map<string, { exitCode: number | null; signal: string | null }>();
      const outputs = new Map<string, string>();
      const terminalClient: AcpTerminalClient & { commandsSeen: string[] } = Object.assign(
        {
          createTerminal: async (params: { args: string[] }) => {
            counter += 1;
            const terminalId = `term-${counter}`;
            const wrapped = params.args[1] ?? "";
            const userCommand = wrapped.replace(/^set -o pipefail;\s*/, "");
            commandsSeen.push(userCommand);
            const isSubmit = userCommand.trim() === "echo BAG_TASK_COMPLETE";
            const isTbCmd = userCommand.includes("pytest -x tests/test_chelpers.py") &&
              !userCommand.includes("sed");
            outputs.set(
              terminalId,
              isSubmit ? "BAG_TASK_COMPLETE\n" : isTbCmd ? tracebackOutput : "ok\n",
            );
            exits.set(terminalId, {
              exitCode: isTbCmd ? 1 : 0,
              signal: null,
            });
            return { terminalId };
          },
          waitForTerminalExit: async ({ terminalId }: { terminalId: string }) =>
            exits.get(terminalId) ?? { exitCode: 0, signal: null },
          terminalOutput: async ({ terminalId }: { terminalId: string }) => ({
            output: outputs.get(terminalId) ?? "",
            truncated: false,
            exitStatus: exits.get(terminalId) ?? null,
          }),
          releaseTerminal: async () => ({}),
        } satisfies AcpTerminalClient,
        { commandsSeen },
      );

      const result = await runAutonomousCodingTurn({
        router: scripted.router,
        client: terminalClient,
        sessionId: "test-session",
        cwd: "/app",
        task: "Run the pyknotid test suite. Echo BAG_TASK_COMPLETE when all tests pass.",
        config: { maxTurns: 8 },
      });

      expect(result.stopReason).toBe("submitted");
      const entries = result.trace.filter(
        (e): e is Extract<typeof e, { kind: "pre_submit_self_check" }> =>
          e.kind === "pre_submit_self_check",
      );
      expect(entries[0]).toMatchObject({ complete: false });
      expect(entries[0].missing.join("\n")).toContain("TRACEBACK");
      expect(entries[0].missing.join("\n")).toContain("AttributeError");

      // The hygiene scan must have surfaced the traceback for the auditor.
      const firstAuditorMessages = JSON.stringify(
        scripted.textCallsTaken[0].messages ?? [],
      );
      expect(firstAuditorMessages).toContain("[Pre-submit hygiene scan]");
      expect(firstAuditorMessages).toContain("AttributeError");
    },
  );
});

// Touch the AutonomousToolResult symbol so the import is treated as type-only
// usage by the linter without affecting runtime.
export type _AutonomousToolResultUsed = AutonomousToolResult;
