import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentSideConnection as AcpConnection } from "@agentclientprotocol/sdk";
import { handleAcpSlashCommand, type AcpSlashRouterDeps } from "../src/acp/slash-router";
import type { BagAcpSession } from "../src/acp/session";
import { defaultConfig } from "../src/config";

const sessionFor = (cwd: string): BagAcpSession => ({
  id: "bag-slash-test",
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
  optimizerPin: {} as never,
  clientCapabilities: {
    fsReadTextFile: true,
    fsWriteTextFile: true,
    terminal: true,
    richDiffContent: true,
    richTerminalContent: true,
    source: "test",
  },
});

const createDeps = (updates: unknown[], messages: string[], calls: string[]): AcpSlashRouterDeps => ({
  connection: {
    sessionUpdate: async (update: unknown) => {
      updates.push(update);
    },
  } as AcpConnection,
  config: defaultConfig(),
  agentMessage: async (_sessionId, text) => {
    messages.push(text);
  },
  listSkills: () => [{ name: "plan-graph", description: "DAG planning", path: "/skills/plan-graph" }],
  runWithTemporaryMode: async (session, activeMode, previousMode, fn) => {
    calls.push(`temporary:${activeMode}:${previousMode}:start-${session.mode}`);
    const result = await fn();
    if (previousMode === "auto") {
      session.mode = "auto";
      updates.push({
        sessionId: session.id,
        update: { sessionUpdate: "current_mode_update", currentModeId: session.mode },
      });
    }
    return result;
  },
  runCodingTurn: async (_session, task) => {
    calls.push(`code:${task}`);
  },
  runPlanningTurn: async (_session, task) => {
    calls.push(`plan:${task}`);
  },
  runAutonomousToolUseTurn: async (_session, task) => {
    calls.push(`tools:${task}`);
  },
  runDagDrivenToolUseTurn: async (_session, task) => {
    calls.push(`dag:${task}`);
  },
  runAdaptiveCodingTurn: async (_session, task) => {
    calls.push(`adaptive:${task}`);
  },
  runMaintenanceCommand: async (_session, task) => {
    calls.push(`maintenance:${task}`);
  },
});

describe("ACP slash router", () => {
  test("ignores normal text and handles safe/yolo without project side effects", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-slash-basic-"));
    const session = sessionFor(cwd);
    const updates: unknown[] = [];
    const messages: string[] = [];
    const calls: string[] = [];
    const deps = createDeps(updates, messages, calls);

    expect(await handleAcpSlashCommand(deps, { session, text: "Ahoj", signal: new AbortController().signal })).toBe(false);
    expect(await handleAcpSlashCommand(deps, { session, text: "/safe", signal: new AbortController().signal })).toBe(true);
    expect(session.yolo).toBe(false);
    expect(await handleAcpSlashCommand(deps, { session, text: "/yolo", signal: new AbortController().signal })).toBe(true);
    expect(session.yolo).toBe(true);

    expect(calls).toEqual([]);
    expect(JSON.stringify(updates)).not.toContain("tool_call");
    expect(messages.join("\n")).toContain("Safe mode enabled");
    expect(messages.join("\n")).toContain("YOLO mode enabled");
  });

  test("routes run and plan tasks through temporary mode restoration", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-slash-routes-"));
    const session = sessionFor(cwd);
    const updates: unknown[] = [];
    const messages: string[] = [];
    const calls: string[] = [];
    const deps = createDeps(updates, messages, calls);

    await handleAcpSlashCommand(deps, { session, text: "/run fix bug", signal: new AbortController().signal });
    await handleAcpSlashCommand(deps, { session, text: "/plan write prd", signal: new AbortController().signal });
    await handleAcpSlashCommand(deps, { session, text: "/maintenance status", signal: new AbortController().signal });

    expect(calls).toEqual([
      "temporary:run:auto:start-run",
      "code:fix bug",
      "temporary:plan:auto:start-plan",
      "plan:write prd",
      "temporary:plan:auto:start-auto",
      "maintenance:status",
    ]);
    expect(session.mode).toBe("auto");
    expect(JSON.stringify(updates)).toContain("\"currentModeId\":\"run\"");
    expect(JSON.stringify(updates)).toContain("\"currentModeId\":\"plan\"");
    expect(JSON.stringify(updates)).toContain("\"currentModeId\":\"auto\"");
  });

  test("preserves newlines and code fences in the task body", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-slash-newlines-"));
    const session = sessionFor(cwd);
    const updates: unknown[] = [];
    const messages: string[] = [];
    const calls: string[] = [];
    const deps = createDeps(updates, messages, calls);

    const taskBody = [
      "Fix the package so the snippet runs:",
      "```python",
      "import foo",
      "from bar import baz",
      "```",
      "and tests pass.",
    ].join("\n");

    await handleAcpSlashCommand(deps, {
      session,
      text: `/run-auto ${taskBody}`,
      signal: new AbortController().signal,
    });

    const adaptive = calls.find((c) => c.startsWith("adaptive:"));
    expect(adaptive).toBeDefined();
    const passedTask = adaptive!.slice("adaptive:".length);
    // Newlines must survive the slash router so downstream code-fence parsing works.
    expect(passedTask).toContain("```python\nimport foo\n");
    expect(passedTask.split("\n").length).toBeGreaterThan(3);
  });

  test("keeps hidden operator and introspection commands inside the router boundary", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-slash-info-"));
    const session = sessionFor(cwd);
    const updates: unknown[] = [];
    const messages: string[] = [];
    const calls: string[] = [];
    const deps = createDeps(updates, messages, calls);

    await handleAcpSlashCommand(deps, { session, text: "/skills", signal: new AbortController().signal });
    await handleAcpSlashCommand(deps, { session, text: "/metrics", signal: new AbortController().signal });
    await handleAcpSlashCommand(deps, { session, text: "/unknown", signal: new AbortController().signal });

    const text = messages.join("\n");
    expect(text).toContain("plan-graph");
    expect(text).toContain("Telemetry JSONL");
    expect(text).toContain("Unknown command: /unknown");
    expect(calls).toEqual([]);
  });
});
