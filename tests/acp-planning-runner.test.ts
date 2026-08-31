import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentSideConnection as AcpConnection } from "@agentclientprotocol/sdk";
import { runAcpPlanningTurn, type AcpPlanningRunnerDeps } from "../src/acp/planning-runner";
import type { BagAcpSession } from "../src/acp/session";
import type { AcpToolInput } from "../src/acp/tool-runner";
import { defaultConfig } from "../src/config";

const sessionFor = (cwd: string): BagAcpSession => ({
  id: "bag-planning-test",
  cwd,
  additionalDirectories: [],
  executorConcurrency: 8,
  mode: "plan",
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

const depsFor = (input: {
  cwd: string;
  updates: unknown[];
  messages: string[];
  runAcpTool: <T>(tool: AcpToolInput) => Promise<T>;
  aborted?: () => boolean;
}): AcpPlanningRunnerDeps => {
  const config = defaultConfig();
  return {
    connection: {
      sessionUpdate: async (update: unknown) => {
        input.updates.push(update);
      },
    } as AcpConnection,
    config,
    agentMessage: async (_sessionId, text) => {
      input.messages.push(text);
    },
    runAcpTool: input.runAcpTool,
    configForSession: () => config,
    throwIfAborted: (signal) => {
      if (signal?.aborted || input.aborted?.() === true) {
        throw new Error("cancelled");
      }
    },
    isAbortError: (error, signal) => signal?.aborted === true || error instanceof Error && error.message === "cancelled",
  };
};

describe("ACP planning runner module", () => {
  test("streams planning updates and writes the expected success artifacts", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-planning-runner-success-"));
    const session = sessionFor(cwd);
    const updates: unknown[] = [];
    const messages: string[] = [];
    const toolNames: string[] = [];

    await runAcpPlanningTurn(
      depsFor({
        cwd,
        updates,
        messages,
        runAcpTool: async <T,>(tool: AcpToolInput) => {
          toolNames.push(tool.toolName);
          switch (tool.toolName) {
            case "bag.knowledge.load":
              return "knowledge\n" as T;
            case "bag.context.scout":
              return [] as T;
            case "bag.context.build":
              return "repo context\n" as T;
            case "bag.interview":
              return {
                question: "ok?",
                rationale: "test",
                acceptedFacts: [],
                openQuestions: [],
                canGeneratePrdNow: true,
                suggestedNextAction: "generate_prd",
              } as T;
            case "bag.prd.generate":
              return {
                documentTitle: "Test PRD",
                sections: [{ key: "problem_statement", title: "Problem Statement", body: "Build it." }],
              } as T;
            case "bag.dag.generate":
              return {
                summary: {
                  planId: "plan.test",
                  title: "Test DAG",
                  status: "ready",
                  issueCount: 0,
                  dependencyCount: 0,
                  chosenTier: "small",
                },
                issues: [],
                dependencies: [],
              } as T;
            case "bag.self.evaluate":
              return { score: 1, passed: true, strengths: [], weaknesses: [], improvementActions: [] } as T;
            case "bag.policy.optimize":
              return { evaluatedRuns: 0, evaluatedMetrics: 0, passRate: 1, notes: [] } as T;
            case "bag.knowledge.codify":
              return join(cwd, ".bag", "knowledge.md") as T;
            default:
              throw new Error(`unexpected tool: ${tool.toolName}`);
          }
        },
      }),
      { session, task: "write a plan", signal: new AbortController().signal },
    );

    const runRoot = latestRunRoot(cwd);
    expect(toolNames).toEqual([
      "bag.knowledge.load",
      "bag.context.scout",
      "bag.context.build",
      "bag.interview",
      "bag.prd.generate",
      "bag.dag.generate",
      "bag.self.evaluate",
      "bag.policy.optimize",
      "bag.knowledge.codify",
    ]);
    expect(existsSync(join(runRoot, "planning-trace.json"))).toBe(true);
    expect(existsSync(join(runRoot, "manifest.json"))).toBe(true);
    expect(existsSync(join(runRoot, "prd.md"))).toBe(true);
    expect(existsSync(join(runRoot, "dag.md"))).toBe(true);
    expect(JSON.stringify(updates)).toContain("\"sessionUpdate\":\"plan\"");
    expect(messages.join("\n")).toContain("ACP planning turn complete");
  });

  test("preserves cancellation artifacts after partial planning progress", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-planning-runner-cancel-"));
    const session = sessionFor(cwd);
    const updates: unknown[] = [];
    const messages: string[] = [];
    const abortController = new AbortController();
    let toolCalls = 0;

    await expect(runAcpPlanningTurn(
      depsFor({
        cwd,
        updates,
        messages,
        aborted: () => abortController.signal.aborted,
        runAcpTool: async <T,>(tool: AcpToolInput) => {
          toolCalls += 1;
          if (tool.toolName !== "bag.knowledge.load") {
            throw new Error(`unexpected planning side effect after cancellation: ${tool.toolName}`);
          }
          abortController.abort();
          return "learned planning context\n" as T;
        },
      }),
      { session, task: "write a PRD", signal: abortController.signal },
    )).rejects.toThrow("cancelled");

    const runRoot = latestRunRoot(cwd);
    const cancellation = JSON.parse(readFileSync(join(runRoot, "cancellation.json"), "utf8")) as {
      partialArtifacts?: Record<string, string>;
      completedPlanEntries?: string[];
    };
    const manifest = JSON.parse(readFileSync(join(runRoot, "manifest.json"), "utf8")) as {
      artifacts?: Record<string, string>;
    };

    expect(toolCalls).toBe(1);
    expect(existsSync(join(runRoot, "planning-trace.json"))).toBe(true);
    expect(existsSync(join(runRoot, "knowledge-input.md"))).toBe(true);
    expect(cancellation.partialArtifacts?.knowledgeInput).toContain("knowledge-input.md");
    expect(cancellation.completedPlanEntries?.join("\n")).toContain("Load learned guidance");
    expect(manifest.artifacts?.trace).toContain("planning-trace.json");
    expect(manifest.artifacts?.cancellation).toContain("cancellation.json");
    expect(messages.join("\n")).toContain("ACP planning turn cancelled");
  });
});

const latestRunRoot = (cwd: string): string => {
  const runsDir = join(cwd, ".bag", "runs");
  const runId = readdirSync(runsDir).find((entry) => entry.startsWith("acp-"));
  expect(runId).toBeDefined();
  return join(runsDir, runId ?? "");
};
