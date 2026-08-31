import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync } from "node:fs";
import type { AgentSideConnection as AcpConnection } from "@agentclientprotocol/sdk";
import {
  handleMaintenanceCommand,
  inspectBackgroundOptimizationTrigger,
  renderMaintenanceStatus,
  renderMaintenanceEvalSummary,
} from "../src/acp/maintenance";
import type { BagAcpSession } from "../src/acp/session";
import { defaultConfig } from "../src/config";

const sessionFor = (cwd: string): BagAcpSession => ({
  id: "bag-maintenance-test",
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
  optimizerPin: {
    telemetry: {
      modelRole: "local",
      modelProfileId: "model.test",
      codebaseProfileId: "codebase.test",
      policyId: "policy.test",
      canonicalToolVersion: "tool.v1",
      renderedToolVersion: "rendered.v1",
      resultStyleVersion: "result.v1",
      verificationPolicyVersion: "verify.v1",
      editStrategyVersion: "edit.v1",
      renderedEditContractVersion: "edit-contract.v1",
      editFallbackPolicyVersion: "fallback.v1",
      editRepairPolicyVersion: "repair.v1",
      editVerifierPolicyVersion: "edit-verify.v1",
      editObjectiveSetId: "objective.v1",
    },
  } as never,
  clientCapabilities: {
    fsReadTextFile: true,
    fsWriteTextFile: true,
    terminal: true,
    richDiffContent: true,
    richTerminalContent: true,
    source: "test",
  },
});

describe("ACP maintenance module", () => {
  test("maintenance status surfaces runtime optimizer gate-suite fail-closed state", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-gate-suite-"));
    const status = renderMaintenanceStatus(defaultConfig(), sessionFor(cwd));

    expect(status).toContain("Runtime optimizer gate suite");
    expect(status).toContain("state: fail_closed");
    expect(status).toContain("promotion allowed: false");
    expect(status).toContain("gate errors: missing:");
  });

  test("runs maintenance eval through compact ACP progress updates", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-module-"));
    const config = defaultConfig();
    const session = sessionFor(cwd);
    const updates: unknown[] = [];
    const messages: string[] = [];

    await handleMaintenanceCommand(
      {
        connection: {
          sessionUpdate: async (update: unknown) => {
            updates.push(update);
          },
        } as AcpConnection,
        config,
        agentMessage: async (_sessionId, text) => {
          messages.push(text);
        },
      },
      session,
      "eval",
    );

    expect(messages.join("\n")).toContain("Maintenance eval summary");
    expect(renderMaintenanceEvalSummary()).toContain("holdout usage: hidden");
    expect(JSON.stringify(updates)).toContain("bag.maintenance.eval_summary");
    expect(JSON.stringify(updates)).toContain("\"status\":\"completed\"");
    expect(JSON.stringify(updates)).toContain("Read configured eval split metadata");
  });

  test("background trigger no-ops without enough metrics and spans", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-noop-"));
    const config = defaultConfig();
    const session = sessionFor(cwd);

    const diagnostic = inspectBackgroundOptimizationTrigger(config, session, {
      source: "test",
      sourceRunId: "run-insufficient",
      enqueue: true,
    });

    expect(diagnostic.triggered).toBe(false);
    expect(diagnostic.reason).toContain("insufficient evidence");
    expect(diagnostic.sideEffects).toEqual([]);
    expect(diagnostic.opportunityPath).toBeUndefined();
    expect(existsSync(join(cwd, ".bag", "maintenance", "opportunities.jsonl"))).toBe(false);
  });

  test("background trigger queues only an opportunity when real failure evidence exists", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-trigger-"));
    const config = defaultConfig();
    const session = sessionFor(cwd);
    mkdirSync(join(cwd, ".bag", "telemetry"), { recursive: true });
    writeFileSync(
      join(cwd, ".bag", "telemetry", "metrics.json"),
      `${JSON.stringify({
        run_1: {
          steps: [metricStep("context.scout", true), metricStep("verify", false), metricStep("context.build", true)],
          llmCalls: [metricLlm(true), metricLlm(true)],
          toolCalls: [metricTool("repo.read", true), metricTool("repo.write", false)],
        },
        run_2: {
          steps: [metricStep("context.scout", true), metricStep("context.build", true)],
          llmCalls: [metricLlm(true)],
          toolCalls: [metricTool("repo.read", true), metricTool("terminal.run", true)],
        },
      })}\n`,
    );
    writeFileSync(
      join(cwd, ".bag", "telemetry", "spans.jsonl"),
      Array.from({ length: 6 }, (_, index) =>
        JSON.stringify({
          trace_id: "trace-bg",
          span_id: `span-${index}`,
          parent_span_id: "root",
          trace_state: "",
          name: index === 0 ? "tool.workspace.repo.write" : "step.context",
          kind: "SPAN_KIND_CLIENT",
          start_time: "2026-04-29T00:00:00.000Z",
          end_time: "2026-04-29T00:00:01.000Z",
          status: { code: index === 0 ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK", message: index === 0 ? "write failed" : "" },
          resource: { attributes: { "service.name": "bleeding-agent" } },
          scope: { name: "bag.telemetry", version: "0.1.0" },
          attributes: {
            "inference.observation_kind": index === 0 ? "TOOL" : "CHAIN",
            "inference.project_id": "bleeding-agent",
            ...(index === 0 ? { "tool.name": "repo.write", "error.message": "write failed" } : {}),
          },
        }),
      ).join("\n") + "\n",
    );

    const diagnostic = inspectBackgroundOptimizationTrigger(config, session, {
      source: "test",
      sourceRunId: "run-enough",
      enqueue: true,
    });

    expect(diagnostic.triggered).toBe(true);
    expect(diagnostic.evidence.runCount).toBe(2);
    expect(diagnostic.evidence.metricObservationCount).toBeGreaterThanOrEqual(12);
    expect(diagnostic.evidence.spanCount).toBe(6);
    expect(diagnostic.sideEffects).toEqual(["append-maintenance-opportunity"]);
    expect(diagnostic.opportunityPath).toBe(join(cwd, ".bag", "maintenance", "opportunities.jsonl"));
    const opportunity = readFileSync(diagnostic.opportunityPath, "utf8");
    expect(opportunity).toContain("background-optimization-opportunity");
    expect(opportunity).toContain("/maintenance optimize");
    expect(opportunity).toContain("no automatic promotion");
    expect(existsSync(join(cwd, ".bag", "optimizer", "active.json"))).toBe(false);
  });
});

const metricStep = (step: string, ok: boolean) => ({
  step,
  startedAt: "2026-04-29T00:00:00.000Z",
  completedAt: "2026-04-29T00:00:01.000Z",
  durationMs: 1000,
  ok,
  modelRole: "deterministic" as const,
  ...(ok ? {} : { error: `${step} failed` }),
});

const metricLlm = (ok: boolean) => ({
  role: "local" as const,
  model: "local-model",
  endpoint: "http://127.0.0.1:18082/v1/chat/completions",
  startedAt: "2026-04-29T00:00:00.000Z",
  completedAt: "2026-04-29T00:00:01.000Z",
  durationMs: 1000,
  ok,
  totalTokens: 42,
  ...(ok ? {} : { error: "llm failed" }),
});

const metricTool = (toolName: string, ok: boolean) => ({
  toolName,
  startedAt: "2026-04-29T00:00:00.000Z",
  completedAt: "2026-04-29T00:00:01.000Z",
  durationMs: 1000,
  ok,
  ...(ok ? {} : { error: `${toolName} failed` }),
});
