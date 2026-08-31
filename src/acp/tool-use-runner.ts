import type {
  AgentSideConnection as AcpConnection,
  TerminalHandle,
} from "@agentclientprotocol/sdk";
import { createArtifactWriter, createRunId } from "../artifacts";
import { createLlmRouter, type LlmRouter } from "../llm";
import {
  classifyTaskShape,
  LONG_WAIT_RUNTIME_HINT,
  type TaskShapeDecision,
} from "../task-shape-router";
import { RunTelemetry } from "../telemetry";
import type { BagConfig } from "../types";
import { buildRepoContext, loadKnowledge, runLocalContextScouts } from "../workspace";
import {
  attachLiveMcpToolsToRouter,
  type AcpMcpToolUseRuntimeServer,
} from "./mcp-bridge";
import type { BagAcpSession } from "./session";
import { buildAcpToolUseClient } from "./terminal";

export type AcpToolUseRunnerDeps = {
  connection: AcpConnection;
  config: BagConfig;
  agentMessage: (sessionId: string, text: string) => Promise<void>;
  createRouter?: (config: BagConfig, telemetry: RunTelemetry) => LlmRouter;
  mcpRuntimeServers?: (
    session: BagAcpSession,
  ) => Promise<readonly AcpMcpToolUseRuntimeServer[]> | readonly AcpMcpToolUseRuntimeServer[];
};

export type AcpToolUseTurnInput = {
  session: BagAcpSession;
  task: string;
  signal: AbortSignal;
};

type AutonomousToolUseResult = {
  trace: unknown;
  stopReason: string;
  turnsUsed: number;
  toolCallsExecuted: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  submittedOutput?: unknown;
};

type DagToolLoopResult = {
  stopReason: string;
  plannedIssueCount: number;
  passedIssueCount: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalBashCalls: number;
  issues: Array<{
    issue: { issueId: string; title: string };
    stopReason: string;
    verifierPassed: boolean;
    verifierExitCodes: number[];
    turnsUsed: number;
    bashCallsExecuted: number;
    repairRoundsUsed: number;
    trace: unknown;
  }>;
};

export const runAcpAdaptiveCodingTurn = async (
  deps: AcpToolUseRunnerDeps,
  input: AcpToolUseTurnInput,
): Promise<void> => {
  const { session, task, signal } = input;
  const runId = `acp-auto-${createRunId()}`;
  const writer = createArtifactWriter(deps.config, runId, session.cwd);
  const telemetry = new RunTelemetry(deps.config, runId, session.cwd, session.optimizerPin.telemetry);
  const router = await createToolUseRouter(deps, session, telemetry, signal);
  await deps.agentMessage(session.id, `Adaptive coding turn started (run ${runId}).`);
  if (!router.masterAvailable) {
    await deps.agentMessage(session.id, "No master model is configured; cannot run adaptive mode.");
    return;
  }
  if (!session.clientCapabilities.terminal) {
    await deps.agentMessage(session.id, "Client does not advertise terminal capability; adaptive mode requires bash.");
    return;
  }

  const repoContext = await buildToolUseRepoContext({ config: deps.config, router, session, task });
  const decision: TaskShapeDecision = await classifyTaskShape({
    router,
    task,
    repoContext,
    cwd: session.cwd,
  });
  const chosenAt = new Date().toISOString();
  writer.writeJson("routing-decision.json", { ...decision, chosenAt, task });
  await deps.agentMessage(
    session.id,
    `Routing decision: shape=${decision.shape} mode=${decision.mode} confidence=${decision.confidence.toFixed(2)} requiresLongWait=${decision.requiresLongWait}. Reason: ${decision.reasoning}`,
  );

  // Semantic dimension: when the classifier marks `requiresLongWait`, surface
  // the background-process + polling hint to the master model. The runner
  // itself does NOT keyword-scan the task — it only reads the boolean.
  const runtimeHint: string | undefined = decision.requiresLongWait
    ? LONG_WAIT_RUNTIME_HINT
    : undefined;

  const handles = new Map<string, TerminalHandle>();
  const client = buildAcpToolUseClient(deps.connection, handles);

  if (decision.mode === "dag-tools") {
    const { runDagToolLoop } = await import("../dag-tool-loop");
    const orchestration = await runDagToolLoop({
      router,
      client,
      sessionId: session.id,
      cwd: session.cwd,
      task,
      repoContext,
      signal,
      ...(runtimeHint === undefined ? {} : { runtimeHint }),
      onPlanned: async (issues) => {
        writer.writeJson("planned-issues.json", issues);
        await deps.agentMessage(
          session.id,
          `Planned ${issues.length} issue(s):\n${issues.map((it, i) => `  ${i + 1}. ${it.title}`).join("\n")}`,
        );
      },
      onIssueStart: async (issue, index, total) => {
        await deps.agentMessage(session.id, `Starting issue ${index + 1}/${total}: ${issue.title}`);
      },
      onIssueComplete: async (result, index, total) => {
        await deps.agentMessage(
          session.id,
          `Finished issue ${index + 1}/${total}: ${result.issue.title} — stopReason=${result.stopReason}, verifierPassed=${result.verifierPassed}, turns=${result.turnsUsed}, bash=${result.bashCallsExecuted}.`,
        );
      },
    });
    writeDagToolLoopArtifacts(writer.writeJson, orchestration.result);
    await deps.agentMessage(
      session.id,
      `Adaptive (dag-tools) complete: ${orchestration.result.stopReason}. ${orchestration.result.passedIssueCount}/${orchestration.result.plannedIssueCount} issues passed verifier. tokens_in=${orchestration.result.totalPromptTokens} tokens_out=${orchestration.result.totalCompletionTokens} bash=${orchestration.result.totalBashCalls}.`,
    );
  } else {
    const { runAutonomousCodingTurn } = await import("../autonomous-coding-turn");
    const { buildVerifierFromInstruction } = await import("../instruction-verifier");
    const verifyAfterSubmit = await buildVerifierFromInstruction({ router, instruction: task });
    const result = await runAutonomousCodingTurn({
      router,
      client,
      sessionId: session.id,
      cwd: session.cwd,
      task,
      signal,
      ...(verifyAfterSubmit === undefined ? {} : { verifyAfterSubmit }),
      ...(runtimeHint === undefined ? {} : { config: { runtimeHint } }),
    });
    writer.writeJson("autonomous-trace.json", result.trace);
    writer.writeJson("autonomous-summary.json", autonomousSummary(result));
    await deps.agentMessage(
      session.id,
      `Adaptive (tools) complete: ${result.stopReason}. turns=${result.turnsUsed} bash_calls=${result.toolCallsExecuted} attempts=${result.attemptsUsed} tokens_in=${result.totalPromptTokens} tokens_out=${result.totalCompletionTokens}. Trace: ${session.cwd}/.bag/runs/${runId}/autonomous-trace.json`,
    );
  }
  void telemetry;
};

export const runAcpDagDrivenToolUseTurn = async (
  deps: AcpToolUseRunnerDeps,
  input: AcpToolUseTurnInput,
): Promise<void> => {
  const { session, task, signal } = input;
  const runId = `acp-dag-tools-${createRunId()}`;
  const writer = createArtifactWriter(deps.config, runId, session.cwd);
  const telemetry = new RunTelemetry(deps.config, runId, session.cwd, session.optimizerPin.telemetry);
  const router = await createToolUseRouter(deps, session, telemetry, signal);
  await deps.agentMessage(session.id, `DAG-driven tool-use turn started (run ${runId}).`);
  if (!router.masterAvailable) {
    await deps.agentMessage(session.id, "No master model is configured; cannot run DAG-tools mode.");
    return;
  }
  if (!session.clientCapabilities.terminal) {
    await deps.agentMessage(session.id, "Client does not advertise terminal capability; DAG-tools mode requires bash.");
    return;
  }
  const handles = new Map<string, TerminalHandle>();
  const client = buildAcpToolUseClient(deps.connection, handles);
  const repoContext = await buildToolUseRepoContext({ config: deps.config, router, session, task });
  const { runDagToolLoop } = await import("../dag-tool-loop");
  const orchestration = await runDagToolLoop({
    router,
    client,
    sessionId: session.id,
    cwd: session.cwd,
    task,
    repoContext,
    signal,
    onPlanned: async (issues) => {
      writer.writeJson("planned-issues.json", issues);
      await deps.agentMessage(
        session.id,
        `Planned ${issues.length} issue(s):\n${issues.map((it, i) => `  ${i + 1}. ${it.title}`).join("\n")}`,
      );
    },
    onIssueStart: async (issue, index, total) => {
      await deps.agentMessage(session.id, `Starting issue ${index + 1}/${total}: ${issue.title}`);
    },
    onIssueComplete: async (result, index, total) => {
      await deps.agentMessage(
        session.id,
        `Finished issue ${index + 1}/${total}: ${result.issue.title} — stopReason=${result.stopReason}, verifierPassed=${result.verifierPassed}, turns=${result.turnsUsed}, bash=${result.bashCallsExecuted}.`,
      );
    },
  });
  writeDagToolLoopArtifacts(writer.writeJson, orchestration.result);
  await deps.agentMessage(
    session.id,
    `DAG-tools complete: ${orchestration.result.stopReason}. ${orchestration.result.passedIssueCount}/${orchestration.result.plannedIssueCount} issues passed verifier. tokens_in=${orchestration.result.totalPromptTokens} tokens_out=${orchestration.result.totalCompletionTokens} bash=${orchestration.result.totalBashCalls}.`,
  );
  void telemetry;
};

export const runAcpAutonomousToolUseTurn = async (
  deps: AcpToolUseRunnerDeps,
  input: AcpToolUseTurnInput,
): Promise<void> => {
  const { session, task, signal } = input;
  const runId = `acp-tools-${createRunId()}`;
  const writer = createArtifactWriter(deps.config, runId, session.cwd);
  const telemetry = new RunTelemetry(deps.config, runId, session.cwd, session.optimizerPin.telemetry);
  const router = await createToolUseRouter(deps, session, telemetry, signal);
  await deps.agentMessage(session.id, `Autonomous tool-use turn started (run ${runId}).`);
  if (!router.masterAvailable) {
    await deps.agentMessage(session.id, "No master model is configured; cannot run autonomous mode.");
    return;
  }
  if (!session.clientCapabilities.terminal) {
    await deps.agentMessage(session.id, "Client does not advertise terminal capability; autonomous mode requires bash.");
    return;
  }
  const handles = new Map<string, TerminalHandle>();
  const client = buildAcpToolUseClient(deps.connection, handles);
  const { runAutonomousCodingTurn } = await import("../autonomous-coding-turn");
  const { buildVerifierFromInstruction } = await import("../instruction-verifier");
  const verifyAfterSubmit = await buildVerifierFromInstruction({ router, instruction: task });
  const result = await runAutonomousCodingTurn({
    router,
    client,
    sessionId: session.id,
    cwd: session.cwd,
    task,
    signal,
    ...(verifyAfterSubmit === undefined ? {} : { verifyAfterSubmit }),
  });
  writer.writeJson("autonomous-trace.json", result.trace);
  writer.writeJson("autonomous-summary.json", autonomousSummary(result));
  await deps.agentMessage(
    session.id,
    `Autonomous turn complete: ${result.stopReason}. turns=${result.turnsUsed} bash_calls=${result.toolCallsExecuted} attempts=${result.attemptsUsed} tokens_in=${result.totalPromptTokens} tokens_out=${result.totalCompletionTokens}. Trace: ${session.cwd}/.bag/runs/${runId}/autonomous-trace.json`,
  );
  void telemetry;
};

const createToolUseRouter = async (
  deps: AcpToolUseRunnerDeps,
  session: BagAcpSession,
  telemetry: RunTelemetry,
  signal: AbortSignal,
): Promise<LlmRouter> => {
  const router = deps.createRouter?.(deps.config, telemetry) ?? createLlmRouter(deps.config, telemetry);
  const servers = await deps.mcpRuntimeServers?.(session);
  if (servers == null || servers.length === 0) {
    return router;
  }
  return attachLiveMcpToolsToRouter(
    { connection: deps.connection },
    { session, telemetry, router, servers, signal },
  );
};

const buildToolUseRepoContext = async (input: {
  config: BagConfig;
  router: ReturnType<typeof createLlmRouter>;
  session: BagAcpSession;
  task: string;
}): Promise<string> => {
  const knowledge = loadKnowledge(input.session.cwd);
  const scoutFindings = await runLocalContextScouts({
    router: input.router,
    config: input.config,
    task: input.task,
    cwd: input.session.cwd,
  });
  const repoContext = buildRepoContext({
    cwd: input.session.cwd,
    config: input.config,
    task: input.task,
    findings: scoutFindings,
  });
  void knowledge;
  return repoContext;
};

const autonomousSummary = (result: AutonomousToolUseResult) => ({
  stopReason: result.stopReason,
  turnsUsed: result.turnsUsed,
  toolCallsExecuted: result.toolCallsExecuted,
  totalPromptTokens: result.totalPromptTokens,
  totalCompletionTokens: result.totalCompletionTokens,
  submittedOutput: result.submittedOutput,
});

const writeDagToolLoopArtifacts = (
  writeJson: (name: string, data: unknown) => string,
  result: DagToolLoopResult,
): void => {
  writeJson("dag-tool-loop-summary.json", {
    stopReason: result.stopReason,
    plannedIssueCount: result.plannedIssueCount,
    passedIssueCount: result.passedIssueCount,
    totalPromptTokens: result.totalPromptTokens,
    totalCompletionTokens: result.totalCompletionTokens,
    totalBashCalls: result.totalBashCalls,
    issues: result.issues.map((r) => ({
      issueId: r.issue.issueId,
      title: r.issue.title,
      stopReason: r.stopReason,
      verifierPassed: r.verifierPassed,
      verifierExitCodes: r.verifierExitCodes,
      turnsUsed: r.turnsUsed,
      bashCallsExecuted: r.bashCallsExecuted,
      repairRoundsUsed: r.repairRoundsUsed,
    })),
  });
  writeJson("dag-tool-loop-traces.json", result.issues.map((r) => ({
    issueId: r.issue.issueId,
    trace: r.trace,
  })));
};
