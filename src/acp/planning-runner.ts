import type {
  AgentSideConnection as AcpConnection,
  PlanEntry,
} from "@agentclientprotocol/sdk";
import { resolve } from "node:path";
import { createArtifactWriter, createRunId, writeManifest } from "../artifacts";
import { generateDag, renderDagMarkdown } from "../dag";
import { runInterview } from "../interview";
import { createLlmRouter } from "../llm";
import { codifyKnowledge, optimizePolicy, type OptimizationReport } from "../optimize";
import { generatePrd, renderPrdMarkdown } from "../prd";
import { deterministicSelfEvaluation, RunTelemetry } from "../telemetry";
import type { BagConfig, ContextScoutFinding, DagPlan, InterviewTurn, PrdArtifact, SelfEvaluation } from "../types";
import { buildRepoContext, loadKnowledge, runLocalContextScouts } from "../workspace";
import type { BagAcpSession } from "./session";
import { artifactLocation, initialPlan, traceEvent, updatePlanEntry, type TraceEvent } from "./surface";
import type { AcpToolInput } from "./tool-runner";

export type AcpPlanningRunnerDeps = {
  connection: AcpConnection;
  config: BagConfig;
  agentMessage: (sessionId: string, text: string) => Promise<void>;
  runAcpTool: <T>(input: AcpToolInput) => Promise<T>;
  configForSession: (session: BagAcpSession) => BagConfig;
  throwIfAborted: (signal?: AbortSignal) => void;
  isAbortError: (error: unknown, signal?: AbortSignal) => boolean;
};

export type AcpPlanningTurnInput = {
  session: BagAcpSession;
  task: string;
  signal: AbortSignal;
};

export const runAcpPlanningTurn = async (
  deps: AcpPlanningRunnerDeps,
  input: AcpPlanningTurnInput,
): Promise<void> => {
  const { session, task, signal } = input;
  const runId = `acp-${createRunId()}`;
  const writer = createArtifactWriter(deps.config, runId, session.cwd);
  const telemetry = new RunTelemetry(deps.config, runId, session.cwd, session.optimizerPin.telemetry);
  const router = createLlmRouter(deps.config, telemetry);
  const trace: TraceEvent[] = [];
  const partialArtifacts: Record<string, string> = {};
  let plan = initialPlan();

  const pushTrace = (entry: TraceEvent) => {
    trace.push(entry);
    telemetry.event("acp.planning.trace", entry as unknown as Record<string, unknown>);
  };
  const setPlan = async (index: number, status: PlanEntry["status"]) => {
    plan = updatePlanEntry(plan, index, status);
    await deps.connection.sessionUpdate({
      sessionId: session.id,
      update: {
        sessionUpdate: "plan",
        entries: plan,
      },
    });
  };

  try {
    telemetry.event("acp.turn.started", {
      sessionId: session.id,
      mode: session.mode,
      task,
    });
    pushTrace(traceEvent("turn", "started", true, { runId, task, cwd: session.cwd }));

    await deps.agentMessage(
      session.id,
      `Starting BleedingAgent ACP turn in ${session.mode} mode. I will stream each planning phase as a visible ACP tool call.`,
    );
    await deps.connection.sessionUpdate({
      sessionId: session.id,
      update: {
        sessionUpdate: "plan",
        entries: plan,
      },
    });

    deps.throwIfAborted(signal);
    await setPlan(0, "in_progress");
    const knowledge = await deps.runAcpTool<string>({
      sessionId: session.id,
      telemetry,
      title: "Load BleedingAgent knowledge",
      toolName: "bag.knowledge.load",
      kind: "read",
      rawInput: { cwd: session.cwd },
      locations: [
        artifactLocation(resolve(session.cwd, ".bag", "knowledge.md")),
        artifactLocation(resolve(session.cwd, ".bag", "tool-guidance.md")),
      ],
      signal,
      fn: async () => loadKnowledge(session.cwd),
    });
    partialArtifacts.knowledgeInput = writer.write("knowledge-input.md", knowledge);
    pushTrace(traceEvent("knowledge", "completed", true, {
      bytes: Buffer.byteLength(knowledge),
      artifact: partialArtifacts.knowledgeInput,
    }));
    await setPlan(0, "completed");

    deps.throwIfAborted(signal);
    await setPlan(1, "in_progress");
    const scoutFindings = await deps.runAcpTool<ContextScoutFinding[]>({
      sessionId: session.id,
      telemetry,
      title: "Scout repository context",
      toolName: "bag.context.scout",
      kind: "search",
      rawInput: {
        task,
        executorConcurrency: session.executorConcurrency,
        localModel: deps.config.local.model,
      },
      signal,
      fn: async () => runLocalContextScouts({ router, config: deps.configForSession(session), task, cwd: session.cwd }),
    });
    partialArtifacts.scoutFindings = writer.writeJson("context-scout-findings.json", scoutFindings);
    pushTrace(traceEvent("context_scout", "completed", true, {
      findingCount: scoutFindings.length,
      artifact: partialArtifacts.scoutFindings,
    }));

    const repoContext = await deps.runAcpTool<string>({
      sessionId: session.id,
      telemetry,
      title: "Build repository context",
      toolName: "bag.context.build",
      kind: "read",
      rawInput: { task, findings: scoutFindings.slice(0, 12) },
      signal,
      fn: async () =>
        buildRepoContext({ cwd: session.cwd, config: deps.configForSession(session), task, findings: scoutFindings }),
    });
    partialArtifacts.repoContext = writer.write("repo-context.md", repoContext);
    pushTrace(traceEvent("repo_context", "completed", true, {
      bytes: Buffer.byteLength(repoContext),
      artifact: partialArtifacts.repoContext,
    }));
    await setPlan(1, "completed");

    deps.throwIfAborted(signal);
    await setPlan(2, "in_progress");
    const interview = await deps.runAcpTool<InterviewTurn>({
      sessionId: session.id,
      telemetry,
      title: "Run interview flow",
      toolName: "bag.interview",
      kind: "think",
      rawInput: { task },
      signal,
      fn: async () => runInterview({ router, task, repoContext, knowledge }),
    });
    partialArtifacts.interview = writer.writeJson("interview.json", interview);
    pushTrace(traceEvent("interview", "completed", true, { artifact: partialArtifacts.interview }));
    await setPlan(2, "completed");

    deps.throwIfAborted(signal);
    await setPlan(3, "in_progress");
    const prd = await deps.runAcpTool<PrdArtifact>({
      sessionId: session.id,
      telemetry,
      title: "Generate PRD",
      toolName: "bag.prd.generate",
      kind: "think",
      rawInput: { task, interview },
      signal,
      fn: async () => generatePrd({ router, task, interview, repoContext, knowledge }),
    });
    partialArtifacts.prdJson = writer.writeJson("prd.json", prd);
    partialArtifacts.prdMarkdown = writer.write("prd.md", renderPrdMarkdown(prd));
    pushTrace(traceEvent("prd", "completed", true, {
      prdJson: partialArtifacts.prdJson,
      prdMarkdown: partialArtifacts.prdMarkdown,
    }));
    await setPlan(3, "completed");

    deps.throwIfAborted(signal);
    await setPlan(4, "in_progress");
    const dag = await deps.runAcpTool<DagPlan>({
      sessionId: session.id,
      telemetry,
      title: "Generate DAG",
      toolName: "bag.dag.generate",
      kind: "think",
      rawInput: { prd },
      signal,
      fn: async () => generateDag({ router, prd, repoContext }),
    });
    partialArtifacts.dagJson = writer.writeJson("dag.json", dag);
    partialArtifacts.dagMarkdown = writer.write("dag.md", renderDagMarkdown(dag));
    pushTrace(traceEvent("dag", "completed", true, {
      dagJson: partialArtifacts.dagJson,
      dagMarkdown: partialArtifacts.dagMarkdown,
    }));
    await setPlan(4, "completed");

    deps.throwIfAborted(signal);
    await setPlan(5, "in_progress");
    const selfEvaluation = await deps.runAcpTool<SelfEvaluation>({
      sessionId: session.id,
      telemetry,
      title: "Self-evaluate ACP turn",
      toolName: "bag.self.evaluate",
      kind: "think",
      rawInput: { artifactCount: Object.keys(partialArtifacts).length },
      signal,
      fn: async () =>
        deterministicSelfEvaluation({
          threshold: deps.config.policy.selfEvalThreshold,
          metrics: telemetry.metrics,
          llmMetrics: telemetry.llmMetrics,
          toolMetrics: telemetry.toolMetrics,
          artifactCount: Object.keys(partialArtifacts).length,
        }),
    });

    const optimization = await deps.runAcpTool<OptimizationReport>({
      sessionId: session.id,
      telemetry,
      title: "Capture telemetry learning",
      toolName: "bag.policy.optimize",
      kind: "think",
      rawInput: { latestSelfEvaluation: selfEvaluation },
      signal,
      fn: async () => optimizePolicy({ config: deps.config, cwd: session.cwd, latestSelfEvaluation: selfEvaluation }),
    });

    const knowledgePath = await deps.runAcpTool<string>({
      sessionId: session.id,
      telemetry,
      title: "Codify learning",
      toolName: "bag.knowledge.codify",
      kind: "edit",
      rawInput: { runId, task, selfEvaluation, optimization },
      locations: [artifactLocation(resolve(session.cwd, deps.config.artifactDir, "knowledge.md"))],
      signal,
      fn: async () =>
        codifyKnowledge({
          config: deps.config,
          runId,
          task,
          report: optimization,
          selfEvaluation,
          cwd: session.cwd,
        }),
    });
    await setPlan(5, "completed");

    pushTrace(traceEvent("turn", "completed", true, {
      runId,
      artifactCount: Object.keys(partialArtifacts).length,
    }));
    const tracePath = writer.writeJson("planning-trace.json", trace);
    const finalArtifacts = {
      ...partialArtifacts,
      trace: tracePath,
      selfEvaluation: writer.writeJson("self-evaluation.json", selfEvaluation),
      optimization: writer.writeJson("optimization.json", optimization),
      knowledge: knowledgePath,
    };
    const manifest = writeManifest({
      config: deps.config,
      command: "acp",
      task,
      runId,
      artifacts: finalArtifacts,
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      selfEvaluation,
      writeJson: writer.writeJson,
    });
    telemetry.event("acp.turn.completed", {
      sessionId: session.id,
      runId,
      selfEvalScore: selfEvaluation.score,
      selfEvalPassed: selfEvaluation.passed,
    });

    await deps.agentMessage(
      session.id,
      [
        `ACP planning turn complete: ${runId}.`,
        `Self-eval: ${selfEvaluation.score} passed=${selfEvaluation.passed}.`,
        `Artifacts root: ${writer.root}`,
        `Manifest: ${manifest}`,
        "",
        "Main artifacts:",
        ...Object.entries(finalArtifacts).map(([name, path]) => `- ${name}: ${path}`),
      ].join("\n"),
    );
  } catch (error) {
    if (!deps.isAbortError(error, signal)) {
      throw error;
    }
    const cancellation = {
      runId,
      sessionId: session.id,
      task,
      cancelledAt: new Date().toISOString(),
      completedPlanEntries: plan.filter((entry) => entry.status === "completed").map((entry) => entry.content),
      inProgressPlanEntries: plan.filter((entry) => entry.status === "in_progress").map((entry) => entry.content),
      partialArtifacts: { ...partialArtifacts },
    };
    pushTrace(traceEvent("turn", "cancelled", false, cancellation));
    const tracePath = writer.writeJson("planning-trace.json", trace);
    const cancellationPath = writer.writeJson("cancellation.json", cancellation);
    const selfEvaluation = deterministicSelfEvaluation({
      threshold: deps.config.policy.selfEvalThreshold,
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      artifactCount: Object.keys(partialArtifacts).length + 2,
    });
    const manifest = writeManifest({
      config: deps.config,
      command: "acp",
      task,
      runId,
      artifacts: {
        ...partialArtifacts,
        trace: tracePath,
        cancellation: cancellationPath,
      },
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      selfEvaluation,
      writeJson: writer.writeJson,
    });
    telemetry.event("acp.turn.cancelled", {
      sessionId: session.id,
      runId,
      manifest,
      tracePath,
      cancellationPath,
      partialArtifactCount: Object.keys(partialArtifacts).length,
    });
    await deps.agentMessage(
      session.id,
      [
        `ACP planning turn cancelled: ${runId}.`,
        `Trace: ${tracePath}`,
        `Cancellation: ${cancellationPath}`,
        `Manifest: ${manifest}`,
      ].join("\n"),
    );
    throw error;
  }
};
