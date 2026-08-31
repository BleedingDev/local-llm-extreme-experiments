import { createArtifactWriter, createRunId, writeManifest } from "./artifacts";
import { generateDag, renderDagMarkdown } from "./dag";
import { runInterview } from "./interview";
import { createLlmRouter } from "./llm";
import { codifyKnowledge, optimizePolicy } from "./optimize";
import { createOptimizerSessionPin } from "./optimizer/session-pin";
import { generatePrd, renderPrdMarkdown } from "./prd";
import { deterministicSelfEvaluation, RunTelemetry } from "./telemetry";
import type { BagConfig } from "./types";
import { buildRepoContext, loadKnowledge, runLocalContextScouts } from "./workspace";

export const runPlanningPipeline = async (input: {
  config: BagConfig;
  task: string;
  command: string;
  cwd?: string;
}) => {
  const cwd = input.cwd ?? process.cwd();
  const runId = createRunId();
  const writer = createArtifactWriter(input.config, runId, cwd);
  const optimizerPin = createOptimizerSessionPin(input.config, cwd);
  const telemetry = new RunTelemetry(input.config, runId, cwd, optimizerPin.telemetry);
  const router = createLlmRouter(input.config, telemetry);

  telemetry.event("run.started", {
    command: input.command,
    task: input.task,
    executorConcurrency: input.config.policy.executorConcurrency,
    maxExecutorConcurrency: input.config.policy.maxExecutorConcurrency,
  });

  const knowledge = await telemetry.measure("knowledge.load", "deterministic", async () => loadKnowledge(cwd));
  const scoutFindings = await telemetry.measure("context.scout", "local", async () =>
    runLocalContextScouts({
      router,
      config: input.config,
      task: input.task,
      cwd,
    }),
  );
  const repoContext = await telemetry.measure("context.build", "deterministic", async () =>
    buildRepoContext({
      cwd,
      config: input.config,
      task: input.task,
      findings: scoutFindings,
    }),
  );
  const interview = await telemetry.measure("interview", router.masterAvailable ? "master" : "deterministic", async () =>
    runInterview({
      router,
      task: input.task,
      repoContext,
      knowledge,
    }),
  );
  const prd = await telemetry.measure("prd.generate", router.masterAvailable ? "master" : "deterministic", async () =>
    generatePrd({
      router,
      task: input.task,
      interview,
      repoContext,
      knowledge,
    }),
  );
  const dag = await telemetry.measure("dag.generate", router.masterAvailable ? "master" : "deterministic", async () =>
    generateDag({
      router,
      prd,
      repoContext,
    }),
  );

  const artifacts = {
    scoutFindings: writer.writeJson("context-scout-findings.json", scoutFindings),
    repoContext: writer.write("repo-context.md", repoContext),
    interview: writer.writeJson("interview.json", interview),
    prdJson: writer.writeJson("prd.json", prd),
    prdMarkdown: writer.write("prd.md", renderPrdMarkdown(prd)),
    dagJson: writer.writeJson("dag.json", dag),
    dagMarkdown: writer.write("dag.md", renderDagMarkdown(dag)),
  };

  const selfEvaluation = await telemetry.measure("self.evaluate", "deterministic", async () =>
    deterministicSelfEvaluation({
      threshold: input.config.policy.selfEvalThreshold,
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      artifactCount: Object.keys(artifacts).length,
    }),
  );
  const optimization = await telemetry.measure("policy.optimize", "deterministic", async () =>
    optimizePolicy({ config: input.config, cwd, latestSelfEvaluation: selfEvaluation }),
  );
  const knowledgePath = await telemetry.measure("knowledge.codify", "deterministic", async () =>
    codifyKnowledge({
      config: input.config,
      runId,
      task: input.task,
      report: optimization,
      selfEvaluation,
      cwd,
    }),
  );

  const finalArtifacts = {
    ...artifacts,
    selfEvaluation: writer.writeJson("self-evaluation.json", selfEvaluation),
    optimization: writer.writeJson("optimization.json", optimization),
    knowledge: knowledgePath,
  };
  const manifest = writeManifest({
    config: input.config,
    command: input.command,
    task: input.task,
    runId,
    artifacts: finalArtifacts,
    metrics: telemetry.metrics,
    llmMetrics: telemetry.llmMetrics,
    toolMetrics: telemetry.toolMetrics,
    selfEvaluation,
    writeJson: writer.writeJson,
  });
  telemetry.event("run.completed", {
    selfEvalScore: selfEvaluation.score,
    selfEvalPassed: selfEvaluation.passed,
    artifactCount: Object.keys(finalArtifacts).length,
  });

  return {
    runId,
    root: writer.root,
    artifacts: { ...finalArtifacts, manifest },
    metrics: telemetry.metrics,
    llmMetrics: telemetry.llmMetrics,
    toolMetrics: telemetry.toolMetrics,
    selfEvaluation,
    optimization,
    interview,
    prd,
    dag,
  };
};
