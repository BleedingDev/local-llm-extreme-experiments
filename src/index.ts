#!/usr/bin/env node
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { readAcpSettingsSnippet, startAcpServer } from "./acp-agent";
import { configPath, defaultConfig, loadConfig, resolveAllModelRoleConfigs } from "./config";
import { generateDag, renderDagMarkdown } from "./dag";
import { EVIDENCE_COMMANDS, isEvidenceCommandName, runEvidenceCommand } from "./evidence/evidence-commands";
import { runInterview } from "./interview";
import { createAxBleedingAgent, createAxServices, createLlmRouter } from "./llm";
import { readMetricsStore, summarizeMetricsStore } from "./metrics";
import { optimizePolicy } from "./optimize";
import { runPromotionWorkflowCommand, type PromotionWorkflowAction } from "./optimizer/promotion-workflow";
import { loadOptimizerRegistry } from "./optimizer/registry";
import { resolveLoadedOptimizerPolicy } from "./optimizer/policy-resolver";
import { runPlanningPipeline } from "./pipeline";
import { generatePrd, renderPrdMarkdown } from "./prd";
import { applySelfOptimization, generateSelfOptimization } from "./self-optimize";
import { buildRepoContext, loadKnowledge, runLocalContextScouts } from "./workspace";

const usage = () => `BleedingAgent (bag)

Usage:
  bag init
  bag doctor
  bag interview <task>
  bag prd <task>
  bag dag <task>
  bag run <task>
  bag optimize
  bag optimizer <promotion-preview|approve|promote|monitor|rollback> [--graph-id <graph-id>] [--selection-hash <selection-hash>] [--candidate-id <candidate-id>] [--promotion-decision-id <decision-id>]
  bag self-optimize [--apply]
  bag apply-optimization [optimization-id]
  bag evidence <${EVIDENCE_COMMANDS.join("|")}> [--write] [--graph-id <graph-id>]
  bag acp
  bag acp-settings [zed]
  bag metrics
  bag ax-smoke <task>

Environment:
  OPENAI_API_KEY             enables GPT master/critic
  bag.config.json            overrides model endpoints, telemetry, and concurrency
`;

const taskFromArgs = (args: string[]): string => args.join(" ").trim();

const argValue = (args: readonly string[], name: string): string | undefined => {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  const value = args[index + 1];
  return value == null || value.startsWith("--") ? undefined : value;
};

const optimizerAction = (value: string | undefined): PromotionWorkflowAction | undefined => {
  switch (value) {
    case "promotion-preview":
      return "preview";
    case "approve":
    case "promote":
    case "monitor":
    case "rollback":
      return value;
    default:
      return undefined;
  }
};

const printArtifacts = (artifacts: Record<string, string>) => {
  for (const [name, path] of Object.entries(artifacts)) {
    console.log(`${name}: ${path}`);
  }
};

const main = async () => {
  const [, , commandRaw, ...rest] = process.argv;
  const command = commandRaw ?? "help";
  const cwd = process.cwd();

  if (command === "help" || command === "--help" || command === "-h") {
    console.log(usage());
    return;
  }

  if (command === "init") {
    const path = configPath(cwd);
    if (existsSync(path)) {
      console.log(`config exists: ${path}`);
      return;
    }
    writeFileSync(path, `${JSON.stringify(defaultConfig(), null, 2)}\n`);
    mkdirSync(resolve(cwd, ".bag"), { recursive: true });
    console.log(`created ${path}`);
    return;
  }

  if (command === "acp") {
    await startAcpServer();
    return;
  }

  const config = loadConfig(cwd);
  const router = createLlmRouter(config);

  if (command === "doctor") {
    const localReady = await router.localAvailable();
    const services = createAxServices(config);
    const registry = loadOptimizerRegistry(config, cwd);
    const providerRoles = resolveAllModelRoleConfigs(config).map((roleConfig) => {
      const resolvedPolicy = resolveLoadedOptimizerPolicy(registry, {
        modelRole: roleConfig.modelRole,
        modelName: roleConfig.model,
      });
      return {
        modelRole: roleConfig.modelRole,
        providerConfigRole: roleConfig.providerConfigRole,
        provider: roleConfig.provider,
        model: roleConfig.model,
        baseUrl: roleConfig.baseUrl,
        endpointKind: roleConfig.endpointKind,
        fallbackModelRole: roleConfig.fallbackModelRole ?? null,
        modelServerId: roleConfig.modelServerId,
        modelServerProfileId: roleConfig.modelServerProfileId,
        providerDiscoverySource: roleConfig.providerDiscoverySource,
        contextWindowTokens: roleConfig.contextWindowTokens,
        contextWindowSource: roleConfig.contextWindowSource,
        maxOutputTokens: roleConfig.maxOutputTokens,
        optimizerSource: resolvedPolicy.source,
        modelProfileId: resolvedPolicy.modelProfileId,
        codebaseProfileId: resolvedPolicy.codebaseProfileId,
        policyId: resolvedPolicy.policyId,
        canonicalToolVersion: resolvedPolicy.canonicalToolVersion,
        renderedToolVersion: resolvedPolicy.renderedToolVersion,
        resultStyleVersion: resolvedPolicy.resultStyleVersion,
        verificationPolicyVersion: resolvedPolicy.verificationPolicyVersion,
        editStrategyVersion: resolvedPolicy.editStrategyVersion,
        renderedEditContractVersion: resolvedPolicy.renderedEditContractVersion,
      };
    });
    console.log(
      JSON.stringify(
        {
          name: "BleedingAgent",
          cli: "bag",
          ax: "@ax-llm/ax",
          masterAvailable: router.masterAvailable,
          localEndpointReady: localReady,
          masterProvider: config.master.provider,
          masterBaseUrl: config.master.baseUrl,
          masterEndpointKind: config.master.endpointKind,
          localBaseUrl: config.local.baseUrl,
          localProvider: config.local.provider,
          localEndpointKind: config.local.endpointKind,
          localModel: config.local.model,
          masterModel: config.master.model,
          providerRoles,
          optimizerRegistryRoot: registry.root,
          optimizerRegistryErrors: registry.errors.length,
          optimizerInvalidRecords: registry.invalidRecords.length,
          activeOptimizerPointer: registry.activePointer ?? null,
          telemetryEnabled: config.telemetry.enabled,
          telemetryJsonl: config.telemetry.jsonl,
          telemetryMetrics: config.telemetry.metrics,
          telemetrySpans: config.telemetry.spans,
          executorConcurrency: config.policy.executorConcurrency,
          maxExecutorConcurrency: config.policy.maxExecutorConcurrency,
          selfEvalThreshold: config.policy.selfEvalThreshold,
          axMasterConfigured: services.master != null,
          axLocalConfigured: services.local != null,
        },
        null,
        2,
      ),
    );
    return;
  }

  if (command === "metrics") {
    const store = readMetricsStore(config, cwd);
    console.log(rest.includes("--json") ? JSON.stringify(store, null, 2) : summarizeMetricsStore(store));
    return;
  }

  if (command === "acp-settings") {
    console.log(readAcpSettingsSnippet(cwd, rest[0] === "zed" ? "zed" : "generic"));
    return;
  }

  if (command === "optimize") {
    const report = optimizePolicy({ config, cwd });
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  if (command === "optimizer") {
    const [optimizerCommand, ...optimizerArgs] = rest;
    const action = optimizerAction(optimizerCommand);
    if (action === undefined) {
      throw new Error(`missing or unknown optimizer command; expected promotion-preview, approve, promote, monitor, or rollback\n\n${usage()}`);
    }
    const graphId = argValue(optimizerArgs, "--graph-id") ?? process.env.BAG_EVIDENCE_GRAPH_ID;
    const selectionHash = argValue(optimizerArgs, "--selection-hash");
    if (graphId === undefined || selectionHash === undefined) {
      throw new Error("optimizer promotion workflow requires --graph-id and --selection-hash");
    }
    const result = runPromotionWorkflowCommand({
      cwd,
      action,
      graphId,
      selectionHash,
      candidatePatchId: argValue(optimizerArgs, "--candidate-id"),
      promotionDecisionId: argValue(optimizerArgs, "--promotion-decision-id"),
    });
    console.log(JSON.stringify(result, null, 2));
    process.exitCode = result.exitCode;
    return;
  }

  if (command === "self-optimize") {
    const result = generateSelfOptimization({ config, cwd });
    console.log(`candidate: ${result.candidate.id}`);
    console.log(`json: ${result.jsonPath}`);
    console.log(`markdown: ${result.markdownPath}`);
    console.log(`summary: ${result.candidate.summary}`);
    if (rest.includes("--apply")) {
      const applied = applySelfOptimization({ config, cwd, candidateId: result.candidate.id });
      console.log(`applied: ${applied.candidate.id}`);
      console.log(`configWritten: ${applied.configWritten} ${applied.configPath}`);
      console.log(`guidanceWritten: ${applied.guidanceWritten} ${applied.guidancePath}`);
    }
    return;
  }

  if (command === "apply-optimization") {
    const candidateId = rest[0];
    const applied = applySelfOptimization({ config, cwd, ...(candidateId == null ? {} : { candidateId }) });
    console.log(`applied: ${applied.candidate.id}`);
    console.log(`configWritten: ${applied.configWritten} ${applied.configPath}`);
    console.log(`guidanceWritten: ${applied.guidanceWritten} ${applied.guidancePath}`);
    return;
  }

  if (command === "evidence") {
    const [evidenceCommandRaw, ...evidenceArgs] = rest;
    if (evidenceCommandRaw == null || !isEvidenceCommandName(evidenceCommandRaw)) {
      throw new Error(`missing or unknown evidence command; expected one of: ${EVIDENCE_COMMANDS.join(", ")}\n\n${usage()}`);
    }
    const graphId = argValue(evidenceArgs, "--graph-id") ?? process.env.BAG_EVIDENCE_GRAPH_ID;
    const result = runEvidenceCommand(evidenceCommandRaw, {
      cwd,
      dryRun: !evidenceArgs.includes("--write"),
      ...(graphId === undefined ? {} : { graphId }),
    });
    console.log(JSON.stringify(result, null, 2));
    process.exitCode = result.exit.code;
    return;
  }

  const task = taskFromArgs(rest);
  if (task === "") {
    throw new Error(`missing task\n\n${usage()}`);
  }

  const knowledge = loadKnowledge(cwd);
  const scoutFindings = await runLocalContextScouts({ router, config, task, cwd });
  const repoContext = buildRepoContext({ cwd, config, task, findings: scoutFindings });

  if (command === "interview") {
    console.log(JSON.stringify(await runInterview({ router, task, repoContext, knowledge }), null, 2));
    return;
  }

  if (command === "prd") {
    const interview = await runInterview({ router, task, repoContext, knowledge });
    const prd = await generatePrd({ router, task, interview, repoContext, knowledge });
    console.log(renderPrdMarkdown(prd));
    return;
  }

  if (command === "dag") {
    const interview = await runInterview({ router, task, repoContext, knowledge });
    const prd = await generatePrd({ router, task, interview, repoContext, knowledge });
    const dag = await generateDag({ router, prd, repoContext });
    console.log(renderDagMarkdown(dag));
    return;
  }

  if (command === "ax-smoke") {
    const bagAgent = createAxBleedingAgent(config);
    const services = createAxServices(config);
    const result = await bagAgent.forward(services.master ?? services.local, {
      task,
      repoContext,
      knowledge,
    });
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (command === "run") {
    const result = await runPlanningPipeline({ config, task, command, cwd });
    console.log(`run: ${result.runId}`);
    console.log(`root: ${result.root}`);
    console.log(`selfEval: ${result.selfEvaluation.score} passed=${result.selfEvaluation.passed}`);
    console.log(`recommendedExecutorConcurrency: ${result.optimization.recommendedExecutorConcurrency}`);
    printArtifacts(result.artifacts);
    return;
  }

  throw new Error(`unknown command: ${command}\n\n${usage()}`);
};

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
