import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { configPath, defaultConfig } from "./config";
import { normalizeRunMetrics, readMetricsStore } from "./metrics";
import { analyzeHaloSpans, readHaloSpans, renderTraceAnalysisMarkdown, type TraceAnalysisReport } from "./trace-analysis";
import { TraceStore, type DatasetOverview } from "./trace-store";
import type { BagConfig, LlmCallMetric, StepMetric, ToolCallMetric } from "./types";

type Severity = "low" | "medium" | "high";

export type OptimizationFinding = {
  area: "tool" | "llm" | "step" | "policy";
  severity: Severity;
  title: string;
  evidence: string[];
  recommendedAction: string;
};

export type ImprovementProposal = {
  target: "tool-guidance" | "prompt-policy" | "eval-suite" | "runtime-policy";
  title: string;
  rationale: string[];
  patchSketch: string;
  evalGate: string;
};

type SafeConfigPatch = {
  policy?: Partial<BagConfig["policy"]>;
};

export type SelfOptimizationCandidate = {
  id: string;
  createdAt: string;
  sourceRuns: number;
  summary: string;
  confidence: number;
  safeToApply: boolean;
  findings: OptimizationFinding[];
  configPatch: SafeConfigPatch;
  toolGuidance: string[];
  traceAnalysis?: TraceAnalysisReport;
  traceDataset?: DatasetOverview;
  improvementProposals: ImprovementProposal[];
  appliedAt?: string;
};

const percentile = (values: number[], ratio: number): number => {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1));
  return sorted[index] ?? 0;
};

const passRate = <T extends { ok: boolean }>(metrics: T[]): number =>
  metrics.length === 0 ? 1 : (metrics.length - metrics.filter((metric) => !metric.ok).length) / metrics.length;

const groupBy = <T>(items: T[], keyOf: (item: T) => string): Map<string, T[]> => {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const key = keyOf(item);
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }
  return groups;
};

const firstLine = (value: string | undefined): string => (value ?? "unknown").split("\n")[0]?.slice(0, 180) ?? "unknown";

const optimizationId = (): string =>
  `opt-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`;

const optimizationsDir = (config: BagConfig, cwd: string): string =>
  resolve(cwd, config.artifactDir, "optimizations");

const latestCandidatePath = (config: BagConfig, cwd: string): string | undefined => {
  const root = optimizationsDir(config, cwd);
  if (!existsSync(root)) {
    return undefined;
  }
  return readdirSync(root)
    .filter((file) => file.endsWith(".json") && !file.endsWith(".applied.json"))
    .sort()
    .map((file) => join(root, file))
    .at(-1);
};

const collectMetrics = (config: BagConfig, cwd: string) => {
  const runs = Object.entries(readMetricsStore(config, cwd)).map(([runId, entry]) => ({
    runId,
    ...normalizeRunMetrics(entry),
  }));
  return {
    runs,
    steps: runs.flatMap((run) => run.steps),
    llmCalls: runs.flatMap((run) => run.llmCalls),
    toolCalls: runs.flatMap((run) => run.toolCalls),
  };
};

const recommendConcurrency = (config: BagConfig, input: {
  steps: StepMetric[];
  llmCalls: LlmCallMetric[];
  toolCalls: ToolCallMetric[];
}): number => {
  const reliability = Math.min(passRate(input.steps), passRate(input.llmCalls), passRate(input.toolCalls));
  const current = config.policy.executorConcurrency;
  if (reliability < 0.8) {
    return Math.max(4, Math.min(10, current - 4));
  }
  if (reliability < 0.9) {
    return Math.max(8, Math.min(12, current - 2));
  }
  if (input.llmCalls.length + input.toolCalls.length >= 50 && reliability > 0.98) {
    return Math.min(config.policy.maxExecutorConcurrency, current + 2);
  }
  return current;
};

export const generateSelfOptimization = (input: {
  config: BagConfig;
  cwd?: string;
}): {
  candidate: SelfOptimizationCandidate;
  jsonPath: string;
  markdownPath: string;
} => {
  const cwd = input.cwd ?? process.cwd();
  const { runs, steps, llmCalls, toolCalls } = collectMetrics(input.config, cwd);
  const traceAnalysis = analyzeHaloSpans(readHaloSpans(input.config, cwd));
  const traceStore = TraceStore.open(input.config, cwd);
  const traceDataset = traceStore.getOverview();
  const findings: OptimizationFinding[] = [];
  const toolGuidance: string[] = [];
  const improvementProposals: ImprovementProposal[] = [];
  const configPatch: SelfOptimizationCandidate["configPatch"] = {};

  const failedToolCalls = toolCalls.filter((call) => !call.ok);
  const toolRate = passRate(toolCalls);
  if (failedToolCalls.length > 0) {
    for (const [toolName, failures] of groupBy(failedToolCalls, (call) => call.toolName)) {
      const total = toolCalls.filter((call) => call.toolName === toolName).length;
      const errorGroups = [...groupBy(failures, (call) => `${call.errorName ?? "Error"}: ${firstLine(call.error)}`).entries()]
        .sort((left, right) => right[1].length - left[1].length)
        .slice(0, 3);
      const failureRate = total === 0 ? 1 : failures.length / total;
      findings.push({
        area: "tool",
        severity: failureRate >= 0.25 ? "high" : "medium",
        title: `${toolName} failed ${failures.length}/${total} calls`,
        evidence: errorGroups.map(([error, rows]) => `${rows.length}x ${error}`),
        recommendedAction:
          "Tighten the tool description and argument schema; add an explicit pre-call checklist and retry only for transient errors.",
      });
      toolGuidance.push(
        [
          `## ${toolName}`,
          `Observed failure rate: ${Math.round(failureRate * 100)}% (${failures.length}/${total}).`,
          "Before calling this tool, validate required arguments against the schema and avoid inventing optional fields.",
          "If the failure is argument-related, repair the arguments once before retrying. If the failure is deterministic, do not retry blindly.",
          "Observed errors:",
          ...errorGroups.map(([error, rows]) => `- ${rows.length}x ${error}`),
        ].join("\n"),
      );
      improvementProposals.push({
        target: "tool-guidance",
        title: `Harden ${toolName} argument discipline`,
        rationale: [
          `${failures.length}/${total} failed calls`,
          ...errorGroups.map(([error, rows]) => `${rows.length}x ${error}`),
        ],
        patchSketch:
          "Add a tool-specific pre-call checklist: required fields, forbidden invented fields, deterministic-error stop rule, and one argument-repair retry.",
        evalGate:
          "Replay or recreate an eval with the same failing argument shape; pass only if tool failure rate decreases without increasing total retries.",
      });
    }
  }

  const failedLlmCalls = llmCalls.filter((call) => !call.ok);
  if (failedLlmCalls.length > 0) {
    const byRole = [...groupBy(failedLlmCalls, (call) => `${call.role}:${call.model}`).entries()];
    for (const [role, failures] of byRole) {
      findings.push({
        area: "llm",
        severity: failures.length >= 3 ? "high" : "medium",
        title: `${role} failed ${failures.length} calls`,
        evidence: [...new Set(failures.map((call) => firstLine(call.error)))].slice(0, 5),
        recommendedAction:
          "Route fragile planning and final judgement to the master model; lower local executor concurrency if failures cluster under load.",
      });
    }
  }

  const failedSteps = steps.filter((step) => !step.ok);
  if (failedSteps.length > 0) {
    for (const [step, failures] of groupBy(failedSteps, (metric) => metric.step)) {
      findings.push({
        area: "step",
        severity: failures.length >= 2 ? "high" : "medium",
        title: `${step} failed ${failures.length} times`,
        evidence: [...new Set(failures.map((metric) => firstLine(metric.error)))].slice(0, 5),
        recommendedAction:
          "Split this step into a smaller DAG node or add targeted validation before handing work to the next step.",
      });
    }
  }

  const slowTools = [...groupBy(toolCalls, (call) => call.toolName).entries()]
    .map(([toolName, calls]) => ({
      toolName,
      p95: percentile(calls.map((call) => call.durationMs), 0.95),
      count: calls.length,
    }))
    .filter((tool) => tool.count >= 5 && tool.p95 > 15_000)
    .sort((left, right) => right.p95 - left.p95);

  for (const tool of slowTools) {
    findings.push({
      area: "tool",
      severity: tool.p95 > 60_000 ? "high" : "medium",
      title: `${tool.toolName} p95 latency is ${Math.round(tool.p95)}ms`,
      evidence: [`${tool.count} calls observed`, `p95=${Math.round(tool.p95)}ms`],
      recommendedAction:
        "Prefer narrower arguments, add a timeout budget, and consider routing large reads/searches through a cheaper prefilter.",
    });
  }

  for (const cluster of traceAnalysis.failureClusters.slice(0, 6)) {
    const area = cluster.observationKind === "LLM" ? "llm" : cluster.observationKind === "TOOL" ? "tool" : "step";
    findings.push({
      area,
      severity: cluster.count >= 3 ? "high" : "medium",
      title: `${cluster.observationKind} trace failures in ${cluster.name}`,
      evidence: [
        `${cluster.count} error spans across ${cluster.traces.length} trace(s)`,
        ...cluster.messages.slice(0, 5),
        cluster.inputHashes.length > 0 ? `inputHashes=${cluster.inputHashes.join(",")}` : "",
      ].filter(Boolean),
      recommendedAction:
        cluster.observationKind === "TOOL"
          ? "Revise this tool's description/schema and add an eval case using the repeated failing input shape."
          : "Inspect the trace cluster, add a regression eval for this failure mode, and route fragile judgement to the master model.",
    });
    improvementProposals.push({
      target: cluster.observationKind === "TOOL" ? "tool-guidance" : "prompt-policy",
      title: `Fix recurring ${cluster.observationKind} failure cluster: ${cluster.name}`,
      rationale: [
        `${cluster.count} error spans`,
        ...cluster.messages.slice(0, 4),
        cluster.inputHashes.length > 0 ? `inputHashes=${cluster.inputHashes.join(",")}` : "",
      ].filter(Boolean),
      patchSketch:
        cluster.observationKind === "TOOL"
          ? "Add a cluster-specific tool call rule and a synthetic eval fixture covering the repeated input hash."
          : "Add a routing/prompt guard for this step and force master judgement when the same failure signature appears.",
      evalGate:
        "Run the coding eval pack before/after; accept only if this cluster disappears or drops while overall pass rate and latency do not regress.",
    });
  }

  for (const cluster of traceAnalysis.latencyClusters.slice(0, 4)) {
    if (cluster.p95Ms < 10_000) {
      continue;
    }
    findings.push({
      area: cluster.observationKind === "TOOL" ? "tool" : cluster.observationKind === "LLM" ? "llm" : "step",
      severity: cluster.p95Ms >= 60_000 ? "high" : "medium",
      title: `${cluster.observationKind} ${cluster.name} p95 latency is ${cluster.p95Ms}ms`,
      evidence: [`count=${cluster.count}`, `p50=${cluster.p50Ms}ms`, `p95=${cluster.p95Ms}ms`],
      recommendedAction:
        "Reduce payload size, tighten retrieval scope, or split the operation before increasing concurrency.",
    });
    improvementProposals.push({
      target: cluster.observationKind === "TOOL" ? "runtime-policy" : "prompt-policy",
      title: `Reduce ${cluster.name} trace latency`,
      rationale: [`${cluster.observationKind} count=${cluster.count}`, `p95=${cluster.p95Ms}ms`],
      patchSketch:
        "Narrow context payloads, split oversized reads/searches, and add timeout-aware routing before raising executor concurrency.",
      evalGate:
        "Accept only if p95 latency improves on the same task class and tool/LLM failure rate does not increase.",
    });
  }

  const recommendedConcurrency = recommendConcurrency(input.config, { steps, llmCalls, toolCalls });
  if (recommendedConcurrency !== input.config.policy.executorConcurrency) {
    configPatch.policy = {
      executorConcurrency: recommendedConcurrency,
      interactiveConcurrency: Math.min(input.config.policy.interactiveConcurrency, recommendedConcurrency),
    };
    findings.push({
      area: "policy",
      severity: "medium",
      title: `Adjust executor concurrency from ${input.config.policy.executorConcurrency} to ${recommendedConcurrency}`,
      evidence: [
        `stepPassRate=${passRate(steps).toFixed(2)}`,
        `llmPassRate=${passRate(llmCalls).toFixed(2)}`,
        `toolPassRate=${toolRate.toFixed(2)}`,
      ],
      recommendedAction: "Apply the safe config patch and re-run the same benchmark task for an A/B comparison.",
    });
    improvementProposals.push({
      target: "runtime-policy",
      title: `Adjust executor concurrency to ${recommendedConcurrency}`,
      rationale: [
        `stepPassRate=${passRate(steps).toFixed(2)}`,
        `llmPassRate=${passRate(llmCalls).toFixed(2)}`,
        `toolPassRate=${toolRate.toFixed(2)}`,
      ],
      patchSketch: `Set executorConcurrency=${recommendedConcurrency} and cap interactiveConcurrency at the same value.`,
      evalGate: "Run the same benchmark task with old/new config and accept only if aggregate tps or latency improves without pass-rate loss.",
    });
  }

  if (findings.length === 0) {
    findings.push({
      area: "policy",
      severity: "low",
      title: "No actionable degradation detected",
      evidence: [
        `runs=${runs.length}`,
        `steps=${steps.length}`,
        `llmCalls=${llmCalls.length}`,
        `toolCalls=${toolCalls.length}`,
      ],
      recommendedAction: "Keep collecting telemetry before changing prompts, schemas, or concurrency.",
    });
  }

  const candidate: SelfOptimizationCandidate = {
    id: optimizationId(),
    createdAt: new Date().toISOString(),
    sourceRuns: runs.length,
    summary: summarizeCandidate(findings, configPatch, toolGuidance),
    confidence: confidenceFor({ runs: runs.length, steps, llmCalls, toolCalls, findings }),
    safeToApply: true,
    findings,
    configPatch,
    toolGuidance,
    traceAnalysis,
    traceDataset,
    improvementProposals,
  };

  const root = optimizationsDir(input.config, cwd);
  mkdirSync(root, { recursive: true });
  const jsonPath = join(root, `${candidate.id}.json`);
  const markdownPath = join(root, `${candidate.id}.md`);
  writeFileSync(jsonPath, `${JSON.stringify(candidate, null, 2)}\n`);
  writeFileSync(markdownPath, renderSelfOptimizationMarkdown(candidate));
  return { candidate, jsonPath, markdownPath };
};

export const applySelfOptimization = (input: {
  config: BagConfig;
  cwd?: string;
  candidateId?: string;
}): {
  candidate: SelfOptimizationCandidate;
  configWritten: boolean;
  guidanceWritten: boolean;
  candidatePath: string;
  configPath: string;
  guidancePath: string;
  planPath: string;
  planWritten: boolean;
} => {
  const cwd = input.cwd ?? process.cwd();
  const root = optimizationsDir(input.config, cwd);
  const candidatePath =
    input.candidateId == null
      ? latestCandidatePath(input.config, cwd)
      : join(root, `${basename(input.candidateId, ".json")}.json`);
  if (candidatePath == null || !existsSync(candidatePath)) {
    throw new Error("no self-optimization candidate found; run `bag self-optimize` first");
  }

  const candidate = JSON.parse(readFileSync(candidatePath, "utf8")) as SelfOptimizationCandidate;
  if (!candidate.safeToApply) {
    throw new Error(`candidate ${candidate.id} is not marked safeToApply`);
  }

  let configWritten = false;
  const nextConfig = mergeConfig(input.config, candidate.configPatch);
  if (Object.keys(candidate.configPatch).length > 0) {
    const path = configPath(cwd);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${JSON.stringify(nextConfig, null, 2)}\n`);
    configWritten = true;
  }

  const guidancePath = resolve(cwd, input.config.artifactDir, "tool-guidance.md");
  let guidanceWritten = false;
  if (candidate.toolGuidance.length > 0) {
    mkdirSync(dirname(guidancePath), { recursive: true });
    const existing = existsSync(guidancePath) ? readFileSync(guidancePath, "utf8").trimEnd() : "# Tool Guidance\n";
    writeFileSync(
      guidancePath,
      `${existing}\n\n# ${candidate.id} ${candidate.createdAt}\n\n${candidate.toolGuidance.join("\n\n")}\n`,
    );
    guidanceWritten = true;
  }

  const planPath = resolve(cwd, input.config.artifactDir, "self-improvement-plan.md");
  let planWritten = false;
  const proposals = candidate.improvementProposals ?? [];
  if (proposals.length > 0) {
    mkdirSync(dirname(planPath), { recursive: true });
    const existing = existsSync(planPath) ? readFileSync(planPath, "utf8").trimEnd() : "# Self-Improvement Plan\n";
    writeFileSync(
      planPath,
      `${existing}\n\n# ${candidate.id} ${candidate.createdAt}\n\n${renderImprovementProposals(proposals)}\n`,
    );
    planWritten = true;
  }

  const applied: SelfOptimizationCandidate = { ...candidate, appliedAt: new Date().toISOString() };
  writeFileSync(candidatePath, `${JSON.stringify(applied, null, 2)}\n`);
  writeFileSync(join(root, `${candidate.id}.applied.json`), `${JSON.stringify(applied, null, 2)}\n`);

  return {
    candidate: applied,
    configWritten,
    guidanceWritten,
    candidatePath,
    configPath: configPath(cwd),
    guidancePath,
    planPath,
    planWritten,
  };
};

export const renderSelfOptimizationMarkdown = (candidate: SelfOptimizationCandidate): string =>
  [
    `# Self-Optimization ${candidate.id}`,
    "",
    `Created: ${candidate.createdAt}`,
    `Source runs: ${candidate.sourceRuns}`,
    `Confidence: ${candidate.confidence}`,
    `Safe to apply: ${candidate.safeToApply}`,
    "",
    "## Summary",
    "",
    candidate.summary,
    "",
    "## Findings",
    "",
    ...candidate.findings.flatMap((finding) => [
      `### ${finding.title}`,
      "",
      `Area: ${finding.area}`,
      `Severity: ${finding.severity}`,
      "",
      "Evidence:",
      ...finding.evidence.map((row) => `- ${row}`),
      "",
      `Action: ${finding.recommendedAction}`,
      "",
    ]),
    "## Safe Config Patch",
    "",
    "```json",
    JSON.stringify(candidate.configPatch, null, 2),
    "```",
    "",
    candidate.traceAnalysis == null ? "" : renderTraceAnalysisMarkdown(candidate.traceAnalysis),
    "",
    "## Trace Dataset Overview",
    "",
    candidate.traceDataset == null
      ? "No trace dataset found."
      : [
          `Traces: ${candidate.traceDataset.traceCount}`,
          `Spans: ${candidate.traceDataset.spanCount}`,
          `Error traces: ${candidate.traceDataset.errorTraceCount}`,
          `Error spans: ${candidate.traceDataset.errorSpanCount}`,
          `Models: ${candidate.traceDataset.models.join(", ") || "none"}`,
          `Observation kinds: ${candidate.traceDataset.observationKinds.join(", ") || "none"}`,
          `Sample trace ids: ${candidate.traceDataset.sampleTraceIds.join(", ") || "none"}`,
        ].join("\n"),
    "",
    "## Improvement Proposals",
    "",
    (candidate.improvementProposals ?? []).length === 0
      ? "No improvement proposals generated."
      : renderImprovementProposals(candidate.improvementProposals ?? []),
    "",
    "## Tool Guidance",
    "",
    candidate.toolGuidance.length === 0 ? "No tool-specific guidance generated." : candidate.toolGuidance.join("\n\n"),
    "",
  ].join("\n");

const summarizeCandidate = (
  findings: OptimizationFinding[],
  configPatch: SelfOptimizationCandidate["configPatch"],
  toolGuidance: string[],
): string => {
  const high = findings.filter((finding) => finding.severity === "high").length;
  const medium = findings.filter((finding) => finding.severity === "medium").length;
  const patchCount = Object.keys(configPatch).length;
  return [
    `${findings.length} findings (${high} high, ${medium} medium).`,
    patchCount > 0 ? "Includes a safe config patch." : "No config patch needed.",
    toolGuidance.length > 0 ? `Includes guidance for ${toolGuidance.length} tool groups.` : "No tool guidance needed.",
    "Includes eval-gated improvement proposals.",
  ].join(" ");
};

const renderImprovementProposals = (proposals: ImprovementProposal[]): string =>
  proposals
    .map((proposal) =>
      [
        `### ${proposal.title}`,
        "",
        `Target: ${proposal.target}`,
        "",
        "Rationale:",
        ...proposal.rationale.map((row) => `- ${row}`),
        "",
        "Patch sketch:",
        proposal.patchSketch,
        "",
        "Eval gate:",
        proposal.evalGate,
      ].join("\n"),
    )
    .join("\n\n");

const confidenceFor = (input: {
  runs: number;
  steps: StepMetric[];
  llmCalls: LlmCallMetric[];
  toolCalls: ToolCallMetric[];
  findings: OptimizationFinding[];
}): number => {
  const volume = Math.min(1, (input.steps.length + input.llmCalls.length + input.toolCalls.length) / 100);
  const runCoverage = Math.min(1, input.runs / 10);
  const severity = input.findings.some((finding) => finding.severity === "high") ? 0.2 : 0.1;
  return Math.round(Math.min(0.95, 0.35 + volume * 0.35 + runCoverage * 0.15 + severity) * 100) / 100;
};

const mergeConfig = (
  current: BagConfig,
  patch: SelfOptimizationCandidate["configPatch"],
): BagConfig => {
  const base = defaultConfig();
  return {
    ...base,
    ...current,
    ...patch,
    policy: {
      ...base.policy,
      ...current.policy,
      ...patch.policy,
    },
    master: {
      ...base.master,
      ...current.master,
    },
    local: {
      ...base.local,
      ...current.local,
    },
    telemetry: {
      ...base.telemetry,
      ...current.telemetry,
    },
  };
};
