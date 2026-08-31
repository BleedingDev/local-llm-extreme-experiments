import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { BagConfig, LlmCallMetric, SelfEvaluation, StepMetric, ToolCallMetric } from "./types";

export type OptimizationReport = {
  evaluatedRuns: number;
  evaluatedMetrics: number;
  passRate: number;
  averageStepDurationMs: number;
  recommendedExecutorConcurrency: number;
  recommendedInteractiveConcurrency: number;
  notes: string[];
};

const readRunMetrics = (
  runsDir: string,
): { steps: StepMetric[]; llmCalls: LlmCallMetric[]; toolCalls: ToolCallMetric[] } => {
  if (!existsSync(runsDir)) {
    return { steps: [], llmCalls: [], toolCalls: [] };
  }

  const rows = readdirSync(runsDir).map((runId) => {
    const manifestPath = join(runsDir, runId, "manifest.json");
    if (!existsSync(manifestPath)) {
      return { steps: [], llmCalls: [], toolCalls: [] };
    }
    try {
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
        metrics?: StepMetric[];
        llmMetrics?: LlmCallMetric[];
        toolMetrics?: ToolCallMetric[];
      };
      return {
        steps: Array.isArray(manifest.metrics) ? manifest.metrics : [],
        llmCalls: Array.isArray(manifest.llmMetrics) ? manifest.llmMetrics : [],
        toolCalls: Array.isArray(manifest.toolMetrics) ? manifest.toolMetrics : [],
      };
    } catch {
      return { steps: [], llmCalls: [], toolCalls: [] };
    }
  });

  return {
    steps: rows.flatMap((row) => row.steps),
    llmCalls: rows.flatMap((row) => row.llmCalls),
    toolCalls: rows.flatMap((row) => row.toolCalls),
  };
};

export const optimizePolicy = (input: {
  config: BagConfig;
  cwd?: string;
  latestSelfEvaluation?: SelfEvaluation;
}): OptimizationReport => {
  const cwd = input.cwd ?? process.cwd();
  const runsDir = resolve(cwd, input.config.artifactDir, "runs");
  const runs = existsSync(runsDir) ? readdirSync(runsDir).filter((name) => !name.startsWith(".")) : [];
  const { steps: metrics, llmCalls, toolCalls } = readRunMetrics(runsDir);
  const failed = metrics.filter((metric) => !metric.ok);
  const failedLlmCalls = llmCalls.filter((metric) => !metric.ok);
  const failedToolCalls = toolCalls.filter((metric) => !metric.ok);
  const avgDuration =
    metrics.length === 0 ? 0 : metrics.reduce((sum, metric) => sum + metric.durationMs, 0) / metrics.length;
  const stepPassRate = metrics.length === 0 ? 1 : (metrics.length - failed.length) / metrics.length;
  const llmPassRate = llmCalls.length === 0 ? 1 : (llmCalls.length - failedLlmCalls.length) / llmCalls.length;
  const toolPassRate = toolCalls.length === 0 ? 1 : (toolCalls.length - failedToolCalls.length) / toolCalls.length;
  const passRate = Math.min(stepPassRate, llmPassRate, toolPassRate);
  const latestScore = input.latestSelfEvaluation?.score ?? 1;
  const shouldBackOff = passRate < 0.9 || latestScore < input.config.policy.selfEvalThreshold;
  const recommendedExecutorConcurrency = shouldBackOff
    ? Math.max(8, Math.min(12, input.config.policy.executorConcurrency))
    : Math.min(20, input.config.policy.maxExecutorConcurrency);

  return {
    evaluatedRuns: runs.length,
    evaluatedMetrics: metrics.length,
    passRate: Math.round(passRate * 100) / 100,
    averageStepDurationMs: Math.round(avgDuration),
    recommendedExecutorConcurrency,
    recommendedInteractiveConcurrency: Math.min(12, recommendedExecutorConcurrency),
    notes: [
      "Current benchmark evidence favors 16-20 local executor calls for coding throughput.",
      "Use 24 only for bulk analysis with long latency tolerance.",
      "Back off automatically when self-eval fails or recent step pass-rate drops below 90%.",
      "Revise tool descriptions, schemas, retries, or timeouts when tool-call pass-rate drops below 90%.",
      "Keep GPT master/critic on planning and final judgement until local model reliability improves.",
    ],
  };
};

export const codifyKnowledge = (input: {
  config: BagConfig;
  runId: string;
  task: string;
  report: OptimizationReport;
  selfEvaluation: SelfEvaluation;
  cwd?: string;
}): string => {
  const cwd = input.cwd ?? process.cwd();
  const path = resolve(cwd, input.config.artifactDir, "knowledge.md");
  mkdirSync(join(cwd, input.config.artifactDir), { recursive: true });
  const existing = existsSync(path) ? readFileSync(path, "utf8") : "# BleedingAgent Knowledge\n";
  const entry = [
    "",
    `## ${new Date().toISOString()} ${input.runId}`,
    `Task: ${input.task}`,
    `Self-eval score: ${input.selfEvaluation.score}`,
    `Self-eval passed: ${input.selfEvaluation.passed}`,
    `Recommended executor concurrency: ${input.report.recommendedExecutorConcurrency}`,
    `Recommended interactive concurrency: ${input.report.recommendedInteractiveConcurrency}`,
    ...input.report.notes.map((note) => `- ${note}`),
    ...input.selfEvaluation.improvementActions.map((action) => `- Improvement: ${action}`),
    "",
  ].join("\n");
  writeFileSync(path, `${existing.trimEnd()}\n${entry}`);
  return path;
};
