import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { BagConfig, LlmCallMetric, StepMetric, ToolCallMetric } from "./types";

export type PersistedMetrics =
  | StepMetric[]
  | {
      steps?: StepMetric[];
      llmCalls?: LlmCallMetric[];
      toolCalls?: ToolCallMetric[];
    };

export type MetricsStore = Record<string, PersistedMetrics>;

export const normalizeRunMetrics = (entry: PersistedMetrics): {
  steps: StepMetric[];
  llmCalls: LlmCallMetric[];
  toolCalls: ToolCallMetric[];
} => {
  if (Array.isArray(entry)) {
    return { steps: entry, llmCalls: [], toolCalls: [] };
  }
  return {
    steps: Array.isArray(entry.steps) ? entry.steps : [],
    llmCalls: Array.isArray(entry.llmCalls) ? entry.llmCalls : [],
    toolCalls: Array.isArray(entry.toolCalls) ? entry.toolCalls : [],
  };
};

export const readMetricsStore = (config: BagConfig, cwd = process.cwd()): MetricsStore => {
  const path = resolve(cwd, config.telemetry.metrics);
  if (!existsSync(path)) {
    return {};
  }
  const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
  return parsed != null && typeof parsed === "object" && !Array.isArray(parsed)
    ? (parsed as MetricsStore)
    : {};
};

export const summarizeMetricsStore = (store: MetricsStore): string => {
  const runs = Object.entries(store).map(([runId, entry]) => ({
    runId,
    ...normalizeRunMetrics(entry),
  }));
  const steps = runs.flatMap((run) => run.steps);
  const llmCalls = runs.flatMap((run) => run.llmCalls);
  const toolCalls = runs.flatMap((run) => run.toolCalls);
  const failedSteps = steps.filter((step) => !step.ok);
  const failedLlmCalls = llmCalls.filter((call) => !call.ok);
  const failedToolCalls = toolCalls.filter((call) => !call.ok);
  const totalPromptTokens = llmCalls.reduce((sum, call) => sum + (call.promptTokens ?? 0), 0);
  const totalCompletionTokens = llmCalls.reduce((sum, call) => sum + (call.completionTokens ?? 0), 0);
  const totalTokens = llmCalls.reduce((sum, call) => sum + (call.totalTokens ?? 0), 0);
  const avgStepMs =
    steps.length === 0 ? 0 : Math.round(steps.reduce((sum, step) => sum + step.durationMs, 0) / steps.length);
  const avgLlmMs =
    llmCalls.length === 0
      ? 0
      : Math.round(llmCalls.reduce((sum, call) => sum + call.durationMs, 0) / llmCalls.length);
  const avgToolMs =
    toolCalls.length === 0
      ? 0
      : Math.round(toolCalls.reduce((sum, call) => sum + call.durationMs, 0) / toolCalls.length);
  const latest = runs.at(-1);
  const topToolFailures = [...new Set(failedToolCalls.map((call) => call.toolName))]
    .map((toolName) => ({
      toolName,
      failed: failedToolCalls.filter((call) => call.toolName === toolName).length,
      total: toolCalls.filter((call) => call.toolName === toolName).length,
    }))
    .sort((left, right) => right.failed - left.failed)
    .slice(0, 5)
    .map((row) => `${row.toolName}=${row.failed}/${row.total}`)
    .join(" ");
  const topLlmFailures = [...new Set(failedLlmCalls.map((call) => `${call.role}:${call.model}`))]
    .map((model) => ({
      model,
      failed: failedLlmCalls.filter((call) => `${call.role}:${call.model}` === model).length,
      total: llmCalls.filter((call) => `${call.role}:${call.model}` === model).length,
    }))
    .sort((left, right) => right.failed - left.failed)
    .slice(0, 5)
    .map((row) => `${row.model}=${row.failed}/${row.total}`)
    .join(" ");

  return [
    `runs: ${runs.length}`,
    `steps: ${steps.length} failed=${failedSteps.length} avgMs=${avgStepMs}`,
    `llmCalls: ${llmCalls.length} failed=${failedLlmCalls.length} avgMs=${avgLlmMs}`,
    `toolCalls: ${toolCalls.length} failed=${failedToolCalls.length} avgMs=${avgToolMs}`,
    `llmFailures: ${topLlmFailures || "none"}`,
    `toolFailures: ${topToolFailures || "none"}`,
    `tokens: prompt=${totalPromptTokens} completion=${totalCompletionTokens} total=${totalTokens}`,
    latest == null
      ? "latest: none"
      : `latest: ${latest.runId} steps=${latest.steps.length} llmCalls=${latest.llmCalls.length} toolCalls=${latest.toolCalls.length}`,
  ].join("\n");
};
