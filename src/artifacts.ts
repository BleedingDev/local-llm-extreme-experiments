import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type {
  BagConfig,
  LlmCallMetric,
  RunManifest,
  SelfEvaluation,
  StepMetric,
  ToolCallMetric,
} from "./types";

export const createRunId = (): string =>
  new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");

export const createArtifactWriter = (config: BagConfig, runId: string, cwd = process.cwd()) => {
  const root = resolve(cwd, config.artifactDir, "runs", runId);
  mkdirSync(root, { recursive: true });

  const write = (relativePath: string, value: string): string => {
    const path = join(root, relativePath);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, value);
    return path;
  };

  const writeJson = (relativePath: string, value: unknown): string =>
    write(relativePath, `${JSON.stringify(value, null, 2)}\n`);

  return { root, write, writeJson };
};

export const writeManifest = (input: {
  config: BagConfig;
  command: string;
  task: string;
  runId: string;
  artifacts: Record<string, string>;
  metrics: StepMetric[];
  llmMetrics: LlmCallMetric[];
  toolMetrics: ToolCallMetric[];
  selfEvaluation: SelfEvaluation;
  writeJson: (relativePath: string, value: unknown) => string;
}): string => {
  const manifest: RunManifest = {
    runId: input.runId,
    createdAt: new Date().toISOString(),
    command: input.command,
    task: input.task,
    config: input.config,
    artifacts: input.artifacts,
    metrics: input.metrics,
    llmMetrics: input.llmMetrics,
    toolMetrics: input.toolMetrics,
    selfEvaluation: input.selfEvaluation,
  };
  return input.writeJson("manifest.json", manifest);
};
