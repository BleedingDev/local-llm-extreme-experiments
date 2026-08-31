#!/usr/bin/env -S node --loader=tsx
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import process from "node:process";

import { loadDatasetEvalRunResults } from "../src/optimizer/dataset-adapter";
import { assessGepaEvidenceReadiness, planGepaDryRunScheduler } from "../src/optimizer/gepa-operations";

type Cli = {
  dataset: string;
  output: string;
  thresholdMinMetricObservations: number;
  thresholdMinRealReplayCases: number;
  thresholdMinVisibleReplayCases: number;
  thresholdMinRepeatedFailureClusters: number;
  thresholdMinEditFailureSignals: number;
  thresholdMinToolFailureSignals: number;
};

const repoRoot = (): string => {
  const cwd = process.cwd();
  const here = resolve(dirname(new URL(import.meta.url).pathname), "..");
  // Prefer cwd if it's a repo root with bench/.bag/optimizer; else use here.
  return cwd.endsWith("scripts") ? here : cwd;
};

const parseArgs = (argv: readonly string[]): Cli => {
  const root = repoRoot();
  const out: Cli = {
    dataset: resolve(root, "bench/.bag/optimizer/dataset.jsonl"),
    output: resolve(root, "bench/.bag/optimizer/gepa-readiness-report.json"),
    // Relax thresholds because we have offline trials only (no replay capture).
    thresholdMinMetricObservations: 30,
    thresholdMinRealReplayCases: 0,
    thresholdMinVisibleReplayCases: 0,
    thresholdMinRepeatedFailureClusters: 0,
    thresholdMinEditFailureSignals: 0,
    thresholdMinToolFailureSignals: 0,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg == null) continue;
    const next = argv[i + 1];
    switch (arg) {
      case "--dataset":
        if (next != null) {
          out.dataset = resolve(root, next);
          i += 1;
        }
        break;
      case "--output":
        if (next != null) {
          out.output = resolve(root, next);
          i += 1;
        }
        break;
      case "--threshold-min-metric-observations":
        if (next != null) {
          out.thresholdMinMetricObservations = Math.max(0, Number(next) | 0);
          i += 1;
        }
        break;
      case "--threshold-min-real-replay-cases":
        if (next != null) {
          out.thresholdMinRealReplayCases = Math.max(0, Number(next) | 0);
          i += 1;
        }
        break;
      case "--threshold-min-visible-replay-cases":
        if (next != null) {
          out.thresholdMinVisibleReplayCases = Math.max(0, Number(next) | 0);
          i += 1;
        }
        break;
      case "--threshold-min-repeated-failure-clusters":
        if (next != null) {
          out.thresholdMinRepeatedFailureClusters = Math.max(0, Number(next) | 0);
          i += 1;
        }
        break;
      case "--threshold-min-edit-failure-signals":
        if (next != null) {
          out.thresholdMinEditFailureSignals = Math.max(0, Number(next) | 0);
          i += 1;
        }
        break;
      case "--threshold-min-tool-failure-signals":
        if (next != null) {
          out.thresholdMinToolFailureSignals = Math.max(0, Number(next) | 0);
          i += 1;
        }
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
        break;
      default:
        if (arg?.startsWith("--") === true) {
          process.stderr.write(`unknown flag: ${arg}\n`);
          printHelp();
          process.exit(2);
        }
    }
  }
  return out;
};

const printHelp = (): void => {
  process.stdout.write([
    "Usage: tsx scripts/run_gepa_scheduler.ts [flags]",
    "",
    "Flags:",
    "  --dataset <path>                          path to dataset.jsonl",
    "  --output <path>                           where to write readiness JSON",
    "  --threshold-min-metric-observations N     default 30",
    "  --threshold-min-real-replay-cases N       default 0",
    "  --threshold-min-visible-replay-cases N    default 0",
    "  --threshold-min-repeated-failure-clusters N default 0",
    "  --threshold-min-edit-failure-signals N    default 0",
    "  --threshold-min-tool-failure-signals N    default 0",
    "",
  ].join("\n"));
};

const main = (): void => {
  const cli = parseArgs(process.argv.slice(2));
  const evalRunResults = loadDatasetEvalRunResults(cli.dataset);
  const failureSignals = evalRunResults.filter((run) => run.status !== "passed").length;
  const readiness = assessGepaEvidenceReadiness({
    evalRunResults,
    metricObservationCount: evalRunResults.length,
    thresholds: {
      minMetricObservationCount: cli.thresholdMinMetricObservations,
      minRealReplayCases: cli.thresholdMinRealReplayCases,
      minVisibleReplayCases: cli.thresholdMinVisibleReplayCases,
      minRepeatedFailureClusters: cli.thresholdMinRepeatedFailureClusters,
      minEditFailureSignals: cli.thresholdMinEditFailureSignals,
      minToolFailureSignals: cli.thresholdMinToolFailureSignals,
    },
  });

  const report = {
    schemaVersion: "gepa-readiness-report.v1",
    generatedAt: new Date().toISOString(),
    datasetPath: cli.dataset,
    datasetRecordCount: evalRunResults.length,
    failureRunCount: failureSignals,
    readiness,
    scheduler: planGepaDryRunScheduler({ readiness }),
  };

  mkdirSync(dirname(cli.output), { recursive: true });
  writeFileSync(cli.output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
};

main();
