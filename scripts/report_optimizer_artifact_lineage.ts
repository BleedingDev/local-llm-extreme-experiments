#!/usr/bin/env -S node --loader=tsx
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";
import { pathToFileURL } from "node:url";
import {
  OptimizerArtifactLineageManifestSchema,
  assessOptimizerArtifactLineage,
} from "../src/optimizer/artifact-lineage";

type Options = {
  manifest?: string;
  out?: string;
  decisionOut?: string;
};

const parseArgs = (argv: readonly string[]): Options => {
  const options: Options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--manifest") {
      options.manifest = requiredValue(argv, ++index, arg);
    } else if (arg === "--out") {
      options.out = requiredValue(argv, ++index, arg);
    } else if (arg === "--decision-out") {
      options.decisionOut = requiredValue(argv, ++index, arg);
    } else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${String(arg)}`);
    }
  }
  if (options.manifest === undefined) {
    throw new Error("--manifest is required");
  }
  return options;
};

const requiredValue = (argv: readonly string[], index: number, flag: string): string => {
  const value = argv[index];
  if (value == null || value.startsWith("--")) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

const printUsage = (): void => {
  process.stdout.write(`usage: tsx scripts/report_optimizer_artifact_lineage.ts --manifest lineage.json [--out report.md] [--decision-out decision.json]\n`);
};

const main = async (): Promise<void> => {
  const options = parseArgs(process.argv.slice(2));
  const manifest = OptimizerArtifactLineageManifestSchema.parse(JSON.parse(await readFile(options.manifest!, "utf8")));
  const decision = assessOptimizerArtifactLineage(manifest);
  if (options.out !== undefined) {
    await writeFile(options.out, decision.report, "utf8");
  } else {
    process.stdout.write(decision.report);
  }
  if (options.decisionOut !== undefined) {
    await writeFile(options.decisionOut, `${JSON.stringify(decision, null, 2)}\n`, "utf8");
  }
};

const directRun = process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;
if (directRun) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  });
}
