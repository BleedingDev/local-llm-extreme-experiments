#!/usr/bin/env -S node --loader=tsx
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";
import { pathToFileURL } from "node:url";
import {
  RealAcpCorpusRunManifestSchema,
  createRealAcpTraceMinedScorecards,
  renderRealAcpTraceMinedScorecardsMarkdown,
} from "../src/replay";

type Options = {
  manifests: string[];
  out?: string;
  scorecardId?: string;
};

const parseArgs = (argv: readonly string[]): Options => {
  const options: Options = { manifests: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--manifest") {
      options.manifests.push(requiredValue(argv, ++index, arg));
    } else if (arg === "--out") {
      options.out = requiredValue(argv, ++index, arg);
    } else if (arg === "--scorecard-id") {
      options.scorecardId = requiredValue(argv, ++index, arg);
    } else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${String(arg)}`);
    }
  }
  if (options.manifests.length === 0) {
    throw new Error("--manifest is required at least once");
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
  process.stdout.write(`usage: tsx scripts/report_real_acp_trace_scorecards.ts --manifest RUN.manifest.json [--manifest RUN2.manifest.json] [--out report.md]\n`);
};

const main = async (): Promise<void> => {
  const options = parseArgs(process.argv.slice(2));
  const manifests = await Promise.all(options.manifests.map(async (manifestPath) =>
    RealAcpCorpusRunManifestSchema.parse(JSON.parse(await readFile(manifestPath, "utf8")))));
  const scorecards = createRealAcpTraceMinedScorecards({
    manifests,
    ...(options.scorecardId === undefined ? {} : { scorecardId: options.scorecardId }),
  });
  const markdown = renderRealAcpTraceMinedScorecardsMarkdown(scorecards);
  if (options.out === undefined) {
    process.stdout.write(markdown);
    return;
  }
  await writeFile(options.out, markdown, "utf8");
  process.stdout.write(`${JSON.stringify({
    status: "complete",
    scorecardId: scorecards.scorecardId,
    out: options.out,
    taskCount: scorecards.taskCount,
    toolCalibrationCount: scorecards.toolCalibration.length,
    argumentPatternCount: scorecards.argumentPatterns.length,
    transitionCount: scorecards.toolTransitions.length,
    editFamilyMatrixCount: scorecards.editFamilyMatrix.length,
  }, null, 2)}\n`);
};

const directRun = process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;
if (directRun) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  });
}
