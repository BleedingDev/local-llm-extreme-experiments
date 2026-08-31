#!/usr/bin/env -S node --loader=tsx
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import process from "node:process";

import { loadDatasetEvalRunResults } from "../src/optimizer/dataset-adapter";
import { buildGepaFeedbackBundle } from "../src/optimizer/gepa-feedback";
import {
  createLlmBackedGepaProposer,
  runGepaOptimizer,
  type GepaLlmProposerClient,
} from "../src/optimizer/gepa-runner";
import { createPromptFragmentProposer } from "../src/optimizer/prompt-fragment-proposer";
import { seedPromptRegistryRecords } from "../src/optimizer/seed-prompt-records";
import type {
  CandidatePatch,
  CandidateScope,
  OptimizerRegistryRecord,
} from "../src/optimizer/types";

type Cli = {
  dataset: string;
  output: string;
  maxIterations: number;
  maxTotalCandidates: number;
  useLlm: boolean;
};

const repoRoot = (): string => {
  const cwd = process.cwd();
  const here = resolve(dirname(new URL(import.meta.url).pathname), "..");
  return cwd.endsWith("scripts") ? here : cwd;
};

const parseArgs = (argv: readonly string[]): Cli => {
  const root = repoRoot();
  const out: Cli = {
    dataset: resolve(root, "bench/.bag/optimizer/dataset.jsonl"),
    output: resolve(root, "bench/.bag/optimizer/candidates.json"),
    maxIterations: 3,
    maxTotalCandidates: 12,
    useLlm: process.env.BAG_GEPA_USE_LLM === "1",
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
      case "--max-iterations":
        if (next != null) {
          out.maxIterations = Math.max(1, Number(next) | 0);
          i += 1;
        }
        break;
      case "--max-total-candidates":
        if (next != null) {
          out.maxTotalCandidates = Math.max(1, Number(next) | 0);
          i += 1;
        }
        break;
      case "--use-llm":
        out.useLlm = true;
        break;
      default:
        if (arg?.startsWith("--") === true) {
          process.stderr.write(`unknown flag: ${arg}\n`);
          process.exit(2);
        }
    }
  }
  return out;
};

/**
 * Synchronous client adapter for `GepaLlmProposerClient`. The runner expects a
 * synchronous response, so we cannot await `LlmRouter.chatText` here. Returning
 * an empty response triggers schema validation failure inside
 * `createLlmBackedGepaProposer`, which then falls back to the deterministic
 * proposer with a recorded diagnostic — exactly the desired behaviour for an
 * offline smoke run while still proving the wiring contract.
 */
const buildLlmProposerClient = (): GepaLlmProposerClient => {
  return (_request) => {
    return {
      // Intentional invalid shape to force fallback path.
      candidates: undefined,
      diagnostics: [],
    };
  };
};

const scopeKey = (scope: CandidateScope): string =>
  `${scope.artifactKind}:${scope.artifactId}`;

const recordArtifactId = (record: OptimizerRegistryRecord): string | undefined => {
  switch (record.recordKind) {
    case "model_profile":
      return record.payload.modelProfileId;
    case "codebase_profile":
      return record.payload.codebaseProfileId;
    case "model_codebase_policy":
      return record.payload.policyId;
    case "canonical_tool_spec":
      return record.payload.canonicalToolId;
    case "rendered_tool_contract":
      return record.payload.renderedToolId;
    default:
      return undefined;
  }
};

const buildExpectedBaseHashes = (
  records: readonly OptimizerRegistryRecord[],
): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const record of records) {
    const artifactId = recordArtifactId(record);
    if (artifactId == null) continue;
    if (record.contentHash == null) continue;
    out[artifactId] = record.contentHash;
  }
  return out;
};

const main = (): void => {
  const cli = parseArgs(process.argv.slice(2));
  const cwd = repoRoot();

  const evalRunResults = loadDatasetEvalRunResults(cli.dataset);
  const failureRuns = evalRunResults.filter((run) => run.status !== "passed");
  const records = seedPromptRegistryRecords(cwd);

  const feedbackBundle = buildGepaFeedbackBundle({
    feedbackBundleId: `gepa-feedback.${new Date().toISOString().replace(/[^0-9]/g, "").slice(0, 14)}`,
    evalRunResults: failureRuns,
  });

  const promptFragmentProposer = createPromptFragmentProposer({ records });
  const proposer = cli.useLlm
    ? createLlmBackedGepaProposer({
        client: buildLlmProposerClient(),
        fallbackProposer: promptFragmentProposer,
      })
    : promptFragmentProposer;

  const expectedBaseHashes = buildExpectedBaseHashes(records);
  const state = runGepaOptimizer({
    feedbackBundle,
    records,
    expectedBaseHashes,
    maxIterations: cli.maxIterations,
    maxTotalCandidates: cli.maxTotalCandidates,
    proposer,
  });

  const validationPassed = state.validations.filter((validation) => validation.valid).length;
  const validationTotal = state.validations.length;
  const scopeDistribution = new Map<string, number>();
  for (const candidate of state.candidates) {
    const key = scopeKey(candidate.scope);
    scopeDistribution.set(key, (scopeDistribution.get(key) ?? 0) + 1);
  }

  const summary = {
    schemaVersion: "gepa-candidates-report.v1",
    generatedAt: new Date().toISOString(),
    datasetPath: cli.dataset,
    datasetRecordCount: evalRunResults.length,
    failureRunCount: failureRuns.length,
    feedbackRecordCount: feedbackBundle.records.length,
    seedPromptRecordCount: records.length,
    candidateCount: state.candidates.length,
    validationTotal,
    validationPassed,
    scopeDistribution: Object.fromEntries(scopeDistribution.entries()),
    iterationCount: state.iterationCount,
    diagnosticsCount: state.diagnostics.length,
    candidates: state.candidates.map((candidate: CandidatePatch) => ({
      candidatePatchId: candidate.candidatePatchId,
      scope: candidate.scope,
      rationale: candidate.rationale,
      operationCount: candidate.operations.length,
    })),
  };

  mkdirSync(dirname(cli.output), { recursive: true });
  writeFileSync(cli.output, `${JSON.stringify({ ...summary, state }, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
};

main();
