import { resolve } from "node:path";
import {
  generateEvidenceIndex,
  generateCanonicalEpoch,
  generateOptimizerGates,
  generateReleaseProof,
  generateScorecards,
  validateEvidence,
  type EvidenceGeneration,
} from "./generators/local-artifact-generators";
import { hasBlockingFailure, type EvidenceArtifactRef, type EvidenceCheck, type EvidenceWriteIntent } from "./generators/artifacts";

export type EvidenceCommandName = "index" | "scorecards" | "optimizer-gates" | "epoch" | "validate" | "release-proof";

export type EvidenceExitIntent =
  | "success"
  | "missing_artifact"
  | "invalid_artifact"
  | "validation_failed";

export type EvidenceCommandOptions = {
  cwd?: string;
  dryRun?: boolean;
  graphId?: string;
};

export type EvidenceCommandResult<TPayload = unknown> = {
  command: EvidenceCommandName;
  cwd: string;
  dryRun: boolean;
  ok: boolean;
  exit: {
    intent: EvidenceExitIntent;
    code: number;
  };
  summary: string;
  payload?: TPayload;
  artifacts: EvidenceArtifactRef[];
  checks: EvidenceCheck[];
  writes: EvidenceWriteIntent[];
  handoffTodos: string[];
};

export type EvidenceCommandPayloads = {
  index: ReturnType<typeof generateEvidenceIndex>["payload"];
  scorecards: ReturnType<typeof generateScorecards>["payload"];
  "optimizer-gates": ReturnType<typeof generateOptimizerGates>["payload"];
  epoch: ReturnType<typeof generateCanonicalEpoch>["payload"];
  validate: ReturnType<typeof validateEvidence>["payload"];
  "release-proof": ReturnType<typeof generateReleaseProof>["payload"];
};

export const EVIDENCE_COMMANDS: readonly EvidenceCommandName[] = [
  "index",
  "scorecards",
  "optimizer-gates",
  "epoch",
  "validate",
  "release-proof",
];

export const runEvidenceCommand = <TCommand extends EvidenceCommandName>(
  command: TCommand,
  options: EvidenceCommandOptions = {},
): EvidenceCommandResult<NonNullable<EvidenceCommandPayloads[TCommand]>> => {
  const cwd = resolve(options.cwd ?? process.cwd());
  const dryRun = options.dryRun ?? true;
  const generationOptions = {
    cwd,
    dryRun,
    ...(options.graphId === undefined ? {} : { graphId: options.graphId }),
  };
  const generation = routeEvidenceCommand(command, generationOptions);
  return commandResult(command, cwd, dryRun, generation) as EvidenceCommandResult<NonNullable<EvidenceCommandPayloads[TCommand]>>;
};

export const isEvidenceCommandName = (value: string): value is EvidenceCommandName =>
  (EVIDENCE_COMMANDS as readonly string[]).includes(value);

const routeEvidenceCommand = (
  command: EvidenceCommandName,
  options: { cwd: string; dryRun: boolean },
): EvidenceGeneration<unknown> => {
  switch (command) {
    case "index":
      return generateEvidenceIndex(options);
    case "scorecards":
      return generateScorecards(options);
    case "optimizer-gates":
      return generateOptimizerGates(options);
    case "epoch":
      return generateCanonicalEpoch(options);
    case "validate":
      return validateEvidence(options);
    case "release-proof":
      return generateReleaseProof(options);
  }
};

const commandResult = (
  command: EvidenceCommandName,
  cwd: string,
  dryRun: boolean,
  generation: EvidenceGeneration<unknown>,
): EvidenceCommandResult => {
  const exit = exitFor(generation);
  return {
    command,
    cwd,
    dryRun,
    ok: exit.code === 0,
    exit,
    summary: generation.summary,
    ...(generation.payload === undefined ? {} : { payload: generation.payload }),
    artifacts: generation.artifacts,
    checks: generation.checks,
    writes: generation.writes,
    handoffTodos: handoffTodosFor(command),
  };
};

const exitFor = (generation: EvidenceGeneration<unknown>): EvidenceCommandResult["exit"] => {
  if (!hasBlockingFailure(generation.checks)) {
    return { intent: "success", code: 0 };
  }

  if (generation.artifacts.some((artifact) => artifact.required && !artifact.exists)) {
    return { intent: "missing_artifact", code: 66 };
  }

  if (generation.checks.some((check) => !check.passed && check.message.includes("Invalid"))) {
    return { intent: "invalid_artifact", code: 65 };
  }

  return { intent: "validation_failed", code: 1 };
};

const handoffTodosFor = (command: EvidenceCommandName): string[] => {
  switch (command) {
    case "index":
      return [
        "Replace local JSONL wrapping with a deterministic index rebuild once source inventory ownership is assigned.",
      ];
    case "scorecards":
      return [
        "Move one-off scorecard mining into deterministic scorecard builders with frozen inputs and explicit split policy.",
      ];
    case "optimizer-gates":
      return [
        "Wire optimizer gate evaluation into runtime scheduler and ACP maintenance status without allowing auto-promotion by default.",
      ];
    case "epoch":
      return [
        "Use the canonical readiness index as the first handoff artifact for downstream blocker-closure lanes.",
      ];
    case "validate":
      return [
        "Use this aggregate command as the later top-level CLI gate before release-proof regeneration.",
      ];
    case "release-proof":
      return [
        "Rebuild release proof from graph metadata and validated command payloads instead of wrapping the existing JSON artifact.",
      ];
  }
};
