#!/usr/bin/env -S node --loader=tsx
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { classifyCodingProgress } from "../src/acp/coding-progress-diagnostics";
import type { CodingEditResult, CodingPatch } from "../src/acp/coding-types";
import type { TerminalCommandResult } from "../src/acp/terminal";
import type { JsonValue } from "../src/optimizer/types";
import {
  RealAcpCorpusRunPurposeSchema,
  RealAcpExecutionModeSchema,
  RealAcpRunMetadataSchema,
  buildRealAcpReplayCorpusIndex,
  createRealAcpConsumerExecutor,
  createRealAcpHeadlessExecutor,
  createRealAcpReplayExportManifest,
  createRealAcpStabilityScorecard,
  createSimulatedRealAcpExecutor,
  resolveRealAcpConsumerReadiness,
  renderRealAcpStabilityScorecardMarkdown,
  runRealAcpCorpus,
  selectRealAcpCorpusTasks,
  serializeRealAcpReplayCorpusIndexJsonl,
  type RealAcpConsumerProtocolRunner,
  type RealAcpConsumerReadiness,
  type RealAcpConsumerReadinessProvider,
  type RealAcpCorpusRunManifest,
  type RealAcpCorpusRunPurpose,
  type RealAcpExecutionMode,
  type RealAcpHeadlessRunnerInput,
  type RealAcpHeadlessRunnerOutput,
  type RealAcpReplayExportManifest,
  type RealAcpRunMetadata,
  type RealAcpStabilityScorecard,
} from "../src/replay";

export const REAL_ACP_CORPUS_OUTPUT_ROOT = join(".bag", "replay-corpus", "real-acp-runs");

export type RunRealAcpCorpusCliOptions = {
  runId: string;
  mode: RealAcpExecutionMode;
  purpose: RealAcpCorpusRunPurpose;
  includeHoldout: boolean;
  metadataPath?: string;
  outputDir: string;
  workspaceBaseDir: string;
  currentRepoPath: string;
  taskIds: string[];
  planOnly: boolean;
  consumer: "zed" | "glass" | "stdio";
  consumerSettingsPath?: string;
  consumerServerKey: string;
  consumerCommand?: string;
  consumerArgs: string[];
};

export type RealAcpCorpusRunArtifacts = {
  manifest: RealAcpCorpusRunManifest;
  replayExport: RealAcpReplayExportManifest;
  scorecard: RealAcpStabilityScorecard;
  indexPath: string;
  rootIndexPath: string;
  exportPath: string;
  scorecardPath: string;
  scorecardMarkdownPath: string;
  indexRecordCount: number;
};

export type RealAcpCorpusRunPlan = {
  schemaVersion: "real-acp-corpus-launch-plan.v1";
  status: "ready" | "blocked";
  runId: string;
  mode: RealAcpExecutionMode;
  purpose: RealAcpCorpusRunPurpose;
  includeHoldout: boolean;
  selectedTaskIds: string[];
  outputDir: string;
  workspaceBaseDir: string;
  currentRepoPath: string;
  metadataPath?: string;
  runnerInvocation: {
    api: "runRealAcpCorpus";
    executionMode: RealAcpExecutionMode;
    executorKind: "simulated" | "headless_acp" | "real_consumer";
    dryRun: boolean;
  };
  safety: {
    outputUnderSafeRoot: boolean;
    workspaceUnderSafeRoot: boolean;
    currentRepoMutationRefused: true;
    actualConsumerLaunch: boolean;
  };
  integrationBlockers: string[];
  realConsumer?: {
    readiness: RealAcpConsumerReadiness;
    protocolEvidence: "acp_stdio_only_not_desktop_ui_parity";
  };
};

export type HeadlessAcpTranscriptRunner = (
  input: RealAcpHeadlessRunnerInput,
) => Promise<HeadlessAcpTranscriptSummary>;

export type RunPlannedRealAcpCorpusDeps = {
  runHeadlessTranscript?: HeadlessAcpTranscriptRunner;
  realConsumerReadinessProvider?: RealAcpConsumerReadinessProvider;
  runRealConsumerProtocol?: RealAcpConsumerProtocolRunner;
};

export type HeadlessAcpTranscriptSummary = {
  stopReason: string;
  trajectoryLength: number;
  counts: {
    fsRead: number;
    fsWrite: number;
    terminalCreate: number;
    terminalExit: number;
    permission: number;
    agentStderr: number;
  };
  trajectory: HeadlessAcpTrajectoryEntry[];
  transcriptPath?: string;
};

type HeadlessAcpTrajectoryEntry =
  | { kind: "fs_read"; path: string; bytes?: number }
  | { kind: "fs_write"; path: string; bytes?: number }
  | { kind: "terminal_create"; terminalId: string; command: string; args: string[] }
  | { kind: "terminal_exit"; terminalId: string; exitCode: number | null; signal: string | null; outputBytes?: number }
  | { kind: "permission"; chosen: string; toolCall?: unknown }
  | { kind: "agent_stderr"; line: string }
  | { kind: string; [key: string]: unknown };

type PartialCliOptions = {
  runId?: string;
  mode?: RealAcpExecutionMode;
  purpose?: RealAcpCorpusRunPurpose;
  includeHoldout?: boolean;
  metadataPath?: string;
  outputDir?: string;
  workspaceBaseDir?: string;
  currentRepoPath?: string;
  taskIds: string[];
  planOnly?: boolean;
  consumer?: "zed" | "glass" | "stdio";
  consumerSettingsPath?: string;
  consumerServerKey?: string;
  consumerCommand?: string;
  consumerArgs: string[];
};

export const parseRunRealAcpCorpusArgs = (
  argv: readonly string[],
  cwd: string = process.cwd(),
): RunRealAcpCorpusCliOptions => {
  const parsed: PartialCliOptions = { taskIds: [], consumerArgs: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--run-id") {
      parsed.runId = requiredValue(argv, ++index, arg);
    } else if (arg === "--mode") {
      parsed.mode = RealAcpExecutionModeSchema.parse(requiredValue(argv, ++index, arg));
    } else if (arg === "--purpose") {
      parsed.purpose = RealAcpCorpusRunPurposeSchema.parse(requiredValue(argv, ++index, arg));
    } else if (arg === "--include-holdout") {
      parsed.includeHoldout = true;
    } else if (arg === "--metadata") {
      parsed.metadataPath = requiredValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      parsed.outputDir = requiredValue(argv, ++index, arg);
    } else if (arg === "--workspace-base-dir") {
      parsed.workspaceBaseDir = requiredValue(argv, ++index, arg);
    } else if (arg === "--current-repo") {
      parsed.currentRepoPath = requiredValue(argv, ++index, arg);
    } else if (arg === "--task-id") {
      parsed.taskIds.push(requiredValue(argv, ++index, arg));
    } else if (arg === "--task-ids") {
      parsed.taskIds.push(...requiredValue(argv, ++index, arg).split(",").map((value) => value.trim()).filter(Boolean));
    } else if (arg === "--plan-only") {
      parsed.planOnly = true;
    } else if (arg === "--consumer") {
      const value = requiredValue(argv, ++index, arg);
      if (value !== "zed" && value !== "glass" && value !== "stdio") {
        throw new Error("--consumer must be zed, glass, or stdio");
      }
      parsed.consumer = value;
    } else if (arg === "--consumer-settings") {
      parsed.consumerSettingsPath = requiredValue(argv, ++index, arg);
    } else if (arg === "--consumer-server-key") {
      parsed.consumerServerKey = requiredValue(argv, ++index, arg);
    } else if (arg === "--consumer-command") {
      parsed.consumerCommand = requiredValue(argv, ++index, arg);
    } else if (arg === "--consumer-arg") {
      parsed.consumerArgs.push(requiredValue(argv, ++index, arg));
    } else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${String(arg)}`);
    }
  }

  const currentRepoPath = resolve(cwd, parsed.currentRepoPath ?? ".");
  const runId = parsed.runId ?? `real-acp-run.${new Date().toISOString().replace(/[^A-Za-z0-9._:-]+/g, "-")}`;
  const outputDir = resolvePath(currentRepoPath, parsed.outputDir ?? join(REAL_ACP_CORPUS_OUTPUT_ROOT, safeId(runId)));
  const workspaceBaseDir = resolvePath(currentRepoPath, parsed.workspaceBaseDir ?? join(outputDir, "workspaces"));
  return {
    runId,
    mode: parsed.mode ?? "dry_run",
    purpose: parsed.purpose ?? "development_eval",
    includeHoldout: parsed.includeHoldout ?? false,
    ...(parsed.metadataPath === undefined ? {} : { metadataPath: resolvePath(cwd, parsed.metadataPath) }),
    outputDir,
    workspaceBaseDir,
    currentRepoPath,
    taskIds: parsed.taskIds,
    planOnly: parsed.planOnly ?? false,
    consumer: parsed.consumer ?? "zed",
    ...(parsed.consumerSettingsPath === undefined ? {} : { consumerSettingsPath: resolvePath(cwd, parsed.consumerSettingsPath) }),
    consumerServerKey: parsed.consumerServerKey ?? "bleeding-agent",
    ...(parsed.consumerCommand === undefined ? {} : { consumerCommand: parsed.consumerCommand }),
    consumerArgs: parsed.consumerArgs,
  };
};

export const loadRealAcpRunMetadata = async (path: string): Promise<RealAcpRunMetadata> => {
  const raw = await readFile(path, "utf8");
  return RealAcpRunMetadataSchema.parse(JSON.parse(raw));
};

export const planRealAcpCorpusRun = (
  options: RunRealAcpCorpusCliOptions,
  deps: Pick<RunPlannedRealAcpCorpusDeps, "realConsumerReadinessProvider"> = {},
): RealAcpCorpusRunPlan => {
  assertUnderSafeOutputRoot(options.outputDir, options.currentRepoPath, "--out-dir");
  assertUnderSafeOutputRoot(options.workspaceBaseDir, options.currentRepoPath, "--workspace-base-dir");
  const selectedTasks = selectRealAcpCorpusTasks({
    purpose: options.purpose,
    includeHoldout: options.includeHoldout,
    ...(options.taskIds.length === 0 ? {} : { taskIds: options.taskIds }),
  });
  const realConsumerReadiness = options.mode === "real_consumer"
    ? (deps.realConsumerReadinessProvider ?? (() => resolveRealAcpConsumerReadiness({
      consumer: options.consumer,
      ...(options.consumerSettingsPath === undefined ? {} : { settingsPath: options.consumerSettingsPath }),
      serverKey: options.consumerServerKey,
      ...(options.consumerCommand === undefined ? {} : { command: options.consumerCommand }),
      args: options.consumerArgs,
      cwd: options.currentRepoPath,
    })))()
    : undefined;
  const integrationBlockers = integrationBlockersForMode(options.mode, realConsumerReadiness);
  const executorKind = executorKindForMode(options.mode);

  return {
    schemaVersion: "real-acp-corpus-launch-plan.v1",
    status: integrationBlockers.length === 0 ? "ready" : "blocked",
    runId: options.runId,
    mode: options.mode,
    purpose: options.purpose,
    includeHoldout: options.includeHoldout,
    selectedTaskIds: selectedTasks.map((task) => task.taskId),
    outputDir: options.outputDir,
    workspaceBaseDir: options.workspaceBaseDir,
    currentRepoPath: options.currentRepoPath,
    ...(options.metadataPath === undefined ? {} : { metadataPath: options.metadataPath }),
    runnerInvocation: {
      api: "runRealAcpCorpus",
      executionMode: options.mode,
      executorKind,
      dryRun: options.mode === "dry_run",
    },
    safety: {
      outputUnderSafeRoot: true,
      workspaceUnderSafeRoot: true,
      currentRepoMutationRefused: true,
      actualConsumerLaunch: options.mode !== "dry_run" && integrationBlockers.length === 0,
    },
    integrationBlockers,
    ...(realConsumerReadiness === undefined ? {} : {
      realConsumer: {
        readiness: realConsumerReadiness,
        protocolEvidence: "acp_stdio_only_not_desktop_ui_parity" as const,
      },
    }),
  };
};

export const runPlannedRealAcpCorpus = async (
  plan: RealAcpCorpusRunPlan,
  metadata: RealAcpRunMetadata,
  createdAt?: string,
  deps: RunPlannedRealAcpCorpusDeps = {},
): Promise<RealAcpCorpusRunManifest> => {
  if (plan.status === "blocked") {
    throw new Error(`real ACP corpus plan is blocked: ${plan.integrationBlockers.join("; ")}`);
  }
  const executor = plan.mode === "headless_acp"
    ? createRealAcpHeadlessExecutor({
      currentRepoPath: plan.currentRepoPath,
      allowedWorkspaceRoot: plan.workspaceBaseDir,
      runTask: (input) => runHeadlessAcpCorpusTask(input, plan, deps),
    })
    : plan.mode === "real_consumer"
      ? createRealAcpConsumerExecutor({
        currentRepoPath: plan.currentRepoPath,
        allowedWorkspaceRoot: plan.workspaceBaseDir,
        readiness: plan.realConsumer?.readiness ?? missingRealConsumerReadiness(),
        ...(deps.runRealConsumerProtocol === undefined ? {} : { runProtocol: deps.runRealConsumerProtocol }),
        transcriptDir: join(plan.outputDir, "transcripts"),
      })
      : createSimulatedRealAcpExecutor();
  const runMetadata = plan.mode === "real_consumer" && plan.realConsumer !== undefined
    ? {
      ...metadata,
      client: plan.realConsumer.readiness.clientMetadata,
    }
    : metadata;
  return runRealAcpCorpus({
    runId: plan.runId,
    metadata: runMetadata,
    executor,
    purpose: plan.purpose,
    executionMode: plan.mode,
    includeHoldout: plan.includeHoldout,
    taskIds: plan.selectedTaskIds,
    outputDir: plan.outputDir,
    workspaceBaseDir: plan.workspaceBaseDir,
    currentRepoPath: plan.currentRepoPath,
    ...(createdAt === undefined ? {} : { createdAt }),
  });
};

export const runPlannedRealAcpCorpusWithArtifacts = async (
  plan: RealAcpCorpusRunPlan,
  metadata: RealAcpRunMetadata,
  createdAt?: string,
  deps: RunPlannedRealAcpCorpusDeps = {},
): Promise<RealAcpCorpusRunArtifacts> => {
  const manifest = await runPlannedRealAcpCorpus(plan, metadata, createdAt, deps);
  const replayExport = createRealAcpReplayExportManifest({
    manifest,
    purpose: plan.purpose,
    status: plan.purpose === "optimizer_input" ? "optimizer_safe" : "evaluation_only",
    includeHoldout: plan.includeHoldout,
  });
  const index = buildRealAcpReplayCorpusIndex({
    runManifests: [manifest],
    reproductionCommand: [
      "tsx",
      "scripts/run_real_acp_corpus.ts",
      "--metadata",
      plan.metadataPath ?? "<metadata.json>",
      "--run-id",
      plan.runId,
      "--mode",
      plan.mode,
      "--purpose",
      plan.purpose,
    ],
    reproductionCwd: plan.currentRepoPath,
  });
  const exportPath = join(plan.outputDir, `${safeId(plan.runId)}.replay-export.json`);
  const scorecardPath = join(plan.outputDir, `${safeId(plan.runId)}.stability-scorecard.json`);
  const scorecardMarkdownPath = join(plan.outputDir, `${safeId(plan.runId)}.stability-scorecard.md`);
  const indexPath = join(plan.outputDir, "index.jsonl");
  const rootIndexPath = join(plan.currentRepoPath, ".bag", "replay-corpus", "index.jsonl");
  const scorecard = createRealAcpStabilityScorecard({
    manifests: [manifest],
    scorecardId: `real-acp-stability.${safeId(plan.runId)}`,
    createdAt: manifest.createdAt,
  });
  await mkdir(plan.outputDir, { recursive: true });
  await mkdir(dirname(rootIndexPath), { recursive: true });
  await writeFile(exportPath, `${JSON.stringify(replayExport, null, 2)}\n`, "utf8");
  await writeFile(scorecardPath, `${JSON.stringify(scorecard, null, 2)}\n`, "utf8");
  await writeFile(scorecardMarkdownPath, renderRealAcpStabilityScorecardMarkdown(scorecard), "utf8");
  const indexJsonl = serializeRealAcpReplayCorpusIndexJsonl(index);
  await writeFile(indexPath, indexJsonl, "utf8");
  await writeFile(rootIndexPath, indexJsonl, "utf8");
  return {
    manifest,
    replayExport,
    scorecard,
    indexPath,
    rootIndexPath,
    exportPath,
    scorecardPath,
    scorecardMarkdownPath,
    indexRecordCount: index.length,
  };
};

const integrationBlockersForMode = (
  mode: RealAcpExecutionMode,
  realConsumerReadiness?: RealAcpConsumerReadiness,
): string[] => {
  if (mode === "dry_run") return [];
  if (mode === "headless_acp") {
    return [];
  }
  if (realConsumerReadiness === undefined) {
    return ["real_consumer readiness provider did not return launch evidence"];
  }
  if (realConsumerReadiness.status === "ready") {
    return [];
  }
  return realConsumerReadiness.blockers.map((blocker) =>
    `${realConsumerReadiness.consumerName} real_consumer readiness blocked: ${blocker}`);
};

const missingRealConsumerReadiness = (): RealAcpConsumerReadiness => ({
  providerId: "real-acp.consumer.missing-plan-readiness",
  consumerName: "stdio",
  status: "blocked",
  blockers: ["real_consumer plan has no readiness evidence"],
  clientMetadata: {
    clientProfileId: "client.real-acp.missing-plan-readiness",
    clientName: "Missing real ACP consumer readiness",
    clientVersion: "unknown",
    transport: "stdio",
    acpConsumerCapabilities: {},
  },
  capabilityEvidence: {},
});

const executorKindForMode = (
  mode: RealAcpExecutionMode,
): "simulated" | "headless_acp" | "real_consumer" => {
  if (mode === "dry_run") return "simulated";
  return mode;
};

const runHeadlessAcpCorpusTask = async (
  input: RealAcpHeadlessRunnerInput,
  plan: RealAcpCorpusRunPlan,
  deps: RunPlannedRealAcpCorpusDeps,
): Promise<RealAcpHeadlessRunnerOutput> => {
  const modelProfileBlockers = deps.runHeadlessTranscript === undefined
    ? headlessQualityPrerequisiteBlockers(input)
    : [];
  if (modelProfileBlockers.length > 0) {
    return blockedHeadlessQualityOutput(input, plan, modelProfileBlockers);
  }
  const transcript = await (deps.runHeadlessTranscript ?? ((runnerInput) =>
    runBagAcpHeadlessTranscript(runnerInput, plan)))(input);
  const terminalCommands = terminalRecordsFromTranscript(input.task.taskId, transcript);
  const toolCalls = toolRecordsFromTranscript(input.task.taskId, transcript);
  const status = statusFromTranscript(input, transcript, terminalCommands);
  const qualityTelemetry = await qualityTelemetryFromTranscript(input, plan, transcript, terminalCommands, status);
  return {
    status,
    route: {
      routeId: `route.${safeId(input.task.taskId)}.headless-acp`,
      selectedMode: status === "cancelled" ? "cancelled" : "coding",
      reason: "Executed through scripts/bag_acp_run.ts headless ACP consumer.",
      confidence: 1,
    },
    editStrategy: {
      strategyId: "edit.headless-acp.consumer.v1",
      family: transcript.counts.fsWrite > 0 ? "diff" : "none",
      selectedBy: transcript.counts.fsWrite > 0 ? "executor" : "not_applicable",
      reason: "Derived from headless ACP transcript side effects.",
    },
    toolCalls,
    terminalCommands,
    verifier: {
      status: verifierStatusFromTranscript(input, status, terminalCommands),
      policy: input.expectedOutcome.verification.policy,
      commandIds: terminalCommands.map((command) => command.commandId),
      ...(input.expectedOutcome.verification.skipReason === undefined
        ? {}
        : { skipReason: input.expectedOutcome.verification.skipReason }),
    },
    repair: {
      attempted: hasTrajectoryDomainAction(transcript, "repair"),
      status: "not_needed",
    },
    rollback: {
      attempted: hasTrajectoryDomainAction(transcript, "rollback"),
      status: "not_needed",
    },
    telemetry: jsonClean({
      headlessAcp: {
        runner: "scripts/bag_acp_run.ts",
        stopReason: transcript.stopReason,
        trajectoryLength: transcript.trajectoryLength,
        counts: transcript.counts,
        ...(transcript.transcriptPath === undefined ? {} : { transcriptPath: transcript.transcriptPath }),
      },
      ...qualityTelemetry,
      corpusLaunch: {
        runId: plan.runId,
        mode: plan.mode,
        purpose: plan.purpose,
      },
    }) as JsonValue,
    ...(status === "failed" || status === "error"
      ? { failureReason: failureReasonFromTranscript(transcript) }
      : {}),
    ...(status === "skipped" ? { skipReason: "headless ACP transcript had no write or verifier signal" } : {}),
  };
};

const runBagAcpHeadlessTranscript = async (
  input: RealAcpHeadlessRunnerInput,
  plan: RealAcpCorpusRunPlan,
): Promise<HeadlessAcpTranscriptSummary> => {
  const transcriptPath = join(plan.outputDir, "transcripts", `${safeId(input.task.taskId)}.json`);
  const repoRoot = process.cwd();
  const tsxBin = resolve(repoRoot, "node_modules/.bin/tsx");
  const headlessRunnerScript = resolve(repoRoot, "scripts/bag_acp_run.ts");
  const args = [
    headlessRunnerScript,
    headlessQualityPrompt(input),
    "--workdir",
    input.workspace.workspacePath,
    "--out",
    transcriptPath,
    "--timeout-ms",
    String(Math.min(input.context.timeoutMs, 120_000)),
    "--terminal-mode",
    "real",
    "--client-profile",
    "capable",
    "--no-resume-check",
    "--close-session",
  ];
  if (input.task.primaryLabel === "cancellation") {
    args.push("--cancel-after-ms", "500");
  }
  const result = await spawnProcess(tsxBin, args, input.workspace.workspacePath, input.context.signal);
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(transcriptPath, "utf8"));
  } catch {
    parsed = {
      stopReason: result.exitCode === 0 ? "missing-transcript" : `error:${result.stderr.slice(0, 200)}`,
      trajectoryLength: 0,
      counts: {},
      trajectory: [],
    };
  }
  return parseHeadlessTranscript(parsed, transcriptPath, result.exitCode, result.stderr);
};

const headlessQualityPrompt = (input: RealAcpHeadlessRunnerInput): string => [
  input.task.userPrompt,
  "",
  "This is an isolated real ACP quality fixture workspace. Do not touch any repository outside this workspace.",
  `Task id: ${input.task.taskId}`,
  `Seed files: ${input.workspace.materializedFilePaths.join(", ") || "(none)"}`,
  `Allowed path prefixes: ${input.workspace.allowedPathPrefixes.join(", ")}`,
  input.workspace.protectedPaths.length === 0
    ? "Protected paths: (none)"
    : `Protected paths: ${input.workspace.protectedPaths.join(", ")}`,
  input.expectedOutcome.expectedChangedPaths.length === 0
    ? "Expected changed paths: (none)"
    : `Expected changed paths: ${input.expectedOutcome.expectedChangedPaths.join(", ")}`,
  input.expectedOutcome.expectedNoChangePaths.length === 0
    ? "Expected no-change paths: (none)"
    : `Expected no-change paths: ${input.expectedOutcome.expectedNoChangePaths.join(", ")}`,
  input.expectedOutcome.verification.commands.length === 0
    ? "Verifier command: none; explain why verification is skipped if you make no terminal call."
    : `Run this verifier before finishing: ${input.expectedOutcome.verification.commands.map((command) => command.join(" ")).join(" && ")}`,
].join("\n");

const headlessQualityPrerequisiteBlockers = (input: RealAcpHeadlessRunnerInput): string[] => {
  const blockers: string[] = [];
  const model = input.run.metadata.model;
  const client = input.run.metadata.client;
  const provider = model.provider.toLowerCase();
  const modelName = model.model.toLowerCase();
  if (provider === "simulated" || provider === "injected" || modelName.includes("simulated")) {
    blockers.push(`model profile ${model.modelProfileId} is ${model.provider}/${model.model}, not a live generation profile`);
  }
  if (model.toolCallingMode === "disabled") {
    blockers.push(`model profile ${model.modelProfileId} has toolCallingMode=disabled`);
  }
  if (client.transport === "simulated") {
    blockers.push(`client profile ${client.clientProfileId} uses simulated transport`);
  }
  if (client.acpConsumerCapabilities.filesystem !== true) {
    blockers.push(`client profile ${client.clientProfileId} does not declare filesystem capability`);
  }
  if (client.acpConsumerCapabilities.terminal !== true) {
    blockers.push(`client profile ${client.clientProfileId} does not declare terminal capability`);
  }
  return blockers;
};

const blockedHeadlessQualityOutput = (
  input: RealAcpHeadlessRunnerInput,
  plan: RealAcpCorpusRunPlan,
  blockers: readonly string[],
): RealAcpHeadlessRunnerOutput => {
  const diagnostic = classifyCodingProgress({
    runId: `${plan.runId}.${input.task.taskId}`,
    patch: emptyPatch(input, {
      modelAvailable: false,
      modelRole: input.run.metadata.model.modelRole === "master" ? "master" : "local",
      modelError: blockers.join("; "),
      rawEditCount: 0,
      rawCommandCount: 0,
    }),
    plannedCommands: input.expectedOutcome.verification.commands,
    commandResults: [],
    terminal: "final",
    evidenceRefs: [`real-acp-model-profile:${input.run.metadata.model.modelProfileId}`],
  });
  return {
    status: "error",
    route: {
      routeId: `route.${safeId(input.task.taskId)}.headless-acp.blocked`,
      selectedMode: "coding",
      reason: "Headless ACP quality run blocked before launch by model/profile prerequisites.",
      confidence: 1,
    },
    editStrategy: {
      strategyId: "edit.none.model-profile-blocked",
      family: "none",
      selectedBy: "not_applicable",
      reason: "No live mutating headless ACP model/profile was available.",
    },
    toolCalls: [],
    terminalCommands: [],
    verifier: {
      status: "not_run",
      policy: input.expectedOutcome.verification.policy,
      commandIds: [],
    },
    repair: {
      attempted: false,
      status: "skipped",
      reason: "quality run blocked before generation",
    },
    rollback: {
      attempted: false,
      status: "skipped",
      reason: "quality run blocked before generation",
    },
    telemetry: jsonClean({
      codingProgressDiagnostic: jsonClean(diagnostic),
      headlessAcp: {
        runner: "scripts/bag_acp_run.ts",
        blocked: true,
        blockerKind: "model_profile_prerequisite",
        blockers: [...blockers],
      },
      corpusLaunch: {
        runId: plan.runId,
        mode: plan.mode,
        purpose: plan.purpose,
      },
    }) as JsonValue,
    failureReason: `headless ACP quality run blocked by model/profile prerequisites: ${blockers.join("; ")}`,
  };
};

const spawnProcess = (
  command: string,
  args: readonly string[],
  cwd: string,
  signal: AbortSignal,
): Promise<{ exitCode: number | null; stderr: string }> =>
  new Promise((resolveProcess, reject) => {
    const child = spawn(command, [...args], { cwd, env: process.env });
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    const abort = () => {
      child.kill("SIGTERM");
      reject(new Error("headless ACP corpus task aborted"));
    };
    signal.addEventListener("abort", abort, { once: true });
    child.once("error", (error) => {
      signal.removeEventListener("abort", abort);
      reject(error);
    });
    child.once("close", (exitCode) => {
      signal.removeEventListener("abort", abort);
      resolveProcess({ exitCode, stderr });
    });
  });

const parseHeadlessTranscript = (
  value: unknown,
  transcriptPath: string,
  processExitCode: number | null,
  processStderr: string,
): HeadlessAcpTranscriptSummary => {
  const object = value != null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
  const trajectory = Array.isArray(object.trajectory)
    ? object.trajectory.filter((entry): entry is HeadlessAcpTrajectoryEntry =>
      entry != null && typeof entry === "object" && typeof (entry as { kind?: unknown }).kind === "string")
    : [];
  const countsInput = object.counts != null && typeof object.counts === "object"
    ? object.counts as Record<string, unknown>
    : {};
  const counts = {
    fsRead: numberCount(countsInput.fsRead),
    fsWrite: numberCount(countsInput.fsWrite),
    terminalCreate: numberCount(countsInput.terminalCreate),
    terminalExit: numberCount(countsInput.terminalExit),
    permission: numberCount(countsInput.permission),
    agentStderr: numberCount(countsInput.agentStderr),
  };
  const stopReason = typeof object.stopReason === "string"
    ? object.stopReason
    : processExitCode === 0 ? "ok" : `error:${processStderr.slice(0, 200)}`;
  return {
    stopReason,
    trajectoryLength: typeof object.trajectoryLength === "number" ? object.trajectoryLength : trajectory.length,
    counts,
    trajectory,
    transcriptPath,
  };
};

const numberCount = (value: unknown): number =>
  Number.isInteger(value) && Number(value) >= 0 ? Number(value) : 0;

const terminalRecordsFromTranscript = (
  taskId: string,
  transcript: HeadlessAcpTranscriptSummary,
) => transcript.trajectory
  .filter((entry): entry is Extract<HeadlessAcpTrajectoryEntry, { kind: "terminal_exit" }> => entry.kind === "terminal_exit")
  .map((entry, index) => {
    const create = transcript.trajectory.find((candidate): candidate is Extract<HeadlessAcpTrajectoryEntry, { kind: "terminal_create" }> =>
      candidate.kind === "terminal_create" && candidate.terminalId === entry.terminalId);
    return {
      commandId: `cmd.${safeId(taskId)}.${index}`,
      command: [create?.command ?? "unknown", ...(create?.args ?? [])],
      status: entry.exitCode === 0 ? "succeeded" as const : "failed" as const,
      exitCode: entry.exitCode,
      durationMs: 0,
    };
  });

const toolRecordsFromTranscript = (
  taskId: string,
  transcript: HeadlessAcpTranscriptSummary,
) => [
  ...transcript.trajectory
    .filter((entry) => entry.kind === "fs_read")
    .map((entry, index) => ({
      toolCallId: `tool.${safeId(taskId)}.fs-read.${index}`,
      namespace: "acp.fs",
      name: "readTextFile",
      status: "succeeded" as const,
      sideEffectLevel: "read" as const,
      telemetry: entry,
    })),
  ...transcript.trajectory
    .filter((entry) => entry.kind === "fs_write")
    .map((entry, index) => ({
      toolCallId: `tool.${safeId(taskId)}.fs-write.${index}`,
      namespace: "acp.fs",
      name: "writeTextFile",
      status: "succeeded" as const,
      sideEffectLevel: "write" as const,
      telemetry: entry,
    })),
].map(({ telemetry: _telemetry, ...record }) => record);

const statusFromTranscript = (
  input: RealAcpHeadlessRunnerInput,
  transcript: HeadlessAcpTranscriptSummary,
  terminalCommands: ReturnType<typeof terminalRecordsFromTranscript>,
): RealAcpHeadlessRunnerOutput["status"] => {
  if (/cancel/i.test(transcript.stopReason)) return "cancelled";
  if (transcript.stopReason.startsWith("error:")) return "error";
  if (terminalCommands.some((command) => command.exitCode !== 0)) return "failed";
  if (input.expectedOutcome.mutation === "no_change") return "passed";
  if (input.expectedOutcome.mutation === "detect_without_final_success") {
    return terminalCommands.some((command) => command.exitCode !== 0) ? "passed" : "failed";
  }
  if (input.expectedOutcome.verification.policy === "required" && terminalCommands.length === 0) return "failed";
  if (transcript.counts.fsWrite > 0) return "passed";
  return "failed";
};

const verifierStatusFromTranscript = (
  input: RealAcpHeadlessRunnerInput,
  status: RealAcpHeadlessRunnerOutput["status"],
  terminalCommands: ReturnType<typeof terminalRecordsFromTranscript>,
) => {
  if (input.expectedOutcome.verification.policy === "must_skip") return "skipped" as const;
  if (terminalCommands.length === 0) return "not_run" as const;
  if (terminalCommands.some((command) => command.exitCode !== 0)) return "failed" as const;
  return "passed" as const;
};

const qualityTelemetryFromTranscript = async (
  input: RealAcpHeadlessRunnerInput,
  plan: RealAcpCorpusRunPlan,
  transcript: HeadlessAcpTranscriptSummary,
  terminalCommands: ReturnType<typeof terminalRecordsFromTranscript>,
  status: RealAcpHeadlessRunnerOutput["status"],
) => {
  const writeEvents = await Promise.all(transcript.trajectory
    .filter((entry): entry is Extract<HeadlessAcpTrajectoryEntry, { kind: "fs_write" }> => entry.kind === "fs_write")
    .map(async (entry) => {
      const path = workspaceRelativePath(input.workspace.workspacePath, entry.path);
      return {
        kind: "fs_write",
        path,
        bytes: entry.bytes ?? 0,
        contentHash: await contentHashForPath(entry.path),
      };
    }));
  const commandResults: TerminalCommandResult[] = terminalCommands.map((command) => ({
    command: command.command[0] ?? "unknown",
    args: command.command.slice(1),
    reason: "headless ACP transcript terminal verifier",
    exitCode: command.exitCode,
    signal: null,
    output: "",
  }));
  const editResults: CodingEditResult[] = writeEvents.map((event) => ({
    path: event.path,
    ok: true,
    reason: "headless ACP transcript recorded fs_write",
    newHash: event.contentHash,
    editStrategyId: "edit.headless-acp.consumer.v1",
    editStatus: "applied",
  }));
  const classified = classifyCodingProgress({
    runId: `${plan.runId}.${input.task.taskId}`,
    patch: transcriptPatch(input, transcript, writeEvents.length),
    editResults,
    plannedCommands: input.expectedOutcome.verification.commands,
    commandResults,
    terminal: "final",
    evidenceRefs: [
      `real-acp-task:${input.task.taskId}`,
      ...(transcript.transcriptPath === undefined ? [] : [`headless-transcript:${transcript.transcriptPath}`]),
    ],
  });
  const codingProgressDiagnostic = transcript.counts.fsWrite === 0 && transcriptHasNoEditVerifier(transcript)
    ? {
      ...classified,
      progressClass: "empty_edits" as const,
      failureSignals: ["generation.empty_edits"],
      reason: "The headless ACP coding runner produced no edit operations for a mutating fixture; the verifier only recorded the no-edit failure.",
    }
    : classified;
  return {
    codingProgressDiagnostic: jsonClean(codingProgressDiagnostic),
    writeEvents,
    postApplyConsistencyStatus: status === "passed" ? "consistent" : status === "failed" ? "inconclusive" : "not_checked",
  };
};

const transcriptPatch = (
  input: RealAcpHeadlessRunnerInput,
  transcript: HeadlessAcpTranscriptSummary,
  writeCount: number,
): CodingPatch => ({
  summary: `Headless ACP transcript for ${input.task.taskId}`,
  editStrategy: {
    strategyId: "edit.headless-acp.consumer.v1",
    strategyFamily: writeCount > 0 ? "unified_diff" : "custom",
    renderedEditToolContractId: "rendered.headless-acp.consumer.v1",
  },
  generation: {
    modelAvailable: true,
    modelRole: input.run.metadata.model.modelRole === "master" ? "master" : "local",
    rawEditCount: writeCount,
    rawCommandCount: transcript.counts.terminalCreate,
  },
  edits: Array.from({ length: writeCount }, (_value, index) => ({
    reason: "headless ACP transcript recorded write",
    editInput: {
      strategyFamily: "whole_file",
      payload: {
        path: `headless-write-${index}`,
        content: "",
      },
    },
    targetFiles: [],
    editStrategyId: "edit.headless-acp.consumer.v1",
    editStrategyFamily: "unified_diff",
    renderedEditToolContractId: "rendered.headless-acp.consumer.v1",
  })),
  commands: input.expectedOutcome.verification.commands.map((command) => ({
    command: command[0] ?? "unknown",
    args: command.slice(1),
    reason: "expected real ACP fixture verifier",
  })),
  risks: [],
  parseFailures: writeCount === 0 && /parse/i.test(transcript.stopReason) ? [transcript.stopReason] : [],
});

const emptyPatch = (
  input: RealAcpHeadlessRunnerInput,
  generation: NonNullable<CodingPatch["generation"]>,
): CodingPatch => ({
  summary: `Headless ACP quality run blocked for ${input.task.taskId}`,
  editStrategy: {
    strategyId: "edit.none.model-profile-blocked",
    strategyFamily: "custom",
    renderedEditToolContractId: "rendered.headless-acp.consumer.v1",
  },
  generation,
  edits: [],
  commands: input.expectedOutcome.verification.commands.map((command) => ({
    command: command[0] ?? "unknown",
    args: command.slice(1),
    reason: "expected real ACP fixture verifier",
  })),
  risks: [],
  parseFailures: [],
});

const contentHashForPath = async (path: string): Promise<string> => {
  try {
    return `sha256:${createHash("sha256").update(await readFile(path)).digest("hex")}`;
  } catch {
    return "sha256:unavailable";
  }
};

const workspaceRelativePath = (workspacePath: string, path: string): string => {
  const relativePath = relative(resolve(workspacePath), resolve(path));
  if (relativePath === "" || relativePath.startsWith("..") || relativePath.includes(`..${sep}`)) {
    return path.replaceAll("\\", "/");
  }
  return relativePath.replaceAll("\\", "/");
};

const jsonClean = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

const transcriptHasNoEditVerifier = (transcript: HeadlessAcpTranscriptSummary): boolean =>
  transcript.trajectory.some((entry) => {
    if (entry.kind !== "terminal_create") return false;
    const command = typeof entry.command === "string" ? entry.command : "";
    const args = Array.isArray(entry.args) ? entry.args.filter((arg): arg is string => typeof arg === "string") : [];
    return `${command} ${args.join(" ")}`.includes("no edit operations were generated");
  });

const failureReasonFromTranscript = (transcript: HeadlessAcpTranscriptSummary): string => {
  if (transcript.counts.fsWrite === 0 && transcriptHasNoEditVerifier(transcript)) {
    return `headless ACP generated no edit operations before verifier failure; stopReason=${transcript.stopReason}`;
  }
  return `headless ACP transcript stopReason=${transcript.stopReason}`;
};

const hasTrajectoryDomainAction = (
  transcript: HeadlessAcpTranscriptSummary,
  action: "repair" | "rollback",
): boolean =>
  transcript.trajectory.some((entry) => entry.kind === action || entry.kind.startsWith(`${action}_`));

const assertUnderSafeOutputRoot = (path: string, currentRepoPath: string, flag: string): void => {
  const safeRoot = resolve(currentRepoPath, REAL_ACP_CORPUS_OUTPUT_ROOT);
  if (!isInsideOrEqual(path, safeRoot)) {
    throw new Error(`${flag} must be under ${REAL_ACP_CORPUS_OUTPUT_ROOT}`);
  }
};

const isInsideOrEqual = (candidatePath: string, rootPath: string): boolean => {
  const relativePath = relative(resolve(rootPath), resolve(candidatePath));
  return relativePath === "" || (!relativePath.startsWith("..") && !relativePath.includes(`..${sep}`));
};

const resolvePath = (base: string, path: string): string =>
  path.startsWith("/") ? resolve(path) : resolve(base, path);

const requiredValue = (argv: readonly string[], index: number, flag: string): string => {
  const value = argv[index];
  if (value == null || value.startsWith("--")) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

const safeId = (value: string): string => value.replace(/[^A-Za-z0-9._:-]+/g, "-");

const printUsage = (): void => {
  process.stdout.write(`usage: tsx scripts/run_real_acp_corpus.ts --metadata metadata.json [options]

Builds a bounded real ACP corpus launch plan and runs only the safe dry-run substrate by default.

Options:
  --mode dry_run|headless_acp|real_consumer
  --purpose optimizer_input|development_eval|holdout_final
  --include-holdout
  --task-id ID | --task-ids ID,ID
  --run-id ID
  --out-dir PATH
  --workspace-base-dir PATH
  --current-repo PATH
  --consumer zed|glass|stdio
  --consumer-settings PATH
  --consumer-server-key NAME
  --consumer-command CMD
  --consumer-arg ARG
  --plan-only
`);
};

const main = async (): Promise<void> => {
  const options = parseRunRealAcpCorpusArgs(process.argv.slice(2));
  if (options.metadataPath === undefined) {
    throw new Error("--metadata is required");
  }
  const metadata = await loadRealAcpRunMetadata(options.metadataPath);
  const plan = planRealAcpCorpusRun(options);
  if (options.planOnly || plan.status === "blocked") {
    process.stdout.write(`${JSON.stringify(plan, null, 2)}\n`);
    if (plan.status === "blocked") {
      process.exitCode = 1;
    }
    return;
  }
  const artifacts = await runPlannedRealAcpCorpusWithArtifacts(plan, metadata);
  process.stdout.write(`${JSON.stringify({
    status: "complete",
    runId: artifacts.manifest.runId,
    mode: artifacts.manifest.executionMode,
    taskCount: artifacts.manifest.summary.total,
    holdout: artifacts.manifest.summary.holdout,
    manifestPath: artifacts.manifest.manifestPath,
    exportPath: artifacts.exportPath,
    scorecardPath: artifacts.scorecardPath,
    scorecardMarkdownPath: artifacts.scorecardMarkdownPath,
    indexPath: artifacts.indexPath,
    rootIndexPath: artifacts.rootIndexPath,
    indexRecordCount: artifacts.indexRecordCount,
  }, null, 2)}\n`);
};

const directRun = process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;
if (directRun) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  });
}
