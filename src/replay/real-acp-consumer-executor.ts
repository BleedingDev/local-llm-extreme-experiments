import { spawn, type ChildProcess, type ChildProcessWithoutNullStreams } from "node:child_process";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { Readable, Writable } from "node:stream";
import * as acp from "@agentclientprotocol/sdk";
import type { JsonValue } from "../optimizer/types";
import {
  RealAcpEditStrategyRecordSchema,
  RealAcpRepairRecordSchema,
  RealAcpRollbackRecordSchema,
  RealAcpRouteRecordSchema,
  RealAcpTerminalRecordSchema,
  RealAcpToolRecordSchema,
  RealAcpVerifierRecordSchema,
  type RealAcpExecutorTaskInput,
  type RealAcpExecutorTaskOutput,
  type RealAcpHeadlessExecutor,
  type RealAcpRunClientMetadata,
  type RealAcpTaskOutcomeStatus,
  type RealAcpTerminalRecord,
  type RealAcpToolRecord,
} from "./real-acp-runner";

export type RealAcpNamedConsumer = "Glass" | "Zed" | "stdio";

export type RealAcpConsumerLaunchConfig = {
  command: string;
  args: readonly string[];
  cwd?: string;
  env?: Readonly<Record<string, string>>;
};

export type RealAcpConsumerAppMetadata = {
  consumer: RealAcpNamedConsumer;
  appPath?: string;
  installed: boolean;
  bundleIdentifier?: string;
  version?: string;
};

export type RealAcpConsumerReadiness = {
  providerId: string;
  consumerName: RealAcpNamedConsumer;
  status: "ready" | "blocked";
  blockers: readonly string[];
  launch?: RealAcpConsumerLaunchConfig;
  app?: RealAcpConsumerAppMetadata;
  clientMetadata: RealAcpRunClientMetadata;
  capabilityEvidence: JsonValue;
};

export type RealAcpConsumerReadinessProvider = () => RealAcpConsumerReadiness;

export type RealAcpConsumerProtocolRunnerInput = {
  task: RealAcpExecutorTaskInput["task"];
  workspacePath: string;
  prompt: string;
  launch: RealAcpConsumerLaunchConfig;
  readiness: RealAcpConsumerReadiness;
  timeoutMs: number;
  signal: AbortSignal;
  transcriptPath?: string;
};

export type RealAcpConsumerTrajectoryEntry =
  | { kind: "protocol_call"; at: string; method: string; phase: "request" | "response" | "error"; sessionId?: string; payload?: unknown }
  | { kind: "session_update"; at: string; update: unknown }
  | { kind: "permission"; at: string; toolCall: unknown; chosen: string }
  | { kind: "fs_read"; at: string; path: string; bytes: number }
  | { kind: "fs_write"; at: string; path: string; bytes: number }
  | { kind: "terminal_create"; at: string; terminalId: string; command: string; args: string[] }
  | { kind: "terminal_output"; at: string; terminalId: string; outputBytes: number; truncated: boolean }
  | { kind: "terminal_exit"; at: string; terminalId: string; exitCode: number | null; signal: string | null; outputBytes?: number }
  | { kind: "terminal_kill"; at: string; terminalId: string }
  | { kind: "terminal_release"; at: string; terminalId: string }
  | { kind: "agent_stderr"; at: string; line: string }
  | { kind: "cancellation"; at: string; phase: string }
  | { kind: "error"; at: string; message: string };

export type RealAcpConsumerTranscriptSummary = {
  stopReason: string;
  trajectoryLength: number;
  counts: {
    protocolCalls: number;
    sessionUpdates: number;
    fsRead: number;
    fsWrite: number;
    terminalCreate: number;
    terminalOutput: number;
    terminalExit: number;
    terminalKill: number;
    terminalRelease: number;
    permission: number;
    agentStderr: number;
    cancellation: number;
    error: number;
  };
  protocolMethods: Record<string, number>;
  trajectory: readonly RealAcpConsumerTrajectoryEntry[];
  transcriptPath?: string;
};

export type RealAcpConsumerProtocolRunner = (
  input: RealAcpConsumerProtocolRunnerInput,
) => Promise<RealAcpConsumerTranscriptSummary>;

export type RealAcpConsumerExecutorOptions = {
  executorId?: string;
  executorVersion?: string;
  currentRepoPath?: string;
  allowedWorkspaceRoot: string;
  readiness: RealAcpConsumerReadiness;
  runProtocol?: RealAcpConsumerProtocolRunner;
  transcriptDir?: string;
};

type AgentServerConfig = {
  type?: string;
  command: string;
  args: string[];
  env: Record<string, string>;
};

const DEFAULT_EXECUTOR_ID = "real-acp.executor.consumer.stdio";
const DEFAULT_EXECUTOR_VERSION = "real-consumer-stdio.v1";

export const createRealAcpConsumerExecutor = (
  options: RealAcpConsumerExecutorOptions,
): RealAcpHeadlessExecutor => ({
  executorId: options.executorId ?? DEFAULT_EXECUTOR_ID,
  executorVersion: options.executorVersion ?? DEFAULT_EXECUTOR_VERSION,
  kind: "real_consumer",
  executeTask: async (input) => executeRealConsumerTask(input, options),
});

export const createStdioAcpConsumerReadiness = (input: {
  command?: string;
  args?: readonly string[];
  cwd?: string;
  env?: Readonly<Record<string, string>>;
  clientName?: string;
  clientVersion?: string;
}): RealAcpConsumerReadiness => {
  const command = input.command?.trim();
  const blockers = command == null || command === "" ? ["stdio ACP consumer command is not configured"] : [];
  return readiness({
    providerId: "real-acp.consumer.stdio",
    consumerName: "stdio",
    blockers,
    ...(blockers.length === 0
      ? { launch: {
        command: command ?? "",
        args: [...(input.args ?? [])],
        ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
        ...(input.env === undefined ? {} : { env: { ...input.env } }),
      } }
      : {}),
    clientName: input.clientName ?? "Configured stdio ACP consumer",
    clientVersion: input.clientVersion ?? "unknown",
    capabilityEvidence: {
      source: "explicit stdio launch config",
      protocol: "ACP over stdio",
    },
  });
};

export const createZedAcpConsumerReadiness = (input: {
  settingsPath: string;
  serverKey: string;
  appPath?: string;
}): RealAcpConsumerReadiness => namedSettingsReadiness({
  providerId: "real-acp.consumer.zed",
  consumerName: "Zed",
  settingsPath: input.settingsPath,
  serverKey: input.serverKey,
  appPath: input.appPath ?? "/Applications/Zed.app",
  clientProfileId: "client.real-acp.zed",
  clientName: "Zed ACP consumer",
});

export const createGlassAcpConsumerReadiness = (input: {
  settingsPath?: string;
  serverKey: string;
  appPath?: string;
}): RealAcpConsumerReadiness => namedSettingsReadiness({
  providerId: "real-acp.consumer.glass",
  consumerName: "Glass",
  ...(input.settingsPath === undefined ? {} : { settingsPath: input.settingsPath }),
  serverKey: input.serverKey,
  appPath: input.appPath ?? "/Applications/Glass.app",
  clientProfileId: "client.real-acp.glass",
  clientName: "Glass ACP consumer",
});

export const resolveRealAcpConsumerReadiness = (input: {
  consumer: "zed" | "glass" | "stdio";
  settingsPath?: string;
  serverKey?: string;
  command?: string;
  args?: readonly string[];
  cwd?: string;
}): RealAcpConsumerReadiness => {
  if (input.consumer === "stdio") {
    return createStdioAcpConsumerReadiness({
      ...(input.command === undefined ? {} : { command: input.command }),
      ...(input.args === undefined ? {} : { args: input.args }),
      ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
    });
  }
  if (input.consumer === "glass") {
    return createGlassAcpConsumerReadiness({
      ...(input.settingsPath === undefined ? {} : { settingsPath: input.settingsPath }),
      serverKey: input.serverKey ?? "bleeding-agent",
    });
  }
  return createZedAcpConsumerReadiness({
    settingsPath: input.settingsPath ?? resolve(process.env.HOME ?? "", ".config", "zed", "settings.json"),
    serverKey: input.serverKey ?? "bleeding-agent",
  });
};

const executeRealConsumerTask = async (
  input: RealAcpExecutorTaskInput,
  options: RealAcpConsumerExecutorOptions,
): Promise<RealAcpExecutorTaskOutput> => {
  assertWorkspaceSafeForRealConsumer(input.workspacePath, options.currentRepoPath ?? process.cwd(), options.allowedWorkspaceRoot);
  if (options.readiness.status !== "ready" || options.readiness.launch === undefined) {
    return blockedOutput(input, options.readiness);
  }
  if (input.context.signal.aborted) {
    return cancelledOutput(input, "real ACP consumer execution cancelled before start", options.readiness, []);
  }

  const transcriptPath = options.transcriptDir === undefined
    ? undefined
    : resolve(options.transcriptDir, `${safeId(input.task.taskId)}.real-consumer-transcript.json`);
  const prompt = realConsumerPrompt(input);
  try {
    const transcript = await (options.runProtocol ?? runStdioAcpConsumerProtocol)({
      task: input.task,
      workspacePath: input.workspacePath,
      prompt,
      launch: options.readiness.launch,
      readiness: options.readiness,
      timeoutMs: Math.min(input.context.timeoutMs, 120_000),
      signal: input.context.signal,
      ...(transcriptPath === undefined ? {} : { transcriptPath }),
    });
    const terminalCommands = terminalRecordsFromConsumerTranscript(input.task.taskId, transcript);
    const toolCalls = toolRecordsFromConsumerTranscript(input.task.taskId, transcript);
    const status = statusFromConsumerTranscript(input, transcript, terminalCommands);
    return {
      status,
      route: RealAcpRouteRecordSchema.parse({
        routeId: `route.${safeId(input.task.taskId)}.real-consumer`,
        selectedMode: status === "cancelled" ? "cancelled" : "coding",
        reason: `Executed through ${options.readiness.consumerName} real ACP consumer over stdio protocol.`,
        confidence: 1,
      }),
      editStrategy: RealAcpEditStrategyRecordSchema.parse({
        strategyId: "edit.real-consumer.acp.v1",
        family: transcript.counts.fsWrite > 0 ? "diff" : "none",
        selectedBy: transcript.counts.fsWrite > 0 ? "executor" : "not_applicable",
        reason: "Derived from real ACP consumer transcript side effects.",
      }),
      toolCalls,
      terminalCommands,
      verifier: RealAcpVerifierRecordSchema.parse({
        status: verifierStatusFromConsumerTranscript(input, status, terminalCommands),
        policy: input.task.expectedOutcome.verification.policy,
        commandIds: terminalCommands.map((command) => command.commandId),
        ...(input.task.expectedOutcome.verification.skipReason === undefined
          ? {}
          : { skipReason: input.task.expectedOutcome.verification.skipReason }),
      }),
      repair: RealAcpRepairRecordSchema.parse({
        attempted: transcript.trajectory.some((entry) => /repair/i.test(JSON.stringify(entry))),
        status: "not_needed",
      }),
      rollback: RealAcpRollbackRecordSchema.parse({
        attempted: transcript.trajectory.some((entry) => /rollback/i.test(JSON.stringify(entry))),
        status: "not_needed",
      }),
      telemetry: jsonClean({
        realConsumer: {
          consumerName: options.readiness.consumerName,
          providerId: options.readiness.providerId,
          protocolBoundary: "ACP over stdio; this is not desktop UI rendering parity evidence.",
          stopReason: transcript.stopReason,
          trajectoryLength: transcript.trajectoryLength,
          counts: transcript.counts,
          protocolMethods: transcript.protocolMethods,
          capabilityEvidence: options.readiness.capabilityEvidence,
          ...(transcript.transcriptPath === undefined ? {} : { transcriptPath: transcript.transcriptPath }),
        },
      }) as JsonValue,
      ...(status === "failed" || status === "error"
        ? { failureReason: failureReasonFromConsumerTranscript(transcript) }
        : {}),
      ...(status === "skipped" ? { skipReason: "real ACP consumer transcript had no write or verifier signal" } : {}),
    };
  } catch (error) {
    if (isAbortLike(error) || input.context.signal.aborted) {
      return cancelledOutput(input, errorMessage(error), options.readiness, []);
    }
    return {
      ...blockedOutput(input, {
        ...options.readiness,
        status: "blocked",
        blockers: [`real ACP consumer protocol runner failed: ${errorMessage(error)}`],
      }),
      status: "error",
      failureReason: errorMessage(error),
    };
  }
};

export const runStdioAcpConsumerProtocol: RealAcpConsumerProtocolRunner = async (input) => {
  const trajectory: RealAcpConsumerTrajectoryEntry[] = [];
  const startedAt = now();
  const child = spawn(input.launch.command, [...input.launch.args], {
    cwd: input.launch.cwd ?? input.workspacePath,
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ...(input.launch.env ?? {}) },
  });
  recordStderr(child, trajectory);
  const exitPromise = new Promise<void>((resolveExit) => {
    child.once("exit", () => resolveExit());
  });
  const abort = () => {
    trajectory.push({ kind: "cancellation", at: now(), phase: "abort-signal" });
    try { child.kill("SIGTERM"); } catch { /* noop */ }
  };
  input.signal.addEventListener("abort", abort, { once: true });

  let stopReason = "ok";
  let sessionId = "";
  try {
    const stream = acp.ndJsonStream(Writable.toWeb(child.stdin), Readable.toWeb(child.stdout));
    const connection = new acp.ClientSideConnection(
      () => createProtocolClient({ workdir: input.workspacePath, trajectory }),
      stream,
    );
    const initialized = await withTimeout(recordProtocolCall(trajectory, "initialize", undefined, {
      protocolVersion: acp.PROTOCOL_VERSION,
    }, () => connection.initialize({
      protocolVersion: acp.PROTOCOL_VERSION,
      clientInfo: {
        name: `bag-real-acp-consumer-${input.readiness.consumerName.toLowerCase()}`,
        version: "1.0.0",
      },
      clientCapabilities: {
        fs: { readTextFile: true, writeTextFile: true },
        terminal: true,
      },
    })), input.timeoutMs, "initialize");
    const session = await withTimeout(recordProtocolCall(trajectory, "session/new", undefined, {
      cwd: input.workspacePath,
      mcpServers: [],
    }, () => connection.newSession({ cwd: input.workspacePath, mcpServers: [] })), input.timeoutMs, "session/new");
    sessionId = session.sessionId;
    const promptResult = await withTimeout(recordProtocolCall(trajectory, "session/prompt", sessionId, {
      text: input.prompt,
      initialized,
    }, () => connection.prompt({
      sessionId,
      prompt: [{ type: "text", text: input.prompt }],
    })), input.timeoutMs, "session/prompt");
    stopReason = typeof promptResult.stopReason === "string" ? promptResult.stopReason : "ok";
  } catch (error) {
    stopReason = isAbortLike(error) || input.signal.aborted ? `cancelled:${errorMessage(error)}` : `error:${errorMessage(error)}`;
    trajectory.push({ kind: "error", at: now(), message: errorMessage(error) });
  } finally {
    input.signal.removeEventListener("abort", abort);
    try { child.kill("SIGTERM"); } catch { /* noop */ }
    await Promise.race([exitPromise, new Promise((resolveExit) => setTimeout(resolveExit, 1_000))]);
    if (child.exitCode == null && child.signalCode == null) {
      try { child.kill("SIGKILL"); } catch { /* noop */ }
    }
  }

  const summary = summarizeConsumerTrajectory({
    stopReason,
    trajectory,
    ...(input.transcriptPath === undefined ? {} : { transcriptPath: input.transcriptPath }),
  });
  if (input.transcriptPath !== undefined) {
    await mkdir(dirname(input.transcriptPath), { recursive: true });
    await writeFile(input.transcriptPath, `${JSON.stringify({
      startedAt,
      completedAt: now(),
      sessionId,
      readiness: input.readiness,
      ...summary,
    }, null, 2)}\n`, "utf8");
  }
  return summary;
};

export const summarizeConsumerTrajectory = (input: {
  stopReason: string;
  trajectory: readonly RealAcpConsumerTrajectoryEntry[];
  transcriptPath?: string;
}): RealAcpConsumerTranscriptSummary => {
  const count = (kind: RealAcpConsumerTrajectoryEntry["kind"]) =>
    input.trajectory.filter((entry) => entry.kind === kind).length;
  const protocolMethods: Record<string, number> = {};
  for (const entry of input.trajectory) {
    if (entry.kind === "protocol_call" && entry.phase === "response") {
      protocolMethods[entry.method] = (protocolMethods[entry.method] ?? 0) + 1;
    }
  }
  return {
    stopReason: input.stopReason,
    trajectoryLength: input.trajectory.length,
    counts: {
      protocolCalls: count("protocol_call"),
      sessionUpdates: count("session_update"),
      fsRead: count("fs_read"),
      fsWrite: count("fs_write"),
      terminalCreate: count("terminal_create"),
      terminalOutput: count("terminal_output"),
      terminalExit: count("terminal_exit"),
      terminalKill: count("terminal_kill"),
      terminalRelease: count("terminal_release"),
      permission: count("permission"),
      agentStderr: count("agent_stderr"),
      cancellation: count("cancellation"),
      error: count("error"),
    },
    protocolMethods,
    trajectory: [...input.trajectory],
    ...(input.transcriptPath === undefined ? {} : { transcriptPath: input.transcriptPath }),
  };
};

const namedSettingsReadiness = (input: {
  providerId: string;
  consumerName: "Glass" | "Zed";
  settingsPath?: string;
  serverKey: string;
  appPath: string;
  clientProfileId: string;
  clientName: string;
}): RealAcpConsumerReadiness => {
  const blockers: string[] = [];
  const app = appMetadata(input.consumerName, input.appPath);
  if (!app.installed) {
    blockers.push(`${input.consumerName} app not found at ${input.appPath}`);
  }
  let launch: RealAcpConsumerLaunchConfig | undefined;
  let settingsEvidence: JsonValue = { settingsPath: input.settingsPath ?? null, serverKey: input.serverKey };
  if (input.settingsPath === undefined || input.settingsPath.trim() === "") {
    blockers.push(`${input.consumerName} ACP settings path is not configured`);
  } else {
    try {
      launch = launchConfigFromSettings(input.settingsPath, input.serverKey);
      settingsEvidence = {
        settingsPath: input.settingsPath,
        serverKey: input.serverKey,
        command: launch.command,
        argCount: launch.args.length,
        hasEnv: Object.keys(launch.env ?? {}).length > 0,
      };
    } catch (error) {
      blockers.push(errorMessage(error));
    }
  }
  return readiness({
    providerId: input.providerId,
    consumerName: input.consumerName,
    blockers,
    ...(launch === undefined ? {} : { launch }),
    app,
    clientProfileId: input.clientProfileId,
    clientName: input.clientName,
    clientVersion: app.version ?? "unknown",
    capabilityEvidence: {
      source: `${input.consumerName} app bundle and ACP settings metadata`,
      app,
      settings: settingsEvidence,
      protocol: "ACP over stdio launch target; desktop UI parity is not asserted",
    },
  });
};

const readiness = (input: {
  providerId: string;
  consumerName: RealAcpNamedConsumer;
  blockers: readonly string[];
  launch?: RealAcpConsumerLaunchConfig;
  app?: RealAcpConsumerAppMetadata;
  clientProfileId?: string;
  clientName: string;
  clientVersion: string;
  capabilityEvidence: JsonValue;
}): RealAcpConsumerReadiness => ({
  providerId: input.providerId,
  consumerName: input.consumerName,
  status: input.blockers.length === 0 && input.launch !== undefined ? "ready" : "blocked",
  blockers: [...input.blockers],
  ...(input.launch === undefined ? {} : { launch: cloneLaunch(input.launch) }),
  ...(input.app === undefined ? {} : { app: input.app }),
  clientMetadata: {
    clientProfileId: input.clientProfileId ?? "client.real-acp.stdio",
    clientName: input.clientName,
    clientVersion: input.clientVersion,
    transport: "stdio",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
      cancellation: true,
      permissions: true,
      transcript: true,
      desktopUiParity: false,
    },
  },
  capabilityEvidence: input.capabilityEvidence,
});

const launchConfigFromSettings = (settingsPath: string, serverKey: string): RealAcpConsumerLaunchConfig => {
  if (!existsSync(settingsPath)) throw new Error(`settings file not found: ${settingsPath}`);
  const settings = parseJsonc(readFileSync(settingsPath, "utf8"));
  const server = findAgentServer(settings, serverKey);
  if (server === undefined) {
    throw new Error(`agent server '${serverKey}' not found in ${settingsPath}`);
  }
  const config = parseAgentServerConfig(server, serverKey);
  return {
    command: config.command,
    args: config.args,
    env: config.env,
  };
};

const findAgentServer = (settings: JsonRecord, serverKey: string): JsonRecord | undefined => {
  const direct = recordAt(settings.agent_servers);
  const fromDirect = direct?.[serverKey];
  if (isJsonRecord(fromDirect)) return fromDirect;
  const namedExamples = recordAt(settings.named_examples);
  for (const example of Object.values(namedExamples ?? {})) {
    if (!isJsonRecord(example)) continue;
    const agentServers = recordAt(example.agent_servers);
    const server = agentServers?.[serverKey];
    if (isJsonRecord(server)) return server;
  }
  const acpServer = settings.acp_server;
  return isJsonRecord(acpServer) ? acpServer : undefined;
};

const parseAgentServerConfig = (raw: JsonRecord, serverKey: string): AgentServerConfig => {
  const command = raw.command;
  const args = raw.args;
  const env = raw.env;
  if (typeof command !== "string" || command.trim() === "") {
    throw new Error(`agent server '${serverKey}' command must be a non-empty string`);
  }
  if (args != null && (!Array.isArray(args) || args.some((arg) => typeof arg !== "string"))) {
    throw new Error(`agent server '${serverKey}' args must be an array of strings`);
  }
  const envRecord: Record<string, string> = {};
  if (env != null) {
    if (!isJsonRecord(env)) throw new Error(`agent server '${serverKey}' env must be an object`);
    for (const [key, value] of Object.entries(env)) {
      if (typeof value !== "string") throw new Error(`agent server '${serverKey}' env.${key} must be a string`);
      envRecord[key] = value;
    }
  }
  return {
    ...(typeof raw.type === "string" ? { type: raw.type } : {}),
    command,
    args: Array.isArray(args) ? [...args] : [],
    env: envRecord,
  };
};

const appMetadata = (consumer: "Glass" | "Zed", appPath: string): RealAcpConsumerAppMetadata => {
  if (!existsSync(appPath)) {
    return { consumer, appPath, installed: false };
  }
  const infoPath = resolve(appPath, "Contents", "Info.plist");
  try {
    const json = execFileSync("plutil", ["-convert", "json", "-o", "-", infoPath], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    const parsed = JSON.parse(json) as JsonRecord;
    return {
      consumer,
      appPath,
      installed: true,
      ...(typeof parsed.CFBundleIdentifier === "string" ? { bundleIdentifier: parsed.CFBundleIdentifier } : {}),
      ...(typeof parsed.CFBundleShortVersionString === "string" ? { version: parsed.CFBundleShortVersionString } : {}),
    };
  } catch {
    return { consumer, appPath, installed: true };
  }
};

type JsonRecord = Record<string, unknown>;

const stripJsonComments = (input: string): string => {
  let output = "";
  let inString = false;
  let quote = "";
  let escaped = false;
  for (let index = 0; index < input.length; index += 1) {
    const char = input[index] ?? "";
    const next = input[index + 1] ?? "";
    if (inString) {
      output += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === quote) {
        inString = false;
        quote = "";
      }
      continue;
    }
    if (char === "\"" || char === "'") {
      inString = true;
      quote = char;
      output += char;
      continue;
    }
    if (char === "/" && next === "/") {
      while (index < input.length && input[index] !== "\n") index += 1;
      output += "\n";
      continue;
    }
    if (char === "/" && next === "*") {
      index += 2;
      while (index < input.length && !(input[index] === "*" && input[index + 1] === "/")) index += 1;
      index += 1;
      continue;
    }
    output += char;
  }
  return output;
};

const parseJsonc = (input: string): JsonRecord => {
  const parsed = JSON.parse(stripJsonComments(input).replace(/,\s*([}\]])/g, "$1")) as unknown;
  if (!isJsonRecord(parsed)) {
    throw new Error("settings root must be an object");
  }
  return parsed;
};

const createProtocolClient = (input: {
  workdir: string;
  trajectory: RealAcpConsumerTrajectoryEntry[];
}) => {
  type TerminalExit = { exitCode: number | null; signal: string | null };
  type TerminalState = {
    command: string;
    args: string[];
    cwd: string;
    output: string;
    outputByteLimit: number | null;
    truncated: boolean;
    proc: ChildProcess | null;
    exitStatus: TerminalExit | null;
    exitPromise: Promise<TerminalExit>;
    resolveExit: (exit: TerminalExit) => void;
  };
  const terminals = new Map<string, TerminalState>();
  return {
    async requestPermission(params: acp.RequestPermissionRequest) {
      const selected = params.options?.find((option) => option.kind === "allow_always")
        ?? params.options?.find((option) => option.kind === "allow_once");
      input.trajectory.push({
        kind: "permission",
        at: now(),
        toolCall: params.toolCall,
        chosen: selected?.optionId ?? "cancelled",
      });
      return selected === undefined
        ? { outcome: { outcome: "cancelled" as const } }
        : { outcome: { outcome: "selected" as const, optionId: selected.optionId } };
    },
    async sessionUpdate(params: acp.SessionNotification) {
      input.trajectory.push({ kind: "session_update", at: now(), update: params.update });
    },
    async readTextFile(params: acp.ReadTextFileRequest): Promise<acp.ReadTextFileResponse> {
      const absolute = resolveClientPath(input.workdir, params.path);
      const raw = readFileSync(absolute, "utf8");
      input.trajectory.push({ kind: "fs_read", at: now(), path: absolute, bytes: Buffer.byteLength(raw) });
      return { content: raw };
    },
    async writeTextFile(params: acp.WriteTextFileRequest): Promise<acp.WriteTextFileResponse> {
      const absolute = resolveClientPath(input.workdir, params.path);
      mkdirSync(dirname(absolute), { recursive: true });
      writeFileSync(absolute, params.content);
      input.trajectory.push({ kind: "fs_write", at: now(), path: absolute, bytes: Buffer.byteLength(params.content) });
      return {};
    },
    async createTerminal(params: acp.CreateTerminalRequest): Promise<acp.CreateTerminalResponse> {
      const terminalId = `term-${Date.now()}-${Math.floor(Math.random() * 1e6).toString(36)}`;
      const cwd = params.cwd == null ? input.workdir : resolveClientPath(input.workdir, params.cwd);
      let resolveExit: (exit: TerminalExit) => void = () => {};
      const exitPromise = new Promise<TerminalExit>((resolvePromise) => {
        resolveExit = resolvePromise;
      });
      const state: TerminalState = {
        command: params.command,
        args: params.args ?? [],
        cwd,
        output: "",
        outputByteLimit: params.outputByteLimit ?? null,
        truncated: false,
        proc: null,
        exitStatus: null,
        exitPromise,
        resolveExit,
      };
      terminals.set(terminalId, state);
      const settle = (exit: TerminalExit) => {
        if (state.exitStatus !== null) return;
        state.exitStatus = exit;
        state.resolveExit(exit);
      };
      const env = { ...process.env };
      for (const variable of params.env ?? []) env[variable.name] = variable.value;
      const proc = spawn(params.command, params.args ?? [], { cwd, env });
      state.proc = proc;
      proc.stdout?.on("data", (chunk: Buffer) => appendTerminalOutput(state, chunk));
      proc.stderr?.on("data", (chunk: Buffer) => appendTerminalOutput(state, chunk));
      proc.once("close", (code, signal) => settle({ exitCode: code, signal: signal == null ? null : String(signal) }));
      proc.once("error", (error) => {
        appendTerminalOutput(state, `[real-acp-consumer terminal error] ${error.message}\n`);
        settle({ exitCode: null, signal: "ERROR" });
      });
      input.trajectory.push({
        kind: "terminal_create",
        at: now(),
        terminalId,
        command: params.command,
        args: params.args ?? [],
      });
      return { terminalId };
    },
    async terminalOutput(params: acp.TerminalOutputRequest): Promise<acp.TerminalOutputResponse> {
      const terminal = terminals.get(params.terminalId);
      const exitStatus = terminal?.exitStatus ?? null;
      input.trajectory.push({
        kind: "terminal_output",
        at: now(),
        terminalId: params.terminalId,
        outputBytes: Buffer.byteLength(terminal?.output ?? ""),
        truncated: terminal?.truncated ?? false,
      });
      const response: acp.TerminalOutputResponse = {
        output: terminal?.output ?? "",
        truncated: terminal?.truncated ?? false,
      };
      if (exitStatus !== null) {
        response.exitStatus = exitStatus;
      }
      return response;
    },
    async waitForTerminalExit(params: acp.WaitForTerminalExitRequest): Promise<acp.WaitForTerminalExitResponse> {
      const terminal = terminals.get(params.terminalId);
      const exit = terminal === undefined ? { exitCode: null, signal: "UNKNOWN_TERMINAL" } : await terminal.exitPromise;
      input.trajectory.push({
        kind: "terminal_exit",
        at: now(),
        terminalId: params.terminalId,
        exitCode: exit.exitCode,
        signal: exit.signal,
        outputBytes: Buffer.byteLength(terminal?.output ?? ""),
      });
      return exit;
    },
    async releaseTerminal(params: acp.ReleaseTerminalRequest): Promise<void> {
      input.trajectory.push({ kind: "terminal_release", at: now(), terminalId: params.terminalId });
      const terminal = terminals.get(params.terminalId);
      if (terminal?.proc != null && terminal.exitStatus === null) {
        try { terminal.proc.kill("SIGTERM"); } catch { /* noop */ }
      }
      terminals.delete(params.terminalId);
    },
    async killTerminal(params: acp.KillTerminalRequest): Promise<void> {
      input.trajectory.push({ kind: "terminal_kill", at: now(), terminalId: params.terminalId });
      const terminal = terminals.get(params.terminalId);
      if (terminal !== undefined) {
        try { terminal.proc?.kill("SIGKILL"); } catch { /* noop */ }
        if (terminal.exitStatus === null) {
          terminal.exitStatus = { exitCode: null, signal: "SIGKILL" };
          terminal.resolveExit(terminal.exitStatus);
        }
      }
    },
  };
};

const appendTerminalOutput = (
  state: {
    output: string;
    outputByteLimit: number | null;
    truncated: boolean;
  },
  chunk: Buffer | string,
): void => {
  const text = Buffer.isBuffer(chunk) ? chunk.toString("utf8") : chunk;
  state.output += text;
  if (state.outputByteLimit === null) return;
  if (state.outputByteLimit <= 0) {
    state.truncated = state.truncated || text !== "";
    state.output = "";
    return;
  }
  if (Buffer.byteLength(state.output) > state.outputByteLimit) {
    const buffer = Buffer.from(state.output, "utf8");
    state.output = buffer.slice(buffer.length - state.outputByteLimit).toString("utf8");
    state.truncated = true;
  }
};

const recordStderr = (
  child: ChildProcessWithoutNullStreams,
  trajectory: RealAcpConsumerTrajectoryEntry[],
): void => {
  child.stderr.on("data", (chunk: Buffer) => {
    for (const line of chunk.toString("utf8").split(/\r?\n/)) {
      if (line.trim() !== "") {
        trajectory.push({ kind: "agent_stderr", at: now(), line });
      }
    }
  });
};

const recordProtocolCall = async <T>(
  trajectory: RealAcpConsumerTrajectoryEntry[],
  method: string,
  sessionId: string | undefined,
  payload: unknown,
  fn: () => Promise<T>,
): Promise<T> => {
  const sessionFields = sessionId === undefined ? {} : { sessionId };
  trajectory.push({ kind: "protocol_call", at: now(), method, phase: "request", ...sessionFields, payload });
  try {
    const result = await fn();
    trajectory.push({ kind: "protocol_call", at: now(), method, phase: "response", ...sessionFields, payload: result });
    return result;
  } catch (error) {
    trajectory.push({ kind: "protocol_call", at: now(), method, phase: "error", ...sessionFields, payload: { error: errorMessage(error) } });
    throw error;
  }
};

const terminalRecordsFromConsumerTranscript = (
  taskId: string,
  transcript: RealAcpConsumerTranscriptSummary,
): RealAcpTerminalRecord[] => transcript.trajectory
  .filter((entry): entry is Extract<RealAcpConsumerTrajectoryEntry, { kind: "terminal_exit" }> => entry.kind === "terminal_exit")
  .map((entry, index) => {
    const create = transcript.trajectory.find((candidate): candidate is Extract<RealAcpConsumerTrajectoryEntry, { kind: "terminal_create" }> =>
      candidate.kind === "terminal_create" && candidate.terminalId === entry.terminalId);
    return RealAcpTerminalRecordSchema.parse({
      commandId: `cmd.${safeId(taskId)}.real-consumer.${index}`,
      command: [create?.command ?? "unknown", ...(create?.args ?? [])],
      status: entry.exitCode === 0 ? "succeeded" : "failed",
      exitCode: entry.exitCode,
      durationMs: 0,
    });
  });

const toolRecordsFromConsumerTranscript = (
  taskId: string,
  transcript: RealAcpConsumerTranscriptSummary,
): RealAcpToolRecord[] => [
  ...transcript.trajectory
    .filter((entry): entry is Extract<RealAcpConsumerTrajectoryEntry, { kind: "fs_read" }> => entry.kind === "fs_read")
    .map((_, index) => RealAcpToolRecordSchema.parse({
      toolCallId: `tool.${safeId(taskId)}.real-consumer.fs-read.${index}`,
      namespace: "acp.fs",
      name: "readTextFile",
      status: "succeeded",
      sideEffectLevel: "read",
    })),
  ...transcript.trajectory
    .filter((entry): entry is Extract<RealAcpConsumerTrajectoryEntry, { kind: "fs_write" }> => entry.kind === "fs_write")
    .map((_, index) => RealAcpToolRecordSchema.parse({
      toolCallId: `tool.${safeId(taskId)}.real-consumer.fs-write.${index}`,
      namespace: "acp.fs",
      name: "writeTextFile",
      status: "succeeded",
      sideEffectLevel: "write",
    })),
];

const statusFromConsumerTranscript = (
  input: RealAcpExecutorTaskInput,
  transcript: RealAcpConsumerTranscriptSummary,
  terminalCommands: readonly RealAcpTerminalRecord[],
): RealAcpTaskOutcomeStatus => {
  if (/cancel/i.test(transcript.stopReason) || transcript.counts.cancellation > 0) return "cancelled";
  if (transcript.stopReason.startsWith("error:") || transcript.counts.error > 0) return "error";
  if (terminalCommands.some((command) => command.exitCode !== 0)) return "failed";
  if (input.task.expectedOutcome.mutation === "no_change") return "passed";
  if (input.task.expectedOutcome.verification.policy === "required" && terminalCommands.length === 0) return "failed";
  if (transcript.counts.fsWrite > 0) return "passed";
  return "failed";
};

const verifierStatusFromConsumerTranscript = (
  input: RealAcpExecutorTaskInput,
  status: RealAcpTaskOutcomeStatus,
  terminalCommands: readonly RealAcpTerminalRecord[],
) => {
  if (input.task.expectedOutcome.verification.policy === "must_skip") return "skipped" as const;
  if (status === "cancelled" || status === "error") return "not_run" as const;
  if (terminalCommands.length === 0) return "not_run" as const;
  if (terminalCommands.some((command) => command.exitCode !== 0)) return "failed" as const;
  return "passed" as const;
};

const blockedOutput = (
  input: RealAcpExecutorTaskInput,
  readinessInput: RealAcpConsumerReadiness,
): RealAcpExecutorTaskOutput => ({
  status: "error",
  route: RealAcpRouteRecordSchema.parse({
    routeId: `route.${safeId(input.task.taskId)}.real-consumer.blocked`,
    selectedMode: "coding",
    reason: "Real ACP consumer execution blocked before launch by readiness checks.",
    confidence: 1,
  }),
  editStrategy: RealAcpEditStrategyRecordSchema.parse({
    strategyId: "edit.none.real-consumer-blocked",
    family: "none",
    selectedBy: "not_applicable",
    reason: "No safe ready real ACP consumer launch target was available.",
  }),
  toolCalls: [],
  terminalCommands: [],
  verifier: RealAcpVerifierRecordSchema.parse({
    status: "not_run",
    policy: input.task.expectedOutcome.verification.policy,
    commandIds: [],
  }),
  repair: { attempted: false, status: "skipped", reason: "real consumer readiness blocked" },
  rollback: { attempted: false, status: "skipped", reason: "real consumer readiness blocked" },
  corrections: [],
  telemetry: jsonClean({
    realConsumer: {
      blocked: true,
      blockerKind: "consumer_readiness",
      consumerName: readinessInput.consumerName,
      providerId: readinessInput.providerId,
      blockers: readinessInput.blockers,
      capabilityEvidence: readinessInput.capabilityEvidence,
      protocolBoundary: "ACP readiness only; desktop UI parity is not asserted",
    },
  }) as JsonValue,
  failureReason: `real ACP consumer readiness blocked: ${readinessInput.blockers.join("; ")}`,
});

const cancelledOutput = (
  input: RealAcpExecutorTaskInput,
  reason: string,
  readinessInput: RealAcpConsumerReadiness,
  trajectory: readonly RealAcpConsumerTrajectoryEntry[],
): RealAcpExecutorTaskOutput => ({
  status: "cancelled",
  route: RealAcpRouteRecordSchema.parse({
    routeId: `route.${safeId(input.task.taskId)}.real-consumer.cancelled`,
    selectedMode: "cancelled",
    reason: "Real ACP consumer execution cancelled.",
  }),
  editStrategy: RealAcpEditStrategyRecordSchema.parse({
    strategyId: "edit.none.real-consumer-cancelled",
    family: "none",
    selectedBy: "not_applicable",
    reason: "Real ACP consumer execution cancelled.",
  }),
  toolCalls: [],
  terminalCommands: [],
  verifier: RealAcpVerifierRecordSchema.parse({
    status: "not_run",
    policy: input.task.expectedOutcome.verification.policy,
    commandIds: [],
  }),
  repair: { attempted: false, status: "skipped", reason: "real consumer cancelled" },
  rollback: { attempted: false, status: "skipped", reason: "real consumer cancelled" },
  corrections: [],
  telemetry: {
    realConsumer: {
      consumerName: readinessInput.consumerName,
      providerId: readinessInput.providerId,
      cancellation: true,
      trajectoryLength: trajectory.length,
    },
  },
  failureReason: reason,
});

const realConsumerPrompt = (input: RealAcpExecutorTaskInput): string => [
  input.task.userPrompt,
  "",
  "This is an isolated real ACP consumer fixture workspace. Do not touch any repository outside this workspace.",
  `Task id: ${input.task.taskId}`,
  `Allowed path prefixes: ${input.task.workspace.allowedPathPrefixes.join(", ")}`,
  input.task.workspace.protectedPaths.length === 0
    ? "Protected paths: (none)"
    : `Protected paths: ${input.task.workspace.protectedPaths.join(", ")}`,
  input.task.expectedOutcome.expectedChangedPaths.length === 0
    ? "Expected changed paths: (none)"
    : `Expected changed paths: ${input.task.expectedOutcome.expectedChangedPaths.join(", ")}`,
  input.task.expectedOutcome.verification.commands.length === 0
    ? "Verifier command: none; explain why verification is skipped if you make no terminal call."
    : `Run this verifier before finishing: ${input.task.expectedOutcome.verification.commands.map((command) => command.join(" ")).join(" && ")}`,
].join("\n");

const failureReasonFromConsumerTranscript = (transcript: RealAcpConsumerTranscriptSummary): string => {
  const error = transcript.trajectory.find((entry): entry is Extract<RealAcpConsumerTrajectoryEntry, { kind: "error" }> =>
    entry.kind === "error");
  return error?.message ?? `real ACP consumer transcript ended with ${transcript.stopReason}`;
};

const assertWorkspaceSafeForRealConsumer = (
  workspacePath: string,
  currentRepoPath: string,
  allowedWorkspaceRoot: string,
): void => {
  const workspace = resolve(workspacePath);
  const allowedRoot = resolve(allowedWorkspaceRoot);
  if (!isInsideOrEqual(workspace, allowedRoot)) {
    throw new Error("real ACP consumer executor refuses workspace outside the configured isolated workspace root");
  }
  if (resolve(workspace) === resolve(currentRepoPath)) {
    throw new Error("real ACP consumer executor refuses to run against the current repository workspace");
  }
};

const resolveClientPath = (workdir: string, path: string): string => {
  const absolute = isAbsolute(path) ? resolve(path) : resolve(workdir, path);
  if (!isInsideOrEqual(absolute, workdir)) {
    throw new Error(`ACP client file operation escapes fixture workspace: ${path}`);
  }
  return absolute;
};

const withTimeout = async <T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> => {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(new Error(`${label} timeout after ${timeoutMs}ms`)), timeoutMs);
        timeout.unref?.();
      }),
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
};

const cloneLaunch = (launch: RealAcpConsumerLaunchConfig): RealAcpConsumerLaunchConfig => ({
  command: launch.command,
  args: [...launch.args],
  ...(launch.cwd === undefined ? {} : { cwd: launch.cwd }),
  ...(launch.env === undefined ? {} : { env: { ...launch.env } }),
});

const recordAt = (value: unknown): JsonRecord | undefined => isJsonRecord(value) ? value : undefined;

const isJsonRecord = (value: unknown): value is JsonRecord =>
  value != null && typeof value === "object" && !Array.isArray(value);

const isInsideOrEqual = (candidatePath: string, rootPath: string): boolean => {
  const relativePath = relative(resolve(rootPath), resolve(candidatePath));
  return relativePath === "" || (!relativePath.startsWith("..") && !relativePath.includes(`..${sep}`));
};

const jsonClean = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(jsonClean).filter((entry) => entry !== undefined);
  if (value != null && typeof value === "object") {
    const output: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) {
      const cleaned = jsonClean(entry);
      if (cleaned !== undefined) output[key] = cleaned;
    }
    return output;
  }
  return value === undefined ? undefined : value;
};

const isAbortLike = (error: unknown): boolean =>
  error instanceof Error && /aborted|abort|cancelled|canceled/i.test(error.message);

const now = (): string => new Date().toISOString();

const safeId = (value: string): string => value.replace(/[^A-Za-z0-9._:-]+/g, "-");

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);
