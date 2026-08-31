#!/usr/bin/env -S node --loader=tsx
import { spawn, type ChildProcess, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { Readable, Writable } from "node:stream";
import { fileURLToPath, pathToFileURL } from "node:url";
import process from "node:process";

import * as acp from "@agentclientprotocol/sdk";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

type Json = unknown;

export type HeadlessTerminalMode = "real" | "stub";

export type HeadlessAcpClientProfileId = "minimal" | "capable";

export type HeadlessAcpConsumerCapabilityProfile = {
  profileId: HeadlessAcpClientProfileId;
  description: string;
  clientInfo: {
    name: string;
    version: string;
  };
  clientCapabilities: acp.ClientCapabilities;
  filesystem: {
    readTextFile: boolean;
    writeTextFile: boolean;
  };
  terminal: {
    create: boolean;
    output: boolean;
    waitForExit: boolean;
    kill: boolean;
    release: boolean;
  };
  richToolContent: {
    diff: boolean;
    terminal: boolean;
    textFallback: boolean;
  };
  artifactLinks: {
    fileLocations: boolean;
    resourceLinks: boolean;
  };
  slashCommands: {
    availableCommandsUpdate: boolean;
    textSlashPrompts: boolean;
  };
  permissions: {
    requestPermission: boolean;
    yoloAutoAllow: boolean;
    safeAutoReject: boolean;
  };
  promptContent: {
    text: boolean;
    image: boolean;
    resource: boolean;
  };
  unsupported: {
    images: boolean;
    resources: boolean;
    nes: boolean;
    provider: boolean;
    forkSession: boolean;
  };
};

const HEADLESS_ACP_CAPABILITY_PROFILES = {
  minimal: {
    profileId: "minimal",
    description: "Text-only ACP consumer: session updates, slash prompts, permissions, artifact file locations, and text fallbacks.",
    clientInfo: { name: "bag-headless-acp-minimal-runner", version: "1.0.0" },
    clientCapabilities: {
      fs: { readTextFile: false, writeTextFile: false },
      terminal: false,
    },
    filesystem: { readTextFile: false, writeTextFile: false },
    terminal: { create: false, output: false, waitForExit: false, kill: false, release: false },
    richToolContent: { diff: false, terminal: false, textFallback: true },
    artifactLinks: { fileLocations: true, resourceLinks: false },
    slashCommands: { availableCommandsUpdate: true, textSlashPrompts: true },
    permissions: { requestPermission: true, yoloAutoAllow: true, safeAutoReject: true },
    promptContent: { text: true, image: false, resource: false },
    unsupported: { images: true, resources: true, nes: true, provider: true, forkSession: true },
  },
  capable: {
    profileId: "capable",
    description: "Full deterministic ACP consumer used for coding transcripts: filesystem, terminal, rich diff/terminal content, permissions, slash prompts, and artifact file locations.",
    clientInfo: { name: "bag-headless-acp-runner", version: "1.0.0" },
    clientCapabilities: {
      fs: { readTextFile: true, writeTextFile: true },
      terminal: true,
    },
    filesystem: { readTextFile: true, writeTextFile: true },
    terminal: { create: true, output: true, waitForExit: true, kill: true, release: true },
    richToolContent: { diff: true, terminal: true, textFallback: true },
    artifactLinks: { fileLocations: true, resourceLinks: false },
    slashCommands: { availableCommandsUpdate: true, textSlashPrompts: true },
    permissions: { requestPermission: true, yoloAutoAllow: true, safeAutoReject: true },
    promptContent: { text: true, image: false, resource: false },
    unsupported: { images: true, resources: true, nes: true, provider: true, forkSession: true },
  },
} satisfies Record<HeadlessAcpClientProfileId, HeadlessAcpConsumerCapabilityProfile>;

export const headlessAcpConsumerCapabilityProfile = (
  profileId: HeadlessAcpClientProfileId = "capable",
): HeadlessAcpConsumerCapabilityProfile => {
  const profile = HEADLESS_ACP_CAPABILITY_PROFILES[profileId];
  return {
    ...profile,
    clientInfo: { ...profile.clientInfo },
    clientCapabilities: {
      fs: { ...profile.clientCapabilities.fs },
      terminal: profile.clientCapabilities.terminal,
    },
    filesystem: { ...profile.filesystem },
    terminal: { ...profile.terminal },
    richToolContent: { ...profile.richToolContent },
    artifactLinks: { ...profile.artifactLinks },
    slashCommands: { ...profile.slashCommands },
    permissions: { ...profile.permissions },
    promptContent: { ...profile.promptContent },
    unsupported: { ...profile.unsupported },
  };
};

type CliArgs = {
  task: string;
  workdir: string;
  outFile: string;
  yolo: boolean;
  promptTimeoutMs: number;
  cancelAfterMs: number | null;
  terminalMode: HeadlessTerminalMode;
  clientProfile: HeadlessAcpClientProfileId;
  resumeCheck: boolean;
  closeSession: boolean;
  mode: "coding" | "tools" | "dag-tools" | "auto";
};

const parseArgs = (argv: string[]): CliArgs => {
  const out: Partial<CliArgs> = {
    yolo: true,
    promptTimeoutMs: 600_000,
    cancelAfterMs: null,
    terminalMode: "real",
    clientProfile: "capable",
    resumeCheck: true,
    closeSession: false,
    mode: "coding",
  };
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] ?? "";
    if (arg === "--workdir" || arg === "-w") {
      out.workdir = argv[++i] ?? "";
    } else if (arg === "--out" || arg === "-o") {
      out.outFile = argv[++i] ?? "";
    } else if (arg === "--no-yolo") {
      out.yolo = false;
    } else if (arg === "--timeout-ms") {
      out.promptTimeoutMs = Number(argv[++i]);
    } else if (arg === "--cancel-after-ms") {
      const value = Number(argv[++i]);
      if (!Number.isFinite(value) || value < 0) {
        throw new Error("--cancel-after-ms must be a non-negative number");
      }
      out.cancelAfterMs = value;
    } else if (arg === "--terminal-mode") {
      const value = argv[++i];
      if (value !== "real" && value !== "stub") {
        throw new Error("--terminal-mode must be real or stub");
      }
      out.terminalMode = value;
    } else if (arg === "--client-profile") {
      const value = argv[++i];
      if (value !== "minimal" && value !== "capable") {
        throw new Error("--client-profile must be minimal or capable");
      }
      out.clientProfile = value;
    } else if (arg === "--no-resume-check") {
      out.resumeCheck = false;
    } else if (arg === "--close-session") {
      out.closeSession = true;
    } else if (arg === "--mode") {
      const value = argv[++i];
      if (value !== "coding" && value !== "tools" && value !== "dag-tools" && value !== "auto") {
        throw new Error("--mode must be coding, tools, dag-tools, or auto");
      }
      out.mode = value;
    } else if (arg.startsWith("--")) {
      throw new Error(`unknown flag: ${arg}`);
    } else {
      positional.push(arg);
    }
  }
  const task = positional.join(" ").trim();
  if (task === "") {
    throw new Error(
      'usage: bag_acp_run.ts "<task>" [--workdir DIR] [--out FILE] [--no-yolo] [--timeout-ms N] [--cancel-after-ms N] [--terminal-mode real|stub] [--client-profile minimal|capable] [--no-resume-check] [--close-session]',
    );
  }
  const workdir = resolve(out.workdir ?? process.cwd());
  const outFile = out.outFile != null
    ? (isAbsolute(out.outFile) ? out.outFile : resolve(workdir, out.outFile))
    : resolve(workdir, ".bag", "acp-runs", `run-${Date.now()}.json`);
  return {
    task,
    workdir,
    outFile,
    yolo: out.yolo ?? true,
    promptTimeoutMs: out.promptTimeoutMs ?? 600_000,
    cancelAfterMs: out.cancelAfterMs ?? null,
    terminalMode: out.terminalMode ?? "real",
    clientProfile: out.clientProfile ?? "capable",
    resumeCheck: out.resumeCheck ?? true,
    closeSession: out.closeSession ?? false,
    mode: out.mode ?? "coding",
  };
};

export type HeadlessAcpProtocolMethod =
  | "initialize"
  | "session/new"
  | "session/list"
  | "session/resume"
  | "session/prompt"
  | "session/cancel"
  | "session/close";

export type TrajectoryEntry =
  | {
      kind: "protocol_call";
      at: string;
      method: HeadlessAcpProtocolMethod;
      phase: "request" | "response" | "error";
      sessionId?: string;
      payload?: Json;
    }
  | { kind: "session_update"; at: string; update: Json }
  | { kind: "permission"; at: string; toolCall: Json; chosen: string }
  | { kind: "fs_read"; at: string; path: string; bytes: number }
  | { kind: "fs_write"; at: string; path: string; bytes: number }
  | { kind: "terminal_create"; at: string; terminalId: string; command: string; args: string[] }
  | { kind: "terminal_exit"; at: string; terminalId: string; exitCode: number | null; signal: string | null; outputBytes: number }
  | { kind: "terminal_output"; at: string; terminalId: string; outputBytes: number; truncated: boolean }
  | { kind: "terminal_kill"; at: string; terminalId: string }
  | { kind: "terminal_release"; at: string; terminalId: string }
  | { kind: "agent_stderr"; at: string; line: string };

export type HeadlessAcpRegressionScenarioId =
  | "greeting-no-side-effect"
  | "read-only-report"
  | "coding-run"
  | "edit-preview-write"
  | "terminal-verification"
  | "rejected-permission"
  | "cancellation"
  | "metrics-traces"
  | "maintenance-isolation";

export type HeadlessAcpRegressionScenario = {
  id: HeadlessAcpRegressionScenarioId;
  prompt: string;
  profileId: HeadlessAcpClientProfileId;
  expected: {
    requiredKinds: TrajectoryEntry["kind"][];
    forbiddenKinds: TrajectoryEntry["kind"][];
    updateSignals: string[];
  };
  compatibilityBoundary: string;
};

export const HEADLESS_ACP_REGRESSION_SCENARIOS: readonly HeadlessAcpRegressionScenario[] = [
  {
    id: "greeting-no-side-effect",
    prompt: "hello",
    profileId: "capable",
    expected: {
      requiredKinds: ["protocol_call", "session_update"],
      forbiddenKinds: ["fs_read", "fs_write", "terminal_create", "permission"],
      updateSignals: ["agent_message_chunk"],
    },
    compatibilityBoundary: "Conversation help must not read files, write files, run terminals, or expose maintenance internals.",
  },
  {
    id: "read-only-report",
    prompt: "/plan summarize repository state without edits",
    profileId: "capable",
    expected: {
      requiredKinds: ["protocol_call", "session_update"],
      forbiddenKinds: ["fs_write", "terminal_create", "permission"],
      updateSignals: ["current_mode_update", "plan", "tool_call", "tool_call_update"],
    },
    compatibilityBoundary: "Planning/reporting may stream read/think progress and artifacts, but it must not commit client file writes or terminal commands.",
  },
  {
    id: "coding-run",
    prompt: "/run make a bounded coding change",
    profileId: "capable",
    expected: {
      requiredKinds: ["protocol_call", "session_update"],
      forbiddenKinds: [],
      updateSignals: ["current_mode_update", "plan"],
    },
    compatibilityBoundary: "Run mode must be explicit through slash command, manual mode, or Auto routing and visible as coding progress.",
  },
  {
    id: "edit-preview-write",
    prompt: "preview and write a changed file",
    profileId: "capable",
    expected: {
      requiredKinds: ["session_update", "fs_write"],
      forbiddenKinds: [],
      updateSignals: ["tool_call", "tool_call_update"],
    },
    compatibilityBoundary: "Edits are previewed before the final ACP whole-file write transport commits content.",
  },
  {
    id: "terminal-verification",
    prompt: "run verification",
    profileId: "capable",
    expected: {
      requiredKinds: ["session_update", "terminal_create", "terminal_output", "terminal_exit"],
      forbiddenKinds: [],
      updateSignals: ["tool_call", "tool_call_update"],
    },
    compatibilityBoundary: "Verification uses ACP terminal lifecycle calls where the consumer exposes terminal support.",
  },
  {
    id: "rejected-permission",
    prompt: "/safe then reject a mutating tool",
    profileId: "capable",
    expected: {
      requiredKinds: ["session_update", "permission"],
      forbiddenKinds: ["fs_write"],
      updateSignals: ["tool_call", "tool_call_update"],
    },
    compatibilityBoundary: "Safe-mode rejection must fail closed, skip the side effect, and surface a traceable failed tool update.",
  },
  {
    id: "cancellation",
    prompt: "cancel an active prompt",
    profileId: "capable",
    expected: {
      requiredKinds: ["protocol_call", "session_update"],
      forbiddenKinds: [],
      updateSignals: ["current_mode_update"],
    },
    compatibilityBoundary: "Cancellation aborts the active prompt and leaves the session reusable; terminal cleanup is covered separately.",
  },
  {
    id: "metrics-traces",
    prompt: "/metrics and /traces",
    profileId: "capable",
    expected: {
      requiredKinds: ["protocol_call", "session_update"],
      forbiddenKinds: ["fs_write", "terminal_create"],
      updateSignals: ["agent_message_chunk"],
    },
    compatibilityBoundary: "Telemetry inspection is text/artifact-path oriented and should stay compact in normal ACP clients.",
  },
  {
    id: "maintenance-isolation",
    prompt: "/maintenance status",
    profileId: "capable",
    expected: {
      requiredKinds: ["protocol_call", "session_update"],
      forbiddenKinds: ["fs_write", "terminal_create"],
      updateSignals: ["plan", "tool_call", "tool_call_update"],
    },
    compatibilityBoundary: "Maintenance commands are hidden from normal suggestions and run as bounded inspections unless explicitly requested.",
  },
];

const now = (): string => new Date().toISOString();

const errorPayload = (error: unknown): Record<string, string> => ({
  message: error instanceof Error ? error.message : String(error),
  name: error instanceof Error ? error.name : "Error",
});

export const recordHeadlessAcpProtocolCall = (
  trajectory: TrajectoryEntry[],
  input: {
    method: HeadlessAcpProtocolMethod;
    phase: "request" | "response" | "error";
    sessionId?: string;
    payload?: Json;
  },
): void => {
  trajectory.push({ kind: "protocol_call", at: now(), ...input });
};

const recordAgentCall = async <T>(
  trajectory: TrajectoryEntry[],
  method: HeadlessAcpProtocolMethod,
  sessionId: string | undefined,
  payload: Json,
  fn: () => Promise<T>,
): Promise<T> => {
  const sessionFields = sessionId === undefined ? {} : { sessionId };
  recordHeadlessAcpProtocolCall(trajectory, { method, phase: "request", ...sessionFields, payload });
  try {
    const result = await fn();
    recordHeadlessAcpProtocolCall(trajectory, { method, phase: "response", ...sessionFields, payload: result });
    return result;
  } catch (error) {
    recordHeadlessAcpProtocolCall(trajectory, { method, phase: "error", ...sessionFields, payload: errorPayload(error) });
    throw error;
  }
};

export const summarizeHeadlessAcpTranscript = (trajectory: TrajectoryEntry[]) => {
  const count = (kind: TrajectoryEntry["kind"]) => trajectory.filter((entry) => entry.kind === kind).length;
  const protocolMethods: Record<string, number> = {};
  for (const entry of trajectory) {
    if (entry.kind === "protocol_call" && entry.phase === "response") {
      protocolMethods[entry.method] = (protocolMethods[entry.method] ?? 0) + 1;
    }
  }
  return {
    entries: trajectory.length,
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
    },
    protocolMethods,
  };
};

type TerminalExitStatus = { exitCode: number | null; signal: string | null };

type TerminalState = {
  terminalId: string;
  command: string;
  args: string[];
  cwd: string | null;
  buffer: string;
  byteLimit: number | null;
  truncated: boolean;
  proc: ChildProcess | null;
  exitStatus: TerminalExitStatus | null;
  exitPromise: Promise<TerminalExitStatus>;
};

const createDeferredExit = () => {
  let settled = false;
  let resolveExit: (exit: TerminalExitStatus) => void = () => {};
  const promise = new Promise<TerminalExitStatus>((resolvePromise) => {
    resolveExit = resolvePromise;
  });
  return {
    promise,
    resolve: (exit: TerminalExitStatus) => {
      if (settled) return;
      settled = true;
      resolveExit(exit);
    },
  };
};

const appendTerminalOutput = (state: TerminalState, chunk: Buffer | string) => {
  const text = Buffer.isBuffer(chunk) ? chunk.toString("utf8") : chunk;
  state.buffer += text;
  if (state.byteLimit == null) return;
  if (state.byteLimit <= 0) {
    state.truncated = state.truncated || text !== "";
    state.buffer = "";
    return;
  }
  if (Buffer.byteLength(state.buffer) > state.byteLimit) {
    const buf = Buffer.from(state.buffer, "utf8");
    state.buffer = buf.slice(buf.length - state.byteLimit).toString("utf8");
    state.truncated = true;
  }
};

export const createHeadlessAcpClientFixture = (input: {
  workdir: string;
  trajectory: TrajectoryEntry[];
  yolo: boolean;
  terminalMode?: HeadlessTerminalMode;
  clientProfile?: HeadlessAcpClientProfileId;
  capabilities?: HeadlessAcpConsumerCapabilityProfile;
}) => {
  const terminals = new Map<string, TerminalState>();
  const terminalMode = input.terminalMode ?? "real";
  const capabilities = input.capabilities ?? headlessAcpConsumerCapabilityProfile(input.clientProfile ?? "capable");
  const unsupported = (method: string): never => {
    throw new Error(`headless ACP client profile ${capabilities.profileId} does not support ${method}`);
  };

  const client = {
    async requestPermission(params: acp.RequestPermissionRequest) {
      if (!capabilities.permissions.requestPermission) unsupported("request_permission");
      const options = params.options ?? [];
      const auto = input.yolo
        ? (options.find((o) => o.kind === "allow_always") ?? options.find((o) => o.kind === "allow_once"))
        : null;
      if (auto != null) {
        input.trajectory.push({ kind: "permission", at: now(), toolCall: params.toolCall, chosen: auto.optionId });
        return { outcome: { outcome: "selected" as const, optionId: auto.optionId } };
      }
      const reject = options.find((o) => o.kind === "reject_once") ?? options.find((o) => o.kind === "reject_always");
      if (reject != null) {
        input.trajectory.push({ kind: "permission", at: now(), toolCall: params.toolCall, chosen: reject.optionId });
        return { outcome: { outcome: "selected" as const, optionId: reject.optionId } };
      }
      return { outcome: { outcome: "cancelled" as const } };
    },

    async sessionUpdate(params: acp.SessionNotification) {
      input.trajectory.push({ kind: "session_update", at: now(), update: params.update });
    },

    async readTextFile(params: acp.ReadTextFileRequest): Promise<acp.ReadTextFileResponse> {
      if (!capabilities.filesystem.readTextFile) unsupported("fs/read_text_file");
      const absolute = isAbsolute(params.path) ? params.path : resolve(input.workdir, params.path);
      const raw = readFileSync(absolute, "utf8");
      let content = raw;
      if (typeof params.line === "number" || typeof params.limit === "number") {
        const lines = raw.split("\n");
        const start = Math.max(0, (params.line ?? 1) - 1);
        const end = params.limit != null ? start + params.limit : lines.length;
        content = lines.slice(start, end).join("\n");
      }
      input.trajectory.push({ kind: "fs_read", at: now(), path: absolute, bytes: Buffer.byteLength(content) });
      return { content };
    },

    async writeTextFile(params: acp.WriteTextFileRequest): Promise<acp.WriteTextFileResponse> {
      if (!capabilities.filesystem.writeTextFile) unsupported("fs/write_text_file");
      const absolute = isAbsolute(params.path) ? params.path : resolve(input.workdir, params.path);
      mkdirSync(dirname(absolute), { recursive: true });
      writeFileSync(absolute, params.content);
      input.trajectory.push({ kind: "fs_write", at: now(), path: absolute, bytes: Buffer.byteLength(params.content) });
      return {};
    },

    async createTerminal(params: acp.CreateTerminalRequest): Promise<acp.CreateTerminalResponse> {
      if (!capabilities.terminal.create) unsupported("terminal/create");
      const cwd = params.cwd ?? input.workdir;
      const terminalId = `term-${Date.now()}-${Math.floor(Math.random() * 1e6).toString(36)}`;
      const deferred = createDeferredExit();
      const state: TerminalState = {
        terminalId,
        command: params.command,
        args: params.args ?? [],
        cwd,
        buffer: "",
        byteLimit: params.outputByteLimit ?? null,
        truncated: false,
        proc: null,
        exitStatus: null,
        exitPromise: deferred.promise,
      };
      const settle = (exit: TerminalExitStatus) => {
        state.exitStatus = exit;
        deferred.resolve(exit);
      };
      if (terminalMode === "stub") {
        appendTerminalOutput(state, `[headless-acp stub terminal] ${params.command} ${(params.args ?? []).join(" ")}\n`);
        settle({ exitCode: 0, signal: null });
      } else {
        const env = { ...process.env };
        for (const v of params.env ?? []) env[v.name] = v.value;
        const proc = spawn(params.command, params.args ?? [], { cwd, env });
        state.proc = proc;
        proc.stdout?.on("data", (chunk: Buffer) => appendTerminalOutput(state, chunk));
        proc.stderr?.on("data", (chunk: Buffer) => appendTerminalOutput(state, chunk));
        proc.once("close", (code, signal) => settle({ exitCode: code, signal: signal == null ? null : String(signal) }));
        proc.once("error", (error) => {
          appendTerminalOutput(state, `[headless-acp terminal error] ${error.message}\n`);
          settle({ exitCode: null, signal: "ERROR" });
        });
      }
      terminals.set(terminalId, state);
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
      if (!capabilities.terminal.output) unsupported("terminal/output");
      const state = terminals.get(params.terminalId);
      if (state == null) return { output: "", truncated: false };
      const exitStatus = state.exitStatus ?? (
        state.proc != null && (state.proc.exitCode != null || state.proc.signalCode != null)
          ? { exitCode: state.proc.exitCode, signal: state.proc.signalCode == null ? null : String(state.proc.signalCode) }
          : null
      );
      input.trajectory.push({
        kind: "terminal_output",
        at: now(),
        terminalId: state.terminalId,
        outputBytes: Buffer.byteLength(state.buffer),
        truncated: state.truncated,
      });
      const response: acp.TerminalOutputResponse = {
        output: state.buffer,
        truncated: state.truncated,
      };
      if (exitStatus != null) {
        response.exitStatus = exitStatus;
      }
      return response;
    },

    async waitForTerminalExit(params: acp.WaitForTerminalExitRequest): Promise<acp.WaitForTerminalExitResponse> {
      if (!capabilities.terminal.waitForExit) unsupported("terminal/wait_for_exit");
      const state = terminals.get(params.terminalId);
      if (state == null) return { exitCode: null, signal: null };
      const exit = await state.exitPromise;
      input.trajectory.push({
        kind: "terminal_exit",
        at: now(),
        terminalId: state.terminalId,
        exitCode: exit.exitCode,
        signal: exit.signal,
        outputBytes: Buffer.byteLength(state.buffer),
      });
      return exit;
    },

    async releaseTerminal(params: acp.ReleaseTerminalRequest): Promise<void> {
      if (!capabilities.terminal.release) unsupported("terminal/release");
      const state = terminals.get(params.terminalId);
      if (state == null) return;
      input.trajectory.push({ kind: "terminal_release", at: now(), terminalId: state.terminalId });
      if (state.proc != null && state.proc.exitCode == null && state.proc.signalCode == null) {
        try { state.proc.kill("SIGTERM"); } catch { /* noop */ }
      }
      terminals.delete(params.terminalId);
    },

    async killTerminal(params: acp.KillTerminalRequest): Promise<void> {
      if (!capabilities.terminal.kill) unsupported("terminal/kill");
      const state = terminals.get(params.terminalId);
      if (state == null) return;
      input.trajectory.push({ kind: "terminal_kill", at: now(), terminalId: state.terminalId });
      if (state.proc == null) {
        state.exitStatus = { exitCode: null, signal: "SIGKILL" };
        return;
      }
      try { state.proc.kill("SIGKILL"); } catch { /* noop */ }
    },
  };

  return { client, terminals, capabilities };
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  const trajectory: TrajectoryEntry[] = [];
  const capabilities = headlessAcpConsumerCapabilityProfile(args.clientProfile);

  const repoRoot = resolve(__dirname, "..");
  const tsxBin = resolve(repoRoot, "node_modules/.bin/tsx");
  const agentEntry = resolve(repoRoot, "src/index.ts");
  const envFile = resolve(repoRoot, ".env");
  // Spawn bag acp with cwd=repo root so it loads our bag.config.json (master+local Anthropic).
  // The actual workspace is set per-session via newSession({ cwd: args.workdir }).
  const agent: ChildProcessWithoutNullStreams = spawn(
    tsxBin,
    ["--env-file=" + envFile, agentEntry, "acp"],
    { stdio: ["pipe", "pipe", "pipe"], cwd: repoRoot, env: process.env },
  );
  agent.stderr.on("data", (chunk) => {
    const text = chunk.toString("utf8");
    process.stderr.write(`[bag-acp] ${text}`);
    for (const line of text.split(/\r?\n/)) {
      if (line.trim() !== "") trajectory.push({ kind: "agent_stderr", at: now(), line });
    }
  });
  agent.on("exit", (code, signal) => {
    process.stderr.write(`[bag-acp] exited code=${code} signal=${signal ?? ""}\n`);
  });

  const input = Writable.toWeb(agent.stdin);
  const output = Readable.toWeb(agent.stdout);
  const stream = acp.ndJsonStream(input, output);
  const { client } = createHeadlessAcpClientFixture({
    workdir: args.workdir,
    trajectory,
    yolo: args.yolo,
    terminalMode: args.terminalMode,
    capabilities,
  });
  const connection = new acp.ClientSideConnection(() => client, stream);

  const startedAt = now();
  let stopReason = "ok";
  let initResult: Awaited<ReturnType<typeof connection.initialize>> | null = null;
  let sessionId = "";
  try {
    const initializeParams = {
      protocolVersion: acp.PROTOCOL_VERSION,
      clientInfo: capabilities.clientInfo,
      clientCapabilities: capabilities.clientCapabilities,
    };
    initResult = await recordAgentCall(trajectory, "initialize", undefined, initializeParams, () =>
      connection.initialize(initializeParams),
    );
    const newSessionParams = { cwd: args.workdir, mcpServers: [] };
    const session = await recordAgentCall(trajectory, "session/new", undefined, newSessionParams, () =>
      connection.newSession(newSessionParams),
    );
    sessionId = session.sessionId;
    if (args.resumeCheck) {
      await recordAgentCall(trajectory, "session/list", sessionId, {}, () => connection.listSessions({}));
      await recordAgentCall(
        trajectory,
        "session/resume",
        sessionId,
        { sessionId, cwd: args.workdir, mcpServers: [] },
        () => connection.resumeSession({ sessionId, cwd: args.workdir, mcpServers: [] }),
      );
    }

    const sendPrompt = async (text: string, timeoutMs: number) => {
      const promptParams = { sessionId, prompt: [{ type: "text" as const, text }] };
      return recordAgentCall(trajectory, "session/prompt", sessionId, { text, timeoutMs }, () =>
        Promise.race([
          connection.prompt(promptParams),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`prompt timeout after ${timeoutMs}ms`)), timeoutMs),
          ),
        ]),
      );
    };
    if (args.yolo) {
      await sendPrompt("/yolo", 30_000);
    }
    let cancelTimer: ReturnType<typeof setTimeout> | null = null;
    if (args.cancelAfterMs != null) {
      cancelTimer = setTimeout(() => {
        void recordAgentCall(trajectory, "session/cancel", sessionId, { cancelAfterMs: args.cancelAfterMs }, () =>
          connection.cancel({ sessionId }),
        ).catch((error) => {
          process.stderr.write(`[bag-acp] cancel failed: ${error instanceof Error ? error.message : String(error)}\n`);
        });
      }, args.cancelAfterMs);
    }
    const slashCmd =
      args.mode === "tools"
        ? "/run-tools"
        : args.mode === "dag-tools"
          ? "/run-dag-tools"
          : args.mode === "auto"
            ? "/run-auto"
            : "/run";
    const promptResult = await sendPrompt(`${slashCmd} ${args.task}`, args.promptTimeoutMs).finally(() => {
      if (cancelTimer != null) clearTimeout(cancelTimer);
    });
    stopReason = promptResult.stopReason;
  } catch (error) {
    stopReason = error instanceof Error ? `error:${error.message}` : `error:${String(error)}`;
  } finally {
    if (args.closeSession && sessionId !== "") {
      try {
        await recordAgentCall(trajectory, "session/close", sessionId, { sessionId }, () =>
          connection.closeSession({ sessionId }),
        );
      } catch (error) {
        process.stderr.write(`[bag-acp] close failed: ${error instanceof Error ? error.message : String(error)}\n`);
      }
    }
    try { agent.kill("SIGTERM"); } catch { /* noop */ }
  }

  const completedAt = now();
  const summary = {
    task: args.task,
    workdir: args.workdir,
    yolo: args.yolo,
    clientProfile: args.clientProfile,
    startedAt,
    completedAt,
    stopReason,
    sessionId,
    initResult,
    consumerCapabilities: capabilities,
    regressionScenarios: HEADLESS_ACP_REGRESSION_SCENARIOS,
    capabilityAssumptions: {
      fs: capabilities.filesystem,
      terminal: { ...capabilities.terminal, mode: args.terminalMode },
      richToolContent: capabilities.richToolContent,
      artifactLinks: capabilities.artifactLinks,
      slashCommands: capabilities.slashCommands,
      permissions: args.yolo ? "auto-select allow_always/allow_once" : "auto-select reject_once/reject_always or cancelled",
      unsupported: capabilities.unsupported,
      transcript: "protocol calls, streamed session updates, permissions, filesystem operations, terminal lifecycle, and agent stderr",
    },
    trajectoryLength: trajectory.length,
    counts: summarizeHeadlessAcpTranscript(trajectory).counts,
    protocolMethods: summarizeHeadlessAcpTranscript(trajectory).protocolMethods,
    trajectory,
  };
  mkdirSync(dirname(args.outFile), { recursive: true });
  writeFileSync(args.outFile, JSON.stringify(summary, null, 2));
  console.error(`[bag-acp] wrote ${args.outFile} (${trajectory.length} entries, stopReason=${stopReason})`);
  console.log(JSON.stringify({ ...summary, trajectory: undefined, _trajectoryFile: args.outFile }, null, 2));

  // Force-tear-down the BAG ACP child process. SIGTERM was sent in the finally
  // block above, but the agent may be mid-sessionUpdate and not respond
  // promptly; in that case node's event loop stays alive (web stream wrappers
  // hold a ref to the child stdio) and the harbor wrapper hangs after
  // BAG_TASK_COMPLETE. Wait briefly for graceful exit, then SIGKILL, then exit
  // the driver process explicitly so node tears the streams down.
  const exitCode = stopReason.startsWith("error:") ? 1 : 0;
  const childExited = new Promise<void>((resolveExit) => {
    if (agent.exitCode != null || agent.signalCode != null) {
      resolveExit();
      return;
    }
    agent.once("exit", () => resolveExit());
  });
  await Promise.race([
    childExited,
    new Promise<void>((resolveTimeout) => setTimeout(() => resolveTimeout(), 2_000)),
  ]);
  if (agent.exitCode == null && agent.signalCode == null) {
    try { agent.kill("SIGKILL"); } catch { /* noop */ }
  }
  // Drop refs so the event loop has nothing to wait on. process.exit forces
  // teardown even if some web stream wrapper is still pending.
  try { agent.stdin?.destroy(); } catch { /* noop */ }
  try { agent.stdout?.destroy(); } catch { /* noop */ }
  try { agent.stderr?.destroy(); } catch { /* noop */ }
  process.exit(exitCode);
};

const directRun = process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;

if (directRun) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  });
}
