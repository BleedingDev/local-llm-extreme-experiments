#!/usr/bin/env -S node --loader=tsx
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { Readable, Writable } from "node:stream";
import { fileURLToPath, pathToFileURL } from "node:url";
import process from "node:process";

import * as acp from "@agentclientprotocol/sdk";

import {
  createHeadlessAcpClientFixture,
  headlessAcpConsumerCapabilityProfile,
  summarizeHeadlessAcpTranscript,
  type HeadlessAcpProtocolMethod,
  type TrajectoryEntry,
} from "./bag_acp_run";

type JsonRecord = Record<string, unknown>;

type CliArgs = {
  settingsPath: string;
  serverKey: string;
  workdir: string;
  outFile: string;
  prompt: string;
  timeoutMs: number;
  skipHandshake: boolean;
};

type BundleEvidence = {
  consumer: "Glass" | "Zed";
  appPath: string;
  installed: boolean;
  bundleIdentifier: string | null;
  version: string | null;
};

type AgentServerConfig = {
  type: string | null;
  command: string;
  args: string[];
  env: Record<string, string>;
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..");

const nowForPath = () => new Date().toISOString().replace(/[:.]/g, "-");

const parseArgs = (argv: string[]): CliArgs => {
  const out: Partial<CliArgs> = {
    settingsPath: resolve(process.env.HOME ?? "", ".config", "zed", "settings.json"),
    serverKey: "bleeding-agent",
    workdir: process.cwd(),
    outFile: resolve(process.cwd(), ".bag", "acp-consumer-fixtures", `local-consumer-validation-${nowForPath()}.json`),
    prompt: "/chat Ahoj, co umis?",
    timeoutMs: 30_000,
    skipHandshake: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] ?? "";
    if (arg === "--settings") {
      out.settingsPath = resolve(argv[++i] ?? "");
    } else if (arg === "--server-key") {
      out.serverKey = argv[++i] ?? "";
    } else if (arg === "--workdir") {
      out.workdir = resolve(argv[++i] ?? "");
    } else if (arg === "--out") {
      const value = argv[++i] ?? "";
      out.outFile = isAbsolute(value) ? value : resolve(process.cwd(), value);
    } else if (arg === "--prompt") {
      out.prompt = argv[++i] ?? "";
    } else if (arg === "--timeout-ms") {
      out.timeoutMs = Number(argv[++i] ?? "");
    } else if (arg === "--skip-handshake") {
      out.skipHandshake = true;
    } else if (arg === "--help" || arg === "-h") {
      throw new Error(
        "usage: verify_acp_consumer_setup.ts [--settings PATH] [--server-key NAME] [--workdir DIR] [--out FILE] [--prompt TEXT] [--timeout-ms N] [--skip-handshake]",
      );
    } else {
      throw new Error(`unknown flag: ${arg}`);
    }
  }
  if ((out.serverKey ?? "").trim() === "") throw new Error("--server-key cannot be empty");
  if ((out.prompt ?? "").trim() === "") throw new Error("--prompt cannot be empty");
  if (!Number.isFinite(out.timeoutMs) || (out.timeoutMs ?? 0) <= 0) throw new Error("--timeout-ms must be positive");
  return {
    settingsPath: out.settingsPath ?? "",
    serverKey: out.serverKey ?? "",
    workdir: resolve(out.workdir ?? process.cwd()),
    outFile: out.outFile ?? "",
    prompt: out.prompt ?? "",
    timeoutMs: out.timeoutMs ?? 30_000,
    skipHandshake: out.skipHandshake ?? false,
  };
};

const stripJsonComments = (input: string): string => {
  let output = "";
  let inString = false;
  let quote = "";
  let escaped = false;
  for (let i = 0; i < input.length; i++) {
    const char = input[i] ?? "";
    const next = input[i + 1] ?? "";
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
      while (i < input.length && input[i] !== "\n") i++;
      output += "\n";
      continue;
    }
    if (char === "/" && next === "*") {
      i += 2;
      while (i < input.length && !(input[i] === "*" && input[i + 1] === "/")) i++;
      i++;
      continue;
    }
    output += char;
  }
  return output;
};

const parseJsonc = (input: string): JsonRecord => {
  const withoutComments = stripJsonComments(input);
  const withoutTrailingCommas = withoutComments.replace(/,\s*([}\]])/g, "$1");
  const parsed = JSON.parse(withoutTrailingCommas) as unknown;
  if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("settings root must be an object");
  }
  return parsed as JsonRecord;
};

const readAgentServerConfig = (settingsPath: string, serverKey: string): AgentServerConfig => {
  if (!existsSync(settingsPath)) throw new Error(`settings file not found: ${settingsPath}`);
  const settings = parseJsonc(readFileSync(settingsPath, "utf8"));
  const agentServers = settings.agent_servers;
  if (agentServers == null || typeof agentServers !== "object" || Array.isArray(agentServers)) {
    throw new Error(`settings file has no agent_servers object: ${settingsPath}`);
  }
  const server = (agentServers as JsonRecord)[serverKey];
  if (server == null || typeof server !== "object" || Array.isArray(server)) {
    throw new Error(`agent server '${serverKey}' not found in ${settingsPath}`);
  }
  const raw = server as JsonRecord;
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
    if (typeof env !== "object" || Array.isArray(env)) throw new Error(`agent server '${serverKey}' env must be an object`);
    for (const [key, value] of Object.entries(env as JsonRecord)) {
      if (typeof value !== "string") throw new Error(`agent server '${serverKey}' env.${key} must be a string`);
      envRecord[key] = value;
    }
  }
  return {
    type: typeof raw.type === "string" ? raw.type : null,
    command,
    args: Array.isArray(args) ? [...args] : [],
    env: envRecord,
  };
};

const bundleEvidence = (consumer: "Glass" | "Zed", appPath: string): BundleEvidence => {
  if (!existsSync(appPath)) {
    return { consumer, appPath, installed: false, bundleIdentifier: null, version: null };
  }
  const infoPath = resolve(appPath, "Contents", "Info.plist");
  try {
    const json = execFileSync("plutil", ["-convert", "json", "-o", "-", infoPath], { encoding: "utf8" });
    const parsed = JSON.parse(json) as JsonRecord;
    return {
      consumer,
      appPath,
      installed: true,
      bundleIdentifier: typeof parsed.CFBundleIdentifier === "string" ? parsed.CFBundleIdentifier : null,
      version: typeof parsed.CFBundleShortVersionString === "string" ? parsed.CFBundleShortVersionString : null,
    };
  } catch {
    return { consumer, appPath, installed: true, bundleIdentifier: null, version: null };
  }
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
    if (timeout != null) clearTimeout(timeout);
  }
};

const runHandshake = async (input: {
  config: AgentServerConfig;
  workdir: string;
  prompt: string;
  timeoutMs: number;
}): Promise<JsonRecord> => {
  const trajectory: TrajectoryEntry[] = [];
  const capabilities = headlessAcpConsumerCapabilityProfile("capable");
  const child: ChildProcessWithoutNullStreams = spawn(input.config.command, input.config.args, {
    cwd: repoRoot,
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ...input.config.env },
  });
  const stderrLines: string[] = [];
  child.stderr.on("data", (chunk) => {
    for (const line of chunk.toString("utf8").split(/\r?\n/)) {
      if (line.trim() !== "") stderrLines.push(line);
    }
  });
  const exitPromise = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolveExit) => {
    child.on("exit", (code, signal) => resolveExit({ code, signal }));
  });

  try {
    const stream = acp.ndJsonStream(Writable.toWeb(child.stdin), Readable.toWeb(child.stdout));
    const { client } = createHeadlessAcpClientFixture({
      workdir: input.workdir,
      trajectory,
      yolo: true,
      terminalMode: "stub",
      capabilities,
    });
    const connection = new acp.ClientSideConnection(() => client, stream);
    const recordAgentCall = async <T>(
      method: HeadlessAcpProtocolMethod,
      sessionId: string | undefined,
      payload: unknown,
      fn: () => Promise<T>,
    ): Promise<T> => {
      const at = new Date().toISOString();
      const sessionFields = sessionId === undefined ? {} : { sessionId };
      trajectory.push({ kind: "protocol_call", at, method, phase: "request", ...sessionFields, payload });
      try {
        const result = await fn();
        trajectory.push({
          kind: "protocol_call",
          at: new Date().toISOString(),
          method,
          phase: "response",
          ...sessionFields,
          payload: result,
        });
        return result;
      } catch (error) {
        trajectory.push({
          kind: "protocol_call",
          at: new Date().toISOString(),
          method,
          phase: "error",
          ...sessionFields,
          payload: { error: error instanceof Error ? error.message : String(error) },
        });
        throw error;
      }
    };
    const initialized = await withTimeout(
      recordAgentCall("initialize", undefined, { protocolVersion: acp.PROTOCOL_VERSION }, () => connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientInfo: { name: "bag-local-consumer-fixture-verifier", version: "1.0.0" },
        clientCapabilities: capabilities.clientCapabilities,
      })),
      input.timeoutMs,
      "initialize",
    );
    const session = await withTimeout(
      recordAgentCall("session/new", undefined, { cwd: input.workdir, mcpServers: [] }, () =>
        connection.newSession({ cwd: input.workdir, mcpServers: [] }),
      ),
      input.timeoutMs,
      "session/new",
    );
    const promptResult = await withTimeout(
      recordAgentCall("session/prompt", session.sessionId, { text: input.prompt }, () =>
        connection.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: input.prompt }] }),
      ),
      input.timeoutMs,
      "session/prompt",
    );
    const summary = summarizeHeadlessAcpTranscript(trajectory);
    const sideEffectCounts = {
      fsRead: summary.counts.fsRead,
      fsWrite: summary.counts.fsWrite,
      terminalCreate: summary.counts.terminalCreate,
      permission: summary.counts.permission,
    };
    return {
      ok: true,
      initialized,
      sessionId: session.sessionId,
      promptResult,
      counts: summary.counts,
      protocolMethods: summary.protocolMethods,
      sideEffectCounts,
      noSideEffects: Object.values(sideEffectCounts).every((count) => count === 0),
      trajectoryLength: trajectory.length,
      stderrLines: stderrLines.slice(-20),
    };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
      counts: summarizeHeadlessAcpTranscript(trajectory).counts,
      protocolMethods: summarizeHeadlessAcpTranscript(trajectory).protocolMethods,
      trajectoryLength: trajectory.length,
      stderrLines: stderrLines.slice(-20),
    };
  } finally {
    try {
      child.kill("SIGTERM");
    } catch {
      // noop
    }
    await Promise.race([exitPromise, new Promise((resolveExit) => setTimeout(resolveExit, 1_000))]);
  }
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  const config = readAgentServerConfig(args.settingsPath, args.serverKey);
  const handshake = args.skipHandshake
    ? { ok: null, skipped: true }
    : await runHandshake({ config, workdir: args.workdir, prompt: args.prompt, timeoutMs: args.timeoutMs });
  const result = {
    generatedAt: new Date().toISOString(),
    verifier: "scripts/verify_acp_consumer_setup.ts",
    scope: "local named ACP consumer launch-target validation; not desktop rendering automation",
    settingsPath: args.settingsPath,
    serverKey: args.serverKey,
    workdir: args.workdir,
    prompt: args.prompt,
    consumers: [
      bundleEvidence("Glass", "/Applications/Glass.app"),
      bundleEvidence("Zed", "/Applications/Zed.app"),
    ],
    agentServer: config,
    checks: {
      settingsExists: existsSync(args.settingsPath),
      glassInstalled: existsSync("/Applications/Glass.app"),
      zedInstalled: existsSync("/Applications/Zed.app"),
      commandResolvedBySpawn: handshake.ok === true,
      promptNoSideEffects: (handshake as { noSideEffects?: unknown }).noSideEffects === true,
    },
    handshake,
  };
  mkdirSync(dirname(args.outFile), { recursive: true });
  writeFileSync(args.outFile, `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify({ ...result, _outFile: args.outFile }, null, 2));
  if (!args.skipHandshake && handshake.ok !== true) {
    process.exitCode = 1;
  }
};

const directRun = process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;

if (directRun) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  });
}
