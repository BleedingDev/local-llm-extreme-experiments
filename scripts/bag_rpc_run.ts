#!/usr/bin/env -S node --loader=tsx
/**
 * BAG RPC adapter — LF-delimited JSONL stdin/stdout.
 *
 * Lets any external process embed BAG by speaking a tiny line-buffered JSON
 * protocol over stdio. Mirrors Pi-mono's RPC mode so existing embedders can
 * point at BAG by swapping the binary.
 *
 * Wire format
 * -----------
 * Every line on stdin is a single JSON command terminated by `\n`. Partial
 * lines are buffered and reassembled — readers can write commands in chunks.
 * Every line on stdout is a single JSON event terminated by `\n`. Stderr is
 * reserved for human-readable diagnostics.
 *
 * Commands (stdin)
 *   { "type": "prompt", "task": "...", "id"?: "..." }
 *     Starts a new run. Only one run can be active at a time; sending
 *     `prompt` while another run is in flight emits an `error` event.
 *   { "type": "cancel" }
 *     Aborts the active run. No-op when idle.
 *   { "type": "steer", "message": "..." }
 *     Queues a mid-task interjection that lands as a user message on the
 *     next assistant turn.
 *   { "type": "shutdown" }
 *     Cancels any active run, drains pending events, and exits cleanly.
 *
 * Events (stdout)
 *   Each line is an `AgentEvent` from `src/sdk/agent-session.ts` plus an
 *   optional `command_id` field copied through from the originating prompt.
 *   Stable kinds: `session_started`, `user_message`, `assistant_message`,
 *   `tool_call`, `tool_result`, `format_error`, `instruction_summarized`,
 *   `attempt`, `self_check`, `steer`, `session_ended`, `error`.
 *
 * CLI flags
 *   --workdir DIR   Working directory for the agent (default cwd).
 *   --max-turns N   Cap on autonomous-coding-turn turns (default config).
 */

import process from "node:process";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { loadConfig } from "../src/config";
import { createLlmRouter } from "../src/llm";
import {
  createBagSession,
  type AgentEvent,
  type BagSession,
} from "../src/sdk/agent-session";

type RpcCommand =
  | { type: "prompt"; task: string; id?: string }
  | { type: "cancel" }
  | { type: "steer"; message: string }
  | { type: "shutdown" };

type RpcEvent = AgentEvent & { command_id?: string };

type CliArgs = {
  workdir: string;
  maxTurns?: number;
};

const parseArgs = (argv: string[]): CliArgs => {
  let workdir: string | null = null;
  let maxTurns: number | undefined;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] ?? "";
    if (arg === "--workdir" || arg === "-w") {
      workdir = argv[++i] ?? "";
    } else if (arg === "--max-turns") {
      const next = Number(argv[++i]);
      if (!Number.isFinite(next) || next <= 0) {
        throw new Error("--max-turns must be a positive integer");
      }
      maxTurns = Math.floor(next);
    } else if (arg === "--help" || arg === "-h") {
      process.stderr.write(
        "usage: bag_rpc_run.ts [--workdir DIR] [--max-turns N]\n",
      );
      process.exit(0);
    } else if (arg.startsWith("--")) {
      throw new Error(`unknown flag: ${arg}`);
    }
  }
  const result: CliArgs = {
    workdir: resolve(workdir ?? process.cwd()),
  };
  if (maxTurns !== undefined) result.maxTurns = maxTurns;
  return result;
};

/**
 * Robust line buffer: feeds chunks (which may contain partial lines or
 * multiple lines per chunk), emits one callback per `\n`-terminated line.
 */
export const createLineBuffer = (
  onLine: (line: string) => void,
): { write: (chunk: string) => void; flush: () => void } => {
  let pending = "";
  return {
    write(chunk: string): void {
      pending += chunk;
      let idx = pending.indexOf("\n");
      while (idx !== -1) {
        const line = pending.slice(0, idx);
        pending = pending.slice(idx + 1);
        // Strip a trailing CR if the producer sent CRLF endings.
        const cleaned = line.endsWith("\r") ? line.slice(0, -1) : line;
        if (cleaned.length > 0) onLine(cleaned);
        idx = pending.indexOf("\n");
      }
    },
    flush(): void {
      const trailing = pending.trim();
      pending = "";
      if (trailing.length > 0) onLine(trailing);
    },
  };
};

const parseCommand = (line: string): RpcCommand | { error: string } => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch (err) {
    return { error: `invalid JSON: ${err instanceof Error ? err.message : String(err)}` };
  }
  if (typeof parsed !== "object" || parsed == null) {
    return { error: "command must be a JSON object" };
  }
  const obj = parsed as Record<string, unknown>;
  const type = obj.type;
  if (type === "prompt") {
    const task = typeof obj.task === "string" ? obj.task : "";
    if (task.trim().length === 0) {
      return { error: "prompt command requires a non-empty 'task' string" };
    }
    const cmd: { type: "prompt"; task: string; id?: string } = {
      type: "prompt",
      task,
    };
    if (typeof obj.id === "string" && obj.id.length > 0) cmd.id = obj.id;
    return cmd;
  }
  if (type === "cancel") return { type: "cancel" };
  if (type === "steer") {
    const message = typeof obj.message === "string" ? obj.message : "";
    if (message.trim().length === 0) {
      return { error: "steer command requires a non-empty 'message' string" };
    }
    return { type: "steer", message };
  }
  if (type === "shutdown") return { type: "shutdown" };
  return { error: `unknown command type: ${String(type)}` };
};

export type RpcStreams = {
  stdin: NodeJS.ReadableStream;
  stdout: { write: (chunk: string) => boolean | void };
  stderr?: { write: (chunk: string) => boolean | void };
};

export type RpcRunOptions = {
  /** Build the session lazily on the first prompt. */
  createSession: () => BagSession;
  streams: RpcStreams;
  /** Per-run config overrides applied to every run. */
  runConfig?: { maxTurns?: number };
};

const writeEvent = (
  out: { write: (chunk: string) => boolean | void },
  event: RpcEvent,
): void => {
  out.write(`${JSON.stringify(event)}\n`);
};

const now = (): string => new Date().toISOString();

/**
 * Core RPC dispatcher. Pulled out from `main()` so tests can drive it
 * synchronously with mock streams without needing to spawn a real process.
 */
export const runRpcLoop = async (options: RpcRunOptions): Promise<void> => {
  let session: BagSession | null = null;
  let activeRunId: string | null = null;
  let activeRun: Promise<void> | null = null;
  let shuttingDown = false;
  const stderr = options.streams.stderr ?? { write: () => undefined };

  const ensureSession = (): BagSession => {
    if (session == null) session = options.createSession();
    return session;
  };

  const startRun = (task: string, commandId: string | undefined): void => {
    if (activeRun != null) {
      writeEvent(options.streams.stdout, {
        kind: "error",
        at: now(),
        message: "another run is already active; cancel it first",
        ...(commandId === undefined ? {} : { command_id: commandId }),
      });
      return;
    }
    const sess = ensureSession();
    activeRunId = commandId ?? null;
    const runOptions: { config?: { maxTurns?: number } } =
      options.runConfig?.maxTurns !== undefined
        ? { config: { maxTurns: options.runConfig.maxTurns } }
        : {};
    const stream = sess.run(task, runOptions);
    activeRun = (async () => {
      try {
        for await (const event of stream) {
          const wire: RpcEvent =
            commandId === undefined ? event : { ...event, command_id: commandId };
          writeEvent(options.streams.stdout, wire);
        }
      } catch (error) {
        writeEvent(options.streams.stdout, {
          kind: "error",
          at: now(),
          message: error instanceof Error ? error.message : String(error),
          ...(commandId === undefined ? {} : { command_id: commandId }),
        });
      } finally {
        activeRun = null;
        activeRunId = null;
      }
    })();
  };

  const handleCommand = (cmd: RpcCommand): void => {
    switch (cmd.type) {
      case "prompt":
        startRun(cmd.task, cmd.id);
        return;
      case "cancel":
        if (session != null) session.cancel();
        return;
      case "steer":
        if (session != null) session.steer(cmd.message);
        return;
      case "shutdown":
        shuttingDown = true;
        if (session != null) session.cancel();
        return;
    }
  };

  const lineBuffer = createLineBuffer((line) => {
    const parsed = parseCommand(line);
    if ("error" in parsed) {
      writeEvent(options.streams.stdout, {
        kind: "error",
        at: now(),
        message: parsed.error,
      });
      stderr.write(`[bag-rpc] ${parsed.error}\n`);
      return;
    }
    handleCommand(parsed);
  });

  await new Promise<void>((resolveLoop) => {
    const onData = (chunk: Buffer | string): void => {
      lineBuffer.write(typeof chunk === "string" ? chunk : chunk.toString("utf8"));
    };
    const finalize = async (): Promise<void> => {
      lineBuffer.flush();
      if (activeRun != null) {
        try {
          await activeRun;
        } catch {
          /* surfaced via error event already */
        }
      }
      resolveLoop();
    };
    options.streams.stdin.on("data", onData);
    options.streams.stdin.once("end", () => {
      void finalize();
    });
    options.streams.stdin.once("close", () => {
      void finalize();
    });
    // Allow shutdown command to break out even if stdin stays open.
    const pollShutdown = setInterval(() => {
      if (shuttingDown) {
        clearInterval(pollShutdown);
        void finalize();
      }
    }, 50);
    pollShutdown.unref?.();
  });

  // Reference the read variable so noUnusedLocals stays clean while we keep
  // the symbol available for future debug-event surfaces.
  void activeRunId;
};

const main = async (): Promise<void> => {
  const args = parseArgs(process.argv.slice(2));
  const config = loadConfig(args.workdir);
  const router = createLlmRouter(config);
  await runRpcLoop({
    createSession: () =>
      createBagSession({
        router,
        cwd: args.workdir,
      }),
    streams: {
      stdin: process.stdin,
      stdout: process.stdout,
      stderr: process.stderr,
    },
    runConfig: args.maxTurns !== undefined ? { maxTurns: args.maxTurns } : {},
  });
};

const directRun =
  process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;

if (directRun) {
  main().catch((error: unknown) => {
    process.stderr.write(
      `[bag-rpc] fatal: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
