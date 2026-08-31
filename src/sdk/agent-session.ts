/**
 * BAG SDK — programmatic agent session for external embedders.
 *
 * Goal: ship the same surface that Pi-mono exposes via
 * `createAgentSession` / `createAgentSessionRuntime` so any host (CLI, IDE,
 * notebook, RPC adapter) can drive BAG without wiring up the full ACP
 * transport stack.
 *
 * Architecture (intentionally thin):
 *   - `createBagSession({ router, cwd, ... })` builds a session bound to a
 *     workspace + LLM router (typically `createLlmRouter(loadConfig(cwd))`).
 *   - `session.run(task, { signal })` returns an `AsyncIterable<AgentEvent>`
 *     that streams normalized events as the underlying autonomous coding
 *     turn executes. Events are GENERIC ("tool_call", "tool_result",
 *     "assistant_message", "submitted", …) — they intentionally do NOT
 *     reference BAG-specific tool names so RPC consumers can adapt them to
 *     their own protocol shapes.
 *   - `session.cancel()` aborts the active run via AbortController.
 *   - `session.steer(message)` queues an interjection that lands as a user
 *     message on the next assistant turn.
 *
 * Reuse: routes calls into `runAutonomousCodingTurn` (no fork). The loop's
 * minimal `onTraceEntry` hook plus an in-memory `AcpTerminalClient` shim
 * (backed by `node:child_process` so we don't need an ACP host) is all the
 * SDK needs. Callers wanting ACP-backed terminals can pass their own
 * `client` override.
 */

import { spawn, type ChildProcess } from "node:child_process";
import process from "node:process";

import {
  runAutonomousCodingTurn,
  type AutonomousTurnConfig,
  type AutonomousTurnResult,
  type AutonomousTurnStopReason,
  type AutonomousTurnTraceEntry,
  type PostSubmitVerifier,
} from "../autonomous-coding-turn";
import type { AcpTerminalClient } from "../autonomous-tools";
import type { LlmRouter } from "../llm";

/**
 * Generic event surface emitted by `BagSession.run()`. Mirrors the
 * autonomous-trace shape but flattened so RPC consumers can serialize a
 * single line per event without reaching into nested objects.
 *
 * NOTE: BAG-specific concepts (BAG_TASK_COMPLETE, bash sentinel, retry
 * attempts) are surfaced via `kind: "submitted"` and `kind: "attempt"` —
 * embedders can ignore the fields they don't recognise.
 */
export type AgentEvent =
  | { kind: "session_started"; at: string; task: string; cwd: string }
  | { kind: "user_message"; at: string; text: string }
  | {
      kind: "instruction_summarized";
      at: string;
      original_chars: number;
      summary_chars: number;
      tokens_saved: number;
    }
  | {
      kind: "assistant_message";
      at: string;
      text: string;
      tool_call_count: number;
    }
  | {
      kind: "tool_call";
      at: string;
      tool_call_id: string;
      tool: string;
      arguments_json: string;
    }
  | {
      kind: "tool_result";
      at: string;
      tool_call_id: string;
      output: string;
      truncated_output: string;
      exit_code: number | null;
      signal: string | null;
      duration_ms: number;
      truncated: boolean;
      submitted: boolean;
    }
  | { kind: "format_error"; at: string; reason: string }
  | { kind: "abort"; at: string }
  | {
      kind: "attempt";
      at: string;
      attempt: number;
      verifier_passed: boolean;
      verifier_output: string;
      verifier_exit_code: number | null;
      prompt_tokens: number;
      completion_tokens: number;
      tool_calls_executed: number;
      turns_used: number;
    }
  | {
      kind: "self_check";
      at: string;
      complete: boolean;
      missing: string[];
    }
  | {
      kind: "code_search";
      at: string;
      tool_call_id: string;
      query: string;
      hit_count: number;
      backend_status: "available" | "unavailable" | "disabled" | "error";
      duration_ms: number;
    }
  | {
      kind: "retry_hint";
      at: string;
      attempt: number;
      source: Extract<AutonomousTurnTraceEntry, { kind: "retry_hint" }>["source"];
      cluster_id?: string;
      library_id?: string;
    }
  | {
      kind: "edit_dispatch";
      at: string;
      strategy: Extract<AutonomousTurnTraceEntry, { kind: "edit_dispatch" }>["strategy"];
      tool: string;
      target: string;
      outcome: Extract<AutonomousTurnTraceEntry, { kind: "edit_dispatch" }>["outcome"];
      bytes_changed: number;
      retries_within_strategy: number;
    }
  | { kind: "steer"; at: string; message: string }
  | {
      kind: "session_ended";
      at: string;
      stop_reason: AutonomousTurnStopReason;
      turns_used: number;
      tool_calls_executed: number;
      total_prompt_tokens: number;
      total_completion_tokens: number;
      submitted_output: string | null;
      attempts_used: number;
    }
  | { kind: "error"; at: string; message: string };

/** Normalize an autonomous-trace entry into an SDK-shaped `AgentEvent`. */
const traceEntryToAgentEvent = (
  entry: AutonomousTurnTraceEntry,
): AgentEvent => {
  switch (entry.kind) {
    case "user":
      return { kind: "user_message", at: entry.at, text: entry.text };
    case "assistant":
      return {
        kind: "assistant_message",
        at: entry.at,
        text: entry.text,
        tool_call_count: entry.toolCallCount,
      };
    case "tool_call":
      return {
        kind: "tool_call",
        at: entry.at,
        tool_call_id: entry.toolCallId,
        tool: entry.tool,
        arguments_json: entry.argumentsJson,
      };
    case "tool_result":
      return {
        kind: "tool_result",
        at: entry.at,
        tool_call_id: entry.toolCallId,
        output: entry.result.output,
        truncated_output: entry.result.truncatedOutput,
        exit_code: entry.result.exitCode,
        signal: entry.result.signal,
        duration_ms: entry.result.durationMs,
        truncated: entry.result.truncated,
        submitted: entry.result.submitted,
      };
    case "format_error":
      return { kind: "format_error", at: entry.at, reason: entry.reason };
    case "abort":
      return { kind: "abort", at: entry.at };
    case "instruction_summarized":
      return {
        kind: "instruction_summarized",
        at: entry.at,
        original_chars: entry.original_chars,
        summary_chars: entry.summary_chars,
        tokens_saved: entry.tokens_saved,
      };
    case "attempt":
      return {
        kind: "attempt",
        at: entry.at,
        attempt: entry.attempt,
        verifier_passed: entry.verifier_passed,
        verifier_output: entry.verifier_output,
        verifier_exit_code: entry.verifier_exit_code,
        prompt_tokens: entry.prompt_tokens,
        completion_tokens: entry.completion_tokens,
        tool_calls_executed: entry.tool_calls_executed,
        turns_used: entry.turns_used,
      };
    case "pre_submit_self_check":
      return {
        kind: "self_check",
        at: entry.at,
        complete: entry.complete,
        missing: entry.missing,
      };
    case "code_search":
      return {
        kind: "code_search",
        at: entry.at,
        tool_call_id: entry.toolCallId,
        query: entry.query,
        hit_count: entry.hitCount,
        backend_status: entry.backendStatus,
        duration_ms: entry.durationMs,
      };
    case "retry_hint":
      return {
        kind: "retry_hint",
        at: entry.at,
        attempt: entry.attempt,
        source: entry.source,
        ...(entry.cluster_id === undefined ? {} : { cluster_id: entry.cluster_id }),
        ...(entry.library_id === undefined ? {} : { library_id: entry.library_id }),
      };
    case "edit_dispatch":
      return {
        kind: "edit_dispatch",
        at: entry.at,
        strategy: entry.strategy,
        tool: entry.tool,
        target: entry.target,
        outcome: entry.outcome,
        bytes_changed: entry.bytes_changed,
        retries_within_strategy: entry.retries_within_strategy,
      };
  }
  const exhaustive: never = entry;
  return exhaustive;
};

/**
 * Default `AcpTerminalClient` shim backed by `node:child_process.spawn`.
 *
 * Lets BAG run autonomous turns without an ACP host. Embedders that want
 * permission gating, sandboxing, or remote workspaces should pass their own
 * `client` override into `createBagSession`.
 */
export const createSubprocessTerminalClient = (): AcpTerminalClient => {
  type State = {
    proc: ChildProcess;
    buffer: string;
    byteLimit: number | null;
    truncated: boolean;
    exit: { exitCode: number | null; signal: string | null } | null;
    exitWaiters: Array<(exit: { exitCode: number | null; signal: string | null }) => void>;
  };
  const terminals = new Map<string, State>();

  const append = (state: State, chunk: string | Buffer): void => {
    const text = Buffer.isBuffer(chunk) ? chunk.toString("utf8") : chunk;
    state.buffer += text;
    if (state.byteLimit == null) return;
    if (state.byteLimit <= 0) {
      state.truncated = state.truncated || text.length > 0;
      state.buffer = "";
      return;
    }
    if (Buffer.byteLength(state.buffer) > state.byteLimit) {
      const buf = Buffer.from(state.buffer, "utf8");
      state.buffer = buf.slice(buf.length - state.byteLimit).toString("utf8");
      state.truncated = true;
    }
  };

  return {
    createTerminal: async ({ command, args, cwd, env, outputByteLimit }) => {
      const terminalId = `bag-sdk-term-${Date.now()}-${Math.floor(Math.random() * 1e6).toString(36)}`;
      const mergedEnv: NodeJS.ProcessEnv = { ...process.env };
      for (const v of env ?? []) mergedEnv[v.name] = v.value;
      const proc = spawn(command, args, {
        cwd: cwd ?? undefined,
        env: mergedEnv,
        stdio: ["ignore", "pipe", "pipe"],
      });
      const state: State = {
        proc,
        buffer: "",
        byteLimit: outputByteLimit ?? null,
        truncated: false,
        exit: null,
        exitWaiters: [],
      };
      terminals.set(terminalId, state);
      proc.stdout?.on("data", (chunk: Buffer) => append(state, chunk));
      proc.stderr?.on("data", (chunk: Buffer) => append(state, chunk));
      const settle = (exitCode: number | null, signal: string | null): void => {
        if (state.exit != null) return;
        state.exit = { exitCode, signal };
        const waiters = state.exitWaiters.splice(0, state.exitWaiters.length);
        for (const w of waiters) w(state.exit);
      };
      proc.once("close", (code, signal) =>
        settle(code, signal == null ? null : String(signal)),
      );
      proc.once("error", (error) => {
        append(state, `[bag-sdk subprocess error] ${error.message}\n`);
        settle(null, "ERROR");
      });
      return { terminalId };
    },
    waitForTerminalExit: async ({ terminalId }) => {
      const state = terminals.get(terminalId);
      if (state == null) return { exitCode: null, signal: null };
      if (state.exit != null) return state.exit;
      return new Promise((resolveExit) => {
        state.exitWaiters.push(resolveExit);
      });
    },
    terminalOutput: async ({ terminalId }) => {
      const state = terminals.get(terminalId);
      if (state == null) {
        return { output: "", truncated: false };
      }
      const exitStatus = state.exit;
      const response: {
        output: string;
        truncated: boolean;
        exitStatus?: { exitCode?: number | null; signal?: string | null } | null;
      } = {
        output: state.buffer,
        truncated: state.truncated,
      };
      if (exitStatus != null) response.exitStatus = exitStatus;
      return response;
    },
    releaseTerminal: async ({ terminalId }) => {
      const state = terminals.get(terminalId);
      if (state == null) return {};
      if (state.proc.exitCode == null && state.proc.signalCode == null) {
        try {
          state.proc.kill("SIGTERM");
        } catch {
          /* noop */
        }
      }
      terminals.delete(terminalId);
      return {};
    },
  };
};

export type SessionConfig = {
  router: LlmRouter;
  cwd: string;
  /** Defaults to `bag-sdk-session-<rand>`. Forwarded to the AcpTerminalClient. */
  sessionId?: string;
  /** Overrides the default subprocess-backed terminal client. */
  client?: AcpTerminalClient;
  /** Forwarded to `runAutonomousCodingTurn` (max turns, sentinel, etc.). */
  config?: Partial<AutonomousTurnConfig>;
  /** Optional post-submit verifier (e.g. project test command). */
  verifyAfterSubmit?: PostSubmitVerifier;
};

export type RunOptions = {
  signal?: AbortSignal;
  /** Per-run config overrides; merged on top of the session-level `config`. */
  config?: Partial<AutonomousTurnConfig>;
};

export type BagSession = {
  /**
   * Run the agent against a task. Returns an `AsyncIterable<AgentEvent>` that
   * yields events as they happen and terminates with `kind: "session_ended"`
   * (or `kind: "error"` on unrecoverable failure).
   *
   * Concurrency: only one run can be active per session at a time. Calling
   * `run()` while a previous run is in flight throws.
   */
  run(task: string, options?: RunOptions): AsyncIterable<AgentEvent>;
  /** Aborts the active run, if any. No-op when idle. */
  cancel(): void;
  /**
   * Queues an interjection. Currently surfaced as a `steer` event in the
   * stream and as a stable property on the session for the loop to consume on
   * the next assistant turn. The hook is intentionally minimal; deeper
   * mid-flight steering will land when `runAutonomousCodingTurn` exposes a
   * messages-mutator hook.
   */
  steer(message: string): void;
  readonly cwd: string;
  readonly sessionId: string;
};

const now = (): string => new Date().toISOString();

/**
 * Lightweight async queue: producers `push` events; consumers `iterate` via
 * for-await. Closing the queue terminates iteration once drained. Failure
 * propagation is via `fail(error)`.
 */
const createEventQueue = <T>() => {
  const buffer: T[] = [];
  const waiters: Array<(result: IteratorResult<T>) => void> = [];
  let closed = false;
  let failure: unknown = null;

  const push = (item: T): void => {
    if (closed) return;
    const waiter = waiters.shift();
    if (waiter !== undefined) {
      waiter({ value: item, done: false });
    } else {
      buffer.push(item);
    }
  };
  const close = (): void => {
    if (closed) return;
    closed = true;
    while (waiters.length > 0) {
      const waiter = waiters.shift();
      if (waiter !== undefined) waiter({ value: undefined as T, done: true });
    }
  };
  const fail = (error: unknown): void => {
    if (failure == null) failure = error;
    close();
  };
  const iterator: AsyncIterator<T> = {
    next(): Promise<IteratorResult<T>> {
      if (failure != null) {
        const error = failure;
        failure = null;
        return Promise.reject(error);
      }
      const item = buffer.shift();
      if (item !== undefined) {
        return Promise.resolve({ value: item, done: false });
      }
      if (closed) return Promise.resolve({ value: undefined as T, done: true });
      return new Promise<IteratorResult<T>>((resolveNext) => {
        waiters.push(resolveNext);
      });
    },
    return(): Promise<IteratorResult<T>> {
      close();
      return Promise.resolve({ value: undefined as T, done: true });
    },
  };
  const iterable: AsyncIterable<T> = {
    [Symbol.asyncIterator](): AsyncIterator<T> {
      return iterator;
    },
  };
  return { push, close, fail, iterable };
};

/**
 * Wrap an `LlmRouter` so that pending `steer(...)` interjections are
 * inserted as user messages immediately before the next `chatTextWithTools`
 * call. This is the minimal "mid-task interjection" hook — it fires on the
 * next assistant turn rather than the current in-flight one, which matches
 * the Pi-mono semantics (interjections land at message-boundaries).
 */
const wrapRouterWithSteer = (
  base: LlmRouter,
  pending: { messages: string[] },
): LlmRouter => ({
  masterAvailable: base.masterAvailable,
  localAvailable: base.localAvailable,
  chatText: base.chatText,
  chatTextWithTools: (options) => {
    if (pending.messages.length > 0) {
      const drained = pending.messages.splice(0, pending.messages.length);
      const enriched = [...options.messages];
      for (const msg of drained) {
        enriched.push({
          role: "user",
          content: `[BAG steer] ${msg}`,
        });
      }
      return base.chatTextWithTools({ ...options, messages: enriched });
    }
    return base.chatTextWithTools(options);
  },
});

export const createBagSession = (config: SessionConfig): BagSession => {
  const sessionId = config.sessionId ?? `bag-sdk-session-${Math.floor(Math.random() * 1e9).toString(36)}`;
  const client = config.client ?? createSubprocessTerminalClient();
  const pendingSteer: { messages: string[] } = { messages: [] };
  const router = wrapRouterWithSteer(config.router, pendingSteer);

  let activeAbort: AbortController | null = null;
  let activeQueue: ReturnType<typeof createEventQueue<AgentEvent>> | null = null;

  const session: BagSession = {
    cwd: config.cwd,
    sessionId,
    run(task: string, options?: RunOptions): AsyncIterable<AgentEvent> {
      if (activeAbort != null) {
        throw new Error("BagSession.run: a previous run is still in flight");
      }
      const abort = new AbortController();
      if (options?.signal != null) {
        if (options.signal.aborted) abort.abort();
        else options.signal.addEventListener("abort", () => abort.abort(), { once: true });
      }
      activeAbort = abort;

      const queue = createEventQueue<AgentEvent>();
      activeQueue = queue;

      queue.push({
        kind: "session_started",
        at: now(),
        task,
        cwd: config.cwd,
      });

      const mergedConfig: Partial<AutonomousTurnConfig> = {
        ...(config.config ?? {}),
        ...(options?.config ?? {}),
      };

      const turnInput: Parameters<typeof runAutonomousCodingTurn>[0] = {
        router,
        client,
        sessionId,
        cwd: config.cwd,
        task,
        signal: abort.signal,
        config: mergedConfig,
        onTraceEntry: (entry) => queue.push(traceEntryToAgentEvent(entry)),
      };
      if (config.verifyAfterSubmit !== undefined) {
        turnInput.verifyAfterSubmit = config.verifyAfterSubmit;
      }

      const finish = (
        result: AutonomousTurnResult | null,
        error: unknown,
      ): void => {
        if (activeAbort === abort) activeAbort = null;
        if (activeQueue === queue) activeQueue = null;
        if (error != null) {
          queue.push({
            kind: "error",
            at: now(),
            message: error instanceof Error ? error.message : String(error),
          });
        }
        if (result != null) {
          queue.push({
            kind: "session_ended",
            at: now(),
            stop_reason: result.stopReason,
            turns_used: result.turnsUsed,
            tool_calls_executed: result.toolCallsExecuted,
            total_prompt_tokens: result.totalPromptTokens,
            total_completion_tokens: result.totalCompletionTokens,
            submitted_output: result.submittedOutput,
            attempts_used: result.attemptsUsed,
          });
        } else {
          queue.push({
            kind: "session_ended",
            at: now(),
            stop_reason: "error",
            turns_used: 0,
            tool_calls_executed: 0,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            submitted_output: null,
            attempts_used: 0,
          });
        }
        queue.close();
      };

      runAutonomousCodingTurn(turnInput).then(
        (result) => finish(result, null),
        (error) => finish(null, error),
      );

      return queue.iterable;
    },
    cancel(): void {
      if (activeAbort != null) {
        activeAbort.abort();
      }
    },
    steer(message: string): void {
      const trimmed = message.trim();
      if (trimmed.length === 0) return;
      pendingSteer.messages.push(trimmed);
      if (activeQueue != null) {
        activeQueue.push({ kind: "steer", at: now(), message: trimmed });
      }
    },
  };

  return session;
};
