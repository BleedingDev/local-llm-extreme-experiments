/**
 * Tests for `scripts/bag_rpc_run.ts`.
 *
 * Goals:
 *   - `createLineBuffer` must be partial-read robust: feeding a multi-line
 *     payload one byte at a time still yields exactly one callback per
 *     `\n`-terminated line. CR-LF endings are stripped.
 *   - `runRpcLoop` consumes JSON commands from a mock stdin and emits one
 *     JSON event per stdout line. We feed a fixture stream of three commands
 *     (prompt → steer → cancel) and assert the wire frames have the right
 *     shape, ordering, and command_id propagation.
 */

import { describe, expect, test } from "bun:test";
import { Readable } from "node:stream";

import {
  createLineBuffer,
  runRpcLoop,
} from "../scripts/bag_rpc_run";
import type {
  AgentEvent,
  BagSession,
  RunOptions,
} from "../src/sdk/agent-session";

describe("createLineBuffer", () => {
  test("emits one callback per LF-terminated line, even when fed byte-by-byte", () => {
    const lines: string[] = [];
    const buf = createLineBuffer((line) => lines.push(line));
    const payload = '{"a":1}\n{"b":2}\n{"c":3}\n';
    for (const ch of payload) buf.write(ch);
    expect(lines).toEqual(['{"a":1}', '{"b":2}', '{"c":3}']);
  });

  test("strips CR before LF (handles CRLF producers)", () => {
    const lines: string[] = [];
    const buf = createLineBuffer((line) => lines.push(line));
    buf.write('{"x":1}\r\n{"y":2}\r\n');
    expect(lines).toEqual(['{"x":1}', '{"y":2}']);
  });

  test("buffers across writes; flush() emits trailing line without LF", () => {
    const lines: string[] = [];
    const buf = createLineBuffer((line) => lines.push(line));
    buf.write('{"a"');
    buf.write(":1}");
    expect(lines).toEqual([]); // no LF yet
    buf.write("\n");
    expect(lines).toEqual(['{"a":1}']);
    buf.write("trailing-no-lf");
    buf.flush();
    expect(lines).toEqual(['{"a":1}', "trailing-no-lf"]);
  });
});

/**
 * Build a stub `BagSession` whose `run()` yields a scripted event sequence.
 * Captures `cancel()` and `steer()` calls so the test can assert dispatch.
 *
 * The counters live on a shared `state` object so the test can read them
 * after the loop completes — using closure variables surfaced through
 * Object.assign getters does NOT work because `Object.assign` snapshots
 * getter return values, not the descriptors.
 */
type StubSessionState = {
  cancelCount: number;
  steerMessages: string[];
  runCount: number;
};

const buildStubSession = (
  scriptedEvents: AgentEvent[][],
): { session: BagSession; state: StubSessionState } => {
  const state: StubSessionState = {
    cancelCount: 0,
    steerMessages: [],
    runCount: 0,
  };
  const session: BagSession = {
    cwd: "/stub",
    sessionId: "stub-session",
    run(_task: string, _options?: RunOptions): AsyncIterable<AgentEvent> {
      const events = scriptedEvents[state.runCount] ?? [];
      state.runCount += 1;
      return {
        async *[Symbol.asyncIterator]() {
          for (const ev of events) {
            // Yield to microtasks so the consumer observes intermediate
            // events before subsequent ones arrive — this also gives the
            // RPC dispatcher a chance to process queued commands such as
            // cancel/steer between events.
            await Promise.resolve();
            yield ev;
          }
        },
      };
    },
    cancel() {
      state.cancelCount += 1;
    },
    steer(message: string) {
      state.steerMessages.push(message);
    },
  };
  return { session, state };
};

const collectStdout = (): {
  write: (chunk: string) => boolean;
  events: AgentEvent[];
  raw: string[];
} => {
  const events: AgentEvent[] = [];
  const raw: string[] = [];
  return {
    write(chunk: string): boolean {
      raw.push(chunk);
      // Each chunk must be exactly one LF-terminated JSON object per the
      // RPC framing contract; assert that explicitly.
      expect(chunk.endsWith("\n")).toBe(true);
      const trimmed = chunk.trimEnd();
      if (trimmed.length > 0) events.push(JSON.parse(trimmed) as AgentEvent);
      return true;
    },
    events,
    raw,
  };
};

describe("runRpcLoop — JSONL framing and command dispatch", () => {
  test("processes a fixture of prompt/steer/cancel and emits one JSON event per stdout line", async () => {
    const scriptedEvents: AgentEvent[] = [
      {
        kind: "session_started",
        at: "2025-01-01T00:00:00.000Z",
        task: "rpc-fixture",
        cwd: "/stub",
      },
      {
        kind: "tool_call",
        at: "2025-01-01T00:00:01.000Z",
        tool_call_id: "tc-1",
        tool: "bash",
        arguments_json: JSON.stringify({ command: "ls" }),
      },
      {
        kind: "tool_result",
        at: "2025-01-01T00:00:02.000Z",
        tool_call_id: "tc-1",
        output: "ok\n",
        truncated_output: "ok\n",
        exit_code: 0,
        signal: null,
        duration_ms: 5,
        truncated: false,
        submitted: false,
      },
      {
        kind: "session_ended",
        at: "2025-01-01T00:00:03.000Z",
        stop_reason: "submitted",
        turns_used: 1,
        tool_calls_executed: 1,
        total_prompt_tokens: 1,
        total_completion_tokens: 1,
        submitted_output: null,
        attempts_used: 1,
      },
    ];

    const { session, state } = buildStubSession([scriptedEvents]);
    const stdoutSink = collectStdout();
    const stdinPayload = [
      JSON.stringify({ type: "prompt", task: "rpc-fixture", id: "req-7" }),
      JSON.stringify({ type: "steer", message: "use rg, not grep" }),
      JSON.stringify({ type: "cancel" }),
      JSON.stringify({ type: "shutdown" }),
    ].join("\n") + "\n";
    const stdin = Readable.from([stdinPayload]);

    await runRpcLoop({
      createSession: () => session,
      streams: { stdin, stdout: stdoutSink },
    });

    // Stdout: every chunk is a single LF-terminated JSON object.
    expect(stdoutSink.raw.length).toBe(scriptedEvents.length);
    for (const chunk of stdoutSink.raw) {
      expect(chunk.split("\n").length).toBe(2); // "<json>\n" → ["<json>", ""]
    }

    // Every event must carry the originating command_id (req-7).
    for (const ev of stdoutSink.events) {
      expect((ev as AgentEvent & { command_id?: string }).command_id).toBe("req-7");
    }
    // Event kinds came through in order.
    expect(stdoutSink.events.map((e) => e.kind)).toEqual([
      "session_started",
      "tool_call",
      "tool_result",
      "session_ended",
    ]);

    // Steer + cancel reached the stub session.
    expect(state.steerMessages).toEqual(["use rg, not grep"]);
    expect(state.cancelCount).toBeGreaterThanOrEqual(1);
    expect(state.runCount).toBe(1);
  });

  test("emits an error event for malformed JSON and continues processing", async () => {
    const { session, state } = buildStubSession([
      [
        {
          kind: "session_started",
          at: "2025-01-01T00:00:00.000Z",
          task: "after-bad-line",
          cwd: "/stub",
        },
        {
          kind: "session_ended",
          at: "2025-01-01T00:00:01.000Z",
          stop_reason: "end_turn",
          turns_used: 0,
          tool_calls_executed: 0,
          total_prompt_tokens: 0,
          total_completion_tokens: 0,
          submitted_output: null,
          attempts_used: 1,
        },
      ],
    ]);
    const stdoutSink = collectStdout();
    const stdinPayload =
      "this is not json\n" +
      JSON.stringify({ type: "prompt", task: "after-bad-line" }) +
      "\n" +
      JSON.stringify({ type: "shutdown" }) +
      "\n";
    const stdin = Readable.from([stdinPayload]);

    await runRpcLoop({
      createSession: () => session,
      streams: { stdin, stdout: stdoutSink },
    });

    // First event must be the malformed-JSON error.
    expect(stdoutSink.events[0]?.kind).toBe("error");
    // Subsequent prompt was still processed (session_started + session_ended).
    const kinds = stdoutSink.events.map((e) => e.kind);
    expect(kinds).toContain("session_started");
    expect(kinds).toContain("session_ended");
    expect(state.runCount).toBe(1);
  });

  test("rejects a second prompt while a run is in flight", async () => {
    // Build a session whose first run blocks until we explicitly release it,
    // so we can fire a second prompt while the first is still active.
    let release: (() => void) | null = null;
    const blocker = new Promise<void>((resolve) => {
      release = resolve;
    });
    const session: BagSession = {
      cwd: "/stub",
      sessionId: "stub-block",
      run(_task: string, _options?: RunOptions): AsyncIterable<AgentEvent> {
        return {
          async *[Symbol.asyncIterator]() {
            yield {
              kind: "session_started",
              at: "2025-01-01T00:00:00.000Z",
              task: "blocking",
              cwd: "/stub",
            };
            await blocker;
            yield {
              kind: "session_ended",
              at: "2025-01-01T00:00:01.000Z",
              stop_reason: "submitted",
              turns_used: 0,
              tool_calls_executed: 0,
              total_prompt_tokens: 0,
              total_completion_tokens: 0,
              submitted_output: null,
              attempts_used: 1,
            };
          },
        };
      },
      cancel() {
        release?.();
      },
      steer() {
        /* noop */
      },
    };
    const stdoutSink = collectStdout();
    const stdinPayload =
      JSON.stringify({ type: "prompt", task: "first", id: "a" }) +
      "\n" +
      JSON.stringify({ type: "prompt", task: "second", id: "b" }) +
      "\n" +
      JSON.stringify({ type: "shutdown" }) +
      "\n";
    const stdin = Readable.from([stdinPayload]);

    await runRpcLoop({
      createSession: () => session,
      streams: { stdin, stdout: stdoutSink },
    });

    // Find the duplicate-run error event and assert it carries command_id="b".
    const errorEvent = stdoutSink.events.find(
      (e): e is AgentEvent & { command_id?: string; kind: "error" } =>
        e.kind === "error" && (e as { command_id?: string }).command_id === "b",
    );
    expect(errorEvent).toBeDefined();
    expect(
      (errorEvent as { message?: string }).message ?? "",
    ).toMatch(/another run is already active/i);
  });
});
