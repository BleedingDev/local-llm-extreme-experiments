/**
 * Tests that the harness-gates flags propagate correctly into
 * `buildVerifierFromInstruction`:
 *   - probeExtractor=false → returns undefined unconditionally (no LLM call)
 *   - probeExtractor=true,  snapshotRestore=true (default) → snapshot fires
 *   - probeExtractor=true,  snapshotRestore=false → snapshot skipped, probes still run
 */
import { describe, expect, test } from "bun:test";
import { buildVerifierFromInstruction } from "../src/instruction-verifier";
import type { HarnessGates } from "../src/harness-gates";
import type { LlmRouter } from "../src/llm";
import type { AcpTerminalClient } from "../src/autonomous-tools";

const FULL_GATES: HarnessGates = {
  probeExtractor: true,
  selfCheck: true,
  snapshotRestore: true,
  viewImage: true,
  codeSearch: true,
  retryPath: true,
  clusterMatcher: true,
  editStrategy: "shell-heredoc",
};

const noopRouter = (responses: string[]): LlmRouter => {
  let i = 0;
  return {
    masterAvailable: true,
    localAvailable: async () => true,
    chatText: async () => {
      const next = responses[i++];
      if (next === undefined) throw new Error("router exhausted");
      return next;
    },
    chatTextWithTools: async () => {
      throw new Error("not used");
    },
  } as LlmRouter;
};

type ExecCall = { command: string };

/**
 * Minimal AcpTerminalClient stub that records the bash commands passed via
 * createTerminal's args and stages a stdout per-prefix. The real client
 * pipes commands through `bash -lc 'set -o pipefail; <cmd>'`, so the
 * captured arg is the wrapped form — we strip the wrapper before testing.
 */
const recordingClient = (
  exitCodeByPrefix: (cmd: string) => number = () => 0,
  stdoutByPrefix: (cmd: string) => string = () => "",
): { client: AcpTerminalClient; calls: ExecCall[] } => {
  const calls: ExecCall[] = [];
  const stagedOutputs = new Map<string, { output: string; exit: number }>();
  let nextId = 0;
  const client: AcpTerminalClient = {
    createTerminal: async (params) => {
      const args = params.args ?? [];
      // args === ["-lc", "set -o pipefail; <command>"]
      const wrapped = args[1] ?? "";
      const command = wrapped.replace(/^set -o pipefail;\s*/, "");
      calls.push({ command });
      const id = `term-${++nextId}`;
      stagedOutputs.set(id, {
        output: stdoutByPrefix(command),
        exit: exitCodeByPrefix(command),
      });
      return { terminalId: id };
    },
    waitForTerminalExit: async ({ terminalId }) => ({
      exitCode: stagedOutputs.get(terminalId)?.exit ?? 0,
      signal: null,
    }),
    terminalOutput: async ({ terminalId }) => ({
      output: stagedOutputs.get(terminalId)?.output ?? "",
      truncated: false,
      exitStatus: { exitCode: stagedOutputs.get(terminalId)?.exit ?? 0, signal: null },
    }),
    releaseTerminal: async () => undefined,
  };
  return { client, calls };
};

describe("buildVerifierFromInstruction × harness gates", () => {
  test("probeExtractor=false returns undefined (no LLM call)", async () => {
    const router: LlmRouter = {
      masterAvailable: true,
      localAvailable: async () => true,
      chatText: async () => {
        throw new Error("router must NOT be called when probeExtractor is gated off");
      },
      chatTextWithTools: async () => {
        throw new Error("not used");
      },
    } as LlmRouter;
    const gates: HarnessGates = { ...FULL_GATES, probeExtractor: false };
    const verifier = await buildVerifierFromInstruction({
      router,
      instruction: "run `curl http://x/foo` and check it returns 200",
      gates,
    });
    expect(verifier).toBeUndefined();
  });

  test("probeExtractor=true with no extractable probes still returns undefined", async () => {
    const router = noopRouter(['{"probes": []}']);
    const verifier = await buildVerifierFromInstruction({
      router,
      instruction: "fix the bug",
      gates: FULL_GATES,
    });
    expect(verifier).toBeUndefined();
  });

  test("snapshotRestore=true (default): snapshot capture fires before probes", async () => {
    const router = noopRouter([
      JSON.stringify({
        probes: [{ cmd: "echo hello", expect: "hello", rationale: "test" }],
      }),
    ]);
    const verifier = await buildVerifierFromInstruction({
      router,
      instruction: "run `echo hello`",
      gates: FULL_GATES,
    });
    expect(verifier).toBeDefined();
    const { client, calls } = recordingClient(
      () => 0,
      (cmd) => (cmd.startsWith("echo hello") ? "hello\n" : ""),
    );
    const result = await verifier!({
      client,
      sessionId: "s1",
      cwd: "/app",
    });
    expect(result.passed).toBe(true);
    // Snapshot capture (find … > /tmp/.bag-probe-snapshot.txt) happens BEFORE
    // the probe; restore (`comm -23 …`) happens AFTER.
    const commands = calls.map((c) => c.command);
    const snapshotIdx = commands.findIndex((c) => c.includes("/tmp/.bag-probe-snapshot.txt") && c.includes("find"));
    const probeIdx = commands.findIndex((c) => c.startsWith("echo hello"));
    const restoreIdx = commands.findIndex((c) => c.includes("comm -23"));
    expect(snapshotIdx).toBeGreaterThanOrEqual(0);
    expect(probeIdx).toBeGreaterThan(snapshotIdx);
    expect(restoreIdx).toBeGreaterThan(probeIdx);
  });

  test("snapshotRestore=false: probes run but no snapshot/restore commands fire", async () => {
    const router = noopRouter([
      JSON.stringify({
        probes: [{ cmd: "echo world", expect: "world", rationale: "test" }],
      }),
    ]);
    const gates: HarnessGates = { ...FULL_GATES, snapshotRestore: false };
    const verifier = await buildVerifierFromInstruction({
      router,
      instruction: "run `echo world`",
      gates,
    });
    expect(verifier).toBeDefined();
    const { client, calls } = recordingClient(
      () => 0,
      (cmd) => (cmd.startsWith("echo world") ? "world\n" : ""),
    );
    const result = await verifier!({
      client,
      sessionId: "s1",
      cwd: "/app",
    });
    expect(result.passed).toBe(true);
    const commands = calls.map((c) => c.command);
    // Probe fired:
    expect(commands.some((c) => c.startsWith("echo world"))).toBe(true);
    // Snapshot/restore did NOT fire:
    expect(commands.some((c) => c.includes("/tmp/.bag-probe-snapshot.txt"))).toBe(false);
    expect(commands.some((c) => c.includes("comm -23"))).toBe(false);
  });
});
