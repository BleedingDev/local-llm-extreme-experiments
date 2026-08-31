/**
 * Tests for the adaptive task-shape router.
 *
 * Two surfaces under test here:
 *   1. The `requiresLongWait` semantic dimension: when the classifier
 *      (mocked Haiku) returns a JSON payload containing
 *      `requiresLongWait: true`, the resulting `TaskShapeDecision` must
 *      surface that flag. Mirror test for `false` to ensure the default
 *      stays sticky when the classifier omits or denies the field.
 *   2. The runtime hint propagation: when an autonomous coding turn is
 *      configured with `runtimeHint`, the user message sent to the master
 *      model must contain the hint string. The test mocks
 *      `chatTextWithTools` to capture the message it sees and asserts the
 *      hint is present.
 */

import { describe, expect, test } from "bun:test";
import {
  classifyTaskShape,
  LONG_WAIT_RUNTIME_HINT,
  type TaskShapeDecision,
} from "../src/task-shape-router";
import { runAutonomousCodingTurn } from "../src/autonomous-coding-turn";
import type { AcpTerminalClient } from "../src/autonomous-tools";
import type { ChatWithToolsOptions, LlmRouter } from "../src/llm";

/**
 * Build a minimal LlmRouter that returns a fixed `chatText` response (used by
 * the classifier) and a no-op `chatTextWithTools` (used by the inner loop).
 * Captures the user-message content passed to `chatTextWithTools` so callers
 * can assert prompt contents.
 */
const buildMockRouter = (input: {
  chatTextResponse: string;
  capture?: { userMessage?: string };
}): LlmRouter => ({
  masterAvailable: true,
  localAvailable: async () => true,
  chatText: async () => input.chatTextResponse,
  chatTextWithTools: async (opts: ChatWithToolsOptions) => {
    // Capture the most recent user-role message so the test can read it.
    const userMsg = opts.messages.find((m) => m.role === "user");
    if (input.capture && typeof userMsg?.content === "string") {
      input.capture.userMessage = userMsg.content;
    }
    // End the turn immediately by returning no tool calls and finish=stop with
    // a phrase that the loop treats as a valid end_turn signal.
    return {
      finishReason: "stop",
      textContent: "complete; nothing to do",
      toolCalls: [],
      promptTokens: 1,
      completionTokens: 1,
    };
  },
});

describe("classifyTaskShape — requiresLongWait dimension", () => {
  test("surfaces requiresLongWait=true when classifier marks the task as needing long async waits", async () => {
    const router = buildMockRouter({
      chatTextResponse: JSON.stringify({
        shape: "monolithic-complex",
        confidence: 0.82,
        reasoning:
          "task boots a QEMU VM under TCG and waits for sshd to bind before configuring it",
        requiresLongWait: true,
      }),
    });
    // Use a long-enough task / repo context to skip the heuristic short-circuit
    // (which fires for <200-char tasks with a near-empty repo).
    const repoContext =
      "files: alpine.iso, qemu-system-x86_64, scripts/boot.sh, scripts/wait-ssh.py, README.md, docs/qemu.md";
    const longTask = [
      "Boot the alpine.iso ISO under QEMU using TCG (no KVM available).",
      "Wait for the serial console to show the login prompt, then SSH into",
      "the VM and configure sshd to allow root login. Verify by running",
      "an ssh command from the host that exits 0.",
    ].join(" ");
    const decision: TaskShapeDecision = await classifyTaskShape({
      router,
      task: longTask,
      repoContext,
      cwd: "/tmp/bag-test",
    });
    expect(decision.requiresLongWait).toBe(true);
    expect(decision.shape).toBe("monolithic-complex");
    expect(decision.mode).toBe("tools");
  });

  test("defaults requiresLongWait=false when classifier omits or denies the field", async () => {
    const router = buildMockRouter({
      chatTextResponse: JSON.stringify({
        shape: "atomic",
        confidence: 0.9,
        reasoning: "single-file fix, no async waits",
        // intentionally no requiresLongWait field
      }),
    });
    const longTask =
      "Fix the off-by-one bug in src/parser.ts at line 142. The off-by-one shows up when the input is exactly 64 bytes long; verify the fix by running the existing test suite.";
    const repoContext =
      "files: src/parser.ts, src/parser.test.ts, src/utils.ts, package.json";
    const decision = await classifyTaskShape({
      router,
      task: longTask,
      repoContext,
      cwd: "/tmp/bag-test",
    });
    expect(decision.requiresLongWait).toBe(false);
  });

  test("does NOT enable requiresLongWait on truthy-but-non-true values (strict bool)", async () => {
    // Defensive: the model occasionally emits "true" as a string. Ensure the
    // parser does NOT enable the hint on those — only the literal JSON
    // boolean `true` should flip it. This guards against false-positive hint
    // injection on tasks where the classifier was uncertain.
    const router = buildMockRouter({
      chatTextResponse: JSON.stringify({
        shape: "compositional",
        confidence: 0.5,
        reasoning: "n/a",
        requiresLongWait: "true",
      }),
    });
    const longTask =
      "Write three independent helper scripts each with their own self-contained smoke check. They share no runtime state; each is verified in isolation before the next is written.";
    const repoContext =
      "files: scripts/a.sh, scripts/b.sh, scripts/c.sh, tests/a.test.sh, tests/b.test.sh, tests/c.test.sh";
    const decision = await classifyTaskShape({
      router,
      task: longTask,
      repoContext,
      cwd: "/tmp/bag-test",
    });
    expect(decision.requiresLongWait).toBe(false);
  });
});

describe("runtime hint wiring — runAutonomousCodingTurn", () => {
  // A no-op terminal client; we never get to a tool call because the mocked
  // chatTextWithTools returns immediately with finish="stop"/text="complete".
  const stubClient: AcpTerminalClient = {
    createTerminal: async () => ({ terminalId: "t1" }),
    waitForTerminalExit: async () => ({ exitCode: 0, signal: null }),
    terminalOutput: async () => ({
      output: "",
      truncated: false,
      exitStatus: { exitCode: 0, signal: null },
    }),
    releaseTerminal: async () => ({}),
  };

  test("appends the long-wait hint to the user message when cfg.runtimeHint is set", async () => {
    const capture: { userMessage?: string } = {};
    const router = buildMockRouter({
      chatTextResponse: "(unused — classifier not invoked here)",
      capture,
    });
    await runAutonomousCodingTurn({
      router,
      client: stubClient,
      sessionId: "s1",
      cwd: "/tmp/bag-test",
      task: "Boot the VM and wait for sshd to come up, then verify.",
      config: { runtimeHint: LONG_WAIT_RUNTIME_HINT, maxTurns: 1 },
    });
    expect(capture.userMessage).toBeDefined();
    // Spot-check two distinctive lines from the hint to be robust against
    // minor wording tweaks in the constant.
    expect(capture.userMessage!).toContain("Task shape hint");
    expect(capture.userMessage!).toContain("long-running async operations");
    expect(capture.userMessage!).toContain("nohup");
  });

  test("does NOT inject the hint into the user message when runtimeHint is unset", async () => {
    const capture: { userMessage?: string } = {};
    const router = buildMockRouter({
      chatTextResponse: "(unused)",
      capture,
    });
    await runAutonomousCodingTurn({
      router,
      client: stubClient,
      sessionId: "s1",
      cwd: "/tmp/bag-test",
      task: "Fix a typo in README.md.",
      config: { maxTurns: 1 },
    });
    expect(capture.userMessage).toBeDefined();
    expect(capture.userMessage!).not.toContain(
      "[Task shape hint - this task involves long-running async operations]",
    );
  });
});
