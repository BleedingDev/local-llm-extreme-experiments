import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  createRealAcpTraceMinedScorecards,
  createSimulatedRealAcpExecutor,
  renderRealAcpTraceMinedScorecardsMarkdown,
  runRealAcpCorpus,
  type RealAcpCorpusRunManifest,
  type RealAcpRunMetadata,
  type RealAcpTaskRunResult,
} from "../src/replay";

const createdAt = "2026-05-04T00:00:00.000Z";

const metadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.trace-scorecard",
    provider: "simulated",
    model: "simulated-acp-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.trace-scorecard",
    rootFingerprint: "sha256:real-acp-trace-scorecard",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.trace-scorecard",
    protectedPathPolicy: "Only task fixture paths may be changed.",
  },
  client: {
    clientProfileId: "client.real-acp.trace-scorecard",
    clientName: "Simulated ACP trace scorecard harness",
    clientVersion: "v1",
    transport: "simulated",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.trace-scorecard",
    optimizerProfileId: "optimizer.real-acp.trace-scorecard",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
};

const makeTempBase = (): Promise<string> => mkdtemp(join(tmpdir(), "real-acp-trace-scorecard-test-"));

const manifestFixture = async (): Promise<RealAcpCorpusRunManifest> => {
  const baseDir = await makeTempBase();
  try {
    return await runRealAcpCorpus({
      runId: "real-acp-run.trace-scorecard.fixture",
      metadata,
      executor: createSimulatedRealAcpExecutor(),
      taskIds: [
        "real-acp.task.simple-edit-greeting",
        "real-acp.task.cart-bugfix-fail-to-pass",
      ],
      workspaceBaseDir: baseDir,
      createdAt,
    });
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
};

const mutateResult = (
  manifest: RealAcpCorpusRunManifest,
  taskId: string,
  patch: (result: RealAcpTaskRunResult) => RealAcpTaskRunResult,
): RealAcpCorpusRunManifest => ({
  ...manifest,
  taskResults: manifest.taskResults.map((result) => result.taskId === taskId ? patch(result) : result),
});

describe("real ACP trace-mined scorecards", () => {
  test("computes tool calibration, argument patterns, transitions, and edit family matrix", async () => {
    const base = await manifestFixture();
    const passed = mutateResult(base, "real-acp.task.simple-edit-greeting", (result) => ({
      ...result,
      status: "passed",
      toolCalls: [
        {
          toolCallId: "tool.trace.read.greeting",
          namespace: "acp.fs",
          name: "readTextFile",
          status: "succeeded",
          sideEffectLevel: "read",
        },
        {
          toolCallId: "tool.trace.write.greeting",
          namespace: "acp.fs",
          name: "writeTextFile",
          status: "succeeded",
          sideEffectLevel: "write",
        },
      ],
      terminalCommands: [
        {
          commandId: "cmd.trace.greeting",
          command: ["bun", "test", "--filter", "greeter"],
          status: "succeeded",
          exitCode: 0,
          durationMs: 50,
        },
      ],
      verifier: { ...result.verifier, status: "passed", commandIds: ["cmd.trace.greeting"] },
    }));
    const failed = mutateResult(passed, "real-acp.task.cart-bugfix-fail-to-pass", (result) => ({
      ...result,
      status: "failed",
      toolCalls: [
        {
          toolCallId: "tool.trace.read.cart",
          namespace: "acp.fs",
          name: "readTextFile",
          status: "succeeded",
          sideEffectLevel: "read",
        },
        {
          toolCallId: "tool.trace.write.cart",
          namespace: "acp.fs",
          name: "writeTextFile",
          status: "failed",
          sideEffectLevel: "write",
          errorCode: "invalid_arguments",
        },
      ],
      terminalCommands: [
        {
          commandId: "cmd.trace.cart",
          command: ["bun", "test", "--filter", "cart"],
          status: "failed",
          exitCode: 1,
          durationMs: 70,
        },
      ],
      verifier: { ...result.verifier, status: "failed", commandIds: ["cmd.trace.cart"] },
      telemetry: {
        codingProgressDiagnostic: {
          schemaVersion: "coding-progress-diagnostics.v1",
          runId: "run.trace",
          classifiedAt: createdAt,
          progressClass: "client_write_failed",
          terminal: "final",
          mutatingProgress: {
            changedFileCount: 0,
            successfulWriteCount: 0,
            failedWriteCount: 1,
            terminalCommandCount: 1,
            terminalExitCount: 1,
          },
          generation: {
            editCount: 1,
            parseFailureCount: 0,
            fallbackAttempted: false,
            fallbackEditCount: 0,
            fallbackParseFailureCount: 0,
          },
          verifier: {
            plannedCommandCount: 1,
            executedCommandCount: 1,
            failedCommandCount: 1,
            missing: false,
          },
          failureSignals: ["write.client_write_failed"],
          evidenceRefs: [],
          reason: "client write failed",
        },
      },
    }));

    const scorecards = createRealAcpTraceMinedScorecards({
      manifests: [failed],
      scorecardId: "real-acp-trace.test",
      createdAt,
    });

    const readSummary = scorecards.toolCalibration.find((summary) => summary.name === "readTextFile");
    expect(readSummary).toMatchObject({
      callCount: 2,
      success: { count: 2, rate: 1 },
    });
    const writeSummary = scorecards.toolCalibration.find((summary) => summary.name === "writeTextFile");
    expect(writeSummary).toMatchObject({
      callCount: 2,
      failed: { count: 1, rate: 0.5 },
      failureClasses: [{ failureClass: "invalid_arguments", count: 1 }],
    });
    expect(scorecards.argumentPatterns.some((summary) =>
      summary.commandName === "bun" && summary.flags.includes("--filter") && summary.failed.count === 1)).toBe(true);
    expect(scorecards.toolTransitions.some((summary) =>
      summary.from === "acp.fs/readTextFile" && summary.to === "acp.fs/writeTextFile" && summary.count === 2)).toBe(true);
    expect(scorecards.editFamilyMatrix.some((summary) =>
      summary.taskShape === "bugfix_fail_to_pass" && summary.verifierFailureRate === 1)).toBe(true);
    expect(scorecards.editFamilyMatrix.some((summary) =>
      summary.codingProgressClasses.some((entry) =>
        entry.failureClass === "client_write_failed" && entry.count === 1))).toBe(true);
  });

  test("renders markdown with caveats for proxy metrics", async () => {
    const scorecards = createRealAcpTraceMinedScorecards({
      manifests: [await manifestFixture()],
      scorecardId: "real-acp-trace.markdown",
      createdAt,
    });
    const markdown = renderRealAcpTraceMinedScorecardsMarkdown(scorecards);

    expect(markdown).toContain("# Real ACP Trace-Mined Scorecards");
    expect(markdown).toContain("## Tool Calibration");
    expect(markdown).toContain("## Argument Patterns");
    expect(markdown).toContain("precisionProxy is the share");
  });
});
