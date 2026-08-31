import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  extractReplayEvalCaseSkeleton,
  replayEvalCaseFromSkeleton,
  runReplayEvalComparison,
  toolCallReplayScenarios,
  routingReplayScenarios,
} from "../src/replay";
import type { ComparisonRunMetadata, EvalComparableContext } from "../src/eval-harness/types";

const now = "2026-05-01T00:00:00.000Z";

const context: EvalComparableContext = {
  policyId: "policy.replay.runner",
  modelProfileId: "model.replay.runner",
  codebaseProfileId: "codebase.replay.runner",
  modelServerId: "server.replay.runner",
  modelServerProfileId: "server-profile.replay.runner",
  canonicalToolVersion: "canonical-tools.replay.runner",
  renderedToolVersion: "rendered-tools.replay.runner",
  resultStyleVersion: "result-style.replay.runner",
  verificationPolicyVersion: "verification.replay.runner",
};

const baseline: ComparisonRunMetadata = {
  comparisonRunId: "compare.replay.runner.baseline",
  runRole: "baseline",
  artifactId: "policy.replay.runner.baseline",
  artifactVersion: "policy.v1",
  context,
};

const candidate: ComparisonRunMetadata = {
  comparisonRunId: "compare.replay.runner.candidate",
  runRole: "candidate",
  artifactId: "candidate.replay.runner",
  artifactVersion: "candidate.v1",
  context,
};

const makeTempBase = (): Promise<string> => mkdtemp(join(tmpdir(), "bleeding-agent-replay-runner-test-"));

describe("replay eval runner integration", () => {
  test("adapts replay skeletons into eval cases with deterministic telemetry assertions", () => {
    const scenario = routingReplayScenarios.find((candidateScenario) =>
      candidateScenario.scenarioKind === "greeting_no_side_effect"
    );
    expect(scenario).toBeDefined();
    if (scenario == null) {
      throw new Error("greeting scenario missing");
    }

    const evalCase = replayEvalCaseFromSkeleton(
      extractReplayEvalCaseSkeleton({
        capture: scenario.capture,
        metadata: scenario.metadata,
      }),
    );

    expect(evalCase.evalCaseId).toBe("replay.eval.routing.greeting-no-side-effect");
    expect(evalCase.fixtureWorkspace.files[0]).toMatchObject({
      path: "REPLAY_CASE.txt",
      executable: false,
    });
    expect(evalCase.assertions.map((assertion) => assertion.assertionKind)).toEqual([
      "json_pointer_equals",
      "json_pointer_equals",
    ]);
  });

  test("runs baseline and candidate replay comparisons through the eval harness", async () => {
    const baseDir = await makeTempBase();
    const scenario = routingReplayScenarios.find((candidateScenario) =>
      candidateScenario.scenarioKind === "read_only_report"
    );
    expect(scenario).toBeDefined();
    if (scenario == null) {
      throw new Error("read-only report scenario missing");
    }

    try {
      const result = await runReplayEvalComparison({
        replayCases: [scenario],
        baseline,
        candidate,
        baseDir,
        createdAt: now,
        candidatePolicy: async ({ workspacePath }) => {
          await mkdir(join(workspacePath, "incidents"), { recursive: true });
          await writeFile(join(workspacePath, "incidents/summary.md"), "Highest-risk services: api, worker\n");
        },
      });
      try {
        expect(result.replayCases.map((replayCase) => replayCase.evalCaseId)).toEqual([
          "replay.eval.routing.read-only-report",
        ]);
        expect(result.baselineResults[0]?.status).toBe("passed");
        expect(result.candidateResults[0]?.status).toBe("failed");
        expect(result.candidateResults[0]?.assertionResults).toContainEqual(expect.objectContaining({
          assertionId: "assert.routing.report.no-edits",
          passed: false,
        }));
        expect(result.scorecards).toHaveLength(1);
        expect(result.scorecards[0]?.split).toBe("dev");
        expect(result.scorecards[0]?.passed).toBe(false);
      } finally {
        await result.cleanup();
      }
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test("replays terminal command records as offline eval command results", async () => {
    const baseDir = await makeTempBase();
    const scenario = toolCallReplayScenarios.find((candidateScenario) =>
      candidateScenario.scenarioKind === "terminal_verification_enforcement"
    );
    expect(scenario).toBeDefined();
    if (scenario == null) {
      throw new Error("terminal verification scenario missing");
    }

    try {
      const result = await runReplayEvalComparison({
        replayCases: [scenario],
        baseline,
        candidate,
        baseDir,
        createdAt: now,
      });
      try {
        expect(result.baselineResults[0]?.status).toBe("passed");
        expect(result.candidateResults[0]?.status).toBe("passed");
        expect(result.candidateResults[0]?.assertionResults).toContainEqual(expect.objectContaining({
          assertionId: "assert.tool.verify.command-exit",
          assertionKind: "command_exit_code",
          passed: true,
          actual: 2,
        }));
        expect(result.scorecards[0]?.passed).toBe(true);
      } finally {
        await result.cleanup();
      }
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  });

  test(
    "default visible replay corpus keeps capture-backed terminal command records",
    async () => {
      const baseDir = await makeTempBase();

      try {
        const result = await runReplayEvalComparison({
          baseline,
          candidate,
          baseDir,
          createdAt: now,
        });
        try {
          const terminalResult = result.candidateResults.find(
            (run) => run.evalCaseId === "replay.eval.tool-call.terminal-verification-enforcement",
          );

          expect(terminalResult).toBeDefined();
          expect(terminalResult?.assertionResults).toContainEqual(expect.objectContaining({
            assertionId: "assert.tool.verify.command-exit",
            assertionKind: "command_exit_code",
            passed: true,
            actual: 2,
          }));
          expect(result.replayCases.map((replayCase) => replayCase.evalCaseId)).not.toContain(
            "replay.eval.tool-call.mcp-call",
          );
        } finally {
          await result.cleanup();
        }
      } finally {
        await rm(baseDir, { recursive: true, force: true });
      }
    },
    // The default visible replay corpus runs the full set of scenarios
    // sequentially through the eval harness; on a cold worker this clocks
    // ~7-9s of real work (filesystem fixtures, executor invocations,
    // assertion roll-ups). Bun's default 5000ms test timeout is too tight
    // for this integration check; raise it to a realistic budget.
    30_000,
  );
});
