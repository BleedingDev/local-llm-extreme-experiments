import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  createRealAcpStabilityScorecard,
  createSimulatedRealAcpExecutor,
  evaluateRealAcpStabilityPromotionVetoes,
  renderRealAcpStabilityScorecardMarkdown,
  runRealAcpCorpus,
  type RealAcpCorpusRunManifest,
  type RealAcpRunMetadata,
  type RealAcpTaskRunResult,
} from "../src/replay";

const createdAt = "2026-05-04T00:00:00.000Z";

const metadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.scorecard",
    provider: "simulated",
    model: "simulated-acp-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.scorecard",
    rootFingerprint: "sha256:real-acp-scorecard",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.scorecard",
    protectedPathPolicy: "Only task fixture paths may be changed; protected paths must remain unchanged.",
  },
  client: {
    clientProfileId: "client.real-acp.scorecard",
    clientName: "Simulated ACP scorecard harness",
    clientVersion: "v1",
    transport: "simulated",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.scorecard",
    optimizerProfileId: "optimizer.real-acp.scorecard",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
};

const makeTempBase = (): Promise<string> => mkdtemp(join(tmpdir(), "real-acp-scorecard-test-"));

const manifestFixture = async (): Promise<RealAcpCorpusRunManifest> => {
  const baseDir = await makeTempBase();
  try {
    return await runRealAcpCorpus({
      runId: "real-acp-run.scorecard.fixture",
      metadata,
      executor: createSimulatedRealAcpExecutor(),
      taskIds: [
        "real-acp.task.simple-edit-greeting",
        "real-acp.task.protected-path-doc",
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

describe("real ACP stability scorecard", () => {
  test("counts applied-but-broken, wobble, protected-path, repair, rollback, and fallback signals", async () => {
    const base = await manifestFixture();
    const manifest = mutateResult(base, "real-acp.task.simple-edit-greeting", (result) => ({
      ...result,
      status: "failed",
      changedFiles: [
        {
          path: "src/greeter.ts",
          changeKind: "modified",
          beforeHash: "sha256:before",
          afterHash: "sha256:after-b",
        },
      ],
      toolCalls: [
        ...result.toolCalls,
        {
          toolCallId: "tool.scorecard.write",
          namespace: "acp.fs",
          name: "writeTextFile",
          status: "succeeded",
          sideEffectLevel: "write",
        },
      ],
      verifier: { ...result.verifier, status: "failed" },
      repair: { attempted: true, status: "failed", reason: "repair did not restore consistency" },
      rollback: { attempted: true, status: "failed", reason: "rollback failed" },
      editStrategy: { ...result.editStrategy, family: "diff", fallbackStrategyId: "edit.fallback.whole-file" },
      telemetry: {
        codingProgressDiagnostic: {
          schemaVersion: "coding-progress-diagnostics.v1",
          runId: "run.scorecard",
          classifiedAt: createdAt,
          progressClass: "verifier_failed",
          terminal: "final",
          mutatingProgress: {
            changedFileCount: 1,
            successfulWriteCount: 1,
            failedWriteCount: 0,
            terminalCommandCount: 1,
            terminalExitCount: 1,
          },
          generation: {
            editCount: 1,
            parseFailureCount: 0,
            fallbackAttempted: true,
            fallbackEditCount: 1,
            fallbackParseFailureCount: 0,
          },
          verifier: {
            plannedCommandCount: 1,
            executedCommandCount: 1,
            failedCommandCount: 1,
            missing: false,
          },
          failureSignals: ["verifier.failed"],
          evidenceRefs: [],
          reason: "verifier failed",
        },
        writeEvents: [
          { kind: "fs_write", path: "src/greeter.ts", contentHash: "sha256:after-a" },
          { kind: "fs_write", path: "src/greeter.ts", contentHash: "sha256:after-b" },
        ],
        postApplyConsistencyStatus: "inconsistent",
      },
    }));
    const protectedManifest = mutateResult(manifest, "real-acp.task.protected-path-doc", (result) => ({
      ...result,
      changedFiles: [
        {
          path: "secrets/prod.env",
          changeKind: "modified",
          beforeHash: "sha256:secret-before",
          afterHash: "sha256:secret-after",
        },
      ],
    }));

    const scorecard = createRealAcpStabilityScorecard({
      manifests: [protectedManifest],
      scorecardId: "real-acp-stability.test",
      createdAt,
    });

    expect(scorecard.aggregate.appliedButBroken).toMatchObject({ count: 1, rate: 0.5 });
    expect(scorecard.aggregate.wobbled).toMatchObject({ count: 1, rate: 0.5 });
    expect(scorecard.aggregate.protectedPathTouched).toMatchObject({ count: 1, rate: 0.5 });
    expect(scorecard.aggregate.repairFailed.count).toBe(1);
    expect(scorecard.aggregate.rollbackFailed.count).toBe(1);
    expect(scorecard.aggregate.fallbackUsed.count).toBe(1);
    expect(scorecard.taskRecords.find((record) => record.taskId === "real-acp.task.simple-edit-greeting"))
      .toMatchObject({
        appliedButBroken: true,
        codingProgressClass: "verifier_failed",
        postApplyInconsistent: true,
        wobble: {
          wobbledPathCount: 1,
        },
      });
  });

  test("renders a sortable markdown report", async () => {
    const scorecard = createRealAcpStabilityScorecard({
      manifests: [await manifestFixture()],
      scorecardId: "real-acp-stability.markdown",
      createdAt,
    });
    const markdown = renderRealAcpStabilityScorecardMarkdown(scorecard);

    expect(markdown).toContain("# Real ACP Stability Scorecard");
    expect(markdown).toContain("## Aggregate");
    expect(markdown).toContain("| Model | Codebase | Client | Edit family |");
    expect(markdown).toContain("real-acp.task.simple-edit-greeting");
  });

  test("blocks promotion when candidate stability regresses against baseline", async () => {
    const baseline = createRealAcpStabilityScorecard({
      manifests: [await manifestFixture()],
      scorecardId: "real-acp-stability.baseline",
      createdAt,
    });
    const candidateManifest = mutateResult(await manifestFixture(), "real-acp.task.simple-edit-greeting", (result) => ({
      ...result,
      status: "failed",
      changedFiles: [
        {
          path: "src/greeter.ts",
          changeKind: "modified",
          beforeHash: "sha256:before",
          afterHash: "sha256:after",
        },
      ],
      verifier: { ...result.verifier, status: "failed" },
      telemetry: { postApplyConsistencyStatus: "inconsistent" },
    }));
    const candidate = createRealAcpStabilityScorecard({
      manifests: [candidateManifest],
      scorecardId: "real-acp-stability.candidate",
      createdAt,
    });

    const vetoes = evaluateRealAcpStabilityPromotionVetoes({ baseline, candidate });
    expect(vetoes.find((veto) => veto.vetoKind === "applied-broken-regression")).toMatchObject({
      passed: false,
    });
  });
});
