import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { EVIDENCE_COMMANDS, runEvidenceCommand } from "./evidence-commands";
import {
  createSimulatedRealAcpExecutor,
  runRealAcpCorpus,
  type RealAcpRunMetadata,
} from "../replay";

const writeJson = (cwd: string, path: string, value: unknown) => {
  writeText(cwd, path, `${JSON.stringify(value, null, 2)}\n`);
};

const writeText = (cwd: string, path: string, value: string) => {
  const absolutePath = join(cwd, path);
  mkdirSync(dirname(absolutePath), { recursive: true });
  writeFileSync(absolutePath, value);
};

const sha256File = (cwd: string, path: string): string =>
  createHash("sha256").update(readFileSync(join(cwd, path))).digest("hex");

const realAcpMetadata: RealAcpRunMetadata = {
  model: {
    modelProfileId: "model.real-acp.test",
    provider: "simulated",
    model: "simulated-acp-model",
    modelRole: "local",
    contextWindowTokens: 128000,
    toolCallingMode: "native",
  },
  codebase: {
    codebaseProfileId: "codebase.real-acp.fixture",
    rootFingerprint: "sha256:real-acp-fixture",
    languageSummary: "TypeScript fixture workspaces",
    testRiskTier: "risk.real-acp.fixture",
    protectedPathPolicy: "Only task fixture paths may be changed.",
  },
  client: {
    clientProfileId: "client.real-acp.simulated",
    clientName: "Simulated ACP harness",
    clientVersion: "v1",
    transport: "simulated",
    acpConsumerCapabilities: {
      filesystem: true,
      terminal: true,
    },
  },
  profile: {
    policyId: "policy.real-acp.test",
    optimizerProfileId: "optimizer.real-acp.test",
    verificationPolicyVersion: "verification.real-acp.v1",
    resultStyleVersion: "result.real-acp.v1",
    canonicalToolVersion: "canonical.real-acp.v1",
    renderedToolVersion: "rendered.real-acp.v1",
  },
};

const createEvidenceFixture = async (overrides: { omitScorecardMarkdown?: boolean; graphId?: string; selectionHash?: string } = {}): Promise<string> => {
  const cwd = await mkdtemp(join(tmpdir(), "evidence-commands-"));
  const graphId = overrides.graphId ?? "graph.test";
  const selectionHash = overrides.selectionHash ?? "abc123";
  const candidatePatchId = "candidate.evidence.fixture";
  const promotionDecisionId = "promotion.evidence.fixture";
  writeText(
    cwd,
    ".bag/evidence/index.jsonl",
    [
      {
        schemaVersion: "local-evidence-index.v1",
        recordKind: "source",
        evidenceId: "evidence.source.a",
        title: "Source A",
        path: "trace-gepa/data/a.jsonl",
        family: "action-tool-supervision",
      },
      {
        schemaVersion: "local-evidence-index.v1",
        recordKind: "source",
        evidenceId: "evidence.source.b",
        title: "Source B",
        path: "trace-gepa/data/b.jsonl",
        family: "recovery-transitions",
      },
      {
        schemaVersion: "local-evidence-index.v1",
        recordKind: "slice",
        evidenceId: "slice.combined",
        title: "Combined slice",
        memberEvidenceIds: ["evidence.source.a", "evidence.source.b"],
        family: "optimizer-state",
      },
    ].map((record) => JSON.stringify(record)).join("\n") + "\n",
  );

  writeJson(cwd, ".bag/evidence/scorecards/index.json", {
    schemaVersion: "local-evidence-scorecard-suite.v1",
    scorecardSuiteId: "scorecard-suite.test",
    graphId,
    generatedAt: "2026-05-04T00:00:00Z",
    sourceIndex: ".bag/evidence/index.jsonl",
    scorecards: [
      {
        scorecardId: "scorecard.tool-routing",
        jsonPath: ".bag/evidence/scorecards/tool-routing.json",
        markdownPath: "docs/local-evidence-scorecard-tool-routing.md",
        primaryUse: "routing evidence",
      },
    ],
    promotionGateInputs: ["schema quality"],
    optimizerReadySlices: ["slice.combined"],
  });
  writeJson(cwd, ".bag/evidence/scorecards/tool-routing.json", {
    schemaVersion: "local-evidence-scorecard.tool-routing.v1",
    scorecardId: "scorecard.tool-routing",
    generatedAt: "2026-05-04T00:00:00Z",
  });
  if (overrides.omitScorecardMarkdown !== true) {
    writeText(cwd, "docs/local-evidence-scorecard-tool-routing.md", "# Tool Routing\n");
  }

  writeJson(cwd, ".bag/evidence/optimizer/index.json", {
    schemaVersion: "local-evidence-optimizer-gate-suite.v1",
    optimizerGateSuiteId: "optimizer-gate-suite.test",
    graphId,
    generatedAt: "2026-05-04T00:00:00Z",
    sourceEvidenceIndex: ".bag/evidence/index.jsonl",
    sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
    contracts: [
      {
        contractId: "optimizer-policy-gates.test",
        jsonPath: ".bag/evidence/optimizer/policy-gates.json",
        markdownPath: "docs/local-evidence-optimizer-policy-gates.md",
        primaryUse: "promotion gates",
      },
    ],
    currentDecision: {
      candidateGeneration: "allowed_as_scoped_dry_run",
      autoPromotion: "blocked",
      promotionReady: false,
      blockingReasons: ["operator approval required"],
      candidatePatchId,
      promotionDecisionId,
    },
    mustFailClosedOn: ["schema quality failure"],
  });
  writeJson(cwd, ".bag/evidence/optimizer/policy-gates.json", {
    schemaVersion: "local-evidence-optimizer-policy-gates.v1",
    graphId,
    promotionGates: [],
  });
  writeText(cwd, "docs/local-evidence-optimizer-policy-gates.md", "# Optimizer Gates\n");

  writeJson(cwd, ".bag/evidence/release-proof.json", {
    schemaVersion: "local-evidence-release-proof.v1",
    releaseProofId: "release-proof.test",
    graphId,
    selectionHash,
    generatedAt: "2026-05-04T00:00:00Z",
    validation: {
      jsonParse: "passed",
      indexJsonlParse: "passed",
    },
    optimizerDecision: {
      candidateGeneration: "allowed_as_scoped_dry_run",
      autoPromotion: "blocked",
      promotionReady: false,
      blockingReasons: ["operator approval required"],
      candidatePatchId,
      promotionDecisionId,
    },
    primaryOutputs: [".bag/evidence/index.jsonl"],
    nextExecutionFrontier: ["wire CLI"],
  });
  writePromotionEvidenceContracts(cwd, { graphId, selectionHash, candidatePatchId, promotionDecisionId });
  writeText(cwd, "docs/local-evidence-flywheel-final-report.md", "# Final Report\n");
  return cwd;
};

const writePromotionEvidenceContracts = (
  cwd: string,
  input: {
    graphId: string;
    selectionHash: string;
    candidatePatchId?: string;
    promotionDecisionId?: string;
    generatedAt?: string;
    observedWindowMs?: number;
    requiredWindowMs?: number;
  },
) => {
  const candidatePatchId = input.candidatePatchId ?? "candidate.evidence.fixture";
  const promotionDecisionId = input.promotionDecisionId ?? "promotion.evidence.fixture";
  const generatedAt = input.generatedAt ?? "2026-05-05T01:00:00Z";
  const common = {
    graphId: input.graphId,
    selectionHash: input.selectionHash,
    planSetHash: "plan-set-test",
    evidenceEpochId: `evidence-epoch.${input.graphId}.${input.selectionHash}`,
    sourceGraph: {
      graphId: input.graphId,
      selectionHash: input.selectionHash,
      planSetHash: "plan-set-test",
      snapshotPath: `.codex/plan-graphs/${input.graphId}/snapshot.json`,
    },
    releaseProofRef: {
      path: ".bag/evidence/release-proof.json",
    },
    candidatePatchId,
    promotionDecisionId,
    generatedAt,
  };
  writeJson(cwd, ".bag/evidence/optimizer/operator-approval.json", {
    ...common,
    schemaVersion: "optimizer-operator-approval.v1",
    approvalId: `approval.${candidatePatchId}`,
    approvalKind: "promotion",
    approved: true,
    approvedBy: "test-operator",
    approvedAt: generatedAt,
    notes: ["fixture approval"],
  });
  writeJson(cwd, ".bag/evidence/optimizer/rollback-checkpoint-proof.json", {
    ...common,
    schemaVersion: "optimizer-rollback-checkpoint-proof.v1",
    checkpointProofId: `rollback-proof.${candidatePatchId}`,
    checkpointPath: `.bag/optimizer/checkpoints/${candidatePatchId}.json`,
    checkpointSha256: "sha256:test-checkpoint",
    checkpointCreatedAt: generatedAt,
    previousPointerHash: "sha256:previous-pointer",
    restoreMode: "dry_run",
    restorable: true,
    rollbackCommand: ["bag", "optimizer", "rollback", "--dry-run"],
  });
  writeJson(cwd, ".bag/evidence/optimizer/post-promotion-monitor-window.json", {
    ...common,
    schemaVersion: "optimizer-post-promotion-monitor-window.v1",
    monitorWindowId: `monitor-window.${candidatePatchId}`,
    promotedPolicyId: "policy.evidence.fixture",
    startedAt: generatedAt,
    completedAt: "2026-05-05T05:00:00Z",
    requiredWindowMs: input.requiredWindowMs ?? 14_400_000,
    observedWindowMs: input.observedWindowMs ?? 14_400_000,
    regressionDetected: false,
    rollbackRequested: false,
    rolledBack: false,
    checkpointPath: `.bag/optimizer/checkpoints/${candidatePatchId}.json`,
    signals: [],
  });
};

const writePlanGraphSnapshot = (cwd: string, input: { graphId: string; selectionHash: string; generatedAt?: string }) => {
  writeJson(cwd, `.codex/plan-graphs/${input.graphId}/snapshot.json`, {
    generated_at: input.generatedAt ?? "2026-05-05T00:00:00Z",
    graph_id: input.graphId,
    plan_set_hash: "plan-set-test",
    selection_hash: input.selectionHash,
    selected_plan_paths: [
      join(cwd, ".codex/plans/live-acp-evidence-readiness/01.plan.md"),
      join(cwd, ".codex/plans/live-acp-evidence-readiness/02.plan.md"),
    ],
    edges: [
      {
        source: "01-live-acp-evidence-regeneration",
        target: "02-current-graph-release-proof-rebuild",
      },
    ],
  });
};

describe("evidence command core", () => {
  test("routes commands to dry-run friendly result shapes", async () => {
    const cwd = await createEvidenceFixture();
    try {
      for (const command of EVIDENCE_COMMANDS.filter((command) => command !== "epoch")) {
        const result = runEvidenceCommand(command, { cwd });
        expect(result.command).toBe(command);
        expect(result.cwd).toBe(cwd);
        expect(result.dryRun).toBe(true);
        expect(result.ok).toBe(true);
        expect(result.exit).toEqual({ intent: "success", code: 0 });
        expect(result.payload).toBeDefined();
        expect(result.artifacts.length).toBeGreaterThan(0);
        expect(result.checks.every((check) => check.passed || check.severity !== "blocking")).toBe(true);
        expect(result.writes.every((write) => write.action === "none" || write.action === "would_write")).toBe(true);
        expect(result.handoffTodos.length).toBeGreaterThan(0);
      }
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("validate aggregates index, scorecards, optimizer gates, and release proof", async () => {
    const cwd = await createEvidenceFixture();
    try {
      const result = runEvidenceCommand("validate", { cwd, dryRun: true });
      expect(result.ok).toBe(true);
      expect(result.payload).toEqual({
        schemaVersion: "evidence-command.validate.v1",
        indexRecords: 3,
        scorecards: 1,
        optimizerContracts: 1,
        releaseProofValidationPassed: true,
        promotionReady: false,
        blockingReasons: ["operator approval required"],
      });
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("missing required artifacts fail closed without payload", async () => {
    const cwd = await createEvidenceFixture({ omitScorecardMarkdown: true });
    try {
      const result = runEvidenceCommand("scorecards", { cwd });
      expect(result.ok).toBe(false);
      expect(result.exit).toEqual({ intent: "missing_artifact", code: 66 });
      expect(result.payload).toBeUndefined();
      expect(result.checks.some((check) =>
        !check.passed &&
        check.severity === "blocking" &&
        check.path === "docs/local-evidence-scorecard-tool-routing.md"
      )).toBe(true);
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("optimizer gates fail closed with precise missing promotion evidence blockers", async () => {
    const cwd = await createEvidenceFixture();
    try {
      await rm(join(cwd, ".bag/evidence/optimizer/operator-approval.json"), { force: true });

      const result = runEvidenceCommand("optimizer-gates", { cwd, dryRun: false });

      expect(result.ok).toBe(false);
      expect(result.exit).toEqual({ intent: "missing_artifact", code: 66 });
      expect(result.payload).toBeUndefined();
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "optimizer.operator-approval.missing",
        passed: false,
        severity: "blocking",
      }));
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "optimizer.promotion-evidence-contracts",
        message: expect.stringContaining("missing operator approval evidence"),
      }));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("optimizer gates fail closed on malformed promotion evidence", async () => {
    const cwd = await createEvidenceFixture();
    try {
      writeText(cwd, ".bag/evidence/optimizer/rollback-checkpoint-proof.json", "{ not json");

      const result = runEvidenceCommand("optimizer-gates", { cwd, dryRun: false });

      expect(result.ok).toBe(false);
      expect(result.exit).toEqual({ intent: "invalid_artifact", code: 65 });
      expect(result.payload).toBeUndefined();
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "optimizer.rollback-checkpoint-proof.parse",
        passed: false,
        severity: "blocking",
      }));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("optimizer gates fail closed on stale or wrong promotion evidence bindings", async () => {
    const cwd = await createEvidenceFixture({ graphId: "current.graph", selectionHash: "current-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
        generatedAt: "2026-05-05T00:00:00Z",
      });
      writePromotionEvidenceContracts(cwd, {
        graphId: "wrong.graph",
        selectionHash: "wrong-selection",
        candidatePatchId: "candidate.evidence.fixture",
        promotionDecisionId: "promotion.evidence.fixture",
        generatedAt: "2026-05-04T00:00:00Z",
      });

      const result = runEvidenceCommand("optimizer-gates", {
        cwd,
        dryRun: false,
        graphId: "current.graph",
      });

      expect(result.ok).toBe(false);
      expect(result.exit).toEqual({ intent: "validation_failed", code: 1 });
      expect(result.payload).toBeUndefined();
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "optimizer.promotion-evidence-contracts",
        message: expect.stringContaining("operator approval targets graph wrong.graph, not current.graph"),
      }));
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "optimizer.promotion-evidence-contracts",
        message: expect.stringContaining("operator approval was generated before the current graph snapshot"),
      }));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("optimizer gates fail closed on wrong candidate promotion evidence", async () => {
    const cwd = await createEvidenceFixture({ graphId: "current.graph", selectionHash: "current-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
        generatedAt: "2026-05-05T00:00:00Z",
      });
      writePromotionEvidenceContracts(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
        candidatePatchId: "candidate.other",
        promotionDecisionId: "promotion.evidence.fixture",
      });

      const result = runEvidenceCommand("optimizer-gates", {
        cwd,
        dryRun: false,
        graphId: "current.graph",
      });

      expect(result.ok).toBe(false);
      expect(result.payload).toBeUndefined();
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "optimizer.promotion-evidence-contracts",
        message: expect.stringContaining("operator approval targets candidate candidate.other, not candidate.evidence.fixture"),
      }));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("release proof dry-run targets the current plan graph instead of stale historical proof", async () => {
    const cwd = await createEvidenceFixture({ graphId: "historical.graph", selectionHash: "old-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
      });

      const result = runEvidenceCommand("release-proof", { cwd, dryRun: true });
      expect(result.ok).toBe(false);
      expect(result.payload).toBeDefined();
      expect(result.payload?.graphId).toBe("current.graph");
      expect(result.payload?.selectionHash).toBe("current-selection");
      expect(result.payload?.proofMode).toBe("current_graph");
      expect(result.payload?.historicalProof).toEqual({
        releaseProofId: "release-proof.test",
        graphId: "historical.graph",
        selectionHash: "old-selection",
        path: ".bag/evidence/history/release-proof.test.old-selection.json",
        staleForCurrentGraph: true,
      });
      expect(result.payload?.validation.scorecardsGraphMatchesCurrent).toBe("failed");
      expect(result.payload?.validation.optimizerGraphMatchesCurrent).toBe("failed");
      expect(result.writes).toContainEqual({
        path: ".bag/evidence/release-proof.json",
        action: "would_write",
        reason: "Release-proof command will rebuild current graph proof from command outputs and plan-graph metadata.",
      });
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("release proof write materializes current graph JSON and markdown when evidence graph matches", async () => {
    const cwd = await createEvidenceFixture({ graphId: "current.graph", selectionHash: "current-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
      });

      const result = runEvidenceCommand("release-proof", { cwd, dryRun: false });
      expect(result.ok).toBe(true);
      expect(result.payload?.graphId).toBe("current.graph");
      expect(result.payload?.selectionHash).toBe("current-selection");
      expect(result.payload?.validationPassed).toBe(true);
      expect(result.payload?.sourceGraph?.dependencyOverlay).toEqual([
        {
          source: "01-live-acp-evidence-regeneration",
          target: "02-current-graph-release-proof-rebuild",
        },
      ]);
      expect(result.payload?.artifactHashes.some((artifact) =>
        artifact.path === ".codex/plan-graphs/current.graph/snapshot.json" &&
        artifact.sha256.length > 0
      )).toBe(true);
      expect(existsSync(join(cwd, ".bag/evidence/release-proof.json"))).toBe(true);
      expect(existsSync(join(cwd, "docs/live-acp-current-release-proof-report.md"))).toBe(true);
      const writtenProof = JSON.parse(readFileSync(join(cwd, ".bag/evidence/release-proof.json"), "utf8")) as { graphId?: string; proofMode?: string };
      expect(writtenProof.graphId).toBe("current.graph");
      expect(writtenProof.proofMode).toBe("current_graph");
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("release proof hashes materialized scorecard and optimizer artifacts after retargeting", async () => {
    const cwd = await createEvidenceFixture({ graphId: "historical.graph", selectionHash: "old-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
      });
      writePromotionEvidenceContracts(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
        candidatePatchId: "candidate.evidence.fixture",
        promotionDecisionId: "promotion.evidence.fixture",
      });

      const result = runEvidenceCommand("release-proof", {
        cwd,
        dryRun: false,
        graphId: "current.graph",
      });

      expect(result.ok).toBe(true);
      const writtenProof = JSON.parse(readFileSync(join(cwd, ".bag/evidence/release-proof.json"), "utf8")) as {
        artifactHashes?: Array<{ path: string; sha256: string }>;
      };
      const artifactHashes = new Map(writtenProof.artifactHashes?.map((artifact) => [artifact.path, artifact.sha256]));
      expect(artifactHashes.get(".bag/evidence/scorecards/index.json")).toBe(sha256File(cwd, ".bag/evidence/scorecards/index.json"));
      expect(artifactHashes.get(".bag/evidence/optimizer/index.json")).toBe(sha256File(cwd, ".bag/evidence/optimizer/index.json"));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("scorecards write materializes edit-attempt records and clears telemetry blocker from optimizer gates", async () => {
    const cwd = await createEvidenceFixture({ graphId: "current.graph", selectionHash: "current-selection" });
    const workspaceBaseDir = await mkdtemp(join(tmpdir(), "evidence-real-acp-workspaces-"));
    try {
      writeJson(cwd, ".bag/evidence/optimizer/index.json", {
        schemaVersion: "local-evidence-optimizer-gate-suite.v1",
        optimizerGateSuiteId: "optimizer-gate-suite.test",
        graphId: "current.graph",
        generatedAt: "2026-05-04T00:00:00Z",
        sourceEvidenceIndex: ".bag/evidence/index.jsonl",
        sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
        contracts: [
          {
            contractId: "optimizer-policy-gates.test",
            jsonPath: ".bag/evidence/optimizer/policy-gates.json",
            markdownPath: "docs/local-evidence-optimizer-policy-gates.md",
            primaryUse: "promotion gates",
          },
        ],
        currentDecision: {
          candidateGeneration: "allowed_as_scoped_dry_run",
          autoPromotion: "blocked",
          promotionReady: false,
          blockingReasons: [
            "edit-policy promotion needs first-class edit attempt telemetry",
            "operator approval required",
            "visible ACP no-write/no-terminal validation blocks promotion: 11/13 case(s) missing required mutation progress",
          ],
        },
        mustFailClosedOn: ["schema quality failure"],
      });
      await runRealAcpCorpus({
        runId: "real-acp-run.test.evidence",
        metadata: realAcpMetadata,
        executor: createSimulatedRealAcpExecutor(),
        purpose: "development_eval",
        executionMode: "dry_run",
        workspaceBaseDir,
        outputDir: join(cwd, ".bag/replay-corpus/real-acp-runs/real-acp-run.test.evidence"),
        currentRepoPath: cwd,
        createdAt: "2026-05-04T00:00:00.000Z",
      });

      const scorecards = runEvidenceCommand("scorecards", {
        cwd,
        dryRun: false,
        graphId: "current.graph",
      });
      expect(scorecards.ok).toBe(true);
      expect(existsSync(join(cwd, ".bag/evidence/edit-attempt-records.jsonl"))).toBe(true);
      expect(existsSync(join(cwd, ".bag/evidence/scorecards/edit-attempt-projection.json"))).toBe(true);
      const records = readFileSync(join(cwd, ".bag/evidence/edit-attempt-records.jsonl"), "utf8")
        .trim()
        .split(/\r?\n/)
        .map((line) => JSON.parse(line) as { schemaVersion?: string; finalOutcome?: string });
      expect(records.length).toBeGreaterThan(0);
      expect(records.every((record) => record.schemaVersion === "acp.edit-attempt-record.v1")).toBe(true);

      const optimizer = runEvidenceCommand("optimizer-gates", {
        cwd,
        dryRun: false,
        graphId: "current.graph",
      });
      expect(optimizer.ok).toBe(true);
      expect(optimizer.payload?.blockingReasons).not.toContain("edit-policy promotion needs first-class edit attempt telemetry");
      expect(optimizer.payload?.blockingReasons).not.toContain("visible ACP no-write/no-terminal validation blocks promotion: 11/13 case(s) missing required mutation progress");
      expect(optimizer.payload?.blockingReasons).not.toContain("operator approval required");
    } finally {
      await rm(cwd, { recursive: true, force: true });
      await rm(workspaceBaseDir, { recursive: true, force: true });
    }
  });

  test("release proof can be pinned to an explicit plan graph id", async () => {
    const cwd = await createEvidenceFixture({ graphId: "current.graph", selectionHash: "current-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
        generatedAt: "2026-05-05T00:00:00Z",
      });
      writePlanGraphSnapshot(cwd, {
        graphId: "newer.but.wrong.graph",
        selectionHash: "wrong-selection",
        generatedAt: "2026-05-06T00:00:00Z",
      });

      const result = runEvidenceCommand("release-proof", {
        cwd,
        dryRun: true,
        graphId: "current.graph",
      });

      expect(result.payload?.graphId).toBe("current.graph");
      expect(result.payload?.selectionHash).toBe("current-selection");
      expect(result.payload?.sourceGraph?.graphId).toBe("current.graph");
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("release proof fails closed when an explicit plan graph id is missing", async () => {
    const cwd = await createEvidenceFixture({ graphId: "current.graph", selectionHash: "current-selection" });
    try {
      const result = runEvidenceCommand("release-proof", {
        cwd,
        dryRun: true,
        graphId: "missing.graph",
      });

      expect(result.ok).toBe(false);
      expect(result.payload).toBeUndefined();
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "release-proof.current-graph-missing",
        passed: false,
        severity: "blocking",
      }));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("canonical epoch indexes the requested graph and blocks stale current-slot artifacts", async () => {
    const cwd = await createEvidenceFixture({ graphId: "historical.graph", selectionHash: "old-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "historical.graph",
        selectionHash: "old-selection",
        generatedAt: "2026-05-03T00:00:00Z",
      });
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
        generatedAt: "2026-05-05T00:00:00Z",
      });

      const result = runEvidenceCommand("epoch", {
        cwd,
        dryRun: false,
        graphId: "current.graph",
      });

      expect(result.ok).toBe(false);
      expect(result.payload?.graphId).toBe("current.graph");
      expect(result.payload?.selectionHash).toBe("current-selection");
      expect(result.payload?.driftStatus).toBe("blocked");
      expect(result.payload?.promotionReady).toBe(false);
      expect(result.payload?.graphInventory).toEqual(expect.arrayContaining([
        expect.objectContaining({
          graphId: "current.graph",
          selectionHash: "current-selection",
          classification: "current",
        }),
        expect.objectContaining({
          graphId: "historical.graph",
          selectionHash: "old-selection",
          classification: "historical",
        }),
      ]));
      expect(result.payload?.stalePaths).toEqual(expect.arrayContaining([
        ".bag/evidence/scorecards/index.json",
        ".bag/evidence/optimizer/index.json",
        ".bag/evidence/release-proof.json",
        "docs/live-acp-current-release-proof-report.md",
      ]));
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "epoch.release-proof.graph",
        passed: false,
        severity: "blocking",
      }));
      expect(existsSync(join(cwd, ".bag/evidence/canonical-epoch.json"))).toBe(true);
      expect(existsSync(join(cwd, "docs/live-acp-canonical-readiness-index.md"))).toBe(true);
      const writtenEpoch = JSON.parse(readFileSync(join(cwd, ".bag/evidence/canonical-epoch.json"), "utf8")) as {
        graphId?: string;
        driftStatus?: string;
        promotionReady?: boolean;
      };
      expect(writtenEpoch).toEqual(expect.objectContaining({
        graphId: "current.graph",
        driftStatus: "blocked",
        promotionReady: false,
      }));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  test("validate includes canonical epoch drift checks for explicit graph validation", async () => {
    const cwd = await createEvidenceFixture({ graphId: "historical.graph", selectionHash: "old-selection" });
    try {
      writePlanGraphSnapshot(cwd, {
        graphId: "current.graph",
        selectionHash: "current-selection",
        generatedAt: "2026-05-05T00:00:00Z",
      });

      const result = runEvidenceCommand("validate", {
        cwd,
        dryRun: true,
        graphId: "current.graph",
      });

      expect(result.ok).toBe(false);
      expect(result.payload).toBeUndefined();
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "epoch.scorecards.graph",
        passed: false,
        severity: "blocking",
      }));
      expect(result.checks).toContainEqual(expect.objectContaining({
        checkId: "epoch.release-proof.selection",
        passed: false,
        severity: "blocking",
      }));
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
  });
});
