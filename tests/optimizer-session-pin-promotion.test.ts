import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createBagAcpSession } from "../src/acp/session";
import { acpClientCapabilityProfileFromInitialize } from "../src/acp/surface";
import { defaultConfig } from "../src/config";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type { ComparisonRunMetadata, EvalComparableContext, EvalRunResult } from "../src/eval-harness/types";
import {
  loadActiveOptimizerPointer,
  loadOptimizerRegistry,
  saveActiveOptimizerPointer,
  saveOptimizerRegistryRecord,
  seedOptimizerRegistry,
} from "../src/optimizer/registry";
import { resolveLoadedOptimizerPolicy } from "../src/optimizer/policy-resolver";
import {
  promoteCandidatePatch,
  rollbackOptimizerPromotion,
} from "../src/optimizer/promotion";
import { createOptimizerSessionPin } from "../src/optimizer/session-pin";
import type { OptimizerArtifactLineageDecision } from "../src/optimizer/artifact-lineage";
import type { CandidatePatch, OptimizerRegistryRecord } from "../src/optimizer/types";
import type { CandidateValidationResult } from "../src/optimizer/validator";

const now = "2026-04-30T00:00:00.000Z";
const previousPromotionAt = "2026-04-29T00:00:00.000Z";

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-session-pin-promotion-"));
  try {
    writePromotionReadyGateSuite(cwd);
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const firstRecord = <Kind extends OptimizerRegistryRecord["recordKind"]>(
  records: OptimizerRegistryRecord[],
  recordKind: Kind,
  predicate: (record: Extract<OptimizerRegistryRecord, { recordKind: Kind }>) => boolean = () => true,
): Extract<OptimizerRegistryRecord, { recordKind: Kind }> => {
  const record = records.find((entry): entry is Extract<OptimizerRegistryRecord, { recordKind: Kind }> =>
    entry.recordKind === recordKind && predicate(entry as Extract<OptimizerRegistryRecord, { recordKind: Kind }>)
  );
  if (record === undefined) {
    throw new Error(`missing ${recordKind} test fixture`);
  }
  return record;
};

const writePromotionReadyGateSuite = (cwd: string): void => {
  const dir = join(cwd, ".bag", "evidence", "optimizer");
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "index.json"),
    `${JSON.stringify({
      schemaVersion: "local-evidence-optimizer-gate-suite.v1",
      optimizerGateSuiteId: "optimizer-gate-suite.session-pin-promotion",
      graphId: "self-evolving-runtime-gates-v1",
      generatedAt: now,
      sourceEvidenceIndex: ".bag/evidence/index.jsonl",
      sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
      contracts: [
        {
          contractId: "optimizer-runtime-readiness.session-pin-promotion",
          jsonPath: ".bag/evidence/optimizer/runtime-readiness.json",
          markdownPath: "docs/local-evidence-optimizer-runtime-readiness.md",
          primaryUse: "session pin promotion test runtime readiness",
        },
      ],
      currentDecision: {
        candidateGeneration: "allowed_as_scoped_dry_run",
        autoPromotion: "allowed",
        promotionReady: true,
        blockingReasons: [],
      },
      mustFailClosedOn: [
        "missing optimizer gate suite",
        "invalid optimizer gate suite",
        "blocking optimizer gate suite decision",
      ],
      policySeparation: {
        dimensions: ["modelProfileId", "codebaseProfileId", "modelCodebasePolicyId"],
        principle: "Promotion applies only to the exact evaluated model/codebase policy tuple.",
      },
    }, null, 2)}\n`,
    "utf8",
  );
};

const runFor = (
  runRole: "baseline" | "candidate",
  context: EvalComparableContext,
): EvalRunResult => ({
  runResultId: `run.session-pin.${runRole}`,
  comparisonRunId: `compare.session-pin.${runRole}`,
  runRole,
  evalCaseId: "eval.session-pin",
  split: "dev",
  context,
  status: "passed",
  score: 1,
  assertionResults: [
    {
      assertionId: `assert.session-pin.${runRole}`,
      assertionKind: "file_contains",
      passed: true,
      severity: "critical",
      message: "ok",
    },
  ],
  objectiveMetrics: [],
  changedFiles: [],
  startedAt: now,
  completedAt: now,
});

const metadataFor = (
  runRole: "baseline" | "candidate",
  context: EvalComparableContext,
): ComparisonRunMetadata => ({
  comparisonRunId: `compare.session-pin.${runRole}`,
  runRole,
  artifactId: `artifact.session-pin.${runRole}`,
  artifactVersion: "policy.v1",
  context,
});

const passingLineageDecision = (candidatePatchId: string): OptimizerArtifactLineageDecision => ({
  schemaVersion: "optimizer-artifact-lineage.v1",
  lineageManifestId: `lineage.${candidatePatchId}`,
  candidatePatchId,
  promotionAllowed: true,
  decision: "would_promote",
  gates: [],
  blockingGateIds: [],
  report: "lineage gates passed",
});

describe("optimizer session pin promotion", () => {
  test("keeps existing ACP session pins stable while promotion applies to new sessions and resolutions", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const localModel = firstRecord(seedRecords, "model_profile", (record) => record.payload.modelRole === "local");
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const seedPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === localModel.payload.modelProfileId &&
        record.payload.codebaseProfileId === codebase.payload.codebaseProfileId
      );
      const previousPointer = saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: localModel.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activeCodebaseRootFingerprint: codebase.payload.rootFingerprint,
          activePolicyId: seedPolicy.payload.policyId,
          promotedAt: previousPromotionAt,
        },
        cwd,
      );
      const sessions = new Map();
      const clientCapabilities = acpClientCapabilityProfileFromInitialize({}, "test");
      const existingSession = createBagAcpSession({
        config,
        sessions,
        cwd,
        additionalDirectories: [],
        id: "bag-session-pin-before-promotion",
        mcpServers: [],
        clientCapabilities,
        createOptimizerSessionPin: (resolvedCwd) => createOptimizerSessionPin(config, resolvedCwd, "local"),
      });
      const existingPinTelemetry = { ...existingSession.optimizerPin.telemetry };

      const promotedPolicyId = "policy.session-pin.promoted";
      saveOptimizerRegistryRecord(
        config,
        {
          ...seedPolicy,
          registryRecordId: "registry.policy.session-pin.promoted",
          status: "promoted",
          createdAt: now,
          updatedAt: now,
          supersedesRecordId: seedPolicy.registryRecordId,
          labels: ["candidate-promotion", "session-pin"],
          payload: {
            ...seedPolicy.payload,
            policyId: promotedPolicyId,
            status: "promoted",
            canonicalToolVersion: "canonical-tools.session-pin-promoted",
            renderedToolVersion: "rendered-tools.session-pin-promoted",
            resultStyleVersion: "result-style.session-pin-promoted",
            verificationPolicyVersion: "verification.session-pin-promoted",
          },
        },
        cwd,
      );
      const context: EvalComparableContext = {
        policyId: promotedPolicyId,
        modelProfileId: localModel.payload.modelProfileId,
        codebaseProfileId: codebase.payload.codebaseProfileId,
        ...(localModel.payload.modelServerId === undefined ? {} : { modelServerId: localModel.payload.modelServerId }),
        ...(localModel.payload.modelServerProfileId === undefined
          ? {}
          : { modelServerProfileId: localModel.payload.modelServerProfileId }),
        canonicalToolVersion: "canonical-tools.session-pin-promoted",
        renderedToolVersion: "rendered-tools.session-pin-promoted",
        resultStyleVersion: "result-style.session-pin-promoted",
        verificationPolicyVersion: "verification.session-pin-promoted",
      };
      const candidate: CandidatePatch = {
        candidatePatchId: "candidate.session-pin.promoted",
        policyId: promotedPolicyId,
        modelProfileId: localModel.payload.modelProfileId,
        codebaseProfileId: codebase.payload.codebaseProfileId,
        baselinePolicyId: seedPolicy.payload.policyId,
        candidatePolicyId: promotedPolicyId,
        codebaseRootFingerprint: codebase.payload.rootFingerprint,
        scope: {
          artifactKind: "model_codebase_policy",
          artifactId: promotedPolicyId,
          allowedJsonPointers: ["/canonicalToolVersion"],
        },
        operations: [{ op: "replace", path: "/canonicalToolVersion", value: "canonical-tools.session-pin-promoted" }],
        rationale: "Promote a measured model/codebase policy for new session pins only.",
        createdAt: now,
        sourceTraceIds: ["trace.session-pin"],
      };
      const validation: CandidateValidationResult = {
        candidatePatchId: candidate.candidatePatchId,
        valid: true,
        issues: [],
      };
      const scorecard = createEvalScorecard({
        scorecardId: "scorecard.session-pin.promoted",
        evalSuiteId: "suite.session-pin",
        split: "dev",
        baseline: metadataFor("baseline", context),
        candidate: metadataFor("candidate", context),
        baselineResults: [runFor("baseline", context)],
        candidateResults: [runFor("candidate", context)],
        createdAt: now,
      });

      const promotion = promoteCandidatePatch({
        config,
        cwd,
        candidate,
        validation,
        candidateEval: scorecard,
        lineageDecision: passingLineageDecision(candidate.candidatePatchId),
        decidedAt: now,
      });
      const newResolution = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });
      const newSession = createBagAcpSession({
        config,
        sessions,
        cwd,
        additionalDirectories: [],
        id: "bag-session-pin-after-promotion",
        mcpServers: [],
        clientCapabilities,
        createOptimizerSessionPin: (resolvedCwd) => createOptimizerSessionPin(config, resolvedCwd, "local"),
      });

      expect(promotion.promoted).toBe(true);
      expect(promotion.decision.appliesToNewSessionsOnly).toBe(true);
      expect(promotion.previousPointer).toMatchObject({
        activePolicyId: previousPointer.activePolicyId,
        promotedAt: previousPointer.promotedAt,
      });
      expect(existingSession.optimizerPin.telemetry).toEqual(existingPinTelemetry);
      expect(existingSession.optimizerPin.telemetry.policyId).toBe(seedPolicy.payload.policyId);
      expect(existingSession.optimizerPin.resolvedPolicy.policyId).toBe(seedPolicy.payload.policyId);
      expect(newResolution.source).toBe("active_pointer");
      expect(newResolution.policyId).toBe(promotedPolicyId);
      expect(newResolution.canonicalToolVersion).toBe("canonical-tools.session-pin-promoted");
      expect(newSession.optimizerPin.telemetry.policyId).toBe(promotedPolicyId);
      expect(newSession.optimizerPin.telemetry.source).toBe("active_pointer");

      const rolledBack = rollbackOptimizerPromotion({ config, cwd, checkpointPath: promotion.checkpointPath });

      expect(rolledBack).toMatchObject({
        activeModelProfileId: previousPointer.activeModelProfileId,
        activeCodebaseProfileId: previousPointer.activeCodebaseProfileId,
        activeCodebaseRootFingerprint: previousPointer.activeCodebaseRootFingerprint,
        activePolicyId: previousPointer.activePolicyId,
        promotedAt: previousPointer.promotedAt,
      });
      expect(loadActiveOptimizerPointer(config, cwd).pointer?.activePolicyId).toBe(seedPolicy.payload.policyId);
      expect(resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" }).policyId).toBe(
        seedPolicy.payload.policyId,
      );
    });
  });

  test("fails closed when active pointers mismatch the requested codebase or model profile", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const localModel = firstRecord(seedRecords, "model_profile", (record) => record.payload.modelRole === "local");
      const masterModel = firstRecord(seedRecords, "model_profile", (record) => record.payload.modelRole === "master");
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const localPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === localModel.payload.modelProfileId &&
        record.payload.codebaseProfileId === codebase.payload.codebaseProfileId
      );
      const masterPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === masterModel.payload.modelProfileId &&
        record.payload.codebaseProfileId === codebase.payload.codebaseProfileId
      );

      saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: localModel.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activeCodebaseRootFingerprint: "sha256:wrong-codebase",
          activePolicyId: localPolicy.payload.policyId,
          promotedAt: now,
        },
        cwd,
      );

      const codebaseMismatch = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(codebaseMismatch.source).toBe("seed");
      expect(codebaseMismatch.policyId).toBe(localPolicy.payload.policyId);
      expect(codebaseMismatch.codebaseRootFingerprint).toBe(codebase.payload.rootFingerprint);

      saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: masterModel.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activeCodebaseRootFingerprint: codebase.payload.rootFingerprint,
          activePolicyId: masterPolicy.payload.policyId,
          promotedAt: now,
        },
        cwd,
      );

      const profileMismatch = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(profileMismatch.source).toBe("seed");
      expect(profileMismatch.modelProfileId).toBe(localModel.payload.modelProfileId);
      expect(profileMismatch.policyId).toBe(localPolicy.payload.policyId);
      expect(profileMismatch.policyId).not.toBe(masterPolicy.payload.policyId);
    });
  });
});
