import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/config";
import { createEvalScorecard } from "../src/eval-harness/scorer";
import type { ComparisonRunMetadata, EvalComparableContext, EvalRunResult, EvalScorecard, EvalSplit } from "../src/eval-harness/types";
import {
  assessOptimizerArtifactLineage,
  buildOptimizerArtifactLineageManifest,
  type OptimizerArtifactLineageDecision,
} from "../src/optimizer/artifact-lineage";
import { buildCandidateEvidenceBundle } from "../src/optimizer/evidence";
import {
  OPTIMIZER_GATE_SUITE_PATH,
  type OptimizerGateSuite,
} from "../src/optimizer/gate-suite";
import { evaluateNoWritePromotionGate } from "../src/optimizer/no-write-gate";
import { promoteCandidatePatch } from "../src/optimizer/promotion";
import {
  loadActiveOptimizerPointer,
  loadOptimizerRegistry,
  saveActiveOptimizerPointer,
} from "../src/optimizer/registry";
import type { CandidatePatch, PromotionDecision } from "../src/optimizer/types";
import type { CandidateValidationResult } from "../src/optimizer/validator";
import type { NoWriteValidationInput } from "../src/replay/no-write-validation";

const now = "2026-05-04T13:00:00.000Z";

const context: EvalComparableContext = {
  policyId: "policy.runtime-lineage.candidate",
  modelProfileId: "model.runtime-lineage",
  codebaseProfileId: "codebase.runtime-lineage",
  modelServerId: "server.runtime-lineage",
  modelServerProfileId: "server-profile.runtime-lineage",
  canonicalToolVersion: "canonical.runtime-lineage.v1",
  renderedToolVersion: "rendered.runtime-lineage.v1",
  resultStyleVersion: "result.runtime-lineage.v1",
  verificationPolicyVersion: "verification.runtime-lineage.v1",
};

const candidate: CandidatePatch = {
  candidatePatchId: "candidate.runtime-lineage",
  policyId: "policy.runtime-lineage.candidate",
  baselinePolicyId: "policy.runtime-lineage.baseline",
  candidatePolicyId: "policy.runtime-lineage.candidate",
  modelProfileId: "model.runtime-lineage",
  codebaseProfileId: "codebase.runtime-lineage",
  codebaseRootFingerprint: "sha256:runtime-lineage",
  evidenceBundleIds: ["evidence.runtime-lineage"],
  scorecardIds: [
    "scorecard.runtime-lineage.train",
    "scorecard.runtime-lineage.dev",
    "scorecard.runtime-lineage.holdout",
  ],
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: "policy.runtime-lineage.candidate",
    allowedJsonPointers: ["/resultStyleVersion"],
  },
  operations: [
    {
      op: "replace",
      path: "/resultStyleVersion",
      value: "result.runtime-lineage.v2",
    },
  ],
  rationale: "Promote only with runtime gate, eval, profile, no-write, and artifact-lineage proof.",
  createdAt: now,
  sourceTraceIds: ["trace.runtime-lineage"],
};

const validation: CandidateValidationResult = {
  candidatePatchId: candidate.candidatePatchId,
  valid: true,
  issues: [],
};

const metadata = (split: EvalSplit, role: "baseline" | "candidate", candidateContext = context): ComparisonRunMetadata => ({
  comparisonRunId: `compare.runtime-lineage.${split}.${role}`,
  runRole: role,
  artifactId: role === "baseline" ? "policy.runtime-lineage.baseline" : "policy.runtime-lineage.candidate",
  artifactVersion: "policy.v1",
  context: role === "baseline" ? context : candidateContext,
});

const run = (split: EvalSplit, role: "baseline" | "candidate", passed: boolean, candidateContext = context): EvalRunResult => ({
  runResultId: `run.runtime-lineage.${split}.${role}.${passed ? "pass" : "fail"}`,
  comparisonRunId: metadata(split, role, candidateContext).comparisonRunId,
  runRole: role,
  evalCaseId: `eval.runtime-lineage.${split}`,
  split,
  context: role === "baseline" ? context : candidateContext,
  status: passed ? "passed" : "failed",
  score: passed ? 1 : 0,
  assertionResults: [
    {
      assertionId: "assert.runtime-lineage",
      assertionKind: "file_contains",
      passed,
      severity: "critical",
      message: passed ? "ok" : "candidate lost required output",
    },
  ],
  objectiveMetrics: [],
  changedFiles: [],
  startedAt: now,
  completedAt: now,
});

const scorecard = (split: EvalSplit, passed = true, candidateContext = context): EvalScorecard =>
  createEvalScorecard({
    scorecardId: `scorecard.runtime-lineage.${split}${passed ? "" : ".failed"}`,
    evalSuiteId: "suite.runtime-lineage",
    split,
    baseline: metadata(split, "baseline"),
    candidate: metadata(split, "candidate", candidateContext),
    baselineResults: [run(split, "baseline", true)],
    candidateResults: [run(split, "candidate", passed, candidateContext)],
    createdAt: now,
  });

const promotionDecisionFixture: PromotionDecision = {
  promotionDecisionId: "promotion.runtime-lineage.preflight",
  decision: "promote",
  policyId: candidate.policyId,
  candidatePatchId: candidate.candidatePatchId,
  evalResultId: "scorecard.runtime-lineage.dev",
  modelProfileId: candidate.modelProfileId,
  codebaseProfileId: candidate.codebaseProfileId,
  baselinePolicyId: candidate.baselinePolicyId,
  candidatePolicyId: candidate.candidatePolicyId,
  codebaseRootFingerprint: candidate.codebaseRootFingerprint,
  canonicalToolVersion: context.canonicalToolVersion,
  renderedToolVersion: context.renderedToolVersion,
  resultStyleVersion: context.resultStyleVersion,
  verificationPolicyVersion: context.verificationPolicyVersion,
  evidenceBundleIds: ["evidence.runtime-lineage"],
  scorecardIds: [
    "scorecard.runtime-lineage.train",
    "scorecard.runtime-lineage.dev",
    "scorecard.runtime-lineage.holdout",
  ],
  rollbackCheckpointPath: ".bag/optimizer/checkpoints/preflight.json",
  reason: "preflight lineage fixture",
  decidedAt: now,
  decidedBy: "deterministic_gate",
  appliesToNewSessionsOnly: true,
};

const passingLineageDecision = (): OptimizerArtifactLineageDecision => {
  const evidence = buildCandidateEvidenceBundle({
    evidenceBundleId: "evidence.runtime-lineage",
    createdAt: now,
    selectedSpanExcerpts: [
      {
        traceId: "trace.runtime-lineage",
        spanId: "span.runtime-lineage",
        text: "Runtime lineage proof spans the exact evaluated model/codebase policy tuple.",
        lineage: {
          modelProfileIds: [candidate.modelProfileId],
          codebaseProfileIds: [candidate.codebaseProfileId],
          policyIds: [candidate.policyId],
        },
      },
    ],
  });
  const manifest = buildOptimizerArtifactLineageManifest({
    candidate,
    validation,
    visibleScorecards: [scorecard("train"), scorecard("dev")],
    holdoutScorecards: [scorecard("holdout")],
    evidenceBundles: [evidence],
    promotionDecision: promotionDecisionFixture,
    rollbackCheckpointPath: ".bag/optimizer/checkpoints/preflight.json",
  });
  return assessOptimizerArtifactLineage(manifest);
};

const blockingLineageDecision = (): OptimizerArtifactLineageDecision =>
  assessOptimizerArtifactLineage(buildOptimizerArtifactLineageManifest({
    candidate: {
      ...candidate,
      evidenceBundleIds: [],
      scorecardIds: [],
    },
    validation,
    visibleScorecards: [scorecard("dev")],
  }));

const noWriteMutationCase: NoWriteValidationInput = {
  recordId: "record.runtime-lineage.no-write",
  taskId: "task.runtime-lineage.no-write",
  routeSelectedMode: "coding",
  expectedMutation: "edit_existing",
  expectedSideEffect: "mutation",
  changedFiles: [],
  fsWriteCount: 0,
  terminalCreateCount: 0,
  terminalExitCount: 0,
  terminalCommandCount: 0,
  stopReason: "end_turn",
  editStrategyFamily: "none",
  verifierStatus: "failed",
  evidenceRefs: [".bag/evidence/scorecards/no-write-runtime-lineage.json"],
};

const baseSuite = (overrides: Partial<OptimizerGateSuite> = {}): OptimizerGateSuite => ({
  schemaVersion: "local-evidence-optimizer-gate-suite.v1",
  optimizerGateSuiteId: "optimizer-gate-suite.runtime-lineage",
  graphId: "self-evolving-runtime-gates-v1",
  generatedAt: now,
  sourceEvidenceIndex: ".bag/evidence/index.jsonl",
  sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
  contracts: [
    {
      contractId: "optimizer-runtime-lineage.test",
      jsonPath: ".bag/evidence/optimizer/runtime-lineage.json",
      markdownPath: "docs/local-evidence-optimizer-runtime-lineage.md",
      primaryUse: "promotion runtime lineage proof",
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
  ...overrides,
});

const withTempCwd = (fn: (cwd: string) => void, suite: OptimizerGateSuite | "missing" = baseSuite()): void => {
  const cwd = mkdtempSync(join(tmpdir(), "optimizer-runtime-lineage-"));
  try {
    if (suite !== "missing") {
      writeSuite(cwd, suite);
    }
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const writeSuite = (cwd: string, value: OptimizerGateSuite): void => {
  const path = join(cwd, OPTIMIZER_GATE_SUITE_PATH);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
};

const assertRejectedWithoutPointerUpdate = (cwd: string, result: ReturnType<typeof promoteCandidatePatch>): void => {
  const config = defaultConfig();
  expect(result.promoted).toBe(false);
  expect(result.decision.decision).toBe("reject");
  expect(result.activePointer).toBeUndefined();
  expect(loadActiveOptimizerPointer(config, cwd).pointer?.activePolicyId).toBe("policy.previous");
  expect(result.registryRecordIds).toHaveLength(2);

  const registry = loadOptimizerRegistry(config, cwd);
  expect(registry.persistedRecords).toEqual(expect.arrayContaining([
    expect.objectContaining({
      recordKind: "candidate_patch",
      status: "rejected",
    }),
    expect.objectContaining({
      recordKind: "promotion_decision",
      status: "rejected",
    }),
  ]));
};

describe("optimizer promotion runtime lineage gates", () => {
  test("updates the active pointer only after runtime, validation, eval, profile, no-write, lineage, and checkpoint gates pass", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      saveActiveOptimizerPointer(config, {
        schemaVersion: "optimizer-active.v1",
        activeModelProfileId: "model.previous",
        activeCodebaseProfileId: "codebase.previous",
        activePolicyId: "policy.previous",
        promotedAt: "2026-05-03T00:00:00.000Z",
      }, cwd);
      const noWriteGate = evaluateNoWritePromotionGate({
        cases: [{
          ...noWriteMutationCase,
          recordId: "record.runtime-lineage.progress",
          changedFiles: ["src/runtime-lineage.ts"],
          fsWriteCount: 1,
          terminalCreateCount: 1,
          terminalExitCount: 1,
          terminalCommandCount: 1,
          editStrategyFamily: "whole_file",
          verifierStatus: "passed",
        }],
        requireEvidence: true,
      });
      const lineageDecision = passingLineageDecision();

      const result = promoteCandidatePatch({
        config,
        cwd,
        candidate,
        validation,
        candidateEval: scorecard("dev"),
        promotionGatePassed: noWriteGate.passed,
        promotionGateReason: noWriteGate.reasons.join("; "),
        lineageDecision,
        decidedAt: now,
      });

      expect(noWriteGate.passed).toBe(true);
      expect(lineageDecision.promotionAllowed).toBe(true);
      expect(result.promoted).toBe(true);
      expect(result.decision.decision).toBe("promote");
      expect(result.previousPointer).toMatchObject({ activePolicyId: "policy.previous" });
      expect(result.activePointer).toMatchObject({
        activeModelProfileId: candidate.modelProfileId,
        activeCodebaseProfileId: candidate.codebaseProfileId,
        activeCodebaseRootFingerprint: candidate.codebaseRootFingerprint,
        activePolicyId: candidate.policyId,
        promotedAt: now,
      });
      expect(result.checkpointPath).toBeString();
      expect(result.decision.rollbackCheckpointPath).toBe(result.checkpointPath);
      expect(existsSync(result.checkpointPath ?? "")).toBe(true);
      expect(loadActiveOptimizerPointer(config, cwd).pointer?.activePolicyId).toBe(candidate.policyId);
    });
  });

  test("rejects and records decisions without active pointer updates for each blocking gate", () => {
    const cases: Array<{
      name: string;
      suite?: OptimizerGateSuite | "missing";
      validation?: CandidateValidationResult;
      candidateEval?: EvalScorecard;
      candidatePatch?: CandidatePatch;
      promotionGatePassed?: boolean;
      promotionGateReason?: string;
      lineageDecision?: OptimizerArtifactLineageDecision;
      omitLineageDecision?: boolean;
      reasonIncludes: string;
    }> = [
      {
        name: "missing runtime gate suite",
        suite: "missing",
        reasonIncludes: "runtime optimizer promotions fail closed",
      },
      {
        name: "blocking runtime gate suite",
        suite: baseSuite({
          currentDecision: {
            candidateGeneration: "allowed_as_scoped_dry_run",
            autoPromotion: "blocked",
            promotionReady: false,
            blockingReasons: ["runtime gate suite blocks auto-promotion"],
          },
        }),
        reasonIncludes: "runtime gate suite blocks auto-promotion",
      },
      {
        name: "candidate validation failure",
        validation: {
          candidatePatchId: candidate.candidatePatchId,
          valid: false,
          issues: [{
            severity: "error",
            code: "scope_violation",
            message: "operation path is outside candidate scope",
            path: "/outside",
          }],
        },
        reasonIncludes: "validation failed",
      },
      {
        name: "eval scorecard failure",
        candidateEval: scorecard("dev", false),
        reasonIncludes: "eval gates failed",
      },
      {
        name: "model/codebase profile mismatch",
        candidatePatch: {
          ...candidate,
          modelProfileId: "model.runtime-lineage.other",
          codebaseProfileId: "codebase.runtime-lineage.other",
        },
        reasonIncludes: "codebase profile gate failed",
      },
      {
        name: "blocking no-write aggregate gate",
        promotionGatePassed: false,
        promotionGateReason: evaluateNoWritePromotionGate({
          cases: [noWriteMutationCase],
          requireEvidence: true,
        }).reasons.join("; "),
        reasonIncludes: "aggregate promotion gates failed",
      },
      {
        name: "supplied artifact lineage rejection",
        lineageDecision: blockingLineageDecision(),
        reasonIncludes: "artifact lineage gates failed",
      },
      {
        name: "missing artifact lineage decision",
        omitLineageDecision: true,
        reasonIncludes: "artifact lineage decision is required",
      },
    ];

    for (const testCase of cases) {
      withTempCwd((cwd) => {
        const config = defaultConfig();
        saveActiveOptimizerPointer(config, {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: "model.previous",
          activeCodebaseProfileId: "codebase.previous",
          activePolicyId: "policy.previous",
          promotedAt: "2026-05-03T00:00:00.000Z",
        }, cwd);

        const result = promoteCandidatePatch({
          config,
          cwd,
          candidate: testCase.candidatePatch ?? candidate,
          validation: testCase.validation ?? validation,
          candidateEval: testCase.candidateEval ?? scorecard("dev"),
          promotionGatePassed: testCase.promotionGatePassed,
          promotionGateReason: testCase.promotionGateReason,
          ...(testCase.omitLineageDecision === true
            ? {}
            : { lineageDecision: testCase.lineageDecision ?? passingLineageDecision() }),
          decidedAt: now,
          decisionId: `promotion.runtime-lineage.${testCase.name.replace(/[^A-Za-z0-9]+/g, "-").toLowerCase()}`,
        });

        assertRejectedWithoutPointerUpdate(cwd, result);
        expect(result.decision.reason).toContain(testCase.reasonIncludes);
      }, testCase.suite ?? baseSuite());
    }
  });
});
