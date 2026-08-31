import { describe, expect, test } from "bun:test";
import { buildGepaFeedbackBundle, type GepaFeedbackBundle } from "../src/optimizer/gepa-feedback";
import {
  createLlmBackedGepaProposer,
  runGepaOptimizer,
  type GepaCandidateProposer,
  type GepaLlmProposerRequest,
} from "../src/optimizer/gepa-runner";
import type { CandidatePatch, OptimizerRegistryRecord } from "../src/optimizer/types";

const now = "2026-04-30T00:00:00.000Z";

const lineage = {
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  policyId: "policy.qwen36.bleeding-agent",
};

const policyRecord: OptimizerRegistryRecord = {
  registryRecordId: "registry.policy.qwen36.bleeding-agent",
  recordKind: "model_codebase_policy",
  schemaVersion: "optimizer-schema.v1",
  recordVersion: "record.v1",
  status: "active",
  createdAt: now,
  updatedAt: now,
  contentHash: "sha256:policy",
  payload: {
    policyId: lineage.policyId,
    modelProfileId: lineage.modelProfileId,
    codebaseProfileId: lineage.codebaseProfileId,
    canonicalToolVersion: "canonical-tools.v1",
    renderedToolVersion: "rendered-tools.v1",
    resultStyleVersion: "result-style.v1",
    verificationPolicyVersion: "verification.v1",
    candidateScopes: [],
    verificationGates: [],
    maxConcurrentEvaluations: 1,
    riskTolerance: "low",
    status: "draft",
  },
};

const policyCandidate = (id = "candidate.gepa.injected"): CandidatePatch => ({
  candidatePatchId: id,
  policyId: lineage.policyId,
  modelProfileId: lineage.modelProfileId,
  codebaseProfileId: lineage.codebaseProfileId,
  scope: {
    artifactKind: "model_codebase_policy",
    artifactId: lineage.policyId,
    allowedJsonPointers: ["/verificationGates/0"],
  },
  operations: [
    {
      op: "add",
      path: "/verificationGates/0",
      value: {
        gateId: "tool-success-rate",
        metric: "tool-call-success-rate",
        comparator: "gte",
        threshold: 0.95,
        required: true,
      },
    },
  ],
  rationale: "Add a deterministic reliability gate from GEPA feedback.",
  createdAt: now,
  sourceTraceIds: [],
});

const feedbackBundle = (): GepaFeedbackBundle =>
  buildGepaFeedbackBundle({
    feedbackBundleId: "gepa.bundle.runner",
    testOutputs: [
      {
        id: "typecheck",
        text: "TS2322: bad tool result shape",
        ...lineage,
      },
      {
        id: "unit",
        text: "expected repo write result to include changed file",
        ...lineage,
      },
    ],
  });

describe("GEPA runner", () => {
  test("runs a deterministic bounded loop and validates generated candidates when hashes are supplied", () => {
    const first = runGepaOptimizer({
      feedbackBundle: feedbackBundle(),
      createdAt: now,
      maxFeedbackRecordsPerIteration: 1,
      maxIterations: 1,
      records: [policyRecord],
      expectedBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
      actualBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
    });
    const second = runGepaOptimizer({
      feedbackBundle: feedbackBundle(),
      createdAt: now,
      maxFeedbackRecordsPerIteration: 1,
      maxIterations: 1,
      records: [policyRecord],
      expectedBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
      actualBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
    });

    expect(first).toEqual(second);
    expect(first.iterationCount).toBe(1);
    expect(first.exhausted).toBe(false);
    expect(first.processedFeedbackIds).toHaveLength(1);
    expect(first.candidates).toHaveLength(1);
    expect(first.validations).toEqual([
      {
        candidatePatchId: first.candidates[0]?.candidatePatchId,
        valid: true,
        issues: [],
      },
    ]);
    expect(first.diagnostics).not.toContainEqual(expect.objectContaining({
      reason: expect.stringContaining("network"),
    }));
  });

  test("respects candidate caps and returns resumable state", () => {
    const state = runGepaOptimizer({
      feedbackBundle: feedbackBundle(),
      createdAt: now,
      maxIterations: 10,
      maxFeedbackRecordsPerIteration: 1,
      maxCandidatesPerIteration: 1,
      maxTotalCandidates: 1,
    });

    expect(state.candidates).toHaveLength(1);
    expect(state.iterationCount).toBe(1);
    expect(state.exhausted).toBe(false);
    expect(state.diagnostics).toContainEqual({
      iteration: 1,
      severity: "info",
      reason: "total candidate cap reached at 1",
    });
  });

  test("reports no feedback without inventing work", () => {
    const empty = buildGepaFeedbackBundle({
      feedbackBundleId: "gepa.bundle.empty",
      testOutputs: [],
    });

    const state = runGepaOptimizer({
      feedbackBundle: empty,
      createdAt: now,
    });

    expect(state.iterationCount).toBe(0);
    expect(state.exhausted).toBe(true);
    expect(state.candidates).toEqual([]);
    expect(state.diagnostics).toEqual([
      {
        severity: "warning",
        reason: "no feedback records available for GEPA optimization",
      },
    ]);
  });

  test("diagnoses missing lineage and validation inputs instead of fabricating ids", () => {
    const bundle = buildGepaFeedbackBundle({
      feedbackBundleId: "gepa.bundle.missing-lineage",
      testOutputs: [
        {
          id: "unscoped",
          text: "tool call failed but lineage was not pinned",
        },
      ],
    });

    const state = runGepaOptimizer({
      feedbackBundle: bundle,
      createdAt: now,
    });

    expect(state.candidates).toEqual([]);
    expect(state.diagnostics.map((diagnostic) => diagnostic.reason)).toContain(
      "feedback has missing or ambiguous lineage: gepa-feedback.gepa.test_output.unscoped",
    );
    expect(state.diagnostics.map((diagnostic) => diagnostic.reason)).toContain(
      "skipped observation with missing or ambiguous policy/model/codebase lineage",
    );
  });

  test("uses an injected proposer without model or network calls", () => {
    const calls: Array<{
      iteration: number;
      feedbackIds: string[];
      maxCandidates: number;
    }> = [];
    const proposer: GepaCandidateProposer = (input) => {
      calls.push({
        iteration: input.iteration,
        feedbackIds: input.feedbackRecords.map((record) => record.feedbackId),
        maxCandidates: input.maxCandidates,
      });
      return {
        evidenceBundleId: input.evidence.evidenceBundleId,
        candidates: [
          policyCandidate("candidate.gepa.injected.a"),
          policyCandidate("candidate.gepa.injected.b"),
        ],
        diagnostics: [],
      };
    };

    const state = runGepaOptimizer({
      feedbackBundle: feedbackBundle(),
      createdAt: now,
      maxIterations: 1,
      maxCandidatesPerIteration: 1,
      proposer,
      records: [policyRecord],
      expectedBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
    });

    expect(calls).toEqual([
      {
        iteration: 0,
        feedbackIds: ["gepa.test_output.typecheck", "gepa.test_output.unit"],
        maxCandidates: 1,
      },
    ]);
    expect(state.candidates).toHaveLength(1);
    expect(state.candidates[0]?.candidatePatchId).toBe("candidate.gepa.injected.a");
    expect(state.validations[0]?.valid).toBe(true);
    expect(state.diagnostics).toContainEqual({
      iteration: 0,
      severity: "info",
      reason: "candidate cap reached at 1",
    });
  });

  test("rejects autonomous candidates outside prompt, tool, edit, and verification policy scope", () => {
    const proposer: GepaCandidateProposer = (input) => ({
      evidenceBundleId: input.evidence.evidenceBundleId,
      candidates: [
        {
          ...policyCandidate("candidate.gepa.risk-tolerance"),
          scope: {
            artifactKind: "model_codebase_policy",
            artifactId: lineage.policyId,
            allowedJsonPointers: ["/riskTolerance"],
          },
          operations: [
            {
              op: "replace",
              path: "/riskTolerance",
              value: "high",
            },
          ],
          rationale: "Unsafe autonomous scope change.",
        },
      ],
      diagnostics: [],
    });

    const state = runGepaOptimizer({
      feedbackBundle: feedbackBundle(),
      createdAt: now,
      maxIterations: 1,
      proposer,
      records: [policyRecord],
      expectedBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
    });

    expect(state.candidates).toEqual([]);
    expect(state.diagnostics).toContainEqual({
      iteration: 0,
      candidatePatchId: "candidate.gepa.risk-tolerance",
      severity: "warning",
      reason:
        "candidate rejected by autonomous GEPA scope: declared allowed path /riskTolerance is outside prompt/tool/edit/verification policy scope",
    });
    expect(state.diagnostics).toContainEqual({
      iteration: 0,
      severity: "warning",
      reason: "GEPA iteration produced no candidate patches",
    });
  });

  test("uses an injected LLM proposer client while keeping calls offline and scoped", () => {
    const requests: GepaLlmProposerRequest[] = [];
    const proposer = createLlmBackedGepaProposer({
      client: (request) => {
        requests.push(request);
        return {
          evidenceBundleId: request.evidenceBundleId,
          candidates: [policyCandidate("candidate.gepa.llm.valid")],
          diagnostics: [
            {
              severity: "info",
              reason: "fake offline LLM response",
            },
          ],
        };
      },
    });

    const state = runGepaOptimizer({
      feedbackBundle: feedbackBundle(),
      createdAt: now,
      maxIterations: 1,
      maxCandidatesPerIteration: 1,
      proposer,
      records: [policyRecord],
      expectedBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
    });

    expect(requests).toHaveLength(1);
    expect(requests[0]?.allowedScopes).toEqual([
      {
        artifactKind: "model_codebase_policy",
        artifactId: lineage.policyId,
        allowedJsonPointers: ["/verificationGates/0"],
      },
    ]);
    expect(state.candidates.map((candidate) => candidate.candidatePatchId)).toEqual([
      "candidate.gepa.llm.valid",
    ]);
    expect(state.diagnostics).toContainEqual({
      iteration: 0,
      severity: "info",
      reason: "fake offline LLM response",
    });
  });

  test("rejects invalid or out-of-scope LLM candidates and falls back deterministically", () => {
    const outOfScope = {
      ...policyCandidate("candidate.gepa.llm.scope-violation"),
      scope: {
        artifactKind: "model_codebase_policy",
        artifactId: lineage.policyId,
        allowedJsonPointers: ["/riskTolerance"],
      },
      operations: [
        {
          op: "replace",
          path: "/riskTolerance",
          value: "high",
        },
      ],
    };
    const proposer = createLlmBackedGepaProposer({
      client: (request) => ({
        evidenceBundleId: request.evidenceBundleId,
        candidates: [
          { candidatePatchId: "candidate.gepa.llm.schema-invalid" },
          outOfScope,
        ],
        diagnostics: [],
      }),
    });

    const state = runGepaOptimizer({
      feedbackBundle: feedbackBundle(),
      createdAt: now,
      maxIterations: 1,
      proposer,
      records: [policyRecord],
      expectedBaseHashes: {
        [lineage.policyId]: "sha256:policy",
      },
    });

    expect(state.candidates.length).toBeGreaterThan(0);
    expect(state.candidates.map((candidate) => candidate.candidatePatchId)).not.toContain(
      "candidate.gepa.llm.scope-violation",
    );
    expect(state.diagnostics.some((diagnostic) =>
      diagnostic.reason.includes("failed candidate schema validation")
    )).toBe(true);
    expect(state.diagnostics.some((diagnostic) =>
      diagnostic.reason.includes("rejected by scope restrictions")
    )).toBe(true);
    expect(state.diagnostics.map((diagnostic) => diagnostic.reason)).toContain(
      "LLM proposer produced no schema-valid in-scope candidates; deterministic fallback used",
    );
  });
});
