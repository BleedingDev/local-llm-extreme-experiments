import { describe, expect, test } from "bun:test";
import { buildCandidateEvidenceBundle } from "../src/optimizer/evidence";
import { createPromptFragmentProposer } from "../src/optimizer/prompt-fragment-proposer";
import {
  CandidatePatchSchema,
  OptimizerRegistryRecordSchema,
  type OptimizerRegistryRecord,
} from "../src/optimizer/types";
import { GepaFeedbackBundleSchema } from "../src/optimizer/gepa-feedback";

const now = "2026-05-02T00:00:00.000Z";

const lineage = {
  modelProfileIds: ["model.claude-opus-4-7"],
  codebaseProfileIds: ["codebase.bleeding-agent"],
  policyIds: ["policy.bleeding-agent"],
  canonicalToolVersions: ["canonical-tools.v1"],
  renderedToolVersions: ["rendered-tools.v1"],
  resultStyleVersions: ["result-style.v1"],
  verificationPolicyVersions: ["verification.v1"],
};

const evalBundle = () =>
  buildCandidateEvidenceBundle({
    evidenceBundleId: "evidence.prompt-fragment.eval",
    createdAt: now,
    evalRunResults: [
      {
        runResultId: "run.fix-vuln",
        comparisonRunId: "compare.fix-vuln",
        runRole: "candidate",
        evalCaseId: "eval.fix-code-vulnerability",
        split: "dev",
        context: {
          policyId: "policy.bleeding-agent",
          modelProfileId: "model.claude-opus-4-7",
          codebaseProfileId: "codebase.bleeding-agent",
          modelServerId: "server.anthropic",
          modelServerProfileId: "server-profile.anthropic",
          canonicalToolVersion: "canonical-tools.v1",
          renderedToolVersion: "rendered-tools.v1",
          resultStyleVersion: "result-style.v1",
          verificationPolicyVersion: "verification.v1",
        },
        status: "failed",
        score: 0,
        assertionResults: [
          {
            assertionId: "assert.exit",
            assertionKind: "command_exit_code",
            passed: false,
            severity: "critical",
            message: "verifier exited non-zero",
          },
        ],
        objectiveMetrics: [],
        changedFiles: [],
        startedAt: now,
        completedAt: now,
      },
    ],
  });

const seedRecord = (id: string, name: string): OptimizerRegistryRecord =>
  OptimizerRegistryRecordSchema.parse({
    registryRecordId: `registry.${id}`,
    recordKind: "rendered_tool_contract",
    schemaVersion: "v1",
    recordVersion: "1.0.0",
    status: "active",
    createdAt: now,
    updatedAt: now,
    contentHash: `hash-${id}`,
    labels: ["prompt-fragment"],
    payload: {
      renderedToolId: id,
      canonicalToolId: id,
      canonicalToolVersion: "v1",
      renderedToolVersion: "1.0.0",
      modelProfileId: "model.claude-opus-4-7",
      renderer: "prompt-fragment.renderer.v1",
      rendererVersion: "1.0.0",
      name,
      description: `Seed prompt fragment for ${name}`,
      inputSchema: { type: "object" },
      resultStyle: "text",
      resultStyleVersion: "v1",
      promptFragments: ["initial prompt body"],
      examples: [],
    },
  });

describe("prompt-fragment proposer", () => {
  test("emits rendered_tool_contract candidates targeting /promptFragments/-", () => {
    const proposer = createPromptFragmentProposer({
      records: [
        seedRecord("prompt.autonomous-coding-turn.system", "prompt.autonomous-coding-turn.system"),
      ],
    });
    const evidence = evalBundle();
    const result = proposer({
      iteration: 0,
      createdAt: now,
      feedbackBundle: GepaFeedbackBundleSchema.parse({
        feedbackBundleId: "fb.test",
        schemaVersion: "gepa-feedback.v1",
        records: [],
        limits: { maxRecords: 10, maxTextChars: 1000 },
      }),
      feedbackRecords: [],
      evidence,
      maxCandidates: 4,
    });

    const promptCandidates = result.candidates.filter(
      (candidate) => candidate.scope.artifactKind === "rendered_tool_contract",
    );
    expect(promptCandidates.length).toBeGreaterThan(0);
    for (const candidate of promptCandidates) {
      expect(candidate.scope.allowedJsonPointers).toEqual(["/promptFragments/-"]);
      expect(candidate.operations[0]?.op).toBe("add");
      expect(candidate.operations[0]?.path).toBe("/promptFragments/-");
      expect(CandidatePatchSchema.parse(candidate)).toEqual(candidate);
    }
  });

  test("cross-products observations across all eligible seed records", () => {
    const proposer = createPromptFragmentProposer({
      records: [
        seedRecord("prompt.autonomous-coding-turn.system", "prompt.autonomous-coding-turn.system"),
        seedRecord("prompt.task-shape-router.classifier", "prompt.task-shape-router.classifier"),
      ],
    });
    const result = proposer({
      iteration: 0,
      createdAt: now,
      feedbackBundle: GepaFeedbackBundleSchema.parse({
        feedbackBundleId: "fb.test",
        schemaVersion: "gepa-feedback.v1",
        records: [],
        limits: { maxRecords: 10, maxTextChars: 1000 },
      }),
      feedbackRecords: [],
      evidence: evalBundle(),
      maxCandidates: 4,
    });

    const targetIds = new Set(result.candidates.map((candidate) => candidate.scope.artifactId));
    expect(targetIds.has("prompt.autonomous-coding-turn.system")).toBe(true);
    expect(targetIds.has("prompt.task-shape-router.classifier")).toBe(true);
  });

  test("falls back to deterministic proposer when no prompt-fragment seeds supplied", () => {
    const proposer = createPromptFragmentProposer({ records: [] });
    const result = proposer({
      iteration: 0,
      createdAt: now,
      feedbackBundle: GepaFeedbackBundleSchema.parse({
        feedbackBundleId: "fb.test",
        schemaVersion: "gepa-feedback.v1",
        records: [],
        limits: { maxRecords: 10, maxTextChars: 1000 },
      }),
      feedbackRecords: [],
      evidence: evalBundle(),
      maxCandidates: 4,
    });

    for (const candidate of result.candidates) {
      expect(candidate.scope.artifactKind).not.toBe("rendered_tool_contract");
    }
  });
});

// ensure unused import is referenced (lineage drives policyId/modelProfileId via context)
void lineage;
