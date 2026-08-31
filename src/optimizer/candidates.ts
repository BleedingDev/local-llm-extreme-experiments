import { z } from "zod";
import type { CandidateEvidenceBundle, CandidateEvidenceObservation } from "./evidence";
import {
  CandidatePatchSchema,
  type CandidatePatch,
  type CandidateScope,
  type CandidatePatchOperation,
} from "./types";

const DEFAULT_MAX_CANDIDATES = 8;
const MAX_CANDIDATES = 50;

export const CandidateGenerationDiagnosticSchema = z.object({
  observationId: z.string().min(1).optional(),
  severity: z.enum(["info", "warning", "error"]),
  reason: z.string().min(1),
}).strict();
export type CandidateGenerationDiagnostic = z.infer<typeof CandidateGenerationDiagnosticSchema>;

export const CandidateGenerationResultSchema = z.object({
  evidenceBundleId: z.string().min(1),
  candidates: z.array(CandidatePatchSchema),
  diagnostics: z.array(CandidateGenerationDiagnosticSchema).default([]),
}).strict();
export type CandidateGenerationResult = z.infer<typeof CandidateGenerationResultSchema>;

export type GenerateCandidatePatchesInput = {
  evidence: CandidateEvidenceBundle;
  createdAt?: string;
  maxCandidates?: number;
};

type BuildCandidatePatchInput = {
  evidence: CandidateEvidenceBundle;
  observation: CandidateEvidenceObservation;
  createdAt: string;
  lineage: RequiredLineage;
  scope: CandidateScope;
  operations: CandidatePatchOperation[];
  rationale: string;
};

type RequiredLineage = {
  policyId: string;
  modelProfileId: string;
  codebaseProfileId: string;
};

export const generateCandidatePatches = (input: GenerateCandidatePatchesInput): CandidateGenerationResult => {
  const createdAt = input.createdAt ?? new Date().toISOString();
  const maxCandidates = boundedInteger(input.maxCandidates, DEFAULT_MAX_CANDIDATES, 1, MAX_CANDIDATES);
  const candidates: CandidatePatch[] = [];
  const diagnostics: CandidateGenerationDiagnostic[] = [];

  for (const observation of [...input.evidence.observations].sort(compareEvidenceObservations)) {
    if (candidates.length >= maxCandidates) {
      diagnostics.push({
        severity: "info",
        reason: `candidate cap reached at ${maxCandidates}`,
      });
      break;
    }

    const lineage = requiredLineage(observation);
    if (lineage == null) {
      diagnostics.push({
        observationId: observation.observationId,
        severity: "warning",
        reason: "skipped observation with missing or ambiguous policy/model/codebase lineage",
      });
      continue;
    }

    const proposal = proposalForObservation(observation, lineage);
    if (proposal == null) {
      diagnostics.push({
        observationId: observation.observationId,
        severity: "info",
        reason: "skipped observation without a conservative candidate mapping",
      });
      continue;
    }

    candidates.push(buildCandidatePatch({
      evidence: input.evidence,
      observation,
      createdAt,
      lineage,
      ...proposal,
    }));
  }

  return CandidateGenerationResultSchema.parse({
    evidenceBundleId: input.evidence.evidenceBundleId,
    candidates,
    diagnostics,
  });
};

const proposalForObservation = (
  observation: CandidateEvidenceObservation,
  lineage: RequiredLineage,
): Pick<BuildCandidatePatchInput, "scope" | "operations" | "rationale"> | undefined => {
  const editProposal = editProposalForObservation(observation, lineage);
  if (editProposal !== undefined) {
    return editProposal;
  }

  if (observation.toolNames.length > 0) {
    const toolName = observation.toolNames[0]!;
    return {
      scope: {
        artifactKind: "rendered_tool_contract",
        artifactId: toolName,
        allowedJsonPointers: [
          "/description",
          "/inputSchema",
          "/resultStyle",
          "/promptFragments/0",
          "/examples/0/expectedResultShape",
        ],
      },
      operations: [
        {
          op: "add",
          path: "/promptFragments/0",
          value: renderedToolGuidance(observation),
        },
      ],
      rationale: `Tighten rendered tool guidance for ${toolName} from evidence ${observation.observationId}.`,
    };
  }

  if (observation.source === "eval_run" || observation.source === "eval_scorecard") {
    return {
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
            gateId: stableId("gate", observation.observationId),
            metric: "aggregate-score",
            comparator: "gte",
            threshold: observation.severity === "critical" ? 1 : 0.8,
            required: true,
          },
        },
      ],
      rationale: `Add a verification gate for eval evidence ${observation.observationId}.`,
    };
  }

  if (observation.source === "trace_latency") {
    return {
      scope: {
        artifactKind: "model_codebase_policy",
        artifactId: lineage.policyId,
        allowedJsonPointers: ["/riskTolerance"],
      },
      operations: [
        {
          op: "replace",
          path: "/riskTolerance",
          value: "low",
        },
      ],
      rationale: `Lower policy risk tolerance after latency evidence ${observation.observationId}.`,
    };
  }

  if (observation.source === "span_excerpt" && isResultStyleObservation(observation)) {
    const suffix = stableId("candidate", observation.observationId).split(".").at(-1) ?? "result-style";
    return {
      scope: {
        artifactKind: "model_codebase_policy",
        artifactId: lineage.policyId,
        allowedJsonPointers: ["/resultStyleVersion", "/verificationGates/0"],
      },
      operations: [
        {
          op: "replace",
          path: "/resultStyleVersion",
          value: `result-style.gepa.${suffix}`,
        },
        {
          op: "add",
          path: "/verificationGates/0",
          value: {
            gateId: stableId("gate", "result-style", observation.observationId),
            metric: "tool-result-parse-success-rate",
            comparator: "gte",
            threshold: observation.severity === "critical" ? 1 : 0.9,
            required: true,
          },
        },
      ],
      rationale: `Adjust result style policy from truncation/result-shape evidence ${observation.observationId}.`,
    };
  }

  if (observation.source === "trace_failure") {
    return {
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
            gateId: stableId("gate", observation.observationId),
            metric: "tool-call-success-rate",
            comparator: "gte",
            threshold: 0.95,
            required: true,
          },
        },
      ],
      rationale: `Add a tool reliability gate for trace failure evidence ${observation.observationId}.`,
    };
  }

  return undefined;
};

const editProposalForObservation = (
  observation: CandidateEvidenceObservation,
  lineage: RequiredLineage,
): Pick<BuildCandidatePatchInput, "scope" | "operations" | "rationale"> | undefined => {
  const renderedEditToolContractId = singleValue(observation.lineage.renderedEditToolContractIds);
  if (renderedEditToolContractId !== undefined) {
    return {
      scope: {
        artifactKind: "rendered_tool_contract",
        artifactId: renderedEditToolContractId,
        allowedJsonPointers: [
          "/description",
          "/inputSchema",
          "/resultStyle",
          "/promptFragments/0",
          "/examples/0",
          "/examples/0/expectedResultShape",
        ],
      },
      operations: [
        {
          op: "add",
          path: "/promptFragments/0",
          value: renderedEditContractGuidance(observation),
        },
      ],
      rationale: `Tighten rendered edit contract ${renderedEditToolContractId} from edit evidence ${observation.observationId}.`,
    };
  }

  if (
    observation.lineage.editStrategyFamilies.length === 0 &&
    observation.lineage.editStrategyIds.length === 0 &&
    observation.lineage.editObjectiveSetIds.length === 0
  ) {
    return undefined;
  }

  const suffix = stableId("candidate", observation.observationId).split(".").at(-1) ?? "edit";
  return {
    scope: {
      artifactKind: "model_codebase_policy",
      artifactId: lineage.policyId,
      allowedJsonPointers: [
        "/editStrategyVersion",
        "/editFallbackPolicyVersion",
        "/editRepairPolicyVersion",
        "/editVerifierPolicyVersion",
        "/editObjectiveSetId",
        "/verificationGates/0",
      ],
    },
    operations: [
      {
        op: "replace",
        path: "/editStrategyVersion",
        value: `edit-strategy.gepa.${suffix}`,
      },
      {
        op: "replace",
        path: "/editFallbackPolicyVersion",
        value: `edit-fallback.gepa.${suffix}`,
      },
      {
        op: "replace",
        path: "/editRepairPolicyVersion",
        value: `edit-repair.gepa.${suffix}`,
      },
      {
        op: "replace",
        path: "/editVerifierPolicyVersion",
        value: `edit-verifier.gepa.${suffix}`,
      },
      {
        op: "replace",
        path: "/editObjectiveSetId",
        value: `edit-objectives.gepa.${suffix}`,
      },
      {
        op: "add",
        path: "/verificationGates/0",
        value: {
          gateId: stableId("gate", "edit", observation.observationId),
          metric: "edit-final-consistency-score",
          comparator: "gte",
          threshold: observation.severity === "critical" ? 1 : 0.9,
          required: true,
        },
      },
    ],
    rationale:
      `Optimize edit strategy/fallback/repair/verifier policy from edit evidence ${observation.observationId}.`,
  };
};

const buildCandidatePatch = (input: BuildCandidatePatchInput): CandidatePatch =>
  CandidatePatchSchema.parse({
    candidatePatchId: stableId("candidate", input.evidence.evidenceBundleId, input.observation.observationId),
    policyId: input.lineage.policyId,
    modelProfileId: input.lineage.modelProfileId,
    codebaseProfileId: input.lineage.codebaseProfileId,
    scope: input.scope,
    operations: input.operations,
    rationale: input.rationale,
    createdAt: input.createdAt,
    sourceTraceIds: input.observation.traceIds.length > 0 ? input.observation.traceIds : input.evidence.sourceTraceIds,
  });

const renderedToolGuidance = (observation: CandidateEvidenceObservation): string =>
  [
    `Evidence ${observation.observationId} showed ${observation.title}.`,
    "Before calling this tool, validate required arguments and avoid inventing fields.",
    observation.argumentHashes.length > 0 ? `Observed argument hashes: ${observation.argumentHashes.join(", ")}` : "",
    observation.excerpts.length > 0 ? `Observed excerpt: ${observation.excerpts[0]?.text ?? ""}` : "",
  ].filter(Boolean).join(" ");

const renderedEditContractGuidance = (observation: CandidateEvidenceObservation): string =>
  [
    `Edit evidence ${observation.observationId} showed ${observation.title}.`,
    "Keep parse, preview, stale-context, protected-path, verification, repair, rollback, and fallback outcomes explicit.",
    "Do not hide an applied-but-broken edit behind fallback; emit structured failure evidence for GEPA.",
    observation.lineage.editStrategyFamilies.length > 0
      ? `Observed edit families: ${observation.lineage.editStrategyFamilies.join(", ")}.`
      : "",
    observation.excerpts.length > 0 ? `Observed excerpt: ${observation.excerpts[0]?.text ?? ""}` : "",
  ].filter(Boolean).join(" ");

const isResultStyleObservation = (observation: CandidateEvidenceObservation): boolean => {
  const text = [
    observation.title,
    ...observation.excerpts.map((excerpt) => excerpt.text),
  ].join("\n").toLowerCase();
  return text.includes("truncat") || text.includes("result style") || text.includes("result-shape") ||
    text.includes("schema shape") || text.includes("parse success");
};

const requiredLineage = (observation: CandidateEvidenceObservation): RequiredLineage | undefined => {
  const policyId = singleValue(observation.lineage.policyIds);
  const modelProfileId = singleValue(observation.lineage.modelProfileIds);
  const codebaseProfileId = singleValue(observation.lineage.codebaseProfileIds);
  if (policyId == null || modelProfileId == null || codebaseProfileId == null) {
    return undefined;
  }
  return { policyId, modelProfileId, codebaseProfileId };
};

const compareEvidenceObservations = (
  left: CandidateEvidenceObservation,
  right: CandidateEvidenceObservation,
): number => {
  const severity = severityRank(right.severity) - severityRank(left.severity);
  if (severity !== 0) {
    return severity;
  }
  const source = left.source.localeCompare(right.source);
  return source === 0 ? left.observationId.localeCompare(right.observationId) : source;
};

const severityRank = (severity: CandidateEvidenceObservation["severity"]): number => {
  switch (severity) {
    case "critical":
      return 4;
    case "high":
      return 3;
    case "medium":
      return 2;
    case "low":
      return 1;
  }
};

const singleValue = (values: readonly string[]): string | undefined =>
  values.length === 1 ? values[0] : undefined;

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 160) || "candidate.empty";

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};
