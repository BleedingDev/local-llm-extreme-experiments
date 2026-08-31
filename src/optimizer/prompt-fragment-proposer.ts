import { generateCandidatePatches } from "./candidates";
import type { CandidateEvidenceObservation } from "./evidence";
import type { GepaCandidateProposer, GepaCandidateProposerInput } from "./gepa-runner";
import {
  CandidatePatchSchema,
  type CandidatePatch,
  type OptimizerRegistryRecord,
} from "./types";

/**
 * Deterministic proposer that targets `prompt-fragment` registry records.
 *
 * The default proposer in `candidates.ts` only emits prompt-targeted patches
 * when an observation carries `toolNames`. Our autonomous-coding-turn dataset
 * surfaces failures as `eval_run` records with empty `toolNames`, so the
 * default proposer falls through to `model_codebase_policy` patches and the
 * actual system prompts in the registry are never touched.
 *
 * This proposer fixes the gap: it walks each evidence observation, picks the
 * best-matching prompt-fragment seed record (by `modelProfileId`), and emits
 * a `rendered_tool_contract` candidate that appends a hint to
 * `/promptFragments/-`. The deterministic policy candidates from the default
 * proposer are concatenated after the prompt-fragment candidates so a single
 * GEPA iteration produces both kinds of patches up to `maxCandidates`.
 */

const PROMPT_FRAGMENT_LABEL = "prompt-fragment";
const FRAGMENT_APPEND_PATH = "/promptFragments/-";
const MAX_HINT_CHARS = 800;
const MAX_EXCERPT_CHARS = 400;

type PromptFragmentRecord = {
  artifactId: string;
  modelProfileId: string;
  promptId: string;
};

const extractPromptFragmentRecords = (
  records: readonly OptimizerRegistryRecord[],
): PromptFragmentRecord[] => {
  const result: PromptFragmentRecord[] = [];
  for (const record of records) {
    if (record.recordKind !== "rendered_tool_contract") continue;
    if (!record.labels.includes(PROMPT_FRAGMENT_LABEL)) continue;
    result.push({
      artifactId: record.payload.renderedToolId,
      modelProfileId: record.payload.modelProfileId,
      promptId: record.payload.name,
    });
  }
  return result;
};

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 160) || "candidate.empty";

const singleValue = (values: readonly string[]): string | undefined =>
  values.length === 1 ? values[0] : undefined;

const observationLineage = (
  observation: CandidateEvidenceObservation,
): { policyId: string; modelProfileId: string; codebaseProfileId: string } | undefined => {
  const policyId = singleValue(observation.lineage.policyIds);
  const modelProfileId = singleValue(observation.lineage.modelProfileIds);
  const codebaseProfileId = singleValue(observation.lineage.codebaseProfileIds);
  if (policyId == null || modelProfileId == null || codebaseProfileId == null) return undefined;
  return { policyId, modelProfileId, codebaseProfileId };
};

const renderHint = (observation: CandidateEvidenceObservation, promptId: string): string => {
  const excerpt = observation.excerpts[0]?.text?.trim().slice(0, MAX_EXCERPT_CHARS) ?? "";
  const lines = [
    `[GEPA hint for ${promptId} — severity=${observation.severity}]`,
    `Observed failure: ${observation.title}`,
  ];
  if (excerpt.length > 0) lines.push(`Excerpt: ${excerpt}`);
  lines.push("Adjust your approach to avoid this failure mode in future tasks.");
  return lines.join("\n").slice(0, MAX_HINT_CHARS);
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

const compareObservations = (
  left: CandidateEvidenceObservation,
  right: CandidateEvidenceObservation,
): number => {
  const sev = severityRank(right.severity) - severityRank(left.severity);
  if (sev !== 0) return sev;
  return left.observationId.localeCompare(right.observationId);
};

export type CreatePromptFragmentProposerInput = {
  records: readonly OptimizerRegistryRecord[];
  fallbackProposer?: GepaCandidateProposer;
};

export const createPromptFragmentProposer = (
  input: CreatePromptFragmentProposerInput,
): GepaCandidateProposer => {
  const fragmentRecords = extractPromptFragmentRecords(input.records);
  const fallback: GepaCandidateProposer = input.fallbackProposer ?? ((proposerInput) =>
    generateCandidatePatches({
      evidence: proposerInput.evidence,
      createdAt: proposerInput.createdAt,
      maxCandidates: proposerInput.maxCandidates,
    }));

  return (proposerInput: GepaCandidateProposerInput) => {
    const fallbackResult = fallback(proposerInput);
    if (fragmentRecords.length === 0) return fallbackResult;

    const promptCandidates: CandidatePatch[] = [];
    const diagnostics = [...fallbackResult.diagnostics];
    const sortedObservations = [...proposerInput.evidence.observations].sort(compareObservations);

    outer: for (const observation of sortedObservations) {
      const lineage = observationLineage(observation);
      if (lineage == null) {
        diagnostics.push({
          observationId: observation.observationId,
          severity: "info",
          reason: "skipped prompt-fragment proposal: observation lacked single-valued lineage",
        });
        continue;
      }
      const eligible = fragmentRecords.filter(
        (record) => record.modelProfileId === lineage.modelProfileId,
      );
      const targets = eligible.length > 0 ? eligible : fragmentRecords;
      for (const target of targets) {
        if (promptCandidates.length >= proposerInput.maxCandidates) break outer;
        promptCandidates.push(
          CandidatePatchSchema.parse({
            candidatePatchId: stableId(
              "gepa-cand.prompt-fragment",
              target.artifactId,
              observation.observationId,
            ),
            policyId: lineage.policyId,
            modelProfileId: target.modelProfileId,
            codebaseProfileId: lineage.codebaseProfileId,
            scope: {
              artifactKind: "rendered_tool_contract",
              artifactId: target.artifactId,
              allowedJsonPointers: [FRAGMENT_APPEND_PATH],
            },
            operations: [
              {
                op: "add",
                path: FRAGMENT_APPEND_PATH,
                value: renderHint(observation, target.promptId),
              },
            ],
            rationale: `Append prompt-fragment hint to ${target.promptId} from observation ${observation.observationId}.`,
            createdAt: proposerInput.createdAt,
            sourceTraceIds:
              observation.traceIds.length > 0
                ? observation.traceIds.slice(0, 5)
                : proposerInput.evidence.sourceTraceIds.slice(0, 5),
          }),
        );
      }
    }

    const combined: CandidatePatch[] = [];
    const seenIds = new Set<string>();
    for (const candidate of [...promptCandidates, ...fallbackResult.candidates]) {
      if (seenIds.has(candidate.candidatePatchId)) continue;
      seenIds.add(candidate.candidatePatchId);
      combined.push(candidate);
      if (combined.length >= proposerInput.maxCandidates) break;
    }

    return {
      evidenceBundleId: fallbackResult.evidenceBundleId,
      candidates: combined,
      diagnostics,
    };
  };
};
