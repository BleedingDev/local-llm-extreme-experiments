import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { z } from "zod";
import type { EvalScorecard } from "../eval-harness/types";
import { CandidateEvidenceBundleSchema, type CandidateEvidenceBundle } from "./evidence";
import { CandidatePatchSchema, PromotionDecisionSchema, type CandidatePatch, type PromotionDecision } from "./types";
import { CandidateValidationResultSchema, type CandidateValidationResult } from "./validator";

const DEFAULT_CANDIDATE_ROOT = ".bag/optimizer/candidates";

export const CandidateArtifactManifestSchema = z.object({
  candidatePatchId: z.string().min(1),
  artifactDir: z.string().min(1),
  files: z.object({
    patch: z.string().min(1),
    evidence: z.string().min(1),
    validation: z.string().min(1),
    report: z.string().min(1),
    baselineEval: z.string().min(1).optional(),
    candidateEval: z.string().min(1).optional(),
    decision: z.string().min(1).optional(),
  }).strict(),
  createdAt: z.string(),
}).strict();
export type CandidateArtifactManifest = z.infer<typeof CandidateArtifactManifestSchema>;

export type MaterializeCandidateArtifactsInput = {
  cwd?: string;
  candidateRoot?: string;
  candidate: CandidatePatch;
  evidence: CandidateEvidenceBundle;
  validation: CandidateValidationResult;
  baselineEval?: EvalScorecard;
  candidateEval?: EvalScorecard;
  decision?: PromotionDecision;
  reportMarkdown?: string;
  createdAt?: string;
};

export const materializeCandidateArtifacts = (
  input: MaterializeCandidateArtifactsInput,
): CandidateArtifactManifest => {
  const candidate = CandidatePatchSchema.parse(input.candidate);
  const evidence = CandidateEvidenceBundleSchema.parse(input.evidence);
  const validation = CandidateValidationResultSchema.parse(input.validation);
  const createdAt = input.createdAt ?? new Date().toISOString();
  const cwd = input.cwd ?? process.cwd();
  const root = resolve(cwd, input.candidateRoot ?? DEFAULT_CANDIDATE_ROOT);
  const artifactDir = join(root, safePathSegment(candidate.candidatePatchId));

  mkdirSync(artifactDir, { recursive: true });

  const files: CandidateArtifactManifest["files"] = {
    patch: "patch.json",
    evidence: "evidence.json",
    validation: "validation.json",
    report: "report.md",
    ...(input.baselineEval == null ? {} : { baselineEval: "baseline-eval.json" }),
    ...(input.candidateEval == null ? {} : { candidateEval: "candidate-eval.json" }),
    ...(input.decision == null ? {} : { decision: "decision.json" }),
  };

  writeJson(join(artifactDir, files.patch), candidate);
  writeJson(join(artifactDir, files.evidence), evidence);
  writeJson(join(artifactDir, files.validation), validation);

  if (input.baselineEval != null && files.baselineEval != null) {
    writeJson(join(artifactDir, files.baselineEval), input.baselineEval);
  }
  if (input.candidateEval != null && files.candidateEval != null) {
    writeJson(join(artifactDir, files.candidateEval), input.candidateEval);
  }
  if (input.decision != null && files.decision != null) {
    writeJson(join(artifactDir, files.decision), PromotionDecisionSchema.parse(input.decision));
  }

  writeFileSync(
    join(artifactDir, files.report),
    input.reportMarkdown ?? renderCandidateReport({ candidate, evidence, validation }),
    "utf8",
  );

  const manifest = CandidateArtifactManifestSchema.parse({
    candidatePatchId: candidate.candidatePatchId,
    artifactDir,
    files,
    createdAt,
  });
  writeJson(join(artifactDir, "manifest.json"), manifest);
  return manifest;
};

const renderCandidateReport = (input: {
  candidate: CandidatePatch;
  evidence: CandidateEvidenceBundle;
  validation: CandidateValidationResult;
}): string =>
  [
    `# Candidate ${input.candidate.candidatePatchId}`,
    "",
    `Policy: ${input.candidate.policyId}`,
    `Model profile: ${input.candidate.modelProfileId}`,
    `Codebase profile: ${input.candidate.codebaseProfileId}`,
    `Scope: ${input.candidate.scope.artifactKind} ${input.candidate.scope.artifactId}`,
    `Validation: ${input.validation.valid ? "valid" : "invalid"}`,
    "",
    "## Evidence",
    "",
    `Evidence bundle: ${input.evidence.evidenceBundleId}`,
    `Trace IDs: ${input.evidence.sourceTraceIds.join(", ") || "none"}`,
    `Eval case IDs: ${input.evidence.sourceEvalCaseIds.join(", ") || "none"}`,
    "",
    "## Rationale",
    "",
    input.candidate.rationale,
    "",
    "## Validation Issues",
    "",
    input.validation.issues.length === 0
      ? "- none"
      : input.validation.issues.map((issue) => `- ${issue.code}: ${issue.message}`).join("\n"),
    "",
  ].join("\n");

const writeJson = (path: string, value: unknown): void => {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
};

const safePathSegment = (value: string): string =>
  value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 160) || "candidate";
