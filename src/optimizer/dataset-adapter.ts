import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import {
  EvalRunResultSchema,
  type EvalRunResult,
  type EvalSplit,
} from "../eval-harness/types";

/**
 * Dataset record produced by `scripts/build_optimizer_dataset.py` and stored at
 * `bench/.bag/optimizer/dataset.jsonl`. The shape is partially typed because the
 * source script is Python-side; we only depend on the fields used here.
 */
export type OptimizerDatasetRecord = {
  trial_id: string;
  task_name: string;
  job_id: string;
  bag_mode: string | null;
  model: string;
  reward: number;
  exception_type: string | null;
  wall_seconds?: number;
  agent_summary?: Record<string, unknown> | null;
  manifest?: { run_id?: string } | null;
  routing?: Record<string, unknown> | null;
  verifier?: { stdout_tail?: string; reward_raw?: string } | null;
  instruction_text?: string;
  source_paths?: Record<string, unknown>;
};

const sanitizeOptimizerId = (value: string): string => {
  const cleaned = value
    .replace(/[^A-Za-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return cleaned.length > 0 ? cleaned : "dataset.unknown";
};

const splitForTrial = (trialId: string): EvalSplit => {
  const hex = createHash("sha256").update(trialId).digest("hex");
  // First 8 hex chars (32 bits) is plenty of entropy for a mod 100 split.
  const bucket = parseInt(hex.slice(0, 8), 16) % 100;
  if (bucket < 70) return "train";
  if (bucket < 85) return "dev";
  return "holdout";
};

const isoTimestamp = (record: OptimizerDatasetRecord): string => {
  // job_id format: 2026-05-01__21-09-19. Convert to a valid ISO-8601 with offset.
  const match = /^(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})$/.exec(record.job_id);
  if (match == null) {
    return new Date(0).toISOString().replace("Z", "+00:00");
  }
  const [, y, mo, d, hh, mm, ss] = match;
  return `${y}-${mo}-${d}T${hh}:${mm}:${ss}.000+00:00`;
};

const completedTimestamp = (record: OptimizerDatasetRecord): string => {
  const start = isoTimestamp(record);
  const wallMs = Math.max(0, Math.round((record.wall_seconds ?? 0) * 1000));
  if (wallMs <= 0) return start;
  const startMs = Date.parse(start);
  if (!Number.isFinite(startMs)) return start;
  const endMs = startMs + wallMs;
  const iso = new Date(endMs).toISOString();
  // toISOString produces ...Z which is rejected by z.string().datetime({offset:true}).
  return iso.replace(/Z$/, "+00:00");
};

const statusForRecord = (record: OptimizerDatasetRecord): EvalRunResult["status"] => {
  if (record.exception_type != null && record.exception_type.length > 0) {
    return "error";
  }
  if (record.reward >= 1) return "passed";
  if (record.reward <= 0) return "failed";
  return "inconclusive";
};

const lastNonEmptyLine = (text: string): string => {
  for (const line of text.split(/\r?\n/).reverse()) {
    const trimmed = line.trim();
    if (trimmed.length > 0) return trimmed.slice(0, 240);
  }
  return "no verifier output";
};

const contextForRecord = (record: OptimizerDatasetRecord): EvalRunResult["context"] => {
  const modelId = sanitizeOptimizerId(`model.${record.model}`);
  const codebaseId = sanitizeOptimizerId(`codebase.${record.task_name}`);
  const policyId = sanitizeOptimizerId(`policy.${record.bag_mode ?? "none"}`);
  return {
    policyId,
    modelProfileId: modelId,
    codebaseProfileId: codebaseId,
    modelServerId: sanitizeOptimizerId(`server.${record.model}`),
    modelServerProfileId: sanitizeOptimizerId(`server.${record.model}.profile`),
    canonicalToolVersion: "v1",
    renderedToolVersion: "v1",
    resultStyleVersion: "v1",
    verificationPolicyVersion: "v1",
  };
};

export const datasetRecordToEvalRunResult = (record: OptimizerDatasetRecord): EvalRunResult => {
  const split = splitForTrial(record.trial_id);
  const runResultId = sanitizeOptimizerId(`run.${record.trial_id}`);
  const evalCaseId = sanitizeOptimizerId(`case.${record.task_name}`);
  const comparisonRunId = sanitizeOptimizerId(`compare.${record.task_name}.${record.job_id}`);
  const startedAt = isoTimestamp(record);
  const completedAt = completedTimestamp(record);
  const status = statusForRecord(record);
  const verifierTail = record.verifier?.stdout_tail ?? "";
  const summaryLine = lastNonEmptyLine(verifierTail);
  const context = contextForRecord(record);
  const score = Math.max(0, Math.min(1, Number(record.reward) || 0));

  const assertion = {
    assertionId: sanitizeOptimizerId(`assertion.${record.trial_id}.verifier`),
    assertionKind: "command_exit_code" as const,
    passed: status === "passed",
    severity: status === "passed" ? "info" : "failure" as const,
    message: summaryLine,
  };

  return EvalRunResultSchema.parse({
    runResultId,
    comparisonRunId,
    runRole: "candidate",
    evalCaseId,
    split,
    context,
    status,
    score,
    assertionResults: [assertion],
    objectiveMetrics: [
      {
        metricId: "metric.reward",
        name: "verifier_reward",
        value: score,
        unit: "score",
        higherIsBetter: true,
      },
    ],
    changedFiles: [],
    startedAt,
    completedAt,
  });
};

export const loadDatasetEvalRunResults = (path: string): EvalRunResult[] => {
  const raw = readFileSync(path, "utf8");
  const out: EvalRunResult[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    const record = JSON.parse(trimmed) as OptimizerDatasetRecord;
    out.push(datasetRecordToEvalRunResult(record));
  }
  return out;
};
