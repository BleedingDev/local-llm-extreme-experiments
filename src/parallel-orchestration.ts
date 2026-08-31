import { z } from "zod";
import type { CodebaseProfile, CommandSpec, ModelCodebasePolicy, ModelProfile } from "./optimizer/types";
import {
  buildCandidateEvidenceBundle,
  type CandidateEvidenceBundle,
  type CandidateEvidenceLineage,
} from "./optimizer/evidence";

export const ParallelLaneKindSchema = z.enum(["exploration", "implementation", "verification"]);
export type ParallelLaneKind = z.infer<typeof ParallelLaneKindSchema>;

export const ParallelLaneStatusSchema = z.enum(["pending", "running", "completed", "failed", "cancelled", "blocked"]);
export type ParallelLaneStatus = z.infer<typeof ParallelLaneStatusSchema>;

export const ParallelSideEffectPolicySchema = z.enum(["read_only", "writes_allowed", "terminal_allowed"]);
export type ParallelSideEffectPolicy = z.infer<typeof ParallelSideEffectPolicySchema>;

export const ParallelIsolationStrategySchema = z.enum([
  "shared_read_only",
  "dry_run_apply_layer",
  "patch_queue",
  "temp_workspace",
  "git_worktree",
]);
export type ParallelIsolationStrategy = z.infer<typeof ParallelIsolationStrategySchema>;

export const ParallelLaneContractSchema = z.object({
  laneId: z.string().min(1),
  title: z.string().min(1),
  laneKind: ParallelLaneKindSchema,
  status: ParallelLaneStatusSchema.default("pending"),
  sideEffectPolicy: ParallelSideEffectPolicySchema,
  targetPaths: z.array(z.string().min(1)).default([]),
  readPaths: z.array(z.string().min(1)).default([]),
  dependsOnLaneIds: z.array(z.string().min(1)).default([]),
  maxTurns: z.number().int().positive().default(6),
  expectedArtifacts: z.array(z.string().min(1)).default([]),
}).strict();
export type ParallelLaneContract = z.infer<typeof ParallelLaneContractSchema>;

export const ParallelLaneConflictSchema = z.object({
  conflictId: z.string().min(1),
  laneIds: z.tuple([z.string().min(1), z.string().min(1)]),
  path: z.string().min(1),
  severity: z.enum(["warning", "blocking"]),
  message: z.string().min(1),
}).strict();
export type ParallelLaneConflict = z.infer<typeof ParallelLaneConflictSchema>;

export const ParallelConcurrencyPolicySchema = z.object({
  recommendedLaneConcurrency: z.number().int().positive(),
  hardLaneConcurrencyCap: z.number().int().positive(),
  reasons: z.array(z.string().min(1)).default([]),
}).strict();
export type ParallelConcurrencyPolicy = z.infer<typeof ParallelConcurrencyPolicySchema>;

export const ParallelOrchestrationPlanSchema = z.object({
  planId: z.string().min(1),
  lanes: z.array(ParallelLaneContractSchema).min(1),
  conflicts: z.array(ParallelLaneConflictSchema).default([]),
  isolationByLaneId: z.record(z.string(), ParallelIsolationStrategySchema),
  concurrency: ParallelConcurrencyPolicySchema,
  acpProgressLabels: z.array(z.string().min(1)).default([]),
  traceLineage: z.object({
    parentTraceId: z.string().min(1).optional(),
    policyId: z.string().min(1).optional(),
    modelProfileId: z.string().min(1).optional(),
    codebaseProfileId: z.string().min(1).optional(),
  }).strict().default({}),
}).strict();
export type ParallelOrchestrationPlan = z.infer<typeof ParallelOrchestrationPlanSchema>;

export const MergeVerificationPlanSchema = z.object({
  verificationRequired: z.boolean(),
  expectedChangedPaths: z.array(z.string().min(1)).default([]),
  commands: z.array(z.object({
    commandId: z.string().min(1),
    command: z.array(z.string().min(1)).min(1),
    required: z.boolean(),
  }).strict()).default([]),
  blockingConflictIds: z.array(z.string().min(1)).default([]),
  rollbackRequiredOnFailure: z.boolean(),
}).strict();
export type MergeVerificationPlan = z.infer<typeof MergeVerificationPlanSchema>;

export const ParallelLaneOutcomeSchema = z.object({
  laneId: z.string().min(1),
  outcome: z.enum([
    "drift",
    "duplicate_work",
    "merge_conflict",
    "verifier_failure",
    "successful_speedup",
    "cost_regression",
    "latency_regression",
    "cancelled",
  ]),
  summary: z.string().min(1),
  durationMs: z.number().int().nonnegative().optional(),
  tokenCost: z.number().int().nonnegative().optional(),
  traceId: z.string().min(1).optional(),
  spanId: z.string().min(1).optional(),
}).strict();
export type ParallelLaneOutcome = z.infer<typeof ParallelLaneOutcomeSchema>;

export type ResolveParallelConcurrencyPolicyInput = {
  modelProfile?: Pick<ModelProfile, "measuredMaxConcurrentRequests" | "measuredConcurrentThroughputTokensPerSecond">;
  modelCodebasePolicy?: Pick<ModelCodebasePolicy, "maxConcurrentEvaluations" | "riskTolerance">;
  maxLaneConcurrency?: number;
  taskRisk?: "low" | "medium" | "high";
  editConflictRisk?: "low" | "medium" | "high";
  toolFailureRate?: number;
  userMode?: "safe" | "auto" | "yolo";
};

export type BuildParallelOrchestrationPlanInput = ResolveParallelConcurrencyPolicyInput & {
  planId?: string;
  lanes: readonly ParallelLaneContract[];
  parentTraceId?: string;
  policyId?: string;
  modelProfileId?: string;
  codebaseProfileId?: string;
  preferGitWorktree?: boolean;
};

export const buildParallelOrchestrationPlan = (
  input: BuildParallelOrchestrationPlanInput,
): ParallelOrchestrationPlan => {
  const lanes = input.lanes.map((lane) => ParallelLaneContractSchema.parse(lane));
  const conflicts = detectParallelLaneConflicts(lanes);
  const concurrency = resolveParallelConcurrencyPolicy(input);
  const isolationByLaneId = Object.fromEntries(lanes.map((lane) => [
    lane.laneId,
    isolationStrategyForLane(lane, conflicts, input.preferGitWorktree === true),
  ]));

  return ParallelOrchestrationPlanSchema.parse({
    planId: input.planId ?? stableId("parallel-plan", ...lanes.map((lane) => lane.laneId)),
    lanes,
    conflicts,
    isolationByLaneId,
    concurrency,
    acpProgressLabels: lanes.map((lane) => `${lane.laneKind}:${lane.laneId}:${lane.status}`),
    traceLineage: {
      ...(input.parentTraceId === undefined ? {} : { parentTraceId: input.parentTraceId }),
      ...(input.policyId === undefined ? {} : { policyId: input.policyId }),
      ...(input.modelProfileId === undefined ? {} : { modelProfileId: input.modelProfileId }),
      ...(input.codebaseProfileId === undefined ? {} : { codebaseProfileId: input.codebaseProfileId }),
    },
  });
};

export const resolveParallelConcurrencyPolicy = (
  input: ResolveParallelConcurrencyPolicyInput,
): ParallelConcurrencyPolicy => {
  const configuredCap = boundedInteger(input.maxLaneConcurrency, 4, 1, 128);
  const modelCap = input.modelProfile?.measuredMaxConcurrentRequests ?? configuredCap;
  const policyCap = input.modelCodebasePolicy?.maxConcurrentEvaluations ?? configuredCap;
  const hardLaneConcurrencyCap = Math.max(1, Math.min(configuredCap, modelCap, policyCap));
  const reasons: string[] = [
    `cap=${hardLaneConcurrencyCap} from configured/model/policy limits`,
  ];
  let recommendedLaneConcurrency = hardLaneConcurrencyCap;

  if (input.userMode === "safe") {
    recommendedLaneConcurrency = Math.min(recommendedLaneConcurrency, 1);
    reasons.push("safe mode keeps orchestration serial");
  }
  if (input.taskRisk === "high" || input.editConflictRisk === "high") {
    recommendedLaneConcurrency = Math.min(recommendedLaneConcurrency, 1);
    reasons.push("high task or edit-conflict risk keeps writes serial");
  } else if (input.taskRisk === "medium" || input.editConflictRisk === "medium") {
    recommendedLaneConcurrency = Math.min(recommendedLaneConcurrency, Math.max(1, Math.floor(hardLaneConcurrencyCap / 2)));
    reasons.push("medium risk halves the lane budget");
  }
  if ((input.toolFailureRate ?? 0) >= 0.2) {
    recommendedLaneConcurrency = Math.min(recommendedLaneConcurrency, Math.max(1, Math.floor(hardLaneConcurrencyCap / 2)));
    reasons.push("recent tool failure rate throttles parallelism");
  }
  if (input.modelCodebasePolicy?.riskTolerance === "low" && input.userMode !== "yolo") {
    recommendedLaneConcurrency = Math.min(recommendedLaneConcurrency, Math.max(1, Math.floor(hardLaneConcurrencyCap / 2)));
    reasons.push("low policy risk tolerance avoids full parallel fanout");
  }

  return ParallelConcurrencyPolicySchema.parse({
    recommendedLaneConcurrency,
    hardLaneConcurrencyCap,
    reasons,
  });
};

export const detectParallelLaneConflicts = (
  lanes: readonly ParallelLaneContract[],
): ParallelLaneConflict[] => {
  const parsed = lanes.map((lane) => ParallelLaneContractSchema.parse(lane));
  const conflicts: ParallelLaneConflict[] = [];
  for (const [leftIndex, left] of parsed.entries()) {
    if (left.sideEffectPolicy === "read_only") continue;
    for (const right of parsed.slice(leftIndex + 1)) {
      if (right.sideEffectPolicy === "read_only") continue;
      for (const leftPath of left.targetPaths) {
        for (const rightPath of right.targetPaths) {
          const conflictPath = overlappingPath(leftPath, rightPath);
          if (conflictPath === undefined) continue;
          conflicts.push(ParallelLaneConflictSchema.parse({
            conflictId: stableId("parallel-conflict", left.laneId, right.laneId, conflictPath),
            laneIds: [left.laneId, right.laneId],
            path: conflictPath,
            severity: "blocking",
            message: `lanes ${left.laneId} and ${right.laneId} both target ${conflictPath}`,
          }));
        }
      }
    }
  }
  return conflicts.sort((left, right) => left.conflictId.localeCompare(right.conflictId));
};

export const buildParallelMergeVerificationPlan = (input: {
  orchestrationPlan: ParallelOrchestrationPlan;
  codebaseProfile?: Pick<CodebaseProfile, "testCommands" | "typecheckCommands" | "lintCommands">;
}): MergeVerificationPlan => {
  const implementationLanes = input.orchestrationPlan.lanes.filter((lane) => lane.laneKind === "implementation");
  const expectedChangedPaths = uniqueSorted(implementationLanes.flatMap((lane) => lane.targetPaths));
  const commands = [
    ...commandRows(input.codebaseProfile?.typecheckCommands ?? []),
    ...commandRows(input.codebaseProfile?.testCommands ?? []),
    ...commandRows(input.codebaseProfile?.lintCommands ?? []),
  ];
  const blockingConflictIds = input.orchestrationPlan.conflicts
    .filter((conflict) => conflict.severity === "blocking")
    .map((conflict) => conflict.conflictId);

  return MergeVerificationPlanSchema.parse({
    verificationRequired: implementationLanes.length > 0,
    expectedChangedPaths,
    commands,
    blockingConflictIds,
    rollbackRequiredOnFailure: implementationLanes.length > 0,
  });
};

export const parallelOrchestrationFeedbackToEvidenceBundle = (input: {
  runId: string;
  createdAt: string;
  outcomes: readonly ParallelLaneOutcome[];
  lineage?: Partial<CandidateEvidenceLineage>;
}): CandidateEvidenceBundle => {
  const outcomes = input.outcomes.map((outcome) => ParallelLaneOutcomeSchema.parse(outcome));
  return buildCandidateEvidenceBundle({
    evidenceBundleId: stableId("parallel-evidence", input.runId),
    createdAt: input.createdAt,
    selectedSpanExcerpts: outcomes.map((outcome, index) => ({
      traceId: outcome.traceId ?? `trace.${input.runId}`,
      spanId: outcome.spanId ?? `span.${input.runId}.${outcome.laneId}.${index}`,
      title: `Parallel lane ${outcome.outcome}: ${outcome.laneId}`,
      text: [
        outcome.summary,
        outcome.durationMs === undefined ? "" : `durationMs=${outcome.durationMs}`,
        outcome.tokenCost === undefined ? "" : `tokenCost=${outcome.tokenCost}`,
      ].filter(Boolean).join("\n"),
      ...(input.lineage === undefined ? {} : { lineage: input.lineage }),
    })),
  });
};

const isolationStrategyForLane = (
  lane: ParallelLaneContract,
  conflicts: readonly ParallelLaneConflict[],
  preferGitWorktree: boolean,
): ParallelIsolationStrategy => {
  if (lane.sideEffectPolicy === "read_only" || lane.laneKind === "exploration") {
    return "shared_read_only";
  }
  if (conflicts.some((conflict) => conflict.laneIds.includes(lane.laneId))) {
    return preferGitWorktree ? "git_worktree" : "temp_workspace";
  }
  if (lane.laneKind === "verification") {
    return "dry_run_apply_layer";
  }
  return "patch_queue";
};

const commandRows = (commands: readonly CommandSpec[]): MergeVerificationPlan["commands"] =>
  commands.map((command) => ({
    commandId: command.commandId,
    command: command.command,
    required: command.required,
  }));

const overlappingPath = (left: string, right: string): string | undefined => {
  const a = normalizePath(left);
  const b = normalizePath(right);
  if (a === b) return a;
  if (a.startsWith(`${b}/`)) return b;
  if (b.startsWith(`${a}/`)) return a;
  return undefined;
};

const normalizePath = (path: string): string =>
  path.replace(/\\/g, "/").replace(/^\.\//, "").replace(/\/+$/g, "");

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values.filter((value) => value.length > 0))].sort((left, right) => left.localeCompare(right));

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};

const stableId = (...parts: readonly string[]): string =>
  parts
    .join(".")
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 180) || "parallel.empty";
