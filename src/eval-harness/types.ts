import { z } from "zod";
import { JsonValueSchema, OptimizerIdSchema } from "../optimizer/types";

const IsoTimestampSchema = z.string().datetime({ offset: true });
const RelativePathSchema = z.string().min(1).regex(/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$)).+$/);
const JsonPointerSchema = z.string().regex(/^\//);
const EvalVersionSchema = z.string().min(1);

export const EvalSplitSchema = z.enum(["train", "dev", "holdout"]);
export type EvalSplit = z.infer<typeof EvalSplitSchema>;

export const EvalComparableContextSchema = z.object({
  policyId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  modelServerId: OptimizerIdSchema,
  modelServerProfileId: OptimizerIdSchema,
  canonicalToolVersion: EvalVersionSchema,
  renderedToolVersion: EvalVersionSchema,
  resultStyleVersion: EvalVersionSchema,
  verificationPolicyVersion: EvalVersionSchema,
}).strict();
export type EvalComparableContext = z.infer<typeof EvalComparableContextSchema>;

export const FixtureWorkspaceFileSchema = z.object({
  path: RelativePathSchema,
  content: z.string(),
  executable: z.boolean().default(false),
}).strict();
export type FixtureWorkspaceFile = z.infer<typeof FixtureWorkspaceFileSchema>;

export const FixtureWorkspaceSchema = z.object({
  fixtureWorkspaceId: OptimizerIdSchema,
  name: z.string().min(1),
  description: z.string().min(1).optional(),
  rootFingerprint: z.string().min(1),
  files: z.array(FixtureWorkspaceFileSchema).min(1),
  protectedPaths: z.array(RelativePathSchema).default([]),
  setupCommands: z.array(z.array(z.string().min(1)).min(1)).default([]),
  verificationCommands: z.array(z.array(z.string().min(1)).min(1)).default([]),
}).strict();
export type FixtureWorkspace = z.infer<typeof FixtureWorkspaceSchema>;

const AssertionBaseSchema = z.object({
  assertionId: OptimizerIdSchema,
  description: z.string().min(1),
  severity: z.enum(["info", "warning", "failure", "critical"]).default("failure"),
});

export const FileContainsAssertionSchema = AssertionBaseSchema.extend({
  assertionKind: z.literal("file_contains"),
  path: RelativePathSchema,
  text: z.string().min(1),
}).strict();

export const FileNotContainsAssertionSchema = AssertionBaseSchema.extend({
  assertionKind: z.literal("file_not_contains"),
  path: RelativePathSchema,
  text: z.string().min(1),
}).strict();

export const CommandExitCodeAssertionSchema = AssertionBaseSchema.extend({
  assertionKind: z.literal("command_exit_code"),
  commandId: OptimizerIdSchema,
  expectedExitCode: z.number().int(),
}).strict();

export const NoForbiddenPathChangedAssertionSchema = AssertionBaseSchema.extend({
  assertionKind: z.literal("no_forbidden_path_changed"),
  paths: z.array(RelativePathSchema).min(1),
}).strict();

export const JsonPointerEqualsAssertionSchema = AssertionBaseSchema.extend({
  assertionKind: z.literal("json_pointer_equals"),
  artifact: z.enum(["result", "telemetry", "scorecard"]),
  pointer: JsonPointerSchema,
  expected: JsonValueSchema,
}).strict();

export const LlmJudgeMinScoreAssertionSchema = AssertionBaseSchema.extend({
  assertionKind: z.literal("llm_judge_min_score"),
  rubricId: OptimizerIdSchema,
  minimumScore: z.number().min(0).max(1),
}).strict();

export const EvalAssertionSchema = z.discriminatedUnion("assertionKind", [
  FileContainsAssertionSchema,
  FileNotContainsAssertionSchema,
  CommandExitCodeAssertionSchema,
  NoForbiddenPathChangedAssertionSchema,
  JsonPointerEqualsAssertionSchema,
  LlmJudgeMinScoreAssertionSchema,
]);
export type EvalAssertion = z.infer<typeof EvalAssertionSchema>;

export const EvalCaseSchema = z.object({
  evalCaseId: OptimizerIdSchema,
  schemaVersion: EvalVersionSchema,
  split: EvalSplitSchema,
  title: z.string().min(1),
  task: z.string().min(1),
  fixtureWorkspace: FixtureWorkspaceSchema,
  assertions: z.array(EvalAssertionSchema).min(1),
  tags: z.array(OptimizerIdSchema).default([]),
  timeoutMs: z.number().int().positive(),
}).strict();
export type EvalCase = z.infer<typeof EvalCaseSchema>;

export const ObjectiveMetricSchema = z.object({
  metricId: OptimizerIdSchema,
  name: z.string().min(1),
  value: z.number().finite(),
  unit: z.enum(["score", "ratio", "count", "ms", "tokens", "bytes"]).default("score"),
  higherIsBetter: z.boolean().default(true),
  baselineValue: z.number().finite().optional(),
  candidateValue: z.number().finite().optional(),
  delta: z.number().finite().optional(),
  threshold: z.number().finite().optional(),
}).strict();
export type ObjectiveMetric = z.infer<typeof ObjectiveMetricSchema>;

export const EvalAssertionResultSchema = z.object({
  assertionId: OptimizerIdSchema,
  assertionKind: z.enum([
    "file_contains",
    "file_not_contains",
    "command_exit_code",
    "no_forbidden_path_changed",
    "json_pointer_equals",
    "llm_judge_min_score",
  ]),
  passed: z.boolean(),
  severity: z.enum(["info", "warning", "failure", "critical"]).default("failure"),
  message: z.string().min(1).optional(),
  expected: JsonValueSchema.optional(),
  actual: JsonValueSchema.optional(),
}).strict();
export type EvalAssertionResult = z.infer<typeof EvalAssertionResultSchema>;

export const EvalRunStatusSchema = z.enum(["passed", "failed", "error", "timeout", "inconclusive"]);
export type EvalRunStatus = z.infer<typeof EvalRunStatusSchema>;

export const EvalRunResultSchema = z.object({
  runResultId: OptimizerIdSchema,
  comparisonRunId: OptimizerIdSchema,
  runRole: z.enum(["baseline", "candidate"]),
  evalCaseId: OptimizerIdSchema,
  split: EvalSplitSchema,
  context: EvalComparableContextSchema,
  candidatePatchId: OptimizerIdSchema.optional(),
  status: EvalRunStatusSchema,
  score: z.number().min(0).max(1),
  assertionResults: z.array(EvalAssertionResultSchema).min(1),
  objectiveMetrics: z.array(ObjectiveMetricSchema).default([]),
  changedFiles: z.array(RelativePathSchema).default([]),
  telemetryArtifactPath: RelativePathSchema.optional(),
  startedAt: IsoTimestampSchema,
  completedAt: IsoTimestampSchema,
}).strict();
export type EvalRunResult = z.infer<typeof EvalRunResultSchema>;

export const CriticalRegressionSchema = z.object({
  regressionId: OptimizerIdSchema,
  evalCaseId: OptimizerIdSchema,
  assertionId: OptimizerIdSchema.optional(),
  metricId: OptimizerIdSchema.optional(),
  reason: z.string().min(1),
  baselineStatus: EvalRunStatusSchema,
  candidateStatus: EvalRunStatusSchema,
  blocksPromotion: z.literal(true),
}).strict();
export type CriticalRegression = z.infer<typeof CriticalRegressionSchema>;

export const CriticalRegressionVetoSchema = z.object({
  vetoed: z.boolean(),
  regressions: z.array(CriticalRegressionSchema).default([]),
}).strict().superRefine((value, ctx) => {
  if (value.regressions.length > 0 && !value.vetoed) {
    ctx.addIssue({
      code: "custom",
      path: ["vetoed"],
      message: "critical regressions must veto promotion",
    });
  }

  if (value.regressions.length === 0 && value.vetoed) {
    ctx.addIssue({
      code: "custom",
      path: ["vetoed"],
      message: "promotion cannot be vetoed without a critical regression",
    });
  }
});
export type CriticalRegressionVeto = z.infer<typeof CriticalRegressionVetoSchema>;

export const ComparisonRunMetadataSchema = z.object({
  comparisonRunId: OptimizerIdSchema,
  runRole: z.enum(["baseline", "candidate"]),
  artifactId: OptimizerIdSchema,
  artifactVersion: EvalVersionSchema,
  context: EvalComparableContextSchema,
}).strict();
export type ComparisonRunMetadata = z.infer<typeof ComparisonRunMetadataSchema>;

export const EvalScorecardSchema = z.object({
  scorecardId: OptimizerIdSchema,
  schemaVersion: EvalVersionSchema,
  evalSuiteId: OptimizerIdSchema,
  split: EvalSplitSchema,
  baseline: ComparisonRunMetadataSchema,
  candidate: ComparisonRunMetadataSchema,
  runResults: z.array(EvalRunResultSchema).min(2),
  objectiveMetrics: z.array(ObjectiveMetricSchema).default([]),
  aggregateScore: z.number().min(0).max(1),
  passed: z.boolean(),
  criticalRegressionVeto: CriticalRegressionVetoSchema,
  createdAt: IsoTimestampSchema,
}).strict().superRefine((value, ctx) => {
  if (value.baseline.runRole !== "baseline") {
    ctx.addIssue({
      code: "custom",
      path: ["baseline", "runRole"],
      message: "baseline metadata must use baseline runRole",
    });
  }

  if (value.candidate.runRole !== "candidate") {
    ctx.addIssue({
      code: "custom",
      path: ["candidate", "runRole"],
      message: "candidate metadata must use candidate runRole",
    });
  }

  const comparableFields = [
    "policyId",
    "modelProfileId",
    "codebaseProfileId",
    "modelServerId",
    "modelServerProfileId",
    "canonicalToolVersion",
    "renderedToolVersion",
    "resultStyleVersion",
    "verificationPolicyVersion",
  ] as const;

  for (const field of comparableFields) {
    if (value.baseline.context[field] !== value.candidate.context[field]) {
      ctx.addIssue({
        code: "custom",
        path: ["candidate", "context", field],
        message: `baseline and candidate are not comparable: ${field} differs`,
      });
    }
  }

  for (const [index, runResult] of value.runResults.entries()) {
    if (runResult.split !== value.split) {
      ctx.addIssue({
        code: "custom",
        path: ["runResults", index, "split"],
        message: "run result split must match scorecard split",
      });
    }

    const metadata = runResult.runRole === "baseline" ? value.baseline : value.candidate;
    if (runResult.comparisonRunId !== metadata.comparisonRunId) {
      ctx.addIssue({
        code: "custom",
        path: ["runResults", index, "comparisonRunId"],
        message: "run result comparisonRunId must match its comparison metadata",
      });
    }

    for (const field of comparableFields) {
      if (runResult.context[field] !== metadata.context[field]) {
        ctx.addIssue({
          code: "custom",
          path: ["runResults", index, "context", field],
          message: `run result context does not match ${runResult.runRole} metadata`,
        });
      }
    }
  }
});
export type EvalScorecard = z.infer<typeof EvalScorecardSchema>;
