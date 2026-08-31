import { z } from "zod";
import {
  CodingProgressClassSchema,
  codingProgressClassFromTelemetry,
} from "../acp/coding-progress-diagnostics";
import { detectAnswerWobble, type WobbleEntry } from "../audit/answer-wobble";
import { EvalSplitSchema } from "../eval-harness/types";
import { OptimizerIdSchema, type JsonValue } from "../optimizer/types";
import {
  RealAcpCorpusRunManifestSchema,
  type RealAcpCorpusRunManifest,
  type RealAcpTaskRunResult,
} from "./real-acp-runner";
import { realAcpCodingCorpusTaskPack, type RealAcpCorpusTask } from "./real-acp-task-pack";

const SCORECARD_SCHEMA_VERSION = "real-acp-stability-scorecard.v1" as const;

const RateSummarySchema = z.object({
  count: z.number().int().nonnegative(),
  rate: z.number().min(0).max(1),
}).strict();
export type RateSummary = z.infer<typeof RateSummarySchema>;

const WobblePathSummarySchema = z.object({
  path: z.string().min(1),
  versionCount: z.number().int().nonnegative(),
  distinctVersionCount: z.number().int().nonnegative(),
}).strict();
export type WobblePathSummary = z.infer<typeof WobblePathSummarySchema>;

export const RealAcpTaskStabilityRecordSchema = z.object({
  taskId: OptimizerIdSchema,
  runResultId: OptimizerIdSchema,
  runId: OptimizerIdSchema,
  split: EvalSplitSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema,
  strategyFamily: z.enum(["whole_file", "diff", "search_replace", "none"]),
  status: z.enum(["passed", "failed", "skipped", "cancelled", "error"]),
  changedFileCount: z.number().int().nonnegative(),
  writeToolCallCount: z.number().int().nonnegative(),
  terminalCommandCount: z.number().int().nonnegative(),
  verifierStatus: z.enum(["passed", "failed", "skipped", "not_run"]),
  repairAttempted: z.boolean(),
  repairStatus: z.enum(["not_needed", "succeeded", "failed", "skipped"]),
  rollbackAttempted: z.boolean(),
  rollbackStatus: z.enum(["not_needed", "succeeded", "failed", "skipped"]),
  correctionCount: z.number().int().nonnegative(),
  fallbackUsed: z.boolean(),
  codingProgressClass: CodingProgressClassSchema.optional(),
  protectedPathTouched: z.boolean(),
  postApplyInconsistent: z.boolean(),
  appliedButBroken: z.boolean(),
  wobble: z.object({
    wobbledPathCount: z.number().int().nonnegative(),
    writeEventCount: z.number().int().nonnegative(),
    distinctContentVersionCount: z.number().int().nonnegative(),
    paths: z.array(WobblePathSummarySchema).default([]),
  }).strict(),
}).strict();
export type RealAcpTaskStabilityRecord = z.infer<typeof RealAcpTaskStabilityRecordSchema>;

export const RealAcpStabilityGroupSummarySchema = z.object({
  groupId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema,
  strategyFamily: z.enum(["whole_file", "diff", "search_replace", "none"]),
  taskCount: z.number().int().nonnegative(),
  passRate: z.number().min(0).max(1),
  failureRate: z.number().min(0).max(1),
  appliedButBrokenRate: z.number().min(0).max(1),
  wobbleRate: z.number().min(0).max(1),
  protectedPathTouchRate: z.number().min(0).max(1),
  repairAttemptRate: z.number().min(0).max(1),
  repairFailureRate: z.number().min(0).max(1),
  rollbackAttemptRate: z.number().min(0).max(1),
  rollbackFailureRate: z.number().min(0).max(1),
  fallbackRate: z.number().min(0).max(1),
}).strict();
export type RealAcpStabilityGroupSummary = z.infer<typeof RealAcpStabilityGroupSummarySchema>;

export const RealAcpStabilityPromotionVetoSchema = z.object({
  vetoId: OptimizerIdSchema,
  vetoKind: z.enum([
    "wobble-regression",
    "applied-broken-regression",
    "protected-path-regression",
    "rollback-regression",
    "repair-churn-regression",
  ]),
  passed: z.boolean(),
  blocking: z.boolean().default(true),
  metric: OptimizerIdSchema,
  baselineRate: z.number().min(0).max(1),
  candidateRate: z.number().min(0).max(1),
  allowedIncrease: z.number().min(0).max(1),
  message: z.string().min(1),
}).strict();
export type RealAcpStabilityPromotionVeto = z.infer<typeof RealAcpStabilityPromotionVetoSchema>;

export const RealAcpStabilityScorecardSchema = z.object({
  schemaVersion: z.literal(SCORECARD_SCHEMA_VERSION),
  scorecardId: OptimizerIdSchema,
  createdAt: z.string().datetime({ offset: true }),
  runIds: z.array(OptimizerIdSchema).min(1),
  taskCount: z.number().int().nonnegative(),
  aggregate: z.object({
    passed: RateSummarySchema,
    failed: RateSummarySchema,
    cancelled: RateSummarySchema,
    errored: RateSummarySchema,
    appliedButBroken: RateSummarySchema,
    wobbled: RateSummarySchema,
    protectedPathTouched: RateSummarySchema,
    repairAttempted: RateSummarySchema,
    repairFailed: RateSummarySchema,
    rollbackAttempted: RateSummarySchema,
    rollbackFailed: RateSummarySchema,
    fallbackUsed: RateSummarySchema,
  }).strict(),
  taskRecords: z.array(RealAcpTaskStabilityRecordSchema),
  groupSummaries: z.array(RealAcpStabilityGroupSummarySchema),
}).strict();
export type RealAcpStabilityScorecard = z.infer<typeof RealAcpStabilityScorecardSchema>;

export type CreateRealAcpStabilityScorecardInput = {
  manifests: readonly RealAcpCorpusRunManifest[];
  scorecardId?: string;
  createdAt?: string;
  taskPack?: { tasks: readonly RealAcpCorpusTask[] };
};

export type RealAcpStabilityVetoThresholds = {
  maxWobbleRateIncrease?: number;
  maxAppliedButBrokenRateIncrease?: number;
  maxProtectedPathTouchRateIncrease?: number;
  maxRollbackFailureRateIncrease?: number;
  maxRepairAttemptRateIncrease?: number;
};

export const createRealAcpStabilityScorecard = (
  input: CreateRealAcpStabilityScorecardInput,
): RealAcpStabilityScorecard => {
  const manifests = input.manifests.map((manifest) => RealAcpCorpusRunManifestSchema.parse(manifest));
  if (manifests.length === 0) {
    throw new Error("at least one real ACP manifest is required");
  }
  const taskById = new Map((input.taskPack?.tasks ?? realAcpCodingCorpusTaskPack.tasks).map((task) => [task.taskId, task]));
  const taskRecords = manifests.flatMap((manifest) =>
    manifest.taskResults.map((result) => taskRecordForResult(manifest, result, taskById.get(result.taskId))),
  );
  const runIds = [...new Set(manifests.map((manifest) => manifest.runId))].sort((left, right) => left.localeCompare(right));
  return RealAcpStabilityScorecardSchema.parse({
    schemaVersion: SCORECARD_SCHEMA_VERSION,
    scorecardId: input.scorecardId ?? `real-acp-stability.${runIds.join("+")}`,
    createdAt: input.createdAt ?? new Date().toISOString(),
    runIds,
    taskCount: taskRecords.length,
    aggregate: aggregateSummary(taskRecords),
    taskRecords,
    groupSummaries: groupSummaries(taskRecords),
  });
};

export const evaluateRealAcpStabilityPromotionVetoes = (input: {
  baseline: RealAcpStabilityScorecard;
  candidate: RealAcpStabilityScorecard;
  thresholds?: RealAcpStabilityVetoThresholds;
}): RealAcpStabilityPromotionVeto[] => {
  const baseline = RealAcpStabilityScorecardSchema.parse(input.baseline);
  const candidate = RealAcpStabilityScorecardSchema.parse(input.candidate);
  const thresholds = input.thresholds ?? {};
  return [
    veto({
      kind: "wobble-regression",
      metric: "wobbleRate",
      baselineRate: baseline.aggregate.wobbled.rate,
      candidateRate: candidate.aggregate.wobbled.rate,
      allowedIncrease: thresholds.maxWobbleRateIncrease ?? 0,
    }),
    veto({
      kind: "applied-broken-regression",
      metric: "appliedButBrokenRate",
      baselineRate: baseline.aggregate.appliedButBroken.rate,
      candidateRate: candidate.aggregate.appliedButBroken.rate,
      allowedIncrease: thresholds.maxAppliedButBrokenRateIncrease ?? 0,
    }),
    veto({
      kind: "protected-path-regression",
      metric: "protectedPathTouchRate",
      baselineRate: baseline.aggregate.protectedPathTouched.rate,
      candidateRate: candidate.aggregate.protectedPathTouched.rate,
      allowedIncrease: thresholds.maxProtectedPathTouchRateIncrease ?? 0,
    }),
    veto({
      kind: "rollback-regression",
      metric: "rollbackFailureRate",
      baselineRate: baseline.aggregate.rollbackFailed.rate,
      candidateRate: candidate.aggregate.rollbackFailed.rate,
      allowedIncrease: thresholds.maxRollbackFailureRateIncrease ?? 0,
    }),
    veto({
      kind: "repair-churn-regression",
      metric: "repairAttemptRate",
      baselineRate: baseline.aggregate.repairAttempted.rate,
      candidateRate: candidate.aggregate.repairAttempted.rate,
      allowedIncrease: thresholds.maxRepairAttemptRateIncrease ?? 0,
    }),
  ];
};

export const renderRealAcpStabilityScorecardMarkdown = (scorecardInput: RealAcpStabilityScorecard): string => {
  const scorecard = RealAcpStabilityScorecardSchema.parse(scorecardInput);
  const rows = scorecard.groupSummaries
    .map((group) => [
      group.modelProfileId,
      group.codebaseProfileId,
      group.clientProfileId,
      group.strategyFamily,
      String(group.taskCount),
      pct(group.passRate),
      pct(group.appliedButBrokenRate),
      pct(group.wobbleRate),
      pct(group.protectedPathTouchRate),
      pct(group.repairAttemptRate),
      pct(group.rollbackFailureRate),
    ].join(" | "));
  const hotTasks = [...scorecard.taskRecords]
    .sort((left, right) =>
      Number(right.appliedButBroken) - Number(left.appliedButBroken) ||
      right.wobble.wobbledPathCount - left.wobble.wobbledPathCount ||
      left.taskId.localeCompare(right.taskId))
    .slice(0, 20)
    .map((task) =>
      `| ${task.taskId} | ${task.split} | ${task.status} | ${task.strategyFamily} | ${task.changedFileCount} | ${task.writeToolCallCount} | ${task.verifierStatus} | ${task.appliedButBroken ? "yes" : "no"} | ${task.wobble.wobbledPathCount} | ${task.repairStatus} | ${task.rollbackStatus} |`);
  return [
    `# Real ACP Stability Scorecard`,
    ``,
    `Scorecard: \`${scorecard.scorecardId}\``,
    ``,
    `Runs: ${scorecard.runIds.map((runId) => `\`${runId}\``).join(", ")}`,
    ``,
    `## Aggregate`,
    ``,
    `| Metric | Count | Rate |`,
    `| --- | ---: | ---: |`,
    metricRow("passed", scorecard.aggregate.passed),
    metricRow("failed", scorecard.aggregate.failed),
    metricRow("cancelled", scorecard.aggregate.cancelled),
    metricRow("errored", scorecard.aggregate.errored),
    metricRow("applied-but-broken", scorecard.aggregate.appliedButBroken),
    metricRow("wobbled", scorecard.aggregate.wobbled),
    metricRow("protected path touched", scorecard.aggregate.protectedPathTouched),
    metricRow("repair attempted", scorecard.aggregate.repairAttempted),
    metricRow("repair failed", scorecard.aggregate.repairFailed),
    metricRow("rollback attempted", scorecard.aggregate.rollbackAttempted),
    metricRow("rollback failed", scorecard.aggregate.rollbackFailed),
    metricRow("fallback used", scorecard.aggregate.fallbackUsed),
    ``,
    `## By Profile And Edit Family`,
    ``,
    `| Model | Codebase | Client | Edit family | Tasks | Pass | Applied broken | Wobble | Protected | Repair | Rollback failed |`,
    `| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |`,
    ...rows.map((row) => `| ${row} |`),
    ``,
    `## Task Records`,
    ``,
    `| Task | Split | Status | Edit family | Changed files | Write tools | Verifier | Applied broken | Wobbled paths | Repair | Rollback |`,
    `| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |`,
    ...hotTasks,
    ``,
  ].join("\n");
};

const taskRecordForResult = (
  manifest: RealAcpCorpusRunManifest,
  result: RealAcpTaskRunResult,
  task: RealAcpCorpusTask | undefined,
): RealAcpTaskStabilityRecord => {
  const writeToolCallCount = result.toolCalls.filter((tool) => tool.sideEffectLevel === "write").length;
  const wobble = wobbleSummary(result);
  const protectedPathTouched = result.changedFiles.some((file) =>
    (task?.workspace.protectedPaths ?? []).some((protectedPath) => pathInsideOrEqual(file.path, protectedPath)),
  );
  const mutationAttempted = result.changedFiles.length > 0 || writeToolCallCount > 0 || result.editStrategy.family !== "none";
  const postApplyInconsistent = hasDomainStatus(result.telemetry, "postApplyConsistencyStatus", "inconsistent") ||
    hasDomainStatus(result.telemetry, "selfDetectedRegressionStatus", "confirmed");
  const appliedButBroken = mutationAttempted && (
    result.status === "failed" ||
    result.status === "error" ||
    result.verifier.status === "failed" ||
    postApplyInconsistent ||
    result.repair.status === "failed" ||
    result.rollback.status === "failed"
  );
  return RealAcpTaskStabilityRecordSchema.parse({
    taskId: result.taskId,
    runResultId: result.runResultId,
    runId: manifest.runId,
    split: result.split,
    modelProfileId: manifest.metadata.model.modelProfileId,
    codebaseProfileId: manifest.metadata.codebase.codebaseProfileId,
    clientProfileId: manifest.metadata.client.clientProfileId,
    strategyFamily: result.editStrategy.family,
    status: result.status,
    changedFileCount: result.changedFiles.length,
    writeToolCallCount,
    terminalCommandCount: result.terminalCommands.length,
    verifierStatus: result.verifier.status,
    repairAttempted: result.repair.attempted,
    repairStatus: result.repair.status,
    rollbackAttempted: result.rollback.attempted,
    rollbackStatus: result.rollback.status,
    correctionCount: result.corrections.length,
    fallbackUsed: result.editStrategy.fallbackStrategyId !== undefined,
    ...(codingProgressClassFromTelemetry(result.telemetry) === undefined
      ? {}
      : { codingProgressClass: codingProgressClassFromTelemetry(result.telemetry) }),
    protectedPathTouched,
    postApplyInconsistent,
    appliedButBroken,
    wobble,
  });
};

const aggregateSummary = (records: readonly RealAcpTaskStabilityRecord[]) => ({
  passed: rate(records, (record) => record.status === "passed"),
  failed: rate(records, (record) => record.status === "failed"),
  cancelled: rate(records, (record) => record.status === "cancelled"),
  errored: rate(records, (record) => record.status === "error"),
  appliedButBroken: rate(records, (record) => record.appliedButBroken),
  wobbled: rate(records, (record) => record.wobble.wobbledPathCount > 0),
  protectedPathTouched: rate(records, (record) => record.protectedPathTouched),
  repairAttempted: rate(records, (record) => record.repairAttempted),
  repairFailed: rate(records, (record) => record.repairStatus === "failed"),
  rollbackAttempted: rate(records, (record) => record.rollbackAttempted),
  rollbackFailed: rate(records, (record) => record.rollbackStatus === "failed"),
  fallbackUsed: rate(records, (record) => record.fallbackUsed),
});

const groupSummaries = (records: readonly RealAcpTaskStabilityRecord[]): RealAcpStabilityGroupSummary[] => {
  const groups = new Map<string, RealAcpTaskStabilityRecord[]>();
  for (const record of records) {
    const key = [record.modelProfileId, record.codebaseProfileId, record.clientProfileId, record.strategyFamily].join("\0");
    groups.set(key, [...(groups.get(key) ?? []), record]);
  }
  return [...groups.entries()]
    .map(([key, groupRecords]) => {
      const [modelProfileId, codebaseProfileId, clientProfileId, strategyFamily] = key.split("\0");
      if (modelProfileId == null || codebaseProfileId == null || clientProfileId == null || strategyFamily == null) {
        throw new Error(`invalid stability group key: ${key}`);
      }
      return RealAcpStabilityGroupSummarySchema.parse({
        groupId: `real-acp-stability.group.${safeId(modelProfileId)}.${safeId(codebaseProfileId)}.${safeId(clientProfileId)}.${strategyFamily}`,
        modelProfileId,
        codebaseProfileId,
        clientProfileId,
        strategyFamily,
        taskCount: groupRecords.length,
        passRate: rate(groupRecords, (record) => record.status === "passed").rate,
        failureRate: rate(groupRecords, (record) => record.status === "failed" || record.status === "error").rate,
        appliedButBrokenRate: rate(groupRecords, (record) => record.appliedButBroken).rate,
        wobbleRate: rate(groupRecords, (record) => record.wobble.wobbledPathCount > 0).rate,
        protectedPathTouchRate: rate(groupRecords, (record) => record.protectedPathTouched).rate,
        repairAttemptRate: rate(groupRecords, (record) => record.repairAttempted).rate,
        repairFailureRate: rate(groupRecords, (record) => record.repairStatus === "failed").rate,
        rollbackAttemptRate: rate(groupRecords, (record) => record.rollbackAttempted).rate,
        rollbackFailureRate: rate(groupRecords, (record) => record.rollbackStatus === "failed").rate,
        fallbackRate: rate(groupRecords, (record) => record.fallbackUsed).rate,
      });
    })
    .sort((left, right) => left.groupId.localeCompare(right.groupId));
};

const wobbleSummary = (result: RealAcpTaskRunResult): RealAcpTaskStabilityRecord["wobble"] => {
  const events = [
    ...writeEventsFromTelemetry(result.telemetry),
    ...result.changedFiles.flatMap((file) => file.afterHash === undefined ? [] : [{
      path: file.path,
      digest: file.afterHash,
    }]),
    ...wobbleEntriesFromTerminalCommands(result),
  ];
  const byPath = new Map<string, string[]>();
  for (const event of events) {
    byPath.set(event.path, [...(byPath.get(event.path) ?? []), event.digest]);
  }
  const paths = [...byPath.entries()]
    .map(([path, digests]) => ({
      path,
      versionCount: digests.length,
      distinctVersionCount: new Set(digests).size,
    }))
    .filter((entry) => entry.versionCount >= 2 && entry.distinctVersionCount >= 2)
    .sort((left, right) => left.path.localeCompare(right.path));
  return {
    wobbledPathCount: paths.length,
    writeEventCount: events.length,
    distinctContentVersionCount: [...new Set(events.map((event) => `${event.path}\0${event.digest}`))].length,
    paths,
  };
};

const wobbleEntriesFromTerminalCommands = (result: RealAcpTaskRunResult): Array<{ path: string; digest: string }> => {
  const report = detectAnswerWobble(result.terminalCommands.map((command) => ({
    command: command.command.join(" "),
    output: "",
    exitCode: command.exitCode ?? 0,
  })));
  return report.wobbledFiles.flatMap((entry: WobbleEntry) =>
    entry.versions.map((version) => ({
      path: entry.path,
      digest: version.contentDigest,
    })),
  );
};

const writeEventsFromTelemetry = (value: JsonValue): Array<{ path: string; digest: string }> => {
  const events: Array<{ path: string; digest: string }> = [];
  const visit = (candidate: JsonValue): void => {
    if (Array.isArray(candidate)) {
      candidate.forEach(visit);
      return;
    }
    if (candidate == null || typeof candidate !== "object") {
      return;
    }
    const object = candidate as Record<string, JsonValue>;
    const path = typeof object.path === "string" ? object.path : undefined;
    const digest = digestFromTelemetryObject(object);
    const kind = typeof object.kind === "string" ? object.kind : undefined;
    const sideEffectLevel = typeof object.sideEffectLevel === "string" ? object.sideEffectLevel : undefined;
    if (path !== undefined && digest !== undefined && (kind?.includes("write") || sideEffectLevel === "write")) {
      events.push({ path, digest });
    }
    Object.values(object).forEach(visit);
  };
  visit(value);
  return events;
};

const digestFromTelemetryObject = (object: Record<string, JsonValue>): string | undefined => {
  for (const key of ["contentHash", "contentDigest", "afterHash", "hash", "digest"]) {
    const value = object[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return undefined;
};

const hasDomainStatus = (value: JsonValue, key: string, expected: string): boolean => {
  if (Array.isArray(value)) {
    return value.some((entry) => hasDomainStatus(entry, key, expected));
  }
  if (value == null || typeof value !== "object") {
    return false;
  }
  const object = value as Record<string, JsonValue>;
  return object[key] === expected || Object.values(object).some((entry) => hasDomainStatus(entry, key, expected));
};

const rate = (
  records: readonly RealAcpTaskStabilityRecord[],
  predicate: (record: RealAcpTaskStabilityRecord) => boolean,
): RateSummary => {
  const count = records.filter(predicate).length;
  return RateSummarySchema.parse({
    count,
    rate: records.length === 0 ? 0 : count / records.length,
  });
};

const veto = (input: {
  kind: RealAcpStabilityPromotionVeto["vetoKind"];
  metric: string;
  baselineRate: number;
  candidateRate: number;
  allowedIncrease: number;
}): RealAcpStabilityPromotionVeto => {
  const delta = input.candidateRate - input.baselineRate;
  const passed = delta <= input.allowedIncrease + 1e-9;
  return RealAcpStabilityPromotionVetoSchema.parse({
    vetoId: `real-acp-stability-veto.${input.kind}`,
    vetoKind: input.kind,
    passed,
    blocking: true,
    metric: input.metric,
    baselineRate: input.baselineRate,
    candidateRate: input.candidateRate,
    allowedIncrease: input.allowedIncrease,
    message: passed
      ? `${input.metric} did not regress beyond allowed increase.`
      : `${input.metric} regressed from ${pct(input.baselineRate)} to ${pct(input.candidateRate)}; allowed increase is ${pct(input.allowedIncrease)}.`,
  });
};

const pathInsideOrEqual = (path: string, protectedPath: string): boolean =>
  path === protectedPath || path.startsWith(`${protectedPath.replace(/\/+$/, "")}/`);

const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;

const metricRow = (name: string, summary: RateSummary): string =>
  `| ${name} | ${summary.count} | ${pct(summary.rate)} |`;

const safeId = (value: string): string => value.replace(/[^A-Za-z0-9._:-]+/g, "-");
