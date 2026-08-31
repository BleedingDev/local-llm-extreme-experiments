import { createHash } from "node:crypto";
import { z } from "zod";
import { codingProgressClassFromTelemetry } from "../acp/coding-progress-diagnostics";
import { EvalSplitSchema } from "../eval-harness/types";
import { OptimizerIdSchema } from "../optimizer/types";
import {
  RealAcpCorpusRunManifestSchema,
  type RealAcpCorpusRunManifest,
  type RealAcpTaskRunResult,
} from "./real-acp-runner";
import { createRealAcpStabilityScorecard } from "./real-acp-scorecard";
import { realAcpCodingCorpusTaskPack, type RealAcpCorpusTask } from "./real-acp-task-pack";

const TRACE_SCORECARD_SCHEMA_VERSION = "real-acp-trace-scorecards.v1" as const;

const CountRateSchema = z.object({
  count: z.number().int().nonnegative(),
  rate: z.number().min(0).max(1),
}).strict();

const FailureClassCountSchema = z.object({
  failureClass: OptimizerIdSchema,
  count: z.number().int().nonnegative(),
}).strict();
export type FailureClassCount = z.infer<typeof FailureClassCountSchema>;

export const RealAcpToolCalibrationSummarySchema = z.object({
  summaryId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  namespace: OptimizerIdSchema,
  name: OptimizerIdSchema,
  sideEffectLevels: z.array(z.enum(["none", "read", "write", "network", "process"])).default([]),
  callCount: z.number().int().nonnegative(),
  success: CountRateSchema,
  failed: CountRateSchema,
  skipped: CountRateSchema,
  blocked: CountRateSchema,
  precisionProxy: CountRateSchema,
  recallProxy: CountRateSchema,
  taskFailureAssociation: CountRateSchema,
  failureClasses: z.array(FailureClassCountSchema).default([]),
  totalLatencyMs: z.number().nonnegative(),
  averageLatencyMs: z.number().nonnegative(),
}).strict();
export type RealAcpToolCalibrationSummary = z.infer<typeof RealAcpToolCalibrationSummarySchema>;

const FlagAssociationSchema = z.object({
  flag: z.string().min(1),
  presentCount: z.number().int().nonnegative(),
  absentCount: z.number().int().nonnegative(),
  presentFailureRate: z.number().min(0).max(1),
  absentFailureRate: z.number().min(0).max(1),
  failureRateDeltaWhenPresent: z.number().min(-1).max(1),
}).strict();
export type FlagAssociation = z.infer<typeof FlagAssociationSchema>;

export const RealAcpArgumentPatternSummarySchema = z.object({
  summaryId: OptimizerIdSchema,
  commandName: z.string().min(1),
  argumentShapeHash: z.string().min(1),
  argumentCount: z.number().int().nonnegative(),
  flags: z.array(z.string().min(1)).default([]),
  samplePreview: z.array(z.string().min(1)).default([]),
  count: z.number().int().nonnegative(),
  success: CountRateSchema,
  failed: CountRateSchema,
  flagAssociations: z.array(FlagAssociationSchema).default([]),
}).strict();
export type RealAcpArgumentPatternSummary = z.infer<typeof RealAcpArgumentPatternSummarySchema>;

export const RealAcpToolTransitionSummarySchema = z.object({
  transitionId: OptimizerIdSchema,
  from: z.string().min(1),
  to: z.string().min(1),
  count: z.number().int().nonnegative(),
  passAssociation: CountRateSchema,
  failureAssociation: CountRateSchema,
}).strict();
export type RealAcpToolTransitionSummary = z.infer<typeof RealAcpToolTransitionSummarySchema>;

export const RealAcpEditFamilyMatrixSummarySchema = z.object({
  summaryId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  strategyFamily: z.enum(["whole_file", "diff", "search_replace", "none"]),
  taskShape: OptimizerIdSchema,
  split: EvalSplitSchema,
  taskCount: z.number().int().nonnegative(),
  passRate: z.number().min(0).max(1),
  changedFileRate: z.number().min(0).max(1),
  verifierFailureRate: z.number().min(0).max(1),
  appliedButBrokenRate: z.number().min(0).max(1),
  fallbackRate: z.number().min(0).max(1),
  codingProgressClasses: z.array(FailureClassCountSchema).default([]),
}).strict();
export type RealAcpEditFamilyMatrixSummary = z.infer<typeof RealAcpEditFamilyMatrixSummarySchema>;

export const RealAcpTraceMinedScorecardsSchema = z.object({
  schemaVersion: z.literal(TRACE_SCORECARD_SCHEMA_VERSION),
  scorecardId: OptimizerIdSchema,
  createdAt: z.string().datetime({ offset: true }),
  runIds: z.array(OptimizerIdSchema).min(1),
  taskCount: z.number().int().nonnegative(),
  toolCalibration: z.array(RealAcpToolCalibrationSummarySchema),
  argumentPatterns: z.array(RealAcpArgumentPatternSummarySchema),
  toolTransitions: z.array(RealAcpToolTransitionSummarySchema),
  editFamilyMatrix: z.array(RealAcpEditFamilyMatrixSummarySchema),
  caveats: z.array(z.string().min(1)).default([]),
}).strict();
export type RealAcpTraceMinedScorecards = z.infer<typeof RealAcpTraceMinedScorecardsSchema>;

export type CreateRealAcpTraceMinedScorecardsInput = {
  manifests: readonly RealAcpCorpusRunManifest[];
  scorecardId?: string;
  createdAt?: string;
  taskPack?: { tasks: readonly RealAcpCorpusTask[] };
};

type ToolObservation = {
  namespace: string;
  name: string;
  status: "succeeded" | "failed" | "skipped" | "blocked";
  sideEffectLevel: "none" | "read" | "write" | "network" | "process";
  errorCode?: string;
  latencyMs: number;
  runResultId: string;
  taskPassed: boolean;
  taskFailed: boolean;
  manifest: RealAcpCorpusRunManifest;
};

type CommandObservation = {
  commandName: string;
  args: string[];
  flags: string[];
  status: "succeeded" | "failed" | "skipped" | "timed_out";
};

type TransitionObservation = {
  from: string;
  to: string;
  taskPassed: boolean;
  taskFailed: boolean;
};

export const createRealAcpTraceMinedScorecards = (
  input: CreateRealAcpTraceMinedScorecardsInput,
): RealAcpTraceMinedScorecards => {
  const manifests = input.manifests.map((manifest) => RealAcpCorpusRunManifestSchema.parse(manifest));
  if (manifests.length === 0) {
    throw new Error("at least one real ACP manifest is required");
  }
  const taskById = new Map((input.taskPack?.tasks ?? realAcpCodingCorpusTaskPack.tasks).map((task) => [task.taskId, task]));
  const taskCount = manifests.reduce((count, manifest) => count + manifest.taskResults.length, 0);
  const runIds = [...new Set(manifests.map((manifest) => manifest.runId))].sort((left, right) => left.localeCompare(right));
  const stability = createRealAcpStabilityScorecard({
    manifests,
    scorecardId: `${input.scorecardId ?? `real-acp-trace.${runIds.join("+")}`}.stability`,
    createdAt: input.createdAt ?? new Date().toISOString(),
    ...(input.taskPack === undefined ? {} : { taskPack: input.taskPack }),
  });
  return RealAcpTraceMinedScorecardsSchema.parse({
    schemaVersion: TRACE_SCORECARD_SCHEMA_VERSION,
    scorecardId: input.scorecardId ?? `real-acp-trace.${runIds.join("+")}`,
    createdAt: input.createdAt ?? new Date().toISOString(),
    runIds,
    taskCount,
    toolCalibration: toolCalibration(manifests),
    argumentPatterns: argumentPatterns(manifests),
    toolTransitions: toolTransitions(manifests),
    editFamilyMatrix: editFamilyMatrix(manifests, taskById, stability),
    caveats: [
      "precisionProxy is the share of tool calls associated with passed tasks, not a true precision oracle.",
      "recallProxy is the share of passed tasks in which the tool appeared, not a true required-tool oracle.",
      "argument flag associations are correlations from visible traces and must not be converted into hardcoded language-specific rules.",
    ],
  });
};

export const renderRealAcpTraceMinedScorecardsMarkdown = (
  scorecardsInput: RealAcpTraceMinedScorecards,
): string => {
  const scorecards = RealAcpTraceMinedScorecardsSchema.parse(scorecardsInput);
  return [
    `# Real ACP Trace-Mined Scorecards`,
    ``,
    `Scorecard: \`${scorecards.scorecardId}\``,
    ``,
    `Runs: ${scorecards.runIds.map((runId) => `\`${runId}\``).join(", ")}`,
    ``,
    `## Tool Calibration`,
    ``,
    `| Tool | Calls | Success | Failed | Precision proxy | Recall proxy | Task failure assoc. | Failure classes | Avg latency ms |`,
    `| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |`,
    ...scorecards.toolCalibration.map((summary) =>
      `| ${summary.namespace}/${summary.name} | ${summary.callCount} | ${pct(summary.success.rate)} | ${pct(summary.failed.rate)} | ${pct(summary.precisionProxy.rate)} | ${pct(summary.recallProxy.rate)} | ${pct(summary.taskFailureAssociation.rate)} | ${summary.failureClasses.map((entry) => `${entry.failureClass}:${entry.count}`).join(", ") || "-"} | ${summary.averageLatencyMs.toFixed(1)} |`),
    ``,
    `## Argument Patterns`,
    ``,
    `| Command | Shape | Count | Flags | Success | Failed | Sample |`,
    `| --- | --- | ---: | --- | ---: | ---: | --- |`,
    ...scorecards.argumentPatterns.map((summary) =>
      `| ${summary.commandName} | ${summary.argumentShapeHash.slice(0, 18)} | ${summary.count} | ${summary.flags.join(" ") || "-"} | ${pct(summary.success.rate)} | ${pct(summary.failed.rate)} | ${summary.samplePreview.join(" ")} |`),
    ``,
    `## Tool Transitions`,
    ``,
    `| From | To | Count | Pass assoc. | Failure assoc. |`,
    `| --- | --- | ---: | ---: | ---: |`,
    ...scorecards.toolTransitions.map((summary) =>
      `| ${summary.from} | ${summary.to} | ${summary.count} | ${pct(summary.passAssociation.rate)} | ${pct(summary.failureAssociation.rate)} |`),
    ``,
    `## Edit Family Matrix`,
    ``,
    `| Edit family | Task shape | Split | Tasks | Pass | Changed files | Verifier failed | Applied broken | Fallback | Progress classes |`,
    `| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |`,
    ...scorecards.editFamilyMatrix.map((summary) =>
      `| ${summary.strategyFamily} | ${summary.taskShape} | ${summary.split} | ${summary.taskCount} | ${pct(summary.passRate)} | ${pct(summary.changedFileRate)} | ${pct(summary.verifierFailureRate)} | ${pct(summary.appliedButBrokenRate)} | ${pct(summary.fallbackRate)} | ${summary.codingProgressClasses.map((entry) => `${entry.failureClass}:${entry.count}`).join(", ") || "-"} |`),
    ``,
    `## Caveats`,
    ``,
    ...scorecards.caveats.map((caveat) => `- ${caveat}`),
    ``,
  ].join("\n");
};

const toolCalibration = (manifests: readonly RealAcpCorpusRunManifest[]): RealAcpToolCalibrationSummary[] => {
  const observations = manifests.flatMap((manifest) =>
    manifest.taskResults.flatMap((result) => toolObservations(manifest, result)));
  const passedTaskCount = manifests.reduce(
    (count, manifest) => count + manifest.taskResults.filter((result) => result.status === "passed").length,
    0,
  );
  const grouped = groupBy(observations, (observation) => [
    observation.manifest.metadata.model.modelProfileId,
    observation.manifest.metadata.codebase.codebaseProfileId,
    observation.manifest.metadata.client.clientProfileId,
    observation.manifest.metadata.profile.policyId,
    observation.namespace,
    observation.name,
  ].join("\0"));
  return [...grouped.entries()].map(([key, group]) => {
    const [modelProfileId, codebaseProfileId, clientProfileId, policyId, namespace, name] = key.split("\0");
    if (modelProfileId == null || codebaseProfileId == null || clientProfileId == null || policyId == null || namespace == null || name == null) {
      throw new Error(`invalid tool calibration key: ${key}`);
    }
    const passedTaskIdsWithTool = new Set(group
      .filter((observation) => observation.taskPassed)
      .map((observation) => observation.runResultId));
    const totalLatencyMs = group.reduce((total, observation) => total + observation.latencyMs, 0);
    return RealAcpToolCalibrationSummarySchema.parse({
      summaryId: `real-acp-tool.${safeId(modelProfileId)}.${safeId(codebaseProfileId)}.${safeId(clientProfileId)}.${safeId(namespace)}.${safeId(name)}`,
      modelProfileId,
      codebaseProfileId,
      clientProfileId,
      policyId,
      namespace,
      name,
      sideEffectLevels: [...new Set(group.map((observation) => observation.sideEffectLevel))].sort(),
      callCount: group.length,
      success: countRate(group, (observation) => observation.status === "succeeded"),
      failed: countRate(group, (observation) => observation.status === "failed"),
      skipped: countRate(group, (observation) => observation.status === "skipped"),
      blocked: countRate(group, (observation) => observation.status === "blocked"),
      precisionProxy: countRate(group, (observation) => observation.taskPassed),
      recallProxy: {
        count: passedTaskIdsWithTool.size,
        rate: passedTaskCount === 0 ? 0 : passedTaskIdsWithTool.size / passedTaskCount,
      },
      taskFailureAssociation: countRate(group, (observation) => observation.taskFailed),
      failureClasses: failureClasses(group),
      totalLatencyMs,
      averageLatencyMs: group.length === 0 ? 0 : totalLatencyMs / group.length,
    });
  }).sort((left, right) => left.summaryId.localeCompare(right.summaryId));
};

const argumentPatterns = (manifests: readonly RealAcpCorpusRunManifest[]): RealAcpArgumentPatternSummary[] => {
  const observations = manifests.flatMap((manifest) =>
    manifest.taskResults.flatMap((result) =>
      result.terminalCommands.map((command): CommandObservation => {
        const [commandName = "unknown", ...args] = command.command;
        return {
          commandName,
          args,
          flags: args.filter((arg) => /^-{1,2}[^-\s]/.test(arg)).sort((left, right) => left.localeCompare(right)),
          status: command.status,
        };
      })));
  const byShape = groupBy(observations, (observation) =>
    [observation.commandName, shapeHash(observation.commandName, observation.args, observation.flags)].join("\0"));
  const byCommand = groupBy(observations, (observation) => observation.commandName);
  return [...byShape.entries()].map(([key, group]) => {
    const [commandName, argumentShapeHash] = key.split("\0");
    if (commandName == null || argumentShapeHash == null) {
      throw new Error(`invalid argument pattern key: ${key}`);
    }
    return RealAcpArgumentPatternSummarySchema.parse({
      summaryId: `real-acp-args.${safeId(commandName)}.${argumentShapeHash.slice(7, 19)}`,
      commandName,
      argumentShapeHash,
      argumentCount: group[0]?.args.length ?? 0,
      flags: [...new Set(group.flatMap((observation) => observation.flags))].sort((left, right) => left.localeCompare(right)),
      samplePreview: group[0]?.args.slice(0, 8) ?? [],
      count: group.length,
      success: countRate(group, (observation) => observation.status === "succeeded"),
      failed: countRate(group, (observation) => observation.status === "failed" || observation.status === "timed_out"),
      flagAssociations: flagAssociations(byCommand.get(commandName) ?? []),
    });
  }).sort((left, right) => left.summaryId.localeCompare(right.summaryId));
};

const toolTransitions = (manifests: readonly RealAcpCorpusRunManifest[]): RealAcpToolTransitionSummary[] => {
  const observations = manifests.flatMap((manifest) =>
    manifest.taskResults.flatMap((result) => {
      const sequence = toolSequence(result);
      const pairs: TransitionObservation[] = [];
      for (let index = 0; index < sequence.length - 1; index += 1) {
        const from = sequence[index];
        const to = sequence[index + 1];
        if (from == null || to == null) continue;
        pairs.push({
          from,
          to,
          taskPassed: result.status === "passed",
          taskFailed: result.status === "failed" || result.status === "error",
        });
      }
      return pairs;
    }));
  const grouped = groupBy(observations, (observation) => `${observation.from}\0${observation.to}`);
  return [...grouped.entries()].map(([key, group]) => {
    const [from, to] = key.split("\0");
    if (from == null || to == null) {
      throw new Error(`invalid transition key: ${key}`);
    }
    return RealAcpToolTransitionSummarySchema.parse({
      transitionId: `real-acp-transition.${safeId(from)}.${safeId(to)}`,
      from,
      to,
      count: group.length,
      passAssociation: countRate(group, (observation) => observation.taskPassed),
      failureAssociation: countRate(group, (observation) => observation.taskFailed),
    });
  }).sort((left, right) => left.transitionId.localeCompare(right.transitionId));
};

const editFamilyMatrix = (
  manifests: readonly RealAcpCorpusRunManifest[],
  taskById: ReadonlyMap<string, RealAcpCorpusTask>,
  stability: ReturnType<typeof createRealAcpStabilityScorecard>,
): RealAcpEditFamilyMatrixSummary[] => {
  const stabilityByRunResult = new Map(stability.taskRecords.map((record) => [record.runResultId, record]));
  const records = manifests.flatMap((manifest) =>
    manifest.taskResults.map((result) => ({
      manifest,
      result,
      taskShape: taskById.get(result.taskId)?.primaryLabel ?? "unknown",
      stability: stabilityByRunResult.get(result.runResultId),
    })));
  const grouped = groupBy(records, (record) => [
    record.manifest.metadata.model.modelProfileId,
    record.manifest.metadata.codebase.codebaseProfileId,
    record.manifest.metadata.client.clientProfileId,
    record.manifest.metadata.profile.policyId,
    record.result.editStrategy.family,
    record.taskShape,
    record.result.split,
  ].join("\0"));
  return [...grouped.entries()].map(([key, group]) => {
    const [modelProfileId, codebaseProfileId, clientProfileId, policyId, strategyFamily, taskShape, split] = key.split("\0");
    if (modelProfileId == null || codebaseProfileId == null || clientProfileId == null || policyId == null || strategyFamily == null || taskShape == null || split == null) {
      throw new Error(`invalid edit matrix key: ${key}`);
    }
    return RealAcpEditFamilyMatrixSummarySchema.parse({
      summaryId: `real-acp-edit-matrix.${safeId(modelProfileId)}.${safeId(codebaseProfileId)}.${safeId(clientProfileId)}.${safeId(strategyFamily)}.${safeId(taskShape)}.${split}`,
      modelProfileId,
      codebaseProfileId,
      clientProfileId,
      policyId,
      strategyFamily,
      taskShape,
      split,
      taskCount: group.length,
      passRate: countRate(group, (record) => record.result.status === "passed").rate,
      changedFileRate: countRate(group, (record) => record.result.changedFiles.length > 0).rate,
      verifierFailureRate: countRate(group, (record) => record.result.verifier.status === "failed").rate,
      appliedButBrokenRate: countRate(group, (record) => record.stability?.appliedButBroken ?? false).rate,
      fallbackRate: countRate(group, (record) => record.result.editStrategy.fallbackStrategyId !== undefined).rate,
      codingProgressClasses: codingProgressClasses(group.map((record) => record.result)),
    });
  }).sort((left, right) => left.summaryId.localeCompare(right.summaryId));
};

const codingProgressClasses = (results: readonly RealAcpTaskRunResult[]): FailureClassCount[] => {
  const classes = new Map<string, number>();
  for (const result of results) {
    const progressClass = codingProgressClassFromTelemetry(result.telemetry);
    if (progressClass === undefined) continue;
    classes.set(progressClass, (classes.get(progressClass) ?? 0) + 1);
  }
  return [...classes.entries()]
    .map(([failureClass, count]) => FailureClassCountSchema.parse({ failureClass, count }))
    .sort((left, right) => left.failureClass.localeCompare(right.failureClass));
};

const toolObservations = (
  manifest: RealAcpCorpusRunManifest,
  result: RealAcpTaskRunResult,
): ToolObservation[] => [
  ...result.toolCalls.map((tool) => ({
    namespace: tool.namespace ?? "tool",
    name: tool.name,
    status: tool.status,
    sideEffectLevel: tool.sideEffectLevel,
    ...(tool.errorCode === undefined ? {} : { errorCode: tool.errorCode }),
    latencyMs: 0,
    runResultId: result.runResultId,
    taskPassed: result.status === "passed",
    taskFailed: result.status === "failed" || result.status === "error",
    manifest,
  })),
  ...result.terminalCommands.map((command) => ({
    namespace: "acp.terminal",
    name: command.command[0] ?? "unknown",
    status: command.status === "timed_out" ? "failed" as const : command.status,
    sideEffectLevel: "process" as const,
    ...(command.exitCode === 0 || command.status === "succeeded" ? {} : { errorCode: terminalFailureClass(command) }),
    latencyMs: command.durationMs,
    runResultId: result.runResultId,
    taskPassed: result.status === "passed",
    taskFailed: result.status === "failed" || result.status === "error",
    manifest,
  })),
];

const toolSequence = (result: RealAcpTaskRunResult): string[] => [
  ...result.toolCalls.map((tool) => `${tool.namespace ?? "tool"}/${tool.name}`),
  ...result.terminalCommands.map((command) => `acp.terminal/${command.command[0] ?? "unknown"}`),
];

const failureClasses = (observations: readonly ToolObservation[]): FailureClassCount[] => {
  const classes = new Map<string, number>();
  for (const observation of observations) {
    if (observation.status === "succeeded") continue;
    const failureClass = observation.errorCode ?? observation.status;
    classes.set(failureClass, (classes.get(failureClass) ?? 0) + 1);
  }
  return [...classes.entries()]
    .map(([failureClass, count]) => FailureClassCountSchema.parse({ failureClass, count }))
    .sort((left, right) => left.failureClass.localeCompare(right.failureClass));
};

const flagAssociations = (observations: readonly CommandObservation[]): FlagAssociation[] => {
  const flags = [...new Set(observations.flatMap((observation) => observation.flags))].sort((left, right) => left.localeCompare(right));
  return flags.map((flag) => {
    const present = observations.filter((observation) => observation.flags.includes(flag));
    const absent = observations.filter((observation) => !observation.flags.includes(flag));
    const presentFailureRate = failureRate(present);
    const absentFailureRate = failureRate(absent);
    return FlagAssociationSchema.parse({
      flag,
      presentCount: present.length,
      absentCount: absent.length,
      presentFailureRate,
      absentFailureRate,
      failureRateDeltaWhenPresent: presentFailureRate - absentFailureRate,
    });
  });
};

const terminalFailureClass = (command: RealAcpTaskRunResult["terminalCommands"][number]): string => {
  if (command.status === "timed_out") return "terminal_timed_out";
  if (command.exitCode == null) return "terminal_unknown_exit";
  return `terminal_exit_${command.exitCode}`;
};

const shapeHash = (commandName: string, args: readonly string[], flags: readonly string[]): string =>
  `sha256:${createHash("sha256")
    .update(JSON.stringify({ commandName, argumentCount: args.length, flags: [...flags].sort() }))
    .digest("hex")
    .slice(0, 16)}`;

const countRate = <T>(
  items: readonly T[],
  predicate: (item: T) => boolean,
): { count: number; rate: number } => {
  const count = items.filter(predicate).length;
  return { count, rate: items.length === 0 ? 0 : count / items.length };
};

const failureRate = (observations: readonly CommandObservation[]): number =>
  countRate(observations, (observation) => observation.status === "failed" || observation.status === "timed_out").rate;

const groupBy = <T>(items: readonly T[], keyFor: (item: T) => string): Map<string, T[]> => {
  const grouped = new Map<string, T[]>();
  for (const item of items) {
    const key = keyFor(item);
    grouped.set(key, [...(grouped.get(key) ?? []), item]);
  }
  return grouped;
};

const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;

const safeId = (value: string): string => value.replace(/[^A-Za-z0-9._:-]+/g, "-");
