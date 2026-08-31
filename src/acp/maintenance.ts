import type { AgentSideConnection as AcpConnection, PlanEntry } from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { evalFixtures } from "../eval-harness/fixtures";
import { summarizeEvalSplits } from "../eval-harness/splits";
import { normalizeRunMetrics, readMetricsStore } from "../metrics";
import { optimizePolicy } from "../optimize";
import { loadOptimizerGateSuiteStatus } from "../optimizer/gate-suite";
import { evaluateNoWritePromotionGate } from "../optimizer/no-write-gate";
import { loadOptimizerRegistry, optimizerRegistryCheckpointsDir } from "../optimizer/registry";
import { CandidatePatchSchema } from "../optimizer/types";
import { buildNoWriteReplaySliceFromCorpus, noWriteValidationInputsFromReplaySlice } from "../replay/no-write-slice";
import { TraceStore } from "../trace-store";
import type { BagConfig } from "../types";
import type { BagAcpSession } from "./session";
import { compactJson, maintenanceCommandHelp, markdownContent, updatePlanEntry } from "./surface";

const BACKGROUND_TRIGGER_MIN_RUNS = 2;
const BACKGROUND_TRIGGER_MIN_METRIC_OBSERVATIONS = 12;
const BACKGROUND_TRIGGER_MIN_SPANS = 6;

export type MaintenanceProgressStep = {
  content: string;
  priority: PlanEntry["priority"];
};

export type MaintenanceProgressInput<T> = {
  connection: AcpConnection;
  session: BagAcpSession;
  title: string;
  toolName: string;
  rawInput?: Record<string, unknown>;
  steps: MaintenanceProgressStep[];
  fn: () => Promise<T> | T;
  summarize: (result: T) => Record<string, unknown>;
};

export type BackgroundOptimizationTriggerEvidence = {
  runCount: number;
  stepMetricCount: number;
  llmCallCount: number;
  toolCallCount: number;
  metricObservationCount: number;
  failedStepCount: number;
  failedLlmCallCount: number;
  failedToolCallCount: number;
  traceCount: number;
  spanCount: number;
  errorTraceCount: number;
  errorSpanCount: number;
};

export type BackgroundOptimizationTriggerDiagnostic = {
  triggered: boolean;
  reason: string;
  evidence: BackgroundOptimizationTriggerEvidence;
  recommendedCommand: string;
  source: string;
  sideEffects: string[];
  sourceRunId?: string;
  opportunityPath?: string;
};

export type BackgroundOptimizationTriggerInput = {
  source: string;
  sourceRunId?: string;
  enqueue?: boolean;
};

export type MaintenanceCommandDeps = {
  connection: AcpConnection;
  config: BagConfig;
  agentMessage: (sessionId: string, text: string) => Promise<void>;
};

export const handleMaintenanceCommand = async (
  deps: MaintenanceCommandDeps,
  session: BagAcpSession,
  task: string,
): Promise<void> => {
  const [rawSubcommand = "", ...rest] = task.trim().split(/\s+/);
  const subcommand = rawSubcommand.toLowerCase();
  const args = rest.filter((arg) => arg.length > 0);

  if (subcommand === "" || subcommand === "help") {
    await deps.agentMessage(session.id, maintenanceCommandHelp());
    return;
  }

  if (subcommand === "status" || subcommand === "inspect") {
    const status = await runMaintenanceProgress({
      connection: deps.connection,
      session,
      title: "Inspect maintenance status",
      toolName: "bag.maintenance.status",
      rawInput: { subcommand },
      steps: [
        { content: "Inspect optimizer registry and session pin", priority: "high" },
        { content: "Inspect runtime optimizer gate suite", priority: "high" },
        { content: "Evaluate no-write replay validation gate", priority: "high" },
        { content: "Summarize background optimization evidence", priority: "medium" },
      ],
      fn: () => renderMaintenanceStatusWithRuntimeGates(deps.config, session),
      summarize: (text) => summarizeMaintenanceText(text),
    });
    await deps.agentMessage(session.id, status);
    return;
  }

  if (subcommand === "eval" || subcommand === "evals") {
    const summary = await runMaintenanceProgress({
      connection: deps.connection,
      session,
      title: "Inspect maintenance eval splits",
      toolName: "bag.maintenance.eval_summary",
      rawInput: { subcommand },
      steps: [
        { content: "Read configured eval split metadata", priority: "high" },
        { content: "Summarize train/dev/holdout visibility", priority: "medium" },
      ],
      fn: () => renderMaintenanceEvalSummary(),
      summarize: (text) => summarizeMaintenanceText(text),
    });
    await deps.agentMessage(session.id, summary);
    return;
  }

  if (subcommand === "optimize") {
    const report = await runMaintenanceProgress({
      connection: deps.connection,
      session,
      title: "Compute maintenance optimize report",
      toolName: "bag.maintenance.optimize",
      rawInput: { subcommand, sideEffects: "none" },
      steps: [
        { content: "Read existing run metrics and manifests", priority: "high" },
        { content: "Compute bounded optimization recommendation", priority: "high" },
        { content: "Render compact maintenance summary", priority: "medium" },
      ],
      fn: () => optimizePolicy({ config: deps.config, cwd: session.cwd }),
      summarize: (optimizationReport) => ({
        evaluatedRuns: optimizationReport.evaluatedRuns,
        evaluatedMetrics: optimizationReport.evaluatedMetrics,
        passRate: optimizationReport.passRate,
        recommendedExecutorConcurrency: optimizationReport.recommendedExecutorConcurrency,
        recommendedInteractiveConcurrency: optimizationReport.recommendedInteractiveConcurrency,
      }),
    });
    await deps.agentMessage(
      session.id,
      [
        "Maintenance optimize report:",
        "- safe existing hook: optimizePolicy",
        "- side effects: none; computed from existing run metrics and manifests",
        `- evaluated runs: ${report.evaluatedRuns}`,
        `- evaluated metrics: ${report.evaluatedMetrics}`,
        `- pass rate: ${report.passRate}`,
        `- average step duration ms: ${report.averageStepDurationMs}`,
        `- recommended executor concurrency: ${report.recommendedExecutorConcurrency}`,
        `- recommended interactive concurrency: ${report.recommendedInteractiveConcurrency}`,
        "Notes:",
        ...report.notes.map((note) => `- ${note}`),
      ].join("\n"),
    );
    return;
  }

  if (subcommand === "promote") {
    await deps.agentMessage(session.id, renderMaintenancePromoteSummary(deps.config, session, args[0]));
    return;
  }

  if (subcommand === "rollback") {
    await deps.agentMessage(session.id, renderMaintenanceRollbackSummary(deps.config, session, args[0]));
    return;
  }

  await deps.agentMessage(session.id, `Unknown maintenance command: ${subcommand}\n\n${maintenanceCommandHelp()}`);
};

export const renderMaintenanceStatus = (config: BagConfig, session: BagAcpSession): string => {
  const registry = loadOptimizerRegistry(config, session.cwd);
  const gateSuiteStatus = loadOptimizerGateSuiteStatus({ cwd: session.cwd });
  const backgroundTrigger = inspectBackgroundOptimizationTrigger(config, session, {
    source: "maintenance-status",
    enqueue: false,
  });
  const recordCounts = registry.records.reduce<Record<string, number>>((counts, record) => {
    counts[record.recordKind] = (counts[record.recordKind] ?? 0) + 1;
    return counts;
  }, {});
  const persistedCounts = registry.persistedRecords.reduce<Record<string, number>>((counts, record) => {
    counts[record.recordKind] = (counts[record.recordKind] ?? 0) + 1;
    return counts;
  }, {});

  return [
    "Maintenance optimizer status:",
    `- registry root: ${registry.root}`,
    `- records: ${registry.records.length} total, ${registry.seedRecords.length} seed, ${registry.persistedRecords.length} persisted`,
    `- record kinds: ${formatCounts(recordCounts)}`,
    `- persisted kinds: ${formatCounts(persistedCounts)}`,
    `- registry errors: ${registry.errors.length}`,
    `- invalid records: ${registry.invalidRecords.length}`,
    `- active pointer: ${registry.activePointer == null ? "none" : compactJson(registry.activePointer)}`,
    "",
    "Runtime optimizer gate suite:",
    `- state: ${gateSuiteStatus.state}`,
    `- suite loaded: ${gateSuiteStatus.suiteLoaded}`,
    `- promotion allowed: ${gateSuiteStatus.promotionAllowed}`,
    `- auto-promotion allowed: ${gateSuiteStatus.autoPromotionAllowed}`,
    `- candidate generation: ${gateSuiteStatus.candidateGeneration}`,
    `- suite path: ${gateSuiteStatus.suitePath}`,
    `- blocking reasons: ${gateSuiteStatus.blockingReasons.join("; ") || "none"}`,
    `- fail-closed triggers: ${gateSuiteStatus.mustFailClosedOn.join("; ") || "none"}`,
    `- gate errors: ${gateSuiteStatus.errors.map((error) => `${error.kind}: ${error.message}`).join("; ") || "none"}`,
    "",
    "Current session pin:",
    `- model role: ${session.optimizerPin.telemetry.modelRole}`,
    `- provider config role: ${session.optimizerPin.telemetry.providerConfigRole ?? "unknown"}`,
    `- provider: ${session.optimizerPin.telemetry.provider ?? "unknown"}`,
    `- endpoint kind: ${session.optimizerPin.telemetry.endpointKind ?? "unknown"}`,
    `- model server: ${session.optimizerPin.telemetry.modelServerId ?? "unknown"}`,
    `- model server profile: ${session.optimizerPin.telemetry.modelServerProfileId ?? "unknown"}`,
    `- provider discovery: ${session.optimizerPin.telemetry.providerDiscoverySource ?? "unknown"}`,
    `- context window tokens: ${session.optimizerPin.telemetry.contextWindowTokens ?? "unknown"}`,
    `- max output tokens: ${session.optimizerPin.telemetry.maxOutputTokens ?? "unknown"}`,
    `- model profile: ${session.optimizerPin.telemetry.modelProfileId}`,
    `- codebase profile: ${session.optimizerPin.telemetry.codebaseProfileId}`,
    `- policy: ${session.optimizerPin.telemetry.policyId}`,
    `- canonical tool version: ${session.optimizerPin.telemetry.canonicalToolVersion}`,
    `- rendered tool version: ${session.optimizerPin.telemetry.renderedToolVersion}`,
    `- result style version: ${session.optimizerPin.telemetry.resultStyleVersion}`,
    `- verification policy version: ${session.optimizerPin.telemetry.verificationPolicyVersion}`,
    `- edit strategy version: ${session.optimizerPin.telemetry.editStrategyVersion}`,
    `- rendered edit contract version: ${session.optimizerPin.telemetry.renderedEditContractVersion}`,
    `- edit fallback policy version: ${session.optimizerPin.telemetry.editFallbackPolicyVersion}`,
    `- edit repair policy version: ${session.optimizerPin.telemetry.editRepairPolicyVersion}`,
    `- edit verifier policy version: ${session.optimizerPin.telemetry.editVerifierPolicyVersion}`,
    `- edit objective set: ${session.optimizerPin.telemetry.editObjectiveSetId}`,
    `- applies to current session: pinned; newly promoted policies apply to new sessions only`,
    "",
    "Background optimization trigger:",
    `- triggered now: ${backgroundTrigger.triggered}`,
    `- reason: ${backgroundTrigger.reason}`,
    `- evidence: runs=${backgroundTrigger.evidence.runCount} metrics=${backgroundTrigger.evidence.metricObservationCount} spans=${backgroundTrigger.evidence.spanCount} errors=${backgroundTrigger.evidence.failedStepCount + backgroundTrigger.evidence.failedLlmCallCount + backgroundTrigger.evidence.failedToolCallCount + backgroundTrigger.evidence.errorSpanCount}`,
    `- recommended command: ${backgroundTrigger.recommendedCommand}`,
    "- side effects: none during status inspection",
  ].join("\n");
};

export const renderMaintenanceStatusWithRuntimeGates = async (
  config: BagConfig,
  session: BagAcpSession,
): Promise<string> => {
  const base = renderMaintenanceStatus(config, session);
  const slice = await buildNoWriteReplaySliceFromCorpus({
    corpusRoot: resolve(session.cwd, ".bag", "replay-corpus"),
  });
  const gate = evaluateNoWritePromotionGate({
    cases: noWriteValidationInputsFromReplaySlice(slice),
    requireEvidence: true,
  });
  return [
    base,
    "",
    "No-write ACP validation gate:",
    `- status: ${gate.status}`,
    `- included replay cases: ${slice.status.includedCases}`,
    `- records seen: ${slice.status.totalRecordsSeen}`,
    `- skipped hidden holdout: ${slice.status.skippedHiddenHoldout}`,
    `- skipped unsafe/excluded: ${slice.status.skippedUnsafeOrExcluded}`,
    `- skipped duplicate: ${slice.status.skippedDuplicate}`,
    `- checked records: ${gate.checkedRecordIds.length}`,
    `- blocked records: ${gate.blockedRecordIds.length}`,
    `- warning records: ${gate.warnedRecordIds.length}`,
    `- reasons: ${gate.reasons.join("; ") || "none"}`,
    `- evidence refs: ${gate.evidenceRefs.length}`,
  ].join("\n");
};

export const renderMaintenanceEvalSummary = (): string => {
  const summaries = summarizeEvalSplits(evalFixtures);
  const total = summaries.reduce((sum, summary) => sum + summary.count, 0);
  return [
    "Maintenance eval summary:",
    "- side effects: none; this is a configured eval-suite summary only",
    `- total eval cases: ${total}`,
    ...summaries.map((summary) =>
      [
        `- ${summary.split}: ${summary.count}`,
        `  evals: ${summary.evalCaseIds.join(", ") || "none"}`,
        `  fixtures: ${summary.fixtureWorkspaceIds.join(", ") || "none"}`,
        `  tags: ${summary.tags.join(", ") || "none"}`,
      ].join("\n"),
    ),
    "- holdout usage: hidden from candidate training and reserved for promotion gates",
  ].join("\n");
};

export const renderMaintenancePromoteSummary = (
  config: BagConfig,
  session: BagAcpSession,
  candidateId: string | undefined,
): string => {
  const registry = loadOptimizerRegistry(config, session.cwd);
  const candidates = registry.persistedRecords.filter((record) => record.recordKind === "candidate_patch");
  const promotionDecisions = registry.persistedRecords.filter((record) => record.recordKind === "promotion_decision");
  const matchingCandidate =
    candidateId == null
      ? undefined
      : candidates.find((record) => {
          const payload = record.payload as { candidatePatchId?: unknown };
          return payload.candidatePatchId === candidateId || record.registryRecordId === candidateId;
        });
  const parsedCandidate = matchingCandidate == null ? undefined : CandidatePatchSchema.safeParse(matchingCandidate.payload);
  const matchingDecisions = parsedCandidate?.success === true
    ? promotionDecisions
        .filter((record) => {
          const payload = record.payload as { candidatePatchId?: unknown; decidedAt?: unknown };
          return payload.candidatePatchId === parsedCandidate.data.candidatePatchId;
        })
        .sort((left, right) => {
          const leftAt = String((left.payload as { decidedAt?: unknown }).decidedAt ?? left.updatedAt);
          const rightAt = String((right.payload as { decidedAt?: unknown }).decidedAt ?? right.updatedAt);
          return leftAt.localeCompare(rightAt);
        })
    : [];
  const readinessBlockers = [
    candidateId == null ? "candidate id required" : undefined,
    candidateId != null && matchingCandidate == null ? "candidate record not found" : undefined,
    matchingCandidate != null && parsedCandidate?.success !== true ? "candidate payload failed CandidatePatchSchema" : undefined,
    parsedCandidate?.success === true && matchingDecisions.length === 0
      ? "no promotion decision or eval lineage recorded for this candidate"
      : undefined,
  ].filter((blocker): blocker is string => blocker != null);
  const lastDecision = matchingDecisions.at(-1)?.payload as
    | { decision?: unknown; reason?: unknown; evalResultId?: unknown; appliesToNewSessionsOnly?: unknown }
    | undefined;
  const dryRunStatus = readinessBlockers.length === 0 ? "ready_for_operator_review" : "blocked";

  return [
    "Maintenance promote inspection:",
    "- side effects: none; this is a dry-run readiness inspection only",
    "- promotion requires a validated candidate patch, candidate eval scorecard, and critical regression veto pass",
    "- promoted policies apply to new ACP sessions only; current sessions stay pinned",
    `- dry-run status: ${dryRunStatus}`,
    `- readiness blockers: ${readinessBlockers.join("; ") || "none"}`,
    `- candidate requested: ${candidateId ?? "none"}`,
    `- persisted candidates: ${candidates.length}`,
    `- candidate found: ${matchingCandidate == null ? "no" : "yes"}`,
    ...(matchingCandidate == null
      ? []
      : [
          `- registry record: ${matchingCandidate.registryRecordId}`,
          `- status: ${matchingCandidate.status}`,
          `- candidate schema valid: ${parsedCandidate?.success === true ? "yes" : "no"}`,
          ...(parsedCandidate?.success === true
            ? [
                `- payload id: ${parsedCandidate.data.candidatePatchId}`,
                `- scope: ${parsedCandidate.data.scope.artifactKind} ${parsedCandidate.data.scope.artifactId}`,
                `- operation count: ${parsedCandidate.data.operations.length}`,
                `- source traces: ${parsedCandidate.data.sourceTraceIds.length}`,
              ]
            : []),
          `- promotion decisions: ${matchingDecisions.length}`,
          ...(lastDecision == null
            ? []
            : [
                `- last decision: ${String(lastDecision.decision ?? "unknown")}`,
                `- last eval result: ${String(lastDecision.evalResultId ?? "none")}`,
                `- new sessions only: ${String(lastDecision.appliesToNewSessionsOnly ?? "unknown")}`,
                `- decision reason: ${String(lastDecision.reason ?? "none")}`,
              ]),
        ]),
  ].join("\n");
};

export const renderMaintenanceRollbackSummary = (
  config: BagConfig,
  session: BagAcpSession,
  checkpointArg: string | undefined,
): string => {
  const checkpointsDir = optimizerRegistryCheckpointsDir(config, session.cwd);
  const checkpoints = existsSync(checkpointsDir)
    ? readdirSync(checkpointsDir)
        .filter((file) => file.endsWith(".json"))
        .sort((left, right) => left.localeCompare(right))
    : [];
  const selected = checkpointArg == null ? checkpoints.at(-1) : checkpoints.includes(checkpointArg) ? checkpointArg : undefined;
  const selectedPath = selected == null ? undefined : join(checkpointsDir, selected);
  const checkpoint = selectedPath == null ? undefined : readRollbackCheckpointSummary(selectedPath);
  const readiness =
    selected == null
      ? "blocked_no_checkpoint"
      : checkpoint?.readable !== true
        ? "blocked_unreadable_checkpoint"
        : checkpoint.previousPointerAvailable
          ? "ready_for_operator_review"
          : "blocked_no_previous_pointer";

  return [
    "Maintenance rollback inspection:",
    "- side effects: none; ACP rollback is intentionally read-only in this lane",
    "- rollback execution must restore a checkpointed active pointer atomically and keep session pins unchanged",
    `- rollback readiness: ${readiness}`,
    `- checkpoints dir: ${checkpointsDir}`,
    `- checkpoints: ${checkpoints.length}`,
    `- requested checkpoint: ${checkpointArg ?? "latest"}`,
    `- selected checkpoint: ${selected ?? "none"}`,
    `- selected exists: ${selected == null ? "no" : "yes"}`,
    `- previous pointer available: ${checkpoint?.previousPointerAvailable === true ? "yes" : "no"}`,
    ...(checkpoint?.candidatePatchId == null ? [] : [`- checkpoint candidate: ${checkpoint.candidatePatchId}`]),
    ...(checkpoint?.createdAt == null ? [] : [`- checkpoint created: ${checkpoint.createdAt}`]),
    ...(checkpoint?.error == null ? [] : [`- checkpoint error: ${checkpoint.error}`]),
    ...(checkpoints.length === 0 ? [] : [`- recent checkpoints: ${checkpoints.slice(-5).join(", ")}`]),
  ].join("\n");
};

export const readRollbackCheckpointSummary = (
  path: string,
): {
  readable: boolean;
  candidatePatchId?: string;
  createdAt?: string;
  previousPointerAvailable: boolean;
  error?: string;
} => {
  try {
    const raw = JSON.parse(readFileSync(path, "utf8")) as {
      candidatePatchId?: unknown;
      createdAt?: unknown;
      previousPointer?: unknown;
    };
    return {
      readable: true,
      previousPointerAvailable: raw.previousPointer != null,
      ...(typeof raw.candidatePatchId === "string" ? { candidatePatchId: raw.candidatePatchId } : {}),
      ...(typeof raw.createdAt === "string" ? { createdAt: raw.createdAt } : {}),
    };
  } catch (error) {
    return {
      readable: false,
      previousPointerAvailable: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
};

export const inspectBackgroundOptimizationTrigger = (
  config: BagConfig,
  session: BagAcpSession,
  input: BackgroundOptimizationTriggerInput,
): BackgroundOptimizationTriggerDiagnostic => {
  const source = input.source;
  const metrics = readBackgroundTriggerMetrics(config, session.cwd);
  const traceOverview = TraceStore.open(config, session.cwd).getOverview();
  const evidence: BackgroundOptimizationTriggerEvidence = {
    runCount: metrics.runs.length,
    stepMetricCount: metrics.steps.length,
    llmCallCount: metrics.llmCalls.length,
    toolCallCount: metrics.toolCalls.length,
    metricObservationCount: metrics.steps.length + metrics.llmCalls.length + metrics.toolCalls.length,
    failedStepCount: metrics.steps.filter((metric) => !metric.ok).length,
    failedLlmCallCount: metrics.llmCalls.filter((metric) => !metric.ok).length,
    failedToolCallCount: metrics.toolCalls.filter((metric) => !metric.ok).length,
    traceCount: traceOverview.traceCount,
    spanCount: traceOverview.spanCount,
    errorTraceCount: traceOverview.errorTraceCount,
    errorSpanCount: traceOverview.errorSpanCount,
  };
  const recommendedCommand = "/maintenance optimize";
  const enoughEvidence =
    evidence.runCount >= BACKGROUND_TRIGGER_MIN_RUNS &&
    evidence.metricObservationCount >= BACKGROUND_TRIGGER_MIN_METRIC_OBSERVATIONS &&
    evidence.spanCount >= BACKGROUND_TRIGGER_MIN_SPANS;
  const errorSignals =
    evidence.failedStepCount +
    evidence.failedLlmCallCount +
    evidence.failedToolCallCount +
    evidence.errorSpanCount;

  if (!enoughEvidence) {
    return backgroundTriggerDiagnostic({
      source,
      sourceRunId: input.sourceRunId,
      triggered: false,
      reason: `insufficient evidence; need at least ${BACKGROUND_TRIGGER_MIN_RUNS} runs, ${BACKGROUND_TRIGGER_MIN_METRIC_OBSERVATIONS} metric observations, and ${BACKGROUND_TRIGGER_MIN_SPANS} spans`,
      evidence,
      recommendedCommand,
      sideEffects: [],
    });
  }

  if (errorSignals === 0) {
    return backgroundTriggerDiagnostic({
      source,
      sourceRunId: input.sourceRunId,
      triggered: false,
      reason: "enough evidence exists, but no failure or trace-error signal needs maintenance yet",
      evidence,
      recommendedCommand,
      sideEffects: [],
    });
  }

  const base = backgroundTriggerDiagnostic({
    source,
    sourceRunId: input.sourceRunId,
    triggered: true,
    reason: "enough real metrics and traces exist with failure signals; queue a maintenance optimization opportunity only",
    evidence,
    recommendedCommand,
    sideEffects: input.enqueue === true ? ["append-maintenance-opportunity"] : [],
  });

  if (input.enqueue !== true) {
    return base;
  }

  const opportunityPath = resolve(session.cwd, config.artifactDir, "maintenance", "opportunities.jsonl");
  mkdirSync(dirname(opportunityPath), { recursive: true });
  appendFileSync(
    opportunityPath,
    `${JSON.stringify({
      createdAt: new Date().toISOString(),
      kind: "background-optimization-opportunity",
      sessionId: session.id,
      source,
      sourceRunId: input.sourceRunId,
      reason: base.reason,
      evidence,
      recommendedCommand,
      sideEffects: ["no automatic promotion", "no policy mutation", "no eval fabrication"],
    })}\n`,
  );
  return backgroundTriggerDiagnostic({
    ...base,
    opportunityPath,
  });
};

export const backgroundTriggerDiagnostic = (
  input: Omit<BackgroundOptimizationTriggerDiagnostic, "sourceRunId" | "opportunityPath"> & {
    sourceRunId?: string | undefined;
    opportunityPath?: string | undefined;
  },
): BackgroundOptimizationTriggerDiagnostic => ({
  triggered: input.triggered,
  reason: input.reason,
  evidence: input.evidence,
  recommendedCommand: input.recommendedCommand,
  source: input.source,
  sideEffects: input.sideEffects,
  ...(input.sourceRunId == null ? {} : { sourceRunId: input.sourceRunId }),
  ...(input.opportunityPath == null ? {} : { opportunityPath: input.opportunityPath }),
});

export const readBackgroundTriggerMetrics = (
  config: BagConfig,
  cwd: string,
): {
  runs: Array<{ runId: string }>;
  steps: Array<{ ok: boolean }>;
  llmCalls: Array<{ ok: boolean }>;
  toolCalls: Array<{ ok: boolean }>;
} => {
  try {
    const runs = Object.entries(readMetricsStore(config, cwd)).map(([runId, entry]) => ({
      runId,
      ...normalizeRunMetrics(entry),
    }));
    return {
      runs,
      steps: runs.flatMap((run) => run.steps),
      llmCalls: runs.flatMap((run) => run.llmCalls),
      toolCalls: runs.flatMap((run) => run.toolCalls),
    };
  } catch {
    return { runs: [], steps: [], llmCalls: [], toolCalls: [] };
  }
};

export const formatCounts = (counts: Record<string, number>): string => {
  const entries = Object.entries(counts).sort(([left], [right]) => left.localeCompare(right));
  return entries.length === 0 ? "none" : entries.map(([key, value]) => `${key}=${value}`).join(", ");
};

export const runMaintenanceProgress = async <T>(input: MaintenanceProgressInput<T>): Promise<T> => {
  const toolCallId = `tool-${randomUUID()}`;
  const rawInput = { toolName: input.toolName, ...(input.rawInput ?? {}) };
  let plan = input.steps.map<PlanEntry>((step) => ({
    content: step.content,
    priority: step.priority,
    status: "pending",
  }));
  const publishPlan = async () => {
    await input.connection.sessionUpdate({
      sessionId: input.session.id,
      update: {
        sessionUpdate: "plan",
        entries: plan,
      },
    });
  };
  const setPlan = async (index: number, status: PlanEntry["status"]) => {
    plan = updatePlanEntry(plan, index, status);
    await publishPlan();
  };

  await publishPlan();
  await input.connection.sessionUpdate({
    sessionId: input.session.id,
    update: {
      sessionUpdate: "tool_call",
      toolCallId,
      title: input.title,
      kind: "think",
      status: "pending",
      rawInput,
    },
  });

  try {
    if (plan.length > 0) {
      await setPlan(0, "in_progress");
    }
    await input.connection.sessionUpdate({
      sessionId: input.session.id,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "in_progress",
        content: [markdownContent(`${input.title} is running as a maintenance-scoped inspection.`)],
      },
    });
    const result = await input.fn();
    for (const index of plan.keys()) {
      await setPlan(index, "completed");
    }
    const rawOutput = { toolName: input.toolName, ...input.summarize(result) };
    await input.connection.sessionUpdate({
      sessionId: input.session.id,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "completed",
        rawOutput,
        content: [markdownContent(`Completed ${input.title}.\n\n\`\`\`json\n${compactJson(rawOutput)}\n\`\`\``)],
      },
    });
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await input.connection.sessionUpdate({
      sessionId: input.session.id,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "failed",
        rawOutput: { error: message },
        content: [markdownContent(`Failed ${input.title}: ${message}`)],
      },
    });
    throw error;
  }
};

export const summarizeMaintenanceText = (text: string): Record<string, unknown> => ({
  lineCount: text.split("\n").filter((line) => line.trim().length > 0).length,
  preview: text.split("\n").slice(0, 4).join("\n"),
});
