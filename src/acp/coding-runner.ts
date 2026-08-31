import type {
  AgentSideConnection as AcpConnection,
  PlanEntry,
} from "@agentclientprotocol/sdk";
import { createArtifactWriter, createRunId, writeManifest } from "../artifacts";
import type { EditAttemptContract } from "../edit-strategy/types";
import { createLlmRouter, type LlmRouter } from "../llm";
import { optimizePolicy } from "../optimize";
import type { EditStrategyFallbackRule } from "../optimizer/edit-policy-router";
import type { AcpReplayCapture } from "../replay";
import { deterministicSelfEvaluation, RunTelemetry } from "../telemetry";
import type { BagConfig, ContextScoutFinding, ToolCallMetric } from "../types";
import { buildRepoContext, detectProjectKind, loadKnowledge, runLocalContextScouts } from "../workspace";
import type {
  CodingCommand,
  CodingEditOperation,
  CodingEditResult,
  CodingFileSelection,
  CodingFileSnapshot,
  CodingPatch,
  LiveEditContext,
  PostApplyConsistencyCheck,
} from "./coding-types";
import type {
  BackgroundOptimizationTriggerDiagnostic,
  BackgroundOptimizationTriggerInput,
} from "./maintenance";
import type { BagAcpSession } from "./session";
import { compactJson, sha256, traceEvent, updatePlanEntry, type TraceEvent } from "./surface";
import type { TerminalCommandResult } from "./terminal";
import type { AcpToolInput } from "./tool-runner";
import {
  classifyCodingProgress,
  type CodingProgressDiagnostic,
} from "./coding-progress-diagnostics";

export type AcpCodingRunnerDeps = {
  connection: AcpConnection;
  config: BagConfig;
  agentMessage: (sessionId: string, text: string) => Promise<void>;
  runAcpTool: <T>(input: AcpToolInput) => Promise<T>;
  configForSession: (session: BagAcpSession) => BagConfig;
  throwIfAborted: (signal?: AbortSignal) => void;
  isAbortError: (error: unknown, signal?: AbortSignal) => boolean;
  absoluteSessionPath: (session: BagAcpSession, path: string) => string;
  sessionRelativePath: (session: BagAcpSession, path: string) => string;
  readClientFile: (input: {
    sessionId: string;
    telemetry: RunTelemetry;
    path: string;
    signal?: AbortSignal;
  }) => Promise<string>;
  selectCodingFiles: (input: {
    router: LlmRouter;
    task: string;
    repoContext: string;
    knowledge: string;
    scoutFindings: ContextScoutFinding[];
  }) => Promise<CodingFileSelection>;
  resolveLiveEditContext: (
    session: BagAcpSession,
    fileSnapshots: CodingFileSnapshot[],
  ) => LiveEditContext;
  serializeLiveEditContext: (context: LiveEditContext) => Record<string, unknown>;
  generateCodingPatch: (input: {
    router: LlmRouter;
    task: string;
    repoContext: string;
    knowledge: string;
    fileSnapshots: CodingFileSnapshot[];
    editContext: LiveEditContext;
    verifierResults?: readonly TerminalCommandResult[];
    postApplyFailures?: readonly { path: string; status: string; reason?: string }[];
    repairRound?: number;
  }) => Promise<CodingPatch>;
  recordPatchParseFailures: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editContext: LiveEditContext;
    patch: CodingPatch;
  }) => void;
  previewAndWriteClientEdit: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    fileSnapshots: CodingFileSnapshot[];
    edit: CodingEditOperation;
    signal?: AbortSignal;
  }) => Promise<CodingEditResult[]>;
  updateFileSnapshotsFromEditResult: (
    session: BagAcpSession,
    fileSnapshots: CodingFileSnapshot[],
    result: CodingEditResult,
  ) => void;
  fallbackTriggerForPatch: (
    patch: CodingPatch,
    results: readonly CodingEditResult[],
  ) => EditStrategyFallbackRule["trigger"] | undefined;
  fallbackLiveEditContext: (
    session: BagAcpSession,
    current: LiveEditContext,
    trigger: EditStrategyFallbackRule["trigger"],
  ) => LiveEditContext | undefined;
  checkPostApplyConsistency: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editResults: readonly CodingEditResult[];
  }) => Promise<PostApplyConsistencyCheck[]>;
  hasPostApplyInconsistency: (checks: readonly PostApplyConsistencyCheck[]) => boolean;
  verificationCommands: (commands: CodingCommand[], cwd: string) => CodingCommand[];
  runTerminalCommand: (input: {
    sessionId: string;
    telemetry: RunTelemetry;
    command: string;
    args: string[];
    reason: string;
    cwd: string;
    signal?: AbortSignal;
  }) => Promise<TerminalCommandResult>;
  rollbackLiveEdits: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    baselineFileSnapshots: readonly CodingFileSnapshot[];
    currentFileSnapshots: readonly CodingFileSnapshot[];
    editResults: readonly CodingEditResult[];
  }) => Promise<CodingEditResult[]>;
  recordFinalEditLifecycleTelemetry: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editResults: readonly CodingEditResult[];
    postApplyChecks: readonly PostApplyConsistencyCheck[];
    commandResults: readonly TerminalCommandResult[];
    rollbackResults: readonly CodingEditResult[];
    artifactRefs: readonly string[];
  }) => EditAttemptContract[];
  buildCodingReplayCapture: (input: {
    session: BagAcpSession;
    runId: string;
    task: string;
    tracePath: string;
    fileSnapshots: readonly CodingFileSnapshot[];
    editAttempts: readonly EditAttemptContract[];
    toolMetrics: readonly ToolCallMetric[];
    commandResults: readonly TerminalCommandResult[];
    artifactRefs: readonly string[];
    codingProgressDiagnostic?: CodingProgressDiagnostic;
  }) => AcpReplayCapture;
  inspectBackgroundOptimizationTrigger: (
    session: BagAcpSession,
    input: BackgroundOptimizationTriggerInput,
  ) => BackgroundOptimizationTriggerDiagnostic;
};

export type AcpCodingTurnInput = {
  session: BagAcpSession;
  task: string;
  signal: AbortSignal;
};

export const runAcpCodingTurn = async (
  deps: AcpCodingRunnerDeps,
  input: AcpCodingTurnInput,
): Promise<void> => {
  const { session, task, signal } = input;
  const runId = `acp-code-${createRunId()}`;
  const writer = createArtifactWriter(deps.config, runId, session.cwd);
  const telemetry = new RunTelemetry(deps.config, runId, session.cwd, session.optimizerPin.telemetry);
  const router = createLlmRouter(deps.config, telemetry);
  const trace: TraceEvent[] = [];
  const plan: PlanEntry[] = [
    { content: "Build repository context and select files", priority: "high", status: "pending" },
    { content: "Read selected files through ACP filesystem", priority: "high", status: "pending" },
    { content: "Generate concrete file edits", priority: "high", status: "pending" },
    { content: "Preview edit strategy, request permission, and write edited files", priority: "high", status: "pending" },
    { content: "Run verification commands in ACP terminal", priority: "high", status: "pending" },
    { content: "Persist traces, metrics, and self-evaluate", priority: "medium", status: "pending" },
  ];
  let currentPlan = plan;

  const pushTrace = (entry: TraceEvent) => {
    trace.push(entry);
    telemetry.event("acp.coding.trace", entry as unknown as Record<string, unknown>);
  };
  const setPlan = async (index: number, status: PlanEntry["status"]) => {
    currentPlan = updatePlanEntry(currentPlan, index, status);
    await deps.connection.sessionUpdate({
      sessionId: session.id,
      update: {
        sessionUpdate: "plan",
        entries: currentPlan,
      },
    });
  };

  try {
    telemetry.event("acp.coding.started", { sessionId: session.id, task });
    pushTrace(traceEvent("turn", "started", true, { runId, task, cwd: session.cwd }));
    await deps.agentMessage(
      session.id,
      `Starting full coding-agent run in ${session.yolo ? "YOLO" : "Safe"} mode. I will read and edit files through ACP, run verification in a terminal, and persist detailed traces/metrics.`,
    );
    await deps.connection.sessionUpdate({
      sessionId: session.id,
      update: { sessionUpdate: "plan", entries: currentPlan },
    });

    deps.throwIfAborted(signal);
    await setPlan(0, "in_progress");
    const knowledge = await deps.runAcpTool<string>({
      sessionId: session.id,
      telemetry,
      title: "Load coding guidance",
      toolName: "bag.coding.knowledge.load",
      kind: "read",
      rawInput: { cwd: session.cwd },
      signal,
      fn: async () => loadKnowledge(session.cwd),
    });
    const scoutFindings = await deps.runAcpTool<ContextScoutFinding[]>({
      sessionId: session.id,
      telemetry,
      title: "Scout candidate files",
      toolName: "bag.coding.context.scout",
      kind: "search",
      rawInput: { task, executorConcurrency: session.executorConcurrency },
      signal,
      fn: async () =>
        runLocalContextScouts({
          router,
          config: deps.configForSession(session),
          task,
          cwd: session.cwd,
        }),
    });
    const repoContext = await deps.runAcpTool<string>({
      sessionId: session.id,
      telemetry,
      title: "Build coding context",
      toolName: "bag.coding.context.build",
      kind: "read",
      rawInput: { task, findings: scoutFindings.slice(0, 20) },
      signal,
      fn: async () =>
        buildRepoContext({
          cwd: session.cwd,
          config: deps.configForSession(session),
          task,
          findings: scoutFindings,
        }),
    });
    const projectKind = detectProjectKind(session.cwd);
    pushTrace(traceEvent("project_kind_detected", "completed", true, {
      cwd: session.cwd,
      projectKind,
    }));
    const selected = await deps.selectCodingFiles({ router, task, repoContext, knowledge, scoutFindings });
    const filesToRead = selected.filesToRead.slice(0, 8).map((path) => deps.absoluteSessionPath(session, path));
    const filesToCreate = (selected.filesToCreate ?? []).slice(0, 8).map((path) => deps.absoluteSessionPath(session, path));
    pushTrace(traceEvent("select_files", "completed", true, {
      approach: selected.approach,
      filesToRead,
      filesToCreate,
      scoutCount: scoutFindings.length,
      projectKind,
    }));
    if (filesToRead.length === 0 && filesToCreate.length > 0) {
      pushTrace(traceEvent("greenfield_detected", "completed", true, {
        filesToCreateCount: filesToCreate.length,
        projectKind,
      }));
    }
    await setPlan(0, "completed");

    deps.throwIfAborted(signal);
    await setPlan(1, "in_progress");
    const fileSnapshots: CodingFileSnapshot[] = [];
    for (const path of filesToRead) {
      const relativePath = deps.sessionRelativePath(session, path);
      try {
        const content = await deps.readClientFile({ sessionId: session.id, telemetry, path, signal });
        fileSnapshots.push({ kind: "existing", path, relativePath, content, hash: sha256(content) });
        pushTrace(traceEvent("read_file", "completed", true, {
          path,
          relativePath,
          bytes: Buffer.byteLength(content),
          hash: fileSnapshots.at(-1)?.hash,
        }));
      } catch (readError) {
        const message = readError instanceof Error ? readError.message : String(readError);
        fileSnapshots.push({ kind: "create", path, relativePath, content: "", hash: sha256("") });
        pushTrace(traceEvent("read_file_promoted_to_create", "completed", true, {
          path,
          relativePath,
          reason: message.slice(0, 240),
        }));
      }
    }
    for (const path of filesToCreate) {
      const relativePath = deps.sessionRelativePath(session, path);
      if (fileSnapshots.some((file) => file.relativePath === relativePath)) {
        continue;
      }
      fileSnapshots.push({ kind: "create", path, relativePath, content: "", hash: sha256("") });
      pushTrace(traceEvent("plan_create_file", "completed", true, { path, relativePath }));
    }
    await setPlan(1, "completed");
    const baselineFileSnapshots = fileSnapshots.map((file) => ({ ...file }));

    deps.throwIfAborted(signal);
    await setPlan(2, "in_progress");
    const editContext = deps.resolveLiveEditContext(session, fileSnapshots);
    writer.writeJson("edit-routing.json", deps.serializeLiveEditContext(editContext));
    pushTrace(traceEvent("edit_route", "completed", true, {
      strategyId: editContext.decision.selectedStrategyId,
      strategyFamily: editContext.decision.selectedStrategyFamily,
      renderedEditToolContractId: editContext.renderedContract.renderedToolId,
      degraded: editContext.decision.degraded,
      candidateCount: editContext.decision.candidates.length,
      warnings: editContext.decision.warnings,
    }));
    const patch = await deps.generateCodingPatch({ router, task, repoContext, knowledge, fileSnapshots, editContext });
    writer.writeJson("coding-patch.json", patch);
    pushTrace(traceEvent("generate_patch", "completed", patch.edits.length > 0, {
      summary: patch.summary,
      editCount: patch.edits.length,
      commandCount: patch.commands.length,
      risks: patch.risks,
      editStrategyId: patch.editStrategy.strategyId,
      editStrategyFamily: patch.editStrategy.strategyFamily,
    }));
    deps.recordPatchParseFailures({ session, telemetry, editContext, patch });
    for (const parseFailure of patch.parseFailures) {
      pushTrace(traceEvent("edit_parse", "failed", false, {
        editStrategyId: patch.editStrategy.strategyId,
        editStrategyFamily: patch.editStrategy.strategyFamily,
        renderedEditToolContractId: patch.editStrategy.renderedEditToolContractId,
        parseFailure,
      }));
    }
    await setPlan(2, "completed");

    deps.throwIfAborted(signal);
    await setPlan(3, "in_progress");
    const editResults: CodingEditResult[] = [];
    const primaryEditResults: CodingEditResult[] = [];
    let fallbackPatch: CodingPatch | undefined;
    for (const edit of patch.edits) {
      const results = await deps.previewAndWriteClientEdit({
        session,
        telemetry,
        fileSnapshots,
        edit,
        signal,
      });
      for (const result of results) {
        editResults.push(result);
        primaryEditResults.push(result);
        deps.updateFileSnapshotsFromEditResult(session, fileSnapshots, result);
        pushTrace(traceEvent("write_file", result.ok ? "completed" : "rejected_or_failed", result.ok, result));
      }
    }
    const fallbackTrigger = deps.fallbackTriggerForPatch(patch, primaryEditResults);
    const fallbackContext =
      fallbackTrigger === undefined
        ? undefined
        : deps.fallbackLiveEditContext(session, editContext, fallbackTrigger);
    const shouldAttemptFallback =
      fallbackContext !== undefined &&
      (primaryEditResults.some((result) => !result.ok) || (patch.edits.length === 0 && patch.parseFailures.length > 0));
    if (shouldAttemptFallback) {
      const activeFallbackTrigger = fallbackTrigger ?? "apply_failed";
      deps.throwIfAborted(signal);
      writer.writeJson("edit-fallback-routing.json", deps.serializeLiveEditContext(fallbackContext));
      pushTrace(traceEvent("edit_fallback_route", "completed", true, {
        trigger: activeFallbackTrigger,
        fromStrategyId: patch.editStrategy.strategyId,
        toStrategyId: fallbackContext.decision.selectedStrategyId,
        toStrategyFamily: fallbackContext.decision.selectedStrategyFamily,
        primaryFailures: primaryEditResults.filter((result) => !result.ok),
        parseFailures: patch.parseFailures,
      }));
      fallbackPatch = await deps.generateCodingPatch({
        router,
        task: [
          task,
          "",
          `Fallback after ${activeFallbackTrigger}. Primary strategy ${patch.editStrategy.strategyId} failed or produced malformed output.`,
          "Preserve the original task intent. Use the fallback contract exactly and do not hide the primary failure.",
          "",
          `Primary edit results:\n${compactJson(primaryEditResults)}`,
          `Primary parse failures:\n${compactJson(patch.parseFailures)}`,
        ].join("\n"),
        repoContext,
        knowledge,
        fileSnapshots,
        editContext: fallbackContext,
      });
      writer.writeJson("coding-fallback-1.json", fallbackPatch);
      deps.recordPatchParseFailures({ session, telemetry, editContext: fallbackContext, patch: fallbackPatch });
      pushTrace(traceEvent("fallback_patch", "completed", fallbackPatch.edits.length > 0, {
        trigger: activeFallbackTrigger,
        summary: fallbackPatch.summary,
        editCount: fallbackPatch.edits.length,
        commandCount: fallbackPatch.commands.length,
        editStrategyId: fallbackPatch.editStrategy.strategyId,
        editStrategyFamily: fallbackPatch.editStrategy.strategyFamily,
        parseFailures: fallbackPatch.parseFailures,
      }));
      for (const edit of fallbackPatch.edits) {
        const results = await deps.previewAndWriteClientEdit({
          session,
          telemetry,
          fileSnapshots,
          edit: {
            ...edit,
            reason: `Fallback after ${activeFallbackTrigger}: ${edit.reason}`,
            fallbackFromStrategyId: patch.editStrategy.strategyId,
            fallbackToStrategyId: fallbackPatch.editStrategy.strategyId,
            fallbackTrigger: activeFallbackTrigger,
          },
          signal,
        });
        for (const result of results) {
          editResults.push(result);
          deps.updateFileSnapshotsFromEditResult(session, fileSnapshots, result);
          pushTrace(traceEvent("fallback_write_file", result.ok ? "completed" : "rejected_or_failed", result.ok, {
            trigger: activeFallbackTrigger,
            ...result,
          }));
        }
      }
    }
    const noEditGenerationFailure = patch.edits.length === 0 && editResults.length === 0;
    if (noEditGenerationFailure) {
      pushTrace(traceEvent("edit_generation_failed", "failed", false, {
        reason: "coding patch generation produced no edit operations",
        summary: patch.summary,
        risks: patch.risks,
        parseFailures: patch.parseFailures,
        editStrategyId: patch.editStrategy.strategyId,
        editStrategyFamily: patch.editStrategy.strategyFamily,
      }));
      await deps.agentMessage(
        session.id,
        "Coding edit generation produced no file edits. I will still run verifier evidence where available and record this as a failed coding attempt.",
      );
    }
    const preVerifierCodingProgressDiagnostic = classifyCodingProgress({
      runId,
      patch,
      ...(fallbackPatch === undefined ? {} : { fallbackPatch }),
      editResults,
      plannedCommands: [],
      commandResults: [],
      toolMetrics: telemetry.toolMetrics,
      terminal: "pre_verifier",
    });
    const preVerifierCodingProgressPath = writer.writeJson(
      "coding-progress-diagnostics.pre-verifier.json",
      preVerifierCodingProgressDiagnostic,
    );
    pushTrace(traceEvent("coding_progress_diagnostics", "completed", preVerifierCodingProgressDiagnostic.progressClass === "verified_edit", {
      phase: "pre_verifier",
      progressClass: preVerifierCodingProgressDiagnostic.progressClass,
      reason: preVerifierCodingProgressDiagnostic.reason,
      artifactRef: preVerifierCodingProgressPath,
    }));
    await setPlan(3, "completed");

    deps.throwIfAborted(signal);
    await setPlan(4, "in_progress");
    const allPostApplyChecks: PostApplyConsistencyCheck[] = [];
    let currentPostApplyChecks = await deps.checkPostApplyConsistency({
      session,
      telemetry,
      editResults,
    });
    allPostApplyChecks.push(...currentPostApplyChecks);
    for (const check of currentPostApplyChecks) {
      pushTrace(traceEvent("post_apply_consistency", check.status, check.status === "consistent", check));
    }

    const allCommandResults: TerminalCommandResult[] = [];
    const commandResults: TerminalCommandResult[] = [];
    const baseVerificationPlan = deps.verificationCommands(patch.commands, session.cwd);
    const verificationPlan = noEditGenerationFailure && baseVerificationPlan.length === 0
      ? [noEditGenerationFailureCommand()]
      : baseVerificationPlan;
    if (verificationPlan.length === 0) {
      pushTrace(traceEvent("verification_skipped", "completed", true, {
        reason: "no verifier available for project kind",
        projectKind,
      }));
    }
    for (const command of verificationPlan) {
      const result = await deps.runTerminalCommand({
        sessionId: session.id,
        telemetry,
        command: command.command,
        args: command.args,
        reason: command.reason,
        cwd: session.cwd,
        signal,
      });
      commandResults.push(result);
      allCommandResults.push(result);
      pushTrace(traceEvent("terminal", "completed", result.exitCode === 0, result));
    }

    for (
      let repairRound = 1;
      repairRound <= 2 &&
        (
          commandResults.some((command) => command.exitCode !== 0) ||
          deps.hasPostApplyInconsistency(currentPostApplyChecks)
        );
      repairRound += 1
    ) {
      deps.throwIfAborted(signal);
      await deps.agentMessage(session.id, `Verification failed; starting repair round ${repairRound}.`);
      const postApplyFailures = currentPostApplyChecks
        .filter((check) => check.status !== "consistent")
        .map((check) => ({ path: check.path, status: check.status, reason: check.reason }));
      const repairPatch = await deps.generateCodingPatch({
        router,
        task: [
          task,
          "",
          `Repair round ${repairRound}. Previous verification or post-apply consistency failed. Fix the edited files and keep the original task intent.`,
        ].join("\n"),
        repoContext,
        knowledge,
        fileSnapshots,
        editContext,
        verifierResults: commandResults,
        postApplyFailures,
        repairRound,
      });
      writer.writeJson(`coding-repair-${repairRound}.json`, repairPatch);
      deps.recordPatchParseFailures({ session, telemetry, editContext, patch: repairPatch });
      pushTrace(traceEvent("repair_patch", "completed", repairPatch.edits.length > 0, {
        repairRound,
        summary: repairPatch.summary,
        editCount: repairPatch.edits.length,
        commandCount: repairPatch.commands.length,
      }));

      if (repairPatch.edits.length === 0) {
        break;
      }

      for (const edit of repairPatch.edits) {
        const results = await deps.previewAndWriteClientEdit({
          session,
          telemetry,
          fileSnapshots,
          edit: {
            ...edit,
            reason: `Repair round ${repairRound}: ${edit.reason}`,
            repairRound,
          },
          signal,
        });
        for (const result of results) {
          editResults.push(result);
          deps.updateFileSnapshotsFromEditResult(session, fileSnapshots, result);
          pushTrace(traceEvent("repair_write_file", result.ok ? "completed" : "rejected_or_failed", result.ok, {
            repairRound,
            ...result,
          }));
        }
      }

      currentPostApplyChecks = await deps.checkPostApplyConsistency({
        session,
        telemetry,
        editResults,
      });
      allPostApplyChecks.push(...currentPostApplyChecks);
      for (const check of currentPostApplyChecks) {
        pushTrace(traceEvent("repair_post_apply_consistency", check.status, check.status === "consistent", {
          repairRound,
          ...check,
        }));
      }

      commandResults.length = 0;
      for (const command of deps.verificationCommands(
        repairPatch.commands.length > 0 ? repairPatch.commands : patch.commands,
        session.cwd,
      )) {
        const result = await deps.runTerminalCommand({
          sessionId: session.id,
          telemetry,
          command: command.command,
          args: command.args,
          reason: `Repair round ${repairRound}: ${command.reason}`,
          cwd: session.cwd,
          signal,
        });
        commandResults.push(result);
        allCommandResults.push(result);
        pushTrace(traceEvent("repair_terminal", "completed", result.exitCode === 0, { repairRound, ...result }));
      }
    }
    const shouldRollback =
      editResults.some((result) => result.ok) &&
      (commandResults.some((command) => command.exitCode !== 0) || deps.hasPostApplyInconsistency(currentPostApplyChecks));
    const rollbackResults = shouldRollback
      ? await deps.rollbackLiveEdits({
          session,
          telemetry,
          baselineFileSnapshots,
          currentFileSnapshots: fileSnapshots,
          editResults,
        })
      : [];
    for (const result of rollbackResults) {
      deps.updateFileSnapshotsFromEditResult(session, fileSnapshots, result);
      pushTrace(traceEvent("rollback_write_file", result.ok ? "completed" : "rejected_or_failed", result.ok, result));
    }
    await setPlan(4, "completed");

    deps.throwIfAborted(signal);
    await setPlan(5, "in_progress");
    const finalCodingProgressDiagnostic = classifyCodingProgress({
      runId,
      patch,
      ...(fallbackPatch === undefined ? {} : { fallbackPatch }),
      editResults,
      postApplyChecks: allPostApplyChecks,
      plannedCommands: verificationPlan,
      commandResults: allCommandResults,
      toolMetrics: telemetry.toolMetrics,
      terminal: "final",
      evidenceRefs: [preVerifierCodingProgressPath],
    });
    const codingProgressPath = writer.writeJson("coding-progress-diagnostics.json", finalCodingProgressDiagnostic);
    telemetry.event("acp.coding.progress_diagnostic", finalCodingProgressDiagnostic as unknown as Record<string, unknown>);
    pushTrace(traceEvent("coding_progress_diagnostics", "completed", finalCodingProgressDiagnostic.progressClass === "verified_edit", {
      phase: "final",
      progressClass: finalCodingProgressDiagnostic.progressClass,
      reason: finalCodingProgressDiagnostic.reason,
      artifactRef: codingProgressPath,
    }));
    const tracePath = writer.writeJson("coding-trace.json", trace);
    const editResultsPath = writer.writeJson("edit-results.json", editResults);
    const postApplyConsistencyPath = writer.writeJson("post-apply-consistency.json", allPostApplyChecks);
    const rollbackResultsPath = writer.writeJson("rollback-results.json", rollbackResults);
    const commandResultsPath = writer.writeJson("command-results.json", commandResults);
    const allCommandResultsPath = writer.writeJson("all-command-results.json", allCommandResults);
    const finalEditAttempts = deps.recordFinalEditLifecycleTelemetry({
      session,
      telemetry,
      editResults,
      postApplyChecks: allPostApplyChecks,
      commandResults,
      rollbackResults,
      artifactRefs: [
        editResultsPath,
        postApplyConsistencyPath,
        rollbackResultsPath,
        commandResultsPath,
        allCommandResultsPath,
        codingProgressPath,
        preVerifierCodingProgressPath,
      ],
    });
    const editLifecyclePath = writer.writeJson("edit-lifecycle-attempts.json", finalEditAttempts);
    const replayCapture: AcpReplayCapture = deps.buildCodingReplayCapture({
      session,
      runId,
      task,
      tracePath,
      fileSnapshots: baselineFileSnapshots,
      editAttempts: finalEditAttempts,
      toolMetrics: telemetry.toolMetrics,
      commandResults: allCommandResults,
      artifactRefs: [
        tracePath,
        editResultsPath,
        postApplyConsistencyPath,
        rollbackResultsPath,
        commandResultsPath,
        allCommandResultsPath,
        codingProgressPath,
        preVerifierCodingProgressPath,
        editLifecyclePath,
      ],
      codingProgressDiagnostic: finalCodingProgressDiagnostic,
    });
    const replayCapturePath = writer.writeJson("replay-capture.json", replayCapture);
    const selfEvaluation = deterministicSelfEvaluation({
      threshold: deps.config.policy.selfEvalThreshold,
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      artifactCount: 13,
    });
    const optimization = optimizePolicy({
      config: deps.config,
      cwd: session.cwd,
      latestSelfEvaluation: selfEvaluation,
    });
    const artifacts = {
      patch: writer.writeJson("coding-patch-final.json", patch),
      trace: tracePath,
      editResults: editResultsPath,
      postApplyConsistency: postApplyConsistencyPath,
      rollbackResults: rollbackResultsPath,
      commandResults: commandResultsPath,
      allCommandResults: allCommandResultsPath,
      codingProgressDiagnostics: codingProgressPath,
      preVerifierCodingProgressDiagnostics: preVerifierCodingProgressPath,
      editLifecycleAttempts: editLifecyclePath,
      replayCapture: replayCapturePath,
      selfEvaluation: writer.writeJson("self-evaluation.json", selfEvaluation),
      optimization: writer.writeJson("optimization.json", optimization),
    };
    const manifest = writeManifest({
      config: deps.config,
      command: "acp-code",
      task,
      runId,
      artifacts,
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      selfEvaluation,
      writeJson: writer.writeJson,
    });
    const backgroundTrigger = deps.inspectBackgroundOptimizationTrigger(session, {
      source: "post-coding-run",
      sourceRunId: runId,
      enqueue: true,
    });
    telemetry.event(
      "maintenance.background_trigger.inspected",
      backgroundTrigger as unknown as Record<string, unknown>,
    );
    const failedCommandCount = allCommandResults.filter((command) => command.exitCode !== 0).length;
    const turnSucceeded = !noEditGenerationFailure &&
      finalCodingProgressDiagnostic.progressClass === "verified_edit" &&
      failedCommandCount === 0 &&
      !deps.hasPostApplyInconsistency(currentPostApplyChecks);
    pushTrace(traceEvent("turn", turnSucceeded ? "completed" : "failed", turnSucceeded, {
      runId,
      editCount: editResults.length,
      appliedEdits: editResults.filter((edit) => edit.ok).length,
      failedCommands: failedCommandCount,
      noEditGenerationFailure,
      codingProgressClass: finalCodingProgressDiagnostic.progressClass,
      manifest,
      backgroundOptimizationTriggered: backgroundTrigger.triggered,
    }));
    writer.writeJson("coding-trace.json", trace);
    await setPlan(5, "completed");

    await deps.agentMessage(
      session.id,
      [
        `Coding run ${turnSucceeded ? "complete" : "failed"}: ${runId}.`,
        `Applied edits: ${editResults.filter((edit) => edit.ok).length}/${editResults.length}.`,
        `Verification failures: ${failedCommandCount}/${allCommandResults.length}.`,
        `Trace: ${tracePath}`,
        `Manifest: ${manifest}`,
      ].join("\n"),
    );
  } catch (error) {
    if (!deps.isAbortError(error, signal)) {
      throw error;
    }
    const cancellation = {
      runId,
      sessionId: session.id,
      task,
      cancelledAt: new Date().toISOString(),
      completedPlanEntries: currentPlan.filter((entry) => entry.status === "completed").map((entry) => entry.content),
      inProgressPlanEntries: currentPlan.filter((entry) => entry.status === "in_progress").map((entry) => entry.content),
    };
    pushTrace(traceEvent("turn", "cancelled", false, cancellation));
    const tracePath = writer.writeJson("coding-trace.json", trace);
    const cancellationPath = writer.writeJson("cancellation.json", cancellation);
    const selfEvaluation = deterministicSelfEvaluation({
      threshold: deps.config.policy.selfEvalThreshold,
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      artifactCount: 2,
    });
    const manifest = writeManifest({
      config: deps.config,
      command: "acp-code",
      task,
      runId,
      artifacts: {
        trace: tracePath,
        cancellation: cancellationPath,
      },
      metrics: telemetry.metrics,
      llmMetrics: telemetry.llmMetrics,
      toolMetrics: telemetry.toolMetrics,
      selfEvaluation,
      writeJson: writer.writeJson,
    });
    telemetry.event("acp.coding.cancelled", {
      sessionId: session.id,
      runId,
      manifest,
      tracePath,
      cancellationPath,
    });
    await deps.agentMessage(
      session.id,
      [
        `Coding run cancelled: ${runId}.`,
        `Trace: ${tracePath}`,
        `Cancellation: ${cancellationPath}`,
        `Manifest: ${manifest}`,
      ].join("\n"),
    );
    throw error;
  }
};

const noEditGenerationFailureCommand = (): CodingCommand => ({
  command: "sh",
  args: [
    "-c",
    "printf '%s\\n' 'BleedingAgent failed this coding turn because no edit operations were generated for a mutating run.' >&2; exit 1",
  ],
  reason: "Fail closed when a mutating coding turn generates no edits and no project verifier is available.",
});
