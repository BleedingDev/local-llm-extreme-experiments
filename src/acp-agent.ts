import {
  AgentSideConnection,
  PROTOCOL_VERSION,
  ndJsonStream,
  type Agent,
  type AgentSideConnection as AcpConnection,
  type AuthenticateRequest,
  type AuthenticateResponse,
  type CancelNotification,
  type CloseSessionRequest,
  type CloseSessionResponse,
  type InitializeRequest,
  type InitializeResponse,
  type ListSessionsRequest,
  type ListSessionsResponse,
  type LoadSessionRequest,
  type LoadSessionResponse,
  type McpServer,
  type NewSessionRequest,
  type NewSessionResponse,
  type PromptRequest,
  type PromptResponse,
  type ResumeSessionRequest,
  type ResumeSessionResponse,
  type SetSessionConfigOptionRequest,
  type SetSessionConfigOptionResponse,
  type SetSessionModeRequest,
  type SetSessionModeResponse,
  type ToolCallContent,
} from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import { Readable, Writable } from "node:stream";
import {
  configForAcpSession,
  createAcpOptimizerSessionPin,
  createBagAcpSession,
  publishAcpAvailableCommands,
  resumeOrCreateBagAcpSession,
  runWithTemporaryAcpMode,
  setAcpSessionModeUpdate,
  type BagAcpSession,
  type BagOptimizerSessionPin,
} from "./acp/session";
import {
  acpClientCapabilityProfileFromInitialize,
  defaultAcpClientCapabilityProfile,
  modeState,
  promptToText,
  sessionConfigOptions,
  type AcpClientCapabilityProfile,
  type AcpSessionMode,
} from "./acp/surface";
import {
  completedAcpToolStatus as completedAcpToolStatusText,
  isAcpAbortError,
  runAcpTool as runAcpToolUpdate,
  sendAcpAgentMessage,
  throwIfAcpAborted,
  waitForAcpTerminalExit,
  type AcpToolInput,
} from "./acp/tool-runner";
import {
  runTerminalCommand as runAcpTerminalCommand,
  type TerminalCommandResult,
} from "./acp/terminal";
import {
  runAcpAdaptiveCodingTurn,
  runAcpAutonomousToolUseTurn,
  runAcpDagDrivenToolUseTurn,
} from "./acp/tool-use-runner";
import {
  absoluteSessionPath as resolveAcpSessionPath,
  displayPathForSessionId as displayAcpPathForSessionId,
  editToolContent as renderAcpEditToolContent,
  readClientFile as readAcpClientFile,
  sessionRelativePath as acpSessionRelativePath,
  writeClientFileWithPermission as writeAcpClientFileWithPermission,
  type AcpEditToolContentInput,
  type AcpWriteClientFileInput,
  type AcpWriteClientFileResult,
} from "./acp/workspace-io";
import {
  handleMaintenanceCommand as runAcpMaintenanceCommand,
  inspectBackgroundOptimizationTrigger as inspectAcpBackgroundOptimizationTrigger,
  type BackgroundOptimizationTriggerDiagnostic,
  type BackgroundOptimizationTriggerInput,
} from "./acp/maintenance";
import { runLiveMcpToolCall as runAcpLiveMcpToolCall } from "./acp/mcp-bridge";
import { runAcpPlanningTurn } from "./acp/planning-runner";
import { decideAcpPromptRoute, runAcpConversationTurn, type AcpPromptRoute } from "./acp/prompt-router";
import { discoverAcpSkills } from "./acp/skills";
import { handleAcpSlashCommand, type SkillSummary } from "./acp/slash-router";
import {
  verificationCommands as acpVerificationCommands,
  type CodingCommand,
  type CodingEditOperation,
  type CodingEditResult,
  type CodingFileSelection,
  type CodingFileSnapshot,
  type CodingPatch,
  type LiveEditContext,
  type PostApplyConsistencyCheck,
} from "./acp/coding-types";
import {
  fallbackLiveEditContext as acpFallbackLiveEditContext,
  fallbackTriggerForPatch as acpFallbackTriggerForPatch,
  resolveLiveEditContext as resolveAcpLiveEditContext,
  serializeLiveEditContext as serializeAcpLiveEditContext,
} from "./acp/edit-routing";
import {
  checkPostApplyConsistency as checkAcpPostApplyConsistency,
  hasPostApplyInconsistency as hasAcpPostApplyInconsistency,
  previewAndWriteClientEdit as previewAndWriteAcpClientEdit,
  rollbackLiveEdits as rollbackAcpLiveEdits,
  updateFileSnapshotsFromEditResult as updateAcpFileSnapshotsFromEditResult,
} from "./acp/edit-lifecycle";
import {
  editAttemptFromParseFailure as buildAcpParseFailureEditAttempt,
  recordFinalEditLifecycleTelemetry as recordAcpFinalEditLifecycleTelemetry,
} from "./acp/edit-telemetry";
import {
  generateCodingPatch as generateAcpCodingPatch,
  selectCodingFiles as selectAcpCodingFiles,
} from "./acp/coding-generation";
import { runAcpCodingTurn } from "./acp/coding-runner";
import {
  buildCodingReplayCapture as buildAcpCodingReplayCapture,
} from "./acp/replay-capture";
import type { CodingProgressDiagnostic } from "./acp/coding-progress-diagnostics";
export { acpConsumerCompatibilityMatrix } from "./acp/surface";
export type {
  AcpConsumerCompatibilityCase,
  AcpConsumerCompatibilityStatus,
  AcpNamedConsumerFixture,
} from "./acp/surface";
export { readAcpSettingsSnippet, readAcpZedSettingsSnippet } from "./acp/settings";
import { loadConfig } from "./config";
import type {
  EditAttemptContract,
} from "./edit-strategy/types";
import type { LlmRouter } from "./llm";
import {
  type McpRuntimeToolCall,
  type McpRuntimeToolExecutor,
  type McpRuntimeToolResult,
  type McpServerMetadata,
} from "./mcp/runtime-tools";
import {
  type EditStrategyFallbackRule,
} from "./optimizer/edit-policy-router";
import type { AcpReplayCapture } from "./replay";
import { RunTelemetry } from "./telemetry";
import type { BagConfig, ContextScoutFinding, ToolCallMetric } from "./types";

type SessionMode = AcpSessionMode;

export class BleedingAcpAgent implements Agent {
  private readonly sessions = new Map<string, BagAcpSession>();
  private readonly config: BagConfig;
  private clientCapabilities: AcpClientCapabilityProfile = defaultAcpClientCapabilityProfile();

  constructor(
    private readonly connection: AcpConnection,
    cwd = process.cwd(),
  ) {
    this.config = loadConfig(cwd);
  }

  async initialize(params: InitializeRequest): Promise<InitializeResponse> {
    this.clientCapabilities = acpClientCapabilityProfileFromInitialize(
      params.clientCapabilities,
      params.clientInfo?.name == null ? "initialize" : `initialize:${params.clientInfo.name}`,
    );
    return {
      protocolVersion: PROTOCOL_VERSION,
      agentCapabilities: {
        loadSession: false,
        promptCapabilities: {
          embeddedContext: true,
          image: false,
          audio: false,
        },
        mcpCapabilities: {
          http: true,
          sse: true,
        },
        sessionCapabilities: {
          close: {},
          list: {},
          resume: {},
          additionalDirectories: {},
        },
      },
      agentInfo: {
        name: "bleeding-agent",
        title: "BleedingAgent",
        version: "1.0.0",
      },
      authMethods: [],
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    const session = this.createSession(params.cwd, params.additionalDirectories ?? [], undefined, params.mcpServers);
    await this.publishAvailableCommands(session);
    return {
      sessionId: session.id,
      modes: modeState(session.mode),
      configOptions: sessionConfigOptions(this.config, session),
    };
  }

  async resumeSession(params: ResumeSessionRequest): Promise<ResumeSessionResponse> {
    const session = this.resumeOrCreateSession(
      params.cwd,
      params.additionalDirectories ?? [],
      params.sessionId,
      params.mcpServers ?? [],
    );
    await this.publishAvailableCommands(session);
    return {
      modes: modeState(session.mode),
      configOptions: sessionConfigOptions(this.config, session),
    };
  }

  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const session = this.resumeOrCreateSession(
      params.cwd,
      params.additionalDirectories ?? [],
      params.sessionId,
      params.mcpServers ?? [],
    );
    await this.publishAvailableCommands(session);
    return {
      modes: modeState(session.mode),
      configOptions: sessionConfigOptions(this.config, session),
    };
  }

  async listSessions(_params: ListSessionsRequest): Promise<ListSessionsResponse> {
    return {
      sessions: [...this.sessions.values()].map((session) => ({
        sessionId: session.id,
        cwd: session.cwd,
        title: session.title,
        updatedAt: session.updatedAt,
      })),
    };
  }

  async closeSession(params: CloseSessionRequest): Promise<CloseSessionResponse> {
    await this.cancel({ sessionId: params.sessionId });
    this.sessions.delete(params.sessionId);
    return {};
  }

  async setSessionMode(params: SetSessionModeRequest): Promise<SetSessionModeResponse> {
    const session = this.requireSession(params.sessionId);
    if (params.modeId !== "auto" && params.modeId !== "chat" && params.modeId !== "plan" && params.modeId !== "run") {
      throw new Error(`unsupported mode: ${params.modeId}`);
    }
    session.mode = params.modeId;
    session.updatedAt = new Date().toISOString();
    await this.connection.sessionUpdate({
      sessionId: session.id,
      update: {
        sessionUpdate: "current_mode_update",
        currentModeId: session.mode,
      },
    });
    return {};
  }

  async setSessionConfigOption(params: SetSessionConfigOptionRequest): Promise<SetSessionConfigOptionResponse> {
    const session = this.sessions.get(params.sessionId);
    if (params.configId === "yolo") {
      if (session != null) {
        session.yolo = Boolean(params.value);
        session.updatedAt = new Date().toISOString();
        await this.agentMessage(session.id, session.yolo ? "YOLO mode enabled." : "Safe mode enabled.");
      }
      return {
        configOptions: sessionConfigOptions(this.config, session),
      };
    }
    if (params.configId !== "executor-concurrency") {
      throw new Error(`unsupported config option: ${params.configId}`);
    }
    const executorConcurrency = Number(params.value);
    if (!Number.isInteger(executorConcurrency) || executorConcurrency <= 0) {
      throw new Error(`invalid executor concurrency: ${String(params.value)}`);
    }
    if (session != null) {
      session.executorConcurrency = Math.min(executorConcurrency, this.config.policy.maxExecutorConcurrency);
      session.updatedAt = new Date().toISOString();
    }
    return {
      configOptions: sessionConfigOptions(this.config, session),
    };
  }

  async authenticate(_params: AuthenticateRequest): Promise<AuthenticateResponse> {
    return {};
  }

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const session = this.requireSession(params.sessionId);
    session.pendingPrompt?.abort();
    const abortController = new AbortController();
    session.pendingPrompt = abortController;

    try {
      const task = promptToText(params.prompt);
      if (task === "") {
        await this.agentMessage(session.id, "I need a text prompt or embedded text resource to start.");
        return { stopReason: "end_turn" };
      }
      const commandHandled = await this.handleSlashCommand(session, task, abortController.signal);
      if (commandHandled) {
        return { stopReason: "end_turn" };
      }

      session.title = task.split("\n")[0]?.slice(0, 80) ?? "BleedingAgent session";
      session.updatedAt = new Date().toISOString();
      if (session.mode === "chat") {
        await this.runConversationTurn(session, task);
        return { stopReason: "end_turn" };
      }
      const previousMode = session.mode;
      const route = previousMode === "auto" ? await this.decidePromptRoute(session, task, abortController.signal) : previousMode;
      if (route === "chat") {
        await this.runConversationTurn(session, task);
      } else if (route === "run") {
        await this.runWithTemporaryMode(session, route, previousMode, () =>
          this.runCodingTurn(session, task, abortController.signal),
        );
      } else {
        await this.runWithTemporaryMode(session, route, previousMode, () =>
          this.runPlanningTurn(session, task, abortController.signal),
        );
      }
      return { stopReason: "end_turn" };
    } catch (error) {
      if (abortController.signal.aborted) {
        return { stopReason: "cancelled" };
      }
      await this.agentMessage(params.sessionId, `ACP turn failed: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    } finally {
      if (session.pendingPrompt === abortController) {
        session.pendingPrompt = null;
      }
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    this.sessions.get(params.sessionId)?.pendingPrompt?.abort();
  }

  async unstable_forkSession(): Promise<never> {
    throw new Error("session/fork is not supported");
  }

  async unstable_setSessionModel(): Promise<never> {
    throw new Error("session model selection is not supported");
  }

  async unstable_listProviders(): Promise<never> {
    throw new Error("provider configuration is not supported");
  }

  async unstable_setProvider(): Promise<never> {
    throw new Error("provider configuration is not supported");
  }

  async unstable_disableProvider(): Promise<never> {
    throw new Error("provider configuration is not supported");
  }

  async unstable_logout(): Promise<Record<string, never>> {
    return {};
  }

  async unstable_startNes(): Promise<never> {
    throw new Error("NES is not supported");
  }

  async unstable_suggestNes(): Promise<never> {
    throw new Error("NES is not supported");
  }

  async unstable_closeNes(): Promise<never> {
    throw new Error("NES is not supported");
  }

  async unstable_didOpenDocument(): Promise<void> {}
  async unstable_didChangeDocument(): Promise<void> {}
  async unstable_didCloseDocument(): Promise<void> {}
  async unstable_didSaveDocument(): Promise<void> {}
  async unstable_didFocusDocument(): Promise<void> {}
  async unstable_acceptNes(): Promise<void> {}
  async unstable_rejectNes(): Promise<void> {}

  async extMethod(_method: string, _params: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async extNotification(_method: string, _params: Record<string, unknown>): Promise<void> {}

  private createSession(
    cwd: string,
    additionalDirectories: string[],
    id = `bag-${randomUUID()}`,
    mcpServers: McpServer[] = [],
  ): BagAcpSession {
    return createBagAcpSession({
      config: this.config,
      sessions: this.sessions,
      cwd,
      additionalDirectories,
      id,
      mcpServers,
      clientCapabilities: this.clientCapabilities,
      createOptimizerSessionPin: (resolvedCwd) => this.createOptimizerSessionPin(resolvedCwd),
    });
  }

  private resumeOrCreateSession(
    cwd: string,
    additionalDirectories: string[],
    id: string,
    mcpServers: McpServer[] = [],
  ): BagAcpSession {
    return resumeOrCreateBagAcpSession({
      config: this.config,
      sessions: this.sessions,
      cwd,
      additionalDirectories,
      id,
      mcpServers,
      clientCapabilities: this.clientCapabilities,
      createOptimizerSessionPin: (resolvedCwd) => this.createOptimizerSessionPin(resolvedCwd),
    });
  }

  private createOptimizerSessionPin(cwd: string): BagOptimizerSessionPin {
    return createAcpOptimizerSessionPin(this.config, cwd);
  }

  private configForSession(session: BagAcpSession): BagConfig {
    return configForAcpSession(this.config, session);
  }

  private async publishAvailableCommands(session: BagAcpSession): Promise<void> {
    await publishAcpAvailableCommands(this.connection, session);
  }

  private async handleSlashCommand(session: BagAcpSession, text: string, signal: AbortSignal): Promise<boolean> {
    return handleAcpSlashCommand(
      {
        connection: this.connection,
        config: this.config,
        agentMessage: (sessionId, message) => this.agentMessage(sessionId, message),
        listSkills: () => this.listSkills(),
        runWithTemporaryMode: (targetSession, activeMode, previousMode, fn) =>
          this.runWithTemporaryMode(targetSession, activeMode, previousMode, fn),
        runCodingTurn: (targetSession, task, targetSignal) => this.runCodingTurn(targetSession, task, targetSignal),
        runPlanningTurn: (targetSession, task, targetSignal) => this.runPlanningTurn(targetSession, task, targetSignal),
        runAutonomousToolUseTurn: (targetSession, task, targetSignal) =>
          this.runAutonomousToolUseTurn(targetSession, task, targetSignal),
        runDagDrivenToolUseTurn: (targetSession, task, targetSignal) =>
          this.runDagDrivenToolUseTurn(targetSession, task, targetSignal),
        runAdaptiveCodingTurn: (targetSession, task, targetSignal) =>
          this.runAdaptiveCodingTurn(targetSession, task, targetSignal),
        runMaintenanceCommand: (targetSession, task) => this.handleMaintenanceCommand(targetSession, task),
      },
      { session, text, signal },
    );
  }

  private async handleMaintenanceCommand(session: BagAcpSession, task: string): Promise<void> {
    await runAcpMaintenanceCommand(
      {
        connection: this.connection,
        config: this.config,
        agentMessage: (sessionId, message) => this.agentMessage(sessionId, message),
      },
      session,
      task,
    );
  }

  private inspectBackgroundOptimizationTrigger(
    session: BagAcpSession,
    input: BackgroundOptimizationTriggerInput,
  ): BackgroundOptimizationTriggerDiagnostic {
    return inspectAcpBackgroundOptimizationTrigger(this.config, session, input);
  }

  private toolUseRunnerDeps() {
    return {
      connection: this.connection,
      config: this.config,
      agentMessage: (sessionId: string, message: string) => this.agentMessage(sessionId, message),
    };
  }

  private async setSessionModeUpdate(session: BagAcpSession, mode: SessionMode): Promise<void> {
    await setAcpSessionModeUpdate(this.connection, session, mode);
  }

  private async runWithTemporaryMode<T>(
    session: BagAcpSession,
    activeMode: "plan" | "run",
    previousMode: SessionMode,
    fn: () => Promise<T>,
  ): Promise<T> {
    return runWithTemporaryAcpMode({
      session,
      activeMode,
      previousMode,
      setMode: (mode) => this.setSessionModeUpdate(session, mode),
      fn,
    });
  }

  private async runConversationTurn(session: BagAcpSession, _text: string): Promise<void> {
    await runAcpConversationTurn(
      { agentMessage: (sessionId, message) => this.agentMessage(sessionId, message) },
      session,
    );
  }

  private async decidePromptRoute(session: BagAcpSession, task: string, signal: AbortSignal): Promise<AcpPromptRoute> {
    return decideAcpPromptRoute(
      {
        config: this.config,
        agentMessage: (sessionId, message) => this.agentMessage(sessionId, message),
        throwIfAborted: (targetSignal) => this.throwIfAborted(targetSignal),
      },
      { session, task, signal },
    );
  }

  private listSkills(): SkillSummary[] {
    return discoverAcpSkills();
  }

  private requireSession(sessionId: string): BagAcpSession {
    const session = this.sessions.get(sessionId);
    if (session == null) {
      throw new Error(`session not found: ${sessionId}`);
    }
    return session;
  }

  private async runPlanningTurn(session: BagAcpSession, task: string, signal: AbortSignal): Promise<void> {
    await runAcpPlanningTurn(
      {
        connection: this.connection,
        config: this.config,
        agentMessage: (sessionId, message) => this.agentMessage(sessionId, message),
        runAcpTool: (input) => this.runAcpTool(input),
        configForSession: (targetSession) => this.configForSession(targetSession),
        throwIfAborted: (targetSignal) => this.throwIfAborted(targetSignal),
        isAbortError: (error, targetSignal) => this.isAbortError(error, targetSignal),
      },
      { session, task, signal },
    );
  }

  private async runAdaptiveCodingTurn(session: BagAcpSession, task: string, signal: AbortSignal): Promise<void> {
    await runAcpAdaptiveCodingTurn(this.toolUseRunnerDeps(), { session, task, signal });
  }

  private async runDagDrivenToolUseTurn(session: BagAcpSession, task: string, signal: AbortSignal): Promise<void> {
    await runAcpDagDrivenToolUseTurn(this.toolUseRunnerDeps(), { session, task, signal });
  }

  private async runAutonomousToolUseTurn(session: BagAcpSession, task: string, signal: AbortSignal): Promise<void> {
    await runAcpAutonomousToolUseTurn(this.toolUseRunnerDeps(), { session, task, signal });
  }

  private async runCodingTurn(session: BagAcpSession, task: string, signal: AbortSignal): Promise<void> {
    await runAcpCodingTurn(
      {
        connection: this.connection,
        config: this.config,
        agentMessage: (sessionId, message) => this.agentMessage(sessionId, message),
        runAcpTool: (input) => this.runAcpTool(input),
        configForSession: (targetSession) => this.configForSession(targetSession),
        throwIfAborted: (targetSignal) => this.throwIfAborted(targetSignal),
        isAbortError: (error, targetSignal) => this.isAbortError(error, targetSignal),
        absoluteSessionPath: (targetSession, path) => this.absoluteSessionPath(targetSession, path),
        sessionRelativePath: (targetSession, path) => this.sessionRelativePath(targetSession, path),
        readClientFile: (input) => this.readClientFile(input),
        selectCodingFiles: (input) => this.selectCodingFiles(input),
        resolveLiveEditContext: (targetSession, fileSnapshots) =>
          this.resolveLiveEditContext(targetSession, fileSnapshots),
        serializeLiveEditContext: (context) => this.serializeLiveEditContext(context),
        generateCodingPatch: (input) => this.generateCodingPatch(input),
        recordPatchParseFailures: (input) => this.recordPatchParseFailures(input),
        previewAndWriteClientEdit: (input) => this.previewAndWriteClientEdit(input),
        updateFileSnapshotsFromEditResult: (targetSession, fileSnapshots, result) =>
          this.updateFileSnapshotsFromEditResult(targetSession, fileSnapshots, result),
        fallbackTriggerForPatch: (patch, results) => this.fallbackTriggerForPatch(patch, results),
        fallbackLiveEditContext: (targetSession, current, trigger) =>
          this.fallbackLiveEditContext(targetSession, current, trigger),
        checkPostApplyConsistency: (input) => this.checkPostApplyConsistency(input),
        hasPostApplyInconsistency: (checks) => this.hasPostApplyInconsistency(checks),
        verificationCommands: (commands, cwd) => this.verificationCommands(commands, cwd),
        runTerminalCommand: (input) => this.runTerminalCommand(input),
        rollbackLiveEdits: (input) => this.rollbackLiveEdits(input),
        recordFinalEditLifecycleTelemetry: (input) => this.recordFinalEditLifecycleTelemetry(input),
        buildCodingReplayCapture: (input) => this.buildCodingReplayCapture(input),
        inspectBackgroundOptimizationTrigger: (targetSession, input) =>
          this.inspectBackgroundOptimizationTrigger(targetSession, input),
      },
      { session, task, signal },
    );
  }

  private absoluteSessionPath(session: BagAcpSession, path: string): string {
    return resolveAcpSessionPath(session, path);
  }

  private async selectCodingFiles(input: {
    router: LlmRouter;
    task: string;
    repoContext: string;
    knowledge: string;
    scoutFindings: ContextScoutFinding[];
  }): Promise<CodingFileSelection> {
    return selectAcpCodingFiles(input);
  }

  private async generateCodingPatch(input: {
    router: LlmRouter;
    task: string;
    repoContext: string;
    knowledge: string;
    fileSnapshots: CodingFileSnapshot[];
    editContext: LiveEditContext;
    verifierResults?: readonly TerminalCommandResult[];
    postApplyFailures?: readonly { path: string; status: string; reason?: string }[];
    repairRound?: number;
  }): Promise<CodingPatch> {
    return generateAcpCodingPatch({ ...input, config: this.config });
  }

  private async readClientFile(input: {
    sessionId: string;
    telemetry: RunTelemetry;
    path: string;
    signal?: AbortSignal;
  }): Promise<string> {
    return readAcpClientFile({
      connection: this.connection,
      requireSession: (sessionId) => this.requireSession(sessionId),
      runAcpTool: (toolInput) => this.runAcpTool(toolInput),
    }, input);
  }

  private sessionRelativePath(session: BagAcpSession, path: string): string {
    return acpSessionRelativePath(session, path);
  }

  private editToolContent(input: AcpEditToolContentInput): ToolCallContent[] {
    return renderAcpEditToolContent(input);
  }

  private async writeClientFileWithPermission(
    input: AcpWriteClientFileInput,
  ): Promise<AcpWriteClientFileResult> {
    return writeAcpClientFileWithPermission({
      connection: this.connection,
      requireSession: (sessionId) => this.requireSession(sessionId),
      runAcpTool: (toolInput) => this.runAcpTool(toolInput),
      editToolContent: (contentInput) => this.editToolContent(contentInput),
    }, input);
  }

  private resolveLiveEditContext(session: BagAcpSession, fileSnapshots: CodingFileSnapshot[]): LiveEditContext {
    return resolveAcpLiveEditContext(session, fileSnapshots);
  }

  private serializeLiveEditContext(context: LiveEditContext): Record<string, unknown> {
    return serializeAcpLiveEditContext(context);
  }

  private fallbackLiveEditContext(
    session: BagAcpSession,
    current: LiveEditContext,
    trigger: EditStrategyFallbackRule["trigger"],
  ): LiveEditContext | undefined {
    return acpFallbackLiveEditContext(session, current, trigger);
  }

  private fallbackTriggerForPatch(patch: CodingPatch, results: readonly CodingEditResult[]): EditStrategyFallbackRule["trigger"] | undefined {
    return acpFallbackTriggerForPatch(patch, results);
  }

  private editLifecycleDeps() {
    return {
      runAcpTool: <T>(toolInput: AcpToolInput) => this.runAcpTool<T>(toolInput),
      readClientFile: (readInput: Parameters<typeof readAcpClientFile>[1]) => this.readClientFile(readInput),
      writeClientFileWithPermission: (writeInput: AcpWriteClientFileInput) =>
        this.writeClientFileWithPermission(writeInput),
      absoluteSessionPath: (session: BagAcpSession, path: string) => this.absoluteSessionPath(session, path),
      sessionRelativePath: (session: BagAcpSession, path: string) => this.sessionRelativePath(session, path),
    };
  }

  private recordPatchParseFailures(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editContext: LiveEditContext;
    patch: CodingPatch;
  }): void {
    for (const parseFailure of input.patch.parseFailures) {
      input.telemetry.recordEditAttempt(
        this.editAttemptFromParseFailure({
          session: input.session,
          editContext: input.editContext,
          parseFailure,
        }),
      );
    }
  }

  private editAttemptFromParseFailure(input: {
    session: BagAcpSession;
    editContext: LiveEditContext;
    parseFailure: string;
  }): EditAttemptContract {
    return buildAcpParseFailureEditAttempt(input);
  }

  private async previewAndWriteClientEdit(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    fileSnapshots: CodingFileSnapshot[];
    edit: CodingEditOperation;
    signal?: AbortSignal;
  }): Promise<CodingEditResult[]>;
  private async previewAndWriteClientEdit(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    path: string;
    oldContent: string;
    newContent: string;
    reason: string;
  }): Promise<CodingEditResult>;
  private async previewAndWriteClientEdit(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    fileSnapshots?: CodingFileSnapshot[];
    edit?: CodingEditOperation;
    path?: string;
    oldContent?: string;
    newContent?: string;
    reason?: string;
    signal?: AbortSignal;
  }): Promise<CodingEditResult[] | CodingEditResult> {
    return previewAndWriteAcpClientEdit(this.editLifecycleDeps(), input as never) as Promise<CodingEditResult[] | CodingEditResult>;
  }

  private updateFileSnapshotsFromEditResult(
    session: BagAcpSession,
    fileSnapshots: CodingFileSnapshot[],
    result: CodingEditResult,
  ): void {
    updateAcpFileSnapshotsFromEditResult(this.editLifecycleDeps(), session, fileSnapshots, result);
  }

  private async checkPostApplyConsistency(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editResults: readonly CodingEditResult[];
  }): Promise<PostApplyConsistencyCheck[]> {
    return checkAcpPostApplyConsistency(this.editLifecycleDeps(), input);
  }

  private hasPostApplyInconsistency(checks: readonly PostApplyConsistencyCheck[]): boolean {
    return hasAcpPostApplyInconsistency(checks);
  }

  private async rollbackLiveEdits(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    baselineFileSnapshots: readonly CodingFileSnapshot[];
    currentFileSnapshots: readonly CodingFileSnapshot[];
    editResults: readonly CodingEditResult[];
  }): Promise<CodingEditResult[]> {
    return rollbackAcpLiveEdits(this.editLifecycleDeps(), input);
  }

  private recordFinalEditLifecycleTelemetry(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editResults: readonly CodingEditResult[];
    postApplyChecks: readonly PostApplyConsistencyCheck[];
    commandResults: readonly TerminalCommandResult[];
    rollbackResults: readonly CodingEditResult[];
    artifactRefs: readonly string[];
  }): EditAttemptContract[] {
    void input.session;
    return recordAcpFinalEditLifecycleTelemetry(input);
  }

  async runLiveMcpToolCall(input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    server: McpServerMetadata;
    call: McpRuntimeToolCall;
    executor: McpRuntimeToolExecutor;
    signal?: AbortSignal;
  }): Promise<McpRuntimeToolResult> {
    return runAcpLiveMcpToolCall({ connection: this.connection }, input);
  }

  private buildCodingReplayCapture(input: {
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
  }): AcpReplayCapture {
    return buildAcpCodingReplayCapture(input);
  }

  private verificationCommands(commands: CodingCommand[], cwd: string): CodingCommand[] {
    return acpVerificationCommands(commands, cwd);
  }

  private async runTerminalCommand(input: {
    sessionId: string;
    telemetry: RunTelemetry;
    command: string;
    args: string[];
    reason: string;
    cwd: string;
    signal?: AbortSignal;
  }): Promise<TerminalCommandResult> {
    return runAcpTerminalCommand({
      connection: this.connection,
      requireSession: (sessionId) => this.requireSession(sessionId),
      waitForTerminalExit: (terminal, signal) => this.waitForTerminalExit(terminal, signal),
    }, input);
  }

  private async runAcpTool<T>(input: AcpToolInput): Promise<T> {
    return runAcpToolUpdate<T>(this.connection, input, {
      displayPathForSessionId: (sessionId, path) => this.displayPathForSessionId(sessionId, path),
      completedStatus: (toolInput, result) => this.completedAcpToolStatus(toolInput, result),
    });
  }

  private completedAcpToolStatus(input: AcpToolInput, result: unknown): string {
    return completedAcpToolStatusText(input, result, (sessionId, path) =>
      this.displayPathForSessionId(sessionId, path),
    );
  }

  private displayPathForSessionId(sessionId: string, path: string): string {
    return displayAcpPathForSessionId(this.sessions, sessionId, path);
  }

  private isAbortError(error: unknown, signal?: AbortSignal): boolean {
    return isAcpAbortError(error, signal);
  }

  private async waitForTerminalExit(
    terminal: Awaited<ReturnType<AcpConnection["createTerminal"]>>,
    signal?: AbortSignal,
  ): Promise<{ exitCode?: number | null; signal?: string | null }> {
    return waitForAcpTerminalExit(terminal, signal);
  }

  private async agentMessage(sessionId: string, text: string): Promise<void> {
    await sendAcpAgentMessage(this.connection, sessionId, text);
  }

  private throwIfAborted(signal?: AbortSignal): void {
    throwIfAcpAborted(signal);
  }
}

export const startAcpServer = async (): Promise<void> => {
  const input = Writable.toWeb(process.stdout);
  const output = Readable.toWeb(process.stdin);
  const stream = ndJsonStream(input, output);
  const connection = new AgentSideConnection((conn) => new BleedingAcpAgent(conn), stream);
  await connection.closed;
};
