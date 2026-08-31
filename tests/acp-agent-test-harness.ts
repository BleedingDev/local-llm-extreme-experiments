import type { BleedingAcpAgent } from "../src/acp-agent";
import type { AcpPromptRoute } from "../src/acp/prompt-router";
import type { BagAcpSession } from "../src/acp/session";
import type {
  AcpWriteClientFileInput,
  AcpWriteClientFileResult,
} from "../src/acp/workspace-io";
import type { TerminalCommandResult } from "../src/acp/terminal";
import type {
  CodingEditResult,
  CodingFileSnapshot,
  PostApplyConsistencyCheck,
} from "../src/acp/coding-types";
import type { EditAttemptContract } from "../src/edit-strategy/types";
import { defaultConfig } from "../src/config";
import { RunTelemetry } from "../src/telemetry";

type AgentInternalsForTest = {
  requireSession: (sessionId: string) => BagAcpSession;
  readClientFile: (input: {
    sessionId: string;
    telemetry: RunTelemetry;
    path: string;
    signal?: AbortSignal;
  }) => Promise<string>;
  writeClientFileWithPermission: (input: AcpWriteClientFileInput) => Promise<AcpWriteClientFileResult>;
  runTerminalCommand: (input: {
    sessionId: string;
    telemetry: RunTelemetry;
    command: string;
    args: string[];
    reason: string;
    cwd: string;
    signal?: AbortSignal;
  }) => Promise<TerminalCommandResult>;
  previewAndWriteClientEdit: (input: unknown) => Promise<unknown>;
  checkPostApplyConsistency: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editResults: readonly CodingEditResult[];
  }) => Promise<PostApplyConsistencyCheck[]>;
  recordFinalEditLifecycleTelemetry: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    editResults: readonly CodingEditResult[];
    postApplyChecks: readonly PostApplyConsistencyCheck[];
    commandResults: readonly TerminalCommandResult[];
    rollbackResults: readonly CodingEditResult[];
    artifactRefs: readonly string[];
  }) => EditAttemptContract[];
  rollbackLiveEdits: (input: {
    session: BagAcpSession;
    telemetry: RunTelemetry;
    baselineFileSnapshots: readonly CodingFileSnapshot[];
    currentFileSnapshots: readonly CodingFileSnapshot[];
    editResults: readonly CodingEditResult[];
  }) => Promise<CodingEditResult[]>;
  runCodingTurn: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<void>;
  runPlanningTurn: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<void>;
  runAcpTool: <T>(input: { toolName: string }) => Promise<T>;
  decidePromptRoute: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<AcpPromptRoute>;
};

const internals = (agent: BleedingAcpAgent): AgentInternalsForTest =>
  agent as unknown as AgentInternalsForTest;

export const requireAgentSessionForTest = (
  agent: BleedingAcpAgent,
  sessionId: string,
): BagAcpSession => internals(agent).requireSession(sessionId);

export const telemetryForAgentSession = (
  agent: BleedingAcpAgent,
  sessionId: string,
  cwd: string,
  runId: string,
): RunTelemetry => new RunTelemetry(
  defaultConfig(),
  runId,
  cwd,
  requireAgentSessionForTest(agent, sessionId).optimizerPin.telemetry,
);

export const readClientFileThroughAgentForTest = (
  agent: BleedingAcpAgent,
  input: Parameters<AgentInternalsForTest["readClientFile"]>[0],
): Promise<string> => internals(agent).readClientFile(input);

export const writeClientFileThroughAgentForTest = (
  agent: BleedingAcpAgent,
  input: AcpWriteClientFileInput,
): Promise<AcpWriteClientFileResult> => internals(agent).writeClientFileWithPermission(input);

export const previewAndWriteClientEditThroughAgentForTest = <TResult = CodingEditResult>(
  agent: BleedingAcpAgent,
  input: unknown,
): Promise<TResult> => internals(agent).previewAndWriteClientEdit(input) as Promise<TResult>;

export const checkPostApplyConsistencyThroughAgentForTest = (
  agent: BleedingAcpAgent,
  input: Parameters<AgentInternalsForTest["checkPostApplyConsistency"]>[0],
): Promise<PostApplyConsistencyCheck[]> => internals(agent).checkPostApplyConsistency(input);

export const recordFinalEditLifecycleTelemetryThroughAgentForTest = (
  agent: BleedingAcpAgent,
  input: Parameters<AgentInternalsForTest["recordFinalEditLifecycleTelemetry"]>[0],
): EditAttemptContract[] => internals(agent).recordFinalEditLifecycleTelemetry(input);

export const rollbackLiveEditsThroughAgentForTest = (
  agent: BleedingAcpAgent,
  input: Parameters<AgentInternalsForTest["rollbackLiveEdits"]>[0],
): Promise<CodingEditResult[]> => internals(agent).rollbackLiveEdits(input);

export const runTerminalCommandThroughAgentForTest = (
  agent: BleedingAcpAgent,
  input: Parameters<AgentInternalsForTest["runTerminalCommand"]>[0],
): Promise<TerminalCommandResult> => internals(agent).runTerminalCommand(input);

export const replaceRunCodingTurnForTest = (
  agent: BleedingAcpAgent,
  fn: AgentInternalsForTest["runCodingTurn"],
): void => {
  internals(agent).runCodingTurn = fn;
};

export const replaceRunPlanningTurnForTest = (
  agent: BleedingAcpAgent,
  fn: AgentInternalsForTest["runPlanningTurn"],
): void => {
  internals(agent).runPlanningTurn = fn;
};

export const replaceRunAcpToolForTest = (
  agent: BleedingAcpAgent,
  fn: AgentInternalsForTest["runAcpTool"],
): void => {
  internals(agent).runAcpTool = fn;
};

export const replaceDecidePromptRouteForTest = (
  agent: BleedingAcpAgent,
  fn: AgentInternalsForTest["decidePromptRoute"],
): void => {
  internals(agent).decidePromptRoute = fn;
};
