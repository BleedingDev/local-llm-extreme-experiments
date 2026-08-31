import type {
  AgentSideConnection as AcpConnection,
  TerminalHandle,
} from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import type { AcpTerminalClient } from "../autonomous-tools";
import type { RunTelemetry } from "../telemetry";
import { resolveSessionPath } from "../workspace-paths";
import { acpFailureRawOutput } from "./permission-outcomes";
import type { BagAcpSession } from "./session";
import { markdownContent } from "./surface";
import { isAcpAbortError, throwIfAcpAborted, waitForAcpTerminalExit } from "./tool-runner";

export type TerminalCommandResult = {
  command: string;
  args: string[];
  reason: string;
  exitCode: number | null;
  signal: string | null;
  output: string;
};

export type AcpRunTerminalCommandInput = {
  sessionId: string;
  telemetry: RunTelemetry;
  command: string;
  args: string[];
  reason: string;
  cwd: string;
  signal?: AbortSignal;
};

export type AcpTerminalDeps = {
  connection: AcpConnection;
  requireSession: (sessionId: string) => BagAcpSession;
  waitForTerminalExit?: (
    terminal: Awaited<ReturnType<AcpConnection["createTerminal"]>>,
    signal?: AbortSignal,
  ) => Promise<{ exitCode?: number | null; signal?: string | null }>;
};

export const buildAcpToolUseClient = (
  connection: AcpConnection,
  handles: Map<string, TerminalHandle>,
): AcpTerminalClient => ({
  createTerminal: async (params) => {
    const handle = await connection.createTerminal({
      sessionId: params.sessionId,
      command: params.command,
      args: params.args,
      ...(params.cwd === undefined || params.cwd === null ? {} : { cwd: params.cwd }),
      ...(params.env === undefined ? {} : { env: params.env }),
      ...(params.outputByteLimit === undefined ? {} : { outputByteLimit: params.outputByteLimit }),
    });
    handles.set(handle.id, handle);
    return { terminalId: handle.id };
  },
  waitForTerminalExit: async (params) => {
    const handle = handles.get(params.terminalId);
    if (handle == null) return { exitCode: null, signal: null };
    const res = await handle.waitForExit();
    return {
      ...(res.exitCode === undefined ? {} : { exitCode: res.exitCode }),
      ...(res.signal === undefined ? {} : { signal: res.signal }),
    };
  },
  terminalOutput: async (params) => {
    const handle = handles.get(params.terminalId);
    if (handle == null) return { output: "", truncated: false };
    const res = await handle.currentOutput();
    return {
      output: res.output,
      truncated: res.truncated,
      ...(res.exitStatus === undefined ? {} : { exitStatus: res.exitStatus }),
    };
  },
  releaseTerminal: async (params) => {
    const handle = handles.get(params.terminalId);
    if (handle == null) return;
    try {
      await handle.release();
    } catch {
      /* noop */
    }
    handles.delete(params.terminalId);
  },
});

export const runTerminalCommand = async (
  deps: AcpTerminalDeps,
  input: AcpRunTerminalCommandInput,
): Promise<TerminalCommandResult> => {
  const toolCallId = `tool-${randomUUID()}`;
  const commandLine = [input.command, ...input.args].join(" ");
  const session = deps.requireSession(input.sessionId);
  const cwd = resolveSessionPath({
    cwd: session.cwd,
    additionalDirectories: session.additionalDirectories,
    path: input.cwd,
    kind: "directory",
  });
  await deps.connection.sessionUpdate({
    sessionId: input.sessionId,
    update: {
      sessionUpdate: "tool_call",
      toolCallId,
      title: `Run ${commandLine}`,
      kind: "execute",
      status: "pending",
      rawInput: { command: input.command, args: input.args, cwd, reason: input.reason },
    },
  });

  try {
    const result = await input.telemetry.measureToolCall({
      toolName: "acp.terminal.create",
      namespace: "bag.acp",
      descriptionVersion: "acp-coding-v1",
      args: { command: input.command, args: input.args, cwd, reason: input.reason },
      fn: async () => {
        throwIfAcpAborted(input.signal);
        if (!session.clientCapabilities.terminal) {
          throw new Error("ACP client does not support terminal/create");
        }
        if (!session.yolo) {
          const permission = await deps.connection.requestPermission({
            sessionId: input.sessionId,
            toolCall: {
              toolCallId,
              title: `Run ${commandLine}`,
              kind: "execute",
              status: "pending",
              rawInput: { command: input.command, args: input.args, cwd, reason: input.reason },
            },
            options: [
              { optionId: "allow", name: "Run command", kind: "allow_once" },
              { optionId: "reject", name: "Skip command", kind: "reject_once" },
            ],
          });
          if (permission.outcome.outcome === "cancelled") {
            throw new Error("cancelled");
          }
          if (permission.outcome.optionId !== "allow") {
            throw new Error("command permission rejected");
          }
        }
        throwIfAcpAborted(input.signal);
        const terminal = await deps.connection.createTerminal({
          sessionId: input.sessionId,
          command: input.command,
          args: input.args,
          cwd,
          outputByteLimit: 80_000,
        });
        await deps.connection.sessionUpdate({
          sessionId: input.sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId,
            status: "in_progress",
            content: session.clientCapabilities.richTerminalContent
              ? [{ type: "terminal", terminalId: terminal.id }]
              : [markdownContent(`Terminal started: ${terminal.id}.`)],
          },
        });
        try {
          const exit = await (deps.waitForTerminalExit ?? waitForAcpTerminalExit)(terminal, input.signal);
          const output = await terminal.currentOutput();
          return {
            command: input.command,
            args: input.args,
            reason: input.reason,
            exitCode: exit.exitCode ?? null,
            signal: exit.signal ?? null,
            output: output.output,
          };
        } finally {
          await terminal.release().catch(() => {});
        }
      },
    });
    await deps.connection.sessionUpdate({
      sessionId: input.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: result.exitCode === 0 ? "completed" : "failed",
        rawOutput: result,
        content: [markdownContent(`Command ${commandLine} exited ${result.exitCode}.\n\n\`\`\`\n${result.output.slice(-12_000)}\n\`\`\``)],
      },
    });
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const cancelled = isAcpAbortError(error, input.signal);
    const result = {
      command: input.command,
      args: input.args,
      reason: input.reason,
      exitCode: null,
      signal: cancelled ? "ABORT_ERR" : null,
      output: message,
    };
    await deps.connection.sessionUpdate({
      sessionId: input.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "failed",
        rawOutput: acpFailureRawOutput(result, { cancelled, message }),
        content: [markdownContent(cancelled ? `Command cancelled: ${commandLine}` : `Command did not run: ${message}`)],
      },
    });
    if (cancelled) {
      throw error;
    }
    return result;
  }
};
