import type {
  AgentSideConnection as AcpConnection,
  ToolKind,
} from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import { acpFailureOutcomeFor } from "./permission-outcomes";
import { markdownContent } from "./surface";
import type { RunTelemetry } from "../telemetry";

export type AcpToolInput = {
  sessionId: string;
  telemetry: RunTelemetry;
  title: string;
  toolName: string;
  kind: ToolKind;
  rawInput?: unknown;
  locations?: Array<{ path: string; line?: number | null }>;
  signal?: AbortSignal;
  fn: () => Promise<unknown>;
};

export const throwIfAcpAborted = (signal?: AbortSignal): void => {
  if (signal?.aborted) {
    throw new Error("cancelled");
  }
};

export const isAcpAbortError = (error: unknown, signal?: AbortSignal): boolean => {
  if (signal?.aborted) {
    return true;
  }
  return error instanceof Error && error.message === "cancelled";
};

export const waitForAcpTerminalExit = async (
  terminal: Awaited<ReturnType<AcpConnection["createTerminal"]>>,
  signal?: AbortSignal,
): Promise<{ exitCode?: number | null; signal?: string | null }> => {
  if (signal == null) {
    return terminal.waitForExit();
  }
  throwIfAcpAborted(signal);
  return new Promise((resolvePromise, rejectPromise) => {
    const abort = () => {
      terminal.kill().catch(() => {}).finally(() => {
        rejectPromise(new Error("cancelled"));
      });
    };
    signal.addEventListener("abort", abort, { once: true });
    terminal.waitForExit().then(resolvePromise, rejectPromise).finally(() => {
      signal.removeEventListener("abort", abort);
    });
  });
};

export const sendAcpAgentMessage = async (
  connection: AcpConnection,
  sessionId: string,
  text: string,
): Promise<void> => {
  await connection.sessionUpdate({
    sessionId,
    update: {
      sessionUpdate: "agent_message_chunk",
      content: {
        type: "text",
        text,
      },
    },
  });
};

export const completedAcpToolStatus = (
  input: AcpToolInput,
  result: unknown,
  displayPathForSessionId: (sessionId: string, path: string) => string,
): string => {
  if (input.toolName.includes("readTextFile")) {
    const path = typeof input.rawInput === "object" && input.rawInput != null
      ? (input.rawInput as { path?: unknown }).path
      : undefined;
    const bytes = typeof result === "string" ? Buffer.byteLength(result) : undefined;
    return `Read ${typeof path === "string" ? displayPathForSessionId(input.sessionId, path) : "file"}${bytes == null ? "" : ` (${bytes} bytes)`}.`;
  }
  if (input.kind === "read" || input.kind === "search") {
    const count = Array.isArray(result) ? ` (${result.length} items)` : "";
    return `${input.title} complete${count}.`;
  }
  if (input.kind === "edit") {
    return `${input.title} complete.`;
  }
  if (input.toolName.includes("self.evaluate")) {
    const score = typeof result === "object" && result != null ? (result as { score?: unknown }).score : undefined;
    return `Self-evaluation complete${typeof score === "number" ? `: ${score.toFixed(2)}` : ""}.`;
  }
  return `${input.title} complete.`;
};

export const runAcpTool = async <T>(
  connection: AcpConnection,
  input: AcpToolInput,
  options: {
    displayPathForSessionId: (sessionId: string, path: string) => string;
    completedStatus?: (input: AcpToolInput, result: unknown) => string;
  },
): Promise<T> => {
  const toolCallId = `tool-${randomUUID()}`;
  await connection.sessionUpdate({
    sessionId: input.sessionId,
    update: {
      sessionUpdate: "tool_call",
      toolCallId,
      title: input.title,
      kind: input.kind,
      status: "pending",
      rawInput: input.rawInput,
      ...(input.locations == null ? {} : { locations: input.locations }),
    },
  });

  try {
    throwIfAcpAborted(input.signal);
    await connection.sessionUpdate({
      sessionId: input.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "in_progress",
      },
    });
    const result = await input.telemetry.measureToolCall({
      toolName: input.toolName,
      namespace: "bag.acp",
      descriptionVersion: "acp-v1",
      args: input.rawInput ?? {},
      fn: async () => {
        throwIfAcpAborted(input.signal);
        const result = await input.fn();
        throwIfAcpAborted(input.signal);
        return result;
      },
    });
    throwIfAcpAborted(input.signal);
    await connection.sessionUpdate({
      sessionId: input.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "completed",
        content: [
          markdownContent(
            options.completedStatus?.(input, result) ??
              completedAcpToolStatus(input, result, options.displayPathForSessionId),
          ),
        ],
        rawOutput: result,
      },
    });
    return result as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const cancelled = isAcpAbortError(error, input.signal);
    await connection.sessionUpdate({
      sessionId: input.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "failed",
        content: [markdownContent(cancelled ? `Cancelled ${input.title}.` : `Failed ${input.title}: ${message}`)],
        rawOutput: { error: message, outcome: acpFailureOutcomeFor({ cancelled, message }) },
      },
    });
    throw error;
  }
};
