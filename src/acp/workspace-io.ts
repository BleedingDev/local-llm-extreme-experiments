import type {
  AgentSideConnection as AcpConnection,
  ToolCallContent,
} from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import type { RunTelemetry } from "../telemetry";
import { resolveSessionPath, sessionRelativePath as safeSessionRelativePath } from "../workspace-paths";
import { acpFailureRawOutput } from "./permission-outcomes";
import { artifactLocation, markdownContent, sha256 } from "./surface";
import type { BagAcpSession } from "./session";
import { isAcpAbortError, throwIfAcpAborted, type AcpToolInput } from "./tool-runner";

export type AcpReadClientFileInput = {
  sessionId: string;
  telemetry: RunTelemetry;
  path: string;
  signal?: AbortSignal;
};

export type AcpWriteClientFileInput = {
  sessionId: string;
  telemetry: RunTelemetry;
  path: string;
  oldContent: string;
  newContent: string;
  reason: string;
  editStrategyId?: string;
  editStrategyFamily?: string;
  renderedEditContractVersion?: string;
  signal?: AbortSignal;
};

export type AcpWriteClientFileResult = {
  path: string;
  ok: boolean;
  reason: string;
  oldHash?: string;
  newHash?: string;
};

export type AcpWorkspaceIoDeps = {
  connection: AcpConnection;
  requireSession: (sessionId: string) => BagAcpSession;
  runAcpTool: <T>(input: AcpToolInput) => Promise<T>;
  editToolContent?: (input: AcpEditToolContentInput) => ToolCallContent[];
};

export type AcpEditToolContentInput = {
  session: BagAcpSession;
  path: string;
  oldContent: string;
  newContent: string;
  oldHash: string;
  newHash: string;
};

export const absoluteSessionPath = (session: BagAcpSession, path: string): string =>
  resolveSessionPath({
    cwd: session.cwd,
    additionalDirectories: session.additionalDirectories,
    path,
  });

export const sessionRelativePath = (session: BagAcpSession, path: string): string =>
  safeSessionRelativePath(session.cwd, session.additionalDirectories, path);

export const displayPathForSessionId = (
  sessions: ReadonlyMap<string, BagAcpSession>,
  sessionId: string,
  path: string,
): string => {
  const session = sessions.get(sessionId);
  return session == null ? path : sessionRelativePath(session, path);
};

export const editToolContent = (input: AcpEditToolContentInput): ToolCallContent[] => {
  if (input.session.clientCapabilities.richDiffContent) {
    return [
      {
        type: "diff",
        path: input.path,
        oldText: input.oldContent,
        newText: input.newContent,
      },
    ];
  }
  return [
    markdownContent(
      [
        `Proposed edit to ${sessionRelativePath(input.session, input.path)}.`,
        `Old hash: ${input.oldHash}`,
        `New hash: ${input.newHash}`,
      ].join("\n"),
    ),
  ];
};

export const readClientFile = async (
  deps: AcpWorkspaceIoDeps,
  input: AcpReadClientFileInput,
): Promise<string> => {
  const session = deps.requireSession(input.sessionId);
  const path = absoluteSessionPath(session, input.path);
  return deps.runAcpTool<string>({
    sessionId: input.sessionId,
    telemetry: input.telemetry,
    title: `Read ${sessionRelativePath(session, path)}`,
    toolName: "acp.fs.readTextFile",
    kind: "read",
    rawInput: { path },
    locations: [artifactLocation(path)],
    ...(input.signal === undefined ? {} : { signal: input.signal }),
    fn: async () => {
      if (!session.clientCapabilities.fsReadTextFile) {
        if (existsSync(path)) {
          return readFileSync(path, "utf8");
        }
        throw new Error("ACP client does not support fs/read_text_file and the file is not readable locally");
      }
      try {
        const response = await deps.connection.readTextFile({ sessionId: input.sessionId, path });
        return response.content;
      } catch (error) {
        if (existsSync(path)) {
          return readFileSync(path, "utf8");
        }
        throw error;
      }
    },
  });
};

export const writeClientFileWithPermission = async (
  deps: AcpWorkspaceIoDeps,
  input: AcpWriteClientFileInput,
): Promise<AcpWriteClientFileResult> => {
  const toolCallId = `tool-${randomUUID()}`;
  const oldHash = sha256(input.oldContent);
  const newHash = sha256(input.newContent);
  const session = deps.requireSession(input.sessionId);
  const path = absoluteSessionPath(session, input.path);
  const displayPath = sessionRelativePath(session, path);
  const content = (deps.editToolContent ?? editToolContent)({
    session,
    path,
    oldContent: input.oldContent,
    newContent: input.newContent,
    oldHash,
    newHash,
  });
  await deps.connection.sessionUpdate({
    sessionId: input.sessionId,
    update: {
      sessionUpdate: "tool_call",
      toolCallId,
      title: `Edit ${displayPath}`,
      kind: "edit",
      status: "pending",
      locations: [artifactLocation(path)],
      rawInput: {
        path,
        reason: input.reason,
        oldHash,
        newHash,
        editStrategyId: input.editStrategyId,
        editStrategyFamily: input.editStrategyFamily,
        renderedEditContractVersion: input.renderedEditContractVersion,
      },
      content,
    },
  });

  try {
    const result = await input.telemetry.measureToolCall({
      toolName: "acp.fs.writeTextFile",
      namespace: "bag.acp",
      descriptionVersion: "acp-coding-v1",
      args: {
        path,
        reason: input.reason,
        oldHash,
        newHash,
        editStrategyId: input.editStrategyId,
        editStrategyFamily: input.editStrategyFamily,
        renderedEditContractVersion: input.renderedEditContractVersion,
      },
      fn: async () => {
        throwIfAcpAborted(input.signal);
        if (!session.clientCapabilities.fsWriteTextFile) {
          throw new Error("ACP client does not support fs/write_text_file");
        }
        if (!session.yolo) {
          const permission = await deps.connection.requestPermission({
            sessionId: input.sessionId,
            toolCall: {
              toolCallId,
              title: `Edit ${displayPath}`,
              kind: "edit",
              status: "pending",
              locations: [artifactLocation(path)],
              rawInput: {
                path,
                reason: input.reason,
                oldHash,
                newHash,
                editStrategyId: input.editStrategyId,
                editStrategyFamily: input.editStrategyFamily,
                renderedEditContractVersion: input.renderedEditContractVersion,
              },
              content,
            },
            options: [
              { optionId: "allow", name: "Apply edit", kind: "allow_once" },
              { optionId: "reject", name: "Reject edit", kind: "reject_once" },
            ],
          });
          if (permission.outcome.outcome === "cancelled") {
            throw new Error("cancelled");
          }
          if (permission.outcome.optionId !== "allow") {
            throw new Error("edit permission rejected");
          }
        }
        throwIfAcpAborted(input.signal);
        await deps.connection.writeTextFile({
          sessionId: input.sessionId,
          path,
          content: input.newContent,
        });
        return { path, oldHash, newHash };
      },
    });
    await deps.connection.sessionUpdate({
      sessionId: input.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "completed",
        rawOutput: result,
        content: [
          ...content,
          markdownContent(`Applied edit to ${displayPath}.`),
        ],
      },
    });
    return { path, ok: true, reason: input.reason, oldHash, newHash };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const cancelled = isAcpAbortError(error, input.signal);
    await deps.connection.sessionUpdate({
      sessionId: input.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "failed",
        rawOutput: acpFailureRawOutput({ error: message, path, oldHash, newHash }, { cancelled, message }),
        content: [markdownContent(cancelled ? `Edit cancelled for ${displayPath}.` : `Edit was not applied to ${displayPath}: ${message}`)],
      },
    });
    if (cancelled) {
      throw error;
    }
    return { path, ok: false, reason: message, oldHash, newHash };
  }
};
