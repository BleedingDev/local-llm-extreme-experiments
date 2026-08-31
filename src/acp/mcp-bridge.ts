import type {
  AgentSideConnection as AcpConnection,
  ToolKind,
} from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import {
  createMcpRuntimeToolBridge,
  mcpRuntimeToolResultToOptimizerFeedback,
  normalizeMcpServerTools,
  prepareMcpRenderedToolContracts,
  type McpRuntimePermissionDecision,
  type McpRuntimeToolCall,
  type McpRuntimeToolExecutor,
  type McpRuntimeToolResult,
  type McpServerMetadata,
} from "../mcp/runtime-tools";
import type {
  AssistantWithToolCalls,
  ChatWithToolsResult,
  LlmRouter,
  ToolDefinition,
  ToolResultMessage,
} from "../llm";
import type { RunTelemetry } from "../telemetry";
import type { BagAcpSession } from "./session";
import { markdownContent } from "./surface";

export type AcpMcpBridgeDeps = {
  connection: AcpConnection;
};

export type AcpLiveMcpToolCallInput = {
  session: BagAcpSession;
  telemetry: RunTelemetry;
  server: McpServerMetadata;
  call: McpRuntimeToolCall;
  executor: McpRuntimeToolExecutor;
  signal?: AbortSignal;
};

export type AcpMcpToolUseRuntimeServer = {
  server: McpServerMetadata;
  executor: McpRuntimeToolExecutor;
};

export type AttachLiveMcpToolsInput = {
  session: BagAcpSession;
  telemetry: RunTelemetry;
  router: LlmRouter;
  servers: readonly AcpMcpToolUseRuntimeServer[];
  signal?: AbortSignal;
  maxMcpToolTurns?: number;
};

type LiveMcpCallable = {
  server: McpServerMetadata;
  executor: McpRuntimeToolExecutor;
  toolName: string;
  definition: ToolDefinition;
};

export const runLiveMcpToolCall = async (
  deps: AcpMcpBridgeDeps,
  input: AcpLiveMcpToolCallInput,
): Promise<McpRuntimeToolResult> => {
  const normalizedTools = normalizeMcpServerTools(input.server, {
    canonicalToolVersion: input.session.optimizerPin.telemetry.canonicalToolVersion,
  });
  const prepared = prepareMcpRenderedToolContracts({
    normalizedTools,
    resolvedPolicy: input.session.optimizerPin.resolvedPolicy,
  });
  const bridge = createMcpRuntimeToolBridge({
    normalizedTools,
    renderedContracts: prepared.renderedContracts,
    executor: input.executor,
    mode: input.session.yolo ? "yolo" : "safe",
    maxRetryCount: 2,
    timeoutMs: 30_000,
    permissionHandler: async (request): Promise<McpRuntimePermissionDecision> => {
      if (input.session.yolo) {
        return "allow";
      }
      const permission = await deps.connection.requestPermission({
        sessionId: input.session.id,
        toolCall: {
          toolCallId: request.callId,
          title: `MCP ${request.serverName}.${request.toolName}`,
          kind: request.policy.risks.runsProcess ? "execute" : request.policy.risks.writesWorkspace ? "edit" : "read",
          status: "pending",
          rawInput: {
            serverId: request.serverId,
            serverName: request.serverName,
            toolName: request.toolName,
            modelFacingToolId: request.modelFacingToolId,
            modelFacingToolName: request.modelFacingToolName,
            renderedToolId: request.renderedToolId,
            renderedToolName: request.renderedToolName,
            sideEffectLevel: request.policy.sideEffectLevel,
            risks: request.policy.risks,
            arguments: request.arguments,
          },
        },
        options: [
          { optionId: "allow", name: "Run MCP tool", kind: "allow_once" },
          { optionId: "reject", name: "Reject MCP tool", kind: "reject_once" },
        ],
      });
      return permission.outcome.outcome === "selected" && permission.outcome.optionId === "allow" ? "allow" : "deny";
    },
  });
  const toolCallId = input.call.callId ?? `mcp-tool-${randomUUID()}`;
  const callable = bridge.callableTools.find((tool) =>
    tool.toolName === input.call.toolName ||
    tool.modelFacingToolName === input.call.modelFacingToolName ||
    tool.modelFacingToolId === input.call.modelFacingToolId ||
    tool.canonicalToolId === input.call.canonicalToolId
  );
  const kind: ToolKind = callable?.policy.risks.runsProcess === true
    ? "execute"
    : callable?.policy.risks.writesWorkspace === true
      ? "edit"
      : "read";
  await deps.connection.sessionUpdate({
    sessionId: input.session.id,
    update: {
      sessionUpdate: "tool_call",
      toolCallId,
      title: `MCP ${input.call.toolName ?? input.call.modelFacingToolName ?? "tool"}`,
      kind,
      status: "pending",
      rawInput: {
        ...input.call,
        modelFacingToolCount: prepared.modelFacingContracts.length,
      },
    },
  });

  const result = await bridge.executeToolCall({
    ...input.call,
    callId: toolCallId,
    ...(input.signal === undefined ? {} : { signal: input.signal }),
  });
  const completedAt = new Date().toISOString();
  const startedAt = new Date(Date.now() - result.metrics.durationMs).toISOString();
  input.telemetry.recordToolCall({
    toolName: result.call.modelFacingToolName ?? result.call.toolName ?? input.call.toolName ?? "unknown_mcp_tool",
    namespace: "mcp",
    descriptionVersion: result.call.renderedToolVersion ?? result.call.canonicalToolVersion,
    startedAt,
    completedAt,
    durationMs: result.metrics.durationMs,
    ok: result.ok,
    retryCount: result.metrics.retryCount,
    argumentBytes: result.metrics.argumentBytes,
    argumentHash: result.argumentShapeHash,
    resultBytes: result.metrics.resultBytes,
    resultKind: result.result === undefined ? "empty" : "json",
    ...(result.ok ? {} : { error: result.error?.message ?? result.failureCode ?? result.status }),
    ...(result.error?.class === undefined ? {} : { errorName: result.error.class }),
  });
  const feedback = mcpRuntimeToolResultToOptimizerFeedback(result, {
    traceId: `trace.${input.session.id}`,
    spanId: `span.${toolCallId}`,
  });
  input.telemetry.event("acp.mcp.tool_call", {
    trace: result.trace as unknown as Record<string, unknown>,
    feedback,
  });
  await deps.connection.sessionUpdate({
    sessionId: input.session.id,
    update: {
      sessionUpdate: "tool_call_update",
      toolCallId,
      status: result.ok ? "completed" : "failed",
      rawOutput: {
        status: result.status,
        failureCode: result.failureCode,
        error: result.error,
        result: result.result,
        trace: result.trace,
        feedback,
      },
      content: [markdownContent(result.ok
        ? `MCP tool completed: ${result.call.modelFacingToolName ?? result.call.toolName ?? toolCallId}.`
        : `MCP tool failed: ${result.failureCode ?? result.status}.`)],
    },
  });
  return result;
};

const parseModelToolArguments = (argumentsJson: string): unknown => {
  try {
    return argumentsJson.trim().length === 0 ? {} : JSON.parse(argumentsJson);
  } catch {
    return argumentsJson;
  }
};

const mcpToolObservation = (result: McpRuntimeToolResult): string =>
  JSON.stringify({
    ok: result.ok,
    status: result.status,
    failureCode: result.failureCode,
    followUpBehavior: result.followUpBehavior,
    result: result.result,
    error: result.error,
    trace: {
      callId: result.callId,
      modelFacingToolName: result.call.modelFacingToolName,
      renderedToolName: result.call.renderedToolName,
      canonicalToolId: result.call.canonicalToolId,
      permissionStatus: result.trace.permissionStatus,
      sideEffectLevel: result.trace.sideEffectLevel,
      retryCount: result.trace.retryCount,
      resultTruncated: result.trace.resultTruncated,
    },
  });

const buildLiveMcpCallables = (input: AttachLiveMcpToolsInput): LiveMcpCallable[] =>
  input.servers.flatMap(({ server, executor }) => {
    const normalizedTools = normalizeMcpServerTools(server, {
      canonicalToolVersion: input.session.optimizerPin.telemetry.canonicalToolVersion,
    });
    const prepared = prepareMcpRenderedToolContracts({
      normalizedTools,
      resolvedPolicy: input.session.optimizerPin.resolvedPolicy,
    });
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools,
      renderedContracts: prepared.renderedContracts,
      executor,
      mode: input.session.yolo ? "yolo" : "safe",
    });
    return bridge.callableTools.map((tool): LiveMcpCallable => ({
      server,
      executor,
      toolName: tool.modelFacingToolName,
      definition: {
        type: "function",
        function: {
          name: tool.modelFacingToolName,
          description: tool.description,
          parameters: tool.inputSchema,
        },
      },
    }));
  });

export const attachLiveMcpToolsToRouter = (
  deps: AcpMcpBridgeDeps,
  input: AttachLiveMcpToolsInput,
): LlmRouter => {
  if (input.servers.length === 0) {
    return input.router;
  }
  const callables = buildLiveMcpCallables(input);
  if (callables.length === 0) {
    return input.router;
  }
  const callableByName = new Map(callables.map((callable) => [callable.toolName, callable] as const));
  const maxMcpToolTurns = Math.max(1, input.maxMcpToolTurns ?? 8);

  return {
    ...input.router,
    chatTextWithTools: async (options): Promise<ChatWithToolsResult> => {
      const tools = [...options.tools, ...callables.map((callable) => callable.definition)];
      const appendMcpResults = async (response: ChatWithToolsResult, mcpToolCalls: ChatWithToolsResult["toolCalls"]) => {
        const assistantMessage: AssistantWithToolCalls = {
          role: "assistant",
          content: response.textContent.length > 0 ? response.textContent : null,
          tool_calls: mcpToolCalls.map((toolCall) => ({
            id: toolCall.id,
            type: "function",
            function: { name: toolCall.name, arguments: toolCall.argumentsJson },
          })),
        };
        options.messages.push(assistantMessage);

        for (const toolCall of mcpToolCalls) {
          const callable = callableByName.get(toolCall.name);
          if (callable == null) {
            continue;
          }
          const result = await runLiveMcpToolCall(deps, {
            session: input.session,
            telemetry: input.telemetry,
            server: callable.server,
            call: {
              callId: toolCall.id,
              modelFacingToolName: toolCall.name,
              arguments: parseModelToolArguments(toolCall.argumentsJson),
              ...(input.signal === undefined ? {} : { signal: input.signal }),
            },
            executor: callable.executor,
            ...(input.signal === undefined ? {} : { signal: input.signal }),
          });
          const toolResult: ToolResultMessage = {
            role: "tool",
            tool_call_id: toolCall.id,
            content: mcpToolObservation(result),
          };
          options.messages.push(toolResult);
        }
      };

      for (let mcpTurn = 0; mcpTurn < maxMcpToolTurns; mcpTurn += 1) {
        const response = await input.router.chatTextWithTools({ ...options, tools });
        const mcpToolCalls = response.toolCalls.filter((toolCall) => callableByName.has(toolCall.name));
        if (mcpToolCalls.length === 0) {
          return response;
        }
        if (mcpToolCalls.length !== response.toolCalls.length) {
          await appendMcpResults(response, mcpToolCalls);
          return {
            ...response,
            toolCalls: response.toolCalls.filter((toolCall) => !callableByName.has(toolCall.name)),
          };
        }
        await appendMcpResults(response, mcpToolCalls);
      }
      return {
        finishReason: "stop",
        textContent: "MCP tool-call budget exhausted. Use a narrower MCP request or continue with the regular coding tools.",
        toolCalls: [],
      };
    },
  };
};
