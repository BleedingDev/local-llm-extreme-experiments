import { createHash } from "node:crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { ResolvedOptimizerPolicy } from "../optimizer/policy-resolver";
import { renderToolContracts } from "../optimizer/tool-renderer";
import type {
  CanonicalToolSpec,
  JsonValue,
  RenderedToolContract,
  ToolResultStyle,
} from "../optimizer/types";

type JsonObject = { [key: string]: JsonValue };
type McpSideEffectLevel = CanonicalToolSpec["sideEffectLevel"];

export type McpPolicyAction = "allow" | "confirm" | "block";

export interface McpToolAnnotations {
  title?: string;
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
}

export interface McpToolExample {
  name?: string;
  input: Record<string, unknown>;
  output?: unknown;
}

export interface McpToolMetadata {
  name: string;
  title?: string;
  description?: string;
  inputSchema?: unknown;
  outputSchema?: unknown;
  annotations?: McpToolAnnotations;
  examples?: readonly McpToolExample[];
  resultStyle?: ToolResultStyle;
}

export interface McpServerMetadata {
  serverId?: string;
  name: string;
  displayName?: string;
  tools: readonly McpToolMetadata[];
}

export interface McpToolPolicy {
  sideEffectLevel: McpSideEffectLevel;
  requiresConfirmation: boolean;
  risks: {
    writesWorkspace: boolean;
    usesNetwork: boolean;
    runsProcess: boolean;
  };
  safeAction: McpPolicyAction;
  yoloAction: McpPolicyAction;
  argumentMaxBytes: number;
  resultMaxBytes: number;
  reasons: string[];
}

export interface NormalizedMcpTool {
  serverId: string;
  serverName: string;
  toolName: string;
  canonicalSpec: CanonicalToolSpec;
  policy: McpToolPolicy;
}

export interface NormalizeMcpToolsOptions {
  canonicalToolVersion?: string;
  namespacePrefix?: string;
}

export interface McpRuntimeToolLineage {
  canonicalToolId: string;
  canonicalToolVersion: string;
  modelFacingToolId: string;
  modelFacingToolName: string;
  renderedToolId?: string;
  renderedToolName?: string;
  renderedToolVersion?: string;
  modelProfileId?: string;
  policyId?: string;
  renderer?: string;
  rendererVersion?: string;
  resultStyleVersion?: string;
}

export interface McpRuntimeResultExample {
  name: string;
  resultStyle: ToolResultStyle;
  value: JsonValue;
}

export interface McpRuntimeResultContract {
  resultStyle: ToolResultStyle;
  resultStyleVersion?: string;
  resultMaxBytes: number;
  truncationMessage: string;
  errorResultStyle: "structured_error";
  resultExamples: McpRuntimeResultExample[];
}

export interface McpModelFacingToolContract {
  kind: "mcp.model_facing_tool_contract";
  modelFacingToolId: string;
  modelFacingToolName: string;
  title: string;
  description: string;
  inputSchema: JsonObject;
  resultStyle: ToolResultStyle;
  promptFragments: string[];
  examples: RenderedToolContract["examples"];
  resultContract: McpRuntimeResultContract;
  lineage: McpRuntimeToolLineage;
  policy: Pick<McpToolPolicy, "sideEffectLevel" | "requiresConfirmation" | "risks" | "safeAction" | "yoloAction" | "reasons">;
  serverId: string;
  serverName: string;
  toolName: string;
  canonicalSpec: CanonicalToolSpec;
  renderedContract?: RenderedToolContract;
}

export interface McpRenderedContractPreparation {
  canonicalToolSpecs: CanonicalToolSpec[];
  renderedContracts: RenderedToolContract[];
  modelFacingContracts: McpModelFacingToolContract[];
  modelFacingContractsById: Record<string, McpModelFacingToolContract>;
  modelFacingContractsByName: Record<string, McpModelFacingToolContract>;
  policiesByCanonicalToolId: Record<string, McpToolPolicy>;
  resultBudgetsByCanonicalToolId: Record<string, {
    resultMaxBytes: number;
    truncationMessage: string;
    errorResultStyle: "structured_error";
  }>;
}

export interface PrepareMcpRenderedContractsInput {
  normalizedTools: readonly NormalizedMcpTool[];
  resolvedPolicy: ResolvedOptimizerPolicy;
}

export type McpRuntimeExecutionMode = "safe" | "yolo";
export type McpRuntimePermissionDecision = "allow" | "deny";
export type McpRuntimePermissionStatus = "not_required" | "requested" | "granted" | "denied" | "blocked";
export type McpRuntimeToolStatus =
  | "success"
  | "error"
  | "invalid_arguments"
  | "denied"
  | "cancelled"
  | "timeout"
  | "retry_exhausted"
  | "missing_server"
  | "unknown_tool";
export type McpRuntimeFailureCode =
  | "malformed_arguments"
  | "schema_mismatch"
  | "missing_server"
  | "unknown_tool"
  | "permission_denied"
  | "policy_blocked"
  | "permission_error"
  | "timeout"
  | "cancelled"
  | "oversized_output"
  | "stale_context"
  | "retry_exhausted"
  | "runtime_exception";
export type McpRuntimeRedactionStatus = "raw_local_only" | "redacted" | "hash_only" | "omitted" | "needs_review";
export type McpRuntimeFollowUpBehavior =
  | "none"
  | "repair_arguments"
  | "refresh_tool_inventory"
  | "request_permission_or_choose_lower_risk_tool"
  | "narrow_request_or_paginate"
  | "retry_with_narrower_scope_or_abort"
  | "inspect_executor_or_choose_alternate_tool";

export interface McpRuntimeCallableTool {
  canonicalToolId: string;
  canonicalToolVersion: string;
  modelFacingToolId: string;
  modelFacingToolName: string;
  renderedToolVersion?: string;
  renderedToolId?: string;
  renderedToolName?: string;
  modelProfileId?: string;
  policyId?: string;
  renderer?: string;
  rendererVersion?: string;
  resultStyleVersion?: string;
  name: string;
  title: string;
  description: string;
  inputSchema: JsonObject;
  resultStyle: ToolResultStyle;
  promptFragments: string[];
  examples: RenderedToolContract["examples"];
  resultContract: McpRuntimeResultContract;
  lineage: McpRuntimeToolLineage;
  serverId: string;
  serverName: string;
  toolName: string;
  policy: McpToolPolicy;
}

export interface McpRuntimePolicyDecision {
  mode: McpRuntimeExecutionMode;
  action: McpPolicyAction;
  permissionStatus: McpRuntimePermissionStatus;
  sideEffectLevel: McpSideEffectLevel;
  requiresConfirmation: boolean;
  reasons: string[];
}

export interface McpRuntimeToolExecutionRequest {
  callId: string;
  serverId: string;
  serverName: string;
  toolName: string;
  canonicalToolId: string;
  modelFacingToolId: string;
  modelFacingToolName: string;
  renderedToolId?: string;
  renderedToolName?: string;
  arguments: JsonObject;
  retryCount?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export type McpRuntimeToolExecutor = (request: McpRuntimeToolExecutionRequest) => Promise<unknown> | unknown;

export interface McpRuntimePermissionRequest {
  callId: string;
  serverId: string;
  serverName: string;
  toolName: string;
  canonicalToolId: string;
  modelFacingToolId: string;
  modelFacingToolName: string;
  renderedToolId?: string;
  renderedToolName?: string;
  policy: McpToolPolicy;
  arguments: JsonObject;
  mode: McpRuntimeExecutionMode;
}

export type McpRuntimePermissionHandler =
  (request: McpRuntimePermissionRequest) => Promise<McpRuntimePermissionDecision> | McpRuntimePermissionDecision;

export interface McpRuntimeToolCall {
  callId?: string;
  serverId?: string;
  serverName?: string;
  toolName?: string;
  canonicalToolId?: string;
  modelFacingToolId?: string;
  modelFacingToolName?: string;
  renderedToolId?: string;
  arguments: unknown;
  retryCount?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface McpRuntimeToolMetrics {
  argumentBytes: number;
  resultBytes: number;
  resultBytesBeforeBounding: number;
  omittedResultBytes: number;
  truncated: boolean;
  retryCount: number;
  durationMs: number;
}

export interface McpRuntimeToolTrace {
  event: "mcp.tool_call";
  spanName: "mcp.tool_call";
  callId: string;
  status: McpRuntimeToolStatus;
  canonicalToolId?: string;
  canonicalToolVersion?: string;
  modelFacingToolId?: string;
  modelFacingToolName?: string;
  renderedToolVersion?: string;
  renderedToolId?: string;
  renderedToolName?: string;
  modelProfileId?: string;
  policyId?: string;
  renderer?: string;
  rendererVersion?: string;
  resultStyleVersion?: string;
  serverId?: string;
  serverName?: string;
  toolName?: string;
  policyAction?: McpPolicyAction;
  permissionStatus?: McpRuntimePermissionStatus;
  sideEffectLevel?: McpSideEffectLevel;
  durationMs: number;
  argumentBytes: number;
  argumentShapeHash: string;
  redactionStatus: McpRuntimeRedactionStatus;
  resultBytes: number;
  resultBytesBeforeBounding: number;
  omittedResultBytes: number;
  resultTruncated: boolean;
  retryCount: number;
  failureCode?: McpRuntimeFailureCode;
  errorClass?: string;
  followUpBehavior: McpRuntimeFollowUpBehavior;
}

export interface McpRuntimeToolResult {
  ok: boolean;
  status: McpRuntimeToolStatus;
  callId: string;
  call: {
    canonicalToolId?: string;
    canonicalToolVersion?: string;
    modelFacingToolId?: string;
    modelFacingToolName?: string;
    renderedToolVersion?: string;
    renderedToolId?: string;
    renderedToolName?: string;
    modelProfileId?: string;
    policyId?: string;
    renderer?: string;
    rendererVersion?: string;
    resultStyleVersion?: string;
    serverId?: string;
    serverName?: string;
    toolName?: string;
  };
  policyDecision?: McpRuntimePolicyDecision;
  metrics: McpRuntimeToolMetrics;
  argumentShapeHash: string;
  redactionStatus: McpRuntimeRedactionStatus;
  followUpBehavior: McpRuntimeFollowUpBehavior;
  failureCode?: McpRuntimeFailureCode;
  result?: JsonValue;
  error?: {
    class: string;
    code: McpRuntimeFailureCode;
    message: string;
    details?: JsonValue;
  };
  trace: McpRuntimeToolTrace;
}

export type McpRuntimeOptimizerFeedbackSeverity = "info" | "warning" | "failure" | "critical";

export interface McpRuntimeOptimizerFeedbackLineage {
  canonicalToolIds: string[];
  modelFacingToolIds: string[];
  modelFacingToolNames: string[];
  renderedToolContractIds: string[];
  canonicalToolVersions: string[];
  renderedToolVersions: string[];
  resultStyleVersions: string[];
  modelProfileIds: string[];
  policyIds: string[];
}

export interface McpRuntimeOptimizerFeedbackRecord {
  feedbackId: string;
  source: "mcp_runtime_tool_call";
  severity: McpRuntimeOptimizerFeedbackSeverity;
  objective: string;
  feedback: string;
  callId: string;
  status: McpRuntimeToolStatus;
  traceIds: string[];
  spanIds: string[];
  canonicalToolId?: string;
  modelFacingToolId?: string;
  modelFacingToolName?: string;
  renderedToolId?: string;
  renderedToolName?: string;
  serverId?: string;
  serverName?: string;
  toolName?: string;
  policyAction?: McpPolicyAction;
  permissionStatus?: McpRuntimePermissionStatus;
  sideEffectLevel?: McpSideEffectLevel;
  durationMs: number;
  argumentBytes: number;
  argumentShapeHash: string;
  redactionStatus: McpRuntimeRedactionStatus;
  resultBytes: number;
  resultBytesBeforeBounding: number;
  omittedResultBytes: number;
  resultTruncated: boolean;
  retryCount: number;
  failureCode?: McpRuntimeFailureCode;
  errorClass?: string;
  followUpBehavior: McpRuntimeFollowUpBehavior;
  lineage: McpRuntimeOptimizerFeedbackLineage;
  redacted: false;
  truncated: boolean;
}

export interface McpRuntimeOptimizerFeedbackOptions {
  traceId?: string;
  spanId?: string;
  objective?: string;
  maxFeedbackChars?: number;
  includeSuccessful?: boolean;
}

export interface CreateMcpRuntimeToolBridgeInput {
  normalizedTools: readonly NormalizedMcpTool[];
  renderedContracts?: readonly RenderedToolContract[];
  executor: McpRuntimeToolExecutor;
  mode?: McpRuntimeExecutionMode;
  permissionHandler?: McpRuntimePermissionHandler;
  timeoutMs?: number;
  maxRetryCount?: number;
  now?: () => number;
  createCallId?: () => string;
}

export interface McpRuntimeToolBridge {
  callableTools: McpRuntimeCallableTool[];
  executeToolCall: (call: McpRuntimeToolCall) => Promise<McpRuntimeToolResult>;
}

export interface McpStdioTransportInput {
  serverId?: string;
  name: string;
  displayName?: string;
  command: string;
  args?: readonly string[];
  cwd?: string;
  env?: Record<string, string | undefined>;
  startupTimeoutMs?: number;
  requestTimeoutMs?: number;
  signal?: AbortSignal;
}

export interface McpStdioRuntimeServer {
  server: McpServerMetadata;
  executor: McpRuntimeToolExecutor;
  close: () => Promise<void>;
  process: ChildProcessWithoutNullStreams;
}

type JsonRpcMessage = {
  jsonrpc?: "2.0";
  id?: string | number | null;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: {
    code?: number;
    message?: string;
    data?: unknown;
  };
};

type JsonRpcPending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

const DEFAULT_CANONICAL_TOOL_VERSION = "canonical-tools.mcp.v1";
const DEFAULT_NAMESPACE_PREFIX = "mcp";
const DEFAULT_ARGUMENT_MAX_BYTES = 32 * 1024;
const DEFAULT_RESULT_MAX_BYTES = 64 * 1024;
const WRITE_RESULT_MAX_BYTES = 32 * 1024;
const RISKY_RESULT_MAX_BYTES = 48 * 1024;
const DEFAULT_FEEDBACK_MAX_CHARS = 1_200;
const MAX_FEEDBACK_CHARS = 8_000;
const MODEL_FACING_TOOL_NAME_MAX_LENGTH = 64;
const MAX_TOOL_TIMEOUT_MS = 10 * 60 * 1000;
const DEFAULT_TRACE_REDACTION_STATUS: McpRuntimeRedactionStatus = "hash_only";

const PROCESS_PATTERN = /\b(shell|command|cmd|exec|spawn|terminal|process|subprocess|script|run)\b/i;
const NETWORK_PATTERN = /\b(http|https|url|uri|fetch|request|download|upload|web|browser|api|network|search)\b/i;
const WRITE_PATTERN = /\b(write|create|update|delete|remove|patch|apply|save|mutate|insert|modify|move|rename)\b/i;
const READ_PATTERN = /\b(read|get|list|find|inspect|lookup|query|show|describe)\b/i;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const toJsonValue = (value: unknown): JsonValue | undefined => {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return Number.isNaN(value) ? undefined : value;
  }

  if (Array.isArray(value)) {
    const entries = value
      .map((entry) => toJsonValue(entry))
      .filter((entry): entry is JsonValue => entry !== undefined);
    return entries;
  }

  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .map(([key, entry]) => [key, toJsonValue(entry)] as const)
        .filter((entry): entry is readonly [string, JsonValue] => entry[1] !== undefined)
        .sort(([left], [right]) => left.localeCompare(right)),
    );
  }

  return undefined;
};

const toJsonObject = (value: unknown, fallback: JsonObject): JsonObject => {
  const jsonValue = toJsonValue(value);
  return isRecord(jsonValue) ? jsonValue as JsonObject : fallback;
};

const stableJson = (value: unknown): string => JSON.stringify(toJsonValue(value) ?? null);

const shortHash = (value: unknown): string =>
  createHash("sha256").update(stableJson(value)).digest("hex").slice(0, 12);

const argumentShapeValue = (value: unknown): JsonValue => {
  if (Array.isArray(value)) {
    return value.map((entry) => argumentShapeValue(entry));
  }

  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .map(([key, entry]) => [key, argumentShapeValue(entry)] as const)
        .sort(([left], [right]) => left.localeCompare(right)),
    );
  }

  if (value === null) {
    return "null";
  }

  return typeof value;
};

const argumentShapeHashFor = (value: unknown): string => `sha256:${shortHash(argumentShapeValue(value))}`;

const uniqueSorted = (values: readonly (string | undefined)[]): string[] =>
  [...new Set(values.filter((value): value is string => value !== undefined && value.length > 0))]
    .sort((left, right) => left.localeCompare(right));

const boundedInteger = (value: number | undefined, fallback: number, min: number, max: number): number => {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};

const boundedOptionalInteger = (value: number | undefined, min: number, max: number): number | undefined => {
  if (value === undefined || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
};

const sanitizeIdentifier = (value: string, fallback: string): string => {
  const sanitized = value
    .trim()
    .replace(/[^A-Za-z0-9._:-]+/g, "_")
    .replace(/^[^A-Za-z0-9]+/, "")
    .replace(/_+/g, "_")
    .replace(/_$/g, "");

  return sanitized.length > 0 ? sanitized : fallback;
};

const titleFromName = (name: string): string =>
  sanitizeIdentifier(name, "tool")
    .replace(/[._:-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const sanitizeModelFacingName = (value: string): string => {
  const sanitized = value
    .trim()
    .replace(/[^A-Za-z0-9_-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+/, "")
    .replace(/_+$/, "");

  return sanitized.length > 0 ? sanitized : "mcp_tool";
};

const modelFacingToolNameFor = (rawName: string, canonicalToolId: string): string => {
  const suffix = shortHash({ canonicalToolId });
  const prefixMaxLength = MODEL_FACING_TOOL_NAME_MAX_LENGTH - suffix.length - 1;
  const sanitized = sanitizeModelFacingName(rawName);
  const prefix = sanitizeModelFacingName(sanitized.slice(0, Math.max(1, prefixMaxLength)));

  return `${prefix}_${suffix}`;
};

const modelFacingToolIdFor = (input: {
  canonicalToolId: string;
  canonicalToolVersion: string;
  renderedToolId?: string;
  renderedToolVersion?: string;
  modelFacingToolName: string;
  resultStyle: ToolResultStyle;
  resultStyleVersion?: string;
  resultMaxBytes: number;
}): string =>
  `mcp.model_facing_tool.${shortHash({
    canonicalToolId: input.canonicalToolId,
    canonicalToolVersion: input.canonicalToolVersion,
    renderedToolId: input.renderedToolId,
    renderedToolVersion: input.renderedToolVersion,
    modelFacingToolName: input.modelFacingToolName,
    resultStyle: input.resultStyle,
    resultStyleVersion: input.resultStyleVersion,
    resultMaxBytes: input.resultMaxBytes,
  })}`;

const defaultInputSchema = (): JsonObject => ({
  type: "object",
  properties: {},
});

const defaultOutputSchema = (): JsonObject => ({
  type: "object",
  properties: {
    result: { type: "string" },
  },
});

const matchTextFor = (tool: Pick<McpToolMetadata, "name" | "description" | "title">): string =>
  [tool.name, tool.title, tool.description].filter((value): value is string => value !== undefined).join(" ");

const strongerSideEffect = (left: McpSideEffectLevel, right: McpSideEffectLevel): McpSideEffectLevel => {
  const rank: Record<McpSideEffectLevel, number> = {
    none: 0,
    read: 1,
    write: 2,
    network: 3,
    process: 4,
  };
  return rank[right] > rank[left] ? right : left;
};

const safeActionFor = (sideEffectLevel: McpSideEffectLevel): McpPolicyAction =>
  sideEffectLevel === "none" || sideEffectLevel === "read" ? "allow" : "confirm";

const yoloActionFor = (sideEffectLevel: McpSideEffectLevel): McpPolicyAction =>
  sideEffectLevel === "process" ? "confirm" : "allow";

const resultBudgetFor = (sideEffectLevel: McpSideEffectLevel): number => {
  switch (sideEffectLevel) {
    case "write":
      return WRITE_RESULT_MAX_BYTES;
    case "network":
    case "process":
      return RISKY_RESULT_MAX_BYTES;
    case "none":
    case "read":
      return DEFAULT_RESULT_MAX_BYTES;
  }
};

export const classifyMcpToolPolicy = (tool: McpToolMetadata): McpToolPolicy => {
  const text = matchTextFor(tool);
  const reasons: string[] = [];
  let sideEffectLevel: McpSideEffectLevel = "read";
  const hasWriteRisk = tool.annotations?.destructiveHint === true || WRITE_PATTERN.test(text);
  const hasNetworkRisk = tool.annotations?.openWorldHint === true || NETWORK_PATTERN.test(text);
  const hasProcessRisk = PROCESS_PATTERN.test(text);

  if (tool.annotations?.readOnlyHint === true) {
    sideEffectLevel = "read";
    reasons.push("annotation:readOnlyHint");
  }

  if (READ_PATTERN.test(text)) {
    reasons.push("name_or_description:read");
  }

  if (hasWriteRisk) {
    sideEffectLevel = strongerSideEffect(sideEffectLevel, "write");
    reasons.push(tool.annotations?.destructiveHint === true ? "annotation:destructiveHint" : "name_or_description:write");
  }

  if (hasNetworkRisk) {
    sideEffectLevel = strongerSideEffect(sideEffectLevel, "network");
    reasons.push(tool.annotations?.openWorldHint === true ? "annotation:openWorldHint" : "name_or_description:network");
  }

  if (hasProcessRisk) {
    sideEffectLevel = strongerSideEffect(sideEffectLevel, "process");
    reasons.push("name_or_description:process");
  }

  if (reasons.length === 0) {
    reasons.push("default:read");
  }

  const risks = {
    writesWorkspace: hasWriteRisk,
    usesNetwork: hasNetworkRisk,
    runsProcess: hasProcessRisk,
  };
  const requiresConfirmation = sideEffectLevel === "write" || sideEffectLevel === "network" || sideEffectLevel === "process";

  return {
    sideEffectLevel,
    requiresConfirmation,
    risks,
    safeAction: safeActionFor(sideEffectLevel),
    yoloAction: yoloActionFor(sideEffectLevel),
    argumentMaxBytes: DEFAULT_ARGUMENT_MAX_BYTES,
    resultMaxBytes: resultBudgetFor(sideEffectLevel),
    reasons,
  };
};

export const normalizeMcpToolToCanonicalSpec = (
  server: Omit<McpServerMetadata, "tools">,
  tool: McpToolMetadata,
  options: NormalizeMcpToolsOptions = {},
): NormalizedMcpTool => {
  const serverId = sanitizeIdentifier(server.serverId ?? server.name, "server");
  const namespace = `${sanitizeIdentifier(options.namespacePrefix ?? DEFAULT_NAMESPACE_PREFIX, "mcp")}.${serverId}`;
  const name = sanitizeIdentifier(tool.name, "tool");
  const policy = classifyMcpToolPolicy(tool);
  const inputSchema = toJsonObject(tool.inputSchema, defaultInputSchema());
  const outputSchema = toJsonObject(tool.outputSchema, defaultOutputSchema());
  const description = tool.description?.trim() || `MCP tool ${tool.name} from ${server.displayName ?? server.name}.`;
  const title = tool.title?.trim() || tool.annotations?.title?.trim() || titleFromName(tool.name);
  const canonicalToolVersion = options.canonicalToolVersion ?? DEFAULT_CANONICAL_TOOL_VERSION;
  const examples = (tool.examples ?? []).map((example, index) => ({
    name: example.name?.trim() || `example ${index + 1}`,
    input: toJsonObject(example.input, {}),
    ...(example.output === undefined ? {} : { output: toJsonValue(example.output) ?? null }),
  }));

  return {
    serverId,
    serverName: server.name,
    toolName: tool.name,
    canonicalSpec: {
      canonicalToolId: [
        "tool",
        namespace,
        name,
        shortHash({
          serverId,
          toolName: tool.name,
          inputSchema,
          outputSchema,
        }),
      ].join("."),
      canonicalToolVersion,
      namespace,
      name,
      title,
      description,
      inputSchema,
      outputSchema,
      resultStyle: tool.resultStyle ?? "json",
      sideEffectLevel: policy.sideEffectLevel,
      requiresConfirmation: policy.requiresConfirmation,
      examples,
    },
    policy,
  };
};

export const normalizeMcpServerTools = (
  server: McpServerMetadata,
  options: NormalizeMcpToolsOptions = {},
): NormalizedMcpTool[] =>
  server.tools.map((tool) => normalizeMcpToolToCanonicalSpec(server, tool, options));

const truncationMessageFor = (policy: McpToolPolicy): string =>
  `If the MCP result exceeds ${policy.resultMaxBytes} bytes, the runtime returns a bounded JSON result with mcpResultTruncated=true, originalBytes, omittedBytes, and preview; treat preview as incomplete and call a narrower tool request if more detail is needed.`;

const renderedNameFallbackFor = (spec: CanonicalToolSpec): string => `${spec.namespace}_${spec.name}`;

const renderedExamplesFallbackFor = (spec: CanonicalToolSpec): RenderedToolContract["examples"] =>
  spec.examples.map((example) => ({
    input: toJsonObject(example.input, {}),
    ...(example.output === undefined ? {} : { expectedResultShape: toJsonValue(example.output) ?? null }),
  }));

const resultExamplesFor = (
  policy: McpToolPolicy,
  resultStyle: ToolResultStyle,
): McpRuntimeResultExample[] => [
  {
    name: "truncated_result",
    resultStyle: "json",
    value: {
      mcpResultTruncated: true,
      originalBytes: policy.resultMaxBytes + 2048,
      omittedBytes: 2048,
      preview: "{\"partial\":true}",
    },
  },
  {
    name: "structured_error",
    resultStyle: "structured_error",
    value: {
      ok: false,
      status: "invalid_arguments",
      error: {
        class: "invalid_arguments",
        message: "MCP tool arguments are missing required field(s).",
        details: {
          reason: "missing_required",
        },
      },
    },
  },
  {
    name: "normal_result_style",
    resultStyle,
    value: {
      resultStyle,
    },
  },
];

const resultContractFor = (
  policy: McpToolPolicy,
  resultStyle: ToolResultStyle,
  resultStyleVersion?: string,
): McpRuntimeResultContract => ({
  resultStyle,
  ...(resultStyleVersion === undefined ? {} : { resultStyleVersion }),
  resultMaxBytes: policy.resultMaxBytes,
  truncationMessage: truncationMessageFor(policy),
  errorResultStyle: "structured_error",
  resultExamples: resultExamplesFor(policy, resultStyle),
});

const modelFacingDescriptionFor = (
  baseDescription: string,
  resultContract: McpRuntimeResultContract,
): string =>
  [
    baseDescription.trim(),
    `Result style: ${resultContract.resultStyle}.`,
    resultContract.truncationMessage,
    "Errors use structured_error with class, message, and optional details.",
  ].filter((part) => part.length > 0).join(" ");

const modelFacingPromptFragmentsFor = (
  renderedName: string,
  promptFragments: readonly string[],
  resultContract: McpRuntimeResultContract,
): string[] => uniqueSorted([
  ...promptFragments,
  `Result contract for ${renderedName}: ${resultContract.truncationMessage}`,
  `Structured error contract for ${renderedName}: return class, message, and optional details; do not hide invalid arguments or permission denials in plain text.`,
]);

const modelFacingPolicyFor = (
  policy: McpToolPolicy,
): McpModelFacingToolContract["policy"] => ({
  sideEffectLevel: policy.sideEffectLevel,
  requiresConfirmation: policy.requiresConfirmation,
  risks: policy.risks,
  safeAction: policy.safeAction,
  yoloAction: policy.yoloAction,
  reasons: policy.reasons,
});

const modelFacingContractFor = (
  tool: NormalizedMcpTool,
  rendered: RenderedToolContract | undefined,
): McpModelFacingToolContract => {
  const spec = tool.canonicalSpec;
  const renderedName = rendered?.name ?? renderedNameFallbackFor(spec);
  const resultStyle = rendered?.resultStyle ?? spec.resultStyle;
  const resultContract = resultContractFor(tool.policy, resultStyle, rendered?.resultStyleVersion);
  const canonicalToolVersion = rendered?.canonicalToolVersion ?? spec.canonicalToolVersion;
  const modelFacingToolName = modelFacingToolNameFor(renderedName, spec.canonicalToolId);
  const modelFacingToolId = modelFacingToolIdFor({
    canonicalToolId: spec.canonicalToolId,
    canonicalToolVersion,
    ...(rendered?.renderedToolId === undefined ? {} : { renderedToolId: rendered.renderedToolId }),
    ...(rendered?.renderedToolVersion === undefined ? {} : { renderedToolVersion: rendered.renderedToolVersion }),
    modelFacingToolName,
    resultStyle,
    ...(rendered?.resultStyleVersion === undefined ? {} : { resultStyleVersion: rendered.resultStyleVersion }),
    resultMaxBytes: tool.policy.resultMaxBytes,
  });
  const lineage: McpRuntimeToolLineage = {
    canonicalToolId: spec.canonicalToolId,
    canonicalToolVersion,
    modelFacingToolId,
    modelFacingToolName,
    ...(rendered?.renderedToolId === undefined ? {} : { renderedToolId: rendered.renderedToolId }),
    renderedToolName: renderedName,
    ...(rendered?.renderedToolVersion === undefined ? {} : { renderedToolVersion: rendered.renderedToolVersion }),
    ...(rendered?.modelProfileId === undefined ? {} : { modelProfileId: rendered.modelProfileId }),
    ...(rendered?.policyId === undefined ? {} : { policyId: rendered.policyId }),
    ...(rendered?.renderer === undefined ? {} : { renderer: rendered.renderer }),
    ...(rendered?.rendererVersion === undefined ? {} : { rendererVersion: rendered.rendererVersion }),
    ...(rendered?.resultStyleVersion === undefined ? {} : { resultStyleVersion: rendered.resultStyleVersion }),
  };

  return {
    kind: "mcp.model_facing_tool_contract",
    modelFacingToolId,
    modelFacingToolName,
    title: spec.title,
    description: modelFacingDescriptionFor(rendered?.description ?? spec.description, resultContract),
    inputSchema: rendered?.inputSchema ?? spec.inputSchema,
    resultStyle,
    promptFragments: modelFacingPromptFragmentsFor(renderedName, rendered?.promptFragments ?? [], resultContract),
    examples: rendered?.examples ?? renderedExamplesFallbackFor(spec),
    resultContract,
    lineage,
    policy: modelFacingPolicyFor(tool.policy),
    serverId: tool.serverId,
    serverName: tool.serverName,
    toolName: tool.toolName,
    canonicalSpec: spec,
    ...(rendered === undefined ? {} : { renderedContract: rendered }),
  };
};

export const buildMcpModelFacingToolContracts = (
  normalizedTools: readonly NormalizedMcpTool[],
  renderedContracts: readonly RenderedToolContract[] = [],
): McpModelFacingToolContract[] => {
  const renderedByCanonicalToolId = new Map(
    renderedContracts.map((contract) => [contract.canonicalToolId, contract] as const),
  );

  return normalizedTools.map((tool) => modelFacingContractFor(
    tool,
    renderedByCanonicalToolId.get(tool.canonicalSpec.canonicalToolId),
  ));
};

export const prepareMcpRenderedToolContracts = (
  input: PrepareMcpRenderedContractsInput,
): McpRenderedContractPreparation => {
  const canonicalToolSpecs = input.normalizedTools.map((tool) => tool.canonicalSpec);
  const renderedContracts = renderToolContracts({
    canonicalToolSpecs,
    resolvedPolicy: input.resolvedPolicy,
  });
  const policiesByCanonicalToolId = Object.fromEntries(
    input.normalizedTools.map((tool) => [tool.canonicalSpec.canonicalToolId, tool.policy]),
  );
  const resultBudgetsByCanonicalToolId = Object.fromEntries(
    input.normalizedTools.map((tool) => [
      tool.canonicalSpec.canonicalToolId,
      {
        resultMaxBytes: tool.policy.resultMaxBytes,
        truncationMessage: truncationMessageFor(tool.policy),
        errorResultStyle: "structured_error" as const,
      },
    ]),
  );
  const modelFacingContracts = buildMcpModelFacingToolContracts(input.normalizedTools, renderedContracts);
  const modelFacingContractsById = Object.fromEntries(
    modelFacingContracts.map((contract) => [contract.modelFacingToolId, contract]),
  );
  const modelFacingContractsByName = Object.fromEntries(
    modelFacingContracts.map((contract) => [contract.modelFacingToolName, contract]),
  );

  return {
    canonicalToolSpecs,
    renderedContracts,
    modelFacingContracts,
    modelFacingContractsById,
    modelFacingContractsByName,
    policiesByCanonicalToolId,
    resultBudgetsByCanonicalToolId,
  };
};

const byteLength = (value: string): number => Buffer.byteLength(value, "utf8");

const jsonByteLength = (value: JsonValue): number => byteLength(stableJson(value));

const truncateUtf8 = (value: string, maxBytes: number): string => {
  if (maxBytes <= 0) {
    return "";
  }

  let truncated = Buffer.from(value, "utf8").subarray(0, maxBytes).toString("utf8");
  while (byteLength(truncated) > maxBytes) {
    truncated = truncated.slice(0, -1);
  }
  return truncated;
};

const fitJsonString = (value: string, maxBytes: number): string => {
  let fitted = value;
  while (byteLength(JSON.stringify(fitted)) > maxBytes && fitted.length > 0) {
    fitted = fitted.slice(0, -1);
  }
  return fitted;
};

const toRuntimeResultValue = (value: unknown): JsonValue => {
  const jsonValue = toJsonValue(value);
  if (jsonValue !== undefined) {
    return jsonValue;
  }

  if (value === undefined) {
    return null;
  }

  return { result: String(value) };
};

const boundResultValue = (
  value: JsonValue,
  maxBytes: number,
): {
  value: JsonValue;
  resultBytes: number;
  resultBytesBeforeBounding: number;
  omittedResultBytes: number;
  truncated: boolean;
} => {
  const serialized = stableJson(value);
  const resultBytesBeforeBounding = byteLength(serialized);

  if (resultBytesBeforeBounding <= maxBytes) {
    return {
      value,
      resultBytes: resultBytesBeforeBounding,
      resultBytesBeforeBounding,
      omittedResultBytes: 0,
      truncated: false,
    };
  }

  const emptyPreview = {
    mcpResultTruncated: true,
    originalBytes: resultBytesBeforeBounding,
    omittedBytes: resultBytesBeforeBounding,
    preview: "",
  };
  const emptyPreviewBytes = jsonByteLength(emptyPreview);
  const previewBudget = Math.max(0, maxBytes - emptyPreviewBytes - 16);
  const preview = truncateUtf8(serialized, previewBudget);
  const previewBytes = byteLength(preview);
  const truncatedValue: JsonObject = {
    mcpResultTruncated: true,
    originalBytes: resultBytesBeforeBounding,
    omittedBytes: Math.max(0, resultBytesBeforeBounding - previewBytes),
    preview,
  };
  const truncatedBytes = jsonByteLength(truncatedValue);

  if (truncatedBytes <= maxBytes) {
    return {
      value: truncatedValue,
      resultBytes: truncatedBytes,
      resultBytesBeforeBounding,
      omittedResultBytes: Math.max(0, resultBytesBeforeBounding - previewBytes),
      truncated: true,
    };
  }

  const fallback = fitJsonString(
    `MCP result truncated from ${resultBytesBeforeBounding} bytes to fit ${maxBytes} bytes.`,
    maxBytes,
  );
  return {
    value: fallback,
    resultBytes: jsonByteLength(fallback),
    resultBytesBeforeBounding,
    omittedResultBytes: resultBytesBeforeBounding,
    truncated: true,
  };
};

const schemaTypeAllows = (schema: unknown, value: unknown): boolean => {
  if (!isRecord(schema)) {
    return true;
  }

  const enumValues = Array.isArray(schema.enum) ? schema.enum : undefined;
  if (enumValues !== undefined && !enumValues.some((entry) => stableJson(entry) === stableJson(value))) {
    return false;
  }

  const rawType = schema.type;
  const allowedTypes = Array.isArray(rawType)
    ? rawType.filter((entry): entry is string => typeof entry === "string")
    : typeof rawType === "string"
      ? [rawType]
      : [];

  if (allowedTypes.length === 0) {
    return true;
  }

  return allowedTypes.some((type) => {
    switch (type) {
      case "array":
        return Array.isArray(value);
      case "boolean":
        return typeof value === "boolean";
      case "integer":
        return typeof value === "number" && Number.isInteger(value);
      case "null":
        return value === null;
      case "number":
        return typeof value === "number" && !Number.isNaN(value);
      case "object":
        return isRecord(value);
      case "string":
        return typeof value === "string";
      default:
        return true;
    }
  });
};

const validateRuntimeArguments = (
  value: unknown,
  schema: JsonObject,
  maxBytes: number,
): { ok: true; arguments: JsonObject; argumentBytes: number } | {
  ok: false;
  argumentBytes: number;
  failureCode: McpRuntimeFailureCode;
  message: string;
  details: JsonValue;
} => {
  const jsonValue = toJsonValue(value);
  const argumentBytes = byteLength(stableJson(jsonValue ?? null));

  if (!isRecord(jsonValue)) {
    return {
      ok: false,
      argumentBytes,
      failureCode: "malformed_arguments",
      message: "MCP tool arguments must be a JSON object.",
      details: { reason: "not_object" },
    };
  }

  if (argumentBytes > maxBytes) {
    return {
      ok: false,
      argumentBytes,
      failureCode: "malformed_arguments",
      message: `MCP tool arguments exceed ${maxBytes} bytes.`,
      details: {
        reason: "arguments_too_large",
        maxBytes,
        actualBytes: argumentBytes,
      },
    };
  }

  const missingRequired = (Array.isArray(schema.required) ? schema.required : [])
    .filter((entry): entry is string => typeof entry === "string")
    .filter((entry) => !(entry in jsonValue));
  if (missingRequired.length > 0) {
    return {
      ok: false,
      argumentBytes,
      failureCode: "schema_mismatch",
      message: `MCP tool arguments are missing required field(s): ${missingRequired.join(", ")}.`,
      details: {
        reason: "missing_required",
        fields: missingRequired,
      },
    };
  }

  const properties = isRecord(schema.properties) ? schema.properties : {};
  const typeErrors = Object.entries(jsonValue)
    .map(([key, entry]) => {
      const propertySchema = properties[key];
      return propertySchema !== undefined && !schemaTypeAllows(propertySchema, entry) ? key : undefined;
    })
    .filter((entry): entry is string => entry !== undefined);

  if (typeErrors.length > 0) {
    return {
      ok: false,
      argumentBytes,
      failureCode: "schema_mismatch",
      message: `MCP tool arguments have invalid field type(s): ${typeErrors.join(", ")}.`,
      details: {
        reason: "invalid_type",
        fields: typeErrors,
      },
    };
  }

  if (schema.additionalProperties === false) {
    const unknownFields = Object.keys(jsonValue).filter((key) => !(key in properties));
    if (unknownFields.length > 0) {
      return {
        ok: false,
        argumentBytes,
        failureCode: "schema_mismatch",
        message: `MCP tool arguments include unknown field(s): ${unknownFields.join(", ")}.`,
        details: {
          reason: "unknown_fields",
          fields: unknownFields,
        },
      };
    }
  }

  return {
    ok: true,
    arguments: jsonValue as JsonObject,
    argumentBytes,
  };
};

const isAbortLikeError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError";

const isTimeoutLikeError = (error: unknown): boolean =>
  error instanceof Error && error.name === "TimeoutError";

const isStaleContextLikeError = (error: unknown): boolean =>
  error instanceof Error && (error.name === "StaleContextError" || error.name === "StaleContext");

const cancellationError = (): Error => Object.assign(new Error("MCP tool call was cancelled."), { name: "AbortError" });

const timeoutError = (timeoutMs: number): Error =>
  Object.assign(new Error(`MCP tool call timed out after ${timeoutMs}ms.`), { name: "TimeoutError" });

const abortReasonFor = (signal: AbortSignal): Error =>
  signal.reason instanceof Error ? signal.reason : cancellationError();

const throwIfAborted = (signal: AbortSignal | undefined): void => {
  if (signal?.aborted === true) {
    throw abortReasonFor(signal);
  }
};

const runWithCancellation = async <T>(operation: Promise<T>, signal: AbortSignal | undefined): Promise<T> => {
  if (signal === undefined) {
    return operation;
  }

  throwIfAborted(signal);
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => {
      reject(abortReasonFor(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    operation.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort);
    });
  });
};

const runtimeSignalFor = (
  externalSignal: AbortSignal | undefined,
  timeoutMs: number | undefined,
): { signal?: AbortSignal; cleanup: () => void } => {
  if (timeoutMs === undefined) {
    return {
      ...(externalSignal === undefined ? {} : { signal: externalSignal }),
      cleanup: () => undefined,
    };
  }

  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | undefined = setTimeout(() => {
    if (!controller.signal.aborted) {
      controller.abort(timeoutError(timeoutMs));
    }
  }, timeoutMs);

  const onExternalAbort = (): void => {
    if (!controller.signal.aborted && externalSignal !== undefined) {
      controller.abort(abortReasonFor(externalSignal));
    }
  };

  if (externalSignal?.aborted === true) {
    onExternalAbort();
  } else {
    externalSignal?.addEventListener("abort", onExternalAbort, { once: true });
  }

  return {
    signal: controller.signal,
    cleanup: () => {
      if (timeoutId !== undefined) {
        clearTimeout(timeoutId);
        timeoutId = undefined;
      }
      externalSignal?.removeEventListener("abort", onExternalAbort);
    },
  };
};

const elapsedMs = (now: () => number, startedAt: number): number => Math.max(0, now() - startedAt);

const followUpBehaviorFor = (
  failureCode: McpRuntimeFailureCode | undefined,
): McpRuntimeFollowUpBehavior => {
  switch (failureCode) {
    case undefined:
      return "none";
    case "malformed_arguments":
    case "schema_mismatch":
      return "repair_arguments";
    case "missing_server":
    case "unknown_tool":
      return "refresh_tool_inventory";
    case "permission_denied":
    case "policy_blocked":
    case "permission_error":
      return "request_permission_or_choose_lower_risk_tool";
    case "oversized_output":
      return "narrow_request_or_paginate";
    case "timeout":
    case "cancelled":
    case "retry_exhausted":
      return "retry_with_narrower_scope_or_abort";
    case "stale_context":
    case "runtime_exception":
      return "inspect_executor_or_choose_alternate_tool";
  }
};

const createTrace = (input: {
  status: McpRuntimeToolStatus;
  callId: string;
  callable?: McpRuntimeCallableTool;
  policyDecision?: McpRuntimePolicyDecision;
  durationMs: number;
  argumentBytes: number;
  argumentShapeHash: string;
  redactionStatus: McpRuntimeRedactionStatus;
  resultBytes: number;
  resultBytesBeforeBounding: number;
  omittedResultBytes: number;
  resultTruncated: boolean;
  retryCount: number;
  failureCode?: McpRuntimeFailureCode;
  errorClass?: string;
}): McpRuntimeToolTrace => {
  const failureCode = input.failureCode ?? (input.resultTruncated ? "oversized_output" : undefined);
  return {
    event: "mcp.tool_call",
    spanName: "mcp.tool_call",
    callId: input.callId,
    status: input.status,
    ...(input.callable === undefined
      ? {}
      : {
        canonicalToolId: input.callable.canonicalToolId,
        canonicalToolVersion: input.callable.canonicalToolVersion,
        modelFacingToolId: input.callable.modelFacingToolId,
        modelFacingToolName: input.callable.modelFacingToolName,
        ...(input.callable.renderedToolVersion === undefined ? {} : { renderedToolVersion: input.callable.renderedToolVersion }),
        ...(input.callable.renderedToolId === undefined ? {} : { renderedToolId: input.callable.renderedToolId }),
        ...(input.callable.renderedToolName === undefined ? {} : { renderedToolName: input.callable.renderedToolName }),
        ...(input.callable.modelProfileId === undefined ? {} : { modelProfileId: input.callable.modelProfileId }),
        ...(input.callable.policyId === undefined ? {} : { policyId: input.callable.policyId }),
        ...(input.callable.renderer === undefined ? {} : { renderer: input.callable.renderer }),
        ...(input.callable.rendererVersion === undefined ? {} : { rendererVersion: input.callable.rendererVersion }),
        ...(input.callable.resultStyleVersion === undefined ? {} : { resultStyleVersion: input.callable.resultStyleVersion }),
        serverId: input.callable.serverId,
        serverName: input.callable.serverName,
        toolName: input.callable.toolName,
      }),
    ...(input.policyDecision === undefined
      ? {}
      : {
        policyAction: input.policyDecision.action,
        permissionStatus: input.policyDecision.permissionStatus,
        sideEffectLevel: input.policyDecision.sideEffectLevel,
    }),
    durationMs: input.durationMs,
    argumentBytes: input.argumentBytes,
    argumentShapeHash: input.argumentShapeHash,
    redactionStatus: input.redactionStatus,
    resultBytes: input.resultBytes,
    resultBytesBeforeBounding: input.resultBytesBeforeBounding,
    omittedResultBytes: input.omittedResultBytes,
    resultTruncated: input.resultTruncated,
    retryCount: input.retryCount,
    ...(failureCode === undefined ? {} : { failureCode }),
    ...(input.errorClass === undefined ? {} : { errorClass: input.errorClass }),
    followUpBehavior: followUpBehaviorFor(failureCode),
  };
};

const createResult = (input: {
  status: McpRuntimeToolStatus;
  callId: string;
  callable?: McpRuntimeCallableTool;
  policyDecision?: McpRuntimePolicyDecision;
  metrics: McpRuntimeToolMetrics;
  argumentShapeHash: string;
  redactionStatus?: McpRuntimeRedactionStatus;
  result?: JsonValue;
  error?: McpRuntimeToolResult["error"];
}): McpRuntimeToolResult => {
  const redactionStatus = input.redactionStatus ?? DEFAULT_TRACE_REDACTION_STATUS;
  const trace = createTrace({
    status: input.status,
    callId: input.callId,
    ...(input.callable === undefined ? {} : { callable: input.callable }),
    ...(input.policyDecision === undefined ? {} : { policyDecision: input.policyDecision }),
    durationMs: input.metrics.durationMs,
    argumentBytes: input.metrics.argumentBytes,
    argumentShapeHash: input.argumentShapeHash,
    redactionStatus,
    resultBytes: input.metrics.resultBytes,
    resultBytesBeforeBounding: input.metrics.resultBytesBeforeBounding,
    omittedResultBytes: input.metrics.omittedResultBytes,
    resultTruncated: input.metrics.truncated,
    retryCount: input.metrics.retryCount,
    ...(input.error === undefined ? {} : { failureCode: input.error.code, errorClass: input.error.class }),
  });
  const failureCode = trace.failureCode;

  return {
    ok: input.status === "success",
    status: input.status,
    callId: input.callId,
    call: input.callable === undefined
      ? {}
      : {
        canonicalToolId: input.callable.canonicalToolId,
        canonicalToolVersion: input.callable.canonicalToolVersion,
        modelFacingToolId: input.callable.modelFacingToolId,
        modelFacingToolName: input.callable.modelFacingToolName,
        ...(input.callable.renderedToolVersion === undefined ? {} : { renderedToolVersion: input.callable.renderedToolVersion }),
        ...(input.callable.renderedToolId === undefined ? {} : { renderedToolId: input.callable.renderedToolId }),
        ...(input.callable.renderedToolName === undefined ? {} : { renderedToolName: input.callable.renderedToolName }),
        ...(input.callable.modelProfileId === undefined ? {} : { modelProfileId: input.callable.modelProfileId }),
        ...(input.callable.policyId === undefined ? {} : { policyId: input.callable.policyId }),
        ...(input.callable.renderer === undefined ? {} : { renderer: input.callable.renderer }),
        ...(input.callable.rendererVersion === undefined ? {} : { rendererVersion: input.callable.rendererVersion }),
        ...(input.callable.resultStyleVersion === undefined ? {} : { resultStyleVersion: input.callable.resultStyleVersion }),
        serverId: input.callable.serverId,
        serverName: input.callable.serverName,
        toolName: input.callable.toolName,
      },
    ...(input.policyDecision === undefined ? {} : { policyDecision: input.policyDecision }),
    metrics: input.metrics,
    argumentShapeHash: input.argumentShapeHash,
    redactionStatus,
    followUpBehavior: trace.followUpBehavior,
    ...(failureCode === undefined ? {} : { failureCode }),
    ...(input.result === undefined ? {} : { result: input.result }),
    ...(input.error === undefined ? {} : { error: input.error }),
    trace,
  };
};

const zeroMetrics = (durationMs: number, argumentBytes = 0, retryCount = 0): McpRuntimeToolMetrics => ({
  argumentBytes,
  resultBytes: 0,
  resultBytesBeforeBounding: 0,
  omittedResultBytes: 0,
  truncated: false,
  retryCount,
  durationMs,
});

const policyActionForMode = (callable: McpRuntimeCallableTool, mode: McpRuntimeExecutionMode): McpPolicyAction =>
  mode === "yolo" ? callable.policy.yoloAction : callable.policy.safeAction;

const preliminaryPolicyDecision = (
  callable: McpRuntimeCallableTool,
  mode: McpRuntimeExecutionMode,
): McpRuntimePolicyDecision => {
  const action = policyActionForMode(callable, mode);
  return {
    mode,
    action,
    permissionStatus: action === "allow" ? "not_required" : action === "block" ? "blocked" : "requested",
    sideEffectLevel: callable.policy.sideEffectLevel,
    requiresConfirmation: callable.policy.requiresConfirmation,
    reasons: callable.policy.reasons,
  };
};

const decidePolicy = async (
  input: {
    callable: McpRuntimeCallableTool;
    callId: string;
    arguments: JsonObject;
    mode: McpRuntimeExecutionMode;
    permissionHandler?: McpRuntimePermissionHandler;
  },
): Promise<McpRuntimePolicyDecision> => {
  const action = policyActionForMode(input.callable, input.mode);
  const base = {
    mode: input.mode,
    action,
    sideEffectLevel: input.callable.policy.sideEffectLevel,
    requiresConfirmation: input.callable.policy.requiresConfirmation,
    reasons: input.callable.policy.reasons,
  };

  if (action === "allow") {
    return {
      ...base,
      permissionStatus: "not_required",
    };
  }

  if (action === "block") {
    return {
      ...base,
      permissionStatus: "blocked",
    };
  }

  if (input.permissionHandler === undefined) {
    return {
      ...base,
      permissionStatus: "denied",
    };
  }

  const permissionRequest: McpRuntimePermissionRequest = {
    callId: input.callId,
    serverId: input.callable.serverId,
    serverName: input.callable.serverName,
    toolName: input.callable.toolName,
    canonicalToolId: input.callable.canonicalToolId,
    modelFacingToolId: input.callable.modelFacingToolId,
    modelFacingToolName: input.callable.modelFacingToolName,
    ...(input.callable.renderedToolId === undefined ? {} : { renderedToolId: input.callable.renderedToolId }),
    ...(input.callable.renderedToolName === undefined ? {} : { renderedToolName: input.callable.renderedToolName }),
    policy: input.callable.policy,
    arguments: input.arguments,
    mode: input.mode,
  };
  const decision = await input.permissionHandler(permissionRequest);
  return {
    ...base,
    permissionStatus: decision === "allow" ? "granted" : "denied",
  };
};

const makeCallId = (): string => `mcp.call.${Date.now().toString(36)}.${Math.random().toString(16).slice(2, 10)}`;

const feedbackSeverityFor = (result: McpRuntimeToolResult): McpRuntimeOptimizerFeedbackSeverity => {
  if (result.status === "error" || result.status === "retry_exhausted") {
    return "failure";
  }
  if (
    result.status === "invalid_arguments"
    || result.status === "denied"
    || result.status === "missing_server"
    || result.status === "unknown_tool"
  ) {
    return "warning";
  }
  if (result.status === "cancelled" || result.status === "timeout" || result.metrics.truncated) {
    return "warning";
  }
  return "info";
};

const feedbackActionHintFor = (result: McpRuntimeToolResult): string => {
  if (result.failureCode === "malformed_arguments" || result.failureCode === "schema_mismatch") {
    return "Optimizer hint: tighten the rendered MCP tool description, schema, and examples so the model supplies exactly the required argument names and types.";
  }
  if (result.failureCode === "missing_server" || result.failureCode === "unknown_tool") {
    return "Optimizer hint: refresh or narrow the model-facing MCP inventory so the model only calls currently available server/tool contracts.";
  }
  if (result.failureCode === "oversized_output") {
    return "Optimizer hint: revise result guidance, result style, or policy budget so large MCP outputs are summarized or paginated before they reach the model.";
  }
  if (
    result.failureCode === "permission_denied"
    || result.failureCode === "policy_blocked"
    || result.failureCode === "permission_error"
  ) {
    return "Optimizer hint: check whether tool choice guidance steered the model toward a side-effecting MCP tool when a lower-risk tool or explicit confirmation path was required.";
  }
  if (
    result.failureCode === "cancelled"
    || result.failureCode === "timeout"
    || result.failureCode === "retry_exhausted"
  ) {
    return "Optimizer hint: inspect timeout, cancellation, and retry guidance for this MCP tool before encouraging repeated calls.";
  }
  if (result.failureCode === "runtime_exception" || result.failureCode === "stale_context") {
    return "Optimizer hint: use the structured error class and lineage to improve MCP tool guidance or retry policy without changing runtime source code.";
  }
  return "Optimizer hint: successful MCP call lineage is available for policy attribution and positive examples.";
};

const boundedFeedbackText = (value: string, maxChars: number): { text: string; truncated: boolean } => {
  if (value.length <= maxChars) {
    return { text: value, truncated: false };
  }

  const marker = `\n[MCP feedback truncated: original ${value.length} chars]`;
  return {
    text: `${value.slice(0, Math.max(0, maxChars - marker.length))}${marker}`,
    truncated: true,
  };
};

const feedbackLineageFor = (result: McpRuntimeToolResult): McpRuntimeOptimizerFeedbackLineage => ({
  canonicalToolIds: uniqueSorted([result.call.canonicalToolId]),
  modelFacingToolIds: uniqueSorted([result.call.modelFacingToolId]),
  modelFacingToolNames: uniqueSorted([result.call.modelFacingToolName]),
  renderedToolContractIds: uniqueSorted([result.call.renderedToolId]),
  canonicalToolVersions: uniqueSorted([result.call.canonicalToolVersion]),
  renderedToolVersions: uniqueSorted([result.call.renderedToolVersion]),
  resultStyleVersions: uniqueSorted([result.call.resultStyleVersion]),
  modelProfileIds: uniqueSorted([result.call.modelProfileId]),
  policyIds: uniqueSorted([result.call.policyId]),
});

const feedbackTextFor = (result: McpRuntimeToolResult): string => {
  const callName = result.call.renderedToolName ?? result.call.toolName ?? result.call.canonicalToolId ?? "unknown_mcp_tool";
  return [
    `MCP tool call ${result.status}: ${callName}`,
    `Contract lineage: canonicalToolId=${result.call.canonicalToolId ?? "unknown"} modelFacingToolId=${result.call.modelFacingToolId ?? "none"} modelFacingToolName=${result.call.modelFacingToolName ?? "none"} renderedToolId=${result.call.renderedToolId ?? "none"} canonicalToolVersion=${result.call.canonicalToolVersion ?? "unknown"} renderedToolVersion=${result.call.renderedToolVersion ?? "none"} modelProfileId=${result.call.modelProfileId ?? "none"} policyId=${result.call.policyId ?? "none"}`,
    `Policy: action=${result.policyDecision?.action ?? result.trace.policyAction ?? "unknown"} permission=${result.policyDecision?.permissionStatus ?? result.trace.permissionStatus ?? "unknown"} sideEffect=${result.policyDecision?.sideEffectLevel ?? result.trace.sideEffectLevel ?? "unknown"}`,
    `Trace evidence: durationMs=${result.metrics.durationMs} argumentShapeHash=${result.argumentShapeHash} redactionStatus=${result.redactionStatus} failureCode=${result.failureCode ?? "none"} followUp=${result.followUpBehavior}`,
    `Sizes: arguments=${result.metrics.argumentBytes}B result=${result.metrics.resultBytes}B originalResult=${result.metrics.resultBytesBeforeBounding}B omitted=${result.metrics.omittedResultBytes}B truncated=${result.metrics.truncated} retries=${result.metrics.retryCount}`,
    result.error === undefined ? "" : `Error: class=${result.error.class} code=${result.error.code} message=${result.error.message}`,
    result.error?.details === undefined ? "" : `Error details: ${stableJson(result.error.details)}`,
    feedbackActionHintFor(result),
  ].filter((line) => line.length > 0).join("\n");
};

export const mcpRuntimeToolResultToOptimizerFeedback = (
  result: McpRuntimeToolResult,
  options: McpRuntimeOptimizerFeedbackOptions = {},
): McpRuntimeOptimizerFeedbackRecord | undefined => {
  if (result.ok && !result.metrics.truncated && options.includeSuccessful !== true) {
    return undefined;
  }

  const maxFeedbackChars = boundedInteger(
    options.maxFeedbackChars,
    DEFAULT_FEEDBACK_MAX_CHARS,
    1,
    MAX_FEEDBACK_CHARS,
  );
  const bounded = boundedFeedbackText(feedbackTextFor(result), maxFeedbackChars);
  const policyAction = result.policyDecision?.action ?? result.trace.policyAction;
  const permissionStatus = result.policyDecision?.permissionStatus ?? result.trace.permissionStatus;
  const sideEffectLevel = result.policyDecision?.sideEffectLevel ?? result.trace.sideEffectLevel;

  return {
    feedbackId: `mcp.feedback.${shortHash({
      callId: result.callId,
      status: result.status,
      canonicalToolId: result.call.canonicalToolId,
      renderedToolId: result.call.renderedToolId,
      errorClass: result.error?.class,
      failureCode: result.failureCode,
      truncated: result.metrics.truncated,
    })}`,
    source: "mcp_runtime_tool_call",
    severity: feedbackSeverityFor(result),
    objective: options.objective ?? "Use MCP runtime trace evidence to improve rendered tool contracts, side-effect policy, retry guidance, or GEPA optimizer feedback.",
    feedback: bounded.text,
    callId: result.callId,
    status: result.status,
    traceIds: options.traceId === undefined ? [] : [options.traceId],
    spanIds: options.spanId === undefined ? [] : [options.spanId],
    ...(result.call.canonicalToolId === undefined ? {} : { canonicalToolId: result.call.canonicalToolId }),
    ...(result.call.modelFacingToolId === undefined ? {} : { modelFacingToolId: result.call.modelFacingToolId }),
    ...(result.call.modelFacingToolName === undefined ? {} : { modelFacingToolName: result.call.modelFacingToolName }),
    ...(result.call.renderedToolId === undefined ? {} : { renderedToolId: result.call.renderedToolId }),
    ...(result.call.renderedToolName === undefined ? {} : { renderedToolName: result.call.renderedToolName }),
    ...(result.call.serverId === undefined ? {} : { serverId: result.call.serverId }),
    ...(result.call.serverName === undefined ? {} : { serverName: result.call.serverName }),
    ...(result.call.toolName === undefined ? {} : { toolName: result.call.toolName }),
    ...(policyAction === undefined ? {} : { policyAction }),
    ...(permissionStatus === undefined ? {} : { permissionStatus }),
    ...(sideEffectLevel === undefined ? {} : { sideEffectLevel }),
    durationMs: result.metrics.durationMs,
    argumentBytes: result.metrics.argumentBytes,
    argumentShapeHash: result.argumentShapeHash,
    redactionStatus: result.redactionStatus,
    resultBytes: result.metrics.resultBytes,
    resultBytesBeforeBounding: result.metrics.resultBytesBeforeBounding,
    omittedResultBytes: result.metrics.omittedResultBytes,
    resultTruncated: result.metrics.truncated,
    retryCount: result.metrics.retryCount,
    ...(result.failureCode === undefined ? {} : { failureCode: result.failureCode }),
    ...(result.error === undefined ? {} : { errorClass: result.error.class }),
    followUpBehavior: result.followUpBehavior,
    lineage: feedbackLineageFor(result),
    redacted: false,
    truncated: bounded.truncated,
  };
};

export const mcpRuntimeToolResultsToOptimizerFeedback = (
  results: readonly McpRuntimeToolResult[],
  options: McpRuntimeOptimizerFeedbackOptions = {},
): McpRuntimeOptimizerFeedbackRecord[] =>
  results
    .map((result) => mcpRuntimeToolResultToOptimizerFeedback(result, options))
    .filter((record): record is McpRuntimeOptimizerFeedbackRecord => record !== undefined);

const mcpStdioRequestError = (message: string, details?: unknown): Error => {
  const error = new Error(message);
  if (details !== undefined) {
    Object.assign(error, { details });
  }
  return error;
};

const encodeMcpStdioMessage = (message: JsonRpcMessage): Buffer => {
  const body = Buffer.from(JSON.stringify(message), "utf8");
  return Buffer.concat([
    Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, "utf8"),
    body,
  ]);
};

const parseMcpStdioFrames = (
  buffer: Buffer<ArrayBufferLike>,
): { messages: JsonRpcMessage[]; remaining: Buffer<ArrayBufferLike> } => {
  const messages: JsonRpcMessage[] = [];
  let remaining = buffer;

  while (remaining.length > 0) {
    const headerEnd = remaining.indexOf("\r\n\r\n");
    if (headerEnd < 0) {
      break;
    }

    const header = remaining.subarray(0, headerEnd).toString("utf8");
    const lengthHeader = header
      .split("\r\n")
      .find((line) => line.toLowerCase().startsWith("content-length:"));
    const contentLength = Number(lengthHeader?.slice("content-length:".length).trim());
    if (!Number.isInteger(contentLength) || contentLength < 0) {
      throw mcpStdioRequestError("MCP stdio frame is missing a valid Content-Length header.");
    }

    const bodyStart = headerEnd + 4;
    const bodyEnd = bodyStart + contentLength;
    if (remaining.length < bodyEnd) {
      break;
    }

    const rawBody = remaining.subarray(bodyStart, bodyEnd).toString("utf8");
    const parsed: unknown = JSON.parse(rawBody);
    if (!isRecord(parsed)) {
      throw mcpStdioRequestError("MCP stdio frame body was not a JSON object.");
    }
    messages.push(parsed as JsonRpcMessage);
    remaining = remaining.subarray(bodyEnd);
  }

  return { messages, remaining };
};

const createMcpStdioJsonRpcClient = (
  input: McpStdioTransportInput,
): {
  process: ChildProcessWithoutNullStreams;
  request: (method: string, params?: unknown, options?: {
    timeoutMs?: number;
    signal?: AbortSignal;
  }) => Promise<unknown>;
  notify: (method: string, params?: unknown) => void;
  close: () => Promise<void>;
} => {
  let nextId = 1;
  let stdoutBuffer: Buffer<ArrayBufferLike> = Buffer.alloc(0);
  const pending = new Map<string, JsonRpcPending>();
  const processEnv = input.env === undefined ? process.env : { ...process.env, ...input.env };
  const child = spawn(input.command, [...(input.args ?? [])], {
    stdio: ["pipe", "pipe", "pipe"],
    ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
    env: processEnv,
  });

  const rejectPending = (error: Error): void => {
    for (const [, entry] of pending) {
      clearTimeout(entry.timer);
      entry.reject(error);
    }
    pending.clear();
  };

  child.stdout.on("data", (chunk: Buffer) => {
    try {
      stdoutBuffer = Buffer.concat([stdoutBuffer, chunk]);
      const parsed = parseMcpStdioFrames(stdoutBuffer);
      stdoutBuffer = parsed.remaining;
      for (const message of parsed.messages) {
        if (message.id === undefined || message.id === null) {
          continue;
        }

        const key = String(message.id);
        const entry = pending.get(key);
        if (entry === undefined) {
          continue;
        }
        pending.delete(key);
        clearTimeout(entry.timer);
        if (message.error !== undefined) {
          entry.reject(mcpStdioRequestError(
            message.error.message ?? "MCP stdio server returned a JSON-RPC error.",
            message.error,
          ));
        } else {
          entry.resolve(message.result);
        }
      }
    } catch (error) {
      rejectPending(error instanceof Error ? error : mcpStdioRequestError("MCP stdio frame parsing failed."));
    }
  });

  child.on("error", (error) => {
    rejectPending(error);
  });
  child.on("exit", (code, signal) => {
    if (pending.size > 0) {
      rejectPending(mcpStdioRequestError(`MCP stdio server exited before replying: code=${code ?? "null"} signal=${signal ?? "null"}.`));
    }
  });

  const writeMessage = (message: JsonRpcMessage): void => {
    if (child.stdin.destroyed || !child.stdin.writable) {
      throw mcpStdioRequestError("MCP stdio server stdin is closed.");
    }
    child.stdin.write(encodeMcpStdioMessage(message));
  };

  const request = (
    method: string,
    params?: unknown,
    options: { timeoutMs?: number; signal?: AbortSignal } = {},
  ): Promise<unknown> => {
    const id = nextId;
    nextId += 1;
    const key = String(id);
    const timeoutMs = boundedOptionalInteger(options.timeoutMs, 1, MAX_TOOL_TIMEOUT_MS)
      ?? boundedOptionalInteger(input.requestTimeoutMs, 1, MAX_TOOL_TIMEOUT_MS)
      ?? 30_000;

    return new Promise<unknown>((resolve, reject) => {
      if (options.signal?.aborted === true) {
        reject(abortReasonFor(options.signal));
        return;
      }

      const timer = setTimeout(() => {
        pending.delete(key);
        reject(timeoutError(timeoutMs));
      }, timeoutMs);
      const onAbort = (): void => {
        clearTimeout(timer);
        pending.delete(key);
        if (options.signal !== undefined) {
          reject(abortReasonFor(options.signal));
        } else {
          reject(cancellationError());
        }
      };

      pending.set(key, {
        resolve: (value) => {
          options.signal?.removeEventListener("abort", onAbort);
          resolve(value);
        },
        reject: (error) => {
          options.signal?.removeEventListener("abort", onAbort);
          reject(error);
        },
        timer,
      });
      options.signal?.addEventListener("abort", onAbort, { once: true });

      try {
        writeMessage({
          jsonrpc: "2.0",
          id,
          method,
          ...(params === undefined ? {} : { params }),
        });
      } catch (error) {
        clearTimeout(timer);
        options.signal?.removeEventListener("abort", onAbort);
        pending.delete(key);
        reject(error instanceof Error ? error : mcpStdioRequestError("MCP stdio write failed."));
      }
    });
  };

  const notify = (method: string, params?: unknown): void => {
    writeMessage({
      jsonrpc: "2.0",
      method,
      ...(params === undefined ? {} : { params }),
    });
  };

  const close = async (): Promise<void> => {
    rejectPending(cancellationError());
    if (child.exitCode !== null || child.killed) {
      return;
    }

    await new Promise<void>((resolve) => {
      const cleanup = (): void => {
        child.off("exit", cleanup);
        resolve();
      };
      child.once("exit", cleanup);
      child.kill("SIGTERM");
      setTimeout(() => {
        if (child.exitCode === null && !child.killed) {
          child.kill("SIGKILL");
        }
        resolve();
      }, 1_000).unref();
    });
  };

  input.signal?.addEventListener("abort", () => {
    void close();
  }, { once: true });

  return {
    process: child,
    request,
    notify,
    close,
  };
};

const normalizeMcpListToolsResult = (server: Omit<McpServerMetadata, "tools">, value: unknown): McpServerMetadata => {
  const tools = isRecord(value) && Array.isArray(value.tools) ? value.tools : [];
  return {
    ...server,
    tools: tools
      .filter(isRecord)
      .map((tool): McpToolMetadata => ({
        name: typeof tool.name === "string" ? tool.name : "tool",
        ...(typeof tool.title === "string" ? { title: tool.title } : {}),
        ...(typeof tool.description === "string" ? { description: tool.description } : {}),
        ...(tool.inputSchema === undefined ? {} : { inputSchema: tool.inputSchema }),
        ...(tool.outputSchema === undefined ? {} : { outputSchema: tool.outputSchema }),
        ...(isRecord(tool.annotations) ? { annotations: tool.annotations as unknown as McpToolAnnotations } : {}),
      })),
  };
};

export const connectMcpStdioRuntimeServer = async (
  input: McpStdioTransportInput,
): Promise<McpStdioRuntimeServer> => {
  const serverIdentity = {
    ...(input.serverId === undefined ? {} : { serverId: input.serverId }),
    name: input.name,
    ...(input.displayName === undefined ? {} : { displayName: input.displayName }),
  };
  const client = createMcpStdioJsonRpcClient(input);
  const startupTimeoutMs = boundedOptionalInteger(input.startupTimeoutMs, 1, MAX_TOOL_TIMEOUT_MS) ?? 10_000;

  try {
    await client.request("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: {
        name: "bleeding-agent-mcp-runtime",
        version: "1.0.0",
      },
    }, {
      timeoutMs: startupTimeoutMs,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
    client.notify("notifications/initialized");
    const listed = await client.request("tools/list", {}, {
      timeoutMs: startupTimeoutMs,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
    const server = normalizeMcpListToolsResult(serverIdentity, listed);

    return {
      server,
      executor: async (request) => {
        const result = await client.request("tools/call", {
          name: request.toolName,
          arguments: request.arguments,
        }, {
          ...(request.timeoutMs === undefined ? {} : { timeoutMs: request.timeoutMs }),
          ...(request.signal === undefined ? {} : { signal: request.signal }),
        });
        return result;
      },
      close: client.close,
      process: client.process,
    };
  } catch (error) {
    await client.close();
    throw error;
  }
};

export const createMcpRuntimeToolBridge = (
  input: CreateMcpRuntimeToolBridgeInput,
): McpRuntimeToolBridge => {
  const mode = input.mode ?? "safe";
  const now = input.now ?? (() => Date.now());
  const createCallId = input.createCallId ?? makeCallId;
  const defaultTimeoutMs = boundedOptionalInteger(input.timeoutMs, 1, MAX_TOOL_TIMEOUT_MS);
  const retryBudgetConfigured = input.maxRetryCount !== undefined;
  const maxRetryCount = boundedInteger(input.maxRetryCount, Number.MAX_SAFE_INTEGER, 0, Number.MAX_SAFE_INTEGER);
  const modelFacingContracts = buildMcpModelFacingToolContracts(input.normalizedTools, input.renderedContracts ?? []);
  const modelFacingByCanonicalToolId = new Map(
    modelFacingContracts.map((contract) => [contract.canonicalSpec.canonicalToolId, contract] as const),
  );
  const callableTools = input.normalizedTools.map((tool): McpRuntimeCallableTool => {
    const modelFacing = modelFacingByCanonicalToolId.get(tool.canonicalSpec.canonicalToolId)
      ?? modelFacingContractFor(tool, undefined);
    return {
      canonicalToolId: modelFacing.lineage.canonicalToolId,
      canonicalToolVersion: modelFacing.lineage.canonicalToolVersion,
      modelFacingToolId: modelFacing.modelFacingToolId,
      modelFacingToolName: modelFacing.modelFacingToolName,
      ...(modelFacing.lineage.renderedToolVersion === undefined ? {} : { renderedToolVersion: modelFacing.lineage.renderedToolVersion }),
      ...(modelFacing.lineage.renderedToolId === undefined ? {} : { renderedToolId: modelFacing.lineage.renderedToolId }),
      ...(modelFacing.lineage.renderedToolName === undefined ? {} : { renderedToolName: modelFacing.lineage.renderedToolName }),
      ...(modelFacing.lineage.modelProfileId === undefined ? {} : { modelProfileId: modelFacing.lineage.modelProfileId }),
      ...(modelFacing.lineage.policyId === undefined ? {} : { policyId: modelFacing.lineage.policyId }),
      ...(modelFacing.lineage.renderer === undefined ? {} : { renderer: modelFacing.lineage.renderer }),
      ...(modelFacing.lineage.rendererVersion === undefined ? {} : { rendererVersion: modelFacing.lineage.rendererVersion }),
      ...(modelFacing.lineage.resultStyleVersion === undefined ? {} : { resultStyleVersion: modelFacing.lineage.resultStyleVersion }),
      name: modelFacing.modelFacingToolName,
      title: modelFacing.title,
      description: modelFacing.description,
      inputSchema: modelFacing.inputSchema,
      resultStyle: modelFacing.resultStyle,
      promptFragments: modelFacing.promptFragments,
      examples: modelFacing.examples,
      resultContract: modelFacing.resultContract,
      lineage: modelFacing.lineage,
      serverId: tool.serverId,
      serverName: tool.serverName,
      toolName: tool.toolName,
      policy: tool.policy,
    };
  });
  const callableByKey = new Map<string, McpRuntimeCallableTool>();
  const serverKeys = new Set<string>();
  const serverToolKey = (serverKey: string, toolName: string): string => `${serverKey}\u0000${toolName}`;
  for (const callable of callableTools) {
    serverKeys.add(callable.serverId);
    serverKeys.add(callable.serverName);
    callableByKey.set(serverToolKey(callable.serverId, callable.toolName), callable);
    callableByKey.set(serverToolKey(callable.serverName, callable.toolName), callable);
    callableByKey.set(callable.canonicalToolId, callable);
    callableByKey.set(callable.modelFacingToolId, callable);
    callableByKey.set(callable.modelFacingToolName, callable);
    callableByKey.set(callable.name, callable);
    callableByKey.set(callable.toolName, callable);
    if (callable.renderedToolId !== undefined) {
      callableByKey.set(callable.renderedToolId, callable);
    }
    if (callable.renderedToolName !== undefined) {
      callableByKey.set(callable.renderedToolName, callable);
    }
  }

  const lookupCallable = (call: McpRuntimeToolCall): McpRuntimeCallableTool | undefined => {
    const serverScopedKeys = [call.serverId, call.serverName]
      .filter((entry): entry is string => entry !== undefined)
      .flatMap((serverKey) => [call.toolName, call.modelFacingToolName]
        .filter((entry): entry is string => entry !== undefined)
        .map((toolKey) => serverToolKey(serverKey, toolKey)));
    const serverScopedMatch = serverScopedKeys
      .map((key) => callableByKey.get(key))
      .find((entry): entry is McpRuntimeCallableTool => entry !== undefined);
    if (serverScopedMatch !== undefined) {
      return serverScopedMatch;
    }

    const keys = [
      call.modelFacingToolId,
      call.modelFacingToolName,
      call.canonicalToolId,
      call.renderedToolId,
      call.toolName,
    ];
    const key = keys.find((entry): entry is string => entry !== undefined);
    return key === undefined ? undefined : callableByKey.get(key);
  };

  const hasMissingServer = (call: McpRuntimeToolCall): boolean => {
    const requestedServerKeys = [call.serverId, call.serverName]
      .filter((entry): entry is string => entry !== undefined);
    return requestedServerKeys.length > 0 && !requestedServerKeys.some((serverKey) => serverKeys.has(serverKey));
  };

  return {
    callableTools,
    executeToolCall: async (call): Promise<McpRuntimeToolResult> => {
      const callId = call.callId ?? createCallId();
      const retryCount = boundedInteger(call.retryCount, 0, 0, Number.MAX_SAFE_INTEGER);
      const effectiveTimeoutMs = boundedOptionalInteger(call.timeoutMs, 1, MAX_TOOL_TIMEOUT_MS) ?? defaultTimeoutMs;
      const startedAt = now();
      const argumentShapeHash = argumentShapeHashFor(call.arguments);
      const rawArgumentBytes = byteLength(stableJson(toJsonValue(call.arguments) ?? null));
      const missingServer = hasMissingServer(call);
      const callable = missingServer ? undefined : lookupCallable(call);

      if (callable === undefined) {
        return createResult({
          status: missingServer ? "missing_server" : "unknown_tool",
          callId,
          argumentShapeHash,
          metrics: zeroMetrics(elapsedMs(now, startedAt), rawArgumentBytes, retryCount),
          error: {
            class: missingServer ? "missing_server" : "unknown_tool",
            code: missingServer ? "missing_server" : "unknown_tool",
            message: missingServer
              ? "MCP runtime bridge could not find the requested server."
              : "MCP runtime bridge could not find the requested tool.",
            details: {
              serverId: call.serverId ?? null,
              serverName: call.serverName ?? null,
              canonicalToolId: call.canonicalToolId ?? null,
              modelFacingToolId: call.modelFacingToolId ?? null,
              modelFacingToolName: call.modelFacingToolName ?? null,
              renderedToolId: call.renderedToolId ?? null,
              toolName: call.toolName ?? null,
            },
          },
        });
      }

      const argumentValidation = validateRuntimeArguments(
        call.arguments,
        callable.inputSchema,
        callable.policy.argumentMaxBytes,
      );
      if (!argumentValidation.ok) {
        return createResult({
          status: "invalid_arguments",
          callId,
          callable,
          policyDecision: preliminaryPolicyDecision(callable, mode),
          argumentShapeHash,
          metrics: zeroMetrics(elapsedMs(now, startedAt), argumentValidation.argumentBytes, retryCount),
          error: {
            class: "invalid_arguments",
            code: argumentValidation.failureCode,
            message: argumentValidation.message,
            details: argumentValidation.details,
          },
        });
      }

      if (retryBudgetConfigured && retryCount > maxRetryCount) {
        return createResult({
          status: "retry_exhausted",
          callId,
          callable,
          policyDecision: preliminaryPolicyDecision(callable, mode),
          argumentShapeHash,
          metrics: zeroMetrics(elapsedMs(now, startedAt), argumentValidation.argumentBytes, retryCount),
          error: {
            class: "retry_exhausted",
            code: "retry_exhausted",
            message: `MCP tool call retry count ${retryCount} exceeds the configured maximum ${maxRetryCount}.`,
            details: {
              retryCount,
              maxRetryCount,
            },
          },
        });
      }

      let policyDecision: McpRuntimePolicyDecision;
      try {
        policyDecision = await decidePolicy({
          callable,
          callId,
          arguments: argumentValidation.arguments,
          mode,
          ...(input.permissionHandler === undefined ? {} : { permissionHandler: input.permissionHandler }),
        });
      } catch (error) {
        return createResult({
          status: "denied",
          callId,
          callable,
          policyDecision: preliminaryPolicyDecision(callable, mode),
          argumentShapeHash,
          metrics: zeroMetrics(elapsedMs(now, startedAt), argumentValidation.argumentBytes, retryCount),
          error: {
            class: "permission_error",
            code: "permission_error",
            message: error instanceof Error ? error.message : "MCP permission handler failed.",
          },
        });
      }

      if (policyDecision.permissionStatus === "blocked" || policyDecision.permissionStatus === "denied") {
        return createResult({
          status: "denied",
          callId,
          callable,
          policyDecision,
          argumentShapeHash,
          metrics: zeroMetrics(elapsedMs(now, startedAt), argumentValidation.argumentBytes, retryCount),
          error: {
            class: policyDecision.permissionStatus === "blocked" ? "policy_blocked" : "permission_denied",
            code: policyDecision.permissionStatus === "blocked" ? "policy_blocked" : "permission_denied",
            message: policyDecision.permissionStatus === "blocked"
              ? "MCP tool call was blocked by side-effect policy."
              : "MCP tool call requires permission and was not approved.",
            details: {
              mode,
              action: policyDecision.action,
              sideEffectLevel: policyDecision.sideEffectLevel,
            },
          },
        });
      }

      const runtimeSignal = runtimeSignalFor(call.signal, effectiveTimeoutMs);
      try {
        throwIfAborted(runtimeSignal.signal);
        const executionRequest: McpRuntimeToolExecutionRequest = {
          callId,
          serverId: callable.serverId,
          serverName: callable.serverName,
          toolName: callable.toolName,
          canonicalToolId: callable.canonicalToolId,
          modelFacingToolId: callable.modelFacingToolId,
          modelFacingToolName: callable.modelFacingToolName,
          ...(callable.renderedToolId === undefined ? {} : { renderedToolId: callable.renderedToolId }),
          ...(callable.renderedToolName === undefined ? {} : { renderedToolName: callable.renderedToolName }),
          arguments: argumentValidation.arguments,
          ...(call.retryCount === undefined ? {} : { retryCount }),
          ...(effectiveTimeoutMs === undefined ? {} : { timeoutMs: effectiveTimeoutMs }),
          ...(runtimeSignal.signal === undefined ? {} : { signal: runtimeSignal.signal }),
        };
        const rawResult = await runWithCancellation(Promise.resolve(input.executor(executionRequest)), runtimeSignal.signal);
        const resultValue = toRuntimeResultValue(rawResult);
        const bounded = boundResultValue(resultValue, callable.policy.resultMaxBytes);

        return createResult({
          status: "success",
          callId,
          callable,
          policyDecision,
          argumentShapeHash,
          metrics: {
            argumentBytes: argumentValidation.argumentBytes,
            resultBytes: bounded.resultBytes,
            resultBytesBeforeBounding: bounded.resultBytesBeforeBounding,
            omittedResultBytes: bounded.omittedResultBytes,
            truncated: bounded.truncated,
            retryCount,
            durationMs: elapsedMs(now, startedAt),
          },
          result: bounded.value,
        });
      } catch (error) {
        const failureCode: McpRuntimeFailureCode = isTimeoutLikeError(error)
          ? "timeout"
          : isAbortLikeError(error)
            ? "cancelled"
            : isStaleContextLikeError(error)
              ? "stale_context"
              : retryBudgetConfigured && retryCount >= maxRetryCount
                ? "retry_exhausted"
                : "runtime_exception";
        const status: McpRuntimeToolStatus = failureCode === "timeout"
          ? "timeout"
          : failureCode === "cancelled"
            ? "cancelled"
            : failureCode === "retry_exhausted"
              ? "retry_exhausted"
              : "error";
        return createResult({
          status,
          callId,
          callable,
          policyDecision,
          argumentShapeHash,
          metrics: zeroMetrics(elapsedMs(now, startedAt), argumentValidation.argumentBytes, retryCount),
          error: {
            class: failureCode === "runtime_exception" ? "execution_error" : failureCode,
            code: failureCode,
            message: error instanceof Error ? error.message : "MCP tool execution failed.",
          },
        });
      } finally {
        runtimeSignal.cleanup();
      }
    },
  };
};
