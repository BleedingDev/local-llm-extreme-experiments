/**
 * Opt-in `LlmRouter` backend backed by `@mariozechner/pi-ai`.
 *
 * Same public shape as the router from `src/llm.ts` — `masterAvailable`,
 * `localAvailable`, `chatText`, `chatTextWithTools`. Routing semantics
 * (purpose tagging, role-based selection, fallback chain, telemetry) are
 * preserved verbatim. Only the *transport* changes: instead of hand-rolled
 * `/chat/completions`, `/responses`, and `/v1/messages` HTTP calls, we hand
 * the request off to pi-ai's unified `complete()` and let it pick the
 * provider-specific wire format.
 *
 * This module never hardcodes a model id, provider id, or base URL — all
 * three flow from `BagConfig.master` / `BagConfig.local`. Pi-ai's `Model`
 * descriptor is constructed on the fly from the resolved role config.
 *
 * Opt in by setting `BAG_USE_PI_AI=1` (the wiring lives in the caller —
 * BAG core still defaults to `createLlmRouter` from `src/llm.ts`).
 */
import type {
  Api,
  AssistantMessage,
  Context as PiContext,
  ImageContent,
  Message as PiMessage,
  Model as PiModel,
  ProviderStreamOptions,
  TextContent,
  Tool as PiTool,
  ToolCall as PiToolCall,
  ToolResultMessage as PiToolResultMessage,
} from "@mariozechner/pi-ai";
import { complete as piComplete } from "@mariozechner/pi-ai";
import { resolveLocalApiKey, resolveMasterApiKey, resolveModelRoleConfig } from "./config";
import type {
  AssistantWithToolCalls,
  ChatContentBlock,
  ChatMessage,
  ChatOptions,
  ChatWithToolsOptions,
  ChatWithToolsResult,
  LlmRouter,
  ToolCall,
  ToolDefinition,
  ToolResultMessage,
  ToolUseTurnMessage,
} from "./llm";
import type {
  BagConfig,
  LlmCallMetric,
  ModelEndpointKind,
  ModelProvider,
  ModelProviderConfigRole,
  ModelRuntimeRole,
} from "./types";

/* ------------------------------------------------------------------------ */
/* Telemetry sink — same contract as `src/llm.ts` so a host that swaps the  */
/* router doesn't have to change anything else.                             */
/* ------------------------------------------------------------------------ */
type LlmTelemetrySink = {
  recordLlmCall: (metric: LlmCallMetric) => void;
};

type CompleteFn = typeof piComplete;

export interface CreateLlmRouterFromPiAiOptions {
  /** Optional injection point for tests — defaults to pi-ai's `complete`. */
  complete?: CompleteFn;
  telemetry?: LlmTelemetrySink;
}

/* ------------------------------------------------------------------------ */
/* BAG-provider → pi-ai-Api/Provider translation. Generic — *no* keyword    */
/* whitelist of model names. Wire format is decided by the BAG provider     */
/* enum. Anyone wiring a new provider just adds it to BAG's enum + the      */
/* table below.                                                             */
/* ------------------------------------------------------------------------ */
const apiForBagProvider = (provider: ModelProvider, endpointKind: ModelEndpointKind): Api => {
  if (provider === "anthropic") return "anthropic-messages";
  if (endpointKind === "responses") return "openai-responses";
  // openai, openai-compatible, vllm, llama.cpp, local-mlx, ollama, custom →
  // they all speak the OpenAI completions wire format (vLLM, LM Studio,
  // Ollama-OpenAI-bridge, Together, Fireworks, OpenRouter, ...).
  return "openai-completions";
};

const piProviderForBagProvider = (provider: ModelProvider): string => {
  // pi-ai's `Provider` is `KnownProvider | string`, so unknown values are
  // legal. We surface BAG's provider id so telemetry / pi-ai's compat
  // detection (which inspects `model.provider` and `model.baseUrl`) sees a
  // stable label.
  switch (provider) {
    case "anthropic":
      return "anthropic";
    case "openai":
      return "openai";
    case "ollama":
      return "ollama";
    case "vllm":
      return "vllm";
    case "llama.cpp":
      return "llama.cpp";
    case "local-mlx":
      return "local-mlx";
    case "openai-compatible":
    case "custom":
      return "openai-compatible";
  }
};

/* ------------------------------------------------------------------------ */
/* Multimodal pass-through.                                                 */
/*                                                                          */
/* BAG's `ChatContentBlock` is the OpenAI-vision shape (`text` /            */
/* `image_url` with a `data:` URL or http(s) URL). Pi-ai's blocks are       */
/* `text` / `image` (raw base64 + mimeType). We map between them losslessly */
/* for `data:` URLs (the `view_image` tool always emits `data:` URLs).      */
/*                                                                          */
/* For plain http(s) image URLs we keep the URL alive as a text marker —    */
/* pi-ai providers do not all support remote URL fetch, but the agent's     */
/* native flow always materialises images locally first via `view_image`.   */
/* ------------------------------------------------------------------------ */
const DATA_URL_PATTERN = /^data:([^;]+);base64,(.+)$/;

const blockToPi = (block: ChatContentBlock): TextContent | ImageContent => {
  if (block.type === "text") return { type: "text", text: block.text };
  const match = DATA_URL_PATTERN.exec(block.image_url.url);
  if (match != null && match[1] != null && match[2] != null) {
    return { type: "image", mimeType: match[1], data: match[2] };
  }
  // Non-data URL: keep as text reference so the model still "sees" the
  // pointer. Providers that support remote images can be wired by
  // upgrading this branch later.
  return { type: "text", text: `[image_url] ${block.image_url.url}` };
};

const contentToPi = (content: ChatMessage["content"]): string | (TextContent | ImageContent)[] => {
  if (typeof content === "string") return content;
  return content.map(blockToPi);
};

/* ------------------------------------------------------------------------ */
/* Build a pi-ai `Model<Api>` from a resolved BAG role. All knobs come from */
/* config — no hardcoding.                                                  */
/* ------------------------------------------------------------------------ */
type SelectedModelRole = {
  requestedRole: ModelRuntimeRole;
  resolvedRole: ModelRuntimeRole;
  providerConfigRole: ModelProviderConfigRole;
  fallbackFromRole?: ModelRuntimeRole;
  provider: ModelProvider;
  baseUrl: string;
  endpointKind: ModelEndpointKind;
  modelServerId: string;
  modelServerProfileId: string;
  contextWindowTokens: number;
  apiKey?: string;
  model: string;
  maxTokens: number;
  maxOutputTokens: number;
  temperature: number;
};

const piModelFor = (selected: SelectedModelRole): PiModel<Api> => {
  const api = apiForBagProvider(selected.provider, selected.endpointKind);
  return {
    id: selected.model,
    name: selected.model,
    api,
    provider: piProviderForBagProvider(selected.provider),
    baseUrl: selected.baseUrl,
    reasoning: false,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: selected.contextWindowTokens,
    maxTokens: selected.maxOutputTokens,
  };
};

/* ------------------------------------------------------------------------ */
/* JSON-Schema parameter shape → TypeBox-compatible `TSchema`.              */
/*                                                                          */
/* pi-ai's `Tool.parameters: TSchema`. At runtime, a TSchema is just a      */
/* plain JSON-Schema object with a `Symbol(TypeBox.Kind)` tag (used only    */
/* by the validator). Providers (OpenAI, Anthropic, Google) all serialise  */
/* the parameters as JSON-Schema, so the unsafe cast is sound for the      */
/* call path. We do not invoke pi-ai's runtime validator on inbound BAG    */
/* tool calls — BAG already validates downstream.                          */
/* ------------------------------------------------------------------------ */
const toolToPi = (tool: ToolDefinition): PiTool => ({
  name: tool.function.name,
  description: tool.function.description,
  // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
  parameters: tool.function.parameters as unknown as PiTool["parameters"],
});

/* ------------------------------------------------------------------------ */
/* BAG turn message → pi-ai message.                                        */
/*                                                                          */
/* BAG's `ToolUseTurnMessage` is one of:                                    */
/*   - `ChatMessage` (system|user|assistant + string|blocks)                */
/*   - `AssistantWithToolCalls`                                             */
/*   - `ToolResultMessage`                                                  */
/* pi-ai's `Message` is `UserMessage | AssistantMessage | ToolResultMessage`*/
/* with `system` carried separately as `Context.systemPrompt`.              */
/* ------------------------------------------------------------------------ */
const piTimestamp = (): number => Date.now();

const isAssistantWithToolCalls = (m: ToolUseTurnMessage): m is AssistantWithToolCalls =>
  (m as AssistantWithToolCalls).role === "assistant" &&
  Array.isArray((m as AssistantWithToolCalls).tool_calls);

const isToolResult = (m: ToolUseTurnMessage): m is ToolResultMessage =>
  (m as ToolResultMessage).role === "tool";

const safeParseJson = (text: string): Record<string, unknown> => {
  try {
    const parsed = JSON.parse(text === "" ? "{}" : text);
    return parsed != null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
};

const turnsToPiContext = (
  selected: SelectedModelRole,
  messages: ToolUseTurnMessage[],
  tools?: ToolDefinition[],
): PiContext => {
  let systemPrompt: string | undefined;
  const piMessages: PiMessage[] = [];

  for (const msg of messages) {
    if (isToolResult(msg)) {
      const result: PiToolResultMessage = {
        role: "toolResult",
        toolCallId: msg.tool_call_id,
        toolName: "", // BAG's ToolResultMessage doesn't preserve name; pi-ai accepts empty
        content: [{ type: "text", text: msg.content }],
        isError: false,
        timestamp: piTimestamp(),
      };
      piMessages.push(result);
      continue;
    }
    if (isAssistantWithToolCalls(msg)) {
      const blocks: (TextContent | PiToolCall)[] = [];
      if (typeof msg.content === "string" && msg.content.length > 0) {
        blocks.push({ type: "text", text: msg.content });
      }
      for (const tc of msg.tool_calls) {
        blocks.push({
          type: "toolCall",
          id: tc.id,
          name: tc.function.name,
          arguments: safeParseJson(tc.function.arguments),
        });
      }
      const assistant: AssistantMessage = {
        role: "assistant",
        content: blocks,
        api: apiForBagProvider(selected.provider, selected.endpointKind),
        provider: piProviderForBagProvider(selected.provider),
        model: selected.model,
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: piTimestamp(),
      };
      piMessages.push(assistant);
      continue;
    }
    // Plain ChatMessage
    const cm = msg satisfies ChatMessage;
    if (cm.role === "system") {
      const text = typeof cm.content === "string"
        ? cm.content
        : cm.content.filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join("\n");
      systemPrompt = systemPrompt == null || systemPrompt === "" ? text : `${systemPrompt}\n\n${text}`;
      continue;
    }
    if (cm.role === "user") {
      piMessages.push({
        role: "user",
        content: contentToPi(cm.content),
        timestamp: piTimestamp(),
      });
      continue;
    }
    // Plain assistant text (no tool calls)
    const text = typeof cm.content === "string" ? cm.content : cm.content
      .filter((b) => b.type === "text")
      .map((b) => (b as { text: string }).text)
      .join("");
    piMessages.push({
      role: "assistant",
      content: [{ type: "text", text }],
      api: apiForBagProvider(selected.provider, selected.endpointKind),
      provider: piProviderForBagProvider(selected.provider),
      model: selected.model,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: piTimestamp(),
    });
  }

  const ctx: PiContext = { messages: piMessages };
  if (systemPrompt != null && systemPrompt !== "") ctx.systemPrompt = systemPrompt;
  if (tools != null && tools.length > 0) ctx.tools = tools.map(toolToPi);
  return ctx;
};

/* ------------------------------------------------------------------------ */
/* AssistantMessage → BAG ChatWithToolsResult.                              */
/* ------------------------------------------------------------------------ */
const assistantToToolResult = (msg: AssistantMessage): ChatWithToolsResult => {
  const text = msg.content
    .filter((b): b is TextContent => b.type === "text")
    .map((b) => b.text)
    .join("");
  const toolCalls: ToolCall[] = msg.content
    .filter((b): b is PiToolCall => b.type === "toolCall")
    .map((tc) => ({
      id: tc.id,
      name: tc.name,
      argumentsJson: JSON.stringify(tc.arguments ?? {}),
    }));
  const finishReason = msg.stopReason === "toolUse"
    ? "tool_calls"
    : msg.stopReason === "length"
      ? "length"
      : msg.stopReason === "aborted" || msg.stopReason === "error"
        ? msg.stopReason
        : "stop";
  return {
    finishReason,
    textContent: text,
    toolCalls,
    ...(msg.usage?.input != null ? { promptTokens: msg.usage.input } : {}),
    ...(msg.usage?.output != null ? { completionTokens: msg.usage.output } : {}),
  };
};

const assistantToText = (msg: AssistantMessage): string =>
  msg.content
    .filter((b): b is TextContent => b.type === "text")
    .map((b) => b.text)
    .join("");

/* ------------------------------------------------------------------------ */
/* Public factory.                                                          */
/* ------------------------------------------------------------------------ */
const now = (): string => new Date().toISOString();

export const createLlmRouterFromPiAi = (
  config: BagConfig,
  options: CreateLlmRouterFromPiAiOptions = {},
): LlmRouter => {
  const completeFn: CompleteFn = options.complete ?? piComplete;
  const telemetry = options.telemetry;

  const masterApiKey = resolveMasterApiKey(config);
  const localApiKey = resolveLocalApiKey(config);

  const apiKeyFor = (providerConfigRole: ModelProviderConfigRole): string | undefined =>
    providerConfigRole === "master" ? masterApiKey : localApiKey;

  const selectRole = (
    requestedRole: ModelRuntimeRole,
    role: ModelRuntimeRole,
    visited: ReadonlySet<ModelRuntimeRole> = new Set(),
    fallbackFromRole?: ModelRuntimeRole,
  ): SelectedModelRole => {
    const roleConfig = resolveModelRoleConfig(config, role);
    const apiKey = apiKeyFor(roleConfig.providerConfigRole);
    if (
      (apiKey == null || apiKey === "") &&
      roleConfig.fallbackModelRole !== undefined &&
      !visited.has(roleConfig.fallbackModelRole)
    ) {
      return selectRole(
        requestedRole,
        roleConfig.fallbackModelRole,
        new Set([...visited, role]),
        role,
      );
    }
    return {
      requestedRole,
      resolvedRole: role,
      providerConfigRole: roleConfig.providerConfigRole,
      ...(fallbackFromRole === undefined ? {} : { fallbackFromRole }),
      provider: roleConfig.provider,
      baseUrl: roleConfig.baseUrl,
      endpointKind: roleConfig.endpointKind,
      modelServerId: roleConfig.modelServerId,
      modelServerProfileId: roleConfig.modelServerProfileId,
      contextWindowTokens: roleConfig.contextWindowTokens,
      ...(apiKey === undefined ? {} : { apiKey }),
      model: roleConfig.model,
      maxTokens: roleConfig.maxTokens,
      maxOutputTokens: roleConfig.maxOutputTokens,
      temperature: roleConfig.temperature,
    };
  };

  const metricRoleFields = (
    selected: SelectedModelRole,
    purpose?: string,
  ): Pick<
    LlmCallMetric,
    | "role"
    | "resolvedRole"
    | "providerConfigRole"
    | "fallbackFromRole"
    | "provider"
    | "endpointKind"
    | "modelServerId"
    | "modelServerProfileId"
    | "contextWindowTokens"
    | "maxOutputTokens"
    | "purpose"
  > => ({
    role: selected.requestedRole,
    resolvedRole: selected.resolvedRole,
    providerConfigRole: selected.providerConfigRole,
    ...(selected.fallbackFromRole === undefined ? {} : { fallbackFromRole: selected.fallbackFromRole }),
    provider: selected.provider,
    endpointKind: selected.endpointKind,
    modelServerId: selected.modelServerId,
    modelServerProfileId: selected.modelServerProfileId,
    contextWindowTokens: selected.contextWindowTokens,
    maxOutputTokens: selected.maxOutputTokens,
    ...(purpose === undefined || purpose.length === 0 ? {} : { purpose }),
  });

  const piEndpointLabel = (selected: SelectedModelRole): string => {
    const api = apiForBagProvider(selected.provider, selected.endpointKind);
    return `pi-ai://${api}/${selected.baseUrl}`;
  };

  const runChat = async (options: ChatOptions): Promise<string> => {
    const selected = selectRole(options.role, options.role);
    const startedAt = now();
    const startedMs = performance.now();
    const endpointLabel = piEndpointLabel(selected);

    if (selected.apiKey == null || selected.apiKey === "") {
      const error = `${selected.resolvedRole} model is unavailable: missing API key`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: endpointLabel,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error,
      });
      throw new Error(error);
    }

    const piModel = piModelFor(selected);
    const ctx = turnsToPiContext(selected, options.messages);
    const piOptions: ProviderStreamOptions = {
      apiKey: selected.apiKey,
      maxTokens: options.maxTokens ?? selected.maxTokens,
      temperature: options.temperature ?? selected.temperature,
    };

    let result: AssistantMessage;
    try {
      result = await completeFn(piModel, ctx, piOptions);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: endpointLabel,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error: message,
      });
      throw error;
    }

    if (result.stopReason === "error" || result.stopReason === "aborted") {
      const error = result.errorMessage ?? `pi-ai stopped with reason ${result.stopReason}`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: endpointLabel,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error,
        ...(result.usage?.input != null ? { promptTokens: result.usage.input } : {}),
        ...(result.usage?.output != null ? { completionTokens: result.usage.output } : {}),
        ...(result.usage?.totalTokens != null ? { totalTokens: result.usage.totalTokens } : {}),
      });
      throw new Error(error);
    }
    telemetry?.recordLlmCall({
      ...metricRoleFields(selected, options.purpose),
      model: selected.model,
      endpoint: endpointLabel,
      startedAt,
      completedAt: now(),
      durationMs: Math.round(performance.now() - startedMs),
      ok: true,
      ...(result.usage?.input != null ? { promptTokens: result.usage.input } : {}),
      ...(result.usage?.output != null ? { completionTokens: result.usage.output } : {}),
      ...(result.usage?.totalTokens != null ? { totalTokens: result.usage.totalTokens } : {}),
    });
    return assistantToText(result);
  };

  const runChatWithTools = async (options: ChatWithToolsOptions): Promise<ChatWithToolsResult> => {
    const selected = selectRole(options.role, options.role);
    const startedAt = now();
    const startedMs = performance.now();
    const endpointLabel = piEndpointLabel(selected);

    if (selected.apiKey == null || selected.apiKey === "") {
      const error = `${selected.resolvedRole} model is unavailable: missing API key`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: endpointLabel,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error,
      });
      throw new Error(error);
    }

    const piModel = piModelFor(selected);
    const ctx = turnsToPiContext(selected, options.messages, options.tools);
    const piOptions: ProviderStreamOptions = {
      apiKey: selected.apiKey,
      maxTokens: options.maxTokens ?? selected.maxTokens,
      temperature: options.temperature ?? selected.temperature,
    };

    let result: AssistantMessage;
    try {
      result = await completeFn(piModel, ctx, piOptions);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: endpointLabel,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error: message,
      });
      throw error;
    }

    if (result.stopReason === "error" || result.stopReason === "aborted") {
      const error = result.errorMessage ?? `pi-ai stopped with reason ${result.stopReason}`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: endpointLabel,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error,
        ...(result.usage?.input != null ? { promptTokens: result.usage.input } : {}),
        ...(result.usage?.output != null ? { completionTokens: result.usage.output } : {}),
        ...(result.usage?.totalTokens != null ? { totalTokens: result.usage.totalTokens } : {}),
      });
      throw new Error(error);
    }
    telemetry?.recordLlmCall({
      ...metricRoleFields(selected, options.purpose),
      model: selected.model,
      endpoint: endpointLabel,
      startedAt,
      completedAt: now(),
      durationMs: Math.round(performance.now() - startedMs),
      ok: true,
      ...(result.usage?.input != null ? { promptTokens: result.usage.input } : {}),
      ...(result.usage?.output != null ? { completionTokens: result.usage.output } : {}),
      ...(result.usage?.totalTokens != null ? { totalTokens: result.usage.totalTokens } : {}),
    });
    return assistantToToolResult(result);
  };

  return {
    masterAvailable: masterApiKey != null,
    localAvailable: async () => {
      if (localApiKey == null || localApiKey === "") return false;
      // Pi-ai does not expose a generic health-check; defer to the same
      // shape as `src/llm.ts` so the public contract stays identical.
      if (config.local.provider === "anthropic") return true;
      try {
        const path = config.local.provider === "ollama" ? "/api/tags" : "/models";
        const url = `${config.local.baseUrl.replace(/\/$/, "")}${path}`;
        const response = await fetch(url, {
          headers: config.local.provider === "ollama"
            ? {}
            : { authorization: `Bearer ${localApiKey}` },
        });
        return response.ok;
      } catch {
        return false;
      }
    },
    chatText: runChat,
    chatTextWithTools: runChatWithTools,
  };
};

/**
 * Returns true when the caller opted into the pi-ai router via the
 * `BAG_USE_PI_AI` environment variable. Centralised so callers don't
 * sprinkle env-var literals across the codebase.
 */
export const piAiRouterEnabled = (env: NodeJS.ProcessEnv = process.env): boolean => {
  const raw = env.BAG_USE_PI_AI?.trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
};
