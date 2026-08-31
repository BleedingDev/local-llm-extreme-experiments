import { AxJSRuntime, AxJSRuntimePermission, agent, ai, type AxAIService } from "@ax-llm/ax";
import { getOAuthHeaders, type OAuthProviderId } from "./auth/oauth-flows";
import { resolveLocalApiKey, resolveMasterApiKey, resolveModelRoleConfig } from "./config";
import type {
  BagConfig,
  LlmCallMetric,
  ModelEndpointKind,
  ModelProvider,
  ModelProviderConfigRole,
  ModelRuntimeRole,
} from "./types";

/**
 * Multimodal content block (OpenAI vision-compat spec). Anthropic's
 * `/v1/chat/completions` compat endpoint accepts these in user-message
 * `content` arrays so the master model can natively perceive images
 * (chess board screenshots, UI mockups, doc page captures, etc.).
 */
export type ChatContentBlock =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: "auto" | "low" | "high" } };

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  /** Either plain text (the common case) OR a multimodal content array. */
  content: string | ChatContentBlock[];
};

export type ChatOptions = {
  role: ModelRuntimeRole;
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  json?: boolean;
  /**
   * Free-form attribution for which BAG subsystem made the call (e.g.
   * "instruction-summarizer", "task-shape-classifier", "probe-extractor",
   * "pre-submit-self-check", "dag-planner", "autonomous-coding-turn"). Logged
   * into the LlmCallMetric so per-model optimisation can group on (model,
   * purpose). BAG core never reads it; it is observability metadata only.
   */
  purpose?: string;
};

export type ToolDefinition = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
};

export type ToolCall = {
  id: string;
  name: string;
  argumentsJson: string;
};

export type ToolResultMessage = {
  role: "tool";
  tool_call_id: string;
  content: string;
};

export type AssistantWithToolCalls = {
  role: "assistant";
  content: string | null;
  tool_calls: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
};

export type ToolUseTurnMessage = ChatMessage | AssistantWithToolCalls | ToolResultMessage;

export type ChatWithToolsOptions = {
  role: ModelRuntimeRole;
  messages: ToolUseTurnMessage[];
  tools: ToolDefinition[];
  maxTokens?: number;
  temperature?: number;
  toolChoice?: "auto" | "required" | "none";
  /** See `ChatOptions.purpose`. */
  purpose?: string;
};

export type ChatWithToolsResult = {
  finishReason: string;
  textContent: string;
  toolCalls: ToolCall[];
  promptTokens?: number;
  completionTokens?: number;
};

export type LlmRouter = {
  masterAvailable: boolean;
  localAvailable: () => Promise<boolean>;
  chatText: (options: ChatOptions) => Promise<string>;
  chatTextWithTools: (options: ChatWithToolsOptions) => Promise<ChatWithToolsResult>;
};

type LlmTelemetrySink = {
  recordLlmCall: (metric: LlmCallMetric) => void;
};

type ChatCompletionUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

const endpoint = (baseUrl: string, path: string): string =>
  `${baseUrl.replace(/\/$/, "")}${path.startsWith("/") ? path : `/${path}`}`;

const now = (): string => new Date().toISOString();

type LlmResponseParser<T> = (payload: unknown) => T;

type LlmHttpRequest<T> = {
  endpoint: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
  parse: LlmResponseParser<T>;
};

type TextResponse = {
  text: string;
  usage?: ChatCompletionUsage;
};

const isAnthropicProvider = (provider: ModelProvider): boolean => provider === "anthropic";

/**
 * Detects whether a request will hit an Anthropic-served endpoint, regardless
 * of which provider config is selected (the OpenAI-compat layer at
 * `https://api.anthropic.com/v1/...` is reachable via either the "openai" or
 * "anthropic" provider entries in BagConfig). Anthropic's Claude 4.x family
 * (Opus 4.7 / Sonnet 4.6 / Haiku 4.5) rejects the `temperature` parameter
 * with `temperature is deprecated for this model`, so we omit it for any
 * Anthropic-served call.
 */
const requestHitsAnthropic = (selected: SelectedModelRole): boolean =>
  isAnthropicProvider(selected.provider) ||
  /(^|\/\/)([^/]*\.)?anthropic\.com/i.test(selected.baseUrl);

const normalizedUsage = (
  promptTokens?: number,
  completionTokens?: number,
  totalTokens?: number,
): ChatCompletionUsage => ({
  ...(promptTokens == null ? {} : { prompt_tokens: promptTokens }),
  ...(completionTokens == null ? {} : { completion_tokens: completionTokens }),
  ...(totalTokens == null ? {} : { total_tokens: totalTokens }),
});

const authHeadersFor = (selected: SelectedModelRole): Record<string, string> => {
  // OAuth branch: when the role was resolved from a `master.authType: "oauth"`
  // (or local) config, `oauthHeaders` was prefetched by selectRole and any
  // refresh/persist already happened. Merge those headers verbatim — they
  // already contain Authorization (Bearer), provider beta flags
  // (anthropic-beta, OpenAI-Beta, chatgpt-account-id, Copilot-Integration-Id, ...).
  if (selected.oauthHeaders != null) {
    return { "content-type": "application/json", ...selected.oauthHeaders };
  }
  return isAnthropicProvider(selected.provider)
    ? {
        "content-type": "application/json",
        "x-api-key": selected.apiKey ?? "",
        "anthropic-version": "2023-06-01",
      }
    : {
        "content-type": "application/json",
        authorization: `Bearer ${selected.apiKey ?? ""}`,
      };
};

const chatCompletionsTextRequest = (selected: SelectedModelRole, options: ChatOptions): LlmHttpRequest<TextResponse> => {
  const skipTemperature = requestHitsAnthropic(selected);
  // Anthropic compat also rejects response_format=json_object; the parser path
  // (parseJsonObject) handles raw text robustly, so it's safe to omit there.
  const skipResponseFormat = skipTemperature;
  return {
    endpoint: endpoint(selected.baseUrl, "/chat/completions"),
    headers: authHeadersFor(selected),
    body: {
      model: selected.model,
      messages: options.messages,
      max_tokens: options.maxTokens ?? selected.maxTokens,
      ...(skipTemperature ? {} : { temperature: options.temperature ?? selected.temperature }),
      ...(options.json === true && !skipResponseFormat ? { response_format: { type: "json_object" } } : {}),
    },
    parse: (payload: unknown) => {
      const parsed = payload as {
        choices?: Array<{ message?: { content?: string } }>;
        usage?: ChatCompletionUsage;
      };
      return {
        text: parsed.choices?.[0]?.message?.content ?? "",
        ...(parsed.usage == null ? {} : { usage: parsed.usage }),
      };
    },
  };
};

const responsesTextRequest = (selected: SelectedModelRole, options: ChatOptions): LlmHttpRequest<TextResponse> => ({
  endpoint: endpoint(selected.baseUrl, "/responses"),
  headers: authHeadersFor(selected),
  body: {
    model: selected.model,
    input: options.messages.map((message) => ({ role: message.role, content: message.content })),
    max_output_tokens: options.maxTokens ?? selected.maxTokens,
    ...(requestHitsAnthropic(selected)
      ? {}
      : { temperature: options.temperature ?? selected.temperature }),
    ...(options.json === true ? { text: { format: { type: "json_object" } } } : {}),
  },
  parse: (payload: unknown) => {
    const parsed = payload as {
      output_text?: string;
      output?: Array<{ content?: Array<{ text?: string; type?: string }> }>;
      usage?: {
        input_tokens?: number;
        output_tokens?: number;
        total_tokens?: number;
      };
    };
    const fromOutput = parsed.output
      ?.flatMap((entry) => entry.content ?? [])
      .map((content) => content.text)
      .filter((text): text is string => typeof text === "string")
      .join("");
    const usage = parsed.usage == null
      ? undefined
      : normalizedUsage(parsed.usage.input_tokens, parsed.usage.output_tokens, parsed.usage.total_tokens);
    return {
      text: parsed.output_text ?? fromOutput ?? "",
      ...(usage == null ? {} : { usage }),
    };
  },
});

const anthropicTextRequest = (selected: SelectedModelRole, options: ChatOptions): LlmHttpRequest<TextResponse> => {
  const system = options.messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n\n");
  const messages = options.messages
    .filter((message) => message.role !== "system")
    .map((message) => ({ role: message.role, content: message.content }));
  return {
    endpoint: endpoint(selected.baseUrl, "/messages"),
    headers: authHeadersFor(selected),
    body: {
      model: selected.model,
      messages,
      max_tokens: options.maxTokens ?? selected.maxTokens,
      // Claude 4.x family rejects `temperature` on the native /messages
      // endpoint; omit unconditionally for the anthropic provider.
      ...(system === "" ? {} : { system }),
    },
    parse: (payload: unknown) => {
      const parsed = payload as {
        content?: Array<{ type?: string; text?: string }>;
        usage?: { input_tokens?: number; output_tokens?: number };
      };
      const text = (parsed.content ?? [])
        .map((content) => content.text)
        .filter((part): part is string => typeof part === "string")
        .join("");
      const usage = parsed.usage == null
        ? undefined
        : normalizedUsage(
            parsed.usage.input_tokens,
            parsed.usage.output_tokens,
            parsed.usage.input_tokens == null || parsed.usage.output_tokens == null
              ? undefined
              : parsed.usage.input_tokens + parsed.usage.output_tokens,
          );
      return {
        text,
        ...(usage == null ? {} : { usage }),
      };
    },
  };
};

const textRequestFor = (selected: SelectedModelRole, options: ChatOptions): LlmHttpRequest<TextResponse> => {
  if (isAnthropicProvider(selected.provider)) {
    return anthropicTextRequest(selected, options);
  }
  if (selected.endpointKind === "responses") {
    return responsesTextRequest(selected, options);
  }
  return chatCompletionsTextRequest(selected, options);
};

const toolUseSupportedBy = (selected: SelectedModelRole): boolean =>
  selected.endpointKind === "chat_completions" && !isAnthropicProvider(selected.provider);

const axProviderNameFor = (provider: ModelProvider): string => {
  switch (provider) {
    case "anthropic":
      return "anthropic";
    case "ollama":
      return "ollama";
    case "openai":
    case "openai-compatible":
    case "local-mlx":
    case "vllm":
    case "llama.cpp":
    case "custom":
      return "openai";
  }
};

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
  /**
   * When the underlying provider config requested OAuth, headers are
   * pre-resolved here (Authorization + provider-specific beta flags). When
   * present, authHeadersFor uses these instead of the api-key path.
   */
  oauthHeaders?: Record<string, string>;
};

export const createLlmRouter = (config: BagConfig, telemetry?: LlmTelemetrySink): LlmRouter => {
  const masterApiKey = resolveMasterApiKey(config);
  const localApiKey = resolveLocalApiKey(config);

  const apiKeyFor = (providerConfigRole: ModelProviderConfigRole): string | undefined =>
    providerConfigRole === "master" ? masterApiKey : localApiKey;

  // OAuth-mode roles bypass the api-key resolver entirely: we resolve headers
  // from ~/.bag/oauth/<oauthProvider>.json (refreshing if needed) and stash
  // them on the SelectedModelRole. Anthropic-on-OAuth is the canonical case
  // (Claude Pro/Max users), but the same path supports OpenAI Codex tokens
  // and GitHub Copilot subscriptions for free.
  const isOAuthRole = (providerConfigRole: ModelProviderConfigRole): boolean => {
    const source = providerConfigRole === "master" ? config.master : config.local;
    return source.authType === "oauth";
  };
  const oauthProviderFor = (providerConfigRole: ModelProviderConfigRole): OAuthProviderId | undefined => {
    const source = providerConfigRole === "master" ? config.master : config.local;
    if (source.authType !== "oauth") return undefined;
    if (source.oauthProvider != null) return source.oauthProvider;
    // Default mapping: provider="anthropic" -> "anthropic", "openai" -> "openai".
    if (source.provider === "anthropic") return "anthropic";
    if (source.provider === "openai") return "openai";
    return undefined;
  };
  const resolveAuth = async (selected: SelectedModelRole): Promise<SelectedModelRole> => {
    if (!isOAuthRole(selected.providerConfigRole)) return selected;
    const provider = oauthProviderFor(selected.providerConfigRole);
    if (provider == null) {
      throw new Error(
        `${selected.providerConfigRole} is configured for authType="oauth" but no oauthProvider could be inferred from provider="${selected.provider}". Set master.oauthProvider to one of "anthropic" | "openai" | "github-copilot".`,
      );
    }
    const result = await getOAuthHeaders(provider);
    return {
      ...selected,
      // OAuth tokens always count as "available" for the apiKey-presence check.
      apiKey: selected.apiKey ?? "<oauth>",
      oauthHeaders: result.headers,
      ...(result.baseUrlOverride == null ? {} : { baseUrl: result.baseUrlOverride }),
    };
  };

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

  const request = async (options: ChatOptions): Promise<string> => {
    const selected = await resolveAuth(selectRole(options.role, options.role));
    const startedAt = now();
    const startedMs = performance.now();
    const requestShape = textRequestFor(selected, options);

    if (selected.apiKey == null) {
      const error = `${selected.resolvedRole} model is unavailable: missing API key`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: requestShape.endpoint,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error,
      });
      throw new Error(error);
    }

    try {
      const response = await fetch(requestShape.endpoint, {
        method: "POST",
        headers: requestShape.headers,
        body: JSON.stringify(requestShape.body),
      });

      if (!response.ok) {
        const text = await response.text();
        const error = `${selected.resolvedRole} request failed ${response.status}: ${text.slice(0, 500)}`;
        telemetry?.recordLlmCall({
          ...metricRoleFields(selected, options.purpose),
          model: selected.model,
          endpoint: requestShape.endpoint,
          startedAt,
          completedAt: now(),
          durationMs: Math.round(performance.now() - startedMs),
          ok: false,
          httpStatus: response.status,
          error,
        });
        throw new Error(error);
      }

      const payload = await response.json();
      const parsed = requestShape.parse(payload);
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: requestShape.endpoint,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: true,
        httpStatus: response.status,
        promptTokens: parsed.usage?.prompt_tokens,
        completionTokens: parsed.usage?.completion_tokens,
        totalTokens: parsed.usage?.total_tokens,
      });
      return parsed.text;
    } catch (error) {
      if (error instanceof Error && error.message.includes("request failed")) {
        throw error;
      }
      const message = error instanceof Error ? error.message : String(error);
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: requestShape.endpoint,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error: message,
      });
      throw error;
    }
  };

  const requestWithTools = async (options: ChatWithToolsOptions): Promise<ChatWithToolsResult> => {
    const selected = await resolveAuth(selectRole(options.role, options.role));
    const startedAt = now();
    const startedMs = performance.now();
    const callEndpoint = endpoint(selected.baseUrl, "/chat/completions");
    if (selected.apiKey == null) {
      const error = `${selected.resolvedRole} model is unavailable: missing API key`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: callEndpoint,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error,
      });
      throw new Error(error);
    }
    if (!toolUseSupportedBy(selected)) {
      const error = `${selected.resolvedRole} tool-use is unsupported for provider ${selected.provider} with ${selected.endpointKind}`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: callEndpoint,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error,
      });
      throw new Error(error);
    }
    const body: Record<string, unknown> = {
      model: selected.model,
      messages: options.messages,
      max_tokens: options.maxTokens ?? selected.maxTokens,
      tools: options.tools,
    };
    if (options.toolChoice != null && options.toolChoice !== "auto") {
      body.tool_choice = options.toolChoice;
    }
    // Claude 4.x rejects `temperature` on Anthropic-served endpoints (both
    // native /messages and the OpenAI-compat /chat/completions surface).
    if (!requestHitsAnthropic(selected)) {
      body.temperature = options.temperature ?? selected.temperature;
    }
    let response: Response;
    try {
      response = await fetch(callEndpoint, {
        method: "POST",
        headers: authHeadersFor(selected),
        body: JSON.stringify(body),
      });
    } catch (fetchError) {
      const message = fetchError instanceof Error ? fetchError.message : String(fetchError);
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: callEndpoint,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        error: message,
      });
      throw fetchError;
    }
    if (!response.ok) {
      const text = await response.text();
      const error = `${selected.resolvedRole} tool-use request failed ${response.status}: ${text.slice(0, 500)}`;
      telemetry?.recordLlmCall({
        ...metricRoleFields(selected, options.purpose),
        model: selected.model,
        endpoint: callEndpoint,
        startedAt,
        completedAt: now(),
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        httpStatus: response.status,
        error,
      });
      throw new Error(error);
    }
    const payload = (await response.json()) as {
      choices?: Array<{
        message?: {
          content?: string | null;
          tool_calls?: Array<{ id?: string; type?: string; function?: { name?: string; arguments?: string } }>;
        };
        finish_reason?: string;
      }>;
      usage?: ChatCompletionUsage;
    };
    telemetry?.recordLlmCall({
      ...metricRoleFields(selected, options.purpose),
      model: selected.model,
      endpoint: callEndpoint,
      startedAt,
      completedAt: now(),
      durationMs: Math.round(performance.now() - startedMs),
      ok: true,
      httpStatus: response.status,
      promptTokens: payload.usage?.prompt_tokens,
      completionTokens: payload.usage?.completion_tokens,
      totalTokens: payload.usage?.total_tokens,
    });
    const choice = payload.choices?.[0];
    const message = choice?.message ?? {};
    const toolCalls: ToolCall[] = (message.tool_calls ?? [])
      .filter((tc) => tc.function != null && typeof tc.function.name === "string")
      .map((tc) => ({
        id: typeof tc.id === "string" && tc.id.length > 0 ? tc.id : `tool-${Math.random().toString(36).slice(2, 10)}`,
        name: tc.function?.name ?? "",
        argumentsJson: tc.function?.arguments ?? "{}",
      }));
    return {
      finishReason: choice?.finish_reason ?? "stop",
      textContent: typeof message.content === "string" ? message.content : "",
      toolCalls,
      ...(payload.usage?.prompt_tokens != null ? { promptTokens: payload.usage.prompt_tokens } : {}),
      ...(payload.usage?.completion_tokens != null ? { completionTokens: payload.usage.completion_tokens } : {}),
    };
  };

  // OAuth-configured roles are considered "available" since we resolve the
  // bearer at request time (and refreshing is a network call, not a check).
  // The actual file presence will be enforced by getOAuthHeaders on first call.
  const masterUsesOAuth = config.master.authType === "oauth";
  return {
    masterAvailable: masterApiKey != null || masterUsesOAuth,
    localAvailable: async () => {
      if (localApiKey == null || localApiKey === "") return false;
      if (config.local.provider === "anthropic") return true;
      if (config.local.provider === "ollama") {
        try {
          const response = await fetch(endpoint(config.local.baseUrl, "/api/tags"));
          return response.ok;
        } catch {
          return false;
        }
      }
      try {
        const response = await fetch(endpoint(config.local.baseUrl, "/models"), {
          headers: { authorization: `Bearer ${localApiKey}` },
        });
        return response.ok;
      } catch {
        return false;
      }
    },
    chatText: request,
    chatTextWithTools: requestWithTools,
  };
};

export const parseJsonObject = <T>(text: string, fallback: T): T => {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim();
  const candidate = fenced ?? trimmed;
  try {
    return JSON.parse(candidate) as T;
  } catch {
    const start = candidate.indexOf("{");
    const end = candidate.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(candidate.slice(start, end + 1)) as T;
      } catch {
        return fallback;
      }
    }
    return fallback;
  }
};

export const createAxServices = (config: BagConfig): {
  master?: AxAIService;
  local: AxAIService;
} => {
  const masterApiKey = resolveMasterApiKey(config);
  const master =
    masterApiKey == null
      ? undefined
      : (ai({
          name: axProviderNameFor(config.master.provider),
          apiKey: masterApiKey,
          apiURL: config.master.baseUrl,
          config: {
            model: config.master.model,
            maxTokens: config.master.maxTokens,
            temperature: config.master.temperature,
          },
        } as never) as AxAIService);

  const local = ai({
    name: axProviderNameFor(config.local.provider),
    apiKey: resolveLocalApiKey(config),
    apiURL: config.local.baseUrl,
    config: {
      model: config.local.model,
      maxTokens: config.local.maxTokens,
      temperature: config.local.temperature,
    },
  } as never) as AxAIService;

  return master == null ? { local } : { master, local };
};

export const createAxBleedingAgent = (config: BagConfig) => {
  const services = createAxServices(config);
  const rootAi = services.master ?? services.local;

  return agent(
    "task:string, repoContext:string, knowledge:string -> plan:string, risks:string, nextActions:string",
    {
      ai: rootAi,
      judgeAI: services.master ?? rootAi,
      agentIdentity: {
        name: "BleedingAgent",
        description:
          "A local-first coding agent that interviews, drafts PRDs, builds DAGs, measures itself, and improves policy from run evidence.",
      },
      contextFields: [
        { field: "repoContext", keepInPromptChars: 4000, reverseTruncate: true },
        { field: "knowledge", keepInPromptChars: 3000, reverseTruncate: true },
      ],
      runtime: new AxJSRuntime({
        timeout: 30_000,
        permissions: [AxJSRuntimePermission.FILESYSTEM],
        captureConsole: true,
      }),
      maxBatchedLlmQueryConcurrency: config.policy.executorConcurrency,
      maxSubAgentCalls: config.policy.maxSubAgentCalls,
      maxTurns: config.policy.maxTurns,
      promptLevel: "detailed",
    },
  );
};
