import { describe, expect, test } from "bun:test";
import type { AssistantMessage, Context as PiContext, Model as PiModel, ProviderStreamOptions } from "@mariozechner/pi-ai";
import { defaultConfig } from "../src/config";
import type { LlmCallMetric } from "../src/types";
import { createLlmRouterFromPiAi, piAiRouterEnabled } from "../src/llm-pi-ai";

type CapturedCall = {
  model: PiModel<string>;
  context: PiContext;
  options: ProviderStreamOptions | undefined;
};

const fakeAssistant = (overrides: Partial<AssistantMessage> = {}): AssistantMessage => ({
  role: "assistant",
  content: [{ type: "text", text: "hello back" }],
  api: "openai-completions",
  provider: "openai",
  model: "fake-model",
  usage: {
    input: 7,
    output: 3,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 10,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  },
  stopReason: "stop",
  timestamp: 0,
  ...overrides,
});

const withMasterKey = async (key: string | undefined, fn: () => Promise<void>): Promise<void> => {
  const previous = process.env["TEST_MASTER_KEY"];
  if (key === undefined) delete process.env["TEST_MASTER_KEY"];
  else process.env["TEST_MASTER_KEY"] = key;
  try {
    await fn();
  } finally {
    if (previous === undefined) delete process.env["TEST_MASTER_KEY"];
    else process.env["TEST_MASTER_KEY"] = previous;
  }
};

const baseConfig = () => {
  const cfg = defaultConfig();
  return {
    ...cfg,
    master: {
      ...cfg.master,
      apiKeyEnv: "TEST_MASTER_KEY",
      provider: "openai" as const,
      model: "configured-master-model",
      baseUrl: "https://example.test/v1",
    },
    local: {
      ...cfg.local,
      apiKey: "local-key",
      apiKeyEnv: undefined,
      provider: "openai-compatible" as const,
      model: "configured-local-model",
      baseUrl: "http://127.0.0.1:8080/v1",
    },
  };
};

describe("createLlmRouterFromPiAi", () => {
  test("masterAvailable reflects API-key presence", async () => {
    await withMasterKey("sk-master", async () => {
      const router = createLlmRouterFromPiAi(baseConfig(), { complete: async () => fakeAssistant() });
      expect(router.masterAvailable).toBe(true);
    });
    await withMasterKey(undefined, async () => {
      const router = createLlmRouterFromPiAi(baseConfig(), { complete: async () => fakeAssistant() });
      expect(router.masterAvailable).toBe(false);
    });
  });

  test("chatText routes the master role through pi-ai with config-driven model id", async () => {
    await withMasterKey("sk-master", async () => {
      const captured: CapturedCall[] = [];
      const metrics: LlmCallMetric[] = [];
      const router = createLlmRouterFromPiAi(baseConfig(), {
        complete: async (model, context, options) => {
          captured.push({ model, context, options });
          return fakeAssistant();
        },
        telemetry: { recordLlmCall: (m) => { metrics.push(m); } },
      });
      const text = await router.chatText({
        role: "master",
        purpose: "instruction-summarizer",
        messages: [
          { role: "system", content: "you are concise" },
          { role: "user", content: "hi" },
        ],
        maxTokens: 256,
        temperature: 0.4,
      });
      expect(text).toBe("hello back");
      expect(captured.length).toBe(1);
      const call = captured[0]!;
      // Model id is config-driven, NOT hardcoded.
      expect(call.model.id).toBe("configured-master-model");
      expect(call.model.baseUrl).toBe("https://example.test/v1");
      expect(call.model.api).toBe("openai-completions");
      expect(call.options?.apiKey).toBe("sk-master");
      expect(call.options?.maxTokens).toBe(256);
      expect(call.options?.temperature).toBe(0.4);
      // System prompt is lifted out, user message carries the rest.
      expect(call.context.systemPrompt).toBe("you are concise");
      expect(call.context.messages.length).toBe(1);
      expect(call.context.messages[0]!.role).toBe("user");
      // Telemetry got the role + purpose.
      expect(metrics.length).toBe(1);
      const metric = metrics[0]!;
      expect(metric.role).toBe("master");
      expect(metric.providerConfigRole).toBe("master");
      expect(metric.purpose).toBe("instruction-summarizer");
      expect(metric.ok).toBe(true);
      expect(metric.promptTokens).toBe(7);
      expect(metric.completionTokens).toBe(3);
      expect(metric.totalTokens).toBe(10);
    });
  });

  test("planner role with no master key falls back to local + records fallbackFromRole", async () => {
    await withMasterKey(undefined, async () => {
      const captured: CapturedCall[] = [];
      const metrics: LlmCallMetric[] = [];
      const router = createLlmRouterFromPiAi(baseConfig(), {
        complete: async (model, context, options) => {
          captured.push({ model, context, options });
          return fakeAssistant({ content: [{ type: "text", text: "from-local" }] });
        },
        telemetry: { recordLlmCall: (m) => { metrics.push(m); } },
      });
      const text = await router.chatText({
        role: "planner",
        messages: [{ role: "user", content: "plan" }],
      });
      expect(text).toBe("from-local");
      expect(captured[0]!.model.id).toBe("configured-local-model");
      expect(captured[0]!.options?.apiKey).toBe("local-key");
      expect(metrics[0]!.role).toBe("planner");
      expect(metrics[0]!.resolvedRole).toBe("local");
      expect(metrics[0]!.fallbackFromRole).toBe("planner");
      expect(metrics[0]!.providerConfigRole).toBe("local");
    });
  });

  test("multimodal user content (text + image_url) is preserved as pi-ai image block", async () => {
    await withMasterKey("sk-master", async () => {
      const captured: CapturedCall[] = [];
      const router = createLlmRouterFromPiAi(baseConfig(), {
        complete: async (model, context, options) => {
          captured.push({ model, context, options });
          return fakeAssistant();
        },
      });
      const PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII=";
      await router.chatText({
        role: "master",
        messages: [
          { role: "system", content: "vision agent" },
          {
            role: "user",
            content: [
              { type: "text", text: "what is in this image?" },
              { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
            ],
          },
        ],
      });
      const userMsg = captured[0]!.context.messages[0]!;
      expect(userMsg.role).toBe("user");
      const blocks = userMsg.content;
      expect(Array.isArray(blocks)).toBe(true);
      const arr = blocks as { type: string }[];
      expect(arr.length).toBe(2);
      expect(arr[0]!.type).toBe("text");
      expect(arr[1]!.type).toBe("image");
      const imageBlock = arr[1]! as { type: "image"; mimeType: string; data: string };
      expect(imageBlock.mimeType).toBe("image/png");
      expect(imageBlock.data).toBe(PNG_B64);
    });
  });

  test("chatTextWithTools surfaces toolCalls + finishReason from pi-ai AssistantMessage", async () => {
    await withMasterKey("sk-master", async () => {
      const router = createLlmRouterFromPiAi(baseConfig(), {
        complete: async () =>
          fakeAssistant({
            stopReason: "toolUse",
            content: [
              { type: "text", text: "I'll use a tool." },
              { type: "toolCall", id: "tc_1", name: "search", arguments: { q: "bag" } },
            ],
          }),
      });
      const result = await router.chatTextWithTools({
        role: "master",
        messages: [{ role: "user", content: "find" }],
        tools: [
          {
            type: "function",
            function: {
              name: "search",
              description: "Search",
              parameters: { type: "object", properties: { q: { type: "string" } }, required: ["q"] },
            },
          },
        ],
      });
      expect(result.finishReason).toBe("tool_calls");
      expect(result.textContent).toBe("I'll use a tool.");
      expect(result.toolCalls.length).toBe(1);
      expect(result.toolCalls[0]!.id).toBe("tc_1");
      expect(result.toolCalls[0]!.name).toBe("search");
      expect(JSON.parse(result.toolCalls[0]!.argumentsJson)).toEqual({ q: "bag" });
      expect(result.promptTokens).toBe(7);
      expect(result.completionTokens).toBe(3);
    });
  });

  test("anthropic provider config selects anthropic-messages api", async () => {
    process.env["TEST_MASTER_KEY"] = "sk-anthropic";
    try {
      const cfg = baseConfig();
      const anthCfg = {
        ...cfg,
        master: { ...cfg.master, provider: "anthropic" as const, baseUrl: "https://api.anthropic.com/v1" },
      };
      const captured: CapturedCall[] = [];
      const router = createLlmRouterFromPiAi(anthCfg, {
        complete: async (model, context, options) => {
          captured.push({ model, context, options });
          return fakeAssistant({ api: "anthropic-messages" });
        },
      });
      await router.chatText({ role: "master", messages: [{ role: "user", content: "hi" }] });
      expect(captured[0]!.model.api).toBe("anthropic-messages");
      expect(captured[0]!.model.provider).toBe("anthropic");
    } finally {
      delete process.env["TEST_MASTER_KEY"];
    }
  });

  test("error stopReason is reported as a thrown error and ok:false telemetry", async () => {
    await withMasterKey("sk-master", async () => {
      const metrics: LlmCallMetric[] = [];
      const router = createLlmRouterFromPiAi(baseConfig(), {
        complete: async () =>
          fakeAssistant({
            stopReason: "error",
            errorMessage: "rate limited",
            content: [],
          }),
        telemetry: { recordLlmCall: (m) => { metrics.push(m); } },
      });
      let threw: Error | undefined;
      try {
        await router.chatText({ role: "master", messages: [{ role: "user", content: "hi" }] });
      } catch (err) {
        threw = err as Error;
      }
      expect(threw?.message).toContain("rate limited");
      expect(metrics.length).toBe(1);
      expect(metrics[0]!.ok).toBe(false);
      expect(metrics[0]!.error).toContain("rate limited");
    });
  });

  test("missing api key short-circuits without invoking pi-ai", async () => {
    await withMasterKey(undefined, async () => {
      // Force a config where local has no key either
      const cfg = baseConfig();
      const noKeysCfg = { ...cfg, local: { ...cfg.local, apiKey: "", apiKeyEnv: undefined } };
      let invoked = false;
      const router = createLlmRouterFromPiAi(noKeysCfg, {
        complete: async () => {
          invoked = true;
          return fakeAssistant();
        },
      });
      let threw: Error | undefined;
      try {
        await router.chatText({ role: "local", messages: [{ role: "user", content: "hi" }] });
      } catch (err) {
        threw = err as Error;
      }
      expect(invoked).toBe(false);
      expect(threw?.message).toMatch(/missing API key/);
    });
  });
});

describe("piAiRouterEnabled", () => {
  test("recognises common truthy strings", () => {
    expect(piAiRouterEnabled({ BAG_USE_PI_AI: "1" })).toBe(true);
    expect(piAiRouterEnabled({ BAG_USE_PI_AI: "true" })).toBe(true);
    expect(piAiRouterEnabled({ BAG_USE_PI_AI: "yes" })).toBe(true);
    expect(piAiRouterEnabled({ BAG_USE_PI_AI: "on" })).toBe(true);
  });
  test("returns false for missing or falsy values", () => {
    expect(piAiRouterEnabled({})).toBe(false);
    expect(piAiRouterEnabled({ BAG_USE_PI_AI: "" })).toBe(false);
    expect(piAiRouterEnabled({ BAG_USE_PI_AI: "0" })).toBe(false);
    expect(piAiRouterEnabled({ BAG_USE_PI_AI: "false" })).toBe(false);
  });
});
