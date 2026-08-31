import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultConfig, resolveAllModelRoleConfigs, resolveModelRoleConfig } from "../src/config";
import { createLlmRouter } from "../src/llm";
import { createOptimizerSessionPin } from "../src/optimizer/session-pin";
import type { LlmCallMetric } from "../src/types";

const withEnv = async (key: string, value: string | undefined, fn: () => Promise<void>): Promise<void> => {
  const previous = process.env[key];
  try {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
    await fn();
  } finally {
    if (previous === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = previous;
    }
  }
};

const fetchInputText = (input: Parameters<typeof fetch>[0]): string => String(input);

describe("provider role model", () => {
  test("describes role mappings with deterministic offline provider profile ids", () => {
    const config = defaultConfig();
    const roles = resolveAllModelRoleConfigs(config);
    const planner = resolveModelRoleConfig(config, "planner");
    const critic = resolveModelRoleConfig(config, "critic");
    const executor = resolveModelRoleConfig(config, "executor");
    const local = resolveModelRoleConfig(config, "local");

    expect(roles.map((role) => role.modelRole)).toEqual([
      "master",
      "local",
      "planner",
      "executor",
      "verifier",
      "critic",
      "summarizer",
      "fast_scout",
      "local_batch_executor",
    ]);
    expect(planner.providerConfigRole).toBe("master");
    expect(planner.fallbackModelRole).toBe("local");
    expect(planner.endpointKind).toBe("chat_completions");
    expect(planner.modelServerId).toMatch(/^server\.master\.openai\.[a-f0-9]{12}$/);
    expect(planner.modelServerProfileId).toMatch(/^server-profile\.master\.[a-f0-9]{12}$/);
    expect(planner.contextWindowTokens).toBe(Math.max(config.master.maxTokens, 8192));
    expect(planner.contextWindowSource).toBe("deterministic_floor");
    expect(planner.maxOutputTokens).toBe(config.master.maxTokens);
    expect(critic.modelServerProfileId).toBe(planner.modelServerProfileId);
    expect(executor.modelServerProfileId).toBe(local.modelServerProfileId);
  });

  test("pins provider and server lineage for run telemetry", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-provider-pin-"));
    const config = defaultConfig();
    const pin = createOptimizerSessionPin(config, cwd, "executor");
    const executor = resolveModelRoleConfig(config, "executor");

    expect(pin.telemetry).toMatchObject({
      modelRole: "executor",
      providerConfigRole: "local",
      provider: config.local.provider,
      endpointKind: "chat_completions",
      modelServerId: executor.modelServerId,
      modelServerProfileId: executor.modelServerProfileId,
      providerDiscoverySource: executor.providerDiscoverySource,
      contextWindowTokens: executor.contextWindowTokens,
      maxOutputTokens: executor.maxOutputTokens,
      modelProfileId: pin.resolvedPolicy.modelProfileId,
      policyId: pin.resolvedPolicy.policyId,
      renderedToolVersion: pin.resolvedPolicy.renderedToolVersion,
      renderedEditContractVersion: pin.resolvedPolicy.renderedEditContractVersion,
    });
  });

  test("keeps explicit master calls on the master provider when the key is missing", async () => {
    await withEnv("OPENAI_API_KEY", undefined, async () => {
      const config = defaultConfig();
      const metrics: LlmCallMetric[] = [];
      const originalFetch = globalThis.fetch;
      let fetchCalled = false;
      globalThis.fetch = (async () => {
        fetchCalled = true;
        throw new Error("fetch should not be called without a master key");
      }) as typeof fetch;
      try {
        const router = createLlmRouter(config, { recordLlmCall: (metric) => metrics.push(metric) });

        await expect(router.chatText({ role: "master", messages: [{ role: "user", content: "hello" }] }))
          .rejects.toThrow("master model is unavailable: missing API key");

        expect(fetchCalled).toBe(false);
        expect(metrics).toHaveLength(1);
        expect(metrics[0]?.role).toBe("master");
        expect(metrics[0]?.resolvedRole).toBe("master");
        expect(metrics[0]?.providerConfigRole).toBe("master");
      } finally {
        globalThis.fetch = originalFetch;
      }
    });
  });

  test("falls planner back to the local provider when the master key is missing", async () => {
    await withEnv("OPENAI_API_KEY", undefined, async () => {
      const config = defaultConfig();
      const metrics: LlmCallMetric[] = [];
      const requests: Array<{ url: string; body: Record<string, unknown>; authorization?: string }> = [];
      const originalFetch = globalThis.fetch;
      globalThis.fetch = (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const authorization = new Headers(init?.headers).get("authorization") ?? undefined;
        requests.push({
          url: fetchInputText(input),
          body: JSON.parse(String(init?.body)) as Record<string, unknown>,
          ...(authorization === undefined ? {} : { authorization }),
        });
        return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as typeof fetch;
      try {
        const router = createLlmRouter(config, { recordLlmCall: (metric) => metrics.push(metric) });
        const localProfile = resolveModelRoleConfig(config, "local");
        const text = await router.chatText({ role: "planner", messages: [{ role: "user", content: "hello" }] });

        expect(text).toBe("ok");
        expect(requests).toHaveLength(1);
        expect(requests[0]?.url).toBe("http://127.0.0.1:18082/v1/chat/completions");
        expect(requests[0]?.body.model).toBe(config.local.model);
        expect(requests[0]?.authorization).toBe("Bearer local");
        expect(metrics[0]?.role).toBe("planner");
        expect(metrics[0]?.resolvedRole).toBe("local");
        expect(metrics[0]?.providerConfigRole).toBe("local");
        expect(metrics[0]?.fallbackFromRole).toBe("planner");
        expect(metrics[0]?.provider).toBe(config.local.provider);
        expect(metrics[0]?.endpointKind).toBe("chat_completions");
        expect(metrics[0]?.modelServerId).toBe(localProfile.modelServerId);
        expect(metrics[0]?.modelServerProfileId).toBe(localProfile.modelServerProfileId);
        expect(metrics[0]?.contextWindowTokens).toBe(localProfile.contextWindowTokens);
        expect(metrics[0]?.maxOutputTokens).toBe(config.local.maxTokens);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });
  });

  test("routes OpenAI Responses text calls through the configured endpoint kind", async () => {
    await withEnv("OPENAI_API_KEY", "responses-key", async () => {
      const base = defaultConfig();
      const config = {
        ...base,
        master: {
          ...base.master,
          endpointKind: "responses" as const,
        },
      };
      const requests: Array<{ url: string; body: Record<string, unknown>; authorization?: string }> = [];
      const originalFetch = globalThis.fetch;
      globalThis.fetch = (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const authorization = new Headers(init?.headers).get("authorization") ?? undefined;
        requests.push({
          url: fetchInputText(input),
          body: JSON.parse(String(init?.body)) as Record<string, unknown>,
          ...(authorization === undefined ? {} : { authorization }),
        });
        return new Response(JSON.stringify({
          output_text: "{\"ok\":true}",
          usage: { input_tokens: 11, output_tokens: 3, total_tokens: 14 },
        }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as typeof fetch;
      try {
        const metrics: LlmCallMetric[] = [];
        const router = createLlmRouter(config, { recordLlmCall: (metric) => metrics.push(metric) });
        const text = await router.chatText({
          role: "master",
          json: true,
          messages: [{ role: "user", content: "return json" }],
        });

        expect(text).toBe("{\"ok\":true}");
        expect(requests).toHaveLength(1);
        expect(requests[0]?.url).toBe("https://api.openai.com/v1/responses");
        expect(requests[0]?.authorization).toBe("Bearer responses-key");
        expect(requests[0]?.body).toEqual(expect.objectContaining({
          model: config.master.model,
          max_output_tokens: config.master.maxTokens,
          text: { format: { type: "json_object" } },
        }));
        expect(metrics[0]).toEqual(expect.objectContaining({
          ok: true,
          endpoint: "https://api.openai.com/v1/responses",
          endpointKind: "responses",
          promptTokens: 11,
          completionTokens: 3,
          totalTokens: 14,
        }));
      } finally {
        globalThis.fetch = originalFetch;
      }
    });
  });

  test("routes Anthropic text calls through /messages with Anthropic headers and system separation", async () => {
    const base = defaultConfig();
    const config = {
      ...base,
      local: {
        ...base.local,
        provider: "anthropic" as const,
        baseUrl: "https://api.anthropic.com/v1",
        apiKey: "anthropic-key",
      },
    };
    const requests: Array<{
      url: string;
      body: Record<string, unknown>;
      apiKey?: string;
      anthropicVersion?: string;
      authorization?: string;
    }> = [];
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
      const headers = new Headers(init?.headers);
      const apiKey = headers.get("x-api-key") ?? undefined;
      const anthropicVersion = headers.get("anthropic-version") ?? undefined;
      const authorization = headers.get("authorization") ?? undefined;
      requests.push({
        url: fetchInputText(input),
        body: JSON.parse(String(init?.body)) as Record<string, unknown>,
        ...(apiKey === undefined ? {} : { apiKey }),
        ...(anthropicVersion === undefined ? {} : { anthropicVersion }),
        ...(authorization === undefined ? {} : { authorization }),
      });
      return new Response(JSON.stringify({
        content: [{ type: "text", text: "anthropic-ok" }],
        usage: { input_tokens: 5, output_tokens: 2 },
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as typeof fetch;
    try {
      const metrics: LlmCallMetric[] = [];
      const router = createLlmRouter(config, { recordLlmCall: (metric) => metrics.push(metric) });
      const text = await router.chatText({
        role: "local",
        messages: [
          { role: "system", content: "system rules" },
          { role: "user", content: "hello" },
        ],
      });

      expect(text).toBe("anthropic-ok");
      expect(requests).toHaveLength(1);
      expect(requests[0]?.url).toBe("https://api.anthropic.com/v1/messages");
      expect(requests[0]?.apiKey).toBe("anthropic-key");
      expect(requests[0]?.anthropicVersion).toBe("2023-06-01");
      expect(requests[0]?.authorization).toBeUndefined();
      expect(requests[0]?.body.system).toBe("system rules");
      expect(requests[0]?.body.messages).toEqual([{ role: "user", content: "hello" }]);
      expect(metrics[0]).toEqual(expect.objectContaining({
        ok: true,
        provider: "anthropic",
        endpoint: "https://api.anthropic.com/v1/messages",
        promptTokens: 5,
        completionTokens: 2,
        totalTokens: 7,
      }));
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("fails closed for tool-use on endpoint kinds without implemented tool protocol support", async () => {
    await withEnv("OPENAI_API_KEY", "responses-key", async () => {
      const base = defaultConfig();
      const config = {
        ...base,
        master: {
          ...base.master,
          endpointKind: "responses" as const,
        },
      };
      const originalFetch = globalThis.fetch;
      let fetchCalled = false;
      globalThis.fetch = (async () => {
        fetchCalled = true;
        throw new Error("fetch should not be called for unsupported tool protocol");
      }) as typeof fetch;
      try {
        const router = createLlmRouter(config);
        await expect(router.chatTextWithTools({
          role: "master",
          messages: [{ role: "user", content: "use a tool" }],
          tools: [{
            type: "function",
            function: {
              name: "noop",
              description: "No-op.",
              parameters: { type: "object", properties: {} },
            },
          }],
        })).rejects.toThrow("tool-use is unsupported");
        expect(fetchCalled).toBe(false);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });
  });
});
