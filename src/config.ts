import { existsSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import {
  BagConfigSchema,
  MODEL_RUNTIME_ROLES,
  type BagConfig,
  type ContextWindowSource,
  type ModelEndpointKind,
  type ModelProvider,
  type ModelProviderConfigRole,
  type ModelRoleBinding,
  type ModelRuntimeRole,
  type ProviderDiscoverySource,
} from "./types";

const CONFIG_FILE = "bag.config.json";

export const defaultConfig = (): BagConfig => BagConfigSchema.parse({});

export const loadConfig = (cwd = process.cwd()): BagConfig => {
  const path = resolve(cwd, CONFIG_FILE);
  if (!existsSync(path)) {
    return defaultConfig();
  }
  return BagConfigSchema.parse(JSON.parse(readFileSync(path, "utf8")) as unknown);
};

export const configPath = (cwd = process.cwd()): string => resolve(cwd, CONFIG_FILE);

export const resolveMasterApiKey = (config: BagConfig): string | undefined => {
  const key = process.env[config.master.apiKeyEnv]?.trim();
  return key === "" ? undefined : key;
};

export const resolveLocalApiKey = (config: BagConfig): string => {
  if (config.local.apiKeyEnv != null) {
    const fromEnv = process.env[config.local.apiKeyEnv]?.trim();
    if (fromEnv != null && fromEnv !== "") return fromEnv;
  }
  return config.local.apiKey;
};

export interface ResolvedModelRoleConfig {
  modelRole: ModelRuntimeRole;
  providerConfigRole: ModelProviderConfigRole;
  provider: ModelProvider;
  model: string;
  baseUrl: string;
  endpointKind: ModelEndpointKind;
  modelServerId: string;
  modelServerProfileId: string;
  providerDiscoverySource: ProviderDiscoverySource;
  contextWindowTokens: number;
  contextWindowSource: ContextWindowSource;
  maxTokens: number;
  maxOutputTokens: number;
  temperature: number;
  fallbackModelRole?: ModelRuntimeRole;
}

export const modelRoleBinding = (config: BagConfig, role: ModelRuntimeRole): ModelRoleBinding =>
  config.modelRoles[role];

type ProviderConfig = BagConfig["master"] | BagConfig["local"];

const stableValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((entry) => stableValue(entry));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, stableValue(entry)]),
    );
  }
  return value;
};

const hashConfig = (value: unknown): string =>
  createHash("sha256").update(JSON.stringify(stableValue(value))).digest("hex").slice(0, 12);

const slug = (value: string): string => {
  const normalized = value.toLowerCase().replace(/[^a-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "");
  return normalized === "" ? "unknown" : normalized.slice(0, 32);
};

const contextWindowFor = (source: ProviderConfig): {
  contextWindowTokens: number;
  contextWindowSource: ContextWindowSource;
} => {
  if (source.contextWindowTokens !== undefined) {
    return {
      contextWindowTokens: source.contextWindowTokens,
      contextWindowSource: "configured",
    };
  }
  return {
    contextWindowTokens: Math.max(source.maxTokens, 8192),
    contextWindowSource: "deterministic_floor",
  };
};

const providerFingerprint = (providerConfigRole: ModelProviderConfigRole, source: ProviderConfig): Record<string, unknown> => ({
  providerConfigRole,
  provider: source.provider,
  baseUrl: source.baseUrl,
});

const serverProfileFingerprint = (
  providerConfigRole: ModelProviderConfigRole,
  source: ProviderConfig,
  contextWindowTokens: number,
): Record<string, unknown> => ({
  ...providerFingerprint(providerConfigRole, source),
  model: source.model,
  endpointKind: source.endpointKind,
  contextWindowTokens,
  maxOutputTokens: source.maxTokens,
});

export const resolveProviderConfigProfile = (
  config: BagConfig,
  providerConfigRole: ModelProviderConfigRole,
): Omit<ResolvedModelRoleConfig, "modelRole" | "fallbackModelRole"> => {
  const source = providerConfigRole === "master" ? config.master : config.local;
  const context = contextWindowFor(source);
  const modelServerId = source.serverId ??
    `server.${providerConfigRole}.${slug(source.provider)}.${hashConfig(providerFingerprint(providerConfigRole, source))}`;
  const modelServerProfileId = source.serverProfileId ??
    `server-profile.${providerConfigRole}.${hashConfig(serverProfileFingerprint(
      providerConfigRole,
      source,
      context.contextWindowTokens,
    ))}`;

  return {
    providerConfigRole,
    provider: source.provider,
    model: source.model,
    baseUrl: source.baseUrl,
    endpointKind: source.endpointKind,
    modelServerId,
    modelServerProfileId,
    providerDiscoverySource: source.serverId === undefined && source.serverProfileId === undefined
      ? "deterministic_default"
      : "configured",
    contextWindowTokens: context.contextWindowTokens,
    contextWindowSource: context.contextWindowSource,
    maxTokens: source.maxTokens,
    maxOutputTokens: source.maxTokens,
    temperature: source.temperature,
  };
};

export const resolveModelRoleConfig = (config: BagConfig, role: ModelRuntimeRole): ResolvedModelRoleConfig => {
  const binding = modelRoleBinding(config, role);
  const providerProfile = resolveProviderConfigProfile(config, binding.source);
  return {
    ...providerProfile,
    modelRole: role,
    ...(binding.fallbackRole === undefined ? {} : { fallbackModelRole: binding.fallbackRole }),
  };
};

export const resolveAllModelRoleConfigs = (config: BagConfig): ResolvedModelRoleConfig[] =>
  MODEL_RUNTIME_ROLES.map((role) => resolveModelRoleConfig(config, role));
