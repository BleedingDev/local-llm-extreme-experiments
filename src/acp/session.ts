import type {
  AgentSideConnection as AcpConnection,
  McpServer,
} from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import { basename, resolve } from "node:path";
import { createOptimizerSessionPin as createOptimizerSessionPinForRun, type OptimizerSessionPin } from "../optimizer/session-pin";
import type { BagConfig } from "../types";
import { availableCommands, type AcpClientCapabilityProfile, type AcpSessionMode } from "./surface";

export type BagOptimizerSessionPin = OptimizerSessionPin;

export type BagAcpSession = {
  id: string;
  cwd: string;
  additionalDirectories: string[];
  executorConcurrency: number;
  mode: AcpSessionMode;
  createdAt: string;
  updatedAt: string;
  pendingPrompt: AbortController | null;
  title: string;
  yolo: boolean;
  mcpServers: McpServer[];
  optimizerPin: BagOptimizerSessionPin;
  clientCapabilities: AcpClientCapabilityProfile;
};

export const createAcpOptimizerSessionPin = (config: BagConfig, cwd: string): BagOptimizerSessionPin =>
  createOptimizerSessionPinForRun(config, cwd);

export const createBagAcpSession = (input: {
  config: BagConfig;
  sessions: Map<string, BagAcpSession>;
  cwd: string;
  additionalDirectories: string[];
  id?: string;
  mcpServers?: McpServer[];
  clientCapabilities: AcpClientCapabilityProfile;
  createOptimizerSessionPin?: (cwd: string) => BagOptimizerSessionPin;
}): BagAcpSession => {
  const now = new Date().toISOString();
  const resolvedCwd = resolve(input.cwd);
  const session: BagAcpSession = {
    id: input.id ?? `bag-${randomUUID()}`,
    cwd: resolvedCwd,
    additionalDirectories: input.additionalDirectories.map((directory) => resolve(resolvedCwd, directory)),
    executorConcurrency: input.config.policy.executorConcurrency,
    mode: "auto",
    createdAt: now,
    updatedAt: now,
    pendingPrompt: null,
    title: `BleedingAgent ${basename(input.cwd)}`,
    yolo: !input.config.policy.requirePermissions,
    mcpServers: input.mcpServers ?? [],
    optimizerPin: input.createOptimizerSessionPin?.(resolvedCwd) ?? createAcpOptimizerSessionPin(input.config, resolvedCwd),
    clientCapabilities: input.clientCapabilities,
  };
  input.sessions.set(session.id, session);
  return session;
};

export const resumeOrCreateBagAcpSession = (input: {
  config: BagConfig;
  sessions: Map<string, BagAcpSession>;
  cwd: string;
  additionalDirectories: string[];
  id: string;
  mcpServers?: McpServer[];
  clientCapabilities: AcpClientCapabilityProfile;
  createOptimizerSessionPin?: (cwd: string) => BagOptimizerSessionPin;
}): BagAcpSession => {
  const existing = input.sessions.get(input.id);
  if (existing != null) {
    existing.additionalDirectories = input.additionalDirectories.map((directory) => resolve(existing.cwd, directory));
    existing.mcpServers = input.mcpServers ?? [];
    existing.updatedAt = new Date().toISOString();
    return existing;
  }
  return createBagAcpSession(input);
};

export const configForAcpSession = (config: BagConfig, session: BagAcpSession): BagConfig => ({
  ...config,
  policy: {
    ...config.policy,
    executorConcurrency: session.executorConcurrency,
  },
});

export const publishAcpAvailableCommands = async (
  connection: AcpConnection,
  session: BagAcpSession,
): Promise<void> => {
  await connection.sessionUpdate({
    sessionId: session.id,
    update: {
      sessionUpdate: "available_commands_update",
      availableCommands: availableCommands(),
    },
  });
};

export const setAcpSessionModeUpdate = async (
  connection: AcpConnection,
  session: BagAcpSession,
  mode: AcpSessionMode,
): Promise<void> => {
  session.mode = mode;
  session.updatedAt = new Date().toISOString();
  await connection.sessionUpdate({
    sessionId: session.id,
    update: { sessionUpdate: "current_mode_update", currentModeId: session.mode },
  });
};

export const runWithTemporaryAcpMode = async <T>(input: {
  session: BagAcpSession;
  activeMode: "plan" | "run";
  previousMode: AcpSessionMode;
  setMode: (mode: AcpSessionMode) => Promise<void>;
  fn: () => Promise<T>;
}): Promise<T> => {
  const shouldRestoreAuto = input.previousMode === "auto";
  if (shouldRestoreAuto && input.session.mode !== input.activeMode) {
    await input.setMode(input.activeMode);
  }
  try {
    return await input.fn();
  } finally {
    if (shouldRestoreAuto) {
      await input.setMode("auto");
    }
  }
};
