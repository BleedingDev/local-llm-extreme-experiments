import { resolveMasterApiKey, resolveModelRoleConfig } from "../config";
import type { BagConfig, ModelRuntimeRole } from "../types";
import { DEFAULT_TOOL_RENDERER_ID, DEFAULT_TOOL_RENDERER_VERSION } from "./tool-renderer";
import { loadOptimizerRegistry } from "./registry";
import { resolveLoadedOptimizerPolicy, type ResolvedOptimizerPolicy } from "./policy-resolver";
import type { OptimizerSessionPinTelemetry } from "../telemetry";

export type OptimizerSessionPin = {
  telemetry: OptimizerSessionPinTelemetry;
  resolvedPolicy: ResolvedOptimizerPolicy;
};

const defaultRuntimeRoleForSession = (config: BagConfig): ModelRuntimeRole =>
  resolveMasterApiKey(config) == null ? "local" : "master";

export const createOptimizerSessionPin = (
  config: BagConfig,
  cwd: string,
  preferredRole: ModelRuntimeRole = defaultRuntimeRoleForSession(config),
): OptimizerSessionPin => {
  const registry = loadOptimizerRegistry(config, cwd);
  const roleConfig = resolveModelRoleConfig(config, preferredRole);
  const resolvedPolicy = resolveLoadedOptimizerPolicy(registry, {
    modelRole: preferredRole,
    modelName: roleConfig.model,
  });

  return {
    resolvedPolicy,
    telemetry: {
      modelRole: resolvedPolicy.modelRole ?? preferredRole,
      ...(resolvedPolicy.providerConfigRole === undefined
        ? { providerConfigRole: roleConfig.providerConfigRole }
        : { providerConfigRole: resolvedPolicy.providerConfigRole }),
      ...(resolvedPolicy.fallbackModelRole === undefined ? {} : { fallbackModelRole: resolvedPolicy.fallbackModelRole }),
      provider: resolvedPolicy.provider,
      endpointKind: resolvedPolicy.endpointKind,
      ...(resolvedPolicy.modelServerId === undefined
        ? { modelServerId: roleConfig.modelServerId }
        : { modelServerId: resolvedPolicy.modelServerId }),
      ...(resolvedPolicy.modelServerProfileId === undefined
        ? { modelServerProfileId: roleConfig.modelServerProfileId }
        : { modelServerProfileId: resolvedPolicy.modelServerProfileId }),
      ...(resolvedPolicy.providerDiscoverySource === undefined
        ? { providerDiscoverySource: roleConfig.providerDiscoverySource }
        : { providerDiscoverySource: resolvedPolicy.providerDiscoverySource }),
      contextWindowTokens: resolvedPolicy.contextWindowTokens,
      maxOutputTokens: resolvedPolicy.maxOutputTokens,
      modelProfileId: resolvedPolicy.modelProfileId,
      codebaseProfileId: resolvedPolicy.codebaseProfileId,
      policyId: resolvedPolicy.policyId,
      canonicalToolVersion: resolvedPolicy.canonicalToolVersion,
      renderedToolVersion: resolvedPolicy.renderedToolVersion,
      resultStyleVersion: resolvedPolicy.resultStyleVersion,
      verificationPolicyVersion: resolvedPolicy.verificationPolicyVersion,
      editStrategyVersion: resolvedPolicy.editStrategyVersion,
      renderedEditContractVersion: resolvedPolicy.renderedEditContractVersion,
      editFallbackPolicyVersion: resolvedPolicy.editFallbackPolicyVersion,
      editRepairPolicyVersion: resolvedPolicy.editRepairPolicyVersion,
      editVerifierPolicyVersion: resolvedPolicy.editVerifierPolicyVersion,
      editObjectiveSetId: resolvedPolicy.editObjectiveSetId,
      source: resolvedPolicy.source,
      registryRoot: registry.root,
      registryErrorCount: registry.errors.length,
      invalidRecordCount: registry.invalidRecords.length,
      rendererId: DEFAULT_TOOL_RENDERER_ID,
      rendererVersion: DEFAULT_TOOL_RENDERER_VERSION,
      modelProfileRecordId: resolvedPolicy.recordIds.modelProfileRecordId,
      codebaseProfileRecordId: resolvedPolicy.recordIds.codebaseProfileRecordId,
      policyRecordId: resolvedPolicy.recordIds.policyRecordId,
    },
  };
};
