import { editApplySupportedFamilies } from "../edit-strategy/apply-layer";
import { renderEditToolContract } from "../edit-strategy/contract-renderer";
import { parseCanonicalEditStrategyDefinitions, type CanonicalEditStrategyDefinition } from "../edit-strategy/taxonomy";
import type { EditStrategyFamily } from "../edit-strategy/types";
import {
  routeEditStrategy,
  type EditStrategyFallbackRule,
  type EditTaskShape,
} from "../optimizer/edit-policy-router";
import { detectProjectKind } from "../workspace";
import type { BagAcpSession } from "./session";
import type { CodingEditResult, CodingFileSnapshot, CodingPatch, LiveEditContext } from "./coding-types";

export const resolveLiveEditContext = (
  session: BagAcpSession,
  fileSnapshots: readonly CodingFileSnapshot[],
): LiveEditContext => {
  const definitions = supportedLiveEditDefinitions();
  const routeDefinitions = requiresCreateCapableStrategy(fileSnapshots)
    ? definitions.filter((definition) => definition.family === "whole_file")
    : definitions;
  const taskShape = editTaskShapeFor(session, fileSnapshots);
  const decision = routeEditStrategy({
    resolvedPolicy: session.optimizerPin.resolvedPolicy,
    taskShape,
    definitions: routeDefinitions,
    minSampleCount: 1,
  });
  const definition = routeDefinitions.find((candidate) => candidate.strategyId === decision.selectedStrategyId) ?? routeDefinitions[0];
  if (definition === undefined) {
    throw new Error("live edit strategy routing requires at least one supported edit definition");
  }
  return {
    taskShape,
    decision,
    definition,
    renderedContract: renderEditToolContract(definition, session.optimizerPin.resolvedPolicy),
  };
};

export const supportedLiveEditDefinitions = (): CanonicalEditStrategyDefinition[] => {
  const supportedFamilies = new Set<EditStrategyFamily>(editApplySupportedFamilies());
  return parseCanonicalEditStrategyDefinitions()
    .filter((definition) => supportedFamilies.has(definition.family))
    .filter((definition) => definition.futureGate === "none");
};

export const editTaskShapeFor = (
  session: BagAcpSession,
  fileSnapshots: readonly CodingFileSnapshot[],
): EditTaskShape => {
  const bytes = fileSnapshots.map((file) => Buffer.byteLength(file.content));
  const protectedPaths = session.optimizerPin.resolvedPolicy.codebaseProfile.protectedPaths;
  const projectKind = detectProjectKind(session.cwd);
  return {
    targetFileCount: fileSnapshots.length,
    estimatedChangedFileCount: Math.max(1, Math.min(fileSnapshots.length, 4)),
    largestTargetFileBytes: bytes.length === 0 ? 0 : Math.max(...bytes),
    totalTargetFileBytes: bytes.reduce((sum, value) => sum + value, 0),
    contextBudgetTokens: session.optimizerPin.resolvedPolicy.modelProfile.contextWindowTokens,
    outputBudgetTokens: session.optimizerPin.resolvedPolicy.modelProfile.maxOutputTokens,
    verifierStrength: projectKind === "unknown" ? "none" : "basic",
    protectedPathRisk: protectedPaths.length > 0 ? "medium" : "low",
    staleContextRisk: "medium",
    requiresMultiFileConsistency: fileSnapshots.length > 1,
  };
};

export const requiresCreateCapableStrategy = (
  fileSnapshots: readonly CodingFileSnapshot[],
): boolean =>
  fileSnapshots.length === 0 || fileSnapshots.some((file) => file.kind === "create");

export const serializeLiveEditContext = (context: LiveEditContext): Record<string, unknown> => ({
  taskShape: context.taskShape,
  decision: context.decision,
  definition: context.definition,
  renderedContract: context.renderedContract,
});

export const fallbackLiveEditContext = (
  session: BagAcpSession,
  current: LiveEditContext,
  trigger: EditStrategyFallbackRule["trigger"],
): LiveEditContext | undefined => {
  const rule = current.decision.fallbackRules.find((candidate) => candidate.trigger === trigger);
  if (rule?.action === "abort" || rule?.toStrategyId == null) {
    return undefined;
  }
  const definitions = supportedLiveEditDefinitions();
  const definition = definitions.find((candidate) => candidate.strategyId === rule.toStrategyId);
  if (definition === undefined) {
    return undefined;
  }
  return {
    taskShape: current.taskShape,
    definition,
    renderedContract: renderEditToolContract(definition, session.optimizerPin.resolvedPolicy),
    decision: {
      ...current.decision,
      selectedStrategyId: definition.strategyId,
      selectedStrategyFamily: definition.family,
      degraded: true,
      warnings: [
        ...current.decision.warnings,
        `fallback selected after ${trigger}: ${rule.reason}`,
      ],
    },
  };
};

export const fallbackTriggerForPatch = (
  patch: CodingPatch,
  results: readonly CodingEditResult[],
): EditStrategyFallbackRule["trigger"] | undefined => {
  if (patch.parseFailures.length > 0) {
    return "parse_failed";
  }
  const failed = results.find((result) => !result.ok);
  if (failed?.errorCode === undefined) {
    return undefined;
  }
  return fallbackTriggerForErrorCode(failed.errorCode);
};

export const fallbackTriggerForErrorCode = (
  errorCode: string,
): EditStrategyFallbackRule["trigger"] | undefined => {
  switch (errorCode) {
    case "parse_error":
    case "path_or_fence_error":
    case "schema_validation_error":
    case "truncation_induced_error":
      return "parse_failed";
    case "hash_mismatch":
    case "anchor_stale":
      return "stale_context";
    case "protected_path_violation":
      return "protected_path_violation";
    case "permission_rejected":
    case "acp_write_failed":
      return undefined;
    default:
      return "apply_failed";
  }
};
