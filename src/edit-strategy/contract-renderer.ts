import type { ResolvedOptimizerPolicy } from "../optimizer/policy-resolver";
import {
  CanonicalToolSpecSchema,
  RenderedToolContractSchema,
  type CanonicalToolSpec,
  type JsonValue,
  type OptimizerRegistryRecord,
  type RenderedToolContract,
} from "../optimizer/types";
import { renderToolContract, renderToolContracts, selectRenderedToolContracts } from "../optimizer/tool-renderer";
import {
  parseCanonicalEditStrategyDefinitions,
  type CanonicalEditStrategyDefinition,
  type EditModelOutputContract,
} from "./taxonomy";

type JsonObject = { [key: string]: JsonValue };

export type RenderEditToolContractsInput = {
  resolvedPolicy: ResolvedOptimizerPolicy;
  definitions?: readonly CanonicalEditStrategyDefinition[];
  includeFutureGated?: boolean;
  initialExperimentOnly?: boolean;
};

export type SelectRenderedEditToolContractsInput = RenderEditToolContractsInput & {
  records: readonly OptimizerRegistryRecord[];
};

const stableObject = (value: JsonObject): JsonObject =>
  Object.fromEntries(
    Object.entries(value)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right)),
  );

const stringSchema = (description: string): JsonObject => ({
  description,
  type: "string",
});

const pathSchema = (): JsonObject => stringSchema("Repository-relative path.");

const editResultSchema = (): JsonObject => ({
  type: "object",
  required: ["editAttemptId", "status"],
  properties: stableObject({
    artifactRefs: {
      type: "array",
      items: { type: "string" },
    },
    editAttemptId: stringSchema("Stable edit attempt id for trace correlation."),
    errorCode: stringSchema("Stable edit error code when parsing, validation, apply, verification, or consistency fails."),
    status: {
      enum: ["parsed", "previewed", "applied", "skipped", "failed"],
      type: "string",
    },
  }),
});

const schemaForOutputContract = (contract: EditModelOutputContract): JsonObject => {
  switch (contract) {
    case "complete_file":
      return {
        type: "object",
        required: ["path", "content"],
        properties: stableObject({
          baseContentHash: stringSchema("Hash of the file content the model used, when available."),
          content: stringSchema("Complete replacement file content."),
          intent: stringSchema("Short edit intent for preview and post-apply consistency checks."),
          path: pathSchema(),
        }),
      };
    case "old_new_text":
      return {
        type: "object",
        required: ["path", "search", "replace"],
        properties: stableObject({
          expectedContentHash: stringSchema("Optional hash of the source file used for stale-context checks."),
          path: pathSchema(),
          replace: stringSchema("Replacement text."),
          search: stringSchema("Exact old text that must match exactly once unless policy says otherwise."),
        }),
      };
    case "edit_list":
      return {
        type: "object",
        required: ["edits"],
        properties: stableObject({
          edits: {
            type: "array",
            items: {
              type: "object",
              required: ["path", "search", "replace"],
              properties: stableObject({
                path: pathSchema(),
                replace: stringSchema("Replacement text."),
                search: stringSchema("Exact old text for this edit."),
              }),
            },
          },
        }),
      };
    case "fenced_blocks":
      return {
        type: "object",
        required: ["blocks"],
        properties: stableObject({
          blocks: {
            type: "array",
            items: {
              type: "object",
              required: ["path", "body"],
              properties: stableObject({
                body: stringSchema("Fenced diff/search-replace block body."),
                path: pathSchema(),
              }),
            },
          },
        }),
      };
    case "unified_patch":
      return {
        type: "object",
        required: ["patch"],
        properties: stableObject({
          patch: stringSchema("Unified/context diff patch text."),
        }),
      };
    case "structured_patch":
      return {
        type: "object",
        required: ["patch"],
        properties: stableObject({
          patch: stringSchema("Structured patch body using the rendered contract syntax."),
        }),
      };
    case "anchor_operations":
      return {
        type: "object",
        required: ["operations"],
        properties: stableObject({
          operations: {
            type: "array",
            items: {
              type: "object",
              required: ["path", "replacement"],
              properties: stableObject({
                endLine: { type: "integer" },
                expectedContentHash: stringSchema("Hash used to reject stale context."),
                path: pathSchema(),
                replacement: stringSchema("Replacement text for the anchored range."),
                startLine: { type: "integer" },
              }),
            },
          },
        }),
      };
    case "lazy_snippet":
      return {
        type: "object",
        required: ["path", "snippet"],
        properties: stableObject({
          mergeIntent: stringSchema("Instructions for the apply/editor model."),
          path: pathSchema(),
          snippet: stringSchema("Snippet to merge into the target file."),
        }),
      };
    case "plan_then_edit":
      return {
        type: "object",
        required: ["plan", "edit"],
        properties: stableObject({
          edit: stringSchema("Serialized downstream edit payload."),
          plan: stringSchema("Planner explanation that must stay consistent with the edit payload."),
        }),
      };
    case "structured_ast_operations":
      return {
        type: "object",
        required: ["operations"],
        properties: stableObject({
          operations: {
            type: "array",
            items: { type: "object" },
          },
        }),
      };
    case "editor_ranges":
      return {
        type: "object",
        required: ["ranges"],
        properties: stableObject({
          ranges: {
            type: "array",
            items: {
              type: "object",
              required: ["path", "replacement"],
              properties: stableObject({
                documentVersion: stringSchema("Fresh editor document version."),
                endOffset: { type: "integer" },
                path: pathSchema(),
                replacement: stringSchema("Replacement text for the editor range."),
                startOffset: { type: "integer" },
              }),
            },
          },
        }),
      };
    case "custom":
      return {
        type: "object",
        additionalProperties: true,
      };
  }
};

const strategyToolName = (definition: CanonicalEditStrategyDefinition): string =>
  definition.family;

export const canonicalEditStrategyToToolSpec = (definition: CanonicalEditStrategyDefinition): CanonicalToolSpec =>
  CanonicalToolSpecSchema.parse({
    canonicalToolId: definition.strategyId,
    canonicalToolVersion: definition.strategyVersion,
    namespace: "edit",
    name: strategyToolName(definition),
    title: definition.displayName,
    description: [
      definition.summary,
      `Output contract: ${definition.modelOutputContract}.`,
      `Trace requirements: ${definition.traceRequirements.join(", ")}.`,
      "Selection is optimizer-controlled per model, codebase, and task shape; this contract is not a global default.",
    ].join(" "),
    inputSchema: schemaForOutputContract(definition.modelOutputContract),
    outputSchema: editResultSchema(),
    resultStyle: "structured_error",
    sideEffectLevel: "write",
    requiresConfirmation: false,
    examples: [
      {
        name: `${definition.family}-minimal`,
        input: exampleInputForOutputContract(definition.modelOutputContract),
        output: {
          editAttemptId: "edit.attempt.example",
          status: "previewed",
        },
      },
    ],
  });

export const canonicalEditToolSpecs = (
  definitions: readonly CanonicalEditStrategyDefinition[] = parseCanonicalEditStrategyDefinitions(),
  options: { includeFutureGated?: boolean; initialExperimentOnly?: boolean } = {},
): CanonicalToolSpec[] =>
  definitions
    .filter((definition) => options.includeFutureGated === true || definition.maturity !== "future_gate")
    .filter((definition) => options.initialExperimentOnly !== true || definition.initialExperimentCandidate)
    .map(canonicalEditStrategyToToolSpec);

const editPolicyForRendering = (resolvedPolicy: ResolvedOptimizerPolicy): ResolvedOptimizerPolicy => ({
  ...resolvedPolicy,
  canonicalToolVersion: resolvedPolicy.editStrategyVersion,
  renderedToolVersion: resolvedPolicy.renderedEditContractVersion,
});

const definitionByStrategyId = (
  definitions: readonly CanonicalEditStrategyDefinition[],
): ReadonlyMap<string, CanonicalEditStrategyDefinition> =>
  new Map(definitions.map((definition) => [definition.strategyId, definition]));

const renderOptions = (
  input: Pick<RenderEditToolContractsInput, "includeFutureGated" | "initialExperimentOnly">,
): { includeFutureGated?: boolean; initialExperimentOnly?: boolean } => ({
  ...(input.includeFutureGated === undefined ? {} : { includeFutureGated: input.includeFutureGated }),
  ...(input.initialExperimentOnly === undefined ? {} : { initialExperimentOnly: input.initialExperimentOnly }),
});

const decorateEditContract = (
  contract: RenderedToolContract,
  definition: CanonicalEditStrategyDefinition,
  resolvedPolicy: ResolvedOptimizerPolicy,
): RenderedToolContract =>
  RenderedToolContractSchema.parse({
    ...contract,
    description: [
      contract.description,
      `Edit strategy id: ${definition.strategyId}.`,
      `Family: ${definition.family}.`,
      "Never treat apply success as final success; post-apply consistency and verification are separate measured phases.",
    ].join(" "),
    promptFragments: [
      ...contract.promptFragments,
      `Edit strategy ${definition.strategyId} uses ${definition.modelOutputContract}; emit only the rendered payload shape.`,
      `Repair policy ${resolvedPolicy.editRepairPolicyVersion}: repairs must cite concrete parse/apply/verification evidence.`,
      `Fallback policy ${resolvedPolicy.editFallbackPolicyVersion}: fallback is measured and must preserve fallbackFrom/fallbackTo lineage.`,
      `Verifier policy ${resolvedPolicy.editVerifierPolicyVersion}: verifier failures and self-detected regressions are not success.`,
      "If context is truncated or stale, report the structured failure instead of guessing a best-effort edit.",
    ],
  });

export const renderEditToolContract = (
  definition: CanonicalEditStrategyDefinition,
  resolvedPolicy: ResolvedOptimizerPolicy,
): RenderedToolContract =>
  decorateEditContract(
    renderToolContract(canonicalEditStrategyToToolSpec(definition), editPolicyForRendering(resolvedPolicy)),
    definition,
    resolvedPolicy,
  );

export const renderEditToolContracts = (input: RenderEditToolContractsInput): RenderedToolContract[] => {
  const definitions = (input.definitions ?? parseCanonicalEditStrategyDefinitions())
    .filter((definition) => input.includeFutureGated === true || definition.maturity !== "future_gate")
    .filter((definition) => input.initialExperimentOnly !== true || definition.initialExperimentCandidate);
  const byStrategyId = definitionByStrategyId(definitions);
  return renderToolContracts({
    canonicalToolSpecs: canonicalEditToolSpecs(definitions, renderOptions(input)),
    resolvedPolicy: editPolicyForRendering(input.resolvedPolicy),
  }).map((contract) => decorateEditContract(contract, byStrategyId.get(contract.canonicalToolId)!, input.resolvedPolicy));
};

export const selectRenderedEditToolContracts = (input: SelectRenderedEditToolContractsInput): RenderedToolContract[] => {
  const definitions = (input.definitions ?? parseCanonicalEditStrategyDefinitions())
    .filter((definition) => input.includeFutureGated === true || definition.maturity !== "future_gate")
    .filter((definition) => input.initialExperimentOnly !== true || definition.initialExperimentCandidate);
  const byStrategyId = definitionByStrategyId(definitions);
  return selectRenderedToolContracts({
    canonicalToolSpecs: canonicalEditToolSpecs(definitions, renderOptions(input)),
    resolvedPolicy: editPolicyForRendering(input.resolvedPolicy),
    records: input.records,
  }).map((contract) => {
    const definition = byStrategyId.get(contract.canonicalToolId);
    return definition === undefined ? contract : decorateEditContract(contract, definition, input.resolvedPolicy);
  });
};

const exampleInputForOutputContract = (contract: EditModelOutputContract): JsonObject => {
  switch (contract) {
    case "complete_file":
      return { path: "src/example.ts", content: "export const value = 1;\n" };
    case "old_new_text":
      return { path: "src/example.ts", search: "value = 0", replace: "value = 1" };
    case "edit_list":
      return { edits: [{ path: "src/example.ts", search: "value = 0", replace: "value = 1" }] };
    case "fenced_blocks":
      return { blocks: [{ path: "src/example.ts", body: "-value = 0\n+value = 1\n" }] };
    case "unified_patch":
      return { patch: "--- a/src/example.ts\n+++ b/src/example.ts\n@@\n-value = 0\n+value = 1\n" };
    case "structured_patch":
      return { patch: "*** Begin Patch\n*** Update File: src/example.ts\n@@\n-value = 0\n+value = 1\n*** End Patch\n" };
    case "anchor_operations":
      return { operations: [{ path: "src/example.ts", startLine: 1, endLine: 1, replacement: "value = 1\n" }] };
    case "lazy_snippet":
      return { path: "src/example.ts", snippet: "value = 1", mergeIntent: "Update the constant." };
    case "plan_then_edit":
      return { plan: "Update the constant.", edit: "value = 1" };
    case "structured_ast_operations":
      return { operations: [{ kind: "replace_literal", value: 1 }] };
    case "editor_ranges":
      return { ranges: [{ path: "src/example.ts", startOffset: 0, endOffset: 9, replacement: "value = 1" }] };
    case "custom":
      return { payload: "custom" };
  }
};
