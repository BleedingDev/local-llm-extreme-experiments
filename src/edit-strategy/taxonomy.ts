import { z } from "zod";
import { OptimizerIdSchema, OptimizerVersionSchema } from "../optimizer/types";
import { EditErrorCodeSchema, EditStrategyFamilySchema, type EditStrategyFamily } from "./types";

export const EditModelOutputContractSchema = z.enum([
  "complete_file",
  "old_new_text",
  "edit_list",
  "fenced_blocks",
  "unified_patch",
  "structured_patch",
  "anchor_operations",
  "lazy_snippet",
  "plan_then_edit",
  "structured_ast_operations",
  "editor_ranges",
  "custom",
]);
export type EditModelOutputContract = z.infer<typeof EditModelOutputContractSchema>;

export const EditApplicationKindSchema = z.enum([
  "direct_write",
  "deterministic_adapter",
  "model_apply",
  "external_capability",
]);
export type EditApplicationKind = z.infer<typeof EditApplicationKindSchema>;

export const EditStrategyMaturitySchema = z.enum([
  "baseline",
  "candidate",
  "experimental",
  "future_gate",
]);
export type EditStrategyMaturity = z.infer<typeof EditStrategyMaturitySchema>;

export const EditFutureGateSchema = z.enum([
  "none",
  "lsp_explicit_approval_required",
  "structured_ast_research_required",
  "apply_model_capacity_required",
]);
export type EditFutureGate = z.infer<typeof EditFutureGateSchema>;

export const EditTraceRequirementSchema = z.enum([
  "read_snapshot_refs",
  "input_output_hashes",
  "parse_status",
  "match_cardinality",
  "anchor_status",
  "preview_diff",
  "permission_status",
  "apply_status",
  "post_apply_consistency",
  "verification_status",
  "repair_status",
  "rollback_status",
  "fallback_path",
  "token_latency_cost",
  "redaction_status",
]);
export type EditTraceRequirement = z.infer<typeof EditTraceRequirementSchema>;

export const CanonicalEditStrategyDefinitionSchema = z.object({
  strategyId: OptimizerIdSchema,
  strategyVersion: OptimizerVersionSchema,
  family: EditStrategyFamilySchema,
  displayName: z.string().min(1),
  summary: z.string().min(1),
  modelOutputContract: EditModelOutputContractSchema,
  applicationKind: EditApplicationKindSchema,
  maturity: EditStrategyMaturitySchema,
  futureGate: EditFutureGateSchema.default("none"),
  deterministicApply: z.boolean(),
  supportsMultiFile: z.boolean(),
  supportsPartialRead: z.boolean(),
  requiresWholeFileOutput: z.boolean(),
  initialExperimentCandidate: z.boolean().default(false),
  expectedFailureCodes: z.array(EditErrorCodeSchema).default([]),
  traceRequirements: z.array(EditTraceRequirementSchema).default([]),
}).strict().superRefine((definition, ctx) => {
  if (definition.maturity !== "future_gate" && definition.futureGate !== "none") {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "futureGate is only valid for future_gate strategies",
      path: ["futureGate"],
    });
  }
  if (definition.maturity === "future_gate" && definition.initialExperimentCandidate) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "future-gated strategies cannot be initial experiment candidates",
      path: ["initialExperimentCandidate"],
    });
  }
});
export type CanonicalEditStrategyDefinition = z.infer<typeof CanonicalEditStrategyDefinitionSchema>;
export type CanonicalEditStrategyDefinitionInput = z.input<typeof CanonicalEditStrategyDefinitionSchema>;

const commonTraceRequirements: EditTraceRequirement[] = [
  "read_snapshot_refs",
  "input_output_hashes",
  "parse_status",
  "preview_diff",
  "permission_status",
  "apply_status",
  "post_apply_consistency",
  "verification_status",
  "repair_status",
  "rollback_status",
  "fallback_path",
  "token_latency_cost",
  "redaction_status",
];

export const CANONICAL_EDIT_STRATEGY_DEFINITIONS = [
  {
    strategyId: "edit.whole-file.acp-write.v1",
    strategyVersion: "v1",
    family: "whole_file",
    displayName: "Whole-file ACP write",
    summary: "Model returns complete file content and the ACP filesystem writes the full file after policy checks.",
    modelOutputContract: "complete_file",
    applicationKind: "direct_write",
    maturity: "baseline",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: false,
    requiresWholeFileOutput: true,
    initialExperimentCandidate: true,
    expectedFailureCodes: [
      "truncation_induced_error",
      "scope_violation",
      "post_apply_syntax_failure",
      "post_apply_behavior_failure",
    ],
    traceRequirements: commonTraceRequirements,
  },
  {
    strategyId: "edit.exact-replace.v1",
    strategyVersion: "v1",
    family: "exact_replace",
    displayName: "Exact search/replace",
    summary: "Model provides one old-text/new-text replacement with deterministic uniqueness and stale-read checks.",
    modelOutputContract: "old_new_text",
    applicationKind: "deterministic_adapter",
    maturity: "candidate",
    deterministicApply: true,
    supportsMultiFile: false,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    initialExperimentCandidate: true,
    expectedFailureCodes: [
      "exact_match_not_found",
      "exact_match_ambiguous",
      "anchor_stale",
      "post_apply_behavior_failure",
    ],
    traceRequirements: [...commonTraceRequirements, "match_cardinality"],
  },
  {
    strategyId: "edit.multi-exact-replace.v1",
    strategyVersion: "v1",
    family: "multi_exact_replace",
    displayName: "Multi exact replace",
    summary: "Model provides an ordered list of exact replacements applied atomically per file.",
    modelOutputContract: "edit_list",
    applicationKind: "deterministic_adapter",
    maturity: "candidate",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    expectedFailureCodes: [
      "exact_match_not_found",
      "exact_match_ambiguous",
      "overlapping_edits",
      "partial_apply",
      "post_apply_behavior_failure",
    ],
    traceRequirements: [...commonTraceRequirements, "match_cardinality"],
  },
  {
    strategyId: "edit.fenced-diff.v1",
    strategyVersion: "v1",
    family: "fenced_diff",
    displayName: "Fenced diff blocks",
    summary: "Model returns fenced search/replace or diff-like blocks where fence/path placement is a measurable contract variant.",
    modelOutputContract: "fenced_blocks",
    applicationKind: "deterministic_adapter",
    maturity: "candidate",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    expectedFailureCodes: [
      "parse_error",
      "path_or_fence_error",
      "exact_match_not_found",
      "post_apply_behavior_failure",
    ],
    traceRequirements: [...commonTraceRequirements, "match_cardinality"],
  },
  {
    strategyId: "edit.unified-diff.v1",
    strategyVersion: "v1",
    family: "unified_diff",
    displayName: "Unified/context diff",
    summary: "Model returns a context patch parsed and applied by a deterministic hunk adapter.",
    modelOutputContract: "unified_patch",
    applicationKind: "deterministic_adapter",
    maturity: "candidate",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    initialExperimentCandidate: true,
    expectedFailureCodes: [
      "parse_error",
      "hunk_context_mismatch",
      "partial_apply",
      "post_apply_behavior_failure",
    ],
    traceRequirements: commonTraceRequirements,
  },
  {
    strategyId: "edit.apply-patch.v1",
    strategyVersion: "v1",
    family: "apply_patch",
    displayName: "Structured apply patch",
    summary: "Model returns structured add/update/delete/move patch operations for a deterministic patch adapter.",
    modelOutputContract: "structured_patch",
    applicationKind: "deterministic_adapter",
    maturity: "candidate",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    initialExperimentCandidate: true,
    expectedFailureCodes: [
      "parse_error",
      "schema_validation_error",
      "hunk_context_mismatch",
      "partial_apply",
      "post_apply_behavior_failure",
    ],
    traceRequirements: commonTraceRequirements,
  },
  {
    strategyId: "edit.hash-range.experimental.v1",
    strategyVersion: "v1",
    family: "hash_range",
    displayName: "Hash/range anchored edit",
    summary: "Model targets line/range anchors with content hashes so stale context and repeated snippets become explicit signals.",
    modelOutputContract: "anchor_operations",
    applicationKind: "deterministic_adapter",
    maturity: "experimental",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    initialExperimentCandidate: true,
    expectedFailureCodes: [
      "anchor_not_found",
      "anchor_stale",
      "anchor_ambiguous",
      "hash_mismatch",
      "range_out_of_bounds",
      "post_apply_behavior_failure",
    ],
    traceRequirements: [...commonTraceRequirements, "anchor_status"],
  },
  {
    strategyId: "edit.apply-model.experimental.v1",
    strategyVersion: "v1",
    family: "apply_model",
    displayName: "Apply/editor model",
    summary: "Planner emits a lazy snippet or intent and a specialized model merges it into file content.",
    modelOutputContract: "lazy_snippet",
    applicationKind: "model_apply",
    maturity: "experimental",
    deterministicApply: false,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    expectedFailureCodes: [
      "post_apply_syntax_failure",
      "post_apply_behavior_failure",
      "fallback_masked_failure",
      "verifier_error",
    ],
    traceRequirements: commonTraceRequirements,
  },
  {
    strategyId: "edit.architect-editor.experimental.v1",
    strategyVersion: "v1",
    family: "architect_editor",
    displayName: "Architect/editor routing",
    summary: "A planner model produces an edit plan and an editor model serializes that plan into another edit strategy.",
    modelOutputContract: "plan_then_edit",
    applicationKind: "model_apply",
    maturity: "experimental",
    deterministicApply: false,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    expectedFailureCodes: [
      "parse_error",
      "post_apply_behavior_failure",
      "fallback_masked_failure",
      "verifier_error",
    ],
    traceRequirements: commonTraceRequirements,
  },
  {
    strategyId: "edit.ast-structured.future.v1",
    strategyVersion: "v1",
    family: "ast_structured",
    displayName: "AST/structured edit",
    summary: "Model emits structured syntax-node operations. This needs language/parser research before activation.",
    modelOutputContract: "structured_ast_operations",
    applicationKind: "external_capability",
    maturity: "future_gate",
    futureGate: "structured_ast_research_required",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    expectedFailureCodes: [
      "schema_validation_error",
      "scope_violation",
      "post_apply_behavior_failure",
    ],
    traceRequirements: commonTraceRequirements,
  },
  {
    strategyId: "edit.range-native.future.v1",
    strategyVersion: "v1",
    family: "range_native",
    displayName: "Range-native editor edit",
    summary: "Editor-buffer range edits tied to fresh document versions. Future-gated with LSP work.",
    modelOutputContract: "editor_ranges",
    applicationKind: "external_capability",
    maturity: "future_gate",
    futureGate: "lsp_explicit_approval_required",
    deterministicApply: true,
    supportsMultiFile: true,
    supportsPartialRead: true,
    requiresWholeFileOutput: false,
    expectedFailureCodes: [
      "range_out_of_bounds",
      "anchor_stale",
      "post_apply_behavior_failure",
    ],
    traceRequirements: commonTraceRequirements,
  },
] as const satisfies readonly CanonicalEditStrategyDefinitionInput[];

export const parseCanonicalEditStrategyDefinitions = (
  definitions: readonly unknown[] = CANONICAL_EDIT_STRATEGY_DEFINITIONS,
): CanonicalEditStrategyDefinition[] =>
  z.array(CanonicalEditStrategyDefinitionSchema).parse(definitions);

export const canonicalEditStrategyDefinitionsByFamily = (): ReadonlyMap<
  EditStrategyFamily,
  CanonicalEditStrategyDefinition[]
> => {
  const byFamily = new Map<EditStrategyFamily, CanonicalEditStrategyDefinition[]>();
  for (const definition of parseCanonicalEditStrategyDefinitions()) {
    const definitions = byFamily.get(definition.family) ?? [];
    definitions.push(definition);
    byFamily.set(definition.family, definitions);
  }
  return byFamily;
};

export const initialExperimentalEditStrategyIds = (): string[] =>
  parseCanonicalEditStrategyDefinitions()
    .filter((definition) => definition.initialExperimentCandidate)
    .map((definition) => definition.strategyId)
    .sort((left, right) => left.localeCompare(right));
