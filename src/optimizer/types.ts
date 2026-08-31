import { z } from "zod";
import {
  ContextWindowSourceSchema,
  ModelEndpointKindSchema,
  ModelProviderConfigRoleSchema,
  ModelProviderSchema,
  ModelRuntimeRoleSchema,
  ProviderDiscoverySourceSchema,
} from "../types";
export { ModelProviderSchema } from "../types";
export type { ModelProvider } from "../types";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    z.record(z.string(), JsonValueSchema),
  ]),
);

export const OptimizerIdSchema = z.string().min(1).regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/);
export type OptimizerId = z.infer<typeof OptimizerIdSchema>;

export const OptimizerVersionSchema = z.string().min(1).default("v1");
export type OptimizerVersion = z.infer<typeof OptimizerVersionSchema>;

export const RegistryRecordKindSchema = z.enum([
  "model_profile",
  "codebase_profile",
  "model_codebase_policy",
  "canonical_tool_spec",
  "rendered_tool_contract",
  "candidate_patch",
  "eval_result",
  "promotion_decision",
]);
export type RegistryRecordKind = z.infer<typeof RegistryRecordKindSchema>;

export const RegistryRecordStatusSchema = z.enum(["draft", "active", "promoted", "retired", "rejected"]);
export type RegistryRecordStatus = z.infer<typeof RegistryRecordStatusSchema>;

export const RegistryRecordSchema = z.object({
  registryRecordId: OptimizerIdSchema,
  recordKind: RegistryRecordKindSchema,
  schemaVersion: OptimizerVersionSchema,
  recordVersion: OptimizerVersionSchema,
  status: RegistryRecordStatusSchema.default("draft"),
  createdAt: z.string(),
  updatedAt: z.string(),
  contentHash: z.string().optional(),
  supersedesRecordId: OptimizerIdSchema.optional(),
  labels: z.array(z.string()).default([]),
});
export type RegistryRecord = z.infer<typeof RegistryRecordSchema>;

export const ToolCallingModeSchema = z.enum(["native", "json", "text", "disabled"]);
export type ToolCallingMode = z.infer<typeof ToolCallingModeSchema>;

export const StructuredOutputModeSchema = z.enum(["json_schema", "json_object", "text"]);
export type StructuredOutputMode = z.infer<typeof StructuredOutputModeSchema>;

export const ModelProfileSchema = z.object({
  modelProfileId: OptimizerIdSchema,
  displayName: z.string().min(1),
  modelRole: ModelRuntimeRoleSchema.optional(),
  providerConfigRole: ModelProviderConfigRoleSchema.optional(),
  fallbackModelRole: ModelRuntimeRoleSchema.optional(),
  provider: ModelProviderSchema,
  model: z.string().min(1),
  modelFamily: z.string().min(1).optional(),
  baseUrl: z.string().min(1).optional(),
  endpointKind: ModelEndpointKindSchema.default("chat_completions"),
  modelServerId: OptimizerIdSchema.optional(),
  modelServerProfileId: OptimizerIdSchema.optional(),
  providerDiscoverySource: ProviderDiscoverySourceSchema.optional(),
  contextWindowTokens: z.number().int().positive(),
  contextWindowSource: ContextWindowSourceSchema.optional(),
  maxOutputTokens: z.number().int().positive(),
  measuredTtftMs: z.number().nonnegative().optional(),
  measuredGenerationTokensPerSecond: z.number().nonnegative().optional(),
  measuredConcurrentThroughputTokensPerSecond: z.number().nonnegative().optional(),
  measuredMaxConcurrentRequests: z.number().int().positive().optional(),
  defaultTemperature: z.number().min(0).max(2).default(0.1),
  toolCallingMode: ToolCallingModeSchema.default("json"),
  structuredOutputMode: StructuredOutputModeSchema.default("json_schema"),
  supportsParallelToolCalls: z.boolean().default(false),
  tokenizer: z.string().optional(),
  promptStyle: z.enum(["system_user", "chatml", "plain_text"]).default("system_user"),
  resultStyleVersion: OptimizerVersionSchema,
  verificationPolicyVersion: OptimizerVersionSchema,
});
export type ModelProfile = z.infer<typeof ModelProfileSchema>;

export const CommandSpecSchema = z.object({
  commandId: OptimizerIdSchema,
  command: z.array(z.string().min(1)).min(1),
  cwd: z.string().optional(),
  timeoutMs: z.number().int().positive().optional(),
  required: z.boolean().default(true),
});
export type CommandSpec = z.infer<typeof CommandSpecSchema>;

export const CodebaseTestRiskTierSchema = z.object({
  tierId: OptimizerIdSchema,
  description: z.string().min(1),
  commandIds: z.array(OptimizerIdSchema).default([]),
  protectedPaths: z.array(z.string()).default([]),
  required: z.boolean().default(true),
});
export type CodebaseTestRiskTier = z.infer<typeof CodebaseTestRiskTierSchema>;

export const CodebaseKnownFailureSchema = z.object({
  failureId: OptimizerIdSchema,
  source: z.enum(["verifier", "profile_observation", "operator_note"]).default("profile_observation"),
  commandId: OptimizerIdSchema.optional(),
  severity: z.enum(["warning", "failure", "critical"]).default("failure"),
  summary: z.string().min(1),
  lastExitCode: z.number().int().optional(),
});
export type CodebaseKnownFailure = z.infer<typeof CodebaseKnownFailureSchema>;

export const AcpClientQuirkSchema = z.object({
  quirkId: OptimizerIdSchema,
  affectedCapability: z.string().min(1),
  behavior: z.string().min(1),
  mitigation: z.string().min(1).optional(),
});
export type AcpClientQuirk = z.infer<typeof AcpClientQuirkSchema>;

export const CodebaseProfileSchema = z.object({
  codebaseProfileId: OptimizerIdSchema,
  displayName: z.string().min(1),
  rootFingerprint: z.string().min(1),
  languages: z.array(z.enum(["typescript", "javascript", "python", "rust", "go", "shell", "markdown", "other"])).default([]),
  packageManagers: z.array(z.enum(["npm", "bun", "pnpm", "yarn", "pip", "cargo", "go", "none"])).default([]),
  primaryPackageManager: z.enum(["npm", "bun", "pnpm", "yarn", "pip", "cargo", "go", "none"]).optional(),
  sourceRoots: z.array(z.string()).default(["src"]),
  generatedDirs: z.array(z.string()).default([]),
  ignoredDirs: z.array(z.string()).default([]),
  testCommands: z.array(CommandSpecSchema).default([]),
  typecheckCommands: z.array(CommandSpecSchema).default([]),
  lintCommands: z.array(CommandSpecSchema).default([]),
  testRiskTiers: z.array(CodebaseTestRiskTierSchema).default([]),
  protectedPaths: z.array(z.string()).default([]),
  conventions: z.array(z.string()).default([]),
  knownFailures: z.array(CodebaseKnownFailureSchema).default([]),
  acpClientQuirks: z.array(AcpClientQuirkSchema).default([]),
  verificationPolicyVersion: OptimizerVersionSchema,
});
export type CodebaseProfile = z.infer<typeof CodebaseProfileSchema>;

export const CandidateScopeSchema = z.object({
  artifactKind: z.enum(["model_profile", "codebase_profile", "model_codebase_policy", "canonical_tool_spec", "rendered_tool_contract"]),
  artifactId: OptimizerIdSchema,
  allowedJsonPointers: z.array(z.string().regex(/^\//)).default([]),
});
export type CandidateScope = z.infer<typeof CandidateScopeSchema>;

export const VerificationGateSchema = z.object({
  gateId: OptimizerIdSchema,
  commandId: OptimizerIdSchema.optional(),
  metric: z.string().min(1).optional(),
  comparator: z.enum(["lt", "lte", "eq", "gte", "gt"]).default("gte"),
  threshold: z.number(),
  required: z.boolean().default(true),
});
export type VerificationGate = z.infer<typeof VerificationGateSchema>;

export const ModelCodebasePolicySchema = z.object({
  policyId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  codebaseRootFingerprint: z.string().min(1).optional(),
  status: z.enum(["draft", "evaluating", "promoted", "retired"]).default("draft"),
  canonicalToolVersion: OptimizerVersionSchema,
  renderedToolVersion: OptimizerVersionSchema,
  resultStyleVersion: OptimizerVersionSchema,
  verificationPolicyVersion: OptimizerVersionSchema,
  editStrategyVersion: OptimizerVersionSchema.default("edit-strategy.v1"),
  renderedEditContractVersion: OptimizerVersionSchema.default("rendered-edit-contract.v1"),
  editFallbackPolicyVersion: OptimizerVersionSchema.default("edit-fallback.v1"),
  editRepairPolicyVersion: OptimizerVersionSchema.default("edit-repair.v1"),
  editVerifierPolicyVersion: OptimizerVersionSchema.default("edit-verifier.v1"),
  editObjectiveSetId: OptimizerIdSchema.default("edit-objectives.default.v1"),
  candidateScopes: z.array(CandidateScopeSchema).default([]),
  verificationGates: z.array(VerificationGateSchema).default([]),
  maxConcurrentEvaluations: z.number().int().positive().default(1),
  riskTolerance: z.enum(["low", "medium", "high"]).default("low"),
});
export type ModelCodebasePolicy = z.infer<typeof ModelCodebasePolicySchema>;

export const ToolResultStyleSchema = z.enum(["text", "json", "artifact_ref", "structured_error"]);
export type ToolResultStyle = z.infer<typeof ToolResultStyleSchema>;

export const CanonicalToolSpecSchema = z.object({
  canonicalToolId: OptimizerIdSchema,
  canonicalToolVersion: OptimizerVersionSchema,
  namespace: OptimizerIdSchema,
  name: OptimizerIdSchema,
  title: z.string().min(1),
  description: z.string().min(1),
  inputSchema: z.record(z.string(), JsonValueSchema),
  outputSchema: z.record(z.string(), JsonValueSchema).optional(),
  resultStyle: ToolResultStyleSchema.default("text"),
  sideEffectLevel: z.enum(["none", "read", "write", "network", "process"]).default("read"),
  requiresConfirmation: z.boolean().default(false),
  examples: z.array(z.object({
    name: z.string().min(1),
    input: z.record(z.string(), JsonValueSchema),
    output: JsonValueSchema.optional(),
  })).default([]),
});
export type CanonicalToolSpec = z.infer<typeof CanonicalToolSpecSchema>;

export const RenderedToolContractSchema = z.object({
  renderedToolId: OptimizerIdSchema,
  canonicalToolId: OptimizerIdSchema,
  canonicalToolVersion: OptimizerVersionSchema,
  renderedToolVersion: OptimizerVersionSchema,
  modelProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema.optional(),
  renderer: OptimizerIdSchema,
  rendererVersion: OptimizerVersionSchema,
  name: OptimizerIdSchema,
  description: z.string().min(1),
  inputSchema: z.record(z.string(), JsonValueSchema),
  resultStyle: ToolResultStyleSchema.default("text"),
  resultStyleVersion: OptimizerVersionSchema,
  promptFragments: z.array(z.string()).default([]),
  examples: z.array(z.object({
    input: z.record(z.string(), JsonValueSchema),
    expectedResultShape: JsonValueSchema.optional(),
  })).default([]),
});
export type RenderedToolContract = z.infer<typeof RenderedToolContractSchema>;

export const JsonPatchAddOperationSchema = z.object({
  op: z.literal("add"),
  path: z.string().regex(/^\//),
  value: JsonValueSchema,
}).strict();

export const JsonPatchReplaceOperationSchema = z.object({
  op: z.literal("replace"),
  path: z.string().regex(/^\//),
  value: JsonValueSchema,
}).strict();

export const JsonPatchRemoveOperationSchema = z.object({
  op: z.literal("remove"),
  path: z.string().regex(/^\//),
}).strict();

export const CandidatePatchOperationSchema = z.discriminatedUnion("op", [
  JsonPatchAddOperationSchema,
  JsonPatchReplaceOperationSchema,
  JsonPatchRemoveOperationSchema,
]);
export type CandidatePatchOperation = z.infer<typeof CandidatePatchOperationSchema>;

export const CandidatePatchSchema = z.object({
  candidatePatchId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema.optional(),
  baselinePolicyId: OptimizerIdSchema.optional(),
  candidatePolicyId: OptimizerIdSchema.optional(),
  codebaseRootFingerprint: z.string().min(1).optional(),
  scope: CandidateScopeSchema,
  operations: z.array(CandidatePatchOperationSchema).min(1),
  rationale: z.string().min(1),
  createdAt: z.string(),
  sourceTraceIds: z.array(z.string()).default([]),
  evidenceBundleIds: z.array(OptimizerIdSchema).optional(),
  scorecardIds: z.array(OptimizerIdSchema).optional(),
  promotionDecisionId: OptimizerIdSchema.optional(),
  rollbackCheckpointPath: z.string().min(1).optional(),
});
export type CandidatePatch = z.infer<typeof CandidatePatchSchema>;

export const EvalMetricSchema = z.object({
  metricId: OptimizerIdSchema,
  value: z.number(),
  unit: z.enum(["score", "ratio", "count", "ms", "tokens", "bytes"]).default("score"),
  higherIsBetter: z.boolean().default(true),
});
export type EvalMetric = z.infer<typeof EvalMetricSchema>;

export const EvalResultSchema = z.object({
  evalResultId: OptimizerIdSchema,
  candidatePatchId: OptimizerIdSchema.optional(),
  policyId: OptimizerIdSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  codebaseRootFingerprint: z.string().min(1).optional(),
  canonicalToolVersion: OptimizerVersionSchema,
  renderedToolVersion: OptimizerVersionSchema,
  resultStyleVersion: OptimizerVersionSchema,
  verificationPolicyVersion: OptimizerVersionSchema,
  status: z.enum(["passed", "failed", "error", "inconclusive"]),
  score: z.number().min(0).max(1),
  metrics: z.array(EvalMetricSchema).default([]),
  commandResults: z.array(z.object({
    commandId: OptimizerIdSchema,
    exitCode: z.number().int(),
    durationMs: z.number().nonnegative(),
    summary: z.string().optional(),
  })).default([]),
  startedAt: z.string(),
  completedAt: z.string(),
});
export type EvalResult = z.infer<typeof EvalResultSchema>;

export const PromotionDecisionSchema = z.object({
  promotionDecisionId: OptimizerIdSchema,
  decision: z.enum(["promote", "reject", "hold"]),
  policyId: OptimizerIdSchema,
  candidatePatchId: OptimizerIdSchema.optional(),
  evalResultId: OptimizerIdSchema.optional(),
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  clientProfileId: OptimizerIdSchema.optional(),
  baselinePolicyId: OptimizerIdSchema.optional(),
  candidatePolicyId: OptimizerIdSchema.optional(),
  codebaseRootFingerprint: z.string().min(1).optional(),
  canonicalToolVersion: OptimizerVersionSchema,
  renderedToolVersion: OptimizerVersionSchema,
  resultStyleVersion: OptimizerVersionSchema,
  verificationPolicyVersion: OptimizerVersionSchema,
  evidenceBundleIds: z.array(OptimizerIdSchema).optional(),
  scorecardIds: z.array(OptimizerIdSchema).optional(),
  rollbackCheckpointPath: z.string().min(1).optional(),
  reason: z.string().min(1),
  decidedAt: z.string(),
  decidedBy: z.enum(["deterministic_gate", "human", "optimizer"]).default("deterministic_gate"),
  appliesToNewSessionsOnly: z.boolean().default(true),
});
export type PromotionDecision = z.infer<typeof PromotionDecisionSchema>;

export const OptimizerRegistryPayloadSchema = z.discriminatedUnion("recordKind", [
  z.object({ recordKind: z.literal("model_profile"), payload: ModelProfileSchema }),
  z.object({ recordKind: z.literal("codebase_profile"), payload: CodebaseProfileSchema }),
  z.object({ recordKind: z.literal("model_codebase_policy"), payload: ModelCodebasePolicySchema }),
  z.object({ recordKind: z.literal("canonical_tool_spec"), payload: CanonicalToolSpecSchema }),
  z.object({ recordKind: z.literal("rendered_tool_contract"), payload: RenderedToolContractSchema }),
  z.object({ recordKind: z.literal("candidate_patch"), payload: CandidatePatchSchema }),
  z.object({ recordKind: z.literal("eval_result"), payload: EvalResultSchema }),
  z.object({ recordKind: z.literal("promotion_decision"), payload: PromotionDecisionSchema }),
]);
export type OptimizerRegistryPayload = z.infer<typeof OptimizerRegistryPayloadSchema>;

export const OptimizerRegistryRecordSchema = RegistryRecordSchema.and(OptimizerRegistryPayloadSchema);
export type OptimizerRegistryRecord = z.infer<typeof OptimizerRegistryRecordSchema>;
