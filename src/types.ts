import { z } from "zod";

export const MODEL_RUNTIME_ROLES = [
  "master",
  "local",
  "planner",
  "executor",
  "verifier",
  "critic",
  "summarizer",
  "fast_scout",
  "local_batch_executor",
] as const;
export const ModelRuntimeRoleSchema = z.enum(MODEL_RUNTIME_ROLES);
export type ModelRuntimeRole = z.infer<typeof ModelRuntimeRoleSchema>;

export const ModelProviderConfigRoleSchema = z.enum(["master", "local"]);
export type ModelProviderConfigRole = z.infer<typeof ModelProviderConfigRoleSchema>;

export const ModelProviderSchema = z.enum([
  "openai",
  "openai-compatible",
  "anthropic",
  "local-mlx",
  "vllm",
  "llama.cpp",
  "ollama",
  "custom",
]);
export type ModelProvider = z.infer<typeof ModelProviderSchema>;

export const ModelEndpointKindSchema = z.enum(["chat_completions", "responses", "custom"]);
export type ModelEndpointKind = z.infer<typeof ModelEndpointKindSchema>;

export const ProviderDiscoverySourceSchema = z.enum(["configured", "deterministic_default", "measured"]);
export type ProviderDiscoverySource = z.infer<typeof ProviderDiscoverySourceSchema>;

export const ContextWindowSourceSchema = z.enum(["configured", "deterministic_floor", "measured"]);
export type ContextWindowSource = z.infer<typeof ContextWindowSourceSchema>;

export const RuntimeProfileIdSchema = z.string().min(1).regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/);

/**
 * Authentication mode for a provider config role.
 *  - "api_key": read a raw API key from process.env[apiKeyEnv] (existing default).
 *  - "oauth":   resolve a subscription token from ~/.bag/oauth/<oauthProvider>.json,
 *               refreshing if expired. Used for Anthropic Pro/Max, ChatGPT Plus/Pro
 *               (Codex), and GitHub Copilot subscriptions.
 *
 * Defaulting to "api_key" preserves backwards compatibility with all existing
 * bag.config.json files that do not set this field.
 */
export const AuthTypeSchema = z.enum(["api_key", "oauth"]);
export type AuthType = z.infer<typeof AuthTypeSchema>;

export const OAuthProviderIdSchema = z.enum(["anthropic", "openai", "github-copilot"]);
export type OAuthProviderIdConfig = z.infer<typeof OAuthProviderIdSchema>;

export const ModelRoleBindingSchema = z.object({
  source: ModelProviderConfigRoleSchema,
  fallbackRole: ModelRuntimeRoleSchema.optional(),
});
export type ModelRoleBinding = z.infer<typeof ModelRoleBindingSchema>;

export const DEFAULT_MODEL_ROLE_BINDINGS = {
  master: { source: "master" },
  local: { source: "local" },
  planner: { source: "master", fallbackRole: "local" },
  executor: { source: "local", fallbackRole: "master" },
  verifier: { source: "master", fallbackRole: "local" },
  critic: { source: "master", fallbackRole: "local" },
  summarizer: { source: "local", fallbackRole: "master" },
  fast_scout: { source: "local", fallbackRole: "master" },
  local_batch_executor: { source: "local", fallbackRole: "executor" },
} as const satisfies Record<ModelRuntimeRole, ModelRoleBinding>;

export const ModelRoleBindingsSchema = z.object({
  master: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.master),
  local: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.local),
  planner: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.planner),
  executor: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.executor),
  verifier: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.verifier),
  critic: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.critic),
  summarizer: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.summarizer),
  fast_scout: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.fast_scout),
  local_batch_executor: ModelRoleBindingSchema.default(DEFAULT_MODEL_ROLE_BINDINGS.local_batch_executor),
}).default(DEFAULT_MODEL_ROLE_BINDINGS);
export type ModelRoleBindings = z.infer<typeof ModelRoleBindingsSchema>;

/**
 * `pathProfile` collects every filesystem-path convention BAG previously hard-
 * coded across the codebase (workspace-snapshot exclusion globs, scratch
 * directories flagged by hygiene audits, the default subprocess `PATH` cited
 * by the pre-submit self-check / executor system prompts) into a SINGLE
 * config block so deployments with non-Linux conventions (Docker images that
 * scratch into `/scratch`, alternate `PATH` layouts, etc.) can override them
 * declaratively. Defaults are the historical Linux conventions, so omitting
 * the block from `bag.config.json` reproduces pre-config behavior byte-for-
 * byte. See `docs/bag-path-profile.md` for the full rationale and override
 * recipes.
 */
export const PathProfileSchema = z.object({
  /**
   * Directories EXCLUDED from workspace snapshots taken by
   * `instruction-verifier.ts` (probe-cleanup snapshot/diff). Defaults to
   * the BAG-private `.bag/` cache and the `.git/` metadata directory.
   * Must contain at least one entry — an empty list would let probes
   * silently mutate `.bag/` artifacts that the harbor verifier scans.
   */
  metadataDirs: z.array(z.string().min(1)).min(1).default([".bag", ".git"]),
  /**
   * Scratch / tmpfs directories flagged by `scratch-hygiene.ts` when the
   * agent writes there without a matching cleanup. Defaults to the POSIX
   * conventions (`/tmp`, `/var/tmp`); container deployments that scratch
   * into `/scratch` can override.
   */
  scratchDirs: z.array(z.string().min(1)).min(1).default(["/tmp", "/var/tmp"]),
  /**
   * Default subprocess `PATH` directories cited by the pre-submit self-
   * check prompt and the autonomous-coding-turn system prompt's SUBPROCESS-
   * PATH GATE. Defaults to the Linux convention
   * (`/usr/local/bin:/usr/bin:/bin`); deployments with custom PATH layouts
   * (e.g. NixOS, alpine, alt FHS) override here.
   */
  systemPathDirs: z
    .array(z.string().min(1))
    .min(1)
    .default(["/usr/local/bin", "/usr/bin", "/bin"]),
}).default({
  metadataDirs: [".bag", ".git"],
  scratchDirs: ["/tmp", "/var/tmp"],
  systemPathDirs: ["/usr/local/bin", "/usr/bin", "/bin"],
});
export type PathProfile = z.infer<typeof PathProfileSchema>;

/** The canonical Linux defaults exposed for direct consumption (tests, fall-back paths). */
export const DEFAULT_PATH_PROFILE: PathProfile = {
  metadataDirs: [".bag", ".git"],
  scratchDirs: ["/tmp", "/var/tmp"],
  systemPathDirs: ["/usr/local/bin", "/usr/bin", "/bin"],
};

export const BagConfigSchema = z.object({
  artifactDir: z.string().default(".bag"),
  pathProfile: PathProfileSchema,
  master: z.object({
    provider: ModelProviderSchema.default("openai"),
    model: z.string().default("gpt-5.5"),
    baseUrl: z.string().default("https://api.openai.com/v1"),
    apiKeyEnv: z.string().default("OPENAI_API_KEY"),
    /**
     * Optional auth-mode override. When "oauth", `apiKeyEnv` is ignored and the
     * router resolves a subscription token from ~/.bag/oauth/<oauthProvider>.json.
     */
    authType: AuthTypeSchema.default("api_key"),
    oauthProvider: OAuthProviderIdSchema.optional(),
    endpointKind: ModelEndpointKindSchema.default("chat_completions"),
    serverId: RuntimeProfileIdSchema.optional(),
    serverProfileId: RuntimeProfileIdSchema.optional(),
    contextWindowTokens: z.number().int().positive().optional(),
    maxTokens: z.number().int().positive().default(4096),
    temperature: z.number().min(0).max(2).default(0.2),
  }).default({
    provider: "openai",
    model: "gpt-5.5",
    baseUrl: "https://api.openai.com/v1",
    apiKeyEnv: "OPENAI_API_KEY",
    authType: "api_key",
    endpointKind: "chat_completions",
    maxTokens: 4096,
    temperature: 0.2,
  }),
  local: z.object({
    provider: ModelProviderSchema.default("openai-compatible"),
    model: z.string().default("majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit"),
    baseUrl: z.string().default("http://127.0.0.1:18082/v1"),
    apiKey: z.string().default("local"),
    apiKeyEnv: z.string().optional(),
    /** See `master.authType` — same semantics for the local provider role. */
    authType: AuthTypeSchema.default("api_key"),
    oauthProvider: OAuthProviderIdSchema.optional(),
    endpointKind: ModelEndpointKindSchema.default("chat_completions"),
    serverId: RuntimeProfileIdSchema.optional(),
    serverProfileId: RuntimeProfileIdSchema.optional(),
    contextWindowTokens: z.number().int().positive().optional(),
    maxTokens: z.number().int().positive().default(2048),
    temperature: z.number().min(0).max(2).default(0.1),
  }).default({
    provider: "openai-compatible",
    model: "majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit",
    baseUrl: "http://127.0.0.1:18082/v1",
    apiKey: "local",
    authType: "api_key",
    endpointKind: "chat_completions",
    maxTokens: 2048,
    temperature: 0.1,
  }),
  modelRoles: ModelRoleBindingsSchema,
  policy: z.object({
    interactiveConcurrency: z.number().int().positive().default(12),
    executorConcurrency: z.number().int().positive().default(16),
    maxExecutorConcurrency: z.number().int().positive().default(24),
    maxSubAgentCalls: z.number().int().positive().default(40),
    maxTurns: z.number().int().positive().default(8),
    contextFiles: z.number().int().positive().default(160),
    contextCharsPerFile: z.number().int().positive().default(6000),
    selfEvalThreshold: z.number().min(0).max(1).default(0.78),
    requirePermissions: z.boolean().default(false),
  }).default({
    interactiveConcurrency: 12,
    executorConcurrency: 16,
    maxExecutorConcurrency: 24,
    maxSubAgentCalls: 40,
    maxTurns: 8,
    contextFiles: 160,
    contextCharsPerFile: 6000,
    selfEvalThreshold: 0.78,
    requirePermissions: false,
  }),
  telemetry: z.object({
    enabled: z.boolean().default(true),
    jsonl: z.string().default(".bag/telemetry/events.jsonl"),
    metrics: z.string().default(".bag/telemetry/metrics.json"),
    spans: z.string().default(".bag/telemetry/spans.jsonl"),
  }).default({
    enabled: true,
    jsonl: ".bag/telemetry/events.jsonl",
    metrics: ".bag/telemetry/metrics.json",
    spans: ".bag/telemetry/spans.jsonl",
  }),
});
export type BagConfig = z.infer<typeof BagConfigSchema>;

export const InterviewTurnSchema = z.object({
  question: z.string(),
  rationale: z.string(),
  acceptedFacts: z.array(z.string()).default([]),
  openQuestions: z.array(z.string()).default([]),
  canGeneratePrdNow: z.boolean().default(false),
  suggestedNextAction: z.enum(["continue_interview", "generate_prd"]).default("continue_interview"),
});
export type InterviewTurn = z.infer<typeof InterviewTurnSchema>;

export const PrdSectionKeySchema = z.enum([
  "table_of_contents",
  "users_and_jobs",
  "problem_statement",
  "product_scope",
  "user_workflows",
  "functional_requirements",
  "integrations_and_constraints",
  "risks_and_open_questions",
  "acceptance_criteria",
  "delivery_plan",
]);

export const PrdArtifactSchema = z.object({
  documentTitle: z.string(),
  sections: z.array(
    z.object({
      key: PrdSectionKeySchema,
      title: z.string(),
      body: z.string(),
    }),
  ),
});
export type PrdArtifact = z.infer<typeof PrdArtifactSchema>;

export const DagIssueSchema = z.object({
  issueId: z.string(),
  issueType: z.enum(["epic", "task"]),
  title: z.string(),
  body: z.string(),
  status: z.enum(["open", "in_progress", "blocked", "closed"]).default("open"),
  sortIndex: z.number().int(),
  plannerMetadata: z.object({
    suggestedOwner: z.enum(["master", "executor", "critic"]).default("executor"),
    expectedFiles: z.array(z.string()).default([]),
    verificationCommands: z.array(z.string()).default([]),
    risk: z.enum(["low", "medium", "high"]).default("medium"),
  }),
});
export type DagIssue = z.infer<typeof DagIssueSchema>;

export const DagDependencySchema = z.object({
  fromIssueId: z.string(),
  toIssueId: z.string(),
  relation: z.enum(["depends_on", "blocks", "relates_to"]).default("depends_on"),
});
export type DagDependency = z.infer<typeof DagDependencySchema>;

export const DagPlanSchema = z.object({
  summary: z.object({
    planId: z.string(),
    title: z.string(),
    status: z.enum(["draft", "ready", "running", "completed", "failed"]).default("draft"),
    issueCount: z.number().int().nonnegative(),
    dependencyCount: z.number().int().nonnegative(),
    chosenTier: z.enum(["small", "medium", "large"]).default("medium"),
  }),
  issues: z.array(DagIssueSchema),
  dependencies: z.array(DagDependencySchema),
});
export type DagPlan = z.infer<typeof DagPlanSchema>;

export const ContextScoutFindingSchema = z.object({
  file: z.string(),
  reason: z.string(),
  signals: z.array(z.string()).default([]),
  confidence: z.number().min(0).max(1).default(0.5),
});
export type ContextScoutFinding = z.infer<typeof ContextScoutFindingSchema>;

export const SelfEvaluationSchema = z.object({
  score: z.number().min(0).max(1),
  passed: z.boolean(),
  strengths: z.array(z.string()).default([]),
  weaknesses: z.array(z.string()).default([]),
  improvementActions: z.array(z.string()).default([]),
});
export type SelfEvaluation = z.infer<typeof SelfEvaluationSchema>;

export const StepMetricSchema = z.object({
  step: z.string(),
  startedAt: z.string(),
  completedAt: z.string(),
  durationMs: z.number(),
  ok: z.boolean(),
  modelRole: z.union([ModelRuntimeRoleSchema, z.literal("deterministic")]).default("deterministic"),
  inputTokens: z.number().int().nonnegative().optional(),
  outputTokens: z.number().int().nonnegative().optional(),
  error: z.string().optional(),
});
export type StepMetric = z.infer<typeof StepMetricSchema>;

export const LlmCallMetricSchema = z.object({
  role: ModelRuntimeRoleSchema,
  resolvedRole: ModelRuntimeRoleSchema.optional(),
  providerConfigRole: ModelProviderConfigRoleSchema.optional(),
  fallbackFromRole: ModelRuntimeRoleSchema.optional(),
  provider: ModelProviderSchema.optional(),
  endpointKind: ModelEndpointKindSchema.optional(),
  modelServerId: z.string().optional(),
  modelServerProfileId: z.string().optional(),
  contextWindowTokens: z.number().int().positive().optional(),
  maxOutputTokens: z.number().int().positive().optional(),
  model: z.string(),
  endpoint: z.string(),
  startedAt: z.string(),
  completedAt: z.string(),
  durationMs: z.number(),
  ok: z.boolean(),
  httpStatus: z.number().int().optional(),
  promptTokens: z.number().int().nonnegative().optional(),
  completionTokens: z.number().int().nonnegative().optional(),
  totalTokens: z.number().int().nonnegative().optional(),
  error: z.string().optional(),
  /**
   * Free-form BAG-subsystem attribution: which call site made this LLM
   * request (e.g. "instruction-summarizer", "task-shape-classifier",
   * "probe-extractor", "pre-submit-self-check", "dag-planner",
   * "autonomous-coding-turn"). Enables per-(model, purpose) cost & quality
   * analysis without mutating BAG's role abstraction.
   */
  purpose: z.string().optional(),
});
export type LlmCallMetric = z.infer<typeof LlmCallMetricSchema>;

export const ToolCallMetricSchema = z.object({
  toolName: z.string(),
  namespace: z.string().optional(),
  descriptionVersion: z.string().optional(),
  startedAt: z.string(),
  completedAt: z.string(),
  durationMs: z.number(),
  ok: z.boolean(),
  retryCount: z.number().int().nonnegative().default(0),
  argumentBytes: z.number().int().nonnegative(),
  argumentHash: z.string(),
  resultBytes: z.number().int().nonnegative().optional(),
  resultKind: z.enum(["text", "json", "binary", "empty", "unknown"]).default("unknown"),
  error: z.string().optional(),
  errorName: z.string().optional(),
});
export type ToolCallMetric = z.infer<typeof ToolCallMetricSchema>;

export const RunManifestSchema = z.object({
  runId: z.string(),
  createdAt: z.string(),
  command: z.string(),
  task: z.string(),
  config: BagConfigSchema,
  artifacts: z.record(z.string(), z.string()),
  metrics: z.array(StepMetricSchema),
  llmMetrics: z.array(LlmCallMetricSchema).default([]),
  toolMetrics: z.array(ToolCallMetricSchema).default([]),
  selfEvaluation: SelfEvaluationSchema,
});
export type RunManifest = z.infer<typeof RunManifestSchema>;
