import { z } from "zod";
import {
  EvalAssertionSchema,
  EvalSplitSchema,
  type EvalAssertion,
  type EvalSplit,
} from "../eval-harness/types";
import { OptimizerIdSchema } from "../optimizer/types";

const REAL_ACP_TASK_PACK_CREATED_AT = "2026-05-04T00:00:00.000Z";
const REAL_ACP_TASK_PACK_SCHEMA_VERSION = "real-acp-corpus-task-pack.v1" as const;
const HIDDEN_SPLIT: EvalSplit = "holdout";
const SPLIT_PATTERN = ["train", "dev", "train", "holdout"] as const satisfies readonly EvalSplit[];
const FORBIDDEN_PATH_BASENAMES = new Set([
  ".gitignore",
  "package.json",
  "package-lock.json",
  "pnpm-lock.yaml",
  "yarn.lock",
  "bun.lock",
  "bun.lockb",
]);

const RelativeSafePathSchema = z.string().min(1)
  .regex(/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$)).+$/)
  .refine((path) => !path.split("/").some((segment) => FORBIDDEN_PATH_BASENAMES.has(segment)), {
    message: "task paths must not target package metadata, lockfiles, or .gitignore",
  });

export const RealAcpTaskLabelSchema = z.enum([
  "simple_edit",
  "greenfield_workspace",
  "bugfix_fail_to_pass",
  "refactor",
  "stale_context",
  "protected_path",
  "cancellation",
  "rollback",
  "applied_but_broken",
  "verifier_skip",
  "mcp_tool_failure",
  "user_correction",
]);
export type RealAcpTaskLabel = z.infer<typeof RealAcpTaskLabelSchema>;

const RealAcpRunTargetSchema = z.enum(["headless_acp", "real_consumer"]);
export type RealAcpRunTarget = z.infer<typeof RealAcpRunTargetSchema>;

const RealAcpWorkspaceKindSchema = z.enum(["fixture", "greenfield"]);
export type RealAcpWorkspaceKind = z.infer<typeof RealAcpWorkspaceKindSchema>;

const RealAcpExpectedMutationSchema = z.enum([
  "edit_existing",
  "create_files",
  "no_change",
  "rollback_to_original",
  "detect_without_final_success",
]);
export type RealAcpExpectedMutation = z.infer<typeof RealAcpExpectedMutationSchema>;

const RealAcpVerifierPolicySchema = z.enum([
  "required",
  "allowed_to_skip",
  "must_skip",
  "expected_to_fail_before_repair",
]);
export type RealAcpVerifierPolicy = z.infer<typeof RealAcpVerifierPolicySchema>;

const RealAcpWorkspaceFileSchema = z.object({
  path: RelativeSafePathSchema,
  content: z.string(),
  executable: z.boolean().default(false),
}).strict();
export type RealAcpWorkspaceFile = z.infer<typeof RealAcpWorkspaceFileSchema>;

const RealAcpWorkspaceSchema = z.object({
  workspaceId: OptimizerIdSchema,
  kind: RealAcpWorkspaceKindSchema,
  description: z.string().min(1),
  files: z.array(RealAcpWorkspaceFileSchema).default([]),
  allowedPathPrefixes: z.array(RelativeSafePathSchema).min(1),
  protectedPaths: z.array(RelativeSafePathSchema).default([]),
}).strict().superRefine((workspace, ctx) => {
  if (workspace.kind === "fixture" && workspace.files.length === 0) {
    ctx.addIssue({
      code: "custom",
      path: ["files"],
      message: "fixture workspaces need at least one seed file",
    });
  }

  const filePaths = new Set(workspace.files.map((file) => file.path));
  if (filePaths.size !== workspace.files.length) {
    ctx.addIssue({
      code: "custom",
      path: ["files"],
      message: "workspace file paths must be unique",
    });
  }
});
export type RealAcpWorkspace = z.infer<typeof RealAcpWorkspaceSchema>;

const RealAcpVerificationSchema = z.object({
  policy: RealAcpVerifierPolicySchema,
  commands: z.array(z.array(z.string().min(1)).min(1)).default([]),
  skipReason: z.string().min(1).optional(),
}).strict().superRefine((verification, ctx) => {
  if (verification.policy === "required" && verification.commands.length === 0) {
    ctx.addIssue({
      code: "custom",
      path: ["commands"],
      message: "required verification needs at least one command",
    });
  }
  if ((verification.policy === "allowed_to_skip" || verification.policy === "must_skip") && verification.skipReason == null) {
    ctx.addIssue({
      code: "custom",
      path: ["skipReason"],
      message: "verifier skip policies need a skip reason",
    });
  }
});
export type RealAcpVerification = z.infer<typeof RealAcpVerificationSchema>;

const RealAcpExpectedOutcomeSchema = z.object({
  mutation: RealAcpExpectedMutationSchema,
  expectedChangedPaths: z.array(RelativeSafePathSchema).default([]),
  expectedNoChangePaths: z.array(RelativeSafePathSchema).default([]),
  assertions: z.array(EvalAssertionSchema).min(1),
  verification: RealAcpVerificationSchema,
  hiddenHoldoutNotes: z.array(z.string().min(1)).default([]),
}).strict();
export type RealAcpExpectedOutcome = z.infer<typeof RealAcpExpectedOutcomeSchema>;

const RealAcpSplitHintSchema = z.object({
  seedOrdinal: z.number().int().nonnegative(),
  assignedBy: z.literal("pack_v1_modulo_pattern"),
  policy: z.string().min(1),
}).strict();
export type RealAcpSplitHint = z.infer<typeof RealAcpSplitHintSchema>;

const RealAcpTaskRunMetadataRequirementsSchema = z.object({
  model: z.array(z.enum([
    "modelProfileId",
    "provider",
    "model",
    "modelRole",
    "contextWindowTokens",
    "toolCallingMode",
  ])).min(1),
  codebase: z.array(z.enum([
    "codebaseProfileId",
    "rootFingerprint",
    "languageSummary",
    "testRiskTier",
    "protectedPathPolicy",
  ])).min(1),
  client: z.array(z.enum([
    "clientProfileId",
    "clientName",
    "clientVersion",
    "transport",
    "acpConsumerCapabilities",
  ])).min(1),
  profile: z.array(z.enum([
    "policyId",
    "optimizerProfileId",
    "verificationPolicyVersion",
    "resultStyleVersion",
    "canonicalToolVersion",
    "renderedToolVersion",
  ])).min(1),
}).strict();
export type RealAcpTaskRunMetadataRequirements = z.infer<typeof RealAcpTaskRunMetadataRequirementsSchema>;

export const RealAcpCorpusTaskSchema = z.object({
  taskId: OptimizerIdSchema,
  title: z.string().min(1),
  primaryLabel: RealAcpTaskLabelSchema,
  labels: z.array(RealAcpTaskLabelSchema).min(1),
  split: EvalSplitSchema,
  splitHint: RealAcpSplitHintSchema,
  optimizationAllowed: z.boolean(),
  runTargets: z.array(RealAcpRunTargetSchema).min(1),
  userPrompt: z.string().min(1),
  workspace: RealAcpWorkspaceSchema,
  expectedOutcome: RealAcpExpectedOutcomeSchema,
  correctionPrompts: z.array(z.string().min(1)).default([]),
  metadataRequirements: RealAcpTaskRunMetadataRequirementsSchema.optional(),
  timeoutMs: z.number().int().positive(),
}).strict().superRefine((task, ctx) => {
  if (!task.labels.includes(task.primaryLabel)) {
    ctx.addIssue({
      code: "custom",
      path: ["labels"],
      message: "task labels must include primaryLabel",
    });
  }
  if (task.split !== deterministicRealAcpTaskSplit(task.splitHint.seedOrdinal)) {
    ctx.addIssue({
      code: "custom",
      path: ["split"],
      message: "task split must match deterministic split hint",
    });
  }
  if (task.split === HIDDEN_SPLIT && task.optimizationAllowed) {
    ctx.addIssue({
      code: "custom",
      path: ["optimizationAllowed"],
      message: "hidden holdout tasks must not be optimizer input",
    });
  }
  if (task.primaryLabel === "greenfield_workspace" && task.workspace.kind !== "greenfield") {
    ctx.addIssue({
      code: "custom",
      path: ["workspace", "kind"],
      message: "greenfield tasks must use a greenfield workspace",
    });
  }
  if (task.primaryLabel === "protected_path" && task.workspace.protectedPaths.length === 0) {
    ctx.addIssue({
      code: "custom",
      path: ["workspace", "protectedPaths"],
      message: "protected path tasks must define protected paths",
    });
  }
  if (task.primaryLabel === "user_correction" && task.correctionPrompts.length === 0) {
    ctx.addIssue({
      code: "custom",
      path: ["correctionPrompts"],
      message: "user correction tasks must include correction prompts",
    });
  }

  const protectedPaths = new Set(task.workspace.protectedPaths);
  const noChangePaths = new Set(task.expectedOutcome.expectedNoChangePaths);
  for (const [index, path] of task.expectedOutcome.expectedChangedPaths.entries()) {
    if (protectedPaths.has(path)) {
      ctx.addIssue({
        code: "custom",
        path: ["expectedOutcome", "expectedChangedPaths", index],
        message: "expected changed paths must not include protected paths",
      });
    }
    if (noChangePaths.has(path)) {
      ctx.addIssue({
        code: "custom",
        path: ["expectedOutcome", "expectedChangedPaths", index],
        message: "a path cannot be both expected changed and expected unchanged",
      });
    }
    if (!task.workspace.allowedPathPrefixes.some((prefix) => path.startsWith(prefix))) {
      ctx.addIssue({
        code: "custom",
        path: ["expectedOutcome", "expectedChangedPaths", index],
        message: "expected changed paths must be under an allowed path prefix",
      });
    }
  }
});
export type RealAcpCorpusTask = z.infer<typeof RealAcpCorpusTaskSchema>;

export const RealAcpTaskPackSchema = z.object({
  taskPackId: OptimizerIdSchema,
  schemaVersion: z.literal(REAL_ACP_TASK_PACK_SCHEMA_VERSION),
  createdAt: z.string().datetime({ offset: true }),
  purpose: z.string().min(1),
  splitPolicy: z.object({
    policyId: OptimizerIdSchema,
    visibleOptimizationSplits: z.array(EvalSplitSchema).min(1),
    hiddenSplits: z.array(EvalSplitSchema).min(1),
    deterministicPattern: z.array(EvalSplitSchema).min(1),
    guidance: z.array(z.string().min(1)).min(1),
  }).strict(),
  runMetadataRequirements: RealAcpTaskRunMetadataRequirementsSchema,
  tasks: z.array(RealAcpCorpusTaskSchema).min(1),
}).strict().superRefine((pack, ctx) => {
  const ids = new Set(pack.tasks.map((task) => task.taskId));
  if (ids.size !== pack.tasks.length) {
    ctx.addIssue({
      code: "custom",
      path: ["tasks"],
      message: "real ACP task ids must be unique",
    });
  }

  const labels = new Set(pack.tasks.flatMap((task) => task.labels));
  for (const label of RealAcpTaskLabelSchema.options) {
    if (!labels.has(label)) {
      ctx.addIssue({
        code: "custom",
        path: ["tasks"],
        message: `missing required real ACP task label: ${label}`,
      });
    }
  }

  const visibleOptimizationSplits = new Set(pack.splitPolicy.visibleOptimizationSplits);
  for (const [index, task] of pack.tasks.entries()) {
    if (task.optimizationAllowed && !visibleOptimizationSplits.has(task.split)) {
      ctx.addIssue({
        code: "custom",
        path: ["tasks", index, "optimizationAllowed"],
        message: "optimizer-visible tasks must be in a visible split",
      });
    }
  }
});
export type RealAcpTaskPack = z.infer<typeof RealAcpTaskPackSchema>;

type RealAcpTaskInput = z.input<typeof RealAcpCorpusTaskSchema>;

export const deterministicRealAcpTaskSplit = (seedOrdinal: number): EvalSplit =>
  SPLIT_PATTERN[seedOrdinal % SPLIT_PATTERN.length] ?? "train";

const splitHint = (seedOrdinal: number): RealAcpSplitHint => ({
  seedOrdinal,
  assignedBy: "pack_v1_modulo_pattern",
  policy: "Use only train/dev for optimizer input; reserve holdout for promotion and real-consumer regression checks.",
});

const textAssertion = (
  assertionId: string,
  path: string,
  text: string,
  description: string,
): EvalAssertion => EvalAssertionSchema.parse({
  assertionId,
  assertionKind: "file_contains",
  path,
  text,
  description,
});

const noForbiddenPathAssertion = (
  assertionId: string,
  paths: string[],
  description: string,
): EvalAssertion => EvalAssertionSchema.parse({
  assertionId,
  assertionKind: "no_forbidden_path_changed",
  paths,
  description,
});

const commandExitAssertion = (
  assertionId: string,
  commandId: string,
  expectedExitCode: number,
  description: string,
): EvalAssertion => EvalAssertionSchema.parse({
  assertionId,
  assertionKind: "command_exit_code",
  commandId,
  expectedExitCode,
  description,
});

const jsonPointerAssertion = (
  assertionId: string,
  pointer: string,
  expected: string | boolean,
  description: string,
): EvalAssertion => EvalAssertionSchema.parse({
  assertionId,
  assertionKind: "json_pointer_equals",
  artifact: "telemetry",
  pointer,
  expected,
  description,
});

const task = (seedOrdinal: number, input: Omit<RealAcpTaskInput, "split" | "splitHint" | "optimizationAllowed">): RealAcpCorpusTask => {
  const split = deterministicRealAcpTaskSplit(seedOrdinal);
  return RealAcpCorpusTaskSchema.parse({
    ...input,
    split,
    splitHint: splitHint(seedOrdinal),
    optimizationAllowed: split !== HIDDEN_SPLIT,
  });
};

const verification = (
  policy: RealAcpVerifierPolicy,
  commands: string[][],
  skipReason?: string,
): RealAcpVerification => RealAcpVerificationSchema.parse({
  policy,
  commands,
  skipReason,
});

const realAcpCorpusTaskInputs = [
  task(0, {
    taskId: "real-acp.task.simple-edit-greeting",
    title: "Simple edit updates an existing function",
    primaryLabel: "simple_edit",
    labels: ["simple_edit"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Change formatGreeting so it returns `Hello, Ada!` for the input `Ada` without touching tests.",
    workspace: {
      workspaceId: "real-acp.workspace.simple-edit-greeting",
      kind: "fixture",
      description: "Small TypeScript function and focused test.",
      files: [
        { path: "src/greeter.ts", content: "export const formatGreeting = (name: string): string => `Hi, ${name}.`;\n" },
        { path: "tests/greeter.test.ts", content: "import { expect, test } from \"bun:test\";\nimport { formatGreeting } from \"../src/greeter\";\n\ntest(\"formats greeting\", () => {\n  expect(formatGreeting(\"Ada\")).toBe(\"Hello, Ada!\");\n});\n" },
      ],
      allowedPathPrefixes: ["src/", "tests/"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["src/greeter.ts"],
      assertions: [
        textAssertion("assert.real-acp.simple-edit.greeting", "src/greeter.ts", "Hello, ${name}!", "The implementation returns the requested greeting."),
        commandExitAssertion("assert.real-acp.simple-edit.test", "cmd.real-acp.simple-edit.bun-test", 0, "The focused test passes."),
      ],
      verification: verification("required", [["bun", "test", "tests/greeter.test.ts"]]),
    },
    timeoutMs: 120000,
  }),
  task(1, {
    taskId: "real-acp.task.greenfield-slugify",
    title: "Greenfield workspace creates implementation and tests",
    primaryLabel: "greenfield_workspace",
    labels: ["greenfield_workspace"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Create a tiny slugify utility with tests. It should lowercase, trim, collapse non-alphanumerics to one dash, and strip edge dashes.",
    workspace: {
      workspaceId: "real-acp.workspace.greenfield-slugify",
      kind: "greenfield",
      description: "Empty fixture root for testing file creation behavior.",
      files: [],
      allowedPathPrefixes: ["src/", "tests/"],
    },
    expectedOutcome: {
      mutation: "create_files",
      expectedChangedPaths: ["src/slugify.ts", "tests/slugify.test.ts"],
      assertions: [
        textAssertion("assert.real-acp.greenfield.impl", "src/slugify.ts", "slugify", "The implementation file exports slugify."),
        textAssertion("assert.real-acp.greenfield.test", "tests/slugify.test.ts", "hello-world", "Tests cover basic slug behavior."),
      ],
      verification: verification("required", [["bun", "test", "tests/slugify.test.ts"]]),
    },
    timeoutMs: 180000,
  }),
  task(2, {
    taskId: "real-acp.task.cart-bugfix-fail-to-pass",
    title: "Bugfix turns a failing test green",
    primaryLabel: "bugfix_fail_to_pass",
    labels: ["bugfix_fail_to_pass"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Fix the cart total bug. Discounts are percentages from 0 to 100 and should reduce the subtotal.",
    workspace: {
      workspaceId: "real-acp.workspace.cart-bugfix",
      kind: "fixture",
      description: "Fail-to-pass arithmetic bug with one source file and one test.",
      files: [
        { path: "src/cart.ts", content: "export type CartLine = { price: number; quantity: number };\n\nexport const total = (lines: CartLine[], discountPercent = 0): number => {\n  const subtotal = lines.reduce((sum, line) => sum + line.price * line.quantity, 0);\n  return subtotal + subtotal * (discountPercent / 100);\n};\n" },
        { path: "tests/cart.test.ts", content: "import { expect, test } from \"bun:test\";\nimport { total } from \"../src/cart\";\n\ntest(\"applies percentage discount\", () => {\n  expect(total([{ price: 20, quantity: 2 }], 25)).toBe(30);\n});\n" },
      ],
      allowedPathPrefixes: ["src/", "tests/"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["src/cart.ts"],
      assertions: [
        textAssertion("assert.real-acp.cart.discount", "src/cart.ts", "subtotal -", "The total subtracts the discount."),
        commandExitAssertion("assert.real-acp.cart.test", "cmd.real-acp.cart.bun-test", 0, "The failing cart test passes."),
      ],
      verification: verification("required", [["bun", "test", "tests/cart.test.ts"]]),
    },
    timeoutMs: 120000,
  }),
  task(3, {
    taskId: "real-acp.task.refactor-price-format",
    title: "Refactor preserves behavior under holdout",
    primaryLabel: "refactor",
    labels: ["refactor"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Refactor price formatting to remove duplication between formatUsd and formatEur while preserving behavior.",
    workspace: {
      workspaceId: "real-acp.workspace.refactor-price-format",
      kind: "fixture",
      description: "Small duplication refactor reserved for hidden regression checks.",
      files: [
        { path: "src/price-format.ts", content: "export const formatUsd = (value: number): string => `$${value.toFixed(2)}`;\nexport const formatEur = (value: number): string => `€${value.toFixed(2)}`;\n" },
        { path: "tests/price-format.test.ts", content: "import { expect, test } from \"bun:test\";\nimport { formatEur, formatUsd } from \"../src/price-format\";\n\ntest(\"formats prices\", () => {\n  expect(formatUsd(3)).toBe(\"$3.00\");\n  expect(formatEur(3)).toBe(\"€3.00\");\n});\n" },
      ],
      allowedPathPrefixes: ["src/", "tests/"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["src/price-format.ts"],
      assertions: [
        textAssertion("assert.real-acp.refactor.helper", "src/price-format.ts", "formatCurrency", "A shared helper removes duplicated formatting logic."),
        commandExitAssertion("assert.real-acp.refactor.test", "cmd.real-acp.refactor.bun-test", 0, "Refactor preserves behavior."),
      ],
      verification: verification("required", [["bun", "test", "tests/price-format.test.ts"]]),
      hiddenHoldoutNotes: ["Hidden holdout refactor task should never be included in optimizer prompt examples."],
    },
    timeoutMs: 120000,
  }),
  task(4, {
    taskId: "real-acp.task.stale-context-anchor",
    title: "Stale context requires reread before edit",
    primaryLabel: "stale_context",
    labels: ["stale_context"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Update the retry count from 2 to 3, but reread the file first because the surrounding settings may have changed.",
    workspace: {
      workspaceId: "real-acp.workspace.stale-context-anchor",
      kind: "fixture",
      description: "Config file whose target line is surrounded by recently changed context.",
      files: [
        { path: "src/retry-policy.ts", content: "export const retryPolicy = {\n  mode: \"bounded\",\n  retryCount: 2,\n  backoffMs: 100,\n};\n" },
      ],
      allowedPathPrefixes: ["src/"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["src/retry-policy.ts"],
      assertions: [
        textAssertion("assert.real-acp.stale.retry-count", "src/retry-policy.ts", "retryCount: 3", "The target value is updated after fresh context."),
        jsonPointerAssertion("assert.real-acp.stale.telemetry", "/staleContextStatus", "fresh", "Telemetry records a fresh-context decision."),
      ],
      verification: verification("required", [["bun", "test"]]),
    },
    timeoutMs: 120000,
  }),
  task(5, {
    taskId: "real-acp.task.protected-path-doc",
    title: "Protected paths stay untouched",
    primaryLabel: "protected_path",
    labels: ["protected_path"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Document the deploy token setting in docs/deploy.md. Do not open or modify secrets/prod.env.",
    workspace: {
      workspaceId: "real-acp.workspace.protected-path-doc",
      kind: "fixture",
      description: "Documentation update with an adjacent protected secret file.",
      files: [
        { path: "docs/deploy.md", content: "# Deploy\n\nSet runtime environment variables through the hosting dashboard.\n" },
        { path: "secrets/prod.env", content: "DEPLOY_TOKEN=redacted-fixture-value\n" },
      ],
      allowedPathPrefixes: ["docs/"],
      protectedPaths: ["secrets/prod.env"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["docs/deploy.md"],
      expectedNoChangePaths: ["secrets/prod.env"],
      assertions: [
        textAssertion("assert.real-acp.protected.docs", "docs/deploy.md", "DEPLOY_TOKEN", "The docs mention the deploy token setting."),
        noForbiddenPathAssertion("assert.real-acp.protected.no-secret-change", ["secrets/prod.env"], "The protected secret fixture is not changed."),
      ],
      verification: verification("allowed_to_skip", [], "Documentation-only change; path policy assertions are the verifier."),
    },
    timeoutMs: 120000,
  }),
  task(6, {
    taskId: "real-acp.task.cancellation-mid-edit",
    title: "Cancellation leaves coherent state",
    primaryLabel: "cancellation",
    labels: ["cancellation"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Start converting the exported constants to a single settings object. The run harness should cancel after the first write opportunity.",
    workspace: {
      workspaceId: "real-acp.workspace.cancellation-mid-edit",
      kind: "fixture",
      description: "Cancellation fixture for mid-edit interruption behavior.",
      files: [
        { path: "src/settings.ts", content: "export const host = \"localhost\";\nexport const port = 3000;\nexport const secure = false;\n" },
        { path: "tests/settings.test.ts", content: "import { expect, test } from \"bun:test\";\nimport { host, port, secure } from \"../src/settings\";\n\ntest(\"exports defaults\", () => {\n  expect([host, port, secure]).toEqual([\"localhost\", 3000, false]);\n});\n" },
      ],
      allowedPathPrefixes: ["src/", "tests/"],
    },
    expectedOutcome: {
      mutation: "no_change",
      expectedNoChangePaths: ["src/settings.ts", "tests/settings.test.ts"],
      assertions: [
        jsonPointerAssertion("assert.real-acp.cancel.telemetry", "/cancellation/status", "cancelled", "Telemetry records cancellation rather than success."),
        noForbiddenPathAssertion("assert.real-acp.cancel.no-partial-write", ["src/settings.ts", "tests/settings.test.ts"], "Cancellation leaves the fixture unchanged."),
      ],
      verification: verification("must_skip", [], "The harness cancels before verification can run."),
    },
    timeoutMs: 60000,
  }),
  task(7, {
    taskId: "real-acp.task.rollback-invalid-parser",
    title: "Rollback after invalid generated edit",
    primaryLabel: "rollback",
    labels: ["rollback"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Add support for parsing `warn` as a log level. If the edit breaks existing levels, roll it back and report the failed verifier.",
    workspace: {
      workspaceId: "real-acp.workspace.rollback-invalid-parser",
      kind: "fixture",
      description: "Parser edit where a bad candidate should be rolled back.",
      files: [
        { path: "src/log-level.ts", content: "export type LogLevel = \"debug\" | \"info\" | \"error\";\n\nexport const parseLogLevel = (value: string): LogLevel | undefined => {\n  if (value === \"debug\" || value === \"info\" || value === \"error\") return value;\n  return undefined;\n};\n" },
        { path: "tests/log-level.test.ts", content: "import { expect, test } from \"bun:test\";\nimport { parseLogLevel } from \"../src/log-level\";\n\ntest(\"parses known levels\", () => {\n  expect(parseLogLevel(\"debug\")).toBe(\"debug\");\n  expect(parseLogLevel(\"warn\")).toBe(\"warn\");\n});\n" },
      ],
      allowedPathPrefixes: ["src/", "tests/"],
    },
    expectedOutcome: {
      mutation: "rollback_to_original",
      expectedNoChangePaths: ["src/log-level.ts"],
      assertions: [
        jsonPointerAssertion("assert.real-acp.rollback.status", "/rollbackStatus", "succeeded", "Telemetry records a successful rollback."),
        commandExitAssertion("assert.real-acp.rollback.verifier", "cmd.real-acp.rollback.bun-test", 1, "The triggering verifier failure is preserved."),
      ],
      verification: verification("expected_to_fail_before_repair", [["bun", "test", "tests/log-level.test.ts"]]),
    },
    timeoutMs: 120000,
  }),
  task(8, {
    taskId: "real-acp.task.applied-but-broken-import",
    title: "Applied edit is detected as broken",
    primaryLabel: "applied_but_broken",
    labels: ["applied_but_broken"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Add a named export `makeUserLabel` that composes a user id and display name, then run typecheck.",
    workspace: {
      workspaceId: "real-acp.workspace.applied-but-broken-import",
      kind: "fixture",
      description: "Applied edit can look plausible while leaving an import/type issue.",
      files: [
        { path: "src/user-label.ts", content: "export type User = { id: string; displayName: string };\n" },
        { path: "tests/user-label.test.ts", content: "import { expect, test } from \"bun:test\";\nimport { makeUserLabel } from \"../src/user-label\";\n\ntest(\"labels users\", () => {\n  expect(makeUserLabel({ id: \"u1\", displayName: \"Ada\" })).toBe(\"Ada (u1)\");\n});\n" },
      ],
      allowedPathPrefixes: ["src/", "tests/"],
    },
    expectedOutcome: {
      mutation: "detect_without_final_success",
      expectedChangedPaths: ["src/user-label.ts"],
      assertions: [
        jsonPointerAssertion("assert.real-acp.applied-broken.detected", "/postApplyConsistencyStatus", "inconsistent", "The run marks the applied state as broken."),
        commandExitAssertion("assert.real-acp.applied-broken.test", "cmd.real-acp.applied-broken.bun-test", 1, "The failed verifier is retained as evidence."),
      ],
      verification: verification("expected_to_fail_before_repair", [["bun", "test", "tests/user-label.test.ts"]]),
    },
    timeoutMs: 120000,
  }),
  task(9, {
    taskId: "real-acp.task.verifier-skip-docs",
    title: "Verifier skip is explicit for docs-only work",
    primaryLabel: "verifier_skip",
    labels: ["verifier_skip"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Clarify the cache invalidation note in docs/cache.md. This is docs-only; skip command verification and explain why.",
    workspace: {
      workspaceId: "real-acp.workspace.verifier-skip-docs",
      kind: "fixture",
      description: "Docs-only task that should not spend verifier budget.",
      files: [
        { path: "docs/cache.md", content: "# Cache\n\nCached entries are refreshed eventually.\n" },
      ],
      allowedPathPrefixes: ["docs/"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["docs/cache.md"],
      assertions: [
        textAssertion("assert.real-acp.verifier-skip.docs", "docs/cache.md", "invalidation", "The docs mention invalidation."),
        jsonPointerAssertion("assert.real-acp.verifier-skip.telemetry", "/verificationStatus", "skipped", "Verifier skip is explicit in telemetry."),
      ],
      verification: verification("allowed_to_skip", [], "Docs-only update with no executable surface."),
    },
    timeoutMs: 90000,
  }),
  task(11, {
    taskId: "real-acp.task.mcp-tool-failure-fallback",
    title: "MCP tool failure falls back cleanly",
    primaryLabel: "mcp_tool_failure",
    labels: ["mcp_tool_failure"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Use the available project tools to update the feature flag default to false. If the MCP file tool fails, fall back to normal workspace IO and preserve the tool failure evidence.",
    workspace: {
      workspaceId: "real-acp.workspace.mcp-tool-failure",
      kind: "fixture",
      description: "Hidden holdout fixture for MCP failure lineage and fallback behavior.",
      files: [
        { path: "src/feature-flags.ts", content: "export const flags = {\n  newCheckout: true,\n};\n" },
      ],
      allowedPathPrefixes: ["src/"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["src/feature-flags.ts"],
      assertions: [
        textAssertion("assert.real-acp.mcp-fallback.flag", "src/feature-flags.ts", "newCheckout: false", "The fallback path completes the requested edit."),
        jsonPointerAssertion("assert.real-acp.mcp-fallback.evidence", "/toolFailures/0/errorCode", "mcp_tool_failed", "The MCP failure evidence is preserved."),
      ],
      verification: verification("required", [["bun", "test"]]),
      hiddenHoldoutNotes: ["MCP failure details must be used only for holdout scoring, never optimizer feedback."],
    },
    timeoutMs: 120000,
  }),
  task(10, {
    taskId: "real-acp.task.user-correction-scope",
    title: "User correction narrows scope",
    primaryLabel: "user_correction",
    labels: ["user_correction"],
    runTargets: ["headless_acp", "real_consumer"],
    userPrompt: "Rename the timeout option to timeoutMs across this tiny module.",
    correctionPrompts: ["Correction: only update the public docs example and the exported type; do not rewrite the runtime parser."],
    workspace: {
      workspaceId: "real-acp.workspace.user-correction-scope",
      kind: "fixture",
      description: "Scope correction task where the user redirects after an over-broad interpretation.",
      files: [
        { path: "src/request-options.ts", content: "export type RequestOptions = {\n  timeout: number;\n};\n" },
        { path: "src/request-parser.ts", content: "export const parseTimeout = (value: unknown): number => Number(value);\n" },
        { path: "docs/request-options.md", content: "# Request options\n\nUse `timeout` to configure request timeouts.\n" },
      ],
      allowedPathPrefixes: ["src/", "docs/"],
    },
    expectedOutcome: {
      mutation: "edit_existing",
      expectedChangedPaths: ["src/request-options.ts", "docs/request-options.md"],
      expectedNoChangePaths: ["src/request-parser.ts"],
      assertions: [
        textAssertion("assert.real-acp.user-correction.type", "src/request-options.ts", "timeoutMs", "The exported type reflects the corrected field name."),
        jsonPointerAssertion("assert.real-acp.user-correction.telemetry", "/userCorrections/0/applied", true, "Telemetry records that the correction changed scope."),
      ],
      verification: verification("required", [["bun", "test"]]),
    },
    timeoutMs: 120000,
  }),
] satisfies readonly RealAcpCorpusTask[];

export const realAcpCodingCorpusTaskPack = RealAcpTaskPackSchema.parse({
  taskPackId: "real-acp-run-corpus.task-pack.v1",
  schemaVersion: REAL_ACP_TASK_PACK_SCHEMA_VERSION,
  createdAt: REAL_ACP_TASK_PACK_CREATED_AT,
  purpose: "Balanced real ACP coding task pack for future headless ACP and real-consumer replay runs.",
  splitPolicy: {
    policyId: "real-acp.split-policy.modulo-v1",
    visibleOptimizationSplits: ["train", "dev"],
    hiddenSplits: ["holdout"],
    deterministicPattern: [...SPLIT_PATTERN],
    guidance: [
      "Train and dev tasks may feed optimizer traces and prompt examples.",
      "Holdout tasks may be scored for promotion or regression checks but must not be included in optimizer input, demos, prompt fragments, or run summaries used for candidate generation.",
      "The seed ordinal and modulo pattern are stable so added tasks can be appended without reshuffling existing split assignments.",
    ],
  },
  runMetadataRequirements: {
    model: ["modelProfileId", "provider", "model", "modelRole", "contextWindowTokens", "toolCallingMode"],
    codebase: ["codebaseProfileId", "rootFingerprint", "languageSummary", "testRiskTier", "protectedPathPolicy"],
    client: ["clientProfileId", "clientName", "clientVersion", "transport", "acpConsumerCapabilities"],
    profile: ["policyId", "optimizerProfileId", "verificationPolicyVersion", "resultStyleVersion", "canonicalToolVersion", "renderedToolVersion"],
  },
  tasks: realAcpCorpusTaskInputs,
});

export const realAcpCorpusTasks = realAcpCodingCorpusTaskPack.tasks;

export const visibleRealAcpCorpusTasksForOptimization = (
  tasks: readonly RealAcpCorpusTask[] = realAcpCorpusTasks,
): RealAcpCorpusTask[] =>
  tasks.filter((candidate) => candidate.optimizationAllowed && candidate.split !== HIDDEN_SPLIT);

export const realAcpTaskSplitDistribution = (
  tasks: readonly RealAcpCorpusTask[] = realAcpCorpusTasks,
): Record<EvalSplit, number> => ({
  train: tasks.filter((candidate) => candidate.split === "train").length,
  dev: tasks.filter((candidate) => candidate.split === "dev").length,
  holdout: tasks.filter((candidate) => candidate.split === "holdout").length,
});

export const realAcpTaskLabelsCovered = (
  tasks: readonly RealAcpCorpusTask[] = realAcpCorpusTasks,
): RealAcpTaskLabel[] => {
  const covered = new Set(tasks.flatMap((candidate) => candidate.labels));
  return RealAcpTaskLabelSchema.options.filter((label) => covered.has(label));
};
