import { createHash } from "node:crypto";
import { z } from "zod";
import {
  EditErrorCodeSchema,
  EditPhaseStatusSchema,
  EditStrategyFamilySchema,
  PostApplyConsistencyStatusSchema,
  SelfDetectedRegressionStatusSchema,
  StaleContextStatusSchema,
  VerificationStatusSchema,
  type EditStrategyFamily,
} from "../edit-strategy/types";
import { OptimizerIdSchema } from "../optimizer/types";
import {
  EvalAssertionSchema,
  EvalSplitSchema,
  FixtureWorkspaceSchema,
  type EvalAssertion,
  type FixtureWorkspace,
} from "./types";

const EDIT_STRATEGY_EVAL_SCHEMA_VERSION = "edit-strategy-eval.v1";
const RelativePathSchema = z.string().min(1).regex(/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$)).+$/);

export const EditStrategyBaselineWholeFileEditSchema = z.object({
  path: RelativePathSchema,
  content: z.string(),
}).strict();
export type EditStrategyBaselineWholeFileEdit = z.infer<typeof EditStrategyBaselineWholeFileEditSchema>;

export const EditStrategyEvalFailureModeSchema = z.enum([
  "small_edit",
  "large_file_localized_edit",
  "multi_file_edit",
  "repeated_snippet",
  "stale_read",
  "malformed_patch",
  "applied_but_broken",
  "self_discovered_inconsistency",
  "no_op_request",
  "formatting_sensitive",
  "protected_path",
  "strategy_specific_failure",
]);
export type EditStrategyEvalFailureMode = z.infer<typeof EditStrategyEvalFailureModeSchema>;

export const EditStrategyProbeSchema = z.object({
  probeId: OptimizerIdSchema,
  strategyFamily: EditStrategyFamilySchema,
  description: z.string().min(1),
  modelOutput: z.string(),
  expectedParseStatus: EditPhaseStatusSchema.default("passed"),
  expectedApplyStatus: EditPhaseStatusSchema.default("passed"),
  expectedErrorCode: EditErrorCodeSchema.optional(),
  expectedStaleContextStatus: StaleContextStatusSchema.optional(),
  expectedVerificationStatus: VerificationStatusSchema.optional(),
  expectedPostApplyConsistencyStatus: PostApplyConsistencyStatusSchema.optional(),
  expectedSelfDetectedRegressionStatus: SelfDetectedRegressionStatusSchema.optional(),
  expectedProtectedPathTouched: z.boolean().optional(),
}).strict().superRefine((probe, ctx) => {
  if (
    (probe.expectedParseStatus === "failed" || probe.expectedApplyStatus === "failed") &&
    probe.expectedErrorCode === undefined
  ) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["expectedErrorCode"],
      message: "failed parse/apply probes must declare the expected stable error code",
    });
  }
});
export type EditStrategyProbe = z.infer<typeof EditStrategyProbeSchema>;

export const EditStrategyEvalCaseSchema = z.object({
  editEvalCaseId: OptimizerIdSchema,
  schemaVersion: z.literal(EDIT_STRATEGY_EVAL_SCHEMA_VERSION).default(EDIT_STRATEGY_EVAL_SCHEMA_VERSION),
  split: EvalSplitSchema,
  title: z.string().min(1),
  task: z.string().min(1),
  fixtureWorkspace: FixtureWorkspaceSchema,
  coveredFailureModes: z.array(EditStrategyEvalFailureModeSchema).min(1),
  targetFiles: z.array(RelativePathSchema).default([]),
  expectedChangedFiles: z.array(RelativePathSchema).default([]),
  forbiddenChangedFiles: z.array(RelativePathSchema).default([]),
  baselineWholeFileEdits: z.array(EditStrategyBaselineWholeFileEditSchema).default([]),
  assertions: z.array(EvalAssertionSchema).default([]),
  probes: z.array(EditStrategyProbeSchema).min(1),
  notes: z.array(z.string().min(1)).default([]),
}).strict().superRefine((evalCase, ctx) => {
  const expected = new Set(evalCase.expectedChangedFiles);
  for (const forbiddenPath of evalCase.forbiddenChangedFiles) {
    if (expected.has(forbiddenPath)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["forbiddenChangedFiles"],
        message: `path cannot be both expected and forbidden: ${forbiddenPath}`,
      });
    }
  }
});
export type EditStrategyEvalCase = z.infer<typeof EditStrategyEvalCaseSchema>;

export const editStrategyEvalCaseIds = [
  "edit-eval.small-targeted-edit",
  "edit-eval.large-localized-edit",
  "edit-eval.multi-file-consistency",
  "edit-eval.repeated-snippet-ambiguity",
  "edit-eval.stale-read-detection",
  "edit-eval.malformed-unified-diff",
  "edit-eval.applied-but-broken",
  "edit-eval.self-discovered-inconsistency",
  "edit-eval.no-op-request",
  "edit-eval.formatting-sensitive-json",
  "edit-eval.protected-path-veto",
  "edit-eval.fenced-diff-path-failure",
] as const;

const noForbiddenChange = (assertionId: string, paths: string[]): EvalAssertion => ({
  assertionId,
  assertionKind: "no_forbidden_path_changed",
  description: "Forbidden files remain unchanged.",
  severity: "critical",
  paths,
});

const fileContains = (assertionId: string, path: string, text: string): EvalAssertion => ({
  assertionId,
  assertionKind: "file_contains",
  description: "Expected content is present.",
  severity: "failure",
  path,
  text,
});

const contentHash = (content: string): string => `sha256:${createHash("sha256").update(content).digest("hex")}`;

const largeSettingsContent = [
  "# Settings",
  ...Array.from({ length: 140 }, (_, index) => {
    const line = String(index + 1).padStart(3, "0");
    return line === "097" ? "setting.097 = disabled" : `setting.${line} = unchanged`;
  }),
  "",
].join("\n");

const rawEditStrategyEvalCases = [
  {
    editEvalCaseId: "edit-eval.small-targeted-edit",
    split: "train",
    title: "Small targeted edit",
    task: "Change the banner from PLACEHOLDER to BleedingAgent without touching package.json.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.small-targeted-edit",
      name: "Small edit fixture",
      rootFingerprint: "sha256:edit-eval-small-targeted-edit-v1",
      files: [
        { path: "src/banner.txt", content: "Hello PLACEHOLDER.\n" },
        { path: "package.json", content: "{\n  \"name\": \"small-edit\"\n}\n" },
      ],
      protectedPaths: ["package.json"],
    },
    coveredFailureModes: ["small_edit", "protected_path"],
    targetFiles: ["src/banner.txt"],
    expectedChangedFiles: ["src/banner.txt"],
    forbiddenChangedFiles: ["package.json"],
    baselineWholeFileEdits: [
      {
        path: "src/banner.txt",
        content: "Hello BleedingAgent.\n",
      },
    ],
    assertions: [
      fileContains("assert.edit-small-banner", "src/banner.txt", "BleedingAgent"),
      noForbiddenChange("assert.edit-small-package-protected", ["package.json"]),
    ],
    probes: [
      {
        probeId: "probe.small.apply-patch",
        strategyFamily: "apply_patch",
        description: "Representative patch output for a one-line localized edit.",
        modelOutput: "*** Begin Patch\n*** Update File: src/banner.txt\n@@\n-Hello PLACEHOLDER.\n+Hello BleedingAgent.\n*** End Patch\n",
      },
      {
        probeId: "probe.small.exact-replace",
        strategyFamily: "exact_replace",
        description: "Representative exact replacement output.",
        modelOutput: "SEARCH: Hello PLACEHOLDER.\\n\nREPLACE: Hello BleedingAgent.\\n\n",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.large-localized-edit",
    split: "dev",
    title: "Large file localized edit",
    task: "In config/settings.txt change only setting.097 from disabled to enabled.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.large-localized-edit",
      name: "Large localized fixture",
      rootFingerprint: "sha256:edit-eval-large-localized-edit-v1",
      files: [
        { path: "config/settings.txt", content: largeSettingsContent },
        { path: "README.md", content: "# Fixture\n\nDo not rewrite the settings file wholesale unless necessary.\n" },
      ],
      protectedPaths: ["README.md"],
    },
    coveredFailureModes: ["large_file_localized_edit"],
    targetFiles: ["config/settings.txt"],
    expectedChangedFiles: ["config/settings.txt"],
    forbiddenChangedFiles: ["README.md"],
    baselineWholeFileEdits: [
      {
        path: "config/settings.txt",
        content: largeSettingsContent.replace("setting.097 = disabled", "setting.097 = enabled"),
      },
    ],
    assertions: [
      fileContains("assert.large-localized-target", "config/settings.txt", "setting.097 = enabled"),
      noForbiddenChange("assert.large-localized-readme-protected", ["README.md"]),
    ],
    probes: [
      {
        probeId: "probe.large.hash-range",
        strategyFamily: "hash_range",
        description: "Range/hash anchored replacement should avoid rewriting unrelated lines.",
        modelOutput:
          JSON.stringify({
            path: "config/settings.txt",
            startLine: 98,
            endLine: 98,
            expectedHash: contentHash(largeSettingsContent),
            replacement: "setting.097 = enabled\n",
          }),
      },
      {
        probeId: "probe.large.unified-diff",
        strategyFamily: "unified_diff",
        description: "Unified diff should apply with enough local context around the target line.",
        modelOutput:
          "--- a/config/settings.txt\n+++ b/config/settings.txt\n@@\n-setting.097 = disabled\n+setting.097 = enabled\n",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.multi-file-consistency",
    split: "train",
    title: "Multi-file consistency edit",
    task: "Rename the exported status value from draft to active and update the related fixture expectation.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.multi-file-consistency",
      name: "Multi-file consistency fixture",
      rootFingerprint: "sha256:edit-eval-multi-file-consistency-v1",
      files: [
        { path: "src/status.ts", content: "export const status = \"draft\";\n" },
        { path: "tests/status.test.ts", content: "expect(status).toBe(\"draft\");\n" },
      ],
      protectedPaths: [],
    },
    coveredFailureModes: ["multi_file_edit"],
    targetFiles: ["src/status.ts", "tests/status.test.ts"],
    expectedChangedFiles: ["src/status.ts", "tests/status.test.ts"],
    baselineWholeFileEdits: [
      {
        path: "src/status.ts",
        content: "export const status = \"active\";\n",
      },
      {
        path: "tests/status.test.ts",
        content: "expect(status).toBe(\"active\");\n",
      },
    ],
    assertions: [
      fileContains("assert.multi-source", "src/status.ts", "\"active\""),
      fileContains("assert.multi-test", "tests/status.test.ts", "\"active\""),
    ],
    probes: [
      {
        probeId: "probe.multi.multi-exact",
        strategyFamily: "multi_exact_replace",
        description: "Multiple exact replacements must apply as one coherent edit attempt.",
        modelOutput:
          "FILE: src/status.ts\nSEARCH: \"draft\"\nREPLACE: \"active\"\n\nFILE: tests/status.test.ts\nSEARCH: \"draft\"\nREPLACE: \"active\"\n",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.repeated-snippet-ambiguity",
    split: "dev",
    title: "Repeated snippet ambiguity",
    task: "Change only the payment handler response to PAID and leave the refund handler unchanged.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.repeated-snippet-ambiguity",
      name: "Repeated snippet fixture",
      rootFingerprint: "sha256:edit-eval-repeated-snippet-ambiguity-v1",
      files: [
        {
          path: "src/handlers.ts",
          content:
            "export function payment() {\n  return \"TODO\";\n}\n\nexport function refund() {\n  return \"TODO\";\n}\n",
        },
      ],
      protectedPaths: [],
    },
    coveredFailureModes: ["repeated_snippet", "strategy_specific_failure"],
    targetFiles: ["src/handlers.ts"],
    expectedChangedFiles: ["src/handlers.ts"],
    baselineWholeFileEdits: [
      {
        path: "src/handlers.ts",
        content:
          "export function payment() {\n  return \"PAID\";\n}\n\nexport function refund() {\n  return \"TODO\";\n}\n",
      },
    ],
    assertions: [fileContains("assert.repeated-payment", "src/handlers.ts", "return \"PAID\";")],
    probes: [
      {
        probeId: "probe.repeated.exact-ambiguous",
        strategyFamily: "exact_replace",
        description: "Underspecified exact replacement matches two identical snippets.",
        modelOutput: "SEARCH: return \"TODO\";\nREPLACE: return \"PAID\";\n",
        expectedApplyStatus: "failed",
        expectedErrorCode: "exact_match_ambiguous",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.stale-read-detection",
    split: "holdout",
    title: "Stale read detection",
    task: "Apply an edit only if the captured file hash still matches the read snapshot.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.stale-read-detection",
      name: "Stale read fixture",
      rootFingerprint: "sha256:edit-eval-stale-read-detection-v1",
      files: [{ path: "src/version.txt", content: "version=2\n" }],
      protectedPaths: [],
    },
    coveredFailureModes: ["stale_read", "strategy_specific_failure"],
    targetFiles: ["src/version.txt"],
    expectedChangedFiles: [],
    baselineWholeFileEdits: [],
    assertions: [fileContains("assert.stale-current-version", "src/version.txt", "version=2")],
    probes: [
      {
        probeId: "probe.stale.hash-mismatch",
        strategyFamily: "hash_range",
        description: "Hash/range edit built from version=1 must reject the current version=2 file.",
        modelOutput:
          JSON.stringify({
            path: "src/version.txt",
            expectedHash: "sha256:version-1",
            replacement: "version=3\n",
          }),
        expectedApplyStatus: "failed",
        expectedErrorCode: "hash_mismatch",
        expectedStaleContextStatus: "stale",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.malformed-unified-diff",
    split: "train",
    title: "Malformed unified diff",
    task: "Reject malformed diff output before any workspace write.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.malformed-unified-diff",
      name: "Malformed diff fixture",
      rootFingerprint: "sha256:edit-eval-malformed-unified-diff-v1",
      files: [{ path: "src/value.txt", content: "value=old\n" }],
      protectedPaths: ["src/value.txt"],
    },
    coveredFailureModes: ["malformed_patch"],
    targetFiles: ["src/value.txt"],
    expectedChangedFiles: [],
    forbiddenChangedFiles: ["src/value.txt"],
    baselineWholeFileEdits: [],
    assertions: [noForbiddenChange("assert.malformed-no-write", ["src/value.txt"])],
    probes: [
      {
        probeId: "probe.malformed.unified-diff",
        strategyFamily: "unified_diff",
        description: "Malformed unified diff must fail at parse time.",
        modelOutput: "--- src/value.txt\n@@ missing-plus-minus-context\nvalue=new\n",
        expectedParseStatus: "failed",
        expectedApplyStatus: "not_started",
        expectedErrorCode: "parse_error",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.applied-but-broken",
    split: "dev",
    title: "Applied but broken edit",
    task: "Fix multiply without leaving syntax or verification failures.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.applied-but-broken",
      name: "Applied but broken fixture",
      rootFingerprint: "sha256:edit-eval-applied-but-broken-v1",
      files: [
        { path: "src/math.mjs", content: "export function multiply(a, b) {\n  return a + b;\n}\n" },
        {
          path: "tests/math.test.mjs",
          content:
            "import assert from 'node:assert/strict';\nimport { multiply } from '../src/math.mjs';\nassert.equal(multiply(3, 4), 12);\n",
        },
      ],
      protectedPaths: ["tests/math.test.mjs"],
      verificationCommands: [["node", "tests/math.test.mjs"]],
    },
    coveredFailureModes: ["applied_but_broken"],
    targetFiles: ["src/math.mjs"],
    expectedChangedFiles: ["src/math.mjs"],
    forbiddenChangedFiles: ["tests/math.test.mjs"],
    baselineWholeFileEdits: [
      {
        path: "src/math.mjs",
        content: "export function multiply(a, b) {\n  return a * b;\n}\n",
      },
    ],
    assertions: [
      fileContains("assert.applied-broken-target", "src/math.mjs", "return a * b"),
      noForbiddenChange("assert.applied-broken-test-protected", ["tests/math.test.mjs"]),
    ],
    probes: [
      {
        probeId: "probe.applied-broken.apply-patch",
        strategyFamily: "apply_patch",
        description: "Patch applies cleanly but leaves invalid syntax and must be penalized after write.",
        modelOutput:
          "*** Begin Patch\n*** Update File: src/math.mjs\n@@\n-  return a + b;\n+  return a * ;\n*** End Patch\n",
        expectedPostApplyConsistencyStatus: "inconsistent",
        expectedVerificationStatus: "failed",
        expectedErrorCode: "post_apply_syntax_failure",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.self-discovered-inconsistency",
    split: "dev",
    title: "Self-discovered inconsistency",
    task: "Update both config and README so the documented feature flag matches the config value.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.self-discovered-inconsistency",
      name: "Self-check inconsistency fixture",
      rootFingerprint: "sha256:edit-eval-self-discovered-inconsistency-v1",
      files: [
        { path: "config/feature.json", content: "{\n  \"newCheckout\": false\n}\n" },
        { path: "README.md", content: "newCheckout is disabled.\n" },
      ],
      protectedPaths: [],
    },
    coveredFailureModes: ["self_discovered_inconsistency", "multi_file_edit"],
    targetFiles: ["config/feature.json", "README.md"],
    expectedChangedFiles: ["config/feature.json", "README.md"],
    baselineWholeFileEdits: [
      {
        path: "config/feature.json",
        content: "{\n  \"newCheckout\": true\n}\n",
      },
      {
        path: "README.md",
        content: "newCheckout is enabled.\n",
      },
    ],
    assertions: [
      fileContains("assert.self-config", "config/feature.json", "\"newCheckout\": true"),
      fileContains("assert.self-readme", "README.md", "newCheckout is enabled"),
    ],
    probes: [
      {
        probeId: "probe.self.whole-file-partial",
        strategyFamily: "whole_file",
        description: "A partial edit that changes config only should be self-detected as inconsistent.",
        modelOutput: "PATH: config/feature.json\nCONTENT:\n{\n  \"newCheckout\": true\n}\n",
        expectedSelfDetectedRegressionStatus: "confirmed",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.no-op-request",
    split: "train",
    title: "No-op request",
    task: "Inspect the workspace and make no file changes because the requested state is already present.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.no-op-request",
      name: "No-op fixture",
      rootFingerprint: "sha256:edit-eval-no-op-request-v1",
      files: [{ path: "src/state.txt", content: "state=ready\n" }],
      protectedPaths: ["src/state.txt"],
    },
    coveredFailureModes: ["no_op_request"],
    targetFiles: [],
    expectedChangedFiles: [],
    forbiddenChangedFiles: ["src/state.txt"],
    baselineWholeFileEdits: [],
    assertions: [noForbiddenChange("assert.no-op-unchanged", ["src/state.txt"])],
    probes: [
      {
        probeId: "probe.no-op.apply-patch",
        strategyFamily: "apply_patch",
        description: "A no-op output should be accepted as skipped, not coerced into a write.",
        modelOutput: "NO_CHANGES\n",
        expectedParseStatus: "passed",
        expectedApplyStatus: "skipped",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.formatting-sensitive-json",
    split: "holdout",
    title: "Formatting-sensitive JSON edit",
    task: "Change version to 2 while preserving indentation, key order, and trailing newline.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.formatting-sensitive-json",
      name: "Formatting sensitive fixture",
      rootFingerprint: "sha256:edit-eval-formatting-sensitive-json-v1",
      files: [{ path: "manifest.json", content: "{\n  \"name\": \"fixture\",\n  \"version\": 1\n}\n" }],
      protectedPaths: [],
    },
    coveredFailureModes: ["formatting_sensitive"],
    targetFiles: ["manifest.json"],
    expectedChangedFiles: ["manifest.json"],
    baselineWholeFileEdits: [
      {
        path: "manifest.json",
        content: "{\n  \"name\": \"fixture\",\n  \"version\": 2\n}\n",
      },
    ],
    assertions: [fileContains("assert.formatting-version", "manifest.json", "\"version\": 2")],
    probes: [
      {
        probeId: "probe.formatting.whole-file",
        strategyFamily: "whole_file",
        description: "Whole-file replacement must preserve stable formatting constraints.",
        modelOutput: "PATH: manifest.json\nCONTENT:\n{\n  \"name\": \"fixture\",\n  \"version\": 2\n}\n",
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.protected-path-veto",
    split: "holdout",
    title: "Protected path veto",
    task: "Update package.json only; package-lock.json is protected and must not be rewritten.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.protected-path-veto",
      name: "Protected path fixture",
      rootFingerprint: "sha256:edit-eval-protected-path-veto-v1",
      files: [
        { path: "package.json", content: "{\n  \"version\": \"1.0.0\"\n}\n" },
        { path: "package-lock.json", content: "{\n  \"lockfileVersion\": 3\n}\n" },
      ],
      protectedPaths: ["package-lock.json"],
    },
    coveredFailureModes: ["protected_path"],
    targetFiles: ["package.json"],
    expectedChangedFiles: ["package.json"],
    forbiddenChangedFiles: ["package-lock.json"],
    baselineWholeFileEdits: [
      {
        path: "package.json",
        content: "{\n  \"version\": \"1.0.1\"\n}\n",
      },
    ],
    assertions: [
      fileContains("assert.protected-package", "package.json", "\"version\": \"1.0.1\""),
      noForbiddenChange("assert.protected-lockfile", ["package-lock.json"]),
    ],
    probes: [
      {
        probeId: "probe.protected.whole-file",
        strategyFamily: "whole_file",
        description: "A generated whole-file edit touching the lockfile must be vetoed.",
        modelOutput:
          "PATH: package-lock.json\nCONTENT:\n{\n  \"lockfileVersion\": 3,\n  \"packages\": {}\n}\n",
        expectedApplyStatus: "failed",
        expectedErrorCode: "protected_path_violation",
        expectedProtectedPathTouched: true,
      },
    ],
  },
  {
    editEvalCaseId: "edit-eval.fenced-diff-path-failure",
    split: "dev",
    title: "Fenced diff path failure",
    task: "Reject fenced diff output that omits the target path fence metadata.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.edit.fenced-diff-path-failure",
      name: "Fenced diff failure fixture",
      rootFingerprint: "sha256:edit-eval-fenced-diff-path-failure-v1",
      files: [{ path: "src/name.txt", content: "name=old\n" }],
      protectedPaths: ["src/name.txt"],
    },
    coveredFailureModes: ["strategy_specific_failure", "malformed_patch"],
    targetFiles: ["src/name.txt"],
    expectedChangedFiles: [],
    forbiddenChangedFiles: ["src/name.txt"],
    baselineWholeFileEdits: [],
    assertions: [noForbiddenChange("assert.fenced-no-write", ["src/name.txt"])],
    probes: [
      {
        probeId: "probe.fenced.path-missing",
        strategyFamily: "fenced_diff",
        description: "Fenced diff without path metadata must fail before write.",
        modelOutput: "```diff\n-name=old\n+name=new\n```\n",
        expectedParseStatus: "failed",
        expectedApplyStatus: "not_started",
        expectedErrorCode: "path_or_fence_error",
      },
    ],
  },
] as const;

export const editStrategyEvalCases: EditStrategyEvalCase[] = rawEditStrategyEvalCases.map((evalCase) => {
  const parsed = EditStrategyEvalCaseSchema.parse({
    schemaVersion: EDIT_STRATEGY_EVAL_SCHEMA_VERSION,
    ...evalCase,
  });
  FixtureWorkspaceSchema.parse(parsed.fixtureWorkspace);
  return parsed;
});

export const editStrategyFixtureWorkspaces: FixtureWorkspace[] = editStrategyEvalCases.map(
  (evalCase) => evalCase.fixtureWorkspace,
);

export const editStrategyEvalCasesByFailureMode = (): Record<EditStrategyEvalFailureMode, EditStrategyEvalCase[]> =>
  Object.fromEntries(
    EditStrategyEvalFailureModeSchema.options.map((failureMode) => [
      failureMode,
      editStrategyEvalCases.filter((evalCase) => evalCase.coveredFailureModes.includes(failureMode)),
    ]),
  ) as Record<EditStrategyEvalFailureMode, EditStrategyEvalCase[]>;

export const editStrategyEvalCasesByStrategyFamily = (): Record<EditStrategyFamily, EditStrategyEvalCase[]> =>
  Object.fromEntries(
    EditStrategyFamilySchema.options.map((family) => [
      family,
      editStrategyEvalCases.filter((evalCase) => evalCase.probes.some((probe) => probe.strategyFamily === family)),
    ]),
  ) as Record<EditStrategyFamily, EditStrategyEvalCase[]>;
