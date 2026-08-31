import { z } from "zod";
import {
  AcpReplayCaptureSchema,
  AcpReplayRecordSchema,
  type AcpReplayCapture,
  type AcpReplayCaptureInput,
} from "./capture";
import {
  ReplayExtractionMetadataSchema,
  extractReplayEvalCaseSkeleton,
  type ReplayEvalCaseSkeleton,
  type ReplayExtractionMetadataInput,
} from "./extraction";
import { EvalSplitSchema } from "../eval-harness/types";
import {
  EditAttemptContractSchema,
  EditStrategyFamilySchema,
  type EditAttemptPhase,
  type EditErrorCode,
  type EditStrategyFamily,
} from "../edit-strategy/types";
import { OptimizerIdSchema } from "../optimizer/types";

const EDIT_FAILURE_SCENARIO_CREATED_AT = "2026-05-01T00:00:00.000Z";
const DEFAULT_TIMEOUT_MS = 120000;
const DEFAULT_CONTEXT = {
  policyId: "policy.replay-edit-failure.synthetic.v1",
  modelProfileId: "model.replay.synthetic",
  codebaseProfileId: "codebase.bleeding-agent.synthetic",
  canonicalToolVersion: "canonical-tools.synthetic.v1",
  renderedToolVersion: "rendered-tools.synthetic.v1",
  resultStyleVersion: "result-style.synthetic.v1",
  verificationPolicyVersion: "verification.synthetic.v1",
};

const ReplayEditFailureScenarioKindSchema = z.enum([
  "parse_failure",
  "apply_failure",
  "stale_context",
  "protected_path",
  "fallback_success_after_primary_failure",
  "applied_but_broken_file",
  "applied_but_broken_verification_failure",
  "self_detected_regression",
  "promotion_veto",
]);
export type ReplayEditFailureScenarioKind = z.infer<typeof ReplayEditFailureScenarioKindSchema>;

const ReplayEditFailureScenarioSchema = z.object({
  scenarioId: OptimizerIdSchema,
  scenarioKind: ReplayEditFailureScenarioKindSchema,
  optimizationAllowed: z.boolean(),
  split: EvalSplitSchema,
  capture: AcpReplayCaptureSchema,
  metadata: ReplayExtractionMetadataSchema,
}).strict().superRefine((scenario, ctx) => {
  if (scenario.split !== scenario.metadata.split) {
    ctx.addIssue({
      code: "custom",
      path: ["metadata", "split"],
      message: "scenario split must match extraction metadata split",
    });
  }

  if (scenario.capture.defaultSplitHint !== undefined && scenario.capture.defaultSplitHint !== scenario.split) {
    ctx.addIssue({
      code: "custom",
      path: ["capture", "defaultSplitHint"],
      message: "capture split hint must match scenario split",
    });
  }

  if (scenario.split === "holdout" && scenario.optimizationAllowed) {
    ctx.addIssue({
      code: "custom",
      path: ["optimizationAllowed"],
      message: "hidden holdout replay scenarios must not be optimization input",
    });
  }
});
export type ReplayEditFailureScenario = z.infer<typeof ReplayEditFailureScenarioSchema>;

type AcpReplayRecordInput = z.input<typeof AcpReplayRecordSchema>;
type EditAttemptInput = z.input<typeof EditAttemptContractSchema>;
type ReplayEditFailureScenarioInput = {
  scenarioId: string;
  scenarioKind: ReplayEditFailureScenarioKind;
  optimizationAllowed?: boolean;
  capture: AcpReplayCaptureInput;
  metadata: ReplayExtractionMetadataInput;
};

const editFailureScenario = (input: ReplayEditFailureScenarioInput): ReplayEditFailureScenario => {
  const metadata = ReplayExtractionMetadataSchema.parse(input.metadata);
  return ReplayEditFailureScenarioSchema.parse({
    scenarioId: input.scenarioId,
    scenarioKind: input.scenarioKind,
    optimizationAllowed: input.optimizationAllowed ?? metadata.split !== "holdout",
    split: metadata.split,
    capture: input.capture,
    metadata,
  });
};

const manualCapture = (
  scenarioId: string,
  records: AcpReplayRecordInput[],
  split: z.infer<typeof EvalSplitSchema>,
): AcpReplayCaptureInput => ({
  captureId: `capture.${scenarioId}`,
  createdAt: EDIT_FAILURE_SCENARIO_CREATED_AT,
  source: {
    sourceType: "manual",
    path: `synthetic://replay/edit-failure/${scenarioId}`,
    sessionId: `session.${scenarioId}`,
    traceIds: [`trace.${scenarioId}`],
  },
  context: DEFAULT_CONTEXT,
  defaultSplitHint: split,
  redactionStatus: "redacted",
  records,
});

const promptRecord = (scenarioId: string, content: string): AcpReplayRecordInput => ({
  recordId: `record.${scenarioId}.prompt`,
  recordKind: "prompt",
  promptRole: "user",
  content,
  contentRedactionStatus: "redacted",
  traceRefs: [{ traceId: `trace.${scenarioId}`, spanId: `span.${scenarioId}.prompt` }],
});

const routeRecord = (scenarioId: string): AcpReplayRecordInput => ({
  recordId: `record.${scenarioId}.route`,
  recordKind: "mode_route",
  promptRecordId: `record.${scenarioId}.prompt`,
  parentRecordIds: [`record.${scenarioId}.prompt`],
  requestedMode: "auto",
  selectedMode: "mutating",
  sideEffectPolicy: "write_allowed",
  reason: "The user requested a bounded workspace edit, so replay should exercise the edit strategy path.",
  traceRefs: [{ traceId: `trace.${scenarioId}`, spanId: `span.${scenarioId}.route` }],
});

const fileReadRecord = (
  scenarioId: string,
  path: string,
  wholeFileSeen = true,
): AcpReplayRecordInput => ({
  recordId: `record.${scenarioId}.file`,
  recordKind: "file_read",
  parentRecordIds: [`record.${scenarioId}.route`],
  path,
  status: "succeeded",
  contentHash: `sha256:${scenarioId}.before`,
  excerpt: wholeFileSeen ? undefined : "redacted excerpt around target hunk",
  redactionStatus: "hash_only",
  ranges: wholeFileSeen ? [{ startLine: 0, endLine: 6 }] : [{ startLine: 3, endLine: 5 }],
  traceRefs: [{ traceId: `trace.${scenarioId}`, spanId: `span.${scenarioId}.file` }],
});

const fixtureWorkspace = (
  scenarioId: string,
  path: string,
  content: string,
  protectedPaths: string[] = [],
): NonNullable<ReplayExtractionMetadataInput["fixtureWorkspace"]> => ({
  fixtureWorkspaceId: `fixture.${scenarioId}`,
  name: scenarioId,
  description: "Synthetic redacted replay workspace for edit-failure evaluation.",
  rootFingerprint: `sha256:${scenarioId}.root`,
  files: [{ path, content }],
  protectedPaths,
  verificationCommands: [["npm", "run", "typecheck"]],
});

const sourceRef = (
  scenarioId: string,
  suffix: string,
  redactionStatus: "redacted" | "hash_only" = "redacted",
): NonNullable<ReplayExtractionMetadataInput["sourceRefs"]>[number] => ({
  sourceKind: "fixture",
  path: `synthetic://replay/edit-failure/${scenarioId}/${suffix}`,
  redactionStatus,
});

const baseAttempt = (
  scenarioId: string,
  suffix: string,
  family: EditStrategyFamily,
  targetFiles: string[],
  overrides: Partial<EditAttemptInput>,
): EditAttemptInput => EditAttemptContractSchema.parse({
  editAttemptId: `edit.${scenarioId}.${suffix}`,
  runId: `run.${scenarioId}`,
  traceId: `trace.${scenarioId}`,
  modelProfileId: DEFAULT_CONTEXT.modelProfileId,
  codebaseProfileId: DEFAULT_CONTEXT.codebaseProfileId,
  policyId: DEFAULT_CONTEXT.policyId,
  editStrategyId: `edit.synthetic.${family}.${suffix}`,
  editStrategyFamily: EditStrategyFamilySchema.parse(family),
  canonicalEditToolSpecId: `tool.edit.${family}`,
  renderedEditToolContractId: `tool.edit.${family}.synthetic`,
  taskShape: {
    replayScenarioKind: scenarioId,
    failurePack: "edit-failure",
  },
  targetFiles,
  readSnapshotRefs: targetFiles.map((path) => ({
    snapshotId: `snapshot.${scenarioId}.${path.replaceAll("/", "_")}.before`,
    path,
    contentHash: `sha256:${scenarioId}.${path}.before`,
    wholeFileSeen: true,
  })),
  inputContentHashes: Object.fromEntries(targetFiles.map((path) => [path, `sha256:${scenarioId}.${path}.before`])),
  verificationStatus: "not_run",
  staleContextStatus: "fresh",
  redactionStatus: "redacted",
  artifactRefs: [`artifact:${scenarioId}.${suffix}.edit-attempt`],
  createdAt: EDIT_FAILURE_SCENARIO_CREATED_AT,
  ...overrides,
});

const failedPhase = (
  phase: EditAttemptPhase,
  errorCode: EditErrorCode,
  scenarioId: string,
  suffix: string,
  message: string,
): NonNullable<EditAttemptInput["phaseResults"]>[number] => ({
  phase,
  status: "failed",
  errorCode,
  message,
  artifactRefs: [`artifact:${scenarioId}.${suffix}.${phase}`],
});

const passedPhase = (phase: EditAttemptPhase): NonNullable<EditAttemptInput["phaseResults"]>[number] => ({
  phase,
  status: "passed",
});

const editRecord = (
  scenarioId: string,
  suffix: string,
  parentRecordIds: string[],
  attempt: EditAttemptInput,
): AcpReplayRecordInput => ({
  recordId: `record.${scenarioId}.${suffix}`,
  recordKind: "edit_attempt",
  parentRecordIds,
  artifactRefs: attempt.artifactRefs,
  traceRefs: [{ traceId: `trace.${scenarioId}`, spanId: `span.${scenarioId}.${suffix}` }],
  attempt,
});

const editFailureReplayScenariosInput = [
  editFailureScenario({
    scenarioId: "replay.edit-failure.parse-failure",
    scenarioKind: "parse_failure",
    capture: manualCapture(
      "replay.edit-failure.parse-failure",
      [
        promptRecord("replay.edit-failure.parse-failure", "Patch src/parser.ts using a structured edit payload."),
        routeRecord("replay.edit-failure.parse-failure"),
        fileReadRecord("replay.edit-failure.parse-failure", "src/parser.ts"),
        editRecord(
          "replay.edit-failure.parse-failure",
          "primary",
          ["record.replay.edit-failure.parse-failure.file"],
          baseAttempt("replay.edit-failure.parse-failure", "primary", "apply_patch", ["src/parser.ts"], {
            phaseResults: [
              passedPhase("generation"),
              failedPhase(
                "parse",
                "parse_error",
                "replay.edit-failure.parse-failure",
                "primary",
                "The model returned text that could not be parsed as the rendered edit contract.",
              ),
            ],
            parseErrorCode: "parse_error",
            staleContextStatus: "not_checked",
            artifactRefs: ["artifact:parse-failure.raw-response", "artifact:parse-failure.parse-error"],
          }),
        ),
      ],
      "train",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.parse-failure",
      title: "Malformed edit payload is captured as parse failure",
      split: "train",
      splitRationale: "Visible synthetic training fixture for edit contract parsing failures.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The replay must expose the parse failure without treating the attempt as an applied patch.",
        assertions: [
          {
            assertionId: "assert.edit.parse.failure-visible",
            assertionKind: "json_pointer_equals",
            description: "The first observed failure is the edit parse error.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "parse_error",
          },
          {
            assertionId: "assert.edit.parse.no-output-hash",
            assertionKind: "json_pointer_equals",
            description: "The parse-failed attempt did not produce changed files.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/phase",
            expected: "parse",
          },
        ],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.parse-failure",
        "src/parser.ts",
        "export const parse = (value: string) => value.trim();\n",
      ),
      sourceRefs: [sourceRef("replay.edit-failure.parse-failure", "raw-response")],
      tags: ["replay", "edit-failure", "parse", "train"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.apply-failure",
    scenarioKind: "apply_failure",
    capture: manualCapture(
      "replay.edit-failure.apply-failure",
      [
        promptRecord("replay.edit-failure.apply-failure", "Replace the legacy status branch in src/status.ts."),
        routeRecord("replay.edit-failure.apply-failure"),
        fileReadRecord("replay.edit-failure.apply-failure", "src/status.ts", false),
        editRecord(
          "replay.edit-failure.apply-failure",
          "primary",
          ["record.replay.edit-failure.apply-failure.file"],
          baseAttempt("replay.edit-failure.apply-failure", "primary", "exact_replace", ["src/status.ts"], {
            readSnapshotRefs: [
              {
                snapshotId: "snapshot.replay.edit-failure.apply-failure.status.range",
                path: "src/status.ts",
                contentHash: "sha256:replay.edit-failure.apply-failure.src/status.ts.before",
                wholeFileSeen: false,
                ranges: [{ startLine: 3, endLine: 5 }],
              },
            ],
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              failedPhase(
                "apply",
                "exact_match_not_found",
                "replay.edit-failure.apply-failure",
                "primary",
                "The expected replacement block was not present in the workspace snapshot.",
              ),
            ],
            applyErrorCode: "exact_match_not_found",
            artifactRefs: ["artifact:apply-failure.patch", "artifact:apply-failure.apply-error"],
          }),
        ),
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.apply-failure",
      title: "Exact replacement failure preserves apply evidence",
      split: "dev",
      splitRationale: "Visible dev fixture for comparing apply-layer recovery behavior.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The replay should distinguish a valid parsed edit from an apply failure caused by a missing exact match.",
        assertions: [
          {
            assertionId: "assert.edit.apply.failure-visible",
            assertionKind: "json_pointer_equals",
            description: "The observed failure records apply as the failed phase.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/phase",
            expected: "apply",
          },
          {
            assertionId: "assert.edit.apply.error-code",
            assertionKind: "json_pointer_equals",
            description: "The apply-layer error code is preserved.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "exact_match_not_found",
          },
        ],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.apply-failure",
        "src/status.ts",
        "export const status = () => 'current';\n",
      ),
      sourceRefs: [sourceRef("replay.edit-failure.apply-failure", "apply-error", "hash_only")],
      tags: ["replay", "edit-failure", "apply", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.stale-context",
    scenarioKind: "stale_context",
    capture: manualCapture(
      "replay.edit-failure.stale-context",
      [
        promptRecord("replay.edit-failure.stale-context", "Update the retry limit in src/retry.ts."),
        routeRecord("replay.edit-failure.stale-context"),
        fileReadRecord("replay.edit-failure.stale-context", "src/retry.ts"),
        editRecord(
          "replay.edit-failure.stale-context",
          "primary",
          ["record.replay.edit-failure.stale-context.file"],
          baseAttempt("replay.edit-failure.stale-context", "primary", "hash_range", ["src/retry.ts"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              failedPhase(
                "stale_context_check",
                "anchor_stale",
                "replay.edit-failure.stale-context",
                "primary",
                "The recorded snapshot hash no longer matched the file before apply.",
              ),
            ],
            staleContextStatus: "stale",
            applyErrorCode: "anchor_stale",
            artifactRefs: ["artifact:stale-context.snapshot", "artifact:stale-context.current-hash"],
          }),
        ),
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.stale-context",
      title: "Stale context blocks hash-range apply",
      split: "dev",
      splitRationale: "Visible dev fixture for stale-context safeguards.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The edit should stop before apply when the snapshot hash is stale and preserve both old and current hash evidence.",
        assertions: [
          {
            assertionId: "assert.edit.stale.phase",
            assertionKind: "json_pointer_equals",
            description: "The stale-context check is the failed phase.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/phase",
            expected: "stale_context_check",
          },
          {
            assertionId: "assert.edit.stale.code",
            assertionKind: "json_pointer_equals",
            description: "The stale anchor error is preserved.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "anchor_stale",
          },
        ],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.stale-context",
        "src/retry.ts",
        "export const retryLimit = 3;\n",
      ),
      sourceRefs: [sourceRef("replay.edit-failure.stale-context", "hashes", "hash_only")],
      tags: ["replay", "edit-failure", "stale-context", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.protected-path",
    scenarioKind: "protected_path",
    capture: manualCapture(
      "replay.edit-failure.protected-path",
      [
        promptRecord("replay.edit-failure.protected-path", "Update src/feature.ts without touching package.json."),
        routeRecord("replay.edit-failure.protected-path"),
        fileReadRecord("replay.edit-failure.protected-path", "package.json"),
        editRecord(
          "replay.edit-failure.protected-path",
          "primary",
          ["record.replay.edit-failure.protected-path.file"],
          baseAttempt("replay.edit-failure.protected-path", "primary", "whole_file", ["package.json"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              failedPhase(
                "permission",
                "protected_path_violation",
                "replay.edit-failure.protected-path",
                "primary",
                "The proposed edit targeted a protected manifest outside the requested scope.",
              ),
            ],
            permissionStatus: "failed",
            protectedPathTouched: true,
            applyErrorCode: "protected_path_violation",
            artifactRefs: ["artifact:protected-path.policy-decision"],
          }),
        ),
      ],
      "train",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.protected-path",
      title: "Protected manifest edit is rejected before write",
      split: "train",
      splitRationale: "Visible training fixture for protected path policy behavior.",
      oracleStrength: "golden",
      expectedBehavior: {
        summary: "The replay must preserve that the model attempted a protected path and that policy rejected the edit.",
        assertions: [
          {
            assertionId: "assert.edit.protected.error",
            assertionKind: "json_pointer_equals",
            description: "The protected path violation appears in observed failures.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "protected_path_violation",
          },
          {
            assertionId: "assert.edit.protected.no-manifest-write",
            assertionKind: "no_forbidden_path_changed",
            description: "The manifest remains protected.",
            severity: "critical",
            paths: ["package.json"],
          },
        ],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.protected-path",
        "package.json",
        "{\"name\":\"synthetic\"}\n",
        ["package.json"],
      ),
      sourceRefs: [sourceRef("replay.edit-failure.protected-path", "policy-decision")],
      tags: ["replay", "edit-failure", "protected-path", "train"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.fallback-success",
    scenarioKind: "fallback_success_after_primary_failure",
    capture: manualCapture(
      "replay.edit-failure.fallback-success",
      [
        promptRecord("replay.edit-failure.fallback-success", "Rename the exported helper in src/helper.ts."),
        routeRecord("replay.edit-failure.fallback-success"),
        fileReadRecord("replay.edit-failure.fallback-success", "src/helper.ts"),
        editRecord(
          "replay.edit-failure.fallback-success",
          "primary",
          ["record.replay.edit-failure.fallback-success.file"],
          baseAttempt("replay.edit-failure.fallback-success", "primary", "unified_diff", ["src/helper.ts"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              failedPhase(
                "apply",
                "hunk_context_mismatch",
                "replay.edit-failure.fallback-success",
                "primary",
                "The primary unified diff hunk did not match the current file context.",
              ),
              {
                phase: "fallback",
                status: "passed",
                message: "Policy routed the edit to the exact replacement fallback.",
                artifactRefs: ["artifact:fallback-success.route"],
              },
            ],
            applyErrorCode: "hunk_context_mismatch",
            fallbackFromStrategyId: "edit.synthetic.unified_diff.primary",
            fallbackToStrategyId: "edit.synthetic.exact_replace.fallback",
            artifactRefs: ["artifact:fallback-success.primary-failure", "artifact:fallback-success.route"],
          }),
        ),
        editRecord(
          "replay.edit-failure.fallback-success",
          "fallback",
          ["record.replay.edit-failure.fallback-success.primary"],
          baseAttempt("replay.edit-failure.fallback-success", "fallback", "exact_replace", ["src/helper.ts"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              passedPhase("apply"),
              passedPhase("verify"),
            ],
            outputContentHashes: {
              "src/helper.ts": "sha256:replay.edit-failure.fallback-success.src/helper.ts.after",
            },
            verificationStatus: "passed",
            fallbackFromStrategyId: "edit.synthetic.unified_diff.primary",
            changedFileCount: 1,
            changedLineCount: 2,
            artifactRefs: ["artifact:fallback-success.fallback-patch", "artifact:fallback-success.verify"],
          }),
        ),
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.fallback-success",
      title: "Fallback success preserves primary failure evidence",
      split: "dev",
      splitRationale: "Visible dev fixture for fallback routing and failure preservation.",
      oracleStrength: "golden",
      expectedBehavior: {
        summary: "A successful fallback must not erase the failed primary apply attempt that caused fallback routing.",
        assertions: [
          {
            assertionId: "assert.edit.fallback.primary-visible",
            assertionKind: "json_pointer_equals",
            description: "The primary hunk-context failure remains the first observed failure.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "hunk_context_mismatch",
          },
          {
            assertionId: "assert.edit.fallback.final-content",
            assertionKind: "file_contains",
            description: "The fallback edit still produces the requested helper rename.",
            path: "src/helper.ts",
            text: "export const renamedHelper",
          },
        ],
        notes: ["Preserve primary failure evidence even when the final edit result succeeds via fallback."],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.fallback-success",
        "src/helper.ts",
        "export const oldHelper = () => true;\n",
      ),
      sourceRefs: [
        sourceRef("replay.edit-failure.fallback-success", "primary-failure"),
        sourceRef("replay.edit-failure.fallback-success", "fallback-patch"),
      ],
      tags: ["replay", "edit-failure", "fallback", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.applied-broken-file",
    scenarioKind: "applied_but_broken_file",
    capture: manualCapture(
      "replay.edit-failure.applied-broken-file",
      [
        promptRecord("replay.edit-failure.applied-broken-file", "Refactor src/render.ts without breaking the exported file."),
        routeRecord("replay.edit-failure.applied-broken-file"),
        fileReadRecord("replay.edit-failure.applied-broken-file", "src/render.ts"),
        editRecord(
          "replay.edit-failure.applied-broken-file",
          "primary",
          ["record.replay.edit-failure.applied-broken-file.file"],
          baseAttempt("replay.edit-failure.applied-broken-file", "primary", "apply_patch", ["src/render.ts"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              passedPhase("apply"),
              passedPhase("write"),
              failedPhase(
                "post_apply_consistency",
                "post_apply_syntax_failure",
                "replay.edit-failure.applied-broken-file",
                "primary",
                "The write landed, but post-apply consistency found a broken TypeScript export.",
              ),
            ],
            outputContentHashes: {
              "src/render.ts": "sha256:replay.edit-failure.applied-broken-file.src/render.ts.after",
            },
            targetContentHashes: [
              {
                path: "src/render.ts",
                beforeHash: "sha256:replay.edit-failure.applied-broken-file.src/render.ts.before",
                afterHash: "sha256:replay.edit-failure.applied-broken-file.src/render.ts.after",
                readSnapshotId: "snapshot.replay.edit-failure.applied-broken-file.src_render.ts.before",
                writeArtifactRef: "artifact:applied-broken-file.write",
              },
            ],
            verificationStatus: "skipped",
            postApplyConsistencyStatus: "inconsistent",
            selfDetectedRegressionStatus: "suspected",
            rollbackStatus: "not_attempted",
            changedFileCount: 1,
            changedLineCount: 4,
            artifactRefs: ["artifact:applied-broken-file.patch", "artifact:applied-broken-file.write"],
          }),
        ),
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.applied-broken-file",
      title: "Applied file write fails post-apply consistency",
      split: "dev",
      splitRationale: "Visible dev fixture for applied-but-broken files before command verification runs.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The replay must show that write/apply succeeded before post-apply consistency detected a broken file.",
        assertions: [
          {
            assertionId: "assert.edit.applied-broken-file.phase",
            assertionKind: "json_pointer_equals",
            description: "Post-apply consistency is the failed phase.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/phase",
            expected: "post_apply_consistency",
          },
          {
            assertionId: "assert.edit.applied-broken-file.hashes",
            assertionKind: "json_pointer_equals",
            description: "Target before/after hash evidence exists on the edit attempt.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "post_apply_syntax_failure",
          },
        ],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.applied-broken-file",
        "src/render.ts",
        "export const render = () => \"ok\";\n",
      ),
      sourceRefs: [
        sourceRef("replay.edit-failure.applied-broken-file", "post-apply-consistency"),
        sourceRef("replay.edit-failure.applied-broken-file", "write", "hash_only"),
      ],
      tags: ["replay", "edit-failure", "applied-but-broken", "post-apply", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.verification-failure",
    scenarioKind: "applied_but_broken_verification_failure",
    capture: manualCapture(
      "replay.edit-failure.verification-failure",
      [
        promptRecord("replay.edit-failure.verification-failure", "Update src/calculator.ts and run verification."),
        routeRecord("replay.edit-failure.verification-failure"),
        fileReadRecord("replay.edit-failure.verification-failure", "src/calculator.ts"),
        editRecord(
          "replay.edit-failure.verification-failure",
          "primary",
          ["record.replay.edit-failure.verification-failure.file"],
          baseAttempt("replay.edit-failure.verification-failure", "primary", "apply_patch", ["src/calculator.ts"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              passedPhase("apply"),
              failedPhase(
                "verify",
                "post_apply_behavior_failure",
                "replay.edit-failure.verification-failure",
                "primary",
                "The patch applied but the focused verification command failed afterward.",
              ),
            ],
            outputContentHashes: {
              "src/calculator.ts": "sha256:replay.edit-failure.verification-failure.src/calculator.ts.after",
            },
            verificationStatus: "failed",
            postApplyConsistencyStatus: "inconsistent",
            rollbackStatus: "not_attempted",
            changedFileCount: 1,
            changedLineCount: 3,
            artifactRefs: ["artifact:verification-failure.patch", "artifact:verification-failure.stderr"],
          }),
        ),
        {
          recordId: "record.replay.edit-failure.verification-failure.verify-command",
          recordKind: "terminal_command",
          parentRecordIds: ["record.replay.edit-failure.verification-failure.primary"],
          commandId: "terminal.replay.edit-failure.verification-failure.typecheck",
          command: ["npm", "run", "typecheck"],
          cwd: "/workspace",
          status: "failed",
          exitCode: 2,
          stderrArtifactRef: "artifact:verification-failure.stderr",
          redactionStatus: "hash_only",
          errorCode: "post_apply_behavior_failure",
          traceRefs: [
            {
              traceId: "trace.replay.edit-failure.verification-failure",
              spanId: "span.replay.edit-failure.verification-failure.verify-command",
            },
          ],
        },
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.verification-failure",
      title: "Applied patch fails verification",
      split: "dev",
      splitRationale: "Visible dev fixture for applied-but-broken edits.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The replay must represent the edit as applied but broken, with verification failure artifacts preserved.",
        assertions: [
          {
            assertionId: "assert.edit.verify.failed",
            assertionKind: "json_pointer_equals",
            description: "The edit verification phase failed after apply passed.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/phase",
            expected: "verify",
          },
          {
            assertionId: "assert.edit.verify.command-exit",
            assertionKind: "command_exit_code",
            description: "The focused verification command failed.",
            commandId: "terminal.replay.edit-failure.verification-failure.typecheck",
            expectedExitCode: 2,
          },
        ],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.verification-failure",
        "src/calculator.ts",
        "export const add = (left: number, right: number) => left + right;\n",
      ),
      sourceRefs: [sourceRef("replay.edit-failure.verification-failure", "stderr", "hash_only")],
      tags: ["replay", "edit-failure", "verification", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.promotion-veto",
    scenarioKind: "promotion_veto",
    optimizationAllowed: false,
    capture: manualCapture(
      "replay.edit-failure.promotion-veto",
      [
        promptRecord("replay.edit-failure.promotion-veto", "Promote the measured edit policy only if replay and holdout gates pass."),
        routeRecord("replay.edit-failure.promotion-veto"),
        fileReadRecord("replay.edit-failure.promotion-veto", "src/veto.ts"),
        editRecord(
          "replay.edit-failure.promotion-veto",
          "candidate",
          ["record.replay.edit-failure.promotion-veto.file"],
          baseAttempt("replay.edit-failure.promotion-veto", "candidate", "apply_patch", ["src/veto.ts"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              passedPhase("apply"),
              passedPhase("write"),
              failedPhase(
                "verify",
                "post_apply_behavior_failure",
                "replay.edit-failure.promotion-veto",
                "candidate",
                "The hidden holdout replay failed after the candidate edit policy applied its patch.",
              ),
            ],
            outputContentHashes: {
              "src/veto.ts": "sha256:replay.edit-failure.promotion-veto.src/veto.ts.after",
            },
            verificationStatus: "failed",
            postApplyConsistencyStatus: "inconsistent",
            rollbackStatus: "not_attempted",
            changedFileCount: 1,
            changedLineCount: 2,
            artifactRefs: ["artifact:promotion-veto.patch", "artifact:promotion-veto.holdout-scorecard"],
          }),
        ),
        {
          recordId: "record.replay.edit-failure.promotion-veto.gate",
          recordKind: "terminal_command",
          parentRecordIds: ["record.replay.edit-failure.promotion-veto.candidate"],
          commandId: "terminal.replay.edit-failure.promotion-veto.gate",
          command: ["bag", "optimizer", "promote-edit-policy"],
          cwd: "/workspace",
          status: "failed",
          exitCode: 1,
          stderrArtifactRef: "artifact:promotion-veto.gate-decision",
          redactionStatus: "hash_only",
          errorCode: "promotion_veto",
          traceRefs: [
            {
              traceId: "trace.replay.edit-failure.promotion-veto",
              spanId: "span.replay.edit-failure.promotion-veto.gate",
            },
          ],
        },
      ],
      "holdout",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.promotion-veto",
      title: "Promotion gate veto preserves failed holdout replay evidence",
      split: "holdout",
      splitRationale: "Synthetic hidden holdout fixture for promotion veto behavior; excluded from optimizer feedback.",
      oracleStrength: "golden",
      expectedBehavior: {
        summary: "Promotion must be vetoed when holdout replay detects an applied-but-broken edit policy candidate.",
        assertions: [
          {
            assertionId: "assert.edit.promotion-veto.verify",
            assertionKind: "json_pointer_equals",
            description: "The holdout candidate edit failed verification.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "post_apply_behavior_failure",
          },
          {
            assertionId: "assert.edit.promotion-veto.gate",
            assertionKind: "json_pointer_equals",
            description: "The promotion command records a gate veto.",
            artifact: "telemetry",
            pointer: "/observedFailures/1/errorCode",
            expected: "promotion_veto",
          },
        ],
        notes: ["This is a fixture-only replay case, not evidence of live ACP extraction."],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.promotion-veto",
        "src/veto.ts",
        "export const gate = () => \"holdout\";\n",
      ),
      sourceRefs: [
        sourceRef("replay.edit-failure.promotion-veto", "holdout-scorecard", "hash_only"),
        sourceRef("replay.edit-failure.promotion-veto", "gate-decision", "hash_only"),
      ],
      tags: ["replay", "edit-failure", "promotion-veto", "holdout"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  editFailureScenario({
    scenarioId: "replay.edit-failure.self-detected-regression",
    scenarioKind: "self_detected_regression",
    optimizationAllowed: false,
    capture: manualCapture(
      "replay.edit-failure.self-detected-regression",
      [
        promptRecord("replay.edit-failure.self-detected-regression", "Simplify src/cache.ts without changing cache semantics."),
        routeRecord("replay.edit-failure.self-detected-regression"),
        fileReadRecord("replay.edit-failure.self-detected-regression", "src/cache.ts"),
        editRecord(
          "replay.edit-failure.self-detected-regression",
          "primary",
          ["record.replay.edit-failure.self-detected-regression.file"],
          baseAttempt("replay.edit-failure.self-detected-regression", "primary", "whole_file", ["src/cache.ts"], {
            phaseResults: [
              passedPhase("generation"),
              passedPhase("parse"),
              passedPhase("apply"),
              failedPhase(
                "self_check",
                "self_detected_regression",
                "replay.edit-failure.self-detected-regression",
                "primary",
                "The agent compared the requested invariant against its patch and detected a behavior regression.",
              ),
            ],
            outputContentHashes: {
              "src/cache.ts": "sha256:replay.edit-failure.self-detected-regression.src/cache.ts.after",
            },
            verificationStatus: "not_run",
            postApplyConsistencyStatus: "inconsistent",
            selfDetectedRegressionStatus: "confirmed",
            selfDetectedRegressionEvidenceRefs: ["artifact:self-detected-regression.self-check"],
            rollbackStatus: "not_attempted",
            changedFileCount: 1,
            changedLineCount: 8,
            artifactRefs: ["artifact:self-detected-regression.patch", "artifact:self-detected-regression.self-check"],
          }),
        ),
      ],
      "holdout",
    ),
    metadata: {
      evalCaseId: "replay.eval.edit-failure.self-detected-regression",
      title: "Self-check catches semantic regression",
      split: "holdout",
      splitRationale: "Synthetic hidden holdout fixture; excluded from optimizer feedback by default.",
      oracleStrength: "golden",
      expectedBehavior: {
        summary: "The agent should preserve self-detected regression evidence and avoid using this holdout case for optimization prompts.",
        assertions: [
          {
            assertionId: "assert.edit.self-check.failed",
            assertionKind: "json_pointer_equals",
            description: "The self-check phase records a confirmed regression.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "self_detected_regression",
          },
          {
            assertionId: "assert.edit.self-check.no-optimization",
            assertionKind: "json_pointer_equals",
            description: "The hidden holdout split is preserved in replay metadata.",
            artifact: "telemetry",
            pointer: "/split",
            expected: "holdout",
          },
        ],
      },
      fixtureWorkspace: fixtureWorkspace(
        "replay.edit-failure.self-detected-regression",
        "src/cache.ts",
        "export const cacheKey = (id: string) => id.toLowerCase();\n",
      ),
      sourceRefs: [sourceRef("replay.edit-failure.self-detected-regression", "self-check")],
      tags: ["replay", "edit-failure", "self-detected-regression", "holdout"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
] satisfies ReplayEditFailureScenario[];

export const editFailureReplayScenarios: ReplayEditFailureScenario[] = editFailureReplayScenariosInput;

export const editFailureReplayScenarioSkeletons: ReplayEvalCaseSkeleton[] =
  editFailureReplayScenarios.map((scenario) => extractReplayEvalCaseSkeleton(scenario));

export const visibleEditFailureReplayScenariosForOptimization = (): ReplayEditFailureScenario[] =>
  editFailureReplayScenarios.filter((scenario) => scenario.optimizationAllowed && scenario.split !== "holdout");

export const extractEditFailureReplayScenarioSkeletons = (
  scenarios: readonly ReplayEditFailureScenario[] = editFailureReplayScenarios,
): ReplayEvalCaseSkeleton[] =>
  scenarios.map((scenario) => extractReplayEvalCaseSkeleton({
    capture: scenario.capture,
    metadata: scenario.metadata,
  }));

export const editFailureReplayCaptures: AcpReplayCapture[] =
  editFailureReplayScenarios.map((scenario) => scenario.capture);
