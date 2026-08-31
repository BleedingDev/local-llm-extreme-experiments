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
import { OptimizerIdSchema } from "../optimizer/types";

const TOOL_CALL_SCENARIO_CREATED_AT = "2026-05-01T00:00:00.000Z";
const DEFAULT_TIMEOUT_MS = 120000;
const DEFAULT_CONTEXT = {
  policyId: "policy.replay-tool-call.synthetic.v1",
  modelProfileId: "model.replay.synthetic",
  codebaseProfileId: "codebase.bleeding-agent.synthetic",
  canonicalToolVersion: "canonical-tools.synthetic.v1",
  renderedToolVersion: "rendered-tools.synthetic.v1",
  resultStyleVersion: "result-style.synthetic.v1",
  verificationPolicyVersion: "verification.synthetic.v1",
};

const ReplayToolCallScenarioKindSchema = z.enum([
  "malformed_tool_arguments",
  "oversized_output",
  "permission_denial",
  "retry_behavior",
  "truncation_visibility",
  "mcp_call",
  "terminal_verification_enforcement",
]);
export type ReplayToolCallScenarioKind = z.infer<typeof ReplayToolCallScenarioKindSchema>;

const ReplayToolCallScenarioSchema = z.object({
  scenarioId: OptimizerIdSchema,
  scenarioKind: ReplayToolCallScenarioKindSchema,
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
export type ReplayToolCallScenario = z.infer<typeof ReplayToolCallScenarioSchema>;

type AcpReplayRecordInput = z.input<typeof AcpReplayRecordSchema>;
type ReplayToolCallScenarioInput = {
  scenarioId: string;
  scenarioKind: ReplayToolCallScenarioKind;
  optimizationAllowed?: boolean;
  capture: AcpReplayCaptureInput;
  metadata: ReplayExtractionMetadataInput;
};

const toolCallScenario = (input: ReplayToolCallScenarioInput): ReplayToolCallScenario => {
  const metadata = ReplayExtractionMetadataSchema.parse(input.metadata);
  return ReplayToolCallScenarioSchema.parse({
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
  createdAt: TOOL_CALL_SCENARIO_CREATED_AT,
  source: {
    sourceType: "manual",
    path: `synthetic://replay/tool-call/${scenarioId}`,
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

const routeRecord = (
  scenarioId: string,
  selectedMode: "read_only" | "mutating" | "safe" = "read_only",
  sideEffectPolicy: "read_only" | "write_allowed" | "terminal_allowed" | "no_side_effects" = "read_only",
): AcpReplayRecordInput => ({
  recordId: `record.${scenarioId}.route`,
  recordKind: "mode_route",
  promptRecordId: `record.${scenarioId}.prompt`,
  parentRecordIds: [`record.${scenarioId}.prompt`],
  requestedMode: "auto",
  selectedMode,
  sideEffectPolicy,
  reason: "The replay exercises tool execution behavior while preserving the routing decision that allowed it.",
  traceRefs: [{ traceId: `trace.${scenarioId}`, spanId: `span.${scenarioId}.route` }],
});

const sourceRef = (
  scenarioId: string,
  suffix: string,
  redactionStatus: "redacted" | "hash_only" = "redacted",
): NonNullable<ReplayExtractionMetadataInput["sourceRefs"]>[number] => ({
  sourceKind: "fixture",
  path: `synthetic://replay/tool-call/${scenarioId}/${suffix}`,
  redactionStatus,
});

const toolCallReplayScenariosInput = [
  toolCallScenario({
    scenarioId: "replay.tool-call.malformed-arguments",
    scenarioKind: "malformed_tool_arguments",
    capture: manualCapture(
      "replay.tool-call.malformed-arguments",
      [
        promptRecord("replay.tool-call.malformed-arguments", "Search for TODO markers in the repo and summarize the matches."),
        routeRecord("replay.tool-call.malformed-arguments"),
        {
          recordId: "record.replay.tool-call.malformed-arguments.tool",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.malformed-arguments.route"],
          toolCallId: "tool.replay.tool-call.malformed-arguments.exec",
          namespace: "functions",
          name: "exec_command",
          status: "malformed_args",
          args: {
            cmd: ["rg", "TODO"],
            cwd: "/workspace",
          },
          result: {
            error: "cmd must be a string",
            receivedShape: "array",
          },
          resultStyle: "structured_error",
          retryCount: 0,
          redactionStatus: "redacted",
          errorCode: "invalid_tool_args",
          artifactRefs: ["artifact:malformed-arguments.validation"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.malformed-arguments",
              spanId: "span.replay.tool-call.malformed-arguments.tool",
              parentSpanId: "span.replay.tool-call.malformed-arguments.route",
            },
          ],
        },
      ],
      "train",
    ),
    metadata: {
      evalCaseId: "replay.eval.tool-call.malformed-arguments",
      title: "Malformed tool arguments stay structured",
      split: "train",
      splitRationale: "Synthetic visible fixture for tool argument validation feedback.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The replay should preserve malformed tool-call arguments as structured telemetry without leaking raw private output.",
        assertions: [
          {
            assertionId: "assert.tool.malformed.failure-kind",
            assertionKind: "json_pointer_equals",
            description: "Malformed arguments are visible as a tool-call failure.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/failureKind",
            expected: "tool_call",
          },
          {
            assertionId: "assert.tool.malformed.error-code",
            assertionKind: "json_pointer_equals",
            description: "The argument validation error code is preserved.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "invalid_tool_args",
          },
        ],
      },
      sourceRefs: [sourceRef("replay.tool-call.malformed-arguments", "schema-validation")],
      tags: ["replay", "tool-call", "malformed-arguments", "train"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  toolCallScenario({
    scenarioId: "replay.tool-call.oversized-output",
    scenarioKind: "oversized_output",
    capture: manualCapture(
      "replay.tool-call.oversized-output",
      [
        promptRecord("replay.tool-call.oversized-output", "List generated bundle files but keep only the useful summary."),
        routeRecord("replay.tool-call.oversized-output"),
        {
          recordId: "record.replay.tool-call.oversized-output.tool",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.oversized-output.route"],
          toolCallId: "tool.replay.tool-call.oversized-output.rg",
          namespace: "functions",
          name: "exec_command",
          status: "truncated",
          args: {
            cmd: "rg --files dist",
            cwd: "/workspace",
          },
          result: {
            artifactRef: "artifact:oversized-output.full",
            originalBytes: 7340032,
            visibleBytes: 4096,
            truncated: true,
          },
          resultStyle: "artifact_ref",
          retryCount: 0,
          redactionStatus: "hash_only",
          errorCode: "tool_output_oversized",
          artifactRefs: ["artifact:oversized-output.full", "artifact:oversized-output.preview"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.oversized-output",
              spanId: "span.replay.tool-call.oversized-output.tool",
              parentSpanId: "span.replay.tool-call.oversized-output.route",
            },
          ],
        },
        {
          recordId: "record.replay.tool-call.oversized-output.artifact",
          recordKind: "artifact_ref",
          parentRecordIds: ["record.replay.tool-call.oversized-output.tool"],
          artifactRef: "artifact:oversized-output.preview",
          artifactRefs: ["artifact:oversized-output.preview"],
          artifactKind: "tool_output",
          path: "artifacts/replay/tool-call/oversized-output.preview.txt",
          contentHash: "sha256:oversized-output.preview",
          redactionStatus: "hash_only",
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.oversized-output",
              spanId: "span.replay.tool-call.oversized-output.artifact",
              parentSpanId: "span.replay.tool-call.oversized-output.tool",
            },
          ],
        },
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.tool-call.oversized-output",
      title: "Oversized tool output is artifact-backed",
      split: "dev",
      splitRationale: "Visible dev fixture covers output-size control and artifact lineage.",
      oracleStrength: "medium",
      expectedBehavior: {
        summary: "The replay should expose that output was too large, keep a bounded preview, and link the redacted artifact.",
        assertions: [
          {
            assertionId: "assert.tool.oversized.status",
            assertionKind: "json_pointer_equals",
            description: "Oversized output is recorded as a visible tool-call failure.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/status",
            expected: "truncated",
          },
          {
            assertionId: "assert.tool.oversized.artifact",
            assertionKind: "json_pointer_equals",
            description: "The oversized output artifact is preserved in observed failure metadata.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/artifactRefs/0",
            expected: "artifact:oversized-output.full",
          },
        ],
      },
      sourceRefs: [sourceRef("replay.tool-call.oversized-output", "preview", "hash_only")],
      tags: ["replay", "tool-call", "oversized-output", "artifact-ref", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  toolCallScenario({
    scenarioId: "replay.tool-call.permission-denial",
    scenarioKind: "permission_denial",
    capture: manualCapture(
      "replay.tool-call.permission-denial",
      [
        promptRecord("replay.tool-call.permission-denial", "Check whether the protected deployment secret exists."),
        routeRecord("replay.tool-call.permission-denial", "safe", "no_side_effects"),
        {
          recordId: "record.replay.tool-call.permission-denial.tool",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.permission-denial.route"],
          toolCallId: "tool.replay.tool-call.permission-denial.read",
          namespace: "functions",
          name: "exec_command",
          status: "permission_denied",
          args: {
            cmd: "cat .env.production",
            cwd: "/workspace",
          },
          result: {
            policy: "protected_path",
            protectedPath: ".env.production",
          },
          resultStyle: "structured_error",
          retryCount: 0,
          redactionStatus: "redacted",
          errorCode: "protected_secret_read_denied",
          artifactRefs: ["artifact:permission-denial.policy"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.permission-denial",
              spanId: "span.replay.tool-call.permission-denial.tool",
              parentSpanId: "span.replay.tool-call.permission-denial.route",
            },
          ],
        },
      ],
      "train",
    ),
    metadata: {
      evalCaseId: "replay.eval.tool-call.permission-denial",
      title: "Permission denial is replay-visible",
      split: "train",
      splitRationale: "Synthetic visible fixture for policy denial handling.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The agent should preserve the permission denial and avoid retrying the protected operation.",
        assertions: [
          {
            assertionId: "assert.tool.permission.status",
            assertionKind: "json_pointer_equals",
            description: "The denied tool call is visible as permission_denied.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/status",
            expected: "permission_denied",
          },
          {
            assertionId: "assert.tool.permission.routing",
            assertionKind: "json_pointer_equals",
            description: "Safe routing remains no-side-effects.",
            artifact: "telemetry",
            pointer: "/routing/sideEffectPolicy",
            expected: "no_side_effects",
          },
        ],
      },
      sourceRefs: [sourceRef("replay.tool-call.permission-denial", "policy")],
      tags: ["replay", "tool-call", "permission", "safe", "train"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  toolCallScenario({
    scenarioId: "replay.tool-call.retry-behavior",
    scenarioKind: "retry_behavior",
    capture: manualCapture(
      "replay.tool-call.retry-behavior",
      [
        promptRecord("replay.tool-call.retry-behavior", "Run the focused package test and retry once if the test runner flakes."),
        routeRecord("replay.tool-call.retry-behavior", "read_only", "terminal_allowed"),
        {
          recordId: "record.replay.tool-call.retry-behavior.first",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.retry-behavior.route"],
          toolCallId: "tool.replay.tool-call.retry-behavior.first",
          namespace: "functions",
          name: "exec_command",
          status: "timed_out",
          args: {
            cmd: "bun test tests/replay-tool-call-scenarios.test.ts",
            cwd: "/workspace",
          },
          result: {
            elapsedMs: 120000,
            timedOut: true,
          },
          resultStyle: "structured_error",
          retryCount: 0,
          redactionStatus: "hash_only",
          errorCode: "tool_timeout",
          artifactRefs: ["artifact:retry-behavior.timeout"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.retry-behavior",
              spanId: "span.replay.tool-call.retry-behavior.first",
              parentSpanId: "span.replay.tool-call.retry-behavior.route",
            },
          ],
        },
        {
          recordId: "record.replay.tool-call.retry-behavior.retry",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.retry-behavior.first"],
          toolCallId: "tool.replay.tool-call.retry-behavior.retry",
          namespace: "functions",
          name: "exec_command",
          status: "succeeded",
          args: {
            cmd: "bun test tests/replay-tool-call-scenarios.test.ts",
            cwd: "/workspace",
          },
          result: {
            exitCode: 0,
            retryOf: "tool.replay.tool-call.retry-behavior.first",
          },
          resultStyle: "json",
          retryCount: 1,
          redactionStatus: "hash_only",
          artifactRefs: ["artifact:retry-behavior.success"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.retry-behavior",
              spanId: "span.replay.tool-call.retry-behavior.retry",
              parentSpanId: "span.replay.tool-call.retry-behavior.first",
            },
          ],
        },
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.tool-call.retry-behavior",
      title: "Retry preserves first failure",
      split: "dev",
      splitRationale: "Visible dev fixture for retry telemetry and failure provenance.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The replay should retain the failed first attempt and the bounded successful retry with retry count metadata.",
        assertions: [
          {
            assertionId: "assert.tool.retry.failure-preserved",
            assertionKind: "json_pointer_equals",
            description: "The first timed-out attempt remains an observed failure.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "tool_timeout",
          },
          {
            assertionId: "assert.tool.retry.success-retry-count",
            assertionKind: "json_pointer_equals",
            description: "The successful retry carries retry count metadata.",
            artifact: "telemetry",
            pointer: "/records/retryCount",
            expected: 1,
          },
        ],
        notes: ["Preserve failed attempt evidence even when a retry later succeeds."],
      },
      sourceRefs: [
        sourceRef("replay.tool-call.retry-behavior", "timeout", "hash_only"),
        sourceRef("replay.tool-call.retry-behavior", "success", "hash_only"),
      ],
      tags: ["replay", "tool-call", "retry", "timeout", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  toolCallScenario({
    scenarioId: "replay.tool-call.truncation-visibility",
    scenarioKind: "truncation_visibility",
    capture: manualCapture(
      "replay.tool-call.truncation-visibility",
      [
        promptRecord("replay.tool-call.truncation-visibility", "Inspect the long benchmark log and call out that the visible output is incomplete."),
        routeRecord("replay.tool-call.truncation-visibility"),
        {
          recordId: "record.replay.tool-call.truncation-visibility.tool",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.truncation-visibility.route"],
          toolCallId: "tool.replay.tool-call.truncation-visibility.log",
          namespace: "functions",
          name: "exec_command",
          status: "truncated",
          args: {
            cmd: "tail -n 10000 artifacts/benchmark.log",
            cwd: "/workspace",
          },
          result: {
            visibleHeadBytes: 2048,
            visibleTailBytes: 2048,
            omittedMiddleBytes: 98304,
            truncationNoticeVisible: true,
          },
          resultStyle: "json",
          retryCount: 0,
          redactionStatus: "hash_only",
          errorCode: "visible_output_truncated",
          artifactRefs: ["artifact:truncation-visibility.log-window"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.truncation-visibility",
              spanId: "span.replay.tool-call.truncation-visibility.tool",
              parentSpanId: "span.replay.tool-call.truncation-visibility.route",
            },
          ],
        },
      ],
      "train",
    ),
    metadata: {
      evalCaseId: "replay.eval.tool-call.truncation-visibility",
      title: "Truncation notice remains visible",
      split: "train",
      splitRationale: "Synthetic visible fixture for model behavior around partial tool output.",
      oracleStrength: "medium",
      expectedBehavior: {
        summary: "The agent should acknowledge partial output and avoid presenting truncated logs as complete evidence.",
        assertions: [
          {
            assertionId: "assert.tool.truncation.status",
            assertionKind: "json_pointer_equals",
            description: "The tool call records explicit truncation.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/status",
            expected: "truncated",
          },
          {
            assertionId: "assert.tool.truncation.error-code",
            assertionKind: "json_pointer_equals",
            description: "The truncation visibility error code is preserved.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "visible_output_truncated",
          },
        ],
      },
      sourceRefs: [sourceRef("replay.tool-call.truncation-visibility", "log-window", "hash_only")],
      tags: ["replay", "tool-call", "truncation", "train"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  toolCallScenario({
    scenarioId: "replay.tool-call.mcp-call",
    scenarioKind: "mcp_call",
    capture: manualCapture(
      "replay.tool-call.mcp-call",
      [
        promptRecord("replay.tool-call.mcp-call", "Read the project note through the MCP filesystem server and summarize it."),
        routeRecord("replay.tool-call.mcp-call"),
        {
          recordId: "record.replay.tool-call.mcp-call.tool",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.mcp-call.route"],
          toolCallId: "tool.replay.tool-call.mcp-call.read",
          namespace: "mcp.filesystem",
          name: "read_file",
          status: "failed",
          args: {
            serverId: "mcp.synthetic.filesystem",
            method: "tools/call",
            toolName: "read_file",
            path: "docs/project-note.md",
          },
          result: {
            requestId: "mcp-request.replay.tool-call.mcp-call.read",
            mcpServerId: "mcp.synthetic.filesystem",
            error: "MCP server returned not_found for docs/project-note.md",
          },
          resultStyle: "structured_error",
          retryCount: 0,
          redactionStatus: "hash_only",
          errorCode: "mcp_tool_not_found",
          artifactRefs: ["artifact:mcp-call.project-note"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.mcp-call",
              spanId: "span.replay.tool-call.mcp-call.mcp-read",
              parentSpanId: "span.replay.tool-call.mcp-call.route",
            },
          ],
        },
        {
          recordId: "record.replay.tool-call.mcp-call.trace",
          recordKind: "artifact_ref",
          parentRecordIds: ["record.replay.tool-call.mcp-call.tool"],
          artifactRef: "artifact:mcp-call.trace",
          artifactRefs: ["artifact:mcp-call.trace"],
          artifactKind: "trace",
          path: "mcp://filesystem/read_file/docs/project-note.md",
          contentHash: "sha256:mcp-call.trace",
          redactionStatus: "hash_only",
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.mcp-call",
              spanId: "span.replay.tool-call.mcp-call.trace",
              parentSpanId: "span.replay.tool-call.mcp-call.mcp-read",
            },
          ],
        },
      ],
      "holdout",
    ),
    metadata: {
      evalCaseId: "replay.eval.tool-call.mcp-call",
      title: "MCP tool-call failure lineage is preserved",
      split: "holdout",
      splitRationale: "Synthetic hidden holdout fixture for MCP lineage; excluded from optimizer feedback by default.",
      oracleStrength: "medium",
      expectedBehavior: {
        summary: "The replay should preserve MCP server, tool, trace, failure class, and artifact lineage without exposing raw MCP content.",
        assertions: [
          {
            assertionId: "assert.tool.mcp.split",
            assertionKind: "json_pointer_equals",
            description: "The MCP case remains hidden holdout.",
            artifact: "telemetry",
            pointer: "/split",
            expected: "holdout",
          },
          {
            assertionId: "assert.tool.mcp.failure-visible",
            assertionKind: "json_pointer_equals",
            description: "The failed MCP read is extracted as observed tool-call failure evidence.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/errorCode",
            expected: "mcp_tool_not_found",
          },
        ],
      },
      sourceRefs: [sourceRef("replay.tool-call.mcp-call", "mcp-trace", "hash_only")],
      tags: ["replay", "tool-call", "mcp", "holdout"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
  toolCallScenario({
    scenarioId: "replay.tool-call.terminal-verification-enforcement",
    scenarioKind: "terminal_verification_enforcement",
    capture: manualCapture(
      "replay.tool-call.terminal-verification-enforcement",
      [
        promptRecord("replay.tool-call.terminal-verification-enforcement", "After changing the replay pack, run the focused test and typecheck before reporting done."),
        routeRecord("replay.tool-call.terminal-verification-enforcement", "mutating", "terminal_allowed"),
        {
          recordId: "record.replay.tool-call.terminal-verification-enforcement.tool",
          recordKind: "tool_call",
          parentRecordIds: ["record.replay.tool-call.terminal-verification-enforcement.route"],
          toolCallId: "tool.replay.tool-call.terminal-verification-enforcement.apply",
          namespace: "functions",
          name: "apply_patch",
          status: "succeeded",
          args: {
            patchArtifactRef: "artifact:terminal-verification.patch",
          },
          result: {
            changedFiles: ["src/replay/tool-call-scenarios.ts"],
          },
          resultStyle: "json",
          retryCount: 0,
          redactionStatus: "redacted",
          artifactRefs: ["artifact:terminal-verification.patch"],
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.terminal-verification-enforcement",
              spanId: "span.replay.tool-call.terminal-verification-enforcement.tool",
              parentSpanId: "span.replay.tool-call.terminal-verification-enforcement.route",
            },
          ],
        },
        {
          recordId: "record.replay.tool-call.terminal-verification-enforcement.command",
          recordKind: "terminal_command",
          parentRecordIds: ["record.replay.tool-call.terminal-verification-enforcement.tool"],
          commandId: "terminal.replay.tool-call.terminal-verification-enforcement.typecheck",
          command: ["npm", "run", "typecheck"],
          cwd: "/workspace",
          status: "failed",
          exitCode: 2,
          stdoutArtifactRef: "artifact:terminal-verification.stdout",
          stderrArtifactRef: "artifact:terminal-verification.stderr",
          redactionStatus: "hash_only",
          errorCode: "verification_failed_after_tool_call",
          traceRefs: [
            {
              traceId: "trace.replay.tool-call.terminal-verification-enforcement",
              spanId: "span.replay.tool-call.terminal-verification-enforcement.command",
              parentSpanId: "span.replay.tool-call.terminal-verification-enforcement.tool",
            },
          ],
        },
      ],
      "dev",
    ),
    metadata: {
      evalCaseId: "replay.eval.tool-call.terminal-verification-enforcement",
      title: "Terminal verification failure blocks completion",
      split: "dev",
      splitRationale: "Visible dev fixture for terminal verification enforcement after tool changes.",
      oracleStrength: "strong",
      expectedBehavior: {
        summary: "The replay should require terminal verification evidence and preserve failed verification output artifacts.",
        assertions: [
          {
            assertionId: "assert.tool.verify.failure-kind",
            assertionKind: "json_pointer_equals",
            description: "The failed verification command is an observed terminal failure.",
            artifact: "telemetry",
            pointer: "/observedFailures/0/failureKind",
            expected: "terminal_command",
          },
          {
            assertionId: "assert.tool.verify.command-exit",
            assertionKind: "command_exit_code",
            description: "Typecheck failure is preserved as the blocking verification result.",
            commandId: "terminal.replay.tool-call.terminal-verification-enforcement.typecheck",
            expectedExitCode: 2,
          },
        ],
      },
      sourceRefs: [sourceRef("replay.tool-call.terminal-verification-enforcement", "stderr", "hash_only")],
      tags: ["replay", "tool-call", "terminal-verification", "dev"],
      timeoutMs: DEFAULT_TIMEOUT_MS,
    },
  }),
] satisfies ReplayToolCallScenario[];

export const toolCallReplayScenarios: ReplayToolCallScenario[] = toolCallReplayScenariosInput;

export const toolCallReplayScenarioSkeletons: ReplayEvalCaseSkeleton[] =
  toolCallReplayScenarios.map((scenario) => extractReplayEvalCaseSkeleton(scenario));

export const visibleToolCallReplayScenariosForOptimization = (): ReplayToolCallScenario[] =>
  toolCallReplayScenarios.filter((scenario) => scenario.optimizationAllowed && scenario.split !== "holdout");

export const extractToolCallReplayScenarioSkeletons = (
  scenarios: readonly ReplayToolCallScenario[] = toolCallReplayScenarios,
): ReplayEvalCaseSkeleton[] =>
  scenarios.map((scenario) => extractReplayEvalCaseSkeleton({
    capture: scenario.capture,
    metadata: scenario.metadata,
  }));

export const toolCallReplayCaptures: AcpReplayCapture[] =
  toolCallReplayScenarios.map((scenario) => scenario.capture);
