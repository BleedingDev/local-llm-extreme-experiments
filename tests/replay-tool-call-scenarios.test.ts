import { describe, expect, test } from "bun:test";
import {
  extractToolCallReplayScenarioSkeletons,
  toolCallReplayCaptures,
  toolCallReplayScenarioSkeletons,
  toolCallReplayScenarios,
  visibleToolCallReplayScenariosForOptimization,
} from "../src/replay";

describe("tool-call replay scenarios", () => {
  test("defines the tool-call scenario pack with redaction-safe split-aware captures", () => {
    expect(toolCallReplayScenarios.map((scenario) => scenario.scenarioKind)).toEqual([
      "malformed_tool_arguments",
      "oversized_output",
      "permission_denial",
      "retry_behavior",
      "truncation_visibility",
      "mcp_call",
      "terminal_verification_enforcement",
    ]);
    expect(toolCallReplayCaptures).toHaveLength(7);
    expect(toolCallReplayScenarios.every((scenario) => scenario.capture.redactionStatus === "redacted")).toBe(true);
    expect(toolCallReplayScenarios.every((scenario) => scenario.capture.defaultSplitHint === scenario.split)).toBe(true);
    expect(toolCallReplayScenarios.every((scenario) => scenario.metadata.expectedBehavior.assertions.length > 0))
      .toBe(true);
    expect(toolCallReplayScenarios.every((scenario) => scenario.metadata.sourceRefs.every(
      (sourceRef) => sourceRef.redactionStatus !== "raw_local_only",
    ))).toBe(true);
  });

  test("extracts observed failures for tool and terminal failure records", () => {
    const failuresByKind = new Map(toolCallReplayScenarioSkeletons.map((skeleton) => {
      const scenario = toolCallReplayScenarios.find((candidate) => candidate.metadata.evalCaseId === skeleton.evalCaseId);
      return [scenario?.scenarioKind, skeleton.observedFailures];
    }));

    expect(failuresByKind.get("malformed_tool_arguments")?.[0]).toMatchObject({
      failureKind: "tool_call",
      status: "malformed_args",
      errorCode: "invalid_tool_args",
    });
    expect(failuresByKind.get("oversized_output")?.[0]).toMatchObject({
      failureKind: "tool_call",
      status: "truncated",
      errorCode: "tool_output_oversized",
    });
    expect(failuresByKind.get("permission_denial")?.[0]).toMatchObject({
      failureKind: "tool_call",
      status: "permission_denied",
      errorCode: "protected_secret_read_denied",
    });
    expect(failuresByKind.get("truncation_visibility")?.[0]).toMatchObject({
      failureKind: "tool_call",
      status: "truncated",
      errorCode: "visible_output_truncated",
    });
    expect(failuresByKind.get("terminal_verification_enforcement")?.[0]).toMatchObject({
      failureKind: "terminal_command",
      status: "failed",
      errorCode: "verification_failed_after_tool_call",
    });
    expect(failuresByKind.get("mcp_call")?.[0]).toMatchObject({
      failureKind: "tool_call",
      status: "failed",
      errorCode: "mcp_tool_not_found",
    });
  });

  test("preserves retry metadata while retaining the failed first attempt", () => {
    const scenario = toolCallReplayScenarios.find((candidate) => candidate.scenarioKind === "retry_behavior");
    expect(scenario).toBeDefined();
    if (scenario == null) {
      throw new Error("retry behavior scenario is missing");
    }

    const toolCalls = scenario.capture.records.filter((record) => record.recordKind === "tool_call");
    expect(toolCalls.map((record) => [record.status, record.retryCount])).toEqual([
      ["timed_out", 0],
      ["succeeded", 1],
    ]);
    expect(toolCalls[1]?.parentRecordIds).toEqual(["record.replay.tool-call.retry-behavior.first"]);

    const skeleton = extractToolCallReplayScenarioSkeletons([scenario])[0];
    expect(skeleton).toBeDefined();
    if (skeleton == null) {
      throw new Error("retry behavior skeleton is missing");
    }

    expect(skeleton.observedFailures).toEqual([
      expect.objectContaining({
        recordId: "record.replay.tool-call.retry-behavior.first",
        status: "timed_out",
        errorCode: "tool_timeout",
      }),
    ]);
    expect(skeleton.oracle.expectedBehavior.notes).toContain(
      "Preserve failed attempt evidence even when a retry later succeeds.",
    );
  });

  test("preserves truncation, permission, source refs, and oracle fields through extraction", () => {
    const truncationScenario = toolCallReplayScenarios.find(
      (scenario) => scenario.scenarioKind === "truncation_visibility",
    );
    expect(truncationScenario).toBeDefined();
    if (truncationScenario == null) {
      throw new Error("truncation visibility scenario is missing");
    }

    const skeleton = extractToolCallReplayScenarioSkeletons([truncationScenario])[0];
    expect(skeleton).toBeDefined();
    if (skeleton == null) {
      throw new Error("truncation visibility skeleton is missing");
    }

    expect(skeleton).toMatchObject({
      evalCaseId: "replay.eval.tool-call.truncation-visibility",
      split: "train",
      captureId: "capture.replay.tool-call.truncation-visibility",
      sourceSessionId: "session.replay.tool-call.truncation-visibility",
      oracle: {
        strength: "medium",
      },
    });
    expect(skeleton.observedFailures[0]).toMatchObject({
      status: "truncated",
      errorCode: "visible_output_truncated",
      artifactRefs: ["artifact:truncation-visibility.log-window"],
    });
    expect(skeleton.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "fixture",
      path: "synthetic://replay/tool-call/replay.tool-call.truncation-visibility/log-window",
      redactionStatus: "hash_only",
    }));
    expect(skeleton.oracle.expectedBehavior.assertions.map((assertion) => assertion.assertionId)).toContain(
      "assert.tool.truncation.error-code",
    );

    const permissionScenario = toolCallReplayScenarios.find((scenario) => scenario.scenarioKind === "permission_denial");
    expect(permissionScenario?.capture.records.find((record) => record.recordKind === "tool_call")).toMatchObject({
      status: "permission_denied",
      retryCount: 0,
      redactionStatus: "redacted",
    });
  });

  test("preserves MCP lineage and keeps hidden holdout out of optimization-visible scenarios", () => {
    const visibleScenarios = visibleToolCallReplayScenariosForOptimization();
    expect(visibleScenarios.map((scenario) => scenario.scenarioKind)).not.toContain("mcp_call");
    expect(visibleScenarios.every((scenario) => scenario.split !== "holdout")).toBe(true);
    expect(visibleScenarios.every((scenario) => scenario.optimizationAllowed)).toBe(true);

    const mcpScenario = toolCallReplayScenarios.find((scenario) => scenario.scenarioKind === "mcp_call");
    expect(mcpScenario).toBeDefined();
    if (mcpScenario == null) {
      throw new Error("MCP scenario is missing");
    }

    const mcpToolCall = mcpScenario.capture.records.find((record) => record.recordKind === "tool_call");
    expect(mcpToolCall).toMatchObject({
      namespace: "mcp.filesystem",
      name: "read_file",
      status: "failed",
      redactionStatus: "hash_only",
      errorCode: "mcp_tool_not_found",
    });
    expect(mcpToolCall?.traceRefs[0]).toMatchObject({
      traceId: "trace.replay.tool-call.mcp-call",
      spanId: "span.replay.tool-call.mcp-call.mcp-read",
      parentSpanId: "span.replay.tool-call.mcp-call.route",
    });

    const mcpSkeleton = toolCallReplayScenarioSkeletons.find(
      (skeleton) => skeleton.evalCaseId === "replay.eval.tool-call.mcp-call",
    );
    expect(mcpSkeleton).toBeDefined();
    if (mcpSkeleton == null) {
      throw new Error("MCP skeleton is missing");
    }

    expect(mcpSkeleton.split).toBe("holdout");
    expect(mcpSkeleton.observedFailures).toEqual([
      expect.objectContaining({
        failureKind: "tool_call",
        recordId: "record.replay.tool-call.mcp-call.tool",
        status: "failed",
        errorCode: "mcp_tool_not_found",
      }),
    ]);
    expect(mcpSkeleton.sourceTraceIds).toEqual(["trace.replay.tool-call.mcp-call"]);
    expect(mcpSkeleton.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "artifact",
      artifactRef: "artifact:mcp-call.trace",
      redactionStatus: "hash_only",
    }));
    expect(mcpSkeleton.sourceRefs).toContainEqual(expect.objectContaining({
      sourceKind: "span",
      traceId: "trace.replay.tool-call.mcp-call",
      spanId: "span.replay.tool-call.mcp-call.mcp-read",
      redactionStatus: "hash_only",
    }));
  });
});
