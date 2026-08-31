import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BleedingAcpAgent, acpConsumerCompatibilityMatrix, readAcpSettingsSnippet, readAcpZedSettingsSnippet } from "../src/acp-agent";
import { resolveLiveEditContext } from "../src/acp/edit-routing";
import { editAttemptFromParseFailure } from "../src/acp/edit-telemetry";
import { inspectBackgroundOptimizationTrigger } from "../src/acp/maintenance";
import { buildCodingReplayCapture } from "../src/acp/replay-capture";
import { defaultConfig } from "../src/config";
import { summarizeMetricsStore } from "../src/metrics";
import { optimizerRegistryCheckpointsDir, saveOptimizerRegistryRecord } from "../src/optimizer/registry";
import type { CandidatePatch, OptimizerRegistryRecord } from "../src/optimizer/types";
import { extractReplayEvalCaseSkeleton, groupAcpReplayRecords, type AcpReplayCapture } from "../src/replay";
import { applySelfOptimization, generateSelfOptimization } from "../src/self-optimize";
import { deterministicSelfEvaluation, RunTelemetry } from "../src/telemetry";
import { analyzeHaloSpans, readHaloSpans, renderTraceAnalysisMarkdown } from "../src/trace-analysis";
import { buildTraceIndex, TraceStore } from "../src/trace-store";
import {
  createHeadlessAcpClientFixture,
  HEADLESS_ACP_REGRESSION_SCENARIOS,
  headlessAcpConsumerCapabilityProfile,
  recordHeadlessAcpProtocolCall,
  summarizeHeadlessAcpTranscript,
  type HeadlessAcpRegressionScenarioId,
  type TrajectoryEntry,
} from "../scripts/bag_acp_run";
import {
  previewAndWriteClientEditThroughAgentForTest,
  recordFinalEditLifecycleTelemetryThroughAgentForTest,
  readClientFileThroughAgentForTest,
  replaceDecidePromptRouteForTest,
  replaceRunAcpToolForTest,
  replaceRunCodingTurnForTest,
  replaceRunPlanningTurnForTest,
  requireAgentSessionForTest,
  rollbackLiveEditsThroughAgentForTest,
  runTerminalCommandThroughAgentForTest,
  telemetryForAgentSession,
  writeClientFileThroughAgentForTest,
} from "./acp-agent-test-harness";

describe("BleedingAgent config", () => {
  test("uses measured local executor defaults", () => {
    const config = defaultConfig();

    expect(config.policy.executorConcurrency).toBe(16);
    expect(config.policy.maxExecutorConcurrency).toBe(24);
    expect(config.local.baseUrl).toBe("http://127.0.0.1:18082/v1");
  });
});

describe("BleedingAgent metrics summary", () => {
  test("summarizes step and LLM call telemetry", () => {
    const summary = summarizeMetricsStore({
      run_1: {
        steps: [
          {
            step: "context.scout",
            startedAt: "2026-04-29T00:00:00.000Z",
            completedAt: "2026-04-29T00:00:01.000Z",
            durationMs: 1000,
            ok: true,
            modelRole: "local",
          },
        ],
        llmCalls: [
          {
            role: "local",
            model: "local-model",
            endpoint: "http://127.0.0.1:18082/v1/chat/completions",
            startedAt: "2026-04-29T00:00:00.000Z",
            completedAt: "2026-04-29T00:00:01.000Z",
            durationMs: 1000,
            ok: true,
            totalTokens: 42,
          },
        ],
        toolCalls: [
          {
            toolName: "repo.read",
            namespace: "workspace",
            descriptionVersion: "v1",
            startedAt: "2026-04-29T00:00:00.000Z",
            completedAt: "2026-04-29T00:00:01.000Z",
            durationMs: 1000,
            ok: false,
            retryCount: 1,
            argumentBytes: 12,
            argumentHash: "abc",
            error: "file missing",
            errorName: "NotFoundError",
          },
        ],
      },
    });

    expect(summary).toContain("runs: 1");
    expect(summary).toContain("llmCalls: 1");
    expect(summary).toContain("toolCalls: 1 failed=1");
    expect(summary).toContain("total=42");
  });
});

describe("BleedingAgent self-evaluation", () => {
  test("passes complete successful runs", () => {
    const evaluation = deterministicSelfEvaluation({
      threshold: 0.78,
      artifactCount: 8,
      toolMetrics: [
        {
          toolName: "repo.read",
          startedAt: "2026-04-29T00:00:00.000Z",
          completedAt: "2026-04-29T00:00:01.000Z",
          durationMs: 1000,
          ok: true,
          retryCount: 0,
          argumentBytes: 12,
          argumentHash: "abc",
          resultBytes: 24,
          resultKind: "text",
        },
      ],
      metrics: [
        {
          step: "interview",
          startedAt: "2026-04-29T00:00:00.000Z",
          completedAt: "2026-04-29T00:00:01.000Z",
          durationMs: 1000,
          ok: true,
          modelRole: "master",
        },
        {
          step: "dag.generate",
          startedAt: "2026-04-29T00:00:01.000Z",
          completedAt: "2026-04-29T00:00:02.000Z",
          durationMs: 1000,
          ok: true,
          modelRole: "master",
        },
      ],
    });

    expect(evaluation.passed).toBe(true);
    expect(evaluation.score).toBeGreaterThanOrEqual(0.78);
  });
});

describe("BleedingAgent HALO-style tracing", () => {
  test("writes OpenInference-shaped spans for steps, LLM calls, and tool calls", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-spans-"));
    const config = defaultConfig();
    const telemetry = new RunTelemetry(config, "run-spans", cwd, {
      modelRole: "local",
      providerConfigRole: "local",
      provider: "openai-compatible",
      endpointKind: "chat_completions",
      modelServerId: "server.local.test",
      modelServerProfileId: "server-profile.local.test",
      providerDiscoverySource: "configured",
      contextWindowTokens: 8192,
      maxOutputTokens: 2048,
      modelProfileId: "model.local.test",
      codebaseProfileId: "codebase.test",
      policyId: "policy.test",
      canonicalToolVersion: "canonical-tools.test",
      renderedToolVersion: "rendered-tools.test",
      resultStyleVersion: "result-style.test",
      verificationPolicyVersion: "verification.test",
      editStrategyVersion: "edit-strategy.test",
      renderedEditContractVersion: "rendered-edit-contract.test",
      editFallbackPolicyVersion: "edit-fallback.test",
      editRepairPolicyVersion: "edit-repair.test",
      editVerifierPolicyVersion: "edit-verifier.test",
      editObjectiveSetId: "edit-objectives.test",
      source: "seed",
      registryRoot: join(cwd, ".bag", "optimizer"),
      rendererId: "renderer.default",
      rendererVersion: "renderer.v1",
    });

    await telemetry.measure("context.scout", "local", async () => "ok");
    telemetry.recordLlmCall({
      role: "local",
      model: "local-model",
      endpoint: "http://127.0.0.1:18082/v1/chat/completions",
      startedAt: "2026-04-29T00:00:00.000Z",
      completedAt: "2026-04-29T00:00:01.000Z",
      durationMs: 1000,
      ok: true,
      totalTokens: 42,
    });
    await telemetry.measureToolCall({
      toolName: "repo.read",
      namespace: "workspace",
      args: { path: "README.md" },
      fn: async () => "hello",
    });

    const spans = readHaloSpans(config, cwd);
    const report = analyzeHaloSpans(spans);

    expect(spans.length).toBeGreaterThanOrEqual(4);
    expect(spans.some((span) => span.attributes["inference.observation_kind"] === "LLM")).toBe(true);
    expect(spans.some((span) => span.attributes["tool.name"] === "repo.read")).toBe(true);
    expect(spans.some((span) => span.attributes["optimizer.policy_id"] === "policy.test")).toBe(true);
    expect(spans.some((span) => span.attributes["optimizer.provider"] === "openai-compatible")).toBe(true);
    expect(spans.some((span) => span.attributes["optimizer.model_server_profile_id"] === "server-profile.local.test"))
      .toBe(true);
    expect(spans.some((span) => span.attributes["optimizer.rendered_tool_version"] === "rendered-tools.test")).toBe(true);
    expect(spans.some((span) => span.attributes["optimizer.edit_strategy_version"] === "edit-strategy.test")).toBe(true);
    expect(JSON.stringify(spans)).toContain('"trace_id"');
    expect(report.observationKinds.TOOL).toBe(1);
    expect(report.optimizerDimensions.modelProfileIds).toContain("model.local.test");
    expect(report.optimizerDimensions.codebaseProfileIds).toContain("codebase.test");
    expect(report.optimizerDimensions.policyIds).toContain("policy.test");
    expect(report.optimizerDimensions.canonicalToolVersions).toContain("canonical-tools.test");
    expect(report.optimizerDimensions.renderedToolVersions).toContain("rendered-tools.test");
    expect(report.optimizerDimensions.resultStyleVersions).toContain("result-style.test");
    expect(report.optimizerDimensions.verificationPolicyVersions).toContain("verification.test");
    expect(report.optimizerDimensions.editStrategyVersions).toContain("edit-strategy.test");
    expect(report.optimizerDimensions.renderedEditContractVersions).toContain("rendered-edit-contract.test");
    expect(report.optimizerDimensions.editObjectiveSetIds).toContain("edit-objectives.test");
    expect(renderTraceAnalysisMarkdown(report)).toContain("policyId: policy.test");
    expect(renderTraceAnalysisMarkdown(report)).toContain("editStrategyVersion: edit-strategy.test");

    const metrics = JSON.parse(readFileSync(join(cwd, ".bag", "telemetry", "metrics.json"), "utf8")) as {
      "run-spans"?: { optimizerPin?: { policyId?: string; canonicalToolVersion?: string } };
    };
    expect(metrics["run-spans"]?.optimizerPin?.policyId).toBe("policy.test");
    expect(metrics["run-spans"]?.optimizerPin?.canonicalToolVersion).toBe("canonical-tools.test");
  });

  test("indexes and queries traces with bounded view/search tools", () => {
    const config = defaultConfig();
    const cwd = mkdtempSync(join(tmpdir(), "bag-trace-store-"));
    mkdirSync(join(cwd, ".bag", "telemetry"), { recursive: true });
    const noisyProjectionPayload = "NOISY-FLAT-PROJECTION-CONTENT ".repeat(1000);
    const span = (input: {
      traceId: string;
      spanId: string;
      name: string;
      kind: "LLM" | "TOOL";
      status?: "STATUS_CODE_OK" | "STATUS_CODE_ERROR";
      attrs?: Record<string, unknown>;
    }) => ({
      trace_id: input.traceId,
      span_id: input.spanId,
      parent_span_id: "root",
      trace_state: "",
      name: input.name,
      kind: "SPAN_KIND_CLIENT",
      start_time: "2026-04-29T00:00:00.000Z",
      end_time: "2026-04-29T00:00:01.000Z",
      status: { code: input.status ?? "STATUS_CODE_OK", message: input.status === "STATUS_CODE_ERROR" ? "boom" : "" },
      resource: { attributes: { "service.name": "bleeding-agent" } },
      scope: { name: "bag.telemetry", version: "0.1.0" },
      attributes: {
        "inference.observation_kind": input.kind,
        "inference.project_id": "bleeding-agent",
        "inference.llm.model_name": input.kind === "LLM" ? "local-model" : undefined,
        "tool.name": input.kind === "TOOL" ? "repo.read" : undefined,
        "input.hash": input.kind === "TOOL" ? "hash-a" : undefined,
        "optimizer.model_profile_id": "model.trace.test",
        "optimizer.codebase_profile_id": "codebase.trace.test",
        "optimizer.policy_id": "policy.trace.test",
        "optimizer.canonical_tool_version": "canonical-tools.trace",
        "optimizer.rendered_tool_version": "rendered-tools.trace",
        "optimizer.result_style_version": "result-style.trace",
        "optimizer.verification_policy_version": "verification.trace",
        "optimizer.edit_strategy_version": "edit-strategy.trace",
        "optimizer.rendered_edit_contract_version": "rendered-edit-contract.trace",
        "optimizer.edit_fallback_policy_version": "edit-fallback.trace",
        "optimizer.edit_repair_policy_version": "edit-repair.trace",
        "optimizer.edit_verifier_policy_version": "edit-verifier.trace",
        "optimizer.edit_objective_set_id": "edit-objectives.trace",
        ...input.attrs,
      },
    });
    writeFileSync(
      join(cwd, ".bag", "telemetry", "spans.jsonl"),
      [
        span({ traceId: "trace-a", spanId: "span-a", name: "llm.local.local-model", kind: "LLM" }),
        span({
          traceId: "trace-a",
          spanId: "span-b",
          name: "tool.workspace.repo.read",
          kind: "TOOL",
          status: "STATUS_CODE_ERROR",
          attrs: {
            "error.message": "missing required path",
            "edit.strategy_id": "edit.apply-patch.v1",
            "edit.strategy_family": "apply_patch",
            "edit.canonical_tool_spec_id": "edit-tool.apply-patch.v1",
            "edit.rendered_tool_contract_id": "rendered-edit.apply-patch.qwen36",
            "edit.verification_status": "failed",
            "edit.post_apply_consistency_status": "inconsistent",
            "edit.self_detected_regression_status": "confirmed",
            "edit.rollback_status": "not_attempted",
            "edit.redaction_status": "redacted",
          },
        }),
        span({
          traceId: "trace-a",
          spanId: "span-d",
          name: "tool.workspace.repo.search",
          kind: "TOOL",
          attrs: {
            "event.message": "retry attempt 1 failed",
            "large.payload": "x".repeat(5000),
            "input.value": JSON.stringify({ prompt: "compact prompt preview marker" }),
            "output.value": JSON.stringify({ answer: "compact output preview marker" }),
            "output.hash": "result-hash-a",
            "output.bytes": 1234,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": noisyProjectionPayload,
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "marker_tool_call",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments_json": JSON.stringify({
              payload: noisyProjectionPayload,
            }),
          },
        }),
        span({
          traceId: "trace-a",
          spanId: "span-e",
          name: "tool.workspace.repo.search",
          kind: "TOOL",
          attrs: { "event.message": "retry attempt 2 failed" },
        }),
        span({
          traceId: "trace-a",
          spanId: "span-f",
          name: "tool.workspace.repo.search",
          kind: "TOOL",
          attrs: { "event.message": "retry attempt 3 failed" },
        }),
        span({ traceId: "trace-b", spanId: "span-c", name: "llm.local.local-model", kind: "LLM" }),
      ]
        .map((row) => JSON.stringify(row))
        .join("\n") + "\n",
    );

    const index = buildTraceIndex({ config, cwd });
    const store = TraceStore.open(config, cwd);
    const overview = store.getOverview();
    const errors = store.queryTraces({ hasErrors: true });
    const byPolicy = store.queryTraces({ policyId: "policy.trace.test" });
    const byRenderedTools = store.queryTraces({ renderedToolVersion: "rendered-tools.trace" });
    const byEditStrategy = store.queryTraces({ editStrategyVersion: "edit-strategy.trace" });
    const byBrokenEdit = store.queryTraces({ editPostApplyConsistencyStatus: "inconsistent" });
    const search = store.searchTrace("trace-a", "missing required path");
    const regexSearch = store.searchTrace("trace-a", "retry attempt [0-9] failed", {
      mode: "regex",
      limit: 2,
      contextChars: 8,
    });
    const invalidRegexSearch = store.searchTrace("trace-a", "[", { mode: "regex" });
    const oversizedPatternSearch = store.searchTrace("trace-a", "x".repeat(513), { mode: "regex" });
    const boundedOutputSearch = store.searchTrace("trace-a", "retry attempt 1", { limit: 1 });
    const noisyProjectionSearch = store.searchTrace("trace-a", "NOISY-FLAT-PROJECTION-CONTENT", { limit: 1 });
    const noisyMarkerSearch = store.searchTrace("trace-a", "marker_tool_call", { limit: 1 });
    const spanView = store.viewSpans("trace-a", ["span-b"]);
    const cappedSpanView = store.viewSpans("trace-a", ["span-d", "span-e", "span-f"], { maxRequested: 2, limit: 1 });
    const noisyProjectionView = store.viewSpans("trace-a", ["span-d"]);
    const missingSpanView = store.viewSpans("trace-a", ["span-missing"]);
    const exactSpanSearch = store.searchSpan("trace-a", "span-b", "missing required path", { contextChars: 8 });
    const boundedSpanSearch = store.searchSpan("trace-a", ["span-d", "span-e", "span-f"], "retry attempt [0-9] failed", {
      mode: "regex",
      limit: 1,
      attrCap: 10,
      contextChars: 5,
    });
    const missingSpanSearch = store.searchSpan("trace-a", "span-missing", "missing required path");

    expect(index.meta.traceCount).toBe(2);
    expect(overview.errorTraceCount).toBe(1);
    expect(overview.models).toContain("local-model");
    expect(overview.modelProfileIds).toContain("model.trace.test");
    expect(overview.codebaseProfileIds).toContain("codebase.trace.test");
    expect(overview.policyIds).toContain("policy.trace.test");
    expect(overview.canonicalToolVersions).toContain("canonical-tools.trace");
    expect(overview.renderedToolVersions).toContain("rendered-tools.trace");
    expect(overview.resultStyleVersions).toContain("result-style.trace");
    expect(overview.verificationPolicyVersions).toContain("verification.trace");
    expect(overview.editStrategyVersions).toContain("edit-strategy.trace");
    expect(overview.renderedEditContractVersions).toContain("rendered-edit-contract.trace");
    expect(overview.editObjectiveSetIds).toContain("edit-objectives.trace");
    expect(overview.editStrategyIds).toContain("edit.apply-patch.v1");
    expect(overview.editStrategyFamilies).toContain("apply_patch");
    expect(overview.renderedEditToolContractIds).toContain("rendered-edit.apply-patch.qwen36");
    expect(overview.editPostApplyConsistencyStatuses).toContain("inconsistent");
    expect(errors.total).toBe(1);
    expect(byPolicy.total).toBe(2);
    expect(byPolicy.traces[0]?.policyIds).toContain("policy.trace.test");
    expect(byRenderedTools.total).toBe(2);
    expect(byEditStrategy.total).toBe(2);
    expect(byBrokenEdit.total).toBe(1);
    expect(search.mode).toBe("literal");
    expect(search.matchCount).toBe(1);
    expect(search.returnedCount).toBe(1);
    expect(search.hasMore).toBe(false);
    expect(regexSearch.matchCount).toBe(3);
    expect(regexSearch.returnedCount).toBe(2);
    expect(regexSearch.hasMore).toBe(true);
    expect(regexSearch.truncated).toBe(true);
    expect(regexSearch.contexts[0]?.match).toBe("retry attempt 1 failed");
    expect(invalidRegexSearch.error?.code).toBe("invalid_regex");
    expect(invalidRegexSearch.matchCount).toBe(0);
    expect(oversizedPatternSearch.error?.code).toBe("pattern_too_long");
    expect(boundedOutputSearch.matchCount).toBe(1);
    expect(boundedOutputSearch.returnedCount).toBe(1);
    expect(String(boundedOutputSearch.matches[0]?.attributes["large.payload"])).toContain("trace-store truncated");
    expect(noisyProjectionSearch.matchCount).toBe(1);
    expect(noisyProjectionSearch.returnedCount).toBe(1);
    expect(JSON.stringify(noisyProjectionSearch.matches[0]?.attributes)).not.toContain(noisyProjectionPayload);
    expect(noisyProjectionSearch.matches[0]?.attributes["llm.input_messages.0.message.role"]).toBe("user");
    expect(noisyProjectionSearch.matches[0]?.attributes["input.value"]).toContain("compact prompt preview marker");
    expect(noisyProjectionSearch.matches[0]?.attributes["output.hash"]).toBe("result-hash-a");
    expect(noisyProjectionSearch.matches[0]?.attributes["trace.sanitized.openinference_flat_projection.groups"]).toContain(
      "llm.input_messages",
    );
    expect(noisyProjectionSearch.matches[0]?.attributes["trace.sanitized.openinference_flat_projection.keys"]).toContain(
      "llm.input_messages.0.message.content",
    );
    expect(noisyMarkerSearch.matchCount).toBe(1);
    expect(noisyMarkerSearch.matches[0]?.attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"]).toBe(
      "marker_tool_call",
    );
    expect(spanView.spans[0]?.attributes["error.message"]).toBe("missing required path");
    expect(JSON.stringify(noisyProjectionView.spans[0]?.attributes)).not.toContain(noisyProjectionPayload);
    expect(noisyProjectionView.spans[0]?.attributes["llm.input_messages.0.message.role"]).toBe("user");
    expect(noisyProjectionView.spans[0]?.attributes["trace.sanitized.openinference_flat_projection.count"]).toBe(2);
    expect(readFileSync(join(cwd, ".bag", "telemetry", "spans.jsonl"), "utf8")).toContain(noisyProjectionPayload);
    expect(cappedSpanView.requestedSpanCount).toBe(3);
    expect(cappedSpanView.spanCount).toBe(2);
    expect(cappedSpanView.returnedCount).toBe(1);
    expect(cappedSpanView.hasMore).toBe(true);
    expect(cappedSpanView.truncated).toBe(true);
    expect(cappedSpanView.omittedSpanCount).toBe(2);
    expect(missingSpanView.spanCount).toBe(0);
    expect(missingSpanView.returnedCount).toBe(0);
    expect(missingSpanView.missingSpanIds).toEqual(["span-missing"]);
    expect(exactSpanSearch.matchCount).toBe(1);
    expect(exactSpanSearch.returnedCount).toBe(1);
    expect(exactSpanSearch.matches[0]?.span_id).toBe("span-b");
    expect(exactSpanSearch.contexts[0]?.spanId).toBe("span-b");
    expect(boundedSpanSearch.matchCount).toBe(3);
    expect(boundedSpanSearch.returnedCount).toBe(1);
    expect(boundedSpanSearch.hasMore).toBe(true);
    expect(String(boundedSpanSearch.matches[0]?.attributes["large.payload"])).toContain("trace-store truncated");
    expect(missingSpanSearch.matchCount).toBe(0);
    expect(missingSpanSearch.searchedSpanCount).toBe(0);
    expect(missingSpanSearch.missingSpanIds).toEqual(["span-missing"]);
  });

  test("surfaces trace source sizing and corrupt JSONL statistics", () => {
    const config = defaultConfig();
    const cwd = mkdtempSync(join(tmpdir(), "bag-trace-sizing-"));
    const tracePath = join(cwd, ".bag", "telemetry", "spans.jsonl");
    mkdirSync(join(cwd, ".bag", "telemetry"), { recursive: true });
    const span = (input: { traceId: string; spanId: string; name: string }) => ({
      trace_id: input.traceId,
      span_id: input.spanId,
      parent_span_id: "root",
      trace_state: "",
      name: input.name,
      kind: "SPAN_KIND_CLIENT",
      start_time: "2026-04-29T00:00:00.000Z",
      end_time: "2026-04-29T00:00:01.000Z",
      status: { code: "STATUS_CODE_OK", message: "" },
      resource: { attributes: { "service.name": "bleeding-agent" } },
      scope: { name: "bag.telemetry", version: "0.1.0" },
      attributes: {
        "inference.observation_kind": "TOOL",
        "inference.project_id": "bleeding-agent",
        "tool.name": "repo.read",
      },
    });
    const firstLine = `${JSON.stringify(span({ traceId: "trace-a", spanId: "span-a", name: "tool.workspace.repo.read" }))}\n`;
    const corruptLine = `{"trace_id":"trace-b","span_id":\n`;
    const blankLine = "\n";
    const secondLine = `${JSON.stringify(span({ traceId: "trace-c", spanId: "span-c", name: "tool.workspace.repo.write" }))}\n`;
    const content = firstLine + corruptLine + blankLine + secondLine;
    writeFileSync(tracePath, content);

    const index = buildTraceIndex({ config, cwd });
    const stat = statSync(tracePath);
    const highResolutionStat = statSync(tracePath, { bigint: true });
    const overview = TraceStore.open(config, cwd).getOverview();
    const traceA = index.rows.find((row) => row.traceId === "trace-a");
    const queryTraceA = TraceStore.open(config, cwd).queryTraces({ projectId: "bleeding-agent" }).traces.find(
      (row) => row.traceId === "trace-a",
    );

    expect(index.meta.sourceBytes).toBe(Buffer.byteLength(content));
    expect(index.meta.sourceSize).toBe(stat.size);
    expect(index.meta.sourceMtimeMs).toBe(stat.mtimeMs);
    expect(index.meta.sourceMtimeNs).toBe(highResolutionStat.mtimeNs.toString());
    expect(index.meta.sourceCtimeNs).toBe(highResolutionStat.ctimeNs.toString());
    expect(index.meta.rawJsonlBytes).toBe(Buffer.byteLength(content));
    expect(index.meta.parsedBytes).toBe(Buffer.byteLength(firstLine + secondLine));
    expect(index.meta.corruptBytes).toBe(Buffer.byteLength(corruptLine));
    expect(index.meta.lineCount).toBe(4);
    expect(index.meta.blankLineCount).toBe(1);
    expect(index.meta.parsedLineCount).toBe(2);
    expect(index.meta.corruptLineCount).toBe(1);
    expect(index.meta.parseErrorCount).toBe(1);
    expect(index.meta.traceCount).toBe(2);
    expect(index.meta.spanCount).toBe(2);
    expect(traceA?.rawJsonlBytes).toBe(Buffer.byteLength(firstLine));
    expect(queryTraceA?.rawJsonlBytes).toBe(Buffer.byteLength(firstLine));
    expect(overview.sourcePath).toBe(tracePath);
    expect(overview.sourceBytes).toBe(Buffer.byteLength(content));
    expect(overview.sourceMtimeNs).toBe(index.meta.sourceMtimeNs);
    expect(overview.rawJsonlBytes).toBe(Buffer.byteLength(firstLine + secondLine));
    expect(overview.parsedBytes).toBe(Buffer.byteLength(firstLine + secondLine));
    expect(overview.corruptBytes).toBe(Buffer.byteLength(corruptLine));
    expect(overview.lineCount).toBe(4);
    expect(overview.blankLineCount).toBe(1);
    expect(overview.parsedLineCount).toBe(2);
    expect(overview.corruptLineCount).toBe(1);
    expect(overview.parseErrorCount).toBe(1);
  });

  test("summarizes repeated trace failures for optimization", () => {
    const config = defaultConfig();
    const cwd = mkdtempSync(join(tmpdir(), "bag-span-opt-"));
    mkdirSync(join(cwd, ".bag", "telemetry"), { recursive: true });
    const baseSpan = {
      trace_id: "trace-a",
      parent_span_id: "root",
      trace_state: "",
      name: "tool.workspace.repo.read",
      kind: "SPAN_KIND_CLIENT",
      start_time: "2026-04-29T00:00:00.000Z",
      end_time: "2026-04-29T00:00:01.000Z",
      resource: { attributes: { "service.name": "bleeding-agent" } },
      scope: { name: "bag.telemetry", version: "0.1.0" },
      attributes: {
        "inference.observation_kind": "TOOL",
        "inference.project_id": "bleeding-agent",
        "tool.name": "repo.read",
        "input.hash": "abc",
        "error.message": "missing required path",
      },
    };
    writeFileSync(
      join(cwd, ".bag", "telemetry", "spans.jsonl"),
      `${JSON.stringify({
        ...baseSpan,
        span_id: "span-a",
        status: { code: "STATUS_CODE_ERROR", message: "missing required path" },
      })}\n${JSON.stringify({
        ...baseSpan,
        span_id: "span-b",
        status: { code: "STATUS_CODE_ERROR", message: "missing required path" },
      })}\n`,
    );

    const generated = generateSelfOptimization({ config, cwd });

    expect(generated.candidate.traceAnalysis?.errorSpanCount).toBe(2);
    expect(generated.candidate.traceDataset?.traceCount).toBe(1);
    expect(generated.candidate.improvementProposals.length).toBeGreaterThan(0);
    expect(generated.candidate.findings.some((finding) => finding.title.includes("trace failures"))).toBe(true);
  });
});

describe("BleedingAgent self-optimization", () => {
  test("generates and applies safe tool guidance from failed tool calls", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-self-opt-"));
    const config = defaultConfig();
    mkdirSync(join(cwd, ".bag", "telemetry"), { recursive: true });
    writeFileSync(
      join(cwd, ".bag", "telemetry", "metrics.json"),
      `${JSON.stringify(
        {
          run_1: {
            steps: [],
            llmCalls: [],
            toolCalls: [
              {
                toolName: "repo.read",
                namespace: "workspace",
                descriptionVersion: "v1",
                startedAt: "2026-04-29T00:00:00.000Z",
                completedAt: "2026-04-29T00:00:01.000Z",
                durationMs: 1000,
                ok: false,
                retryCount: 0,
                argumentBytes: 12,
                argumentHash: "abc",
                error: "missing required path",
                errorName: "ValidationError",
              },
            ],
          },
        },
        null,
        2,
      )}\n`,
    );

    const generated = generateSelfOptimization({ config, cwd });

    expect(generated.candidate.findings.some((finding) => finding.area === "tool")).toBe(true);
    expect(generated.candidate.toolGuidance.join("\n")).toContain("repo.read");

    const applied = applySelfOptimization({ config, cwd, candidateId: generated.candidate.id });

    expect(applied.guidanceWritten).toBe(true);
    expect(applied.planWritten).toBe(true);
    expect(readFileSync(join(cwd, ".bag", "tool-guidance.md"), "utf8")).toContain("missing required path");
    expect(readFileSync(join(cwd, ".bag", "self-improvement-plan.md"), "utf8")).toContain("Harden repo.read");
  });
});

describe("BleedingAgent ACP", () => {
  const agentMessageText = (updates: unknown[]): string =>
    updates
      .map((update) => (update as { update?: { content?: { text?: string } } }).update?.content?.text)
      .filter((text): text is string => typeof text === "string")
      .join("\n");

  const availableCommandNames = (updates: unknown[]): string[] =>
    updates
      .flatMap((update) => {
        const availableCommands = (update as { update?: { availableCommands?: Array<{ name?: unknown }> } }).update
          ?.availableCommands;
        return Array.isArray(availableCommands) ? availableCommands : [];
      })
      .map((command) => command.name)
      .filter((name): name is string => typeof name === "string");

  const sessionUpdates = <T extends { sessionUpdate?: string }>(updates: unknown[], sessionUpdate: string): T[] =>
    updates
      .map((update) => (update as { update?: T }).update)
      .filter((update): update is T => update?.sessionUpdate === sessionUpdate);

  const minimalAcpConsumerCapabilities = headlessAcpConsumerCapabilityProfile("minimal");
  const capableAcpConsumerCapabilities = headlessAcpConsumerCapabilityProfile("capable");
  const capableAcpClient = capableAcpConsumerCapabilities.clientCapabilities;

  const createOfflineAcpSmokeHarness = (cwd = mkdtempSync(join(tmpdir(), "bag-acp-smoke-"))) => {
    const updates: unknown[] = [];
    const writes: Array<{ path: string; content: string }> = [];
    const permissionRequests: unknown[] = [];
    const permissionOutcomes: Array<{ outcome: "cancelled" } | { outcome: "selected"; optionId: string }> = [];
    const terminalRuns: Array<{ command: string; args: string[]; cwd: string }> = [];
    const transcript: Array<Record<string, unknown>> = [];
    const recordCall = async <T>(method: string, fn: () => Promise<T>): Promise<T> => {
      transcript.push({ kind: "protocol_call", phase: "request", method });
      try {
        const result = await fn();
        transcript.push({ kind: "protocol_call", phase: "response", method });
        return result;
      } catch (error) {
        transcript.push({
          kind: "protocol_call",
          phase: "error",
          method,
          message: error instanceof Error ? error.message : String(error),
        });
        throw error;
      }
    };
    const connection = {
        sessionUpdate: async (update: unknown) => {
          transcript.push({ kind: "session_update", update });
          updates.push(update);
        },
        readTextFile: async (input: { path: string }) => {
          transcript.push({ kind: "fs_read", path: input.path });
          return {
            content: input.path.endsWith("example.ts") ? "export const value = 1;\n" : "offline fixture\n",
          };
        },
        writeTextFile: async (input: { path: string; content: string }) => {
          transcript.push({ kind: "fs_write", path: input.path, bytes: Buffer.byteLength(input.content) });
          writes.push(input);
          return {};
        },
        requestPermission: async (input: unknown) => {
          transcript.push({ kind: "permission", input });
          permissionRequests.push(input);
          return { outcome: permissionOutcomes.shift() ?? { outcome: "selected", optionId: "allow" } };
        },
        createTerminal: async (input: { command: string; args: string[]; cwd: string }) => {
          transcript.push({ kind: "terminal_create", command: input.command, args: input.args, cwd: input.cwd });
          terminalRuns.push(input);
          return {
            id: `terminal-${terminalRuns.length}`,
            waitForExit: async () => {
              transcript.push({ kind: "terminal_exit", exitCode: 0 });
              return { exitCode: 0, signal: null };
            },
            currentOutput: async () => {
              transcript.push({ kind: "terminal_output", output: "ok\n" });
              return { output: "ok\n" };
            },
            kill: async () => {
              transcript.push({ kind: "terminal_kill" });
              return {};
            },
            release: async () => {
              transcript.push({ kind: "terminal_release" });
              return {};
            },
          };
        },
      };
    const agent = new BleedingAcpAgent(connection as never, cwd);
    return { cwd, updates, writes, permissionRequests, permissionOutcomes, terminalRuns, transcript, recordCall, connection, agent };
  };

  test("advertises ACP capabilities and creates sessions", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );

    const initialized = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
    });
    const session = await agent.newSession({
      cwd,
      mcpServers: [],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/metrics" }],
    });

    expect(initialized.protocolVersion).toBe(1);
    expect(initialized.agentInfo?.name).toBe("bleeding-agent");
    expect(initialized.agentCapabilities.promptCapabilities?.embeddedContext).toBe(true);
    expect(session.sessionId).toStartWith("bag-");
    expect(session.modes?.currentModeId).toBe("auto");
    const serialized = JSON.stringify(updates);
    expect(serialized).toContain('"sessionUpdate":"available_commands_update"');
    expect(agentMessageText(updates)).toContain("Telemetry JSONL:");
    expect(agentMessageText(updates)).toContain("Active tuning: pinned for this session");
    expect(agentMessageText(updates)).not.toContain("Optimizer pin:");
    expect(agentMessageText(updates)).not.toContain("canonical-tools.v1/rendered-tools.v1");
  });

  test("documents ACP contract coverage separately from named consumer fixtures", () => {
    const matrix = acpConsumerCompatibilityMatrix();

    expect(matrix.map((entry) => entry.id)).toEqual([
      "session-start",
      "greeting",
      "plan-report",
      "edit-run",
      "terminal-verification",
      "permissions",
      "slash-commands",
      "cancellation",
      "trace-artifacts",
    ]);
    expect(matrix.every((entry) => entry.acpContract === "tested-offline")).toBe(true);
    expect(matrix.every((entry) => entry.namedConsumerFixtures.length > 0)).toBe(true);
    expect(matrix.every((entry) => !("zed" in entry) && !("glass" in entry))).toBe(true);
    expect(matrix.map((entry) => entry.expectedBehavior).join("\n")).toContain("YOLO bypasses prompts by default");
    expect(matrix.map((entry) => entry.smokeSignal).join("\n")).toContain("available_commands_update");
  });

  test("models minimal and capable ACP consumer capabilities explicitly", () => {
    expect(minimalAcpConsumerCapabilities).toMatchObject({
      profileId: "minimal",
      clientCapabilities: {
        fs: { readTextFile: false, writeTextFile: false },
        terminal: false,
      },
      filesystem: { readTextFile: false, writeTextFile: false },
      terminal: { create: false, output: false, waitForExit: false, kill: false, release: false },
      richToolContent: { diff: false, terminal: false, textFallback: true },
      artifactLinks: { fileLocations: true, resourceLinks: false },
      slashCommands: { availableCommandsUpdate: true, textSlashPrompts: true },
      permissions: { requestPermission: true, yoloAutoAllow: true, safeAutoReject: true },
      promptContent: { text: true, image: false, resource: false },
      unsupported: { images: true, resources: true, nes: true, provider: true, forkSession: true },
    });
    expect(capableAcpConsumerCapabilities).toMatchObject({
      profileId: "capable",
      clientCapabilities: {
        fs: { readTextFile: true, writeTextFile: true },
        terminal: true,
      },
      filesystem: { readTextFile: true, writeTextFile: true },
      terminal: { create: true, output: true, waitForExit: true, kill: true, release: true },
      richToolContent: { diff: true, terminal: true, textFallback: true },
      artifactLinks: { fileLocations: true, resourceLinks: false },
      slashCommands: { availableCommandsUpdate: true, textSlashPrompts: true },
      permissions: { requestPermission: true, yoloAutoAllow: true, safeAutoReject: true },
      promptContent: { text: true, image: false, resource: false },
      unsupported: { images: true, resources: true, nes: true, provider: true, forkSession: true },
    });
    expect(JSON.stringify(capableAcpConsumerCapabilities.clientCapabilities)).not.toContain("nes");
    expect(JSON.stringify(capableAcpConsumerCapabilities.clientCapabilities)).not.toContain("provider");
    expect(JSON.stringify(capableAcpConsumerCapabilities.clientCapabilities)).not.toContain("fork");
  });

  test("headless ACP client fixture records protocol, filesystem, terminal, permission, and update transcripts", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-headless-client-"));
    writeFileSync(join(cwd, "input.txt"), "line 1\nline 2\n");
    const trajectory: TrajectoryEntry[] = [];
    const { client } = createHeadlessAcpClientFixture({
      workdir: cwd,
      trajectory,
      yolo: false,
      terminalMode: "stub",
    });

    recordHeadlessAcpProtocolCall(trajectory, {
      method: "initialize",
      phase: "response",
      payload: { protocolVersion: 1 },
    });
    recordHeadlessAcpProtocolCall(trajectory, {
      method: "session/new",
      phase: "response",
      sessionId: "bag-test",
    });
    recordHeadlessAcpProtocolCall(trajectory, {
      method: "session/resume",
      phase: "response",
      sessionId: "bag-test",
    });
    recordHeadlessAcpProtocolCall(trajectory, {
      method: "session/cancel",
      phase: "response",
      sessionId: "bag-test",
    });
    await client.sessionUpdate({
      sessionId: "bag-test",
      update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "streamed" } },
    } as never);
    const read = await client.readTextFile({ path: "input.txt", line: 2, limit: 1 } as never);
    await client.writeTextFile({ path: "out/generated.txt", content: "generated\n" } as never);
    const permission = await client.requestPermission({
      options: [{ kind: "reject_once", optionId: "reject", name: "Reject once" }],
      toolCall: { title: "write generated file", kind: "edit" },
    } as never);
    const terminal = await client.createTerminal({
      command: "npm",
      args: ["test"],
      cwd,
      outputByteLimit: 2048,
    } as never);
    const output = await client.terminalOutput({ terminalId: terminal.terminalId } as never);
    const exit = await client.waitForTerminalExit({ terminalId: terminal.terminalId } as never);
    await client.killTerminal({ terminalId: terminal.terminalId } as never);
    await client.releaseTerminal({ terminalId: terminal.terminalId } as never);

    const summary = summarizeHeadlessAcpTranscript(trajectory);
    expect(read.content).toBe("line 2");
    expect(readFileSync(join(cwd, "out", "generated.txt"), "utf8")).toBe("generated\n");
    expect(permission.outcome).toEqual({ outcome: "selected", optionId: "reject" });
    expect(output.output).toContain("[headless-acp stub terminal] npm test");
    expect(exit).toEqual({ exitCode: 0, signal: null });
    expect(summary.protocolMethods).toEqual({
      initialize: 1,
      "session/new": 1,
      "session/resume": 1,
      "session/cancel": 1,
    });
    expect(summary.counts).toEqual(
      expect.objectContaining({
        sessionUpdates: 1,
        fsRead: 1,
        fsWrite: 1,
        terminalCreate: 1,
        terminalOutput: 1,
        terminalExit: 1,
        terminalKill: 1,
        terminalRelease: 1,
        permission: 1,
      }),
    );
  });

  test("covers labeled ACP regression scenario transcripts in the offline headless harness", async () => {
    type OfflineHarness = ReturnType<typeof createOfflineAcpSmokeHarness>;
    const scenarioIds = HEADLESS_ACP_REGRESSION_SCENARIOS.map((scenario) => scenario.id);
    const scenarioById = new Map(HEADLESS_ACP_REGRESSION_SCENARIOS.map((scenario) => [scenario.id, scenario]));
    const covered = new Set<HeadlessAcpRegressionScenarioId>();
    const updateSignals = (updates: unknown[]): string[] =>
      updates
        .map((update) => (update as { update?: { sessionUpdate?: unknown } }).update?.sessionUpdate)
        .filter((signal): signal is string => typeof signal === "string");
    const transcriptKinds = (transcript: Array<Record<string, unknown>>): string[] =>
      transcript.map((entry) => String(entry.kind));
    const resetScenario = (harness: OfflineHarness) => {
      harness.updates.length = 0;
      harness.writes.length = 0;
      harness.permissionRequests.length = 0;
      harness.permissionOutcomes.length = 0;
      harness.terminalRuns.length = 0;
      harness.transcript.length = 0;
    };
    const startScenario = async () => {
      const harness = createOfflineAcpSmokeHarness();
      await harness.recordCall("initialize", () =>
        harness.agent.initialize({ protocolVersion: 1, clientCapabilities: capableAcpClient }),
      );
      const session = await harness.recordCall("session/new", () => harness.agent.newSession({ cwd: harness.cwd, mcpServers: [] }));
      return { harness, session };
    };
    const assertScenario = (id: HeadlessAcpRegressionScenarioId, harness: OfflineHarness) => {
      const scenario = scenarioById.get(id);
      expect(scenario).toBeDefined();
      const kinds = transcriptKinds(harness.transcript);
      const signals = updateSignals(harness.updates);
      for (const kind of scenario?.expected.requiredKinds ?? []) {
        expect(kinds).toContain(kind);
      }
      for (const kind of scenario?.expected.forbiddenKinds ?? []) {
        expect(kinds).not.toContain(kind);
      }
      for (const signal of scenario?.expected.updateSignals ?? []) {
        expect(signals).toContain(signal);
      }
      covered.add(id);
    };
    const publishScenarioProgress = async (
      harness: OfflineHarness,
      sessionId: string,
      input: { title: string; toolCallId: string; planEntry: string },
    ) => {
      await harness.connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [{ content: input.planEntry, priority: "high", status: "completed" }],
        },
      });
      await harness.connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: input.toolCallId,
          title: input.title,
          kind: "think",
          status: "pending",
        },
      });
      await harness.connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId: input.toolCallId,
          status: "completed",
          rawOutput: { sideEffects: [] },
          content: [{ type: "content", content: { type: "text", text: `${input.title} complete.` } }],
        },
      });
    };

    {
      const { harness, session } = await startScenario();
      await harness.agent.setSessionMode({ sessionId: session.sessionId, modeId: "chat" });
      resetScenario(harness);
      await harness.recordCall("session/prompt", () =>
        harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "hello" }] }),
      );

      assertScenario("greeting-no-side-effect", harness);
      expect(agentMessageText(harness.updates)).toContain("ACP coding agent");
      expect(agentMessageText(harness.updates)).not.toMatch(/\b(optimizer|policyId|modelProfileId|registry)\b/i);
      expect(harness.writes).toHaveLength(0);
      expect(harness.permissionRequests).toHaveLength(0);
      expect(harness.terminalRuns).toHaveLength(0);
    }

    {
      const { harness, session } = await startScenario();
      replaceRunPlanningTurnForTest(harness.agent, async (acpSession) => {
        await publishScenarioProgress(harness, acpSession.id, {
          title: "Render read-only report",
          toolCallId: "scenario-report",
          planEntry: "Render read-only report without ACP side effects",
        });
      });
      resetScenario(harness);
      await harness.recordCall("session/prompt", () =>
        harness.agent.prompt({
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "/plan summarize repository state without edits" }],
        }),
      );

      assertScenario("read-only-report", harness);
      expect(harness.writes).toHaveLength(0);
      expect(harness.permissionRequests).toHaveLength(0);
      expect(harness.terminalRuns).toHaveLength(0);
    }

    {
      const { harness, session } = await startScenario();
      let codingRuns = 0;
      replaceRunCodingTurnForTest(harness.agent, async (acpSession) => {
        codingRuns += 1;
        await publishScenarioProgress(harness, acpSession.id, {
          title: "Run bounded coding task",
          toolCallId: "scenario-coding",
          planEntry: "Execute coding task through Run mode",
        });
      });
      resetScenario(harness);
      await harness.recordCall("session/prompt", () =>
        harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/run make a bounded coding change" }] }),
      );

      assertScenario("coding-run", harness);
      expect(codingRuns).toBe(1);
    }

    {
      const { harness, session } = await startScenario();
      const acpSession = requireAgentSessionForTest(harness.agent, session.sessionId);
      const telemetry = telemetryForAgentSession(harness.agent, session.sessionId, harness.cwd, "test-acp-regression-edit");
      resetScenario(harness);
      const edit = await previewAndWriteClientEditThroughAgentForTest(harness.agent, {
        session: acpSession,
        telemetry,
        path: join(harness.cwd, "example.ts"),
        oldContent: "export const value = 1;\n",
        newContent: "export const value = 2;\n",
        reason: "ACP regression edit preview/write scenario",
      });

      assertScenario("edit-preview-write", harness);
      expect(edit.ok).toBe(true);
      expect(JSON.stringify(harness.updates)).toContain("Preview edit strategy");
      expect(JSON.stringify(harness.updates)).toContain('"type":"diff"');
      expect(harness.writes).toEqual([expect.objectContaining({ content: "export const value = 2;\n" })]);
    }

    {
      const { harness, session } = await startScenario();
      const telemetry = telemetryForAgentSession(harness.agent, session.sessionId, harness.cwd, "test-acp-regression-terminal");
      resetScenario(harness);
      const terminal = await runTerminalCommandThroughAgentForTest(harness.agent, {
        sessionId: session.sessionId,
        telemetry,
        command: "npm",
        args: ["run", "typecheck"],
        reason: "ACP regression terminal verification scenario",
        cwd: harness.cwd,
      });

      assertScenario("terminal-verification", harness);
      expect(terminal.exitCode).toBe(0);
      expect(harness.terminalRuns).toEqual([expect.objectContaining({ command: "npm", args: ["run", "typecheck"] })]);
      expect(JSON.stringify(harness.updates)).toContain('"type":"terminal"');
    }

    {
      const { harness, session } = await startScenario();
      await harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/safe" }] });
      const telemetry = telemetryForAgentSession(harness.agent, session.sessionId, harness.cwd, "test-acp-regression-permission");
      harness.permissionOutcomes.push({ outcome: "selected", optionId: "reject" });
      resetScenario(harness);
      harness.permissionOutcomes.push({ outcome: "selected", optionId: "reject" });
      const write = await writeClientFileThroughAgentForTest(harness.agent, {
        sessionId: session.sessionId,
        telemetry,
        path: join(harness.cwd, "example.ts"),
        oldContent: "export const value = 1;\n",
        newContent: "export const value = 2;\n",
        reason: "ACP regression permission rejection scenario",
      });

      assertScenario("rejected-permission", harness);
      expect(write).toEqual(expect.objectContaining({ ok: false, reason: "edit permission rejected" }));
      expect(harness.permissionRequests).toHaveLength(1);
      expect(harness.writes).toHaveLength(0);
      expect(JSON.stringify(harness.updates)).toContain("permission_rejected");
    }

    {
      const { harness, session } = await startScenario();
      let runStartedResolve: () => void = () => {};
      const runStarted = new Promise<void>((resolvePromise) => {
        runStartedResolve = resolvePromise;
      });
      replaceRunCodingTurnForTest(harness.agent, async (_session, _task, signal) => {
        runStartedResolve();
        await new Promise<void>((_resolve, reject) => {
          if (signal.aborted) {
            reject(new Error("cancelled"));
            return;
          }
          signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
        });
      });
      resetScenario(harness);
      const runningPrompt = harness.recordCall("session/prompt", () =>
        harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/run long running task" }] }),
      );
      await runStarted;
      await harness.recordCall("session/cancel", () => harness.agent.cancel({ sessionId: session.sessionId }));
      await expect(runningPrompt).resolves.toEqual({ stopReason: "cancelled" });

      assertScenario("cancellation", harness);
      expect(transcriptKinds(harness.transcript)).toContain("protocol_call");
      await expect(
        harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/chat" }] }),
      ).resolves.toEqual({ stopReason: "end_turn" });
    }

    {
      const { harness, session } = await startScenario();
      resetScenario(harness);
      await harness.recordCall("session/prompt", () =>
        harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/metrics" }] }),
      );
      await harness.recordCall("session/prompt", () =>
        harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/traces" }] }),
      );

      assertScenario("metrics-traces", harness);
      expect(agentMessageText(harness.updates)).toContain("Telemetry JSONL:");
      expect(agentMessageText(harness.updates)).toContain("HALO-style trace dataset");
    }

    {
      const { harness, session } = await startScenario();
      const names = availableCommandNames(harness.updates);
      expect(names).not.toContain("maintenance");
      expect(names).not.toContain("optimize");
      expect(names).not.toContain("promote");
      expect(names).not.toContain("rollback");
      resetScenario(harness);
      await harness.recordCall("session/prompt", () =>
        harness.agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/maintenance status" }] }),
      );

      assertScenario("maintenance-isolation", harness);
      expect(agentMessageText(harness.updates)).toContain("Maintenance optimizer status");
      expect(harness.writes).toHaveLength(0);
      expect(harness.terminalRuns).toHaveLength(0);
    }

    expect([...covered].sort()).toEqual([...scenarioIds].sort());
  });

  test("contrasts minimal and capable ACP client transcripts for rich content and side effects", async () => {
    const exerciseProfile = async (profileId: "minimal" | "capable") => {
      const profile = headlessAcpConsumerCapabilityProfile(profileId);
      const { agent, updates, writes, permissionRequests, terminalRuns, transcript, recordCall, cwd } =
        createOfflineAcpSmokeHarness();
      await recordCall("initialize", () =>
        agent.initialize({ protocolVersion: 1, clientCapabilities: profile.clientCapabilities }),
      );
      const sessionResponse = await recordCall("session/new", () => agent.newSession({ cwd, mcpServers: [] }));
      const telemetry = telemetryForAgentSession(agent, sessionResponse.sessionId, cwd, `test-acp-${profileId}-capabilities`);

      const write = await writeClientFileThroughAgentForTest(agent, {
        sessionId: sessionResponse.sessionId,
        telemetry,
        path: join(cwd, "example.ts"),
        oldContent: "export const value = 1;\n",
        newContent: "export const value = 2;\n",
        reason: `${profileId} capability write`,
      });
      const command = await runTerminalCommandThroughAgentForTest(agent, {
        sessionId: sessionResponse.sessionId,
        telemetry,
        command: "npm",
        args: ["test"],
        reason: `${profileId} capability terminal`,
        cwd,
      });

      return { command, permissionRequests, profile, telemetry, terminalRuns, transcript, updates, write, writes };
    };

    const minimal = await exerciseProfile("minimal");
    const capable = await exerciseProfile("capable");
    const minimalSerialized = JSON.stringify(minimal.updates);
    const capableSerialized = JSON.stringify(capable.updates);
    const minimalTranscriptKinds = minimal.transcript.map((entry) => entry.kind);
    const capableTranscriptKinds = capable.transcript.map((entry) => entry.kind);

    expect(minimal.write).toEqual(
      expect.objectContaining({ ok: false, reason: "ACP client does not support fs/write_text_file" }),
    );
    expect(minimal.command).toEqual(
      expect.objectContaining({ exitCode: null, output: "ACP client does not support terminal/create" }),
    );
    expect(minimal.permissionRequests).toHaveLength(0);
    expect(minimal.writes).toHaveLength(0);
    expect(minimal.terminalRuns).toHaveLength(0);
    expect(minimalSerialized).toContain("Proposed edit to");
    expect(minimalSerialized).toContain("ACP client does not support fs/write_text_file");
    expect(minimalSerialized).toContain("ACP client does not support terminal/create");
    expect(minimalSerialized).not.toContain('"type":"diff"');
    expect(minimalSerialized).not.toContain('"type":"terminal"');
    expect(minimalTranscriptKinds).not.toContain("fs_write");
    expect(minimalTranscriptKinds).not.toContain("terminal_create");
    expect(minimal.telemetry.toolMetrics.filter((metric) => !metric.ok).map((metric) => metric.toolName)).toEqual([
      "acp.fs.writeTextFile",
      "acp.terminal.create",
    ]);

    expect(capable.write).toEqual(expect.objectContaining({ ok: true }));
    expect(capable.command).toEqual(expect.objectContaining({ exitCode: 0, output: "ok\n" }));
    expect(capable.permissionRequests).toHaveLength(0);
    expect(capable.writes).toHaveLength(1);
    expect(capable.terminalRuns).toHaveLength(1);
    expect(capableSerialized).toContain('"type":"diff"');
    expect(capableSerialized).toContain('"type":"terminal"');
    expect(capableSerialized).toContain("Applied edit to example.ts");
    expect(capableTranscriptKinds).toEqual(expect.arrayContaining(["fs_write", "terminal_create", "terminal_exit", "terminal_output"]));
    expect(capable.telemetry.toolMetrics.filter((metric) => metric.ok).map((metric) => metric.toolName)).toEqual([
      "acp.fs.writeTextFile",
      "acp.terminal.create",
    ]);
  });

  test("runs an offline headless ACP consumer smoke transcript", async () => {
    const { agent, updates, writes, permissionRequests, terminalRuns, transcript, recordCall, cwd } =
      createOfflineAcpSmokeHarness();
    const initialized = await recordCall("initialize", () =>
      agent.initialize({ protocolVersion: 1, clientCapabilities: capableAcpClient }),
    );
    const session = await recordCall("session/new", () => agent.newSession({ cwd, mcpServers: [] }));
    await recordCall("session/resume", () => agent.resumeSession({ sessionId: session.sessionId, cwd, mcpServers: [] }));
    await recordCall("session/cancel", () => agent.cancel({ sessionId: session.sessionId }));
    const acpSession = requireAgentSessionForTest(agent, session.sessionId);
    const telemetry = telemetryForAgentSession(agent, session.sessionId, cwd, "test-acp-consumer-smoke");

    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "chat" });
    await recordCall("session/prompt", () =>
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "hello" }] }),
    );
    await recordCall("session/prompt", () =>
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/auto" }] }),
    );
    const read = await readClientFileThroughAgentForTest(agent, { sessionId: session.sessionId, telemetry, path: join(cwd, "example.ts") });
    const edit = await previewAndWriteClientEditThroughAgentForTest(agent, {
      session: acpSession,
      telemetry,
      path: join(cwd, "example.ts"),
      oldContent: read,
      newContent: "export const value = 2;\n",
      reason: "offline smoke edit",
    });
    const terminal = await runTerminalCommandThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      command: "npm",
      args: ["run", "typecheck"],
      reason: "offline smoke verification",
      cwd,
    });
    await recordCall("session/prompt", () =>
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/metrics" }] }),
    );
    await recordCall("session/prompt", () =>
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "/traces" }] }),
    );

    const serialized = JSON.stringify(updates);
    const transcriptMethods = transcript
      .filter((entry) => entry.kind === "protocol_call" && entry.phase === "response")
      .map((entry) => entry.method);
    const transcriptKinds = transcript.map((entry) => entry.kind);
    const toolUpdateText = sessionUpdates<{ sessionUpdate: string; content?: Array<{ content?: { text?: string } }> }>(
      updates,
      "tool_call_update",
    )
      .flatMap((update) => update.content ?? [])
      .map((content) => content.content?.text)
      .filter((text): text is string => typeof text === "string")
      .join("\n");

    expect(initialized.protocolVersion).toBe(1);
    expect(session.modes?.currentModeId).toBe("auto");
    expect(agentMessageText(updates)).toContain("ACP coding agent");
    expect(edit.ok).toBe(true);
    expect(terminal.exitCode).toBe(0);
    expect(writes).toEqual([expect.objectContaining({ content: "export const value = 2;\n" })]);
    expect(permissionRequests).toHaveLength(0);
    expect(terminalRuns).toEqual([expect.objectContaining({ command: "npm", args: ["run", "typecheck"] })]);
    expect(transcriptMethods).toEqual(
      expect.arrayContaining(["initialize", "session/new", "session/resume", "session/cancel", "session/prompt"]),
    );
    expect(transcriptKinds).toEqual(
      expect.arrayContaining(["session_update", "fs_read", "fs_write", "terminal_create", "terminal_exit", "terminal_output"]),
    );
    expect(serialized).toContain('"sessionUpdate":"available_commands_update"');
    expect(serialized).toContain('"type":"diff"');
    expect(serialized).toContain('"type":"terminal"');
    expect(agentMessageText(updates)).toContain("Telemetry JSONL:");
    expect(agentMessageText(updates)).toContain("HALO-style trace dataset");
    expect(toolUpdateText).toContain("Read example.ts");
    expect(toolUpdateText).toContain("Applied edit to example.ts");
    expect(toolUpdateText).toContain("Command npm run typecheck exited 0");
    expect(toolUpdateText).not.toContain("```json");
    expect(toolUpdateText).not.toMatch(/\b(policyId|modelProfileId|canonicalToolVersion|renderedToolVersion)\b/);
  });

  test("keeps normal ACP command surface coding-first without maintenance controls", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-surface-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );

    await agent.newSession({ cwd, mcpServers: [] });

    expect(availableCommandNames(updates)).toEqual([
      "run",
      "plan",
      "chat",
      "auto",
      "yolo",
      "safe",
      "skills",
      "mcp",
      "metrics",
      "traces",
    ]);
    expect(availableCommandNames(updates)).not.toContain("maintenance");
    expect(availableCommandNames(updates)).not.toContain("optimize");
    expect(availableCommandNames(updates)).not.toContain("promote");
    expect(availableCommandNames(updates)).not.toContain("rollback");
    const serialized = JSON.stringify(updates);
    expect(serialized).toContain("coding task");
    expect(serialized).toContain("planning or reporting task");
    expect(serialized).toContain("telemetry, metrics, trace");
    expect(serialized).not.toMatch(/\b(optimize|optimizer|promote|rollback|registry|policyId|modelProfileId)\b/i);
  });

  test("answers normal chat with capability help without noisy optimizer internals", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-chat-surface-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });
    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "chat" });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "Ahoj, co umíš?" }],
    });

    const text = agentMessageText(updates);
    expect(text).toContain("ACP coding agent");
    expect(text).not.toContain("Glass");
    expect(text).not.toContain("Zed");
    expect(text).toContain("/run <task>");
    expect(text).toContain("/plan <task>");
    expect(text).toContain("/metrics");
    expect(text).toContain("/traces");
    expect(text).toContain("/yolo");
    expect(text).toContain("/safe");
    expect(text).toContain("/skills");
    expect(text).toContain("/mcp");
    expect(text).not.toMatch(/\b(optimizer|policyId|modelProfileId|canonical|rendered|registry|promote|rollback)\b/i);
  });

  test("prints a Zed settings snippet for bag acp", () => {
    const snippet = readAcpZedSettingsSnippet();
    expect(snippet).toContain('"command": "bag"');
    expect(snippet).toContain('"acp"');
  });

  test("prints generic ACP settings with named consumer examples", () => {
    const snippet = JSON.parse(readAcpSettingsSnippet()) as {
      acp_server?: { command?: string; args?: string[] };
      named_examples?: Record<string, unknown>;
    };
    expect(snippet.acp_server?.command).toBe("bag");
    expect(snippet.acp_server?.args).toEqual(["acp"]);
    expect(Object.keys(snippet.named_examples ?? {}).sort()).toEqual(["glass", "zed"]);
    expect(snippet.named_examples?.glass).toBeDefined();
    expect(snippet.named_examples?.zed).toBeDefined();
  });

  test("auto mode routes plain text through agent decision", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-auto-"));
    const updates: unknown[] = [];
    let codingRuns = 0;
    let planningRuns = 0;
    const routes: Array<"chat" | "plan" | "run"> = ["chat", "plan", "run"];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    replaceDecidePromptRouteForTest(agent, async () => routes.shift() ?? "chat");
    replaceRunCodingTurnForTest(agent, async () => {
      codingRuns += 1;
    });
    replaceRunPlanningTurnForTest(agent, async () => {
      planningRuns += 1;
    });
    const session = await agent.newSession({ cwd, mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "Ahoj, co umíš?" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "Vygeneruj report o stavu codebase." }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "Oprav failing tests a ověř to." }],
    });

    const serialized = JSON.stringify(updates);
    expect(serialized).toContain("V Auto módu rozhoduju");
    expect(codingRuns).toBe(1);
    expect(planningRuns).toBe(1);
    expect(
      updates
        .map((update) => (update as { update?: { sessionUpdate?: string; currentModeId?: string } }).update)
        .filter((update) => update?.sessionUpdate === "current_mode_update")
        .map((update) => update?.currentModeId),
    ).toEqual(["plan", "auto", "run", "auto"]);
  });

  test("routes slash commands and explicit non-chat modes to execution", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-routing-"));
    const updates: unknown[] = [];
    let codingRuns = 0;
    let planningRuns = 0;
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    replaceDecidePromptRouteForTest(agent, async () => "run");
    replaceRunCodingTurnForTest(agent, async () => {
      codingRuns += 1;
    });
    replaceRunPlanningTurnForTest(agent, async () => {
      planningRuns += 1;
    });
    const session = await agent.newSession({ cwd, mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/run ahoj" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/plan navrhni flow" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/chat" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "plain text in chat mode" }],
    });
    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "run" });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "plain text after manual run-mode opt-in" }],
    });
    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "plan" });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "plain text after manual plan-mode opt-in" }],
    });

    expect(codingRuns).toBe(2);
    expect(planningRuns).toBe(2);
    expect(JSON.stringify(updates)).toContain("Plain messages will not read, edit, or run the project");
    expect(
      updates
        .map((update) => (update as { update?: { sessionUpdate?: string; currentModeId?: string } }).update)
        .filter((update) => update?.sessionUpdate === "current_mode_update")
        .map((update) => update?.currentModeId),
    ).toEqual(["run", "auto", "plan", "auto", "chat", "run", "plan"]);
  });

  test("writes files in default YOLO mode without ACP permission prompt", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-"));
    const updates: unknown[] = [];
    const writes: Array<{ path: string; content: string }> = [];
    let permissionRequests = 0;
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        requestPermission: async () => {
          permissionRequests += 1;
          return {
            outcome: {
              outcome: "selected",
              optionId: "allow",
            },
          };
        },
        writeTextFile: async (input: { path: string; content: string }) => {
          writes.push(input);
          return {};
        },
      } as never,
      cwd,
    );
    const telemetry = new RunTelemetry(defaultConfig(), "test-acp-edit", cwd);
    const session = await agent.newSession({ cwd, mcpServers: [] });

    const result = await writeClientFileThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      path: join(cwd, "example.ts"),
      oldContent: "export const value = 1;\n",
      newContent: "export const value = 2;\n",
      reason: "exercise ACP write path",
    });

    expect(result.ok).toBe(true);
    expect(permissionRequests).toBe(0);
    expect(writes).toHaveLength(1);
    expect(writes[0]?.content).toContain("value = 2");
    expect(telemetry.toolMetrics.some((metric) => metric.toolName === "acp.fs.writeTextFile" && metric.ok)).toBe(true);
    expect(JSON.stringify(updates)).toContain('"sessionUpdate":"tool_call"');
    expect(JSON.stringify(updates)).toContain('"type":"diff"');
  });

  test("prompts in Safe mode for writes and terminal commands with traceable rejection outcomes", async () => {
    const { agent, updates, writes, permissionRequests, permissionOutcomes, terminalRuns, cwd } =
      createOfflineAcpSmokeHarness();
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    await agent.prompt({ sessionId: sessionResponse.sessionId, prompt: [{ type: "text", text: "/safe" }] });
    permissionOutcomes.push({ outcome: "selected", optionId: "reject" }, { outcome: "selected", optionId: "reject" });
    const telemetry = telemetryForAgentSession(agent, sessionResponse.sessionId, cwd, "test-acp-safe-rejections");

    const write = await writeClientFileThroughAgentForTest(agent, {
      sessionId: sessionResponse.sessionId,
      telemetry,
      path: join(cwd, "example.ts"),
      oldContent: "export const value = 1;\n",
      newContent: "export const value = 2;\n",
      reason: "safe write rejection",
    });
    const command = await runTerminalCommandThroughAgentForTest(agent, {
      sessionId: sessionResponse.sessionId,
      telemetry,
      command: "npm",
      args: ["test"],
      reason: "safe command rejection",
      cwd,
    });

    expect(write).toEqual(expect.objectContaining({ ok: false, reason: "edit permission rejected" }));
    expect(command).toEqual(expect.objectContaining({ exitCode: null, output: "command permission rejected" }));
    expect(permissionRequests).toHaveLength(2);
    expect(writes).toHaveLength(0);
    expect(terminalRuns).toHaveLength(0);
    const failedUpdates = sessionUpdates<{ sessionUpdate: string; status?: string; rawOutput?: Record<string, unknown> }>(
      updates,
      "tool_call_update",
    ).filter((update) => update.status === "failed");
    expect(JSON.stringify(failedUpdates)).toContain("permission_rejected");
    expect(JSON.stringify(failedUpdates)).toContain("edit permission rejected");
    expect(JSON.stringify(failedUpdates)).toContain("command permission rejected");
    expect(telemetry.toolMetrics.filter((metric) => !metric.ok).map((metric) => metric.toolName)).toEqual([
      "acp.fs.writeTextFile",
      "acp.terminal.create",
    ]);
  });

  test("cancels terminal verification with failed update and terminal cleanup", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-terminal-cancel-"));
    const updates: unknown[] = [];
    let killed = 0;
    let released = 0;
    let terminalCreatedResolve: () => void = () => {};
    const terminalCreated = new Promise<void>((resolvePromise) => {
      terminalCreatedResolve = resolvePromise;
    });
    let waitStartedResolve: () => void = () => {};
    const waitStarted = new Promise<void>((resolvePromise) => {
      waitStartedResolve = resolvePromise;
    });
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        createTerminal: async () => {
          terminalCreatedResolve();
          return {
            id: "terminal-cancel",
            waitForExit: async () => {
              waitStartedResolve();
              return new Promise<{ exitCode: number | null; signal: string | null }>(() => {});
            },
            currentOutput: async () => ({ output: "still running\n" }),
            kill: async () => {
              killed += 1;
              return {};
            },
            release: async () => {
              released += 1;
              return {};
            },
          };
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });
    const telemetry = telemetryForAgentSession(agent, session.sessionId, cwd, "test-acp-terminal-cancel");
    const controller = new AbortController();
    const run = runTerminalCommandThroughAgentForTest(agent, {
      sessionId: session.sessionId,
      telemetry,
      command: "npm",
      args: ["test"],
      reason: "cancel smoke",
      cwd,
      signal: controller.signal,
    });

    await terminalCreated;
    await waitStarted;
    controller.abort();

    await expect(run).rejects.toThrow("cancelled");
    expect(killed).toBe(1);
    expect(released).toBe(1);
    const failedUpdates = sessionUpdates<{ sessionUpdate: string; status?: string; rawOutput?: Record<string, unknown> }>(
      updates,
      "tool_call_update",
    ).filter((update) => update.status === "failed");
    expect(JSON.stringify(failedUpdates)).toContain("cancelled");
    expect(JSON.stringify(failedUpdates)).toContain("ABORT_ERR");
  });

  test("cancels prompt-level coding runs, clears pending prompt, and reuses resumed sessions", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-prompt-cancel-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });
    let codingRuns = 0;
    let runStartedResolve: () => void = () => {};
    const runStarted = new Promise<void>((resolvePromise) => {
      runStartedResolve = resolvePromise;
    });
    replaceRunCodingTurnForTest(agent, async (_session, _task, signal) => {
      codingRuns += 1;
      runStartedResolve();
      await new Promise<void>((_resolve, reject) => {
        if (signal.aborted) {
          reject(new Error("cancelled"));
          return;
        }
        signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
      });
    });

    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "run" });
    const runningPrompt = agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "change the project and verify it" }],
    });
    await runStarted;
    await agent.resumeSession({ sessionId: session.sessionId, cwd, mcpServers: [] });
    await agent.loadSession({ sessionId: session.sessionId, cwd, mcpServers: [] });
    await agent.cancel({ sessionId: session.sessionId });

    await expect(runningPrompt).resolves.toEqual({ stopReason: "cancelled" });
    expect(codingRuns).toBe(1);
    expect(requireAgentSessionForTest(agent, session.sessionId).pendingPrompt).toBeNull();

    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "chat" });
    await expect(
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "hello after cancel" }] }),
    ).resolves.toEqual({ stopReason: "end_turn" });
    expect(agentMessageText(updates)).toContain("ACP coding agent");
  });

  test("persists planning cancellation artifacts and manifest for started runs", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-planning-cancel-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });
    let toolCalls = 0;
    replaceRunAcpToolForTest(agent, async <T,>(input: { toolName: string }) => {
      toolCalls += 1;
      if (input.toolName === "bag.knowledge.load") {
        await agent.cancel({ sessionId: session.sessionId });
        return "learned planning context\n" as T;
      }
      throw new Error(`unexpected planning side effect after cancellation: ${input.toolName}`);
    });

    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "plan" });
    await expect(
      agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "write a PRD for this repo" }] }),
    ).resolves.toEqual({ stopReason: "cancelled" });

    const runsDir = join(cwd, ".bag", "runs");
    const runId = readdirSync(runsDir).find((entry) => entry.startsWith("acp-"));
    expect(runId).toBeDefined();
    const runRoot = join(runsDir, runId ?? "");
    const cancellation = JSON.parse(readFileSync(join(runRoot, "cancellation.json"), "utf8")) as {
      partialArtifacts?: Record<string, string>;
      completedPlanEntries?: string[];
    };
    const manifest = JSON.parse(readFileSync(join(runRoot, "manifest.json"), "utf8")) as {
      artifacts?: Record<string, string>;
    };

    expect(toolCalls).toBe(1);
    expect(existsSync(join(runRoot, "planning-trace.json"))).toBe(true);
    expect(existsSync(join(runRoot, "knowledge-input.md"))).toBe(true);
    expect(cancellation.partialArtifacts?.knowledgeInput).toContain("knowledge-input.md");
    expect(cancellation.completedPlanEntries?.join("\n")).toContain("Load learned guidance");
    expect(manifest.artifacts?.trace).toContain("planning-trace.json");
    expect(manifest.artifacts?.cancellation).toContain("cancellation.json");
    expect(agentMessageText(updates)).toContain("ACP planning turn cancelled");
  });

  test("previews ACP edits through the edit strategy layer and records edit attempts", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-strategy-"));
    const updates: unknown[] = [];
    const writes: Array<{ path: string; content: string }> = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        writeTextFile: async (input: { path: string; content: string }) => {
          writes.push(input);
          return {};
        },
      } as never,
      cwd,
    );
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    const session = requireAgentSessionForTest(agent, sessionResponse.sessionId);
    const config = defaultConfig();
    const telemetry = new RunTelemetry(config, "test-acp-edit-strategy", cwd, session.optimizerPin.telemetry);

    const result = await previewAndWriteClientEditThroughAgentForTest(agent, {
      session,
      telemetry,
      path: join(cwd, "example.ts"),
      oldContent: "export const value = 1;\n",
      newContent: "export const value = 2;\n",
      reason: "exercise edit strategy preview path",
    });

    const serialized = JSON.stringify(updates);
    const spans = readHaloSpans(config, cwd);
    const editSpan = spans.find((span) => span.attributes["edit.strategy_id"] === "edit.whole-file.acp-write.v1");

    expect(result).toMatchObject({
      ok: true,
      editStrategyId: "edit.whole-file.acp-write.v1",
      editStatus: "applied",
    });
    expect(writes).toHaveLength(1);
    expect(writes[0]?.content).toContain("value = 2");
    expect(serialized).toContain("Preview edit strategy");
    expect(serialized).toContain("edit.whole-file.acp-write.v1");
    expect(editSpan?.attributes["inference.observation_kind"]).toBe("EDIT");
    expect(editSpan?.attributes["edit.verification_status"]).toBe("not_run");
    expect(editSpan?.attributes["edit.changed_file_count"]).toBe(1);
  });

  test("applies routed edit operation envelopes through ACP writes and traces strategy lineage", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-envelope-"));
    const updates: unknown[] = [];
    const writes: Array<{ path: string; content: string }> = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        writeTextFile: async (input: { path: string; content: string }) => {
          writes.push(input);
          return {};
        },
      } as never,
      cwd,
    );
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    const session = requireAgentSessionForTest(agent, sessionResponse.sessionId);
    const config = defaultConfig();
    const telemetry = new RunTelemetry(config, "test-acp-edit-envelope", cwd, session.optimizerPin.telemetry);
    const oldContent = "export const value = 1;\n";

    const results = await previewAndWriteClientEditThroughAgentForTest<Array<{ ok: boolean; editStrategyId: string; editStatus: string; newContent?: string }>>(agent, {
      session,
      telemetry,
      fileSnapshots: [{
        path: join(cwd, "example.ts"),
        relativePath: "example.ts",
        content: oldContent,
        hash: "old-hash",
      }],
      edit: {
        reason: "exercise routed apply_patch edit envelope",
        editInput: {
          strategyFamily: "apply_patch",
          payload: {
            patch: [
              "*** Begin Patch",
              "*** Update File: example.ts",
              "@@",
              "-export const value = 1;",
              "+export const value = 3;",
              "*** End Patch",
            ].join("\n"),
          },
        },
        targetFiles: ["example.ts"],
        editStrategyId: "edit.apply-patch.v1",
        editStrategyFamily: "apply_patch",
        renderedEditToolContractId: "rendered.edit.apply-patch.v1.model.local.test.rendered-edit-contract.v1",
        fallbackFromStrategyId: "edit.unified-diff.v1",
        fallbackToStrategyId: "edit.apply-patch.v1",
        fallbackTrigger: "apply_failed",
      },
    });

    const spans = readHaloSpans(config, cwd);
    const editSpan = spans.find((span) => span.attributes["edit.strategy_id"] === "edit.apply-patch.v1");

    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({
      ok: true,
      editStrategyId: "edit.apply-patch.v1",
      editStatus: "applied",
    });
    expect(results[0]?.newContent).toContain("value = 3");
    expect(writes).toHaveLength(1);
    expect(writes[0]?.content).toContain("value = 3");
    expect(JSON.stringify(updates)).toContain("Preview edit strategy");
    expect(editSpan?.attributes["edit.strategy_family"]).toBe("apply_patch");
    expect(editSpan?.attributes["edit.rendered_tool_contract_id"]).toBe(
      "rendered.edit.apply-patch.v1.model.local.test.rendered-edit-contract.v1",
    );
    expect(editSpan?.attributes["edit.fallback_from_strategy_id"]).toBe("edit.unified-diff.v1");
    expect(editSpan?.attributes["edit.fallback_to_strategy_id"]).toBe("edit.apply-patch.v1");
    expect(editSpan?.attributes["edit.phase.fallback.status"]).toBe("passed");
  });

  test("records malformed edit payloads as first-class parse-failure edit attempts", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-parse-failure-"));
    const agent = new BleedingAcpAgent({ sessionUpdate: async () => {} } as never, cwd);
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    const session = requireAgentSessionForTest(agent, sessionResponse.sessionId);
    const editContext = resolveLiveEditContext(session, []);
    const config = defaultConfig();
    const telemetry = new RunTelemetry(config, "test-acp-edit-parse-failure", cwd, session.optimizerPin.telemetry);

    telemetry.recordEditAttempt(editAttemptFromParseFailure({
      session,
      editContext,
      parseFailure: "edit 1: Required field content is missing",
    }));

    const spans = readHaloSpans(config, cwd);
    const editSpan = spans.find((span) => span.attributes["edit.phase.parse.status"] === "failed");

    expect(editSpan?.attributes["inference.observation_kind"]).toBe("EDIT");
    expect(editSpan?.attributes["edit.phase.parse.error_code"]).toBe("schema_validation_error");
    expect(editSpan?.attributes["edit.error_codes"]).toContain("schema_validation_error");
    expect(editSpan?.attributes["edit.changed_file_count"]).toBe(0);
  });

  test("builds live ACP replay captures with session lineage, file reads, edits, tools, and terminal records", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-live-replay-"));
    const agent = new BleedingAcpAgent({ sessionUpdate: async () => {} } as never, cwd);
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    const session = requireAgentSessionForTest(agent, sessionResponse.sessionId);
    const content = "export const value = 1;\n";
    const editContext = resolveLiveEditContext(session, []);
    const editAttempt = editAttemptFromParseFailure({
      session,
      editContext,
      parseFailure: "model emitted malformed edit JSON",
    });

    const capture: AcpReplayCapture = buildCodingReplayCapture({
      session,
      runId: "acp-code-live-replay-test",
      task: "Fix src/example.ts and verify it.",
      tracePath: ".bag/runs/acp-code-live-replay-test/coding-trace.json",
      fileSnapshots: [{
        path: join(cwd, "src/example.ts"),
        relativePath: "src/example.ts",
        content,
        hash: createHash("sha256").update(content).digest("hex"),
      }],
      editAttempts: [editAttempt],
      toolMetrics: [metricTool("mcp.workspace.read_file", false)],
      commandResults: [{
        command: "npm",
        args: ["run", "typecheck"],
        reason: "verify edited code",
        exitCode: 1,
        signal: null,
        output: "typecheck failed",
      }],
      artifactRefs: [".bag/runs/acp-code-live-replay-test/coding-trace.json"],
    });

    const groups = groupAcpReplayRecords(capture);
    const skeleton = extractReplayEvalCaseSkeleton({
      capture,
      metadata: {
        evalCaseId: "replay.eval.live-acp.generated",
        title: "Generated live ACP replay capture",
        expectedBehavior: {
          summary: "The replay must preserve live coding lineage and observed failures.",
        },
        oracleStrength: "medium",
      },
    });

    expect(capture.context.modelProfileId).toBeDefined();
    expect(capture.context.modelServerProfileId).toBeDefined();
    expect(capture.context.acpConsumerCapabilities).toMatchObject({
      fsReadTextFile: true,
      fsWriteTextFile: true,
      terminal: true,
    });
    expect(groups.prompts).toHaveLength(1);
    expect(groups.fileReads[0]).toMatchObject({
      path: "src/example.ts",
      redactionStatus: "hash_only",
    });
    expect(groups.editAttempts).toHaveLength(1);
    expect(groups.toolCalls[0]).toMatchObject({
      name: "mcp.workspace.read_file",
      status: "failed",
    });
    expect(groups.terminalCommands[0]).toMatchObject({
      command: ["npm", "run", "typecheck"],
      status: "failed",
      errorCode: "verifier_error",
    });
    expect(skeleton.sourceSessionId).toBe(sessionResponse.sessionId);
    expect(skeleton.observedFailures.map((failure) => failure.failureKind)).toEqual(
      expect.arrayContaining(["tool_call", "terminal_command"]),
    );
  });

  test("runs MCP tool calls through ACP updates, permission policy, and optimizer telemetry", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-live-mcp-"));
    const updates: unknown[] = [];
    const permissionRequests: unknown[] = [];
    let executorCalls = 0;
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        requestPermission: async (request: unknown) => {
          permissionRequests.push(request);
          return { outcome: { outcome: "selected", optionId: "reject" } };
        },
      } as never,
      cwd,
    );
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    const session = requireAgentSessionForTest(agent, sessionResponse.sessionId);
    const telemetry = new RunTelemetry(defaultConfig(), "test-acp-live-mcp", cwd, session.optimizerPin.telemetry);
    const server = {
      serverId: "workspace",
      name: "workspace",
      displayName: "Workspace tools",
      tools: [
        {
          name: "read_file",
          description: "Read a file from the workspace.",
          inputSchema: {
            type: "object",
            properties: { path: { type: "string" } },
            required: ["path"],
          },
          annotations: { readOnlyHint: true },
        },
        {
          name: "write_file",
          description: "Write a file in the workspace.",
          inputSchema: {
            type: "object",
            properties: { path: { type: "string" }, content: { type: "string" } },
            required: ["path", "content"],
          },
          annotations: { destructiveHint: true },
        },
      ],
    };

    const success = await agent.runLiveMcpToolCall({
      session,
      telemetry,
      server,
      call: { toolName: "read_file", arguments: { path: "README.md" } },
      executor: async (request) => {
        executorCalls += 1;
        return { content: `read ${request.arguments.path}` };
      },
    });
    session.yolo = false;
    const denied = await agent.runLiveMcpToolCall({
      session,
      telemetry,
      server,
      call: { toolName: "write_file", arguments: { path: "out.txt", content: "blocked" } },
      executor: async () => {
        executorCalls += 1;
        return { written: true };
      },
    });
    session.yolo = true;
    const malformed = await agent.runLiveMcpToolCall({
      session,
      telemetry,
      server,
      call: { toolName: "read_file", arguments: {} },
      executor: async () => {
        executorCalls += 1;
        return { unreachable: true };
      },
    });
    const retryExhausted = await agent.runLiveMcpToolCall({
      session,
      telemetry,
      server,
      call: { toolName: "read_file", arguments: { path: "README.md" }, retryCount: 3 },
      executor: async () => {
        executorCalls += 1;
        return { unreachable: true };
      },
    });
    const oversized = await agent.runLiveMcpToolCall({
      session,
      telemetry,
      server,
      call: { toolName: "read_file", arguments: { path: "README.md" } },
      executor: async () => {
        executorCalls += 1;
        return { content: "x".repeat(100_000) };
      },
    });

    expect(success.ok).toBe(true);
    expect(success.call.modelFacingToolName).toContain("mcp_workspace_read_file");
    expect(denied.ok).toBe(false);
    expect(denied.failureCode).toBe("permission_denied");
    expect(malformed.ok).toBe(false);
    expect(malformed.failureCode).toBe("schema_mismatch");
    expect(retryExhausted.ok).toBe(false);
    expect(retryExhausted.failureCode).toBe("retry_exhausted");
    expect(oversized.ok).toBe(true);
    expect(oversized.failureCode).toBe("oversized_output");
    expect(oversized.metrics.truncated).toBe(true);
    expect(permissionRequests).toHaveLength(1);
    expect(executorCalls).toBe(2);
    expect(telemetry.toolMetrics.map((metric) => metric.namespace)).toEqual(["mcp", "mcp", "mcp", "mcp", "mcp"]);
    expect(telemetry.toolMetrics[0]).toMatchObject({ ok: true, resultKind: "json" });
    expect(telemetry.toolMetrics[1]).toMatchObject({ ok: false, errorName: "permission_denied" });
    expect(telemetry.toolMetrics.slice(2).map((metric) => metric.errorName)).toEqual([
      "invalid_arguments",
      "retry_exhausted",
      undefined,
    ]);
    const toolUpdates = sessionUpdates<{ sessionUpdate: string; status?: string; rawOutput?: Record<string, unknown> }>(
      updates,
      "tool_call_update",
    );
    expect(toolUpdates.map((update) => update.status)).toEqual(["completed", "failed", "failed", "failed", "completed"]);
    expect(JSON.stringify(toolUpdates)).toContain("permission_denied");
    expect(JSON.stringify(toolUpdates)).toContain("oversized_output");
  });

  test("records final edit lifecycle telemetry for verification failure and rollback", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-lifecycle-"));
    const updates: unknown[] = [];
    const writes: Array<{ path: string; content: string }> = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        writeTextFile: async (input: { path: string; content: string }) => {
          writes.push(input);
          return {};
        },
      } as never,
      cwd,
    );
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    const session = requireAgentSessionForTest(agent, sessionResponse.sessionId);
    const config = defaultConfig();
    const telemetry = new RunTelemetry(config, "test-acp-edit-lifecycle", cwd, session.optimizerPin.telemetry);
    const originalContent = "export const value = 1;\n";
    const editedContent = "export const value = 2;\n";
    const originalHash = createHash("sha256").update(originalContent).digest("hex");

    const editResults = await previewAndWriteClientEditThroughAgentForTest<Array<{
      path: string;
      ok: boolean;
      editStrategyId: string;
      editStatus: string;
      newHash?: string;
      editAttempt?: unknown;
    }>>(agent, {
      session,
      telemetry,
      fileSnapshots: [{
        path: join(cwd, "example.ts"),
        relativePath: "example.ts",
        content: originalContent,
        hash: originalHash,
      }],
      edit: {
        reason: "exercise lifecycle telemetry",
        editInput: {
          strategyFamily: "whole_file",
          payload: {
            path: "example.ts",
            content: editedContent,
            intent: "exercise lifecycle telemetry",
          },
        },
        targetFiles: ["example.ts"],
        editStrategyId: "edit.whole-file.lifecycle-test.v1",
        editStrategyFamily: "whole_file",
        renderedEditToolContractId: "rendered.edit.whole-file.lifecycle-test.v1",
      },
    });
    const editedHash = editResults[0]?.newHash;
    expect(editedHash).toBeDefined();

    const finalized = recordFinalEditLifecycleTelemetryThroughAgentForTest(agent, {
      session,
      telemetry,
      editResults,
      postApplyChecks: [{
        path: "example.ts",
        status: "consistent",
        expectedHash: editedHash,
        actualHash: editedHash,
        reason: "client file content matches the written edit hash",
      }],
      commandResults: [{
        command: "npm",
        args: ["test"],
        reason: "verification",
        exitCode: 1,
        signal: null,
        output: "failing test output",
      }],
      rollbackResults: [{
        path: join(cwd, "example.ts"),
        ok: true,
        reason: "rollback",
        oldHash: editedHash,
        newHash: originalHash,
        editStrategyId: "edit.rollback.acp-write.v1",
        editStatus: "rollback_applied",
      }],
      artifactRefs: ["artifact:command-results.json", "artifact:rollback-results.json"],
    });

    const spans = readHaloSpans(config, cwd);
    const finalSpan = spans.find(
      (span) =>
        span.attributes["edit.strategy_id"] === "edit.whole-file.lifecycle-test.v1" &&
        span.attributes["edit.verification_status"] === "failed",
    );

    expect(finalized[0]).toMatchObject({
      verificationStatus: "failed",
      postApplyConsistencyStatus: "consistent",
      selfDetectedRegressionStatus: "suspected",
      rollbackStatus: "succeeded",
    });
    expect(finalSpan?.attributes["edit.phase.verify.status"]).toBe("failed");
    expect(finalSpan?.attributes["edit.phase.verify.error_code"]).toBe("verifier_error");
    expect(finalSpan?.attributes["edit.phase.rollback.status"]).toBe("passed");
    expect(finalSpan?.attributes["edit.rollback_status"]).toBe("succeeded");
    expect(finalSpan?.attributes["edit.error_codes"]).toContain("verifier_error");
    expect(writes).toHaveLength(1);
  });

  test("rolls back changed live files through the ACP write boundary", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-rollback-"));
    const updates: unknown[] = [];
    const writes: Array<{ path: string; content: string }> = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        writeTextFile: async (input: { path: string; content: string }) => {
          writes.push(input);
          return {};
        },
      } as never,
      cwd,
    );
    const sessionResponse = await agent.newSession({ cwd, mcpServers: [] });
    const session = requireAgentSessionForTest(agent, sessionResponse.sessionId);
    const config = defaultConfig();
    const telemetry = new RunTelemetry(config, "test-acp-edit-rollback", cwd, session.optimizerPin.telemetry);
    const baselineContent = "export const value = 1;\n";
    const currentContent = "export const value = 2;\n";

    const rollbackResults = await rollbackLiveEditsThroughAgentForTest(agent, {
      session,
      telemetry,
      baselineFileSnapshots: [{
        path: join(cwd, "example.ts"),
        relativePath: "example.ts",
        content: baselineContent,
        hash: createHash("sha256").update(baselineContent).digest("hex"),
      }],
      currentFileSnapshots: [{
        path: join(cwd, "example.ts"),
        relativePath: "example.ts",
        content: currentContent,
        hash: createHash("sha256").update(currentContent).digest("hex"),
      }],
      editResults: [{
        path: join(cwd, "example.ts"),
        ok: true,
        reason: "edited",
        oldHash: createHash("sha256").update(baselineContent).digest("hex"),
        newHash: createHash("sha256").update(currentContent).digest("hex"),
        editStrategyId: "edit.whole-file.rollback-test.v1",
        editStatus: "applied",
        newContent: currentContent,
      }],
    });

    expect(rollbackResults).toEqual([
      expect.objectContaining({
        ok: true,
        editStrategyId: "edit.rollback.acp-write.v1",
        editStatus: "rollback_applied",
        newContent: baselineContent,
      }),
    ]);
    expect(writes).toEqual([expect.objectContaining({ path: join(cwd, "example.ts"), content: baselineContent })]);
    expect(JSON.stringify(updates)).toContain("Rollback after unrepaired verification or post-apply consistency failure");
  });

  test("supports ACP slash commands for safe mode and MCP/skills visibility", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-commands-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({
      cwd,
      mcpServers: [{ name: "local-tools", command: "node", args: ["server.js"], env: [] }],
    });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/safe" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/mcp" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/traces" }],
    });

    const serialized = JSON.stringify(updates);
    expect(serialized).toContain('"sessionUpdate":"available_commands_update"');
    expect(serialized).toContain("Safe mode enabled");
    expect(serialized).toContain("local-tools");
    expect(serialized).toContain("HALO-style trace dataset");
  });

  test("supports hidden maintenance slash commands without exposing them in normal command surface", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance status" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance eval" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance promote candidate.test" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance rollback" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance optimize" }],
    });

    const names = availableCommandNames(updates);
    expect(names).not.toContain("maintenance");
    expect(names).not.toContain("optimize");
    expect(names).not.toContain("promote");
    expect(names).not.toContain("rollback");

    const text = agentMessageText(updates);
    expect(text).toContain("Maintenance optimizer status");
    expect(text).toContain("Current session pin");
    expect(text).toContain("Maintenance eval summary");
    expect(text).toContain("holdout usage");
    expect(text).toContain("Maintenance promote inspection");
    expect(text).toContain("side effects: none");
    expect(text).toContain("dry-run status: blocked");
    expect(text).toContain("Maintenance rollback inspection");
    expect(text).toContain("rollback readiness: blocked_no_checkpoint");
    expect(text).toContain("Maintenance optimize report");
    expect(text).toContain("safe existing hook: optimizePolicy");
    expect(
      updates
        .map((update) => (update as { update?: { sessionUpdate?: string; currentModeId?: string } }).update)
        .filter((update) => update?.sessionUpdate === "current_mode_update")
        .map((update) => update?.currentModeId),
    ).toEqual(["plan", "auto", "plan", "auto", "plan", "auto", "plan", "auto", "plan", "auto"]);
  });

  test("surfaces maintenance promotion dry-run and rollback readiness without applying side effects", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-readiness-"));
    const updates: unknown[] = [];
    const config = defaultConfig();
    const now = "2026-05-01T10:00:00.000Z";
    const candidate: CandidatePatch = {
      candidatePatchId: "candidate.maintenance.dryrun",
      policyId: "policy.maintenance",
      modelProfileId: "model.local.maintenance",
      codebaseProfileId: "codebase.maintenance",
      scope: {
        artifactKind: "model_codebase_policy",
        artifactId: "policy.maintenance",
        allowedJsonPointers: ["/verificationGates/0"],
      },
      operations: [
        {
          op: "add",
          path: "/verificationGates/0",
          value: {
            gateId: "tool-call-success-rate",
            metric: "tool-call-success-rate",
            comparator: "gte",
            threshold: 0.95,
            required: true,
          },
        },
      ],
      rationale: "Tighten tool-call gate after observed failures.",
      createdAt: now,
      sourceTraceIds: ["trace.maintenance"],
    };
    const records: OptimizerRegistryRecord[] = [
      {
        registryRecordId: "registry.candidate.maintenance.dryrun",
        recordKind: "candidate_patch",
        schemaVersion: "optimizer-schema.v1",
        recordVersion: "record.v1",
        status: "draft",
        createdAt: now,
        updatedAt: now,
        labels: ["maintenance-test"],
        payload: candidate,
      },
      {
        registryRecordId: "registry.promotion.maintenance.dryrun",
        recordKind: "promotion_decision",
        schemaVersion: "optimizer-schema.v1",
        recordVersion: "record.v1",
        status: "promoted",
        createdAt: now,
        updatedAt: now,
        labels: ["maintenance-test"],
        payload: {
          promotionDecisionId: "promotion.maintenance.dryrun",
          decision: "promote",
          policyId: candidate.policyId,
          candidatePatchId: candidate.candidatePatchId,
          evalResultId: "eval.maintenance.dryrun",
          modelProfileId: candidate.modelProfileId,
          codebaseProfileId: candidate.codebaseProfileId,
          canonicalToolVersion: "canonical-tools.v1",
          renderedToolVersion: "rendered-tools.v1",
          resultStyleVersion: "result-style.v1",
          verificationPolicyVersion: "verification.v1",
          reason: "Candidate passed dry-run gates in fixture.",
          decidedAt: now,
          decidedBy: "deterministic_gate",
          appliesToNewSessionsOnly: true,
        },
      },
    ];
    for (const record of records) {
      saveOptimizerRegistryRecord(config, record, cwd);
    }
    const checkpointDir = optimizerRegistryCheckpointsDir(config, cwd);
    mkdirSync(checkpointDir, { recursive: true });
    const checkpointFile = "2026-05-01T10:00:00.000Z.candidate.maintenance.dryrun.json";
    writeFileSync(
      join(checkpointDir, checkpointFile),
      `${JSON.stringify({
        candidatePatchId: candidate.candidatePatchId,
        createdAt: now,
        previousPointer: {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: "model.previous",
          activeCodebaseProfileId: "codebase.previous",
          activePolicyId: "policy.previous",
          promotedAt: "2026-04-30T10:00:00.000Z",
        },
      })}\n`,
    );

    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
      config,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance promote candidate.maintenance.dryrun" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: `/maintenance rollback ${checkpointFile}` }],
    });

    const text = agentMessageText(updates);
    expect(text).toContain("dry-run status: ready_for_operator_review");
    expect(text).toContain("readiness blockers: none");
    expect(text).toContain("scope: model_codebase_policy policy.maintenance");
    expect(text).toContain("operation count: 1");
    expect(text).toContain("promotion decisions: 1");
    expect(text).toContain("last decision: promote");
    expect(text).toContain("new sessions only: true");
    expect(text).toContain("rollback readiness: ready_for_operator_review");
    expect(text).toContain("previous pointer available: yes");
    expect(text).toContain("checkpoint candidate: candidate.maintenance.dryrun");
    expect(existsSync(join(cwd, ".bag", "optimizer", "active.json"))).toBe(false);
  });

  test("returns auto sessions from temporary maintenance mode while preserving manual modes", async () => {
    const currentModeIds = (updates: unknown[]) =>
      updates
        .map((update) => (update as { update?: { sessionUpdate?: string; currentModeId?: string } }).update)
        .filter((update) => update?.sessionUpdate === "current_mode_update")
        .map((update) => update?.currentModeId);

    const autoCwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-auto-mode-"));
    const autoUpdates: unknown[] = [];
    const autoAgent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          autoUpdates.push(update);
        },
      } as never,
      autoCwd,
    );
    const autoSession = await autoAgent.newSession({ cwd: autoCwd, mcpServers: [] });

    await autoAgent.prompt({
      sessionId: autoSession.sessionId,
      prompt: [{ type: "text", text: "/maintenance status" }],
    });
    await autoAgent.prompt({
      sessionId: autoSession.sessionId,
      prompt: [{ type: "text", text: "/metrics" }],
    });

    expect(currentModeIds(autoUpdates)).toEqual(["plan", "auto"]);
    expect(agentMessageText(autoUpdates)).toContain("Current mode: auto");

    for (const manualMode of ["chat", "plan", "run"] as const) {
      const cwd = mkdtempSync(join(tmpdir(), `bag-acp-maintenance-${manualMode}-mode-`));
      const updates: unknown[] = [];
      const agent = new BleedingAcpAgent(
        {
          sessionUpdate: async (update: unknown) => {
            updates.push(update);
          },
        } as never,
        cwd,
      );
      const session = await agent.newSession({ cwd, mcpServers: [] });
      await agent.setSessionMode({ sessionId: session.sessionId, modeId: manualMode });
      updates.length = 0;

      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "/maintenance eval" }],
      });
      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "/metrics" }],
      });

      expect(currentModeIds(updates)).toEqual([]);
      expect(agentMessageText(updates)).toContain(`Current mode: ${manualMode}`);
    }
  });

  test("emits compact ACP progress for maintenance status eval and optimize", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-maintenance-progress-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance status" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance eval" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance optimize" }],
    });

    const toolCalls = sessionUpdates<{ sessionUpdate: string; title?: string; kind?: string; status?: string }>(
      updates,
      "tool_call",
    );
    const toolCallUpdates = sessionUpdates<{
      sessionUpdate: string;
      status?: string;
      rawOutput?: Record<string, unknown>;
    }>(updates, "tool_call_update");
    const plans = sessionUpdates<{ sessionUpdate: string; entries?: Array<{ content?: string; status?: string }> }>(
      updates,
      "plan",
    );

    expect(toolCalls.map((call) => call.title)).toEqual([
      "Inspect maintenance status",
      "Inspect maintenance eval splits",
      "Compute maintenance optimize report",
    ]);
    expect(toolCalls.every((call) => call.kind === "think" && call.status === "pending")).toBe(true);
    expect(toolCallUpdates.filter((update) => update.status === "completed")).toHaveLength(3);
    expect(JSON.stringify(toolCallUpdates)).toContain("evaluatedRuns");
    expect(JSON.stringify(plans)).toContain("Inspect optimizer registry and session pin");
    expect(JSON.stringify(plans)).toContain("Read configured eval split metadata");
    expect(JSON.stringify(plans)).toContain("Compute bounded optimization recommendation");
    expect(plans.filter((plan) => plan.entries?.every((entry) => entry.status === "completed")).length).toBeGreaterThanOrEqual(
      3,
    );
    expect(agentMessageText(updates)).toContain("Maintenance optimize report");
  });

  test("does not emit maintenance progress for normal chat or coding routes", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-no-maintenance-progress-"));
    const updates: unknown[] = [];
    let codingRuns = 0;
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    replaceRunCodingTurnForTest(agent, async () => {
      codingRuns += 1;
    });
    const session = await agent.newSession({ cwd, mcpServers: [] });

    await agent.setSessionMode({ sessionId: session.sessionId, modeId: "chat" });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "Ahoj, co umíš?" }],
    });
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/run oprav drobnost" }],
    });

    expect(codingRuns).toBe(1);
    const serialized = JSON.stringify(updates);
    expect(serialized).not.toContain("bag.maintenance");
    expect(serialized).not.toContain("Inspect maintenance status");
    expect(serialized).not.toContain("Compute maintenance optimize report");
    expect(serialized).not.toContain("Read configured eval split metadata");
  });

  test("background optimization trigger no-ops when evidence is insufficient", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-bg-noop-"));
    const agent = new BleedingAcpAgent({ sessionUpdate: async () => {} } as never, cwd);
    const session = await agent.newSession({ cwd, mcpServers: [] });

    const diagnostic = inspectBackgroundOptimizationTrigger(defaultConfig(), requireAgentSessionForTest(agent, session.sessionId), {
      source: "test",
      sourceRunId: "run-insufficient",
      enqueue: true,
    });

    expect(diagnostic.triggered).toBe(false);
    expect(diagnostic.reason).toContain("insufficient evidence");
    expect(diagnostic.sideEffects).toEqual([]);
    expect(diagnostic.opportunityPath).toBeUndefined();
    expect(existsSync(join(cwd, ".bag", "maintenance", "opportunities.jsonl"))).toBe(false);
  });

  test("background optimization trigger queues opportunity when enough metrics and traces exist", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-bg-trigger-"));
    const config = defaultConfig();
    mkdirSync(join(cwd, ".bag", "telemetry"), { recursive: true });
    writeFileSync(
      join(cwd, ".bag", "telemetry", "metrics.json"),
      `${JSON.stringify(
        {
          run_1: {
            steps: [
              metricStep("context.scout", true),
              metricStep("context.build", true),
              metricStep("verify", false),
            ],
            llmCalls: [metricLlm(true), metricLlm(true)],
            toolCalls: [metricTool("repo.read", true), metricTool("repo.write", false)],
          },
          run_2: {
            steps: [metricStep("context.scout", true), metricStep("context.build", true)],
            llmCalls: [metricLlm(true)],
            toolCalls: [metricTool("repo.read", true), metricTool("terminal.run", true)],
          },
        },
        null,
        2,
      )}\n`,
    );
    writeFileSync(
      join(cwd, ".bag", "telemetry", "spans.jsonl"),
      Array.from({ length: 6 }, (_, index) =>
        JSON.stringify({
          trace_id: "trace-bg",
          span_id: `span-${index}`,
          parent_span_id: "root",
          trace_state: "",
          name: index === 0 ? "tool.workspace.repo.write" : "step.context",
          kind: "SPAN_KIND_CLIENT",
          start_time: "2026-04-29T00:00:00.000Z",
          end_time: "2026-04-29T00:00:01.000Z",
          status: { code: index === 0 ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK", message: index === 0 ? "write failed" : "" },
          resource: { attributes: { "service.name": "bleeding-agent" } },
          scope: { name: "bag.telemetry", version: "0.1.0" },
          attributes: {
            "inference.observation_kind": index === 0 ? "TOOL" : "CHAIN",
            "inference.project_id": "bleeding-agent",
            "tool.name": index === 0 ? "repo.write" : undefined,
            "error.message": index === 0 ? "write failed" : undefined,
          },
        }),
      ).join("\n") + "\n",
    );
    const agent = new BleedingAcpAgent({ sessionUpdate: async () => {} } as never, cwd);
    const session = await agent.newSession({ cwd, mcpServers: [] });

    const diagnostic = inspectBackgroundOptimizationTrigger(config, requireAgentSessionForTest(agent, session.sessionId), {
      source: "test",
      sourceRunId: "run-enough",
      enqueue: true,
    });

    expect(diagnostic.triggered).toBe(true);
    expect(diagnostic.evidence.runCount).toBe(2);
    expect(diagnostic.evidence.metricObservationCount).toBeGreaterThanOrEqual(12);
    expect(diagnostic.evidence.spanCount).toBe(6);
    expect(diagnostic.evidence.errorSpanCount).toBe(1);
    expect(diagnostic.sideEffects).toEqual(["append-maintenance-opportunity"]);
    expect(diagnostic.opportunityPath).toBe(join(cwd, ".bag", "maintenance", "opportunities.jsonl"));
    const opportunity = readFileSync(diagnostic.opportunityPath, "utf8");
    expect(opportunity).toContain("background-optimization-opportunity");
    expect(opportunity).toContain("/maintenance optimize");
    expect(opportunity).toContain("no automatic promotion");
    expect(existsSync(join(cwd, ".bag", "optimizer", "active.json"))).toBe(false);
    expect(config.artifactDir).toBe(".bag");
  });

  test("background optimization status inspection does not promote or mutate active policy", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-bg-status-"));
    const updates: unknown[] = [];
    const agent = new BleedingAcpAgent(
      {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
      } as never,
      cwd,
    );
    const session = await agent.newSession({ cwd, mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/maintenance status" }],
    });

    const text = agentMessageText(updates);
    expect(text).toContain("Background optimization trigger");
    expect(text).toContain("side effects: none during status inspection");
    expect(existsSync(join(cwd, ".bag", "maintenance", "opportunities.jsonl"))).toBe(false);
    expect(existsSync(join(cwd, ".bag", "optimizer", "active.json"))).toBe(false);
  });
});

const metricStep = (step: string, ok: boolean) => ({
  step,
  startedAt: "2026-04-29T00:00:00.000Z",
  completedAt: "2026-04-29T00:00:01.000Z",
  durationMs: 1000,
  ok,
  modelRole: "deterministic" as const,
  ...(ok ? {} : { error: `${step} failed` }),
});

const metricLlm = (ok: boolean) => ({
  role: "local" as const,
  model: "local-model",
  endpoint: "http://127.0.0.1:18082/v1/chat/completions",
  startedAt: "2026-04-29T00:00:00.000Z",
  completedAt: "2026-04-29T00:00:01.000Z",
  durationMs: 1000,
  ok,
  totalTokens: 42,
  ...(ok ? {} : { error: "llm failed" }),
});

const metricTool = (toolName: string, ok: boolean) => ({
  toolName,
  namespace: "workspace",
  descriptionVersion: "v1",
  startedAt: "2026-04-29T00:00:00.000Z",
  completedAt: "2026-04-29T00:00:01.000Z",
  durationMs: 1000,
  ok,
  retryCount: 0,
  argumentBytes: 12,
  argumentHash: `${toolName}-args`,
  resultBytes: ok ? 24 : undefined,
  resultKind: ok ? ("text" as const) : ("unknown" as const),
  ...(ok ? {} : { error: `${toolName} failed`, errorName: "Error" }),
});
