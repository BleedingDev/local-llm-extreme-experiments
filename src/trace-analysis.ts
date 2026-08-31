import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { BagConfig } from "./types";
import type { HaloSpan } from "./telemetry";

export type TraceOptimizerDimensions = {
  modelProfileIds: string[];
  codebaseProfileIds: string[];
  policyIds: string[];
  canonicalToolVersions: string[];
  renderedToolVersions: string[];
  resultStyleVersions: string[];
  verificationPolicyVersions: string[];
  editStrategyVersions: string[];
  renderedEditContractVersions: string[];
  editFallbackPolicyVersions: string[];
  editRepairPolicyVersions: string[];
  editVerifierPolicyVersions: string[];
  editObjectiveSetIds: string[];
  editStrategyIds: string[];
  editStrategyFamilies: string[];
  canonicalEditToolSpecIds: string[];
  renderedEditToolContractIds: string[];
};

export type TraceFailureCluster = {
  name: string;
  observationKind: string;
  count: number;
  traces: string[];
  messages: string[];
  inputHashes: string[];
  optimizerDimensions: TraceOptimizerDimensions;
};

export type TraceLatencyCluster = {
  name: string;
  observationKind: string;
  count: number;
  p50Ms: number;
  p95Ms: number;
  optimizerDimensions: TraceOptimizerDimensions;
};

export type TraceAnalysisReport = {
  spanCount: number;
  traceCount: number;
  errorSpanCount: number;
  observationKinds: Record<string, number>;
  optimizerDimensions: TraceOptimizerDimensions;
  failureClusters: TraceFailureCluster[];
  latencyClusters: TraceLatencyCluster[];
};

const percentile = (values: number[], ratio: number): number => {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1));
  return sorted[index] ?? 0;
};

const firstLine = (value: unknown): string => String(value ?? "").split("\n")[0]?.slice(0, 180) ?? "";

const spanDurationMs = (span: HaloSpan): number => {
  const explicit = span.attributes["inference.duration_ms"];
  if (typeof explicit === "number" && Number.isFinite(explicit)) {
    return explicit;
  }
  const started = Date.parse(span.start_time);
  const ended = Date.parse(span.end_time);
  return Number.isFinite(started) && Number.isFinite(ended) ? Math.max(0, ended - started) : 0;
};

const observationKind = (span: HaloSpan): string => {
  const kind = span.attributes["inference.observation_kind"] ?? span.attributes["openinference.span.kind"];
  return typeof kind === "string" && kind.length > 0 ? kind : "SPAN";
};

const groupKey = (span: HaloSpan): string => `${observationKind(span)}:${span.name}`;

const unique = (values: string[], limit: number): string[] => [...new Set(values.filter(Boolean))].slice(0, limit);

const uniqueSorted = (values: string[], limit: number): string[] =>
  [...new Set(values.filter(Boolean))].sort().slice(0, limit);

const optimizerDimensions = (spans: HaloSpan[], limit = 50): TraceOptimizerDimensions => ({
  modelProfileIds: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.model_profile_id"] ?? "")),
    limit,
  ),
  codebaseProfileIds: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.codebase_profile_id"] ?? "")),
    limit,
  ),
  policyIds: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.policy_id"] ?? "")),
    limit,
  ),
  canonicalToolVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.canonical_tool_version"] ?? "")),
    limit,
  ),
  renderedToolVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.rendered_tool_version"] ?? "")),
    limit,
  ),
  resultStyleVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.result_style_version"] ?? "")),
    limit,
  ),
  verificationPolicyVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.verification_policy_version"] ?? "")),
    limit,
  ),
  editStrategyVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.edit_strategy_version"] ?? "")),
    limit,
  ),
  renderedEditContractVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.rendered_edit_contract_version"] ?? "")),
    limit,
  ),
  editFallbackPolicyVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.edit_fallback_policy_version"] ?? "")),
    limit,
  ),
  editRepairPolicyVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.edit_repair_policy_version"] ?? "")),
    limit,
  ),
  editVerifierPolicyVersions: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.edit_verifier_policy_version"] ?? "")),
    limit,
  ),
  editObjectiveSetIds: uniqueSorted(
    spans.map((span) => String(span.attributes["optimizer.edit_objective_set_id"] ?? "")),
    limit,
  ),
  editStrategyIds: uniqueSorted(
    spans.map((span) => String(span.attributes["edit.strategy_id"] ?? "")),
    limit,
  ),
  editStrategyFamilies: uniqueSorted(
    spans.map((span) => String(span.attributes["edit.strategy_family"] ?? "")),
    limit,
  ),
  canonicalEditToolSpecIds: uniqueSorted(
    spans.map((span) => String(span.attributes["edit.canonical_tool_spec_id"] ?? "")),
    limit,
  ),
  renderedEditToolContractIds: uniqueSorted(
    spans.map((span) => String(span.attributes["edit.rendered_tool_contract_id"] ?? "")),
    limit,
  ),
});

export const readHaloSpans = (config: BagConfig, cwd = process.cwd()): HaloSpan[] => {
  const path = resolve(cwd, config.telemetry.spans);
  if (!existsSync(path)) {
    return [];
  }
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as HaloSpan];
      } catch {
        return [];
      }
    });
};

export const analyzeHaloSpans = (spans: HaloSpan[]): TraceAnalysisReport => {
  const observationKinds: Record<string, number> = {};
  const traces = new Set<string>();
  const failed = spans.filter((span) => span.status.code === "STATUS_CODE_ERROR");
  const groupedFailures = new Map<string, HaloSpan[]>();
  const groupedLatencies = new Map<string, HaloSpan[]>();

  for (const span of spans) {
    traces.add(span.trace_id);
    const kind = observationKind(span);
    observationKinds[kind] = (observationKinds[kind] ?? 0) + 1;
    const key = groupKey(span);
    groupedLatencies.set(key, [...(groupedLatencies.get(key) ?? []), span]);
    if (span.status.code === "STATUS_CODE_ERROR") {
      groupedFailures.set(key, [...(groupedFailures.get(key) ?? []), span]);
    }
  }

  const failureClusters = [...groupedFailures.entries()]
    .map(([key, rows]) => {
      const [kind, ...nameParts] = key.split(":");
      return {
        name: nameParts.join(":"),
        observationKind: kind ?? "SPAN",
        count: rows.length,
        traces: unique(rows.map((span) => span.trace_id), 8),
        messages: unique(
          rows.flatMap((span) => [
            firstLine(span.status.message),
            firstLine(span.attributes["error.message"]),
            firstLine(span.attributes["error.type"]),
          ]),
          8,
        ),
        inputHashes: unique(rows.map((span) => String(span.attributes["input.hash"] ?? "")), 8),
        optimizerDimensions: optimizerDimensions(rows, 8),
      };
    })
    .sort((left, right) => right.count - left.count)
    .slice(0, 20);

  const latencyClusters = [...groupedLatencies.entries()]
    .map(([key, rows]) => {
      const [kind, ...nameParts] = key.split(":");
      const durations = rows.map(spanDurationMs);
      return {
        name: nameParts.join(":"),
        observationKind: kind ?? "SPAN",
        count: rows.length,
        p50Ms: Math.round(percentile(durations, 0.5)),
        p95Ms: Math.round(percentile(durations, 0.95)),
        optimizerDimensions: optimizerDimensions(rows, 8),
      };
    })
    .filter((cluster) => cluster.count >= 3 || cluster.p95Ms >= 10_000)
    .sort((left, right) => right.p95Ms - left.p95Ms)
    .slice(0, 20);

  return {
    spanCount: spans.length,
    traceCount: traces.size,
    errorSpanCount: failed.length,
    observationKinds,
    optimizerDimensions: optimizerDimensions(spans),
    failureClusters,
    latencyClusters,
  };
};

const renderDimension = (label: string, values: string[]): string =>
  values.length === 0 ? `- ${label}: none` : `- ${label}: ${values.join(", ")}`;

export const renderTraceAnalysisMarkdown = (report: TraceAnalysisReport): string =>
  [
    "## HALO-Style Trace Analysis",
    "",
    `Spans: ${report.spanCount}`,
    `Traces: ${report.traceCount}`,
    `Error spans: ${report.errorSpanCount}`,
    "",
    "Observation kinds:",
    ...Object.entries(report.observationKinds).map(([kind, count]) => `- ${kind}: ${count}`),
    "",
    "Optimizer dimensions:",
    renderDimension("modelProfileId", report.optimizerDimensions.modelProfileIds),
    renderDimension("codebaseProfileId", report.optimizerDimensions.codebaseProfileIds),
    renderDimension("policyId", report.optimizerDimensions.policyIds),
    renderDimension("canonicalToolVersion", report.optimizerDimensions.canonicalToolVersions),
    renderDimension("renderedToolVersion", report.optimizerDimensions.renderedToolVersions),
    renderDimension("resultStyleVersion", report.optimizerDimensions.resultStyleVersions),
    renderDimension("verificationPolicyVersion", report.optimizerDimensions.verificationPolicyVersions),
    renderDimension("editStrategyVersion", report.optimizerDimensions.editStrategyVersions),
    renderDimension("renderedEditContractVersion", report.optimizerDimensions.renderedEditContractVersions),
    renderDimension("editFallbackPolicyVersion", report.optimizerDimensions.editFallbackPolicyVersions),
    renderDimension("editRepairPolicyVersion", report.optimizerDimensions.editRepairPolicyVersions),
    renderDimension("editVerifierPolicyVersion", report.optimizerDimensions.editVerifierPolicyVersions),
    renderDimension("editObjectiveSetId", report.optimizerDimensions.editObjectiveSetIds),
    renderDimension("editStrategyId", report.optimizerDimensions.editStrategyIds),
    renderDimension("editStrategyFamily", report.optimizerDimensions.editStrategyFamilies),
    renderDimension("canonicalEditToolSpecId", report.optimizerDimensions.canonicalEditToolSpecIds),
    renderDimension("renderedEditToolContractId", report.optimizerDimensions.renderedEditToolContractIds),
    "",
    "Failure clusters:",
    report.failureClusters.length === 0
      ? "- none"
      : report.failureClusters
          .map((cluster) =>
            [
              `- ${cluster.observationKind} ${cluster.name}: ${cluster.count}`,
              cluster.messages.length > 0 ? `  messages: ${cluster.messages.join(" | ")}` : "",
              cluster.inputHashes.length > 0 ? `  input hashes: ${cluster.inputHashes.join(", ")}` : "",
            ]
              .filter(Boolean)
              .join("\n"),
          )
          .join("\n"),
    "",
    "Latency clusters:",
    report.latencyClusters.length === 0
      ? "- none"
      : report.latencyClusters
          .map(
            (cluster) =>
              `- ${cluster.observationKind} ${cluster.name}: count=${cluster.count} p50=${cluster.p50Ms}ms p95=${cluster.p95Ms}ms`,
          )
          .join("\n"),
    "",
  ].join("\n");
