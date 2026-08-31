import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash, randomBytes } from "node:crypto";
import { dirname, resolve } from "node:path";
import {
  EditAttemptContractSchema,
  REAL_EDIT_ATTEMPT_REQUIRED_PHASES,
  editAttemptCaptureIssues,
  editAttemptTargetHashRows,
  missingRequiredEditAttemptPhases,
  type EditAttemptContract,
  type EditFallbackPathStep,
  type EditPhaseResult,
} from "./edit-strategy/types";
import type {
  BagConfig,
  LlmCallMetric,
  ModelEndpointKind,
  ModelProvider,
  ModelProviderConfigRole,
  ModelRuntimeRole,
  ProviderDiscoverySource,
  SelfEvaluation,
  StepMetric,
  ToolCallMetric,
} from "./types";

type TelemetryEvent = {
  type: string;
  runId: string;
  timestamp: string;
  payload: Record<string, unknown>;
};

export type OptimizerSessionPinTelemetry = {
  modelRole?: ModelRuntimeRole;
  providerConfigRole?: ModelProviderConfigRole;
  fallbackModelRole?: ModelRuntimeRole;
  provider?: ModelProvider;
  endpointKind?: ModelEndpointKind;
  modelServerId?: string;
  modelServerProfileId?: string;
  providerDiscoverySource?: ProviderDiscoverySource;
  contextWindowTokens?: number;
  maxOutputTokens?: number;
  modelProfileId: string;
  codebaseProfileId: string;
  policyId: string;
  canonicalToolVersion: string;
  renderedToolVersion: string;
  resultStyleVersion: string;
  verificationPolicyVersion: string;
  editStrategyVersion: string;
  renderedEditContractVersion: string;
  editFallbackPolicyVersion: string;
  editRepairPolicyVersion: string;
  editVerifierPolicyVersion: string;
  editObjectiveSetId: string;
  source: "active_pointer" | "registry" | "seed";
  registryRoot?: string;
  registryErrorCount?: number;
  invalidRecordCount?: number;
  rendererId?: string;
  rendererVersion?: string;
  modelProfileRecordId?: string;
  codebaseProfileRecordId?: string;
  policyRecordId?: string;
};

export type HaloSpan = {
  trace_id: string;
  span_id: string;
  parent_span_id: string;
  trace_state: string;
  name: string;
  kind: "SPAN_KIND_INTERNAL" | "SPAN_KIND_CLIENT";
  start_time: string;
  end_time: string;
  status: {
    code: "STATUS_CODE_OK" | "STATUS_CODE_ERROR";
    message: string;
  };
  resource: {
    attributes: Record<string, unknown>;
  };
  scope: {
    name: string;
    version: string;
  };
  attributes: Record<string, unknown>;
};

const now = (): string => new Date().toISOString();

const appendJsonl = (path: string, event: TelemetryEvent): void => {
  mkdirSync(dirname(path), { recursive: true });
  appendFileSync(path, `${JSON.stringify(event)}\n`);
};

const appendRawJsonl = (path: string, value: unknown): void => {
  mkdirSync(dirname(path), { recursive: true });
  appendFileSync(path, `${JSON.stringify(value)}\n`);
};

export class RunTelemetry {
  readonly metrics: StepMetric[] = [];
  readonly llmMetrics: LlmCallMetric[] = [];
  readonly toolMetrics: ToolCallMetric[] = [];
  private readonly eventsPath: string;
  private readonly metricsPath: string;
  private readonly spansPath: string;
  private readonly traceId: string;
  private readonly rootSpanId: string;
  private readonly cwd: string;

  constructor(
    private readonly config: BagConfig,
    private readonly runId: string,
    cwd = process.cwd(),
    private readonly optimizerPin?: OptimizerSessionPinTelemetry,
  ) {
    this.cwd = cwd;
    this.eventsPath = resolve(cwd, config.telemetry.jsonl);
    this.metricsPath = resolve(cwd, config.telemetry.metrics);
    this.spansPath = resolve(cwd, config.telemetry.spans);
    this.traceId = hashString(`trace:${runId}`).slice(0, 32);
    this.rootSpanId = hashString(`root:${runId}`).slice(0, 16);
    this.recordHaloSpan({
      name: "agent.run",
      observationKind: "AGENT",
      startedAt: now(),
      completedAt: now(),
      ok: true,
      parentSpanId: "",
      attributes: {
        "inference.agent_name": "BleedingAgent",
        "inference.run_id": runId,
        "inference.cwd": cwd,
        ...this.optimizerHaloAttributes(),
      },
    });
  }

  event(type: string, payload: Record<string, unknown> = {}): void {
    if (!this.config.telemetry.enabled) {
      return;
    }
    appendJsonl(this.eventsPath, {
      type,
      runId: this.runId,
      timestamp: now(),
      payload: this.withOptimizerPayload(payload),
    });
  }

  async measure<T>(
    step: string,
    modelRole: StepMetric["modelRole"],
    fn: () => Promise<T>,
  ): Promise<T> {
    const startedAt = now();
    const startedMs = performance.now();
    this.event("step.started", { step, modelRole });
    try {
      const value = await fn();
      const completedAt = now();
      const metric: StepMetric = {
        step,
        startedAt,
        completedAt,
        durationMs: Math.round(performance.now() - startedMs),
        ok: true,
        modelRole,
      };
      this.metrics.push(metric);
      this.event("step.completed", { step, modelRole, durationMs: metric.durationMs });
      this.recordHaloSpan({
        name: `step.${step}`,
        observationKind: "CHAIN",
        startedAt,
        completedAt,
        ok: true,
        attributes: {
          "inference.step.name": step,
          "inference.step.model_role": modelRole,
          "inference.duration_ms": metric.durationMs,
        },
      });
      this.flushMetrics();
      return value;
    } catch (error) {
      const completedAt = now();
      const metric: StepMetric = {
        step,
        startedAt,
        completedAt,
        durationMs: Math.round(performance.now() - startedMs),
        ok: false,
        modelRole,
        error: error instanceof Error ? error.message : String(error),
      };
      this.metrics.push(metric);
      this.event("step.failed", { step, modelRole, durationMs: metric.durationMs, error: metric.error });
      this.recordHaloSpan({
        name: `step.${step}`,
        observationKind: "CHAIN",
        startedAt,
        completedAt,
        ok: false,
        error: metric.error,
        attributes: {
          "inference.step.name": step,
          "inference.step.model_role": modelRole,
          "inference.duration_ms": metric.durationMs,
        },
      });
      this.flushMetrics();
      throw error;
    }
  }

  recordLlmCall(metric: LlmCallMetric): void {
    this.llmMetrics.push(metric);
    this.event("llm.call", {
      role: metric.role,
      model: metric.model,
      durationMs: metric.durationMs,
      ok: metric.ok,
      promptTokens: metric.promptTokens,
      completionTokens: metric.completionTokens,
      totalTokens: metric.totalTokens,
      error: metric.error,
    });
    this.recordHaloSpan({
      name: `llm.${metric.role}.${metric.model}`,
      kind: "SPAN_KIND_CLIENT",
      observationKind: "LLM",
      startedAt: metric.startedAt,
      completedAt: metric.completedAt,
      ok: metric.ok,
      error: metric.error,
      attributes: {
        "llm.model_name": metric.model,
        "inference.llm.model_name": metric.model,
        "inference.llm.role": metric.role,
        "inference.llm.endpoint": metric.endpoint,
        "inference.llm.http_status": metric.httpStatus,
        "llm.token_count.prompt": metric.promptTokens,
        "llm.token_count.completion": metric.completionTokens,
        "llm.token_count.total": metric.totalTokens,
        "inference.llm.input_tokens": metric.promptTokens,
        "inference.llm.output_tokens": metric.completionTokens,
        "inference.duration_ms": metric.durationMs,
      },
    });
    this.flushMetrics();
  }

  recordToolCall(metric: ToolCallMetric): void {
    this.toolMetrics.push(metric);
    this.event("tool.call", {
      toolName: metric.toolName,
      namespace: metric.namespace,
      descriptionVersion: metric.descriptionVersion,
      durationMs: metric.durationMs,
      ok: metric.ok,
      retryCount: metric.retryCount,
      argumentBytes: metric.argumentBytes,
      resultBytes: metric.resultBytes,
      resultKind: metric.resultKind,
      error: metric.error,
      errorName: metric.errorName,
    });
    this.flushMetrics();
  }

  recordEditAttempt(attempt: EditAttemptContract): EditAttemptContract {
    const parsed = EditAttemptContractSchema.parse(attempt);
    const health = classifyEditAttemptHealth(parsed);
    const startedAt = parsed.createdAt;
    const completedAt = parsed.completedAt ?? now();
    const targetHashRows = editAttemptTargetHashRows(parsed);
    const missingRequiredPhases = missingRequiredEditAttemptPhases(parsed.phaseResults);
    const captureIssues = editAttemptCaptureIssues(parsed);
    const fallbackPath = parsed.fallbackPath ?? [];
    this.event("edit.attempt", {
      attempt: parsed,
      ok: health.ok,
      error: health.message,
    });
    this.recordHaloSpan({
      name: `edit.${parsed.editStrategyFamily}.${parsed.editStrategyId}`,
      observationKind: "EDIT",
      startedAt,
      completedAt,
      ok: health.ok,
      error: health.ok ? undefined : health.message,
      attributes: {
        "edit.attempt_id": parsed.editAttemptId,
        "edit.schema_version": parsed.schemaVersion,
        "edit.strategy_id": parsed.editStrategyId,
        "edit.strategy_family": parsed.editStrategyFamily,
        "edit.canonical_tool_spec_id": parsed.canonicalEditToolSpecId,
        "edit.rendered_tool_contract_id": parsed.renderedEditToolContractId,
        "edit.rendered_contract_version": parsed.renderedEditContractVersion,
        "edit.model_profile_id": parsed.modelProfileId,
        "edit.codebase_profile_id": parsed.codebaseProfileId,
        "edit.policy_id": parsed.policyId,
        "edit.run_id": parsed.runId,
        "edit.trace_id": parsed.traceId,
        "edit.target_files": parsed.targetFiles,
        "edit.target_file_count": parsed.targetFiles.length,
        "edit.read_snapshot_count": parsed.readSnapshotRefs.length,
        "edit.input_hash_count": Object.keys(parsed.inputContentHashes).length,
        "edit.output_hash_count": Object.keys(parsed.outputContentHashes).length,
        "edit.target_hash_paths": targetHashRows.map((target) => target.path),
        "edit.target_hash_count": targetHashRows.length,
        "edit.target_hash.before_count": targetHashRows.filter((target) => target.beforeHash !== undefined).length,
        "edit.target_hash.after_count": targetHashRows.filter((target) => target.afterHash !== undefined).length,
        "edit.target_hash.missing_before_paths": targetHashRows
          .filter((target) => target.beforeHash === undefined)
          .map((target) => target.path),
        "edit.target_hash.missing_after_paths": targetHashRows
          .filter((target) => target.afterHash === undefined)
          .map((target) => target.path),
        "edit.stale_context_status": parsed.staleContextStatus,
        "edit.permission_status": parsed.permissionStatus,
        "edit.verification_status": parsed.verificationStatus,
        "edit.post_apply_consistency_status": parsed.postApplyConsistencyStatus,
        "edit.self_detected_regression_status": parsed.selfDetectedRegressionStatus,
        "edit.self_detected_regression_evidence_refs": parsed.selfDetectedRegressionEvidenceRefs,
        "edit.self_detected_regression_evidence_count": parsed.selfDetectedRegressionEvidenceRefs.length +
          (parsed.selfDetectedRegressionEvidence?.length ?? 0),
        "edit.repair_attempt_count": parsed.repairAttemptCount,
        "edit.repair_attempt_ref_count": parsed.repairAttemptRefs?.length ?? 0,
        "edit.rollback_status": parsed.rollbackStatus,
        "edit.fallback_from_strategy_id": parsed.fallbackFromStrategyId,
        "edit.fallback_to_strategy_id": parsed.fallbackToStrategyId,
        "edit.fallback_path": fallbackPath.map(fallbackPathStepLabel),
        "edit.fallback_path_length": fallbackPath.length,
        "edit.token_count.prompt": parsed.tokenUsage.promptTokens,
        "edit.token_count.completion": parsed.tokenUsage.completionTokens,
        "edit.token_count.total": parsed.tokenUsage.totalTokens,
        "edit.changed_file_count": parsed.changedFileCount,
        "edit.changed_line_count": parsed.changedLineCount,
        "edit.protected_path_touched": parsed.protectedPathTouched,
        "edit.redaction_status": parsed.redactionStatus,
        "edit.artifact_refs": parsed.artifactRefs,
        "edit.artifact_ref_count": parsed.artifactRefs.length,
        "edit.error_codes": editErrorCodes(parsed.phaseResults, parsed.parseErrorCode, parsed.applyErrorCode),
        "edit.failed_phases": parsed.phaseResults
          .filter((phase) => phase.status === "failed")
          .map((phase) => phase.phase),
        "edit.required_phases": REAL_EDIT_ATTEMPT_REQUIRED_PHASES,
        "edit.missing_required_phases": missingRequiredPhases,
        "edit.required_phase_coverage_status": missingRequiredPhases.length === 0 ? "complete" : "incomplete",
        "edit.capture_status": captureIssues.length === 0 ? "complete" : "partial",
        "edit.capture_issues": captureIssues,
        "inference.duration_ms": parsed.latencyMs ?? editPhaseDuration(parsed.phaseResults),
        ...editPhaseAttributes(parsed.phaseResults),
      },
    });
    return parsed;
  }

  async measureToolCall<T>(input: {
    toolName: string;
    namespace?: string;
    descriptionVersion?: string;
    args: unknown;
    retryCount?: number;
    fn: () => Promise<T>;
  }): Promise<T> {
    const startedAt = now();
    const startedMs = performance.now();
    const encodedArgs = stableJson(input.args);
    const argumentHash = hashString(encodedArgs);
    try {
      const value = await input.fn();
      const result = classifyResult(value);
      const completedAt = now();
      const durationMs = Math.round(performance.now() - startedMs);
      const encodedResult = previewValue(value);
      this.recordToolCall({
        toolName: input.toolName,
        namespace: input.namespace,
        descriptionVersion: input.descriptionVersion,
        startedAt,
        completedAt,
        durationMs,
        ok: true,
        retryCount: input.retryCount ?? 0,
        argumentBytes: Buffer.byteLength(encodedArgs),
        argumentHash,
        resultBytes: result.bytes,
        resultKind: result.kind,
      });
      this.recordHaloSpan({
        name: `tool.${input.namespace == null ? input.toolName : `${input.namespace}.${input.toolName}`}`,
        kind: "SPAN_KIND_CLIENT",
        observationKind: "TOOL",
        startedAt,
        completedAt,
        ok: true,
        attributes: {
          "tool.name": input.toolName,
          "tool.namespace": input.namespace,
          "tool.description_version": input.descriptionVersion,
          "tool.retry_count": input.retryCount ?? 0,
          "input.value": previewValue(input.args),
          "input.hash": argumentHash,
          "input.bytes": Buffer.byteLength(encodedArgs),
          "output.value": encodedResult,
          "output.bytes": result.bytes,
          "output.kind": result.kind,
          "inference.duration_ms": durationMs,
        },
      });
      return value;
    } catch (error) {
      const completedAt = now();
      const durationMs = Math.round(performance.now() - startedMs);
      const message = error instanceof Error ? error.message : String(error);
      this.recordToolCall({
        toolName: input.toolName,
        namespace: input.namespace,
        descriptionVersion: input.descriptionVersion,
        startedAt,
        completedAt,
        durationMs,
        ok: false,
        retryCount: input.retryCount ?? 0,
        argumentBytes: Buffer.byteLength(encodedArgs),
        argumentHash,
        resultKind: "unknown",
        error: message,
        errorName: error instanceof Error ? error.name : undefined,
      });
      this.recordHaloSpan({
        name: `tool.${input.namespace == null ? input.toolName : `${input.namespace}.${input.toolName}`}`,
        kind: "SPAN_KIND_CLIENT",
        observationKind: "TOOL",
        startedAt,
        completedAt,
        ok: false,
        error: message,
        attributes: {
          "tool.name": input.toolName,
          "tool.namespace": input.namespace,
          "tool.description_version": input.descriptionVersion,
          "tool.retry_count": input.retryCount ?? 0,
          "input.value": previewValue(input.args),
          "input.hash": argumentHash,
          "input.bytes": Buffer.byteLength(encodedArgs),
          "error.type": error instanceof Error ? error.name : "Error",
          "error.message": message,
          "inference.duration_ms": durationMs,
        },
      });
      throw error;
    }
  }

  flushMetrics(): void {
    if (!this.config.telemetry.enabled) {
      return;
    }
    mkdirSync(dirname(this.metricsPath), { recursive: true });
    const previous = existsSync(this.metricsPath)
      ? (JSON.parse(readFileSync(this.metricsPath, "utf8")) as unknown)
      : {};
    const root =
      previous != null && typeof previous === "object" && !Array.isArray(previous)
        ? (previous as Record<string, unknown>)
        : {};
    root[this.runId] = {
      ...(this.optimizerPin === undefined ? {} : { optimizerPin: this.optimizerPin }),
      steps: this.metrics,
      llmCalls: this.llmMetrics,
      toolCalls: this.toolMetrics,
    };
    writeFileSync(this.metricsPath, `${JSON.stringify(root, null, 2)}\n`);
  }

  private withOptimizerPayload(payload: Record<string, unknown>): Record<string, unknown> {
    if (this.optimizerPin === undefined) {
      return payload;
    }
    return {
      ...payload,
      optimizerPin: this.optimizerPin,
    };
  }

  private optimizerHaloAttributes(): Record<string, unknown> {
    if (this.optimizerPin === undefined) {
      return {};
    }
    return {
      "optimizer.model_role": this.optimizerPin.modelRole,
      "optimizer.provider_config_role": this.optimizerPin.providerConfigRole,
      "optimizer.fallback_model_role": this.optimizerPin.fallbackModelRole,
      "optimizer.provider": this.optimizerPin.provider,
      "optimizer.endpoint_kind": this.optimizerPin.endpointKind,
      "optimizer.model_server_id": this.optimizerPin.modelServerId,
      "optimizer.model_server_profile_id": this.optimizerPin.modelServerProfileId,
      "optimizer.provider_discovery_source": this.optimizerPin.providerDiscoverySource,
      "optimizer.context_window_tokens": this.optimizerPin.contextWindowTokens,
      "optimizer.max_output_tokens": this.optimizerPin.maxOutputTokens,
      "optimizer.model_profile_id": this.optimizerPin.modelProfileId,
      "optimizer.codebase_profile_id": this.optimizerPin.codebaseProfileId,
      "optimizer.policy_id": this.optimizerPin.policyId,
      "optimizer.canonical_tool_version": this.optimizerPin.canonicalToolVersion,
      "optimizer.rendered_tool_version": this.optimizerPin.renderedToolVersion,
      "optimizer.result_style_version": this.optimizerPin.resultStyleVersion,
      "optimizer.verification_policy_version": this.optimizerPin.verificationPolicyVersion,
      "optimizer.edit_strategy_version": this.optimizerPin.editStrategyVersion,
      "optimizer.rendered_edit_contract_version": this.optimizerPin.renderedEditContractVersion,
      "optimizer.edit_fallback_policy_version": this.optimizerPin.editFallbackPolicyVersion,
      "optimizer.edit_repair_policy_version": this.optimizerPin.editRepairPolicyVersion,
      "optimizer.edit_verifier_policy_version": this.optimizerPin.editVerifierPolicyVersion,
      "optimizer.edit_objective_set_id": this.optimizerPin.editObjectiveSetId,
      "optimizer.source": this.optimizerPin.source,
      "optimizer.registry_root": this.optimizerPin.registryRoot,
      "optimizer.registry_error_count": this.optimizerPin.registryErrorCount,
      "optimizer.invalid_record_count": this.optimizerPin.invalidRecordCount,
      "optimizer.renderer_id": this.optimizerPin.rendererId,
      "optimizer.renderer_version": this.optimizerPin.rendererVersion,
      "optimizer.model_profile_record_id": this.optimizerPin.modelProfileRecordId,
      "optimizer.codebase_profile_record_id": this.optimizerPin.codebaseProfileRecordId,
      "optimizer.policy_record_id": this.optimizerPin.policyRecordId,
    };
  }

  private recordHaloSpan(input: {
    name: string;
    kind?: HaloSpan["kind"];
    observationKind: "AGENT" | "CHAIN" | "LLM" | "TOOL" | "EDIT";
    startedAt: string;
    completedAt: string;
    ok: boolean;
    error?: string | undefined;
    parentSpanId?: string;
    attributes?: Record<string, unknown>;
  }): void {
    if (!this.config.telemetry.enabled) {
      return;
    }
    const span: HaloSpan = {
      trace_id: this.traceId,
      span_id: input.parentSpanId === "" ? this.rootSpanId : randomSpanId(),
      parent_span_id: input.parentSpanId ?? this.rootSpanId,
      trace_state: "",
      name: input.name,
      kind: input.kind ?? "SPAN_KIND_INTERNAL",
      start_time: input.startedAt,
      end_time: input.completedAt,
      status: {
        code: input.ok ? "STATUS_CODE_OK" : "STATUS_CODE_ERROR",
        message: input.error ?? "",
      },
      resource: {
        attributes: {
          "service.name": "bleeding-agent",
          "service.version": "0.1.0",
          "deployment.environment": process.env.NODE_ENV ?? "local",
          "telemetry.sdk.language": "typescript",
        },
      },
      scope: {
        name: "bag.telemetry",
        version: "0.1.0",
      },
      attributes: cleanRecord({
        "openinference.span.kind": input.observationKind,
        "inference.observation_kind": input.observationKind,
        "inference.project_id": "bleeding-agent",
        "inference.export.schema_version": 1,
        "inference.run_id": this.runId,
        "inference.cwd": this.cwd,
        ...this.optimizerHaloAttributes(),
        ...input.attributes,
      }),
    };
    appendRawJsonl(this.spansPath, span);
  }
}

const editPhaseDuration = (phaseResults: readonly EditPhaseResult[]): number | undefined => {
  const durations = phaseResults
    .map((phase) => phase.durationMs)
    .filter((duration): duration is number => typeof duration === "number" && Number.isFinite(duration));
  if (durations.length === 0) {
    return undefined;
  }
  return Math.round(durations.reduce((sum, duration) => sum + duration, 0));
};

const editErrorCodes = (
  phaseResults: readonly EditPhaseResult[],
  parseErrorCode: EditAttemptContract["parseErrorCode"],
  applyErrorCode: EditAttemptContract["applyErrorCode"],
): string[] => {
  const codes = new Set<string>();
  for (const code of [parseErrorCode, applyErrorCode, ...phaseResults.map((phase) => phase.errorCode)]) {
    if (typeof code === "string" && code.length > 0) {
      codes.add(code);
    }
  }
  return [...codes];
};

const editPhaseAttributes = (phaseResults: readonly EditPhaseResult[]): Record<string, unknown> =>
  Object.fromEntries(
    phaseResults.flatMap((phase) => [
      [`edit.phase.${phase.phase}.status`, phase.status],
      [`edit.phase.${phase.phase}.error_code`, phase.errorCode],
      [`edit.phase.${phase.phase}.duration_ms`, phase.durationMs],
      [`edit.phase.${phase.phase}.artifact_ref_count`, phase.artifactRefs.length],
    ]),
  );

const fallbackPathStepLabel = (step: EditFallbackPathStep): string =>
  `${step.trigger}:${step.fromStrategyId}->${step.toStrategyId}:${step.status}`;

const classifyEditAttemptHealth = (attempt: EditAttemptContract): { ok: boolean; message: string } => {
  const reasons = [
    ...attempt.phaseResults
      .filter((phase) => phase.status === "failed")
      .map((phase) => `${phase.phase}:${phase.errorCode ?? "failed"}`),
    attempt.parseErrorCode == null ? undefined : `parse:${attempt.parseErrorCode}`,
    attempt.applyErrorCode == null ? undefined : `apply:${attempt.applyErrorCode}`,
    attempt.staleContextStatus === "stale" || attempt.staleContextStatus === "conflict"
      ? `stale-context:${attempt.staleContextStatus}`
      : undefined,
    attempt.permissionStatus === "rejected" || attempt.permissionStatus === "failed"
      ? `permission:${attempt.permissionStatus}`
      : undefined,
    attempt.verificationStatus === "failed" || attempt.verificationStatus === "error"
      ? `verification:${attempt.verificationStatus}`
      : undefined,
    attempt.postApplyConsistencyStatus === "inconsistent"
      ? `post-apply:${attempt.postApplyConsistencyStatus}`
      : undefined,
    attempt.selfDetectedRegressionStatus === "confirmed"
      ? `self-detected:${attempt.selfDetectedRegressionStatus}`
      : undefined,
    attempt.rollbackStatus === "failed" || attempt.rollbackStatus === "partial"
      ? `rollback:${attempt.rollbackStatus}`
      : undefined,
    attempt.protectedPathTouched ? "protected-path:touched" : undefined,
  ].filter((reason): reason is string => reason !== undefined);
  return {
    ok: reasons.length === 0,
    message: reasons.join("; "),
  };
};

export const deterministicSelfEvaluation = (input: {
  threshold: number;
  metrics: StepMetric[];
  llmMetrics?: LlmCallMetric[];
  toolMetrics?: ToolCallMetric[];
  artifactCount: number;
}): SelfEvaluation => {
  const failed = input.metrics.filter((metric) => !metric.ok);
  const failedLlmCalls = (input.llmMetrics ?? []).filter((metric) => !metric.ok);
  const failedToolCalls = (input.toolMetrics ?? []).filter((metric) => !metric.ok);
  const avgDuration =
    input.metrics.length === 0
      ? 0
      : input.metrics.reduce((sum, metric) => sum + metric.durationMs, 0) / input.metrics.length;
  const artifactScore = Math.min(1, input.artifactCount / 8);
  const reliabilityScore = failed.length === 0 ? 1 : Math.max(0, 1 - failed.length / input.metrics.length);
  const llmReliabilityScore =
    input.llmMetrics == null || input.llmMetrics.length === 0
      ? 0.8
      : Math.max(0, 1 - failedLlmCalls.length / input.llmMetrics.length);
  const toolReliabilityScore =
    input.toolMetrics == null || input.toolMetrics.length === 0
      ? 0.85
      : Math.max(0, 1 - failedToolCalls.length / input.toolMetrics.length);
  const latencyScore = avgDuration === 0 ? 0.5 : avgDuration < 10_000 ? 1 : avgDuration < 60_000 ? 0.8 : 0.55;
  const score =
    Math.round(
      (artifactScore * 0.25 +
        reliabilityScore * 0.3 +
        llmReliabilityScore * 0.18 +
        toolReliabilityScore * 0.17 +
        latencyScore * 0.1) *
        100,
    ) / 100;

  return {
    score,
    passed: score >= input.threshold,
    strengths: [
      `Captured ${input.metrics.length} step metrics.`,
      `Captured ${input.llmMetrics?.length ?? 0} LLM call metrics.`,
      `Captured ${input.toolMetrics?.length ?? 0} tool call metrics.`,
      `Persisted ${input.artifactCount} artifacts.`,
    ],
    weaknesses: [
      ...failed.map((metric) => `${metric.step}: ${metric.error ?? "failed"}`),
      ...failedLlmCalls.map((metric) => `${metric.role}:${metric.model}: ${metric.error ?? "failed"}`),
      ...failedToolCalls.map((metric) => `${metric.toolName}: ${metric.error ?? "failed"}`),
    ],
    improvementActions:
      score >= input.threshold
        ? ["Keep current policy; collect more runs before changing concurrency."]
        : [
            "Reduce executor concurrency for interactive runs.",
            "Prefer master judgement on failed planning or schema-validation steps.",
            "Inspect failed tool-call arguments and revise tool descriptions or schemas.",
            "Add targeted eval examples for the failed step.",
          ],
  };
};

const stableJson = (value: unknown): string => {
  const normalize = (input: unknown): unknown => {
    if (Array.isArray(input)) {
      return input.map(normalize);
    }
    if (input != null && typeof input === "object") {
      return Object.fromEntries(
        Object.entries(input as Record<string, unknown>)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, nested]) => [key, normalize(nested)]),
      );
    }
    return input;
  };

  try {
    return JSON.stringify(normalize(value));
  } catch {
    return String(value);
  }
};

const hashString = (value: string): string => createHash("sha256").update(value).digest("hex");

const randomSpanId = (): string => randomBytes(8).toString("hex");

const previewValue = (value: unknown, maxChars = 4096): string => {
  const encoded = stableJson(value);
  if (encoded.length <= maxChars) {
    return encoded;
  }
  return `${encoded.slice(0, maxChars)}[BleedingAgent truncated: original ${encoded.length} chars]`;
};

const cleanRecord = (input: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(input).filter(([, value]) => value !== undefined));

const classifyResult = (
  value: unknown,
): {
  bytes: number;
  kind: ToolCallMetric["resultKind"];
} => {
  if (value == null) {
    return { bytes: 0, kind: "empty" };
  }
  if (typeof value === "string") {
    return { bytes: Buffer.byteLength(value), kind: "text" };
  }
  if (value instanceof Uint8Array) {
    return { bytes: value.byteLength, kind: "binary" };
  }
  const encoded = stableJson(value);
  return { bytes: Buffer.byteLength(encoded), kind: "json" };
};
