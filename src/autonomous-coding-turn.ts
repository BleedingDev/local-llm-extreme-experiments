/**
 * Autonomous coding turn — single-tool (bash) tool-use loop, mini-swe-agent
 * style. Used by the `/run-tools` ACP slash command.
 *
 * Contract:
 *   - The agent receives a task description and the workspace cwd.
 *   - The LLM is exposed exactly one tool: bash.
 *   - The loop continues until either:
 *       * the model emits `BAG_TASK_COMPLETE` as the first non-blank line of
 *         a bash command's stdout (stopReason='submitted'),
 *       * finish_reason !== 'tool_calls' AND the model returned no tool_calls
 *         (stopReason='end_turn'),
 *       * max_turns reached (stopReason='max_turns'),
 *       * cost_limit reached (stopReason='cost_limit'),
 *       * an abort signal fires (stopReason='cancelled'),
 *   - Output elision (head 5k + tail 5k) is applied at message render time.
 *   - Format errors (no tool_calls, unknown tool, malformed JSON) are
 *     reinjected as a user-role correction message; the loop continues.
 */

import {
  BASH_TOOL_DEFINITION,
  BASH_TOOL_NAME,
  CODE_SEARCH_TOOL_DEFINITION,
  CODE_SEARCH_TOOL_NAME,
  VIEW_IMAGE_TOOL_DEFINITION,
  VIEW_IMAGE_TOOL_NAME,
  executeViewImageTool,
  executeBashTool,
  renderBashObservation,
  SUBMIT_SENTINEL,
  type AcpTerminalClient,
  type AutonomousToolResult,
  type CodeSearchToolResult,
} from "./autonomous-tools";
import {
  colgrepBackend,
  renderHitsAsObservation,
  type CodebaseSearchBackend,
} from "./codebase-index/colgrep-bridge";
import { matchVerifierSignature, renderHintForRetry } from "./verifier-signature-library";
import {
  loadFailureClusters,
  matchClusterByVerifierOutput,
  type FailureClustersDocument,
} from "./optimizer/failure-clusters";
import { maybeSummarizeInstruction } from "./instruction-summarizer";
import { loadOptimizedExecutorPrompt } from "./optimized-prompt-loader";
import {
  runPreSubmitSelfCheck,
  type BashTraceTailEntry,
  SELF_CHECK_BASH_TAIL_MAX_CALLS,
} from "./pre-submit-self-check";
import { buildSystemPrompt as buildModularSystemPrompt } from "./prompts/loader";
import { DEFAULT_PATH_PROFILE, type PathProfile } from "./types";
import { loadHarnessGates, type HarnessGates } from "./harness-gates";
import {
  createEditStrategy,
  type EditDispatchOutcome,
  type EditStrategy,
  type EditStrategyId,
} from "./edit-strategies/registry";

/**
 * Lazy-loaded failure cluster index. Auto-discovered patterns are now the
 * PRIMARY retry-hint source — the curated verifier-signature library
 * remains as a soft-deprecated safety net (see
 * `docs/bag-verifier-signature-retirement.md`).
 */
let CACHED_FAILURE_CLUSTERS: FailureClustersDocument | null = null;
let FAILURE_CLUSTERS_LOADED = false;
/**
 * Threshold for the trigram-Jaccard cluster matcher when invoked from the
 * autonomous coding turn. Tuned 2026-05-02 against the 143-failure corpus —
 * see `bench/.bag/optimizer/failure-clusters-config.json` and
 * `tests/verifier-signature-vs-clusters-parity.test.ts`. Lower than the
 * library default (0.45) so the cluster matcher is the primary signal,
 * higher than 0.25 so a known spurious overlap (acp-internal-error vs
 * FileNotFoundError cluster) doesn't pollute retry hints.
 */
const FAILURE_CLUSTER_MATCH_THRESHOLD = 0.3;
const getFailureClusters = (): FailureClustersDocument | null => {
  if (FAILURE_CLUSTERS_LOADED) return CACHED_FAILURE_CLUSTERS;
  FAILURE_CLUSTERS_LOADED = true;
  try {
    CACHED_FAILURE_CLUSTERS = loadFailureClusters(process.cwd());
  } catch {
    CACHED_FAILURE_CLUSTERS = null;
  }
  return CACHED_FAILURE_CLUSTERS;
};
import type {
  AssistantWithToolCalls,
  ChatMessage,
  ChatWithToolsOptions,
  LlmRouter,
  ToolResultMessage,
  ToolUseTurnMessage,
} from "./llm";

export type AutonomousTurnStopReason =
  | "submitted"
  | "submitted_but_failed"
  | "end_turn"
  | "max_turns"
  | "cost_limit"
  | "cancelled"
  | "error";

export type AutonomousTurnTraceEntry =
  | { kind: "user"; at: string; text: string }
  | { kind: "assistant"; at: string; text: string; toolCallCount: number }
  | { kind: "tool_call"; at: string; toolCallId: string; tool: string; argumentsJson: string }
  | { kind: "tool_result"; at: string; toolCallId: string; result: AutonomousToolResult }
  | { kind: "format_error"; at: string; reason: string }
  | { kind: "abort"; at: string }
  | {
      kind: "instruction_summarized";
      at: string;
      original_chars: number;
      summary_chars: number;
      tokens_saved: number;
    }
  | {
      kind: "attempt";
      at: string;
      attempt: number;
      verifier_passed: boolean;
      verifier_output: string;
      verifier_exit_code: number | null;
      prompt_tokens: number;
      completion_tokens: number;
      tool_calls_executed: number;
      turns_used: number;
    }
  | {
      kind: "pre_submit_self_check";
      at: string;
      complete: boolean;
      missing: string[];
      /**
       * `true` whenever this entry was emitted because the self-check gate
       * actually executed (either the auditor returned a verdict, or the
       * auditor call threw and we failed open). Audit tools use this field
       * to distinguish "gate ran and approved" (`gate_reached: true,
       * complete: true`) from "gate never ran" (no entry at all). Always
       * `true` in the current implementation — the field exists so the
       * shape is forward-compatible with future "gate_reached: false"
       * sentinels (e.g. trace markers emitted from a path that explicitly
       * skipped the gate).
       */
      gate_reached: true;
      /**
       * Set when the gate ran but failed open because the auditor LLM call
       * threw (network, parse, etc.). The submission still proceeds, but
       * the lift analyzer can subtract these from the "fired and produced
       * a verdict" denominator.
       */
      error?: string;
    }
  | {
      kind: "code_search";
      at: string;
      toolCallId: string;
      query: string;
      hitCount: number;
      backendStatus: CodeSearchToolResult["backendStatus"];
      durationMs: number;
    }
  | {
      /**
       * Emitted on every verifier-failure retry to record which retry-hint
       * source fired for the next attempt. Audit pipelines use this to track
       * the curated `verifier-signature-library` retirement readiness:
       * when `library` hit-rate drops below 5% over 30 BAG runs, the
       * curated library can be deleted (see
       * `docs/bag-verifier-signature-retirement.md`).
       *
       * `source` semantics:
       *   - "cluster": auto-discovered failure-cluster matched (PRIMARY)
       *   - "library": curated verifier-signature-library matched (fallback)
       *   - "both":   both fired — record for retirement audit
       *   - "none":   neither fired — generic verifier feedback only
       */
      kind: "retry_hint";
      at: string;
      attempt: number;
      source: "cluster" | "library" | "both" | "none";
      cluster_id?: string;
      library_id?: string;
    }
  | {
      /**
       * Emitted whenever the structured edit-strategy registry dispatches a
       * tool call. Used by the edit-strategy study aggregator
       * (`bench/edit_strategy_study/aggregate.py`) to compute the per-cell
       * applied/match_failed/stale_context/syntax_error rate. Never emitted
       * when `gates.editStrategy === "shell-heredoc"` because that strategy
       * delegates to bash and has no structured dispatch.
       */
      kind: "edit_dispatch";
      at: string;
      strategy: EditStrategyId;
      tool: string;
      target: string;
      outcome: EditDispatchOutcome;
      bytes_changed: number;
      retries_within_strategy: number;
    };

export type AutonomousTurnResult = {
  stopReason: AutonomousTurnStopReason;
  turnsUsed: number;
  toolCallsExecuted: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  trace: AutonomousTurnTraceEntry[];
  submittedOutput: string | null;
  attemptsUsed: number;
};

export type PostSubmitVerifierInput = {
  client: AcpTerminalClient;
  sessionId: string;
  cwd: string;
};

export type PostSubmitVerifierResult = {
  passed: boolean;
  output: string;
  exitCode: number | null;
};

export type PostSubmitVerifier = (
  input: PostSubmitVerifierInput,
) => Promise<PostSubmitVerifierResult>;

export type AutonomousTurnConfig = {
  maxTurns: number;
  maxConsecutiveFormatErrors: number;
  perCallTimeoutSec: number;
  costLimitUsd?: number;
  systemPromptOverride?: string;
  /**
   * The sentinel string the model echoes via bash to signal completion.
   * Default: BAG_TASK_COMPLETE. Per-issue scoped loops should use
   * BAG_ISSUE_COMPLETE so a single bash call can submit a sub-goal without
   * collapsing the outer DAG-level run.
   */
  submitSentinel?: string;
  /**
   * Extra user-message context appended after the standard task header.
   * Used by per-issue scoped loops to inject the previous issue results,
   * the current issue's expected files, and verifier hints.
   */
  contextSuffix?: string;
  /**
   * Optional runtime hint (e.g. background-process + polling guidance for
   * tasks that require long async waits). When set, appended to the user
   * message before the final "Begin by listing..." instruction so the master
   * model sees it on every chat round through the standard message history.
   *
   * Distinct from `contextSuffix` to keep per-shape advisory hints separate
   * from per-issue DAG context (the dag-tool-loop already uses
   * `contextSuffix` for issue-scoped data and we don't want to clobber it).
   */
  runtimeHint?: string;
  /**
   * Maximum number of additional retry attempts when a `verifyAfterSubmit`
   * callback rejects a submission. Default: 2 (so up to 3 total attempts,
   * 1 initial + 2 retries). Ignored when no verifier is supplied.
   */
  maxAttemptRetries?: number;
  /**
   * Optional dependency-injected `CodebaseSearchBackend`. When omitted, the
   * default `colgrepBackend()` is used. Tests inject a mock backend here so
   * the dispatch loop can be exercised without the colgrep binary.
   *
   * When the env var `BAG_CODE_SEARCH=0` is set the tool is REMOVED from the
   * tool list entirely so the model cannot call it — used by the A/B harness
   * to compare with-vs-without code_search at runtime.
   */
  codeSearchBackend?: CodebaseSearchBackend;
};

const DEFAULT_CONFIG: AutonomousTurnConfig = {
  maxTurns: 80,
  maxConsecutiveFormatErrors: 3,
  perCallTimeoutSec: 90,
  maxAttemptRetries: 2,
};

/**
 * Build the executor system prompt from a PathProfile.
 *
 * The prompt is now assembled from `src/prompts/principles.md` +
 * `src/prompts/tactics/*.md` via `buildSystemPrompt()` so each forensic
 * gate / clause is a separately-auditable file with YAML frontmatter
 * (incident pointer, introduction date, review date, trigger). See
 * `src/prompts/README.md` for the full design.
 *
 * `pathProfile` supplies three placeholder substitutions that the
 * markdown sources reference:
 *   - `${SCRATCH}`         — `pathProfile.scratchDirs[0]` (the cited scratch dir).
 *   - `${PATH_JOINED}`     — `pathProfile.systemPathDirs.join(":")` (clean PATH for SUBPROCESS-PATH GATE).
 *   - `${PERSIST_TARGET}`  — `pathProfile.systemPathDirs[0]` (where to install binaries system-wide).
 *
 * The default-profile output is byte-equivalent to the historical
 * hard-coded string, modulo the trailing
 * `[Tactics loaded: N — auditable in src/prompts/tactics/]` attestation
 * footer the loader appends.
 */
export const buildExecutorSystemPrompt = (
  pathProfile: PathProfile = DEFAULT_PATH_PROFILE,
): string => {
  const scratch = pathProfile.scratchDirs[0] ?? "/tmp";
  const pathJoined = pathProfile.systemPathDirs.join(":");
  const persistTarget = pathProfile.systemPathDirs[0] ?? "/usr/local/bin";
  return buildModularSystemPrompt({
    sentinel: SUBMIT_SENTINEL,
    placeholders: {
      SCRATCH: scratch,
      PATH_JOINED: pathJoined,
      PERSIST_TARGET: persistTarget,
    },
  });
};

/**
 * Backwards-compatible default executor system prompt. Internal callers that
 * need to render for a different profile should call
 * `buildExecutorSystemPrompt(profile)`.
 */
export const SYSTEM_PROMPT_DEFAULT = buildExecutorSystemPrompt(DEFAULT_PATH_PROFILE);

const now = (): string => new Date().toISOString();

const renderObservationMessage = (
  toolCallId: string,
  result: AutonomousToolResult,
): ToolResultMessage => ({
  role: "tool",
  tool_call_id: toolCallId,
  content: renderBashObservation(result),
});

const userReminder = (text: string): ChatMessage => ({ role: "user", content: text });

const parseBashArguments = (argumentsJson: string): { command: string; timeoutSec?: number; error?: string } => {
  try {
    const parsed = JSON.parse(argumentsJson) as { command?: unknown; timeout_sec?: unknown };
    const command = typeof parsed.command === "string" ? parsed.command : "";
    if (command.trim().length === 0) {
      return { command: "", error: "bash requires a non-empty 'command' string" };
    }
    const timeoutSec =
      typeof parsed.timeout_sec === "number" && Number.isFinite(parsed.timeout_sec) && parsed.timeout_sec > 0
        ? parsed.timeout_sec
        : undefined;
    return timeoutSec === undefined ? { command } : { command, timeoutSec };
  } catch (err) {
    return { command: "", error: `bash arguments JSON parse failure: ${err instanceof Error ? err.message : String(err)}` };
  }
};

export const runAutonomousCodingTurn = async (input: {
  router: LlmRouter;
  client: AcpTerminalClient;
  sessionId: string;
  cwd: string;
  task: string;
  signal?: AbortSignal;
  config?: Partial<AutonomousTurnConfig>;
  verifyAfterSubmit?: PostSubmitVerifier;
  /**
   * Optional path-convention overrides flowed through to
   * `runPreSubmitSelfCheck` (and any inner audit). When omitted the helper
   * defaults to the Linux conventions baked into `BagConfigSchema.pathProfile`.
   */
  pathProfile?: PathProfile;
  /**
   * Optional event hook invoked synchronously every time a trace entry is
   * appended. Used by `src/sdk/agent-session.ts` to stream live events to
   * external embedders (SDK consumers, RPC adapters) without forking the
   * loop. Throwing from the hook is caught and ignored so subscribers cannot
   * destabilise the turn; subscribers wanting cancellation should signal via
   * the `signal` parameter.
   */
  onTraceEntry?: (entry: AutonomousTurnTraceEntry) => void;
  /**
   * Optional harness-gate snapshot. When omitted, gates are read from the
   * process env via `loadHarnessGates()`. Tests inject directly so they
   * don't have to mutate `process.env`. Production callers rely on
   * env-var resolution — the ablation harness sets BAG_MODE_BARE_ENV /
   * BAG_MODE_MINIMAL_ENV before invoking.
   */
  gates?: HarnessGates;
}): Promise<AutonomousTurnResult> => {
  const cfg: AutonomousTurnConfig = { ...DEFAULT_CONFIG, ...(input.config ?? {}) };
  const gates: HarnessGates = input.gates ?? loadHarnessGates();
  const sentinel = cfg.submitSentinel ?? SUBMIT_SENTINEL;
  const optimized = loadOptimizedExecutorPrompt();
  const exec_system = optimized?.system ?? SYSTEM_PROMPT_DEFAULT;
  if (optimized) console.log(`[bag] using optimized executor prompt run=${optimized.runId}`);
  // Edit-strategy registry — when `gates.editStrategy !== "shell-heredoc"`
  // the chosen strategy contributes additional tool definitions (e.g. `edit`,
  // `apply_patch`, `fs_write_text_file`) that the model can call instead of
  // shelling out via bash. The default strategy ships zero extra tools so
  // existing BAG behaviour is byte-equivalent. See
  // `src/edit-strategies/registry.ts` and `docs/bag-edit-strategy-study.md`.
  const editStrategy: EditStrategy = createEditStrategy(gates.editStrategy);
  const editStrategyToolDefs = editStrategy.toolDefinitions();
  const editStrategyToolNames = new Set(
    editStrategyToolDefs.map((def) => def.function.name),
  );
  const baseSystemPrompt =
    cfg.systemPromptOverride != null
      ? cfg.systemPromptOverride
      : exec_system.replace(/BAG_TASK_COMPLETE/g, sentinel);
  // Append the strategy's tactic fragment ONLY when a non-default strategy is
  // in play. shell-heredoc adds no tool surface and the existing executor
  // prompt already explains shell-edit semantics, so we leave it untouched
  // for the default path.
  const systemPrompt =
    gates.editStrategy === "shell-heredoc"
      ? baseSystemPrompt
      : `${baseSystemPrompt}\n\n${editStrategy.systemPromptFragment()}`;
  const trace: AutonomousTurnTraceEntry[] = [];
  const onTraceEntry = input.onTraceEntry;
  const pushTrace = (entry: AutonomousTurnTraceEntry): void => {
    trace.push(entry);
    if (onTraceEntry) {
      try {
        onTraceEntry(entry);
      } catch {
        // Subscriber failures must not destabilise the turn.
      }
    }
  };
  // Universal lesson #1: pre-summarize long instructions via the cheap local
  // role so the master loop doesn't re-burn tokens on every turn. Below the
  // threshold this is a no-op passthrough.
  const instructionSummary = await maybeSummarizeInstruction({
    router: input.router,
    task: input.task,
  });
  if (instructionSummary.summarized) {
    pushTrace({
      kind: "instruction_summarized",
      at: now(),
      original_chars: instructionSummary.original.length,
      summary_chars: instructionSummary.summary.length,
      tokens_saved: instructionSummary.tokensSaved,
    });
  }
  const taskForPrompt = instructionSummary.summary;
  const userParts: string[] = [
    `Task:\n${taskForPrompt.trim()}`,
    ``,
    `Workspace cwd: ${input.cwd}`,
  ];
  if (cfg.contextSuffix && cfg.contextSuffix.trim().length > 0) {
    userParts.push("");
    userParts.push(cfg.contextSuffix.trim());
  }
  if (cfg.runtimeHint && cfg.runtimeHint.trim().length > 0) {
    userParts.push("");
    userParts.push(cfg.runtimeHint.trim());
  }
  userParts.push("");
  userParts.push(
    `Begin by listing the workspace, reading any instruction.md, then plan and execute. Submit with \`echo ${sentinel}\` when done.`,
  );
  const messages: ToolUseTurnMessage[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userParts.join("\n") },
  ];
  pushTrace({ kind: "user", at: now(), text: input.task });

  let turnsUsed = 0;
  let toolCallsExecuted = 0;
  let totalPromptTokens = 0;
  let totalCompletionTokens = 0;
  let consecutiveFormatErrors = 0;
  let stopReason: AutonomousTurnStopReason = "end_turn";
  let submittedOutput: string | null = null;

  // Tool-surface gating: each tool can be removed independently via the
  // matching `BAG_TOOL_*=0` env var (see src/harness-gates.ts). The A/B/C
  // ablation harness uses BAG_MODE_BARE_ENV (gates off, multi-tool on) and
  // BAG_MODE_MINIMAL_ENV (everything off) to enumerate the contribution of
  // each gate / tool to BAG's pass rate per model tier. Defaults are ON so
  // existing call sites are byte-equivalent.
  const codeSearchEnabled = gates.codeSearch;
  const viewImageEnabled = gates.viewImage;
  const codeSearchBackend: CodebaseSearchBackend =
    cfg.codeSearchBackend ?? colgrepBackend();
  const tools: ChatWithToolsOptions["tools"] = [
    BASH_TOOL_DEFINITION,
    ...(viewImageEnabled ? [VIEW_IMAGE_TOOL_DEFINITION] : []),
    ...(codeSearchEnabled ? [CODE_SEARCH_TOOL_DEFINITION] : []),
    ...editStrategyToolDefs,
  ];
  // Pending images queued by `view_image` tool calls — they get attached to
  // the NEXT outgoing user message as multimodal content blocks. See the
  // VIEW_IMAGE_TOOL_DEFINITION docstring for the rationale.
  const pendingImages: Array<{ mimeType: string; base64: string; path: string }> = [];

  /**
   * Dispatch a `code_search` tool call. Pure-async — no terminal channel.
   * Parses the model-supplied JSON arguments, hands them to the injected
   * backend, and renders a compact observation string. Backend
   * unavailability returns a STRUCTURED error (not a crash) so the model can
   * fall back to bash + rg on the next turn. Closes over `input.cwd` and
   * `input.signal` from the surrounding `runAutonomousCodingTurn` call.
   */
  const dispatchCodeSearch = async (argumentsJson: string): Promise<CodeSearchToolResult> => {
    const startedMs = Date.now();
    let parsed:
      | {
          query: string;
          topK?: number;
          mode?: "semantic" | "hybrid";
          pathFilter?: string;
          languageFilter?: string;
        }
      | { error: string };
    try {
      const obj = JSON.parse(argumentsJson) as Record<string, unknown>;
      const query = typeof obj.query === "string" ? obj.query.trim() : "";
      if (query.length === 0) {
        parsed = { error: "code_search requires a non-empty 'query' string" };
      } else {
        const topKRaw = obj.top_k;
        const modeRaw = obj.mode;
        const pathRaw = obj.path_filter;
        const langRaw = obj.language_filter;
        const out: {
          query: string;
          topK?: number;
          mode?: "semantic" | "hybrid";
          pathFilter?: string;
          languageFilter?: string;
        } = { query };
        if (typeof topKRaw === "number" && Number.isFinite(topKRaw) && topKRaw > 0) {
          out.topK = Math.min(100, Math.floor(topKRaw));
        }
        if (modeRaw === "semantic" || modeRaw === "hybrid") out.mode = modeRaw;
        if (typeof pathRaw === "string" && pathRaw.length > 0) out.pathFilter = pathRaw;
        if (typeof langRaw === "string" && langRaw.length > 0) out.languageFilter = langRaw;
        parsed = out;
      }
    } catch (err) {
      parsed = {
        error: `code_search arguments JSON parse failure: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
    if ("error" in parsed) {
      return {
        ok: false,
        observation: `code_search error: ${parsed.error}`,
        hits: [],
        backendStatus: "error",
        durationMs: Date.now() - startedMs,
        errorMessage: parsed.error,
      };
    }
    const available = await codeSearchBackend.isAvailable();
    if (!available) {
      const message =
        "code_search backend unavailable: `colgrep` binary not on PATH. Fall back to `bash` + `rg` for this query, or run `installColgrep()` once on the host.";
      return {
        ok: false,
        observation: message,
        hits: [],
        backendStatus: "unavailable",
        durationMs: Date.now() - startedMs,
        errorMessage: message,
      };
    }
    try {
      // Best-effort index ensure — we don't fail the search if ensureIndex
      // returns "skipped" (which happens on first call when the binary is
      // missing OR when the workspace is unreachable).
      await codeSearchBackend
        .ensureIndex({ cwd: input.cwd, ...(input.signal ? { signal: input.signal } : {}) })
        .catch(() => undefined);
      const searchOptions: Parameters<CodebaseSearchBackend["search"]>[0] = {
        cwd: input.cwd,
        query: parsed.query,
      };
      if (parsed.topK !== undefined) searchOptions.topK = parsed.topK;
      if (parsed.mode !== undefined) searchOptions.mode = parsed.mode;
      if (parsed.pathFilter !== undefined) searchOptions.pathFilter = parsed.pathFilter;
      if (parsed.languageFilter !== undefined) searchOptions.languageFilter = parsed.languageFilter;
      if (input.signal) searchOptions.signal = input.signal;
      const hits = await codeSearchBackend.search(searchOptions);
      const observation = renderHitsAsObservation(hits);
      return {
        ok: true,
        observation,
        hits,
        backendStatus: "available",
        durationMs: Date.now() - startedMs,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        ok: false,
        observation: `code_search failed: ${message}. Fall back to \`bash\` + \`rg\` for this query.`,
        hits: [],
        backendStatus: "error",
        durationMs: Date.now() - startedMs,
        errorMessage: message,
      };
    }
  };

  const verifier = input.verifyAfterSubmit;
  const maxAttemptRetries = Math.max(
    0,
    cfg.maxAttemptRetries ?? DEFAULT_CONFIG.maxAttemptRetries ?? 0,
  );
  // Gate: retry path disabled → cap attempts at 1 even with a verifier
  // present. The verifier still runs (so we record pass/fail telemetry),
  // but a failure ends the turn instead of injecting feedback + looping.
  const effectiveMaxRetries = gates.retryPath ? maxAttemptRetries : 0;
  const totalAllowedAttempts = verifier ? 1 + effectiveMaxRetries : 1;

  let attemptsUsed = 0;
  let attemptStart = {
    promptTokens: totalPromptTokens,
    completionTokens: totalCompletionTokens,
    toolCalls: toolCallsExecuted,
    turns: turnsUsed,
  };
  let lastVerifierPassed = false;
  let anyAttemptPassed = false;
  // Pre-submit self-check retries are accounted SEPARATELY from the
  // verifier-driven Best-of-N retries so the two gates compose without
  // double-charging the attempt budget. Cap total self-check retries at 1
  // across the whole turn (defence against pathological loops where the
  // auditor keeps flagging the same items). When `selfCheckRetryQueued` is
  // true the outer attempts loop runs an extra iteration without
  // incrementing `attemptsUsed`.
  // Allow 3 self-check rounds: empirically a single round catches "you forgot
  // file X", a second catches "format mismatch in X", and a third catches the
  // "shell-session-only fix" pattern (the change works in your session but
  // not in a fresh subshell). Each round spawns ONE local-role call so the
  // cost is bounded.
  const SELF_CHECK_MAX_RETRIES = 3;
  let selfCheckRetriesUsed = 0;
  let selfCheckRetryQueued = false;

  // Reconstruct the most-recent bash calls from the trace to feed the
  // pre-submit self-check auditor. Pulls the last N tool_call/tool_result
  // pairs in chronological order.
  const collectBashTraceTail = (): BashTraceTailEntry[] => {
    const callsById = new Map<
      string,
      { command: string; index: number }
    >();
    let order = 0;
    const ordered: Array<{
      index: number;
      command: string;
      output: string;
      exitCode: number | null;
    }> = [];
    for (const entry of trace) {
      if (entry.kind === "tool_call" && entry.tool === BASH_TOOL_NAME) {
        order += 1;
        let command = "";
        try {
          const parsed = JSON.parse(entry.argumentsJson) as { command?: unknown };
          if (typeof parsed.command === "string") command = parsed.command;
        } catch {
          // ignore — we'll just emit an empty command for malformed args
        }
        callsById.set(entry.toolCallId, { command, index: order });
      } else if (entry.kind === "tool_result") {
        const meta = callsById.get(entry.toolCallId);
        if (meta == null) continue;
        ordered.push({
          index: meta.index,
          command: meta.command,
          output: entry.result.output ?? "",
          exitCode: entry.result.exitCode,
        });
        callsById.delete(entry.toolCallId);
      }
    }
    ordered.sort((a, b) => a.index - b.index);
    return ordered
      .slice(-SELF_CHECK_BASH_TAIL_MAX_CALLS)
      .map(({ command, output, exitCode }) => ({ command, output, exitCode }));
  };

  // Pre-submit self-check gate. Returns true when the submission should be
  // accepted; false when the auditor flagged unmet requirements AND we still
  // have retry budget (in which case feedback was already pushed to the
  // conversation and a self-check retry was queued). Fails open: any throw
  // or empty-missing payload accepts the submission.
  //
  // CONTRACT (measurability): every invocation of this helper pushes EXACTLY
  // ONE `pre_submit_self_check` trace entry — even when the auditor approves
  // or the call fails open. This is what `bench/audit/self_check_lift.py`
  // (and the broader BAG forensic tooling) depends on to distinguish
  // "gate ran and approved" (entry present, `complete: true`) from
  // "gate never reached on this attempt" (no entry).
  const runSelfCheckGate = async (): Promise<boolean> => {
    // Gate: self-check disabled → accept the submission unconditionally.
    // No trace entry is emitted (mirrors the "gate never reached" signal
    // documented in the contract above; lift analyzers count cells with no
    // pre_submit_self_check trace entry as "gate skipped"). The ablation
    // harness uses this to isolate self-check's contribution per model tier.
    if (!gates.selfCheck) return true;
    let result: { complete: boolean; missing: string[]; error?: string };
    let gateError: string | null = null;
    try {
      result = await runPreSubmitSelfCheck({
        router: input.router,
        instruction: instructionSummary.original,
        bashTraceTail: collectBashTraceTail(),
        ...(input.pathProfile === undefined ? {} : { pathProfile: input.pathProfile }),
      });
    } catch (err) {
      // `runPreSubmitSelfCheck` already converts auditor-level errors
      // (chatText throws, parse failures) into `{complete: true,
      // missing: [], error: "..."}`, so reaching this catch implies an
      // unexpected throw (router itself unavailable, bug in the helper).
      // Fail open AND emit a trace entry tagged with the error so the
      // lift analyzer can count gate-runs that produced no verdict.
      gateError = err instanceof Error ? err.message : String(err);
      result = { complete: true, missing: [] };
    }
    // Surface the helper's own (caught) error too — same audit path.
    const errorMessage = gateError ?? result.error ?? null;
    pushTrace({
      kind: "pre_submit_self_check",
      at: now(),
      complete: result.complete,
      missing: result.missing,
      gate_reached: true,
      ...(errorMessage !== null ? { error: errorMessage } : {}),
    });
    if (errorMessage !== null) return true;
    if (result.complete || result.missing.length === 0) return true;
    if (
      selfCheckRetriesUsed >= SELF_CHECK_MAX_RETRIES ||
      turnsUsed >= cfg.maxTurns ||
      input.signal?.aborted
    ) {
      return true;
    }
    const lines = [
      "[BAG pre-submit self-check] You declared the task complete, but these requirements appear unmet:",
      ...result.missing.map((item) => `- ${item}`),
      "Fix them and submit again.",
    ];
    messages.push(userReminder(lines.join("\n")));
    selfCheckRetriesUsed += 1;
    selfCheckRetryQueued = true;
    return false;
  };

  while (attemptsUsed < totalAllowedAttempts || selfCheckRetryQueued) {
    // A self-check retry is "free" — it doesn't consume an attempt slot,
    // it just lets the inner loop run again with feedback already injected.
    const isSelfCheckRetry = selfCheckRetryQueued;
    selfCheckRetryQueued = false;
    if (!isSelfCheckRetry) {
      attemptsUsed += 1;
      attemptStart = {
        promptTokens: totalPromptTokens,
        completionTokens: totalCompletionTokens,
        toolCalls: toolCallsExecuted,
        turns: turnsUsed,
      };
    }
    // Reset per-attempt mutable book-keeping; turn budget is shared across
    // attempts via the cfg.maxTurns cap, format errors reset between attempts.
    consecutiveFormatErrors = 0;
    stopReason = "end_turn";
    submittedOutput = null;
    let submittedThisAttempt = false;

    while (turnsUsed < cfg.maxTurns) {
      if (input.signal?.aborted) {
        pushTrace({ kind: "abort", at: now() });
        stopReason = "cancelled";
        break;
      }
      turnsUsed += 1;
      const response = await input.router.chatTextWithTools({
        role: "master",
        messages,
        tools,
        toolChoice: "auto",
        maxTokens: 4096,
        purpose: "autonomous-coding-turn",
      });
      if (response.promptTokens != null) totalPromptTokens += response.promptTokens;
      if (response.completionTokens != null) totalCompletionTokens += response.completionTokens;

      const assistantMessage: AssistantWithToolCalls = {
        role: "assistant",
        content: response.textContent.length > 0 ? response.textContent : null,
        tool_calls: response.toolCalls.map((tc) => ({
          id: tc.id,
          type: "function" as const,
          function: { name: tc.name, arguments: tc.argumentsJson },
        })),
      };
      messages.push(assistantMessage);
      pushTrace({
        kind: "assistant",
        at: now(),
        text: response.textContent,
        toolCallCount: response.toolCalls.length,
      });

      if (response.toolCalls.length === 0) {
        // Format error: model produced text but no tool call. Two paths:
        //   (a) finish_reason is 'stop' AND content suggests completion — accept end_turn.
        //   (b) finish_reason is 'stop' but no completion signal — coax model.
        const looksLikeFinish =
          response.finishReason === "stop" &&
          (response.textContent.toLowerCase().includes("complete") ||
            response.textContent.toLowerCase().includes("done"));
        if (looksLikeFinish) {
          stopReason = "end_turn";
          break;
        }
        consecutiveFormatErrors += 1;
        pushTrace({
          kind: "format_error",
          at: now(),
          reason: `assistant turn produced no tool_calls (finish=${response.finishReason})`,
        });
        if (consecutiveFormatErrors >= cfg.maxConsecutiveFormatErrors) {
          stopReason = "error";
          break;
        }
        messages.push(
          userReminder(
            `Every response MUST include exactly one bash tool call. To finish, call bash with \`echo ${SUBMIT_SENTINEL}\` as the only command. Continue.`,
          ),
        );
        continue;
      }

      consecutiveFormatErrors = 0;

      let submittedThisTurn = false;
      for (const toolCall of response.toolCalls) {
        if (input.signal?.aborted) {
          pushTrace({ kind: "abort", at: now() });
          stopReason = "cancelled";
          break;
        }
        const isCodeSearchCall =
          toolCall.name === CODE_SEARCH_TOOL_NAME && codeSearchEnabled;
        const isViewImageCall =
          toolCall.name === VIEW_IMAGE_TOOL_NAME && viewImageEnabled;
        const isEditStrategyCall = editStrategyToolNames.has(toolCall.name);
        if (
          toolCall.name !== BASH_TOOL_NAME &&
          !isViewImageCall &&
          !isCodeSearchCall &&
          !isEditStrategyCall
        ) {
          const knownTools = [
            BASH_TOOL_NAME,
            ...(viewImageEnabled ? [VIEW_IMAGE_TOOL_NAME] : []),
            ...(codeSearchEnabled ? [CODE_SEARCH_TOOL_NAME] : []),
            ...editStrategyToolDefs.map((def) => def.function.name),
          ];
          const known = knownTools.join(", ");
          pushTrace({
            kind: "format_error",
            at: now(),
            reason: `unknown tool '${toolCall.name}', only '${known}' are available`,
          });
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: `error: unknown tool '${toolCall.name}'. Available tools: ${known}.`,
          });
          continue;
        }

        if (isEditStrategyCall) {
          let parsedArgs: unknown;
          try {
            parsedArgs = JSON.parse(toolCall.argumentsJson || "{}");
          } catch (parseErr) {
            const message = parseErr instanceof Error ? parseErr.message : String(parseErr);
            pushTrace({
              kind: "format_error",
              at: now(),
              reason: `${toolCall.name} arguments JSON parse failure: ${message}`,
            });
            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: `error: ${toolCall.name} arguments JSON parse failure: ${message}. Re-emit with valid JSON arguments.`,
            });
            continue;
          }
          pushTrace({
            kind: "tool_call",
            at: now(),
            toolCallId: toolCall.id,
            tool: toolCall.name,
            argumentsJson: toolCall.argumentsJson,
          });
          toolCallsExecuted += 1;
          try {
            const editResult = await editStrategy.dispatch(toolCall.name, parsedArgs, {
              cwd: input.cwd,
              emit: (entry) => {
                pushTrace({
                  kind: "edit_dispatch",
                  at: now(),
                  strategy: entry.strategy,
                  tool: entry.tool,
                  target: entry.target,
                  outcome: entry.outcome,
                  bytes_changed: entry.bytesChanged,
                  retries_within_strategy: entry.retriesWithinStrategy,
                });
              },
            });
            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: editResult.observation,
            });
          } catch (execError) {
            const message = execError instanceof Error ? execError.message : String(execError);
            pushTrace({
              kind: "format_error",
              at: now(),
              reason: `${toolCall.name} dispatch failed: ${message}`,
            });
            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: `error: ${message}`,
            });
          }
          continue;
        }

        if (isCodeSearchCall) {
          pushTrace({
            kind: "tool_call",
            at: now(),
            toolCallId: toolCall.id,
            tool: toolCall.name,
            argumentsJson: toolCall.argumentsJson,
          });
          toolCallsExecuted += 1;
          let codeSearchResult: CodeSearchToolResult;
          try {
            codeSearchResult = await dispatchCodeSearch(toolCall.argumentsJson);
          } catch (execError) {
            const message = execError instanceof Error ? execError.message : String(execError);
            codeSearchResult = {
              ok: false,
              observation: `code_search dispatch crashed: ${message}. Fall back to bash + rg.`,
              hits: [],
              backendStatus: "error",
              durationMs: 0,
              errorMessage: message,
            };
          }
          let querySnippet = "";
          try {
            const obj = JSON.parse(toolCall.argumentsJson) as { query?: unknown };
            if (typeof obj.query === "string") querySnippet = obj.query.slice(0, 240);
          } catch {
            /* ignore */
          }
          pushTrace({
            kind: "code_search",
            at: now(),
            toolCallId: toolCall.id,
            query: querySnippet,
            hitCount: codeSearchResult.hits.length,
            backendStatus: codeSearchResult.backendStatus,
            durationMs: codeSearchResult.durationMs,
          });
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: codeSearchResult.observation,
          });
          continue;
        }

        if (toolCall.name === VIEW_IMAGE_TOOL_NAME) {
          let parsedPath: string | null = null;
          let parseError: string | null = null;
          try {
            const obj = JSON.parse(toolCall.argumentsJson) as { path?: unknown };
            parsedPath = typeof obj.path === "string" && obj.path.length > 0 ? obj.path : null;
            if (parsedPath == null) parseError = "view_image requires a non-empty 'path' string";
          } catch (parseErr) {
            parseError = `view_image arguments JSON parse failure: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`;
          }
          if (parseError != null || parsedPath == null) {
            pushTrace({ kind: "format_error", at: now(), reason: parseError ?? "missing path" });
            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: `error: ${parseError ?? "missing path"}. Re-emit view_image with {"path":"..."}.`,
            });
            continue;
          }
          pushTrace({
            kind: "tool_call",
            at: now(),
            toolCallId: toolCall.id,
            tool: toolCall.name,
            argumentsJson: toolCall.argumentsJson,
          });
          toolCallsExecuted += 1;
          try {
            const imageResult = await executeViewImageTool({
              client: input.client,
              sessionId: input.sessionId,
              cwd: input.cwd,
              path: parsedPath,
            });
            pushTrace({
              kind: "tool_result",
              at: now(),
              toolCallId: toolCall.id,
              result: {
                output: imageResult.observation,
                truncatedOutput: imageResult.observation,
                exitCode: imageResult.ok ? 0 : -1,
                signal: null,
                durationMs: 0,
                truncated: false,
                submitted: false,
              },
            });
            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: imageResult.observation,
            });
            if (imageResult.ok && imageResult.base64 != null) {
              pendingImages.push({
                mimeType: imageResult.mimeType,
                base64: imageResult.base64,
                path: imageResult.path,
              });
            }
          } catch (execError) {
            const message = execError instanceof Error ? execError.message : String(execError);
            pushTrace({ kind: "format_error", at: now(), reason: `view_image failed: ${message}` });
            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: `error: ${message}`,
            });
          }
          continue;
        }

        // toolCall.name === BASH_TOOL_NAME
        const { command, timeoutSec, error: parseError } = parseBashArguments(toolCall.argumentsJson);
        if (parseError != null) {
          pushTrace({ kind: "format_error", at: now(), reason: parseError });
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: `error: ${parseError}. Re-emit the bash call with valid JSON arguments {"command":"..."}.`,
          });
          continue;
        }
        pushTrace({
          kind: "tool_call",
          at: now(),
          toolCallId: toolCall.id,
          tool: toolCall.name,
          argumentsJson: toolCall.argumentsJson,
        });
        toolCallsExecuted += 1;

        try {
          const result = await executeBashTool({
            sessionId: input.sessionId,
            cwd: input.cwd,
            command,
            ...(timeoutSec === undefined ? {} : { timeoutSec }),
            client: input.client,
            submitSentinel: sentinel,
          });
          pushTrace({ kind: "tool_result", at: now(), toolCallId: toolCall.id, result });
          messages.push(renderObservationMessage(toolCall.id, result));
          if (result.submitted) {
            submittedOutput = result.output;
            submittedThisTurn = true;
            stopReason = "submitted";
            break;
          }
        } catch (execError) {
          const message = execError instanceof Error ? execError.message : String(execError);
          pushTrace({ kind: "format_error", at: now(), reason: `bash execution failed: ${message}` });
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: `error: ${message}`,
          });
        }
      }
      // After tool calls finish for this turn: if any view_image queued an
      // image, append it as a multimodal user message so the next assistant
      // turn perceives it directly.
      if (pendingImages.length > 0) {
        const blocks: Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }> = [
          { type: "text", text: `[BAG] ${pendingImages.length} image(s) attached for visual analysis:` },
        ];
        for (const img of pendingImages) {
          blocks.push({
            type: "image_url",
            image_url: { url: `data:${img.mimeType};base64,${img.base64}` },
          });
          blocks.push({ type: "text", text: `(${img.path})` });
        }
        messages.push({ role: "user", content: blocks });
        pendingImages.length = 0;
      }
      if (submittedThisTurn) {
        submittedThisAttempt = true;
        break;
      }
      if (stopReason === "cancelled") break;
    }

    // Promote end_turn -> max_turns if turn budget exhausted (parity with prior behavior).
    if (stopReason === "end_turn" && turnsUsed >= cfg.maxTurns) {
      stopReason = "max_turns";
    }

    // Post-submit verification path. Only runs when:
    //   - the attempt actually submitted (BAG_TASK_COMPLETE sentinel), and
    //   - a verifier callback was provided.
    if (submittedThisAttempt && verifier) {
      let verifierResult: PostSubmitVerifierResult;
      try {
        verifierResult = await verifier({
          client: input.client,
          sessionId: input.sessionId,
          cwd: input.cwd,
        });
      } catch (verifyError) {
        const msg = verifyError instanceof Error ? verifyError.message : String(verifyError);
        verifierResult = {
          passed: false,
          output: `verifier callback threw: ${msg}`,
          exitCode: null,
        };
      }
      lastVerifierPassed = verifierResult.passed;
      pushTrace({
        kind: "attempt",
        at: now(),
        attempt: attemptsUsed,
        verifier_passed: verifierResult.passed,
        verifier_output: verifierResult.output,
        verifier_exit_code: verifierResult.exitCode,
        prompt_tokens: totalPromptTokens - attemptStart.promptTokens,
        completion_tokens: totalCompletionTokens - attemptStart.completionTokens,
        tool_calls_executed: toolCallsExecuted - attemptStart.toolCalls,
        turns_used: turnsUsed - attemptStart.turns,
      });
      if (verifierResult.passed) {
        // Verifier passed — run the generic pre-submit self-check as a
        // separate gate. If the auditor flags unmet requirements AND a
        // self-check retry is still available, the gate pushes a feedback
        // message and queues a retry; the outer loop will re-enter the
        // inner conversation without consuming a verifier-style attempt.
        const accepted = await runSelfCheckGate();
        if (!accepted) continue;
        anyAttemptPassed = true;
        stopReason = "submitted";
        break;
      }
      // Verifier failed. If retries remain AND we have turn budget, append
      // feedback message and retry within the same conversation. Match the
      // verifier complaint against (1) the auto-discovered failure-cluster
      // index PRIMARY, (2) the curated verifier-signature-library as a
      // safety-net fallback. The clusters-first order reflects the
      // retirement plan in `docs/bag-verifier-signature-retirement.md`:
      // cluster matcher recall is now ≥5/8 vs the curated library, and we
      // expect that gap to close as the BAG corpus grows. The curated
      // library remains for: typecheck-missing-import (TS errors that the
      // pytest-shaped corpus has no equivalent for), acp-internal-error
      // (rare ACP crashes — corpus doesn't have enough volume yet), and
      // the catchall (generic verifier-rejection nudge).
      //
      // Each retry emits a `retry_hint` trace entry with `source` so audit
      // pipelines (bench/audit/...) can track library-vs-cluster hit rates;
      // when library hit-rate drops below 5% over 30 BAG runs, retire.
      const retriesRemaining = totalAllowedAttempts - attemptsUsed;
      if (retriesRemaining > 0 && turnsUsed < cfg.maxTurns && !input.signal?.aborted) {
        const exitCodeStr = verifierResult.exitCode == null ? "n/a" : String(verifierResult.exitCode);
        const taskNameSnippet: string = input.task.split("\n")[0]?.slice(0, 80) ?? "";
        // PRIMARY: auto-discovered failure clusters (data-driven, retirement target).
        // Gate: clusterMatcher disabled → skip cluster lookup entirely; the
        // curated verifier-signature-library still has a chance to fire below.
        const clustersDoc = gates.clusterMatcher ? getFailureClusters() : null;
        const matchedCluster = clustersDoc
          ? matchClusterByVerifierOutput(
              clustersDoc,
              verifierResult.output,
              FAILURE_CLUSTER_MATCH_THRESHOLD,
            )
          : null;
        // FALLBACK: curated library (soft-deprecated, kept as safety net).
        const matchedSignature = matchVerifierSignature({
          taskName: taskNameSnippet,
          verifierOutput: verifierResult.output,
        });
        const messageLines: string[] = [];
        if (matchedCluster !== null) {
          messageLines.push(
            [
              "[BAG corpus auto-cluster — past trials clustered to this failure mode]:",
              `  cluster: ${matchedCluster.name} (size ${matchedCluster.size}, tasks: ${matchedCluster.tasks.slice(0, 4).join(", ")})`,
              `  exemplar excerpt: ${matchedCluster.exemplarVerifierExcerpt.slice(0, 240)}`,
              `  prior trial sample: ${matchedCluster.trialIds.slice(0, 3).join(", ")}`,
            ].join("\n"),
          );
          messageLines.push("");
        }
        if (matchedSignature !== null) {
          messageLines.push(renderHintForRetry(matchedSignature));
          messageLines.push("");
        }
        // Telemetry: which retry-hint source fired? Used by the retirement
        // audit (when `library` < 5% of fires across 30 runs, delete the
        // curated library).
        const hintSource: "cluster" | "library" | "both" | "none" =
          matchedCluster !== null && matchedSignature !== null
            ? "both"
            : matchedCluster !== null
              ? "cluster"
              : matchedSignature !== null
                ? "library"
                : "none";
        const hintEntry: AutonomousTurnTraceEntry = {
          kind: "retry_hint",
          at: now(),
          attempt: attemptsUsed,
          source: hintSource,
          ...(matchedCluster !== null ? { cluster_id: matchedCluster.id } : {}),
          ...(matchedSignature !== null ? { library_id: matchedSignature.id } : {}),
        };
        pushTrace(hintEntry);
        messageLines.push(
          `The submitted solution failed verification. Verifier output:`,
          verifierResult.output,
          `<exit code> ${exitCodeStr}`,
          `Fix the issues and submit again with \`echo ${sentinel}\`.`,
        );
        messages.push(userReminder(messageLines.join("\n")));
        continue;
      }
      // No retry budget — but BEFORE accepting `submitted_but_failed` as the
      // terminal state, run the generic self-check gate one last time. The
      // self-check looks at the full bash trace and the original instruction;
      // it can flag patterns the probe-based verifier missed (e.g. a fix that
      // works only in the agent's session because of `export PATH=` and won't
      // be visible to a fresh subprocess like the harbor verifier). If the
      // self-check finds something AND we still have turn budget, the outer
      // loop runs another conversation round without consuming an attempt.
      const accepted = await runSelfCheckGate();
      if (!accepted) continue;
      stopReason = "submitted_but_failed";
      break;
    }

    // No verifier OR did not submit. Behavior matches single-attempt original.
    if (submittedThisAttempt) {
      // Submitted with no verifier configured — gate via the generic
      // pre-submit self-check. Same retry-queueing semantics as the
      // verifier-pass path: on auditor flag with budget remaining, the
      // outer loop re-runs the inner conversation without burning an
      // attempt slot.
      const accepted = await runSelfCheckGate();
      if (!accepted) continue;
      anyAttemptPassed = true;
      stopReason = "submitted";
      break;
    }
    // Did not submit (end_turn / max_turns / cancelled / error). Exit.
    break;
  }

  // Final stopReason adjustment: if any attempt passed, force "submitted".
  if (anyAttemptPassed) {
    stopReason = "submitted";
  } else if (
    verifier &&
    submittedOutput != null &&
    !lastVerifierPassed &&
    stopReason !== "cancelled" &&
    stopReason !== "error"
  ) {
    stopReason = "submitted_but_failed";
  }

  return {
    stopReason,
    turnsUsed,
    toolCallsExecuted,
    totalPromptTokens,
    totalCompletionTokens,
    trace,
    submittedOutput,
    attemptsUsed: Math.max(1, attemptsUsed),
  };
};
