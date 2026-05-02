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

/**
 * Lazy-loaded failure cluster index. Auto-discovered patterns complement
 * the curated verifier-signature library — same retry-hint role but
 * data-driven from the BAG trial corpus.
 */
let CACHED_FAILURE_CLUSTERS: FailureClustersDocument | null = null;
let FAILURE_CLUSTERS_LOADED = false;
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

export const SYSTEM_PROMPT_DEFAULT = `\
You are BleedingAgent in autonomous coding mode. You have access to these tools: \`bash\`, \`view_image\`, \`code_search\`.

Tool selection guide:
- \`bash\` + \`rg\`: EXACT tokens, identifiers, error strings, file paths. Always your default for "find this literal string".
- \`code_search\`: CONCEPTUAL questions ("where is auth middleware", "how is rate limiting handled", "where is the DAG cached"). Returns ranked file/line/symbol hits — read full bodies via \`bash\` once you have localized.
- \`view_image\`: load an image into your visual context (only when verification depends on perceiving an image).
Don't read large file bodies until you've localized via search.

Workflow:
1. Read the task carefully.
2. Investigate the workspace via bash (\`ls\`, \`cat\`, \`grep\`, etc.) before editing.
3. Reproduce the failure or required behaviour with a small bash check before making changes.
4. Edit files using here-docs (\`cat <<'EOF' > path ... EOF\`), \`sed -i\`, \`printf >> path\`, etc.
5. Test in /tmp (\`cp file /tmp/x && cd /tmp && gcc x ...\`) when you can — keeps the workspace clean for the verifier.
6. Re-run any verification command the task implies.
7. **Before submitting: \`ls -la\` the workspace and remove any compiled binaries, .o files, __pycache__, *.pyc, /tmp test artifacts, or other byproducts of testing that the verifier may not expect.** Verifiers frequently assert exact file lists; one stray binary == reward 0.
   COMPILED-LANGUAGE GATE — when the task's deliverable is a SOURCE file but you compiled an executable to test it (gcc, g++, cc, clang, rustc, cargo build, go build, make, etc.), explicitly remove the compiled artifact (\`rm -f <binary>\`) before submitting. Verifiers that assert the deliverable directory contains an exact set of files reject any byproduct.
   SCRATCH-DIR HYGIENE — anything you wrote under \`/tmp/\` (test scripts, log captures, build outputs, repro snippets, data dumps) MUST be removed before submitting unless the task explicitly placed a deliverable there. Today's verifier may not probe \`/tmp/\`, but tomorrow's clean-room verifier will. Run \`rm -rf /tmp/<your-paths>\` (or \`rm -rf /tmp/* 2>/dev/null || true\` if you do not own anything else under \`/tmp/\`) as part of your pre-submit pass.
8. **CRITICAL — pre-submit final-check pass:** before \`echo ${SUBMIT_SENTINEL}\`, do these checks in one bash call (chained with \`&&\` or in a heredoc):
   (a) Re-read the original task instruction line by line. Watch for plurals ("print **all** moves", "for **each** input"), edge cases ("if there are multiple X"), and END-TO-END flows ("then \`curl http://...\` should return Y").
   (b) For end-to-end flows, **literally run the verification command from the task description** (e.g. \`curl -s http://localhost:8080/hello.html\` and inspect output, not just that the service is up).
   (c) Confirm every output the task specified actually exists with the expected content (\`cat /app/move.txt\`, \`diff <expected> <actual>\`).
   (d) **SUBPROCESS-PATH GATE** — when the task says a tool must be "in PATH" / "available system-wide" / "callable as \`X\`", the actual verifier runs in a fresh subprocess (typically \`subprocess.run(['X', ...])\` from Python) which has the DEFAULT system PATH (\`/usr/local/bin:/usr/bin:/bin\`) and does NOT inherit your shell's \`export PATH=\`, aliases, or virtualenv activation. Verify your fix from a clean shell: \`bash -c 'unset PATH; PATH=/usr/local/bin:/usr/bin:/bin command -v X'\` must succeed. If it doesn't, persist the binary system-wide via one of: \`ln -s /full/path/X /usr/local/bin/X\`, \`cp X /usr/local/bin/\`, \`pip install --user X\`, or place the binary in a directory already on default PATH. \`export PATH=\` alone is insufficient.
   If any check disagrees, fix it BEFORE submitting. Do not submit on partial matches.
9. When everything is verified end-to-end, run \`echo ${SUBMIT_SENTINEL}\` as the only command in a single bash call to submit.

Hard rules:
- Each bash call runs in a NEW subshell. cwd and env do NOT persist across calls. Always chain \`cd /workdir && ...\` if you need a directory.
- Never run \`echo ${SUBMIT_SENTINEL}\` together with anything else; it must be the only command.
- If a command's output is elided, do NOT keep retrying — narrow with \`head\`, \`tail\`, \`sed -n\`, or write to a temp file then read it.
- Do not ask the user clarifying questions. Make a reasonable assumption and proceed; if you are blocked, write a short bash comment explaining and then submit.
- Prefer small, observable steps; you can always run another command.

Available tools: \`bash(command, timeout_sec?)\`, \`view_image(path)\`, \`code_search(query, top_k?, mode?, path_filter?, language_filter?)\`. Always include exactly one tool call per assistant turn.
`;

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
   * Optional event hook invoked synchronously every time a trace entry is
   * appended. Used by `src/sdk/agent-session.ts` to stream live events to
   * external embedders (SDK consumers, RPC adapters) without forking the
   * loop. Throwing from the hook is caught and ignored so subscribers cannot
   * destabilise the turn; subscribers wanting cancellation should signal via
   * the `signal` parameter.
   */
  onTraceEntry?: (entry: AutonomousTurnTraceEntry) => void;
}): Promise<AutonomousTurnResult> => {
  const cfg: AutonomousTurnConfig = { ...DEFAULT_CONFIG, ...(input.config ?? {}) };
  const sentinel = cfg.submitSentinel ?? SUBMIT_SENTINEL;
  const optimized = loadOptimizedExecutorPrompt();
  const exec_system = optimized?.system ?? SYSTEM_PROMPT_DEFAULT;
  if (optimized) console.log(`[bag] using optimized executor prompt run=${optimized.runId}`);
  const systemPrompt =
    cfg.systemPromptOverride != null
      ? cfg.systemPromptOverride
      : exec_system.replace(/BAG_TASK_COMPLETE/g, sentinel);
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

  // Code-search tool gating: `BAG_CODE_SEARCH=0` removes the tool from the
  // surface so the A/B harness can compare with-vs-without on the same model
  // / task. Default = enabled (the tool is always REGISTERED; whether the
  // backend binary is available is a separate runtime concern handled in
  // executeCodeSearchTool below).
  const codeSearchEnabled = process.env.BAG_CODE_SEARCH !== "0";
  const codeSearchBackend: CodebaseSearchBackend =
    cfg.codeSearchBackend ?? colgrepBackend();
  const tools: ChatWithToolsOptions["tools"] = codeSearchEnabled
    ? [BASH_TOOL_DEFINITION, VIEW_IMAGE_TOOL_DEFINITION, CODE_SEARCH_TOOL_DEFINITION]
    : [BASH_TOOL_DEFINITION, VIEW_IMAGE_TOOL_DEFINITION];
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
  const totalAllowedAttempts = verifier ? 1 + maxAttemptRetries : 1;

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
    let result: { complete: boolean; missing: string[]; error?: string };
    let gateError: string | null = null;
    try {
      result = await runPreSubmitSelfCheck({
        router: input.router,
        instruction: instructionSummary.original,
        bashTraceTail: collectBashTraceTail(),
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
        if (
          toolCall.name !== BASH_TOOL_NAME &&
          toolCall.name !== VIEW_IMAGE_TOOL_NAME &&
          !isCodeSearchCall
        ) {
          const known = codeSearchEnabled
            ? `${BASH_TOOL_NAME}, ${VIEW_IMAGE_TOOL_NAME}, ${CODE_SEARCH_TOOL_NAME}`
            : `${BASH_TOOL_NAME}, ${VIEW_IMAGE_TOOL_NAME}`;
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
      // feedback message and retry within the same conversation. Also try to
      // match the verifier complaint against the curated VERIFIER_SIGNATURE_LIBRARY
      // — if a known historical pattern matches, prepend its actionable fix
      // hint so the model has prior-art guidance, not just the raw verifier
      // output. This is the runtime hook for `docs/bag-failure-pattern-digest.md`.
      const retriesRemaining = totalAllowedAttempts - attemptsUsed;
      if (retriesRemaining > 0 && turnsUsed < cfg.maxTurns && !input.signal?.aborted) {
        const exitCodeStr = verifierResult.exitCode == null ? "n/a" : String(verifierResult.exitCode);
        const taskNameSnippet: string = input.task.split("\n")[0]?.slice(0, 80) ?? "";
        const matchedSignature = matchVerifierSignature({
          taskName: taskNameSnippet,
          verifierOutput: verifierResult.output,
        });
        const clustersDoc = getFailureClusters();
        const matchedCluster = clustersDoc
          ? matchClusterByVerifierOutput(clustersDoc, verifierResult.output)
          : null;
        const messageLines: string[] = [];
        if (matchedSignature !== null) {
          messageLines.push(renderHintForRetry(matchedSignature));
          messageLines.push("");
        }
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
