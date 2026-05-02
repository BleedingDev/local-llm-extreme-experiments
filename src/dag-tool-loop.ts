/**
 * DAG-driven tool-use loop.
 *
 * Combines BAG's planning instinct with the bash-only autonomous loop:
 *   1. A single LLM call ("lite planner") decomposes the task into 1-5 issues
 *      with optional verifier commands.
 *   2. For each issue (in topological order), a scoped tool-use loop runs
 *      until the model emits BAG_ISSUE_COMPLETE.
 *   3. Optional per-issue verification gates progression. On verifier failure
 *      the loop re-enters the same issue with the verifier output prepended
 *      as context (max repair rounds).
 *   4. Aggregate per-issue results into a single AutonomousTurnResult-shape
 *      summary.
 */

import {
  runAutonomousCodingTurn,
  type AutonomousTurnTraceEntry,
  type PostSubmitVerifier,
} from "./autonomous-coding-turn";
import type { AcpTerminalClient } from "./autonomous-tools";
import { executeBashTool, renderBashObservation } from "./autonomous-tools";
import type { LlmRouter } from "./llm";
import { parseJsonObject } from "./llm";
import { maybeSummarizeInstruction } from "./instruction-summarizer";

export const ISSUE_SUBMIT_SENTINEL = "BAG_ISSUE_COMPLETE";

export type DagToolLoopIssue = {
  issueId: string;
  title: string;
  body: string;
  expectedFiles: string[];
  verifierCommands: string[];
};

export type DagToolLoopIssueResult = {
  issue: DagToolLoopIssue;
  stopReason: string;
  turnsUsed: number;
  bashCallsExecuted: number;
  promptTokens: number;
  completionTokens: number;
  trace: AutonomousTurnTraceEntry[];
  verifierExitCodes: number[];
  verifierPassed: boolean;
  verifierOutputs: string[];
  repairRoundsUsed: number;
};

export type DagToolLoopResult = {
  stopReason: "submitted" | "blocked" | "error" | "cancelled";
  issues: DagToolLoopIssueResult[];
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalBashCalls: number;
  plannedIssueCount: number;
  passedIssueCount: number;
};

const SYSTEM_PROMPT_PLANNER = `\
You are BleedingAgent's lite planner for autonomous coding.
Given a task, decompose it into the SMALLEST sequence of issues a coding agent must solve, in dependency order.

Rules:
- 1 issue is acceptable when the task is atomic ("create greet.py", "fix one regex"). Do NOT pad.
- Maximum 5 issues. Each issue should be solvable in <30 bash calls.
- Issues are SEQUENTIAL; later issues may assume previous ones succeeded.
- Each issue has a verifier: a list of bash commands whose exit codes must all be 0 for the issue to be considered complete. Verifiers should be CONCRETE and CHEAP — \`test -f path\`, \`python3 -c "import x"\`, \`grep -q ... file\`, etc. Empty array if no obvious verifier.

Return JSON ONLY (no prose, no fences):
{"issues":[{"issueId":"task-1-...","title":"...","body":"...","expectedFiles":["relative/path"],"verifierCommands":["bash -c '...'"]}]}
`;

const buildPlannerPrompt = (task: string, repoContext: string): string =>
  [
    `Task:\n${task.trim()}`,
    "",
    `Workspace listing & repo context:`,
    repoContext.slice(0, 12000),
  ].join("\n");

export const planDagIssues = async (input: {
  router: LlmRouter;
  task: string;
  repoContext: string;
}): Promise<DagToolLoopIssue[]> => {
  const fallback: DagToolLoopIssue[] = [
    {
      issueId: "task-1-direct",
      title: "Solve the task directly",
      body: input.task,
      expectedFiles: [],
      verifierCommands: [],
    },
  ];
  if (!input.router.masterAvailable) return fallback;

  // Universal lesson #1 (long files = token hemorrhage): when the task instruction
  // is large, summarize via the cheap local role before passing to the planner.
  let plannerTask = input.task;
  try {
    const summary = await maybeSummarizeInstruction({ router: input.router, task: input.task });
    if (summary.summarized) {
      plannerTask = summary.summary;
    }
  } catch {
    // Summarizer failure is non-fatal — fall back to original.
  }

  try {
    // classifier runs on the cheap local role — cost split.
    const raw = await input.router.chatText({
      role: "local",
      json: true,
      maxTokens: 1500,
      purpose: "dag-planner",
      messages: [
        { role: "system", content: SYSTEM_PROMPT_PLANNER },
        { role: "user", content: buildPlannerPrompt(plannerTask, input.repoContext) },
      ],
    });
    const parsed = parseJsonObject<{ issues?: unknown }>(raw, { issues: [] });
    const candidates = Array.isArray(parsed.issues) ? parsed.issues : [];
    const issues: DagToolLoopIssue[] = candidates
      .slice(0, 5)
      .filter((entry): entry is Record<string, unknown> => entry != null && typeof entry === "object")
      .map((entry, index) => {
        const issueId =
          typeof entry.issueId === "string" && entry.issueId.length > 0
            ? entry.issueId
            : `task-${index + 1}`;
        const title =
          typeof entry.title === "string" && entry.title.length > 0 ? entry.title : `Issue ${index + 1}`;
        const body = typeof entry.body === "string" ? entry.body : "";
        const expectedFiles = Array.isArray(entry.expectedFiles)
          ? entry.expectedFiles.map(String).filter((s) => s.length > 0).slice(0, 16)
          : [];
        const verifierCommands = Array.isArray(entry.verifierCommands)
          ? entry.verifierCommands.map(String).filter((s) => s.length > 0).slice(0, 8)
          : [];
        return { issueId, title, body, expectedFiles, verifierCommands };
      })
      .filter((issue) => issue.title.length > 0 && issue.body.length > 0);
    return issues.length > 0 ? issues : fallback;
  } catch {
    return fallback;
  }
};

const runIssueVerifiers = async (input: {
  client: AcpTerminalClient;
  sessionId: string;
  cwd: string;
  commands: readonly string[];
}): Promise<{ exitCodes: number[]; outputs: string[]; passed: boolean }> => {
  const exitCodes: number[] = [];
  const outputs: string[] = [];
  for (const cmd of input.commands) {
    try {
      const r = await executeBashTool({
        client: input.client,
        sessionId: input.sessionId,
        cwd: input.cwd,
        command: cmd,
        timeoutSec: 60,
      });
      exitCodes.push(r.exitCode ?? -1);
      outputs.push(renderBashObservation(r));
    } catch (e) {
      exitCodes.push(-1);
      outputs.push(`verifier execution error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  const passed = input.commands.length === 0 ? true : exitCodes.every((c) => c === 0);
  return { exitCodes, outputs, passed };
};

const ISSUE_SYSTEM_PROMPT = (sentinel: string): string => `\
You are BleedingAgent in autonomous coding mode, working on ONE specific issue from a larger plan. You have exactly one tool: \`bash\`.

Workflow:
1. Read the issue body carefully. Note the expected files and the verifier commands you must satisfy.
2. Investigate the workspace via bash (\`ls\`, \`cat\`, \`grep\`).
3. Make changes via here-docs (\`cat <<'EOF' > path ... EOF\`), \`sed -i\`, \`printf >> path\`, etc.
4. Locally re-run the verifier commands to confirm they pass.
5. When the verifier commands pass, run \`echo ${sentinel}\` as the only command to mark this issue complete.

Hard rules:
- Each bash call runs in a NEW subshell. cwd and env do NOT persist across calls. Always chain \`cd <workdir> && ...\`.
- Never run \`echo ${sentinel}\` together with anything else; it must be the only command.
- Do not jump to the next issue — submit this one first; the orchestrator will start the next one.
- If output is elided, narrow with \`head\`/\`tail\`/\`sed -n\`.
- Do not ask questions; proceed with reasonable defaults.

Available tool: \`bash(command, timeout_sec?)\`.
`;

const buildIssueContextSuffix = (
  issue: DagToolLoopIssue,
  index: number,
  total: number,
  previousResults: readonly DagToolLoopIssueResult[],
  repairFeedback?: string,
): string => {
  const lines: string[] = [
    `Current issue: ${index + 1}/${total} — ${issue.title}`,
    `Issue body:\n${issue.body}`,
  ];
  if (issue.expectedFiles.length > 0) {
    lines.push(`Expected files:\n  ${issue.expectedFiles.join("\n  ")}`);
  }
  if (issue.verifierCommands.length > 0) {
    lines.push(`Verifier commands (must all exit 0):\n  ${issue.verifierCommands.join("\n  ")}`);
  } else {
    lines.push("No machine verifier for this issue. Submit when the body is satisfied.");
  }
  if (previousResults.length > 0) {
    lines.push("");
    lines.push(`Previous issues completed: ${previousResults.length}.`);
    for (const prev of previousResults.slice(-3)) {
      lines.push(
        `  - ${prev.issue.title}: stopReason=${prev.stopReason}, verifierPassed=${prev.verifierPassed}`,
      );
    }
  }
  if (repairFeedback != null) {
    lines.push("");
    lines.push("Verifier results from previous attempt at THIS issue:");
    lines.push(repairFeedback);
    lines.push("");
    lines.push("Adjust your changes so every verifier command exits 0, then submit again.");
  }
  return lines.join("\n");
};

export const runDagToolLoop = async (input: {
  router: LlmRouter;
  client: AcpTerminalClient;
  sessionId: string;
  cwd: string;
  task: string;
  repoContext: string;
  signal?: AbortSignal;
  maxRepairRoundsPerIssue?: number;
  maxTurnsPerIssue?: number;
  /**
   * Per-attempt retry budget for the inner `runAutonomousCodingTurn`. When a
   * per-issue submit fails verification AND retries remain, the inner loop
   * stays in the same conversation and asks the model to fix-and-resubmit
   * before falling out to the outer repair round.
   */
  maxAttemptRetriesPerIssue?: number;
  /**
   * Optional caller-provided post-submit verifier. When supplied, replaces the
   * synthesized verifier built from `issue.verifierCommands`. The default
   * (synthesized) verifier executes each `verifierCommands` entry and passes
   * iff every command exits 0.
   */
  verifyAfterSubmit?: PostSubmitVerifier;
  /**
   * Optional runtime hint propagated from the task-shape router (e.g. when
   * `requiresLongWait === true`). Forwarded verbatim into each per-issue
   * `runAutonomousCodingTurn` invocation so the master model sees the hint
   * on every issue regardless of which one happens to schedule the long-wait
   * step.
   */
  runtimeHint?: string;
  onIssueStart?: (issue: DagToolLoopIssue, index: number, total: number) => Promise<void> | void;
  onIssueComplete?: (result: DagToolLoopIssueResult, index: number, total: number) => Promise<void> | void;
  onPlanned?: (issues: DagToolLoopIssue[]) => Promise<void> | void;
}): Promise<{ planned: DagToolLoopIssue[]; result: DagToolLoopResult }> => {
  const maxRepair = input.maxRepairRoundsPerIssue ?? 1;
  const maxTurnsPerIssue = input.maxTurnsPerIssue ?? 30;
  // Default 0 to preserve historical DAG-tools behavior: prior callers got a
  // single inner attempt + outer repair-rounds. Best-of-N inner retries are
  // opt-in via maxAttemptRetriesPerIssue or an explicit verifyAfterSubmit.
  const maxAttemptRetriesPerIssue = input.maxAttemptRetriesPerIssue ?? 0;

  const planned = await planDagIssues({
    router: input.router,
    task: input.task,
    repoContext: input.repoContext,
  });
  await input.onPlanned?.(planned);

  const issueResults: DagToolLoopIssueResult[] = [];
  let totalPromptTokens = 0;
  let totalCompletionTokens = 0;
  let totalBashCalls = 0;

  for (let i = 0; i < planned.length; i++) {
    if (input.signal?.aborted) {
      return {
        planned,
        result: {
          stopReason: "cancelled",
          issues: issueResults,
          totalPromptTokens,
          totalCompletionTokens,
          totalBashCalls,
          plannedIssueCount: planned.length,
          passedIssueCount: issueResults.filter((r) => r.verifierPassed && r.stopReason === "submitted").length,
        },
      };
    }
    const issue = planned[i];
    if (issue == null) continue;
    await input.onIssueStart?.(issue, i, planned.length);

    let repairRounds = 0;
    let lastVerifier: { exitCodes: number[]; outputs: string[]; passed: boolean } = {
      exitCodes: [],
      outputs: [],
      passed: false,
    };
    let lastTurn: Awaited<ReturnType<typeof runAutonomousCodingTurn>> | null = null;
    let repairFeedback: string | undefined;

    // Build a per-issue post-submit verifier. If the caller passed one, use
    // it verbatim; otherwise (and only if best-of-N is enabled) synthesize
    // one from the planner's verifier commands. Used by
    // `runAutonomousCodingTurn`'s best-of-N retry loop so a sentinel-
    // submitted-but-verifier-failed attempt can be retried within the same
    // conversation before falling out to the outer repair round.
    const synthesizeFromVerifierCommands =
      maxAttemptRetriesPerIssue > 0 && issue.verifierCommands.length > 0;
    const issueVerifier: PostSubmitVerifier | undefined =
      input.verifyAfterSubmit ??
      (synthesizeFromVerifierCommands
        ? async (vIn) => {
            const v = await runIssueVerifiers({
              client: vIn.client,
              sessionId: vIn.sessionId,
              cwd: vIn.cwd,
              commands: issue.verifierCommands,
            });
            const output = v.outputs
              .map((line, idx) => `[${idx + 1}] ${line}`)
              .join("\n\n")
              .slice(0, 4000);
            const firstNonZero = v.exitCodes.find((c) => c !== 0);
            return {
              passed: v.passed,
              output,
              exitCode: firstNonZero ?? (v.exitCodes[0] ?? null),
            };
          }
        : undefined);

    while (true) {
      const turn = await runAutonomousCodingTurn({
        router: input.router,
        client: input.client,
        sessionId: input.sessionId,
        cwd: input.cwd,
        task: input.task,
        ...(input.signal === undefined ? {} : { signal: input.signal }),
        ...(issueVerifier === undefined ? {} : { verifyAfterSubmit: issueVerifier }),
        config: {
          submitSentinel: ISSUE_SUBMIT_SENTINEL,
          systemPromptOverride: ISSUE_SYSTEM_PROMPT(ISSUE_SUBMIT_SENTINEL),
          contextSuffix: buildIssueContextSuffix(issue, i, planned.length, issueResults, repairFeedback),
          maxTurns: maxTurnsPerIssue,
          maxAttemptRetries: maxAttemptRetriesPerIssue,
          ...(input.runtimeHint === undefined ? {} : { runtimeHint: input.runtimeHint }),
        },
      });
      lastTurn = turn;
      totalPromptTokens += turn.totalPromptTokens;
      totalCompletionTokens += turn.totalCompletionTokens;
      totalBashCalls += turn.toolCallsExecuted;

      lastVerifier = await runIssueVerifiers({
        client: input.client,
        sessionId: input.sessionId,
        cwd: input.cwd,
        commands: issue.verifierCommands,
      });

      if (lastVerifier.passed && turn.stopReason === "submitted") break;
      if (turn.stopReason === "cancelled") break;
      if (repairRounds >= maxRepair) break;
      repairRounds += 1;
      repairFeedback = lastVerifier.outputs
        .map((line, idx) => `[${idx + 1}] ${line}`)
        .join("\n\n")
        .slice(0, 4000);
    }

    const result: DagToolLoopIssueResult = {
      issue,
      stopReason: lastTurn?.stopReason ?? "error",
      turnsUsed: lastTurn?.turnsUsed ?? 0,
      bashCallsExecuted: lastTurn?.toolCallsExecuted ?? 0,
      promptTokens: lastTurn?.totalPromptTokens ?? 0,
      completionTokens: lastTurn?.totalCompletionTokens ?? 0,
      trace: lastTurn?.trace ?? [],
      verifierExitCodes: lastVerifier.exitCodes,
      verifierOutputs: lastVerifier.outputs,
      verifierPassed: lastVerifier.passed,
      repairRoundsUsed: repairRounds,
    };
    issueResults.push(result);
    await input.onIssueComplete?.(result, i, planned.length);

    if (!result.verifierPassed && result.stopReason !== "submitted") {
      // Hard block — issue did not submit AND verifier failed. Stop the chain.
      return {
        planned,
        result: {
          stopReason: "blocked",
          issues: issueResults,
          totalPromptTokens,
          totalCompletionTokens,
          totalBashCalls,
          plannedIssueCount: planned.length,
          passedIssueCount: issueResults.filter((r) => r.verifierPassed && r.stopReason === "submitted").length,
        },
      };
    }
  }

  const passed = issueResults.filter((r) => r.verifierPassed && r.stopReason === "submitted").length;
  return {
    planned,
    result: {
      stopReason: passed === planned.length ? "submitted" : "blocked",
      issues: issueResults,
      totalPromptTokens,
      totalCompletionTokens,
      totalBashCalls,
      plannedIssueCount: planned.length,
      passedIssueCount: passed,
    },
  };
};
