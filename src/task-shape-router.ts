/**
 * Adaptive task-shape router (Stage A).
 *
 * Classifies an incoming coding task into one of four shapes and maps it to
 * the BAG coding mode that is most likely to succeed. The empirical lessons
 * driving this router are captured in `docs/bag-tb-tool-use-vs-dag-tools.md`:
 *
 *   - atomic              -> tools     (no planning overhead)
 *   - compositional       -> dag-tools (planning is the win)
 *   - monolithic-complex  -> tools     (preserve global view)
 *   - hard / ambiguous    -> tools     (let the model improvise)
 *
 * The router makes a single chat-text master call with strict JSON output and
 * falls back to a safe default ("atomic" -> tools) on parse failure. A cheap
 * heuristic short-circuit handles trivial single-file tasks without an LLM
 * round trip.
 */

import { z } from "zod";
import { parseJsonObject, type LlmRouter } from "./llm";

export type TaskShape = "atomic" | "compositional" | "monolithic-complex" | "hard";
export type TaskShapeMode = "tools" | "dag-tools";

/**
 * Zod schema for `TaskShapeDecision`. Centralised so consumers / persisters can
 * validate routing artifacts without re-stating the field shape.
 *
 * `requiresLongWait` is additive (defaults to `false`) so artifacts written
 * before this dimension existed remain parseable.
 */
export const TaskShapeDecisionSchema = z.object({
  shape: z.enum(["atomic", "compositional", "monolithic-complex", "hard"]),
  mode: z.enum(["tools", "dag-tools"]),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
  requiresLongWait: z.boolean().default(false),
  tokens: z.object({ in: z.number(), out: z.number() }),
});

export type TaskShapeDecision = {
  shape: TaskShape;
  mode: TaskShapeMode;
  confidence: number;
  reasoning: string;
  /**
   * Semantic flag set by the classifier when the task involves long-running
   * asynchronous waits (>~30s wall clock) that are observable via
   * filesystem/log polling — VM boots, package source builds, network installs,
   * service startup before integration check, etc. When `true`, downstream
   * runners inject a runtime hint advising background-process + polling
   * patterns instead of blocking `sleep N` calls.
   *
   * Defaults to `false`. Inferred semantically by the classifier; consumers
   * MUST treat this as a signal, not a keyword match.
   */
  requiresLongWait: boolean;
  tokens: { in: number; out: number };
};

/**
 * Hint injected into the autonomous-turn user message when
 * `requiresLongWait === true`. Tuned for the qemu-alpine-ssh failure mode:
 * 72 bash calls / 14m blew the prompt timeout because the agent gated each VM
 * boot / package install behind a blocking `sleep`. The hint surfaces the
 * background-process + polling pattern so the agent can do parallel work while
 * a slow op runs.
 */
export const LONG_WAIT_RUNTIME_HINT = `\
[Task shape hint — this task involves long-running async operations]
You have a hard wall-clock budget for this whole task (typically 14-15 minutes).
Every blocking \`sleep N\` directly consumes that budget. The way to fit a
long async operation into a tight budget is to RUN IT IN THE BACKGROUND and
POLL for readiness, not to block.

Strict rules for long-wait operations:
1. NEVER call \`sleep\` with a duration over 30 seconds. If you think you need
   to wait longer, you are using the wrong pattern — switch to background +
   poll. Long sleeps stack with token-burn from each turn and exhaust the
   prompt budget catastrophically.
2. Spawn slow operations with \`nohup CMD > /tmp/X.log 2>&1 & echo $! > /tmp/X.pid\`.
   The agent's bash tool runs each call in a NEW subshell, so a foreground
   process dies when the call returns — you MUST detach with nohup + &.
3. Poll a readiness signal in a tight loop with bounded sleep granularity:
     \`until grep -q READY /tmp/X.log; do sleep 2; done\`  (or curl, or pgrep)
   Set a wall-clock guard:
     \`for i in $(seq 1 60); do grep -q READY /tmp/X.log && break; sleep 2; done\`
4. While a slow op is starting up, INVESTIGATE in parallel — read configs,
   inspect state, prepare the verification command — so when the readiness
   signal fires you submit immediately.
5. If a probe / verification step takes minutes (e.g. cold VM boot, source
   build, package install), KICK IT OFF FIRST in the background, then do the
   rest of the task while it boots. Don't serialise.

Anti-pattern (DO NOT DO):
   \`sleep 90; tail -50 /tmp/serial.log\`            # blocks for 90s
   \`sleep 120 && check_status\`                     # blocks for 2 minutes
Pattern (DO):
   \`nohup boot_vm.sh > /tmp/boot.log 2>&1 &\`
   ...do other prep work...
   \`for i in $(seq 1 120); do grep -q "login:" /tmp/boot.log && break; sleep 2; done\``;

const MODE_FOR_SHAPE: Record<TaskShape, TaskShapeMode> = {
  atomic: "tools",
  compositional: "dag-tools",
  "monolithic-complex": "tools",
  hard: "tools",
};

const SAFE_FALLBACK: TaskShapeDecision = {
  shape: "atomic",
  mode: "tools",
  confidence: 0.3,
  reasoning: "classifier failed; defaulting to safest",
  requiresLongWait: false,
  tokens: { in: 0, out: 0 },
};

export const SYSTEM_PROMPT_CLASSIFIER = `\
You are BleedingAgent's adaptive task-shape router. Your only job is to classify a
coding task into ONE of four shapes so the orchestrator can pick the right
execution mode. You see the task text and a compact repo-context snapshot.

Shapes (decide based on the natural structure of the work):
- "atomic": single file, single concept, one verifier check.
  Examples: "create greet.py that prints Hello, World!", "fix the off-by-one in foo.py line 42".
- "compositional": multiple INDEPENDENT sub-goals — each sub-goal can be
  designed, written, AND verified IN ISOLATION before the next begins. The
  sub-goals do not share mutable state during execution.
  Positive examples:
    "build X with gcov instrumentation, then make it available in PATH"
    "vendor library Y, write a wrapper script Z, add a smoke test"
    "create script A and script B, each with its own test"
  CRITICAL anti-pattern — these are NOT compositional, classify as
  "monolithic-complex" or "atomic" instead:
    "boot a VM, wait for SSH, authenticate, verify a service" (sequentially
    DEPENDENT — each step needs runtime state from the previous; fragmenting
    breaks ordering, the verifier checks the FINAL state, not intermediate)
    "start service X then configure it then run a query" (single coherent
    runtime; sub-goals share state)
    "spawn process A in the background, then exec B against it" (process
    lifetime spans the whole task)
  In doubt: if a verifier could pass for sub-goal N without sub-goal N+1
  having started, it's compositional; otherwise it's monolithic.
- "monolithic-complex": one cohesive change spanning multiple files that must
  all move together, OR a sequentially-stateful task where each step depends
  on runtime state from the previous one. Decomposing loses global view OR
  breaks ordering.
  Examples: "fix CVE-... in bottle.py", "refactor the routing layer", "boot
  a VM and ssh into it and verify a daemon", "set up a multi-service git
  webserver that survives a restart".
  HARD RULE — build pipelines are ALWAYS monolithic-complex, never compositional:
  if the task mentions ./configure, make, cmake, autotools, ninja, cargo build,
  go build, npm run build, mvn package, gradle, setup.py build_ext, OR involves
  "compile X with flag Y and install/expose to PATH", classify as monolithic-complex.
  These pipelines look compositional (extract → configure → build → install)
  but they share runtime state (config flags must propagate to compile, paths
  must propagate to install). Decomposing them loses flag continuity and
  the verifier checks the FINAL artifact, not intermediates.
- "hard": ambiguous, dual-constraint, novel, or polyglot tasks where neither
  decomposition nor a naive single pass is obviously right.
  Examples: "produce a polyglot file that is valid Python and valid Bash",
  "implement feature in both Rust and TS with matching APIs".

Mode mapping (binding):
  atomic -> tools
  compositional -> dag-tools
  monolithic-complex -> tools
  hard -> tools

Long-wait dimension (orthogonal to shape — set independently):
- Set \`requiresLongWait: true\` when the task requires waiting for an
  asynchronous operation that takes more than ~30 seconds AND is observable
  via filesystem / log polling. The signal is wall-clock latency, NOT the
  type of tool involved.
  Positive examples (set true):
    "boot alpine.iso in QEMU under TCG, then ssh in" (cold VM boot = minutes)
    "build openssl from source then link the test program" (./configure +
    make can be many minutes; agent must wait for the artifact)
    "install package X via apt/yum/pip from the network then exercise it"
    "start service S, wait for it to bind, then run an integration check"
    "download a multi-GB dataset / model then run preprocessing on it"
  Negative examples (set false):
    "fix off-by-one in foo.py line 42" (no waiting at all)
    "run the existing pytest suite" (test runtime is bounded; not a polled
    readiness wait)
    "create greet.py and verify with python3 greet.py" (sub-second)
    "edit three config files atomically" (no async readiness)
  Why this matters: when true, the orchestrator advises the executing model
  to spawn slow ops in the background (nohup ... &) and poll a log/file for
  readiness, rather than burning the prompt budget on consecutive \`sleep 60\`
  calls. Reason about whether the task contains a step that the agent
  literally has to wait on the wall-clock for, not whether the task name
  contains a particular keyword.

Return STRICT JSON ONLY (no prose, no fences):
{"shape":"atomic|compositional|monolithic-complex|hard","confidence":0.0-1.0,"reasoning":"short why","requiresLongWait":true|false}
`;

const buildClassifierPrompt = (task: string, repoContext: string, cwd: string): string =>
  [
    `Workspace cwd: ${cwd}`,
    "",
    "Task:",
    task.trim(),
    "",
    "Repo context (truncated):",
    repoContext.slice(0, 6000),
  ].join("\n");

const SOURCE_FILE_RE = /\b[A-Za-z0-9_./-]+\.(?:py|js|ts|tsx|jsx|go|rs|c|h|cc|cpp|hpp|sh|rb|java|kt|swift|md|json|yaml|yml|toml)\b/g;

const countSourceFilesMentioned = (repoContext: string): number => {
  const seen = new Set<string>();
  for (const match of repoContext.match(SOURCE_FILE_RE) ?? []) {
    seen.add(match);
  }
  return seen.size;
};

const heuristicAtomicShortCircuit = (task: string, repoContext: string): boolean => {
  if (task.trim().length >= 200) return false;
  // Only short-circuit when the repo context references at most one source file
  // (typical of an empty workspace where the agent will create a single file).
  return countSourceFilesMentioned(repoContext) <= 1;
};

const normalizeShape = (raw: unknown): TaskShape => {
  if (typeof raw !== "string") return "atomic";
  const lower = raw.toLowerCase().trim();
  if (lower === "atomic") return "atomic";
  if (lower === "compositional") return "compositional";
  if (lower === "monolithic-complex" || lower === "monolithic_complex" || lower === "monolithic") {
    return "monolithic-complex";
  }
  if (lower === "hard" || lower === "ambiguous") return "hard";
  return "atomic";
};

const clampConfidence = (value: unknown): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0.5;
  return Math.max(0, Math.min(1, value));
};

export const classifyTaskShape = async (input: {
  router: LlmRouter;
  task: string;
  repoContext: string;
  cwd: string;
}): Promise<TaskShapeDecision> => {
  if (heuristicAtomicShortCircuit(input.task, input.repoContext)) {
    return {
      shape: "atomic",
      mode: MODE_FOR_SHAPE.atomic,
      confidence: 0.85,
      reasoning:
        "heuristic: instruction <200 chars and repo context mentions <=1 source file -> atomic",
      requiresLongWait: false,
      tokens: { in: 0, out: 0 },
    };
  }

  if (!input.router.masterAvailable) {
    return SAFE_FALLBACK;
  }

  let raw: string;
  let metricIn = 0;
  let metricOut = 0;
  // Capture token usage by intercepting the next LlmCallMetric. The router
  // already records via telemetry; we read the same numbers off the response
  // by approximating from the input/output text length when usage is not
  // exposed on chatText. parseJsonObject handles malformed output.
  try {
    // classifier runs on the cheap local role — cost split.
    raw = await input.router.chatText({
      role: "local",
      json: true,
      maxTokens: 256,
      temperature: 0.0,
      purpose: "task-shape-classifier",
      messages: [
        { role: "system", content: SYSTEM_PROMPT_CLASSIFIER },
        { role: "user", content: buildClassifierPrompt(input.task, input.repoContext, input.cwd) },
      ],
    });
    // chatText itself does not surface tokens, but metrics are recorded via
    // RunTelemetry / LlmCallMetric (see createLlmRouter). We approximate here
    // for the per-decision artifact; the canonical numbers live in metrics.
    metricIn = Math.ceil(
      (SYSTEM_PROMPT_CLASSIFIER.length + buildClassifierPrompt(input.task, input.repoContext, input.cwd).length) / 4,
    );
    metricOut = Math.ceil(raw.length / 4);
  } catch (error) {
    return {
      ...SAFE_FALLBACK,
      reasoning: `classifier call threw: ${error instanceof Error ? error.message : String(error)}`,
    };
  }

  const parsed = parseJsonObject<{
    shape?: unknown;
    confidence?: unknown;
    reasoning?: unknown;
    requiresLongWait?: unknown;
  }>(raw, {
    shape: "atomic",
    confidence: 0.3,
    reasoning: "classifier returned unparseable JSON",
    requiresLongWait: false,
  });

  const shape = normalizeShape(parsed.shape);
  const confidence = clampConfidence(parsed.confidence);
  const reasoning =
    typeof parsed.reasoning === "string" && parsed.reasoning.trim().length > 0
      ? parsed.reasoning.trim()
      : "no reasoning provided";
  // Strict boolean coerce — only `true` (the literal JSON boolean) flips the
  // flag. Avoids accidentally enabling the hint when the model emits "true"
  // strings or 1/0 ints.
  const requiresLongWait = parsed.requiresLongWait === true;

  return {
    shape,
    mode: MODE_FOR_SHAPE[shape],
    confidence,
    reasoning,
    requiresLongWait,
    tokens: { in: metricIn, out: metricOut },
  };
};
