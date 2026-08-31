/**
 * Autonomous-mode tool surface.
 *
 * Inspired by mini-swe-agent's "bash-only" architecture (see docs/plans/
 * autonomous-coding-mode.md). The LLM is exposed a single bash tool. File
 * reads/writes/edits all flow through `cat`, `sed`, here-docs, etc. — the
 * same primitives the model already has high training signal on.
 *
 * Submission: when the agent emits `echo BAG_TASK_COMPLETE` (or
 * `printf 'BAG_TASK_COMPLETE\n'`) as the FIRST non-blank line of stdout, the
 * loop terminates with stopReason='submitted'.
 */

export const SUBMIT_SENTINEL = "BAG_TASK_COMPLETE";

export const BASH_TOOL_NAME = "bash";

export const BASH_TOOL_DEFINITION = {
  type: "function" as const,
  function: {
    name: BASH_TOOL_NAME,
    description: [
      "Execute a single bash command in the workspace and return its stdout/stderr/exit code.",
      "Each call runs in a NEW subshell — working directory and environment do NOT persist across calls.",
      "If you need a particular cwd, chain it: `cd /path && command`.",
      "If you need an env var, prefix it: `MY_VAR=x command`.",
      "Use here-docs (`cat <<'EOF' > file ... EOF`), `sed -i`, `printf >>` etc. for file edits.",
      `When the task is complete, run \`echo ${SUBMIT_SENTINEL}\` as the first line of a bash command (no other commands on the same line).`,
      "Output is truncated to ~10000 characters with a head+tail elision; use `head`/`tail`/`sed -n 'M,Np'`/redirection to a file when output is long.",
    ].join("\n"),
    parameters: {
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "The bash command line to execute. Single line; chain with && or use a here-doc.",
        },
        timeout_sec: {
          type: "number",
          description: "Optional timeout in seconds; defaults to 60. Cap for the call.",
        },
      },
      required: ["command"],
      additionalProperties: false,
    },
  },
} as const;

export const CODE_SEARCH_TOOL_NAME = "code_search";

/**
 * `code_search` exposes a generic late-interaction / semantic codebase search
 * to the master model. The first impl wraps ColGrep (LightOn's late-
 * interaction code search), but the contract is intentionally backend-
 * agnostic — see `src/codebase-index/colgrep-bridge.ts` for the
 * `CodebaseSearchBackend` interface.
 *
 * Use-case split with `bash`/`rg`:
 *   - `bash` + `rg`:    EXACT tokens, identifiers, error strings, paths.
 *   - `code_search`:    CONCEPTUAL queries ("where is the auth middleware?",
 *                       "how is rate limiting handled?").
 *
 * Returns ranked file/line/symbol hits and short snippets — the model still
 * needs to read full bodies via `bash` once it has localized.
 */
export const CODE_SEARCH_TOOL_DEFINITION = {
  type: "function" as const,
  function: {
    name: CODE_SEARCH_TOOL_NAME,
    description: [
      "Semantic codebase search via late-interaction retrieval. Use for CONCEPTUAL queries ('where is the retry/backoff logic', 'how is the DAG cached'). For exact tokens/identifiers/error messages, use `bash` with `rg`. Returns ranked file/line/symbol hits, no large bodies.",
      "Args: query (required), top_k (default 10, max 100), mode ('semantic'|'hybrid', default 'hybrid'), path_filter (optional glob), language_filter (optional language tag).",
      "When the backend is unavailable (binary missing, container without internet to fetch the model), the tool returns a structured error message — fall back to `bash` + `rg`.",
    ].join("\n"),
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Conceptual search query in natural language. Avoid quoting; let the retriever embed it.",
        },
        top_k: {
          type: "number",
          description: "Maximum number of hits to return. Default 10, capped at 100.",
        },
        mode: {
          type: "string",
          enum: ["semantic", "hybrid"],
          description: "'semantic' = pure late-interaction; 'hybrid' = late-interaction + lexical re-rank (default).",
        },
        path_filter: {
          type: "string",
          description: "Optional path glob (e.g. 'src/**/*.ts') to restrict the search.",
        },
        language_filter: {
          type: "string",
          description: "Optional language filter ('typescript', 'python', 'rust', …).",
        },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
} as const;

export const VIEW_IMAGE_TOOL_NAME = "view_image";

/**
 * `view_image` lets the master model NATIVELY perceive an image file from the
 * workspace. The tool reads the file, base64-encodes it, and queues it as a
 * multimodal `image_url` content block in the NEXT user message sent to the
 * model. The tool's textual reply confirms the load and surfaces the file's
 * size — the actual visual content arrives on the subsequent turn so the
 * model can reason about it directly instead of guessing pixels via PIL.
 *
 * This is the same mainstream capability mainstream coding agents (Claude
 * Code, Cursor, Aider, …) ship: native vision input via OpenAI-compat
 * `image_url` blocks. BAG previously stripped images at the ACP boundary;
 * this tool restores the channel.
 */
export const VIEW_IMAGE_TOOL_DEFINITION = {
  type: "function" as const,
  function: {
    name: VIEW_IMAGE_TOOL_NAME,
    description: [
      "Load an image file from the workspace into your visual context so you can SEE it directly on the next turn.",
      "Use this for any task whose verification depends on perceiving an image — a chess board screenshot, a UI mockup, a graph, a doc page capture, a generated diagram, etc.",
      "Supported formats: PNG, JPEG, GIF, WEBP. The file is base64-encoded and attached to the next user message as a multimodal content block.",
      "After calling, the next message you receive will literally contain the image; analyze it visually rather than reading pixels via PIL.",
      "Returns text confirmation of the load (path, mime, byte count). The visual data arrives on the next assistant turn.",
    ].join("\n"),
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Absolute or workspace-relative path to the image file (e.g. /app/chess_board.png).",
        },
      },
      required: ["path"],
      additionalProperties: false,
    },
  },
} as const;

export type AutonomousToolResult = {
  output: string;
  truncatedOutput: string;
  exitCode: number | null;
  signal: string | null;
  durationMs: number;
  truncated: boolean;
  submitted: boolean;
};

const ELISION_THRESHOLD = 10_000;
const ELISION_HEAD_TAIL = 5_000;

const elideOutput = (output: string): { display: string; truncated: boolean } => {
  if (output.length <= ELISION_THRESHOLD) {
    return { display: output, truncated: false };
  }
  const head = output.slice(0, ELISION_HEAD_TAIL);
  const tail = output.slice(-ELISION_HEAD_TAIL);
  const elided = output.length - ELISION_HEAD_TAIL * 2;
  const display = `${head}\n...\n[${elided} bytes elided — use head/tail/sed/redirect to file]\n...\n${tail}`;
  return { display, truncated: true };
};

const detectSentinel = (output: string, sentinel: string = SUBMIT_SENTINEL): boolean => {
  for (const rawLine of output.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line.length === 0) continue;
    return line === sentinel;
  }
  return false;
};

export type AcpTerminalClient = {
  createTerminal: (params: {
    sessionId: string;
    command: string;
    args: string[];
    cwd?: string | null;
    env?: { name: string; value: string }[];
    outputByteLimit?: number;
  }) => Promise<{ terminalId: string }>;
  waitForTerminalExit: (params: { sessionId: string; terminalId: string }) => Promise<{
    exitCode?: number | null;
    signal?: string | null;
  }>;
  terminalOutput: (params: { sessionId: string; terminalId: string }) => Promise<{
    output: string;
    truncated: boolean;
    exitStatus?: { exitCode?: number | null; signal?: string | null } | null;
  }>;
  releaseTerminal: (params: { sessionId: string; terminalId: string }) => Promise<unknown>;
};

/**
 * Result from `executeViewImageTool`. When `ok=true`, the agent's next user
 * message should embed the image as an OpenAI-vision-compat content block:
 *   { type: "image_url", image_url: { url: "data:<mimeType>;base64,<base64>" } }
 * The textual `observation` is what shows up as the tool result for the
 * model on the current turn.
 */
export type ViewImageResult = {
  ok: boolean;
  path: string;
  mimeType: string;
  base64?: string;
  bytes: number;
  observation: string;
};

const VIEW_IMAGE_BYTE_CAP = 5 * 1024 * 1024;

export const executeViewImageTool = async (input: {
  sessionId: string;
  cwd: string;
  path: string;
  client: AcpTerminalClient;
}): Promise<ViewImageResult> => {
  const safePath = JSON.stringify(input.path);
  // We read the file from inside the container via bash so we get the real
  // workspace state. The output frames base64 between sentinels so we can
  // strip header lines reliably.
  // MIME detection chain (containers may lack `file`, so we cascade):
  //   1. `file -b --mime-type` if available (canonical)
  //   2. magic-byte sniff via `head -c 12 | od` (works on every POSIX shell)
  //   3. extension fallback (.png/.jpg/.jpeg/.gif/.webp)
  // The last stage is generic — any future image extension can extend the
  // case statement without touching agent code.
  const cmd = [
    `f=${safePath}`,
    `if [ ! -f "$f" ]; then echo "ERROR: not a regular file"; exit 2; fi`,
    `size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)`,
    `if [ "$size" -gt ${VIEW_IMAGE_BYTE_CAP} ]; then echo "ERROR: file too large ($size bytes); cap is ${VIEW_IMAGE_BYTE_CAP} bytes"; exit 3; fi`,
    `mt=$(file -b --mime-type "$f" 2>/dev/null || true)`,
    `if [ -z "$mt" ] || [ "$mt" = "application/octet-stream" ] || [ "$mt" = "regular file" ]; then`,
    `  hex=$(head -c 12 "$f" 2>/dev/null | od -An -tx1 -v 2>/dev/null | tr -d ' \\n')`,
    `  case "$hex" in`,
    `    89504e470d0a1a0a*) mt="image/png" ;;`,
    `    ffd8ff*) mt="image/jpeg" ;;`,
    `    47494638376*|47494638396*) mt="image/gif" ;;`,
    `    52494646????57454250*) mt="image/webp" ;;`,
    `    424d*) mt="image/bmp" ;;`,
    `    49492a00*|4d4d002a*) mt="image/tiff" ;;`,
    `    *)`,
    `      case "$f" in`,
    `        *.png|*.PNG) mt="image/png" ;;`,
    `        *.jpg|*.jpeg|*.JPG|*.JPEG) mt="image/jpeg" ;;`,
    `        *.gif|*.GIF) mt="image/gif" ;;`,
    `        *.webp|*.WEBP) mt="image/webp" ;;`,
    `        *.bmp|*.BMP) mt="image/bmp" ;;`,
    `        *.tif|*.tiff|*.TIF|*.TIFF) mt="image/tiff" ;;`,
    `        *) mt="application/octet-stream" ;;`,
    `      esac`,
    `      ;;`,
    `  esac`,
    `fi`,
    `echo "MIME=$mt"`,
    `echo "BYTES=$size"`,
    `echo "B64START"`,
    `base64 -w0 "$f" 2>/dev/null || base64 "$f" | tr -d '\\n'`,
    `echo`,
    `echo "B64END"`,
  ].join("; ");

  const r = await executeBashTool({
    client: input.client,
    sessionId: input.sessionId,
    cwd: input.cwd,
    command: cmd,
    timeoutSec: 60,
    outputByteLimit: 8 * 1024 * 1024,
  });

  if (r.exitCode !== 0) {
    return {
      ok: false,
      path: input.path,
      mimeType: "",
      bytes: 0,
      observation: `view_image failed (exit ${r.exitCode ?? "null"}): ${r.output.slice(0, 300)}`,
    };
  }

  const out = r.output;
  const mimeMatch = /^MIME=(\S+)$/m.exec(out);
  const bytesMatch = /^BYTES=(\d+)$/m.exec(out);
  const b64Match = /B64START\n([\s\S]*?)\nB64END/.exec(out);
  if (b64Match == null) {
    return {
      ok: false,
      path: input.path,
      mimeType: "",
      bytes: 0,
      observation: `view_image: could not parse base64 envelope. raw head: ${out.slice(0, 300)}`,
    };
  }
  const mimeType = mimeMatch?.[1] ?? "application/octet-stream";
  const bytes = bytesMatch != null ? Number(bytesMatch[1]) : 0;
  const base64 = (b64Match[1] ?? "").replace(/\s/g, "");
  if (!mimeType.startsWith("image/")) {
    return {
      ok: false,
      path: input.path,
      mimeType,
      bytes,
      observation: `view_image: ${input.path} is ${mimeType}, not an image. Use bash to inspect non-image files.`,
    };
  }
  return {
    ok: true,
    path: input.path,
    mimeType,
    bytes,
    base64,
    observation: `Image loaded: path=${input.path}, mime=${mimeType}, bytes=${bytes}. The image is attached to your next turn — analyse it visually.`,
  };
};

export const executeBashTool = async (input: {
  sessionId: string;
  cwd: string;
  command: string;
  timeoutSec?: number;
  client: AcpTerminalClient;
  outputByteLimit?: number;
  submitSentinel?: string;
}): Promise<AutonomousToolResult> => {
  const startedMs = performance.now();
  const timeoutSec = Math.max(1, Math.min(900, input.timeoutSec ?? 60));
  const outputByteLimit = input.outputByteLimit ?? 256_000;
  const created = await input.client.createTerminal({
    sessionId: input.sessionId,
    command: "bash",
    args: ["-lc", `set -o pipefail; ${input.command}`],
    cwd: input.cwd,
    outputByteLimit,
  });
  const terminalId = created.terminalId;
  let exit: { exitCode?: number | null; signal?: string | null } = {};
  try {
    exit = await Promise.race([
      input.client.waitForTerminalExit({ sessionId: input.sessionId, terminalId }),
      new Promise<{ exitCode: null; signal: "SIGTERM" }>((resolveTimeout) =>
        setTimeout(() => resolveTimeout({ exitCode: null, signal: "SIGTERM" }), timeoutSec * 1_000),
      ),
    ]);
  } finally {
    // Always pull the buffered output, even if we timed out the wait.
  }
  const outputResp = await input.client.terminalOutput({ sessionId: input.sessionId, terminalId });
  await input.client.releaseTerminal({ sessionId: input.sessionId, terminalId });
  const fullOutput = outputResp.output ?? "";
  const elided = elideOutput(fullOutput);
  const submitted = detectSentinel(fullOutput, input.submitSentinel);
  const exitCode =
    exit.exitCode ?? outputResp.exitStatus?.exitCode ?? null;
  const signal =
    exit.signal != null
      ? String(exit.signal)
      : outputResp.exitStatus?.signal != null
        ? String(outputResp.exitStatus.signal)
        : null;
  return {
    output: fullOutput,
    truncatedOutput: elided.display,
    exitCode,
    signal,
    durationMs: Math.round(performance.now() - startedMs),
    truncated: elided.truncated || (outputResp.truncated ?? false),
    submitted,
  };
};

/**
 * Match raw bash output against the Codex-corpus-mined recovery playbooks
 * (`src/codex-trace-distilled.ts:RECOVERY_PLAYBOOKS`). Returns a one-line hint
 * if a known pattern matches — appended to the bash observation so the model
 * sees prior-art recovery alongside the raw error.
 *
 * Lazy-imported to keep the autonomous-tools module dep-light. Imports may
 * fail if the artifact is missing; in that case, we silently skip.
 */
type RecoveryPlaybookEntry = {
  triggerError: string;
  successfulRecovery: string;
  count?: number;
};

let CACHED_PLAYBOOKS: ReadonlyArray<RecoveryPlaybookEntry> | null = null;
let PLAYBOOKS_LOAD_ATTEMPTED = false;
const loadRecoveryPlaybooks = async (): Promise<ReadonlyArray<RecoveryPlaybookEntry>> => {
  if (PLAYBOOKS_LOAD_ATTEMPTED) return CACHED_PLAYBOOKS ?? [];
  PLAYBOOKS_LOAD_ATTEMPTED = true;
  try {
    const mod = (await import("./codex-trace-distilled")) as {
      RECOVERY_PLAYBOOKS?: ReadonlyArray<RecoveryPlaybookEntry>;
    };
    CACHED_PLAYBOOKS = mod.RECOVERY_PLAYBOOKS ?? [];
  } catch {
    CACHED_PLAYBOOKS = [];
  }
  return CACHED_PLAYBOOKS ?? [];
};

const matchRecoveryPlaybook = (
  output: string,
  playbooks: ReadonlyArray<RecoveryPlaybookEntry>,
): RecoveryPlaybookEntry | null => {
  if (output.length === 0) return null;
  for (const pb of playbooks) {
    try {
      const re = new RegExp(pb.triggerError, "im");
      if (re.test(output)) return pb;
    } catch {
      // invalid regex — skip
    }
  }
  return null;
};

/**
 * Result from `executeCodeSearchTool`. Mirrors the shape of `ViewImageResult`
 * — the `observation` string is what the master model sees as the tool's
 * reply on the current turn. `hits` is retained on the entry so trace
 * processors can post-hoc analyse retrieval quality without re-parsing the
 * observation.
 */
export type CodeSearchToolResult = {
  ok: boolean;
  observation: string;
  hits: ReadonlyArray<{
    file: string;
    lineRange: [number, number];
    symbol?: string;
    unitKind?: string;
    score: number;
    snippet?: string;
  }>;
  backendStatus: "available" | "unavailable" | "disabled" | "error";
  durationMs: number;
  errorMessage?: string;
};

export const renderBashObservation = (result: AutonomousToolResult): string => {
  const lines: string[] = [];
  lines.push(`exit_code: ${result.exitCode ?? "null"}${result.signal != null ? ` (signal=${result.signal})` : ""}`);
  lines.push(`duration_ms: ${result.durationMs}`);
  if (result.truncated) lines.push("truncated: true");
  if (result.submitted) lines.push("submitted: true");
  lines.push("output:");
  lines.push(result.truncatedOutput);
  // Synchronous-by-design wrapper. The async playbook load happens once and
  // is cached; subsequent calls hit the cache. We don't await here so the
  // first observation is hint-free; from the second tool call onward the
  // cache is warm and matches show up. Acceptable trade for keeping the
  // signature pure-synchronous.
  if (CACHED_PLAYBOOKS != null && (result.exitCode ?? 0) !== 0) {
    const match = matchRecoveryPlaybook(result.truncatedOutput, CACHED_PLAYBOOKS);
    if (match) {
      lines.push("");
      lines.push(
        `[BAG corpus playbook — past sessions recovered from this error pattern via]: ${match.successfulRecovery}`,
      );
    }
  } else if (!PLAYBOOKS_LOAD_ATTEMPTED) {
    // Fire-and-forget: warm the cache for next call.
    void loadRecoveryPlaybooks();
  }
  return lines.join("\n");
};

