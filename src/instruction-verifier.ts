/**
 * LLM-driven post-submit verifier built from the task's instruction text.
 *
 * BAG runs sealed verifiers AFTER it exits the container, but the instruction
 * usually mentions an observable contract: a curl that should return a body,
 * a CLI that must print a value, a `pytest -rA` invocation, a file the agent
 * must produce. We extract those contracts at submit-time and run them BEFORE
 * accepting the agent's `BAG_TASK_COMPLETE` sentinel. A failing probe routes
 * the result back into the Best-of-N retry loop with the verifier-signature
 * library + failure-cluster matcher injecting actionable hints.
 *
 * Extraction pipeline:
 *   1. Primary — LLM (cheap local role) reads the instruction and emits a
 *      JSON list of {cmd, expect?, rationale}. The model understands ANY
 *      verification command family (pytest, cargo test, npm test, curl …)
 *      without a hardcoded keyword list.
 *   2. Fallback — when the LLM call fails or returns nothing usable, a
 *      conservative regex scan with a safe-verb whitelist. The whitelist
 *      exists purely to AVOID running destructive commands (rm/dd/sudo) by
 *      accident; it is not the source-of-truth for what counts as a probe.
 *   3. Hardening — every curl probe gains `-fsS` so HTTP 4xx/5xx → non-zero
 *      exit (otherwise the verifier passes a 404 page as "OK" because the
 *      transport succeeded). Idempotent: if `-f` / `--fail` / `-fsS` is
 *      already present we skip the rewrite.
 *
 * Probe execution:
 *   - Each probe runs in-container via the existing bash tool.
 *   - Pass condition is exit==0 AND (no `expect` || stdout contains `expect`).
 *   - First failure aborts the chain and the failing section is surfaced into
 *     the retry feedback so the verifier-signature-library can match it.
 */

import { executeBashTool, type AcpTerminalClient } from "./autonomous-tools";
import type { PostSubmitVerifier, PostSubmitVerifierResult } from "./autonomous-coding-turn";
import type { LlmRouter } from "./llm";
import { parseJsonObject } from "./llm";
import { DEFAULT_PATH_PROFILE, type PathProfile } from "./types";
import { loadHarnessGates, type HarnessGates } from "./harness-gates";

const MAX_PROBES = 5;
const PROBE_TIMEOUT_SEC = 30;
const LLM_INSTRUCTION_BUDGET_CHARS = 4000;

export type ProbeCommand = {
  /** Bash command to execute. */
  cmd: string;
  /** Optional substring/regex required in the probe's stdout for success. */
  expect?: string;
  /** Free-form note on why the probe was extracted (for trace/debugging). */
  rationale: string;
  /** Source of the probe — useful for telemetry attribution. */
  source: "llm" | "regex";
};

const PROBE_EXTRACTOR_SYSTEM_PROMPT = [
  "You audit a coding task instruction and identify the OBSERVABLE checks the user",
  "expects to pass. Output strictly JSON of the form:",
  '  {"probes": [{"cmd": "...", "expect": "...", "rationale": "..."}]}',
  "",
  "ABSOLUTE RULES — IF YOU VIOLATE THESE, THE PROBE WILL POISON THE WORKSPACE:",
  "  PROBES MUST NEVER CREATE FILES IN THE WORKSPACE. No gcc / g++ / cc / make /",
  "  cargo build / npm install / pip install / apt-get / curl -o / mv / cp / rm /",
  "  mkdir / sed -i / tee / shell redirection (>, >>) anywhere in cmd. These are",
  "  the agent's job. Probes ONLY observe.",
  "",
  "  When the instruction lists ALTERNATIVE invocation forms (e.g. \"I can run",
  "  python3 X.py N OR gcc X.c -o Y && ./Y N\"), pick ONLY the source-only form",
  "  (python3 X.py N). Skip the compile-and-run form entirely. The user said",
  "  \"OR\", so testing one path is enough. Compiling artifacts may violate the",
  "  task's final-state invariant (\"only main.py.c may exist\").",
  "",
  "  If the only verification path requires compilation/installation, return",
  "  {\"probes\": []}. A non-firing verifier is strictly safer than one that",
  "  poisons the deliverable directory. The post-submit harbor verifier will",
  "  judge the workspace independently.",
  "",
  "Rules:",
  "  - Probes are STRICTLY NON-MUTATING. They observe end-state only. NEVER",
  "    emit commands that compile, build, install, write files, modify the",
  "    workspace, or have side effects on the filesystem / network / processes.",
  "    Forbidden verbs: gcc, g++, cc, make, cargo build, npm install, pip install,",
  "    apt-get, curl with -o/-O writing files, mv, cp, rm, mkdir, sed -i, tee,",
  "    redirection (>, >>) to a workspace file, service start/stop, kill, echo",
  '    "..." > file. The agent is responsible for those; probes only check what',
  "    they produced.",
  "  - Probes test DELIVERABLES (artifacts the instruction explicitly names",
  "    as outputs the user wants to keep), not TRANSIENT VERIFICATION",
  "    ARTIFACTS (binaries the agent compiles to test correctness but is",
  "    not asked to retain). If the instruction says 'create file X.py' and",
  "    'run with python3 X.py' AND ALSO 'compile with gcc X.c → ./prog' for",
  "    verification, the deliverable is X (the source). The compiled `./prog`",
  "    is a transient — the agent may delete it before submission and a probe",
  "    that runs ./prog will FAIL even though the task is correct. Probe the",
  "    invocation form using the deliverable directly (`python3 X.py …`) and",
  "    skip the transient form unless the binary is explicitly named as a",
  "    file to deliver. When unsure whether something is a deliverable, OMIT",
  "    the probe — a verifier that fails because the agent cleaned up too",
  "    aggressively is worse than no verifier.",
  "  - Allowed verbs are read-only or pure observers: curl GET (without -o), cat,",
  "    ls, stat, file, head, tail, wc, diff, grep, test, [, sha256sum, md5sum,",
  "    pytest (read-only when tests don't mutate), python/python3 invoking a",
  "    script the agent built, node, cargo test (in read-only project tree),",
  "    npm test, go test, make test/check, which, command -v.",
  "  - Use the EXACT command string the user wrote when one is present (curl",
  "    URLs, pytest invocations, file checks). Do not paraphrase, do not insert",
  "    extra build steps the user did not write.",
  "  - DO NOT INVENT structural assertions about internal layout. If the user",
  "    did not literally name a path / file / config option in the instruction,",
  "    do NOT probe it. Examples of forbidden inferences: 'I'll check that the",
  "    git repo is bare by looking for HEAD without a .git/ subdir', 'I'll grep",
  "    /etc/passwd for the user', 'I'll inspect systemd unit files'. The agent",
  "    chose those internal details; verifying them is not the user's contract.",
  "  - When the instruction contains no observable verification at all (just",
  "    'fix bug X', 'refactor Y'), return {\"probes\": []}. Do not synthesize.",
  "    A non-firing verifier is strictly safer than a misleading one.",
  "  - For HTTP probes, prefer `curl -fsS` so HTTP errors exit non-zero. If the",
  '    user said `curl http://x/y`, emit `cmd: "curl -fsS http://x/y"`.',
  "  - When the task says output X, use `expect` to assert that X appears in",
  '    stdout. Example: instruction "must return hello world" → expect: "hello world".',
  "  - Skip probes you cannot run from a generic shell (e.g. interactive REPLs,",
  "    GUI checks, anything requiring credentials we don't have).",
  "  - If the instruction has NO observable check at all, return {\"probes\": []}.",
  "  - Maximum 5 probes. Order matters — the first failure short-circuits.",
  "  - Output ONLY the JSON object. No prose, no fences.",
].join("\n");

const SAFE_VERB_REGEX_FALLBACK = [
  "curl ",
  "wget ",
  "cat ",
  "ls ",
  "stat ",
  "file ",
  "head ",
  "tail ",
  "wc ",
  "diff ",
  "grep ",
  "test ",
  "[ ",
  "python ",
  "python3 ",
  "node ",
  "pytest ",
  "pytest\n",
  "npm test",
  "npm run",
  "yarn test",
  "bun test",
  "cargo test",
  "go test",
  "make test",
  "make check",
  "rake test",
  "phpunit",
  "mvn test",
  "gradle test",
  "sha256sum ",
  "md5sum ",
];

const isSafeFallbackPrefix = (line: string): boolean =>
  SAFE_VERB_REGEX_FALLBACK.some((p) => line.startsWith(p));

/** Trims, collapses double-spaces, drops trailing punctuation. */
const normalizeProbe = (raw: string): string => raw.trim().replace(/[.;,\s]+$/, "").trim();

/** Idempotent: rewrites `curl …` to `curl -fsS …` so HTTP errors propagate. */
const hardenCurl = (cmd: string): string => {
  const trimmed = cmd.trimStart();
  if (!/^curl(\s|$)/.test(trimmed)) return cmd;
  // Already hardened? -f / --fail / -fsS / -f anywhere as a flag → leave it.
  if (/(?:^|\s)-[A-Za-z]*f[A-Za-z]*(?:\s|$)/.test(trimmed)) return cmd;
  if (/(?:^|\s)--fail(?:\s|$)/.test(trimmed)) return cmd;
  // Otherwise prepend -fsS right after the curl token.
  return trimmed.replace(/^curl(\s+|$)/, "curl -fsS$1");
};

/**
 * Deterministic regex-based extractor — used ONLY as fallback when the LLM
 * extractor is unavailable or returns nothing useful. The safe-verb list is a
 * SAFETY filter (don't run `rm` by accident), not the canonical truth about
 * what counts as a probe.
 */
export const extractProbeCommandsRegex = (instruction: string): ProbeCommand[] => {
  const probes: ProbeCommand[] = [];
  const seen = new Set<string>();
  const push = (cmd: string, rationale: string): void => {
    const norm = normalizeProbe(cmd);
    if (norm.length < 4 || norm.length > 300) return;
    if (norm.includes("`")) return;
    if (!isSafeFallbackPrefix(norm)) return;
    if (seen.has(norm)) return;
    seen.add(norm);
    probes.push({ cmd: norm, rationale, source: "regex" });
  };

  // 1. Single-line backtick spans. Reject if pre-char is also a backtick
  // (= we landed inside a ```language…``` fence whose label is not bash/sh).
  for (const match of instruction.matchAll(/`([^`\n]{4,300})`/g)) {
    const raw = match[1];
    if (raw == null) continue;
    const start = match.index ?? 0;
    if (start > 0 && instruction.charAt(start - 1) === "`") continue;
    push(raw, "single-backtick span");
  }
  // 2. Code-fence blocks ```bash``` / ```sh``` / ``` ``` — line-by-line.
  for (const block of instruction.matchAll(/```(?:bash|sh)?\n([\s\S]*?)```/g)) {
    const body = block[1] ?? "";
    for (const line of body.split("\n")) {
      const trimmed = line.trim();
      if (trimmed.length === 0 || trimmed.startsWith("#")) continue;
      push(trimmed, "code-fence line");
    }
  }
  // 3. Inline curl mentions. Use \S+ (not [^\s.,;]+) so URLs with dots survive.
  for (const match of instruction.matchAll(/\b(curl\s+\S+)/g)) {
    const raw = match[1];
    if (raw != null) push(raw, "inline curl mention");
  }
  return probes.slice(0, MAX_PROBES);
};

const parseLlmProbeResponse = (raw: string): ProbeCommand[] => {
  type LlmShape = { probes?: Array<{ cmd?: unknown; expect?: unknown; rationale?: unknown }> };
  const parsed = parseJsonObject<LlmShape>(raw, { probes: [] });
  const list = Array.isArray(parsed.probes) ? parsed.probes : [];
  const probes: ProbeCommand[] = [];
  const seen = new Set<string>();
  for (const entry of list) {
    if (entry == null || typeof entry !== "object") continue;
    const cmdRaw = typeof entry.cmd === "string" ? entry.cmd : null;
    if (cmdRaw == null) continue;
    const norm = normalizeProbe(cmdRaw);
    if (norm.length === 0 || norm.length > 300) continue;
    if (seen.has(norm)) continue;
    seen.add(norm);
    const expect =
      typeof entry.expect === "string" && entry.expect.trim().length > 0
        ? entry.expect.trim().slice(0, 200)
        : undefined;
    const rationale =
      typeof entry.rationale === "string" && entry.rationale.trim().length > 0
        ? entry.rationale.trim().slice(0, 200)
        : "llm extractor";
    probes.push({
      cmd: norm,
      ...(expect === undefined ? {} : { expect }),
      rationale,
      source: "llm",
    });
    if (probes.length >= MAX_PROBES) break;
  }
  return probes;
};

/**
 * Primary extractor — calls the cheap local role to produce probes. Falls
 * back to the regex extractor on any error or empty result. All probes pass
 * through `hardenCurl` before being returned.
 */
export const extractProbeCommands = async (input: {
  router: LlmRouter;
  instruction: string;
}): Promise<ProbeCommand[]> => {
  const truncated =
    input.instruction.length > LLM_INSTRUCTION_BUDGET_CHARS
      ? input.instruction.slice(0, LLM_INSTRUCTION_BUDGET_CHARS)
      : input.instruction;

  let llmProbes: ProbeCommand[] = [];
  try {
    const raw = await input.router.chatText({
      role: "local",
      json: true,
      maxTokens: 600,
      purpose: "probe-extractor",
      messages: [
        { role: "system", content: PROBE_EXTRACTOR_SYSTEM_PROMPT },
        { role: "user", content: truncated },
      ],
    });
    llmProbes = parseLlmProbeResponse(raw);
  } catch {
    llmProbes = [];
  }

  // Principled choice: if the LLM extractor returned nothing (or threw), we
  // emit ZERO probes rather than falling back to a hardcoded keyword regex.
  // Empty probes -> verifyAfterSubmit returns undefined -> the principled
  // pre-submit self-check gate is the single source of truth. The regex
  // fallback (`extractProbeCommandsRegex` + `SAFE_VERB_REGEX_FALLBACK`) is
  // retained in this file as a debugging utility but is NOT in the live path.
  return llmProbes.map((p) => ({ ...p, cmd: hardenCurl(p.cmd) }));
};

const stdoutMatchesExpect = (stdout: string, expect: string | undefined): boolean => {
  if (expect === undefined) return true;
  if (stdout.includes(expect)) return true;
  // Try as a regex too — best effort, ignore invalid patterns.
  try {
    return new RegExp(expect, "i").test(stdout);
  } catch {
    return false;
  }
};

/**
 * Snapshot the file list of `cwd` so we can detect (and erase) any new files
 * the probes leave behind. The probe extractor system prompt asks the LLM to
 * emit non-mutating commands, but the LLM is non-deterministic — sometimes it
 * still emits a `gcc … -o BIN && ./BIN` form. Without restoration, the BIN
 * lingers in the deliverable directory and the harbor verifier (which checks
 * end-state) rejects the submission. The snapshot/restore pair is a generic
 * defence: whatever the probe writes, we wipe — the workspace looks the same
 * after probes as it did before, regardless of LLM compliance.
 */
const SNAPSHOT_PATH = "/tmp/.bag-probe-snapshot.txt";

/**
 * Render `-not -path` flags for `find` from a PathProfile's metadataDirs.
 * Each dir D produces `-not -path '*<slash>D<slash>*'` (slashes elided here
 * to avoid closing this JSDoc block comment). Exposed for tests so we can
 * assert the snapshot command picks up overrides without spinning bash.
 */
export const renderFindMetadataExcludes = (metadataDirs: ReadonlyArray<string>): string =>
  metadataDirs
    .map((d) => {
      const trimmed = d.replace(/^\/+|\/+$/g, "");
      // Single-quote escape: ' → '\''; safe for the embedded shell literal.
      const escaped = trimmed.replace(/'/g, "'\\''");
      return `-not -path '*/${escaped}/*'`;
    })
    .join(" ");

/**
 * Build the snapshot bash command from a PathProfile. Pure-string builder so
 * tests can assert the exclusion globs propagate from config to the actual
 * `find` invocation without executing bash.
 */
export const buildSnapshotCommand = (input: {
  cwd: string;
  pathProfile: PathProfile;
  outputPath: string;
}): string => {
  const excludes = renderFindMetadataExcludes(input.pathProfile.metadataDirs);
  return `find ${JSON.stringify(input.cwd)} -type f ${excludes} 2>/dev/null | sort > ${input.outputPath} || true`;
};

const captureWorkspaceSnapshot = async (input: {
  client: AcpTerminalClient;
  sessionId: string;
  cwd: string;
  pathProfile: PathProfile;
}): Promise<void> => {
  // 2>/dev/null swallows permission denials on /proc-style mounts. We list
  // every regular file under cwd and store the inventory in /tmp.
  await executeBashTool({
    client: input.client,
    sessionId: input.sessionId,
    cwd: input.cwd,
    command: buildSnapshotCommand({
      cwd: input.cwd,
      pathProfile: input.pathProfile,
      outputPath: SNAPSHOT_PATH,
    }),
    timeoutSec: 30,
  });
};

const restoreWorkspaceFromSnapshot = async (input: {
  client: AcpTerminalClient;
  sessionId: string;
  cwd: string;
  pathProfile: PathProfile;
}): Promise<{ removedFiles: number }> => {
  const excludes = renderFindMetadataExcludes(input.pathProfile.metadataDirs);
  // Diff the post-probe inventory against the snapshot, delete additions.
  const r = await executeBashTool({
    client: input.client,
    sessionId: input.sessionId,
    cwd: input.cwd,
    command: [
      `find ${JSON.stringify(input.cwd)} -type f ${excludes} 2>/dev/null | sort > /tmp/.bag-probe-after.txt`,
      `comm -23 /tmp/.bag-probe-after.txt ${SNAPSHOT_PATH} > /tmp/.bag-probe-new.txt`,
      `count=$(wc -l < /tmp/.bag-probe-new.txt | tr -d ' ')`,
      `xargs -r -d '\\n' rm -f < /tmp/.bag-probe-new.txt`,
      `echo "removed=${"$count"}"`,
    ].join("; "),
    timeoutSec: 30,
  });
  const match = /removed=(\d+)/.exec(r.truncatedOutput);
  const removedFiles = match ? Number(match[1]) : 0;
  return { removedFiles };
};

/**
 * Build a `PostSubmitVerifier` from instruction text. Returns undefined when
 * no probes are extractable — caller falls back to default Best-of-N behavior.
 *
 * Probe execution is wrapped in a workspace snapshot/restore pair so that any
 * file a probe creates (compiled binary, temp output, generated config) is
 * deleted after the probe chain finishes. The agent's deliverable state is
 * preserved end-to-end even when the probe extractor mis-emits a mutating
 * command. This is a defensive fail-safe for non-deterministic LLM probe
 * output.
 */
export const buildVerifierFromInstruction = async (input: {
  router: LlmRouter;
  instruction: string;
  /**
   * Optional path-convention overrides. When omitted, falls back to the
   * Linux defaults baked into `BagConfigSchema.pathProfile` so existing
   * call sites that have not yet been threaded with config remain byte-
   * equivalent. Override to redirect snapshot exclusion globs (Docker
   * images that use a non-`.git/` metadata layout, etc.).
   */
  pathProfile?: PathProfile;
  /**
   * Optional harness-gate snapshot. When omitted, gates are read from
   * the process env via `loadHarnessGates()`. The ablation harness uses
   * env-var presets (BAG_MODE_BARE_ENV / BAG_MODE_MINIMAL_ENV) to flip
   * gates without touching code. Tests inject this directly to avoid
   * env mutation.
   */
  gates?: HarnessGates;
}): Promise<PostSubmitVerifier | undefined> => {
  const gates = input.gates ?? loadHarnessGates();
  // Gate: probe extractor disabled → no verifier ever fires. Caller will
  // skip Best-of-N retry and self-check looks at the bash trace alone.
  if (!gates.probeExtractor) return undefined;
  const probes = await extractProbeCommands(input);
  if (probes.length === 0) return undefined;
  const pathProfile = input.pathProfile ?? DEFAULT_PATH_PROFILE;
  const snapshotEnabled = gates.snapshotRestore;

  return async (callbackInput: {
    client: AcpTerminalClient;
    sessionId: string;
    cwd: string;
  }): Promise<PostSubmitVerifierResult> => {
    const sections: string[] = [];
    let firstFailureExit: number | null = null;
    let allPassed = true;
    let lastExit: number | null = 0;

    if (snapshotEnabled) {
      await captureWorkspaceSnapshot({ ...callbackInput, pathProfile });
    }

    try {
      for (const probe of probes) {
        try {
          const r = await executeBashTool({
            client: callbackInput.client,
            sessionId: callbackInput.sessionId,
            cwd: callbackInput.cwd,
            command: probe.cmd,
            timeoutSec: PROBE_TIMEOUT_SEC,
          });
          const exit = r.exitCode ?? null;
          const stdoutMatch = stdoutMatchesExpect(r.truncatedOutput, probe.expect);
          const probePassed = exit === 0 && stdoutMatch;
          const expectNote =
            probe.expect === undefined
              ? ""
              : `\nexpect: ${probe.expect} (matched: ${stdoutMatch})`;
          sections.push(
            `$ ${probe.cmd}\nexit=${exit ?? "null"}${expectNote}\nrationale: ${probe.rationale}\n${r.truncatedOutput.slice(0, 1500)}`,
          );
          lastExit = exit;
          if (!probePassed) {
            allPassed = false;
            if (firstFailureExit === null) firstFailureExit = exit ?? -1;
            break;
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          sections.push(`$ ${probe.cmd}\nexec error: ${message}`);
          allPassed = false;
          if (firstFailureExit === null) firstFailureExit = -1;
          break;
        }
      }
    } finally {
      if (snapshotEnabled) {
        try {
          const cleanup = await restoreWorkspaceFromSnapshot({ ...callbackInput, pathProfile });
          if (cleanup.removedFiles > 0) {
            sections.push(
              `[probe-cleanup] removed ${cleanup.removedFiles} file(s) created by probes; workspace restored to pre-probe state.`,
            );
          }
        } catch {
          // Cleanup is best-effort — never let it crash the verifier callback.
        }
      }
    }

    return {
      passed: allPassed,
      output: sections.join("\n\n"),
      exitCode: allPassed ? 0 : firstFailureExit ?? lastExit ?? -1,
    };
  };
};
