/**
 * Scratch-hygiene auditor — pure-TS heuristics over a bash trace tail.
 *
 * Motivation: forensic survey (`docs/bag-successful-runs-deep-dive.md`)
 * found that 52.8 % of winning trials leak scratch into `/tmp/` (would fail a
 * clean-room verifier) and 100 % of `build-cython-ext` wins contain a Python
 * Traceback in their bash trace (median 21.6 % non-zero exit rate). Today's
 * verifier doesn't see either signal — the pre-submit self-check LLM auditor
 * also misses them because it has to re-derive everything from the raw trace
 * tail it is shown.
 *
 * This helper extracts those hygiene signals with cheap regex heuristics
 * BEFORE the LLM call so the auditor receives a structured, citeable
 * pre-scan it can pin its decision on. The signals are intentionally
 * conservative — false negatives are fine (the LLM has the raw trace too)
 * but false positives waste auditor budget, so we keep the patterns tight.
 *
 * Design constraints:
 *   1. Pure function, no I/O.
 *   2. Generic — no project-specific paths, no allowlist of "known good"
 *      tools. We just describe semantic patterns.
 *   3. Bounded — output sizes are capped so the prompt injection stays
 *      within the auditor's token budget.
 */

import type { BashTraceTailEntry } from "./pre-submit-self-check";
import { DEFAULT_PATH_PROFILE, type PathProfile } from "./types";

/** Maximum number of scratch-dir writes we report (defence-in-depth). */
export const SCRATCH_HYGIENE_MAX_TMP_WRITES = 12;

/** Maximum number of distinct traceback signatures we report. */
export const SCRATCH_HYGIENE_MAX_TRACEBACKS = 8;

/** Maximum length of a quoted error signature. */
export const SCRATCH_HYGIENE_SIGNATURE_MAX_CHARS = 200;

/** Maximum length of a quoted command for the structured signal. */
export const SCRATCH_HYGIENE_COMMAND_MAX_CHARS = 240;

export type ScratchHygieneSignal = {
  /**
   * Paths under `/tmp/` that the agent wrote (via `>`, `cat > /tmp/...`,
   * `cp ... /tmp/`, `mv ... /tmp/`, `mkdir /tmp/...`, `tee /tmp/...`,
   * `touch /tmp/...`) and for which we did NOT see a later cleanup
   * (`rm -rf /tmp/...`, `rm /tmp/...`) within the trace tail.
   */
  tmpWrites: { path: string; commandIdx: number }[];
  /**
   * Distinct exception / panic signatures observed in command output.
   * Each entry quotes the FIRST line of the offending exception (e.g.
   * `AttributeError: module 'numpy.array_api' has no attribute 'foo'`).
   */
  tracebacks: { signature: string; commandIdx: number }[];
  /**
   * The longest run of consecutive non-zero exit codes whose final
   * command was never re-run successfully later in the tail. Empty
   * `commands` array means the tail is clean.
   */
  nonZeroChain: { commands: string[]; exitCodes: number[] };
};

const truncate = (text: string, max: number): string => {
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(0, max - 3))}...`;
};

/** Escape a directory string for safe inclusion inside a regex character body. */
const escapeRegex = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Normalize a configured scratch dir to an absolute, no-trailing-slash form.
 * `/tmp` and `/tmp/` both produce `/tmp`; this keeps the synthesised regexes
 * stable regardless of how the operator authored the override.
 */
const normalizeScratchDir = (dir: string): string => {
  const trimmed = dir.trim().replace(/\/+$/g, "");
  return trimmed.length === 0 ? dir : trimmed;
};

/**
 * Extract scratch-dir write targets from a single bash command string.
 * Recognises shell redirection (`> <dir>/x`, `>> <dir>/x`), heredoc-style
 * `cat > <dir>/x`, copy / move / mkdir / touch / tee invocations. Each
 * configured scratch dir from `pathProfile.scratchDirs` participates; the
 * returned paths are absolute and start with one of those dirs.
 */
const extractTmpWrites = (command: string, scratchDirs: ReadonlyArray<string>): string[] => {
  const out = new Set<string>();
  const addAll = (re: RegExp): void => {
    for (const match of command.matchAll(re)) {
      const path = match[1];
      if (typeof path === "string" && path.length > 0) out.add(path);
    }
  };
  for (const rawDir of scratchDirs) {
    const dir = normalizeScratchDir(rawDir);
    const dirRe = escapeRegex(dir);
    // The captured path includes the dir prefix and the file/subpath after it.
    // Shell redirection: `> <dir>/foo` or `>> <dir>/foo` or `2> <dir>/foo`.
    addAll(new RegExp(String.raw`(?:^|[\s|;&])\d?>>?\s*(${dirRe}\/[\w./@+\-]+)`, "g"));
    // tee: `tee <dir>/foo`, `tee -a <dir>/foo`.
    addAll(new RegExp(String.raw`\btee(?:\s+-[aip]+)*\s+(${dirRe}\/[\w./@+\-]+)`, "g"));
    // cp/mv/install: last <dir>/ arg in the command.
    addAll(new RegExp(String.raw`\b(?:cp|mv|install)\b[^|;&]*?(${dirRe}\/[\w./@+\-]+)`, "g"));
    // mkdir: `mkdir -p <dir>/foo`.
    addAll(new RegExp(String.raw`\bmkdir\b(?:\s+-[a-z]+)*\s+(${dirRe}\/[\w./@+\-]+)`, "g"));
    // touch: `touch <dir>/foo`.
    addAll(new RegExp(String.raw`\btouch\b(?:\s+-[a-z]+)*\s+(${dirRe}\/[\w./@+\-]+)`, "g"));
  }
  return Array.from(out);
};

/**
 * Detect cleanup of a scratch-dir path in a single command. We accept any
 * `rm` invocation that names the path, names a parent directory of the
 * path, or sweeps any of the configured scratch root directories. Conservative —
 * false negatives are OK; false positives would silently drop a real leak signal.
 */
const cleansTmpPath = (
  command: string,
  path: string,
  scratchDirs: ReadonlyArray<string>,
): boolean => {
  if (!/\brm\b/.test(command)) return false;
  if (command.includes(path)) return true;
  // rm -rf <dir>/* — sweeping cleanup of any configured scratch root.
  for (const rawDir of scratchDirs) {
    const dir = normalizeScratchDir(rawDir);
    const dirRe = escapeRegex(dir);
    const sweepRe = new RegExp(String.raw`\brm\b[^|;&]*\s${dirRe}(?:\/\*+)?\s*(?:$|[|;&])`);
    if (sweepRe.test(command)) return true;
  }
  // Cleanup of an enclosing directory: walk upward.
  const segments = path.split("/").filter(Boolean); // ['tmp', 'a', 'b', 'c.txt']
  for (let i = segments.length - 1; i > 1; i -= 1) {
    const parent = `/${segments.slice(0, i).join("/")}`;
    if (command.includes(parent) && /\brm\b/.test(command)) return true;
  }
  return false;
};

const TRACEBACK_PATTERNS: { name: string; re: RegExp; sigGroup: number }[] = [
  // Python: "Traceback (most recent call last):" followed eventually by an
  // ExceptionType: message line. We capture the first matching exception
  // line because that's the actionable signal.
  {
    name: "python-traceback",
    re: /Traceback \(most recent call last\):[\s\S]*?\n([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning):.*)/m,
    sigGroup: 1,
  },
  // Go panic.
  { name: "go-panic", re: /^(panic:\s.+)$/m, sigGroup: 1 },
  // Rust panic.
  {
    name: "rust-panic",
    re: /^(thread '.+?' panicked at .+)$/m,
    sigGroup: 1,
  },
  // Segfault.
  {
    name: "segfault",
    re: /(Segmentation fault(?:\s\(core dumped\))?)/,
    sigGroup: 1,
  },
  // Generic compilation failure.
  { name: "compile-fail", re: /^(.*[Cc]ompilation (?:failed|terminated).*)$/m, sigGroup: 1 },
  // pytest summary line listing failures.
  { name: "pytest-failures", re: /^(=+\s*\d+\s+failed.*?=+)$/m, sigGroup: 1 },
];

const detectTracebackSignature = (output: string): string | null => {
  if (!output) return null;
  for (const { re, sigGroup } of TRACEBACK_PATTERNS) {
    const m = output.match(re);
    if (m) {
      const sig = (m[sigGroup] ?? m[0]).trim();
      if (sig.length === 0) continue;
      return truncate(sig, SCRATCH_HYGIENE_SIGNATURE_MAX_CHARS);
    }
  }
  return null;
};

/**
 * Identify the longest run of consecutive non-zero exit codes whose
 * final command was NEVER re-run with a zero exit later in the tail.
 * Two commands are "the same" iff their first 60 chars match — this
 * catches `pytest` retries that vary in args without being too sloppy.
 */
const detectNonZeroChain = (
  trace: ReadonlyArray<BashTraceTailEntry>,
): { commands: string[]; exitCodes: number[] } => {
  const empty = { commands: [], exitCodes: [] };
  if (trace.length === 0) return empty;
  type Run = { startIdx: number; commands: string[]; exitCodes: number[] };
  let bestRun: Run | null = null;
  let currentRun: Run | null = null;
  for (let i = 0; i < trace.length; i += 1) {
    const entry = trace[i];
    if (entry == null) continue;
    const exit = entry.exitCode;
    if (exit != null && exit !== 0) {
      if (currentRun == null) {
        currentRun = { startIdx: i, commands: [], exitCodes: [] };
      }
      currentRun.commands.push(entry.command);
      currentRun.exitCodes.push(exit);
    } else {
      if (currentRun && (bestRun == null || currentRun.commands.length > bestRun.commands.length)) {
        bestRun = currentRun;
      }
      currentRun = null;
    }
  }
  if (currentRun && (bestRun == null || currentRun.commands.length > bestRun.commands.length)) {
    bestRun = currentRun;
  }
  if (bestRun == null || bestRun.commands.length < 2) return empty;

  // Filter: skip if the last failing command was successfully re-run later.
  const lastFailingCommand = bestRun.commands[bestRun.commands.length - 1] ?? "";
  const lastFailingPrefix = lastFailingCommand.slice(0, 60);
  const lastFailingIdx = bestRun.startIdx + bestRun.commands.length - 1;
  for (let j = lastFailingIdx + 1; j < trace.length; j += 1) {
    const candidate = trace[j];
    if (candidate == null) continue;
    if (
      candidate.exitCode === 0 &&
      candidate.command.slice(0, 60) === lastFailingPrefix
    ) {
      return empty;
    }
  }
  return {
    commands: bestRun.commands.map((c) => truncate(c, SCRATCH_HYGIENE_COMMAND_MAX_CHARS)),
    exitCodes: bestRun.exitCodes,
  };
};

/**
 * Audit a bash trace tail for scratch-dir pollution, ignored tracebacks,
 * and uncleared non-zero exit chains. Returns a structured signal that
 * can be injected into the pre-submit self-check prompt verbatim.
 *
 * The function is pure and idempotent — it never reads from disk or the
 * network. It is safe to call on every pre-submit-self-check round.
 *
 * Accepts an optional `pathProfile` so deployments with non-standard scratch
 * conventions (Docker images that scratch into `/scratch`, etc.) can supply
 * their own list. When omitted, the Linux defaults (`/tmp`, `/var/tmp`)
 * baked into `BagConfigSchema.pathProfile` are used — preserving the
 * pre-config behavior byte-for-byte.
 */
export const auditScratchHygiene = (
  trace: ReadonlyArray<BashTraceTailEntry>,
  pathProfile: PathProfile = DEFAULT_PATH_PROFILE,
): ScratchHygieneSignal => {
  const tmpWrites: { path: string; commandIdx: number }[] = [];
  const tracebacks: { signature: string; commandIdx: number }[] = [];
  const seenTracebackSignatures = new Set<string>();
  const scratchDirs = pathProfile.scratchDirs;

  for (let i = 0; i < trace.length; i += 1) {
    const entry = trace[i];
    if (entry == null) continue;
    const commandIdx = i + 1; // 1-indexed for human readability
    // Scratch-dir writes — track each path, then drop ones that get
    // cleaned up later in the tail.
    for (const path of extractTmpWrites(entry.command ?? "", scratchDirs)) {
      let cleanedLater = false;
      for (let j = i + 1; j < trace.length; j += 1) {
        const later = trace[j];
        if (later == null) continue;
        if (cleansTmpPath(later.command ?? "", path, scratchDirs)) {
          cleanedLater = true;
          break;
        }
      }
      if (!cleanedLater && tmpWrites.length < SCRATCH_HYGIENE_MAX_TMP_WRITES) {
        tmpWrites.push({ path, commandIdx });
      }
    }
    // Tracebacks / panics / segfaults / compilation failures.
    const sig = detectTracebackSignature(entry.output ?? "");
    if (sig != null && !seenTracebackSignatures.has(sig)) {
      seenTracebackSignatures.add(sig);
      if (tracebacks.length < SCRATCH_HYGIENE_MAX_TRACEBACKS) {
        tracebacks.push({ signature: sig, commandIdx });
      }
    }
  }

  const nonZeroChain = detectNonZeroChain(trace);
  return { tmpWrites, tracebacks, nonZeroChain };
};

/**
 * Render the structured signal as a human-readable system-context block
 * that can be injected into the LLM prompt. Returns an empty string when
 * the audit is clean — callers can use that to skip injection entirely.
 */
export const renderScratchHygieneBlock = (
  signal: ScratchHygieneSignal,
): string => {
  const lines: string[] = [];
  if (signal.tmpWrites.length > 0) {
    const items = signal.tmpWrites
      .map((w) => `${w.path} (call #${w.commandIdx})`)
      .join(", ");
    lines.push(`Scratch writes detected (no cleanup observed): ${items}`);
  }
  if (signal.tracebacks.length > 0) {
    const items = signal.tracebacks
      .map((t) => `${t.signature} (call #${t.commandIdx})`)
      .join("; ");
    lines.push(`Tracebacks detected: ${items}`);
  }
  if (signal.nonZeroChain.commands.length > 0) {
    const pairs = signal.nonZeroChain.commands
      .map((c, i) => `"${c}" (exit ${signal.nonZeroChain.exitCodes[i]})`)
      .join(", ");
    lines.push(`Consecutive non-zero exits never re-run successfully: ${pairs}`);
  }
  if (lines.length === 0) return "";
  return ["[Pre-submit hygiene scan]", ...lines].join("\n");
};
