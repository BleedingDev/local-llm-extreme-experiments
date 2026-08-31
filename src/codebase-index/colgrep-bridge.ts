/**
 * Generic codebase-search backend interface + concrete ColGrep adapter.
 *
 * Design notes:
 *   - The `CodebaseSearchBackend` interface intentionally hides the concrete
 *     implementation. ColGrep (LightOn's late-interaction code search) is the
 *     first impl, but the same interface accepts any future late-interaction
 *     or vector-search backend (Vera, Sourcegraph, Cody, …).
 *   - No model-name-specific or query-keyword-specific logic. The caller
 *     formulates the query; the backend is a black box.
 *   - Graceful degradation: when the colgrep binary is absent (most CI /
 *     containers initially), `isAvailable()` returns false. Callers must
 *     surface a structured error to the model rather than crashing.
 *   - Index lifecycle state lives at `<cwd>/.bag/codebase-index/`. That path
 *     is excluded from the probe-snapshot/restore cleaner used by
 *     `instruction-verifier.ts` (see find with `-not -path '<star>/.bag/<star>'`),
 *     so building the index never pollutes a workspace probe.
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdir, readFile, writeFile, stat } from "node:fs/promises";
import { resolve as resolvePath, join as joinPath } from "node:path";

export type CodebaseSearchHit = {
  file: string;
  lineRange: [number, number];
  symbol?: string;
  unitKind?: string;
  score: number;
  snippet?: string;
};

export type EnsureIndexInput = {
  cwd: string;
  signal?: AbortSignal;
};

export type EnsureIndexResult = {
  status: "fresh" | "rebuilt" | "incremental" | "skipped";
  bytes?: number;
  durationMs?: number;
};

export type SearchInput = {
  cwd: string;
  query: string;
  topK?: number;
  mode?: "semantic" | "hybrid";
  pathFilter?: string;
  languageFilter?: string;
  signal?: AbortSignal;
};

export interface CodebaseSearchBackend {
  ensureIndex(input: EnsureIndexInput): Promise<EnsureIndexResult>;
  search(input: SearchInput): Promise<CodebaseSearchHit[]>;
  invalidate?(input: { cwd: string }): Promise<void>;
  isAvailable(): Promise<boolean>;
}

/** Minimal subprocess runner shape. Substitutable for tests. */
export type SubprocessRunner = (input: {
  command: string;
  args: string[];
  cwd: string;
  env?: NodeJS.ProcessEnv;
  signal?: AbortSignal;
  stdin?: string;
  timeoutMs?: number;
}) => Promise<{ stdout: string; stderr: string; exitCode: number | null }>;

const defaultRunner: SubprocessRunner = (input) =>
  new Promise((resolveFn) => {
    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawn(input.command, input.args, {
        cwd: input.cwd,
        env: input.env ?? process.env,
      });
    } catch (err) {
      resolveFn({ stdout: "", stderr: String(err), exitCode: 127 });
      return;
    }
    let stdout = "";
    let stderr = "";
    let timer: NodeJS.Timeout | null = null;
    if (input.timeoutMs && input.timeoutMs > 0) {
      timer = setTimeout(() => {
        try {
          child.kill("SIGTERM");
        } catch {
          /* ignore */
        }
      }, input.timeoutMs);
    }
    if (input.signal) {
      const onAbort = (): void => {
        try {
          child.kill("SIGTERM");
        } catch {
          /* ignore */
        }
      };
      if (input.signal.aborted) onAbort();
      else input.signal.addEventListener("abort", onAbort, { once: true });
    }
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (err) => {
      if (timer) clearTimeout(timer);
      resolveFn({ stdout, stderr: stderr + String(err), exitCode: 127 });
    });
    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      resolveFn({ stdout, stderr, exitCode: code });
    });
    if (input.stdin) {
      child.stdin.write(input.stdin);
      child.stdin.end();
    }
  });

/**
 * Where we persist per-workspace index lifecycle state. This file holds:
 *   - lastBuiltAt: ISO timestamp of last full rebuild
 *   - lastIncrementalAt: ISO timestamp of last incremental update
 *   - sourceFingerprint: a coarse hash of (file count + total bytes + max mtime)
 *     so we can detect whether the corpus drifted enough to need an update
 * Persisted under `<cwd>/.bag/codebase-index/` which is already excluded from
 * the probe snapshot/restore in `instruction-verifier.ts`.
 */
type IndexState = {
  lastBuiltAt?: string;
  lastIncrementalAt?: string;
  sourceFingerprint?: string;
  binaryVersion?: string;
};

const INDEX_DIR_REL = ".bag/codebase-index";
const STATE_FILE = "colgrep.idx-state.json";

const stateFilePath = (cwd: string): string =>
  joinPath(cwd, INDEX_DIR_REL, STATE_FILE);

const readIndexState = async (cwd: string): Promise<IndexState | null> => {
  try {
    const raw = await readFile(stateFilePath(cwd), "utf8");
    return JSON.parse(raw) as IndexState;
  } catch {
    return null;
  }
};

const writeIndexState = async (cwd: string, state: IndexState): Promise<void> => {
  const dir = joinPath(cwd, INDEX_DIR_REL);
  await mkdir(dir, { recursive: true });
  await writeFile(stateFilePath(cwd), JSON.stringify(state, null, 2), "utf8");
};

/**
 * Coarse source fingerprint — list tracked files via `git ls-files`, fall back
 * to find. We hash (count, total-bytes, max-mtime) into one short string. Not
 * cryptographic — only used to short-circuit "nothing changed" rebuilds.
 */
const computeSourceFingerprint = async (
  cwd: string,
  runner: SubprocessRunner,
): Promise<string> => {
  // Try git first
  let listing = "";
  const gitRes = await runner({
    command: "git",
    args: ["ls-files", "-z"],
    cwd,
    timeoutMs: 10_000,
  });
  if (gitRes.exitCode === 0 && gitRes.stdout.length > 0) {
    listing = gitRes.stdout;
  } else {
    const findRes = await runner({
      command: "find",
      args: [
        ".",
        "-type",
        "f",
        "-not",
        "-path",
        "./.bag/*",
        "-not",
        "-path",
        "./.git/*",
        "-not",
        "-path",
        "./node_modules/*",
        "-print0",
      ],
      cwd,
      timeoutMs: 30_000,
    });
    listing = findRes.stdout;
  }
  const files = listing
    .split("\0")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  let count = 0;
  let totalBytes = 0;
  let maxMtimeMs = 0;
  for (const rel of files) {
    try {
      const st = await stat(joinPath(cwd, rel));
      if (!st.isFile()) continue;
      count += 1;
      totalBytes += st.size;
      if (st.mtimeMs > maxMtimeMs) maxMtimeMs = st.mtimeMs;
    } catch {
      // skipped
    }
  }
  return `c=${count};b=${totalBytes};m=${Math.floor(maxMtimeMs)}`;
};

export type ColgrepBackendOptions = {
  /** Override for tests; defaults to a real spawn() runner. */
  runner?: SubprocessRunner;
  /** Path to the colgrep binary. Default: lookup on PATH. */
  binary?: string;
  /** When true, force fresh rebuild even if fingerprints match. */
  forceRebuild?: boolean;
  /** Optional logger sink for non-fatal diagnostics. */
  logger?: (msg: string) => void;
};

const DEFAULT_BINARY = "colgrep";

export const colgrepBackend = (options: ColgrepBackendOptions = {}): CodebaseSearchBackend => {
  const runner = options.runner ?? defaultRunner;
  const binary = options.binary ?? DEFAULT_BINARY;
  const log = options.logger ?? ((_m: string): void => undefined);

  const isAvailable = async (): Promise<boolean> => {
    const r = await runner({
      command: binary,
      args: ["--version"],
      cwd: process.cwd(),
      timeoutMs: 5_000,
    });
    return r.exitCode === 0;
  };

  const ensureIndex = async (input: EnsureIndexInput): Promise<EnsureIndexResult> => {
    const cwd = resolvePath(input.cwd);
    const startedMs = Date.now();
    if (!(await isAvailable())) {
      return { status: "skipped" };
    }
    const prev = await readIndexState(cwd);
    const fingerprint = await computeSourceFingerprint(cwd, runner);
    const sameFingerprint =
      prev?.sourceFingerprint === fingerprint && (prev?.lastBuiltAt || prev?.lastIncrementalAt);
    if (sameFingerprint && !options.forceRebuild) {
      return { status: "fresh", durationMs: Date.now() - startedMs };
    }
    if (prev?.lastBuiltAt && !options.forceRebuild) {
      // Incremental update path. We feed `colgrep update` (delegating the
      // actual diff scan to the colgrep binary, which knows what its own
      // index already contains).
      const updateOpts: Parameters<SubprocessRunner>[0] = {
        command: binary,
        args: ["update", "-y"],
        cwd,
        timeoutMs: 600_000,
      };
      if (input.signal) updateOpts.signal = input.signal;
      const r = await runner(updateOpts);
      if (r.exitCode === 0) {
        await writeIndexState(cwd, {
          ...prev,
          sourceFingerprint: fingerprint,
          lastIncrementalAt: new Date().toISOString(),
        });
        return { status: "incremental", durationMs: Date.now() - startedMs };
      }
      log(`colgrep update failed (rc=${r.exitCode}); falling back to full init: ${r.stderr.slice(0, 200)}`);
    }
    // Full (re)build path.
    const initOpts: Parameters<SubprocessRunner>[0] = {
      command: binary,
      args: ["init", "-y"],
      cwd,
      timeoutMs: 1_800_000,
    };
    if (input.signal) initOpts.signal = input.signal;
    const r = await runner(initOpts);
    if (r.exitCode !== 0) {
      throw new Error(
        `colgrep init failed (rc=${r.exitCode}): ${r.stderr.slice(0, 500) || r.stdout.slice(0, 500)}`,
      );
    }
    const state: IndexState = {
      lastBuiltAt: new Date().toISOString(),
      sourceFingerprint: fingerprint,
    };
    await writeIndexState(cwd, state);
    return {
      status: prev ? "rebuilt" : "fresh",
      durationMs: Date.now() - startedMs,
    };
  };

  const parseHits = (raw: string): CodebaseSearchHit[] => {
    const trimmed = raw.trim();
    if (trimmed.length === 0) return [];
    // Two supported formats:
    //   1. JSON array of hit objects (preferred — colgrep --json)
    //   2. JSONL — one hit per line (older colgrep versions)
    let parsedAny: unknown;
    try {
      parsedAny = JSON.parse(trimmed);
    } catch {
      // Try JSONL fallback
      const out: CodebaseSearchHit[] = [];
      for (const line of trimmed.split(/\r?\n/)) {
        const t = line.trim();
        if (t.length === 0) continue;
        try {
          out.push(normalizeHit(JSON.parse(t)));
        } catch {
          // skip malformed line
        }
      }
      return out;
    }
    if (Array.isArray(parsedAny)) {
      return parsedAny.map((h) => normalizeHit(h));
    }
    if (parsedAny && typeof parsedAny === "object") {
      const obj = parsedAny as { hits?: unknown };
      if (Array.isArray(obj.hits)) return obj.hits.map((h) => normalizeHit(h));
    }
    return [];
  };

  const search = async (input: SearchInput): Promise<CodebaseSearchHit[]> => {
    const cwd = resolvePath(input.cwd);
    if (!(await isAvailable())) {
      throw new Error(
        "code_search backend unavailable: `colgrep` binary not found on PATH. " +
          "Install via `brew install lightonai/tap/colgrep` (macOS) or `cargo install colgrep`. " +
          "Fall back to `bash` + `rg` for this query.",
      );
    }
    const args: string[] = ["search", "--json"];
    const topK = Math.max(1, Math.min(100, input.topK ?? 10));
    args.push("--top-k", String(topK));
    const mode = input.mode ?? "hybrid";
    if (mode === "semantic") args.push("--mode", "semantic");
    else args.push("--mode", "hybrid");
    if (input.pathFilter && input.pathFilter.length > 0) {
      args.push("--path", input.pathFilter);
    }
    if (input.languageFilter && input.languageFilter.length > 0) {
      args.push("--lang", input.languageFilter);
    }
    args.push("--", input.query);
    const searchOpts: Parameters<SubprocessRunner>[0] = {
      command: binary,
      args,
      cwd,
      timeoutMs: 60_000,
    };
    if (input.signal) searchOpts.signal = input.signal;
    const r = await runner(searchOpts);
    if (r.exitCode !== 0) {
      throw new Error(
        `colgrep search failed (rc=${r.exitCode}): ${r.stderr.slice(0, 400) || r.stdout.slice(0, 400)}`,
      );
    }
    return parseHits(r.stdout);
  };

  const invalidate = async (input: { cwd: string }): Promise<void> => {
    const cwd = resolvePath(input.cwd);
    try {
      await writeIndexState(cwd, {});
    } catch {
      // file may not exist; that's fine
    }
  };

  return { ensureIndex, search, invalidate, isAvailable };
};

const normalizeHit = (raw: unknown): CodebaseSearchHit => {
  const r = (raw ?? {}) as Record<string, unknown>;
  const file = typeof r.file === "string" ? r.file : typeof r.path === "string" ? r.path : "";
  let lineRange: [number, number] = [0, 0];
  if (Array.isArray(r.lineRange) && r.lineRange.length === 2) {
    const a = Number(r.lineRange[0]);
    const b = Number(r.lineRange[1]);
    if (Number.isFinite(a) && Number.isFinite(b)) lineRange = [a, b];
  } else if (typeof r.line_start === "number" && typeof r.line_end === "number") {
    lineRange = [r.line_start, r.line_end];
  } else if (typeof r.start === "number" && typeof r.end === "number") {
    lineRange = [r.start, r.end];
  } else if (typeof r.line === "number") {
    lineRange = [r.line, r.line];
  }
  const score =
    typeof r.score === "number"
      ? r.score
      : typeof r.similarity === "number"
        ? r.similarity
        : 0;
  const hit: CodebaseSearchHit = { file, lineRange, score };
  if (typeof r.symbol === "string") hit.symbol = r.symbol;
  if (typeof r.unitKind === "string") hit.unitKind = r.unitKind;
  else if (typeof r.kind === "string") hit.unitKind = r.kind;
  if (typeof r.snippet === "string") hit.snippet = r.snippet;
  else if (typeof r.text === "string") hit.snippet = r.text;
  return hit;
};

/**
 * Render a list of hits as a compact text observation suitable for
 * tool-result message content. Caller can override `maxSnippetChars` to
 * trim per-hit body. Returns "no results" when empty.
 */
export const renderHitsAsObservation = (
  hits: ReadonlyArray<CodebaseSearchHit>,
  options: { maxSnippetChars?: number } = {},
): string => {
  if (hits.length === 0) return "code_search: no results.";
  const cap = options.maxSnippetChars ?? 240;
  const lines: string[] = [`code_search: ${hits.length} hit(s).`];
  hits.forEach((hit, i) => {
    const span = `${hit.file}:${hit.lineRange[0]}-${hit.lineRange[1]}`;
    const sym = hit.symbol ? ` [${hit.symbol}${hit.unitKind ? ` ${hit.unitKind}` : ""}]` : "";
    lines.push(`#${i + 1} score=${hit.score.toFixed(3)} ${span}${sym}`);
    if (hit.snippet && hit.snippet.length > 0) {
      const snippet = hit.snippet.length > cap ? `${hit.snippet.slice(0, cap)}…` : hit.snippet;
      lines.push(snippet);
    }
  });
  return lines.join("\n");
};
