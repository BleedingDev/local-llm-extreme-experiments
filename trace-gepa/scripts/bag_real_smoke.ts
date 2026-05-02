/**
 * Wave-3 Agent #R smoke test (Wave-3 #V update):
 * Run BAG's `planDagIssues` end-to-end on a REAL coding task (not the toy
 * `echo hello` from Wave-1 Agent A). Capture the planner's actual JSON-shaped
 * output and assert validity.
 *
 * As of Wave-3 #V, the optimised prompt is wired into the EXECUTOR step
 * (action-selection, in `runAutonomousCodingTurn`), NOT the planner. So:
 *   - `planDagIssues` always uses the seed prompt and should produce a real,
 *     file-aware plan with verifiers regardless of BAG_DISABLE_OPTIMIZED_PROMPT.
 *   - The optimised executor-prompt log line `[bag] using optimized executor
 *     prompt run=...` will NOT appear in this script because we don't invoke
 *     `runAutonomousCodingTurn` here.
 *
 * Differences from `bag_smoke.ts`:
 *   - Uses a realistic, file-referencing coding task as the planner input.
 *   - Provides a real `repoContext` listing actual repo files so the planner
 *     can name them in `expectedFiles` / `verifierCommands`.
 *   - Honours `BAG_DISABLE_OPTIMIZED_PROMPT` for A/B compare (planner output
 *     should be unchanged in either mode now — the disabled-flag is no longer
 *     load-bearing for the planner step).
 *
 * Usage:
 *   bun run trace-gepa/scripts/bag_real_smoke.ts                # default
 *   BAG_DISABLE_OPTIMIZED_PROMPT=1 bun run trace-gepa/scripts/bag_real_smoke.ts
 *
 * Exits 0 on PASS, 1 on FAIL. Read-only on src/. Does NOT modify BAG sources.
 */

import { existsSync, readFileSync, readdirSync, readlinkSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..");
const TRACE_GEPA_ROOT = resolve(REPO_ROOT, "trace-gepa");
const LATEST_LINK = resolve(TRACE_GEPA_ROOT, "artifacts/optimized-prompts/latest");

// --- 1. Ensure .env (ANTHROPIC_AUTH_TOKEN) is loaded into process.env ---
const envFile = resolve(REPO_ROOT, ".env");
if (existsSync(envFile)) {
  const lines = readFileSync(envFile, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const m = /^([A-Z0-9_]+)=(.*)$/i.exec(line.trim());
    if (m && process.env[m[1]!] == null) {
      process.env[m[1]!] = m[2]!;
    }
  }
}

// --- 2. Loader plumbing for the optimised-prompt artefact ---
// The loader treats BAG_REPO_ROOT as authoritative when set. Pin it to the
// trace-gepa root so the artefact lookup matches the same layout the smoke
// test in `bag_smoke.ts` uses, regardless of cwd. Honour the operator's
// BAG_DISABLE_OPTIMIZED_PROMPT to opt out of the optimised path.
process.env.BAG_REPO_ROOT = TRACE_GEPA_ROOT;
const disabled =
  /^(1|true)$/i.test(process.env.BAG_DISABLE_OPTIMIZED_PROMPT?.trim() ?? "");

// --- 3. Capture stdout to grep for the [bag] line ---
const captured: string[] = [];
const origLog = console.log.bind(console);
console.log = (...args: unknown[]) => {
  const line = args
    .map((a) => (typeof a === "string" ? a : JSON.stringify(a)))
    .join(" ");
  captured.push(line);
  origLog(...args);
};

const fail = (msg: string, extra?: unknown): never => {
  origLog(`SMOKE FAIL: ${msg}`);
  if (extra !== undefined) origLog(extra);
  process.exit(1);
};

// --- 4. Build a real repo-context string for the planner ---
const repoContext = (): string => {
  const srcRoot = resolve(REPO_ROOT, "src");
  const adapters = resolve(srcRoot, "source-adapters");
  const adapterFiles = readdirSync(adapters).sort();
  const topSrc = readdirSync(srcRoot)
    .filter((name) => !name.startsWith("."))
    .sort();
  return [
    "Repository: bleeding-agent (BAG) — TypeScript ACP coding agent.",
    "",
    "Top-level src/ entries:",
    ...topSrc.map((n) => `  - src/${n}`),
    "",
    "Files under src/source-adapters/:",
    ...adapterFiles.map((n) => `  - src/source-adapters/${n}`),
    "",
    "Tests live under tests/ at the repo root (Bun test runner). Some adapters",
    "have a sibling tests/source-adapters/<name>.test.ts file; some don't.",
  ].join("\n");
};

// --- 5. Choose a small, realistic coding task (< 30 words). ---
const TASK =
  "List every TypeScript file under src/source-adapters/ that does NOT have a corresponding tests/source-adapters/<name>.test.ts file, and write the list to artifacts/missing_adapter_tests.txt.";

const main = async () => {
  const mode = disabled ? "seed" : "optimised";
  origLog(`[smoke] mode=${mode}`);
  origLog(`[smoke] BAG_REPO_ROOT=${TRACE_GEPA_ROOT}`);
  origLog(`[smoke] BAG_DISABLE_OPTIMIZED_PROMPT=${process.env.BAG_DISABLE_OPTIMIZED_PROMPT ?? "(unset)"}`);

  // Lazy imports so env vars are set before any module reads them.
  const { loadConfig } = await import(resolve(REPO_ROOT, "src/config.ts"));
  const { createLlmRouter } = await import(resolve(REPO_ROOT, "src/llm.ts"));
  const { planDagIssues } = await import(resolve(REPO_ROOT, "src/dag-tool-loop.ts"));

  // Sanity: artefact must exist where loader expects it (only matters when not disabled).
  const expectedArtifact = resolve(LATEST_LINK, "best_candidate.json");
  if (!disabled) {
    if (!existsSync(expectedArtifact)) {
      fail(`expected artifact missing at ${expectedArtifact}`);
    }
    const linkTarget = (() => {
      try {
        return readlinkSync(LATEST_LINK);
      } catch {
        return null;
      }
    })();
    origLog(`[smoke] latest -> ${linkTarget ?? "(not a symlink)"}`);
  }

  // Load config from the BAG repo root so master/local point at Anthropic.
  const config = loadConfig(REPO_ROOT);
  const router = createLlmRouter(config);
  if (!router.masterAvailable) {
    fail("router.masterAvailable is false; ANTHROPIC_AUTH_TOKEN missing or empty");
  }

  origLog(`[smoke] task: ${TASK}`);
  const ctx = repoContext();

  // 5-minute hard wallclock budget.
  const BUDGET_MS = 5 * 60 * 1000;
  const t0 = Date.now();
  let issues: unknown;
  let timedOut = false;
  try {
    issues = await Promise.race([
      planDagIssues({ router, task: TASK, repoContext: ctx }),
      new Promise<never>((_, reject) =>
        setTimeout(() => {
          timedOut = true;
          reject(new Error(`planDagIssues exceeded budget=${BUDGET_MS}ms`));
        }, BUDGET_MS),
      ),
    ]);
  } catch (e) {
    if (timedOut) {
      fail(`planner timed out after ${BUDGET_MS}ms`);
    }
    fail(`planDagIssues threw: ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
  }
  const ms = Date.now() - t0;
  origLog(`[smoke] planDagIssues returned in ${ms}ms`);
  origLog("[smoke] === issues JSON ===");
  origLog(JSON.stringify(issues, null, 2));
  origLog("[smoke] === end issues ===");

  // --- Assertions ---

  // (a) Wave-3 #V: optimised prompt is wired at the EXECUTOR step, not the
  // planner. We must NEVER see the stale planner-prompt log line. (The
  // executor log line `[bag] using optimized executor prompt run=...` would
  // only appear if we drove `runAutonomousCodingTurn`, which this script does
  // not.)
  const stalePlannerLog = captured.find((l) =>
    l.includes("[bag] using optimized planner prompt run="),
  );
  if (stalePlannerLog) {
    fail(
      "saw stale planner-prompt optimised log line — this should be wired to the executor step now",
      stalePlannerLog,
    );
  }
  const logLine: string | undefined = undefined; // retained for snapshot shape

  // (b) issues array parses (planDagIssues already does the JSON.parse internally;
  // we re-serialize and re-parse to confirm the shape is JSON-clean).
  const reparsed = JSON.parse(JSON.stringify(issues)) as unknown;

  // (c) issues array has >= 1 entry
  if (!Array.isArray(reparsed) || reparsed.length === 0) {
    fail("planner returned no issues (expected >= 1)", reparsed);
  }

  // (d) at least one issue's content references a real BAG repo file.
  // We accept any issue field whose body / title / expectedFiles / verifier
  // mentions a path that exists on disk, OR mentions a known BAG path token.
  const repoTokens = [
    "src/source-adapters",
    "src/",
    "tests/source-adapters",
    "tests/",
    "package.json",
    "artifacts/",
    "missing_adapter_tests.txt",
  ];
  const issueHasRepoRef = (entry: unknown): boolean => {
    if (entry == null || typeof entry !== "object") return false;
    const e = entry as Record<string, unknown>;
    const haystack = [
      typeof e.title === "string" ? e.title : "",
      typeof e.body === "string" ? e.body : "",
      ...(Array.isArray(e.expectedFiles) ? e.expectedFiles.map(String) : []),
      ...(Array.isArray(e.verifierCommands) ? e.verifierCommands.map(String) : []),
    ].join("\n");
    return repoTokens.some((t) => haystack.includes(t));
  };
  const refCount = (reparsed as unknown[]).filter(issueHasRepoRef).length;
  if (refCount === 0) {
    fail(
      "no issue references a known BAG repo path (src/, tests/, artifacts/, etc.)",
      reparsed,
    );
  }

  // (e) Wave-3 #V regression guard: confirm the planner did NOT degrade to its
  // hard-coded fallback (`task-1-direct` with no expectedFiles, no verifiers).
  // Pre-fix, the optimised planner-prompt mis-wiring caused every realistic
  // task to return exactly that fallback. The seed prompt should produce a
  // real plan with verifiers for this task.
  const isFallback = (entry: unknown): boolean => {
    if (entry == null || typeof entry !== "object") return false;
    const e = entry as Record<string, unknown>;
    const id = typeof e.issueId === "string" ? e.issueId : "";
    const expectedFiles = Array.isArray(e.expectedFiles) ? e.expectedFiles : [];
    const verifierCommands = Array.isArray(e.verifierCommands) ? e.verifierCommands : [];
    return id === "task-1-direct" && expectedFiles.length === 0 && verifierCommands.length === 0;
  };
  const fallbackCount = (reparsed as unknown[]).filter(isFallback).length;
  if (fallbackCount === (reparsed as unknown[]).length) {
    fail(
      "planner returned only the hard-coded fallback (task-1-direct, no verifiers) — wiring regression",
      reparsed,
    );
  }

  // --- Persist the run for the comparison report ---
  const outDir = resolve(TRACE_GEPA_ROOT, "artifacts/real_smoke");
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outFile = resolve(outDir, `${mode}-${stamp}.json`);
  writeFileSync(
    outFile,
    JSON.stringify(
      {
        mode,
        task: TASK,
        elapsedMs: ms,
        optimisedLogLine: logLine ?? null,
        issueCount: (reparsed as unknown[]).length,
        repoRefCount: refCount,
        issues: reparsed,
      },
      null,
      2,
    ),
  );

  origLog("");
  origLog(`SMOKE PASS: mode=${mode}`);
  origLog(`SMOKE PASS: planner produced ${(reparsed as unknown[]).length} issue(s); ${refCount} reference repo paths`);
  if (logLine) origLog(`SMOKE PASS: log line='${logLine}'`);
  origLog(`SMOKE PASS: wrote run snapshot -> ${outFile}`);
};

const teardown = (): void => {
  delete process.env.BAG_REPO_ROOT;
};

main()
  .then(() => {
    teardown();
    process.exit(0);
  })
  .catch((e: unknown) => {
    teardown();
    origLog(`SMOKE FAIL (uncaught): ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
    process.exit(1);
  });
