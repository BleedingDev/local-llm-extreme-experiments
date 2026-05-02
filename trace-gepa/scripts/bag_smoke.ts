/**
 * Wave-1 Action 1 smoke test (Wave-3 #V update):
 * Verify that BAG's `planDagIssues` plans an issue successfully and that the
 * optimised-prompt loader artefact resolves correctly under
 * `<BAG_REPO_ROOT>/artifacts/optimized-prompts/latest/best_candidate.json`.
 *
 * Note: as of Wave-3 #V, the optimised prompt is wired into the EXECUTOR step
 * (`runAutonomousCodingTurn` in src/autonomous-coding-turn.ts), not the
 * PLANNER step. So `planDagIssues` no longer logs the `[bag] using optimized
 * planner prompt run=...` line. This script now only asserts that the planner
 * succeeds and that the artefact is resolvable.
 *
 * Usage:
 *   bun run trace-gepa/scripts/bag_smoke.ts
 *
 * Verification-only. Does NOT modify any source files. Sets/unsets env vars
 * locally for the duration of this process only.
 */

import { existsSync, readFileSync, readlinkSync } from "node:fs";
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

// --- 2. Pin BAG_REPO_ROOT for THIS process only.
// Loader is default-on; ensure we are NOT disabled by an inherited env.
process.env.BAG_REPO_ROOT = TRACE_GEPA_ROOT;
delete process.env.BAG_DISABLE_OPTIMIZED_PROMPT;

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

const main = async () => {
  // Lazy imports so env vars are set before any module reads them.
  const { loadConfig } = await import(resolve(REPO_ROOT, "src/config.ts"));
  const { createLlmRouter } = await import(resolve(REPO_ROOT, "src/llm.ts"));
  const { planDagIssues } = await import(resolve(REPO_ROOT, "src/dag-tool-loop.ts"));

  // Sanity: artifact must exist where loader expects it.
  const expectedArtifact = resolve(LATEST_LINK, "best_candidate.json");
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
  origLog(`[smoke] BAG_REPO_ROOT=${TRACE_GEPA_ROOT}`);
  origLog(`[smoke] latest -> ${linkTarget ?? "(not a symlink)"}`);

  // Load config from real BAG repo root so master/local point at Anthropic.
  const config = loadConfig(REPO_ROOT);
  const router = createLlmRouter(config);
  if (!router.masterAvailable) {
    fail("router.masterAvailable is false; ANTHROPIC_AUTH_TOKEN missing or empty");
  }

  // Tiny planner input. planDagIssues uses role=local and json=true.
  const t0 = Date.now();
  let issues: unknown;
  try {
    issues = await planDagIssues({
      router,
      task: "echo hello",
      repoContext: "Repo: smoke test. Files: none.",
    });
  } catch (e) {
    fail(`planDagIssues threw: ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
  }
  const ms = Date.now() - t0;
  origLog(`[smoke] planDagIssues returned in ${ms}ms`);
  origLog(`[smoke] issues=${JSON.stringify(issues, null, 2)}`);

  // --- Assertions ---
  // Wave-3 #V: optimised prompt is now wired at the executor step, NOT the
  // planner. The planner uses the seed prompt unconditionally — no optimised
  // log line should be emitted from `planDagIssues`.
  const plannerLogLine = captured.find((l) =>
    l.includes("[bag] using optimized planner prompt run="),
  );
  if (plannerLogLine) {
    fail(
      "saw stale planner-prompt optimised log line — this should be wired to the executor step now",
      plannerLogLine,
    );
  }

  if (!Array.isArray(issues) || issues.length === 0) {
    fail("planner returned no issues (expected at least the fallback)", issues);
  }

  origLog("");
  origLog(`SMOKE PASS: planner uses seed prompt (no optimised override at planner step)`);
  origLog(`SMOKE PASS: optimised artefact resolvable -> ${linkTarget}`);
  origLog(`SMOKE PASS: planner produced ${(issues as unknown[]).length} issue(s)`);
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
