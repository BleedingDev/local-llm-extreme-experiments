/**
 * Wave-2 Agent #J verification (Wave-3 #V update):
 * Same as `bag_smoke.ts` but DOES NOT set `BAG_REPO_ROOT`. Confirms that the
 * top-level `<repo>/artifacts/optimized-prompts -> trace-gepa/...` symlink lets
 * the loader find the artefact under the default BAG repo root, and that the
 * planner step does NOT pick up the optimised prompt (which is now wired into
 * the executor step instead).
 *
 * Usage:
 *   bun run trace-gepa/scripts/bag_smoke_no_root.ts
 */

import { existsSync, readFileSync, readlinkSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..");
const LATEST_LINK = resolve(REPO_ROOT, "artifacts/optimized-prompts/latest");

// --- 1. Load .env ---
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

// --- 2. Loader is default-on; intentionally NO BAG_REPO_ROOT.
// We deliberately DO NOT clear BAG_DISABLE_OPTIMIZED_PROMPT — operators may
// pass it in to verify the seed-prompt fallback path.
delete process.env.BAG_REPO_ROOT;

// --- 3. Capture stdout ---
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
  const { loadConfig } = await import(resolve(REPO_ROOT, "src/config.ts"));
  const { createLlmRouter } = await import(resolve(REPO_ROOT, "src/llm.ts"));
  const { planDagIssues } = await import(resolve(REPO_ROOT, "src/dag-tool-loop.ts"));

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
  origLog(`[smoke] BAG_REPO_ROOT=<unset>`);
  origLog(`[smoke] resolved via top-level symlink at ${LATEST_LINK}`);
  origLog(`[smoke] latest -> ${linkTarget ?? "(not a symlink)"}`);

  const config = loadConfig(REPO_ROOT);
  const router = createLlmRouter(config);
  if (!router.masterAvailable) {
    fail("router.masterAvailable is false; ANTHROPIC_AUTH_TOKEN missing or empty");
  }

  const disabled = (() => {
    const v = process.env.BAG_DISABLE_OPTIMIZED_PROMPT;
    if (v == null) return false;
    const s = v.trim().toLowerCase();
    return s === "1" || s === "true";
  })();
  origLog(`[smoke] BAG_DISABLE_OPTIMIZED_PROMPT=${disabled ? "1 (fallback expected)" : "<unset>"}`);

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

  // Wave-3 #V: optimised prompt is wired into the EXECUTOR step now. The
  // planner always uses the seed prompt — no optimised log line should ever
  // be emitted by `planDagIssues`, regardless of disabled flag.
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
  if (linkTarget) origLog(`SMOKE PASS: optimised artefact resolvable -> ${linkTarget}`);
  origLog(`SMOKE PASS: planner produced ${(issues as unknown[]).length} issue(s)`);
};

const teardown = (): void => {
  // No-op: this script does not set any persistent env vars.
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
