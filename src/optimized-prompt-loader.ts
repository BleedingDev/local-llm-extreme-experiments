import { existsSync, lstatSync, readFileSync, readlinkSync } from "node:fs";
import { basename, dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DISABLE_FLAG = "BAG_DISABLE_OPTIMIZED_PROMPT";
const ENABLE_FLAG = "BAG_USE_OPTIMIZED_PROMPT";
const ARTIFACT_REL = "artifacts/optimized-prompts/latest/best_candidate.json";

const isDisabled = (): boolean => {
  const v = process.env[DISABLE_FLAG];
  if (v == null) return false;
  const s = v.trim().toLowerCase();
  return s === "1" || s === "true";
};

const isEnabled = (): boolean => {
  const v = process.env[ENABLE_FLAG];
  if (v == null) return false;
  const s = v.trim().toLowerCase();
  return s === "1" || s === "true" || s === "yes" || s === "on";
};

const repoRootCandidates = (): string[] => {
  const fromEnv = process.env.BAG_REPO_ROOT;
  // When BAG_REPO_ROOT is explicitly set, it wins authoritatively — no fallthrough
  // to cwd/here. This keeps tests hermetic and matches operator expectations
  // ("if I pin a root, that's the only place you look").
  if (fromEnv && fromEnv.length > 0) return [fromEnv];
  const cwd = process.cwd();
  // Module file lives under <repoRoot>/src/, so two levels up resolves the repo root
  // even when the process is launched from a sub-directory.
  const here = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  return [cwd, here];
};

const resolveRunId = (latestPath: string): string => {
  // <repoRoot>/artifacts/optimized-prompts/latest is expected to be a symlink
  // pointing at a sibling timestamped directory. Fall back to the parent dir
  // basename when it is a regular directory (e.g. test fixtures).
  try {
    const st = lstatSync(latestPath);
    if (st.isSymbolicLink()) {
      const target = readlinkSync(latestPath);
      return basename(target.replace(/\/+$/, ""));
    }
  } catch {
    // ignore — fall through to basename of parent
  }
  return basename(dirname(latestPath));
};

export const loadOptimizedExecutorPrompt = (): { system: string; runId: string } | null => {
  // Optimized prompts are an explicit promotion/runtime policy, not filesystem
  // discovery. A rotated `latest` artifact only takes effect when the caller
  // opts in; the disable flag remains an emergency override.
  if (isDisabled() || !isEnabled()) return null;
  for (const root of repoRootCandidates()) {
    const file = isAbsolute(ARTIFACT_REL) ? ARTIFACT_REL : resolve(root, ARTIFACT_REL);
    if (!existsSync(file)) continue;
    let raw: string;
    try {
      raw = readFileSync(file, "utf8");
    } catch {
      return null;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return null;
    }
    if (parsed == null || typeof parsed !== "object") return null;
    const obj = parsed as Record<string, unknown>;
    const system =
      typeof obj.system === "string" && obj.system.length > 0
        ? obj.system
        : typeof obj.prompt === "string" && obj.prompt.length > 0
          ? obj.prompt
          : null;
    if (system == null) return null;
    const latestDir = resolve(root, "artifacts/optimized-prompts/latest");
    const runId =
      typeof obj.runId === "string" && obj.runId.length > 0 ? obj.runId : resolveRunId(latestDir);
    return { system, runId };
  }
  return null;
};
