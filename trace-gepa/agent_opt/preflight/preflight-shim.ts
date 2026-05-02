/**
 * Preflight shim — calls the Python deterministic checker before tool dispatch.
 *
 * NOT wired into BAG yet. Scaffold only. Wire site:
 *   src/dispatch.ts → before `tool.handler(input)` add:
 *     const veto = await preflightCheck(action, ctx);
 *     if (!veto.passed) return { is_error: true, content: veto.blocked_by.join("\n") };
 */
import { spawnSync } from "node:child_process";

export interface PreflightAction {
  name: string;
  input: string | Record<string, unknown>;
}

export interface PreflightContext {
  recent_actions?: string[];
  available_skills?: string[];
}

export interface PreflightResult {
  passed: boolean;
  blocked_by: string[];
  warnings: string[];
  fired: string[];
}

const PYTHON =
  process.env.PREFLIGHT_PYTHON ??
  // default to project venv
  ".venv-gepa/bin/python";

export function preflightCheck(
  action: PreflightAction,
  context: PreflightContext = {},
): PreflightResult {
  const proc = spawnSync(
    PYTHON,
    [
      "-m",
      "agent_opt.preflight.cli",
      "--action",
      JSON.stringify(action),
      "--context",
      JSON.stringify(context),
    ],
    { encoding: "utf8", timeout: 2000 },
  );
  if (proc.error || proc.status === null) {
    // fail-open: never crash the dispatcher because preflight bombed
    return { passed: true, blocked_by: [], warnings: [`preflight error: ${proc.error}`], fired: [] };
  }
  try {
    return JSON.parse(proc.stdout) as PreflightResult;
  } catch {
    return { passed: true, blocked_by: [], warnings: ["preflight: bad json output"], fired: [] };
  }
}
