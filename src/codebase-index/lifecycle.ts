/**
 * Codebase-index lifecycle helpers.
 *
 * Hosts:
 *   - `installColgrep()` — best-effort install of the colgrep binary on the
 *     current host. macOS uses Homebrew tap, Linux falls back to `cargo
 *     install`. Returns silently when neither path is available (graceful
 *     degradation: BAG keeps working without code_search).
 *   - `indexStatus(cwd)` — inspect the persisted state file under
 *     `<cwd>/.bag/codebase-index/`.
 *   - `rebuildIndex(cwd)` — force a full rebuild via the colgrep adapter.
 *
 * Skips entirely when the colgrep binary is absent — surfaces an
 * `available: false` flag the caller can present to the operator.
 */

import { spawn } from "node:child_process";
import { stat, readFile } from "node:fs/promises";
import { join as joinPath } from "node:path";
import { colgrepBackend } from "./colgrep-bridge";

const which = async (binary: string): Promise<boolean> =>
  new Promise((resolveFn) => {
    const child = spawn("sh", ["-c", `command -v ${binary} >/dev/null 2>&1`]);
    child.on("close", (code) => resolveFn(code === 0));
    child.on("error", () => resolveFn(false));
  });

const runShell = async (cmd: string, timeoutMs = 600_000): Promise<number> =>
  new Promise((resolveFn) => {
    const child = spawn("sh", ["-c", cmd]);
    let timer: NodeJS.Timeout | null = null;
    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        try {
          child.kill("SIGTERM");
        } catch {
          /* ignore */
        }
      }, timeoutMs);
    }
    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      resolveFn(code ?? 1);
    });
    child.on("error", () => {
      if (timer) clearTimeout(timer);
      resolveFn(127);
    });
  });

export type InstallColgrepResult = {
  installed: boolean;
  alreadyPresent: boolean;
  method?: "brew" | "cargo" | "noop";
  reason?: string;
};

export const installColgrep = async (): Promise<InstallColgrepResult> => {
  if (await which("colgrep")) {
    return { installed: true, alreadyPresent: true, method: "noop" };
  }
  if (process.platform === "darwin" && (await which("brew"))) {
    const rc = await runShell("brew install lightonai/tap/colgrep");
    if (rc === 0 && (await which("colgrep"))) {
      return { installed: true, alreadyPresent: false, method: "brew" };
    }
    return {
      installed: false,
      alreadyPresent: false,
      method: "brew",
      reason: `brew install lightonai/tap/colgrep failed (rc=${rc})`,
    };
  }
  if (await which("cargo")) {
    const rc = await runShell("cargo install colgrep");
    if (rc === 0 && (await which("colgrep"))) {
      return { installed: true, alreadyPresent: false, method: "cargo" };
    }
    return {
      installed: false,
      alreadyPresent: false,
      method: "cargo",
      reason: `cargo install colgrep failed (rc=${rc})`,
    };
  }
  return {
    installed: false,
    alreadyPresent: false,
    reason:
      "no install path available (no brew, no cargo); operator must install colgrep manually",
  };
};

export type IndexStatus = {
  available: boolean;
  exists: boolean;
  sizeBytes?: number;
  lastBuilt?: string;
  lastIncremental?: string;
  staleFiles?: number;
  reason?: string;
};

const INDEX_DIR_REL = ".bag/codebase-index";
const STATE_FILE = "colgrep.idx-state.json";

export const indexStatus = async (cwd: string): Promise<IndexStatus> => {
  const available = await which("colgrep");
  const stateFile = joinPath(cwd, INDEX_DIR_REL, STATE_FILE);
  let exists = false;
  let sizeBytes: number | undefined;
  let lastBuilt: string | undefined;
  let lastIncremental: string | undefined;
  try {
    const st = await stat(stateFile);
    exists = st.isFile();
    sizeBytes = st.size;
    if (exists) {
      const body = await readFile(stateFile, "utf8");
      const parsed = JSON.parse(body) as {
        lastBuiltAt?: string;
        lastIncrementalAt?: string;
      };
      lastBuilt = parsed.lastBuiltAt;
      lastIncremental = parsed.lastIncrementalAt;
    }
  } catch {
    /* not present yet */
  }
  const result: IndexStatus = { available, exists };
  if (sizeBytes !== undefined) result.sizeBytes = sizeBytes;
  if (lastBuilt !== undefined) result.lastBuilt = lastBuilt;
  if (lastIncremental !== undefined) result.lastIncremental = lastIncremental;
  if (!available) result.reason = "colgrep binary not on PATH";
  return result;
};

export const rebuildIndex = async (cwd: string): Promise<{ status: string; durationMs?: number }> => {
  const backend = colgrepBackend({ forceRebuild: true });
  if (!(await backend.isAvailable())) {
    return { status: "skipped" };
  }
  const r = await backend.ensureIndex({ cwd });
  const out: { status: string; durationMs?: number } = { status: r.status };
  if (r.durationMs !== undefined) out.durationMs = r.durationMs;
  return out;
};
