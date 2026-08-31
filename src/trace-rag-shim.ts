/// <reference types="node" />
/**
 * Thin TS shim around the Python trace-RAG CLI. Spawns a subprocess and parses
 * its JSON stdout. Designed to be called from BAG's planner/executor when the
 * model is uncertain and wants past similar contexts.
 *
 * Lookup is disabled unless BAG_USE_TRACE_RAG is explicitly enabled.
 * Repo-relative paths are auto-resolved via env vars or reasonable defaults.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);

const USE_TRACE_RAG_FLAG = "BAG_USE_TRACE_RAG";

const isEnabled = (value: string | undefined): boolean => {
  if (value == null) return false;
  const normalized = value.trim().toLowerCase();
  return (
    normalized === "1" ||
    normalized === "true" ||
    normalized === "yes" ||
    normalized === "on"
  );
};

const isTraceRagEnabled = (): boolean => isEnabled(process.env[USE_TRACE_RAG_FLAG]);

const repoRoot = (): string => {
  const fromEnv = process.env.BAG_REPO_ROOT;
  if (fromEnv != null && fromEnv.length > 0) return fromEnv;
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
};

const traceGepaDir = (): string => path.join(repoRoot(), "trace-gepa");

const pyBin = (): string =>
  process.env.TRACE_RAG_PY || path.join(repoRoot(), ".venv-gepa", "bin", "python");

const indexDir = (): string =>
  process.env.TRACE_RAG_INDEX_DIR || path.join(traceGepaDir(), "artifacts", "rag_index_v2");

export interface TraceRagHit {
  rank: number;
  similarity: number;
  id?: string | null;
  src?: string | null;
  src_path?: string | null;
  label?: string | null;
  failure_category?: string | null;
  observed_tool?: string | null;
  user_request_excerpt?: string;
  next_user_message_excerpt?: string;
}

export interface LookupOptions {
  k?: number;
  timeoutMs?: number;
}

export async function lookupSimilarSituation(
  query: string,
  opts: LookupOptions = {},
): Promise<TraceRagHit[]> {
  if (!query || !query.trim()) return [];
  if (!isTraceRagEnabled()) return [];
  const k = Math.max(1, Math.min(opts.k ?? 5, 20));
  const timeoutMs = opts.timeoutMs ?? 8000;

  try {
    const { stdout } = await execFileAsync(
      pyBin(),
      [
        "-m",
        "agent_opt.rag.cli",
        "--query",
        query,
        "--k",
        String(k),
        "--index-dir",
        indexDir(),
      ],
      {
        cwd: traceGepaDir(),
        env: { ...process.env, PYTHONPATH: traceGepaDir() },
        timeout: timeoutMs,
        maxBuffer: 8 * 1024 * 1024,
      },
    );
    const parsed = JSON.parse(stdout) as { results?: TraceRagHit[] };
    return parsed.results ?? [];
  } catch (err) {
    // Failsafe: never block BAG on retrieval issues. Log and return empty.
    if (process.env.BAG_TRACE_RAG_DEBUG) {
      console.error(`[trace-rag-shim] lookup failed: ${(err as Error).message}`);
    }
    return [];
  }
}

export function summariseHitsForPrompt(hits: TraceRagHit[]): string {
  if (!hits.length) return "";
  const lines: string[] = ["Similar past situations from your traces:"];
  for (const h of hits) {
    const cat = h.failure_category ? ` cat=${h.failure_category}` : "";
    const lbl = h.label ? ` label=${h.label}` : "";
    const tool = h.observed_tool ? ` tool=${h.observed_tool}` : "";
    const reqExc = (h.user_request_excerpt || "").slice(0, 120).replace(/\s+/g, " ").trim();
    const nxtExc = (h.next_user_message_excerpt || "").slice(0, 120).replace(/\s+/g, " ").trim();
    lines.push(`- (#${h.rank} sim=${h.similarity.toFixed(3)}${tool}${lbl}${cat}) ${reqExc}`);
    if (nxtExc) lines.push(`  user-followup: ${nxtExc}`);
  }
  return lines.join("\n");
}
