/// <reference types="node" />
/**
 * Thin TS shim around the Python trace-RAG CLI. Spawns a subprocess and parses
 * its JSON stdout. Designed to be called from BAG's planner/executor when the
 * model is uncertain and wants past similar contexts.
 *
 * Repo-relative paths are auto-resolved via env vars set in .mcp.json or
 * via reasonable defaults.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as path from "node:path";

const execFileAsync = promisify(execFile);

const REPO_ROOT =
  process.env.BAG_REPO_ROOT ||
  path.resolve(__dirname, "..");
const PY_BIN =
  process.env.TRACE_RAG_PY ||
  path.join(REPO_ROOT, ".venv-gepa", "bin", "python");
const TRACE_GEPA = path.join(REPO_ROOT, "trace-gepa");
const INDEX_DIR =
  process.env.TRACE_RAG_INDEX_DIR ||
  path.join(TRACE_GEPA, "artifacts", "rag_index_v2");

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
  const k = Math.max(1, Math.min(opts.k ?? 5, 20));
  const timeoutMs = opts.timeoutMs ?? 8000;

  try {
    const { stdout } = await execFileAsync(
      PY_BIN,
      [
        "-m",
        "agent_opt.rag.cli",
        "--query",
        query,
        "--k",
        String(k),
        "--index-dir",
        INDEX_DIR,
      ],
      {
        cwd: TRACE_GEPA,
        env: { ...process.env, PYTHONPATH: TRACE_GEPA },
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
