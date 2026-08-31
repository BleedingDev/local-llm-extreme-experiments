import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Auto-discovered failure cluster (built by `scripts/build_failure_clusters.py`).
 * Sibling to the curated `verifier-signature-library` — same shape role, but
 * data-driven from the BAG trial corpus.
 */
export type FailureCluster = {
  id: string;
  name: string;
  size: number;
  trialIds: string[];
  signature: string;
  tasks: string[];
  firstSeen: string;
  lastSeen: string;
  exemplarVerifierExcerpt: string;
};

export type FailureClustersDocument = {
  generatedAt: string;
  totalFailures: number;
  clusters: FailureCluster[];
};

const CLUSTERS_RELATIVE_PATH = "bench/.bag/optimizer/failure-clusters.json";
const TRIGRAM_N = 3;
const DEFAULT_MATCH_THRESHOLD = 0.45;

type RawCluster = {
  id?: string;
  name?: string;
  size?: number;
  trial_ids?: string[];
  signature?: string;
  tasks?: string[];
  first_seen?: string;
  last_seen?: string;
  exemplar_verifier_excerpt?: string;
};

type RawDocument = {
  generated_at?: string;
  total_failures?: number;
  clusters?: RawCluster[];
};

const normalizeCluster = (raw: RawCluster): FailureCluster | null => {
  const id = raw.id ?? raw.name;
  const signature = raw.signature ?? "";
  if (!id || signature.length === 0) return null;
  return {
    id,
    name: raw.name ?? id,
    size: raw.size ?? (raw.trial_ids?.length ?? 0),
    trialIds: raw.trial_ids ?? [],
    signature,
    tasks: raw.tasks ?? [],
    firstSeen: raw.first_seen ?? "",
    lastSeen: raw.last_seen ?? "",
    exemplarVerifierExcerpt: raw.exemplar_verifier_excerpt ?? "",
  };
};

/**
 * Load the auto-discovered failure clusters document. Returns `null` when the
 * artifact has not been generated yet (e.g. fresh checkout) — callers should
 * treat that as "no auto-cluster signal available" rather than an error.
 */
export const loadFailureClusters = (
  cwd: string,
): FailureClustersDocument | null => {
  const path = join(cwd, CLUSTERS_RELATIVE_PATH);
  if (!existsSync(path)) return null;
  let parsed: RawDocument;
  try {
    parsed = JSON.parse(readFileSync(path, "utf-8")) as RawDocument;
  } catch {
    return null;
  }
  const clusters = (parsed.clusters ?? [])
    .map(normalizeCluster)
    .filter((c): c is FailureCluster => c !== null);
  return {
    generatedAt: parsed.generated_at ?? "",
    totalFailures: parsed.total_failures ?? clusters.length,
    clusters,
  };
};

const trigrams = (text: string): Set<string> => {
  const s = text.replace(/\s+/g, " ").trim().toLowerCase();
  if (s.length < TRIGRAM_N) return s.length > 0 ? new Set([s]) : new Set();
  const out = new Set<string>();
  for (let i = 0; i <= s.length - TRIGRAM_N; i += 1) {
    out.add(s.slice(i, i + TRIGRAM_N));
  }
  return out;
};

const jaccard = (a: Set<string>, b: Set<string>): number => {
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter += 1;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
};

/**
 * Extract a likely failure signature from a verifier output blob. Mirrors the
 * Python script's heuristic so live matching uses the same shape.
 */
const extractSignature = (verifierOutput: string): string => {
  const lines = verifierOutput
    .split(/\r?\n/)
    .map((ln) => ln.trimEnd())
    .filter((ln) => ln.trim().length > 0);
  if (lines.length === 0) return "";

  const eLines = lines
    .filter((ln) => /^E\s/.test(ln))
    .map((ln) => ln.slice(1).trim())
    .filter((ln) => ln.length > 5);

  const headerRx = /^(?:[A-Z][A-Za-z]*Error|[A-Z][A-Za-z]*Exception|assert\b)/;
  for (let i = eLines.length - 1; i >= 0; i -= 1) {
    const sig = eLines[i];
    if (sig !== undefined && headerRx.test(sig)) return sig;
  }
  for (let i = eLines.length - 1; i >= 0; i -= 1) {
    const sig = eLines[i];
    if (sig === undefined) continue;
    const low = sig.toLowerCase();
    if (low.startsWith("+ ") || low.startsWith("where ") || low.startsWith("use -")) {
      continue;
    }
    return sig;
  }
  if (eLines.length > 0) {
    const last = eLines[eLines.length - 1];
    if (last !== undefined) return last;
  }

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const candidate = lines[i];
    if (candidate === undefined) continue;
    if (/\b[A-Z][A-Za-z]*Error\b\s*:/.test(candidate)) return candidate.trim();
  }
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const candidate = lines[i];
    if (candidate === undefined) continue;
    if (/^\s*FAILED\s/.test(candidate)) return candidate.trim();
  }
  return lines.slice(-3).join(" | ");
};

/**
 * Match a verifier output against the auto-discovered clusters. Returns the
 * single best match by trigram-Jaccard similarity over the extracted
 * signature, or `null` if the best score is below `threshold`.
 */
export const matchClusterByVerifierOutput = (
  doc: FailureClustersDocument,
  verifierOutput: string,
  threshold: number = DEFAULT_MATCH_THRESHOLD,
): FailureCluster | null => {
  if (doc.clusters.length === 0) return null;
  const sig = extractSignature(verifierOutput);
  if (sig.length === 0) return null;
  const sigGr = trigrams(sig);

  let best: FailureCluster | null = null;
  let bestScore = 0;
  for (const cluster of doc.clusters) {
    const score = jaccard(sigGr, trigrams(cluster.signature));
    if (score > bestScore) {
      bestScore = score;
      best = cluster;
    }
  }
  if (best === null || bestScore < threshold) return null;
  return best;
};

/** Internal helper exported for tests only. */
export const __test = { extractSignature, trigrams, jaccard };
