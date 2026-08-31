#!/usr/bin/env -S npx tsx
/**
 * BAG tactics audit — list every tactic with its forensic incident pointer,
 * introduction date, review-by date, and status. Flag tactics whose
 * `review_by` has passed; suggest retirement candidates whose incident
 * hasn't recurred in the last N runs (default 30).
 *
 * Usage:
 *   tsx scripts/bag_tactics_audit.ts                     # human-readable
 *   tsx scripts/bag_tactics_audit.ts --json              # machine-readable
 *   tsx scripts/bag_tactics_audit.ts --recurrence-window 60
 *
 * Exit code is 0 unless `--strict` is passed AND there are overdue tactics.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadAllTactics, type Tactic } from "../src/prompts/loader";

type AuditRow = {
  id: string;
  status: "active" | "deprecated";
  order: number | null;
  incident: string;
  introduced: string | null;
  review_by: string | null;
  trigger: string | null;
  merged_into: string | null;
  overdue: boolean;
  daysUntilReview: number | null;
  recurrenceHits: number;
  retirementCandidate: boolean;
  path: string;
};

const today = (): Date => {
  const d = new Date();
  return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
};

const parseDate = (s: string | null | undefined): Date | null => {
  if (s == null || s.length === 0) return null;
  // Accept YYYY-MM-DD only (matches our frontmatter convention).
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s.trim());
  if (m == null) return null;
  return new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
};

const daysBetween = (from: Date, to: Date): number => {
  const ms = to.getTime() - from.getTime();
  return Math.floor(ms / (24 * 60 * 60 * 1000));
};

const parseArgs = (argv: string[]): {
  json: boolean;
  recurrenceWindow: number;
  strict: boolean;
  repoRoot: string;
  runsDir: string | null;
} => {
  let json = false;
  let recurrenceWindow = 30;
  let strict = false;
  let repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  let runsDir: string | null = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--json") json = true;
    else if (a === "--strict") strict = true;
    else if (a === "--recurrence-window" || a === "-N") {
      const next = argv[i + 1];
      if (next == null || Number.isNaN(Number(next))) {
        throw new Error("--recurrence-window requires a numeric argument (default 30)");
      }
      recurrenceWindow = Math.max(1, Math.floor(Number(next)));
      i++;
    } else if (a === "--repo-root") {
      const next = argv[i + 1];
      if (next == null) throw new Error("--repo-root requires a path");
      repoRoot = next;
      i++;
    } else if (a === "--runs-dir") {
      const next = argv[i + 1];
      if (next == null) throw new Error("--runs-dir requires a path");
      runsDir = next;
      i++;
    } else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`unknown flag: ${a}`);
    }
  }
  return { json, recurrenceWindow, strict, repoRoot, runsDir };
};

const printHelp = (): void => {
  // eslint-disable-next-line no-console
  console.log(`bag_tactics_audit.ts

Lists every tactic under src/prompts/tactics/ with its forensic pointer,
introduction date, and review date. Flags tactics whose review_by has
passed; suggests retirement candidates whose incident hasn't recurred in
the last N runs.

Flags:
  --json                       emit machine-readable JSON instead of a table
  --recurrence-window N        recurrence window in days (default 30)
  --runs-dir PATH              where to scan runs (default <repo>/.bag/runs)
  --repo-root PATH             override repo root (default: this script's parent)
  --strict                     exit non-zero if any tactic is overdue
  -h, --help                   print this message
`);
};

/**
 * Walk recent BAG runs and count, per tactic id, how many runs have a
 * verifier transcript / forensic note that mentions the tactic id or its
 * trigger keywords. Best-effort: missing runs dir → all hits = 0 (we
 * don't fail the audit for a clean checkout).
 */
const recurrenceHitsByTactic = (
  runsDir: string,
  windowDays: number,
  tactics: Tactic[],
): Map<string, number> => {
  const out = new Map<string, number>();
  for (const t of tactics) out.set(t.id, 0);
  if (!existsSync(runsDir)) return out;
  let runEntries: string[];
  try {
    runEntries = readdirSync(runsDir);
  } catch {
    return out;
  }
  const cutoff = today();
  cutoff.setUTCDate(cutoff.getUTCDate() - windowDays);
  const cutoffMs = cutoff.getTime();
  for (const entry of runEntries) {
    const runPath = join(runsDir, entry);
    let st: ReturnType<typeof statSync>;
    try {
      st = statSync(runPath);
    } catch {
      continue;
    }
    if (!st.isDirectory()) continue;
    if (st.mtimeMs < cutoffMs) continue;
    // Scan all *.jsonl, *.json, *.log under the run for tactic mentions.
    let blob = "";
    try {
      blob = collectRunText(runPath);
    } catch {
      continue;
    }
    for (const t of tactics) {
      if (blob.includes(t.id)) {
        out.set(t.id, (out.get(t.id) ?? 0) + 1);
        continue;
      }
      // Soft match by trigger keyword (e.g. "SUBPROCESS-PATH GATE").
      if (
        t.frontmatter.trigger != null &&
        t.frontmatter.trigger.length > 12 &&
        blob.includes(t.frontmatter.trigger)
      ) {
        out.set(t.id, (out.get(t.id) ?? 0) + 1);
      }
    }
  }
  return out;
};

const collectRunText = (runPath: string): string => {
  const parts: string[] = [];
  const stack = [runPath];
  let budget = 8 * 1024 * 1024; // 8 MiB cap per run to keep the audit fast.
  while (stack.length > 0 && budget > 0) {
    const cur = stack.pop()!;
    let entries: string[];
    try {
      entries = readdirSync(cur);
    } catch {
      continue;
    }
    for (const e of entries) {
      const p = join(cur, e);
      let st: ReturnType<typeof statSync>;
      try {
        st = statSync(p);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        stack.push(p);
        continue;
      }
      if (!/\.(jsonl|json|log|txt|md)$/.test(e)) continue;
      const take = Math.min(st.size, budget);
      if (take <= 0) continue;
      try {
        const txt = readFileSync(p, "utf8").slice(0, take);
        parts.push(txt);
        budget -= txt.length;
      } catch {
        continue;
      }
    }
  }
  return parts.join("\n");
};

const buildAudit = (opts: {
  repoRoot: string;
  recurrenceWindow: number;
  runsDir: string | null;
}): AuditRow[] => {
  const tactics = loadAllTactics(opts.repoRoot);
  const runsDir = opts.runsDir ?? join(opts.repoRoot, ".bag", "runs");
  const hits = recurrenceHitsByTactic(runsDir, opts.recurrenceWindow, tactics);
  const now = today();
  const rows: AuditRow[] = tactics.map((t) => {
    const reviewDate = parseDate(t.frontmatter.review_by ?? null);
    const overdue = reviewDate != null && reviewDate.getTime() < now.getTime();
    const daysUntilReview = reviewDate ? daysBetween(now, reviewDate) : null;
    const recurrenceHits = hits.get(t.id) ?? 0;
    const retirementCandidate =
      t.status === "active" && recurrenceHits === 0 && overdue;
    return {
      id: t.id,
      status: t.status,
      order: t.frontmatter.order ?? null,
      incident: t.frontmatter.incident ?? "(unknown)",
      introduced: t.frontmatter.introduced ?? null,
      review_by: t.frontmatter.review_by ?? null,
      trigger: t.frontmatter.trigger ?? null,
      merged_into: t.frontmatter.merged_into ?? null,
      overdue,
      daysUntilReview,
      recurrenceHits,
      retirementCandidate,
      path: t.path,
    };
  });
  rows.sort((a, b) => {
    if (a.status !== b.status) return a.status === "active" ? -1 : 1;
    if ((a.order ?? Number.POSITIVE_INFINITY) !== (b.order ?? Number.POSITIVE_INFINITY)) {
      return (a.order ?? Number.POSITIVE_INFINITY) - (b.order ?? Number.POSITIVE_INFINITY);
    }
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return rows;
};

const renderTable = (rows: AuditRow[], windowDays: number): string => {
  const lines: string[] = [];
  lines.push(`BAG tactics audit — ${rows.length} tactic(s) discovered`);
  lines.push(`recurrence-window: ${windowDays}d`);
  lines.push("");
  lines.push(
    [
      "STATUS".padEnd(11),
      "ID".padEnd(28),
      "INTRO".padEnd(12),
      "REVIEW".padEnd(12),
      "DUE".padEnd(8),
      "HITS".padEnd(6),
      "FLAGS",
    ].join("  "),
  );
  lines.push("-".repeat(96));
  for (const r of rows) {
    const flags: string[] = [];
    if (r.overdue) flags.push("OVERDUE");
    if (r.retirementCandidate) flags.push("RETIRE?");
    if (r.merged_into != null) flags.push(`MERGED→${r.merged_into}`);
    const due =
      r.daysUntilReview == null
        ? "—"
        : r.daysUntilReview < 0
          ? `${-r.daysUntilReview}d ago`
          : `in ${r.daysUntilReview}d`;
    lines.push(
      [
        r.status.padEnd(11),
        r.id.padEnd(28),
        (r.introduced ?? "—").padEnd(12),
        (r.review_by ?? "—").padEnd(12),
        due.padEnd(8),
        String(r.recurrenceHits).padEnd(6),
        flags.join(" ") || "—",
      ].join("  "),
    );
  }
  lines.push("");
  const overdueCount = rows.filter((r) => r.overdue).length;
  const candidateCount = rows.filter((r) => r.retirementCandidate).length;
  lines.push(
    `summary: ${overdueCount} overdue review(s), ${candidateCount} retirement candidate(s)`,
  );
  if (candidateCount > 0) {
    lines.push("");
    lines.push("retirement candidates (overdue + 0 hits in recurrence window):");
    for (const r of rows.filter((r) => r.retirementCandidate)) {
      lines.push(`  - ${r.id}  (last incident: ${r.incident})`);
    }
  }
  return lines.join("\n") + "\n";
};

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));
  const rows = buildAudit({
    repoRoot: args.repoRoot,
    recurrenceWindow: args.recurrenceWindow,
    runsDir: args.runsDir,
  });
  if (args.json) {
    process.stdout.write(JSON.stringify({ tactics: rows, windowDays: args.recurrenceWindow }, null, 2));
    process.stdout.write("\n");
  } else {
    process.stdout.write(renderTable(rows, args.recurrenceWindow));
  }
  if (args.strict && rows.some((r) => r.overdue)) process.exit(2);
};

main();
