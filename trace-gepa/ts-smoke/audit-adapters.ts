import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  detectSourceJsonl,
  detectSourceRecords,
  sourceAdapters,
  type SourceAdapterType,
  type SourceDetectionResult,
} from "../../src/source-adapters/boundary";
import { canonicalizeSourceRecords } from "../../src/source-adapters/canonical";
import { canonicalizeCcSessionV2 } from "../../src/source-adapters/cc-session-v2";

type AuditCase = {
  label: string;
  path: string;
  expectedAdapter: SourceAdapterType;
  sampleLines: number;
};

const CASES: AuditCase[] = [
  {
    label: "claude-code (cc405b87)",
    path: join(
      homedir(),
      ".claude/projects/-Users-satan-side-experiments-supergemma-dflash-ddtree-mlx/cc405b87-4ce5-4ac5-bb3f-cb19d3a3b6d0.jsonl",
    ),
    expectedAdapter: "acp-session-jsonl",
    sampleLines: 50,
  },
  {
    label: "claude-code-v2 (cc405b87)",
    path: join(
      homedir(),
      ".claude/projects/-Users-satan-side-experiments-supergemma-dflash-ddtree-mlx/cc405b87-4ce5-4ac5-bb3f-cb19d3a3b6d0.jsonl",
    ),
    expectedAdapter: "cc-session-jsonl-v2" as SourceAdapterType,
    sampleLines: 50,
  },
  {
    label: "codex (rollout-019d542e)",
    path: "/Users/satan/.codex/sessions/2026/04/03/rollout-2026-04-03T23-30-43-019d542e-c1fb-7ed1-b8a2-2d3f44cdc7c3.jsonl",
    expectedAdapter: "codex-session-jsonl",
    sampleLines: 50,
  },
];

const readNLines = (path: string, n: number): string => {
  const raw = readFileSync(path, "utf8");
  return raw.split(/\r?\n/).slice(0, n).filter((l) => l.trim().length > 0).join("\n");
};

const summarize = (label: string, expected: SourceAdapterType, jsonl: string) => {
  console.log(`\n=== ${label} (expected adapter: ${expected}) ===`);
  const detected = detectSourceJsonl(jsonl, { path: label });
  console.log(`detect.ok=${detected.ok}`);
  if (!detected.ok) {
    console.log(`detect.diagnostics=${JSON.stringify(detected.diagnostics, null, 2)}`);
  } else {
    console.log(`detect.sourceType=${detected.source.sourceType}`);
    console.log(`detect.signals=${JSON.stringify(detected.source.detectedSignals)}`);
    console.log(`detect.sessionId=${detected.source.sessionId ?? "<none>"}`);
    console.log(`detect.schemaVersion=${detected.source.schemaVersion ?? "<none>"}`);
  }

  // Try forced detection against expected adapter
  const records: unknown[] = jsonl
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0)
    .map((l) => {
      try {
        return JSON.parse(l);
      } catch {
        return null;
      }
    })
    .filter((r) => r != null);
  console.log(`parsed.records=${records.length}`);

  const forced = detectSourceRecords(records, { path: label, maxInspectionRecords: records.length }, expected);
  console.log(`force(${expected}).ok=${forced.ok}`);
  if (!forced.ok) {
    console.log(`force.diagnostics=${JSON.stringify(forced.diagnostics, null, 2)}`);
  }

  // Tally raw record types in the sample
  const typeCounts = new Map<string, number>();
  for (const rec of records) {
    if (typeof rec === "object" && rec != null && !Array.isArray(rec)) {
      const t = (rec as Record<string, unknown>).type;
      const k = typeof t === "string" ? t : "<no-type>";
      typeCounts.set(k, (typeCounts.get(k) ?? 0) + 1);
    }
  }
  console.log(`raw.type counts=${JSON.stringify(Object.fromEntries(typeCounts))}`);

  // If we managed any detection, run the canonicalizer.
  let usedSource: SourceDetectionResult = forced.ok ? forced : detected;
  if (usedSource.ok) {
    const sourceMeta = usedSource.source;
    const canon = sourceMeta.sourceType === "cc-session-jsonl-v2"
      ? canonicalizeCcSessionV2({ source: sourceMeta, records })
      : canonicalizeSourceRecords({ source: sourceMeta, records });
    console.log(`canonical.spans=${canon.records.length}`);
    console.log(`canonical.diagnostics.count=${canon.diagnostics.length}`);
    const diagByCode = new Map<string, number>();
    const diagByRecordType = new Map<string, number>();
    for (const d of canon.diagnostics) {
      diagByCode.set(d.code, (diagByCode.get(d.code) ?? 0) + 1);
      const rt = d.recordType ?? "<unknown>";
      diagByRecordType.set(rt, (diagByRecordType.get(rt) ?? 0) + 1);
    }
    console.log(`canonical.diagnostics.byCode=${JSON.stringify(Object.fromEntries(diagByCode))}`);
    console.log(`canonical.diagnostics.byRecordType=${JSON.stringify(Object.fromEntries(diagByRecordType))}`);

    const eventKindCounts = new Map<string, number>();
    const obsKindCounts = new Map<string, number>();
    for (const r of canon.records) {
      const ek = r.span.attributes["source.adapter.event_kind"];
      const ok = r.span.attributes["inference.observation_kind"];
      eventKindCounts.set(String(ek), (eventKindCounts.get(String(ek)) ?? 0) + 1);
      obsKindCounts.set(String(ok), (obsKindCounts.get(String(ok)) ?? 0) + 1);
    }
    console.log(`canonical.event_kinds=${JSON.stringify(Object.fromEntries(eventKindCounts))}`);
    console.log(`canonical.observation_kinds=${JSON.stringify(Object.fromEntries(obsKindCounts))}`);

    if (canon.records[0]) {
      const first = canon.records[0];
      console.log(
        `first.span={name:${first.span.name}, ek:${first.span.attributes["source.adapter.event_kind"]}, ok:${first.span.attributes["inference.observation_kind"]}, status:${first.span.status.code}}`,
      );
    }
  } else {
    console.log("skipping canonicalize: no successful detection");
  }
};

console.log(`registered adapters: ${sourceAdapters.map((a) => a.sourceType).join(", ")}`);

for (const c of CASES) {
  try {
    const sample = readNLines(c.path, c.sampleLines);
    summarize(c.label, c.expectedAdapter, sample);
  } catch (e) {
    console.error(`FAIL ${c.label}: ${e instanceof Error ? e.message : String(e)}`);
  }
}
