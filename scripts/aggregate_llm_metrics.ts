#!/usr/bin/env -S node --loader=tsx
/**
 * Aggregate `events.jsonl` LLM call metrics into a per-(model, purpose, role)
 * breakdown so we can identify per-model optimisation targets without baking
 * any model-name knowledge into BAG core.
 *
 * Usage:
 *   tsx scripts/aggregate_llm_metrics.ts <events.jsonl path...>
 *   tsx scripts/aggregate_llm_metrics.ts --harbor-job bench/jobs/<RUN>
 *
 * Reads each `{type: "llm.call", payload: {role, model, purpose, ...}}`
 * record. Emits JSON of the form:
 *   {
 *     totals: { calls, in, out, ms, errors },
 *     byPurpose: { "<purpose>": { calls, in, out, p50_ms, p95_ms, errors } },
 *     byModel:   { "<model>":   { calls, in, out, p50_ms, p95_ms, errors } },
 *     byPurposeModel: { "<purpose>::<model>": {...} },
 *     untaggedCalls: number,   // calls without purpose — fix the source
 *   }
 *
 * The untaggedCalls counter is the canary: every BAG LLM call should declare
 * a `purpose`. A non-zero count after a run means a call site was added
 * without telemetry attribution.
 */

import { readFileSync, readdirSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

type Record_ = {
  role?: string;
  model?: string;
  purpose?: string;
  durationMs?: number;
  promptTokens?: number;
  completionTokens?: number;
  ok?: boolean;
};

type BucketStats = {
  calls: number;
  in: number;
  out: number;
  errors: number;
  durations: number[];
};

const newBucket = (): BucketStats => ({ calls: 0, in: 0, out: 0, errors: 0, durations: [] });

const accumulate = (bucket: BucketStats, record: Record_): void => {
  bucket.calls += 1;
  bucket.in += record.promptTokens ?? 0;
  bucket.out += record.completionTokens ?? 0;
  if (record.ok === false) bucket.errors += 1;
  if (typeof record.durationMs === "number") bucket.durations.push(record.durationMs);
};

const percentile = (values: number[], p: number): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * (sorted.length - 1)));
  return sorted[idx] ?? 0;
};

const finalize = (bucket: BucketStats) => ({
  calls: bucket.calls,
  in: bucket.in,
  out: bucket.out,
  errors: bucket.errors,
  p50_ms: percentile(bucket.durations, 50),
  p95_ms: percentile(bucket.durations, 95),
});

const findEventFiles = (path: string): string[] => {
  if (!existsSync(path)) return [];
  const stat = statSync(path);
  if (stat.isFile()) return [path];
  if (!stat.isDirectory()) return [];
  // Walk recursively, picking up any `events.jsonl` under the directory.
  const out: string[] = [];
  const stack: string[] = [path];
  while (stack.length > 0) {
    const dir = stack.pop()!;
    let entries: Array<{ name: string; isDirectory: () => boolean; isFile: () => boolean }>;
    try {
      entries = readdirSync(dir, { withFileTypes: true }) as Array<{
        name: string;
        isDirectory: () => boolean;
        isFile: () => boolean;
      }>;
    } catch {
      continue;
    }
    for (const entry of entries) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.isFile() && entry.name === "events.jsonl") {
        out.push(full);
      }
    }
  }
  return out;
};

const main = (): void => {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    process.stderr.write(
      "usage: aggregate_llm_metrics.ts <events.jsonl|run-dir>...\n",
    );
    process.exit(2);
  }

  const totals = newBucket();
  const byPurpose = new Map<string, BucketStats>();
  const byModel = new Map<string, BucketStats>();
  const byPurposeModel = new Map<string, BucketStats>();
  let untaggedCalls = 0;
  let parsedFiles = 0;
  let parsedRecords = 0;

  const files = args.flatMap((a) => findEventFiles(a));
  for (const file of files) {
    let raw: string;
    try {
      raw = readFileSync(file, "utf8");
    } catch {
      continue;
    }
    parsedFiles += 1;
    for (const line of raw.split(/\r?\n/)) {
      if (line.trim().length === 0) continue;
      let parsed: { type?: string; payload?: Record_ };
      try {
        parsed = JSON.parse(line);
      } catch {
        continue;
      }
      if (parsed.type !== "llm.call" || parsed.payload == null) continue;
      const payload = parsed.payload;
      parsedRecords += 1;
      accumulate(totals, payload);
      const purpose = payload.purpose ?? "<untagged>";
      if (payload.purpose == null) untaggedCalls += 1;
      const model = payload.model ?? "<unknown>";
      const purposeBucket = byPurpose.get(purpose) ?? newBucket();
      accumulate(purposeBucket, payload);
      byPurpose.set(purpose, purposeBucket);
      const modelBucket = byModel.get(model) ?? newBucket();
      accumulate(modelBucket, payload);
      byModel.set(model, modelBucket);
      const pmKey = `${purpose}::${model}`;
      const pmBucket = byPurposeModel.get(pmKey) ?? newBucket();
      accumulate(pmBucket, payload);
      byPurposeModel.set(pmKey, pmBucket);
    }
  }

  const mapToObject = <T>(m: Map<string, BucketStats>, fn: (b: BucketStats) => T): Record<string, T> => {
    const entries = [...m.entries()].map(([k, v]) => [k, fn(v)] as const);
    entries.sort((a, b) => a[0].localeCompare(b[0]));
    return Object.fromEntries(entries);
  };

  const report = {
    parsedFiles,
    parsedRecords,
    untaggedCalls,
    totals: finalize(totals),
    byPurpose: mapToObject(byPurpose, finalize),
    byModel: mapToObject(byModel, finalize),
    byPurposeModel: mapToObject(byPurposeModel, finalize),
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
};

main();
