import { describe, expect, test } from "bun:test";
import { appendFileSync, existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  appendConsolidationGroup,
  appendKnowledgeEntry,
  buildKnowledgeSummaryDocument,
  consolidateKnowledgeEntries,
  readConsolidationGroups,
  readKnowledgeEntries,
  readKnowledgeSummaryMarkdown,
  renderKnowledgeSummaryDocument,
  resolveKnowledgeStorePaths,
  writeKnowledgeSummaryFromEntries,
} from "../src/knowledge/store";
import type { ConsolidationGroup, KnowledgeEntry } from "../src/knowledge/types";

const now = "2026-04-30T00:00:00.000Z";

const tempProject = (): string => mkdtempSync(join(tmpdir(), "bag-knowledge-store-"));

const factEntry = (entryId = "knowledge.fact.store"): KnowledgeEntry => ({
  entryId,
  schemaVersion: "knowledge-schema.v1",
  kind: "fact",
  status: "active",
  title: "Knowledge store uses project-local files",
  body: "Knowledge entries are stored below .bag/knowledge for the current project.",
  tags: ["knowledge", "storage"],
  confidence: 0.95,
  retention: { retention: "project" },
  sourceRefs: [],
  dedupeKeys: [],
  createdAt: now,
  updatedAt: now,
  acceptedByUser: false,
  redaction: {
    state: "not_required",
    redactionKinds: [],
    replacementCount: 0,
  },
  subject: "knowledge store",
  statement: "Entries live in .bag/knowledge/entries.jsonl.",
  affectedPaths: ["src/knowledge/store.ts"],
});

const correctionEntry = (entryId = "knowledge.correction.store"): KnowledgeEntry => ({
  entryId,
  schemaVersion: "knowledge-schema.v1",
  kind: "accepted_user_correction",
  status: "active",
  title: "Use entries.jsonl for project knowledge",
  body: "Replace generic memory summary with the accepted user correction.",
  summary: "User correction: entries belong in .bag/knowledge/entries.jsonl.",
  tags: ["user_correction", "knowledge"],
  confidence: 1,
  retention: { retention: "project" },
  sourceRefs: [],
  dedupeKeys: [
    {
      strategy: "normalized_text",
      value: "knowledge store location",
      generatedAt: now,
    },
  ],
  createdAt: now,
  updatedAt: now,
  acceptedByUser: true,
  redaction: {
    state: "not_required",
    redactionKinds: [],
    replacementCount: 0,
  },
  correction: {
    correctionId: "correction.store.location",
    original: "Knowledge can be summarized anywhere under .bag.",
    corrected: "Knowledge entries belong in .bag/knowledge/entries.jsonl.",
    acceptedAt: now,
    acceptedBy: "user",
    appliesToEntryIds: [],
    sourceRefs: [],
    redaction: {
      state: "not_required",
      redactionKinds: [],
      replacementCount: 0,
    },
  },
});

describe("knowledge store", () => {
  test("resolves project-local paths and reads an empty store", () => {
    const cwd = tempProject();
    const paths = resolveKnowledgeStorePaths({ cwd });
    const entries = readKnowledgeEntries({ cwd });
    const groups = readConsolidationGroups({ cwd });

    expect(paths.rootDir).toBe(join(cwd, ".bag", "knowledge"));
    expect(paths.entriesPath).toBe(join(cwd, ".bag", "knowledge", "entries.jsonl"));
    expect(paths.summaryPath).toBe(join(cwd, ".bag", "knowledge", "AI.md"));
    expect(entries.entries).toEqual([]);
    expect(entries.invalidRows).toEqual([]);
    expect(groups.groups).toEqual([]);
    expect(groups.invalidRows).toEqual([]);
    expect(readKnowledgeSummaryMarkdown({ cwd })).toBeUndefined();
  });

  test("appends and reads validated entries and consolidation groups", () => {
    const cwd = tempProject();
    const entry = appendKnowledgeEntry(factEntry(), { cwd });
    const group: ConsolidationGroup = {
      consolidationGroupId: "knowledge.group.store",
      status: "open",
      memberEntryIds: [entry.entryId],
      dedupeKeys: [],
      summary: "Group for project-local knowledge store entries.",
      createdAt: now,
      updatedAt: now,
    };

    appendConsolidationGroup(group, { cwd });

    const entries = readKnowledgeEntries({ cwd });
    const groups = readConsolidationGroups({ cwd });

    expect(entries.invalidRows).toEqual([]);
    expect(entries.entries).toHaveLength(1);
    expect(entries.entries[0]?.entryId).toBe("knowledge.fact.store");
    expect(entries.entries[0]?.schemaVersion).toBe("knowledge-schema.v1");
    expect(groups.invalidRows).toEqual([]);
    expect(groups.groups[0]?.consolidationGroupId).toBe("knowledge.group.store");
  });

  test("reports invalid JSON and schema rows without crashing reads", () => {
    const cwd = tempProject();
    const paths = resolveKnowledgeStorePaths({ cwd });

    appendKnowledgeEntry(factEntry("knowledge.fact.valid"), { cwd });
    appendFileSync(paths.entriesPath, "{not json}\n");
    appendFileSync(paths.entriesPath, `${JSON.stringify({ entryId: "knowledge.fact.invalid", kind: "fact" })}\n`);

    const result = readKnowledgeEntries({ cwd });

    expect(result.entries.map((entry) => entry.entryId)).toEqual(["knowledge.fact.valid"]);
    expect(result.invalidRows).toHaveLength(2);
    expect(result.invalidRows[0]?.reason).toBe("invalid_json");
    expect(result.invalidRows[0]?.lineNumber).toBe(2);
    expect(result.invalidRows[1]?.reason).toBe("invalid_schema");
    expect(result.invalidRows[1]?.lineNumber).toBe(3);
  });

  test("renders and writes a human-readable AI.md summary", () => {
    const cwd = tempProject();
    const entry = factEntry();
    const document = buildKnowledgeSummaryDocument([entry], now);
    const markdown = renderKnowledgeSummaryDocument(document);
    const path = writeKnowledgeSummaryFromEntries([entry], { cwd }, now);

    expect(document.schemaVersion).toBe("knowledge-schema.v1");
    expect(markdown).toContain("# BleedingAgent Project Knowledge");
    expect(markdown).toContain("## Facts");
    expect(markdown).toContain("Knowledge store uses project-local files");
    expect(path).toBe(join(cwd, ".bag", "knowledge", "AI.md"));
    expect(existsSync(path)).toBe(true);
    expect(readFileSync(path, "utf8")).toBe(markdown);
    expect(readKnowledgeSummaryMarkdown({ cwd })).toBe(markdown);
  });

  test("consolidates duplicate knowledge while preserving accepted user corrections", () => {
    const generic = {
      ...factEntry("knowledge.fact.generic-store"),
      title: "Generic store location summary",
      body: "Knowledge store location was discussed.",
      summary: "Generic summary of the knowledge store location.",
      dedupeKeys: [
        {
          strategy: "normalized_text" as const,
          value: "knowledge store location",
          generatedAt: now,
        },
      ],
    };
    const correction = correctionEntry();
    const result = consolidateKnowledgeEntries([generic, correction], { generatedAt: "2026-04-30T01:00:00.000Z" });

    expect(result.groups).toHaveLength(1);
    expect(result.groups[0]).toMatchObject({
      primaryEntryId: correction.entryId,
      status: "consolidated",
    });
    expect(result.groups[0]?.summary).toContain("Preserved accepted correction");
    expect(result.supersededEntryIds).toEqual([generic.entryId]);
    expect(result.preservedCorrectionEntryIds).toEqual([correction.entryId]);

    const consolidatedGeneric = result.entries.find((entry) => entry.entryId === generic.entryId);
    const consolidatedCorrection = result.entries.find((entry) => entry.entryId === correction.entryId);
    expect(consolidatedGeneric?.status).toBe("superseded");
    expect(consolidatedCorrection).toMatchObject({
      status: "active",
      summary: "User correction: entries belong in .bag/knowledge/entries.jsonl.",
      acceptedByUser: true,
    });

    const document = buildKnowledgeSummaryDocument(result.entries, now);
    const correctionSection = document.sections.find((section) => section.sectionId === "summary.accepted-user-corrections");
    expect(correctionSection?.items[0]?.summary).toBe("User correction: entries belong in .bag/knowledge/entries.jsonl.");
    expect(document.sections.some((section) =>
      section.items.some((item) => item.entryId === generic.entryId),
    )).toBe(false);
  });
});
