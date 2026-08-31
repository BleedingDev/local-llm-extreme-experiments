import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  ConsolidationGroupSchema,
  KnowledgeEntrySchema,
  KnowledgeSummaryDocumentSchema,
  type ConsolidationGroup,
  type KnowledgeEntry,
  type KnowledgeSummaryDocument,
  type KnowledgeSummarySection,
} from "./types";

export type KnowledgeStoreOptions = {
  cwd?: string;
  artifactDir?: string;
};

export type KnowledgeStorePaths = {
  rootDir: string;
  entriesPath: string;
  consolidationGroupsPath: string;
  summaryPath: string;
};

export type InvalidKnowledgeRow = {
  lineNumber: number;
  raw: string;
  reason: "invalid_json" | "invalid_schema";
  error: string;
};

export type KnowledgeEntriesReadResult = {
  path: string;
  entries: KnowledgeEntry[];
  invalidRows: InvalidKnowledgeRow[];
};

export type ConsolidationGroupsReadResult = {
  path: string;
  groups: ConsolidationGroup[];
  invalidRows: InvalidKnowledgeRow[];
};

export type KnowledgeConsolidationOptions = {
  generatedAt?: string;
};

export type KnowledgeConsolidationResult = {
  entries: KnowledgeEntry[];
  groups: ConsolidationGroup[];
  supersededEntryIds: string[];
  preservedCorrectionEntryIds: string[];
};

const DEFAULT_ARTIFACT_DIR = ".bag";
const KNOWLEDGE_DIR = "knowledge";
const ENTRIES_FILE = "entries.jsonl";
const CONSOLIDATION_GROUPS_FILE = "consolidation-groups.jsonl";
const SUMMARY_FILE = "AI.md";

const nowIso = (): string => new Date().toISOString();

const messageForError = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

export const resolveKnowledgeStorePaths = (options: KnowledgeStoreOptions = {}): KnowledgeStorePaths => {
  const cwd = options.cwd ?? process.cwd();
  const artifactDir = options.artifactDir ?? DEFAULT_ARTIFACT_DIR;
  const rootDir = resolve(cwd, artifactDir, KNOWLEDGE_DIR);

  return {
    rootDir,
    entriesPath: resolve(rootDir, ENTRIES_FILE),
    consolidationGroupsPath: resolve(rootDir, CONSOLIDATION_GROUPS_FILE),
    summaryPath: resolve(rootDir, SUMMARY_FILE),
  };
};

const ensureParentDir = (path: string): void => {
  mkdirSync(dirname(path), { recursive: true });
};

const appendJsonl = (path: string, value: unknown): void => {
  ensureParentDir(path);
  appendFileSync(path, `${JSON.stringify(value)}\n`);
};

export const appendKnowledgeEntry = (
  entry: KnowledgeEntry,
  options: KnowledgeStoreOptions = {},
): KnowledgeEntry => {
  const parsed = KnowledgeEntrySchema.parse(entry);
  appendJsonl(resolveKnowledgeStorePaths(options).entriesPath, parsed);
  return parsed;
};

export const readKnowledgeEntries = (options: KnowledgeStoreOptions = {}): KnowledgeEntriesReadResult => {
  const path = resolveKnowledgeStorePaths(options).entriesPath;
  const entries: KnowledgeEntry[] = [];
  const invalidRows: InvalidKnowledgeRow[] = [];

  if (!existsSync(path)) {
    return { path, entries, invalidRows };
  }

  readFileSync(path, "utf8").split("\n").forEach((raw, index) => {
    if (raw.trim() === "") {
      return;
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(raw) as unknown;
    } catch (error) {
      invalidRows.push({
        lineNumber: index + 1,
        raw,
        reason: "invalid_json",
        error: messageForError(error),
      });
      return;
    }

    const parsedEntry = KnowledgeEntrySchema.safeParse(parsedJson);
    if (!parsedEntry.success) {
      invalidRows.push({
        lineNumber: index + 1,
        raw,
        reason: "invalid_schema",
        error: parsedEntry.error.message,
      });
      return;
    }

    entries.push(parsedEntry.data);
  });

  return { path, entries, invalidRows };
};

export const appendConsolidationGroup = (
  group: ConsolidationGroup,
  options: KnowledgeStoreOptions = {},
): ConsolidationGroup => {
  const parsed = ConsolidationGroupSchema.parse(group);
  appendJsonl(resolveKnowledgeStorePaths(options).consolidationGroupsPath, parsed);
  return parsed;
};

export const readConsolidationGroups = (
  options: KnowledgeStoreOptions = {},
): ConsolidationGroupsReadResult => {
  const path = resolveKnowledgeStorePaths(options).consolidationGroupsPath;
  const groups: ConsolidationGroup[] = [];
  const invalidRows: InvalidKnowledgeRow[] = [];

  if (!existsSync(path)) {
    return { path, groups, invalidRows };
  }

  readFileSync(path, "utf8").split("\n").forEach((raw, index) => {
    if (raw.trim() === "") {
      return;
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(raw) as unknown;
    } catch (error) {
      invalidRows.push({
        lineNumber: index + 1,
        raw,
        reason: "invalid_json",
        error: messageForError(error),
      });
      return;
    }

    const parsedGroup = ConsolidationGroupSchema.safeParse(parsedJson);
    if (!parsedGroup.success) {
      invalidRows.push({
        lineNumber: index + 1,
        raw,
        reason: "invalid_schema",
        error: parsedGroup.error.message,
      });
      return;
    }

    groups.push(parsedGroup.data);
  });

  return { path, groups, invalidRows };
};

const normalizeDedupeValue = (value: string): string => value.toLowerCase().replace(/\s+/g, " ").trim();

const dedupeIdentitiesForEntry = (entry: KnowledgeEntry): string[] => [
  ...new Set(entry.dedupeKeys.map((key) => `${key.strategy}:${normalizeDedupeValue(key.value)}`)),
];

const isProtectedCorrection = (entry: KnowledgeEntry): boolean =>
  entry.kind === "accepted_user_correction" || entry.acceptedByUser;

const primaryEntryForGroup = (entries: KnowledgeEntry[]): KnowledgeEntry =>
  [...entries].sort((left, right) => {
    const correctionCompare = Number(isProtectedCorrection(right)) - Number(isProtectedCorrection(left));
    if (correctionCompare !== 0) {
      return correctionCompare;
    }

    const activeCompare = Number(right.status === "active") - Number(left.status === "active");
    if (activeCompare !== 0) {
      return activeCompare;
    }

    if (right.confidence !== left.confidence) {
      return right.confidence - left.confidence;
    }

    const updatedCompare = right.updatedAt.localeCompare(left.updatedAt);
    if (updatedCompare !== 0) {
      return updatedCompare;
    }

    return left.entryId.localeCompare(right.entryId);
  })[0]!;

const groupIdForDedupeIdentity = (identity: string): string =>
  `knowledge.group.${identity.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80)}`;

export const consolidateKnowledgeEntries = (
  entries: KnowledgeEntry[],
  options: KnowledgeConsolidationOptions = {},
): KnowledgeConsolidationResult => {
  const generatedAt = options.generatedAt ?? nowIso();
  const parsedEntries = entries.map((entry) => KnowledgeEntrySchema.parse(entry));
  const groupsByIdentity = new Map<string, KnowledgeEntry[]>();

  for (const entry of parsedEntries) {
    for (const identity of dedupeIdentitiesForEntry(entry)) {
      const group = groupsByIdentity.get(identity) ?? [];
      group.push(entry);
      groupsByIdentity.set(identity, group);
    }
  }

  const entriesById = new Map(parsedEntries.map((entry) => [entry.entryId, entry]));
  const groups: ConsolidationGroup[] = [];
  const supersededEntryIds = new Set<string>();
  const preservedCorrectionEntryIds = new Set<string>();

  for (const [identity, matchingEntries] of [...groupsByIdentity.entries()].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    const uniqueEntries = [...new Map(matchingEntries.map((entry) => [entry.entryId, entry])).values()];
    if (uniqueEntries.length < 2) {
      continue;
    }

    const primary = primaryEntryForGroup(uniqueEntries);
    const memberEntryIds = uniqueEntries.map((entry) => entry.entryId).sort();
    const groupId = groupIdForDedupeIdentity(identity);

    groups.push(ConsolidationGroupSchema.parse({
      consolidationGroupId: groupId,
      status: "consolidated",
      primaryEntryId: primary.entryId,
      memberEntryIds,
      dedupeKeys: primary.dedupeKeys,
      summary: isProtectedCorrection(primary)
        ? `Preserved accepted correction: ${primary.summary ?? primary.body}`
        : `Consolidated duplicate knowledge entries into ${primary.entryId}.`,
      rationale: isProtectedCorrection(primary)
        ? "Accepted user corrections are authoritative project memory and must not be overwritten by generic summaries."
        : "Entries shared a dedupe key and the primary entry had the strongest active/confidence/update ordering.",
      createdAt: generatedAt,
      updatedAt: generatedAt,
    }));

    for (const entry of uniqueEntries) {
      if (isProtectedCorrection(entry)) {
        preservedCorrectionEntryIds.add(entry.entryId);
      }

      if (entry.entryId === primary.entryId || isProtectedCorrection(entry)) {
        entriesById.set(entry.entryId, {
          ...entry,
          consolidationGroupId: groupId,
          updatedAt: entry.updatedAt,
        });
        continue;
      }

      supersededEntryIds.add(entry.entryId);
      entriesById.set(entry.entryId, {
        ...entry,
        status: "superseded",
        consolidationGroupId: groupId,
        updatedAt: generatedAt,
      });
    }
  }

  return {
    entries: parsedEntries.map((entry) => KnowledgeEntrySchema.parse(entriesById.get(entry.entryId) ?? entry)),
    groups,
    supersededEntryIds: [...supersededEntryIds].sort(),
    preservedCorrectionEntryIds: [...preservedCorrectionEntryIds].sort(),
  };
};

const sectionLabels: Record<KnowledgeEntry["kind"], { sectionId: string; title: string; purpose: string }> = {
  command: {
    sectionId: "summary.commands",
    title: "Commands",
    purpose: "Useful project-local commands and when to run them.",
  },
  convention: {
    sectionId: "summary.conventions",
    title: "Conventions",
    purpose: "Repository conventions that should guide future changes.",
  },
  gotcha: {
    sectionId: "summary.gotchas",
    title: "Gotchas",
    purpose: "Known failure modes and mitigations.",
  },
  decision: {
    sectionId: "summary.decisions",
    title: "Decisions",
    purpose: "Project decisions that remain relevant for future work.",
  },
  fact: {
    sectionId: "summary.facts",
    title: "Facts",
    purpose: "Stable facts about this codebase.",
  },
  accepted_user_correction: {
    sectionId: "summary.accepted-user-corrections",
    title: "Accepted User Corrections",
    purpose: "User corrections that should be preserved as project knowledge.",
  },
};

const summaryTextForEntry = (entry: KnowledgeEntry): string => entry.summary ?? entry.body;

export const buildKnowledgeSummaryDocument = (
  entries: KnowledgeEntry[],
  generatedAt = nowIso(),
): KnowledgeSummaryDocument => {
  const validatedEntries = entries.map((entry) => KnowledgeEntrySchema.parse(entry));
  const sections: KnowledgeSummarySection[] = [];

  for (const kind of Object.keys(sectionLabels) as KnowledgeEntry["kind"][]) {
    const label = sectionLabels[kind];
    const items = validatedEntries
      .filter((entry) => entry.kind === kind && entry.status === "active")
      .map((entry) => ({
        entryId: entry.entryId,
        title: entry.title,
        summary: summaryTextForEntry(entry),
        tags: entry.tags,
      }));

    if (items.length === 0) {
      continue;
    }

    sections.push({
      sectionId: label.sectionId,
      title: label.title,
      purpose: label.purpose,
      items,
      sourceEntryIds: items.map((item) => item.entryId),
      updatedAt: generatedAt,
    });
  }

  return KnowledgeSummaryDocumentSchema.parse({
    generatedAt,
    sections,
  });
};

export const renderKnowledgeSummaryDocument = (document: KnowledgeSummaryDocument): string => {
  const parsed = KnowledgeSummaryDocumentSchema.parse(document);
  const lines = [
    "# BleedingAgent Project Knowledge",
    "",
    `Generated: ${parsed.generatedAt}`,
    "",
    "This file is generated from `.bag/knowledge/entries.jsonl`.",
  ];

  if (parsed.sections.length === 0) {
    lines.push("", "No active project knowledge entries have been recorded yet.");
    return `${lines.join("\n")}\n`;
  }

  for (const section of parsed.sections) {
    lines.push("", `## ${section.title}`);
    if (section.purpose != null) {
      lines.push("", section.purpose);
    }

    for (const item of section.items) {
      const tags = item.tags.length > 0 ? ` [${item.tags.join(", ")}]` : "";
      lines.push("", `- **${item.title}** (${item.entryId})${tags}`, `  ${item.summary}`);
    }
  }

  return `${lines.join("\n")}\n`;
};

export const writeKnowledgeSummaryDocument = (
  document: KnowledgeSummaryDocument,
  options: KnowledgeStoreOptions = {},
): string => {
  const path = resolveKnowledgeStorePaths(options).summaryPath;
  const markdown = renderKnowledgeSummaryDocument(document);
  ensureParentDir(path);
  writeFileSync(path, markdown);
  return path;
};

export const writeKnowledgeSummaryFromEntries = (
  entries: KnowledgeEntry[],
  options: KnowledgeStoreOptions = {},
  generatedAt = nowIso(),
): string => writeKnowledgeSummaryDocument(buildKnowledgeSummaryDocument(entries, generatedAt), options);

export const readKnowledgeSummaryMarkdown = (options: KnowledgeStoreOptions = {}): string | undefined => {
  const path = resolveKnowledgeStorePaths(options).summaryPath;
  if (!existsSync(path)) {
    return undefined;
  }
  return readFileSync(path, "utf8");
};
