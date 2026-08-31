import { describe, expect, test } from "bun:test";
import {
  AcceptedUserCorrectionKnowledgeEntrySchema,
  CommandKnowledgeEntrySchema,
  ConsolidationGroupSchema,
  DecisionKnowledgeEntrySchema,
  KnowledgeEntrySchema,
  KnowledgeSummaryDocumentSchema,
} from "../src/knowledge/types";

const now = "2026-04-30T00:00:00.000Z";

const sourceRef = {
  sourceKind: "file",
  path: "src/workspace.ts",
  lineStart: 1,
  lineEnd: 20,
  observedAt: now,
  excerpt: "loadKnowledge reads project-local knowledge files before fallback docs.",
};

const commandEntry = {
  entryId: "knowledge.command.typecheck",
  kind: "command",
  title: "Run TypeScript typecheck",
  body: "Use the package script when validating source-level TypeScript changes.",
  tags: ["typescript", "verification"],
  confidence: 0.9,
  sourceRefs: [sourceRef],
  dedupeKeys: [
    {
      strategy: "command",
      value: "npm run typecheck",
      contentHash: "sha256:typecheck",
    },
  ],
  createdAt: now,
  updatedAt: now,
  command: ["npm", "run", "typecheck"],
  purpose: "Validate src/**/*.ts without emitting build artifacts.",
};

describe("knowledge schemas", () => {
  test("parse representative command, decision, correction, and summary records", () => {
    const parsedCommand = CommandKnowledgeEntrySchema.parse(commandEntry);
    expect(parsedCommand.schemaVersion).toBe("knowledge-schema.v1");
    expect(parsedCommand.retention.retention).toBe("project");
    expect(parsedCommand.redaction.state).toBe("not_required");

    const decision = DecisionKnowledgeEntrySchema.parse({
      entryId: "knowledge.decision.project-not-model",
      kind: "decision",
      title: "Keep project knowledge separate from optimizer profiles",
      body: "Knowledge records describe repo facts and decisions, not model-specific tool behavior.",
      confidence: 1,
      createdAt: now,
      updatedAt: now,
      decision: "Store codebase facts separately from model/codebase optimizer policy artifacts.",
      rationale: ["Project knowledge should transfer across models.", "Optimizer profiles can remain model-specific."],
      alternativesConsidered: ["Mix knowledge and optimizer guidance into one blob."],
    });
    expect(decision.kind).toBe("decision");

    const correction = AcceptedUserCorrectionKnowledgeEntrySchema.parse({
      entryId: "knowledge.correction.scope",
      kind: "accepted_user_correction",
      title: "Respect declared write scope",
      body: "When a graph lane owns a narrow scope, do not edit unrelated runtime or docs files.",
      acceptedByUser: true,
      confidence: 1,
      createdAt: now,
      updatedAt: now,
      correction: {
        correctionId: "correction.scope.1",
        original: "Edit optimizer modules as needed.",
        corrected: "Edit only the files declared in the lane write scope.",
        acceptedAt: now,
      },
    });
    expect(correction.correction.acceptedBy).toBe("user");

    const summary = KnowledgeSummaryDocumentSchema.parse({
      generatedAt: now,
      sections: [
        {
          sectionId: "summary.commands",
          title: "Commands",
          updatedAt: now,
          items: [
            {
              entryId: parsedCommand.entryId,
              title: parsedCommand.title,
              summary: parsedCommand.body,
              tags: parsedCommand.tags,
            },
          ],
          sourceEntryIds: [parsedCommand.entryId],
        },
      ],
    });
    expect(summary.sections[0]?.items[0]?.entryId).toBe("knowledge.command.typecheck");
  });

  test("parse dedupe and consolidation records", () => {
    const group = ConsolidationGroupSchema.parse({
      consolidationGroupId: "knowledge.group.typecheck",
      status: "consolidated",
      primaryEntryId: "knowledge.command.typecheck",
      memberEntryIds: ["knowledge.command.typecheck", "knowledge.command.tsc"],
      dedupeKeys: [
        {
          strategy: "normalized_text",
          value: "run typescript typecheck",
          generatedAt: now,
        },
      ],
      summary: "Merged duplicate typecheck command learnings into one command entry.",
      createdAt: now,
      updatedAt: now,
    });

    expect(group.memberEntryIds).toHaveLength(2);
    expect(group.dedupeKeys[0]?.strategy).toBe("normalized_text");
  });

  test("rejects secret-looking unredacted content and accepts explicit redaction metadata", () => {
    const unredacted = KnowledgeEntrySchema.safeParse({
      ...commandEntry,
      body: "Never store password=supersecretvalue12345 in project knowledge.",
    });
    expect(unredacted.success).toBe(false);

    const redacted = KnowledgeEntrySchema.parse({
      ...commandEntry,
      body: "Never store password=[REDACTED] in project knowledge.",
      redaction: {
        state: "redacted",
        redactionKinds: ["password"],
        replacementCount: 1,
        originalContentHash: "sha256:redacted-original",
        redactedAt: now,
        redactedBy: "agent",
      },
    });
    expect(redacted.redaction.state).toBe("redacted");
  });

  test("rejects strict schema failures", () => {
    expect(KnowledgeEntrySchema.safeParse({
      ...commandEntry,
      confidence: 1.5,
    }).success).toBe(false);

    expect(KnowledgeEntrySchema.safeParse({
      ...commandEntry,
      optimizerProfileId: "model.qwen36.local",
    }).success).toBe(false);

    expect(KnowledgeEntrySchema.safeParse({
      ...commandEntry,
      sourceRefs: [
        {
          ...sourceRef,
          lineEnd: 1,
          lineStart: 2,
        },
      ],
    }).success).toBe(false);
  });
});
