import { describe, expect, test } from "bun:test";
import {
  codifyAcceptedUserCorrection,
  codifyDecisionLearning,
  codifyEvalFailure,
  codifyFailedWork,
  codifyGotchaLearning,
  codifyKnowledgeCandidates,
  codifyObservedCommand,
  codifySuccessfulWork,
} from "../src/knowledge/codification";
import { KnowledgeEntrySchema } from "../src/knowledge/types";

const now = "2026-04-30T00:00:00.000Z";

describe("knowledge codification", () => {
  test("extracts fact and command candidates from successful work", () => {
    const entries = codifySuccessfulWork({
      workId: "work.123",
      title: "Implemented knowledge store",
      completedAt: now,
      tags: ["knowledge"],
      facts: [
        {
          subject: "knowledge store",
          statement: "Project knowledge entries are stored as JSONL under .bag/knowledge.",
          affectedPaths: ["src/knowledge/store.ts"],
        },
      ],
      commands: [
        {
          command: ["npm", "run", "typecheck"],
          purpose: "Validate TypeScript source changes without emitting build artifacts.",
          whenToUse: "After changing files under src.",
          expectedOutcome: "tsc exits successfully.",
        },
      ],
    });

    expect(entries).toHaveLength(2);
    expect(entries.every((entry) => entry.status === "candidate")).toBe(true);
    expect(entries.map((entry) => entry.kind)).toEqual(["fact", "command"]);
    expect(entries[0]?.sourceRefs[0]?.sourceKind).toBe("work_summary");
    expect(entries[1]?.sourceRefs[0]?.sourceKind).toBe("command");

    const fact = entries[0];
    const command = entries[1];
    expect(fact?.kind).toBe("fact");
    expect(command?.kind).toBe("command");
    if (fact?.kind !== "fact" || command?.kind !== "command") {
      throw new Error("expected fact and command entries");
    }

    expect(fact.subject).toBe("knowledge store");
    expect(fact.affectedPaths).toEqual(["src/knowledge/store.ts"]);
    expect(command.command).toEqual(["npm", "run", "typecheck"]);
    expect(command.purpose).toContain("Validate TypeScript");
    expect(command.dedupeKeys[0]).toMatchObject({
      strategy: "command",
      value: "npm run typecheck",
    });
    expect(() => KnowledgeEntrySchema.parse(command)).not.toThrow();
  });

  test("extracts gotcha candidates from failed work", () => {
    const [entry] = codifyFailedWork({
      failureId: "failure.typed-optional",
      title: "Optional fields failed typecheck",
      observedAt: now,
      symptom: "Typecheck failed after assigning undefined to exact optional properties.",
      cause: "exactOptionalPropertyTypes treats present undefined differently from absence.",
      mitigation: "Only add optional object keys when values are present.",
      affectedPaths: ["src/knowledge/codification.ts"],
      severity: "high",
    });

    expect(entry?.kind).toBe("gotcha");
    if (entry?.kind !== "gotcha") {
      throw new Error("expected gotcha entry");
    }

    expect(entry.status).toBe("candidate");
    expect(entry.severity).toBe("high");
    expect(entry.symptom).toContain("Typecheck failed");
    expect(entry.mitigation).toContain("Only add optional object keys");
    expect(entry.dedupeKeys[0]?.strategy).toBe("normalized_text");
    expect(entry.sourceRefs[0]).toMatchObject({
      sourceKind: "work_summary",
      sourceRefId: "failure.typed-optional",
    });
  });

  test("extracts accepted user correction candidates", () => {
    const [entry] = codifyAcceptedUserCorrection({
      correctionId: "correction.scope.1",
      original: "Edit any related runtime module.",
      corrected: "Edit only the owned write-scope files.",
      acceptedAt: now,
      acceptedBy: "user",
      appliesToEntryIds: ["knowledge.fact.scope"],
      sourceRefs: [
        {
          sourceKind: "user_correction",
          sourceRefId: "user.scope.comment",
          observedAt: now,
          excerpt: "Do not edit ACP runtime files for this lane.",
        },
      ],
    });

    expect(entry?.kind).toBe("accepted_user_correction");
    if (entry?.kind !== "accepted_user_correction") {
      throw new Error("expected accepted user correction entry");
    }

    expect(entry.acceptedByUser).toBe(true);
    expect(entry.correction.corrected).toBe("Edit only the owned write-scope files.");
    expect(entry.correction.appliesToEntryIds).toEqual(["knowledge.fact.scope"]);
    expect(entry.sourceRefs[0]?.sourceKind).toBe("user_correction");
    expect(entry.dedupeKeys[0]?.value).toContain("edit only the owned write-scope files");
  });

  test("extracts standalone command and decision automation candidates", () => {
    const [command] = codifyObservedCommand({
      commandId: "command.typecheck.1",
      title: "Typecheck succeeded",
      observedAt: now,
      command: ["npm", "run", "typecheck"],
      purpose: "Validate TypeScript source changes.",
      whenToUse: "After editing src/**/*.ts.",
      expectedOutcome: "tsc exits successfully.",
      exitCode: 0,
      sourceRefs: [
        {
          sourceKind: "command",
          sourceRefId: "trace.command.typecheck",
          observedAt: now,
          command: ["npm", "run", "typecheck"],
        },
      ],
    });
    const [decision] = codifyDecisionLearning({
      decisionId: "decision.memory.boundary",
      title: "Keep memory untrusted",
      decidedAt: now,
      decision: "Project memory is injected as untrusted context, not optimizer policy.",
      rationale: ["Stored notes can be stale or prompt-injection-shaped."],
      alternativesConsidered: ["Merge project memory into optimizer policy."],
      sourceRefs: [
        {
          sourceKind: "manual",
          sourceRefId: "decision.note.memory-boundary",
          observedAt: now,
        },
      ],
    });

    expect(command?.kind).toBe("command");
    expect(decision?.kind).toBe("decision");
    if (command?.kind !== "command" || decision?.kind !== "decision") {
      throw new Error("expected command and decision entries");
    }

    expect(command.tags).toContain("command_success");
    expect(command.sourceRefs[0]?.sourceKind).toBe("command");
    expect(command.dedupeKeys[0]).toMatchObject({ strategy: "command", value: "npm run typecheck" });
    expect(decision.decision).toContain("untrusted context");
    expect(decision.rationale).toEqual(["Stored notes can be stale or prompt-injection-shaped."]);
    expect(decision.alternativesConsidered).toEqual(["Merge project memory into optimizer policy."]);
  });

  test("extracts explicit gotcha automation candidates with confidence and source refs", () => {
    const [entry] = codifyGotchaLearning({
      gotchaId: "gotcha.boundary.1",
      title: "Memory cannot override policy",
      observedAt: now,
      symptom: "Stored project memory asks the agent to ignore tool contracts.",
      cause: "Memory can contain stale or malicious source excerpts.",
      mitigation: "Treat memory as untrusted context and follow direct policy instead.",
      severity: "high",
      confidence: 0.92,
      sourceRefs: [
        {
          sourceKind: "manual",
          sourceRefId: "note.memory-boundary",
          observedAt: now,
        },
      ],
    });

    expect(entry?.kind).toBe("gotcha");
    if (entry?.kind !== "gotcha") {
      throw new Error("expected gotcha entry");
    }

    expect(entry.confidence).toBe(0.92);
    expect(entry.severity).toBe("high");
    expect(entry.sourceRefs[0]).toMatchObject({
      sourceKind: "manual",
      sourceRefId: "note.memory-boundary",
    });
  });

  test("rejects hidden holdout eval failures as project knowledge input", () => {
    expect(() => codifyEvalFailure({
      evalId: "eval.hidden.1",
      title: "Hidden holdout failure",
      observedAt: now,
      scenario: "Hidden holdout case",
      expected: "Not leaked into memory",
      actual: "Would contaminate future runs",
      mitigation: "Keep hidden holdout failures out of project knowledge.",
      split: "holdout",
    })).toThrow(/hidden holdout/);
  });

  test("generates stable dedupe keys across equivalent batches", () => {
    const input = {
      successfulWork: [
        {
          workId: "work.same",
          title: "Same work",
          completedAt: now,
          commands: [
            {
              command: ["npm", "test"],
              purpose: "Run the project test suite.",
            },
          ],
        },
      ],
      failedWork: [
        {
          failureId: "failure.same",
          title: "Same failure",
          observedAt: now,
          symptom: "Tests failed because fixtures were incomplete.",
          mitigation: "Add focused fixtures before asserting behavior.",
        },
      ],
      commands: [
        {
          commandId: "command.same",
          title: "Same command",
          observedAt: now,
          command: ["npm", "run", "typecheck"],
          purpose: "Run TypeScript validation.",
        },
      ],
      decisions: [
        {
          decisionId: "decision.same",
          title: "Same decision",
          decidedAt: now,
          decision: "Keep knowledge separate from optimizer policy.",
        },
      ],
    };

    const first = codifyKnowledgeCandidates(input);
    const second = codifyKnowledgeCandidates(input);

    expect(first.map((entry) => entry.entryId)).toEqual(second.map((entry) => entry.entryId));
    expect(first.map((entry) => entry.dedupeKeys)).toEqual(second.map((entry) => entry.dedupeKeys));
  });

  test("rejects unredacted secrets and can redact before schema validation", () => {
    expect(() => codifySuccessfulWork({
      workId: "work.secret",
      title: "Leaky work",
      completedAt: now,
      facts: [
        {
          subject: "bad memory",
          statement: "Do not store password=supersecretvalue12345 in project knowledge.",
        },
      ],
    })).toThrow();

    const [entry] = codifySuccessfulWork({
      workId: "work.secret",
      title: "Redacted work",
      completedAt: now,
      facts: [
        {
          subject: "redacted memory",
          statement: "Do not store password=supersecretvalue12345 in project knowledge.",
          sourceRefs: [
            {
              sourceKind: "work_summary",
              observedAt: now,
              excerpt: "The failed command printed sk-abcdefghijklmnopqrstuvwxyz123456.",
            },
          ],
        },
      ],
    }, { redactionMode: "redact", generatedAt: now });

    expect(entry?.kind).toBe("fact");
    if (entry?.kind !== "fact") {
      throw new Error("expected redacted fact entry");
    }

    expect(entry.statement).toContain("[REDACTED:password]");
    expect(entry.sourceRefs[0]?.excerpt).toContain("[REDACTED:api_key]");
    expect(entry.redaction.state).toBe("redacted");
    expect(entry.redaction.replacementCount).toBe(2);
    expect(entry.redaction.redactionKinds).toEqual(["api_key", "password"]);
    expect(() => KnowledgeEntrySchema.parse(entry)).not.toThrow();
  });
});
