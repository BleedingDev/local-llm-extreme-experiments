import { z } from "zod";

const SECRET_LIKE_PATTERNS = [
  /\bsk-[A-Za-z0-9_-]{20,}\b/,
  /\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)\b\s*[:=]\s*["']?[A-Za-z0-9_./+=-]{12,}/i,
  /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/,
  /-----BEGIN (?:RSA |EC |OPENSSH |PRIVATE )?PRIVATE KEY-----/,
];

const DEFAULT_REDACTION_METADATA = {
  state: "not_required" as const,
  redactionKinds: [],
  replacementCount: 0,
};

export const hasSecretLikeContent = (value: string): boolean =>
  SECRET_LIKE_PATTERNS.some((pattern) => pattern.test(value));

export const KnowledgeIdSchema = z.string().min(1).regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/);
export type KnowledgeId = z.infer<typeof KnowledgeIdSchema>;

export const KnowledgeSchemaVersionSchema = z.literal("knowledge-schema.v1").default("knowledge-schema.v1");
export type KnowledgeSchemaVersion = z.infer<typeof KnowledgeSchemaVersionSchema>;

export const KnowledgeTagSchema = z.string().min(1).max(64).regex(/^[a-z0-9][a-z0-9._:-]*$/);
export type KnowledgeTag = z.infer<typeof KnowledgeTagSchema>;

export const ConfidenceSchema = z.number().min(0).max(1);
export type Confidence = z.infer<typeof ConfidenceSchema>;

export const RedactionKindSchema = z.enum([
  "api_key",
  "access_token",
  "password",
  "private_key",
  "personal_data",
  "repository_secret",
  "other",
]);
export type RedactionKind = z.infer<typeof RedactionKindSchema>;

export const RedactionMetadataSchema = z.object({
  state: z.enum(["not_required", "redacted"]).default("not_required"),
  redactionKinds: z.array(RedactionKindSchema).default([]),
  replacementCount: z.number().int().nonnegative().default(0),
  originalContentHash: z.string().min(1).optional(),
  redactedAt: z.string().min(1).optional(),
  redactedBy: z.enum(["agent", "user", "importer"]).optional(),
}).strict();
export type RedactionMetadata = z.infer<typeof RedactionMetadataSchema>;

export const KnowledgeSourceKindSchema = z.enum([
  "user_correction",
  "work_summary",
  "review",
  "eval_failure",
  "trace",
  "file",
  "command",
  "documentation",
  "manual",
]);
export type KnowledgeSourceKind = z.infer<typeof KnowledgeSourceKindSchema>;

export const KnowledgeSourceRefSchema = z.object({
  sourceRefId: KnowledgeIdSchema.optional(),
  sourceKind: KnowledgeSourceKindSchema,
  title: z.string().min(1).optional(),
  uri: z.string().min(1).optional(),
  path: z.string().min(1).optional(),
  lineStart: z.number().int().positive().optional(),
  lineEnd: z.number().int().positive().optional(),
  traceId: z.string().min(1).optional(),
  spanId: z.string().min(1).optional(),
  command: z.array(z.string().min(1)).optional(),
  excerpt: z.string().min(1).optional(),
  observedAt: z.string().min(1),
  contentHash: z.string().min(1).optional(),
  redaction: RedactionMetadataSchema.default(DEFAULT_REDACTION_METADATA),
}).strict().superRefine((sourceRef, context) => {
  if (sourceRef.lineStart != null && sourceRef.lineEnd != null && sourceRef.lineEnd < sourceRef.lineStart) {
    context.addIssue({
      code: "custom",
      message: "lineEnd must be greater than or equal to lineStart",
      path: ["lineEnd"],
    });
  }

  if (sourceRef.excerpt != null && sourceRef.redaction.state !== "redacted" && hasSecretLikeContent(sourceRef.excerpt)) {
    context.addIssue({
      code: "custom",
      message: "source excerpt appears to contain an unredacted secret",
      path: ["excerpt"],
    });
  }
});
export type KnowledgeSourceRef = z.infer<typeof KnowledgeSourceRefSchema>;

export const RetentionPolicySchema = z.object({
  retention: z.enum(["ephemeral", "short_term", "project", "long_term", "pinned"]).default("project"),
  reviewAfter: z.string().min(1).optional(),
  expiresAt: z.string().min(1).optional(),
  reason: z.string().min(1).optional(),
}).strict();
export type RetentionPolicy = z.infer<typeof RetentionPolicySchema>;

export const DedupeKeySchema = z.object({
  keyId: KnowledgeIdSchema.optional(),
  strategy: z.enum(["exact", "normalized_text", "semantic", "source_ref", "command"]),
  value: z.string().min(1),
  contentHash: z.string().min(1).optional(),
  generatedAt: z.string().min(1).optional(),
}).strict();
export type DedupeKey = z.infer<typeof DedupeKeySchema>;

const CommonKnowledgeEntryFields = {
  entryId: KnowledgeIdSchema,
  schemaVersion: KnowledgeSchemaVersionSchema,
  status: z.enum(["candidate", "active", "superseded", "rejected", "archived"]).default("active"),
  title: z.string().min(1).max(200),
  body: z.string().min(1),
  summary: z.string().min(1).optional(),
  tags: z.array(KnowledgeTagSchema).default([]),
  confidence: ConfidenceSchema.default(0.6),
  retention: RetentionPolicySchema.default({ retention: "project" }),
  sourceRefs: z.array(KnowledgeSourceRefSchema).default([]),
  dedupeKeys: z.array(DedupeKeySchema).default([]),
  consolidationGroupId: KnowledgeIdSchema.optional(),
  createdAt: z.string().min(1),
  updatedAt: z.string().min(1),
  acceptedByUser: z.boolean().default(false),
  redaction: RedactionMetadataSchema.default(DEFAULT_REDACTION_METADATA),
} as const;

export const CommandKnowledgeEntrySchema = z.object({
  ...CommonKnowledgeEntryFields,
  kind: z.literal("command"),
  command: z.array(z.string().min(1)).min(1),
  cwd: z.string().min(1).optional(),
  purpose: z.string().min(1),
  whenToUse: z.string().min(1).optional(),
  expectedOutcome: z.string().min(1).optional(),
  verification: z.enum(["manual", "automated", "informational"]).default("automated"),
}).strict();
export type CommandKnowledgeEntry = z.infer<typeof CommandKnowledgeEntrySchema>;

export const ConventionKnowledgeEntrySchema = z.object({
  ...CommonKnowledgeEntryFields,
  kind: z.literal("convention"),
  scope: z.string().min(1),
  rule: z.string().min(1),
  rationale: z.string().min(1).optional(),
  examples: z.array(z.string().min(1)).default([]),
}).strict();
export type ConventionKnowledgeEntry = z.infer<typeof ConventionKnowledgeEntrySchema>;

export const GotchaKnowledgeEntrySchema = z.object({
  ...CommonKnowledgeEntryFields,
  kind: z.literal("gotcha"),
  severity: z.enum(["low", "medium", "high"]).default("medium"),
  symptom: z.string().min(1),
  cause: z.string().min(1).optional(),
  mitigation: z.string().min(1),
  affectedPaths: z.array(z.string().min(1)).default([]),
}).strict();
export type GotchaKnowledgeEntry = z.infer<typeof GotchaKnowledgeEntrySchema>;

export const DecisionKnowledgeEntrySchema = z.object({
  ...CommonKnowledgeEntryFields,
  kind: z.literal("decision"),
  decision: z.string().min(1),
  rationale: z.array(z.string().min(1)).default([]),
  alternativesConsidered: z.array(z.string().min(1)).default([]),
  decidedAt: z.string().min(1).optional(),
  supersedesEntryIds: z.array(KnowledgeIdSchema).default([]),
}).strict();
export type DecisionKnowledgeEntry = z.infer<typeof DecisionKnowledgeEntrySchema>;

export const FactKnowledgeEntrySchema = z.object({
  ...CommonKnowledgeEntryFields,
  kind: z.literal("fact"),
  subject: z.string().min(1),
  statement: z.string().min(1),
  affectedPaths: z.array(z.string().min(1)).default([]),
}).strict();
export type FactKnowledgeEntry = z.infer<typeof FactKnowledgeEntrySchema>;

export const AcceptedUserCorrectionSchema = z.object({
  correctionId: KnowledgeIdSchema,
  original: z.string().min(1),
  corrected: z.string().min(1),
  acceptedAt: z.string().min(1),
  acceptedBy: z.enum(["user", "maintainer"]).default("user"),
  appliesToEntryIds: z.array(KnowledgeIdSchema).default([]),
  sourceRefs: z.array(KnowledgeSourceRefSchema).default([]),
  redaction: RedactionMetadataSchema.default(DEFAULT_REDACTION_METADATA),
}).strict().superRefine((correction, context) => {
  if (correction.redaction.state !== "redacted") {
    for (const [field, value] of Object.entries({ original: correction.original, corrected: correction.corrected })) {
      if (hasSecretLikeContent(value)) {
        context.addIssue({
          code: "custom",
          message: "accepted correction appears to contain an unredacted secret",
          path: [field],
        });
      }
    }
  }
});
export type AcceptedUserCorrection = z.infer<typeof AcceptedUserCorrectionSchema>;

export const AcceptedUserCorrectionKnowledgeEntrySchema = z.object({
  ...CommonKnowledgeEntryFields,
  kind: z.literal("accepted_user_correction"),
  correction: AcceptedUserCorrectionSchema,
}).strict();
export type AcceptedUserCorrectionKnowledgeEntry = z.infer<typeof AcceptedUserCorrectionKnowledgeEntrySchema>;

export const KnowledgeEntrySchema = z.discriminatedUnion("kind", [
  CommandKnowledgeEntrySchema,
  ConventionKnowledgeEntrySchema,
  GotchaKnowledgeEntrySchema,
  DecisionKnowledgeEntrySchema,
  FactKnowledgeEntrySchema,
  AcceptedUserCorrectionKnowledgeEntrySchema,
]).superRefine((entry, context) => {
  const text = [
    entry.title,
    entry.body,
    entry.summary,
    entry.kind === "command" ? [entry.purpose, entry.whenToUse, entry.expectedOutcome, entry.command.join(" ")] : undefined,
    entry.kind === "convention" ? [entry.scope, entry.rule, entry.rationale, ...entry.examples] : undefined,
    entry.kind === "gotcha" ? [entry.symptom, entry.cause, entry.mitigation, ...entry.affectedPaths] : undefined,
    entry.kind === "decision" ? [entry.decision, ...entry.rationale, ...entry.alternativesConsidered] : undefined,
    entry.kind === "fact" ? [entry.subject, entry.statement, ...entry.affectedPaths] : undefined,
  ].flat(2).filter((value): value is string => typeof value === "string");

  if (entry.redaction.state !== "redacted" && text.some(hasSecretLikeContent)) {
    context.addIssue({
      code: "custom",
      message: "knowledge entry appears to contain an unredacted secret",
      path: ["redaction"],
    });
  }
});
export type KnowledgeEntry = z.infer<typeof KnowledgeEntrySchema>;

export const ConsolidationGroupSchema = z.object({
  consolidationGroupId: KnowledgeIdSchema,
  status: z.enum(["open", "consolidated", "superseded", "rejected"]).default("open"),
  primaryEntryId: KnowledgeIdSchema.optional(),
  memberEntryIds: z.array(KnowledgeIdSchema).min(1),
  dedupeKeys: z.array(DedupeKeySchema).default([]),
  summary: z.string().min(1),
  rationale: z.string().min(1).optional(),
  createdAt: z.string().min(1),
  updatedAt: z.string().min(1),
}).strict();
export type ConsolidationGroup = z.infer<typeof ConsolidationGroupSchema>;

export const KnowledgeSummaryItemSchema = z.object({
  entryId: KnowledgeIdSchema,
  title: z.string().min(1),
  summary: z.string().min(1),
  tags: z.array(KnowledgeTagSchema).default([]),
}).strict();
export type KnowledgeSummaryItem = z.infer<typeof KnowledgeSummaryItemSchema>;

export const KnowledgeSummarySectionSchema = z.object({
  sectionId: KnowledgeIdSchema,
  title: z.string().min(1),
  purpose: z.string().min(1).optional(),
  items: z.array(KnowledgeSummaryItemSchema).default([]),
  sourceEntryIds: z.array(KnowledgeIdSchema).default([]),
  updatedAt: z.string().min(1),
}).strict();
export type KnowledgeSummarySection = z.infer<typeof KnowledgeSummarySectionSchema>;

export const KnowledgeSummaryDocumentSchema = z.object({
  schemaVersion: KnowledgeSchemaVersionSchema,
  generatedAt: z.string().min(1),
  sections: z.array(KnowledgeSummarySectionSchema).default([]),
}).strict();
export type KnowledgeSummaryDocument = z.infer<typeof KnowledgeSummaryDocumentSchema>;
