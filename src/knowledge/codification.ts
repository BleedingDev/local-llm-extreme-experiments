import { createHash } from "node:crypto";
import {
  KnowledgeEntrySchema,
  hasSecretLikeContent,
  type AcceptedUserCorrection,
  type DedupeKey,
  type KnowledgeEntry,
  type KnowledgeSourceKind,
  type KnowledgeSourceRef,
  type RedactionKind,
  type RedactionMetadata,
} from "./types";

type RedactionMode = "reject" | "redact";

export type CodificationOptions = {
  generatedAt?: string | undefined;
  redactionMode?: RedactionMode | undefined;
};

export type CodificationSourceRefInput = {
  sourceRefId?: string | undefined;
  sourceKind: KnowledgeSourceKind;
  title?: string | undefined;
  uri?: string | undefined;
  path?: string | undefined;
  lineStart?: number | undefined;
  lineEnd?: number | undefined;
  traceId?: string | undefined;
  spanId?: string | undefined;
  command?: string[] | undefined;
  excerpt?: string | undefined;
  observedAt?: string | undefined;
  contentHash?: string | undefined;
  redaction?: RedactionMetadata | undefined;
};

export type SuccessfulWorkFact = {
  subject: string;
  statement: string;
  affectedPaths?: string[] | undefined;
  confidence?: number | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type SuccessfulWorkCommand = {
  command: string[];
  purpose: string;
  cwd?: string | undefined;
  whenToUse?: string | undefined;
  expectedOutcome?: string | undefined;
  confidence?: number | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type SuccessfulWorkSummary = {
  workId: string;
  title: string;
  completedAt: string;
  tags?: string[] | undefined;
  facts?: SuccessfulWorkFact[] | undefined;
  commands?: SuccessfulWorkCommand[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type FailedWorkSummary = {
  failureId: string;
  title: string;
  observedAt: string;
  symptom: string;
  mitigation: string;
  cause?: string | undefined;
  severity?: "low" | "medium" | "high" | undefined;
  affectedPaths?: string[] | undefined;
  tags?: string[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type ReviewLearningSummary = {
  reviewId: string;
  title: string;
  observedAt: string;
  finding: string;
  recommendation: string;
  severity?: "low" | "medium" | "high" | undefined;
  affectedPaths?: string[] | undefined;
  tags?: string[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type EvalFailureSummary = {
  evalId: string;
  title: string;
  observedAt: string;
  scenario: string;
  expected: string;
  actual: string;
  mitigation: string;
  rootCause?: string | undefined;
  severity?: "low" | "medium" | "high" | undefined;
  affectedPaths?: string[] | undefined;
  tags?: string[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
  split?: "train" | "validation" | "holdout" | undefined;
  hiddenHoldout?: boolean | undefined;
};

export type AcceptedUserCorrectionInput = {
  correctionId: string;
  original: string;
  corrected: string;
  acceptedAt: string;
  acceptedBy?: "user" | "maintainer" | undefined;
  appliesToEntryIds?: string[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
  redaction?: RedactionMetadata | undefined;
  title?: string | undefined;
  tags?: string[] | undefined;
};

export type ObservedCommandSummary = {
  commandId: string;
  title: string;
  observedAt: string;
  command: string[];
  purpose: string;
  cwd?: string | undefined;
  whenToUse?: string | undefined;
  expectedOutcome?: string | undefined;
  exitCode?: number | null | undefined;
  confidence?: number | undefined;
  tags?: string[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type GotchaLearningSummary = {
  gotchaId: string;
  title: string;
  observedAt: string;
  symptom: string;
  mitigation: string;
  cause?: string | undefined;
  severity?: "low" | "medium" | "high" | undefined;
  affectedPaths?: string[] | undefined;
  confidence?: number | undefined;
  tags?: string[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type DecisionLearningSummary = {
  decisionId: string;
  title: string;
  decidedAt: string;
  decision: string;
  rationale?: string[] | undefined;
  alternativesConsidered?: string[] | undefined;
  supersedesEntryIds?: string[] | undefined;
  confidence?: number | undefined;
  tags?: string[] | undefined;
  sourceRefs?: CodificationSourceRefInput[] | undefined;
};

export type KnowledgeCodificationInput = {
  successfulWork?: SuccessfulWorkSummary[] | undefined;
  failedWork?: FailedWorkSummary[] | undefined;
  reviews?: ReviewLearningSummary[] | undefined;
  evalFailures?: EvalFailureSummary[] | undefined;
  corrections?: AcceptedUserCorrectionInput[] | undefined;
  commands?: ObservedCommandSummary[] | undefined;
  gotchas?: GotchaLearningSummary[] | undefined;
  decisions?: DecisionLearningSummary[] | undefined;
};

const DEFAULT_REDACTION_METADATA: RedactionMetadata = {
  state: "not_required",
  redactionKinds: [],
  replacementCount: 0,
};

const SECRET_REDACTION_RULES: Array<{
  pattern: RegExp;
  kind: RedactionKind;
  replace: (match: string, ...groups: string[]) => string;
}> = [
  {
    pattern: /\bsk-[A-Za-z0-9_-]{20,}\b/g,
    kind: "api_key",
    replace: () => "[REDACTED:api_key]",
  },
  {
    pattern: /\b((?:api[_-]?key|secret)\b\s*[:=]\s*["']?)[A-Za-z0-9_./+=-]{12,}/gi,
    kind: "api_key",
    replace: (_match, prefix: string) => `${prefix}[REDACTED:api_key]`,
  },
  {
    pattern: /\b((?:access[_-]?token|auth[_-]?token)\b\s*[:=]\s*["']?)[A-Za-z0-9_./+=-]{12,}/gi,
    kind: "access_token",
    replace: (_match, prefix: string) => `${prefix}[REDACTED:access_token]`,
  },
  {
    pattern: /\b(password\b\s*[:=]\s*["']?)[A-Za-z0-9_./+=-]{12,}/gi,
    kind: "password",
    replace: (_match, prefix: string) => `${prefix}[REDACTED:password]`,
  },
  {
    pattern: /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g,
    kind: "repository_secret",
    replace: () => "[REDACTED:repository_secret]",
  },
  {
    pattern: /-----BEGIN (?:RSA |EC |OPENSSH |PRIVATE )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |PRIVATE )?PRIVATE KEY-----/g,
    kind: "private_key",
    replace: () => "[REDACTED:private_key]",
  },
];

type RedactionAccumulator = {
  originalParts: string[];
  redactionKinds: Set<RedactionKind>;
  replacementCount: number;
};

const sha256 = (value: string): string => `sha256:${createHash("sha256").update(value).digest("hex")}`;

const normalizeText = (value: string): string => value.toLowerCase().replace(/[^a-z0-9._:/-]+/g, " ").trim();

const compactTags = (tags: string[] | undefined): string[] =>
  [
    ...new Set(
      (tags ?? [])
        .map((tag) => normalizeText(tag).replace(/[^a-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, ""))
        .filter((tag) => tag !== ""),
    ),
  ].sort();

const compactStrings = (values: string[] | undefined): string[] =>
  [...new Set((values ?? []).map((value) => value.trim()).filter((value) => value !== ""))].sort((left, right) =>
    left.localeCompare(right),
  );

const safeIdPart = (value: string): string => {
  const normalized = normalizeText(value).replace(/[^a-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "");
  return normalized === "" ? "entry" : normalized.slice(0, 48);
};

const entryId = (kind: KnowledgeEntry["kind"], sourceId: string, body: string): string =>
  `knowledge.${kind}.${safeIdPart(sourceId)}.${sha256(body).slice(7, 19)}`;

const dedupeKey = (strategy: DedupeKey["strategy"], value: string, generatedAt: string): DedupeKey => ({
  strategy,
  value,
  contentHash: sha256(value),
  generatedAt,
});

const emptyRedactionAccumulator = (): RedactionAccumulator => ({
  originalParts: [],
  redactionKinds: new Set(),
  replacementCount: 0,
});

const mergeRedaction = (target: RedactionAccumulator, source: RedactionAccumulator): void => {
  target.originalParts.push(...source.originalParts);
  for (const kind of source.redactionKinds) {
    target.redactionKinds.add(kind);
  }
  target.replacementCount += source.replacementCount;
};

const redactionMetadata = (accumulator: RedactionAccumulator, generatedAt: string): RedactionMetadata => {
  if (accumulator.replacementCount === 0) {
    return DEFAULT_REDACTION_METADATA;
  }

  return {
    state: "redacted",
    redactionKinds: [...accumulator.redactionKinds].sort(),
    replacementCount: accumulator.replacementCount,
    originalContentHash: sha256(accumulator.originalParts.join("\n")),
    redactedAt: generatedAt,
    redactedBy: "agent",
  };
};

const redactText = (value: string, mode: RedactionMode): { value: string; redaction: RedactionAccumulator } => {
  const redaction = emptyRedactionAccumulator();
  if (mode === "reject" || !hasSecretLikeContent(value)) {
    return { value, redaction };
  }

  let redacted = value;
  for (const rule of SECRET_REDACTION_RULES) {
    redacted = redacted.replace(rule.pattern, (match, ...groups: string[]) => {
      redaction.originalParts.push(match);
      redaction.redactionKinds.add(rule.kind);
      redaction.replacementCount += 1;
      return rule.replace(match, ...groups);
    });
  }

  return { value: redacted, redaction };
};

const redactStringArray = (
  values: string[] | undefined,
  mode: RedactionMode,
): { values: string[]; redaction: RedactionAccumulator } => {
  const redaction = emptyRedactionAccumulator();
  const redactedValues = (values ?? []).map((value) => {
    const redacted = redactText(value, mode);
    mergeRedaction(redaction, redacted.redaction);
    return redacted.value;
  });

  return { values: redactedValues, redaction };
};

const commonSourceRef = (
  input: CodificationSourceRefInput,
  fallbackObservedAt: string,
  mode: RedactionMode,
): { sourceRef: KnowledgeSourceRef; redaction: RedactionAccumulator } => {
  const redaction = emptyRedactionAccumulator();
  const sourceRef: KnowledgeSourceRef = {
    sourceKind: input.sourceKind,
    observedAt: input.observedAt ?? fallbackObservedAt,
    redaction: input.redaction ?? DEFAULT_REDACTION_METADATA,
  };

  const setText = <Key extends keyof KnowledgeSourceRef>(key: Key, value: string | undefined): void => {
    if (value == null) {
      return;
    }
    const redacted = redactText(value, mode);
    mergeRedaction(redaction, redacted.redaction);
    Object.assign(sourceRef, { [key]: redacted.value });
  };

  setText("sourceRefId", input.sourceRefId);
  setText("title", input.title);
  setText("uri", input.uri);
  setText("path", input.path);
  setText("traceId", input.traceId);
  setText("spanId", input.spanId);
  setText("excerpt", input.excerpt);
  setText("contentHash", input.contentHash);

  if (input.lineStart != null) {
    sourceRef.lineStart = input.lineStart;
  }
  if (input.lineEnd != null) {
    sourceRef.lineEnd = input.lineEnd;
  }
  if (input.command != null) {
    const redactedCommand = redactStringArray(input.command, mode);
    mergeRedaction(redaction, redactedCommand.redaction);
    sourceRef.command = redactedCommand.values;
  }

  if (redaction.replacementCount > 0 && input.redaction == null) {
    sourceRef.redaction = redactionMetadata(redaction, fallbackObservedAt);
  }

  return { sourceRef, redaction };
};

const commonSourceRefs = (
  inputs: CodificationSourceRefInput[] | undefined,
  fallbackObservedAt: string,
  mode: RedactionMode,
): { sourceRefs: KnowledgeSourceRef[]; redaction: RedactionAccumulator } => {
  const redaction = emptyRedactionAccumulator();
  const sourceRefs = (inputs ?? []).map((input) => {
    const sourceRef = commonSourceRef(input, fallbackObservedAt, mode);
    mergeRedaction(redaction, sourceRef.redaction);
    return sourceRef.sourceRef;
  });

  return { sourceRefs, redaction };
};

const parseEntry = (entry: KnowledgeEntry): KnowledgeEntry => KnowledgeEntrySchema.parse(entry);

const defaultSourceRef = (
  sourceKind: KnowledgeSourceKind,
  sourceRefId: string,
  title: string,
  observedAt: string,
): CodificationSourceRefInput => ({
  sourceKind,
  sourceRefId,
  title,
  observedAt,
});

const commandEntry = (
  sourceId: string,
  title: string,
  observedAt: string,
  input: SuccessfulWorkCommand,
  tags: string[],
  options: CodificationOptions,
): KnowledgeEntry => {
  const generatedAt = options.generatedAt ?? observedAt;
  const mode = options.redactionMode ?? "reject";
  const redaction = emptyRedactionAccumulator();
  const commandText = redactStringArray(input.command, mode);
  const purpose = redactText(input.purpose, mode);
  const whenToUse = input.whenToUse == null ? undefined : redactText(input.whenToUse, mode);
  const expectedOutcome = input.expectedOutcome == null ? undefined : redactText(input.expectedOutcome, mode);
  const cwd = input.cwd == null ? undefined : redactText(input.cwd, mode);
  const sourceRefs = commonSourceRefs(
    [
      ...(input.sourceRefs ?? []),
      {
        ...defaultSourceRef("command", `${sourceId}.command`, title, observedAt),
        command: commandText.values,
      },
    ],
    observedAt,
    mode,
  );
  mergeRedaction(redaction, commandText.redaction);
  mergeRedaction(redaction, purpose.redaction);
  if (whenToUse != null) {
    mergeRedaction(redaction, whenToUse.redaction);
  }
  if (expectedOutcome != null) {
    mergeRedaction(redaction, expectedOutcome.redaction);
  }
  if (cwd != null) {
    mergeRedaction(redaction, cwd.redaction);
  }
  mergeRedaction(redaction, sourceRefs.redaction);

  const commandValue = commandText.values.join(" ");
  const dedupeValue = normalizeText(commandValue);
  return parseEntry({
    entryId: entryId("command", sourceId, `command:${dedupeValue}`),
    schemaVersion: "knowledge-schema.v1",
    kind: "command",
    status: "candidate",
    title: `Run ${commandValue}`.slice(0, 200),
    body: purpose.value,
    summary: purpose.value,
    tags: compactTags([...tags, "command"]),
    confidence: input.confidence ?? 0.8,
    retention: { retention: "project" },
    sourceRefs: sourceRefs.sourceRefs,
    dedupeKeys: [dedupeKey("command", dedupeValue, generatedAt)],
    createdAt: generatedAt,
    updatedAt: generatedAt,
    acceptedByUser: false,
    redaction: redactionMetadata(redaction, generatedAt),
    command: commandText.values,
    ...(cwd == null ? {} : { cwd: cwd.value }),
    purpose: purpose.value,
    ...(whenToUse == null ? {} : { whenToUse: whenToUse.value }),
    ...(expectedOutcome == null ? {} : { expectedOutcome: expectedOutcome.value }),
    verification: "automated",
  });
};

export const codifySuccessfulWork = (
  summary: SuccessfulWorkSummary,
  options: CodificationOptions = {},
): KnowledgeEntry[] => {
  const generatedAt = options.generatedAt ?? summary.completedAt;
  const mode = options.redactionMode ?? "reject";
  const baseTags = compactTags(["work_summary", "successful_work", ...(summary.tags ?? [])]);
  const entries: KnowledgeEntry[] = [];

  for (const fact of summary.facts ?? []) {
    const redaction = emptyRedactionAccumulator();
    const subject = redactText(fact.subject, mode);
    const statement = redactText(fact.statement, mode);
    const affectedPaths = redactStringArray(fact.affectedPaths, mode);
    const sourceRefs = commonSourceRefs(
      [...(summary.sourceRefs ?? []), ...(fact.sourceRefs ?? []), defaultSourceRef("work_summary", summary.workId, summary.title, summary.completedAt)],
      summary.completedAt,
      mode,
    );
    mergeRedaction(redaction, subject.redaction);
    mergeRedaction(redaction, statement.redaction);
    mergeRedaction(redaction, affectedPaths.redaction);
    mergeRedaction(redaction, sourceRefs.redaction);

    const dedupeValue = normalizeText(["fact", subject.value, statement.value, ...affectedPaths.values].join(" "));
    entries.push(parseEntry({
      entryId: entryId("fact", summary.workId, dedupeValue),
      schemaVersion: "knowledge-schema.v1",
      kind: "fact",
      status: "candidate",
      title: `${subject.value}: ${statement.value}`.slice(0, 200),
      body: statement.value,
      summary: statement.value,
      tags: baseTags,
      confidence: fact.confidence ?? 0.75,
      retention: { retention: "project" },
      sourceRefs: sourceRefs.sourceRefs,
      dedupeKeys: [dedupeKey("normalized_text", dedupeValue, generatedAt)],
      createdAt: generatedAt,
      updatedAt: generatedAt,
      acceptedByUser: false,
      redaction: redactionMetadata(redaction, generatedAt),
      subject: subject.value,
      statement: statement.value,
      affectedPaths: compactStrings(affectedPaths.values),
    }));
  }

  for (const command of summary.commands ?? []) {
    entries.push(commandEntry(summary.workId, summary.title, summary.completedAt, {
      ...command,
      sourceRefs: [...(summary.sourceRefs ?? []), ...(command.sourceRefs ?? [])],
    }, baseTags, options));
  }

  return entries;
};

const gotchaEntry = (
  sourceId: string,
  sourceKind: KnowledgeSourceKind,
  title: string,
  observedAt: string,
  symptomInput: string,
  mitigationInput: string,
  options: CodificationOptions,
  details: {
    cause?: string;
    severity?: "low" | "medium" | "high";
    affectedPaths?: string[];
    tags?: string[];
    sourceRefs?: CodificationSourceRefInput[];
  } = {},
): KnowledgeEntry => {
  const generatedAt = options.generatedAt ?? observedAt;
  const mode = options.redactionMode ?? "reject";
  const redaction = emptyRedactionAccumulator();
  const symptom = redactText(symptomInput, mode);
  const mitigation = redactText(mitigationInput, mode);
  const cause = details.cause == null ? undefined : redactText(details.cause, mode);
  const affectedPaths = redactStringArray(details.affectedPaths, mode);
  const sourceRefs = commonSourceRefs(
    [...(details.sourceRefs ?? []), defaultSourceRef(sourceKind, sourceId, title, observedAt)],
    observedAt,
    mode,
  );
  mergeRedaction(redaction, symptom.redaction);
  mergeRedaction(redaction, mitigation.redaction);
  if (cause != null) {
    mergeRedaction(redaction, cause.redaction);
  }
  mergeRedaction(redaction, affectedPaths.redaction);
  mergeRedaction(redaction, sourceRefs.redaction);

  const dedupeValue = normalizeText([
    "gotcha",
    sourceKind,
    symptom.value,
    cause?.value ?? "",
    mitigation.value,
    ...affectedPaths.values,
  ].join(" "));

  return parseEntry({
    entryId: entryId("gotcha", sourceId, dedupeValue),
    schemaVersion: "knowledge-schema.v1",
    kind: "gotcha",
    status: "candidate",
    title,
    body: `${symptom.value} Mitigation: ${mitigation.value}`,
    summary: `${symptom.value} -> ${mitigation.value}`,
    tags: compactTags([sourceKind, "gotcha", ...(details.tags ?? [])]),
    confidence: sourceKind === "eval_failure" ? 0.7 : 0.75,
    retention: { retention: "project" },
    sourceRefs: sourceRefs.sourceRefs,
    dedupeKeys: [dedupeKey("normalized_text", dedupeValue, generatedAt)],
    createdAt: generatedAt,
    updatedAt: generatedAt,
    acceptedByUser: false,
    redaction: redactionMetadata(redaction, generatedAt),
    severity: details.severity ?? "medium",
    symptom: symptom.value,
    ...(cause == null ? {} : { cause: cause.value }),
    mitigation: mitigation.value,
    affectedPaths: compactStrings(affectedPaths.values),
  });
};

export const codifyFailedWork = (
  summary: FailedWorkSummary,
  options: CodificationOptions = {},
): KnowledgeEntry[] => [
  gotchaEntry(summary.failureId, "work_summary", summary.title, summary.observedAt, summary.symptom, summary.mitigation, options, {
    ...(summary.cause == null ? {} : { cause: summary.cause }),
    ...(summary.severity == null ? {} : { severity: summary.severity }),
    ...(summary.affectedPaths == null ? {} : { affectedPaths: summary.affectedPaths }),
    tags: ["failed_work", ...(summary.tags ?? [])],
    ...(summary.sourceRefs == null ? {} : { sourceRefs: summary.sourceRefs }),
  }),
];

export const codifyReviewLearning = (
  summary: ReviewLearningSummary,
  options: CodificationOptions = {},
): KnowledgeEntry[] => [
  gotchaEntry(summary.reviewId, "review", summary.title, summary.observedAt, summary.finding, summary.recommendation, options, {
    severity: summary.severity ?? "medium",
    ...(summary.affectedPaths == null ? {} : { affectedPaths: summary.affectedPaths }),
    tags: ["review", ...(summary.tags ?? [])],
    ...(summary.sourceRefs == null ? {} : { sourceRefs: summary.sourceRefs }),
  }),
];

export const codifyEvalFailure = (
  summary: EvalFailureSummary,
  options: CodificationOptions = {},
): KnowledgeEntry[] => {
  if (summary.hiddenHoldout === true || summary.split === "holdout") {
    throw new Error("hidden holdout eval failures must not be codified as project knowledge");
  }

  return [gotchaEntry(
    summary.evalId,
    "eval_failure",
    summary.title,
    summary.observedAt,
    `${summary.scenario} expected ${summary.expected} but got ${summary.actual}.`,
    summary.mitigation,
    options,
    {
      ...(summary.rootCause == null ? {} : { cause: summary.rootCause }),
      severity: summary.severity ?? "high",
      ...(summary.affectedPaths == null ? {} : { affectedPaths: summary.affectedPaths }),
      tags: ["eval_failure", ...(summary.tags ?? [])],
      ...(summary.sourceRefs == null ? {} : { sourceRefs: summary.sourceRefs }),
    },
  )];
};

export const codifyObservedCommand = (
  summary: ObservedCommandSummary,
  options: CodificationOptions = {},
): KnowledgeEntry[] => [
  commandEntry(
    summary.commandId,
    summary.title,
    summary.observedAt,
    {
      command: summary.command,
      purpose: summary.purpose,
      ...(summary.cwd == null ? {} : { cwd: summary.cwd }),
      ...(summary.whenToUse == null ? {} : { whenToUse: summary.whenToUse }),
      ...(summary.expectedOutcome == null ? {} : { expectedOutcome: summary.expectedOutcome }),
      ...(summary.confidence == null ? {} : { confidence: summary.confidence }),
      ...(summary.sourceRefs == null ? {} : { sourceRefs: summary.sourceRefs }),
    },
    [
      "observed_command",
      ...(summary.exitCode == null ? [] : [summary.exitCode === 0 ? "command_success" : "command_failure"]),
      ...(summary.tags ?? []),
    ],
    options,
  ),
];

export const codifyGotchaLearning = (
  summary: GotchaLearningSummary,
  options: CodificationOptions = {},
): KnowledgeEntry[] => {
  const entry = gotchaEntry(
    summary.gotchaId,
    "manual",
    summary.title,
    summary.observedAt,
    summary.symptom,
    summary.mitigation,
    options,
    {
      ...(summary.cause == null ? {} : { cause: summary.cause }),
      ...(summary.severity == null ? {} : { severity: summary.severity }),
      ...(summary.affectedPaths == null ? {} : { affectedPaths: summary.affectedPaths }),
      tags: ["gotcha_learning", ...(summary.tags ?? [])],
      ...(summary.sourceRefs == null ? {} : { sourceRefs: summary.sourceRefs }),
    },
  );

  if (summary.confidence == null) {
    return [entry];
  }

  return [parseEntry({ ...entry, confidence: summary.confidence })];
};

export const codifyDecisionLearning = (
  summary: DecisionLearningSummary,
  options: CodificationOptions = {},
): KnowledgeEntry[] => {
  const generatedAt = options.generatedAt ?? summary.decidedAt;
  const mode = options.redactionMode ?? "reject";
  const redaction = emptyRedactionAccumulator();
  const decision = redactText(summary.decision, mode);
  const rationale = redactStringArray(summary.rationale, mode);
  const alternativesConsidered = redactStringArray(summary.alternativesConsidered, mode);
  const sourceRefs = commonSourceRefs(
    [...(summary.sourceRefs ?? []), defaultSourceRef("manual", summary.decisionId, summary.title, summary.decidedAt)],
    summary.decidedAt,
    mode,
  );
  mergeRedaction(redaction, decision.redaction);
  mergeRedaction(redaction, rationale.redaction);
  mergeRedaction(redaction, alternativesConsidered.redaction);
  mergeRedaction(redaction, sourceRefs.redaction);

  const rationaleText = rationale.values.length === 0 ? "" : ` Rationale: ${rationale.values.join(" ")}`;
  const dedupeValue = normalizeText(["decision", decision.value, ...rationale.values].join(" "));

  return [parseEntry({
    entryId: entryId("decision", summary.decisionId, dedupeValue),
    schemaVersion: "knowledge-schema.v1",
    kind: "decision",
    status: "candidate",
    title: summary.title,
    body: `${decision.value}${rationaleText}`.trim(),
    summary: decision.value,
    tags: compactTags(["decision", ...(summary.tags ?? [])]),
    confidence: summary.confidence ?? 0.85,
    retention: { retention: "project" },
    sourceRefs: sourceRefs.sourceRefs,
    dedupeKeys: [dedupeKey("normalized_text", dedupeValue, generatedAt)],
    createdAt: generatedAt,
    updatedAt: generatedAt,
    acceptedByUser: false,
    redaction: redactionMetadata(redaction, generatedAt),
    decision: decision.value,
    rationale: rationale.values,
    alternativesConsidered: alternativesConsidered.values,
    decidedAt: summary.decidedAt,
    supersedesEntryIds: summary.supersedesEntryIds ?? [],
  })];
};

export const codifyAcceptedUserCorrection = (
  correction: AcceptedUserCorrectionInput,
  options: CodificationOptions = {},
): KnowledgeEntry[] => {
  const generatedAt = options.generatedAt ?? correction.acceptedAt;
  const mode = options.redactionMode ?? "reject";
  const redaction = emptyRedactionAccumulator();
  const original = redactText(correction.original, mode);
  const corrected = redactText(correction.corrected, mode);
  const sourceRefs = commonSourceRefs(correction.sourceRefs, correction.acceptedAt, mode);
  mergeRedaction(redaction, original.redaction);
  mergeRedaction(redaction, corrected.redaction);
  mergeRedaction(redaction, sourceRefs.redaction);

  const correctionRedaction = redactionMetadata(redaction, generatedAt);
  const dedupeValue = normalizeText(["accepted_user_correction", original.value, corrected.value].join(" "));
  const parsedCorrection: AcceptedUserCorrection = {
    correctionId: correction.correctionId,
    original: original.value,
    corrected: corrected.value,
    acceptedAt: correction.acceptedAt,
    acceptedBy: correction.acceptedBy ?? "user",
    appliesToEntryIds: correction.appliesToEntryIds ?? [],
    sourceRefs: sourceRefs.sourceRefs,
    redaction: correction.redaction?.state === "redacted" ? correction.redaction : correctionRedaction,
  };

  return [parseEntry({
    entryId: entryId("accepted_user_correction", correction.correctionId, dedupeValue),
    schemaVersion: "knowledge-schema.v1",
    kind: "accepted_user_correction",
    status: "candidate",
    title: correction.title ?? `Accepted correction ${correction.correctionId}`,
    body: `Replace "${original.value}" with "${corrected.value}".`,
    summary: corrected.value,
    tags: compactTags(["user_correction", ...(correction.tags ?? [])]),
    confidence: 1,
    retention: { retention: "project" },
    sourceRefs: sourceRefs.sourceRefs,
    dedupeKeys: [dedupeKey("normalized_text", dedupeValue, generatedAt)],
    createdAt: generatedAt,
    updatedAt: generatedAt,
    acceptedByUser: (correction.acceptedBy ?? "user") === "user",
    redaction: correction.redaction?.state === "redacted" ? correction.redaction : correctionRedaction,
    correction: parsedCorrection,
  })];
};

export const codifyKnowledgeCandidates = (
  input: KnowledgeCodificationInput,
  options: CodificationOptions = {},
): KnowledgeEntry[] => [
  ...(input.successfulWork ?? []).flatMap((summary) => codifySuccessfulWork(summary, options)),
  ...(input.failedWork ?? []).flatMap((summary) => codifyFailedWork(summary, options)),
  ...(input.reviews ?? []).flatMap((summary) => codifyReviewLearning(summary, options)),
  ...(input.evalFailures ?? []).flatMap((summary) => codifyEvalFailure(summary, options)),
  ...(input.corrections ?? []).flatMap((correction) => codifyAcceptedUserCorrection(correction, options)),
  ...(input.commands ?? []).flatMap((command) => codifyObservedCommand(command, options)),
  ...(input.gotchas ?? []).flatMap((gotcha) => codifyGotchaLearning(gotcha, options)),
  ...(input.decisions ?? []).flatMap((decision) => codifyDecisionLearning(decision, options)),
];
