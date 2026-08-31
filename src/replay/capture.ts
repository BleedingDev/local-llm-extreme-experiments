import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { z } from "zod";
import {
  EditAttemptContractSchema,
  RedactionStatusSchema,
  type RedactionStatus,
} from "../edit-strategy/types";
import { EvalSplitSchema } from "../eval-harness/types";
import { JsonValueSchema, OptimizerIdSchema } from "../optimizer/types";

const IsoTimestampSchema = z.string().datetime({ offset: true });
const ArtifactRefSchema = z.string().min(1);
const RelativeOrArtifactPathSchema = z.string().min(1);

export const ReplayCaptureSchemaVersion = "acp-replay-capture.v1" as const;

export const ReplayTraceRefSchema = z.object({
  traceId: z.string().min(1),
  spanId: z.string().min(1).optional(),
  parentSpanId: z.string().min(1).optional(),
}).strict();
export type ReplayTraceRef = z.infer<typeof ReplayTraceRefSchema>;

const ReplayRecordBaseSchema = z.object({
  recordId: OptimizerIdSchema,
  at: IsoTimestampSchema.optional(),
  parentRecordIds: z.array(OptimizerIdSchema).default([]),
  traceRefs: z.array(ReplayTraceRefSchema).default([]),
  artifactRefs: z.array(ArtifactRefSchema).default([]),
});

export const AcpReplayPromptRecordSchema = ReplayRecordBaseSchema.extend({
  recordKind: z.literal("prompt"),
  promptRole: z.enum(["system", "developer", "user", "assistant"]),
  promptEvent: z.enum(["message", "user_correction"]).default("message"),
  content: z.string().min(1),
  contentRedactionStatus: RedactionStatusSchema.default("raw_local_only"),
}).strict();
export type AcpReplayPromptRecord = z.infer<typeof AcpReplayPromptRecordSchema>;

export const AcpReplayModeSchema = z.enum([
  "chat",
  "read_only",
  "mutating",
  "auto",
  "safe",
  "yolo",
  "unknown",
]);
export type AcpReplayMode = z.infer<typeof AcpReplayModeSchema>;

export const AcpReplayModeRoutingRecordSchema = ReplayRecordBaseSchema.extend({
  recordKind: z.literal("mode_route"),
  promptRecordId: OptimizerIdSchema,
  requestedMode: AcpReplayModeSchema.optional(),
  selectedMode: AcpReplayModeSchema,
  restoredMode: AcpReplayModeSchema.optional(),
  sideEffectPolicy: z.enum([
    "no_side_effects",
    "read_only",
    "write_allowed",
    "terminal_allowed",
    "unknown",
  ]).default("unknown"),
  reason: z.string().min(1).optional(),
}).strict();
export type AcpReplayModeRoutingRecord = z.infer<typeof AcpReplayModeRoutingRecordSchema>;

export const AcpReplayFileReadRecordSchema = ReplayRecordBaseSchema.extend({
  recordKind: z.literal("file_read"),
  path: RelativeOrArtifactPathSchema,
  status: z.enum(["succeeded", "failed", "truncated", "omitted"]),
  contentHash: z.string().min(1).optional(),
  excerpt: z.string().optional(),
  redactionStatus: RedactionStatusSchema.default("raw_local_only"),
  ranges: z.array(z.object({
    startLine: z.number().int().nonnegative(),
    endLine: z.number().int().nonnegative(),
  }).strict().refine((range) => range.endLine >= range.startLine, {
    message: "endLine must be greater than or equal to startLine",
  })).default([]),
  errorCode: z.string().min(1).optional(),
}).strict();
export type AcpReplayFileReadRecord = z.infer<typeof AcpReplayFileReadRecordSchema>;

export const AcpReplayEditAttemptRecordSchema = ReplayRecordBaseSchema.extend({
  recordKind: z.literal("edit_attempt"),
  attempt: EditAttemptContractSchema,
}).strict();
export type AcpReplayEditAttemptRecord = z.infer<typeof AcpReplayEditAttemptRecordSchema>;

export const AcpReplayToolCallRecordSchema = ReplayRecordBaseSchema.extend({
  recordKind: z.literal("tool_call"),
  toolCallId: OptimizerIdSchema,
  namespace: OptimizerIdSchema.optional(),
  name: OptimizerIdSchema,
  status: z.enum([
    "succeeded",
    "failed",
    "malformed_args",
    "permission_denied",
    "timed_out",
    "truncated",
  ]),
  args: JsonValueSchema.optional(),
  result: JsonValueSchema.optional(),
  resultStyle: z.enum(["text", "json", "artifact_ref", "structured_error"]).default("text"),
  retryCount: z.number().int().nonnegative().default(0),
  redactionStatus: RedactionStatusSchema.default("raw_local_only"),
  errorCode: z.string().min(1).optional(),
}).strict();
export type AcpReplayToolCallRecord = z.infer<typeof AcpReplayToolCallRecordSchema>;

export const AcpReplayTerminalCommandRecordSchema = ReplayRecordBaseSchema.extend({
  recordKind: z.literal("terminal_command"),
  commandId: OptimizerIdSchema,
  command: z.array(z.string().min(1)).min(1),
  cwd: z.string().min(1).optional(),
  status: z.enum(["succeeded", "failed", "timed_out", "permission_denied"]),
  exitCode: z.number().int().nullable().optional(),
  signal: z.string().min(1).nullable().optional(),
  stdoutArtifactRef: ArtifactRefSchema.optional(),
  stderrArtifactRef: ArtifactRefSchema.optional(),
  redactionStatus: RedactionStatusSchema.default("raw_local_only"),
  errorCode: z.string().min(1).optional(),
}).strict();
export type AcpReplayTerminalCommandRecord = z.infer<typeof AcpReplayTerminalCommandRecordSchema>;

export const AcpReplayArtifactRefRecordSchema = ReplayRecordBaseSchema.extend({
  recordKind: z.literal("artifact_ref"),
  artifactRef: ArtifactRefSchema,
  artifactKind: z.enum([
    "trace",
    "source_snapshot",
    "tool_output",
    "terminal_output",
    "edit_patch",
    "verification",
    "other",
  ]),
  path: RelativeOrArtifactPathSchema.optional(),
  contentHash: z.string().min(1).optional(),
  redactionStatus: RedactionStatusSchema.default("raw_local_only"),
}).strict();
export type AcpReplayArtifactRefRecord = z.infer<typeof AcpReplayArtifactRefRecordSchema>;

export const AcpReplayRecordSchema = z.discriminatedUnion("recordKind", [
  AcpReplayPromptRecordSchema,
  AcpReplayModeRoutingRecordSchema,
  AcpReplayFileReadRecordSchema,
  AcpReplayEditAttemptRecordSchema,
  AcpReplayToolCallRecordSchema,
  AcpReplayTerminalCommandRecordSchema,
  AcpReplayArtifactRefRecordSchema,
]);
export type AcpReplayRecord = z.infer<typeof AcpReplayRecordSchema>;

export const AcpReplayCaptureSourceSchema = z.object({
  sourceType: z.enum([
    "acp-session-jsonl",
    "cc-session-jsonl-v2",
    "codex-session-jsonl",
    "manual",
    "pi-session-jsonl",
    "spans-jsonl",
  ]),
  path: z.string().min(1).optional(),
  sessionId: z.string().min(1).optional(),
  traceIds: z.array(z.string().min(1)).default([]),
}).strict();
export type AcpReplayCaptureSource = z.infer<typeof AcpReplayCaptureSourceSchema>;

export const AcpReplayCaptureContextSchema = z.object({
  modelRole: z.string().min(1).optional(),
  provider: z.string().min(1).optional(),
  providerConfigRole: z.string().min(1).optional(),
  policyId: OptimizerIdSchema.optional(),
  modelProfileId: OptimizerIdSchema.optional(),
  codebaseProfileId: OptimizerIdSchema.optional(),
  modelServerId: OptimizerIdSchema.optional(),
  modelServerProfileId: OptimizerIdSchema.optional(),
  canonicalToolVersion: z.string().min(1).optional(),
  renderedToolVersion: z.string().min(1).optional(),
  resultStyleVersion: z.string().min(1).optional(),
  verificationPolicyVersion: z.string().min(1).optional(),
  acpConsumerCapabilities: JsonValueSchema.optional(),
}).strict();
export type AcpReplayCaptureContext = z.infer<typeof AcpReplayCaptureContextSchema>;

export const AcpReplayCaptureSchema = z.object({
  captureId: OptimizerIdSchema,
  schemaVersion: z.literal(ReplayCaptureSchemaVersion).default(ReplayCaptureSchemaVersion),
  createdAt: IsoTimestampSchema,
  source: AcpReplayCaptureSourceSchema,
  context: AcpReplayCaptureContextSchema.default({}),
  defaultSplitHint: EvalSplitSchema.optional(),
  redactionStatus: RedactionStatusSchema.default("raw_local_only"),
  records: z.array(AcpReplayRecordSchema).min(1),
}).strict().superRefine((capture, ctx) => {
  const recordsById = new Map(capture.records.map((record) => [record.recordId, record]));
  if (recordsById.size !== capture.records.length) {
    ctx.addIssue({
      code: "custom",
      path: ["records"],
      message: "replay capture record ids must be unique",
    });
  }

  const hasPrompt = capture.records.some((record) => record.recordKind === "prompt");
  if (!hasPrompt) {
    ctx.addIssue({
      code: "custom",
      path: ["records"],
      message: "replay capture requires at least one prompt record",
    });
  }

  for (const [index, record] of capture.records.entries()) {
    for (const parentRecordId of record.parentRecordIds) {
      if (!recordsById.has(parentRecordId)) {
        ctx.addIssue({
          code: "custom",
          path: ["records", index, "parentRecordIds"],
          message: `unknown parent record id: ${parentRecordId}`,
        });
      }
    }

    if (record.recordKind === "mode_route") {
      const promptRecord = recordsById.get(record.promptRecordId);
      if (promptRecord?.recordKind !== "prompt") {
        ctx.addIssue({
          code: "custom",
          path: ["records", index, "promptRecordId"],
          message: "mode routing record must reference a prompt record",
        });
      }
    }
  }
});
export type AcpReplayCapture = z.infer<typeof AcpReplayCaptureSchema>;
export type AcpReplayCaptureInput = z.input<typeof AcpReplayCaptureSchema>;

export type AcpReplayRecordGroups = {
  prompts: AcpReplayPromptRecord[];
  modeRoutes: AcpReplayModeRoutingRecord[];
  fileReads: AcpReplayFileReadRecord[];
  editAttempts: AcpReplayEditAttemptRecord[];
  toolCalls: AcpReplayToolCallRecord[];
  terminalCommands: AcpReplayTerminalCommandRecord[];
  artifactRefs: AcpReplayArtifactRefRecord[];
};

export const parseAcpReplayCapture = (capture: unknown): AcpReplayCapture =>
  AcpReplayCaptureSchema.parse(capture);

export const replayCaptureToJson = (capture: AcpReplayCaptureInput): string =>
  `${JSON.stringify(AcpReplayCaptureSchema.parse(capture), null, 2)}\n`;

export const writeAcpReplayCaptureFile = async (
  path: string,
  capture: AcpReplayCaptureInput,
): Promise<AcpReplayCapture> => {
  const parsed = AcpReplayCaptureSchema.parse(capture);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, replayCaptureToJson(parsed), "utf8");
  return parsed;
};

export const readAcpReplayCaptureFile = async (path: string): Promise<AcpReplayCapture> => {
  const raw = await readFile(path, "utf8");
  return AcpReplayCaptureSchema.parse(JSON.parse(raw));
};

export const groupAcpReplayRecords = (
  captureInput: AcpReplayCaptureInput,
): AcpReplayRecordGroups => {
  const capture = AcpReplayCaptureSchema.parse(captureInput);
  return {
    prompts: capture.records.filter((record): record is AcpReplayPromptRecord => record.recordKind === "prompt"),
    modeRoutes: capture.records.filter(
      (record): record is AcpReplayModeRoutingRecord => record.recordKind === "mode_route",
    ),
    fileReads: capture.records.filter(
      (record): record is AcpReplayFileReadRecord => record.recordKind === "file_read",
    ),
    editAttempts: capture.records.filter(
      (record): record is AcpReplayEditAttemptRecord => record.recordKind === "edit_attempt",
    ),
    toolCalls: capture.records.filter(
      (record): record is AcpReplayToolCallRecord => record.recordKind === "tool_call",
    ),
    terminalCommands: capture.records.filter(
      (record): record is AcpReplayTerminalCommandRecord => record.recordKind === "terminal_command",
    ),
    artifactRefs: capture.records.filter(
      (record): record is AcpReplayArtifactRefRecord => record.recordKind === "artifact_ref",
    ),
  };
};

export const redactionStatusForReplayRecord = (
  record: AcpReplayRecord,
): RedactionStatus | undefined => {
  switch (record.recordKind) {
    case "prompt":
      return record.contentRedactionStatus;
    case "file_read":
    case "tool_call":
    case "terminal_command":
    case "artifact_ref":
      return record.redactionStatus;
    case "edit_attempt":
      return record.attempt.redactionStatus;
    case "mode_route":
      return undefined;
  }
};
