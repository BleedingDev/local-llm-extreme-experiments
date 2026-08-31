import { createHash } from "node:crypto";
import {
  AcpReplayCaptureSchema,
  groupAcpReplayRecords,
  type AcpReplayCapture,
  type AcpReplayCaptureInput,
} from "./capture";
import {
  extractReplayEvalCaseSkeleton,
  ReplayExtractionMetadataSchema,
  type ReplayEvalCaseSkeleton,
  type ReplayExtractionMetadata,
  type ReplayExtractionMetadataInput,
  type ReplayOracleStrength,
} from "./extraction";
import {
  redactAcpReplayCaptureForLocalSafeUse,
  type ReplayRedactionOptions,
  type ReplayRedactionReport,
} from "./redaction";
import type { EvalSplit } from "../eval-harness/types";

export type ReplayDatasetExtractionOptions = {
  sourcePath?: string;
  metadata?: Partial<ReplayExtractionMetadataInput>;
  redaction?: ReplayRedactionOptions;
};

export type ReplayDatasetCase = {
  capture: AcpReplayCapture;
  replayCase: ReplayEvalCaseSkeleton;
  redactionReport: ReplayRedactionReport;
};

export const extractReplayDatasetCaseFromCapture = (
  captureInput: AcpReplayCaptureInput,
  options: ReplayDatasetExtractionOptions = {},
): ReplayDatasetCase => {
  const rawCapture = AcpReplayCaptureSchema.parse(captureInput);
  const redacted = redactAcpReplayCaptureForLocalSafeUse(rawCapture, options.redaction);
  const metadata = deriveReplayExtractionMetadata(redacted.capture, {
    ...(options.sourcePath === undefined ? {} : { sourcePath: options.sourcePath }),
    ...(options.metadata === undefined ? {} : { overrides: options.metadata }),
  });
  return {
    capture: redacted.capture,
    replayCase: extractReplayEvalCaseSkeleton({
      capture: redacted.capture,
      metadata,
    }),
    redactionReport: redacted.report,
  };
};

export const extractReplayDatasetCasesFromCaptures = (
  captures: readonly AcpReplayCaptureInput[],
  options: ReplayDatasetExtractionOptions = {},
): ReplayDatasetCase[] =>
  captures
    .map((capture) => extractReplayDatasetCaseFromCapture(capture, options))
    .sort((left, right) => left.replayCase.evalCaseId.localeCompare(right.replayCase.evalCaseId));

export const deriveReplayExtractionMetadata = (
  captureInput: AcpReplayCaptureInput,
  input: {
    sourcePath?: string;
    overrides?: Partial<ReplayExtractionMetadataInput>;
  } = {},
): ReplayExtractionMetadata => {
  const capture = AcpReplayCaptureSchema.parse(captureInput);
  const groups = groupAcpReplayRecords(capture);
  const primaryPrompt = groups.prompts.find((prompt) => prompt.promptRole === "user") ?? groups.prompts[0];
  const classification = classifyReplayCapture(capture);
  const split = input.overrides?.split ?? capture.defaultSplitHint ?? deterministicReplaySplit(capture.captureId);
  const defaultSourceRef = {
    sourceKind: "capture" as const,
    captureId: capture.captureId,
    ...(input.sourcePath === undefined ? {} : { path: input.sourcePath }),
    redactionStatus: capture.redactionStatus,
  };
  const metadata: ReplayExtractionMetadataInput = {
    evalCaseId: input.overrides?.evalCaseId ?? `replay.eval.live.${stableId(capture.captureId)}`,
    title: input.overrides?.title ?? classification.title,
    task: input.overrides?.task ?? primaryPrompt?.content ?? classification.title,
    split,
    splitRationale: input.overrides?.splitRationale ?? splitRationale(split, capture.defaultSplitHint),
    oracleStrength: input.overrides?.oracleStrength ?? classification.oracleStrength,
    expectedBehavior: input.overrides?.expectedBehavior ?? {
      summary: classification.expectedBehaviorSummary,
      assertions: [],
      notes: [
        "Generated from a redacted live ACP replay capture; strengthen the oracle before using as a golden test.",
      ],
    },
    sourceRefs: input.overrides?.sourceRefs ?? [defaultSourceRef],
    ...(input.overrides?.fixtureWorkspace === undefined ? {} : {
      fixtureWorkspace: input.overrides.fixtureWorkspace,
    }),
    tags: input.overrides?.tags ?? classification.tags,
    timeoutMs: input.overrides?.timeoutMs ?? 120_000,
  };
  return ReplayExtractionMetadataSchema.parse(metadata);
};

export const deterministicReplaySplit = (key: string): EvalSplit => {
  const value = Number.parseInt(createHash("sha256").update(key).digest("hex").slice(0, 8), 16) % 10;
  if (value === 0) return "holdout";
  if (value <= 3) return "dev";
  return "train";
};

const classifyReplayCapture = (
  capture: AcpReplayCapture,
): {
  title: string;
  expectedBehaviorSummary: string;
  oracleStrength: ReplayOracleStrength;
  tags: string[];
} => {
  const groups = groupAcpReplayRecords(capture);
  const failedEdit = groups.editAttempts.find((record) =>
    record.attempt.phaseResults.some((phase) => phase.status === "failed") ||
    record.attempt.verificationStatus === "failed" ||
    record.attempt.postApplyConsistencyStatus === "inconsistent" ||
    record.attempt.selfDetectedRegressionStatus === "confirmed",
  );
  const failedTool = groups.toolCalls.find((record) => record.status !== "succeeded");
  const failedTerminal = groups.terminalCommands.find((record) => record.status !== "succeeded");
  const route = groups.modeRoutes[0];
  if (failedEdit != null) {
    return {
      title: "Live ACP edit failure or recovery replay",
      expectedBehaviorSummary: "The candidate policy must preserve the edit failure phase, recovery signal, and final safety outcome.",
      oracleStrength: "medium",
      tags: ["replay", "live", "edit-failure"],
    };
  }
  if (failedTool != null) {
    return {
      title: "Live ACP tool failure replay",
      expectedBehaviorSummary: "The candidate policy must preserve the tool failure class and avoid masking the failed tool call.",
      oracleStrength: "medium",
      tags: ["replay", "live", "tool-failure"],
    };
  }
  if (failedTerminal != null) {
    return {
      title: "Live ACP terminal verification replay",
      expectedBehaviorSummary: "The candidate policy must preserve terminal verification failure evidence and avoid false success.",
      oracleStrength: "medium",
      tags: ["replay", "live", "terminal-failure"],
    };
  }
  if (route?.sideEffectPolicy === "no_side_effects") {
    return {
      title: "Live ACP no-side-effect chat replay",
      expectedBehaviorSummary: "The candidate policy must keep the turn conversational without filesystem, terminal, or permission side effects.",
      oracleStrength: "strong",
      tags: ["replay", "live", "routing", "no-side-effect"],
    };
  }
  return {
    title: "Live ACP successful session replay",
    expectedBehaviorSummary: "The candidate policy should preserve the successful outcome while keeping trace and artifact lineage comparable.",
    oracleStrength: "weak",
    tags: ["replay", "live", "successful-session"],
  };
};

const splitRationale = (split: EvalSplit, captureHint: EvalSplit | undefined): string =>
  captureHint === undefined
    ? `Deterministic replay split assignment selected ${split}; holdout remains hidden from optimizer input.`
    : `Capture default split hint selected ${split}; holdout remains hidden from optimizer input.`;

const stableId = (value: string): string =>
  value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+|-+$/g, "") || `capture.${createHash("sha256").update(value).digest("hex").slice(0, 12)}`;
