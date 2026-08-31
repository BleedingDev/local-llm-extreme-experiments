import type { SourceAdapterType } from "./boundary";

export type SourceAdapterFailureKind =
  | "bash_nonzero"
  | "cancellation"
  | "command_not_found"
  | "edit_before_read"
  | "generic_error"
  | "hallucinated_skill"
  | "malformed_args"
  | "non_unique_edit_string"
  | "permission_denied"
  | "timeout"
  | "user_correction";

export type SourceAdapterFailureClassification = {
  kind: SourceAdapterFailureKind;
  errorCode: string;
  phase: "edit" | "terminal" | "tool" | "user_feedback";
  statusMessage: string;
  statusCode: "STATUS_CODE_ERROR";
};

export type ClassifySourceAdapterFailureInput = {
  sourceType: SourceAdapterType;
  eventKind: string;
  observationKind: "AGENT" | "CHAIN" | "LLM" | "TOOL";
  statusCode?: "STATUS_CODE_OK" | "STATUS_CODE_ERROR" | "STATUS_CODE_UNSET" | undefined;
  statusMessage?: string | undefined;
  attributes: Record<string, unknown>;
};

export const classifySourceAdapterFailure = (
  input: ClassifySourceAdapterFailureInput,
): SourceAdapterFailureClassification | undefined => {
  const text = searchableText(input);
  const toolName = lower(stringValue(input.attributes["tool.name"]));
  const toolStatus = lower(stringValue(input.attributes["tool.status"]));
  const isErrorStatus = input.statusCode === "STATUS_CODE_ERROR"
    || input.attributes["tool.is_error"] === true
    || toolStatus === "failed"
    || toolStatus === "error"
    || toolStatus === "malformed_args"
    || toolStatus === "permission_denied"
    || toolStatus === "timed_out"
    || toolStatus === "truncated";

  if (input.eventKind.includes("user_message") && correctionPattern.test(text)) {
    return failure("user_correction", "user_correction", "user_feedback", "User correction observed in baseline transcript.");
  }
  if (/\b(cancelled|canceled|aborted|user interrupted|interrupt signal)\b/i.test(text)) {
    return failure("cancellation", "cancellation", "tool", firstLine(input.statusMessage, text));
  }
  if (/\b(timed out|timeout|time limit exceeded|deadline exceeded)\b/i.test(text)) {
    return failure("timeout", "timeout", phaseForTool(toolName), firstLine(input.statusMessage, text));
  }
  if (/(?:^|\n|\b)(?:zsh|bash|sh|fish|cmd|powershell)?[:\s-]*(?:\d+:\s*)?command not found\b/i.test(text)
    || /\bnot recognized as (?:an internal|a cmdlet|a command)\b/i.test(text)) {
    return failure("command_not_found", "command_not_found", "terminal", firstLine(input.statusMessage, text));
  }
  if (/\b(?:exit code|exit status|exited with code|process exited with code)\s+[1-9]\d*\b/i.test(text)
    || (isShellTool(toolName) && /\bcommand failed\b/i.test(text))) {
    return failure("bash_nonzero", "bash_nonzero", "terminal", firstLine(input.statusMessage, text));
  }
  if (/\b(?:no such skill|unknown skill|skill .*not found|could not load skill|missing skill)\b/i.test(text)) {
    return failure("hallucinated_skill", "hallucinated_skill", "tool", firstLine(input.statusMessage, text));
  }
  if (/\b(?:not unique|non-unique|multiple matches|matched multiple|found \d+ matches|ambiguous .*replace|old_string .*unique)\b/i.test(text)) {
    return failure("non_unique_edit_string", "non_unique_edit_string", "edit", firstLine(input.statusMessage, text));
  }
  if (/\b(?:must read .* before (?:edit|editing)|read .* before (?:edit|editing)|has not been read|edit-before-read|edit before read)\b/i.test(text)) {
    return failure("edit_before_read", "edit_before_read", "edit", firstLine(input.statusMessage, text));
  }
  if (/\b(?:permission denied|permission rejected|not allowed|operation not permitted)\b/i.test(text)) {
    return failure("permission_denied", "permission_denied", phaseForTool(toolName), firstLine(input.statusMessage, text));
  }
  if (toolStatus === "malformed_args" || /\b(?:malformed args|invalid tool args|invalid arguments)\b/i.test(text)) {
    return failure("malformed_args", "malformed_args", "tool", firstLine(input.statusMessage, text));
  }
  if (isErrorStatus) {
    return failure("generic_error", "generic_error", phaseForTool(toolName), firstLine(input.statusMessage, text));
  }
  return undefined;
};

export const sourceAdapterFailureAttributes = (
  classification: SourceAdapterFailureClassification | undefined,
): Record<string, unknown> => classification == null ? {} : {
  "source.failure.kind": classification.kind,
  "source.failure.error_code": classification.errorCode,
  "source.failure.phase": classification.phase,
  "source.baseline.role": "observed_baseline",
  "source.baseline.gold": false,
};

const failure = (
  kind: SourceAdapterFailureKind,
  errorCode: string,
  phase: SourceAdapterFailureClassification["phase"],
  statusMessage: string,
): SourceAdapterFailureClassification => ({
  kind,
  errorCode,
  phase,
  statusMessage,
  statusCode: "STATUS_CODE_ERROR",
});

const correctionPattern =
  /\b(?:no,|not quite|that's wrong|that is wrong|incorrect|wrong file|you (?:missed|changed|broke|forgot)|actually,|please (?:undo|revert|fix)|this (?:failed|is not right))\b/i;

const searchableText = (input: ClassifySourceAdapterFailureInput): string =>
  [
    input.statusMessage,
    input.attributes["error.message"],
    input.attributes["tool.status"],
    input.attributes["tool.name"],
    input.attributes["input.value"],
    input.attributes["output.value"],
    input.attributes["source.record.redacted"],
  ].map(stringifySearchValue).filter((value) => value.length > 0).join("\n");

const stringifySearchValue = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (value == null) return "";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const phaseForTool = (toolName: string | undefined): SourceAdapterFailureClassification["phase"] => {
  if (toolName == null) return "tool";
  if (isShellTool(toolName)) return "terminal";
  if (/\b(?:edit|multiedit|write|apply_patch|str_replace|replace)\b/i.test(toolName)) return "edit";
  return "tool";
};

const isShellTool = (toolName: string | undefined): boolean =>
  toolName != null && /^(?:bash|shell|sh|exec|exec_command|terminal|run_command)$/i.test(toolName);

const firstLine = (...values: readonly (string | undefined)[]): string => {
  const value = values.find((candidate) => candidate != null && candidate.trim().length > 0) ?? "";
  return value.split(/\r?\n/, 1)[0] ?? "";
};

const stringValue = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const lower = (value: string | undefined): string | undefined => value?.toLowerCase();
