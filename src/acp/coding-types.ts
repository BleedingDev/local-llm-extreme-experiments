import {
  EditApplyInputSchema,
  type EditApplyInput,
} from "../edit-strategy/apply-layer";
import type {
  EditAttemptContract,
  EditErrorCode,
  EditStrategyFamily,
  PostApplyConsistencyStatus,
} from "../edit-strategy/types";
import type { EditStrategyFallbackRule, EditStrategyRouterDecision, EditTaskShape } from "../optimizer/edit-policy-router";
import type { RenderedToolContract } from "../optimizer/types";
import type { TerminalCommandResult } from "./terminal";
import { detectProjectKind, type ProjectKind } from "../workspace";
import type { CanonicalEditStrategyDefinition } from "../edit-strategy/taxonomy";
import type {
  CodingGenerationDiagnostic,
  CodingStructuredImpossibility,
} from "./coding-progress-diagnostics";

export type CodingFileSelection = {
  approach: string;
  filesToRead: string[];
  filesToCreate: string[];
};

export type CodingFileSnapshot = {
  kind: "existing" | "create";
  path: string;
  relativePath: string;
  content: string;
  hash: string;
};

export type CodingCommand = {
  command: string;
  args: string[];
  reason: string;
};

export type LiveEditContext = {
  taskShape: EditTaskShape;
  decision: EditStrategyRouterDecision;
  definition: CanonicalEditStrategyDefinition;
  renderedContract: RenderedToolContract;
};

export type CodingEditOperation = {
  reason: string;
  editInput: EditApplyInput;
  targetFiles: string[];
  editStrategyId: string;
  editStrategyFamily: EditStrategyFamily;
  renderedEditToolContractId: string;
  fallbackFromStrategyId?: string;
  fallbackToStrategyId?: string;
  fallbackTrigger?: EditStrategyFallbackRule["trigger"];
  repairRound?: number;
};

export type CodingPatch = {
  summary: string;
  editStrategy: {
    strategyId: string;
    strategyFamily: EditStrategyFamily;
    renderedEditToolContractId: string;
  };
  generation?: CodingGenerationDiagnostic;
  structuredImpossibility?: CodingStructuredImpossibility;
  edits: CodingEditOperation[];
  commands: CodingCommand[];
  risks: string[];
  parseFailures: string[];
};

export type CodingEditResult = {
  path: string;
  ok: boolean;
  reason: string;
  oldHash?: string;
  newHash?: string;
  editStrategyId: string;
  editStatus: string;
  errorCode?: string;
  newContent?: string;
  editAttempt?: EditAttemptContract;
};

export type PostApplyConsistencyCheck = {
  path: string;
  status: Exclude<PostApplyConsistencyStatus, "not_checked" | "pre_existing_failure">;
  expectedHash?: string;
  actualHash?: string;
  reason: string;
  errorCode?: EditErrorCode;
};

export const renderVerifierResultsForLlm = (
  results: readonly { command: string; args: string[]; exitCode: number | null; output: string }[],
  repairRound?: number,
): string => {
  const previousLabel =
    repairRound == null || repairRound <= 1
      ? "the initial attempt"
      : `repair round ${repairRound - 1}`;
  const heading = `Verifier results from ${previousLabel}`;
  const lines: string[] = [`${heading}:`];
  for (const r of results) {
    const cmdline = [r.command, ...r.args].join(" ");
    const verdict = r.exitCode === 0 ? "PASSED" : `FAILED (exit ${r.exitCode ?? "?"})`;
    lines.push(`- $ ${cmdline}`);
    lines.push(`  ${verdict}`);
    const tail = (r.output ?? "").slice(-3000);
    if (tail.trim().length > 0) {
      lines.push("  output:");
      for (const line of tail.split("\n")) lines.push(`    ${line}`);
    }
  }
  lines.push("");
  lines.push("Use the verifier output as concrete evidence of what is currently wrong. Adjust the proposed edits to make every command exit 0 while preserving the original task intent.");
  return lines.join("\n");
};

export const defaultVerificationCommands = (kind: ProjectKind): CodingCommand[] => {
  switch (kind) {
    case "node":
      return [
        { command: "npm", args: ["run", "typecheck"], reason: "Default TypeScript verification." },
      ];
    case "python":
      return [
        { command: "python3", args: ["-m", "compileall", "-q", "."], reason: "Default Python compileall verification." },
      ];
    case "shell":
      return [
        { command: "sh", args: ["-c", "set -e; for f in $(find . -maxdepth 3 -name '*.sh' 2>/dev/null); do bash -n \"$f\" || exit $?; done"], reason: "Default shell syntax check via bash -n." },
      ];
    case "rust":
      return [
        { command: "cargo", args: ["check", "--quiet"], reason: "Default Rust cargo check." },
      ];
    case "go":
      return [
        { command: "go", args: ["build", "./..."], reason: "Default Go build." },
      ];
    case "unknown":
    default:
      return [];
  }
};

export const verificationCommands = (commands: readonly CodingCommand[], cwd: string): CodingCommand[] => {
  if (commands.length > 0) {
    return commands.slice(0, 4);
  }
  return defaultVerificationCommands(detectProjectKind(cwd));
};

export const modelPayloadForSelectedEditStrategy = (
  edit: Record<string, unknown>,
  strategyFamily: EditStrategyFamily,
  fileSnapshots: readonly CodingFileSnapshot[],
): unknown => {
  const explicitPayload = edit.payload != null && typeof edit.payload === "object"
    ? edit.payload as Record<string, unknown>
    : undefined;
  const payload = explicitPayload ?? edit;
  switch (strategyFamily) {
    case "whole_file":
      return normalizeModelPayloadPaths({
        path: payload.path,
        content: payload.content,
        baseContentHash: payload.baseContentHash,
        intent: payload.intent ?? edit.reason,
      }, fileSnapshots);
    case "exact_replace":
      return normalizeModelPayloadPaths({
        path: payload.path,
        search: payload.search,
        replace: payload.replace,
        expectedContentHash: payload.expectedContentHash,
      }, fileSnapshots);
    case "hash_range":
      return normalizeModelPayloadPaths(payload, fileSnapshots);
    case "unified_diff":
    case "apply_patch":
      return payload;
    default:
      return payload;
  }
};

export const normalizeModelPayloadPaths = (
  payload: unknown,
  fileSnapshots: readonly CodingFileSnapshot[],
): unknown => {
  if (Array.isArray(payload)) {
    return payload.map((item) => normalizeModelPayloadPaths(item, fileSnapshots));
  }
  if (payload == null || typeof payload !== "object") {
    return payload;
  }
  return Object.fromEntries(
    Object.entries(payload as Record<string, unknown>).map(([key, value]) => [
      key,
      key === "path" && typeof value === "string"
        ? normalizeModelPath(value, fileSnapshots)
        : normalizeModelPayloadPaths(value, fileSnapshots),
    ]),
  );
};

export const normalizeModelPath = (path: string, fileSnapshots: readonly CodingFileSnapshot[]): string => {
  const matched = fileSnapshots.find((file) => file.path === path || file.relativePath === path);
  if (matched !== undefined) {
    return matched.relativePath;
  }
  return path.replace(/^[ab]\//, "");
};

export const targetFilesForEditInput = (
  input: EditApplyInput,
  fileSnapshots: readonly CodingFileSnapshot[],
): string[] => {
  switch (input.strategyFamily) {
    case "whole_file":
    case "exact_replace":
      return [input.payload.path];
    case "hash_range":
      return [...new Set(input.payload.operations.map((operation) => operation.path))];
    case "unified_diff":
    case "apply_patch": {
      const paths = extractPatchTargetFiles(input.payload.patch);
      return paths.length > 0 ? paths : fileSnapshots.map((file) => file.relativePath);
    }
  }
};

export const extractPatchTargetFiles = (patch: string): string[] => {
  const paths = new Set<string>();
  for (const line of patch.split(/\r?\n/)) {
    if (line.startsWith("+++ ")) {
      const path = line.slice(4).trim().replace(/^[ab]\//, "");
      if (path !== "/dev/null" && path.length > 0) {
        paths.add(path);
      }
    }
    if (line.startsWith("*** Update File: ")) {
      const path = line.slice("*** Update File: ".length).trim();
      if (path.length > 0) {
        paths.add(path);
      }
    }
  }
  return [...paths].sort();
};

export const parseCodingEditOperation = (input: {
  rawEdit: Record<string, unknown>;
  index: number;
  editContext: LiveEditContext;
  fileSnapshots: readonly CodingFileSnapshot[];
}): { edit?: CodingEditOperation; parseFailure?: string } => {
  const payload = modelPayloadForSelectedEditStrategy(
    input.rawEdit,
    input.editContext.decision.selectedStrategyFamily,
    input.fileSnapshots,
  );
  const editInput = EditApplyInputSchema.safeParse({
    strategyFamily: input.editContext.decision.selectedStrategyFamily,
    payload,
  });
  if (!editInput.success) {
    return {
      parseFailure: `edit ${input.index + 1}: ${editInput.error.issues.map((issue) => issue.message).join("; ")}`,
    };
  }
  return {
    edit: {
      reason: String(input.rawEdit.reason ?? "Model-proposed code edit."),
      editInput: editInput.data,
      targetFiles: targetFilesForEditInput(editInput.data, input.fileSnapshots),
      editStrategyId: input.editContext.decision.selectedStrategyId,
      editStrategyFamily: input.editContext.decision.selectedStrategyFamily,
      renderedEditToolContractId: input.editContext.renderedContract.renderedToolId,
    },
  };
};

export const terminalResultsForVerifierPrompt = (
  results: readonly TerminalCommandResult[],
  repairRound?: number,
): string => renderVerifierResultsForLlm(results, repairRound);
