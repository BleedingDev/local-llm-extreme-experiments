import { createHash } from "node:crypto";
import { z } from "zod";
import {
  EditErrorCodeSchema,
  EditStrategyFamilySchema,
  type EditErrorCode,
  type EditStrategyFamily,
} from "./types";

const RelativePathSchema = z.string().min(1).regex(/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$)).+$/);

export const EditWorkspaceFileSchema = z.object({
  path: RelativePathSchema,
  content: z.string(),
  contentHash: z.string().optional(),
}).strict();
export type EditWorkspaceFile = z.infer<typeof EditWorkspaceFileSchema>;

export const EditApplyWorkspaceSchema = z.object({
  files: z.array(EditWorkspaceFileSchema).default([]),
  protectedPaths: z.array(RelativePathSchema).default([]),
}).strict();
export type EditApplyWorkspace = z.infer<typeof EditApplyWorkspaceSchema>;

export const WholeFileEditInputSchema = z.object({
  path: RelativePathSchema,
  content: z.string(),
  baseContentHash: z.string().optional(),
  intent: z.string().optional(),
}).strict();
export type WholeFileEditInput = z.infer<typeof WholeFileEditInputSchema>;

export const ExactReplaceEditInputSchema = z.object({
  path: RelativePathSchema,
  search: z.string().min(1),
  replace: z.string(),
  expectedContentHash: z.string().optional(),
}).strict();
export type ExactReplaceEditInput = z.infer<typeof ExactReplaceEditInputSchema>;

export const UnifiedDiffEditInputSchema = z.object({
  patch: z.string().min(1),
}).strict();
export type UnifiedDiffEditInput = z.infer<typeof UnifiedDiffEditInputSchema>;

export const ApplyPatchEditInputSchema = z.object({
  patch: z.string().min(1),
}).strict();
export type ApplyPatchEditInput = z.infer<typeof ApplyPatchEditInputSchema>;

export const HashRangeOperationSchema = z.object({
  path: RelativePathSchema,
  startLine: z.number().int().positive(),
  endLine: z.number().int().positive(),
  expectedContentHash: z.string().optional(),
  replacement: z.string(),
}).strict().refine((operation) => operation.endLine >= operation.startLine, {
  message: "endLine must be greater than or equal to startLine",
});
export type HashRangeOperation = z.infer<typeof HashRangeOperationSchema>;

export const HashRangeEditInputSchema = z.object({
  operations: z.array(HashRangeOperationSchema).min(1),
}).strict();
export type HashRangeEditInput = z.infer<typeof HashRangeEditInputSchema>;

export const EditApplyInputSchema = z.discriminatedUnion("strategyFamily", [
  z.object({ strategyFamily: z.literal("whole_file"), payload: WholeFileEditInputSchema }).strict(),
  z.object({ strategyFamily: z.literal("exact_replace"), payload: ExactReplaceEditInputSchema }).strict(),
  z.object({ strategyFamily: z.literal("unified_diff"), payload: UnifiedDiffEditInputSchema }).strict(),
  z.object({ strategyFamily: z.literal("apply_patch"), payload: ApplyPatchEditInputSchema }).strict(),
  z.object({ strategyFamily: z.literal("hash_range"), payload: HashRangeEditInputSchema }).strict(),
]);
export type EditApplyInput = z.infer<typeof EditApplyInputSchema>;

export type AppliedEditFile = {
  path: string;
  beforeContent?: string | undefined;
  afterContent?: string | undefined;
  changeKind: "added" | "modified" | "deleted";
};

export type EditApplyResult = {
  strategyFamily: EditStrategyFamily;
  status: "applied" | "skipped" | "failed";
  changedFiles: AppliedEditFile[];
  errorCode?: EditErrorCode;
  errorMessage?: string;
  previewDiff: string;
  protectedPathTouched: boolean;
};

type WorkspaceState = {
  files: Map<string, string>;
  protectedPaths: Set<string>;
};

type FilePatch = {
  path: string;
  hunks: string[][];
};

export const applyEdit = (workspaceInput: EditApplyWorkspace, input: EditApplyInput): EditApplyResult => {
  const workspace = parseWorkspace(workspaceInput);
  const parsed = EditApplyInputSchema.parse(input);
  switch (parsed.strategyFamily) {
    case "whole_file":
      return applyWholeFile(workspace, parsed.payload);
    case "exact_replace":
      return applyExactReplace(workspace, parsed.payload);
    case "unified_diff":
      return applyUnifiedDiff(workspace, parsed.payload);
    case "apply_patch":
      return applyApplyPatch(workspace, parsed.payload);
    case "hash_range":
      return applyHashRange(workspace, parsed.payload);
  }
};

const parseWorkspace = (workspaceInput: EditApplyWorkspace): WorkspaceState => {
  const workspace = EditApplyWorkspaceSchema.parse(workspaceInput);
  return {
    files: new Map(workspace.files.map((file) => [file.path, file.content])),
    protectedPaths: new Set(workspace.protectedPaths),
  };
};

const sha256 = (content: string): string => `sha256:${createHash("sha256").update(content).digest("hex")}`;

const isProtected = (workspace: WorkspaceState, path: string): boolean =>
  [...workspace.protectedPaths].some((protectedPath) => path === protectedPath || path.startsWith(`${protectedPath}/`));

const protectedFailure = (strategyFamily: EditStrategyFamily, path: string): EditApplyResult => ({
  strategyFamily,
  status: "failed",
  changedFiles: [],
  errorCode: "protected_path_violation",
  errorMessage: `protected path touched: ${path}`,
  previewDiff: "",
  protectedPathTouched: true,
});

const failure = (
  strategyFamily: EditStrategyFamily,
  errorCode: EditErrorCode,
  errorMessage: string,
  protectedPathTouched = false,
): EditApplyResult => ({
  strategyFamily,
  status: "failed",
  changedFiles: [],
  errorCode,
  errorMessage,
  previewDiff: "",
  protectedPathTouched,
});

const success = (strategyFamily: EditStrategyFamily, changedFiles: AppliedEditFile[]): EditApplyResult => ({
  strategyFamily,
  status: changedFiles.length === 0 ? "skipped" : "applied",
  changedFiles,
  previewDiff: changedFiles.map(renderPreviewDiff).join("\n"),
  protectedPathTouched: false,
});

const contentFor = (workspace: WorkspaceState, strategyFamily: EditStrategyFamily, path: string): string | EditApplyResult => {
  const content = workspace.files.get(path);
  return content === undefined ? failure(strategyFamily, "scope_violation", `file not found: ${path}`) : content;
};

const assertExpectedHash = (
  strategyFamily: EditStrategyFamily,
  content: string,
  expectedHash: string | undefined,
): EditApplyResult | undefined => {
  if (expectedHash !== undefined && expectedHash !== sha256(content)) {
    return failure(strategyFamily, "hash_mismatch", "expected content hash does not match current file content");
  }
  return undefined;
};

const applyWholeFile = (workspace: WorkspaceState, payloadInput: WholeFileEditInput): EditApplyResult => {
  const payload = WholeFileEditInputSchema.parse(payloadInput);
  if (isProtected(workspace, payload.path)) {
    return protectedFailure("whole_file", payload.path);
  }
  const before = workspace.files.get(payload.path);
  if (before !== undefined) {
    const hashFailure = assertExpectedHash("whole_file", before, payload.baseContentHash);
    if (hashFailure !== undefined) {
      return hashFailure;
    }
  }
  if (before === payload.content) {
    return success("whole_file", []);
  }
  return success("whole_file", [
    {
      path: payload.path,
      beforeContent: before,
      afterContent: payload.content,
      changeKind: before === undefined ? "added" : "modified",
    },
  ]);
};

const applyExactReplace = (workspace: WorkspaceState, payloadInput: ExactReplaceEditInput): EditApplyResult => {
  const payload = ExactReplaceEditInputSchema.parse(payloadInput);
  if (isProtected(workspace, payload.path)) {
    return protectedFailure("exact_replace", payload.path);
  }
  const content = contentFor(workspace, "exact_replace", payload.path);
  if (typeof content !== "string") {
    return content;
  }
  const hashFailure = assertExpectedHash("exact_replace", content, payload.expectedContentHash);
  if (hashFailure !== undefined) {
    return hashFailure;
  }
  const count = countOccurrences(content, payload.search);
  if (count === 0) {
    return failure("exact_replace", "exact_match_not_found", "search text was not found");
  }
  if (count > 1) {
    return failure("exact_replace", "exact_match_ambiguous", "search text matched more than once");
  }
  const after = content.replace(payload.search, payload.replace);
  return success("exact_replace", [
    {
      path: payload.path,
      beforeContent: content,
      afterContent: after,
      changeKind: "modified",
    },
  ]);
};

const applyHashRange = (workspace: WorkspaceState, payloadInput: HashRangeEditInput): EditApplyResult => {
  const payload = HashRangeEditInputSchema.parse(payloadInput);
  const working = new Map(workspace.files);
  const changed = new Map<string, AppliedEditFile>();

  for (const operation of payload.operations) {
    if (isProtected(workspace, operation.path)) {
      return protectedFailure("hash_range", operation.path);
    }
    const content = working.get(operation.path);
    if (content === undefined) {
      return failure("hash_range", "scope_violation", `file not found: ${operation.path}`);
    }
    const hashFailure = assertExpectedHash("hash_range", content, operation.expectedContentHash);
    if (hashFailure !== undefined) {
      return hashFailure;
    }
    const lines = splitContentLines(content);
    if (operation.endLine > lines.body.length || operation.startLine > lines.body.length + 1) {
      return failure("hash_range", "range_out_of_bounds", `line range is outside ${operation.path}`);
    }
    const replacementLines = splitReplacementLines(operation.replacement);
    const afterLines = [
      ...lines.body.slice(0, operation.startLine - 1),
      ...replacementLines,
      ...lines.body.slice(operation.endLine),
    ];
    const after = joinContentLines(afterLines, lines.finalNewline);
    working.set(operation.path, after);
    changed.set(operation.path, {
      path: operation.path,
      beforeContent: workspace.files.get(operation.path),
      afterContent: after,
      changeKind: "modified",
    });
  }

  return success("hash_range", [...changed.values()].filter((file) => file.beforeContent !== file.afterContent));
};

const applyUnifiedDiff = (workspace: WorkspaceState, payloadInput: UnifiedDiffEditInput): EditApplyResult => {
  const payload = UnifiedDiffEditInputSchema.parse(payloadInput);
  const parsed = parseUnifiedPatch(payload.patch);
  if ("errorCode" in parsed) {
    return failure("unified_diff", parsed.errorCode, parsed.errorMessage);
  }
  return applyLinePatches("unified_diff", workspace, parsed);
};

const applyApplyPatch = (workspace: WorkspaceState, payloadInput: ApplyPatchEditInput): EditApplyResult => {
  const payload = ApplyPatchEditInputSchema.parse(payloadInput);
  const parsed = parseApplyPatch(payload.patch);
  if ("errorCode" in parsed) {
    return failure("apply_patch", parsed.errorCode, parsed.errorMessage);
  }
  return applyLinePatches("apply_patch", workspace, parsed);
};

const applyLinePatches = (
  strategyFamily: "unified_diff" | "apply_patch",
  workspace: WorkspaceState,
  patches: FilePatch[],
): EditApplyResult => {
  const working = new Map(workspace.files);
  const changed = new Map<string, AppliedEditFile>();

  for (const patch of patches) {
    if (isProtected(workspace, patch.path)) {
      return protectedFailure(strategyFamily, patch.path);
    }
    const content = working.get(patch.path);
    if (content === undefined) {
      return failure(strategyFamily, "scope_violation", `file not found: ${patch.path}`);
    }
    const applied = applyHunksByExactBlock(strategyFamily, content, patch.hunks);
    if (typeof applied !== "string") {
      return applied;
    }
    working.set(patch.path, applied);
    changed.set(patch.path, {
      path: patch.path,
      beforeContent: workspace.files.get(patch.path),
      afterContent: applied,
      changeKind: "modified",
    });
  }

  return success(strategyFamily, [...changed.values()].filter((file) => file.beforeContent !== file.afterContent));
};

const applyHunksByExactBlock = (
  strategyFamily: "unified_diff" | "apply_patch",
  content: string,
  hunks: string[][],
): string | EditApplyResult => {
  let current = content;
  for (const hunk of hunks) {
    const parsed = hunkSearchReplace(hunk);
    if (parsed === undefined) {
      return failure(strategyFamily, "parse_error", "hunk did not include a removable or addable edit");
    }
    const search = parsed.search.join("\n") + (parsed.search.length > 0 ? "\n" : "");
    const replace = parsed.replace.join("\n") + (parsed.replace.length > 0 ? "\n" : "");
    if (search.length === 0) {
      current += replace;
      continue;
    }
    const count = countOccurrences(current, search);
    if (count === 0) {
      return failure(strategyFamily, "hunk_context_mismatch", "hunk context was not found");
    }
    if (count > 1) {
      return failure(strategyFamily, "exact_match_ambiguous", "hunk context matched more than once");
    }
    current = current.replace(search, replace);
  }
  return current;
};

const hunkSearchReplace = (hunk: string[]): { search: string[]; replace: string[] } | undefined => {
  const search: string[] = [];
  const replace: string[] = [];
  let touched = false;
  for (const line of hunk) {
    const marker = line[0];
    const body = line.slice(1);
    if (marker === " ") {
      search.push(body);
      replace.push(body);
      continue;
    }
    if (marker === "-") {
      search.push(body);
      touched = true;
      continue;
    }
    if (marker === "+") {
      replace.push(body);
      touched = true;
      continue;
    }
    return undefined;
  }
  return touched ? { search, replace } : undefined;
};

const parseUnifiedPatch = (patch: string): FilePatch[] | { errorCode: EditErrorCode; errorMessage: string } => {
  const lines = patch.split(/\r?\n/);
  const patches: FilePatch[] = [];
  let index = 0;
  while (index < lines.length) {
    if (lines[index]?.startsWith("--- ") !== true) {
      index += 1;
      continue;
    }
    index += 1;
    const next = lines[index];
    if (next?.startsWith("+++ ") !== true) {
      return { errorCode: "parse_error", errorMessage: "unified patch missing +++ file header" };
    }
    const path = normalizePatchPath(next.slice(4).trim());
    index += 1;
    const hunks: string[][] = [];
    while (index < lines.length && lines[index]?.startsWith("--- ") !== true) {
      if (lines[index]?.startsWith("@@") !== true) {
        index += 1;
        continue;
      }
      index += 1;
      const hunk: string[] = [];
      while (
        index < lines.length &&
        lines[index]?.startsWith("@@") !== true &&
        lines[index]?.startsWith("--- ") !== true
      ) {
        const line = lines[index] ?? "";
        if (line.length > 0 && [" ", "-", "+"].includes(line[0]!)) {
          hunk.push(line);
        }
        index += 1;
      }
      if (hunk.length === 0) {
        return { errorCode: "parse_error", errorMessage: "unified patch contains an empty hunk" };
      }
      hunks.push(hunk);
    }
    if (hunks.length === 0) {
      return { errorCode: "parse_error", errorMessage: "unified patch has no hunks" };
    }
    patches.push({ path, hunks });
  }
  return patches.length === 0 ? { errorCode: "parse_error", errorMessage: "unified patch has no file patches" } : patches;
};

const parseApplyPatch = (patch: string): FilePatch[] | { errorCode: EditErrorCode; errorMessage: string } => {
  const lines = patch.split(/\r?\n/);
  if (lines[0] !== "*** Begin Patch" || !lines.includes("*** End Patch")) {
    return { errorCode: "parse_error", errorMessage: "apply_patch payload must include begin/end markers" };
  }
  const patches: FilePatch[] = [];
  let index = 1;
  while (index < lines.length) {
    const line = lines[index] ?? "";
    if (line === "*** End Patch") {
      break;
    }
    if (!line.startsWith("*** Update File: ")) {
      return { errorCode: "parse_error", errorMessage: `unsupported apply_patch section: ${line}` };
    }
    const path = line.slice("*** Update File: ".length).trim();
    index += 1;
    const hunks: string[][] = [];
    while (index < lines.length && !lines[index]?.startsWith("*** ")) {
      if (lines[index]?.startsWith("@@") !== true) {
        index += 1;
        continue;
      }
      index += 1;
      const hunk: string[] = [];
      while (
        index < lines.length &&
        lines[index]?.startsWith("@@") !== true &&
        !lines[index]?.startsWith("*** ")
      ) {
        const hunkLine = lines[index] ?? "";
        if (hunkLine.length > 0 && [" ", "-", "+"].includes(hunkLine[0]!)) {
          hunk.push(hunkLine);
        }
        index += 1;
      }
      if (hunk.length === 0) {
        return { errorCode: "parse_error", errorMessage: "apply_patch contains an empty hunk" };
      }
      hunks.push(hunk);
    }
    if (hunks.length === 0) {
      return { errorCode: "parse_error", errorMessage: "apply_patch update has no hunks" };
    }
    patches.push({ path, hunks });
  }
  return patches.length === 0 ? { errorCode: "parse_error", errorMessage: "apply_patch has no file updates" } : patches;
};

const normalizePatchPath = (rawPath: string): string => rawPath.replace(/^[ab]\//, "");

const splitContentLines = (content: string): { body: string[]; finalNewline: boolean } => {
  const finalNewline = content.endsWith("\n");
  const body = finalNewline ? content.slice(0, -1).split("\n") : content.split("\n");
  return { body: body.length === 1 && body[0] === "" ? [] : body, finalNewline };
};

const splitReplacementLines = (replacement: string): string[] => {
  const finalNewline = replacement.endsWith("\n");
  const body = finalNewline ? replacement.slice(0, -1).split("\n") : replacement.split("\n");
  return body.length === 1 && body[0] === "" ? [] : body;
};

const joinContentLines = (lines: readonly string[], finalNewline: boolean): string =>
  `${lines.join("\n")}${finalNewline ? "\n" : ""}`;

const countOccurrences = (content: string, search: string): number => {
  let count = 0;
  let index = 0;
  while (true) {
    const found = content.indexOf(search, index);
    if (found < 0) {
      return count;
    }
    count += 1;
    index = found + search.length;
  }
};

const renderPreviewDiff = (file: AppliedEditFile): string => [
  `--- ${file.path}`,
  `+++ ${file.path}`,
  `change: ${file.changeKind}`,
  ...(file.beforeContent === undefined ? [] : [`before-sha256: ${sha256(file.beforeContent)}`]),
  ...(file.afterContent === undefined ? [] : [`after-sha256: ${sha256(file.afterContent)}`]),
].join("\n");

export const editApplySupportedFamilies = (): EditStrategyFamily[] =>
  EditStrategyFamilySchema.options.filter((family) =>
    ["whole_file", "exact_replace", "unified_diff", "apply_patch", "hash_range"].includes(family),
  );

export const parseEditApplyErrorCode = (value: unknown): EditErrorCode => EditErrorCodeSchema.parse(value);
