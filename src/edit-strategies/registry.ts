/**
 * Edit-strategy registry — runtime tool dispatcher.
 *
 * The "autonomous coding turn" historically exposed only `bash`, and the model
 * edited files with shell primitives (`cat <<'EOF' > file`, `sed -i`, etc.).
 * The Aider-Polyglot run forensics showed BAG made 17 terminal_create calls
 * and ZERO `fs/write_text_file` calls — so we have NEVER measured whether
 * shell-edit is actually optimal per-model.
 *
 * This module defines a generic `EditStrategy` interface and ships five
 * concrete strategies that reflect the file-editing modalities used by
 * mainstream coding agents:
 *
 *   1. shell-heredoc          — current default; bash + here-docs
 *   2. fs-write-whole-file    — ACP-style: model emits a full file body
 *   3. edit-tool-stringreplace — Claude Code style: {path, old_string, new_string}
 *   4. apply-patch-unified    — Codex style: unified diff via apply_patch
 *   5. edit-diff-blocks       — Pi-style: model emits block ranges (line-based)
 *
 * The model emits edits via tool calls; the strategy interprets the tool
 * arguments and writes to disk. Every strategy returns an `EditResult` whose
 * `outcome` field is fed to the autonomous-trace `edit_dispatch` telemetry
 * entry so we can measure per-model edit-error rates.
 *
 * Note: this file is intentionally INDEPENDENT from `src/edit-strategy/`
 * (singular) — that module is the heavyweight EditAttemptContract for
 * downstream telemetry. This module is the thin runtime tool dispatcher.
 * The contract module wraps results emitted here into telemetry rows.
 */

import { promises as fs } from "node:fs";
import path from "node:path";

/* ---------- public types --------------------------------------------------- */

export type EditStrategyId =
  | "shell-heredoc"
  | "fs-write-whole-file"
  | "edit-tool-stringreplace"
  | "apply-patch-unified"
  | "edit-diff-blocks";

export const EDIT_STRATEGY_IDS: readonly EditStrategyId[] = [
  "shell-heredoc",
  "fs-write-whole-file",
  "edit-tool-stringreplace",
  "apply-patch-unified",
  "edit-diff-blocks",
] as const;

/**
 * Outcome of dispatching a single tool call. The taxonomy is deliberately
 * coarse — the goal is "did the edit land or not, and if not, why" — finer-
 * grained codes live in `src/edit-strategy/types.ts::EditErrorCodeSchema`
 * for downstream contract-style telemetry.
 */
export type EditDispatchOutcome =
  | "applied"
  | "match_failed"
  | "stale_context"
  | "syntax_error"
  | "permission_denied"
  | "delegated_to_bash";

export type EditResult = {
  outcome: EditDispatchOutcome;
  /** Workspace-relative path the edit targeted (or "" for delegated calls). */
  target: string;
  /** Net signed bytes added (positive) or removed (negative). 0 for failures. */
  bytesChanged: number;
  /** Number of automatic retries attempted within this strategy's apply step. */
  retriesWithinStrategy: number;
  /** Human-readable observation to feed back to the model as a tool result. */
  observation: string;
  /** When set, the strategy reports it should fall back to shell-heredoc. */
  fallbackToShell?: boolean;
};

export type ToolDefinition = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, unknown>;
      required?: string[];
      additionalProperties: false;
    };
  };
};

/**
 * Context handed to `EditStrategy.dispatch`. The strategy uses `cwd` to
 * resolve workspace-relative paths and resolves edits inside that root.
 * Strategies MUST refuse to write outside `cwd` to prevent path traversal.
 */
export type EditContext = {
  cwd: string;
  /** Optional: bag-style telemetry hook for emitting `edit_dispatch` entries. */
  emit?: (entry: {
    strategy: EditStrategyId;
    tool: string;
    target: string;
    outcome: EditDispatchOutcome;
    bytesChanged: number;
    retriesWithinStrategy: number;
  }) => void;
};

export interface EditStrategy {
  readonly id: EditStrategyId;
  toolDefinitions(): ToolDefinition[];
  dispatch(toolName: string, args: unknown, ctx: EditContext): Promise<EditResult>;
  /**
   * Tactic fragment appended to the executor system prompt that explains how
   * to use this strategy's tool(s). Kept short — the goal is one paragraph
   * that disambiguates from bash. Returned without a trailing newline.
   */
  systemPromptFragment(): string;
}

/* ---------- shared helpers ------------------------------------------------- */

const ensureInsideCwd = (cwd: string, target: string): string => {
  const absolute = path.isAbsolute(target)
    ? path.normalize(target)
    : path.normalize(path.join(cwd, target));
  const cwdNormalized = path.normalize(cwd);
  const rel = path.relative(cwdNormalized, absolute);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error(`refusing edit outside workspace: ${target}`);
  }
  return absolute;
};

const readIfExists = async (absolute: string): Promise<string | null> => {
  try {
    return await fs.readFile(absolute, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw err;
  }
};

const byteDelta = (before: string | null, after: string): number => {
  return Buffer.byteLength(after, "utf8") - (before == null ? 0 : Buffer.byteLength(before, "utf8"));
};

const writeAtomic = async (absolute: string, content: string): Promise<void> => {
  await fs.mkdir(path.dirname(absolute), { recursive: true });
  await fs.writeFile(absolute, content, "utf8");
};

const stringFromArg = (value: unknown, key: string): string => {
  if (typeof value !== "string") {
    throw new Error(`argument '${key}' must be a string`);
  }
  return value;
};

const optionalStringFromArg = (value: unknown, key: string): string | undefined => {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string") {
    throw new Error(`argument '${key}' must be a string when provided`);
  }
  return value;
};

/* ---------- 1. shell-heredoc (default) ------------------------------------ */

/**
 * The current state of the world. The autonomous-coding-turn already exposes
 * `bash`; this strategy declares NO additional tool definitions and reports
 * `delegated_to_bash` on every dispatch — its only effect is to inform the
 * harness that the bash tool MUST stay registered.
 *
 * We keep it as a first-class strategy so the registry is uniform: the
 * harness queries `strategy.toolDefinitions()` and `strategy.systemPromptFragment()`
 * without conditional logic, and the bench sweep can compare it directly
 * against the structured strategies.
 */
export class ShellHeredocStrategy implements EditStrategy {
  readonly id: EditStrategyId = "shell-heredoc";

  toolDefinitions(): ToolDefinition[] {
    // Bash is registered separately by the autonomous-coding-turn; this
    // strategy does not introduce any new tool surface.
    return [];
  }

  async dispatch(toolName: string, _args: unknown, _ctx: EditContext): Promise<EditResult> {
    return {
      outcome: "delegated_to_bash",
      target: "",
      bytesChanged: 0,
      retriesWithinStrategy: 0,
      observation: `shell-heredoc strategy does not handle structured tool '${toolName}'; use bash here-docs / sed / printf instead.`,
      fallbackToShell: true,
    };
  }

  systemPromptFragment(): string {
    return [
      "EDIT MODE: shell-heredoc.",
      "Edit files via the existing `bash` tool only. Use here-docs (`cat <<'EOF' > file ... EOF`),",
      "`sed -i`, `printf >>`, or `echo >>`. No structured edit tool is available; do not attempt",
      "to call `fs/write_text_file`, `edit`, or `apply_patch` — they are not registered.",
    ].join(" ");
  }
}

/* ---------- 2. fs-write-whole-file (ACP-style) ---------------------------- */

const WRITE_TEXT_FILE_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: "fs_write_text_file",
    description: [
      "Write a full text file to disk. The 'content' field must contain the COMPLETE final file body.",
      "Use this tool for ALL file edits — single-line tweaks AND large rewrites. The runtime overwrites",
      "the file atomically; line numbers, partial diffs, and 'unchanged' regions are not supported.",
      "Path is relative to the workspace root and must not escape it (no '..').",
    ].join(" "),
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Workspace-relative file path." },
        content: { type: "string", description: "Full file body to write." },
      },
      required: ["path", "content"],
      additionalProperties: false,
    },
  },
};

export class FsWriteWholeFileStrategy implements EditStrategy {
  readonly id: EditStrategyId = "fs-write-whole-file";

  toolDefinitions(): ToolDefinition[] {
    return [WRITE_TEXT_FILE_TOOL];
  }

  async dispatch(toolName: string, args: unknown, ctx: EditContext): Promise<EditResult> {
    if (toolName !== WRITE_TEXT_FILE_TOOL.function.name) {
      return delegatedResult(toolName);
    }
    const obj = (args ?? {}) as Record<string, unknown>;
    let target = "";
    try {
      const targetArg = stringFromArg(obj.path, "path");
      const content = stringFromArg(obj.content, "content");
      target = targetArg;
      const absolute = ensureInsideCwd(ctx.cwd, targetArg);
      const before = await readIfExists(absolute);
      await writeAtomic(absolute, content);
      const delta = byteDelta(before, content);
      const result: EditResult = {
        outcome: "applied",
        target: targetArg,
        bytesChanged: delta,
        retriesWithinStrategy: 0,
        observation: `wrote ${targetArg} (${content.length} chars, delta ${delta} bytes)`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const result: EditResult = {
        outcome: "syntax_error",
        target,
        bytesChanged: 0,
        retriesWithinStrategy: 0,
        observation: `fs_write_text_file failed: ${message}`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    }
  }

  systemPromptFragment(): string {
    return [
      "EDIT MODE: fs-write-whole-file.",
      "Edit files exclusively via the `fs_write_text_file` tool. Each call REPLACES the file with the",
      "full body you supply, so always include the entire final content — even unchanged lines.",
      "Do NOT use bash to mutate files; bash is for reading, listing, and running tests only.",
    ].join(" ");
  }
}

/* ---------- 3. edit-tool-stringreplace (Claude Code style) ---------------- */

const STRING_REPLACE_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: "edit",
    description: [
      "Replace exactly one occurrence of `old_string` with `new_string` in the file at `path`.",
      "The match is LITERAL (no regex). If `old_string` is empty, the file is created with `new_string` as its body.",
      "If `old_string` appears zero or multiple times, the call FAILS — make `old_string` unique by including",
      "enough surrounding context (typically 3-5 lines). Preserve indentation/whitespace exactly.",
      "Set `replace_all=true` only when you intentionally want every occurrence replaced.",
    ].join(" "),
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Workspace-relative file path." },
        old_string: { type: "string", description: "Exact text to find. Must be unique unless replace_all=true." },
        new_string: { type: "string", description: "Replacement text." },
        replace_all: { type: "boolean", description: "Replace every occurrence. Default false." },
      },
      required: ["path", "old_string", "new_string"],
      additionalProperties: false,
    },
  },
};

export class EditToolStringReplaceStrategy implements EditStrategy {
  readonly id: EditStrategyId = "edit-tool-stringreplace";

  toolDefinitions(): ToolDefinition[] {
    return [STRING_REPLACE_TOOL];
  }

  async dispatch(toolName: string, args: unknown, ctx: EditContext): Promise<EditResult> {
    if (toolName !== STRING_REPLACE_TOOL.function.name) {
      return delegatedResult(toolName);
    }
    const obj = (args ?? {}) as Record<string, unknown>;
    let target = "";
    try {
      const targetArg = stringFromArg(obj.path, "path");
      const oldString = stringFromArg(obj.old_string, "old_string");
      const newString = stringFromArg(obj.new_string, "new_string");
      const replaceAll = obj.replace_all === true;
      target = targetArg;
      const absolute = ensureInsideCwd(ctx.cwd, targetArg);
      const before = await readIfExists(absolute);
      // Empty old_string == file-create semantics.
      if (oldString.length === 0) {
        if (before !== null && before.length > 0) {
          const result: EditResult = {
            outcome: "match_failed",
            target: targetArg,
            bytesChanged: 0,
            retriesWithinStrategy: 0,
            observation: `edit refused: old_string is empty but ${targetArg} already has content. Pass a non-empty old_string to anchor the edit.`,
          };
          ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
          return result;
        }
        await writeAtomic(absolute, newString);
        const result: EditResult = {
          outcome: "applied",
          target: targetArg,
          bytesChanged: byteDelta(before, newString),
          retriesWithinStrategy: 0,
          observation: `created ${targetArg} (${newString.length} chars)`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      if (before === null) {
        const result: EditResult = {
          outcome: "stale_context",
          target: targetArg,
          bytesChanged: 0,
          retriesWithinStrategy: 0,
          observation: `edit refused: ${targetArg} does not exist. Pass an empty old_string to create it.`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      const occurrences = countOccurrences(before, oldString);
      if (occurrences === 0) {
        const result: EditResult = {
          outcome: "match_failed",
          target: targetArg,
          bytesChanged: 0,
          retriesWithinStrategy: 0,
          observation: `edit refused: old_string not found in ${targetArg}. Read the file again and supply a unique anchor.`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      if (occurrences > 1 && !replaceAll) {
        const result: EditResult = {
          outcome: "match_failed",
          target: targetArg,
          bytesChanged: 0,
          retriesWithinStrategy: 0,
          observation: `edit refused: old_string occurs ${occurrences}x in ${targetArg}. Add surrounding context or set replace_all=true.`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      const after = replaceAll
        ? before.split(oldString).join(newString)
        : before.replace(oldString, newString);
      await writeAtomic(absolute, after);
      const result: EditResult = {
        outcome: "applied",
        target: targetArg,
        bytesChanged: byteDelta(before, after),
        retriesWithinStrategy: 0,
        observation: `edit applied to ${targetArg} (${replaceAll ? `${occurrences} replacements` : "1 replacement"}, delta ${byteDelta(before, after)} bytes)`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const result: EditResult = {
        outcome: "syntax_error",
        target,
        bytesChanged: 0,
        retriesWithinStrategy: 0,
        observation: `edit failed: ${message}`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    }
  }

  systemPromptFragment(): string {
    return [
      "EDIT MODE: edit-tool-stringreplace.",
      "Edit files exclusively via the `edit` tool: supply {path, old_string, new_string}. The match is",
      "literal — preserve indentation and whitespace. To create a new file pass an empty old_string.",
      "Do NOT use bash to mutate files; bash is for reading, listing, and running tests only.",
    ].join(" ");
  }
}

/* ---------- 4. apply-patch-unified (Codex-style) -------------------------- */

const APPLY_PATCH_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: "apply_patch",
    description: [
      "Apply a unified-diff patch to one or more files. The 'patch' argument must be a complete unified diff",
      "starting with a `--- a/<path>` / `+++ b/<path>` header per file, followed by `@@` hunks.",
      "Hunk context lines (no leading +/-) must match the current file content EXACTLY. New files use",
      "`--- /dev/null` and `+++ b/<path>` headers. Multi-file patches are supported in a single call.",
    ].join(" "),
    parameters: {
      type: "object",
      properties: {
        patch: { type: "string", description: "Unified diff text (full --- / +++ / @@ format)." },
      },
      required: ["patch"],
      additionalProperties: false,
    },
  },
};

type ParsedPatchHunk = {
  oldStart: number;
  oldLines: string[];
  newLines: string[];
};

type ParsedPatchFile = {
  path: string;
  isNewFile: boolean;
  isDelete: boolean;
  hunks: ParsedPatchHunk[];
};

const parseUnifiedDiff = (patch: string): ParsedPatchFile[] => {
  const files: ParsedPatchFile[] = [];
  const lines = patch.split(/\r?\n/);
  let i = 0;
  let current: ParsedPatchFile | null = null;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    if (line.startsWith("--- ")) {
      const header = line.slice(4).trim();
      const next = lines[i + 1] ?? "";
      if (!next.startsWith("+++ ")) {
        throw new Error(`malformed diff: expected '+++ ' after '--- ' at line ${i + 1}`);
      }
      const newHeader = next.slice(4).trim();
      const isNewFile = header === "/dev/null";
      const isDelete = newHeader === "/dev/null";
      const stripPrefix = (h: string) => (h.startsWith("a/") || h.startsWith("b/") ? h.slice(2) : h);
      const targetPath = isDelete ? stripPrefix(header) : stripPrefix(newHeader);
      current = { path: targetPath, isNewFile, isDelete, hunks: [] };
      files.push(current);
      i += 2;
      continue;
    }
    if (line.startsWith("@@")) {
      if (current === null) {
        throw new Error(`malformed diff: hunk before file header at line ${i + 1}`);
      }
      const headerMatch = line.match(/^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@/);
      if (headerMatch === null) {
        throw new Error(`malformed hunk header at line ${i + 1}: ${line}`);
      }
      const oldStart = Number.parseInt(headerMatch[1] ?? "1", 10);
      const oldLines: string[] = [];
      const newLines: string[] = [];
      i += 1;
      while (i < lines.length) {
        const hunkLine = lines[i] ?? "";
        if (hunkLine.startsWith("@@") || hunkLine.startsWith("--- ")) break;
        if (hunkLine.length === 0) {
          oldLines.push("");
          newLines.push("");
        } else if (hunkLine.startsWith(" ")) {
          oldLines.push(hunkLine.slice(1));
          newLines.push(hunkLine.slice(1));
        } else if (hunkLine.startsWith("-")) {
          oldLines.push(hunkLine.slice(1));
        } else if (hunkLine.startsWith("+")) {
          newLines.push(hunkLine.slice(1));
        } else if (hunkLine.startsWith("\\")) {
          // "\ No newline at end of file" — ignored for now.
        } else {
          break;
        }
        i += 1;
      }
      current.hunks.push({ oldStart, oldLines, newLines });
      continue;
    }
    i += 1;
  }
  if (files.length === 0) {
    throw new Error("no file headers found in patch");
  }
  return files;
};

const applyHunk = (
  fileLines: string[],
  hunk: ParsedPatchHunk,
): { applied: boolean; result: string[] } => {
  // Try at the hunk's reported oldStart first, then scan ±50 lines as a tolerance.
  const startCandidates = [hunk.oldStart - 1];
  for (let offset = 1; offset <= 50; offset += 1) {
    startCandidates.push(hunk.oldStart - 1 - offset);
    startCandidates.push(hunk.oldStart - 1 + offset);
  }
  for (const start of startCandidates) {
    if (start < 0) continue;
    if (start + hunk.oldLines.length > fileLines.length) continue;
    let matches = true;
    for (let k = 0; k < hunk.oldLines.length; k += 1) {
      if (fileLines[start + k] !== hunk.oldLines[k]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      const result = [
        ...fileLines.slice(0, start),
        ...hunk.newLines,
        ...fileLines.slice(start + hunk.oldLines.length),
      ];
      return { applied: true, result };
    }
  }
  return { applied: false, result: fileLines };
};

export class ApplyPatchUnifiedStrategy implements EditStrategy {
  readonly id: EditStrategyId = "apply-patch-unified";

  toolDefinitions(): ToolDefinition[] {
    return [APPLY_PATCH_TOOL];
  }

  async dispatch(toolName: string, args: unknown, ctx: EditContext): Promise<EditResult> {
    if (toolName !== APPLY_PATCH_TOOL.function.name) {
      return delegatedResult(toolName);
    }
    const obj = (args ?? {}) as Record<string, unknown>;
    let firstTarget = "";
    try {
      const patch = stringFromArg(obj.patch, "patch");
      const files = parseUnifiedDiff(patch);
      firstTarget = files[0]?.path ?? "";
      let totalDelta = 0;
      const targets: string[] = [];
      for (const file of files) {
        const absolute = ensureInsideCwd(ctx.cwd, file.path);
        const before = await readIfExists(absolute);
        if (file.isDelete) {
          if (before === null) {
            const result: EditResult = {
              outcome: "stale_context",
              target: file.path,
              bytesChanged: 0,
              retriesWithinStrategy: 0,
              observation: `apply_patch: cannot delete ${file.path} — file not found.`,
            };
            ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
            return result;
          }
          await fs.unlink(absolute);
          totalDelta -= Buffer.byteLength(before, "utf8");
          targets.push(file.path);
          continue;
        }
        let lines = before === null ? [] : before.split("\n");
        if (file.isNewFile && before !== null) {
          const result: EditResult = {
            outcome: "match_failed",
            target: file.path,
            bytesChanged: 0,
            retriesWithinStrategy: 0,
            observation: `apply_patch: ${file.path} marked as new file but already exists.`,
          };
          ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
          return result;
        }
        // New-file hunks: oldStart=0 with empty oldLines. Build the body by
        // concatenating the hunks' newLines instead of running applyHunk on an
        // empty file (`applyHunk` searches for context which doesn't exist).
        if (file.isNewFile) {
          const newBody: string[] = [];
          for (const hunk of file.hunks) {
            for (const line of hunk.newLines) newBody.push(line);
          }
          lines = newBody;
        } else {
          for (const hunk of file.hunks) {
            const applied = applyHunk(lines, hunk);
            if (!applied.applied) {
              const result: EditResult = {
                outcome: "match_failed",
                target: file.path,
                bytesChanged: 0,
                retriesWithinStrategy: 0,
                observation: `apply_patch: hunk @@ -${hunk.oldStart} did not match in ${file.path}. Re-read the file and regenerate the diff.`,
              };
              ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
              return result;
            }
            lines = applied.result;
          }
        }
        const after = lines.join("\n");
        await writeAtomic(absolute, after);
        totalDelta += byteDelta(before, after);
        targets.push(file.path);
      }
      const result: EditResult = {
        outcome: "applied",
        target: firstTarget,
        bytesChanged: totalDelta,
        retriesWithinStrategy: 0,
        observation: `apply_patch: applied to ${targets.length} file(s) (${targets.join(", ")}); delta ${totalDelta} bytes`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const result: EditResult = {
        outcome: "syntax_error",
        target: firstTarget,
        bytesChanged: 0,
        retriesWithinStrategy: 0,
        observation: `apply_patch failed: ${message}`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    }
  }

  systemPromptFragment(): string {
    return [
      "EDIT MODE: apply-patch-unified.",
      "Edit files exclusively via the `apply_patch` tool with a unified diff. Each diff must include a",
      "`--- a/<path>` / `+++ b/<path>` header per file and `@@` hunks. Context lines must match the file",
      "EXACTLY. Use `--- /dev/null` to add a new file. Do NOT use bash to mutate files.",
    ].join(" ");
  }
}

/* ---------- 5. edit-diff-blocks (Pi-style) -------------------------------- */

const EDIT_DIFF_BLOCK_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: "edit_diff_block",
    description: [
      "Replace lines [start_line, end_line] (1-indexed, inclusive) of `path` with `new_content`.",
      "The runtime reads the file, replaces ONLY the given line range, and writes the result back.",
      "Useful when you know the exact line span you want to change and want to avoid re-emitting whole files.",
      "Pass start_line=0 and end_line=0 with a non-existent path to create a new file from `new_content`.",
      "If end_line < start_line or the range exceeds the file length, the call fails.",
    ].join(" "),
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Workspace-relative file path." },
        start_line: { type: "number", description: "1-indexed inclusive start of the replaced span (0 = whole-file create)." },
        end_line: { type: "number", description: "1-indexed inclusive end of the replaced span (0 = whole-file create)." },
        new_content: { type: "string", description: "Replacement block (multiple lines allowed; trailing newline optional)." },
        expected_old_block: {
          type: "string",
          description: "Optional: expected current content of the line range. When provided, the call refuses if the file does not match (stale-context guard).",
        },
      },
      required: ["path", "start_line", "end_line", "new_content"],
      additionalProperties: false,
    },
  },
};

const numberFromArg = (value: unknown, key: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`argument '${key}' must be a finite number`);
  }
  return Math.trunc(value);
};

export class EditDiffBlocksStrategy implements EditStrategy {
  readonly id: EditStrategyId = "edit-diff-blocks";

  toolDefinitions(): ToolDefinition[] {
    return [EDIT_DIFF_BLOCK_TOOL];
  }

  async dispatch(toolName: string, args: unknown, ctx: EditContext): Promise<EditResult> {
    if (toolName !== EDIT_DIFF_BLOCK_TOOL.function.name) {
      return delegatedResult(toolName);
    }
    const obj = (args ?? {}) as Record<string, unknown>;
    let target = "";
    try {
      const targetArg = stringFromArg(obj.path, "path");
      const startLine = numberFromArg(obj.start_line, "start_line");
      const endLine = numberFromArg(obj.end_line, "end_line");
      const newContent = stringFromArg(obj.new_content, "new_content");
      const expectedOldBlock = optionalStringFromArg(obj.expected_old_block, "expected_old_block");
      target = targetArg;
      const absolute = ensureInsideCwd(ctx.cwd, targetArg);
      const before = await readIfExists(absolute);
      // Whole-file create.
      if (startLine === 0 && endLine === 0) {
        if (before !== null && before.length > 0) {
          const result: EditResult = {
            outcome: "match_failed",
            target: targetArg,
            bytesChanged: 0,
            retriesWithinStrategy: 0,
            observation: `edit_diff_block: cannot create ${targetArg} — file already exists. Use a real line range instead.`,
          };
          ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
          return result;
        }
        await writeAtomic(absolute, newContent);
        const result: EditResult = {
          outcome: "applied",
          target: targetArg,
          bytesChanged: byteDelta(before, newContent),
          retriesWithinStrategy: 0,
          observation: `edit_diff_block: created ${targetArg} (${newContent.length} chars)`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      if (before === null) {
        const result: EditResult = {
          outcome: "stale_context",
          target: targetArg,
          bytesChanged: 0,
          retriesWithinStrategy: 0,
          observation: `edit_diff_block: ${targetArg} does not exist. Use start_line=0 end_line=0 to create it.`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      if (startLine < 1 || endLine < startLine) {
        const result: EditResult = {
          outcome: "syntax_error",
          target: targetArg,
          bytesChanged: 0,
          retriesWithinStrategy: 0,
          observation: `edit_diff_block: bad range [${startLine}, ${endLine}]; must be 1-indexed and end_line >= start_line.`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      const lines = before.split("\n");
      if (endLine > lines.length) {
        const result: EditResult = {
          outcome: "stale_context",
          target: targetArg,
          bytesChanged: 0,
          retriesWithinStrategy: 0,
          observation: `edit_diff_block: end_line ${endLine} exceeds file length ${lines.length}. Re-read ${targetArg}.`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      const oldBlock = lines.slice(startLine - 1, endLine).join("\n");
      if (expectedOldBlock !== undefined && expectedOldBlock !== oldBlock) {
        const result: EditResult = {
          outcome: "stale_context",
          target: targetArg,
          bytesChanged: 0,
          retriesWithinStrategy: 0,
          observation: `edit_diff_block: expected_old_block does not match lines [${startLine}, ${endLine}] of ${targetArg}. Re-read the file.`,
        };
        ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
        return result;
      }
      const newBlockLines = newContent.endsWith("\n")
        ? newContent.slice(0, -1).split("\n")
        : newContent.split("\n");
      const after = [
        ...lines.slice(0, startLine - 1),
        ...newBlockLines,
        ...lines.slice(endLine),
      ].join("\n");
      await writeAtomic(absolute, after);
      const result: EditResult = {
        outcome: "applied",
        target: targetArg,
        bytesChanged: byteDelta(before, after),
        retriesWithinStrategy: 0,
        observation: `edit_diff_block: replaced lines [${startLine}, ${endLine}] of ${targetArg} (delta ${byteDelta(before, after)} bytes)`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const result: EditResult = {
        outcome: "syntax_error",
        target,
        bytesChanged: 0,
        retriesWithinStrategy: 0,
        observation: `edit_diff_block failed: ${message}`,
      };
      ctx.emit?.({ strategy: this.id, tool: toolName, ...resultEvent(result) });
      return result;
    }
  }

  systemPromptFragment(): string {
    return [
      "EDIT MODE: edit-diff-blocks.",
      "Edit files exclusively via the `edit_diff_block` tool: supply {path, start_line, end_line, new_content}.",
      "Lines are 1-indexed and the range is inclusive. Pass start_line=0 end_line=0 to create a new file. Use the",
      "optional `expected_old_block` for a stale-context guard. Do NOT use bash to mutate files.",
    ].join(" ");
  }
}

/* ---------- registry & factories ------------------------------------------ */

const STRATEGIES: Readonly<Record<EditStrategyId, () => EditStrategy>> = {
  "shell-heredoc": () => new ShellHeredocStrategy(),
  "fs-write-whole-file": () => new FsWriteWholeFileStrategy(),
  "edit-tool-stringreplace": () => new EditToolStringReplaceStrategy(),
  "apply-patch-unified": () => new ApplyPatchUnifiedStrategy(),
  "edit-diff-blocks": () => new EditDiffBlocksStrategy(),
};

export const createEditStrategy = (id: EditStrategyId): EditStrategy => {
  const factory = STRATEGIES[id];
  if (factory === undefined) {
    throw new Error(`unknown edit strategy: ${id}`);
  }
  return factory();
};

export const isEditStrategyId = (value: unknown): value is EditStrategyId => {
  return typeof value === "string" && (EDIT_STRATEGY_IDS as readonly string[]).includes(value);
};

/* ---------- internal helpers ---------------------------------------------- */

const delegatedResult = (toolName: string): EditResult => ({
  outcome: "delegated_to_bash",
  target: "",
  bytesChanged: 0,
  retriesWithinStrategy: 0,
  observation: `tool '${toolName}' is not registered by this edit strategy.`,
  fallbackToShell: true,
});

const resultEvent = (
  r: EditResult,
): { target: string; outcome: EditDispatchOutcome; bytesChanged: number; retriesWithinStrategy: number } => ({
  target: r.target,
  outcome: r.outcome,
  bytesChanged: r.bytesChanged,
  retriesWithinStrategy: r.retriesWithinStrategy,
});

const countOccurrences = (haystack: string, needle: string): number => {
  if (needle.length === 0) return 0;
  let count = 0;
  let idx = 0;
  while (true) {
    const next = haystack.indexOf(needle, idx);
    if (next === -1) break;
    count += 1;
    idx = next + needle.length;
  }
  return count;
};
