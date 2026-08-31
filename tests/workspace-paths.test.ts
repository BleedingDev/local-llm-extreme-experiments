import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveSessionPath, sessionRelativePath, WorkspacePathError } from "../src/workspace-paths";

describe("workspace path safety", () => {
  test("allows files inside cwd and additional directories", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-path-root-"));
    const extra = mkdtempSync(join(tmpdir(), "bag-path-extra-"));
    writeFileSync(join(cwd, "a.ts"), "export const a = 1;\n");
    writeFileSync(join(extra, "b.ts"), "export const b = 1;\n");

    expect(resolveSessionPath({ cwd, path: "a.ts" })).toBe(join(cwd, "a.ts"));
    expect(resolveSessionPath({ cwd, additionalDirectories: [extra], path: join(extra, "b.ts") })).toBe(
      join(extra, "b.ts"),
    );
    expect(sessionRelativePath(cwd, [extra], join(cwd, "a.ts"))).toBe("a.ts");
    expect(sessionRelativePath(cwd, [extra], join(extra, "b.ts"))).toBe(join(extra, "b.ts"));
  });

  test("rejects parent traversal, absolute outside paths, directories as files, and symlink escapes", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-path-safe-"));
    const outside = mkdtempSync(join(tmpdir(), "bag-path-outside-"));
    mkdirSync(join(cwd, "src"));
    writeFileSync(join(outside, "secret.txt"), "secret\n");
    symlinkSync(join(outside, "secret.txt"), join(cwd, "src", "secret-link.txt"));

    expect(() => resolveSessionPath({ cwd, path: "../secret.txt" })).toThrow(WorkspacePathError);
    expect(() => resolveSessionPath({ cwd, path: join(outside, "secret.txt") })).toThrow(WorkspacePathError);
    expect(() => resolveSessionPath({ cwd, path: "src" })).toThrow(WorkspacePathError);
    expect(() => resolveSessionPath({ cwd, path: "src/secret-link.txt" })).toThrow(WorkspacePathError);
  });

  test("requires terminal cwd paths to be existing directories inside allowed roots", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-path-terminal-"));
    mkdirSync(join(cwd, "work"));
    writeFileSync(join(cwd, "file.txt"), "x\n");

    expect(resolveSessionPath({ cwd, path: "work", kind: "directory" })).toBe(join(cwd, "work"));
    expect(() => resolveSessionPath({ cwd, path: "file.txt", kind: "directory" })).toThrow(WorkspacePathError);
    expect(() => resolveSessionPath({ cwd, path: "missing", kind: "directory" })).toThrow(WorkspacePathError);
  });
});
