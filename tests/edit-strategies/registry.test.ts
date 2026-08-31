import { afterAll, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  EDIT_STRATEGY_IDS,
  createEditStrategy,
  isEditStrategyId,
  type EditContext,
  type EditDispatchOutcome,
} from "../../src/edit-strategies/registry";

const ROOT = mkdtempSync(path.join(tmpdir(), "bag-edit-strategy-"));
afterAll(() => {
  rmSync(ROOT, { recursive: true, force: true });
});

const newCwd = (label: string): string => {
  const dir = path.join(ROOT, `${label}-${Math.random().toString(36).slice(2, 8)}`);
  return dir;
};

const captured = (): {
  ctx: EditContext;
  events: Array<{ outcome: EditDispatchOutcome; tool: string; target: string }>;
} => {
  const events: Array<{ outcome: EditDispatchOutcome; tool: string; target: string }> = [];
  return {
    events,
    ctx: {
      cwd: "",
      emit: (entry) => {
        events.push({ outcome: entry.outcome, tool: entry.tool, target: entry.target });
      },
    },
  };
};

describe("EDIT_STRATEGY_IDS surface", () => {
  test("registry exposes the canonical 5 strategy ids", () => {
    expect([...EDIT_STRATEGY_IDS].sort()).toEqual(
      [
        "shell-heredoc",
        "fs-write-whole-file",
        "edit-tool-stringreplace",
        "apply-patch-unified",
        "edit-diff-blocks",
      ].sort(),
    );
  });

  test("isEditStrategyId narrows correctly", () => {
    expect(isEditStrategyId("shell-heredoc")).toBe(true);
    expect(isEditStrategyId("nope")).toBe(false);
    expect(isEditStrategyId(42)).toBe(false);
  });
});

describe("ShellHeredocStrategy", () => {
  test("declares no tool definitions and reports delegated_to_bash", async () => {
    const strat = createEditStrategy("shell-heredoc");
    expect(strat.toolDefinitions()).toEqual([]);
    expect(strat.systemPromptFragment().toLowerCase()).toContain("shell-heredoc");
    const cwd = newCwd("shell");
    await fs.mkdir(cwd, { recursive: true });
    const result = await strat.dispatch("anything", {}, { cwd });
    expect(result.outcome).toBe("delegated_to_bash");
    expect(result.fallbackToShell).toBe(true);
  });
});

describe("FsWriteWholeFileStrategy", () => {
  test("writes a full file body and reports applied", async () => {
    const strat = createEditStrategy("fs-write-whole-file");
    const cwd = newCwd("fswrite");
    await fs.mkdir(cwd, { recursive: true });
    const cap = captured();
    cap.ctx.cwd = cwd;
    const result = await strat.dispatch(
      "fs_write_text_file",
      { path: "out.txt", content: "hello\nworld\n" },
      cap.ctx,
    );
    expect(result.outcome).toBe("applied");
    expect(result.target).toBe("out.txt");
    expect(result.bytesChanged).toBe(Buffer.byteLength("hello\nworld\n", "utf8"));
    const onDisk = await fs.readFile(path.join(cwd, "out.txt"), "utf8");
    expect(onDisk).toBe("hello\nworld\n");
    expect(cap.events).toHaveLength(1);
    expect(cap.events[0]?.outcome).toBe("applied");
  });

  test("refuses paths that escape the workspace", async () => {
    const strat = createEditStrategy("fs-write-whole-file");
    const cwd = newCwd("fswrite-escape");
    await fs.mkdir(cwd, { recursive: true });
    const result = await strat.dispatch(
      "fs_write_text_file",
      { path: "../escaped.txt", content: "no" },
      { cwd },
    );
    expect(result.outcome).toBe("syntax_error");
    expect(result.observation).toContain("refusing edit outside workspace");
  });

  test("reports syntax_error on missing required args", async () => {
    const strat = createEditStrategy("fs-write-whole-file");
    const cwd = newCwd("fswrite-args");
    await fs.mkdir(cwd, { recursive: true });
    const result = await strat.dispatch("fs_write_text_file", {}, { cwd });
    expect(result.outcome).toBe("syntax_error");
  });
});

describe("EditToolStringReplaceStrategy", () => {
  beforeEach(() => undefined);

  test("applies a single unique replacement", async () => {
    const strat = createEditStrategy("edit-tool-stringreplace");
    const cwd = newCwd("strrepl");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.ts");
    await fs.writeFile(file, "const x = 1;\nconst y = 2;\n");
    const cap = captured();
    cap.ctx.cwd = cwd;
    const result = await strat.dispatch(
      "edit",
      { path: "src.ts", old_string: "const y = 2;", new_string: "const y = 99;" },
      cap.ctx,
    );
    expect(result.outcome).toBe("applied");
    expect(await fs.readFile(file, "utf8")).toBe("const x = 1;\nconst y = 99;\n");
    expect(cap.events[0]?.outcome).toBe("applied");
  });

  test("reports match_failed when old_string is absent", async () => {
    const strat = createEditStrategy("edit-tool-stringreplace");
    const cwd = newCwd("strrepl-miss");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.ts");
    await fs.writeFile(file, "const x = 1;\n");
    const result = await strat.dispatch(
      "edit",
      { path: "src.ts", old_string: "MISSING", new_string: "REPLACED" },
      { cwd },
    );
    expect(result.outcome).toBe("match_failed");
    expect(await fs.readFile(file, "utf8")).toBe("const x = 1;\n");
  });

  test("reports match_failed when old_string is ambiguous without replace_all", async () => {
    const strat = createEditStrategy("edit-tool-stringreplace");
    const cwd = newCwd("strrepl-amb");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.ts");
    await fs.writeFile(file, "AAA\nAAA\nAAA\n");
    const result = await strat.dispatch(
      "edit",
      { path: "src.ts", old_string: "AAA", new_string: "BBB" },
      { cwd },
    );
    expect(result.outcome).toBe("match_failed");
    expect(await fs.readFile(file, "utf8")).toBe("AAA\nAAA\nAAA\n");
  });

  test("replace_all rewrites every occurrence", async () => {
    const strat = createEditStrategy("edit-tool-stringreplace");
    const cwd = newCwd("strrepl-all");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.ts");
    await fs.writeFile(file, "AAA\nAAA\nAAA\n");
    const result = await strat.dispatch(
      "edit",
      { path: "src.ts", old_string: "AAA", new_string: "BBB", replace_all: true },
      { cwd },
    );
    expect(result.outcome).toBe("applied");
    expect(await fs.readFile(file, "utf8")).toBe("BBB\nBBB\nBBB\n");
  });

  test("empty old_string creates the file when missing", async () => {
    const strat = createEditStrategy("edit-tool-stringreplace");
    const cwd = newCwd("strrepl-create");
    await fs.mkdir(cwd, { recursive: true });
    const result = await strat.dispatch(
      "edit",
      { path: "new.txt", old_string: "", new_string: "hello" },
      { cwd },
    );
    expect(result.outcome).toBe("applied");
    expect(await fs.readFile(path.join(cwd, "new.txt"), "utf8")).toBe("hello");
  });

  test("stale_context when file is missing and old_string is non-empty", async () => {
    const strat = createEditStrategy("edit-tool-stringreplace");
    const cwd = newCwd("strrepl-stale");
    await fs.mkdir(cwd, { recursive: true });
    const result = await strat.dispatch(
      "edit",
      { path: "missing.ts", old_string: "anything", new_string: "x" },
      { cwd },
    );
    expect(result.outcome).toBe("stale_context");
  });
});

describe("ApplyPatchUnifiedStrategy", () => {
  test("applies a hunk that matches existing context", async () => {
    const strat = createEditStrategy("apply-patch-unified");
    const cwd = newCwd("apatch");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.ts");
    await fs.writeFile(file, "alpha\nbeta\ngamma\n");
    const patch = [
      "--- a/src.ts",
      "+++ b/src.ts",
      "@@ -1,3 +1,3 @@",
      " alpha",
      "-beta",
      "+BETA",
      " gamma",
      "",
    ].join("\n");
    const cap = captured();
    cap.ctx.cwd = cwd;
    const result = await strat.dispatch("apply_patch", { patch }, cap.ctx);
    expect(result.outcome).toBe("applied");
    expect(await fs.readFile(file, "utf8")).toBe("alpha\nBETA\ngamma\n");
    expect(cap.events[0]?.outcome).toBe("applied");
  });

  test("reports match_failed when context lines do not match", async () => {
    const strat = createEditStrategy("apply-patch-unified");
    const cwd = newCwd("apatch-miss");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.ts");
    await fs.writeFile(file, "alpha\nbeta\ngamma\n");
    const patch = [
      "--- a/src.ts",
      "+++ b/src.ts",
      "@@ -1,3 +1,3 @@",
      " WRONG",
      "-beta",
      "+BETA",
      " gamma",
      "",
    ].join("\n");
    const result = await strat.dispatch("apply_patch", { patch }, { cwd });
    expect(result.outcome).toBe("match_failed");
    // File untouched.
    expect(await fs.readFile(file, "utf8")).toBe("alpha\nbeta\ngamma\n");
  });

  test("creates a new file via /dev/null source", async () => {
    const strat = createEditStrategy("apply-patch-unified");
    const cwd = newCwd("apatch-new");
    await fs.mkdir(cwd, { recursive: true });
    const patch = [
      "--- /dev/null",
      "+++ b/created.txt",
      "@@ -0,0 +1,2 @@",
      "+hello",
      "+world",
      "",
    ].join("\n");
    const result = await strat.dispatch("apply_patch", { patch }, { cwd });
    expect(result.outcome).toBe("applied");
    const out = await fs.readFile(path.join(cwd, "created.txt"), "utf8");
    expect(out.split("\n")[0]).toBe("hello");
    expect(out).toContain("world");
  });

  test("reports syntax_error on malformed diff", async () => {
    const strat = createEditStrategy("apply-patch-unified");
    const cwd = newCwd("apatch-syntax");
    await fs.mkdir(cwd, { recursive: true });
    const result = await strat.dispatch("apply_patch", { patch: "no headers here" }, { cwd });
    expect(result.outcome).toBe("syntax_error");
  });
});

describe("EditDiffBlocksStrategy", () => {
  test("replaces a 1-indexed inclusive line range", async () => {
    const strat = createEditStrategy("edit-diff-blocks");
    const cwd = newCwd("diffblock");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.txt");
    await fs.writeFile(file, "L1\nL2\nL3\nL4\n");
    const cap = captured();
    cap.ctx.cwd = cwd;
    const result = await strat.dispatch(
      "edit_diff_block",
      { path: "src.txt", start_line: 2, end_line: 3, new_content: "X1\nX2\nX3" },
      cap.ctx,
    );
    expect(result.outcome).toBe("applied");
    expect(await fs.readFile(file, "utf8")).toBe("L1\nX1\nX2\nX3\nL4\n");
    expect(cap.events[0]?.outcome).toBe("applied");
  });

  test("reports stale_context when expected_old_block disagrees", async () => {
    const strat = createEditStrategy("edit-diff-blocks");
    const cwd = newCwd("diffblock-stale");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.txt");
    await fs.writeFile(file, "L1\nL2\nL3\n");
    const result = await strat.dispatch(
      "edit_diff_block",
      {
        path: "src.txt",
        start_line: 2,
        end_line: 2,
        new_content: "X",
        expected_old_block: "WRONG",
      },
      { cwd },
    );
    expect(result.outcome).toBe("stale_context");
  });

  test("creates new files via 0,0 sentinel", async () => {
    const strat = createEditStrategy("edit-diff-blocks");
    const cwd = newCwd("diffblock-new");
    await fs.mkdir(cwd, { recursive: true });
    const result = await strat.dispatch(
      "edit_diff_block",
      { path: "fresh.txt", start_line: 0, end_line: 0, new_content: "hi" },
      { cwd },
    );
    expect(result.outcome).toBe("applied");
    expect(await fs.readFile(path.join(cwd, "fresh.txt"), "utf8")).toBe("hi");
  });

  test("reports syntax_error on bad ranges", async () => {
    const strat = createEditStrategy("edit-diff-blocks");
    const cwd = newCwd("diffblock-bad");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.txt");
    await fs.writeFile(file, "L1\nL2\n");
    const result = await strat.dispatch(
      "edit_diff_block",
      { path: "src.txt", start_line: 3, end_line: 2, new_content: "x" },
      { cwd },
    );
    expect(result.outcome).toBe("syntax_error");
  });

  test("reports stale_context when end_line exceeds file length", async () => {
    const strat = createEditStrategy("edit-diff-blocks");
    const cwd = newCwd("diffblock-oob");
    await fs.mkdir(cwd, { recursive: true });
    const file = path.join(cwd, "src.txt");
    await fs.writeFile(file, "L1\nL2\n");
    const result = await strat.dispatch(
      "edit_diff_block",
      { path: "src.txt", start_line: 1, end_line: 99, new_content: "x" },
      { cwd },
    );
    expect(result.outcome).toBe("stale_context");
  });
});
