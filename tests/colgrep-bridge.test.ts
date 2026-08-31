import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  colgrepBackend,
  renderHitsAsObservation,
  type SubprocessRunner,
} from "../src/codebase-index/colgrep-bridge";

type RunnerCall = {
  command: string;
  args: string[];
  cwd: string;
};

const makeMockRunner = (
  responder: (call: RunnerCall) => { stdout: string; stderr: string; exitCode: number | null },
): { runner: SubprocessRunner; calls: RunnerCall[] } => {
  const calls: RunnerCall[] = [];
  const runner: SubprocessRunner = async (input) => {
    calls.push({ command: input.command, args: input.args, cwd: input.cwd });
    return responder({ command: input.command, args: input.args, cwd: input.cwd });
  };
  return { runner, calls };
};

const makeTmpWorkspace = (): string => {
  const dir = mkdtempSync(join(tmpdir(), "bag-colgrep-bridge-"));
  return dir;
};

const cleanup = (dir: string): void => {
  try {
    rmSync(dir, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
};

describe("colgrep-bridge", () => {
  test("isAvailable returns false when colgrep --version fails (exit code 127)", async () => {
    const { runner } = makeMockRunner(() => ({
      stdout: "",
      stderr: "command not found",
      exitCode: 127,
    }));
    const backend = colgrepBackend({ runner });
    expect(await backend.isAvailable()).toBe(false);
  });

  test("isAvailable returns true when colgrep --version succeeds", async () => {
    const { runner } = makeMockRunner(() => ({
      stdout: "colgrep 0.4.2",
      stderr: "",
      exitCode: 0,
    }));
    const backend = colgrepBackend({ runner });
    expect(await backend.isAvailable()).toBe(true);
  });

  test("ensureIndex returns 'skipped' when binary unavailable", async () => {
    const { runner } = makeMockRunner(() => ({ stdout: "", stderr: "", exitCode: 127 }));
    const backend = colgrepBackend({ runner });
    const dir = makeTmpWorkspace();
    try {
      const result = await backend.ensureIndex({ cwd: dir });
      expect(result.status).toBe("skipped");
    } finally {
      cleanup(dir);
    }
  });

  test("ensureIndex on first call (no prior state) does a full init and writes state file", async () => {
    const dir = makeTmpWorkspace();
    try {
      const { runner, calls } = makeMockRunner((call) => {
        if (call.args[0] === "--version") return { stdout: "colgrep 0.4", stderr: "", exitCode: 0 };
        if (call.args[0] === "init") return { stdout: "indexed 12 files", stderr: "", exitCode: 0 };
        if (call.command === "git") return { stdout: "", stderr: "", exitCode: 1 };
        if (call.command === "find") return { stdout: "", stderr: "", exitCode: 0 };
        return { stdout: "", stderr: "unexpected", exitCode: 1 };
      });
      const backend = colgrepBackend({ runner });
      const result = await backend.ensureIndex({ cwd: dir });
      expect(result.status).toBe("fresh");
      // Verify a state file was persisted
      const statePath = join(dir, ".bag/codebase-index/colgrep.idx-state.json");
      expect(existsSync(statePath)).toBe(true);
      const state = JSON.parse(readFileSync(statePath, "utf8")) as {
        lastBuiltAt?: string;
        sourceFingerprint?: string;
      };
      expect(state.lastBuiltAt).toBeDefined();
      expect(state.sourceFingerprint).toBeDefined();
      // We saw an `init` invocation
      const initCall = calls.find((c) => c.args[0] === "init");
      expect(initCall).toBeDefined();
    } finally {
      cleanup(dir);
    }
  });

  test("search throws structured error when binary missing", async () => {
    const { runner } = makeMockRunner(() => ({ stdout: "", stderr: "", exitCode: 127 }));
    const backend = colgrepBackend({ runner });
    const dir = makeTmpWorkspace();
    try {
      await expect(backend.search({ cwd: dir, query: "auth" })).rejects.toThrow(
        /code_search backend unavailable/,
      );
    } finally {
      cleanup(dir);
    }
  });

  test("search parses JSON-array output into typed hits", async () => {
    const dir = makeTmpWorkspace();
    try {
      const fakeHits = [
        {
          file: "src/auth.ts",
          line_start: 12,
          line_end: 28,
          symbol: "verifyToken",
          kind: "function",
          score: 0.91,
          snippet: "export function verifyToken(t: string)",
        },
        {
          file: "src/server.ts",
          lineRange: [44, 60],
          score: 0.71,
        },
      ];
      const { runner, calls } = makeMockRunner((call) => {
        if (call.args[0] === "--version") return { stdout: "colgrep", stderr: "", exitCode: 0 };
        if (call.args[0] === "search") {
          return { stdout: JSON.stringify(fakeHits), stderr: "", exitCode: 0 };
        }
        return { stdout: "", stderr: "", exitCode: 1 };
      });
      const backend = colgrepBackend({ runner });
      const hits = await backend.search({
        cwd: dir,
        query: "auth middleware",
        topK: 5,
        mode: "hybrid",
      });
      expect(hits).toHaveLength(2);
      expect(hits[0]).toMatchObject({
        file: "src/auth.ts",
        lineRange: [12, 28],
        symbol: "verifyToken",
        unitKind: "function",
        score: 0.91,
      });
      expect(hits[1]?.lineRange).toEqual([44, 60]);
      // The CLI was invoked with the right shape
      const searchCall = calls.find((c) => c.args[0] === "search");
      expect(searchCall).toBeDefined();
      expect(searchCall?.args).toContain("--json");
      expect(searchCall?.args).toContain("--top-k");
      expect(searchCall?.args).toContain("5");
      expect(searchCall?.args).toContain("--mode");
      expect(searchCall?.args).toContain("hybrid");
      expect(searchCall?.args).toContain("auth middleware");
    } finally {
      cleanup(dir);
    }
  });

  test("search parses JSONL fallback when stdout is one-hit-per-line", async () => {
    const dir = makeTmpWorkspace();
    try {
      const jsonl = [
        JSON.stringify({ file: "a.py", line: 5, score: 0.5 }),
        JSON.stringify({ file: "b.py", line_start: 3, line_end: 9, score: 0.4 }),
      ].join("\n");
      const { runner } = makeMockRunner((call) => {
        if (call.args[0] === "--version") return { stdout: "colgrep", stderr: "", exitCode: 0 };
        if (call.args[0] === "search") return { stdout: jsonl, stderr: "", exitCode: 0 };
        return { stdout: "", stderr: "", exitCode: 1 };
      });
      const backend = colgrepBackend({ runner });
      const hits = await backend.search({ cwd: dir, query: "thing" });
      expect(hits).toHaveLength(2);
      expect(hits[0]?.lineRange).toEqual([5, 5]);
      expect(hits[1]?.lineRange).toEqual([3, 9]);
    } finally {
      cleanup(dir);
    }
  });

  test("invalidate clears the persisted state file", async () => {
    const dir = makeTmpWorkspace();
    try {
      const { runner } = makeMockRunner((call) => {
        if (call.args[0] === "--version") return { stdout: "colgrep", stderr: "", exitCode: 0 };
        if (call.args[0] === "init") return { stdout: "ok", stderr: "", exitCode: 0 };
        if (call.command === "git") return { stdout: "", stderr: "", exitCode: 1 };
        if (call.command === "find") return { stdout: "", stderr: "", exitCode: 0 };
        return { stdout: "", stderr: "", exitCode: 1 };
      });
      const backend = colgrepBackend({ runner });
      await backend.ensureIndex({ cwd: dir });
      const statePath = join(dir, ".bag/codebase-index/colgrep.idx-state.json");
      expect(existsSync(statePath)).toBe(true);
      // Invalidate writes an empty {} to the state file
      await backend.invalidate?.({ cwd: dir });
      const after = JSON.parse(readFileSync(statePath, "utf8")) as Record<string, unknown>;
      expect(Object.keys(after)).toHaveLength(0);
    } finally {
      cleanup(dir);
    }
  });

  test("renderHitsAsObservation summarizes results compactly", () => {
    const obs = renderHitsAsObservation([
      {
        file: "a.ts",
        lineRange: [1, 5],
        score: 0.9,
        symbol: "foo",
        unitKind: "function",
        snippet: "function foo() {}",
      },
    ]);
    expect(obs).toContain("code_search: 1 hit(s)");
    expect(obs).toContain("a.ts:1-5");
    expect(obs).toContain("[foo function]");
    expect(obs).toContain("function foo() {}");
  });

  test("renderHitsAsObservation handles empty hit list", () => {
    expect(renderHitsAsObservation([])).toBe("code_search: no results.");
  });
});
