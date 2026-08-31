import { describe, expect, test } from "bun:test";
import {
  auditScratchHygiene,
  renderScratchHygieneBlock,
} from "../src/scratch-hygiene";
import type { BashTraceTailEntry } from "../src/pre-submit-self-check";

const trace = (
  entries: Array<{ command: string; output?: string; exitCode?: number | null }>,
): BashTraceTailEntry[] =>
  entries.map((e) => ({
    command: e.command,
    output: e.output ?? "",
    exitCode: e.exitCode === undefined ? 0 : e.exitCode,
  }));

describe("auditScratchHygiene — /tmp/ writes", () => {
  test("flags a redirected write to /tmp/ that is never cleaned up", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "echo hello > /tmp/build_log.txt" },
        { command: "ls /app" },
      ]),
    );
    expect(result.tmpWrites).toEqual([
      { path: "/tmp/build_log.txt", commandIdx: 1 },
    ]);
  });

  test("flags cat-heredoc into /tmp/ when no cleanup follows", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "cat > /tmp/repro.py <<'EOF'\nprint(1)\nEOF" },
        { command: "python3 /tmp/repro.py" },
      ]),
    );
    expect(result.tmpWrites.map((w) => w.path)).toEqual(["/tmp/repro.py"]);
  });

  test("flags cp/mv/mkdir/touch/tee to /tmp/", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "cp /app/foo.py /tmp/foo.py.c" },
        { command: "mv build/output /tmp/snapshot" },
        { command: "mkdir -p /tmp/scratch_dir" },
        { command: "touch /tmp/marker" },
        { command: "echo log | tee /tmp/run.log" },
      ]),
    );
    const paths = result.tmpWrites.map((w) => w.path).sort();
    expect(paths).toEqual([
      "/tmp/foo.py.c",
      "/tmp/marker",
      "/tmp/run.log",
      "/tmp/scratch_dir",
      "/tmp/snapshot",
    ]);
  });

  test("does NOT flag a /tmp/ write that is later removed", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "echo data > /tmp/scratch.txt" },
        { command: "cat /tmp/scratch.txt" },
        { command: "rm -f /tmp/scratch.txt" },
      ]),
    );
    expect(result.tmpWrites).toEqual([]);
  });

  test("does NOT flag /tmp/ writes when a sweeping rm -rf /tmp/* follows", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "echo a > /tmp/a.txt" },
        { command: "echo b > /tmp/b.txt" },
        { command: "rm -rf /tmp/*" },
      ]),
    );
    expect(result.tmpWrites).toEqual([]);
  });

  test("flags only the still-present writes when SOME are cleaned up", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "echo data > /tmp/keep.txt" },
        { command: "echo data > /tmp/drop.txt" },
        { command: "rm -f /tmp/drop.txt" },
      ]),
    );
    expect(result.tmpWrites.map((w) => w.path)).toEqual(["/tmp/keep.txt"]);
  });

  test("does not match writes outside /tmp/", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "echo data > /app/output.json" },
        { command: "cp file /var/log/x" },
      ]),
    );
    expect(result.tmpWrites).toEqual([]);
  });
});

describe("auditScratchHygiene — tracebacks and panics", () => {
  test("captures a Python Traceback signature line", () => {
    const result = auditScratchHygiene(
      trace([
        {
          command: "pytest -x",
          output: [
            "============================ test session starts =============================",
            "Traceback (most recent call last):",
            "  File \"/app/pyknotid/cinvariants.py\", line 12, in <module>",
            "    from numpy.array_api import asarray",
            "AttributeError: module 'numpy' has no attribute 'array_api'",
          ].join("\n"),
          exitCode: 1,
        },
      ]),
    );
    expect(result.tracebacks).toHaveLength(1);
    expect(result.tracebacks[0].commandIdx).toBe(1);
    expect(result.tracebacks[0].signature).toContain("AttributeError");
    expect(result.tracebacks[0].signature).toContain("numpy");
  });

  test("captures a Go panic", () => {
    const result = auditScratchHygiene(
      trace([
        {
          command: "go run main.go",
          output: "panic: runtime error: index out of range [5] with length 3\n",
          exitCode: 2,
        },
      ]),
    );
    expect(result.tracebacks[0].signature).toContain("panic:");
  });

  test("captures a Rust panic", () => {
    const result = auditScratchHygiene(
      trace([
        {
          command: "cargo run",
          output: "thread 'main' panicked at 'unwrap on None', src/main.rs:4:14\n",
          exitCode: 101,
        },
      ]),
    );
    expect(result.tracebacks[0].signature).toContain("panicked");
  });

  test("captures a Segmentation fault", () => {
    const result = auditScratchHygiene(
      trace([
        {
          command: "./a.out",
          output: "Segmentation fault (core dumped)\n",
          exitCode: 139,
        },
      ]),
    );
    expect(result.tracebacks[0].signature).toContain("Segmentation fault");
  });

  test("deduplicates identical traceback signatures across calls", () => {
    const result = auditScratchHygiene(
      trace([
        {
          command: "pytest -k a",
          output:
            "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nValueError: bad arg",
          exitCode: 1,
        },
        {
          command: "pytest -k b",
          output:
            "Traceback (most recent call last):\n  File \"y.py\", line 2, in <module>\nValueError: bad arg",
          exitCode: 1,
        },
      ]),
    );
    expect(result.tracebacks).toHaveLength(1);
  });

  test("emits no tracebacks for clean output", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "ls /app", output: "file.py\n", exitCode: 0 },
      ]),
    );
    expect(result.tracebacks).toEqual([]);
  });
});

describe("auditScratchHygiene — non-zero exit chain", () => {
  test("reports the longest run of consecutive non-zero exits", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "ls", exitCode: 0 },
        { command: "pytest -x", exitCode: 1 },
        { command: "pytest -x --pdb", exitCode: 1 },
        { command: "pytest -x --tb=short", exitCode: 1 },
        { command: "echo done", exitCode: 0 },
      ]),
    );
    expect(result.nonZeroChain.commands).toHaveLength(3);
    expect(result.nonZeroChain.exitCodes).toEqual([1, 1, 1]);
  });

  test("clears the chain when the same command later succeeds", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "pytest -k mytest", exitCode: 1 },
        { command: "pytest -k mytest", exitCode: 1 },
        // Same command (matched by 60-char prefix), exit 0 → chain is cleared.
        { command: "pytest -k mytest", exitCode: 0 },
      ]),
    );
    expect(result.nonZeroChain.commands).toEqual([]);
  });

  test("returns empty chain when there is only a single non-zero exit", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "ls /missing", exitCode: 2 },
        { command: "ls /app", exitCode: 0 },
      ]),
    );
    expect(result.nonZeroChain.commands).toEqual([]);
  });

  test("ignores null exit codes (incomplete tool calls)", () => {
    const result = auditScratchHygiene(
      trace([
        { command: "long-running", exitCode: null },
        { command: "long-running-2", exitCode: null },
      ]),
    );
    expect(result.nonZeroChain.commands).toEqual([]);
  });
});

describe("renderScratchHygieneBlock", () => {
  test("renders a non-empty block with all three categories", () => {
    const block = renderScratchHygieneBlock({
      tmpWrites: [
        { path: "/tmp/build_log.txt", commandIdx: 5 },
        { path: "/tmp/foo.py.c", commandIdx: 12 },
      ],
      tracebacks: [
        {
          signature: "AttributeError: module 'numpy.array_api' has no attribute 'foo'",
          commandIdx: 18,
        },
      ],
      nonZeroChain: { commands: ["pytest", "pytest --pdb"], exitCodes: [1, 1] },
    });
    expect(block).toContain("[Pre-submit hygiene scan]");
    expect(block).toContain("/tmp/build_log.txt (call #5)");
    expect(block).toContain("/tmp/foo.py.c (call #12)");
    expect(block).toContain("AttributeError");
    expect(block).toContain("call #18");
    expect(block).toContain("(exit 1)");
  });

  test("returns empty string for clean signal", () => {
    const block = renderScratchHygieneBlock({
      tmpWrites: [],
      tracebacks: [],
      nonZeroChain: { commands: [], exitCodes: [] },
    });
    expect(block).toBe("");
  });
});

describe("auditScratchHygiene — empty trace", () => {
  test("returns an empty signal", () => {
    const result = auditScratchHygiene([]);
    expect(result).toEqual({
      tmpWrites: [],
      tracebacks: [],
      nonZeroChain: { commands: [], exitCodes: [] },
    });
  });
});
