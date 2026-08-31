import { describe, expect, test } from "bun:test";

import {
  detectAnswerWobble,
  renderWobbleScanBlock,
} from "../src/audit/answer-wobble";
import type { BashTraceTailEntry } from "../src/pre-submit-self-check";

const trace = (commands: string[]): BashTraceTailEntry[] =>
  commands.map((command) => ({ command, output: "", exitCode: 0 }));

describe("detectAnswerWobble", () => {
  test("single write to deliverable yields no wobble", () => {
    const report = detectAnswerWobble(
      trace(["cat > /app/move.txt <<'EOF'\ne2e4\nEOF\ncat /app/move.txt"]),
    );
    expect(report.wobbledFiles).toEqual([]);
  });

  test("two writes with identical content do not trigger wobble", () => {
    const report = detectAnswerWobble(
      trace([
        "cat > /app/move.txt <<'EOF'\ne2e4\nEOF",
        "echo 'sanity check' > /tmp/log",
        "cat > /app/move.txt <<'EOF'\ne2e4\nEOF",
      ]),
    );
    expect(report.wobbledFiles).toEqual([]);
  });

  test("two writes with different content trigger wobble and capture both versions", () => {
    const report = detectAnswerWobble(
      trace([
        "cat > /app/move.txt <<'EOF'\ne2e4\nEOF",
        "ls /app",
        "cat > /app/move.txt <<'EOF'\ng2g4\nEOF",
      ]),
    );
    expect(report.wobbledFiles.length).toBe(1);
    const entry = report.wobbledFiles[0];
    if (!entry) throw new Error("expected wobbled entry");
    expect(entry.path).toBe("/app/move.txt");
    expect(entry.versions.length).toBe(2);
    const [v1, v2] = entry.versions;
    if (!v1 || !v2) throw new Error("expected two versions");
    expect(v1.commandIdx).toBe(1);
    expect(v2.commandIdx).toBe(3);
    expect(v1.contentDigest).not.toBe(v2.contentDigest);
    // Each digest is a 64-char hex SHA-256.
    expect(v1.contentDigest).toMatch(/^[0-9a-f]{64}$/);
    expect(v2.contentDigest).toMatch(/^[0-9a-f]{64}$/);
    expect(v1.bytes).toBeGreaterThan(0);
    expect(v2.bytes).toBeGreaterThan(0);
  });

  test("opaque writes (cp / pipe / variable interpolation) are conservatively skipped", () => {
    // The agent first writes 'e2e4' literally, then later does `cp x /app/move.txt`
    // (opaque source). We MUST NOT flag this as wobble because we cannot prove
    // the cp produced a different answer.
    const report = detectAnswerWobble(
      trace([
        "echo 'e2e4' > /app/move.txt",
        "cp /tmp/answer /app/move.txt",
        // also a piped command, fully opaque:
        "stockfish | head -1 > /app/move.txt",
      ]),
    );
    expect(report.wobbledFiles).toEqual([]);
  });

  test("echo with `>>` (append) is not treated as a fresh write", () => {
    // The detector only tracks write redirections (`>`), so two appends
    // followed by one fresh write should still register as a single
    // recoverable write.
    const report = detectAnswerWobble(
      trace([
        "echo 'a' >> /app/log.txt",
        "echo 'b' >> /app/log.txt",
        "echo 'final' > /app/log.txt",
      ]),
    );
    expect(report.wobbledFiles).toEqual([]);
  });

  test("printf with literal format string is recovered", () => {
    const report = detectAnswerWobble(
      trace([
        "printf 'one' > /app/answer.txt",
        "printf 'two' > /app/answer.txt",
      ]),
    );
    expect(report.wobbledFiles.length).toBe(1);
    const entry = report.wobbledFiles[0];
    if (!entry) throw new Error("expected wobbled entry");
    expect(entry.path).toBe("/app/answer.txt");
  });

  test("printf with a format specifier is treated as opaque", () => {
    // `printf '%s\n' 'X'` we will not interpret to avoid disagreeing with
    // shell semantics; it must NOT trip wobble detection on its own.
    const report = detectAnswerWobble(
      trace([
        "printf '%s\\n' 'one' > /app/answer.txt",
        "printf '%s\\n' 'two' > /app/answer.txt",
      ]),
    );
    expect(report.wobbledFiles).toEqual([]);
  });

  test("redirect targets that are devices or directories are ignored", () => {
    const report = detectAnswerWobble(
      trace([
        "echo 'a' > /dev/null",
        "echo 'b' > /dev/null",
      ]),
    );
    expect(report.wobbledFiles).toEqual([]);
  });

  test("renderWobbleScanBlock emits a structured block", () => {
    const report = detectAnswerWobble(
      trace([
        "cat > /app/move.txt <<'EOF'\ne2e4\nEOF",
        "cat > /app/move.txt <<'EOF'\ng2g4\nEOF",
      ]),
    );
    const block = renderWobbleScanBlock(report);
    expect(block.startsWith("[Wobble scan]")).toBe(true);
    expect(block).toContain("/app/move.txt");
    expect(block).toContain("versions=2");
    expect(block).toContain("cmd#1");
    expect(block).toContain("cmd#2");
  });

  test("renderWobbleScanBlock indicates a clean scan when nothing wobbled", () => {
    const block = renderWobbleScanBlock({ wobbledFiles: [] });
    expect(block).toContain("[Wobble scan]");
    expect(block).toContain("(no wobble detected)");
  });
});
