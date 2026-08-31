import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { acpFailureOutcomeFor } from "../src/acp/permission-outcomes";
import type { BagAcpSession } from "../src/acp/session";
import { runTerminalCommand } from "../src/acp/terminal";
import {
  absoluteSessionPath,
  displayPathForSessionId,
  editToolContent,
  sessionRelativePath,
} from "../src/acp/workspace-io";
import type { RunTelemetry } from "../src/telemetry";

const sessionFor = (input: {
  cwd: string;
  additionalDirectories?: string[];
  richDiffContent?: boolean;
  terminal?: boolean;
  yolo?: boolean;
}): BagAcpSession => ({
  id: "bag-test",
  cwd: input.cwd,
  additionalDirectories: input.additionalDirectories ?? [],
  executorConcurrency: 8,
  mode: "auto",
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
  pendingPrompt: null,
  title: "test",
  yolo: input.yolo ?? true,
  mcpServers: [],
  optimizerPin: {} as never,
  clientCapabilities: {
    fsReadTextFile: true,
    fsWriteTextFile: true,
    terminal: input.terminal ?? true,
    richDiffContent: input.richDiffContent ?? true,
    richTerminalContent: true,
    source: "test",
  },
});

const fakeTelemetry = (): RunTelemetry => ({
  measureToolCall: async (input: { fn: () => Promise<unknown> }) => input.fn(),
}) as RunTelemetry;

describe("ACP tool boundary modules", () => {
  test("resolve and display workspace paths without duplicating path policy", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-io-root-"));
    const extra = mkdtempSync(join(tmpdir(), "bag-acp-io-extra-"));
    const rootFile = join(cwd, "src.ts");
    const extraFile = join(extra, "note.md");
    writeFileSync(rootFile, "root\n");
    writeFileSync(extraFile, "extra\n");
    const session = sessionFor({ cwd, additionalDirectories: [extra] });

    expect(absoluteSessionPath(session, "src.ts")).toBe(rootFile);
    expect(absoluteSessionPath(session, extraFile)).toBe(extraFile);
    expect(sessionRelativePath(session, rootFile)).toBe("src.ts");
    expect(sessionRelativePath(session, extraFile)).toBe(extraFile);
    expect(displayPathForSessionId(new Map([[session.id, session]]), session.id, extraFile)).toBe(extraFile);
  });

  test("renders rich diff content and text fallback from the same edit boundary", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-content-"));
    const rich = sessionFor({ cwd, richDiffContent: true });
    const textOnly = sessionFor({ cwd, richDiffContent: false });

    expect(editToolContent({
      session: rich,
      path: join(cwd, "file.ts"),
      oldContent: "old\n",
      newContent: "new\n",
      oldHash: "old-hash",
      newHash: "new-hash",
    })[0]).toMatchObject({ type: "diff", oldText: "old\n", newText: "new\n" });

    expect(JSON.stringify(editToolContent({
      session: textOnly,
      path: join(cwd, "file.ts"),
      oldContent: "old\n",
      newContent: "new\n",
      oldHash: "old-hash",
      newHash: "new-hash",
    }))).toContain("Proposed edit to file.ts");
  });

  test("normalizes permission and terminal failure outcomes", async () => {
    expect(acpFailureOutcomeFor({ cancelled: true, message: "cancelled" })).toBe("cancelled");
    expect(acpFailureOutcomeFor({ cancelled: false, message: "command permission rejected" })).toBe("permission_rejected");
    expect(acpFailureOutcomeFor({ cancelled: false, message: "boom" })).toBe("failed");

    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-terminal-module-"));
    const session = sessionFor({ cwd });
    const updates: unknown[] = [];
    const releases: string[] = [];
    const result = await runTerminalCommand({
      connection: {
        sessionUpdate: async (update: unknown) => {
          updates.push(update);
        },
        createTerminal: async () => ({
          id: "terminal-1",
          waitForExit: async () => ({ exitCode: 9, signal: null }),
          currentOutput: async () => ({ output: "failed\n" }),
          kill: async () => ({}),
          release: async () => {
            releases.push("terminal-1");
            return {};
          },
        }),
      } as never,
      requireSession: () => session,
    }, {
      sessionId: session.id,
      telemetry: fakeTelemetry(),
      command: "npm",
      args: ["test"],
      reason: "module test",
      cwd,
    });

    expect(result).toMatchObject({ exitCode: 9, output: "failed\n" });
    expect(releases).toEqual(["terminal-1"]);
    expect(JSON.stringify(updates)).toContain('"status":"failed"');
  });
});
