import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  exportAdapterReplayCorpus,
  type AdapterReplayExportOptions,
} from "../scripts/export_adapter_replay_corpus";

const now = "2026-05-04T00:00:00.000Z";

const tempDir = async () => mkdtemp(join(tmpdir(), "adapter-replay-export-"));

const writeJsonl = async (path: string, records: readonly unknown[]) => {
  await writeFile(path, records.map((record) => JSON.stringify(record)).join("\n") + "\n", "utf8");
};

describe("adapter replay corpus export script", () => {
  test("writes redacted captures, cases, and complete manifest for safe distinct sessions", async () => {
    const root = await tempDir();
    const outDir = join(root, ".bag", "replay-corpus", "source-adapters", "adapter-replay-export");
    await writeJsonl(join(root, "codex-a.jsonl"), codexSession("codex-a"));
    await writeJsonl(join(root, "codex-b.jsonl"), codexSession("codex-b"));
    await writeFile(join(root, "not-a-session.jsonl"), "{\"nope\":true}\n", "utf8");

    const manifest = await exportAdapterReplayCorpus(options(root, outDir, 2), now);

    expect(manifest.status).toBe("complete");
    expect(manifest.discovery).toMatchObject({
      candidateFileCount: 3,
      detectedSourceFileCount: 2,
      exportedSessionCount: 2,
      distinctSessionCount: 2,
    });
    expect(manifest.counts.bySourceKind).toEqual({ "codex-session-jsonl": 2 });
    expect(manifest.counts.bySplit).toEqual({ dev: 1, train: 1 });
    expect(manifest.counts.redaction.rawLocalContentRetained).toBe(false);
    expect(manifest.exportedSessions).toHaveLength(2);
    expect(manifest.rejectedSessions).toHaveLength(1);
    expect(manifest.reproductionCommand).toContain("scripts/export_adapter_replay_corpus.ts");

    const manifestRaw = await readFile(join(outDir, "manifest.json"), "utf8");
    const captureRaw = await readFile(join(root, manifest.exportedSessions[0]!.capturePath), "utf8");
    const caseRaw = await readFile(join(root, manifest.exportedSessions[0]!.replayCasePath), "utf8");
    expect(`${manifestRaw}\n${captureRaw}\n${caseRaw}`).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz123456");
    expect(captureRaw).toContain("path:sha256:");
    expect(caseRaw).toContain("observed baselines rather than golden expected behavior");
  });

  test("writes explicit blocker when too few safe sessions are exported", async () => {
    const root = await tempDir();
    const outDir = join(root, ".bag", "replay-corpus", "source-adapters", "adapter-replay-export");
    await writeJsonl(join(root, "codex-a.jsonl"), codexSession("codex-a"));

    const manifest = await exportAdapterReplayCorpus(options(root, outDir, 2), now);

    expect(manifest.status).toBe("blocked");
    expect(manifest.blocker).toMatchObject({
      code: "insufficient_safe_sessions",
      safeDistinctSessionCount: 1,
      requiredDistinctSessionCount: 2,
    });
  });
});

const options = (root: string, outDir: string, minDistinctSessions: number): AdapterReplayExportOptions => ({
  roots: [root],
  outDir,
  limit: 2,
  minDistinctSessions,
  maxCandidateFiles: 10,
  maxFileBytes: 1024 * 1024,
  maxRecordsPerSession: 20,
  maxTextExcerptChars: 96,
  splitPattern: ["train", "dev"],
  rootPath: root,
});

const codexSession = (id: string): unknown[] => [
  {
    timestamp: now,
    type: "session_meta",
    payload: {
      id,
      timestamp: now,
      cwd: "/Users/example/private-project",
      cli_version: "0.99.0",
      source: "codex",
      model_provider: "openai",
    },
  },
  {
    timestamp: now,
    type: "response_item",
    payload: {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "Run the adapter replay export and keep the token private." }],
    },
  },
  {
    timestamp: now,
    type: "response_item",
    payload: {
      type: "function_call",
      id: `tool-${id}`,
      call_id: `call-${id}`,
      name: "exec_command",
      arguments: JSON.stringify({
        cmd: "npm test",
        token: "ghp_abcdefghijklmnopqrstuvwxyz123456",
      }),
    },
  },
  {
    timestamp: now,
    type: "response_item",
    payload: {
      type: "function_call_output",
      id: `out-${id}`,
      call_id: `call-${id}`,
      status: "failed",
      output: "Process exited with code 1",
    },
  },
];
