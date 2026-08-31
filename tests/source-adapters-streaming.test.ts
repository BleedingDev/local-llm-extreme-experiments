import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  detectSourceFile,
  discoverSourceFiles,
  streamJsonlFile,
  streamSourceDirectory,
  streamSourceFile,
} from "../src/source-adapters/streaming";

const now = "2026-04-30T00:00:00.000Z";

const codexMeta = {
  type: "session_meta",
  timestamp: now,
  payload: {
    id: "codex-session-a",
    cwd: "/repo",
    cli_version: "0.99.0",
    source: "codex",
    model_provider: "openai",
  },
};

const codexMessage = {
  type: "response_item",
  timestamp: now,
  payload: {
    type: "message",
    role: "user",
    content: "hello",
  },
};

const spanRecord = {
  trace_id: "trace-a",
  span_id: "span-a",
  parent_span_id: "root",
  trace_state: "",
  name: "tool.workspace.repo.read",
  kind: "SPAN_KIND_INTERNAL",
  start_time: now,
  end_time: now,
  status: {
    code: "STATUS_CODE_OK",
    message: "",
  },
  resource: {
    attributes: {
      "service.name": "bleeding-agent",
    },
  },
  scope: {
    name: "bleeding-agent",
    version: "1.0.0",
  },
  attributes: {
    "openinference.span.kind": "TOOL",
  },
};

const writeJsonl = async (path: string, records: readonly unknown[]) => {
  await writeFile(path, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
};

const tempDir = async () => mkdtemp(join(tmpdir(), "bleeding-agent-source-streaming-"));

describe("source adapter streaming", () => {
  test("streams JSONL records incrementally with line and record indexes", async () => {
    const dir = await tempDir();
    const path = join(dir, "session.jsonl");
    await writeJsonl(path, [codexMeta, codexMessage]);

    const items = [];
    for await (const item of streamJsonlFile(path)) {
      items.push(item);
    }

    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({
      kind: "record",
      path,
      line: 1,
      recordIndex: 0,
      record: codexMeta,
    });
    expect(items[1]).toMatchObject({
      kind: "record",
      line: 2,
      recordIndex: 1,
      record: codexMessage,
    });
  });

  test("emits malformed line and non-object diagnostics without throwing", async () => {
    const dir = await tempDir();
    const path = join(dir, "broken.jsonl");
    await writeFile(path, `${JSON.stringify(codexMeta)}\n{"type":\n42\n`);

    const items = [];
    for await (const item of streamJsonlFile(path)) {
      items.push(item);
    }

    expect(items.filter((item) => item.kind === "record")).toHaveLength(1);
    const diagnostics = items.filter((item) => item.kind === "diagnostic");
    expect(diagnostics).toHaveLength(2);
    expect(diagnostics[0]).toMatchObject({
      kind: "diagnostic",
      diagnostic: {
        code: "malformed_jsonl",
        path,
        line: 2,
      },
    });
    expect(diagnostics[1]).toMatchObject({
      kind: "diagnostic",
      diagnostic: {
        code: "non_object_record",
        path,
        line: 3,
        recordIndex: 1,
      },
    });
  });

  test("detects source files from a bounded sample and streams with source metadata", async () => {
    const dir = await tempDir();
    const path = join(dir, "codex-session.jsonl");
    await writeJsonl(path, [codexMeta, codexMessage]);

    const detection = await detectSourceFile(path, { maxInspectionRecords: 1 });
    expect(detection.ok).toBe(true);
    if (!detection.ok) {
      throw new Error("expected supported source");
    }
    expect(detection.source).toMatchObject({
      sourceType: "codex-session-jsonl",
      path,
      sessionId: "codex-session-a",
      schemaVersion: "0.99.0",
      inspectedRecordCount: 1,
    });
    expect(detection.source.recordCountEstimate).toEqual({ value: 1, kind: "sample" });

    const streamed = [];
    for await (const item of streamSourceFile(path)) {
      streamed.push(item);
    }

    expect(streamed).toHaveLength(2);
    expect(streamed[0]).toMatchObject({
      kind: "record",
      source: {
        sourceType: "codex-session-jsonl",
        sessionId: "codex-session-a",
      },
    });
  });

  test("redacts streamed source records only through explicit opt-in", async () => {
    const dir = await tempDir();
    const path = join(dir, "codex-session.jsonl");
    const secretMessage = {
      ...codexMessage,
      payload: {
        ...codexMessage.payload,
        content: "sk-testsecretvalue1234567890",
      },
    };
    await writeJsonl(path, [codexMeta, secretMessage]);

    const records = [];
    for await (const item of streamSourceFile(path, { redact: true })) {
      records.push(item);
    }

    expect(records[1]).toMatchObject({
      kind: "record",
      redacted: {
        lineage: {
          sessionId: "codex-session-a",
          role: "user",
        },
      },
    });
    expect(JSON.stringify(records[1])).not.toContain("sk-testsecretvalue1234567890");
    expect(JSON.stringify(records[1])).toContain("[REDACTED:openai_api_key]");
  });

  test("discovers supported JSONL files in directories and streams them", async () => {
    const dir = await tempDir();
    const nested = join(dir, "nested");
    await mkdir(nested);
    const codexPath = join(dir, "codex.jsonl");
    const spanPath = join(nested, "spans.ndjson");
    const ignoredPath = join(dir, "notes.txt");
    await writeJsonl(codexPath, [codexMeta, codexMessage]);
    await writeJsonl(spanPath, [spanRecord]);
    await writeFile(ignoredPath, "not jsonl");

    const discovered = [];
    for await (const item of discoverSourceFiles(dir)) {
      discovered.push(item);
    }

    const filePaths = discovered
      .filter((item) => item.kind === "file" && item.detection.ok)
      .map((item) => item.path)
      .sort();
    expect(filePaths).toEqual([codexPath, spanPath].sort());

    const streamed = [];
    for await (const item of streamSourceDirectory(dir)) {
      streamed.push(item);
    }
    expect(streamed.filter((item) => item.kind === "record")).toHaveLength(3);
  });

  test("enforces max file size, max record, and max file caps with diagnostics", async () => {
    const dir = await tempDir();
    const first = join(dir, "first.jsonl");
    const second = join(dir, "second.jsonl");
    await writeJsonl(first, [codexMeta, codexMessage, { ...codexMessage, payload: { role: "assistant" } }]);
    await writeJsonl(second, [codexMeta]);

    const tooLarge = [];
    for await (const item of streamJsonlFile(first, { maxFileBytes: 1 })) {
      tooLarge.push(item);
    }
    expect(tooLarge).toEqual([
      {
        kind: "diagnostic",
        diagnostic: expect.objectContaining({
          code: "file_too_large",
          path: first,
          limit: 1,
        }),
      },
    ]);

    const cappedRecords = [];
    for await (const item of streamJsonlFile(first, { maxRecords: 1 })) {
      cappedRecords.push(item);
    }
    expect(cappedRecords.filter((item) => item.kind === "record")).toHaveLength(1);
    expect(cappedRecords.at(-1)).toMatchObject({
      kind: "diagnostic",
      diagnostic: {
        code: "max_records_reached",
        path: first,
        limit: 1,
      },
    });

    const cappedFiles = [];
    for await (const item of discoverSourceFiles(dir, { maxFiles: 1 })) {
      cappedFiles.push(item);
    }
    expect(cappedFiles.filter((item) => item.kind === "file")).toHaveLength(1);
    expect(cappedFiles.at(-1)).toMatchObject({
      kind: "diagnostic",
      diagnostic: {
        code: "max_files_reached",
        path: dir,
        limit: 1,
      },
    });
  });
});
