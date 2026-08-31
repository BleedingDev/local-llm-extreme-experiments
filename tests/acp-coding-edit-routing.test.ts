import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  defaultVerificationCommands,
  extractPatchTargetFiles,
  parseCodingEditOperation,
  renderVerifierResultsForLlm,
  targetFilesForEditInput,
  verificationCommands,
  type CodingFileSnapshot,
} from "../src/acp/coding-types";
import {
  fallbackTriggerForErrorCode,
  fallbackTriggerForPatch,
  requiresCreateCapableStrategy,
  resolveLiveEditContext,
  serializeLiveEditContext,
} from "../src/acp/edit-routing";
import {
  buildCodingReplayCapture,
  replayToolErrorCodeForMetric,
  replayToolStatusForMetric,
} from "../src/acp/replay-capture";
import type { BagAcpSession } from "../src/acp/session";
import { defaultConfig } from "../src/config";
import { createOptimizerSessionPin } from "../src/optimizer/session-pin";

const sessionFor = (cwd: string): BagAcpSession => ({
  id: "bag-test",
  cwd,
  additionalDirectories: [],
  executorConcurrency: 8,
  mode: "auto",
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
  pendingPrompt: null,
  title: "test",
  yolo: true,
  mcpServers: [],
  optimizerPin: createOptimizerSessionPin(defaultConfig(), cwd, "executor"),
  clientCapabilities: {
    fsReadTextFile: true,
    fsWriteTextFile: true,
    terminal: true,
    richDiffContent: true,
    richTerminalContent: true,
    source: "test",
  },
});

const snapshot = (overrides: Partial<CodingFileSnapshot> = {}): CodingFileSnapshot => ({
  kind: "existing",
  path: "/repo/src/a.ts",
  relativePath: "src/a.ts",
  content: "export const value = 1;\n",
  hash: "hash-a",
  ...overrides,
});

describe("ACP coding edit routing helpers", () => {
  test("formats verifier output for repair prompts without losing failure evidence", () => {
    const text = renderVerifierResultsForLlm([
      { command: "npm", args: ["test"], exitCode: 1, output: "first\nsecond\n" },
    ], 2);

    expect(text).toContain("Verifier results from repair round 1");
    expect(text).toContain("$ npm test");
    expect(text).toContain("FAILED (exit 1)");
    expect(text).toContain("second");
  });

  test("normalizes model paths and parses selected strategy payloads", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-coding-route-"));
    const file = snapshot({ path: join(cwd, "src/a.ts") });
    const editContext = resolveLiveEditContext(sessionFor(cwd), [file]);
    const parsed = parseCodingEditOperation({
      rawEdit: {
        reason: "update value",
        payload: {
          path: file.path,
          content: "export const value = 2;\n",
        },
      },
      index: 0,
      editContext: {
        ...editContext,
        decision: {
          ...editContext.decision,
          selectedStrategyId: "edit.whole-file.v1",
          selectedStrategyFamily: "whole_file",
        },
      },
      fileSnapshots: [file],
    });

    expect(parsed.parseFailure).toBeUndefined();
    expect(parsed.edit).toMatchObject({
      reason: "update value",
      targetFiles: ["src/a.ts"],
      editStrategyFamily: "whole_file",
    });
    expect(parsed.edit?.editInput).toMatchObject({
      strategyFamily: "whole_file",
      payload: { path: "src/a.ts" },
    });
  });

  test("routes empty and create-target workspaces to whole-file create support", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-greenfield-route-"));
    const emptyContext = resolveLiveEditContext(sessionFor(cwd), []);
    const createContext = resolveLiveEditContext(sessionFor(cwd), [
      snapshot({
        kind: "create",
        path: join(cwd, "answer.py"),
        relativePath: "answer.py",
        content: "",
        hash: "empty",
      }),
    ]);

    expect(requiresCreateCapableStrategy([])).toBe(true);
    expect(emptyContext.decision.selectedStrategyFamily).toBe("whole_file");
    expect(emptyContext.taskShape).toMatchObject({
      targetFileCount: 0,
      verifierStrength: "none",
    });
    expect(createContext.decision.selectedStrategyFamily).toBe("whole_file");

    const parsed = parseCodingEditOperation({
      rawEdit: {
        reason: "create script",
        payload: {
          path: "answer.py",
          content: "print('ok')\n",
        },
      },
      index: 0,
      editContext: createContext,
      fileSnapshots: [],
    });
    expect(parsed.parseFailure).toBeUndefined();
    expect(parsed.edit).toMatchObject({
      targetFiles: ["answer.py"],
      editStrategyFamily: "whole_file",
      editInput: {
        strategyFamily: "whole_file",
        payload: { path: "answer.py", content: "print('ok')\n" },
      },
    });
  });

  test("derives patch targets and verification fallback without hardcoding one edit strategy", () => {
    expect(extractPatchTargetFiles("diff --git a/src/a.ts b/src/a.ts\n--- a/src/a.ts\n+++ b/src/a.ts\n"))
      .toEqual(["src/a.ts"]);
    expect(extractPatchTargetFiles("*** Begin Patch\n*** Update File: src/b.ts\n@@\n-old\n+new\n*** End Patch\n"))
      .toEqual(["src/b.ts"]);
    expect(targetFilesForEditInput({
      strategyFamily: "apply_patch",
      payload: { patch: "no explicit target" },
    }, [snapshot()])).toEqual(["src/a.ts"]);

    expect(verificationCommands([{ command: "bun", args: ["test"], reason: "requested" }], process.cwd()))
      .toEqual([{ command: "bun", args: ["test"], reason: "requested" }]);
    expect(verificationCommands([], process.cwd()).map((command) => command.command)).toContain("npm");
  });

  test("does not synthesize npm typecheck for unknown non-Node projects", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-unknown-verify-"));

    expect(defaultVerificationCommands("unknown")).toEqual([]);
    expect(verificationCommands([], cwd)).toEqual([]);
  });

  test("keeps edit fallback routing measurable by trigger instead of fixing one strategy", () => {
    expect(fallbackTriggerForErrorCode("schema_validation_error")).toBe("parse_failed");
    expect(fallbackTriggerForErrorCode("anchor_stale")).toBe("stale_context");
    expect(fallbackTriggerForErrorCode("protected_path_violation")).toBe("protected_path_violation");
    expect(fallbackTriggerForErrorCode("permission_rejected")).toBeUndefined();
    expect(fallbackTriggerForErrorCode("unknown_error")).toBe("apply_failed");

    const trigger = fallbackTriggerForPatch({
      summary: "bad payload",
      editStrategy: {
        strategyId: "edit.apply-patch.v1",
        strategyFamily: "apply_patch",
        renderedEditToolContractId: "rendered.edit.apply-patch.v1.test",
      },
      edits: [],
      commands: [],
      risks: [],
      parseFailures: ["edit 1: malformed"],
    }, []);
    expect(trigger).toBe("parse_failed");
  });

  test("serializes live edit route with optimizer evidence dimensions", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-edit-route-"));
    const context = resolveLiveEditContext(sessionFor(cwd), [snapshot({ path: join(cwd, "src/a.ts") })]);
    const serialized = serializeLiveEditContext(context);

    expect(serialized).toMatchObject({
      taskShape: expect.objectContaining({ targetFileCount: 1 }),
      decision: expect.objectContaining({
        selectedStrategyId: expect.any(String),
        selectedStrategyFamily: expect.any(String),
      }),
      renderedContract: expect.objectContaining({ renderedToolId: expect.any(String) }),
    });
  });

  test("builds replay capture records with tool failure classes preserved for optimization", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-replay-route-"));
    const session = sessionFor(cwd);
    const capture = buildCodingReplayCapture({
      session,
      runId: "acp-code-test",
      task: "change code",
      tracePath: join(cwd, ".bag", "runs", "trace.json"),
      fileSnapshots: [snapshot({ path: join(cwd, "src/a.ts") })],
      editAttempts: [],
      commandResults: [{
        command: "npm",
        args: ["test"],
        reason: "verify",
        exitCode: 1,
        signal: null,
        output: "failed\n",
      }],
      toolMetrics: [{
        toolName: "write_file",
        namespace: "acp",
        startedAt: "2026-01-01T00:00:00.000Z",
        completedAt: "2026-01-01T00:00:01.000Z",
        durationMs: 1000,
        ok: false,
        retryCount: 0,
        argumentBytes: 32,
        argumentHash: "arg-hash",
        resultKind: "json",
        error: "permission rejected",
        errorName: "PermissionRejected",
      }],
      artifactRefs: [join(cwd, ".bag", "runs", "trace.json")],
    });

    expect(capture.records.map((record) => record.recordKind)).toContain("tool_call");
    expect(capture.records).toContainEqual(expect.objectContaining({
      recordKind: "terminal_command",
      status: "failed",
      errorCode: "verifier_error",
    }));
    expect(replayToolStatusForMetric({
      toolName: "edit",
      startedAt: "2026-01-01T00:00:00.000Z",
      completedAt: "2026-01-01T00:00:01.000Z",
      durationMs: 1000,
      ok: false,
      retryCount: 0,
      argumentBytes: 10,
      argumentHash: "arg-hash",
      resultKind: "empty",
      error: "malformed arguments",
    })).toBe("malformed_args");
    expect(replayToolErrorCodeForMetric({
      toolName: "edit",
      startedAt: "2026-01-01T00:00:00.000Z",
      completedAt: "2026-01-01T00:00:01.000Z",
      durationMs: 1000,
      ok: false,
      retryCount: 0,
      argumentBytes: 10,
      argumentHash: "arg-hash",
      resultKind: "empty",
      error: "permission rejected",
    })).toBe("permission_denied");
  });

  test("replay capture records planned create targets separately from successful reads", () => {
    const cwd = mkdtempSync(join(tmpdir(), "bag-acp-replay-create-"));
    const capture = buildCodingReplayCapture({
      session: sessionFor(cwd),
      runId: "acp-code-create",
      task: "create answer.py",
      tracePath: join(cwd, ".bag", "runs", "trace.json"),
      fileSnapshots: [{
        kind: "create",
        path: join(cwd, "answer.py"),
        relativePath: "answer.py",
        content: "",
        hash: "empty",
      }],
      editAttempts: [],
      commandResults: [],
      toolMetrics: [],
      artifactRefs: [],
    });

    expect(capture.records).toContainEqual(expect.objectContaining({
      recordKind: "file_read",
      path: "answer.py",
      status: "omitted",
      errorCode: "planned_create_target",
    }));
  });
});
