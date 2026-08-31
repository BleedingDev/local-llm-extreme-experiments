import { describe, expect, test } from "bun:test";
import {
  classifyCodingProgress,
  type CodingProgressClass,
} from "../acp/coding-progress-diagnostics";
import type { CodingEditResult, CodingPatch } from "../acp/coding-types";
import {
  noWriteValidationInputFromRealAcpTaskRunResult,
  validateNoWriteProgress,
  type NoWriteValidationInput,
} from "./no-write-validation";
import type { RealAcpTaskRunResult } from "./real-acp-runner";

const baseInput: NoWriteValidationInput = {
  recordId: "record.no-write.base",
  taskId: "task.no-write.base",
  routeSelectedMode: "coding",
  expectedMutation: "edit_existing",
  expectedSideEffect: "mutation",
  changedFiles: [],
  fsWriteCount: 0,
  terminalCreateCount: 0,
  terminalExitCount: 0,
  terminalCommandCount: 0,
  stopReason: "end_turn",
  editStrategyFamily: "none",
  verifierStatus: "failed",
  evidenceRefs: ["scorecard:tool-routing.visible-acp"],
};

describe("no-write validation oracle", () => {
  test("blocks mutation-expected coding tasks with no writes, changed files, or terminal progress", () => {
    const result = validateNoWriteProgress(baseInput);

    expect(result).toMatchObject({
      passed: false,
      severity: "block",
      classification: "mutation_progress_missing",
      missingProgressSignals: ["changed_files", "fs_write", "terminal_create", "terminal_exit"],
      observed: {
        routeSelectedMode: "coding",
        expectedMutation: "edit_existing",
        fsWriteCount: 0,
        terminalCreateCount: 0,
        stopReason: "end_turn",
        editStrategyFamily: "none",
      },
    });
  });

  test("passes legitimate read-only tasks without requiring mutation progress", () => {
    const result = validateNoWriteProgress({
      ...baseInput,
      recordId: "record.no-write.read-only",
      taskId: "task.no-write.read-only",
      routeSelectedMode: "read_only",
      expectedMutation: "no_change",
      expectedSideEffect: "read",
      verifierStatus: "not_run",
    });

    expect(result).toMatchObject({
      passed: true,
      severity: "pass",
      classification: "read_only_legitimate",
      missingProgressSignals: [],
    });
  });

  test("warns but passes when verifier skip is explicit and justified", () => {
    const result = validateNoWriteProgress({
      ...baseInput,
      recordId: "record.no-write.verifier-skip",
      taskId: "task.no-write.verifier-skip",
      verifierStatus: "skipped",
      verifierSkippedJustification: {
        present: true,
        policy: "allowed_to_skip",
        reason: "docs surface has no executable verifier",
      },
    });

    expect(result).toMatchObject({
      passed: true,
      severity: "warn",
      classification: "verifier_skip_justified",
      observed: {
        verifierStatus: "skipped",
        verifierSkippedJustificationPresent: true,
      },
    });
  });

  test("passes when write or terminal progress is present", () => {
    const writeResult = validateNoWriteProgress({
      ...baseInput,
      recordId: "record.no-write.fs-write",
      changedFiles: [{ path: "src/greeter.ts", changeKind: "modified" }],
      fsWriteCount: 1,
      editStrategyFamily: "whole_file",
      verifierStatus: "passed",
    });
    const terminalResult = validateNoWriteProgress({
      ...baseInput,
      recordId: "record.no-write.terminal",
      terminalCreateCount: 1,
      terminalExitCount: 1,
      terminalCommandCount: 1,
      verifierStatus: "passed",
    });

    expect(writeResult).toMatchObject({
      passed: true,
      severity: "pass",
      classification: "write_or_terminal_progress",
    });
    expect(terminalResult).toMatchObject({
      passed: true,
      severity: "pass",
      classification: "write_or_terminal_progress",
    });
  });

  test("normalizes visible real ACP task-result fields without parsing failure text", () => {
    const result = validateNoWriteProgress(noWriteValidationInputFromRealAcpTaskRunResult({
      result: realAcpTaskResultFixture(),
      expectedMutation: "edit_existing",
    }));

    expect(result).toMatchObject({
      recordId: "real-acp-run.visible.real-acp.task.simple-edit-greeting",
      taskId: "real-acp.task.simple-edit-greeting",
      passed: false,
      classification: "mutation_progress_missing",
      observed: {
        fsWriteCount: 0,
        terminalCreateCount: 0,
        terminalExitCount: 0,
        stopReason: "end_turn",
      },
    });
    expect(result.evidenceRefs).toContain("real-acp-task-result:real-acp-run.visible.real-acp.task.simple-edit-greeting");
    expect(result.evidenceRefs).toContain("/tmp/transcript.json");
  });

  test("uses precise coding progress classes for mutation-expected no-progress blocks", () => {
    const result = validateNoWriteProgress({
      ...baseInput,
      recordId: "record.no-write.no-model",
      codingProgressClass: "no_model",
      verifierStatus: "failed",
    });

    expect(result).toMatchObject({
      passed: false,
      severity: "block",
      classification: "no_model",
    });
    expect(result.reasons.join("\n")).toContain("no_model");
  });

  test("accepts structured impossibility as explicit non-mutating progress", () => {
    const result = validateNoWriteProgress({
      ...baseInput,
      recordId: "record.no-write.structured-impossibility",
      codingProgressClass: "structured_impossibility",
      verifierStatus: "not_run",
    });

    expect(result).toMatchObject({
      passed: true,
      severity: "warn",
      classification: "structured_impossibility",
    });
  });
});

describe("coding progress diagnostics classifier", () => {
  test.each([
    ["no_model", { patch: patch({ generation: { modelAvailable: false } }) }],
    ["model_error", { patch: patch({ generation: { modelAvailable: true, modelError: "upstream 500" } }) }],
    ["empty_edits", { patch: patch() }],
    ["parse_rejected", { patch: patch({ parseFailures: ["edit 1 malformed"] }) }],
    ["fallback_empty", { patch: patch({ parseFailures: ["edit 1 malformed"] }), fallbackPatch: patch() }],
    ["executor_failed", {
      patch: patch({ editCount: 1 }),
      editResults: [editResult({ ok: false, errorCode: "schema_validation_error", reason: "preview failed" })],
    }],
    ["permission_rejected", {
      patch: patch({ editCount: 1 }),
      editResults: [editResult({ ok: false, errorCode: "permission_rejected", reason: "edit permission rejected" })],
    }],
    ["client_write_failed", {
      patch: patch({ editCount: 1 }),
      editResults: [editResult({ ok: false, errorCode: "acp_write_failed", reason: "ACP client does not support fs/write_text_file" })],
    }],
    ["verifier_missing", {
      patch: patch({ editCount: 1 }),
      editResults: [editResult({ ok: true })],
    }],
    ["verifier_failed", {
      patch: patch({ editCount: 1 }),
      editResults: [editResult({ ok: true })],
      plannedCommands: [{}],
      commandResults: [{ command: "bun", args: ["test"], reason: "verify", exitCode: 1, signal: null, output: "fail" }],
    }],
    ["verified_edit", {
      patch: patch({ editCount: 1 }),
      editResults: [editResult({ ok: true })],
      plannedCommands: [{}],
      commandResults: [{ command: "bun", args: ["test"], reason: "verify", exitCode: 0, signal: null, output: "ok" }],
    }],
    ["structured_impossibility", {
      patch: patch({
        structuredImpossibility: { reason: "required secret is unavailable", evidenceRefs: ["task"] },
      }),
    }],
  ] satisfies readonly [CodingProgressClass, Omit<Parameters<typeof classifyCodingProgress>[0], "runId">][])(
    "classifies %s",
    (expected: CodingProgressClass, input: Omit<Parameters<typeof classifyCodingProgress>[0], "runId">) => {
      expect(classifyCodingProgress({
        runId: `run.${expected}`,
        ...input,
      }).progressClass).toBe(expected);
    },
  );
});

const realAcpTaskResultFixture = (): RealAcpTaskRunResult => ({
  schemaVersion: "real-acp-task-result.v1",
  runResultId: "real-acp-run.visible.real-acp.task.simple-edit-greeting",
  taskId: "real-acp.task.simple-edit-greeting",
  split: "train",
  optimizationAllowed: true,
  status: "failed",
  startedAt: "2026-05-04T00:00:00.000Z",
  completedAt: "2026-05-04T00:00:01.000Z",
  workspaceFingerprintBefore: "sha256:before",
  workspaceFingerprintAfter: "sha256:after",
  changedFiles: [],
  route: {
    routeId: "route.real-acp.task.simple-edit-greeting",
    selectedMode: "coding",
    reason: "headless ACP consumer",
  },
  editStrategy: {
    strategyId: "edit.headless-acp.consumer.v1",
    family: "none",
    selectedBy: "not_applicable",
  },
  toolCalls: [
    {
      toolCallId: "tool.real-acp.task.simple-edit-greeting.read",
      namespace: "acp.fs",
      name: "readTextFile",
      status: "succeeded",
      sideEffectLevel: "read",
    },
  ],
  terminalCommands: [],
  verifier: {
    status: "failed",
    policy: "required",
    commandIds: [],
  },
  repair: {
    attempted: false,
    status: "not_needed",
  },
  rollback: {
    attempted: false,
    status: "not_needed",
  },
  corrections: [],
  lineage: {
    taskId: "real-acp.task.simple-edit-greeting",
    runResultId: "real-acp-run.visible.real-acp.task.simple-edit-greeting",
    sourceTaskPackId: "real-acp-task-pack.visible",
  },
  telemetry: {
    headlessAcp: {
      stopReason: "end_turn",
      counts: {
        fsWrite: 0,
        terminalCreate: 0,
        terminalExit: 0,
      },
      transcriptPath: "/tmp/transcript.json",
    },
  },
  redaction: {
    rawLocalStatus: "raw_local_only",
    optimizerSafe: true,
    excludedFromOptimizerReasons: [],
  },
  failureReason: "headless ACP transcript stopReason=end_turn",
});

const patch = (input: {
  editCount?: number;
  parseFailures?: string[];
  generation?: CodingPatch["generation"];
  structuredImpossibility?: CodingPatch["structuredImpossibility"];
} = {}): CodingPatch => ({
  summary: "patch",
  editStrategy: {
    strategyId: "edit.whole-file.v1",
    strategyFamily: "whole_file",
    renderedEditToolContractId: "rendered.edit.whole-file.v1",
  },
  ...(input.generation === undefined ? {} : { generation: input.generation }),
  ...(input.structuredImpossibility === undefined ? {} : { structuredImpossibility: input.structuredImpossibility }),
  edits: Array.from({ length: input.editCount ?? 0 }, (_, index) => ({
    reason: `edit ${index}`,
    editInput: {
      strategyFamily: "whole_file",
      payload: {
        path: "src/a.ts",
        content: "export const value = 2;\n",
      },
    },
    targetFiles: ["src/a.ts"],
    editStrategyId: "edit.whole-file.v1",
    editStrategyFamily: "whole_file",
    renderedEditToolContractId: "rendered.edit.whole-file.v1",
  })),
  commands: [],
  risks: [],
  parseFailures: input.parseFailures ?? [],
});

const editResult = (input: {
  ok: boolean;
  reason?: string;
  errorCode?: string;
}): CodingEditResult => ({
  path: "src/a.ts",
  ok: input.ok,
  reason: input.reason ?? "edit",
  oldHash: "sha256:old",
  newHash: "sha256:new",
  editStrategyId: "edit.whole-file.v1",
  editStatus: "applied",
  ...(input.errorCode === undefined ? {} : { errorCode: input.errorCode }),
});
